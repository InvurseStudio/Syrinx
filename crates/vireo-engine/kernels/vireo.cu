// Vireo's CUDA kernels: everything one token of Llama, Qwen 3.5 and Gemma 4 needs.
//
// Compiled to PTX by build.rs and loaded by src/cuda/mod.rs. The quantized block
// layouts match src/quant.rs (and ggml), and every kernel does the same arithmetic as
// the CPU reference in f32, so the two can be compared number for number. These are
// the first, straightforward versions: correct first, then tuned.

#include <cuda_fp16.h>

typedef unsigned char u8;

__device__ __forceinline__ float half_at(const u8* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// The sum of `v` over the block, returned to every thread. `scratch` holds 32 floats.
__device__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();  // scratch may still be in use from a previous call
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x + 31) / 32 ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

__device__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x + 31) / 32 ? scratch[threadIdx.x] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) scratch[0] = v;
    __syncthreads();
    return scratch[0];
}

// ---- Matrix-vector products: y = W x, one warp per row of W -------------------------

// Plain f32 rows.
extern "C" __global__ void matvec_f32(const float* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* r = w + (size_t)row * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 32) sum += r[c] * x[c];
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// f16 rows.
extern "C" __global__ void matvec_f16(const __half* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const __half* r = w + (size_t)row * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 32) sum += __half2float(r[c]) * x[c];
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Q8_0: blocks of 32 in 34 bytes (f16 scale, 32 int8s). Lane l takes number l of each block.
extern "C" __global__ void matvec_q8_0(const u8* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int blocks = cols / 32;
    const u8* r = w + (size_t)row * blocks * 34;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const u8* block = r + b * 34;
        sum += half_at(block) * (float)(signed char)block[2 + lane] * x[b * 32 + lane];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Q4_0: blocks of 32 in 18 bytes (f16 scale, 16 bytes of 4-bit numbers, offset 8).
// Byte i holds number i (low nibble) and i + 16 (high): lanes 0-15 take the low
// nibbles, lanes 16-31 the high.
extern "C" __global__ void matvec_q4_0(const u8* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int blocks = cols / 32;
    const u8* r = w + (size_t)row * blocks * 18;
    float sum = 0.0f;
    const int byte = lane & 15, shift = lane < 16 ? 0 : 4;
    for (int b = 0; b < blocks; b++) {
        const u8* block = r + b * 18;
        const int q = (block[2 + byte] >> shift) & 0xF;
        sum += half_at(block) * (float)(q - 8) * x[b * 32 + lane];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Q4_K's 6-bit scale and minimum for sub-block j (see q4_k_scale_min in quant.rs).
__device__ __forceinline__ void q4_k_scale_min(int j, const u8* q, float& scale, float& min) {
    if (j < 4) {
        scale = q[j] & 63;
        min = q[j + 4] & 63;
    } else {
        scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K: blocks of 256 in 144 bytes. Lane l reads byte l of each 32-byte group of
// numbers, which holds number l of one sub-block (low nibble) and of the next (high).
extern "C" __global__ void matvec_q4_k(const u8* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int blocks = cols / 256;
    const u8* r = w + (size_t)row * blocks * 144;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const u8* block = r + b * 144;
        const float d = half_at(block), dmin = half_at(block + 2);
        const u8* scales = block + 4;
        const u8* qs = block + 16;
        const float* xb = x + b * 256;
        for (int pair = 0; pair < 4; pair++) {
            float s1, m1, s2, m2;
            q4_k_scale_min(2 * pair, scales, s1, m1);
            q4_k_scale_min(2 * pair + 1, scales, s2, m2);
            const u8 q = qs[pair * 32 + lane];
            sum += (d * s1 * (float)(q & 0xF) - dmin * m1) * xb[pair * 64 + lane];
            sum += (d * s2 * (float)(q >> 4) - dmin * m2) * xb[pair * 64 + 32 + lane];
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Q5_K: Q4_K plus 32 bytes of fifth bits (qh) before the 4-bit numbers; number l of
// sub-block j takes its fifth bit from bit j of qh[l].
extern "C" __global__ void matvec_q5_k(const u8* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int blocks = cols / 256;
    const u8* r = w + (size_t)row * blocks * 176;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const u8* block = r + b * 176;
        const float d = half_at(block), dmin = half_at(block + 2);
        const u8* scales = block + 4;
        const u8 qh = block[16 + lane];
        const u8* qs = block + 48;
        const float* xb = x + b * 256;
        for (int pair = 0; pair < 4; pair++) {
            float s1, m1, s2, m2;
            q4_k_scale_min(2 * pair, scales, s1, m1);
            q4_k_scale_min(2 * pair + 1, scales, s2, m2);
            const u8 q = qs[pair * 32 + lane];
            const float high1 = (qh >> (2 * pair)) & 1 ? 16.0f : 0.0f;
            const float high2 = (qh >> (2 * pair + 1)) & 1 ? 16.0f : 0.0f;
            sum += (d * s1 * ((float)(q & 0xF) + high1) - dmin * m1) * xb[pair * 64 + lane];
            sum += (d * s2 * ((float)(q >> 4) + high2) - dmin * m2) * xb[pair * 64 + 32 + lane];
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Q6_K: blocks of 256 in 210 bytes (ql 128, qh 64, 16 int8 scales, f16 d). Each half
// of a block is 4 runs of 32; lane l takes number l of each run.
extern "C" __global__ void matvec_q6_k(const u8* w, const float* x, float* y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int blocks = cols / 256;
    const u8* r = w + (size_t)row * blocks * 210;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const u8* block = r + b * 210;
        const float d = half_at(block + 208);
        const float* xb = x + b * 256;
        for (int half = 0; half < 2; half++) {
            const u8* ql = block + half * 64;
            const u8 qh = block[128 + half * 32 + lane];
            const signed char* sc = reinterpret_cast<const signed char*>(block + 192 + half * 8);
            for (int k = 0; k < 4; k++) {
                const u8 low_byte = ql[lane + (k & 1) * 32];
                const int low = k < 2 ? (low_byte & 0xF) : (low_byte >> 4);
                const int high = (qh >> (2 * k)) & 3;
                const float q = (float)((low | (high << 4)) - 32);
                sum += d * (float)sc[lane / 16 + 2 * k] * q * xb[half * 128 + k * 32 + lane];
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ---- Everything else in a layer ----------------------------------------------------

// RMSNorm of rows of n numbers, one block per row (so per head, or per layer slice).
// `weight` (n numbers, the same for every row) may be null: no weight. In place is fine.
extern "C" __global__ void rms_norm(const float* x, const float* weight, float* out, int n, float eps) {
    __shared__ float scratch[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* o = out + (size_t)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i] * xr[i];
    sum = block_sum(sum, scratch);
    const float scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * scale * (weight ? weight[i] : 1.0f);
}

// x / max(|x|, eps) for rows of n numbers, one block per row, in place (Qwen 3.5's q and k).
extern "C" __global__ void l2_norm(float* x, int n, float eps) {
    __shared__ float scratch[32];
    float* xr = x + (size_t)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i] * xr[i];
    sum = block_sum(sum, scratch);
    const float scale = 1.0f / fmaxf(sqrtf(sum), eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= scale;
}

// RoPE on the first `rope_dims` numbers of each head at `pos`: adjacent pairs (ggml's
// NORM, Llama) or halves (`neox`: number j with j + rope_dims/2; Qwen, Gemma).
// `factors` (rope_freqs) may be null.
extern "C" __global__ void rope(float* x, int heads, int head_dim, int rope_dims, int pos, float base,
                                const float* factors, int neox) {
    const int pairs = rope_dims / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * pairs) return;
    const int head = i / pairs, p = i % pairs;
    float theta = (float)pos * powf(base, -2.0f * (float)p / (float)rope_dims);
    if (factors) theta /= factors[p];
    float s, c;
    sincosf(theta, &s, &c);
    float* v = x + head * head_dim;
    const int i0 = neox ? p : 2 * p, i1 = neox ? p + pairs : 2 * p + 1;
    const float a = v[i0], b = v[i1];
    v[i0] = a * c - b * s;
    v[i1] = a * s + b * c;
}

// Attention for one new token: one block per query head, over cached positions
// first..len (first > 0 for a sliding window). `scores` is scratch space, `capacity`
// floats per head.
extern "C" __global__ void attention(const float* q, const float* keys, const float* values, float* out,
                                     float* scores, int first, int len, int head_dim, int kv_dim, int group,
                                     int capacity, float scale) {
    __shared__ float scratch[32];
    const int head = blockIdx.x, kv = (head / group) * head_dim;
    const float* qh = q + head * head_dim;
    float* s = scores + (size_t)head * capacity;

    float local_max = -INFINITY;
    for (int t = first + threadIdx.x; t < len; t += blockDim.x) {
        const float* k = keys + (size_t)t * kv_dim + kv;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * k[d];
        s[t] = dot * scale;
        local_max = fmaxf(local_max, s[t]);
    }
    const float max = block_max(local_max, scratch);
    float local_sum = 0.0f;
    for (int t = first + threadIdx.x; t < len; t += blockDim.x) {
        s[t] = expf(s[t] - max);
        local_sum += s[t];
    }
    const float sum = block_sum(local_sum, scratch);  // also makes every s[t] visible
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = first; t < len; t++) acc += s[t] * values[(size_t)t * kv_dim + kv + d];
        out[head * head_dim + d] = acc / sum;
    }
}

// gate = SiLU(gate) * up
extern "C" __global__ void swiglu(float* gate, const float* up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

// x += y
extern "C" __global__ void add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// gate = GELU(gate) * up, GELU with the tanh approximation (Gemma 4's feed-forward).
extern "C" __global__ void geglu(float* gate, const float* up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = gate[i];
    gate[i] = 0.5f * x * (1.0f + tanhf(0.7978846f * x * (1.0f + 0.044715f * x * x))) * up[i];
}

// x *= s
extern "C" __global__ void scale(float* x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// x = cap * tanh(x / cap) (Gemma's logit softcap)
extern "C" __global__ void softcap(float* x, float cap, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// ---- Qwen 3.5 ------------------------------------------------------------------------

// Gated attention's query: each head's [query | gate] -> just the queries.
extern "C" __global__ void split_query(const float* query_gate, float* q, int heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * head_dim) return;
    const int h = i / head_dim, d = i % head_dim;
    q[i] = query_gate[h * 2 * head_dim + d];
}

// out *= sigmoid(gate), the gate being each head's second half of query_gate.
extern "C" __global__ void apply_gate(float* out, const float* query_gate, int heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * head_dim) return;
    const int h = i / head_dim, d = i % head_dim;
    out[i] *= 1.0f / (1.0f + expf(-query_gate[h * 2 * head_dim + head_dim + d]));
}

// The causal convolution over each channel's last `kernel` values (oldest first), then
// SiLU; the history moves along by one token. One thread per channel.
extern "C" __global__ void conv_silu(const float* current, float* history, const float* weights, float* out,
                                     int channels, int kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* w = weights + (size_t)c * kernel;
    float sum = 0.0f;
    for (int i = 0; i < kernel - 1; i++) sum += history[(size_t)i * channels + c] * w[i];
    const float now = current[c];
    sum += now * w[kernel - 1];
    out[c] = sum / (1.0f + expf(-sum));
    for (int i = 0; i < kernel - 2; i++) history[(size_t)i * channels + c] = history[(size_t)(i + 1) * channels + c];
    history[(size_t)(kernel - 2) * channels + c] = now;
}

// One Gated DeltaNet head's step for one token: one block of n threads per value head
// (n = the state size, at most 256). Thread j owns row j of the head's transposed state
// (state[j*n + i] = S[i][j]), as the CPU reference and llama.cpp store it:
//   S = exp(g) S; delta_j = (v_j - S[:,j].k) beta; S[:,j] += k delta_j; o_j = S[:,j].q / sqrt(n)
// then out = RMSNorm(o) * norm * SiLU(z), where beta = sigmoid(beta_raw) and
// g = softplus(alpha_raw + dt_bias) * a.
extern "C" __global__ void delta_net(const float* q, const float* k, const float* v, const float* beta_raw,
                                     const float* alpha_raw, const float* dt_bias, const float* a,
                                     float* state, const float* z, const float* norm, float* out,
                                     int n, int k_heads, float eps) {
    __shared__ float scratch[32];
    __shared__ float ks[256];
    __shared__ float qs[256];
    const int h = blockIdx.x, j = threadIdx.x, kh = h % k_heads;
    ks[j] = k[kh * n + j];
    qs[j] = q[kh * n + j];
    __syncthreads();
    const float beta = 1.0f / (1.0f + expf(-beta_raw[h]));
    const float x = alpha_raw[h] + dt_bias[h];
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    const float decay = expf(softplus * a[h]);

    float* row = state + ((size_t)h * n + j) * n;
    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        row[i] *= decay;
        dot += row[i] * ks[i];
    }
    const float delta = (v[h * n + j] - dot) * beta;
    float o = 0.0f;
    for (int i = 0; i < n; i++) {
        row[i] += ks[i] * delta;
        o += row[i] * qs[i];
    }
    o *= 1.0f / sqrtf((float)n);

    const float sum = block_sum(o * o, scratch);
    const float normed = o / sqrtf(sum / (float)n + eps) * norm[j];
    const float zj = z[h * n + j];
    out[h * n + j] = normed * (zj / (1.0f + expf(-zj)));
}
