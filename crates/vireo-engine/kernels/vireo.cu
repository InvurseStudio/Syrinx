// Vireo's CUDA kernels: everything one token of a Llama-style model needs.
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

// RMSNorm of n numbers, in one block.
extern "C" __global__ void rms_norm(const float* x, const float* weight, float* out, int n, float eps) {
    __shared__ float scratch[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    sum = block_sum(sum, scratch);
    const float scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * weight[i];
}

// RoPE, adjacent pairs (ggml's NORM), for `heads` heads of `head_dim` at `pos`.
// `factors` (Llama 3.1+'s rope_freqs) may be null.
extern "C" __global__ void rope_norm(float* x, int heads, int head_dim, int pos, float base, const float* factors) {
    const int pairs = head_dim / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * pairs) return;
    const int head = i / pairs, p = i % pairs;
    float theta = (float)pos * powf(base, -2.0f * (float)p / (float)head_dim);
    if (factors) theta /= factors[p];
    float s, c;
    sincosf(theta, &s, &c);
    float* v = x + head * head_dim + 2 * p;
    const float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
}

// Attention for one new token: one block per query head, over `len` cached positions.
// `scores` is scratch space, `capacity` floats per head.
extern "C" __global__ void attention(const float* q, const float* keys, const float* values, float* out,
                                     float* scores, int len, int head_dim, int kv_dim, int group,
                                     int capacity, float scale) {
    __shared__ float scratch[32];
    const int head = blockIdx.x, kv = (head / group) * head_dim;
    const float* qh = q + head * head_dim;
    float* s = scores + (size_t)head * capacity;

    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const float* k = keys + (size_t)t * kv_dim + kv;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * k[d];
        s[t] = dot * scale;
        local_max = fmaxf(local_max, s[t]);
    }
    const float max = block_max(local_max, scratch);
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        s[t] = expf(s[t] - max);
        local_sum += s[t];
    }
    const float sum = block_sum(local_sum, scratch);  // also makes every s[t] visible
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; t++) acc += s[t] * values[(size_t)t * kv_dim + kv + d];
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
