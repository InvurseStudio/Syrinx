// Vireo's Metal kernels: the Mac's port of kernels/vireo.cu, everything Llama, Qwen 3.5 and
// Gemma 4 need, for one token at a time (generating) or many (reading a prompt).
//
// Compiled when the engine opens the GPU (src/metal/mod.rs) and run through Metal. Each
// kernel is the CUDA kernel of the same name, with the same arguments, block layouts and
// arithmetic; only the spelling differs:
//
// - Arguments come as one constant buffer of 8-byte slots, in the CUDA kernel's parameter
//   order: a GPU address (MTLBuffer.gpuAddress plus an offset), or an int or float in the
//   slot's low 4 bytes. ARG_PTR, ARG_INT and ARG_F32 read them.
// - A CUDA block is a threadgroup, a warp a SIMD group (32 threads on Apple GPUs), shared
//   memory threadgroup memory (at most 32 KB).
// - Apple GPUs have no dp4a or CUDA tensor cores, so the 8-bit-activation and tensor-core
//   kernels aren't here: the backend reports no 8-bit kernels, and the models use these.
//
// Kernels that depend on the position read it from GPU memory (`pos`), so one token's
// dispatches are the same every time and can be recorded once and replayed.

#include <metal_stdlib>
using namespace metal;

typedef uchar u8;
typedef ushort u16;
typedef uint u32;

#define ARGS constant ulong* A [[buffer(0)]]
#define KP                                                                       \
    uint3 gid [[threadgroup_position_in_grid]], uint3 ngrid [[threadgroups_per_grid]], \
        uint tid [[thread_index_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]
#define ARG_PTR(T, i) reinterpret_cast<device T*>(A[i])
#define ARG_INT(i) (as_type<int2>(A[i]).x)
#define ARG_F32(i) (as_type<float2>(A[i]).x)
#define NONNULL(p) ((ulong)(p) != 0)

#define BARRIER threadgroup_barrier(mem_flags::mem_threadgroup)

static inline float half_at(device const u8* p) { return float(*reinterpret_cast<device const half*>(p)); }

// Four bytes from an address that's only 2-byte aligned.
static inline u32 u32_at2(device const u8* p) {
    device const u16* h = reinterpret_cast<device const u16*>(p);
    return (u32)h[0] | ((u32)h[1] << 16);
}

static inline float4 f4(device const float* p) { return *reinterpret_cast<device const float4*>(p); }

// The sum of `v` over the threadgroup, returned to every thread. `scratch` holds 32 floats.
static inline float block_sum(float v, threadgroup float* scratch, uint tid, uint threads) {
    const uint lane = tid & 31, warp = tid >> 5;
    v = simd_sum(v);
    BARRIER;  // scratch may still be in use from a previous call
    if (lane == 0) scratch[warp] = v;
    BARRIER;
    v = tid < (threads + 31) / 32 ? scratch[tid] : 0.0f;
    if (warp == 0) v = simd_sum(v);
    if (tid == 0) scratch[0] = v;
    BARRIER;
    return scratch[0];
}

static inline float block_max(float v, threadgroup float* scratch, uint tid, uint threads) {
    const uint lane = tid & 31, warp = tid >> 5;
    v = simd_max(v);
    BARRIER;
    if (lane == 0) scratch[warp] = v;
    BARRIER;
    v = tid < (threads + 31) / 32 ? scratch[tid] : -INFINITY;
    if (warp == 0) v = simd_max(v);
    if (tid == 0) scratch[0] = v;
    BARRIER;
    return scratch[0];
}

// log(1 + x), accurate for small x (MSL has no log1p).
static inline float log1p_(float x) {
    const float u = 1.0f + x;
    return u == 1.0f ? x : log(u) * x / (u - 1.0f);
}

static inline float softplus_(float x) { return x > 20.0f ? x : log1p_(exp(x)); }

// Q4_K's 6-bit scale and minimum for sub-block j.
static inline void q4_k_scale_min(int j, device const u8* q, thread float& scale, thread float& mn) {
    if (j < 4) {
        scale = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_0 on the GPU: each row's 16-byte runs of 4-bit numbers first, then its f16 scales,
// padded to 16 bytes (as in vireo.cu; Gpu::upload_weights repacks GGUF's blocks so).
static inline ulong q4_0_row_bytes(int cols) {
    const int blocks = cols / 32;
    return (ulong)blocks * 16 + ((blocks * 2 + 15) & ~15);
}

static inline float q4_0_scale(device const u8* row, int blocks, int b) {
    return half_at(row + (ulong)blocks * 16 + b * 2);
}

// Q6_K's 210-byte blocks padded to 224 on the GPU.
#define Q6K_BYTES 224

// ---- Matrix-vector products: y = W x, one SIMD group per row of W -------------------

#define MATVEC_ROW                                         \
    const int row = (gid.x * tpg.x + tid) >> 5, lane = tid & 31; \
    if (row >= rows) return;

kernel void matvec_f32(ARGS, KP) {
    device const float* w = ARG_PTR(const float, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4);
    MATVEC_ROW
    device const float* r = w + (ulong)row * cols;
    float sum = 0.0f;
    if ((cols & 3) == 0) {
        for (int c = lane * 4; c < cols; c += 128) {
            const float4 a = f4(r + c), b = f4(x + c);
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int c = lane; c < cols; c += 32) sum += r[c] * x[c];
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}

kernel void matvec_f16(ARGS, KP) {
    device const half* w = ARG_PTR(const half, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4);
    MATVEC_ROW
    device const half* r = w + (ulong)row * cols;
    float sum = 0.0f;
    if ((cols & 7) == 0) {
        for (int c = lane * 8; c < cols; c += 256) {
            const half4 h0 = *reinterpret_cast<device const half4*>(r + c);
            const half4 h1 = *reinterpret_cast<device const half4*>(r + c + 4);
            const float4 x0 = f4(x + c), x1 = f4(x + c + 4);
            const float4 a = float4(h0), b = float4(h1);
            sum += a.x * x0.x + a.y * x0.y + a.z * x0.z + a.w * x0.w;
            sum += b.x * x1.x + b.y * x1.y + b.z * x1.z + b.w * x1.w;
        }
    } else {
        for (int c = lane; c < cols; c += 32) sum += float(r[c]) * x[c];
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}

// A small whole number (under 2^23) as a float, minus `offset`, exactly.
static inline float small_int(u32 v, float offset) {
    return as_type<float>(0x4B000000u | v) - (8388608.0f + offset);
}

#define MV_ROWS 2
#define MATVEC_ROWS                                                \
    const int warp = (gid.x * tpg.x + tid) >> 5, lane = tid & 31; \
    const int row0 = warp * MV_ROWS;                               \
    if (row0 >= rows) return;                                      \
    const int nrows = min(MV_ROWS, rows - row0);                   \
    float sum[MV_ROWS];                                            \
    for (int r = 0; r < MV_ROWS; r++) sum[r] = 0.0f;

#define MATVEC_STORE                                     \
    for (int r = 0; r < MV_ROWS; r++) {                  \
        const float total = simd_sum(sum[r]);            \
        if (lane == 0 && r < nrows) y[row0 + r] = total; \
    }

static inline float half_lo(u32 v) { return float(as_type<half>((ushort)(v & 0xFFFF))); }
static inline float half_hi(u32 v) { return float(as_type<half>((ushort)(v >> 16))); }

// Σ nibble_i × x_i over 8 nibbles of `q` (the low or high nibbles of 8 bytes).
static inline float nibble_dot8(uint2 q, int shift, float4 a, float4 b) {
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu, hi = (q.y >> shift) & 0x0F0F0F0Fu;
    return small_int(lo & 0xFF, 0.0f) * a.x + small_int((lo >> 8) & 0xFF, 0.0f) * a.y +
           small_int((lo >> 16) & 0xFF, 0.0f) * a.z + small_int(lo >> 24, 0.0f) * a.w +
           small_int(hi & 0xFF, 0.0f) * b.x + small_int((hi >> 8) & 0xFF, 0.0f) * b.y +
           small_int((hi >> 16) & 0xFF, 0.0f) * b.z + small_int(hi >> 24, 0.0f) * b.w;
}

// Σ bit_i × x_i over the bits at `shift` of 8 bytes.
static inline float bit_dot8(uint2 q, int shift, float4 a, float4 b) {
    const u32 lo = (q.x >> shift) & 0x01010101u, hi = (q.y >> shift) & 0x01010101u;
    return small_int(lo & 1, 0.0f) * a.x + small_int((lo >> 8) & 1, 0.0f) * a.y +
           small_int((lo >> 16) & 1, 0.0f) * a.z + small_int(lo >> 24, 0.0f) * a.w + small_int(hi & 1, 0.0f) * b.x +
           small_int((hi >> 8) & 1, 0.0f) * b.y + small_int((hi >> 16) & 1, 0.0f) * b.z +
           small_int(hi >> 24, 0.0f) * b.w;
}

#define MATVEC_ARGS                               \
    device const u8* w = ARG_PTR(const u8, 0);    \
    device const float* x = ARG_PTR(const float, 1); \
    device float* y = ARG_PTR(float, 2);          \
    const int rows = ARG_INT(3), cols = ARG_INT(4);

kernel void matvec_q8_0(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS
    const int blocks = cols / 32;
    const ulong row_bytes = (ulong)blocks * 34;
    const int sub = lane >> 3, j = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 4) {
        const float4 xv = f4(x + b * 32 + j);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * 34;
            const u32 q = u32_at2(block + 2 + j);
            const float dot = small_int((q & 0xFF) ^ 0x80, 128.0f) * xv.x +
                              small_int(((q >> 8) & 0xFF) ^ 0x80, 128.0f) * xv.y +
                              small_int(((q >> 16) & 0xFF) ^ 0x80, 128.0f) * xv.z +
                              small_int((q >> 24) ^ 0x80, 128.0f) * xv.w;
            sum[r] += half_at(block) * dot;
        }
    }
    MATVEC_STORE
}

kernel void matvec_q4_0(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub; b < blocks; b += 16) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + j);
            const float dot = nibble_dot8(q, 0, lo0, lo1) + nibble_dot8(q, 4, hi0, hi1) -
                              8.0f * ((lo0.x + lo0.y) + (lo0.z + lo0.w) + (lo1.x + lo1.y) + (lo1.z + lo1.w) +
                                      (hi0.x + hi0.y) + (hi0.z + hi0.w) + (hi1.x + hi1.y) + (hi1.z + hi1.w));
            sum[r] += q4_0_scale(row, blocks, b) * dot;
        }
    }
    MATVEC_STORE
}

static inline void q4_k_scale_min_w(int j, u32 s0, u32 s1, u32 s2, thread float& scale, thread float& mn) {
    const int at = (j & 3) * 8;
    const u32 b0 = (s0 >> at) & 0xFF, b1 = (s1 >> at) & 0xFF, b2 = (s2 >> at) & 0xFF;
    if (j < 4) {
        scale = b0 & 63;
        mn = b1 & 63;
    } else {
        scale = (b2 & 0xF) | ((b0 >> 6) << 4);
        mn = (b2 >> 4) | ((b1 >> 6) << 4);
    }
}

kernel void matvec_q4_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 144;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        device const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * 144;
            const uint4 head = *reinterpret_cast<device const uint4*>(block);
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<device const uint2*>(block + 16 + pair * 32 + part);
            sum[r] += d * s1 * nibble_dot8(q, 0, a1, b1) - dmin * m1 * x1 + d * s2 * nibble_dot8(q, 4, a2, b2) -
                      dmin * m2 * x2;
        }
    }
    MATVEC_STORE
}

kernel void matvec_q5_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 176;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        device const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * 176;
            const uint4 head = *reinterpret_cast<device const uint4*>(block);
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<device const uint2*>(block + 48 + pair * 32 + part);
            const uint2 qh = *reinterpret_cast<device const uint2*>(block + 16 + part);
            const float dot1 = nibble_dot8(q, 0, a1, b1) + 16.0f * bit_dot8(qh, 2 * pair, a1, b1);
            const float dot2 = nibble_dot8(q, 4, a2, b2) + 16.0f * bit_dot8(qh, 2 * pair + 1, a2, b2);
            sum[r] += d * s1 * dot1 - dmin * m1 * x1 + d * s2 * dot2 - dmin * m2 * x2;
        }
    }
    MATVEC_STORE
}

kernel void matvec_q6_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * Q6K_BYTES;
    const int sub = lane >> 4, hf = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 2) {
        device const float* xb = x + b * 256 + hf * 128 + l;
        float4 xv[4];
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at2(block + hf * 64 + l), ql1 = u32_at2(block + hf * 64 + 32 + l);
            const u32 qh = u32_at2(block + 128 + hf * 32 + l);
            device const char* sc = reinterpret_cast<device const char*>(block + 192 + hf * 8 + l / 16);
            float bsum = 0.0f;
            for (int k = 0; k < 4; k++) {
                const u32 low = ((k & 1) ? ql1 : ql0) >> (k < 2 ? 0 : 4);
                const float q0 = small_int((low & 0xF) | (((qh >> (2 * k)) & 3) << 4), 32.0f);
                const float q1 = small_int(((low >> 8) & 0xF) | (((qh >> (8 + 2 * k)) & 3) << 4), 32.0f);
                const float q2 = small_int(((low >> 16) & 0xF) | (((qh >> (16 + 2 * k)) & 3) << 4), 32.0f);
                const float q3 = small_int(((low >> 24) & 0xF) | (((qh >> (24 + 2 * k)) & 3) << 4), 32.0f);
                bsum += (float)sc[2 * k] * (q0 * xv[k].x + q1 * xv[k].y + q2 * xv[k].z + q3 * xv[k].w);
            }
            sum[r] += d * bsum;
        }
    }
    MATVEC_STORE
}

// ---- Matrix-matrix products: Y = X Wᵀ for many tokens at once ------------------------

enum { K_F32, K_F16, K_Q4_0, K_Q8_0, K_Q4_K, K_Q5_K, K_Q6_K };

// Eight numbers of a row, from column k (a multiple of 8), decoded as quant.rs does.
template <int KIND>
static inline void decode8(device const u8* row, int blocks, int k, thread float* v);

template <>
inline void decode8<K_F32>(device const u8* row, int blocks, int k, thread float* v) {
    device const float* r = reinterpret_cast<device const float*>(row) + k;
    for (int i = 0; i < 8; i++) v[i] = r[i];
}

template <>
inline void decode8<K_F16>(device const u8* row, int blocks, int k, thread float* v) {
    device const half* r = reinterpret_cast<device const half*>(row) + k;
    for (int i = 0; i < 8; i++) v[i] = float(r[i]);
}

template <>
inline void decode8<K_Q4_0>(device const u8* row, int blocks, int k, thread float* v) {
    const int b = k / 32, j = k % 32;
    const float d = q4_0_scale(row, blocks, b);
    const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + (j & 15));
    const int shift = j < 16 ? 0 : 4;
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu, hi = (q.y >> shift) & 0x0F0F0F0Fu;
    for (int i = 0; i < 4; i++) {
        v[i] = d * (float)((int)((lo >> (8 * i)) & 0xF) - 8);
        v[4 + i] = d * (float)((int)((hi >> (8 * i)) & 0xF) - 8);
    }
}

template <>
inline void decode8<K_Q8_0>(device const u8* row, int blocks, int k, thread float* v) {
    device const u8* block = row + (k / 32) * 34;
    const float d = half_at(block);
    const int j = k % 32;
    for (int i = 0; i < 8; i++) v[i] = d * (float)(char)block[2 + j + i];
}

template <>
inline void decode8<K_Q4_K>(device const u8* row, int blocks, int k, thread float* v) {
    device const u8* block = row + (k / 256) * 144;
    const int n = k % 256, j = n / 32, l = n % 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    device const u8* q = block + 16 + (j / 2) * 32 + l;
    const int shift = (j & 1) * 4;
    for (int i = 0; i < 8; i++) v[i] = d1 * (float)((q[i] >> shift) & 0xF) - min1;
}

template <>
inline void decode8<K_Q5_K>(device const u8* row, int blocks, int k, thread float* v) {
    device const u8* block = row + (k / 256) * 176;
    const int n = k % 256, j = n / 32, l = n % 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    device const u8* q = block + 48 + (j / 2) * 32 + l;
    device const u8* qh = block + 16 + l;
    const int shift = (j & 1) * 4;
    for (int i = 0; i < 8; i++) {
        const float high = (qh[i] >> j) & 1 ? 16.0f : 0.0f;
        v[i] = d1 * ((float)((q[i] >> shift) & 0xF) + high) - min1;
    }
}

template <>
inline void decode8<K_Q6_K>(device const u8* row, int blocks, int k, thread float* v) {
    device const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, hf = n / 128, run = (n % 128) / 32, l = n % 32;
    const float d = half_at(block + 208);
    device const u8* ql = block + hf * 64 + (run & 1) * 32 + l;
    device const u8* qh = block + 128 + hf * 32 + l;
    device const char* sc = reinterpret_cast<device const char*>(block + 192 + hf * 8 + l / 16 + 2 * run);
    const float ds = d * (float)sc[0];
    for (int i = 0; i < 8; i++) {
        const int low = run < 2 ? (ql[i] & 0xF) : (ql[i] >> 4);
        const int q = (low | (((qh[i] >> (2 * run)) & 3) << 4)) - 32;
        v[i] = ds * (float)q;
    }
}

template <int KIND>
static inline ulong mm_row_bytes(int cols) {
    switch (KIND) {
        case K_F32: return (ulong)cols * 4;
        case K_F16: return (ulong)cols * 2;
        case K_Q4_0: return q4_0_row_bytes(cols);
        case K_Q8_0: return (ulong)(cols / 32) * 34;
        case K_Q4_K: return (ulong)(cols / 256) * 144;
        case K_Q5_K: return (ulong)(cols / 256) * 176;
        default: return (ulong)(cols / 256) * Q6K_BYTES;
    }
}

#define MM_ROWS 128
#define MM_TOKENS 64
#define MM_K 32

// Tiles of 128 rows × 64 tokens; each of the 256 threads works out 8 rows × 4 tokens.
template <int KIND>
kernel void matmul(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    threadgroup float ws[MM_K][MM_ROWS + 4];
    threadgroup float xs[MM_K][MM_TOKENS + 4];
    const int row0 = gid.x * MM_ROWS, tok0 = gid.y * MM_TOKENS;
    const int load_row = tid >> 2, load_k = (tid & 3) * 8;
    const int ty = tid >> 4, tx = tid & 15;
    float acc[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += MM_K) {
        for (int hf = 0; hf < 2; hf++) {
            float v[8];
            const int lr = load_row + hf * 64, r = row0 + lr;
            if (r < rows) {
                decode8<KIND>(w + (ulong)r * row_bytes, cols / 32, k0 + load_k, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) ws[load_k + i][lr] = v[i];
        }
        const int t = tok0 + load_row;
        if (t < tokens) {
            device const float* xr = x + (ulong)t * cols + k0 + load_k;
            const float4 a = f4(xr), b = f4(xr + 4);
            xs[load_k + 0][load_row] = a.x;
            xs[load_k + 1][load_row] = a.y;
            xs[load_k + 2][load_row] = a.z;
            xs[load_k + 3][load_row] = a.w;
            xs[load_k + 4][load_row] = b.x;
            xs[load_k + 5][load_row] = b.y;
            xs[load_k + 6][load_row] = b.z;
            xs[load_k + 7][load_row] = b.w;
        } else {
            for (int i = 0; i < 8; i++) xs[load_k + i][load_row] = 0.0f;
        }
        BARRIER;
        for (int k = 0; k < MM_K; k++) {
            const float4 a0 = *reinterpret_cast<threadgroup const float4*>(&ws[k][ty * 4]);
            const float4 a1 = *reinterpret_cast<threadgroup const float4*>(&ws[k][64 + ty * 4]);
            const float4 b = *reinterpret_cast<threadgroup const float4*>(&xs[k][tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
        }
        BARRIER;
    }
    for (int j = 0; j < 4; j++) {
        const int t = tok0 + tx * 4 + j;
        for (int i = 0; i < 8; i++) {
            const int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
            if (t < tokens && r < rows) y[(ulong)t * rows + r] = acc[i][j];
        }
    }
}

template [[host_name("matmul_f32")]] kernel void matmul<K_F32>(ARGS, KP);
template [[host_name("matmul_f16")]] kernel void matmul<K_F16>(ARGS, KP);
template [[host_name("matmul_q4_0")]] kernel void matmul<K_Q4_0>(ARGS, KP);
template [[host_name("matmul_q8_0")]] kernel void matmul<K_Q8_0>(ARGS, KP);
template [[host_name("matmul_q4_k")]] kernel void matmul<K_Q4_K>(ARGS, KP);
template [[host_name("matmul_q5_k")]] kernel void matmul<K_Q5_K>(ARGS, KP);
template [[host_name("matmul_q6_k")]] kernel void matmul<K_Q6_K>(ARGS, KP);

// ---- Norms, RoPE and the residual stream ---------------------------------------------

kernel void rms_norm(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device const float* weight = ARG_PTR(const float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n = ARG_INT(3);
    const float eps = ARG_F32(4);
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)gid.x * n;
    device float* o = out + (ulong)gid.x * n;
    const bool has_weight = NONNULL(weight);
    float sum = 0.0f;
    for (int i = tid; i < n; i += tpg.x) sum += xr[i] * xr[i];
    sum = block_sum(sum, scratch, tid, tpg.x);
    const float scale = 1.0f / sqrt(sum / (float)n + eps);
    for (int i = tid; i < n; i += tpg.x) o[i] = xr[i] * scale * (has_weight ? weight[i] : 1.0f);
}

kernel void residual_norm(ARGS, KP) {
    device float* x = ARG_PTR(float, 0);
    device const float* p = ARG_PTR(const float, 1);
    device const float* post = ARG_PTR(const float, 2);
    const float scale = ARG_F32(3);
    device const float* next = ARG_PTR(const float, 4);
    device float* h = ARG_PTR(float, 5);
    const int n = ARG_INT(6);
    const float eps = ARG_F32(7);
    threadgroup float scratch[32];
    const bool has_p = NONNULL(p), has_post = NONNULL(post), has_next = NONNULL(next);
    x += (ulong)gid.x * n;
    float p_scale = 1.0f;
    if (has_p) {
        p += (ulong)gid.x * n;
        if (has_post) {
            float sum = 0.0f;
            for (int i = tid; i < n; i += tpg.x) sum += p[i] * p[i];
            sum = block_sum(sum, scratch, tid, tpg.x);
            p_scale = 1.0f / sqrt(sum / (float)n + eps);
        }
    }
    float sum = 0.0f;
    for (int i = tid; i < n; i += tpg.x) {
        float v = x[i];
        if (has_p) v += has_post ? p[i] * p_scale * post[i] : p[i];
        if (scale != 1.0f) v *= scale;
        x[i] = v;
        sum += v * v;
    }
    if (!has_next) return;
    sum = block_sum(sum, scratch, tid, tpg.x);
    const float s = 1.0f / sqrt(sum / (float)n + eps);
    h += (ulong)gid.x * n;
    for (int i = tid; i < n; i += tpg.x) h[i] = x[i] * s * next[i];
}

// See qkv_prep in vireo.cu.
#define QKV_TOKENS 8
kernel void qkv_prep(ARGS, KP) {
    device const float* q_src = ARG_PTR(const float, 0);
    const int q_stride = ARG_INT(1), q_head_stride = ARG_INT(2);
    device const float* q_norm = ARG_PTR(const float, 3);
    device float* q_dst = ARG_PTR(float, 4);
    device const float* k_src = ARG_PTR(const float, 5);
    device const float* k_norm = ARG_PTR(const float, 6);
    device const float* v_src = ARG_PTR(const float, 7);
    const int v_norm = ARG_INT(8);
    const ulong k_cache = A[9], v_cache = A[10];
    const int heads = ARG_INT(11), kv_heads = ARG_INT(12), hd = ARG_INT(13), rope_dims = ARG_INT(14);
    const float base = ARG_F32(15);
    device const float* factors = ARG_PTR(const float, 16);
    const int neox = ARG_INT(17);
    device const int* pos = ARG_PTR(const int, 18);
    const float eps = ARG_F32(19);
    const int half_cache = ARG_INT(20), tokens = ARG_INT(21);

    threadgroup float scratch[32];
    threadgroup float buf[512];
    const int slot = gid.x, kv_dim = kv_heads * hd;
    const int t0 = gid.y * QKV_TOKENS, t1 = min(t0 + QKV_TOKENS, tokens);
    const int kind = slot < heads ? 0 : slot < heads + kv_heads ? 1 : 2;  // q, k, v
    const int h = kind == 0 ? slot : kind == 1 ? slot - heads : slot - heads - kv_heads;
    device const float* weight = kind == 0 ? q_norm : kind == 1 ? k_norm : (device const float*)nullptr;
    const bool has_weight = NONNULL(weight);
    const bool norm = kind == 2 ? v_norm != 0 : has_weight;
    const bool rope = kind != 2;
    const bool has_factors = NONNULL(factors);

    const int pairs = rope ? rope_dims / 2 : 0, rotated = 2 * pairs;
    const int items = pairs + hd - rotated;
    float freq = 0.0f;
    if ((int)tid < pairs) freq = pow(base, -2.0f * (float)tid / (float)rope_dims);

    for (int t = t0; t < t1; t++) {
        const int at = *pos + t;
        device const float* src;
        device float* dst = nullptr;
        device half* half_dst = nullptr;
        if (kind == 0) {
            src = q_src + (ulong)t * q_stride + (ulong)h * q_head_stride;
            dst = q_dst + ((ulong)t * heads + h) * hd;
        } else {
            src = (kind == 1 ? k_src : v_src) + (ulong)t * kv_dim + h * hd;
            const ulong at_cache = (ulong)at * kv_dim + h * hd;
            const ulong cache = kind == 1 ? k_cache : v_cache;
            if (half_cache)
                half_dst = reinterpret_cast<device half*>(cache) + at_cache;
            else
                dst = reinterpret_cast<device float*>(cache) + at_cache;
        }
        const bool to_half = half_cache && kind != 0;
        BARRIER;  // the last token is done with buf
        float sum = 0.0f;
        for (int i = tid; i < hd; i += tpg.x) {
            buf[i] = src[i];
            sum += buf[i] * buf[i];
        }
        if (norm) {
            sum = block_sum(sum, scratch, tid, tpg.x);
            const float scale = 1.0f / sqrt(sum / (float)hd + eps);
            for (int i = tid; i < hd; i += tpg.x) buf[i] = buf[i] * scale * (has_weight ? weight[i] : 1.0f);
        }
        BARRIER;
        for (int j = tid; j < items; j += tpg.x) {
            if (j >= pairs) {
                const int i = rotated + (j - pairs);
                if (to_half) half_dst[i] = half(buf[i]); else dst[i] = buf[i];
                continue;
            }
            const int p = j;
            const int i0 = neox ? p : 2 * p, i1 = neox ? p + pairs : 2 * p + 1;
            float theta = (float)at * freq;
            if (has_factors) theta /= factors[p];
            float c;
            const float s = sincos(theta, c);
            const float a = buf[i0], b = buf[i1];
            const float r0 = a * c - b * s, r1 = a * s + b * c;
            if (to_half) {
                half_dst[i0] = half(r0);
                half_dst[i1] = half(r1);
            } else {
                dst[i0] = r0;
                dst[i1] = r1;
            }
        }
    }
}

// ---- Attention -----------------------------------------------------------------------

static inline float4 load4(device const float* p) { return f4(p); }
static inline float4 load4(device const half* p) { return float4(*reinterpret_cast<device const half4*>(p)); }
static inline float load1(device const float* p) { return *p; }
static inline float load1(device const half* p) { return float(*p); }

// One threadgroup of 128 per query head and token (see attention_one in vireo.cu).
template <typename KV>
kernel void attention_one(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5), hd = ARG_INT(6), kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    threadgroup float scratch[32];
    threadgroup float qs[512];
    threadgroup float ps[128];
    const int head = gid.x, heads = ngrid.x, t = gid.y;
    const int len = *pos + t + 1;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int kv = (head / group) * hd;
    device const float* qh = q + ((ulong)t * heads + head) * hd;
    for (int d = tid; d < hd; d += tpg.x) qs[d] = qh[d];
    BARRIER;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_so_far = -INFINITY, total = 0.0f;
    for (int start = first; start < len; start += 128) {
        const int at = start + tid;
        float s = -INFINITY;
        if (at < len) {
            device const KV* k = keys + (ulong)at * kv_dim + kv;
            float dot = 0.0f;
            for (int d = 0; d < hd; d += 4) {
                const float4 kv4 = load4(k + d);
                const float4 q4 = *reinterpret_cast<threadgroup const float4*>(qs + d);
                dot += q4.x * kv4.x;
                dot += q4.y * kv4.y;
                dot += q4.z * kv4.z;
                dot += q4.w * kv4.w;
            }
            s = dot * scale;
        }
        const float new_max = max(max_so_far, block_max(s, scratch, tid, tpg.x));
        const float p = at < len ? exp(s - new_max) : 0.0f;
        ps[tid] = p;
        const float correction = exp(max_so_far - new_max);
        total = total * correction + block_sum(p, scratch, tid, tpg.x);
        const int count = min(128, len - start);
        for (int j = 0; j < 4; j++) {
            const int d = tid + j * 128;
            if (d >= hd) break;
            float a = acc[j] * correction;
            device const KV* v = values + (ulong)start * kv_dim + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + (ulong)i * kv_dim);
            acc[j] = a;
        }
        max_so_far = new_max;
    }
    device float* o = out + ((ulong)t * heads + head) * hd;
    for (int j = 0; j < 4; j++) {
        const int d = tid + j * 128;
        if (d < hd) o[d] = acc[j] / total;
    }
}

template [[host_name("attention")]] kernel void attention_one<float>(ARGS, KP);
template [[host_name("attention_h")]] kernel void attention_one<half>(ARGS, KP);

// One new token's attention split along the context (see attention_part in vireo.cu).
template <typename KV>
kernel void attention_part(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* partial = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5), hd = ARG_INT(6), kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    threadgroup float scratch[32];
    threadgroup float qs[512];
    threadgroup float ps[128];
    const int head = gid.x, part = gid.y, parts = ngrid.y;
    const int len = *pos + 1;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    const int kv = (head / group) * hd;
    for (int d = tid; d < hd; d += tpg.x) qs[d] = q[head * hd + d];
    BARRIER;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_so_far = -INFINITY, total = 0.0f;
    for (int start = lo; start < hi; start += 128) {
        const int at = start + tid;
        float s = -INFINITY;
        if (at < hi) {
            device const KV* k = keys + (ulong)at * kv_dim + kv;
            float dot = 0.0f;
            for (int d = 0; d < hd; d += 4) {
                const float4 kv4 = load4(k + d);
                const float4 q4 = *reinterpret_cast<threadgroup const float4*>(qs + d);
                dot += q4.x * kv4.x;
                dot += q4.y * kv4.y;
                dot += q4.z * kv4.z;
                dot += q4.w * kv4.w;
            }
            s = dot * scale;
        }
        const float new_max = max(max_so_far, block_max(s, scratch, tid, tpg.x));
        const float p = at < hi ? exp(s - new_max) : 0.0f;
        ps[tid] = p;
        const float correction = exp(max_so_far - new_max);
        total = total * correction + block_sum(p, scratch, tid, tpg.x);
        const int count = min(128, hi - start);
        for (int j = 0; j < 4; j++) {
            const int d = tid + j * 128;
            if (d >= hd) break;
            float a = acc[j] * correction;
            device const KV* v = values + (ulong)start * kv_dim + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + (ulong)i * kv_dim);
            acc[j] = a;
        }
        max_so_far = new_max;
    }
    device float* o = partial + ((ulong)head * parts + part) * (2 + hd);
    if (tid == 0) {
        o[0] = max_so_far;
        o[1] = total;
    }
    for (int j = 0; j < 4; j++) {
        const int d = tid + j * 128;
        if (d < hd) o[2 + d] = acc[j];
    }
}

template [[host_name("attention_part_f32")]] kernel void attention_part<float>(ARGS, KP);
template [[host_name("attention_part_f16")]] kernel void attention_part<half>(ARGS, KP);

kernel void attention_combine(ARGS, KP) {
    device const float* partial = ARG_PTR(const float, 0);
    device float* out = ARG_PTR(float, 1);
    const int parts = ARG_INT(2), hd = ARG_INT(3);
    const int head = gid.x;
    device const float* p = partial + (ulong)head * parts * (2 + hd);
    float mx = -INFINITY;
    for (int i = 0; i < parts; i++) mx = max(mx, p[i * (2 + hd)]);
    float total = 0.0f;
    for (int i = 0; i < parts; i++) {
        const float m = p[i * (2 + hd)];
        if (m != -INFINITY) total += exp(m - mx) * p[i * (2 + hd) + 1];
    }
    for (int d = tid; d < hd; d += tpg.x) {
        float acc = 0.0f;
        for (int i = 0; i < parts; i++) {
            const float m = p[i * (2 + hd)];
            if (m != -INFINITY) acc += exp(m - mx) * p[i * (2 + hd) + 2 + d];
        }
        out[head * hd + d] = acc / total;
    }
}

// A prompt's attention, tiled (see attention_tiled in vireo.cu).
template <int LANES, int PER, int CHUNK>
kernel void attention_tiled(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const float* keys = ARG_PTR(const float, 1);
    device const float* values = ARG_PTR(const float, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5), kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const int tokens = ARG_INT(10);
    constexpr int HD = LANES * PER, ROWS = 256 / LANES;
    threadgroup float ks[CHUNK][HD];
    threadgroup float vs[CHUNK][HD];
    const int head = gid.x, heads = ngrid.x;
    const int tq = tid / LANES, sub = tid % LANES;
    const int t0 = gid.y * ROWS, p0 = *pos;
    const int t = t0 + tq;
    const bool live = t < tokens;
    const int at = p0 + t;
    const int last_row = p0 + min(t0 + ROWS, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;
    const int row_first = window > 0 ? max(0, at + 1 - window) : 0;
    const int kv = (head / group) * HD;

    float qr[PER], acc[PER];
    device const float* qh = q + ((ulong)min(t, tokens - 1) * heads + head) * HD;
    for (int j = 0; j < PER; j++) {
        qr[j] = qh[sub * PER + j];
        acc[j] = 0.0f;
    }
    float max_so_far = -INFINITY, total = 0.0f;

    for (int start = lo; start <= last_row; start += CHUNK) {
        const int count = min(CHUNK, last_row + 1 - start);
        for (int i = tid; i < CHUNK * HD / 4; i += 256) {
            const int r = (i * 4) / HD, d = (i * 4) % HD;
            float4 kv4 = float4(0.0f), vv4 = float4(0.0f);
            if (r < count) {
                kv4 = f4(keys + (ulong)(start + r) * kv_dim + kv + d);
                vv4 = f4(values + (ulong)(start + r) * kv_dim + kv + d);
            }
            *reinterpret_cast<threadgroup float4*>(&ks[r][d]) = kv4;
            *reinterpret_cast<threadgroup float4*>(&vs[r][d]) = vv4;
        }
        BARRIER;

        float s[CHUNK];
        float chunk_max = -INFINITY;
        for (int k = 0; k < CHUNK; k++) {
            float part = 0.0f;
            for (int j = 0; j < PER; j += 4) {
                const float4 kk = *reinterpret_cast<threadgroup const float4*>(&ks[k][sub * PER + j]);
                part += qr[j] * kk.x;
                part += qr[j + 1] * kk.y;
                part += qr[j + 2] * kk.z;
                part += qr[j + 3] * kk.w;
            }
            for (int o = LANES / 2; o > 0; o >>= 1) part += simd_shuffle_xor(part, (ushort)o);
            const int kpos = start + k;
            const bool seen = live && k < count && kpos <= at && kpos >= row_first;
            s[k] = seen ? part * scale : -INFINITY;
            chunk_max = max(chunk_max, s[k]);
        }
        const float new_max = max(max_so_far, chunk_max);
        if (new_max != -INFINITY) {
            const float correction = exp(max_so_far - new_max);
            float chunk_sum = 0.0f;
            for (int k = 0; k < CHUNK; k++) {
                s[k] = s[k] == -INFINITY ? 0.0f : exp(s[k] - new_max);
                chunk_sum += s[k];
            }
            total = total * correction + chunk_sum;
            max_so_far = new_max;
            for (int j = 0; j < PER; j++) acc[j] *= correction;
            for (int k = 0; k < CHUNK; k++) {
                for (int j = 0; j < PER; j += 4) {
                    const float4 v = *reinterpret_cast<threadgroup const float4*>(&vs[k][sub * PER + j]);
                    acc[j] += s[k] * v.x;
                    acc[j + 1] += s[k] * v.y;
                    acc[j + 2] += s[k] * v.z;
                    acc[j + 3] += s[k] * v.w;
                }
            }
        }
        BARRIER;
    }
    if (!live) return;
    device float* o = out + ((ulong)t * heads + head) * HD;
    for (int j = 0; j < PER; j++) o[sub * PER + j] = acc[j] / total;
}

template [[host_name("attention_tiled_64")]] kernel void attention_tiled<8, 8, 32>(ARGS, KP);
template [[host_name("attention_tiled_128")]] kernel void attention_tiled<8, 16, 32>(ARGS, KP);
template [[host_name("attention_tiled_256")]] kernel void attention_tiled<16, 16, 16>(ARGS, KP);
template [[host_name("attention_tiled_512")]] kernel void attention_tiled<32, 16, 8>(ARGS, KP);

// ---- Element-wise steps --------------------------------------------------------------

#define PER_NUMBER const int i = gid.x * tpg.x + tid;

kernel void swiglu(ARGS, KP) {
    device float* gate = ARG_PTR(float, 0);
    device const float* up = ARG_PTR(const float, 1);
    const int n = ARG_INT(2), up_stride = ARG_INT(3), total = ARG_INT(4);
    PER_NUMBER
    if (i >= total) return;
    const float u = up[(ulong)(i / n) * up_stride + i % n];
    gate[i] = gate[i] / (1.0f + exp(-gate[i])) * u;
}

kernel void geglu(ARGS, KP) {
    device float* gate = ARG_PTR(float, 0);
    device const float* up = ARG_PTR(const float, 1);
    const int n = ARG_INT(2), up_stride = ARG_INT(3), total = ARG_INT(4);
    PER_NUMBER
    if (i >= total) return;
    const float x = gate[i];
    const float u = up[(ulong)(i / n) * up_stride + i % n];
    gate[i] = 0.5f * x * (1.0f + tanh(0.7978846f * x * (1.0f + 0.044715f * x * x))) * u;
}

kernel void add(ARGS, KP) {
    device float* x = ARG_PTR(float, 0);
    device const float* y = ARG_PTR(const float, 1);
    const int n = ARG_INT(2);
    PER_NUMBER
    if (i < n) x[i] += y[i];
}

kernel void softcap(ARGS, KP) {
    device float* x = ARG_PTR(float, 0);
    const float cap = ARG_F32(1);
    const int n = ARG_INT(2);
    PER_NUMBER
    if (i < n) x[i] = cap * tanh(x[i] / cap);
}

// ---- Qwen 3.5 ------------------------------------------------------------------------

kernel void apply_gate(ARGS, KP) {
    device float* out = ARG_PTR(float, 0);
    device const float* query_gate = ARG_PTR(const float, 1);
    const int heads = ARG_INT(2), head_dim = ARG_INT(3);
    PER_NUMBER
    if (i >= heads * head_dim) return;
    const int h = i / head_dim, d = i % head_dim;
    out[i] *= 1.0f / (1.0f + exp(-query_gate[h * 2 * head_dim + head_dim + d]));
}

#define CONV_DISPATCH(fn, ...)              \
    switch (kernel_size) {                  \
        case 2: fn<2>(__VA_ARGS__); return; \
        case 3: fn<3>(__VA_ARGS__); return; \
        case 4: fn<4>(__VA_ARGS__); return; \
        case 5: fn<5>(__VA_ARGS__); return; \
        case 6: fn<6>(__VA_ARGS__); return; \
        case 7: fn<7>(__VA_ARGS__); return; \
        default: fn<8>(__VA_ARGS__); return; \
    }

template <int K>
static inline void conv_silu_tokens(device const float* current, device float* history, device const float* weights,
                                    device float* out, int channels, int tokens, int c) {
    if (c >= channels) return;
    float w[K], past[K - 1];
    for (int i = 0; i < K; i++) w[i] = weights[(ulong)c * K + i];
    for (int i = 0; i < K - 1; i++) past[i] = history[(ulong)i * channels + c];
    for (int t = 0; t < tokens; t++) {
        float sum = 0.0f;
        for (int i = 0; i < K - 1; i++) sum += past[i] * w[i];
        const float now = current[(ulong)t * channels + c];
        sum += now * w[K - 1];
        out[(ulong)t * channels + c] = sum / (1.0f + exp(-sum));
        for (int i = 0; i < K - 2; i++) past[i] = past[i + 1];
        past[K - 2] = now;
    }
    for (int i = 0; i < K - 1; i++) history[(ulong)i * channels + c] = past[i];
}

kernel void conv_silu(ARGS, KP) {
    device const float* current = ARG_PTR(const float, 0);
    device float* history = ARG_PTR(float, 1);
    device const float* weights = ARG_PTR(const float, 2);
    device float* out = ARG_PTR(float, 3);
    const int channels = ARG_INT(4), kernel_size = ARG_INT(5), tokens = ARG_INT(6);
    const int c = gid.x * tpg.x + tid;
    CONV_DISPATCH(conv_silu_tokens, current, history, weights, out, channels, tokens, c)
}

#define CONV_TOKENS 16
template <int K>
static inline void conv_silu_run(device const float* current, device const float* history, device const float* weights,
                                 device float* out, int channels, int tokens, int c, int t0) {
    if (c >= channels) return;
    const int t1 = min(t0 + CONV_TOKENS, tokens);
    float w[K], past[K - 1];
    for (int k = 0; k < K; k++) w[k] = weights[(ulong)c * K + k];
    for (int k = 0; k < K - 1; k++) {
        const int from = t0 - (K - 1) + k;
        past[k] = from >= 0 ? current[(ulong)from * channels + c] : history[(ulong)(K - 1 + from) * channels + c];
    }
    for (int t = t0; t < t1; t++) {
        const float now = current[(ulong)t * channels + c];
        float sum = 0.0f;
        for (int k = 0; k < K - 1; k++) sum += past[k] * w[k];
        sum += now * w[K - 1];
        out[(ulong)t * channels + c] = sum / (1.0f + exp(-sum));
        for (int k = 0; k < K - 2; k++) past[k] = past[k + 1];
        past[K - 2] = now;
    }
}

kernel void conv_silu_parallel(ARGS, KP) {
    device const float* current = ARG_PTR(const float, 0);
    device const float* history = ARG_PTR(const float, 1);
    device const float* weights = ARG_PTR(const float, 2);
    device float* out = ARG_PTR(float, 3);
    const int channels = ARG_INT(4), kernel_size = ARG_INT(5), tokens = ARG_INT(6);
    const int c = gid.x * tpg.x + tid, t0 = gid.y * CONV_TOKENS;
    CONV_DISPATCH(conv_silu_run, current, history, weights, out, channels, tokens, c, t0)
}

template <int K>
static inline void conv_history_run(device const float* current, device float* history, int channels, int tokens,
                                    int c) {
    if (c >= channels) return;
    float past[K - 1];
    for (int k = 0; k < K - 1; k++) {
        const int from = tokens - (K - 1) + k;
        past[k] = from >= 0 ? current[(ulong)from * channels + c] : history[(ulong)(K - 1 + from) * channels + c];
    }
    for (int k = 0; k < K - 1; k++) history[(ulong)k * channels + c] = past[k];
}

kernel void conv_history(ARGS, KP) {
    device const float* current = ARG_PTR(const float, 0);
    device float* history = ARG_PTR(float, 1);
    const int channels = ARG_INT(2), kernel_size = ARG_INT(3), tokens = ARG_INT(4);
    const int c = gid.x * tpg.x + tid;
    CONV_DISPATCH(conv_history_run, current, history, channels, tokens, c)
}

// Gated DeltaNet token by token, the state in registers: 4 threads to a column (see
// delta_net in vireo.cu). n a multiple of 8 up to 128.
kernel void delta_net(ARGS, KP) {
    device const float* conv = ARG_PTR(const float, 0);
    const int channels = ARG_INT(1), key_dim = ARG_INT(2);
    device const float* beta_raw = ARG_PTR(const float, 3);
    device const float* alpha_raw = ARG_PTR(const float, 4);
    device const float* dt_bias = ARG_PTR(const float, 5);
    device const float* a = ARG_PTR(const float, 6);
    device float* state = ARG_PTR(float, 7);
    device const float* z = ARG_PTR(const float, 8);
    device const float* norm = ARG_PTR(const float, 9);
    device float* out = ARG_PTR(float, 10);
    const int n = ARG_INT(11), k_heads = ARG_INT(12);
    const float eps = ARG_F32(13);
    const int tokens = ARG_INT(14);
    threadgroup float scratch[32];
    threadgroup float ks[128];
    threadgroup float qs[128];
    const int h = gid.x, v_heads = ngrid.x, kh = h % k_heads;
    const int j = tid >> 2, part = tid & 3, per = n >> 2;
    device float* column = state + ((ulong)h * n + j) * n;
    float st[32];
    for (int r = 0; r < 32; r++)
        if (r < per) st[r] = column[part + 4 * r];
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int t = 0; t < tokens; t++) {
        device const float* c = conv + (ulong)t * channels;
        float qj = 0.0f, kj = 0.0f;
        if (part == 0) {
            qj = c[kh * n + j];
            kj = c[key_dim + kh * n + j];
        }
        const float q_len = sqrt(block_sum(qj * qj, scratch, tid, tpg.x));
        const float k_len = sqrt(block_sum(kj * kj, scratch, tid, tpg.x));
        if (part == 0) {
            qs[j] = qj * (1.0f / max(q_len, eps));
            ks[j] = kj * (1.0f / max(k_len, eps));
        }
        BARRIER;
        const int head_at = t * v_heads + h;
        const float beta = 1.0f / (1.0f + exp(-beta_raw[head_at]));
        const float decay = exp(softplus_(alpha_raw[head_at] + dt_bias[h]) * a[h]);

        float dot = 0.0f;
        for (int r = 0; r < 32; r++) {
            if (r < per) {
                st[r] *= decay;
                dot += st[r] * ks[part + 4 * r];
            }
        }
        dot += simd_shuffle_xor(dot, (ushort)1);
        dot += simd_shuffle_xor(dot, (ushort)2);
        const float delta = (c[2 * key_dim + h * n + j] - dot) * beta;
        float o = 0.0f;
        for (int r = 0; r < 32; r++) {
            if (r < per) {
                st[r] += ks[part + 4 * r] * delta;
                o += st[r] * qs[part + 4 * r];
            }
        }
        o += simd_shuffle_xor(o, (ushort)1);
        o += simd_shuffle_xor(o, (ushort)2);
        o *= inv_sqrt_n;

        const float sum = block_sum(part == 0 ? o * o : 0.0f, scratch, tid, tpg.x);
        if (part == 0) {
            const float normed = o / sqrt(sum / (float)n + eps) * norm[j];
            const ulong at = (ulong)head_at * n + j;
            const float zj = z[at];
            out[at] = normed * (zj / (1.0f + exp(-zj)));
        }
    }
    for (int r = 0; r < 32; r++)
        if (r < per) column[part + 4 * r] = st[r];
}

// Gated DeltaNet for state sizes over 128: a thread per row of the state.
kernel void delta_net_wide(ARGS, KP) {
    device const float* conv = ARG_PTR(const float, 0);
    const int channels = ARG_INT(1), key_dim = ARG_INT(2);
    device const float* beta_raw = ARG_PTR(const float, 3);
    device const float* alpha_raw = ARG_PTR(const float, 4);
    device const float* dt_bias = ARG_PTR(const float, 5);
    device const float* a = ARG_PTR(const float, 6);
    device float* state = ARG_PTR(float, 7);
    device const float* z = ARG_PTR(const float, 8);
    device const float* norm = ARG_PTR(const float, 9);
    device float* out = ARG_PTR(float, 10);
    const int n = ARG_INT(11), k_heads = ARG_INT(12);
    const float eps = ARG_F32(13);
    const int tokens = ARG_INT(14);
    threadgroup float scratch[32];
    threadgroup float ks[256];
    threadgroup float qs[256];
    const int h = gid.x, v_heads = ngrid.x, j = tid, kh = h % k_heads;
    device float* row = state + ((ulong)h * n + j) * n;
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int t = 0; t < tokens; t++) {
        device const float* c = conv + (ulong)t * channels;
        const float qj = c[kh * n + j], kj = c[key_dim + kh * n + j];
        const float q_len = sqrt(block_sum(qj * qj, scratch, tid, tpg.x));
        const float k_len = sqrt(block_sum(kj * kj, scratch, tid, tpg.x));
        qs[j] = qj * (1.0f / max(q_len, eps));
        ks[j] = kj * (1.0f / max(k_len, eps));
        BARRIER;
        const int head_at = t * v_heads + h;
        const float beta = 1.0f / (1.0f + exp(-beta_raw[head_at]));
        const float decay = exp(softplus_(alpha_raw[head_at] + dt_bias[h]) * a[h]);
        float dot = 0.0f;
        for (int i = 0; i < n; i++) {
            row[i] *= decay;
            dot += row[i] * ks[i];
        }
        const float delta = (c[2 * key_dim + h * n + j] - dot) * beta;
        float o = 0.0f;
        for (int i = 0; i < n; i++) {
            row[i] += ks[i] * delta;
            o += row[i] * qs[i];
        }
        o *= inv_sqrt_n;
        const float sum = block_sum(o * o, scratch, tid, tpg.x);
        const float normed = o / sqrt(sum / (float)n + eps) * norm[j];
        const ulong at = (ulong)head_at * n + j;
        const float zj = z[at];
        out[at] = normed * (zj / (1.0f + exp(-zj)));
    }
}

// q and k of each token L2-normalized in place: one threadgroup of n per row of n.
kernel void delta_l2(ARGS, KP) {
    device float* conv = ARG_PTR(float, 0);
    const int channels = ARG_INT(1), k_heads = ARG_INT(2), n = ARG_INT(3);
    const float eps = ARG_F32(4);
    threadgroup float scratch[32];
    const int t = gid.x / (2 * k_heads), which = gid.x % (2 * k_heads);
    device float* p = conv + (ulong)t * channels + which * n;
    const float v = p[tid];
    const float len = sqrt(block_sum(v * v, scratch, tid, tpg.x));
    p[tid] = v * (1.0f / max(len, eps));
}

// The state update for a prompt, columns split over threadgroups (see delta_scan in
// vireo.cu).
#define DS_COLS 16
#define DS_CHUNK 16
kernel void delta_scan(ARGS, KP) {
    device const float* conv = ARG_PTR(const float, 0);
    const int channels = ARG_INT(1), key_dim = ARG_INT(2);
    device const float* beta_raw = ARG_PTR(const float, 3);
    device const float* alpha_raw = ARG_PTR(const float, 4);
    device const float* dt_bias = ARG_PTR(const float, 5);
    device const float* a = ARG_PTR(const float, 6);
    device float* state = ARG_PTR(float, 7);
    device float* out = ARG_PTR(float, 8);
    const int n = ARG_INT(9), k_heads = ARG_INT(10), tokens = ARG_INT(11);
    threadgroup float ks[DS_CHUNK][128], qs[DS_CHUNK][128];
    threadgroup float betas[DS_CHUNK], decays[DS_CHUNK];
    const int h = gid.x, v_heads = ngrid.x, kh = h % k_heads;
    const int j = gid.y * DS_COLS + (tid >> 2), part = tid & 3, per = n >> 2;
    device float* column = state + ((ulong)h * n + j) * n;
    float st[32];
    for (int r = 0; r < 32; r++)
        if (r < per) st[r] = column[part + 4 * r];
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int t0 = 0; t0 < tokens; t0 += DS_CHUNK) {
        const int count = min(DS_CHUNK, tokens - t0);
        BARRIER;
        for (int i = tid; i < count * n; i += tpg.x) {
            const int tt = i / n, e = i % n;
            device const float* c = conv + (ulong)(t0 + tt) * channels;
            qs[tt][e] = c[kh * n + e];
            ks[tt][e] = c[key_dim + kh * n + e];
        }
        if ((int)tid < count) {
            const int head_at = (t0 + tid) * v_heads + h;
            betas[tid] = 1.0f / (1.0f + exp(-beta_raw[head_at]));
            decays[tid] = exp(softplus_(alpha_raw[head_at] + dt_bias[h]) * a[h]);
        }
        BARRIER;
        for (int tt = 0; tt < count; tt++) {
            const float decay = decays[tt], beta = betas[tt];
            float dot = 0.0f;
            for (int r = 0; r < 32; r++) {
                if (r < per) {
                    st[r] *= decay;
                    dot += st[r] * ks[tt][part + 4 * r];
                }
            }
            dot += simd_shuffle_xor(dot, (ushort)1);
            dot += simd_shuffle_xor(dot, (ushort)2);
            const float delta = (conv[(ulong)(t0 + tt) * channels + 2 * key_dim + h * n + j] - dot) * beta;
            float o = 0.0f;
            for (int r = 0; r < 32; r++) {
                if (r < per) {
                    st[r] += ks[tt][part + 4 * r] * delta;
                    o += st[r] * qs[tt][part + 4 * r];
                }
            }
            o += simd_shuffle_xor(o, (ushort)1);
            o += simd_shuffle_xor(o, (ushort)2);
            if (part == 0) out[((ulong)(t0 + tt) * v_heads + h) * n + j] = o * inv_sqrt_n;
        }
    }
    for (int r = 0; r < 32; r++)
        if (r < per) column[part + 4 * r] = st[r];
}

// out = RMSNorm(o) × norm × SiLU(z) in place, one threadgroup of n per token and head.
kernel void delta_out(ARGS, KP) {
    device float* out = ARG_PTR(float, 0);
    device const float* z = ARG_PTR(const float, 1);
    device const float* norm = ARG_PTR(const float, 2);
    const int n = ARG_INT(3);
    const float eps = ARG_F32(4);
    threadgroup float scratch[32];
    const ulong at = (ulong)gid.x * n + tid;
    const float o = out[at];
    const float sum = block_sum(o * o, scratch, tid, tpg.x);
    const float normed = o / sqrt(sum / (float)n + eps) * norm[tid];
    const float zj = z[at];
    out[at] = normed * (zj / (1.0f + exp(-zj)));
}

// ---- Copies --------------------------------------------------------------------------

// `bytes` from `src` to `dst`, in order with the other work: 16 bytes a thread when both
// are 16-byte aligned, otherwise one.
kernel void copy_bytes(ARGS, KP) {
    const ulong dst = A[0], src = A[1], bytes = A[2];
    const ulong i = (ulong)gid.x * tpg.x + tid;
    if (((dst | src | bytes) & 15) == 0) {
        if (i * 16 < bytes)
            reinterpret_cast<device uint4*>(dst)[i] = reinterpret_cast<device const uint4*>(src)[i];
    } else if (i < bytes) {
        reinterpret_cast<device u8*>(dst)[i] = reinterpret_cast<device const u8*>(src)[i];
    }
}
