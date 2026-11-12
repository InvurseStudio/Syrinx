// Vireo's CUDA kernels: everything Llama, Qwen 3.5 and Gemma 4 need, for one token at a
// time (generating) or many (reading a prompt).
//
// Compiled to PTX by build.rs and loaded by src/cuda/mod.rs. The quantized block
// layouts match src/quant.rs (and ggml), except Q4_0's, repacked on upload (see
// q4_0_row_bytes). Most kernels work in f32 on the exact weights and activations, as the
// CPU reference does, so the two can be compared number for number (they differ only in
// the order of sums); the "8-bit activations" section rounds the activations first, as
// llama.cpp does, for speed (the engine's default; Precision::Exact uses the others).

// Kernels that depend on the position read it from GPU memory (`pos`), so one token's
// launches are the same every time and can be recorded once as a CUDA graph.

#include <cuda_fp16.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

// Small whole numbers in each byte as signed bytes, minus their offset: 4-bit n (0..15) as
// n − 8 and 6-bit n (0..63) as n − 32. n ^ 8 (n ^ 32) is the answer in 4-bit (6-bit) two's
// complement; copying its sign bit into the byte's top bits (a multiply with no carries
// between bytes) widens it. What __vsubss4 computes, in 3 instructions instead of the ~8
// it compiles to.
__device__ __forceinline__ u32 sext4(u32 n) {
    const u32 t = n ^ 0x08080808u;
    return t | ((t & 0x08080808u) * 0x1Eu);
}

__device__ __forceinline__ u32 sext6(u32 n) {
    const u32 t = n ^ 0x20202020u;
    return t | ((t & 0x20202020u) * 0x06u);
}

__device__ __forceinline__ float half_at(const u8* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// Four bytes from an address that's only 2-byte aligned (Q4_0, Q8_0 and Q6_K blocks
// have odd sizes, so their contents are only 2-byte aligned).
__device__ __forceinline__ u32 u32_at2(const u8* p) {
    const u16* h = reinterpret_cast<const u16*>(p);
    return (u32)h[0] | ((u32)h[1] << 16);
}

__device__ __forceinline__ float4 f4(const float* p) { return *reinterpret_cast<const float4*>(p); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_sum_all(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
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

// Q4_0 on the GPU: each row's 16-byte runs of 4-bit numbers first, then its f16 scales,
// padded to 16 bytes (Gpu::upload_weights repacks GGUF's 18-byte blocks so). Block b's
// numbers are at row + 16b, aligned for 16-byte loads, and its scale at row + 16 × blocks
// + 2b. Byte i holds number i (low nibble) and i + 16 (high), offset 8, as in GGUF.
__device__ __forceinline__ size_t q4_0_row_bytes(int cols) {
    const int blocks = cols / 32;
    return (size_t)blocks * 16 + ((blocks * 2 + 15) & ~15);
}

__device__ __forceinline__ float q4_0_scale(const u8* row, int blocks, int b) {
    return half_at(row + (size_t)blocks * 16 + b * 2);
}

// Q6_K on the GPU: GGUF's 210-byte blocks padded to 224, so each starts 16-byte aligned
// (Gpu::upload_weights). The contents are as in GGUF.
#define Q6K_BYTES 224

// ---- Matrix-vector products: y = W x, one warp per row of W -------------------------
//
// Generating a token reads every weight once, so these are what limit its speed. Each
// lane reads several bytes of a block at once (not one), and a warp works on several
// blocks per step, so plenty of memory reads are in flight.

#define MATVEC_ROW                                                              \
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; \
    if (row >= rows) return;

// Plain f32 rows.
extern "C" __global__ void matvec_f32(const float* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROW
    const float* r = w + (size_t)row * cols;
    float sum = 0.0f;
    if ((cols & 3) == 0) {
        for (int c = lane * 4; c < cols; c += 128) {
            const float4 a = f4(r + c), b = f4(x + c);
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int c = lane; c < cols; c += 32) sum += r[c] * x[c];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// f16 rows: 8 numbers (16 bytes) per lane per step when the rows allow it.
extern "C" __global__ void matvec_f16(const __half* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROW
    const __half* r = w + (size_t)row * cols;
    float sum = 0.0f;
    if ((cols & 7) == 0) {
        for (int c = lane * 8; c < cols; c += 256) {
            const uint4 packed = *reinterpret_cast<const uint4*>(r + c);
            const __half2* h = reinterpret_cast<const __half2*>(&packed);
            const float4 x0 = f4(x + c), x1 = f4(x + c + 4);
            const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
            const float2 cc = __half22float2(h[2]), d = __half22float2(h[3]);
            sum += a.x * x0.x + a.y * x0.y + b.x * x0.z + b.y * x0.w;
            sum += cc.x * x1.x + cc.y * x1.y + d.x * x1.z + d.y * x1.w;
        }
    } else {
        for (int c = lane; c < cols; c += 32) sum += __half2float(r[c]) * x[c];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

// The quantized kernels below give each warp MV_ROWS rows: the slice of x a lane needs
// is read once and used for all of them, which halves the reads of x per weight byte.
// A small whole number (under 2^23) as a float, minus `offset`, exactly: the number
// goes into the mantissa of 2^23, and 2^23 + offset comes off. Cheaper than converting
// with I2F, whose rate on these GPUs is what limited the kernels below.
__device__ __forceinline__ float small_int(u32 v, float offset) {
    return __int_as_float(0x4B000000u | v) - (8388608.0f + offset);
}

#define MV_ROWS 2
#define MATVEC_ROWS                                                                         \
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; \
    const int row0 = warp * MV_ROWS;                                                        \
    if (row0 >= rows) return;                                                               \
    const int nrows = min(MV_ROWS, rows - row0);                                            \
    float sum[MV_ROWS];                                                                     \
    for (int r = 0; r < MV_ROWS; r++) sum[r] = 0.0f;

#define MATVEC_STORE                                     \
    for (int r = 0; r < MV_ROWS; r++) {                  \
        const float total = warp_sum(sum[r]);            \
        if (lane == 0 && r < nrows) y[row0 + r] = total; \
    }

__device__ __forceinline__ float half_lo(u32 v) { return __half2float(__ushort_as_half((unsigned short)(v & 0xFFFF))); }
__device__ __forceinline__ float half_hi(u32 v) { return __half2float(__ushort_as_half((unsigned short)(v >> 16))); }

// Σ nibble_i × x_i over 8 nibbles of `q` (the low or high nibbles of 8 bytes).
__device__ __forceinline__ float nibble_dot8(uint2 q, int shift, float4 a, float4 b) {
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu, hi = (q.y >> shift) & 0x0F0F0F0Fu;
    return small_int(lo & 0xFF, 0.0f) * a.x + small_int((lo >> 8) & 0xFF, 0.0f) * a.y + small_int((lo >> 16) & 0xFF, 0.0f) * a.z +
           small_int(lo >> 24, 0.0f) * a.w + small_int(hi & 0xFF, 0.0f) * b.x + small_int((hi >> 8) & 0xFF, 0.0f) * b.y +
           small_int((hi >> 16) & 0xFF, 0.0f) * b.z + small_int(hi >> 24, 0.0f) * b.w;
}

// Σ bit_i × x_i over the bits at `shift` of 8 bytes.
__device__ __forceinline__ float bit_dot8(uint2 q, int shift, float4 a, float4 b) {
    const u32 lo = (q.x >> shift) & 0x01010101u, hi = (q.y >> shift) & 0x01010101u;
    return small_int(lo & 1, 0.0f) * a.x + small_int((lo >> 8) & 1, 0.0f) * a.y + small_int((lo >> 16) & 1, 0.0f) * a.z +
           small_int(lo >> 24, 0.0f) * a.w + small_int(hi & 1, 0.0f) * b.x + small_int((hi >> 8) & 1, 0.0f) * b.y +
           small_int((hi >> 16) & 1, 0.0f) * b.z + small_int(hi >> 24, 0.0f) * b.w;
}

// Q8_0: blocks of 32 in 34 bytes (f16 scale, 32 int8s). 8 lanes per block, 4 numbers
// each; a warp takes 4 blocks per step.
extern "C" __global__ void matvec_q8_0(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const int sub = lane >> 3, j = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 4) {
        const float4 xv = f4(x + b * 32 + j);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 34;
            const u32 q = u32_at2(block + 2 + j);
            const float dot = small_int((q & 0xFF) ^ 0x80, 128.0f) * xv.x + small_int(((q >> 8) & 0xFF) ^ 0x80, 128.0f) * xv.y +
                              small_int(((q >> 16) & 0xFF) ^ 0x80, 128.0f) * xv.z + small_int((q >> 24) ^ 0x80, 128.0f) * xv.w;
            sum[r] += half_at(block) * dot;
        }
    }
    MATVEC_STORE
}

// Q4_0 (the GPU layout, see q4_0_row_bytes): 2 lanes per block, 8 bytes (16 numbers)
// each; a warp takes 16 blocks per step.
extern "C" __global__ void matvec_q4_0(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub; b < blocks; b += 16) {
        const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<const uint2*>(row + b * 16 + j);
            const float dot = nibble_dot8(q, 0, lo0, lo1) + nibble_dot8(q, 4, hi0, hi1) -
                              8.0f * ((lo0.x + lo0.y) + (lo0.z + lo0.w) + (lo1.x + lo1.y) + (lo1.z + lo1.w) +
                                      (hi0.x + hi0.y) + (hi0.z + hi0.w) + (hi1.x + hi1.y) + (hi1.z + hi1.w));
            sum[r] += q4_0_scale(row, blocks, b) * dot;
        }
    }
    MATVEC_STORE
}

// Q4_K's scale and minimum for sub-block j from its 12 scale bytes read as 3 words, as
// q4_k_scale_min does from bytes.
__device__ __forceinline__ void q4_k_scale_min_w(int j, u32 s0, u32 s1, u32 s2, float& scale, float& min) {
    const int at = (j & 3) * 8;
    const u32 b0 = (s0 >> at) & 0xFF, b1 = (s1 >> at) & 0xFF, b2 = (s2 >> at) & 0xFF;
    if (j < 4) {
        scale = b0 & 63;
        min = b1 & 63;
    } else {
        scale = (b2 & 0xF) | ((b0 >> 6) << 4);
        min = (b2 >> 4) | ((b1 >> 6) << 4);
    }
}

// Q4_K: blocks of 256 in 144 bytes. 16 lanes per block: lane j reads 8 bytes of one
// 32-byte group of numbers, which hold 8 numbers of one sub-block (low nibbles) and 8 of
// the next (high nibbles). A warp takes 2 blocks per step.
extern "C" __global__ void matvec_q4_k(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * 144;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 144;
            const uint4 head = *reinterpret_cast<const uint4*>(block);  // d, dmin and the 12 scale bytes
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<const uint2*>(block + 16 + pair * 32 + part);
            sum[r] += d * s1 * nibble_dot8(q, 0, a1, b1) - dmin * m1 * x1 + d * s2 * nibble_dot8(q, 4, a2, b2) -
                      dmin * m2 * x2;
        }
    }
    MATVEC_STORE
}

// Q5_K: Q4_K plus 32 bytes of fifth bits (qh) before the 4-bit numbers; number l of
// sub-block j takes its fifth bit from bit j of qh[l]. Lanes as in Q4_K.
extern "C" __global__ void matvec_q5_k(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * 176;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 176;
            const uint4 head = *reinterpret_cast<const uint4*>(block);
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<const uint2*>(block + 48 + pair * 32 + part);
            const uint2 qh = *reinterpret_cast<const uint2*>(block + 16 + part);
            // The fifth bits add 16 to some numbers.
            const float dot1 = nibble_dot8(q, 0, a1, b1) + 16.0f * bit_dot8(qh, 2 * pair, a1, b1);
            const float dot2 = nibble_dot8(q, 4, a2, b2) + 16.0f * bit_dot8(qh, 2 * pair + 1, a2, b2);
            sum[r] += d * s1 * dot1 - dmin * m1 * x1 + d * s2 * dot2 - dmin * m2 * x2;
        }
    }
    MATVEC_STORE
}

// Q6_K: blocks of 256 in 210 bytes (ql 128, qh 64, 16 int8 scales, f16 d), padded to
// Q6K_BYTES on the GPU so they start 16-byte aligned (Gpu::upload_weights). Each half
// of a block is 4 runs of 32 numbers. 16 lanes per block: lane j takes 4 positions of
// one half (16 numbers, one from each run). A warp takes 2 blocks per step.
extern "C" __global__ void matvec_q6_k(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * Q6K_BYTES;
    const int sub = lane >> 4, half = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 2) {
        const float* xb = x + b * 256 + half * 128 + l;
        float4 xv[4];
#pragma unroll
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at2(block + half * 64 + l), ql1 = u32_at2(block + half * 64 + 32 + l);
            const u32 qh = u32_at2(block + 128 + half * 32 + l);
            const signed char* sc = reinterpret_cast<const signed char*>(block + 192 + half * 8 + l / 16);
            float block_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const u32 low = ((k & 1) ? ql1 : ql0) >> (k < 2 ? 0 : 4);
                const float q0 = small_int((low & 0xF) | (((qh >> (2 * k)) & 3) << 4), 32.0f);
                const float q1 = small_int(((low >> 8) & 0xF) | (((qh >> (8 + 2 * k)) & 3) << 4), 32.0f);
                const float q2 = small_int(((low >> 16) & 0xF) | (((qh >> (16 + 2 * k)) & 3) << 4), 32.0f);
                const float q3 = small_int(((low >> 24) & 0xF) | (((qh >> (24 + 2 * k)) & 3) << 4), 32.0f);
                block_sum += (float)sc[2 * k] * (q0 * xv[k].x + q1 * xv[k].y + q2 * xv[k].z + q3 * xv[k].w);
            }
            sum[r] += d * block_sum;
        }
    }
    MATVEC_STORE
}

// ---- Matrix-matrix products: Y = X Wᵀ for many tokens at once ------------------------
//
// Reading a prompt: each weight is decoded once per tile of 64 tokens instead of once per
// token. Tiles of 64 rows × 64 tokens, 32 columns at a time: the block decodes the
// weights into shared memory as f32, and each of its 256 threads works out 4 × 4 results.
// x is `tokens` rows of `cols`, y is `tokens` rows of `rows`.

// Eight numbers of a row, from column k (a multiple of 8), decoded as quant.rs does.
template <int KIND>
__device__ __forceinline__ void decode8(const u8* row, int k, float* v);

enum { K_F32, K_F16, K_Q4_0, K_Q8_0, K_Q4_K, K_Q5_K, K_Q6_K };

template <>
__device__ __forceinline__ void decode8<K_F32>(const u8* row, int k, float* v) {
    const float* r = reinterpret_cast<const float*>(row) + k;
    for (int i = 0; i < 8; i++) v[i] = r[i];
}

template <>
__device__ __forceinline__ void decode8<K_F16>(const u8* row, int k, float* v) {
    const __half* r = reinterpret_cast<const __half*>(row) + k;
    for (int i = 0; i < 8; i++) v[i] = __half2float(r[i]);
}

template <>
__device__ __forceinline__ void decode8<K_Q4_0>(const u8* row, int k, float* v) {
    // `row` is the row's start in the GPU layout; the row's length isn't known here, so
    // the scale's place comes from decode8_q4_0_blocks.
    (void)row;
    (void)k;
    (void)v;
}

template <>
__device__ __forceinline__ void decode8<K_Q8_0>(const u8* row, int k, float* v) {
    const u8* block = row + (k / 32) * 34;
    const float d = half_at(block);
    const int j = k % 32;
    for (int i = 0; i < 8; i++) v[i] = d * (float)(signed char)block[2 + j + i];
}

template <>
__device__ __forceinline__ void decode8<K_Q4_K>(const u8* row, int k, float* v) {
    const u8* block = row + (k / 256) * 144;
    const int n = k % 256, j = n / 32, l = n % 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    const u8* q = block + 16 + (j / 2) * 32 + l;
    const int shift = (j & 1) * 4;
    for (int i = 0; i < 8; i++) v[i] = d1 * (float)((q[i] >> shift) & 0xF) - min1;
}

template <>
__device__ __forceinline__ void decode8<K_Q5_K>(const u8* row, int k, float* v) {
    const u8* block = row + (k / 256) * 176;
    const int n = k % 256, j = n / 32, l = n % 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    const u8* q = block + 48 + (j / 2) * 32 + l;
    const u8* qh = block + 16 + l;
    const int shift = (j & 1) * 4;
    for (int i = 0; i < 8; i++) {
        const float high = (qh[i] >> j) & 1 ? 16.0f : 0.0f;
        v[i] = d1 * ((float)((q[i] >> shift) & 0xF) + high) - min1;
    }
}

template <>
__device__ __forceinline__ void decode8<K_Q6_K>(const u8* row, int k, float* v) {
    const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, half = n / 128, run = (n % 128) / 32, l = n % 32;
    const float d = half_at(block + 208);
    const u8* ql = block + half * 64 + (run & 1) * 32 + l;
    const u8* qh = block + 128 + half * 32 + l;
    const signed char* sc = reinterpret_cast<const signed char*>(block + 192 + half * 8 + l / 16 + 2 * run);
    const float ds = d * (float)sc[0];
    for (int i = 0; i < 8; i++) {
        const int low = run < 2 ? (ql[i] & 0xF) : (ql[i] >> 4);
        const int q = (low | (((qh[i] >> (2 * run)) & 3) << 4)) - 32;
        v[i] = ds * (float)q;
    }
}

#define MM_ROWS 128
#define MM_TOKENS 64
#define MM_K 32

// Tiles of 128 rows × 64 tokens; each of the 256 threads works out 8 rows × 4 tokens.
// Q4_0 needs the row's block count to find its scales (see q4_0_row_bytes).
__device__ __forceinline__ void decode8_q4_0(const u8* row, int blocks, int k, float* v) {
    const int b = k / 32, j = k % 32;
    const float d = q4_0_scale(row, blocks, b);
    const uint2 q = *reinterpret_cast<const uint2*>(row + b * 16 + (j & 15));
    const int shift = j < 16 ? 0 : 4;
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu, hi = (q.y >> shift) & 0x0F0F0F0Fu;
    for (int i = 0; i < 4; i++) {
        v[i] = d * (float)((int)((lo >> (8 * i)) & 0xF) - 8);
        v[4 + i] = d * (float)((int)((hi >> (8 * i)) & 0xF) - 8);
    }
}

template <int KIND>
__device__ void matmul(const u8* w, size_t row_bytes, const float* x, float* y, int rows, int cols, int tokens) {
    __shared__ __align__(16) float ws[MM_K][MM_ROWS + 4];
    __shared__ __align__(16) float xs[MM_K][MM_TOKENS + 4];
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * MM_ROWS, tok0 = blockIdx.y * MM_TOKENS;
    const int load_row = tid >> 2, load_k = (tid & 3) * 8;
    const int ty = tid >> 4, tx = tid & 15;
    float acc[8][4] = {};

    for (int k0 = 0; k0 < cols; k0 += MM_K) {
        for (int half = 0; half < 2; half++) {
            float v[8];
            const int lr = load_row + half * 64, r = row0 + lr;
            if (r < rows) {
                if (KIND == K_Q4_0)
                    decode8_q4_0(w + (size_t)r * row_bytes, cols / 32, k0 + load_k, v);
                else
                    decode8<KIND>(w + (size_t)r * row_bytes, k0 + load_k, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) ws[load_k + i][lr] = v[i];
        }
        const int t = tok0 + load_row;
        if (t < tokens) {
            const float* xr = x + (size_t)t * cols + k0 + load_k;
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
        __syncthreads();
        // Unrolled only 4 steps at a time: all 32 would be more code than the
        // instruction cache holds.
#pragma unroll 4
        for (int k = 0; k < MM_K; k++) {
            // Rows ty*4..+4 and 64+ty*4..+4: two float4 reads without bank conflicts.
            const float4 a0 = *reinterpret_cast<const float4*>(&ws[k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&ws[k][64 + ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&xs[k][tx * 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    // Fully unrolled, so `acc` stays in registers (a loop the compiler can't unroll
    // would put it in memory).
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int t = tok0 + tx * 4 + j;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
            if (t < tokens && r < rows) y[(size_t)t * rows + r] = acc[i][j];
        }
    }
}

#define MATMUL(name, KIND, ROW_BYTES)                                                                  \
    extern "C" __global__ void name(const u8* w, const float* x, float* y, int rows, int cols, int tokens) { \
        matmul<KIND>(w, (size_t)(ROW_BYTES), x, y, rows, cols, tokens);                               \
    }

MATMUL(matmul_f32, K_F32, cols * 4)
MATMUL(matmul_f16, K_F16, cols * 2)
MATMUL(matmul_q4_0, K_Q4_0, q4_0_row_bytes(cols))
MATMUL(matmul_q8_0, K_Q8_0, cols / 32 * 34)
MATMUL(matmul_q4_k, K_Q4_K, cols / 256 * 144)
MATMUL(matmul_q5_k, K_Q5_K, cols / 256 * 176)
MATMUL(matmul_q6_k, K_Q6_K, cols / 256 * Q6K_BYTES)

// ---- 8-bit activations: faster, not exact ---------------------------------------------
//
// The kernels above work in f32 on the exact weights and activations. Those below round
// the activations (x) to 8 bits first, as llama.cpp does, so the weights' small whole
// numbers can be multiplied 4 at a time (dp4a) or on the tensor cores (mma). Each block
// of 32 numbers of x gets a scale d = max|x| / 127 and q = round(x / d); `xs` keeps
// d × Σq for each 16, for the weights' minimums. Rows of x are `cols` long: xq holds cols
// bytes per row, xd cols / 32 floats and xs cols / 16.

// One warp per block of 32: rounds `x` to 8 bits.
extern "C" __global__ void quantize_q8(const float* x, signed char* xq, float* xd, float* xs, int blocks) {
    const int block = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (block >= blocks) return;
    const float v = x[(size_t)block * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    xq[(size_t)block * 32 + lane] = (signed char)q;
    // The two halves' sums of q.
    int sum = q;
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) xd[block] = d;
    if (lane == 0 || lane == 16) xs[(size_t)block * 2 + lane / 16] = d * (float)sum;
}

__device__ __forceinline__ int i32_at(const signed char* p) { return *reinterpret_cast<const int*>(p); }

// The dp4a matrix-vector kernels: as the f32 ones (MV_ROWS rows a warp, lanes laid out
// the same), with x as 8-bit numbers.

// Q8_0: d × dx × Σ q qx per block of 32.
extern "C" __global__ void matvec8_q8_0(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const int sub = lane >> 3, j = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 4) {
        const int xv = i32_at(xq + b * 32 + j);
        const float dx = xd[b];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 34;
            sum[r] += half_at(block) * dx * (float)__dp4a((int)u32_at2(block + 2 + j), xv, 0);
        }
    }
    MATVEC_STORE
}

// Q4_0 (the GPU layout): d × dx × Σ (q − 8) qx per block of 32. 2 lanes per block, 8
// bytes (16 numbers) each; a warp takes 16 blocks per step.
extern "C" __global__ void matvec8_q4_0(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub; b < blocks; b += 16) {
        const int2 lo = *reinterpret_cast<const int2*>(xq + b * 32 + j);
        const int2 hi = *reinterpret_cast<const int2*>(xq + b * 32 + 16 + j);
        const float dx = xd[b];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<const uint2*>(row + b * 16 + j);
            int dot = __dp4a((int)sext4(q.x & 0x0F0F0F0Fu), lo.x, 0);
            dot = __dp4a((int)sext4(q.y & 0x0F0F0F0Fu), lo.y, dot);
            dot = __dp4a((int)sext4((q.x >> 4) & 0x0F0F0F0Fu), hi.x, dot);
            dot = __dp4a((int)sext4((q.y >> 4) & 0x0F0F0F0Fu), hi.y, dot);
            sum[r] += q4_0_scale(row, blocks, b) * dx * (float)dot;
        }
    }
    MATVEC_STORE
}

// Q4_K: per sub-block of 32, d × scale × dx × Σ q qx − dmin × min × dx × Σ qx. Lanes as in
// matvec_q4_k: 8 numbers of a sub-block and 8 of the next.
extern "C" __global__ void matvec8_q4_k(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * 144;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        const signed char* x1 = xq + b * 256 + pair * 64 + part;
        const int x1a = i32_at(x1), x1b = i32_at(x1 + 4), x2a = i32_at(x1 + 32), x2b = i32_at(x1 + 36);
        const int sum1 = __dp4a(0x01010101, x1b, __dp4a(0x01010101, x1a, 0));
        const int sum2 = __dp4a(0x01010101, x2b, __dp4a(0x01010101, x2a, 0));
        const float dx1 = xd[b * 8 + 2 * pair], dx2 = xd[b * 8 + 2 * pair + 1];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 144;
            const uint4 head = *reinterpret_cast<const uint4*>(block);
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<const uint2*>(block + 16 + pair * 32 + part);
            const int dot1 = __dp4a((int)(q.y & 0x0F0F0F0Fu), x1b, __dp4a((int)(q.x & 0x0F0F0F0Fu), x1a, 0));
            const int dot2 =
                __dp4a((int)((q.y >> 4) & 0x0F0F0F0Fu), x2b, __dp4a((int)((q.x >> 4) & 0x0F0F0F0Fu), x2a, 0));
            sum[r] += dx1 * (d * s1 * (float)dot1 - dmin * m1 * (float)sum1) +
                      dx2 * (d * s2 * (float)dot2 - dmin * m2 * (float)sum2);
        }
    }
    MATVEC_STORE
}

// Q5_K: Q4_K with the fifth bits put into the bytes before multiplying.
extern "C" __global__ void matvec8_q5_k(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * 176;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 2) {
        const signed char* x1 = xq + b * 256 + pair * 64 + part;
        const int x1a = i32_at(x1), x1b = i32_at(x1 + 4), x2a = i32_at(x1 + 32), x2b = i32_at(x1 + 36);
        const int sum1 = __dp4a(0x01010101, x1b, __dp4a(0x01010101, x1a, 0));
        const int sum2 = __dp4a(0x01010101, x2b, __dp4a(0x01010101, x2a, 0));
        const float dx1 = xd[b * 8 + 2 * pair], dx2 = xd[b * 8 + 2 * pair + 1];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * 176;
            const uint4 head = *reinterpret_cast<const uint4*>(block);
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = *reinterpret_cast<const uint2*>(block + 48 + pair * 32 + part);
            const uint2 qh = *reinterpret_cast<const uint2*>(block + 16 + part);
            const int s = 2 * pair;
            const int q1a = (int)((q.x & 0x0F0F0F0Fu) | (((qh.x >> s) & 0x01010101u) << 4));
            const int q1b = (int)((q.y & 0x0F0F0F0Fu) | (((qh.y >> s) & 0x01010101u) << 4));
            const int q2a = (int)(((q.x >> 4) & 0x0F0F0F0Fu) | (((qh.x >> (s + 1)) & 0x01010101u) << 4));
            const int q2b = (int)(((q.y >> 4) & 0x0F0F0F0Fu) | (((qh.y >> (s + 1)) & 0x01010101u) << 4));
            const int dot1 = __dp4a(q1b, x1b, __dp4a(q1a, x1a, 0));
            const int dot2 = __dp4a(q2b, x2b, __dp4a(q2a, x2a, 0));
            sum[r] += dx1 * (d * s1 * (float)dot1 - dmin * m1 * (float)sum1) +
                      dx2 * (d * s2 * (float)dot2 - dmin * m2 * (float)sum2);
        }
    }
    MATVEC_STORE
}

// Q6_K: d × scale × dx × Σ (q − 32) qx over each lane's 4 numbers of a run. Lanes as in
// matvec_q6_k.
extern "C" __global__ void matvec8_q6_k(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = (size_t)blocks * Q6K_BYTES;
    const int sub = lane >> 4, half = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 2) {
        const signed char* xb = xq + b * 256 + half * 128 + l;
        int xv[4];
        float dx[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            xv[k] = i32_at(xb + k * 32);
            dx[k] = xd[b * 8 + half * 4 + k];
        }
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at2(block + half * 64 + l), ql1 = u32_at2(block + half * 64 + 32 + l);
            const u32 qh = u32_at2(block + 128 + half * 32 + l);
            const signed char* sc = reinterpret_cast<const signed char*>(block + 192 + half * 8 + l / 16);
            float block_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const u32 low = (((k & 1) ? ql1 : ql0) >> (k < 2 ? 0 : 4)) & 0x0F0F0F0Fu;
                const u32 q = low | (((qh >> (2 * k)) & 0x03030303u) << 4);
                const int dot = __dp4a((int)sext6(q), xv[k], 0);
                block_sum += (float)sc[2 * k] * dx[k] * (float)dot;
            }
            sum[r] += d * block_sum;
        }
    }
    MATVEC_STORE
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800  // (the host side of a full nvcc build sees them too)
// ---- Tensor-core matrix-matrix products (RTX 30 series on) ---------------------------
//
// Tiles of 128 rows × 64 tokens, 128 columns at a time: the weights' columns are
// unpacked to 8-bit whole numbers in shared memory with a scale (and, for Q4_K and Q5_K,
// a minimum) per GROUP columns; x comes already rounded (quantize_q8). Each of the 8
// warps multiplies 32 rows × 32 tokens on the tensor cores (mma.m16n8k16, 8-bit in,
// 32-bit sums out), and the sums are scaled into f32 once per GROUP columns.
//
// Measured on an RTX 5060 Laptop (Q4_K, 8192 × 2048, 512 tokens): ~6 TOPS. The scaling
// is about half the time: with a scale per 32 columns on both sides (as llama.cpp's
// too), each result takes about as many float operations to scale as multiply-adds on
// the tensor cores. Unpacking the weights is most of the rest.

#define M8_ROWS 128
#define M8_TOKENS 64
#define M8_STRIDE 48  // bytes per row of 32 in shared memory: no bank conflicts reading fragments

// The 16 numbers of row r from column k (a multiple of 16) as bytes, with their scale and
// minimum (per 16: Q6_K; per 32 for the rest, given for both halves).
template <int KIND>
__device__ __forceinline__ void unpack16(const u8* row, int k, uint4& q, float& scale, float& min);

template <>
__device__ __forceinline__ void unpack16<K_Q8_0>(const u8* row, int k, uint4& q, float& scale, float& min) {
    const u8* block = row + (k / 32) * 34;
    const u8* p = block + 2 + (k % 32);
    q = make_uint4(u32_at2(p), u32_at2(p + 4), u32_at2(p + 8), u32_at2(p + 12));
    scale = half_at(block);
    min = 0.0f;
}


template <>
__device__ __forceinline__ void unpack16<K_Q4_K>(const u8* row, int k, uint4& q, float& scale, float& min) {
    const u8* block = row + (k / 256) * 144;
    const int n = k % 256, j = n / 32, l = n % 32;
    const uint4 head = *reinterpret_cast<const uint4*>(block);
    float sc, m;
    q4_k_scale_min_w(j, head.y, head.z, head.w, sc, m);
    const uint4 w = *reinterpret_cast<const uint4*>(block + 16 + (j / 2) * 32 + l);
    const int shift = (j & 1) * 4;
    q = make_uint4((w.x >> shift) & 0x0F0F0F0Fu, (w.y >> shift) & 0x0F0F0F0Fu, (w.z >> shift) & 0x0F0F0F0Fu,
                   (w.w >> shift) & 0x0F0F0F0Fu);
    scale = half_lo(head.x) * sc;
    min = half_hi(head.x) * m;
}

template <>
__device__ __forceinline__ void unpack16<K_Q5_K>(const u8* row, int k, uint4& q, float& scale, float& min) {
    const u8* block = row + (k / 256) * 176;
    const int n = k % 256, j = n / 32, l = n % 32;
    const uint4 head = *reinterpret_cast<const uint4*>(block);
    float sc, m;
    q4_k_scale_min_w(j, head.y, head.z, head.w, sc, m);
    const uint4 w = *reinterpret_cast<const uint4*>(block + 48 + (j / 2) * 32 + l);
    const uint4 h = *reinterpret_cast<const uint4*>(block + 16 + l);
    const int shift = (j & 1) * 4;
    q = make_uint4(((w.x >> shift) & 0x0F0F0F0Fu) | (((h.x >> j) & 0x01010101u) << 4),
                   ((w.y >> shift) & 0x0F0F0F0Fu) | (((h.y >> j) & 0x01010101u) << 4),
                   ((w.z >> shift) & 0x0F0F0F0Fu) | (((h.z >> j) & 0x01010101u) << 4),
                   ((w.w >> shift) & 0x0F0F0F0Fu) | (((h.w >> j) & 0x01010101u) << 4));
    scale = half_lo(head.x) * sc;
    min = half_hi(head.x) * m;
}

template <>
__device__ __forceinline__ void unpack16<K_Q6_K>(const u8* row, int k, uint4& q, float& scale, float& min) {
    const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, half = n / 128, run = (n % 128) / 32, l = n % 32;
    const u8* ql = block + half * 64 + (run & 1) * 32 + l;
    const u8* qh = block + 128 + half * 32 + l;
    const int shift = run < 2 ? 0 : 4;
    u32 out[4];
    for (int i = 0; i < 4; i++) {
        const u32 low = (u32_at2(ql + 4 * i) >> shift) & 0x0F0F0F0Fu;
        const u32 high = ((u32_at2(qh + 4 * i) >> (2 * run)) & 0x03030303u) << 4;
        out[i] = sext6(low | high);
    }
    q = make_uint4(out[0], out[1], out[2], out[3]);
    scale = half_at(block + 208) * (float)reinterpret_cast<const signed char*>(block + 192 + half * 8 + l / 16 + 2 * run)[0];
    min = 0.0f;
}

__device__ __forceinline__ void mma_s8(int (&c)[4], u32 a0, u32 a1, u32 b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// 16 rows × 8 tokens × 32 columns: A's registers hold columns tig*4.. and 16 + tig*4.. of
// rows gid and gid + 8; B's, columns tig*4.. and 16 + tig*4.. of token gid.
__device__ __forceinline__ void mma_s8_k32(int (&c)[4], const u32 (&a)[4], const u32 (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int KIND, int GROUP>
__device__ void matmul8(const u8* w, size_t row_bytes, const signed char* xq, const float* xd, const float* xs,
                        float* y, int rows, int cols, int tokens) {
    // 128 columns a step: 64 tensor-core products per warp between syncs. Rows of 128
    // bytes padded to 144 in shared memory, so reading fragments has no bank conflicts.
    constexpr int K = 128, KS = K + 16, NG = K / GROUP, NX = K / 32;
    __shared__ __align__(16) u8 wq[M8_ROWS][KS];
    __shared__ __align__(16) u8 xqs[M8_TOKENS][KS];
    // Scales by group, then row or token, so a warp reads them in pairs (a row's scale
    // with its minimum; a thread's two tokens' scales or sums) without bank conflicts.
    // x_sum is per 32 columns: the pairs of 16 added as they're stored.
    __shared__ __align__(16) float2 w_sm[NG][M8_ROWS];
    __shared__ __align__(16) float x_scale[NX][M8_TOKENS], x_sum[NX][M8_TOKENS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * M8_ROWS, tok0 = blockIdx.y * M8_TOKENS;
    const int wr = (warp & 3) * 32, wt = (warp >> 2) * 32;  // this warp's rows and tokens in the tile
    constexpr bool has_min = KIND == K_Q4_K || KIND == K_Q5_K;
    float acc[2][4][4] = {};
    const int xd_cols = cols / 32, xs_cols = cols / 16;

    // The next step's raw bytes are read into registers before this step's products, and
    // unpacked into shared memory after them, so the reads' wait overlaps the arithmetic.
    // (Q8_0's 2-byte-aligned words are put together as they're read.)
    struct Raw {
        uint4 head, v[2], h[2];  // Q4_K, Q5_K: a pair of sub-blocks; Q6_K: v low, h high bits
        uint4 q[4];              // Q4_0, Q8_0: 4 runs of 16
        u32 sc[4];               // their scales (Q4_0: f16 bits; Q8_0: f32 bits); Q6_K: d, scales
        bool ok[4];
        uint4 xq[2];
        float xsum[2], xscale[2];
    } raw;
    auto fetch = [&](int k0) {
        if constexpr (KIND == K_Q4_K || KIND == K_Q5_K) {
            const int r = tid >> 1, p = tid & 1, k = k0 + p * 64;
            raw.ok[0] = row0 + r < rows && k < cols;
            if (raw.ok[0]) {
                constexpr int bytes = KIND == K_Q4_K ? 144 : 176, qs = KIND == K_Q4_K ? 16 : 48;
                const u8* block = w + (size_t)(row0 + r) * row_bytes + (k / 256) * bytes;
                const int j = (k % 256) / 32;
                raw.head = *reinterpret_cast<const uint4*>(block);
                for (int i = 0; i < 2; i++) {
                    raw.v[i] = *reinterpret_cast<const uint4*>(block + qs + (j / 2) * 32 + i * 16);
                    if (KIND == K_Q5_K) raw.h[i] = *reinterpret_cast<const uint4*>(block + 16 + i * 16);
                }
            }
        } else if constexpr (KIND == K_Q6_K) {
            const int r = tid >> 1, p = tid & 1;
            raw.ok[0] = row0 + r < rows && k0 < cols;
            if (raw.ok[0]) {
                const u8* block = w + (size_t)(row0 + r) * row_bytes + (k0 / 256) * Q6K_BYTES;
                const int half = (k0 % 256) / 128;
                raw.sc[0] = *reinterpret_cast<const u16*>(block + 208);
                raw.sc[1] = *reinterpret_cast<const u16*>(block + 192 + half * 8 + 2 * p);
                raw.sc[2] = *reinterpret_cast<const u16*>(block + 192 + half * 8 + 2 * (p + 2));
                for (int i = 0; i < 2; i++) {
                    raw.v[i] = *reinterpret_cast<const uint4*>(block + half * 64 + p * 32 + i * 16);
                    raw.h[i] = *reinterpret_cast<const uint4*>(block + 128 + half * 32 + i * 16);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int c = tid + i * 256, r = c >> 3, h = c & 7, k = k0 + h * 16;
                raw.ok[i] = row0 + r < rows && k < cols;
                if (raw.ok[i]) {
                    const u8* row = w + (size_t)(row0 + r) * row_bytes;
                    if constexpr (KIND == K_Q4_0) {
                        const int blocks = cols / 32, b = k / 32;
                        raw.q[i] = *reinterpret_cast<const uint4*>(row + b * 16);
                        raw.sc[i] = *reinterpret_cast<const u16*>(row + (size_t)blocks * 16 + b * 2);
                    } else {
                        float scale, min;
                        unpack16<KIND>(row, k, raw.q[i], scale, min);
                        raw.sc[i] = __float_as_uint(scale);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * 256, t = c >> 3, h = c & 7, k = k0 + h * 16;
            raw.xq[i] = make_uint4(0, 0, 0, 0);
            raw.xsum[i] = raw.xscale[i] = 0.0f;
            if (tok0 + t < tokens && k < cols) {
                raw.xq[i] = *reinterpret_cast<const uint4*>(xq + (size_t)(tok0 + t) * cols + k);
                raw.xsum[i] = xs[(size_t)(tok0 + t) * xs_cols + k / 16];
                raw.xscale[i] = xd[(size_t)(tok0 + t) * xd_cols + k / 32];
            }
        }
    };
    auto store = [&](int k0) {
        if constexpr (KIND == K_Q4_K || KIND == K_Q5_K) {
            // A pair of sub-blocks a thread: 32 bytes hold 32 numbers of one sub-block (low
            // nibbles) and 32 of the next (high), so one scale header gives 64 numbers.
            const int r = tid >> 1, p = tid & 1, k = k0 + p * 64;
            uint4 lo[2] = {}, hi[2] = {};
            float s1 = 0.0f, m1 = 0.0f, s2 = 0.0f, m2 = 0.0f;
            if (raw.ok[0]) {
                const int j = (k % 256) / 32;  // the pair's first sub-block (even)
                const uint4 head = raw.head;
                q4_k_scale_min_w(j, head.y, head.z, head.w, s1, m1);
                q4_k_scale_min_w(j + 1, head.y, head.z, head.w, s2, m2);
                const float d = half_lo(head.x), dmin = half_hi(head.x);
                s1 *= d;
                s2 *= d;
                m1 *= dmin;
                m2 *= dmin;
                for (int i = 0; i < 2; i++) {
                    const uint4 v = raw.v[i];
                    lo[i] = make_uint4(v.x & 0x0F0F0F0Fu, v.y & 0x0F0F0F0Fu, v.z & 0x0F0F0F0Fu, v.w & 0x0F0F0F0Fu);
                    hi[i] = make_uint4((v.x >> 4) & 0x0F0F0F0Fu, (v.y >> 4) & 0x0F0F0F0Fu, (v.z >> 4) & 0x0F0F0F0Fu,
                                       (v.w >> 4) & 0x0F0F0F0Fu);
                    if (KIND == K_Q5_K) {
                        // Fifth bits: bit j of qh[l] for the first sub-block, j + 1 for the next.
                        const uint4 h = raw.h[i];
                        lo[i].x |= ((h.x >> j) & 0x01010101u) << 4;
                        lo[i].y |= ((h.y >> j) & 0x01010101u) << 4;
                        lo[i].z |= ((h.z >> j) & 0x01010101u) << 4;
                        lo[i].w |= ((h.w >> j) & 0x01010101u) << 4;
                        hi[i].x |= ((h.x >> (j + 1)) & 0x01010101u) << 4;
                        hi[i].y |= ((h.y >> (j + 1)) & 0x01010101u) << 4;
                        hi[i].z |= ((h.z >> (j + 1)) & 0x01010101u) << 4;
                        hi[i].w |= ((h.w >> (j + 1)) & 0x01010101u) << 4;
                    }
                }
            }
            *reinterpret_cast<uint4*>(&wq[r][p * 64]) = lo[0];
            *reinterpret_cast<uint4*>(&wq[r][p * 64 + 16]) = lo[1];
            *reinterpret_cast<uint4*>(&wq[r][p * 64 + 32]) = hi[0];
            *reinterpret_cast<uint4*>(&wq[r][p * 64 + 48]) = hi[1];
            w_sm[2 * p][r] = make_float2(s1, m1);
            w_sm[2 * p + 1][r] = make_float2(s2, m2);
        } else if constexpr (KIND == K_Q6_K) {
            // Q6_K: a step of 128 columns is half a block, 4 runs of 32. One thread per row
            // and pair of runs (p and p + 2), which share their 4-bit bytes (low and high
            // nibbles) and all the high bits.
            const int r = tid >> 1, p = tid & 1;
            uint4 lo[2] = {}, hi[2] = {};
            float sc[4] = {};
            if (raw.ok[0]) {
                const float d = __half2float(__ushort_as_half((unsigned short)raw.sc[0]));
                sc[0] = d * (float)(signed char)(raw.sc[1] & 0xFF);
                sc[1] = d * (float)(signed char)(raw.sc[1] >> 8);
                sc[2] = d * (float)(signed char)(raw.sc[2] & 0xFF);
                sc[3] = d * (float)(signed char)(raw.sc[2] >> 8);
                for (int i = 0; i < 2; i++) {
                    const uint4 l = raw.v[i], h = raw.h[i];
                    const u32 lw[4] = {l.x, l.y, l.z, l.w}, hw[4] = {h.x, h.y, h.z, h.w};
                    u32 a[4], b[4];
                    for (int x = 0; x < 4; x++) {
                        a[x] = sext6((lw[x] & 0x0F0F0F0Fu) | (((hw[x] >> (2 * p)) & 0x03030303u) << 4));
                        b[x] = sext6(((lw[x] >> 4) & 0x0F0F0F0Fu) | (((hw[x] >> (2 * p + 4)) & 0x03030303u) << 4));
                    }
                    lo[i] = make_uint4(a[0], a[1], a[2], a[3]);
                    hi[i] = make_uint4(b[0], b[1], b[2], b[3]);
                }
            }
            *reinterpret_cast<uint4*>(&wq[r][p * 32]) = lo[0];
            *reinterpret_cast<uint4*>(&wq[r][p * 32 + 16]) = lo[1];
            *reinterpret_cast<uint4*>(&wq[r][(p + 2) * 32]) = hi[0];
            *reinterpret_cast<uint4*>(&wq[r][(p + 2) * 32 + 16]) = hi[1];
            w_sm[2 * p][r].x = sc[0];
            w_sm[2 * p + 1][r].x = sc[1];
            w_sm[2 * (p + 2)][r].x = sc[2];
            w_sm[2 * (p + 2) + 1][r].x = sc[3];
        } else {
            // The weights: 128 rows × 8 runs of 16, 4 runs a thread.
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int c = tid + i * 256, r = c >> 3, h = c & 7, k = k0 + h * 16;
                uint4 q = make_uint4(0, 0, 0, 0);
                float scale = 0.0f;
                if (raw.ok[i]) {
                    if constexpr (KIND == K_Q4_0) {
                        const int shift = (k % 32) ? 4 : 0;  // the first 16 numbers are the low nibbles
                        const uint4 v = raw.q[i];
                        q = make_uint4(sext4((v.x >> shift) & 0x0F0F0F0Fu), sext4((v.y >> shift) & 0x0F0F0F0Fu),
                                       sext4((v.z >> shift) & 0x0F0F0F0Fu), sext4((v.w >> shift) & 0x0F0F0F0Fu));
                        scale = __half2float(__ushort_as_half((unsigned short)raw.sc[i]));
                    } else {
                        q = raw.q[i];
                        scale = __uint_as_float(raw.sc[i]);
                    }
                }
                *reinterpret_cast<uint4*>(&wq[r][h * 16]) = q;
                if (GROUP == 16 || (h & 1) == 0) w_sm[GROUP == 16 ? h : h / 2][r].x = scale;
            }
        }
        // x: 64 tokens × 8 runs of 16, 2 a thread.
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * 256, t = c >> 3, h = c & 7;
            *reinterpret_cast<uint4*>(&xqs[t][h * 16]) = raw.xq[i];
            // The runs h and h + 1 of a group are on neighbouring lanes.
            const float sum32 = raw.xsum[i] + __shfl_xor_sync(0xFFFFFFFFu, raw.xsum[i], 1);
            if ((h & 1) == 0) {
                x_scale[h / 2][t] = raw.xscale[i];
                x_sum[h / 2][t] = sum32;
            }
        }
    };

    // Not for Q6_K: it runs two blocks per SM, and the raw bytes would spill there (one
    // block with the read ahead measured 3% slower than two without).
    constexpr bool prefetch = KIND != K_Q6_K;
    if constexpr (prefetch) fetch(0);
    for (int k0 = 0; k0 < cols; k0 += K) {
        if constexpr (!prefetch) fetch(k0);
        store(k0);
        __syncthreads();
        if constexpr (prefetch)
            if (k0 + K < cols) fetch(k0 + K);

#pragma unroll
        for (int g = 0; g < NG; g++) {
            int c[2][4][4] = {};
            if (GROUP == 32) {
                // A whole group in one product a tile (mma.m16n8k32). (Loading the
                // fragments with ldmatrix measured 2% slower than these loads.)
                const int kk = g * 32;
                u32 a[2][4], b[4][2];
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    const u8* r0 = &wq[wr + mi * 16 + gid][kk + tig * 4];
                    const u8* r1 = &wq[wr + mi * 16 + gid + 8][kk + tig * 4];
                    a[mi][0] = *reinterpret_cast<const u32*>(r0);
                    a[mi][1] = *reinterpret_cast<const u32*>(r1);
                    a[mi][2] = *reinterpret_cast<const u32*>(r0 + 16);
                    a[mi][3] = *reinterpret_cast<const u32*>(r1 + 16);
                }
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    const u8* t = &xqs[wt + ni * 8 + gid][kk + tig * 4];
                    b[ni][0] = *reinterpret_cast<const u32*>(t);
                    b[ni][1] = *reinterpret_cast<const u32*>(t + 16);
                }
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
#pragma unroll
                    for (int ni = 0; ni < 4; ni++) mma_s8_k32(c[mi][ni], a[mi], b[ni]);
            } else {
#pragma unroll
                for (int kk = g * GROUP; kk < (g + 1) * GROUP; kk += 16) {
                    u32 a[2][2], b[4];
#pragma unroll
                    for (int mi = 0; mi < 2; mi++) {
                        a[mi][0] = *reinterpret_cast<const u32*>(&wq[wr + mi * 16 + gid][kk + tig * 4]);
                        a[mi][1] = *reinterpret_cast<const u32*>(&wq[wr + mi * 16 + gid + 8][kk + tig * 4]);
                    }
#pragma unroll
                    for (int ni = 0; ni < 4; ni++)
                        b[ni] = *reinterpret_cast<const u32*>(&xqs[wt + ni * 8 + gid][kk + tig * 4]);
#pragma unroll
                    for (int mi = 0; mi < 2; mi++)
#pragma unroll
                        for (int ni = 0; ni < 4; ni++) mma_s8(c[mi][ni], a[mi][0], a[mi][1], b[ni]);
                }
            }
            // Scale this group's whole-number sums into the results, reading each row's and
            // token's scales once.
            float ws[2][2], wm[2][2], xsc[4][2], xsm[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = wr + mi * 16 + gid + h * 8;
                    if constexpr (has_min) {
                        const float2 sm = w_sm[g][r];
                        ws[mi][h] = sm.x;
                        wm[mi][h] = -sm.y;
                    } else {
                        ws[mi][h] = w_sm[g][r].x;
                        wm[mi][h] = 0.0f;
                    }
                }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const int t = wt + ni * 8 + tig * 2;
                const float2 s = *reinterpret_cast<const float2*>(&x_scale[g * GROUP / 32][t]);
                xsc[ni][0] = s.x;
                xsc[ni][1] = s.y;
                if constexpr (has_min) {
                    const float2 m = *reinterpret_cast<const float2*>(&x_sum[g][t]);
                    xsm[ni][0] = m.x;
                    xsm[ni][1] = m.y;
                } else {
                    xsm[ni][0] = xsm[ni][1] = 0.0f;
                }
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        // Two fused multiply-adds straight into the sum (the minimum's term
                        // added on its own, not first subtracted from the product).
                        const int h = i >> 1, j = i & 1;
                        acc[mi][ni][i] = fmaf((float)c[mi][ni][i], ws[mi][h] * xsc[ni][j], acc[mi][ni][i]);
                        if (has_min) acc[mi][ni][i] = fmaf(wm[mi][h], xsm[ni][j], acc[mi][ni][i]);
                    }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = row0 + wr + mi * 16 + gid + (i >> 1) * 8, t = tok0 + wt + ni * 8 + tig * 2 + (i & 1);
                if (r < rows && t < tokens) y[(size_t)t * rows + r] = acc[mi][ni][i];
            }
}

#define MATMUL8(name, KIND, GROUP, ROW_BYTES, BLOCKS)                                                      \
    extern "C" __global__ void __launch_bounds__(256, BLOCKS)                                            \
        name(const u8* w, const signed char* xq, const float* xd, const float* xs, float* y, int rows,   \
             int cols, int tokens) {                                                                     \
        matmul8<KIND, GROUP>(w, (size_t)(ROW_BYTES), xq, xd, xs, y, rows, cols, tokens);                 \
    }

// One block per SM, with the next step's bytes read ahead into registers (fetch): faster
// than two blocks without (Q4_0 -6%, Q8_0 -25%), and Q4_K and Q5_K would spill with two.
// Q6_K keeps two blocks and no read-ahead.
MATMUL8(matmul8_q4_0, K_Q4_0, 32, q4_0_row_bytes(cols), 1)
MATMUL8(matmul8_q8_0, K_Q8_0, 32, cols / 32 * 34, 1)
MATMUL8(matmul8_q4_k, K_Q4_K, 32, cols / 256 * 144, 1)
MATMUL8(matmul8_q5_k, K_Q5_K, 32, cols / 256 * 176, 1)
MATMUL8(matmul8_q6_k, K_Q6_K, 16, cols / 256 * Q6K_BYTES, 2)

// ---- Attention on the tensor cores (8-bit mode: an f16 cache) ------------------------
//
// Flash attention for a prompt's tokens: a block per query head and 16 × ROWS tokens.
// Keys and values come KC positions at a time into shared memory by cp.async, as they
// are in the cache (no transposing: the second product reads values with
// ldmatrix.trans). The loads overlap the arithmetic: a chunk's values load while its
// S = Q Kᵀ runs, and the next chunk's keys while P V runs, so one buffer of each is
// enough. S and O += P V run on the tensor cores in f16 with f32 sums (mma.m16n8k16),
// with an online softmax in f32 in between. The softmax's numbers come out of the first
// product laid out exactly as the second takes them in, so P never goes through shared
// memory.
//
// Wide heads (512) keep too many sums for one warp, so SPLIT warps share each 16 rows:
// each works out S over its 1/SPLIT of the head's numbers, the parts are added through
// shared memory (in the same order by every warp, so all of them see the same S), and
// each makes its own 1/SPLIT of the output. S is worked out once.
// Shared memory: dynamic, attention_mma_bytes on the host.

__device__ __forceinline__ void mma_f16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const u32*>(&h);
}

__device__ __forceinline__ u32 u32_of(const __half* p) { return *reinterpret_cast<const u32*>(p); }

__device__ __forceinline__ u32 shared_address(const void* p) {
    return static_cast<u32>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared memory without going through registers; zeros when
// `valid` is false (nothing is read then).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_address(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Waits until at most one group this thread committed is still in flight.
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

// Four 8 × 8 f16 matrices, transposed: rows (positions) from lanes 0–7, 8–15, 16–23,
// 24–31; each thread gets two B fragments of mma.m16n8k16 for two tiles of 8 columns.
__device__ __forceinline__ void ldmatrix_x4_trans(u32 (&r)[4], const __half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(shared_address(p)));
}

template <int D, int ROWS, int SPLIT, int KC>
__device__ void attention_mma(const float* q, const __half* keys, const __half* values, float* out,
                              const int* pos, int window, int kv_dim, int group, float scale, int tokens) {
    constexpr int WARPS = ROWS * SPLIT, THREADS = 32 * WARPS, QT = 16 * ROWS;
    constexpr int DS = D + 8;            // padded rows: no bank conflicts
    constexpr int DP = D / SPLIT;        // the numbers of a head each warp takes
    constexpr int NS = KC / 8 * 4;       // a thread's share of S
    static_assert(DP % 16 == 0 && KC % 16 == 0, "tile sizes");
    extern __shared__ __align__(16) unsigned char smem[];
    __half* qs = reinterpret_cast<__half*>(smem);  // QT × DS
    __half* ks = qs + QT * DS;                      // KC × DS
    __half* vs = ks + KC * DS;                      // KC × DS
    float* sx = reinterpret_cast<float*>(vs + KC * DS);  // SPLIT > 1: [ROWS][SPLIT][NS][32]

    const int head = blockIdx.x, heads = gridDim.x;
    const int t0 = (gridDim.y - 1 - blockIdx.y) * QT;  // the longest rows first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int rw = warp / SPLIT, sp = warp % SPLIT, dp = sp * DP;
    const int p0 = *pos, kv = (head / group) * D;
    const int last_row = p0 + min(t0 + QT, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    auto load = [&](__half* dst, const __half* src, int start) {
        for (int i = tid; i < KC * D / 8; i += THREADS) {
            const int r = i / (D / 8), d = (i % (D / 8)) * 8;
            const bool valid = start + r <= last_row;
            cp_async16(dst + r * DS + d, src + (size_t)(valid ? start + r : 0) * kv_dim + kv + d, valid);
        }
        cp_async_commit();
    };
    load(ks, keys, lo);

    // The queries, as f16.
    for (int i = tid; i < QT * D / 4; i += THREADS) {
        const int r = (i * 4) / D, d = (i * 4) % D;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t0 + r < tokens) v = f4(q + ((size_t)(t0 + r) * heads + head) * D + d);
        *reinterpret_cast<__half2*>(&qs[r * DS + d]) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(&qs[r * DS + d + 2]) = __floats2half2_rn(v.z, v.w);
    }

    // This thread's two rows (gid and gid + 8 of the warp's 16) and where they look.
    const int row0 = rw * 16 + gid;
    int at[2], first[2];
    bool live[2];
    for (int h = 0; h < 2; h++) {
        const int r = row0 + h * 8;
        live[h] = t0 + r < tokens;
        at[h] = p0 + t0 + r;
        first[h] = window > 0 ? max(0, at[h] + 1 - window) : 0;
    }
    float o[DP / 8][4];
#pragma unroll
    for (int n = 0; n < DP / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (int start = lo; start <= last_row; start += KC) {
        load(vs, values, start);  // the last chunk's P V is done with vs (the sync below)
        cp_async_wait1();         // this chunk's keys are in
        __syncthreads();

        // S = Q Kᵀ for the warp's 16 rows × KC positions (over its part of the head).
        float s[KC / 8][4];
#pragma unroll
        for (int n = 0; n < KC / 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll 4
        for (int kk = dp; kk < dp + DP; kk += 16) {
            const __half* qa = qs + row0 * DS + kk + tig * 2;
            const u32 a0 = u32_of(qa), a1 = u32_of(qa + 8 * DS), a2 = u32_of(qa + 8), a3 = u32_of(qa + 8 * DS + 8);
#pragma unroll
            for (int n = 0; n < KC / 8; n++) {
                const __half* kb = ks + (n * 8 + gid) * DS + kk + tig * 2;
                mma_f16(s[n], a0, a1, a2, a3, u32_of(kb), u32_of(kb + 8));
            }
        }
        if constexpr (SPLIT > 1) {
            float* mine = sx + ((rw * SPLIT + sp) * NS) * 32 + lane;
#pragma unroll
            for (int n = 0; n < KC / 8; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) mine[(n * 4 + i) * 32] = s[n][i];
            __syncthreads();
            const float* all = sx + (rw * SPLIT * NS) * 32 + lane;
#pragma unroll
            for (int n = 0; n < KC / 8; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float sum = all[(n * 4 + i) * 32];
#pragma unroll
                    for (int w = 1; w < SPLIT; w++) sum += all[((w * NS) + n * 4 + i) * 32];
                    s[n][i] = sum;
                }
        }
        __syncthreads();  // every warp is done with ks (and sx)
        if (start + KC <= last_row) {
            load(ks, keys, start + KC);
        } else {
            cp_async_commit();  // an empty group, so the waits count the same
        }

        // The online softmax, per row (the row's KC positions are spread over 4 lanes).
        float row_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < KC / 8; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int h = i >> 1, key = start + n * 8 + tig * 2 + (i & 1);
                const bool seen = live[h] && key <= at[h] && key >= first[h];
                s[n][i] = seen ? s[n][i] * scale : -INFINITY;
                row_max[h] = fmaxf(row_max[h], s[n][i]);
            }
        float correction[2], row_sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int h = 0; h < 2; h++) {
            row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 1));
            row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 2));
            const float new_max = fmaxf(m[h], row_max[h]);
            correction[h] = new_max == -INFINITY ? 1.0f : expf(m[h] - new_max);
            m[h] = new_max;
        }
#pragma unroll
        for (int n = 0; n < KC / 8; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int h = i >> 1;
                s[n][i] = s[n][i] == -INFINITY ? 0.0f : expf(s[n][i] - m[h]);
                row_sum[h] += s[n][i];
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 1);
            row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 2);
            l[h] = l[h] * correction[h] + row_sum[h];
        }
        // Once a row's largest score settles, the correction is 1 (multiplying by it
        // changes nothing, so skipping it gives the same numbers).
        if (__any_sync(0xffffffffu, correction[0] != 1.0f || correction[1] != 1.0f)) {
#pragma unroll
            for (int n = 0; n < DP / 8; n++) {
                o[n][0] *= correction[0];
                o[n][1] *= correction[0];
                o[n][2] *= correction[1];
                o[n][3] *= correction[1];
            }
        }

        cp_async_wait1();  // this chunk's values are in
        __syncthreads();

        // O += P V: P's numbers are already where the A operand wants them.
        const int mat = lane >> 3, mrow = lane & 7;
#pragma unroll
        for (int kk = 0; kk < KC / 16; kk++) {
            const u32 a0 = pack_half2(s[2 * kk][0], s[2 * kk][1]), a1 = pack_half2(s[2 * kk][2], s[2 * kk][3]);
            const u32 a2 = pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            const u32 a3 = pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            const __half* vrow = vs + (kk * 16 + (mat & 1) * 8 + mrow) * DS + dp + (mat >> 1) * 8;
#pragma unroll
            for (int n = 0; n < DP / 8; n += 2) {
                u32 b[4];
                ldmatrix_x4_trans(b, vrow + n * 8);
                mma_f16(o[n], a0, a1, a2, a3, b[0], b[1]);
                mma_f16(o[n + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
        __syncthreads();  // every warp is done with vs
    }

#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (!live[h]) continue;
        float* dst = out + ((size_t)(t0 + row0 + h * 8) * heads + head) * D + dp;
#pragma unroll
        for (int n = 0; n < DP / 8; n++) {
            dst[n * 8 + tig * 2] = o[n][2 * h] / l[h];
            dst[n * 8 + tig * 2 + 1] = o[n][2 * h + 1] / l[h];
        }
    }
}

#define ATTENTION_MMA(name, D, ROWS, SPLIT, KC)                                                              \
    extern "C" __global__ void __launch_bounds__(32 * ROWS * SPLIT) name(                                    \
        const float* q, const __half* keys, const __half* values, float* out, const int* pos, int window,    \
        int hd, int kv_dim, int group, float scale, int tokens) {                                            \
        attention_mma<D, ROWS, SPLIT, KC>(q, keys, values, out, pos, window, kv_dim, group, scale, tokens);  \
    }

// Head sizes 64 to 512: (rows of 16 tokens, warps per row, positions per chunk), the
// fastest of the shapes measured on an RTX 5060 Laptop. The host side's ATTENTION_MMA
// table must match.
ATTENTION_MMA(attention_mma_64, 64, 4, 1, 32)
ATTENTION_MMA(attention_mma_128, 128, 4, 1, 32)
ATTENTION_MMA(attention_mma_256, 256, 4, 1, 32)
ATTENTION_MMA(attention_mma_512, 512, 2, 2, 16)
#endif

// ---- Everything else in a layer, for `tokens` rows at a time --------------------------

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

// The residual stream between steps, one block per token (row of n numbers):
//   x = (x + p') × scale, p' = p, or RMSNorm(p) × post with a `post` weight (Gemma);
//   then h = RMSNorm(x) × next, the input of the next step (none if `next` is null).
// `p` may be null (just the norm). Run with 256 threads: the sums then come out exactly
// as rms_norm's.
extern "C" __global__ void residual_norm(float* x, const float* p, const float* post, float scale,
                                         const float* next, float* h, int n, float eps) {
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * n;
    // Rows of up to R numbers a thread (6,144 at 256 threads): every number read into
    // registers at once, so the reads overlap instead of each waiting on the last (one
    // token's norm is one block: its time is the reads' latency). The same sums in the
    // same order as the loop below.
    constexpr int R = 24;
    const int bd = blockDim.x, tid = threadIdx.x;
    if (n <= R * bd) {
        float xv[R], pv[R], wv[R];
        if (p) p += (size_t)blockIdx.x * n;
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int i = tid + j * bd;
            xv[j] = i < n ? x[i] : 0.0f;
            pv[j] = p && i < n ? p[i] : 0.0f;
            wv[j] = post && i < n ? post[i] : 0.0f;
        }
        float p_scale = 1.0f;
        if (p && post) {
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < R; j++)
                if (tid + j * bd < n) sum += pv[j] * pv[j];
            sum = block_sum(sum, scratch);
            p_scale = 1.0f / sqrtf(sum / (float)n + eps);
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int i = tid + j * bd;
            if (i >= n) break;
            float v = xv[j];
            if (p) v += post ? pv[j] * p_scale * wv[j] : pv[j];
            if (scale != 1.0f) v *= scale;
            xv[j] = v;
            x[i] = v;
            sum += v * v;
        }
        if (!next) return;
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int i = tid + j * bd;
            wv[j] = i < n ? next[i] : 0.0f;
        }
        sum = block_sum(sum, scratch);
        const float s = 1.0f / sqrtf(sum / (float)n + eps);
        h += (size_t)blockIdx.x * n;
#pragma unroll
        for (int j = 0; j < R; j++) {
            const int i = tid + j * bd;
            if (i < n) h[i] = xv[j] * s * wv[j];
        }
        return;
    }
    float p_scale = 1.0f;
    if (p) {
        p += (size_t)blockIdx.x * n;
        if (post) {
            float sum = 0.0f;
            for (int i = threadIdx.x; i < n; i += blockDim.x) sum += p[i] * p[i];
            sum = block_sum(sum, scratch);
            p_scale = 1.0f / sqrtf(sum / (float)n + eps);
        }
    }
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (p) v += post ? p[i] * p_scale * post[i] : p[i];
        if (scale != 1.0f) v *= scale;
        x[i] = v;
        sum += v * v;
    }
    if (!next) return;
    sum = block_sum(sum, scratch);
    const float s = 1.0f / sqrtf(sum / (float)n + eps);
    h += (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[i] = x[i] * s * next[i];
}

// Gets tokens' queries, keys and values ready for attention, one block per head and run
// of QKV_TOKENS tokens (a thread per number of a head, up to 256): an optional RMSNorm per
// head, then RoPE at the token's position on the first `rope_dims` numbers (adjacent
// pairs, or halves with `neox`); keys and values go into the cache at that position.
// Blocks: `heads` query heads, then (with a cache) `kv_heads` key heads and `kv_heads`
// value heads; blockIdx.y is the run of tokens. Each pair's frequency (a powf, the
// costly part) is worked out once for the block's tokens.
//   q: token t's head h at q_src + t × q_stride + h × q_head_stride (Qwen's queries sit in
//      [query | gate] pairs), normed with `q_norm` (may be null), out to q_dst.
//   k, v: token t at k_src/v_src + t × kv_dim; k normed with `k_norm` (may be null), v
//      with no weight if `v_norm`.
// `factors` (rope_freqs) may be null. Positions are *pos + t. With `half_cache` the cache
// holds f16 (8-bit mode), otherwise f32.
#define QKV_TOKENS 8
extern "C" __global__ void qkv_prep(const float* q_src, int q_stride, int q_head_stride, const float* q_norm,
                                    float* q_dst, const float* k_src, const float* k_norm, const float* v_src,
                                    int v_norm, void* k_cache, void* v_cache, int heads, int kv_heads,
                                    int hd, int rope_dims, float base, const float* factors, int neox,
                                    const int* pos, float eps, int half_cache, int tokens) {
    __shared__ float scratch[32];
    __shared__ float buf[512];
    const int slot = blockIdx.x, kv_dim = kv_heads * hd;
    const int t0 = blockIdx.y * QKV_TOKENS, t1 = min(t0 + QKV_TOKENS, tokens);
    const int kind = slot < heads ? 0 : slot < heads + kv_heads ? 1 : 2;  // q, k, v
    const int h = kind == 0 ? slot : kind == 1 ? slot - heads : slot - heads - kv_heads;
    const float* weight = kind == 0 ? q_norm : kind == 1 ? k_norm : nullptr;
    const bool norm = kind == 2 ? v_norm != 0 : weight != nullptr;
    const bool rope = kind != 2;

    // Work items: each rotated pair (its frequency fixed for the block), then the numbers
    // past rope_dims as they are.
    const int pairs = rope ? rope_dims / 2 : 0, rotated = 2 * pairs;
    const int items = pairs + hd - rotated;
    float freq = 0.0f;  // (blockDim ≥ hd / 2 ≥ pairs: a thread has at most one pair)
    if ((int)threadIdx.x < pairs) freq = powf(base, -2.0f * (float)threadIdx.x / (float)rope_dims);

    for (int t = t0; t < t1; t++) {
        const int at = *pos + t;
        const float* src;
        float* dst = nullptr;
        __half* half_dst = nullptr;  // instead of dst, for an f16 cache
        if (kind == 0) {
            src = q_src + (size_t)t * q_stride + (size_t)h * q_head_stride;
            dst = q_dst + ((size_t)t * heads + h) * hd;
        } else {
            src = (kind == 1 ? k_src : v_src) + (size_t)t * kv_dim + h * hd;
            const size_t at_cache = (size_t)at * kv_dim + h * hd;
            void* cache = kind == 1 ? k_cache : v_cache;
            if (half_cache)
                half_dst = static_cast<__half*>(cache) + at_cache;
            else
                dst = static_cast<float*>(cache) + at_cache;
        }
        auto put = [&](int i, float v) {
            if (half_dst)
                half_dst[i] = __float2half(v);
            else
                dst[i] = v;
        };
        __syncthreads();  // the last token is done with buf
        float sum = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            buf[i] = src[i];
            sum += buf[i] * buf[i];
        }
        if (norm) {
            sum = block_sum(sum, scratch);
            const float scale = 1.0f / sqrtf(sum / (float)hd + eps);
            for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = buf[i] * scale * (weight ? weight[i] : 1.0f);
        }
        __syncthreads();
        for (int j = threadIdx.x; j < items; j += blockDim.x) {
            if (j >= pairs) {
                const int i = rotated + (j - pairs);
                put(i, buf[i]);
                continue;
            }
            const int p = j;  // the pair: numbers i0 and i1 (halves with neox, else adjacent)
            const int i0 = neox ? p : 2 * p, i1 = neox ? p + pairs : 2 * p + 1;
            float theta = (float)at * freq;
            if (factors) theta /= factors[p];
            float s, c;
            sincosf(theta, &s, &c);
            const float a = buf[i0], b = buf[i1];
            put(i0, a * c - b * s);
            put(i1, a * s + b * c);
        }
    }
}

// Causal attention, one block of 128 threads per query head and token. Token t (at
// position *pos + t) looks at cached positions first..its own, first > 0 for a sliding
// window of `window` (0 for none). Keys are taken 128 at a time with a running maximum
// (online softmax), so no scratch memory is needed however long the context; up to 128
// positions it's exactly the plain softmax. Head size at most 512. The cache holds KV:
// float (attention) or __half (attention_h, 8-bit mode).
__device__ __forceinline__ float4 load4(const float* p) { return f4(p); }
__device__ __forceinline__ float4 load4(const __half* p) {
    const uint2 v = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float load1(const float* p) { return *p; }
__device__ __forceinline__ float load1(const __half* p) { return __half2float(*p); }

template <typename KV>
__device__ void attention_one(const float* q, const KV* keys, const KV* values, float* out, const int* pos,
                              int window, int hd, int kv_dim, int group, float scale) {
    __shared__ float scratch[32];
    __shared__ __align__(16) float qs[512];
    __shared__ float ps[128];
    const int head = blockIdx.x, heads = gridDim.x, t = blockIdx.y, tid = threadIdx.x;
    const int len = *pos + t + 1;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int kv = (head / group) * hd;
    const float* qh = q + ((size_t)t * heads + head) * hd;
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = qh[d];
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_so_far = -INFINITY, total = 0.0f;
    for (int start = first; start < len; start += 128) {
        const int at = start + tid;
        float s = -INFINITY;
        if (at < len) {
            const KV* k = keys + (size_t)at * kv_dim + kv;
            float dot = 0.0f;
            for (int d = 0; d < hd; d += 4) {
                const float4 kv4 = load4(k + d), q4 = f4(qs + d);
                dot += q4.x * kv4.x;
                dot += q4.y * kv4.y;
                dot += q4.z * kv4.z;
                dot += q4.w * kv4.w;
            }
            s = dot * scale;
        }
        const float new_max = fmaxf(max_so_far, block_max(s, scratch));
        const float p = at < len ? expf(s - new_max) : 0.0f;
        ps[tid] = p;
        const float correction = expf(max_so_far - new_max);  // 0 on the first chunk
        total = total * correction + block_sum(p, scratch);  // the sum also syncs ps
        const int count = min(128, len - start);
        for (int j = 0; j < 4; j++) {
            const int d = tid + j * 128;
            if (d >= hd) break;
            float a = acc[j] * correction;
            const KV* v = values + (size_t)start * kv_dim + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + (size_t)i * kv_dim);
            acc[j] = a;
        }
        max_so_far = new_max;
    }
    float* o = out + ((size_t)t * heads + head) * hd;
    for (int j = 0; j < 4; j++) {
        const int d = tid + j * 128;
        if (d < hd) o[d] = acc[j] / total;
    }
}

extern "C" __global__ void attention(const float* q, const float* keys, const float* values, float* out,
                                     const int* pos, int window, int hd, int kv_dim, int group, float scale) {
    attention_one<float>(q, keys, values, out, pos, window, hd, kv_dim, group, scale);
}

extern "C" __global__ void attention_h(const float* q, const __half* keys, const __half* values, float* out,
                                       const int* pos, int window, int hd, int kv_dim, int group, float scale) {
    attention_one<__half>(q, keys, values, out, pos, window, hd, kv_dim, group, scale);
}

// Attention for one new token, split along the context: block (head, part) takes part
// `part` of the cached positions (an equal share of first..len, worked out from *pos on
// the GPU, so a recorded graph fits every length) and leaves its running maximum, sum and
// unnormalized output in `partial` (heads × parts rows of 2 + hd floats); then
// attention_combine merges each head's parts. With one block per head, a long context
// leaves most of the GPU idle; this spreads it over heads × parts blocks.
template <typename KV>
__device__ void attention_part(const float* q, const KV* keys, const KV* values, float* partial, const int* pos,
                               int window, int hd, int kv_dim, int group, float scale) {
    __shared__ float scratch[32];
    __shared__ __align__(16) float qs[512];
    __shared__ float ps[128];
    const int head = blockIdx.x, part = blockIdx.y, parts = gridDim.y, tid = threadIdx.x;
    const int len = *pos + 1;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    const int kv = (head / group) * hd;
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = q[head * hd + d];
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_so_far = -INFINITY, total = 0.0f;
    for (int start = lo; start < hi; start += 128) {
        const int at = start + tid;
        float s = -INFINITY;
        if (at < hi) {
            const KV* k = keys + (size_t)at * kv_dim + kv;
            float dot = 0.0f;
            for (int d = 0; d < hd; d += 4) {
                const float4 kv4 = load4(k + d), q4 = f4(qs + d);
                dot += q4.x * kv4.x;
                dot += q4.y * kv4.y;
                dot += q4.z * kv4.z;
                dot += q4.w * kv4.w;
            }
            s = dot * scale;
        }
        const float new_max = fmaxf(max_so_far, block_max(s, scratch));
        const float p = at < hi ? expf(s - new_max) : 0.0f;
        ps[tid] = p;
        const float correction = expf(max_so_far - new_max);
        total = total * correction + block_sum(p, scratch);
        const int count = min(128, hi - start);
        for (int j = 0; j < 4; j++) {
            const int d = tid + j * 128;
            if (d >= hd) break;
            float a = acc[j] * correction;
            const KV* v = values + (size_t)start * kv_dim + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + (size_t)i * kv_dim);
            acc[j] = a;
        }
        max_so_far = new_max;
    }
    float* out = partial + ((size_t)head * parts + part) * (2 + hd);
    if (tid == 0) {
        out[0] = max_so_far;  // -inf for an empty part
        out[1] = total;
    }
    for (int j = 0; j < 4; j++) {
        const int d = tid + j * 128;
        if (d < hd) out[2 + d] = acc[j];
    }
}

extern "C" __global__ void attention_part_f32(const float* q, const float* keys, const float* values, float* partial,
                                              const int* pos, int window, int hd, int kv_dim, int group,
                                              float scale) {
    attention_part<float>(q, keys, values, partial, pos, window, hd, kv_dim, group, scale);
}

extern "C" __global__ void attention_part_f16(const float* q, const __half* keys, const __half* values,
                                              float* partial, const int* pos, int window, int hd, int kv_dim,
                                              int group, float scale) {
    attention_part<__half>(q, keys, values, partial, pos, window, hd, kv_dim, group, scale);
}

// attention_part for an f16 cache (8-bit mode), reading each key and value once for all
// the query heads that share them. Block (x, part): x is a key/value head and a set of up
// to 4 of its query heads (`group` / 4 rounded up sets a head); part as in attention_part.
// Each of the 4 warps takes every 4th position of the part, its lanes splitting the head
// (HD / 32 numbers each, read as 4–32 contiguous bytes, so a warp reads a row whole), and
// keeps an online softmax per query head in base 2; the warps are merged at the end into
// attention_part's partial rows (natural-log maximum), for attention_combine.
template <int N>
__device__ __forceinline__ void halves_to_float(const __half* p, float (&out)[N]) {
    if constexpr (N >= 8) {
#pragma unroll
        for (int i = 0; i < N / 8; i++) {
            const uint4 u = reinterpret_cast<const uint4*>(p)[i];
            const u32 w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[j]));
                out[i * 8 + 2 * j] = f.x;
                out[i * 8 + 2 * j + 1] = f.y;
            }
        }
    } else if constexpr (N == 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        out[0] = a.x, out[1] = a.y, out[2] = b.x, out[3] = b.y;
    } else {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(p));
        out[0] = a.x, out[1] = a.y;
    }
}

template <int HD>
__device__ void attention_part_kv(const float* q, const __half* keys, const __half* values, float* partial,
                                  const int* pos, int window, int kv_dim, int group, float scale) {
    constexpr int PER = HD / 32, G = 4, WARPS = 4;
    __shared__ float wm[WARPS][G], wl[WARPS][G];
    __shared__ __align__(16) float wacc[WARPS][G][HD];
    const int sets = (group + G - 1) / G;
    const int kvh = blockIdx.x / sets, set = blockIdx.x % sets;
    const int head0 = kvh * group + set * G, ng = min(G, group - set * G);
    const int part = blockIdx.y, parts = gridDim.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int len = *pos + 1;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);

    // The queries, scaled so a score is in base 2.
    const float qscale = scale * 1.4426950408889634f;
    float qr[G][PER];
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
        for (int e = 0; e < PER; e++)
            qr[g][e] = g < ng ? q[(size_t)(head0 + g) * HD + lane * PER + e] * qscale : 0.0f;

    float m[G], l[G], acc[G][PER];
#pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int e = 0; e < PER; e++) acc[g][e] = 0.0f;
    }
    const size_t off = (size_t)kvh * HD + lane * PER;
    // The scores of positions in N, all heads at once: the dot products, then their sums
    // across the warp together, so the shuffles overlap. (Heads past ng have zero queries.)
    auto scores = [&](const float (&k)[PER], float (&sc)[G]) {
#pragma unroll
        for (int g = 0; g < G; g++) {
            float s = 0.0f;
#pragma unroll
            for (int e = 0; e < PER; e++) s = fmaf(qr[g][e], k[e], s);
            sc[g] = s;
        }
    };
    auto reduce = [&](float (&sc)[G]) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int g = 0; g < G; g++) sc[g] += __shfl_xor_sync(0xFFFFFFFFu, sc[g], o);
    };
    // U positions a step (4, or 2 for 512-wide heads, by registers), all read before any is
    // used so more reads are in flight, and one rescaling of the sums for them all.
    constexpr int U = HD >= 512 ? 2 : 4;
    int at = lo + warp;
    for (; at + (U - 1) * WARPS < hi; at += U * WARPS) {
        float k[U][PER], v[U][PER], sc[U][G];
#pragma unroll
        for (int u = 0; u < U; u++) halves_to_float(keys + (size_t)(at + u * WARPS) * kv_dim + off, k[u]);
#pragma unroll
        for (int u = 0; u < U; u++) halves_to_float(values + (size_t)(at + u * WARPS) * kv_dim + off, v[u]);
#pragma unroll
        for (int u = 0; u < U; u++) scores(k[u], sc[u]);
#pragma unroll
        for (int u = 0; u < U; u++) reduce(sc[u]);
#pragma unroll
        for (int g = 0; g < G; g++) {
            float mn = m[g];
#pragma unroll
            for (int u = 0; u < U; u++) mn = fmaxf(mn, sc[u][g]);
            const float c = exp2f(m[g] - mn);
            float p[U], psum = 0.0f;
#pragma unroll
            for (int u = 0; u < U; u++) {
                p[u] = exp2f(sc[u][g] - mn);
                psum += p[u];
            }
            m[g] = mn;
            l[g] = fmaf(l[g], c, psum);
#pragma unroll
            for (int e = 0; e < PER; e++) {
                float pv = p[0] * v[0][e];
#pragma unroll
                for (int u = 1; u < U; u++) pv = fmaf(p[u], v[u][e], pv);
                acc[g][e] = fmaf(acc[g][e], c, pv);
            }
        }
    }
    for (; at < hi; at += WARPS) {
        float k[PER], v[PER], sc[G];
        halves_to_float(keys + (size_t)at * kv_dim + off, k);
        halves_to_float(values + (size_t)at * kv_dim + off, v);
        scores(k, sc);
        reduce(sc);
#pragma unroll
        for (int g = 0; g < G; g++) {
            const float mn = fmaxf(m[g], sc[g]);
            const float c = exp2f(m[g] - mn), p = exp2f(sc[g] - mn);
            m[g] = mn;
            l[g] = fmaf(l[g], c, p);
#pragma unroll
            for (int e = 0; e < PER; e++) acc[g][e] = fmaf(acc[g][e], c, p * v[e]);
        }
    }

    // Merge the warps.
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (lane == 0) {
            wm[warp][g] = m[g];
            wl[warp][g] = l[g];
        }
#pragma unroll
        for (int e = 0; e < PER; e++) wacc[warp][g][lane * PER + e] = acc[g][e];
    }
    __syncthreads();
    for (int g = 0; g < ng; g++) {
        float mx = -INFINITY;
#pragma unroll
        for (int w = 0; w < WARPS; w++) mx = fmaxf(mx, wm[w][g]);
        float f[WARPS];
#pragma unroll
        for (int w = 0; w < WARPS; w++) f[w] = mx == -INFINITY ? 0.0f : exp2f(wm[w][g] - mx);
        float* out = partial + ((size_t)(head0 + g) * parts + part) * (2 + HD);
        if (threadIdx.x == 0) {
            float total = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; w++) total += f[w] * wl[w][g];
            out[0] = mx * 0.6931471805599453f;  // natural log; -inf for an empty part
            out[1] = total;
        }
        for (int d = threadIdx.x; d < HD; d += 32 * WARPS) {
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; w++) a += f[w] * wacc[w][g][d];
            out[2 + d] = a;
        }
    }
}

#define ATTENTION_PART_KV(HD)                                                                             \
    extern "C" __global__ void __launch_bounds__(128, 2) attention_part_kv_##HD(                           \
        const float* q, const __half* keys, const __half* values, float* partial, const int* pos,           \
        int window, int kv_dim, int group, float scale) {                                                   \
        attention_part_kv<HD>(q, keys, values, partial, pos, window, kv_dim, group, scale);                \
    }
ATTENTION_PART_KV(64)
ATTENTION_PART_KV(128)
ATTENTION_PART_KV(256)
ATTENTION_PART_KV(512)

// Merges attention_part's parts: one block per head.
extern "C" __global__ void attention_combine(const float* partial, float* out, int parts, int hd) {
    const int head = blockIdx.x;
    const float* p = partial + (size_t)head * parts * (2 + hd);
    float max = -INFINITY;
    for (int i = 0; i < parts; i++) max = fmaxf(max, p[i * (2 + hd)]);
    float total = 0.0f;
    for (int i = 0; i < parts; i++) {
        const float m = p[i * (2 + hd)];
        if (m != -INFINITY) total += expf(m - max) * p[i * (2 + hd) + 1];
    }
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < parts; i++) {
            const float m = p[i * (2 + hd)];
            if (m != -INFINITY) acc += expf(m - max) * p[i * (2 + hd) + 2 + d];
        }
        out[head * hd + d] = acc / total;
    }
}

// Causal attention for a prompt's tokens, tiled, for head size LANES × PER: one block
// of 256 threads per query head and 256 / LANES tokens, so each key and value is read
// once per block rather than once per query. Keys and values come CHUNK positions at a
// time into shared memory. Each token has LANES threads (neighbouring lanes); thread
// `sub` holds PER numbers of the token's query, from sub × PER on, in registers and
// works out those numbers of its output (reading 4 at a time from shared memory), and
// they add their parts of each score together with shuffles. As in `attention`, an
// online softmax.
template <int LANES, int PER, int CHUNK>
__device__ void attention_tiled(const float* q, const float* keys, const float* values, float* out,
                                const int* pos, int window, int kv_dim, int group, float scale, int tokens) {
    constexpr int HD = LANES * PER, ROWS = 256 / LANES;
    __shared__ __align__(16) float ks[CHUNK][HD];
    __shared__ __align__(16) float vs[CHUNK][HD];
    const int head = blockIdx.x, heads = gridDim.x, tid = threadIdx.x;
    const int tq = tid / LANES, sub = tid % LANES;
    const int t0 = blockIdx.y * ROWS, p0 = *pos;
    const int t = t0 + tq;
    const bool live = t < tokens;
    const int at = p0 + t;  // this token's position
    const int last_row = p0 + min(t0 + ROWS, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;
    const int row_first = window > 0 ? max(0, at + 1 - window) : 0;
    const int kv = (head / group) * HD;

    float qr[PER], acc[PER];
    const float* qh = q + ((size_t)min(t, tokens - 1) * heads + head) * HD;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        qr[j] = qh[sub * PER + j];
        acc[j] = 0.0f;
    }
    float max_so_far = -INFINITY, total = 0.0f;

    for (int start = lo; start <= last_row; start += CHUNK) {
        const int count = min(CHUNK, last_row + 1 - start);
        for (int i = tid; i < CHUNK * HD / 4; i += 256) {
            const int r = (i * 4) / HD, d = (i * 4) % HD;
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f), vv4 = kv4;
            if (r < count) {
                kv4 = f4(keys + (size_t)(start + r) * kv_dim + kv + d);
                vv4 = f4(values + (size_t)(start + r) * kv_dim + kv + d);
            }
            *reinterpret_cast<float4*>(&ks[r][d]) = kv4;
            *reinterpret_cast<float4*>(&vs[r][d]) = vv4;
        }
        __syncthreads();

        float s[CHUNK];
        float chunk_max = -INFINITY;
#pragma unroll
        for (int k = 0; k < CHUNK; k++) {
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; j += 4) {
                const float4 kk = *reinterpret_cast<const float4*>(&ks[k][sub * PER + j]);
                part += qr[j] * kk.x;
                part += qr[j + 1] * kk.y;
                part += qr[j + 2] * kk.z;
                part += qr[j + 3] * kk.w;
            }
#pragma unroll
            for (int o = LANES / 2; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            const int kpos = start + k;
            const bool seen = live && k < count && kpos <= at && kpos >= row_first;
            s[k] = seen ? part * scale : -INFINITY;
            chunk_max = fmaxf(chunk_max, s[k]);
        }
        const float new_max = fmaxf(max_so_far, chunk_max);
        if (new_max != -INFINITY) {
            const float correction = expf(max_so_far - new_max);
            float chunk_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < CHUNK; k++) {
                s[k] = s[k] == -INFINITY ? 0.0f : expf(s[k] - new_max);
                chunk_sum += s[k];
            }
            total = total * correction + chunk_sum;
            max_so_far = new_max;
#pragma unroll
            for (int j = 0; j < PER; j++) acc[j] *= correction;
#pragma unroll
            for (int k = 0; k < CHUNK; k++) {
#pragma unroll
                for (int j = 0; j < PER; j += 4) {
                    const float4 v = *reinterpret_cast<const float4*>(&vs[k][sub * PER + j]);
                    acc[j] += s[k] * v.x;
                    acc[j + 1] += s[k] * v.y;
                    acc[j + 2] += s[k] * v.z;
                    acc[j + 3] += s[k] * v.w;
                }
            }
        }
        __syncthreads();  // done with ks, vs before the next chunk
    }
    if (!live) return;
    float* o = out + ((size_t)t * heads + head) * HD;
#pragma unroll
    for (int j = 0; j < PER; j++) o[sub * PER + j] = acc[j] / total;
}

#define ATTENTION_TILED(name, LANES, PER, CHUNK)                                                             extern "C" __global__ void __launch_bounds__(256) name(const float* q, const float* keys,                                                                        const float* values, float* out, const int* pos,                                                            int window, int hd, int kv_dim, int group,                                                                float scale, int tokens) {                             attention_tiled<LANES, PER, CHUNK>(q, keys, values, out, pos, window, kv_dim, group, scale, tokens);     }

// Head sizes 64, 128, 256 and 512, with as many tokens per block as fit.
ATTENTION_TILED(attention_tiled_64, 8, 8, 32)
ATTENTION_TILED(attention_tiled_128, 8, 16, 32)
ATTENTION_TILED(attention_tiled_256, 16, 16, 16)
ATTENTION_TILED(attention_tiled_512, 32, 16, 8)

// gate = SiLU(gate) × up, over `total` numbers in rows of n; up's rows are `up_stride` apart.
extern "C" __global__ void swiglu(float* gate, const float* up, int n, int up_stride, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float u = up[(size_t)(i / n) * up_stride + i % n];
    gate[i] = gate[i] / (1.0f + expf(-gate[i])) * u;
}

// gate = GELU(gate) × up (tanh approximation; Gemma 4), laid out as in swiglu.
extern "C" __global__ void geglu(float* gate, const float* up, int n, int up_stride, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float x = gate[i];
    const float u = up[(size_t)(i / n) * up_stride + i % n];
    gate[i] = 0.5f * x * (1.0f + tanhf(0.7978846f * x * (1.0f + 0.044715f * x * x))) * u;
}

// x += y
extern "C" __global__ void add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// x = cap × tanh(x / cap) (Gemma's logit softcap)
extern "C" __global__ void softcap(float* x, float cap, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// ---- Qwen 3.5 ------------------------------------------------------------------------

// out ×= sigmoid(gate), the gate being each head's second half of query_gate.
extern "C" __global__ void apply_gate(float* out, const float* query_gate, int heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * head_dim) return;
    const int h = i / head_dim, d = i % head_dim;
    out[i] *= 1.0f / (1.0f + expf(-query_gate[h * 2 * head_dim + head_dim + d]));
}

// The convolution kernels take its size (2–8) as a constant, so their windows stay in
// registers: with the size known only at run time, the arrays went to local memory.
#define CONV_DISPATCH(fn, ...)              \
    switch (kernel) {                       \
        case 2: return fn<2>(__VA_ARGS__);  \
        case 3: return fn<3>(__VA_ARGS__);  \
        case 4: return fn<4>(__VA_ARGS__);  \
        case 5: return fn<5>(__VA_ARGS__);  \
        case 6: return fn<6>(__VA_ARGS__);  \
        case 7: return fn<7>(__VA_ARGS__);  \
        default: return fn<8>(__VA_ARGS__); \
    }

// The causal convolution over each channel's last `kernel` values (oldest first), then
// SiLU, for `tokens` tokens in order; the history moves along with them. One thread per
// channel; `current` and `out` hold a row of `channels` per token. Kernel at most 8.

template <int K>
__device__ void conv_silu_tokens(const float* current, float* history, const float* weights, float* out,
                                 int channels, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float w[K], past[K - 1];
#pragma unroll
    for (int i = 0; i < K; i++) w[i] = weights[(size_t)c * K + i];
#pragma unroll
    for (int i = 0; i < K - 1; i++) past[i] = history[(size_t)i * channels + c];
    for (int t = 0; t < tokens; t++) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < K - 1; i++) sum += past[i] * w[i];
        const float now = current[(size_t)t * channels + c];
        sum += now * w[K - 1];
        out[(size_t)t * channels + c] = sum / (1.0f + expf(-sum));
#pragma unroll
        for (int i = 0; i < K - 2; i++) past[i] = past[i + 1];
        past[K - 2] = now;
    }
#pragma unroll
    for (int i = 0; i < K - 1; i++) history[(size_t)i * channels + c] = past[i];
}

extern "C" __global__ void conv_silu(const float* current, float* history, const float* weights, float* out,
                                     int channels, int kernel, int tokens) {
    CONV_DISPATCH(conv_silu_tokens, current, history, weights, out, channels, tokens)
}

// conv_silu for a prompt: each output depends only on inputs (the token's and the
// `kernel` − 1 before it, from `history` before the first), so every channel and run of
// CONV_TOKENS tokens goes at once: a thread walks its channel's run, keeping the weights
// and the last `kernel` − 1 inputs in registers, with the same sums in the same order.
// Blocks: x over channels, y over runs of tokens. The history isn't moved along here:
// conv_history does that after.
#define CONV_TOKENS 16
template <int K>
__device__ void conv_silu_run(const float* current, const float* history, const float* weights, float* out,
                              int channels, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int t0 = blockIdx.y * CONV_TOKENS, t1 = min(t0 + CONV_TOKENS, tokens);
    float w[K], past[K - 1];  // past[k]: the input for tap k (the oldest first)
#pragma unroll
    for (int k = 0; k < K; k++) w[k] = weights[(size_t)c * K + k];
#pragma unroll
    for (int k = 0; k < K - 1; k++) {
        const int from = t0 - (K - 1) + k;  // the token this tap reads, < 0 for history
        past[k] = from >= 0 ? current[(size_t)from * channels + c] : history[(size_t)(K - 1 + from) * channels + c];
    }
    for (int t = t0; t < t1; t++) {
        const float now = current[(size_t)t * channels + c];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K - 1; k++) sum += past[k] * w[k];
        sum += now * w[K - 1];
        out[(size_t)t * channels + c] = sum / (1.0f + expf(-sum));
        // Move the window along: the oldest tap drops out, `now` comes in last.
#pragma unroll
        for (int k = 0; k < K - 2; k++) past[k] = past[k + 1];
        past[K - 2] = now;
    }
}

extern "C" __global__ void conv_silu_parallel(const float* current, const float* history, const float* weights,
                                              float* out, int channels, int kernel, int tokens) {
    CONV_DISPATCH(conv_silu_run, current, history, weights, out, channels, tokens)
}

// After conv_silu_parallel: the history becomes the last `kernel` − 1 inputs (older ones
// from the history itself when there were fewer tokens). One thread per channel.
template <int K>
__device__ void conv_history_run(const float* current, float* history, int channels, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float past[K - 1];
#pragma unroll
    for (int k = 0; k < K - 1; k++) {
        const int from = tokens - (K - 1) + k;
        past[k] = from >= 0 ? current[(size_t)from * channels + c] : history[(size_t)(K - 1 + from) * channels + c];
    }
#pragma unroll
    for (int k = 0; k < K - 1; k++) history[(size_t)k * channels + c] = past[k];
}

extern "C" __global__ void conv_history(const float* current, float* history, int channels, int kernel,
                                        int tokens) {
    CONV_DISPATCH(conv_history_run, current, history, channels, tokens)
}

// Gated DeltaNet for `tokens` tokens in order, one block of 4n threads per value head
// (n = the state size, a multiple of 8 up to 128). Token t's convolved [q | k | v] is
// row t of `conv` (`channels` wide; q and k are `key_dim` each). Per token: q and k
// L2-normalized (to at least `eps`), then, with S the head's state stored transposed
// (state[j*n + i] = S[i][j]) as the CPU reference and llama.cpp store it:
//   S = exp(g) S; delta_j = (v_j - S[:,j].k) beta; S[:,j] += k delta_j; o_j = S[:,j].q / sqrt(n)
// then out = RMSNorm(o) * norm * SiLU(z), where beta = sigmoid(beta_raw) and
// g = softplus(alpha_raw + dt_bias) * a. beta_raw and alpha_raw have a row of v_heads
// per token; z and out a row of v_heads × n.
//
// Column j of S belongs to 4 neighbouring threads, each holding every 4th number of it
// in registers for the whole batch: the state is read once and written once, however
// many tokens there are.
extern "C" __global__ void __launch_bounds__(512) delta_net(const float* conv, int channels, int key_dim,
                                     const float* beta_raw, const float* alpha_raw, const float* dt_bias,
                                     const float* a, float* state, const float* z, const float* norm,
                                     float* out, int n, int k_heads, float eps, int tokens) {
    __shared__ float scratch[32];
    __shared__ float ks[128];
    __shared__ float qs[128];
    const int h = blockIdx.x, v_heads = gridDim.x, kh = h % k_heads;
    const int j = threadIdx.x >> 2, part = threadIdx.x & 3, per = n >> 2;
    float* column = state + ((size_t)h * n + j) * n;
    float st[32];
#pragma unroll
    for (int r = 0; r < 32; r++)
        if (r < per) st[r] = column[part + 4 * r];
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int t = 0; t < tokens; t++) {
        const float* c = conv + (size_t)t * channels;
        float qj = 0.0f, kj = 0.0f;
        if (part == 0) {
            qj = c[kh * n + j];
            kj = c[key_dim + kh * n + j];
        }
        const float q_len = sqrtf(block_sum(qj * qj, scratch));
        const float k_len = sqrtf(block_sum(kj * kj, scratch));
        if (part == 0) {
            qs[j] = qj * (1.0f / fmaxf(q_len, eps));
            ks[j] = kj * (1.0f / fmaxf(k_len, eps));
        }
        __syncthreads();
        const int head_at = t * v_heads + h;
        const float beta = 1.0f / (1.0f + expf(-beta_raw[head_at]));
        const float x = alpha_raw[head_at] + dt_bias[h];
        const float softplus = x > 20.0f ? x : log1pf(expf(x));
        const float decay = expf(softplus * a[h]);

        float dot = 0.0f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            if (r < per) {
                st[r] *= decay;
                dot += st[r] * ks[part + 4 * r];
            }
        }
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        const float delta = (c[2 * key_dim + h * n + j] - dot) * beta;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            if (r < per) {
                st[r] += ks[part + 4 * r] * delta;
                o += st[r] * qs[part + 4 * r];
            }
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        o *= inv_sqrt_n;

        const float sum = block_sum(part == 0 ? o * o : 0.0f, scratch);  // also: done with ks, qs
        if (part == 0) {
            const float normed = o / sqrtf(sum / (float)n + eps) * norm[j];
            const size_t at = (size_t)head_at * n + j;
            const float zj = z[at];
            out[at] = normed * (zj / (1.0f + expf(-zj)));
        }
    }
#pragma unroll
    for (int r = 0; r < 32; r++)
        if (r < per) column[part + 4 * r] = st[r];
}

// Gated DeltaNet for state sizes over 128 (see delta_net): one block of n threads per
// value head (n a multiple of 32 up to 256), the state read and written in GPU memory
// for every token. Token t's convolved [q | k | v] is
// row t of `conv` (`channels` wide; q and k are `key_dim` each). Per token: q and k
// L2-normalized (to at least `eps`), then with thread j owning row j of the head's
// transposed state (state[j*n + i] = S[i][j]), as the CPU reference and llama.cpp
// store it:
//   S = exp(g) S; delta_j = (v_j - S[:,j].k) beta; S[:,j] += k delta_j; o_j = S[:,j].q / sqrt(n)
// then out = RMSNorm(o) * norm * SiLU(z), where beta = sigmoid(beta_raw) and
// g = softplus(alpha_raw + dt_bias) * a. beta_raw and alpha_raw have a row of v_heads
// per token; z and out a row of v_heads × n.
extern "C" __global__ void delta_net_wide(const float* conv, int channels, int key_dim, const float* beta_raw,
                                     const float* alpha_raw, const float* dt_bias, const float* a, float* state,
                                     const float* z, const float* norm, float* out, int n, int k_heads,
                                     float eps, int tokens) {
    __shared__ float scratch[32];
    __shared__ float ks[256];
    __shared__ float qs[256];
    const int h = blockIdx.x, v_heads = gridDim.x, j = threadIdx.x, kh = h % k_heads;
    float* row = state + ((size_t)h * n + j) * n;
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int t = 0; t < tokens; t++) {
        const float* c = conv + (size_t)t * channels;
        const float qj = c[kh * n + j], kj = c[key_dim + kh * n + j];
        const float q_len = sqrtf(block_sum(qj * qj, scratch));
        const float k_len = sqrtf(block_sum(kj * kj, scratch));
        qs[j] = qj * (1.0f / fmaxf(q_len, eps));
        ks[j] = kj * (1.0f / fmaxf(k_len, eps));
        __syncthreads();
        const int head_at = t * v_heads + h;
        const float beta = 1.0f / (1.0f + expf(-beta_raw[head_at]));
        const float x = alpha_raw[head_at] + dt_bias[h];
        const float softplus = x > 20.0f ? x : log1pf(expf(x));
        const float decay = expf(softplus * a[h]);

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

        const float sum = block_sum(o * o, scratch);  // also: everyone is done with ks, qs
        const float normed = o / sqrtf(sum / (float)n + eps) * norm[j];
        const size_t at = (size_t)head_at * n + j;
        const float zj = z[at];
        out[at] = normed * (zj / (1.0f + expf(-zj)));
    }
}

// ---- Gated DeltaNet for a prompt, in three steps ---------------------------------------
//
// Of a DeltaNet token's work, only the state update depends on the token before; the
// L2 norms of q and k before it and the gated RMSNorm after it don't. So a prompt runs
// them apart: delta_l2 normalizes every token's q and k at once, delta_scan updates the
// state token by token, its columns split over several blocks (each column of the state
// is independent: the only sums across columns were the norms), and delta_out applies
// the gated RMSNorm to every token at once. The arithmetic is delta_net's.

// q and k of each token L2-normalized in place (to at least `eps`): one block of n
// threads per row of n, rows being token t's 2 × k_heads heads of q then k.
extern "C" __global__ void delta_l2(float* conv, int channels, int k_heads, int n, float eps) {
    __shared__ float scratch[32];
    const int t = blockIdx.x / (2 * k_heads), which = blockIdx.x % (2 * k_heads);
    float* p = conv + (size_t)t * channels + which * n;
    const float v = p[threadIdx.x];
    const float len = sqrtf(block_sum(v * v, scratch));
    p[threadIdx.x] = v * (1.0f / fmaxf(len, eps));
}

// The state update for `tokens` tokens in order, for columns blockIdx.y × DS_COLS.. of value
// head blockIdx.x: 4 threads to a column, each holding every 4th number of it in
// registers, as delta_net. q and k (normalized by delta_l2) and each token's beta and
// decay come into shared memory 16 tokens at a time. Writes o / sqrt(n) per token,
// head and column to `out` (rows of v_heads × n), for delta_out.
#define DS_COLS 16
#define DS_CHUNK 16
extern "C" __global__ void __launch_bounds__(4 * DS_COLS)
    delta_scan(const float* conv, int channels, int key_dim, const float* beta_raw, const float* alpha_raw,
               const float* dt_bias, const float* a, float* state, float* out, int n, int k_heads, int tokens) {
    __shared__ float ks[DS_CHUNK][128], qs[DS_CHUNK][128];
    __shared__ float betas[DS_CHUNK], decays[DS_CHUNK];
    const int h = blockIdx.x, v_heads = gridDim.x, kh = h % k_heads;
    const int j = blockIdx.y * DS_COLS + (threadIdx.x >> 2), part = threadIdx.x & 3, per = n >> 2;
    float* column = state + ((size_t)h * n + j) * n;
    float st[32];
#pragma unroll
    for (int r = 0; r < 32; r++)
        if (r < per) st[r] = column[part + 4 * r];
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int t0 = 0; t0 < tokens; t0 += DS_CHUNK) {
        const int count = min(DS_CHUNK, tokens - t0);
        __syncthreads();  // done with the last chunk's q and k
        for (int i = threadIdx.x; i < count * n; i += blockDim.x) {
            const int tt = i / n, e = i % n;
            const float* c = conv + (size_t)(t0 + tt) * channels;
            qs[tt][e] = c[kh * n + e];
            ks[tt][e] = c[key_dim + kh * n + e];
        }
        if (threadIdx.x < count) {
            const int head_at = (t0 + threadIdx.x) * v_heads + h;
            betas[threadIdx.x] = 1.0f / (1.0f + expf(-beta_raw[head_at]));
            const float x = alpha_raw[head_at] + dt_bias[h];
            const float softplus = x > 20.0f ? x : log1pf(expf(x));
            decays[threadIdx.x] = expf(softplus * a[h]);
        }
        __syncthreads();
        for (int tt = 0; tt < count; tt++) {
            const float decay = decays[tt], beta = betas[tt];
            float dot = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; r++) {
                if (r < per) {
                    st[r] *= decay;
                    dot += st[r] * ks[tt][part + 4 * r];
                }
            }
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2);
            const float delta = (conv[(size_t)(t0 + tt) * channels + 2 * key_dim + h * n + j] - dot) * beta;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; r++) {
                if (r < per) {
                    st[r] += ks[tt][part + 4 * r] * delta;
                    o += st[r] * qs[tt][part + 4 * r];
                }
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (part == 0) out[((size_t)(t0 + tt) * v_heads + h) * n + j] = o * inv_sqrt_n;
        }
    }
#pragma unroll
    for (int r = 0; r < 32; r++)
        if (r < per) column[part + 4 * r] = st[r];
}

// out = RMSNorm(o) × norm × SiLU(z) in place, one block of n threads per token and head.
extern "C" __global__ void delta_out(float* out, const float* z, const float* norm, int n, float eps) {
    __shared__ float scratch[32];
    const size_t at = (size_t)blockIdx.x * n + threadIdx.x;
    const float o = out[at];
    const float sum = block_sum(o * o, scratch);
    const float normed = o / sqrtf(sum / (float)n + eps) * norm[threadIdx.x];
    const float zj = z[at];
    out[at] = normed * (zj / (1.0f + expf(-zj)));
}

// ---- Gated DeltaNet for a prompt in chunks (the WY form) -------------------------------
//
// Token by token, a head's state S (n × n: S[i][j] = state[j*n + i]) goes
//   S_t = g_t S_{t-1} + k_t δ_tᵀ,  δ_t = β_t (v_t − (g_t S_{t-1})ᵀ k_t),  o_t = S_tᵀ q_t / √n
// (as delta_net). Over a chunk of DC tokens from a state S0, with γ_i = g_1 ⋯ g_i:
//   S_i = γ_i S0 + Σ_{j≤i} (γ_i/γ_j) k_j δ_jᵀ, so the δ's solve a triangular system
//   (I + A) δ = diag(β) (V − diag(γ) K S0),  A_ij = β_i (γ_i/γ_j) k_i·k_j for j < i.
// With T = (I + A)⁻¹: δ = U − W S0, where U = T diag(β) V and W = T diag(β γ) K don't
// depend on S0. Then
//   O = diag(γ) Q S0 + P δ,  P_ij = (γ_i/γ_j) q_i·k_j for j ≤ i,
//   S_DC = γ_DC S0 + K̂ᵀ δ,   K̂_j = (γ_DC/γ_j) k_j.
// delta_chunk_prep works out W, U, P, Q̂ = diag(γ) Q, K̂ and γ_DC for every chunk at once
// (they need only the chunk's own q, k, v, β and g); delta_chunk_scan then runs the
// chunks in order, each a few small products with the state. The same arithmetic as
// token by token in another order (llama.cpp's chunked form too). q and k come
// normalized by delta_l2; delta_out applies the gated RMSNorm after.
//
// Scratch per chunk and value head (delta_scratch_floats on the host): W, U, Q̂, K̂ (DC × n
// each), P (DC × DC), γ_DC.

#define DC 64

__device__ __forceinline__ size_t delta_chunk_stride(int n) { return (size_t)4 * DC * n + DC * DC + 4; }

// One block of 256 threads per value head (blockIdx.x) and chunk (blockIdx.y); n ≤ 128.
// Dynamic shared memory (delta_chunk_prep_bytes on the host): the chunk's k, which A
// takes the place of once k·k is worked out, then the cumulative log decays and β. q is
// read from GPU memory (through L1): with it in shared memory too, only one block fit an
// SM.
extern "C" __global__ void __launch_bounds__(256)
    delta_chunk_prep(const float* conv, int channels, int key_dim, const float* beta_raw, const float* alpha_raw,
                     const float* dt_bias, const float* a, float* scratch, int n, int k_heads, int tokens) {
    extern __shared__ __align__(16) float sm[];
    const int NS = n + 4;  // padded rows
    float* ks = sm;
    float* As = sm;  // DC × (DC + 1), over ks once A is worked out
    float* G = sm + DC * NS;
    float* bs = G + DC;
    const int h = blockIdx.x, v_heads = gridDim.x, ch = blockIdx.y, kh = h % k_heads, tid = threadIdx.x;
    const int t0 = ch * DC, len = min(DC, tokens - t0);
    float* base = scratch + ((size_t)ch * v_heads + h) * delta_chunk_stride(n);
    float *W = base, *U = W + DC * n, *Qh = U + DC * n, *Kh = Qh + DC * n, *P = Kh + DC * n;
    // Row i's q, k and v in conv (rows past the chunk's end read as 0).
    auto q_row = [&](int i) { return conv + (size_t)(t0 + i) * channels + kh * n; };
    auto k_row = [&](int i) { return conv + (size_t)(t0 + i) * channels + key_dim + kh * n; };

    for (int idx = tid; idx < DC * n; idx += blockDim.x) {
        const int i = idx / n, d = idx % n;
        ks[i * NS + d] = i < len ? k_row(i)[d] : 0.0f;
    }
    if (tid < DC) {
        float log_g = 0.0f, beta = 0.0f;  // rows past the end: no decay, no update
        if (tid < len) {
            const int head_at = (t0 + tid) * v_heads + h;
            beta = 1.0f / (1.0f + expf(-beta_raw[head_at]));
            const float x = alpha_raw[head_at] + dt_bias[h];
            const float softplus = x > 20.0f ? x : log1pf(expf(x));
            log_g = softplus * a[h];
        }
        G[tid] = log_g;
        bs[tid] = beta;
    }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f;
        for (int i = 0; i < DC; i++) G[i] = g += G[i];  // log γ_i
    }
    __syncthreads();

    // k·k and q·k in 4 × 4 tiles: thread → rows 4 ti.., columns tj + 16 b (spread out so
    // the rows read together sit in different banks).
    const int ti = tid >> 4, tj = tid & 15;
    float kk[4][4] = {}, qk[4][4] = {};
    for (int d = 0; d < n; d += 4) {
        float4 ki[4], qi[4], kj[4];
#pragma unroll
        for (int x = 0; x < 4; x++) {
            const int i = 4 * ti + x;
            ki[x] = *reinterpret_cast<const float4*>(&ks[i * NS + d]);
            qi[x] = i < len ? *reinterpret_cast<const float4*>(q_row(i) + d) : make_float4(0.f, 0.f, 0.f, 0.f);
            kj[x] = *reinterpret_cast<const float4*>(&ks[(tj + 16 * x) * NS + d]);
        }
#pragma unroll
        for (int x = 0; x < 4; x++)
#pragma unroll
            for (int y = 0; y < 4; y++) {
                kk[x][y] += ki[x].x * kj[y].x + ki[x].y * kj[y].y + ki[x].z * kj[y].z + ki[x].w * kj[y].w;
                qk[x][y] += qi[x].x * kj[y].x + qi[x].y * kj[y].y + qi[x].z * kj[y].z + qi[x].w * kj[y].w;
            }
    }
    // Q̂ and K̂, and γ_DC (the decay over the chunk's real tokens: later rows add 0).
    const float g_last = G[DC - 1];
    for (int idx = tid; idx < DC * n; idx += blockDim.x) {
        const int i = idx / n, d = idx % n;
        Qh[idx] = i < len ? expf(G[i]) * q_row(i)[d] : 0.0f;
        Kh[idx] = i < len ? expf(g_last - G[i]) * ks[i * NS + d] : 0.0f;
    }
    if (tid == 0) P[DC * DC] = expf(g_last);
    __syncthreads();  // done with ks: A goes in its place
#pragma unroll
    for (int x = 0; x < 4; x++)
#pragma unroll
        for (int y = 0; y < 4; y++) {
            const int i = 4 * ti + x, j = tj + 16 * y;
            const float decay = j <= i ? expf(G[i] - G[j]) : 0.0f;
            As[i * (DC + 1) + j] = j < i ? bs[i] * decay * kk[x][y] : 0.0f;
            P[i * DC + j] = j <= i && i < len ? decay * qk[x][y] : 0.0f;
        }
    __syncthreads();

    // W and U by forward substitution, a thread per column of [W | U]: X_i = B_i −
    // Σ_{m<i} A_im X_m, where B_i = β_i γ_i k_i (W) or β_i v_i (U). A column needs only
    // its own earlier rows, kept in registers. The sum in 4 parts, so the multiply-adds
    // don't all wait on one another.
    const int col = tid;
    if (col < 2 * n) {
        const bool is_w = col < n;
        const int d = is_w ? col : col - n;
        float* X = is_w ? W : U;
        float x[DC];
#pragma unroll
        for (int i = 0; i < DC; i++) {
            float b = 0.0f;
            if (i < len)
                b = is_w ? bs[i] * expf(G[i]) * k_row(i)[d]
                         : bs[i] * conv[(size_t)(t0 + i) * channels + 2 * key_dim + h * n + d];
            float part[4] = {};
#pragma unroll
            for (int m = 0; m < i; m++) part[m & 3] += As[i * (DC + 1) + m] * x[m];
            b -= (part[0] + part[1]) + (part[2] + part[3]);
            x[i] = b;
            X[i * n + d] = b;
        }
    }
}

// The chunks in order for columns blockIdx.y × DSC_COLS.. of value head blockIdx.x's
// state (each column of S is independent): 64 threads, the state's columns and the
// chunk's δ in shared memory. A thread works out 2 rows × 8 columns of δ and O and 2 rows
// of the new state, so each value read from shared memory serves 2 multiply-adds (with
// one row a thread, the shared-memory loads were the limit). Writes o / √n per token,
// head and column to `out` (rows of v_heads × n), for delta_out; leaves the state after
// the last token.
#define DSC_COLS 16
#define DSC_STRIDE (DSC_COLS + 4)  // padded rows, 16-byte aligned: no bank conflicts by row
extern "C" __global__ void __launch_bounds__(64)
    delta_chunk_scan(const float* scratch, float* state, float* out, int n, int tokens) {
    __shared__ __align__(16) float Ss[128][DSC_STRIDE];  // S0[r][c]
    __shared__ __align__(16) float ds[DC][DSC_COLS];     // δ[i][c]
    const int h = blockIdx.x, v_heads = gridDim.x, j0 = blockIdx.y * DSC_COLS, tid = threadIdx.x;
    const int i = tid >> 1, c0 = (tid & 1) * 8;  // δ and O: rows i and i + 32, 8 columns from c0
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int idx = tid; idx < n * DSC_COLS; idx += blockDim.x) {
        const int c = idx / n, r = idx % n;
        Ss[r][c] = state[((size_t)h * n + j0 + c) * n + r];
    }
    // acc[x][0..7] += w[x] × row[c0..c0 + 7] for both rows, the row read 16 bytes at a time.
    auto axpy2x8 = [&](float (&acc)[2][8], float w0, float w1, const float* row) {
        const float4 a = *reinterpret_cast<const float4*>(row + c0);
        const float4 b = *reinterpret_cast<const float4*>(row + c0 + 4);
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int c = 0; c < 8; c++) {
            acc[0][c] += w0 * v[c];
            acc[1][c] += w1 * v[c];
        }
    };
    const int chunks = (tokens + DC - 1) / DC;
    for (int ch = 0; ch < chunks; ch++) {
        const float* base = scratch + ((size_t)ch * v_heads + h) * delta_chunk_stride(n);
        const float *W = base, *U = W + DC * n, *Qh = U + DC * n, *Kh = Qh + DC * n, *P = Kh + DC * n;
        const float gamma = P[DC * DC];
        const int i1 = i + DC / 2;
        __syncthreads();  // the state is in (or updated)

        // δ = U − W S0
        float acc[2][8] = {};
#pragma unroll 4
        for (int r = 0; r < n; r += 4) {
            const float4 w0 = *reinterpret_cast<const float4*>(&W[i * n + r]);
            const float4 w1 = *reinterpret_cast<const float4*>(&W[i1 * n + r]);
            axpy2x8(acc, w0.x, w1.x, Ss[r]);
            axpy2x8(acc, w0.y, w1.y, Ss[r + 1]);
            axpy2x8(acc, w0.z, w1.z, Ss[r + 2]);
            axpy2x8(acc, w0.w, w1.w, Ss[r + 3]);
        }
#pragma unroll
        for (int c = 0; c < 8; c++) {
            ds[i][c0 + c] = U[i * n + j0 + c0 + c] - acc[0][c];
            ds[i1][c0 + c] = U[i1 * n + j0 + c0 + c] - acc[1][c];
        }
        __syncthreads();

        // O = Q̂ S0 + P δ
        float o[2][8] = {};
#pragma unroll 4
        for (int r = 0; r < n; r += 4) {
            const float4 q0 = *reinterpret_cast<const float4*>(&Qh[i * n + r]);
            const float4 q1 = *reinterpret_cast<const float4*>(&Qh[i1 * n + r]);
            axpy2x8(o, q0.x, q1.x, Ss[r]);
            axpy2x8(o, q0.y, q1.y, Ss[r + 1]);
            axpy2x8(o, q0.z, q1.z, Ss[r + 2]);
            axpy2x8(o, q0.w, q1.w, Ss[r + 3]);
        }
        // P is 0 past the diagonal: row i stops at i, row i1 at i1 (its first half with i's).
#pragma unroll 4
        for (int m = 0; m <= i1; m++) axpy2x8(o, m <= i ? P[i * DC + m] : 0.0f, P[i1 * DC + m], ds[m]);
#pragma unroll
        for (int x = 0; x < 2; x++) {
            const int t = ch * DC + (x ? i1 : i);
            if (t < tokens) {
                float* dst = out + ((size_t)t * v_heads + h) * n + j0 + c0;
#pragma unroll
                for (int c = 0; c < 8; c++) dst[c] = o[x][c] * inv_sqrt_n;
            }
        }
        __syncthreads();  // done reading S0

        // S = γ S0 + K̂ᵀ δ, rows r and r + 64
        for (int r = tid; r < n; r += 2 * blockDim.x) {
            const int r1 = r + blockDim.x;
            const bool two = r1 < n;
            float4 s0[DSC_COLS / 4], s1[DSC_COLS / 4];
#pragma unroll
            for (int c = 0; c < DSC_COLS / 4; c++) {
                s0[c] = *reinterpret_cast<const float4*>(&Ss[r][4 * c]);
                s1[c] = two ? *reinterpret_cast<const float4*>(&Ss[r1][4 * c]) : make_float4(0.f, 0.f, 0.f, 0.f);
                s0[c].x *= gamma, s0[c].y *= gamma, s0[c].z *= gamma, s0[c].w *= gamma;
                s1[c].x *= gamma, s1[c].y *= gamma, s1[c].z *= gamma, s1[c].w *= gamma;
            }
#pragma unroll 8
            for (int m = 0; m < DC; m++) {
                const float k0 = Kh[m * n + r], k1 = two ? Kh[m * n + r1] : 0.0f;
#pragma unroll
                for (int c = 0; c < DSC_COLS / 4; c++) {
                    const float4 d = *reinterpret_cast<const float4*>(&ds[m][4 * c]);
                    s0[c].x += k0 * d.x, s0[c].y += k0 * d.y, s0[c].z += k0 * d.z, s0[c].w += k0 * d.w;
                    s1[c].x += k1 * d.x, s1[c].y += k1 * d.y, s1[c].z += k1 * d.z, s1[c].w += k1 * d.w;
                }
            }
#pragma unroll
            for (int c = 0; c < DSC_COLS / 4; c++) {
                *reinterpret_cast<float4*>(&Ss[r][4 * c]) = s0[c];
                if (two) *reinterpret_cast<float4*>(&Ss[r1][4 * c]) = s1[c];
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < n * DSC_COLS; idx += blockDim.x) {
        const int c = idx / n, r = idx % n;
        state[((size_t)h * n + j0 + c) * n + r] = Ss[r][c];
    }
}
