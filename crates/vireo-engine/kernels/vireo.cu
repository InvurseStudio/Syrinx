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

// Where a position's keys and values sit in a layer's cache: its row, or for a ring (a
// sliding-window layer that keeps only its last `ring` positions) its row modulo the ring.
// The ring travels in the window argument's high 16 bits (Gpu::attention_ring), as on
// Metal.
__device__ __forceinline__ size_t ring_row(int t, int ring) { return (size_t)(ring > 0 ? t % ring : t); }

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

// Four bytes from an address that's only 2-byte aligned (Q8_0's blocks have an odd size,
// so their contents are only 2-byte aligned).
__device__ __forceinline__ u32 u32_at2(const u8* p) {
    const u16* h = reinterpret_cast<const u16*>(p);
    return (u32)h[0] | ((u32)h[1] << 16);
}

// Four bytes from a 4-byte-aligned address: Q6_K's words (its blocks start 16-byte aligned).
__device__ __forceinline__ u32 u32_at4(const u8* p) { return *reinterpret_cast<const u32*>(p); }

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

// Q8_0 on the GPU, as Q4_0: each row's numbers first (32 bytes a block, so any 4 or 16 of
// them are one aligned load), then its f16 scales, padded to 16 bytes
// (Gpu::upload_weights repacks GGUF's 34-byte blocks so). A word of a 34-byte block was
// two loads of a half, and a token's pass waited on the loads.
__device__ __forceinline__ size_t q8_0_row_bytes(int cols) {
    const int blocks = cols / 32;
    return (size_t)blocks * 32 + ((blocks * 2 + 15) & ~15);
}

__device__ __forceinline__ float q8_0_scale(const u8* row, int blocks, int b) {
    return half_at(row + (size_t)blocks * 32 + b * 2);
}

// Q6_K on the GPU: each row's blocks without their f16 scale d (208 bytes: ql 128, qh 64,
// 16 int8 scales, as in GGUF), so every block starts 16-byte aligned, then the row's d's,
// padded to 16 bytes (Gpu::upload_weights). 210 bytes a block and a few a row, where
// padding each block to 224 was 6% more to read for every token written (the Mac pads to
// 212 and reads words; here the tensor-core products read 16 bytes at a time).
#define Q6K_BYTES 208

__device__ __forceinline__ size_t q6_k_row_bytes(int cols) {
    const int blocks = cols / 256;
    return (size_t)blocks * Q6K_BYTES + ((blocks * 2 + 15) & ~15);
}

// Block b's d, as its 16 bits.
__device__ __forceinline__ u32 q6_k_d_bits(const u8* row, int blocks, int b) {
    return *reinterpret_cast<const u16*>(row + (size_t)blocks * Q6K_BYTES + b * 2);
}

// A lane's four scales (one a run of 32: the 16 its 4 numbers of that run are in) from one
// 8-byte load of its half's 8, not four loads of a byte. l is the lane's place in a run.
struct Q6Scales {
    uint2 w;
    int shift;
    __device__ __forceinline__ int at(int k) const {
        return (int)(signed char)(((k < 2 ? w.x : w.y) >> (shift + (k & 1) * 16)) & 0xFF);
    }
};

__device__ __forceinline__ Q6Scales q6_k_scales(const u8* block, int half, int l) {
    return {*reinterpret_cast<const uint2*>(block + 192 + half * 8), (l / 16) * 8};
}

__device__ __forceinline__ float q6_k_d(const u8* row, int blocks, int b) {
    return half_at(row + (size_t)blocks * Q6K_BYTES + b * 2);
}

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

// Q8_0 (the GPU layout, see q8_0_row_bytes). 8 lanes per block, 4 numbers each; a warp
// takes 4 blocks per step.
extern "C" __global__ void matvec_q8_0(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 3, j = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 4) {
        const float4 xv = f4(x + b * 32 + j);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const u32 q = u32_at4(row + b * 32 + j);
            const float dot = small_int((q & 0xFF) ^ 0x80, 128.0f) * xv.x + small_int(((q >> 8) & 0xFF) ^ 0x80, 128.0f) * xv.y +
                              small_int(((q >> 16) & 0xFF) ^ 0x80, 128.0f) * xv.z + small_int((q >> 24) ^ 0x80, 128.0f) * xv.w;
            sum[r] += q8_0_scale(row, blocks, b) * dot;
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

// Q6_K: blocks of 256 in 210 bytes (ql 128, qh 64, 16 int8 scales, f16 d), laid out on
// the GPU as Q6K_BYTES above says. Each half of a block is 4 runs of 32 numbers. 16 lanes per block: lane j takes 4 positions of
// one half (16 numbers, one from each run). A warp takes 2 blocks per step.
extern "C" __global__ void matvec_q6_k(const u8* w, const float* x, float* y, int rows, int cols) {
    MATVEC_ROWS
    const int blocks = cols / 256;
    const size_t row_bytes = q6_k_row_bytes(cols);
    const int sub = lane >> 4, half = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 2) {
        const float* xb = x + b * 256 + half * 128 + l;
        float4 xv[4];
#pragma unroll
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const u8* block = row + b * Q6K_BYTES;
            const float d = q6_k_d(row, blocks, b);
            const u32 ql0 = u32_at4(block + half * 64 + l), ql1 = u32_at4(block + half * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + half * 32 + l);
            const Q6Scales sc = q6_k_scales(block, half, l);
            float block_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const u32 low = ((k & 1) ? ql1 : ql0) >> (k < 2 ? 0 : 4);
                const float q0 = small_int((low & 0xF) | (((qh >> (2 * k)) & 3) << 4), 32.0f);
                const float q1 = small_int(((low >> 8) & 0xF) | (((qh >> (8 + 2 * k)) & 3) << 4), 32.0f);
                const float q2 = small_int(((low >> 16) & 0xF) | (((qh >> (16 + 2 * k)) & 3) << 4), 32.0f);
                const float q3 = small_int(((low >> 24) & 0xF) | (((qh >> (24 + 2 * k)) & 3) << 4), 32.0f);
                block_sum += (float)sc.at(k) * (q0 * xv[k].x + q1 * xv[k].y + q2 * xv[k].z + q3 * xv[k].w);
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
    // As Q4_0: the row's scales are past its numbers, so decode8_q8_0 takes the block count.
    (void)row;
    (void)k;
    (void)v;
}

__device__ __forceinline__ void decode8_q8_0(const u8* row, int blocks, int k, float* v) {
    const float d = q8_0_scale(row, blocks, k / 32);
    for (int i = 0; i < 8; i++) v[i] = d * (float)(signed char)row[k + i];
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
    // As Q4_0: the row's d's are past its blocks, so decode8_q6_k takes the block count.
    (void)row;
    (void)k;
    (void)v;
}

__device__ __forceinline__ void decode8_q6_k(const u8* row, int blocks, int k, float* v) {
    const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, half = n / 128, run = (n % 128) / 32, l = n % 32;
    const float d = q6_k_d(row, blocks, k / 256);
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
                else if (KIND == K_Q8_0)
                    decode8_q8_0(w + (size_t)r * row_bytes, cols / 32, k0 + load_k, v);
                else if (KIND == K_Q6_K)
                    decode8_q6_k(w + (size_t)r * row_bytes, cols / 256, k0 + load_k, v);
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
MATMUL(matmul_q8_0, K_Q8_0, q8_0_row_bytes(cols))
MATMUL(matmul_q4_k, K_Q4_K, cols / 256 * 144)
MATMUL(matmul_q5_k, K_Q5_K, cols / 256 * 176)
MATMUL(matmul_q6_k, K_Q6_K, q6_k_row_bytes(cols))

// ---- 8-bit activations: faster, not exact ---------------------------------------------
//
// The kernels above work in f32 on the exact weights and activations. Those below round
// the activations (x) to 8 bits first, as llama.cpp does, so the weights' small whole
// numbers can be multiplied 4 at a time (dp4a) or on the tensor cores (mma). Each block
// of 32 numbers of x gets a scale d = max|x| / 127 and q = round(x / d); `xs` keeps
// d × Σq for each 16, for the weights' minimums. Rows of x are `cols` long: xq holds cols
// bytes per row, xd cols / 32 floats and xs cols / 16.

// A batch's layout (the tensor-core kernels read it, matmul8): the bytes as above (a
// token's 128 for a step of the product on one 128-byte line), and in place of xd and xs
// a head of X8_HEAD floats for each token and 128 numbers: its four scales, then the four
// sums d × Σq of its 32s (the two 16s' sums added, as that kernel adds them). A step
// reads a token's scales and sums in one place, and half the sums. For models of
// K-quants (matmul8's HEAD): a ~2.3K-token read, Qwen 3.5 9B +5% and 4B +3%; Gemma 4
// E4B's Q4_0 products were 3% slower so, and Q4_0 models keep xd and xs. (All of it in one
// block of 160 bytes, bytes after head, cost Q4_0 9%: the bytes off their 128-byte line.)
#define X8_HEAD 8
__device__ __forceinline__ size_t x8_heads(int cols) { return (size_t)((cols + 127) / 128); }

// A warp's block of 32 (number g of token t's row of `cols`) into that layout: lane
// `lane`'s byte q, the scale d, and the sum of q over the lane's 16.
__device__ __forceinline__ void x8_store(signed char* xq, float* head, int cols, int t, int g, int lane, int q,
                                         float d, int sum16) {
    xq[(size_t)t * cols + g * 32 + lane] = (signed char)q;
    // Each 16's sum times d, then the two added: neither contracted, so the same f32 as
    // the one-token layout's two numbers added.
    const float part = __fmul_rn(d, (float)sum16);
    const float other = __shfl_xor_sync(0xffffffffu, part, 16);
    if (lane == 0) {
        float* h = head + ((size_t)t * x8_heads(cols) + g / 4) * X8_HEAD;
        h[g % 4] = d;
        h[4 + g % 4] = __fadd_rn(part, other);
    }
}

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

// quantize_q8 for x = act(gate) × up (SiLU, or Gemma's tanh-GELU), worked out here and
// not written anywhere as f32: the feed-forward's middle is only ever the next product's
// input. One pass over gate and up instead of the activation's pass and the rounding's
// (a 2,400-token prompt: Gemma 4 E4B's geglu 20 ms and about half of quantize_q8's 23).
// Rows of n (a multiple of 32); up's rows are `up_stride` apart.
template <bool GELU>
__device__ __forceinline__ float gated_value(float x, float u) {
    return GELU ? 0.5f * x * (1.0f + tanhf(0.7978846f * x * (1.0f + 0.044715f * x * x))) * u
                : x / (1.0f + expf(-x)) * u;
}

template <bool GELU>
__device__ __forceinline__ void gated_q8(const float* gate, const float* up, int n, int up_stride, signed char* xq,
                                         float* xd, float* xs, int blocks) {
    const int block = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (block >= blocks) return;
    const size_t i = (size_t)block * 32 + lane;
    const float x = gate[i], u = up[(i / n) * up_stride + i % n];
    const float v = gated_value<GELU>(x, u);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    xq[i] = (signed char)q;
    int sum = q;
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) xd[block] = d;
    if (lane == 0 || lane == 16) xs[(size_t)block * 2 + lane / 16] = d * (float)sum;
}
// quantize_q8 and the gated kernels below, into a batch's layout: a block of 8 warps takes
// 8 of a token's 32s, the token is blockIdx.y (no dividing to find where a 32 goes).
extern "C" __global__ void quantize_q8_tile(const float* x, signed char* xq, float* head, int cols) {
    const int t = blockIdx.y, g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (g >= cols / 32) return;
    const float v = x[(size_t)t * cols + g * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    int sum = q;
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    x8_store(xq, head, cols, t, g, lane, q, d, sum);
}
template <bool GELU>
__device__ __forceinline__ void gated_q8_tile(const float* gate, const float* up, int n, int up_stride,
                                              signed char* xq, float* head) {
    const int t = blockIdx.y, g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (g >= n / 32) return;
    const float v = gated_value<GELU>(gate[(size_t)t * n + g * 32 + lane], up[(size_t)t * up_stride + g * 32 + lane]);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    int sum = q;
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    x8_store(xq, head, n, t, g, lane, q, d, sum);
}
extern "C" __global__ void swiglu_q8_tile(const float* gate, const float* up, int n, int up_stride, signed char* xq,
                                          float* head) {
    gated_q8_tile<false>(gate, up, n, up_stride, xq, head);
}
extern "C" __global__ void geglu_q8_tile(const float* gate, const float* up, int n, int up_stride, signed char* xq,
                                         float* head) {
    gated_q8_tile<true>(gate, up, n, up_stride, xq, head);
}
extern "C" __global__ void swiglu_q8(const float* gate, const float* up, int n, int up_stride, signed char* xq,
                                     float* xd, float* xs, int blocks) {
    gated_q8<false>(gate, up, n, up_stride, xq, xd, xs, blocks);
}
extern "C" __global__ void geglu_q8(const float* gate, const float* up, int n, int up_stride, signed char* xq,
                                    float* xd, float* xs, int blocks) {
    gated_q8<true>(gate, up, n, up_stride, xq, xd, xs, blocks);
}

__device__ __forceinline__ int i32_at(const signed char* p) { return *reinterpret_cast<const int*>(p); }

// The dp4a matrix-vector kernels: as the f32 ones (MV_ROWS rows a warp, lanes laid out
// the same), with x as 8-bit numbers.

// Q8_0: d × dx × Σ q qx per block of 32. 2 lanes per block, 16 bytes each in one load; a
// warp takes 16 blocks (512 bytes of a row) per step. With 8 lanes of 4 bytes a block a
// step was 128 bytes of a row, and the reads' latency, not memory, set the pace
// (Qwen 3.5 4B's ssm_out 49 us for 11 MB; its 32-row ssm_alpha 15 us).
#define Q8_STEP 16
__device__ __forceinline__ int dot16(uint4 q, uint4 x) {
    return __dp4a((int)q.w, (int)x.w, __dp4a((int)q.z, (int)x.z, __dp4a((int)q.y, (int)x.y, __dp4a((int)q.x, (int)x.x, 0))));
}

extern "C" __global__ void matvec8_q8_0(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    const int blocks = cols / 32;
    const size_t row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 16;
    for (int b = sub; b < blocks; b += Q8_STEP) {
        const uint4 xv = *reinterpret_cast<const uint4*>(xq + b * 32 + j);
        const float dx = xd[b];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const uint4 q = *reinterpret_cast<const uint4*>(row + b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * dx * (float)dot16(q, xv);
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
// matvec_q6_k. A row and step is 5 loads a lane (three words, the half's scales, d) where
// it was 11 (each word as two halves, each scale a byte), and x's four scales one: the
// loads were what a token's pass waited on.
// W words (1, 2 or 4) from a 4W-byte-aligned address, in one load.
template <int W>
__device__ __forceinline__ void words_at(const void* p, u32* out) {
    if constexpr (W == 1) {
        out[0] = *reinterpret_cast<const u32*>(p);
    } else if constexpr (W == 2) {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        out[0] = v.x, out[1] = v.y;
    } else {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        out[0] = v.x, out[1] = v.y, out[2] = v.z, out[3] = v.w;
    }
}

// Words a lane: 1. With 2 (8 bytes of each run a lane, 4 blocks a step) writing was 6-9%
// faster while the laptop's GPU was held to 45 W and level to 2% slower at full power, and
// the sums are added in another order (Gemma 4 E4B's second token changed).
#define Q6_W 1
extern "C" __global__ void matvec8_q6_k(const u8* w, const signed char* xq, const float* xd, float* y, int rows,
                                        int cols) {
    MATVEC_ROWS
    // A lane takes W words of each run of a half (4W numbers of each of its 4 runs), so
    // 16 / W lanes a block and 2W blocks a step.
    constexpr int W = Q6_W, LB = 16 / W;
    const int blocks = cols / 256;
    const size_t row_bytes = q6_k_row_bytes(cols);
    const int sub = lane / LB, half = (lane % LB) / (LB / 2), l = (lane % (LB / 2)) * 4 * W;
    for (int b = sub; b < blocks; b += 32 / LB) {
        const signed char* xb = xq + b * 256 + half * 128 + l;
        u32 xv[4][W];
        float dx[4];
#pragma unroll
        for (int k = 0; k < 4; k++) words_at<W>(xb + k * 32, xv[k]);
        // The half's four scales of x in one load (xd is 16-byte aligned, 8 a block).
        const float4 dxv = f4(xd + b * 8 + half * 4);
        dx[0] = dxv.x, dx[1] = dxv.y, dx[2] = dxv.z, dx[3] = dxv.w;
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const u8* block = row + b * Q6K_BYTES;
            const float d = q6_k_d(row, blocks, b);
            u32 ql0[W], ql1[W], qh[W];
            words_at<W>(block + half * 64 + l, ql0);
            words_at<W>(block + half * 64 + 32 + l, ql1);
            words_at<W>(block + 128 + half * 32 + l, qh);
            const Q6Scales sc = q6_k_scales(block, half, l);
            float block_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int dot = 0;
#pragma unroll
                for (int i = 0; i < W; i++) {
                    const u32 low = (((k & 1) ? ql1[i] : ql0[i]) >> (k < 2 ? 0 : 4)) & 0x0F0F0F0Fu;
                    const u32 q = low | (((qh[i] >> (2 * k)) & 0x03030303u) << 4);
                    dot = __dp4a((int)sext6(q), (int)xv[k][i], dot);
                }
                block_sum += (float)sc.at(k) * dx[k] * (float)dot;
            }
            sum[r] += d * block_sum;
        }
    }
    MATVEC_STORE
}

// ---- A few tokens at once: the dp4a kernels with T rows of x ---------------------------
//
// A draft's guesses checked in one pass, a short tool result: 2 to 8 tokens. The tensor-core
// tiles below cost the same for 2 tokens as for 64 (about two tokens' time: a tile is
// unpacked through shared memory whatever it holds), so these read the weights as the
// one-token kernels do, at the memory's speed, and multiply each word by every token's x
// while it's in registers. The same sums in the same order as one token at a time. Token
// t's x is row t of xq (cols bytes) and xd (cols / 32), its results row t of y; a token
// from `tokens` on is skipped (T is the kernel's size, `tokens` what the batch holds).

#define FEW_ROWS(T)                                                                         \
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; \
    const int row0 = warp * MV_ROWS;                                                        \
    if (row0 >= rows) return;                                                               \
    const int nrows = min(MV_ROWS, rows - row0);                                            \
    const int xd_cols = cols / 32;                                                          \
    float sum[MV_ROWS][T];                                                                  \
    for (int r = 0; r < MV_ROWS; r++)                                                       \
        for (int t = 0; t < T; t++) sum[r][t] = 0.0f;

#define FEW_STORE(T)                                                                          \
    for (int r = 0; r < MV_ROWS; r++)                                                         \
        for (int t = 0; t < T; t++) {                                                         \
            const float total = warp_sum(sum[r][t]);                                          \
            if (lane == 0 && r < nrows && t < tokens) y[(size_t)t * rows + row0 + r] = total; \
        }

template <int T>
__device__ __forceinline__ void matvec8_few_q8_0(const u8* w, const signed char* xq, const float* xd, float* y,
                                                 int rows, int cols, int tokens) {
    FEW_ROWS(T)
    const int blocks = cols / 32;
    const size_t row_bytes = q8_0_row_bytes(cols);
    // Lanes as in matvec8_q8_0, so the sums are its sums.
    const int sub = lane >> 1, j = (lane & 1) * 16;
    for (int b = sub; b < blocks; b += Q8_STEP) {
        uint4 xv[T];
        float dx[T];
#pragma unroll
        for (int t = 0; t < T; t++) {
            const bool on = t < tokens;
            xv[t] = on ? *reinterpret_cast<const uint4*>(xq + (size_t)t * cols + b * 32 + j) : make_uint4(0, 0, 0, 0);
            dx[t] = on ? xd[(size_t)t * xd_cols + b] : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const float d = q8_0_scale(row, blocks, b);
            const uint4 q = *reinterpret_cast<const uint4*>(row + b * 32 + j);
#pragma unroll
            for (int t = 0; t < T; t++) sum[r][t] += d * dx[t] * (float)dot16(q, xv[t]);
        }
    }
    FEW_STORE(T)
}

template <int T>
__device__ __forceinline__ void matvec8_few_q4_0(const u8* w, const signed char* xq, const float* xd, float* y,
                                                 int rows, int cols, int tokens) {
    FEW_ROWS(T)
    const int blocks = cols / 32;
    const size_t row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub; b < blocks; b += 16) {
        int2 lo[T], hi[T];
        float dx[T];
#pragma unroll
        for (int t = 0; t < T; t++) {
            const bool on = t < tokens;
            const signed char* xt = xq + (size_t)t * cols + b * 32 + j;
            lo[t] = on ? *reinterpret_cast<const int2*>(xt) : make_int2(0, 0);
            hi[t] = on ? *reinterpret_cast<const int2*>(xt + 16) : make_int2(0, 0);
            dx[t] = on ? xd[(size_t)t * xd_cols + b] : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<const uint2*>(row + b * 16 + j);
            const int a0 = (int)sext4(q.x & 0x0F0F0F0Fu), a1 = (int)sext4(q.y & 0x0F0F0F0Fu);
            const int a2 = (int)sext4((q.x >> 4) & 0x0F0F0F0Fu), a3 = (int)sext4((q.y >> 4) & 0x0F0F0F0Fu);
            const float d = q4_0_scale(row, blocks, b);
#pragma unroll
            for (int t = 0; t < T; t++) {
                int dot = __dp4a(a0, lo[t].x, 0);
                dot = __dp4a(a1, lo[t].y, dot);
                dot = __dp4a(a2, hi[t].x, dot);
                dot = __dp4a(a3, hi[t].y, dot);
                sum[r][t] += d * dx[t] * (float)dot;
            }
        }
    }
    FEW_STORE(T)
}

// Q4_K and Q5_K (FIFTH: the fifth bits put into the bytes), lanes as in matvec8_q4_k.
template <int T, bool FIFTH>
__device__ __forceinline__ void matvec8_few_q45_k(const u8* w, const signed char* xq, const float* xd, float* y,
                                                  int rows, int cols, int tokens) {
    FEW_ROWS(T)
    const int blocks = cols / 256;
    constexpr int bytes = FIFTH ? 176 : 144, qs = FIFTH ? 48 : 16;
    const size_t row_bytes = (size_t)blocks * bytes;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    struct Words {
        uint4 head;
        uint2 q, qh;
    };
    auto fetch = [&](int r, int b) {
        const u8* block = w + (row0 + r) * row_bytes + b * bytes;
        Words v;
        v.head = *reinterpret_cast<const uint4*>(block);
        v.q = *reinterpret_cast<const uint2*>(block + qs + pair * 32 + part);
        v.qh = FIFTH ? *reinterpret_cast<const uint2*>(block + 16 + part) : make_uint2(0, 0);
        return v;
    };
    Words ahead[MV_ROWS] = {};
    if (sub < blocks) {
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++)
            if (r < nrows) ahead[r] = fetch(r, sub);
    }
    for (int b = sub; b < blocks; b += 2) {
        int x1a[T], x1b[T], x2a[T], x2b[T];
        float sum1[T], sum2[T], dx1[T], dx2[T];
#pragma unroll
        for (int t = 0; t < T; t++) {
            const bool on = t < tokens;
            const signed char* x1 = xq + (size_t)t * cols + b * 256 + pair * 64 + part;
            x1a[t] = on ? i32_at(x1) : 0;
            x1b[t] = on ? i32_at(x1 + 4) : 0;
            x2a[t] = on ? i32_at(x1 + 32) : 0;
            x2b[t] = on ? i32_at(x1 + 36) : 0;
            sum1[t] = (float)__dp4a(0x01010101, x1b[t], __dp4a(0x01010101, x1a[t], 0));
            sum2[t] = (float)__dp4a(0x01010101, x2b[t], __dp4a(0x01010101, x2a[t], 0));
            dx1[t] = on ? xd[(size_t)t * xd_cols + b * 8 + 2 * pair] : 0.0f;
            dx2[t] = on ? xd[(size_t)t * xd_cols + b * 8 + 2 * pair + 1] : 0.0f;
        }
        // The next step's weights, read before this step's arithmetic: with several
        // tokens' numbers in registers an SM holds fewer warps, so each keeps a read in
        // flight while it multiplies.
        Words now[MV_ROWS];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) now[r] = ahead[r];
        if (b + 2 < blocks) {
#pragma unroll
            for (int r = 0; r < MV_ROWS; r++)
                if (r < nrows) ahead[r] = fetch(r, b + 2);
        }
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const uint4 head = now[r].head;
            float s1, m1, s2, m2;
            q4_k_scale_min_w(2 * pair, head.y, head.z, head.w, s1, m1);
            q4_k_scale_min_w(2 * pair + 1, head.y, head.z, head.w, s2, m2);
            const float d = half_lo(head.x), dmin = half_hi(head.x);
            const uint2 q = now[r].q;
            int q1a = (int)(q.x & 0x0F0F0F0Fu), q1b = (int)(q.y & 0x0F0F0F0Fu);
            int q2a = (int)((q.x >> 4) & 0x0F0F0F0Fu), q2b = (int)((q.y >> 4) & 0x0F0F0F0Fu);
            if constexpr (FIFTH) {
                const uint2 qh = now[r].qh;
                const int s = 2 * pair;
                q1a |= (int)(((qh.x >> s) & 0x01010101u) << 4);
                q1b |= (int)(((qh.y >> s) & 0x01010101u) << 4);
                q2a |= (int)(((qh.x >> (s + 1)) & 0x01010101u) << 4);
                q2b |= (int)(((qh.y >> (s + 1)) & 0x01010101u) << 4);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int dot1 = __dp4a(q1b, x1b[t], __dp4a(q1a, x1a[t], 0));
                const int dot2 = __dp4a(q2b, x2b[t], __dp4a(q2a, x2a[t], 0));
                sum[r][t] += dx1[t] * (d * s1 * (float)dot1 - dmin * m1 * sum1[t]) +
                             dx2[t] * (d * s2 * (float)dot2 - dmin * m2 * sum2[t]);
            }
        }
    }
    FEW_STORE(T)
}

template <int T>
__device__ __forceinline__ void matvec8_few_q6_k(const u8* w, const signed char* xq, const float* xd, float* y,
                                                 int rows, int cols, int tokens) {
    FEW_ROWS(T)
    // Lanes as in matvec8_q6_k (W words of each run a lane), so the sums are its sums.
    constexpr int W = Q6_W, LB = 16 / W, STEP = 32 / LB;
    const int blocks = cols / 256;
    const size_t row_bytes = q6_k_row_bytes(cols);
    const int sub = lane / LB, half = (lane % LB) / (LB / 2), l = (lane % (LB / 2)) * 4 * W;
    // A step's weight words: the low bits of two runs' bytes, the high bits, d, and the
    // half's 8 scales.
    struct Words {
        u32 ql0[W], ql1[W], qh[W], d;
        Q6Scales sc;
    };
    auto fetch = [&](int r, int b) {
        const u8* row = w + (row0 + r) * row_bytes;
        const u8* block = row + b * Q6K_BYTES;
        Words v;
        words_at<W>(block + half * 64 + l, v.ql0);
        words_at<W>(block + half * 64 + 32 + l, v.ql1);
        words_at<W>(block + 128 + half * 32 + l, v.qh);
        v.d = q6_k_d_bits(row, blocks, b);
        v.sc = q6_k_scales(block, half, l);
        return v;
    };
    Words ahead[MV_ROWS] = {};
    if (sub < blocks) {
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++)
            if (r < nrows) ahead[r] = fetch(r, sub);
    }
    for (int b = sub; b < blocks; b += STEP) {
        u32 xv[T][4][W];
        float dx[T][4];
#pragma unroll
        for (int t = 0; t < T; t++) {
            const bool on = t < tokens;
            const signed char* xb = xq + (size_t)t * cols + b * 256 + half * 128 + l;
#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int i = 0; i < W; i++) xv[t][k][i] = 0;
                if (on) words_at<W>(xb + k * 32, xv[t][k]);
            }
            const float4 dxv = on ? f4(xd + (size_t)t * xd_cols + b * 8 + half * 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            dx[t][0] = dxv.x, dx[t][1] = dxv.y, dx[t][2] = dxv.z, dx[t][3] = dxv.w;
        }
        // The next step's weights before this step's arithmetic (see Q4_K's).
        Words now[MV_ROWS];
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) now[r] = ahead[r];
        if (b + STEP < blocks) {
#pragma unroll
            for (int r = 0; r < MV_ROWS; r++)
                if (r < nrows) ahead[r] = fetch(r, b + STEP);
        }
#pragma unroll
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            const float d = __half2float(__ushort_as_half((unsigned short)now[r].d));
            int q[4][W];
            float scale[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int i = 0; i < W; i++) {
                    const u32 low = (((k & 1) ? now[r].ql1[i] : now[r].ql0[i]) >> (k < 2 ? 0 : 4)) & 0x0F0F0F0Fu;
                    q[k][i] = (int)sext6(low | (((now[r].qh[i] >> (2 * k)) & 0x03030303u) << 4));
                }
                scale[k] = (float)now[r].sc.at(k);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                float block_sum = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    int dot = 0;
#pragma unroll
                    for (int i = 0; i < W; i++) dot = __dp4a(q[k][i], (int)xv[t][k][i], dot);
                    block_sum += scale[k] * dx[t][k] * (float)dot;
                }
                sum[r][t] += d * block_sum;
            }
        }
    }
    FEW_STORE(T)
}

#define MATVEC8_FEW(T)                                                                                     \
    extern "C" __global__ void matvec8_few##T##_q4_0(const u8* w, const signed char* xq, const float* xd, \
                                                     float* y, int rows, int cols, int tokens) {          \
        matvec8_few_q4_0<T>(w, xq, xd, y, rows, cols, tokens);                                            \
    }                                                                                                      \
    extern "C" __global__ void matvec8_few##T##_q8_0(const u8* w, const signed char* xq, const float* xd, \
                                                     float* y, int rows, int cols, int tokens) {          \
        matvec8_few_q8_0<T>(w, xq, xd, y, rows, cols, tokens);                                            \
    }                                                                                                      \
    extern "C" __global__ void matvec8_few##T##_q4_k(const u8* w, const signed char* xq, const float* xd, \
                                                     float* y, int rows, int cols, int tokens) {          \
        matvec8_few_q45_k<T, false>(w, xq, xd, y, rows, cols, tokens);                                    \
    }                                                                                                      \
    extern "C" __global__ void matvec8_few##T##_q5_k(const u8* w, const signed char* xq, const float* xd, \
                                                     float* y, int rows, int cols, int tokens) {          \
        matvec8_few_q45_k<T, true>(w, xq, xd, y, rows, cols, tokens);                                     \
    }                                                                                                      \
    extern "C" __global__ void matvec8_few##T##_q6_k(const u8* w, const signed char* xq, const float* xd, \
                                                     float* y, int rows, int cols, int tokens) {          \
        matvec8_few_q6_k<T>(w, xq, xd, y, rows, cols, tokens);                                            \
    }
MATVEC8_FEW(2)
MATVEC8_FEW(3)
MATVEC8_FEW(4)
MATVEC8_FEW(5)
MATVEC8_FEW(6)
MATVEC8_FEW(7)
MATVEC8_FEW(8)

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

// Four 8 x 8 matrices of 2-byte numbers from shared memory (their rows from lanes 0-7,
// 8-15, 16-23, 24-31): thread (gid, tig) gets bytes 4 tig to 4 tig + 3 of row gid of each.
__device__ __forceinline__ void ldmatrix_x4_bytes(u32 (&r)[4], const u8* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(static_cast<u32>(__cvta_generic_to_shared(p))));
}

// NT is the tile's tokens: 64, or 32 or 16 for a short batch (a tool's result, a new turn),
// whose 8 warps then take 16 rows each and multiply only the tokens there are: a tile of
// 64 costs the same for 9 tokens as for 64. Or 128, a warp taking 32 rows by 64 tokens:
// the rows' weights are then read and unpacked once for twice the tokens, which pays for
// the K-quants (the host's WIDE_TOKENS).
//
// GLU (1 SiLU, 2 Gemma's tanh-GELU) is a feed-forward's first half in one kernel: w is
// its gate matrix and w2 its up matrix, a block takes 64 rows of each (16 of one, then the
// same 16 of the other, so a thread holds both sums of a row and token), and what it
// writes is act(gate) x up rounded to 8 bits as quantize_q8 would round it (oq, od, os):
// the next product's input. Neither product is written as f32 (rows x tokens x 8 bytes a
// layer, read back once by the activation). Tiles of 64 or 128 tokens; rows in 64s.
//
// HEAD is how x's scales and sums come (and how GLU writes them): false, xa is xd and xb
// xs; true, xa is a batch's heads (X8_HEAD) and xb isn't read. oa and ob likewise.
template <int KIND, int GROUP, int NT, int GLU = 0, bool HEAD = false>
__device__ void matmul8(const u8* w, const u8* w2, size_t row_bytes, const signed char* xq, const float* xa,
                        const float* xb, float* y, signed char* oq, float* oa, float* ob, int rows, int cols,
                        int tokens) {
    static_assert(GLU == 0 || NT >= 64, "the gated kernel needs a warp's two tiles of 16 rows");
    // 128 columns a step: 64 tensor-core products per warp between syncs. Rows of 128
    // bytes padded to 144 in shared memory, so reading fragments has no bank conflicts.
    constexpr int K = 128, KS = K + 16, NG = K / GROUP, NX = K / 32;
    // Row tiles of 16 and token tiles of 8 a warp, and the threads' turns storing x.
    constexpr int MI = NT >= 64 ? 2 : 1, NI = NT == 128 ? 8 : NT == 16 ? 2 : 4, XI = NT == 128 ? 4 : NT == 64 ? 2 : 1;
    __shared__ __align__(16) u8 wq[M8_ROWS][KS];
    __shared__ __align__(16) u8 xqs[NT][KS];
    // Scales by group, then row or token, so a warp reads them in pairs (a row's scale
    // with its minimum; a thread's two tokens' scales or sums) without bank conflicts.
    // x_sum is per 32 columns: the pairs of 16 added as they're stored.
    __shared__ __align__(16) float2 w_sm[NG][M8_ROWS];
    __shared__ __align__(16) float x_scale[NX][NT], x_sum[NX][NT];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * (GLU ? M8_ROWS / 2 : M8_ROWS), tok0 = blockIdx.y * NT;
    // The matrix's row at the tile's row r, and which matrix.
    auto tile_row = [&](int r) -> int { return GLU ? row0 + (r >> 5) * 16 + (r & 15) : row0 + r; };
    auto tile_w = [&](int r) -> const u8* { return GLU && (r & 16) ? w2 : w; };
    // This warp's rows and tokens in the tile.
    const int wr = NT >= 64 ? (warp & 3) * 32 : warp * 16, wt = NT >= 64 ? (warp >> 2) * (NT / 2) : 0;
    constexpr bool has_min = KIND == K_Q4_K || KIND == K_Q5_K;
    float acc[MI][NI][4] = {};
    const size_t x_heads = x8_heads(cols);
    const int xd_cols = cols / 32, xs_cols = cols / 16;

    // The next step's raw bytes are read into registers before this step's products, and
    // unpacked into shared memory after them, so the reads' wait overlaps the arithmetic.
    struct Raw {
        uint4 head, v[2], h[2];  // Q4_K, Q5_K: a pair of sub-blocks; Q6_K: v low, h high bits
        uint4 q[4];              // Q4_0, Q8_0: 4 runs of 16
        u32 sc[4];               // their scales (f16 bits); Q6_K: d, scales
        bool ok[4];
        uint4 xq[XI];
        float xsum[XI], xscale[XI];
    } raw;
    auto fetch = [&](int k0) {
        if constexpr (KIND == K_Q4_K || KIND == K_Q5_K) {
            const int r = tid >> 1, p = tid & 1, k = k0 + p * 64;
            raw.ok[0] = tile_row(r) < rows && k < cols;
            if (raw.ok[0]) {
                constexpr int bytes = KIND == K_Q4_K ? 144 : 176, qs = KIND == K_Q4_K ? 16 : 48;
                const u8* block = tile_w(r) + (size_t)tile_row(r) * row_bytes + (k / 256) * bytes;
                const int j = (k % 256) / 32;
                raw.head = *reinterpret_cast<const uint4*>(block);
                for (int i = 0; i < 2; i++) {
                    raw.v[i] = *reinterpret_cast<const uint4*>(block + qs + (j / 2) * 32 + i * 16);
                    if (KIND == K_Q5_K) raw.h[i] = *reinterpret_cast<const uint4*>(block + 16 + i * 16);
                }
            }
        } else if constexpr (KIND == K_Q6_K) {
            const int r = tid >> 1, p = tid & 1;
            raw.ok[0] = tile_row(r) < rows && k0 < cols;
            if (raw.ok[0]) {
                const u8* row = tile_w(r) + (size_t)tile_row(r) * row_bytes;
                const u8* block = row + (k0 / 256) * Q6K_BYTES;
                const int half = (k0 % 256) / 128;
                raw.sc[0] = q6_k_d_bits(row, cols / 256, k0 / 256);
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
                raw.ok[i] = tile_row(r) < rows && k < cols;
                if (raw.ok[i]) {
                    const u8* row = tile_w(r) + (size_t)tile_row(r) * row_bytes;
                    if constexpr (KIND == K_Q4_0) {
                        const int blocks = cols / 32, b = k / 32;
                        raw.q[i] = *reinterpret_cast<const uint4*>(row + b * 16);
                        raw.sc[i] = *reinterpret_cast<const u16*>(row + (size_t)blocks * 16 + b * 2);
                    } else {
                        // Q8_0: a run of 16 is 16 bytes of the row.
                        const int blocks = cols / 32, b = k / 32;
                        raw.q[i] = *reinterpret_cast<const uint4*>(row + k);
                        raw.sc[i] = *reinterpret_cast<const u16*>(row + (size_t)blocks * 32 + b * 2);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < XI; i++) {
            const int c = tid + i * 256, t = c >> 3, h = c & 7, k = k0 + h * 16;
            raw.xq[i] = make_uint4(0, 0, 0, 0);
            raw.xsum[i] = raw.xscale[i] = 0.0f;
            if (t < NT && tok0 + t < tokens && k < cols) {
                raw.xq[i] = *reinterpret_cast<const uint4*>(xq + (size_t)(tok0 + t) * cols + k);
                if constexpr (HEAD) {
                    // The scale and the sum of this thread's 32, from the step's head.
                    const float* head = xa + ((size_t)(tok0 + t) * x_heads + k0 / K) * X8_HEAD;
                    raw.xscale[i] = head[h / 2];
                    if constexpr (has_min) raw.xsum[i] = head[4 + h / 2];
                } else {
                    raw.xsum[i] = xb[(size_t)(tok0 + t) * xs_cols + k / 16];
                    raw.xscale[i] = xa[(size_t)(tok0 + t) * xd_cols + k / 32];
                }
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
            // A run of 32's two scales stay whole numbers (as two 16-bit halves of a word) and
            // the block's d a float: the products scale a 32's two sums together (below).
            float d = 0.0f;
            int sc[2] = {};
            if (raw.ok[0]) {
                d = __half2float(__ushort_as_half((unsigned short)raw.sc[0]));
                sc[0] = ((int)(signed char)(raw.sc[1] & 0xFF) & 0xFFFF) | ((int)(signed char)(raw.sc[1] >> 8) << 16);
                sc[1] = ((int)(signed char)(raw.sc[2] & 0xFF) & 0xFFFF) | ((int)(signed char)(raw.sc[2] >> 8) << 16);
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
            w_sm[p][r] = make_float2(d, __int_as_float(sc[0]));
            w_sm[p + 2][r] = make_float2(d, __int_as_float(sc[1]));
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
                        scale = __half2float(__ushort_as_half((unsigned short)raw.sc[i]));
                    }
                }
                *reinterpret_cast<uint4*>(&wq[r][h * 16]) = q;
                if (GROUP == 16 || (h & 1) == 0) w_sm[GROUP == 16 ? h : h / 2][r].x = scale;
            }
        }
        // x: NT tokens × 8 runs of 16, 2 a thread for 64 (one or none for fewer).
#pragma unroll
        for (int i = 0; i < XI; i++) {
            const int c = tid + i * 256, t = c >> 3, h = c & 7;
            // A head has a 32's sum; else the runs h and h + 1 of a 32 are on neighbouring lanes.
            const float sum32 = HEAD ? raw.xsum[i] : raw.xsum[i] + __shfl_xor_sync(0xFFFFFFFFu, raw.xsum[i], 1);
            if (t >= NT) continue;
            *reinterpret_cast<uint4*>(&xqs[t][h * 16]) = raw.xq[i];
            if ((h & 1) == 0) {
                x_scale[h / 2][t] = raw.xscale[i];
                x_sum[h / 2][t] = sum32;
            }
        }
    };

    // Not for Q6_K's tiles of 64 tokens and under: they run two blocks per SM, and the raw
    // bytes would spill there (one block with the read ahead measured 3% slower than two
    // without). Its tile of 128, one block an SM, reads ahead (Qwen 3.5 9B's ffn_down,
    // 1,024 tokens: 3.50 -> 3.14 ms).
    constexpr bool prefetch = KIND != K_Q6_K || NT == 128 || GLU != 0;
    if constexpr (prefetch) fetch(0);
    for (int k0 = 0; k0 < cols; k0 += K) {
        if constexpr (!prefetch) fetch(k0);
        store(k0);
        __syncthreads();
        if constexpr (prefetch)
            if (k0 + K < cols) fetch(k0 + K);

        if constexpr (KIND == K_Q6_K) {
            // Q6_K has a scale every 16 numbers, a whole number, and x one every 32. A 32's
            // two sums are put together as whole numbers, each times its scale (under 2^24,
            // so exact in f32 too), and scaled into the results once: half the float work
            // of scaling each 16 (which was 15% of a read: a row's scale read, a multiply
            // and a multiply-add for every result and 16). Four token tiles at a time, so
            // both 16s' sums are in registers together.
            constexpr int NB = 1;
#pragma unroll
            for (int g = 0; g < K / 32; g++) {
                const int kk = g * 32;
                u32 a[MI][2][2];
                float wd[MI][2];
                int s1[MI][2], s2[MI][2];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int r = wr + mi * 16 + gid + h * 8;
                        a[mi][0][h] = *reinterpret_cast<const u32*>(&wq[r][kk + tig * 4]);
                        a[mi][1][h] = *reinterpret_cast<const u32*>(&wq[r][kk + 16 + tig * 4]);
                        const float2 sm = w_sm[g][r];
                        const int both = __float_as_int(sm.y);
                        wd[mi][h] = sm.x;
                        s1[mi][h] = (short)(both & 0xFFFF);
                        s2[mi][h] = both >> 16;
                    }
#pragma unroll
                for (int nh = 0; nh < NI; nh += NB) {
                    int c1[MI][NB][4] = {}, c2[MI][NB][4] = {};
                    float xsc[NB][2];
#pragma unroll
                    for (int x = 0; x < NB; x++) {
                        const u8* row = &xqs[wt + (nh + x) * 8 + gid][kk + tig * 4];
                        const u32 b1 = *reinterpret_cast<const u32*>(row);
                        const u32 b2 = *reinterpret_cast<const u32*>(row + 16);
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) {
                            mma_s8(c1[mi][x], a[mi][0][0], a[mi][0][1], b1);
                            mma_s8(c2[mi][x], a[mi][1][0], a[mi][1][1], b2);
                        }
                        const float2 s = *reinterpret_cast<const float2*>(&x_scale[g][wt + (nh + x) * 8 + tig * 2]);
                        xsc[x][0] = s.x;
                        xsc[x][1] = s.y;
                    }
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int x = 0; x < NB; x++)
#pragma unroll
                            for (int i = 0; i < 4; i++) {
                                const int h = i >> 1, j = i & 1;
                                const int both = c1[mi][x][i] * s1[mi][h] + c2[mi][x][i] * s2[mi][h];
                                acc[mi][nh + x][i] = fmaf((float)both, wd[mi][h] * xsc[x][j], acc[mi][nh + x][i]);
                            }
                }
            }
        } else {
#pragma unroll
        for (int g = 0; g < NG; g++) {
            int c[MI][NI][4] = {};
            if (GROUP == 32) {
                // A whole group in one product a tile (mma.m16n8k32). One side's
                // fragments come by ldmatrix, four reads in one: the weights' where there
                // are no minimums, x's where there are (a ~2.3K-token read: Gemma 4 E4B
                // 377 -> 357 ms, Qwen 3.5 4B 653 -> 636; the other side for each a little
                // less, and both sides at once 366 and 675: the reads queue for shared
                // memory, and which mix suits a kernel was only found by timing).
                constexpr bool LDA = !has_min, LDB = has_min;
                const int kk = g * 32;
                u32 a[MI][4], b[NI][2];
                const int mat = lane >> 3, mrow = lane & 7;
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    if (LDA) {
                        ldmatrix_x4_bytes(a[mi], &wq[wr + mi * 16 + (mat & 1) * 8 + mrow][kk + (mat >> 1) * 16]);
                    } else {
                        const u8* r0 = &wq[wr + mi * 16 + gid][kk + tig * 4];
                        const u8* r1 = &wq[wr + mi * 16 + gid + 8][kk + tig * 4];
                        a[mi][0] = *reinterpret_cast<const u32*>(r0);
                        a[mi][1] = *reinterpret_cast<const u32*>(r1);
                        a[mi][2] = *reinterpret_cast<const u32*>(r0 + 16);
                        a[mi][3] = *reinterpret_cast<const u32*>(r1 + 16);
                    }
                }
#pragma unroll
                for (int ni = 0; ni < NI; ni += 2) {
                    if (LDB) {
                        u32 r[4];
                        ldmatrix_x4_bytes(r, &xqs[wt + (ni + (mat >> 1)) * 8 + mrow][kk + (mat & 1) * 16]);
                        b[ni][0] = r[0], b[ni][1] = r[1], b[ni + 1][0] = r[2], b[ni + 1][1] = r[3];
                    } else {
#pragma unroll
                        for (int x = ni; x < ni + 2; x++) {
                            const u8* t = &xqs[wt + x * 8 + gid][kk + tig * 4];
                            b[x][0] = *reinterpret_cast<const u32*>(t);
                            b[x][1] = *reinterpret_cast<const u32*>(t + 16);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int ni = 0; ni < NI; ni++) mma_s8_k32(c[mi][ni], a[mi], b[ni]);
            } else {
#pragma unroll
                for (int kk = g * GROUP; kk < (g + 1) * GROUP; kk += 16) {
                    u32 a[MI][2], b[NI];
#pragma unroll
                    for (int mi = 0; mi < MI; mi++) {
                        a[mi][0] = *reinterpret_cast<const u32*>(&wq[wr + mi * 16 + gid][kk + tig * 4]);
                        a[mi][1] = *reinterpret_cast<const u32*>(&wq[wr + mi * 16 + gid + 8][kk + tig * 4]);
                    }
#pragma unroll
                    for (int ni = 0; ni < NI; ni++)
                        b[ni] = *reinterpret_cast<const u32*>(&xqs[wt + ni * 8 + gid][kk + tig * 4]);
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int ni = 0; ni < NI; ni++) mma_s8(c[mi][ni], a[mi][0], a[mi][1], b[ni]);
                }
            }
            // Scale this group's whole-number sums into the results, reading each row's and
            // token's scales once.
            float ws[MI][2], wm[MI][2], xsc[NI][2], xsm[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
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
            for (int ni = 0; ni < NI; ni++) {
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
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++)
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        // Two fused multiply-adds straight into the sum (the minimum's term
                        // added on its own, not first subtracted from the product).
                        const int h = i >> 1, j = i & 1;
                        acc[mi][ni][i] = fmaf((float)c[mi][ni][i], ws[mi][h] * xsc[ni][j], acc[mi][ni][i]);
                        if (has_min) acc[mi][ni][i] = fmaf(wm[mi][h], xsm[ni][j], acc[mi][ni][i]);
                    }
        }
        }
        __syncthreads();
    }
    if constexpr (GLU != 0) {
        // This thread's rows gid and gid + 8 of the warp's 16, for two tokens a token tile:
        // acc[0] is gate's sums and acc[1] up's. A token's 32 rows (one scale) are this
        // warp's 16 and the next warp's (warp ^ 1), so each leaves its largest in shared
        // memory for the other; a token's sum of 16 is the warp's own.
        __shared__ float largest[8][NT / 2];
        __shared__ float sums[8][NT / 2];
        const int orow = row0 + (warp & 3) * 16;
        const size_t o_heads = x8_heads(rows);
        float scale[NI][2];
        float v[NI][4];
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
#pragma unroll
            for (int i = 0; i < 4; i++) v[ni][i] = gated_value<GLU == 2>(acc[0][ni][i], acc[1][ni][i]);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                float m = fmaxf(fabsf(v[ni][j]), fabsf(v[ni][2 + j]));
                for (int o = 4; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                if (gid == 0) largest[warp][ni * 8 + tig * 2 + j] = m;
            }
        }
        __syncthreads();
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int at = ni * 8 + tig * 2 + j, t = tok0 + wt + at;
                const float amax = fmaxf(largest[warp][at], largest[warp ^ 1][at]);
                const float d = amax / 127.0f;
                const int q0 = amax == 0.0f ? 0 : __float2int_rn(v[ni][j] / d);
                const int q1 = amax == 0.0f ? 0 : __float2int_rn(v[ni][2 + j] / d);
                int sum = q0 + q1;
                for (int o = 4; o < 32; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
                scale[ni][j] = d;
                if (HEAD && gid == 0) sums[warp][at] = __fmul_rn(d, (float)sum);
                if (orow < rows && t < tokens) {
                    signed char* out = oq + (size_t)t * rows + orow + gid;
                    out[0] = (signed char)q0;
                    out[8] = (signed char)q1;
                    if (!HEAD && gid == 0) {
                        ob[(size_t)t * (rows / 16) + orow / 16] = d * (float)sum;
                        if ((warp & 1) == 0) oa[(size_t)t * (rows / 32) + orow / 32] = d;
                    }
                }
            }
        if constexpr (HEAD) __syncthreads();
        // A head: a 32's scale, and its sum from the two warps' 16s, by its first warp.
        if (HEAD && gid == 0 && (warp & 1) == 0 && orow < rows) {
#pragma unroll
            for (int ni = 0; ni < NI; ni++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int at = ni * 8 + tig * 2 + j, t = tok0 + wt + at;
                    if (t >= tokens) continue;
                    float* head = oa + ((size_t)t * o_heads + orow / 128) * X8_HEAD;
                    const int g = (orow % 128) / 32;
                    head[g] = scale[ni][j];
                    head[4 + g] = __fadd_rn(sums[warp][at], sums[warp ^ 1][at]);
                }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int ni = 0; ni < NI; ni++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int r = row0 + wr + mi * 16 + gid + (i >> 1) * 8, t = tok0 + wt + ni * 8 + tig * 2 + (i & 1);
                    if (r < rows && t < tokens) y[(size_t)t * rows + r] = acc[mi][ni][i];
                }
    }
}

#define MATMUL8(name, KIND, GROUP, NT, ROW_BYTES, BLOCKS, HEAD)                                            \
    extern "C" __global__ void __launch_bounds__(256, BLOCKS)                                            \
        name(const u8* w, const signed char* xq, const float* xa, const float* xb, float* y, int rows,   \
             int cols, int tokens) {                                                                     \
        matmul8<KIND, GROUP, NT, 0, HEAD>(w, nullptr, (size_t)(ROW_BYTES), xq, xa, xb, y, nullptr,       \
                                          nullptr, nullptr, rows, cols, tokens);                         \
    }
// A feed-forward's gate and up products with its activation, rounded for the next product
// (matmul8's GLU): `rows` of each matrix, a block an SM.
#define MATMUL8_GLU(name, KIND, GROUP, NT, ROW_BYTES, GLU, HEAD)                                           \
    extern "C" __global__ void __launch_bounds__(256, 1)                                                 \
        name(const u8* gate, const u8* up, const signed char* xq, const float* xa, const float* xb,      \
             signed char* oq, float* oa, float* ob, int rows, int cols, int tokens) {                    \
        matmul8<KIND, GROUP, NT, GLU, HEAD>(gate, up, (size_t)(ROW_BYTES), xq, xa, xb, nullptr, oq, oa,  \
                                            ob, rows, cols, tokens);                                     \
    }

// One block per SM, with the next step's bytes read ahead into registers (fetch): faster
// than two blocks without (Q4_0 -6%, Q8_0 -25%), and Q4_K and Q5_K would spill with two.
// Q6_K keeps two blocks and no read-ahead.
// Tiles of 32 and 16 tokens are for batches of up to that many (two blocks an SM, which
// their fewer registers allow, measured slower for Q4_0, 74 -> 86 us, and level for the
// rest); tiles of 128 run one block an SM, Q6_K's as well (the sums take twice the
// registers).
//
// A type's kernels at every tile, plain and gated. The K-quants read a batch's heads
// (HEAD); Q4_0 and Q8_0 come both ways, the heads' kernels named `_h`, for what little a
// K-quant model keeps in them (Qwen's ssm_alpha and ssm_beta).
#define MATMUL8_ALL(q, sfx, KIND, GROUP, ROW_BYTES, B64, HEAD)                                  \
    MATMUL8(matmul8_##q##sfx, KIND, GROUP, 64, ROW_BYTES, B64, HEAD)                            \
    MATMUL8(matmul8_t32_##q##sfx, KIND, GROUP, 32, ROW_BYTES, B64, HEAD)                        \
    MATMUL8(matmul8_t16_##q##sfx, KIND, GROUP, 16, ROW_BYTES, B64, HEAD)                        \
    MATMUL8(matmul8_t128_##q##sfx, KIND, GROUP, 128, ROW_BYTES, 1, HEAD)                        \
    MATMUL8_GLU(matmul8_swiglu_t64_##q##sfx, KIND, GROUP, 64, ROW_BYTES, 1, HEAD)               \
    MATMUL8_GLU(matmul8_swiglu_t128_##q##sfx, KIND, GROUP, 128, ROW_BYTES, 1, HEAD)             \
    MATMUL8_GLU(matmul8_geglu_t64_##q##sfx, KIND, GROUP, 64, ROW_BYTES, 2, HEAD)                \
    MATMUL8_GLU(matmul8_geglu_t128_##q##sfx, KIND, GROUP, 128, ROW_BYTES, 2, HEAD)
MATMUL8_ALL(q4_0, , K_Q4_0, 32, q4_0_row_bytes(cols), 1, false)
MATMUL8_ALL(q8_0, , K_Q8_0, 32, q8_0_row_bytes(cols), 1, false)
MATMUL8_ALL(q4_0, _h, K_Q4_0, 32, q4_0_row_bytes(cols), 1, true)
MATMUL8_ALL(q8_0, _h, K_Q8_0, 32, q8_0_row_bytes(cols), 1, true)
MATMUL8_ALL(q4_k, , K_Q4_K, 32, cols / 256 * 144, 1, true)
MATMUL8_ALL(q5_k, , K_Q5_K, 32, cols / 256 * 176, 1, true)
MATMUL8_ALL(q6_k, , K_Q6_K, 16, q6_k_row_bytes(cols), 2, true)

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

// Four 8 x 8 f16 matrices as they lie (rows from lanes 0-7, 8-15, 16-23, 24-31): each
// thread gets the B fragments of mma.m16n8k16 for two tiles (r0, r1 and r2, r3) when a
// matrix's rows are positions and its columns a head's numbers.
__device__ __forceinline__ void ldmatrix_x4(u32 (&r)[4], const __half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(shared_address(p)));
}

// A batch's queries rounded to f16 where they lie, a row's halves at the start of its
// floats: a block a row (a token's head, up to 512 numbers), each thread reading its four
// numbers before any is written. Each 32 numbers are laid out as attention_mma's threads
// take them, so one 16-byte read gives a thread its part of two steps of the first
// product: the thread of a group of four that takes numbers 2t, 2t + 1 of each 8 finds
// those of the four 8s side by side, at 8t.
extern "C" __global__ void q_rows_to_half(float* q, int d) {
    float* row = q + (size_t)blockIdx.x * d;
    const int i = threadIdx.x;
    const bool mine = i * 4 < d;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (mine) v = f4(row + i * 4);
    __syncthreads();
    if (mine) {
        // Numbers 4i to 4i + 3: of the 32 they are in, the 8 numbered (i / 2) % 4, its
        // first or second half.
        const int g = i >> 3, b = (i >> 1) & 3, t = (i & 1) * 2;
        __half2* dst = reinterpret_cast<__half2*>(row) + g * 16 + b;
        dst[t * 4] = __floats2half2_rn(v.x, v.y);
        dst[(t + 1) * 4] = __floats2half2_rn(v.z, v.w);
    }
}

//
// PART, for a batch of a few dozen tokens deep in a chat (a tool's result, a new turn):
// it has too few blocks to fill the GPU, each reading the whole context alone. So a block
// takes ROWS sets of 16 tokens of query heads that share a key/value head (not 16 × ROWS
// tokens of one head, most of them past the batch's end), and the context is split into
// gridDim.y parts, a block each: block (x, part) leaves its rows' sums as attention_part
// does (the largest score, the sum, the weighted values), for attention_combine. No
// sliding windows (they are short already). `all_heads` is the query heads.
template <int D, int ROWS, int SPLIT, int KC, bool PART>
__device__ void attention_mma(const float* q, const __half* keys, const __half* values, float* out,
                              const int* pos, int window, int all_heads, int kv_dim, int group, float scale,
                              int tokens) {
    const int ring = window >> 16;  // see ring_row
    window &= 0xFFFF;
    constexpr int WARPS = ROWS * SPLIT, THREADS = 32 * WARPS, QT = 16 * ROWS;
    constexpr int DS = D + 8;            // padded rows: no bank conflicts
    constexpr int DP = D / SPLIT;        // the numbers of a head each warp takes
    constexpr int NS = KC / 8 * 4;       // a thread's share of S
    static_assert(DP % 32 == 0 && KC % 16 == 0, "tile sizes");
    extern __shared__ __align__(16) unsigned char smem[];
    __half* ks = reinterpret_cast<__half*>(smem);  // KC × DS
    __half* vs = ks + KC * DS;                      // KC × DS
    float* sx = reinterpret_cast<float*>(vs + KC * DS);  // SPLIT > 1: [ROWS][SPLIT][NS][32]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int rw = warp / SPLIT, sp = warp % SPLIT, dp = sp * DP;
    const int p0 = *pos;
    // The warp's query head and its first token less 16 rw (so row r of the block is token
    // t0 + r), the positions the block reads (lo to end) and the last any of its rows sees.
    int head, heads, t0, kv, last_row, lo, end;
    bool warp_live = true;
    if constexpr (PART) {
        const int tiles = (tokens + 15) / 16, units = group * tiles;
        const int per_kv = (units + ROWS - 1) / ROWS;
        const int kvh = blockIdx.x / per_kv, unit = (blockIdx.x % per_kv) * ROWS + rw;
        warp_live = unit < units;
        heads = all_heads;
        head = kvh * group + (warp_live ? unit / tiles : 0);
        t0 = (warp_live ? unit % tiles : 0) * 16 - rw * 16;
        kv = kvh * D;
        last_row = p0 + tokens - 1;
        const int chunks = last_row / KC + 1, share = (chunks + (int)gridDim.y - 1) / (int)gridDim.y;
        lo = (int)blockIdx.y * share * KC;
        end = min(last_row, lo + share * KC - 1);
    } else {
        head = blockIdx.x, heads = gridDim.x;
        t0 = (gridDim.y - 1 - blockIdx.y) * QT;  // the longest rows first
        kv = (head / group) * D;
        last_row = p0 + min(t0 + QT, tokens) - 1;
        lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;
        end = last_row;
    }

    auto load = [&](__half* dst, const __half* src, int start) {
        for (int i = tid; i < KC * D / 8; i += THREADS) {
            const int r = i / (D / 8), d = (i % (D / 8)) * 8;
            const bool valid = start + r <= last_row;
            cp_async16(dst + r * DS + d, src + ring_row(valid ? start + r : 0, ring) * kv_dim + kv + d, valid);
        }
        cp_async_commit();
    };
    load(ks, keys, lo);

    // This thread's two rows (gid and gid + 8 of the warp's 16) and where they look.
    // Their queries are read where they lie, in f16 (q_rows_to_half): out of shared
    // memory, a block's is its two chunks only, and an SM holds more blocks. (A row past
    // the batch's end reads the last token's; its scores are never used.)
    const int row0 = rw * 16 + gid;
    const __half* q0 =
        reinterpret_cast<const __half*>(q + ((size_t)min(t0 + row0, tokens - 1) * heads + head) * D);
    const __half* q1 =
        reinterpret_cast<const __half*>(q + ((size_t)min(t0 + row0 + 8, tokens - 1) * heads + head) * D);
    int at[2], first[2];
    bool live[2];
    for (int h = 0; h < 2; h++) {
        const int r = row0 + h * 8;
        live[h] = warp_live && t0 + r < tokens;
        at[h] = p0 + t0 + r;
        first[h] = window > 0 ? max(0, at[h] + 1 - window) : 0;
    }
    float o[DP / 8][4];
#pragma unroll
    for (int n = 0; n < DP / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    // The queries of the next two steps of S = Q Kᵀ (32 of a head's numbers each), read
    // ahead: see the loop.
    constexpr int AHEAD = 2;
    uint4 qna[AHEAD], qnb[AHEAD];
#pragma unroll
    for (int a = 0; a < AHEAD; a++) {
        qna[a] = *reinterpret_cast<const uint4*>(q0 + dp + a * 32 + tig * 8);
        qnb[a] = *reinterpret_cast<const uint4*>(q1 + dp + a * 32 + tig * 8);
    }
    for (int start = lo; start <= end; start += KC) {
        load(vs, values, start);  // the last chunk's P V is done with vs (the sync below)
        cp_async_wait1();         // this chunk's keys are in
        __syncthreads();

        // S = Q Kᵀ for the warp's 16 rows × KC positions (over its part of the head).
        float s[KC / 8][4];
#pragma unroll
        for (int n = 0; n < KC / 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll 2
        for (int kk = dp; kk < dp + DP; kk += 32) {
            // Two steps' queries come in one read each (q_rows_to_half laid them out so),
            // and are asked for AHEAD steps before their products (after the last steps,
            // the first ones', for the next chunk). A block's queries are larger than
            // what is left of the L1 cache beside its shared memory, so most reads go to
            // L2, and a warp that waited for each lost a fifth of its time (Nsight
            // Compute: 32% of the reads found in L1; a ~30K-token batch 28.0 -> 24.5 ms
            // 512 wide and 21.1 -> 18.8 ms 256 wide, one step ahead about half of that).
            const uint4 qa = qna[0], qb = qnb[0];
            {
#pragma unroll
                for (int a = 0; a + 1 < AHEAD; a++) qna[a] = qna[a + 1], qnb[a] = qnb[a + 1];
                int next = kk + 32 * AHEAD;
                if (next >= dp + DP) next -= DP;
                qna[AHEAD - 1] = *reinterpret_cast<const uint4*>(q0 + next + tig * 8);
                qnb[AHEAD - 1] = *reinterpret_cast<const uint4*>(q1 + next + tig * 8);
            }
#pragma unroll
            for (int n = 0; n < KC / 8; n += 2) {
                u32 b[4];
                ldmatrix_x4(b, ks + ((n + (lane >> 4)) * 8 + (lane & 7)) * DS + kk + ((lane >> 3) & 1) * 8);
                mma_f16(s[n], qa.x, qb.x, qa.y, qb.y, b[0], b[1]);
                mma_f16(s[n + 1], qa.x, qb.x, qa.y, qb.y, b[2], b[3]);
            }
#pragma unroll
            for (int n = 0; n < KC / 8; n += 2) {
                u32 b[4];
                ldmatrix_x4(b, ks + ((n + (lane >> 4)) * 8 + (lane & 7)) * DS + kk + 16 + ((lane >> 3) & 1) * 8);
                mma_f16(s[n], qa.z, qb.z, qa.w, qb.w, b[0], b[1]);
                mma_f16(s[n + 1], qa.z, qb.z, qa.w, qb.w, b[2], b[3]);
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
        if (start + KC <= end) {
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

    if constexpr (PART) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (!live[h]) continue;
            float* dst = out + (((size_t)(t0 + row0 + h * 8) * heads + head) * gridDim.y + blockIdx.y) * (2 + D);
            if (tig == 0 && sp == 0) {
                dst[0] = m[h];  // -inf for a part the row sees nothing of
                dst[1] = l[h];
            }
#pragma unroll
            for (int n = 0; n < DP / 8; n++) {
                dst[2 + dp + n * 8 + tig * 2] = o[n][2 * h];
                dst[2 + dp + n * 8 + tig * 2 + 1] = o[n][2 * h + 1];
            }
        }
        return;
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

#define ATTENTION_MMA_AS(name, D, ROWS, SPLIT, KC, PART)                                                     \
    extern "C" __global__ void __launch_bounds__(32 * ROWS * SPLIT) name(                                    \
        const float* q, const __half* keys, const __half* values, float* out, const int* pos, int window,    \
        int hd, int kv_dim, int group, float scale, int tokens) {                                            \
        attention_mma<D, ROWS, SPLIT, KC, PART>(q, keys, values, out, pos, window, hd, kv_dim, group, scale, \
                                                tokens);                                                     \
    }
#define ATTENTION_MMA(name, D, ROWS, SPLIT, KC) ATTENTION_MMA_AS(name, D, ROWS, SPLIT, KC, false)

// Head sizes 64 to 512: (rows of 16 tokens, warps per row, positions per chunk), the
// fastest of the shapes measured on an RTX 5060 Laptop. The host side's ATTENTION_MMA
// tables must match. With the queries out of shared memory a block of twice the rows
// and positions fits, and reads a long prompt faster (a ~30K-token prompt: Gemma 4 E2B
// 3.70 -> 3.25 s, E4B 7.85 -> 7.24, Qwen 3.5 4B 11.30 -> 10.70; 512-wide 4 x 2 x 32 best
// of 4x2x16, 2x2x32, 6x2x32, 8x2x16, 8x2x32, 4x4x32 and 8x1x32, 256-wide 8 x 1 x 64 of
// 8x1x32, 4x1x64, 12x1x64, 16x1x32 and 16x1x64). A batch of up to 64 tokens keeps the
// smaller shapes (`_short`): it has too few of the larger blocks to fill the GPU (Qwen
// 3.5 4B, 16 tokens 27K deep: 41.5 ms against 47.9).
ATTENTION_MMA(attention_mma_64, 64, 4, 1, 32)
ATTENTION_MMA(attention_mma_128, 128, 4, 1, 32)
ATTENTION_MMA(attention_mma_256, 256, 8, 1, 64)
ATTENTION_MMA(attention_mma_512, 512, 4, 2, 32)
ATTENTION_MMA(attention_mma_256_short, 256, 4, 1, 32)
ATTENTION_MMA(attention_mma_512_short, 512, 2, 2, 16)
// The short shapes with the context in parts (PART above; `hd` is the query heads there).
ATTENTION_MMA_AS(attention_mma_64_part, 64, 4, 1, 32, true)
ATTENTION_MMA_AS(attention_mma_128_part, 128, 4, 1, 32, true)
ATTENTION_MMA_AS(attention_mma_256_part, 256, 4, 1, 32, true)
ATTENTION_MMA_AS(attention_mma_512_part, 512, 2, 2, 16, true)
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
                                    const int* pos, float eps, int half_cache, int tokens, int ring) {
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
            const size_t at_cache = ring_row(at, ring) * kv_dim + h * hd;
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
    const int ring = window >> 16;  // see ring_row
    window &= 0xFFFF;
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
            const KV* k = keys + ring_row(at, ring) * kv_dim + kv;
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
            const KV* v = values + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + ring_row(start + i, ring) * kv_dim);
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
    const int ring = window >> 16;  // see ring_row
    window &= 0xFFFF;
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
            const KV* k = keys + ring_row(at, ring) * kv_dim + kv;
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
            const KV* v = values + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + ring_row(start + i, ring) * kv_dim);
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
// A few tokens at once (a draft's guesses checked in one pass) take a grid layer each:
// token z of the batch is at position *pos + z, its queries row z of q (`heads` heads),
// its partial rows after those of the tokens before it.
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
                                  const int* pos, int window, int kv_dim, int group, float scale, int heads) {
    const int ring = window >> 16;  // see ring_row
    window &= 0xFFFF;
    constexpr int PER = HD / 32, G = 4, WARPS = 4;
    __shared__ float wm[WARPS][G], wl[WARPS][G];
    __shared__ __align__(16) float wacc[WARPS][G][HD];
    const int sets = (group + G - 1) / G;
    const int kvh = blockIdx.x / sets, set = blockIdx.x % sets;
    const int head0 = kvh * group + set * G, ng = min(G, group - set * G);
    const int part = blockIdx.y, parts = gridDim.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int token = blockIdx.z;
    const int len = *pos + 1 + token;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    q += (size_t)token * heads * HD;
    partial += (size_t)token * heads * parts * (2 + HD);

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
        for (int u = 0; u < U; u++) halves_to_float(keys + ring_row(at + u * WARPS, ring) * kv_dim + off, k[u]);
#pragma unroll
        for (int u = 0; u < U; u++) halves_to_float(values + ring_row(at + u * WARPS, ring) * kv_dim + off, v[u]);
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
        halves_to_float(keys + ring_row(at, ring) * kv_dim + off, k);
        halves_to_float(values + ring_row(at, ring) * kv_dim + off, v);
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
        int window, int kv_dim, int group, float scale, int heads) {                                        \
        attention_part_kv<HD>(q, keys, values, partial, pos, window, kv_dim, group, scale, heads);         \
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
    const int ring = window >> 16;  // see ring_row
    window &= 0xFFFF;
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
                kv4 = f4(keys + ring_row(start + r, ring) * kv_dim + kv + d);
                vv4 = f4(values + ring_row(start + r, ring) * kv_dim + kv + d);
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
// many tokens there are. With `steps` (not null) it's also written there as it was before
// the batch and after each token but the last, a whole state each (step m: the state
// after the batch's first m tokens), so a draft's rejected guesses can be undone.
extern "C" __global__ void __launch_bounds__(512) delta_net(const float* conv, int channels, int key_dim,
                                     const float* beta_raw, const float* alpha_raw, const float* dt_bias,
                                     const float* a, float* state, const float* z, const float* norm,
                                     float* out, int n, int k_heads, float eps, int tokens, float* steps) {
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
    const size_t step_len = (size_t)v_heads * n * n;
    for (int t = 0; t < tokens; t++) {
        if (steps) {
            float* kept = steps + (size_t)t * step_len + ((size_t)h * n + j) * n;
#pragma unroll
            for (int r = 0; r < 32; r++)
                if (r < per) kept[part + 4 * r] = st[r];
        }
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

// An f32 as two bf16 numbers in one word, for the tensor cores (delta_chunk_scan_mma): the
// nearest bf16 in the low half and, in the high half, the nearest bf16 to what that left
// over. A product of two such numbers as hi x hi + hi x lo + lo x hi is the f32 product
// to about 2^-16 of itself (only lo x lo is dropped), where one bf16 each would give 2^-8.
__device__ __forceinline__ u32 bf16_bits(float x) {
    const u32 b = __float_as_uint(x);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;  // to nearest, ties to even
}
__device__ __forceinline__ u32 split_bf16(float x) {
    const u32 hi = bf16_bits(x);
    return hi | (bf16_bits(x - __uint_as_float(hi << 16)) << 16);
}

// One block of 256 threads per value head (blockIdx.x) and chunk (blockIdx.y); n ≤ 128.
// Dynamic shared memory (delta_chunk_prep_bytes on the host): the chunk's k, which A
// takes the place of once k·k is worked out, then the cumulative log decays and β. q is
// read from GPU memory (through L1): with it in shared memory too, only one block fit an
// SM.
extern "C" __global__ void __launch_bounds__(256)
    delta_chunk_prep(const float* conv, int channels, int key_dim, const float* beta_raw, const float* alpha_raw,
                     const float* dt_bias, const float* a, float* scratch, int n, int k_heads, int tokens,
                     int split) {
    // With `split`, for delta_chunk_scan_mma: W and P are written as split_bf16 words in
    // the same places and U stays f32. Q̂ and K̂ are not written: they are q and k scaled a
    // row, which that scan reads where they are, and Q̂'s place holds the scales (γ_i, then
    // γ_DC / γ_i, a number a row). The scratch is what both kernels wait on, written here
    // and read back there, and those two were 64 of its 147 KB a chunk and head.
    extern __shared__ __align__(16) float sm[];
    u32* const scratch_bits = reinterpret_cast<u32*>(scratch);
    auto put = [&](float* at, float x) {
        if (split)
            scratch_bits[at - scratch] = split_bf16(x);
        else
            *at = x;
    };
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
    if (split) {
        if (tid < DC) {
            Qh[tid] = expf(G[tid]);
            Qh[DC + tid] = expf(g_last - G[tid]);
        }
    } else {
        for (int idx = tid; idx < DC * n; idx += blockDim.x) {
            const int i = idx / n, d = idx % n;
            Qh[idx] = i < len ? expf(G[i]) * q_row(i)[d] : 0.0f;
            Kh[idx] = i < len ? expf(g_last - G[i]) * ks[i * NS + d] : 0.0f;
        }
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
            put(&P[i * DC + j], j <= i && i < len ? decay * qk[x][y] : 0.0f);
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
            if (is_w)
                put(&X[i * n + d], b);
            else
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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// ---- The chunks' scan on the tensor cores (RTX 30 series on) ---------------------------
//
// delta_chunk_scan's products as f32 sums of bf16 products (mma.m16n8k16), each number in
// two bf16 parts (split_bf16) and each product in three, as llama.cpp's
// gated-delta-net-mma does: the state stays f32, in registers, and what is multiplied is
// right to about 2^-16. After delta_chunk_prep with `split`. n = 128.
//
// A block of 4 warps per value head (blockIdx.x) and 32 columns of its state (blockIdx.y):
// the columns are independent. Warp w keeps the state's rows 16 w.. and 64 + 16 w.. (two
// tiles of 16 rows by 4 of 8 columns, laid out as the products' results are), and takes
// rows 16 w.. of the chunk's δ and O. A chunk: the state into shared memory in two parts,
// D1 = W S0 and D2 = Q̂ S0, δ = U − D1 into shared memory, then O = D2 + P δ and
// S = γ S0 + K̂ᵀ δ. W and P are read from the scratch as the products take them; Q̂ and K̂ᵀ
// are made here from q and k (normalized by delta_l2) and the scratch's scale a row.
#define DM_COLS 32
#define DM_S (128 + 8)  // the state's columns in shared memory, padded
#define DM_D (DC + 8)   // δ's

__device__ __forceinline__ void mma_bf16(float (&c)[4], const u32 (&a)[4], const u32 (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two neighbouring split words as a register of their first parts and one of their second.
__device__ __forceinline__ void split_pair(const u32* p, u32& hi, u32& lo) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    hi = (w.x & 0xFFFFu) | (w.y << 16);
    lo = (w.x >> 16) | (w.y & 0xFFFF0000u);
}

// c += a b with both in two parts: the three products that matter.
__device__ __forceinline__ void mma_split(float (&c)[4], const u32 (&ah)[4], const u32 (&al)[4],
                                          const u32 (&bh)[2], const u32 (&bl)[2]) {
    mma_bf16(c, al, bh);
    mma_bf16(c, ah, bl);
    mma_bf16(c, ah, bh);
}

extern "C" __global__ void __launch_bounds__(128)
    delta_chunk_scan_mma(const float* scratch, float* state, float* out, int n, int tokens, const float* conv,
                         int channels, int key_dim, int k_heads) {
    __shared__ __align__(16) u32 Sb[DM_COLS][DM_S];  // S0[r][c] split, a column's rows together
    __shared__ __align__(16) u32 Db[DM_COLS][DM_D];  // δ[m][c] split, likewise
    const int h = blockIdx.x, v_heads = gridDim.x, j0 = blockIdx.y * DM_COLS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    // A result's 4 numbers a thread: rows gid and gid + 8 of its 16, columns 2 tig and 2 tig + 1 of its 8.
    auto row_of = [&](int e) { return gid + (e >> 1) * 8; };
    auto col_of = [&](int ct, int e) { return ct * 8 + tig * 2 + (e & 1); };
    // A's registers for rows gid and gid + 8 of 16 from `rows` (rows of `stride` split
    // words), columns k.. of 16; B's for 8 columns from ct of rows k.. of `from`.
    auto load_a = [&](u32 (&ah)[4], u32 (&al)[4], const u32* rows, int stride, int k) {
        const u32* r0 = rows + (size_t)gid * stride + k + tig * 2;
        const u32* r1 = r0 + (size_t)8 * stride;
        split_pair(r0, ah[0], al[0]);
        split_pair(r1, ah[1], al[1]);
        split_pair(r0 + 8, ah[2], al[2]);
        split_pair(r1 + 8, ah[3], al[3]);
    };

    float s[2][4][4];
#pragma unroll
    for (int x = 0; x < 2; x++)
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                s[x][ct][e] = state[((size_t)h * n + j0 + col_of(ct, e)) * n + (warp + 4 * x) * 16 + row_of(e)];

    const int chunks = (tokens + DC - 1) / DC;
    for (int ch = 0; ch < chunks; ch++) {
        const float* base = scratch + ((size_t)ch * v_heads + h) * delta_chunk_stride(n);
        const u32* W = reinterpret_cast<const u32*>(base);
        const float* U = base + DC * n;
        const float* up = U + DC * n;  // γ_i
        const float* down = up + DC;   // γ_DC / γ_i
        const u32* P = reinterpret_cast<const u32*>(base + (size_t)4 * DC * n);
        const float gamma = base[(size_t)4 * DC * n + DC * DC];
        // The chunk's q and k, a row a token, and how many tokens it has.
        const int len = min(DC, tokens - ch * DC);
        const float* q0 = conv + (size_t)ch * DC * channels + (h % k_heads) * n;
        const float* k0 = q0 + key_dim;
        // Rows 16 warp + gid and + 8 of Q̂ from column k: q scaled and split.
        auto load_q = [&](u32 (&ah)[4], u32 (&al)[4], int k) {
#pragma unroll
            for (int x = 0; x < 4; x++) {
                const int i = warp * 16 + gid + (x & 1) * 8;
                float2 q = make_float2(0.0f, 0.0f);
                if (i < len) {
                    q = *reinterpret_cast<const float2*>(q0 + (size_t)i * channels + k + (x >> 1) * 8 + tig * 2);
                    const float g = up[i];
                    q.x *= g, q.y *= g;
                }
                const u32 w0 = split_bf16(q.x), w1 = split_bf16(q.y);
                ah[x] = (w0 & 0xFFFFu) | (w1 << 16);
                al[x] = (w0 >> 16) | (w1 & 0xFFFF0000u);
            }
        };
        // Rows r0 + gid and + 8 of K̂ᵀ from column m: tokens m and m + 1's k at a row.
        auto load_kt = [&](u32 (&ah)[4], u32 (&al)[4], int r0, int k) {
#pragma unroll
            for (int x = 0; x < 4; x++) {
                const int r = r0 + gid + (x & 1) * 8, m = k + (x >> 1) * 8 + tig * 2;
                float e0 = 0.0f, e1 = 0.0f;
                if (m < len) e0 = down[m] * k0[(size_t)m * channels + r];
                if (m + 1 < len) e1 = down[m + 1] * k0[(size_t)(m + 1) * channels + r];
                const u32 w0 = split_bf16(e0), w1 = split_bf16(e1);
                ah[x] = (w0 & 0xFFFFu) | (w1 << 16);
                al[x] = (w0 >> 16) | (w1 & 0xFFFF0000u);
            }
        };

#pragma unroll
        for (int x = 0; x < 2; x++)
#pragma unroll
            for (int ct = 0; ct < 4; ct++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    Sb[col_of(ct, e)][(warp + 4 * x) * 16 + row_of(e)] = split_bf16(s[x][ct][e]);
        __syncthreads();

        // D1 = W S0 (acc[0]) and D2 = Q̂ S0 (acc[1]) for rows 16 warp.. of the chunk.
        float acc[2][4][4] = {};
#pragma unroll 2
        for (int k = 0; k < 128; k += 16) {
            u32 ah[2][4], al[2][4], bh[4][2], bl[4][2];
            load_a(ah[0], al[0], W + (size_t)warp * 16 * n, n, k);
            load_q(ah[1], al[1], k);
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                const u32* b = &Sb[ct * 8 + gid][k + tig * 2];
                split_pair(b, bh[ct][0], bl[ct][0]);
                split_pair(b + 8, bh[ct][1], bl[ct][1]);
            }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                mma_split(acc[0][ct], ah[0], al[0], bh[ct], bl[ct]);
                mma_split(acc[1][ct], ah[1], al[1], bh[ct], bl[ct]);
            }
        }
        // δ = U − D1
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int i = warp * 16 + row_of(e), c = col_of(ct, e);
                Db[c][i] = split_bf16(U[(size_t)i * n + j0 + c] - acc[0][ct][e]);
            }
        __syncthreads();

        // O = D2 + P δ and S = γ S0 + K̂ᵀ δ, a step's δ read once for both.
#pragma unroll
        for (int x = 0; x < 2; x++)
#pragma unroll
            for (int ct = 0; ct < 4; ct++)
#pragma unroll
                for (int e = 0; e < 4; e++) s[x][ct][e] *= gamma;
#pragma unroll 2
        for (int k = 0; k < DC; k += 16) {
            u32 ph[4], pl[4], kh[2][4], kl[2][4], bh[4][2], bl[4][2];
            load_a(ph, pl, P + (size_t)warp * 16 * DC, DC, k);
            load_kt(kh[0], kl[0], warp * 16, k);
            load_kt(kh[1], kl[1], (warp + 4) * 16, k);
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                const u32* b = &Db[ct * 8 + gid][k + tig * 2];
                split_pair(b, bh[ct][0], bl[ct][0]);
                split_pair(b + 8, bh[ct][1], bl[ct][1]);
            }
#pragma unroll
            for (int ct = 0; ct < 4; ct++) {
                mma_split(acc[1][ct], ph, pl, bh[ct], bl[ct]);
                mma_split(s[0][ct], kh[0], kl[0], bh[ct], bl[ct]);
                mma_split(s[1][ct], kh[1], kl[1], bh[ct], bl[ct]);
            }
        }
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int t = ch * DC + warp * 16 + row_of(e);
                if (t < tokens) out[((size_t)t * v_heads + h) * n + j0 + col_of(ct, e)] = acc[1][ct][e] * inv_sqrt_n;
            }
        // The next chunk's state goes into Sb, which nothing reads now; its sync comes
        // before δ is written again.
    }
#pragma unroll
    for (int x = 0; x < 2; x++)
#pragma unroll
        for (int ct = 0; ct < 4; ct++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                state[((size_t)h * n + j0 + col_of(ct, e)) * n + (warp + 4 * x) * 16 + row_of(e)] = s[x][ct][e];
}
#endif
