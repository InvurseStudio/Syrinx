// Vireo's Metal kernels: the Mac's port of kernels/syrinx.cu, everything Llama, Qwen 3.5 and
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

// Four bytes from a 4-byte-aligned address: Q6_K's numbers, its blocks padded to 212 bytes
// on the GPU (Q4_0's and Q8_0's blocks keep GGUF's odd sizes, so they use u32_at2).
static inline u32 u32_at4(device const u8* p) { return *reinterpret_cast<device const u32*>(p); }

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

// block_sum with one barrier, for a `scratch` (32 floats) no other call uses in the
// kernel: every SIMD group adds up the partial sums itself, in the order block_sum's
// first group does, so the result is the same to the bit.
static inline float block_sum_once(float v, threadgroup float* scratch, uint tid, uint threads) {
    const uint lane = tid & 31, warp = tid >> 5;
    v = simd_sum(v);
    if (lane == 0) scratch[warp] = v;
    BARRIER;
    return simd_sum(lane < (threads + 31) / 32 ? scratch[lane] : 0.0f);
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
// padded to 16 bytes (as in syrinx.cu; Gpu::upload_weights repacks GGUF's blocks so).
static inline ulong q4_0_row_bytes(int cols) {
    const int blocks = cols / 32;
    return (ulong)blocks * 16 + ((blocks * 2 + 15) & ~15);
}

static inline float q4_0_scale(device const u8* row, int blocks, int b) {
    return half_at(row + (ulong)blocks * 16 + b * 2);
}

// Q8_0 on the GPU: each row's 32-byte runs of int8 numbers first (16-byte aligned), then
// its f16 scales, padded to 16 bytes (Gpu::upload_weights repacks GGUF's 34-byte blocks).
static inline ulong q8_0_row_bytes(int cols) {
    const int blocks = cols / 32;
    return (ulong)blocks * 32 + ((blocks * 2 + 15) & ~15);
}

static inline float q8_0_scale(device const u8* row, int blocks, int b) {
    return half_at(row + (ulong)blocks * 32 + b * 2);
}

static inline float q8_0_dot8(uint2 q, float4 x0, float4 x1) {
    return dot(float4(as_type<char4>(q.x)), x0) + dot(float4(as_type<char4>(q.y)), x1);
}

// Q6_K's 210-byte blocks padded to 212 on the GPU: whole 4-byte words, and no more of
// them than that (one token's pass reads every weight once, and is bound by it).
#define Q6K_BYTES 212

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

// Split-K F32 matvec (4 SIMD groups = 128 threads per row): saturates the GPU for
// small-row F32 matrices like the 128-expert router (128 x 2304).
kernel void matvec_split_f32(ARGS, KP) {
    device const float* w = ARG_PTR(const float, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int cols = ARG_INT(4);
    const int row = (int)gid.x;
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    device const float* r = w + (ulong)row * cols;
    float sum = 0.0f;
    for (int c = (sgi * 32 + lane) * 4; c < cols; c += 512) {
        sum += dot(f4(r + c), f4(x + c));
    }
    sum = simd_sum(sum);
    threadgroup float parts[4];
    if (lane == 0) parts[sgi] = sum;
    BARRIER;
    if (tid == 0) y[row] = parts[0] + parts[1] + parts[2] + parts[3];
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
    const ulong row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 2, j = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (ulong)(row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + (ulong)b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * q8_0_dot8(q, x0, x1);
        }
    }
    MATVEC_STORE
}

constant float KVALUES_IQ4NL_F[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

static inline float iq4nl_dot8(uint2 q, int shift, float4 a, float4 b) {
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu;
    const u32 hi = (q.y >> shift) & 0x0F0F0F0Fu;
    const float4 va(KVALUES_IQ4NL_F[lo & 0xF], KVALUES_IQ4NL_F[(lo >> 8) & 0xF],
                    KVALUES_IQ4NL_F[(lo >> 16) & 0xF], KVALUES_IQ4NL_F[lo >> 24]);
    const float4 vb(KVALUES_IQ4NL_F[hi & 0xF], KVALUES_IQ4NL_F[(hi >> 8) & 0xF],
                    KVALUES_IQ4NL_F[(hi >> 16) & 0xF], KVALUES_IQ4NL_F[hi >> 24]);
    return dot(va, a) + dot(vb, b);
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

kernel void matvec_iq4_nl(ARGS, KP) {
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
            sum[r] += q4_0_scale(row, blocks, b) * (iq4nl_dot8(q, 0, lo0, lo1) + iq4nl_dot8(q, 4, hi0, hi1));
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
            const u32 ql0 = u32_at4(block + hf * 64 + l), ql1 = u32_at4(block + hf * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + hf * 32 + l);
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

// The quantized matrix-vector products with each pair of rows split over the
// threadgroup's MV_SPLIT SIMD groups: each takes every MV_SPLIT-th step of the row's
// blocks, and their sums meet in threadgroup memory. The arithmetic is the kernels' above;
// a row then takes a quarter of the sequential steps, which is what limits a small
// matrix (one token's time is mostly the latency of those steps, not the bandwidth).
#define MV_SPLIT 4
#define MATVEC_ROWS_SPLIT                                   \
    threadgroup float mv_parts[MV_SPLIT][MV_ROWS];          \
    const int lane = tid & 31, sgi = tid >> 5;              \
    const int row0 = gid.x * MV_ROWS;                       \
    if (row0 >= rows) return;                               \
    const int nrows = min(MV_ROWS, rows - row0);            \
    float sum[MV_ROWS];                                     \
    for (int r = 0; r < MV_ROWS; r++) sum[r] = 0.0f;

#define MATVEC_STORE_SPLIT                                  \
    for (int r = 0; r < MV_ROWS; r++) {                     \
        const float part = simd_sum(sum[r]);                \
        if (lane == 0) mv_parts[sgi][r] = part;             \
    }                                                       \
    BARRIER;                                                \
    if ((int)tid < nrows) {                                 \
        float total = 0.0f;                                 \
        for (int g = 0; g < MV_SPLIT; g++) total += mv_parts[g][tid]; \
        if (EPI == 1) {                                     \
            /* tanh-GELU, times a row of argument 5: geglu's arithmetic */ \
            const float u = ARG_PTR(const float, 5)[row0 + tid]; \
            total = 0.5f * total * (1.0f + tanh(0.7978846f * total * (1.0f + 0.044715f * total * total))) * u; \
        }                                                   \
        y[row0 + tid] = total;                              \
    }

template <int EPI>
kernel void matvec_split_q8_0_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 2, j = (lane & 3) * 8;
    for (int b = sub + 8 * sgi; b < blocks; b += 8 * MV_SPLIT) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (ulong)(row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + (ulong)b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * q8_0_dot8(q, x0, x1);
        }
    }
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_q8_0")]] kernel void matvec_split_q8_0_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_q8_0")]] kernel void matvec_split_q8_0_t<1>(ARGS, KP);

// Two Q8_0 matvecs sharing the same input `x` in one dispatch (4 rows/TG, 1 SG/row, zero barrier).
kernel void matvec_split_q8_0_pair(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    const int cols = ARG_INT(1);
    const int r0 = ARG_INT(4), r1 = ARG_INT(7);
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int g_row = (int)gid.x * 4 + sgi;
    device const u8* w;
    device float* y;
    int row;
    if (g_row < r0) {
        w = ARG_PTR(const u8, 2);
        y = ARG_PTR(float, 3);
        row = g_row;
    } else if (g_row < r0 + r1) {
        w = ARG_PTR(const u8, 5);
        y = ARG_PTR(float, 6);
        row = g_row - r0;
    } else {
        return;
    }
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    device const u8* rptr = w + (ulong)row * row_bytes;
    const int sub = lane >> 2, j = (lane & 3) * 8;
    float sum = 0.0f;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        const uint2 q = *reinterpret_cast<device const uint2*>(rptr + (ulong)b * 32 + j);
        sum += q8_0_scale(rptr, blocks, b) * q8_0_dot8(q, x0, x1);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Four Q8_0 matvecs sharing the same input `x` in one dispatch (4 rows/TG, 1 SG/row, zero barrier).
kernel void matvec_split_q8_0_multi4(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    const int cols = ARG_INT(1);
    const int r0 = ARG_INT(4), r1 = ARG_INT(7), r2 = ARG_INT(10), r3 = ARG_INT(13);
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int g_row = (int)gid.x * 4 + sgi;
    device const u8* w;
    device float* y;
    int row;
    if (g_row < r0) {
        w = ARG_PTR(const u8, 2);
        y = ARG_PTR(float, 3);
        row = g_row;
    } else if (g_row < r0 + r1) {
        w = ARG_PTR(const u8, 5);
        y = ARG_PTR(float, 6);
        row = g_row - r0;
    } else if (g_row < r0 + r1 + r2) {
        w = ARG_PTR(const u8, 8);
        y = ARG_PTR(float, 9);
        row = g_row - r0 - r1;
    } else if (g_row < r0 + r1 + r2 + r3) {
        w = ARG_PTR(const u8, 11);
        y = ARG_PTR(float, 12);
        row = g_row - r0 - r1 - r2;
    } else {
        return;
    }
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    device const u8* rptr = w + (ulong)row * row_bytes;
    const int sub = lane >> 2, j = (lane & 3) * 8;
    float sum = 0.0f;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        const uint2 q = *reinterpret_cast<device const uint2*>(rptr + (ulong)b * 32 + j);
        sum += q8_0_scale(rptr, blocks, b) * q8_0_dot8(q, x0, x1);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Up to 8 Q8_0 attention projections across 3 inputs (h0: q, gate, rel, sink; h1: k, k_taps;
// h2: v, v_taps) in a single dispatch (4 rows/TG, 1 SG/row, zero barrier).
kernel void ink_qkv_q8_0(ARGS, KP) {
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int g_row = (int)gid.x * 4 + sgi;
    const int2 r01 = as_type<int2>(A[19]);
    const int2 r23 = as_type<int2>(A[20]);
    const int2 r45 = as_type<int2>(A[21]);
    const int2 r67 = as_type<int2>(A[22]);
    const int cols = ARG_INT(23);
    const int rs[8] = {r01.x, r01.y, r23.x, r23.y, r45.x, r45.y, r67.x, r67.y};
    int m = 0, off = g_row;
    while (m < 8 && off >= rs[m]) {
        off -= rs[m];
        m++;
    }
    if (m >= 8) return;
    const int x_idx = m < 4 ? 0 : ((m == 4 || m == 6) ? 1 : 2);
    device const float* x = ARG_PTR(const float, x_idx);
    device const u8* w = ARG_PTR(const u8, 3 + m);
    device float* y = ARG_PTR(float, 11 + m);
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    device const u8* rptr = w + (ulong)off * row_bytes;
    const int sub = lane >> 2, j = (lane & 3) * 8;
    float sum = 0.0f;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        const uint2 q = *reinterpret_cast<device const uint2*>(rptr + (ulong)b * 32 + j);
        sum += q8_0_scale(rptr, blocks, b) * q8_0_dot8(q, x0, x1);
    }
    sum = simd_sum(sum);
    if (lane == 0) y[off] = sum;
}

// Single-token fused Q8_0 attn_output matvec + optional new_block stack save +
// attn_post_conv + residual add/overwrite into x (eliminating ink_conv & copy_bytes).
kernel void ink_o_conv_q8_0(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device const float* x_in = ARG_PTR(const float, 1);
    device float* history = ARG_PTR(float, 2);
    device const float* conv_w = ARG_PTR(const float, 3);
    device float* x_out = ARG_PTR(float, 4);
    device float* stack_dst = ARG_PTR(float, 5);  // non-null iff new_block
    const int rows = ARG_INT(6), cols = ARG_INT(7), channels = ARG_INT(8), conv = ARG_INT(9);
    threadgroup float mv_parts[MV_SPLIT][MV_ROWS];
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int row0 = (int)gid.x * MV_ROWS;
    const int nrows = min(MV_ROWS, rows - row0);
    float sum[MV_ROWS];
    for (int r = 0; r < MV_ROWS; r++) sum[r] = 0.0f;
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 2, j = (lane & 3) * 8;
    for (int b = sub + 8 * sgi; b < blocks; b += 8 * MV_SPLIT) {
        device const float* xb = x_in + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (ulong)(row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + (ulong)b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * q8_0_dot8(q, x0, x1);
        }
    }
    for (int r = 0; r < MV_ROWS; r++) {
        const float part = simd_sum(sum[r]);
        if (lane == 0) mv_parts[sgi][r] = part;
    }
    BARRIER;
    if ((int)tid < nrows) {
        const int r = row0 + (int)tid;
        float val = 0.0f;
        for (int g = 0; g < MV_SPLIT; g++) val += mv_parts[g][tid];
        float window[8];
        for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + r];
        window[conv - 1] = val;
        float csum = val;
        device const float* cw = conv_w + (ulong)r * conv;
        for (int k = 0; k < conv; k++) csum += cw[k] * window[k];
        const float y = csum / sqrt((float)(conv + 1));
        if (NONNULL(stack_dst)) {
            stack_dst[r] = x_out[r];
            x_out[r] = y;
        } else {
            x_out[r] += y;
        }
        for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + r] = window[k + 1];
    }
}

template <int EPI>
kernel void matvec_split_q4_0_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub + 16 * sgi; b < blocks; b += 16 * MV_SPLIT) {
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
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_q4_0")]] kernel void matvec_split_q4_0_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_q4_0")]] kernel void matvec_split_q4_0_t<1>(ARGS, KP);

template <int EPI>
kernel void matvec_split_iq4_nl_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub + 16 * sgi; b < blocks; b += 16 * MV_SPLIT) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + j);
            sum[r] += q4_0_scale(row, blocks, b) * (iq4nl_dot8(q, 0, lo0, lo1) + iq4nl_dot8(q, 4, hi0, hi1));
        }
    }
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_iq4_nl")]] kernel void matvec_split_iq4_nl_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_iq4_nl")]] kernel void matvec_split_iq4_nl_t<1>(ARGS, KP);

template <int EPI>
kernel void matvec_split_q4_k_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 144;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
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
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_q4_k")]] kernel void matvec_split_q4_k_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_q4_k")]] kernel void matvec_split_q4_k_t<1>(ARGS, KP);

template <int EPI>
kernel void matvec_split_q5_k_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 176;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
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
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_q5_k")]] kernel void matvec_split_q5_k_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_q5_k")]] kernel void matvec_split_q5_k_t<1>(ARGS, KP);

template <int EPI>
kernel void matvec_split_q6_k_t(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_SPLIT
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * Q6K_BYTES;
    const int sub = lane >> 4, hf = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
        device const float* xb = x + b * 256 + hf * 128 + l;
        float4 xv[4];
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at4(block + hf * 64 + l), ql1 = u32_at4(block + hf * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + hf * 32 + l);
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
    MATVEC_STORE_SPLIT
}
template [[host_name("matvec_split_q6_k")]] kernel void matvec_split_q6_k_t<0>(ARGS, KP);
template [[host_name("matvec_split_gelu_q6_k")]] kernel void matvec_split_q6_k_t<1>(ARGS, KP);

// Q6_K's own few-token products, used for 2 and 3 tokens (matvec_few, below, is faster
// from 4 on). A few tokens' products at once (a draft's guesses, checked in one pass): y[t] = W x[t]
// for `tokens` ≤ MT tokens, rows of x and y a token each. The split kernel's layout, each
// lane's share of a block's weights decoded once and used for every token, so the
// weights are read once as for one token. The prompt kernels' tiles hold 128 tokens: for
// 2 to 8 they do 16 to 64 times the arithmetic needed (Gemma 4 31B checking 4 guesses:
// about 190 ms against 50 for one token). Arguments: w, x, y, rows, cols; one kernel a
// token count, unrolled.
// MR rows a threadgroup (MULTI_ROWS: 8 for up to 4 tokens, 4 above), so the tokens'
// activations, read again by every threadgroup, cost no more than one token's do in the
// single-token kernels; each block's x is loaded once for all of its rows.
template <int MT, int MR>
kernel void matvec_multi_q6_k(ARGS, KP) {
    MATVEC_ARGS
    threadgroup float mv_parts[MV_SPLIT][MR * MT];
    const int lane = tid & 31, sgi = tid >> 5;
    const int row0 = gid.x * MR;
    if (row0 >= rows) return;
    const int nrows = min(MR, rows - row0);
    float sum[MR][MT];
    for (int r = 0; r < MR; r++)
        for (int t = 0; t < MT; t++) sum[r][t] = 0.0f;
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * Q6K_BYTES;
    const int sub = lane >> 4, hf = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
        float4 xv[MT][4];
        for (int t = 0; t < MT; t++) {
            device const float* xb = x + (ulong)t * cols + b * 256 + hf * 128 + l;
            for (int k = 0; k < 4; k++) xv[t][k] = f4(xb + k * 32);
        }
        for (int r = 0; r < MR; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at4(block + hf * 64 + l), ql1 = u32_at4(block + hf * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + hf * 32 + l);
            device const char* sc = reinterpret_cast<device const char*>(block + 192 + hf * 8 + l / 16);
            for (int k = 0; k < 4; k++) {
                const u32 low = ((k & 1) ? ql1 : ql0) >> (k < 2 ? 0 : 4);
                const float f = d * (float)sc[2 * k];
                const float4 wq = f * float4(small_int((low & 0xF) | (((qh >> (2 * k)) & 3) << 4), 32.0f),
                                             small_int(((low >> 8) & 0xF) | (((qh >> (8 + 2 * k)) & 3) << 4), 32.0f),
                                             small_int(((low >> 16) & 0xF) | (((qh >> (16 + 2 * k)) & 3) << 4), 32.0f),
                                             small_int(((low >> 24) & 0xF) | (((qh >> (24 + 2 * k)) & 3) << 4), 32.0f));
                for (int t = 0; t < MT; t++) sum[r][t] += dot(wq, xv[t][k]);
            }
        }
    }
    for (int r = 0; r < MR; r++)
        for (int t = 0; t < MT; t++) {
            const float part = simd_sum(sum[r][t]);
            if (lane == 0) mv_parts[sgi][r * MT + t] = part;
        }
    BARRIER;
    for (int i = tid; i < nrows * MT; i += tpg.x) {
        const int r = i / MT, t = i % MT;
        float total = 0.0f;
        for (int g = 0; g < MV_SPLIT; g++) total += mv_parts[g][r * MT + t];
        y[(ulong)t * rows + row0 + r] = total;
    }
}
template [[host_name("matvec_multi_q6_k_2")]] kernel void matvec_multi_q6_k<2, 4>(ARGS, KP);
template [[host_name("matvec_multi_q6_k_3")]] kernel void matvec_multi_q6_k<3, 4>(ARGS, KP);

// One token's gate and up projections and the activation in one dispatch: y =
// act(W_gate x) × (W_up x), act SiLU (0) or tanh-GELU (1). The split kernels with each
// threadgroup taking 2 rows of both matrices (as 4 rows: r & 1 the row, r >> 1 which
// matrix), x read once for both. Arguments: w_gate, x, y, rows, cols, w_up, act.
#define GLU_ROWS 4
#define MATVEC_ROWS_GLU                                     \
    device const u8* wu = ARG_PTR(const u8, 5);             \
    const int act = ARG_INT(6);                             \
    threadgroup float mv_parts[MV_SPLIT][GLU_ROWS];         \
    const int lane = tid & 31, sgi = tid >> 5;              \
    const int row0 = gid.x * MV_ROWS;                       \
    if (row0 >= rows) return;                               \
    const int nrows = min(MV_ROWS, rows - row0);            \
    float sum[GLU_ROWS];                                    \
    for (int r = 0; r < GLU_ROWS; r++) sum[r] = 0.0f;

#define MATVEC_STORE_GLU                                    \
    for (int r = 0; r < GLU_ROWS; r++) {                    \
        const float part = simd_sum(sum[r]);                \
        if (lane == 0) mv_parts[sgi][r] = part;             \
    }                                                       \
    BARRIER;                                                \
    if ((int)tid < nrows) {                                 \
        float g = 0.0f, u = 0.0f;                           \
        for (int k = 0; k < MV_SPLIT; k++) {                \
            g += mv_parts[k][tid];                          \
            u += mv_parts[k][2 + tid];                      \
        }                                                   \
        const float a = act == 0 ? g / (1.0f + exp(-g))     \
                                 : 0.5f * g * (1.0f + tanh(0.7978846f * g * (1.0f + 0.044715f * g * g))); \
        y[row0 + tid] = a * u;                              \
    }

kernel void matvec_glu_q8_0(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    const int sub = lane >> 2, j = (lane & 3) * 8;
    for (int b = sub + 8 * sgi; b < blocks; b += 8 * MV_SPLIT) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* row = ((r >> 1) ? wu : w) + (ulong)(row0 + (r & 1)) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + (ulong)b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * q8_0_dot8(q, x0, x1);
        }
    }
    MATVEC_STORE_GLU
}

kernel void matvec_glu_q4_0(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub + 16 * sgi; b < blocks; b += 16 * MV_SPLIT) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* row = ((r >> 1) ? wu : w) + (row0 + (r & 1)) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + j);
            const float dot = nibble_dot8(q, 0, lo0, lo1) + nibble_dot8(q, 4, hi0, hi1) -
                              8.0f * ((lo0.x + lo0.y) + (lo0.z + lo0.w) + (lo1.x + lo1.y) + (lo1.z + lo1.w) +
                                      (hi0.x + hi0.y) + (hi0.z + hi0.w) + (hi1.x + hi1.y) + (hi1.z + hi1.w));
            sum[r] += q4_0_scale(row, blocks, b) * dot;
        }
    }
    MATVEC_STORE_GLU
}

kernel void matvec_glu_iq4_nl(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub + 16 * sgi; b < blocks; b += 16 * MV_SPLIT) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* row = ((r >> 1) ? wu : w) + (row0 + (r & 1)) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + j);
            sum[r] += q4_0_scale(row, blocks, b) * (iq4nl_dot8(q, 0, lo0, lo1) + iq4nl_dot8(q, 4, hi0, hi1));
        }
    }
    MATVEC_STORE_GLU
}

kernel void matvec_glu_q4_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 144;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
        device const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* block = ((r >> 1) ? wu : w) + (row0 + (r & 1)) * row_bytes + b * 144;
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
    MATVEC_STORE_GLU
}

kernel void matvec_glu_q5_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * 176;
    const int sub = lane >> 4, pair = (lane & 15) >> 2, part = (lane & 3) * 8;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
        device const float* xb = x + b * 256 + pair * 64 + part;
        const float4 a1 = f4(xb), b1 = f4(xb + 4), a2 = f4(xb + 32), b2 = f4(xb + 36);
        const float x1 = (a1.x + a1.y) + (a1.z + a1.w) + (b1.x + b1.y) + (b1.z + b1.w);
        const float x2 = (a2.x + a2.y) + (a2.z + a2.w) + (b2.x + b2.y) + (b2.z + b2.w);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* block = ((r >> 1) ? wu : w) + (row0 + (r & 1)) * row_bytes + b * 176;
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
    MATVEC_STORE_GLU
}

kernel void matvec_glu_q6_k(ARGS, KP) {
    MATVEC_ARGS
    MATVEC_ROWS_GLU
    const int blocks = cols / 256;
    const ulong row_bytes = (ulong)blocks * Q6K_BYTES;
    const int sub = lane >> 4, hf = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub + 2 * sgi; b < blocks; b += 2 * MV_SPLIT) {
        device const float* xb = x + b * 256 + hf * 128 + l;
        float4 xv[4];
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
        for (int r = 0; r < GLU_ROWS; r++) {
            if ((r & 1) >= nrows) continue;
            device const u8* block = ((r >> 1) ? wu : w) + (row0 + (r & 1)) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at4(block + hf * 64 + l), ql1 = u32_at4(block + hf * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + hf * 32 + l);
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
    MATVEC_STORE_GLU
}

// ---- Matrix-matrix products: Y = X Wᵀ for many tokens at once ------------------------

enum { K_F32, K_F16, K_Q4_0, K_Q8_0, K_Q4_K, K_Q5_K, K_Q6_K, K_IQ4_NL };

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
inline void decode8<K_IQ4_NL>(device const u8* row, int blocks, int k, thread float* v) {
    const int b = k / 32, j = k % 32;
    const float d = q4_0_scale(row, blocks, b);
    const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + (j & 15));
    const int shift = j < 16 ? 0 : 4;
    const u32 lo = (q.x >> shift) & 0x0F0F0F0Fu, hi = (q.y >> shift) & 0x0F0F0F0Fu;
    for (int i = 0; i < 4; i++) {
        v[i] = d * KVALUES_IQ4NL_F[(lo >> (8 * i)) & 0xF];
        v[4 + i] = d * KVALUES_IQ4NL_F[(hi >> (8 * i)) & 0xF];
    }
}

template <>
inline void decode8<K_Q8_0>(device const u8* row, int blocks, int k, thread float* v) {
    const int b = k / 32;
    const float d = q8_0_scale(row, blocks, b);
    const uint2 q = *reinterpret_cast<device const uint2*>(row + k);
    const char4 q0 = as_type<char4>(q.x), q1 = as_type<char4>(q.y);
    v[0] = d * (float)q0.x;
    v[1] = d * (float)q0.y;
    v[2] = d * (float)q0.z;
    v[3] = d * (float)q0.w;
    v[4] = d * (float)q1.x;
    v[5] = d * (float)q1.y;
    v[6] = d * (float)q1.z;
    v[7] = d * (float)q1.w;
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

// Thirty-two numbers of a row from column k (a multiple of 32), decoded as decode8 does
// and stored to threadgroup memory as f16: one Q4_0 or Q8_0 block, or one sub-block of a
// K-quant, so its scale is worked out once and its bits read 16 bytes at a time.
template <int KIND>
static inline void decode32(device const u8* row, int blocks, int k, threadgroup half* out) {
    float v[8];
    for (int i = 0; i < 4; i++) {
        decode8<KIND>(row, blocks, k + 8 * i, v);
        for (int j = 0; j < 8; j++) out[8 * i + j] = half(v[j]);
    }
}

// Four f16 numbers from the low (shift 0) or high (shift 4) nibbles of 4 bytes, as
// scale × nibble − offset.
static inline half4 nibbles4(u32 b, int shift, float scale, float offset) {
    const u32 q = (b >> shift) & 0x0F0F0F0Fu;
    return half4(float4((float)(q & 0xFF), (float)((q >> 8) & 0xFF), (float)((q >> 16) & 0xFF), (float)(q >> 24)) *
                     scale - offset);
}

template <>
inline void decode32<K_F32>(device const u8* row, int blocks, int k, threadgroup half* out) {
    device const packed_float4* r = reinterpret_cast<device const packed_float4*>(row + (ulong)k * 4);
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    for (int i = 0; i < 8; i++) o[i] = half4(float4(r[i]));
}

template <>
inline void decode32<K_F16>(device const u8* row, int blocks, int k, threadgroup half* out) {
    device const packed_half4* r = reinterpret_cast<device const packed_half4*>(row + (ulong)k * 2);
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    for (int i = 0; i < 8; i++) o[i] = half4(r[i]);
}

template <>
inline void decode32<K_Q8_0>(device const u8* row, int blocks, int k, threadgroup half* out) {
    const int b = k / 32;
    const half d = half(q8_0_scale(row, blocks, b));
    device const uint4* q = reinterpret_cast<device const uint4*>(row + (ulong)b * 32);
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    const uint4 q0 = q[0], q1 = q[1];
    o[0] = half4(as_type<char4>(q0.x)) * d;
    o[1] = half4(as_type<char4>(q0.y)) * d;
    o[2] = half4(as_type<char4>(q0.z)) * d;
    o[3] = half4(as_type<char4>(q0.w)) * d;
    o[4] = half4(as_type<char4>(q1.x)) * d;
    o[5] = half4(as_type<char4>(q1.y)) * d;
    o[6] = half4(as_type<char4>(q1.z)) * d;
    o[7] = half4(as_type<char4>(q1.w)) * d;
}

template <>
inline void decode32<K_Q4_0>(device const u8* row, int blocks, int k, threadgroup half* out) {
    const int b = k / 32;
    const float d = q4_0_scale(row, blocks, b);
    const uint4 q = *reinterpret_cast<device const uint4*>(row + b * 16);
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    const u32 w[4] = {q.x, q.y, q.z, q.w};
    for (int i = 0; i < 4; i++) {
        o[i] = nibbles4(w[i], 0, d, 8.0f * d);      // numbers 4i.. (low nibbles)
        o[4 + i] = nibbles4(w[i], 4, d, 8.0f * d);  // numbers 16 + 4i.. (high)
    }
}

static inline half4 iq4nl4(u32 b, int shift, half scale) {
    const u32 q = (b >> shift) & 0x0F0F0F0Fu;
    return half4(KVALUES_IQ4NL_F[q & 0xF], KVALUES_IQ4NL_F[(q >> 8) & 0xF],
                 KVALUES_IQ4NL_F[(q >> 16) & 0xF], KVALUES_IQ4NL_F[q >> 24]) * scale;
}

template <>
inline void decode32<K_IQ4_NL>(device const u8* row, int blocks, int k, threadgroup half* out) {
    const int b = k / 32;
    const half d = half(q4_0_scale(row, blocks, b));
    const uint4 q = *reinterpret_cast<device const uint4*>(row + b * 16);
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    const u32 w[4] = {q.x, q.y, q.z, q.w};
    for (int i = 0; i < 4; i++) {
        o[i] = iq4nl4(w[i], 0, d);
        o[4 + i] = iq4nl4(w[i], 4, d);
    }
}

template <>
inline void decode32<K_Q4_K>(device const u8* row, int blocks, int k, threadgroup half* out) {
    device const u8* block = row + (k / 256) * 144;
    const int j = (k % 256) / 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    device const uint4* q = reinterpret_cast<device const uint4*>(block + 16 + (j / 2) * 32);
    const int shift = (j & 1) * 4;
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    for (int h = 0; h < 2; h++) {
        const uint4 qq = q[h];
        const u32 w[4] = {qq.x, qq.y, qq.z, qq.w};
        for (int i = 0; i < 4; i++) o[4 * h + i] = nibbles4(w[i], shift, d1, min1);
    }
}

template <>
inline void decode32<K_Q5_K>(device const u8* row, int blocks, int k, threadgroup half* out) {
    device const u8* block = row + (k / 256) * 176;
    const int j = (k % 256) / 32;
    float sc, m;
    q4_k_scale_min(j, block + 4, sc, m);
    const float d1 = half_at(block) * sc, min1 = half_at(block + 2) * m;
    device const uint4* q = reinterpret_cast<device const uint4*>(block + 48 + (j / 2) * 32);
    device const uint4* qh = reinterpret_cast<device const uint4*>(block + 16);
    const int shift = (j & 1) * 4;
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    for (int h = 0; h < 2; h++) {
        const uint4 qq = q[h], hh = qh[h];
        const u32 w[4] = {qq.x, qq.y, qq.z, qq.w}, hw[4] = {hh.x, hh.y, hh.z, hh.w};
        for (int i = 0; i < 4; i++) {
            // the fifth bit of number l is bit j of qh[l]: 16 more
            const u32 bits = (hw[i] >> j) & 0x01010101u;
            const u32 qv = ((w[i] >> shift) & 0x0F0F0F0Fu) | (bits << 4);
            o[4 * h + i] = half4(float4((float)(qv & 0xFF), (float)((qv >> 8) & 0xFF), (float)((qv >> 16) & 0xFF),
                                        (float)(qv >> 24)) * d1 - min1);
        }
    }
}

template <>
inline void decode32<K_Q6_K>(device const u8* row, int blocks, int k, threadgroup half* out) {
    device const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, hf = n / 128, run = (n % 128) / 32;
    const float d = half_at(block + 208);
    // (packed: a block starts at a 4-byte word, not always a 16-byte one.)
    device const packed_uint4* ql = reinterpret_cast<device const packed_uint4*>(block + hf * 64 + (run & 1) * 32);
    device const packed_uint4* qh = reinterpret_cast<device const packed_uint4*>(block + 128 + hf * 32);
    device const char* sc = reinterpret_cast<device const char*>(block + 192 + hf * 8 + 2 * run);
    const int shift = run < 2 ? 0 : 4;
    threadgroup half4* o = reinterpret_cast<threadgroup half4*>(out);
    for (int h = 0; h < 2; h++) {  // numbers 16h.. take scale sc[h]
        const float ds = d * (float)sc[h];
        const uint4 lq = ql[h], hq = qh[h];
        const u32 lw[4] = {lq.x, lq.y, lq.z, lq.w}, hw[4] = {hq.x, hq.y, hq.z, hq.w};
        for (int i = 0; i < 4; i++) {
            const u32 qv = ((lw[i] >> shift) & 0x0F0F0F0Fu) | (((hw[i] >> (2 * run)) & 0x03030303u) << 4);
            o[4 * h + i] = half4((float4((float)(qv & 0xFF), (float)((qv >> 8) & 0xFF), (float)((qv >> 16) & 0xFF),
                                         (float)(qv >> 24)) - 32.0f) * ds);
        }
    }
}

template <int KIND>
static inline ulong mm_row_bytes(int cols) {
    switch (KIND) {
        case K_F32: return (ulong)cols * 4;
        case K_F16: return (ulong)cols * 2;
        case K_Q4_0:
        case K_IQ4_NL: return q4_0_row_bytes(cols);
        case K_Q8_0: return q8_0_row_bytes(cols);
        case K_Q4_K: return (ulong)(cols / 256) * 144;
        case K_Q5_K: return (ulong)(cols / 256) * 176;
        default: return (ulong)(cols / 256) * Q6K_BYTES;
    }
}

// A few tokens' products (up to 4: forced text, a short tool result, a draft's guesses):
// y[t] = W x[t], rows of x and y a token each. The prompt kernels' tiles hold 128
// tokens, so a handful cost as much as 128 (Gemma 4 E4B: 82-120 µs a matrix for 2
// tokens, 18-30 for one). Here a threadgroup takes FEW_ROWS rows, its 4 SIMD groups
// every 4th run of 256 columns (as the split kernels do), each lane 8 columns at a time:
// each weight is decoded once and used for every token, so the weights are read once,
// as for one token. MT is the most tokens (2 or 4); `tokens` may be fewer. On the M5
// Max, of 1, 2, 4 and 8 rows a threadgroup, and 8 tokens at once (slower than two
// dispatches of 4), 2 rows read fastest. Arguments: w, x, y, rows, cols, tokens.
template <int KIND, int MT, int FEW_ROWS>
kernel void matvec_few(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    device const u8* w = ARG_PTR(const u8, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    threadgroup float parts[4][FEW_ROWS][MT];
    const int row0 = (int)gid.x * FEW_ROWS;
    if (row0 >= rows) return;
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    const int blocks = cols / 32;
    float acc[FEW_ROWS][MT];
    for (int r = 0; r < FEW_ROWS; r++)
        for (int t = 0; t < MT; t++) acc[r][t] = 0.0f;
    // The 4 SIMD groups take every 4th run of 256 columns.
    for (int k = ((int)sg * 32 + (int)lane) * 8; k < cols; k += 1024) {
        float4 xa[MT], xb[MT];
        for (int t = 0; t < MT; t++) {
            if (t < tokens) {
                xa[t] = f4(x + (ulong)t * cols + k);
                xb[t] = f4(x + (ulong)t * cols + k + 4);
            } else {
                xa[t] = xb[t] = float4(0.0f);
            }
        }
        for (int r = 0; r < FEW_ROWS; r++) {
            if (row0 + r >= rows) break;
            float v[8];
            decode8<KIND>(w + (ulong)(row0 + r) * row_bytes, blocks, k, v);
            const float4 va = float4(v[0], v[1], v[2], v[3]), vb = float4(v[4], v[5], v[6], v[7]);
            for (int t = 0; t < MT; t++) acc[r][t] += dot(va, xa[t]) + dot(vb, xb[t]);
        }
    }
    for (int r = 0; r < FEW_ROWS; r++)
        for (int t = 0; t < MT; t++) {
            const float part = simd_sum(acc[r][t]);
            if (lane == 0) parts[sg][r][t] = part;
        }
    BARRIER;
    if ((int)tid < FEW_ROWS * MT) {
        const int r = (int)tid / MT, t = (int)tid % MT;
        if (row0 + r < rows && t < tokens)
            y[(ulong)t * rows + row0 + r] = ((parts[0][r][t] + parts[1][r][t]) + parts[2][r][t]) + parts[3][r][t];
    }
}

#define FEW_KINDS(MT, FR)                                                                                 \
    template [[host_name("matvec_few" #MT "_f32")]] kernel void matvec_few<K_F32, MT, FR>(ARGS, KP, uint, uint);   \
    template [[host_name("matvec_few" #MT "_f16")]] kernel void matvec_few<K_F16, MT, FR>(ARGS, KP, uint, uint);   \
    template [[host_name("matvec_few" #MT "_q4_0")]] kernel void matvec_few<K_Q4_0, MT, FR>(ARGS, KP, uint, uint); \
    template [[host_name("matvec_few" #MT "_q8_0")]] kernel void matvec_few<K_Q8_0, MT, FR>(ARGS, KP, uint, uint); \
    template [[host_name("matvec_few" #MT "_q4_k")]] kernel void matvec_few<K_Q4_K, MT, FR>(ARGS, KP, uint, uint); \
    template [[host_name("matvec_few" #MT "_q5_k")]] kernel void matvec_few<K_Q5_K, MT, FR>(ARGS, KP, uint, uint); \
    template [[host_name("matvec_few" #MT "_q6_k")]] kernel void matvec_few<K_Q6_K, MT, FR>(ARGS, KP, uint, uint); \
    template [[host_name("matvec_few" #MT "_iq4_nl")]] kernel void matvec_few<K_IQ4_NL, MT, FR>(ARGS, KP, uint, uint);
FEW_KINDS(2, 2)
FEW_KINDS(3, 2)
FEW_KINDS(4, 2)


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

// The same product on the SIMD-group matrix units: tiles of SG_ROWS rows × SG_TOKENS
// tokens, SG_K columns at a time. The 128 threads decode the tile's weights (as f32,
// with decode8) and copy its activations into threadgroup memory; each of the 4 SIMD
// groups then multiplies 8 × 8 blocks for 32 rows × 16 tokens of the tile, in f32 as the
// CPU does. C = W Xᵀ (rows × tokens) is stored transposed, as y's rows of `rows`.
#define SG_ROWS 64
#define SG_TOKENS 32
#define SG_K 32
#define SG_PAD 4
template <int KIND>
kernel void matmul_sg(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]]) {
    device const u8* w = ARG_PTR(const u8, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    constexpr int SA = SG_K + SG_PAD, SB = SG_K + SG_PAD;
    threadgroup float sa[SG_ROWS * SA];      // weights: [row][k]
    threadgroup float sb[SG_TOKENS * SB];    // activations: [token][k]
    threadgroup float cs[SG_TOKENS * SG_ROWS];  // the results of a partial tile: [token][row]
    const int row0 = gid.x * SG_ROWS, tok0 = gid.y * SG_TOKENS;
    // Loading: thread → row tid / 2, columns (tid % 2) × 16.. of the weights; token
    // tid / 4, columns (tid % 4) × 8.. of the activations.
    const int lr = tid >> 1, lk = (tid & 1) * 16;
    const int lt = tid >> 2, lx = (tid & 3) * 8;
    const bool row_live = row0 + lr < rows, tok_live = tok0 + lt < tokens;
    device const u8* wrow = w + (ulong)(row0 + lr) * row_bytes;
    device const float* xrow = x + (ulong)(tok0 + lt) * cols;
    // This SIMD group's part of the tile: rows sr.., tokens st..
    const int sr = (sg & 1) * 32, st = (sg >> 1) * 16;
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < cols; k0 += SG_K) {
        float v[8];
        for (int part = 0; part < 2; part++) {
            if (row_live) {
                decode8<KIND>(wrow, cols / 32, k0 + lk + part * 8, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) sa[lr * SA + lk + part * 8 + i] = v[i];
        }
        if (tok_live) {
            const float4 a = f4(xrow + k0 + lx), b = f4(xrow + k0 + lx + 4);
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx]) = a;
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx + 4]) = b;
        } else {
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx]) = float4(0.0f);
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx + 4]) = float4(0.0f);
        }
        BARRIER;
        for (int kk = 0; kk < SG_K; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], &sa[(sr + 8 * i) * SA + kk], SA);
            // Xᵀ: the activations' [token][k] read transposed, as k × tokens.
            for (int j = 0; j < 2; j++) simdgroup_load(b[j], &sb[(st + 8 * j) * SB + kk], SB, ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        BARRIER;
    }
    if (row0 + SG_ROWS <= rows && tok0 + SG_TOKENS <= tokens) {
        // A whole tile: straight to y, transposed (y holds a row of `rows` per token).
        device float* out = y + (ulong)tok0 * rows + row0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                simdgroup_store(acc[i][j], out + (ulong)(st + 8 * j) * rows + sr + 8 * i, (ulong)rows, ulong2(0, 0), true);
        return;
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            simdgroup_store(acc[i][j], &cs[(st + 8 * j) * SG_ROWS + sr + 8 * i], SG_ROWS, ulong2(0, 0), true);
    BARRIER;
    for (int e = tid; e < SG_ROWS * SG_TOKENS; e += tpg.x) {
        const int t = e / SG_ROWS, r = e % SG_ROWS;
        if (tok0 + t < tokens && row0 + r < rows) y[(ulong)(tok0 + t) * rows + row0 + r] = cs[e];
    }
}

template [[host_name("matmul_sg_f32")]] kernel void matmul_sg<K_F32>(ARGS, KP, uint);
template [[host_name("matmul_sg_f16")]] kernel void matmul_sg<K_F16>(ARGS, KP, uint);
template [[host_name("matmul_sg_q4_0")]] kernel void matmul_sg<K_Q4_0>(ARGS, KP, uint);
template [[host_name("matmul_sg_q8_0")]] kernel void matmul_sg<K_Q8_0>(ARGS, KP, uint);
template [[host_name("matmul_sg_q4_k")]] kernel void matmul_sg<K_Q4_K>(ARGS, KP, uint);
template [[host_name("matmul_sg_q5_k")]] kernel void matmul_sg<K_Q5_K>(ARGS, KP, uint);
template [[host_name("matmul_sg_q6_k")]] kernel void matmul_sg<K_Q6_K>(ARGS, KP, uint);
template [[host_name("matmul_sg_iq4_nl")]] kernel void matmul_sg<K_IQ4_NL>(ARGS, KP, uint);

template [[host_name("matmul_f32")]] kernel void matmul<K_F32>(ARGS, KP);
template [[host_name("matmul_f16")]] kernel void matmul<K_F16>(ARGS, KP);
template [[host_name("matmul_q4_0")]] kernel void matmul<K_Q4_0>(ARGS, KP);
template [[host_name("matmul_q8_0")]] kernel void matmul<K_Q8_0>(ARGS, KP);
template [[host_name("matmul_q4_k")]] kernel void matmul<K_Q4_K>(ARGS, KP);
template [[host_name("matmul_q5_k")]] kernel void matmul<K_Q5_K>(ARGS, KP);
template [[host_name("matmul_q6_k")]] kernel void matmul<K_Q6_K>(ARGS, KP);
template [[host_name("matmul_iq4_nl")]] kernel void matmul<K_IQ4_NL>(ARGS, KP);

// A batch's slots of a mixture of experts (see moe_pick), put in order of their expert
// so each expert's matrix is read once for all the tokens that chose it, not once a
// token: `order` is the slots sorted by expert (in slot order within one), `tiles` cuts
// that order into runs of up to `tile_slots` slots of one expert, three numbers a tile
// (the expert, where its run starts in `order`, how many slots; 0 for a tile not used),
// and `place` is where each slot went in `order`. Two passes of a threadgroup an expert:
// moe_count counts each expert's slots into `counts`; moe_place, from the counts before
// its expert, lays out that expert's tiles and puts its slots in order (a SIMD prefix sum
// a chunk); the last expert's also empties the unused tiles. (One threadgroup doing it
// all took 0.78 ms for 32K slots.) Arguments: idx, order, tiles, slots, experts,
// max_tiles, tile_slots, place, counts.
#define MOE_MAX_EXPERTS 512
kernel void moe_count(ARGS, KP) {
    device const int* idx = ARG_PTR(const int, 0);
    const int slots = ARG_INT(3);
    device int* counts = ARG_PTR(int, 8);
    const int e = gid.x;
    int n = 0;
    for (int s = tid; s < slots; s += tpg.x) n += idx[s] == e ? 1 : 0;
    threadgroup int scratch[32];
    n = simd_sum(n);
    if ((tid & 31) == 0) scratch[tid >> 5] = n;
    BARRIER;
    if (tid == 0) {
        int total = 0;
        for (uint w = 0; w < (tpg.x + 31) / 32; w++) total += scratch[w];
        counts[e] = total;
    }
}

kernel void moe_place(ARGS, KP) {
    device const int* idx = ARG_PTR(const int, 0);
    device int* order = ARG_PTR(int, 1);
    device int* tiles = ARG_PTR(int, 2);
    const int slots = ARG_INT(3), experts = ARG_INT(4), max_tiles = ARG_INT(5), tile_slots = ARG_INT(6);
    device int* place = ARG_PTR(int, 7);
    device const int* counts = ARG_PTR(const int, 8);
    const int e = gid.x;
    threadgroup int base[2];
    threadgroup int warp_sums[32];
    if (tid == 0) {
        int start = 0, tile0 = 0;
        for (int x = 0; x < e; x++) {
            start += counts[x];
            tile0 += (counts[x] + tile_slots - 1) / tile_slots;
        }
        base[0] = start;
        base[1] = tile0;
        const int n = counts[e];
        int t = tile0;
        for (int first = 0; first < n && t < max_tiles; first += tile_slots, t++) {
            tiles[3 * t] = e;
            tiles[3 * t + 1] = start + first;
            tiles[3 * t + 2] = min(tile_slots, n - first);
        }
        if (e == experts - 1) {
            for (; t < max_tiles; t++) tiles[3 * t + 2] = 0;
        }
    }
    BARRIER;
    int at = base[0];
    const uint lane = tid & 31, warp = tid >> 5, warps = (tpg.x + 31) / 32;
    for (int c = 0; c < slots; c += tpg.x) {
        const int s = c + tid;
        const int mine = s < slots && idx[s] == e ? 1 : 0;
        const int before = simd_prefix_exclusive_sum(mine);
        const int in_warp = simd_sum(mine);
        if (lane == 0) warp_sums[warp] = in_warp;
        BARRIER;
        int offset = 0, total = 0;
        for (uint w = 0; w < warps; w++) {
            if (w < warp) offset += warp_sums[w];
            total += warp_sums[w];
        }
        if (mine) {
            order[at + offset + before] = s;
            place[s] = at + offset + before;
        }
        at += total;
        BARRIER;
    }
}


// `matmul_sg` for one tile of moe_group's: its expert's rows × its slots, the tile a
// threadgroup row (gid.y), on the SIMD-group matrix units. The slots' inputs are gathered
// through `order` and their products stored each to its own slot, as moe_matvec does.
// (As `matmul`, 128 rows × 64 slots in scalar f32, it ran at a seventh of this: 8.6 ms a
// layer's gate and up rows for 1,024 tokens on the M5 Max.) Arguments: w, x, y, rows (an
// expert's), cols, order, tiles, used, x_stride, x_per_slot, y_stride.
template <int KIND>
kernel void moe_matmul(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]]) {
    const int rows = ARG_INT(3), cols = ARG_INT(4);
    device const int* order = ARG_PTR(const int, 5);
    device const int* tile = ARG_PTR(const int, 6) + 3 * gid.y;
    const int used = ARG_INT(7), x_stride = ARG_INT(8), x_per_slot = ARG_INT(9), y_stride = ARG_INT(10);
    const int tokens = tile[2];
    if (tokens == 0) return;
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    device const u8* w = ARG_PTR(const u8, 0) + (ulong)tile[0] * (ulong)rows * row_bytes;
    device const float* x = ARG_PTR(const float, 1);
    device float* y = ARG_PTR(float, 2);
    device const int* slots = order + tile[1];
    constexpr int SA = SG_K + SG_PAD, SB = SG_K + SG_PAD;
    threadgroup float sa[SG_ROWS * SA];
    threadgroup float sb[SG_TOKENS * SB];
    threadgroup float cs[SG_TOKENS * SG_ROWS];
    const int row0 = gid.x * SG_ROWS;
    const int lr = tid >> 1, lk = (tid & 1) * 16;
    const int lt = tid >> 2, lx = (tid & 3) * 8;
    const bool row_live = row0 + lr < rows, tok_live = lt < tokens;
    device const u8* wrow = w + (ulong)(row0 + lr) * row_bytes;
    device const float* xrow = x;
    if (tok_live) {
        const int slot = slots[lt];
        xrow = x + (ulong)(x_per_slot ? slot : slot / used) * x_stride;
    }
    const int sr = (sg & 1) * 32, st = (sg >> 1) * 16;
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < cols; k0 += SG_K) {
        float v[8];
        for (int part = 0; part < 2; part++) {
            if (row_live) {
                decode8<KIND>(wrow, cols / 32, k0 + lk + part * 8, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) sa[lr * SA + lk + part * 8 + i] = v[i];
        }
        if (tok_live) {
            const float4 a = f4(xrow + k0 + lx), b = f4(xrow + k0 + lx + 4);
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx]) = a;
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx + 4]) = b;
        } else {
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx]) = float4(0.0f);
            *reinterpret_cast<threadgroup float4*>(&sb[lt * SB + lx + 4]) = float4(0.0f);
        }
        BARRIER;
        for (int kk = 0; kk < SG_K; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (int i = 0; i < 4; i++) simdgroup_load(a[i], &sa[(sr + 8 * i) * SA + kk], SA);
            for (int j = 0; j < 2; j++) simdgroup_load(b[j], &sb[(st + 8 * j) * SB + kk], SB, ulong2(0, 0), true);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        BARRIER;
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            simdgroup_store(acc[i][j], &cs[(st + 8 * j) * SG_ROWS + sr + 8 * i], SG_ROWS, ulong2(0, 0), true);
    BARRIER;
    for (int e = tid; e < SG_ROWS * SG_TOKENS; e += tpg.x) {
        const int t = e / SG_ROWS, r = e % SG_ROWS;
        if (t < tokens && row0 + r < rows) y[(ulong)slots[t] * y_stride + row0 + r] = cs[e];
    }
}
template [[host_name("moe_matmul_f32")]] kernel void moe_matmul<K_F32>(ARGS, KP, uint);
template [[host_name("moe_matmul_f16")]] kernel void moe_matmul<K_F16>(ARGS, KP, uint);
template [[host_name("moe_matmul_q4_0")]] kernel void moe_matmul<K_Q4_0>(ARGS, KP, uint);
template [[host_name("moe_matmul_q8_0")]] kernel void moe_matmul<K_Q8_0>(ARGS, KP, uint);
template [[host_name("moe_matmul_q4_k")]] kernel void moe_matmul<K_Q4_K>(ARGS, KP, uint);
template [[host_name("moe_matmul_q5_k")]] kernel void moe_matmul<K_Q5_K>(ARGS, KP, uint);
template [[host_name("moe_matmul_q6_k")]] kernel void moe_matmul<K_Q6_K>(ARGS, KP, uint);
template [[host_name("moe_matmul_iq4_nl")]] kernel void moe_matmul<K_IQ4_NL>(ARGS, KP, uint);

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
    // Optionally rows_per_weight > 0: rows [k × rows_per_weight, ...) use weight k (0, 1, 2:
    // arguments 1, 5, 6), so several norms of contiguous blocks are one dispatch.
    const int rows_per_weight = ARG_INT(7);
    if (rows_per_weight > 0) {
        const int k = (int)gid.x / rows_per_weight;
        if (k == 1) weight = ARG_PTR(const float, 5);
        else if (k == 2) weight = ARG_PTR(const float, 6);
    }
    const bool has_weight = NONNULL(weight);
    float sum = 0.0f;
    for (int i = tid; i < n; i += tpg.x) sum += xr[i] * xr[i];
    sum = block_sum_once(sum, scratch, tid, tpg.x);
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
    threadgroup float scratch[32], scratch2[32];
    const bool has_p = NONNULL(p), has_post = NONNULL(post), has_next = NONNULL(next);
    x += (ulong)gid.x * n;
    float p_scale = 1.0f;
    if (has_p) {
        p += (ulong)gid.x * n;
        if (has_post) {
            float sum = 0.0f;
            for (int i = tid; i < n; i += tpg.x) sum += p[i] * p[i];
            sum = block_sum_once(sum, scratch2, tid, tpg.x);
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
    sum = block_sum_once(sum, scratch, tid, tpg.x);
    const float s = 1.0f / sqrt(sum / (float)n + eps);
    h += (ulong)gid.x * n;
    for (int i = tid; i < n; i += tpg.x) h[i] = x[i] * s * next[i];
}

// Where a position's keys and values sit in a layer's cache: its row, or for a ring (a
// sliding-window layer that keeps only the last `ring` positions) its row modulo the ring.
// The ring travels in the window argument's high 16 bits (see Gpu::attention).
inline ulong ring_row(int t, int ring) { return (ulong)(ring > 0 ? t % ring : t); }

// See qkv_prep in syrinx.cu.
#define QKV_TOKENS 8
struct BatchChat {
    ulong keys;
    ulong values;
    int pos;
    int ring;
};

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
    const int half_cache = ARG_INT(20), tokens = ARG_INT(21), ring = ARG_INT(22);
    // The queries once more in f16, a key/value head's together across tokens, for
    // attention_tcg (see q_group_half in syrinx_tensor.metal), when given.
    device half* q_half = ARG_PTR(half, 23);

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
        int at, c_ring;
        ulong c_k, c_v;
        if (NONNULL(pos)) {
            at = *pos + t;
            c_ring = ring;
            c_k = k_cache;
            c_v = v_cache;
        } else {
            device const BatchChat* bc = reinterpret_cast<device const BatchChat*>(k_cache) + t;
            at = bc->pos;
            c_ring = bc->ring;
            c_k = bc->keys;
            c_v = bc->values;
        }
        device const float* src;
        device float* dst = nullptr;
        device half* half_dst = nullptr;
        device half* q2 = nullptr;
        if (kind == 0) {
            src = q_src + (ulong)t * q_stride + (ulong)h * q_head_stride;
            dst = q_dst + ((ulong)t * heads + h) * hd;
            const int group = heads / kv_heads;
            if (NONNULL(q_half)) q2 = q_half + (((ulong)(h / group) * tokens + t) * group + h % group) * hd;
        } else {
            src = (kind == 1 ? k_src : v_src) + (ulong)t * kv_dim + h * hd;
            const ulong at_cache = ring_row(at, c_ring) * kv_dim + h * hd;
            const ulong cache = kind == 1 ? c_k : c_v;
            if (half_cache == 1)
                half_dst = reinterpret_cast<device half*>(cache) + at_cache;
            else if (half_cache == 0)
                dst = reinterpret_cast<device float*>(cache) + at_cache;
        }
        const bool to_half = half_cache == 1 && kind != 0;
        const bool to_q8 = half_cache == 2 && kind != 0;
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
                if (to_half) half_dst[i] = half(buf[i]); else if (!to_q8) dst[i] = buf[i];
                if (NONNULL(q2)) q2[i] = half(buf[i]);
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
            if (to_q8) {
                buf[i0] = r0;
                buf[i1] = r1;
            } else if (to_half) {
                half_dst[i0] = half(r0);
                half_dst[i1] = half(r1);
            } else {
                dst[i0] = r0;
                dst[i1] = r1;
            }
            if (NONNULL(q2)) {
                q2[i0] = half(r0);
                q2[i1] = half(r1);
            }
        }
        if (to_q8) {
            BARRIER;
            const ulong cache = kind == 1 ? c_k : c_v;
            const ulong head_bytes = (ulong)hd + (ulong)(hd / 32) * 2;
            device char* q8_dst =
                reinterpret_cast<device char*>(cache) + (ring_row(at, c_ring) * (ulong)kv_heads + (ulong)h) * head_bytes;
            device half* sc_dst = reinterpret_cast<device half*>(q8_dst + hd);
            const int sg = (int)(tid >> 5), lane = (int)(tid & 31), nsg = (int)(tpg.x >> 5);
            for (int b = sg; b < hd / 32; b += nsg) {
                const float v = buf[b * 32 + lane];
                const float amax = simd_max(fabs(v));
                const float d = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                q8_dst[b * 32 + lane] = (char)(int)rint(v * id);
                if (lane == 0) sc_dst[b] = half(d);
            }
        }
    }
}

// Dequantize a Q8_0 KV cache (layout: per (row, kv_head), hd int8s + hd/32 half scales)
// into a contiguous f16 KV scratch buffer for prompt attention (attention_tcg / attention_tc).
// Grid: (kv_heads, stride_blocks), threads: 256 (8 SIMD groups, 1 SIMD group per 32-elt block).
// With a sliding window and no ring, only the rows the batch's first token reads from on.
kernel void dequant_kv_q8_to_half(ARGS, KP) {
    device const char* src = ARG_PTR(const char, 0);
    device half* dst = ARG_PTR(half, 1);
    device const int* pos = ARG_PTR(const int, 2);
    const int tokens = ARG_INT(3), ring = ARG_INT(4), kv_heads = ARG_INT(5), hd = ARG_INT(6);
    const int window = ARG_INT(7);
    const int rows = ring > 0 ? ring : (*pos + tokens);
    const int first = ring == 0 && window > 0 ? max(0, *pos + 1 - window) : 0;
    const int h = gid.x;
    const ulong head_bytes = (ulong)hd + (ulong)(hd / 32) * 2;
    const ulong row_bytes = (ulong)kv_heads * head_bytes;
    const ulong kv_dim = (ulong)kv_heads * (ulong)hd;
    const int sg = (int)(tid >> 5), lane = (int)(tid & 31), nsg = (int)(tpg.x >> 5);
    const int blocks = hd / 32;
    for (int r = first + (int)gid.y; r < rows; r += (int)ngrid.y) {
        device const char* q8 = src + (ulong)r * row_bytes + (ulong)h * head_bytes;
        device const half* sc = reinterpret_cast<device const half*>(q8 + hd);
        device half* out = dst + (ulong)r * kv_dim + (ulong)h * (ulong)hd;
        for (int b = sg; b < blocks; b += nsg) {
            const float d = float(sc[b]);
            out[b * 32 + lane] = half(float(q8[b * 32 + lane]) * d);
        }
    }
}

// ---- Attention -----------------------------------------------------------------------

static inline float4 load4(device const float* p) { return f4(p); }
static inline float4 load4(device const half* p) { return float4(*reinterpret_cast<device const half4*>(p)); }
static inline float4 load4(device const char* p) { return float4(*reinterpret_cast<device const char4*>(p)); }
static inline float load1(device const float* p) { return *p; }
static inline float load1(device const half* p) { return float(*p); }

template <typename KV, int D>
struct KvAccess {
    static inline device const KV* head_ptr(device const KV* base, ulong row, int kv_heads, int kv_head) {
        return base + (row * (ulong)kv_heads + (ulong)kv_head) * (ulong)D;
    }
    static inline float lane_scale(device const KV*, int) { return 1.0f; }
};
template <int D>
struct KvAccess<char, D> {
    static inline device const char* head_ptr(device const char* base, ulong row, int kv_heads, int kv_head) {
        constexpr ulong HEAD_BYTES = (ulong)D + (ulong)(D / 32) * 2;
        return base + (row * (ulong)kv_heads + (ulong)kv_head) * HEAD_BYTES;
    }
    static inline float lane_scale(device const char* head_ptr, int d0) {
        return float(reinterpret_cast<device const half*>(head_ptr + D)[d0 >> 5]);
    }
};

// One threadgroup of 128 per query head and token (see attention_one in syrinx.cu).
template <typename KV>
kernel void attention_one(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, hd = ARG_INT(6), kv_dim = ARG_INT(7), group = ARG_INT(8);
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
            device const KV* k = keys + ring_row(at, ring) * kv_dim + kv;
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
            device const KV* v = values + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + ring_row(start + i, ring) * kv_dim);
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

// One new token's attention split along the context (see attention_part in syrinx.cu).
template <typename KV>
kernel void attention_part(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* partial = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, hd = ARG_INT(6), kv_dim = ARG_INT(7), group = ARG_INT(8);
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
            device const KV* k = keys + ring_row(at, ring) * kv_dim + kv;
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
            device const KV* v = values + kv + d;
            for (int i = 0; i < count; i++) a += ps[i] * load1(v + ring_row(start + i, ring) * kv_dim);
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

// One new token's attention in one dispatch: threadgroups of 256 (8 SIMD groups) for each
// query head (gid.x) and part of the context (gid.y; an equal share of first..len, worked
// out from *pos, so a recorded run fits every length). In a SIMD group each lane takes a
// position, 32 at a time (its score from q in threadgroup memory), and the group keeps its
// own online softmax with SIMD reductions, adding the values up with each lane holding
// D / 32 of the output: no barriers until the 8 groups are merged at the end. With more
// than one part, each leaves its maximum, sum and unnormalized output in `partial`, and the
// last of a head's parts to finish (counted in `counters`, with device-scope fences) merges
// them, so no second dispatch is needed. Exact for any context: up to rounding, the same
// softmax as attention_part + attention_combine.
template <typename KV, int D, int HG>
kernel void attention_decode(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    // HG query heads of one key/value head per threadgroup (gid.x × HG..), each key and
    // value read once for all of them: a lane scores its position against each head's
    // query and adds the value row into each head's output. Built for HG = 1 only: more
    // was slower on the M5 Max (the repeated reads come from the cache).
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    device float* partial = ARG_PTR(float, 10);
    device atomic_uint* counters = ARG_PTR(atomic_uint, 11);
    // A few tokens at once (gid.z: a draft's guesses, a short batch): each its own query
    // and output rows, its own length (causal), and its own share of the scratch.
    {
        const int z = (int)gid.z, nheads = (int)ngrid.x * HG;
        q += (ulong)z * nheads * D;
        out += (ulong)z * nheads * D;
        partial += (ulong)z * nheads * ngrid.y * (2 + D);
        counters += (ulong)z * ngrid.x;
    }
    constexpr int SG = 8, DPL = D / 32;  // SIMD groups; output numbers per lane and head
    threadgroup float qs[HG][D];
    threadgroup float sm[SG], sl[SG];
    threadgroup float so[SG][D];
    threadgroup bool last;
    const int head0 = gid.x * HG, part = gid.y, parts = ngrid.y;
    const int len = *pos + 1 + (int)gid.z;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    const int kv = (head0 / group) * D;
    for (int i = tid; i < HG * D; i += 256) qs[i / D][i % D] = q[head0 * D + i];
    BARRIER;

    float m[HG], l[HG], o[HG][DPL];
    for (int h = 0; h < HG; h++) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int j = 0; j < DPL; j++) o[h][j] = 0.0f;
    }
    for (int base = lo + (int)sg * 32; base < hi; base += SG * 32) {
        const int at = base + (int)lane;
        float s[HG];
        for (int h = 0; h < HG; h++) s[h] = 0.0f;
        if (at < hi) {
            device const KV* k = keys + ring_row(at, ring) * kv_dim + kv;
            for (int d = 0; d < D; d += 4) {
                const float4 k4 = load4(k + d);
                for (int h = 0; h < HG; h++) {
                    const float4 q4 = *reinterpret_cast<threadgroup const float4*>(&qs[h][d]);
                    s[h] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
            }
        }
        float p[HG];
        for (int h = 0; h < HG; h++) {
            const float sc = at < hi ? s[h] * scale : -INFINITY;
            const float new_max = max(m[h], simd_max(sc));
            const float correction = exp(m[h] - new_max);  // 0 on the group's first block
            p[h] = at < hi ? exp(sc - new_max) : 0.0f;
            l[h] = l[h] * correction + simd_sum(p[h]);
            for (int j = 0; j < DPL; j++) o[h][j] *= correction;
            m[h] = new_max;
        }
        const int count = min(32, hi - base);
        device const KV* v = values + kv + lane * DPL;
        for (int i = 0; i < count; i++) {
            device const KV* vr = v + ring_row(base + i, ring) * kv_dim;
            float vv[DPL];
            for (int j = 0; j < DPL; j++) vv[j] = load1(vr + j);
            for (int h = 0; h < HG; h++) {
                const float pi = simd_shuffle(p[h], (ushort)i);
                for (int j = 0; j < DPL; j++) o[h][j] += pi * vv[j];
            }
        }
    }

    // Each head in turn: merge the SIMD groups, then write it out (one part) or leave the
    // part's maximum, sum and output in `partial`.
    for (int h = 0; h < HG; h++) {
        const int head = head0 + h;
        BARRIER;  // the last head is done with sm, sl and so
        if (lane == 0) {
            sm[sg] = m[h];
            sl[sg] = l[h];
        }
        for (int j = 0; j < DPL; j++) so[sg][lane * DPL + j] = o[h][j];
        BARRIER;
        float mx = -INFINITY;
        for (int g = 0; g < SG; g++) mx = max(mx, sm[g]);
        float total = 0.0f;
        for (int g = 0; g < SG; g++)
            if (sm[g] != -INFINITY) total += exp(sm[g] - mx) * sl[g];
        device float* mine = partial + ((ulong)head * parts + part) * (2 + D);
        if (parts > 1 && tid == 0) {
            mine[0] = mx;  // -inf for an empty part
            mine[1] = total;
        }
        for (int d = tid; d < D; d += 256) {
            float acc = 0.0f;
            for (int g = 0; g < SG; g++)
                if (sm[g] != -INFINITY) acc += exp(sm[g] - mx) * so[g][d];
            if (parts == 1)
                out[head * D + d] = acc / total;
            else
                mine[2 + d] = acc;
        }
    }
    if (parts == 1) return;
    // The last part of these heads to get here merges them all.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    BARRIER;
    if (tid == 0) {
        const uint done = atomic_fetch_add_explicit(&counters[gid.x], 1u, memory_order_relaxed);
        last = done == (uint)parts - 1;
        if (last) atomic_store_explicit(&counters[gid.x], 0u, memory_order_relaxed);
    }
    BARRIER;
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    for (int h = 0; h < HG; h++) {
        const int head = head0 + h;
        device const float* all = partial + (ulong)head * parts * (2 + D);
        float pm = -INFINITY;
        for (int i = 0; i < parts; i++) pm = max(pm, all[i * (2 + D)]);
        float pt = 0.0f;
        for (int i = 0; i < parts; i++) {
            const float mi = all[i * (2 + D)];
            if (mi != -INFINITY) pt += exp(mi - pm) * all[i * (2 + D) + 1];
        }
        for (int d = tid; d < D; d += 256) {
            float acc = 0.0f;
            for (int i = 0; i < parts; i++) {
                const float mi = all[i * (2 + D)];
                if (mi != -INFINITY) acc += exp(mi - pm) * all[i * (2 + D) + 2 + d];
            }
            out[head * D + d] = acc / pt;
        }
    }
}

template [[host_name("attention_decode_f32_64_1")]] kernel void attention_decode<float, 64, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f32_128_1")]] kernel void attention_decode<float, 128, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f32_256_1")]] kernel void attention_decode<float, 256, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f32_512_1")]] kernel void attention_decode<float, 512, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f16_64_1")]] kernel void attention_decode<half, 64, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f16_128_1")]] kernel void attention_decode<half, 128, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f16_256_1")]] kernel void attention_decode<half, 256, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_f16_512_1")]] kernel void attention_decode<half, 512, 1>(ARGS, KP, uint, uint);

// One token's attention for wide heads (256, 512): attention_decode with the work laid
// out the other way. A SIMD group takes one position at a time (the 8 groups take every
// 8th), its 32 lanes each holding D / 32 of the head: the key row is read once,
// coalesced, the score is one simd_sum, and the online softmax is the group's (the same
// for every lane). attention_decode gives each lane a position, reading a whole key row
// alone; for wide heads that was the slow part. Parts and merging as attention_decode's.
template <typename KV, int D>
kernel void attention_decode_wide(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF;
    int ring = ARG_INT(5) >> 16;
    const int kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    device float* partial = ARG_PTR(float, 10);
    device atomic_uint* counters = ARG_PTR(atomic_uint, 11);
    // A few tokens at once (gid.z: a draft's guesses, a short batch), or one token for
    // each of several chats (when pos is null and arg 1 is a BatchChat table).
    int len;
    {
        const int z = (int)gid.z, nheads = (int)ngrid.x;
        q += (ulong)z * nheads * D;
        out += (ulong)z * nheads * D;
        partial += (ulong)z * nheads * ngrid.y * (2 + D);
        counters += (ulong)z * ngrid.x;
        if (NONNULL(pos)) {
            len = *pos + 1 + z;
        } else {
            device const BatchChat* bc = reinterpret_cast<device const BatchChat*>(ARG_PTR(const u8, 1)) + z;
            keys = reinterpret_cast<device const KV*>(bc->keys);
            values = reinterpret_cast<device const KV*>(bc->values);
            len = bc->pos + 1;
            ring = bc->ring;
        }
    }
    constexpr int SG = 8, DPL = D / 32;
    threadgroup float sm[SG], sl[SG];
    threadgroup float so[SG][D];
    threadgroup bool last;
    const int head = gid.x, part = gid.y, parts = ngrid.y;
    const int first = window > 0 ? max(0, len - window) : 0;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    const int kv_heads = kv_dim / D, kv_head = head / group, d0 = (int)lane * DPL;
    float qv[DPL];
    const float scale_log2 = scale * 1.4426950408889634f;
    for (int j = 0; j < DPL; j++) qv[j] = q[head * D + d0 + j] * scale_log2;

    float m = -INFINITY, l = 0.0f, o[DPL];
    for (int j = 0; j < DPL; j++) o[j] = 0.0f;
    for (int at = lo + (int)sg; at < hi; at += SG) {
        const ulong r = ring_row(at, ring);
        device const KV* kh = KvAccess<KV, D>::head_ptr(keys, r, kv_heads, kv_head);
        const float ks = KvAccess<KV, D>::lane_scale(kh, d0);
        device const KV* k = kh + d0;
        float dot = 0.0f;
        for (int j = 0; j < DPL; j += 4) {
            const float4 k4 = load4(k + j);
            dot += qv[j] * k4.x + qv[j + 1] * k4.y + qv[j + 2] * k4.z + qv[j + 3] * k4.w;
        }
        const float s = simd_sum(dot * ks);
        const float new_max = max(m, s);
        const float correction = metal::fast::exp2(m - new_max);
        const float p = metal::fast::exp2(s - new_max);
        l = l * correction + p;
        device const KV* vh = KvAccess<KV, D>::head_ptr(values, r, kv_heads, kv_head);
        const float ps = p * KvAccess<KV, D>::lane_scale(vh, d0);
        device const KV* v = vh + d0;
        for (int j = 0; j < DPL; j += 4) {
            const float4 v4 = load4(v + j);
            o[j] = o[j] * correction + ps * v4.x;
            o[j + 1] = o[j + 1] * correction + ps * v4.y;
            o[j + 2] = o[j + 2] * correction + ps * v4.z;
            o[j + 3] = o[j + 3] * correction + ps * v4.w;
        }
        m = new_max;
    }
    if (lane == 0) {
        sm[sg] = m;
        sl[sg] = l;
    }
    for (int j = 0; j < DPL; j++) so[sg][d0 + j] = o[j];
    BARRIER;
    float mx = -INFINITY;
    for (int g = 0; g < SG; g++) mx = max(mx, sm[g]);
    float total = 0.0f;
    for (int g = 0; g < SG; g++)
        if (sm[g] != -INFINITY) total += metal::fast::exp2(sm[g] - mx) * sl[g];
    device float* mine = partial + ((ulong)head * parts + part) * (2 + D);
    if (parts > 1 && tid == 0) {
        mine[0] = mx;
        mine[1] = total;
    }
    for (int d = tid; d < D; d += 256) {
        float acc = 0.0f;
        for (int g = 0; g < SG; g++)
            if (sm[g] != -INFINITY) acc += metal::fast::exp2(sm[g] - mx) * so[g][d];
        if (parts == 1)
            out[head * D + d] = acc / total;
        else
            mine[2 + d] = acc;
    }
    if (parts == 1) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    BARRIER;
    if (tid == 0) {
        const uint done = atomic_fetch_add_explicit(&counters[head], 1u, memory_order_relaxed);
        last = done == (uint)parts - 1;
        if (last) atomic_store_explicit(&counters[head], 0u, memory_order_relaxed);
    }
    BARRIER;
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    device const float* all = partial + (ulong)head * parts * (2 + D);
    float pm = -INFINITY;
    for (int i = 0; i < parts; i++) pm = max(pm, all[i * (2 + D)]);
    float pt = 0.0f;
    for (int i = 0; i < parts; i++) {
        const float mi = all[i * (2 + D)];
        if (mi != -INFINITY) pt += metal::fast::exp2(mi - pm) * all[i * (2 + D) + 1];
    }
    for (int d = tid; d < D; d += 256) {
        float acc = 0.0f;
        for (int i = 0; i < parts; i++) {
            const float mi = all[i * (2 + D)];
            if (mi != -INFINITY) acc += metal::fast::exp2(mi - pm) * all[i * (2 + D) + 2 + d];
        }
        out[head * D + d] = acc / pt;
    }
}

template [[host_name("attention_decode_wide_f32_128")]] kernel void attention_decode_wide<float, 128>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_f32_256")]] kernel void attention_decode_wide<float, 256>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_f32_512")]] kernel void attention_decode_wide<float, 512>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_f16_128")]] kernel void attention_decode_wide<half, 128>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_f16_256")]] kernel void attention_decode_wide<half, 256>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_f16_512")]] kernel void attention_decode_wide<half, 512>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_q8_128")]] kernel void attention_decode_wide<char, 128>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_q8_256")]] kernel void attention_decode_wide<char, 256>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_wide_q8_512")]] kernel void attention_decode_wide<char, 512>(ARGS, KP, uint, uint);

// One token's attention for all the query heads that share a key/value head (gid.x is
// the key/value head, gid.y the part of the context): attention_decode_wide's layout (a
// SIMD group a position, each lane D / 32 of the head), but each key and value row read
// once for the whole group. Reading it once per query head, as attention_decode_wide does,
// was the slow part deep in a context: Qwen 3.8 27B's 24 heads over 4 key/value heads at
// 30K positions took 760 µs a layer, against 280 for the 4 heads alone. A SIMD group
// holds H heads' outputs; with group / H slices of the heads, the 8 groups are that many
// slices times 8 / slices streams of positions (every streams-th). The queries, scaled,
// sit in threadgroup memory. Parts and merging as attention_decode's.
constant constexpr int GROUP_SCRATCH = 7936;
// The most parts attention_decode_group merges (the host keeps to it).
constant constexpr int MAX_GROUP_PARTS = 64;
template <typename KV, int D, int H>
kernel void attention_decode_group(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF;
    int ring = ARG_INT(5) >> 16;
    const int kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    device float* partial = ARG_PTR(float, 10);
    device atomic_uint* counters = ARG_PTR(atomic_uint, 11);
    int len;
    {
        const int z = (int)gid.z, nheads = (int)ngrid.x * group;
        q += (ulong)z * nheads * D;
        out += (ulong)z * nheads * D;
        partial += (ulong)z * nheads * ngrid.y * (2 + D);
        counters += (ulong)z * ngrid.x;
        if (NONNULL(pos)) {
            len = *pos + 1 + z;
        } else {
            device const BatchChat* bc = reinterpret_cast<device const BatchChat*>(ARG_PTR(const u8, 1)) + z;
            keys = reinterpret_cast<device const KV*>(bc->keys);
            values = reinterpret_cast<device const KV*>(bc->values);
            len = bc->pos + 1;
            ring = bc->ring;
        }
    }
    constexpr int SG = 8, DPL = D / 32;
    // The group's queries, then the streams' outputs being merged: the host keeps
    // (group + streams) × D within GROUP_SCRATCH.
    threadgroup float scratch[GROUP_SCRATCH];
    threadgroup float sm[SG], sl[SG];
    threadgroup bool last;
    // SIMD groups past slices × streams (when slices doesn't divide 8) only help merge.
    const int slices = group / H, streams = SG / slices;
    const int slice = (int)sg % slices, stream = (int)sg / slices;
    const bool active = stream < streams;
    const int first = window > 0 ? max(0, len - window) : 0;
    // A short context in fewer parts (at least min_share positions each): the rest of the
    // grid has nothing to do, and the merge has less.
    const int min_share = ARG_INT(12);
    const int head0 = gid.x * group, part = gid.y;
    const int parts = clamp((len - first + min_share - 1) / min_share, 1, (int)ngrid.y);
    if (part >= parts) return;
    const int share = (len - first + parts - 1) / parts;
    const int lo = first + part * share, hi = min(len, lo + share);
    const int kv_heads = kv_dim / D, kv_head = (int)gid.x, d0 = (int)lane * DPL;
    threadgroup float* qs = scratch;
    threadgroup float* so = scratch + group * D;
    const float scale_log2 = scale * 1.4426950408889634f;
    for (int i = tid; i < group * D; i += 256) qs[i] = q[head0 * D + i] * scale_log2;
    BARRIER;

    float m[H], l[H], o[H][DPL];
    for (int h = 0; h < H; h++) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int j = 0; j < DPL; j++) o[h][j] = 0.0f;
    }
    float qv[H][DPL];
    for (int h = 0; h < H; h++)
        for (int j = 0; j < DPL; j++) qv[h][j] = qs[(slice * H + h) * D + d0 + j];
    for (int at = active ? lo + stream : hi; at < hi; at += streams) {
        const ulong r = ring_row(at, ring);
        device const KV* kh = KvAccess<KV, D>::head_ptr(keys, r, kv_heads, kv_head);
        device const KV* vh = KvAccess<KV, D>::head_ptr(values, r, kv_heads, kv_head);
        const float ks = KvAccess<KV, D>::lane_scale(kh, d0);
        const float vs = KvAccess<KV, D>::lane_scale(vh, d0);
        device const KV* k = kh + d0;
        device const KV* v = vh + d0;
        float kk[DPL], vv[DPL];
        for (int j = 0; j < DPL; j += 4) {
            const float4 k4 = load4(k + j);
            kk[j] = k4.x; kk[j + 1] = k4.y; kk[j + 2] = k4.z; kk[j + 3] = k4.w;
        }
        for (int j = 0; j < DPL; j += 4) {
            const float4 v4 = load4(v + j);
            vv[j] = v4.x; vv[j + 1] = v4.y; vv[j + 2] = v4.z; vv[j + 3] = v4.w;
        }
        float s[H];
        for (int h = 0; h < H; h++) {
            float dot = 0.0f;
            for (int j = 0; j < DPL; j++) dot += qv[h][j] * kk[j];
            s[h] = dot * ks;
        }
        for (int h = 0; h < H; h++) s[h] = simd_sum(s[h]);
        for (int h = 0; h < H; h++) {
            const float new_max = max(m[h], s[h]);
            const float correction = metal::fast::exp2(m[h] - new_max);
            const float p = metal::fast::exp2(s[h] - new_max);
            const float ps = p * vs;
            l[h] = l[h] * correction + p;
            for (int j = 0; j < DPL; j++) o[h][j] = o[h][j] * correction + ps * vv[j];
            m[h] = new_max;
        }
    }

    // Each head in turn: merge its streams, then write it out (one part) or leave the
    // part's maximum, sum and output in `partial`.
    for (int h = 0; h < H; h++) {
        for (int sl_i = 0; sl_i < slices; sl_i++) {
            const int head = head0 + sl_i * H + h;
            BARRIER;
            if (active && slice == sl_i) {
                if (lane == 0) {
                    sm[stream] = m[h];
                    sl[stream] = l[h];
                }
                for (int j = 0; j < DPL; j++) so[stream * D + lane * DPL + j] = o[h][j];
            }
            BARRIER;
            float mx = -INFINITY;
            for (int g = 0; g < streams; g++) mx = max(mx, sm[g]);
            float total = 0.0f;
            for (int g = 0; g < streams; g++)
                if (sm[g] != -INFINITY) total += metal::fast::exp2(sm[g] - mx) * sl[g];
            device float* mine = partial + ((ulong)head * parts + part) * (2 + D);
            if (parts > 1 && tid == 0) {
                mine[0] = mx;
                mine[1] = total;
            }
            for (int d = tid; d < D; d += 256) {
                float acc = 0.0f;
                for (int g = 0; g < streams; g++)
                    if (sm[g] != -INFINITY) acc += metal::fast::exp2(sm[g] - mx) * so[g * D + d];
                if (parts == 1)
                    out[head * D + d] = acc / total;
                else
                    mine[2 + d] = acc;
            }
        }
    }
    if (parts == 1) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    BARRIER;
    if (tid == 0) {
        const uint done = atomic_fetch_add_explicit(&counters[gid.x], 1u, memory_order_relaxed);
        last = done == (uint)parts - 1;
        if (last) atomic_store_explicit(&counters[gid.x], 0u, memory_order_relaxed);
    }
    BARRIER;
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    // Merge the parts: a SIMD group a head, its lanes taking the parts to find their
    // weights (kept in the scratch), then the head's numbers.
    for (int hh = (int)sg; hh < group; hh += SG) {
        const int head = head0 + hh;
        device const float* all = partial + (ulong)head * parts * (2 + D);
        threadgroup float* w = scratch + sg * MAX_GROUP_PARTS;
        float pm = -INFINITY;
        for (int i = (int)lane; i < parts; i += 32) pm = max(pm, all[i * (2 + D)]);
        pm = simd_max(pm);
        float pt = 0.0f;
        for (int i = (int)lane; i < parts; i += 32) {
            const float mi = all[i * (2 + D)];
            const float wi = mi != -INFINITY ? metal::fast::exp2(mi - pm) : 0.0f;
            w[i] = wi;
            pt += wi * all[i * (2 + D) + 1];
        }
        pt = simd_sum(pt);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float acc[DPL];
        for (int j = 0; j < DPL; j++) acc[j] = 0.0f;
        for (int i = 0; i < parts; i++) {
            const float wi = w[i];
            device const float* row = all + i * (2 + D) + 2;
            for (int j = 0; j < DPL; j++) acc[j] += wi * row[j * 32 + lane];
        }
        for (int j = 0; j < DPL; j++) out[head * D + j * 32 + lane] = acc[j] / pt;
    }
}



template [[host_name("attention_decode_group_f32_256_1")]] kernel void attention_decode_group<float, 256, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f32_256_2")]] kernel void attention_decode_group<float, 256, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f32_256_3")]] kernel void attention_decode_group<float, 256, 3>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f32_256_4")]] kernel void attention_decode_group<float, 256, 4>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f32_512_1")]] kernel void attention_decode_group<float, 512, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f32_512_2")]] kernel void attention_decode_group<float, 512, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_256_1")]] kernel void attention_decode_group<half, 256, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_256_2")]] kernel void attention_decode_group<half, 256, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_256_3")]] kernel void attention_decode_group<half, 256, 3>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_256_4")]] kernel void attention_decode_group<half, 256, 4>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_512_1")]] kernel void attention_decode_group<half, 512, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_f16_512_2")]] kernel void attention_decode_group<half, 512, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_256_1")]] kernel void attention_decode_group<char, 256, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_256_2")]] kernel void attention_decode_group<char, 256, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_256_3")]] kernel void attention_decode_group<char, 256, 3>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_256_4")]] kernel void attention_decode_group<char, 256, 4>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_512_1")]] kernel void attention_decode_group<char, 512, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_decode_group_q8_512_2")]] kernel void attention_decode_group<char, 512, 2>(ARGS, KP, uint, uint);

// A prompt's attention, tiled (see attention_tiled in syrinx.cu).
template <typename KV, int LANES, int PER, int CHUNK>
kernel void attention_tiled(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const KV* keys = ARG_PTR(const KV, 1);
    device const KV* values = ARG_PTR(const KV, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
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
                kv4 = load4(keys + ring_row(start + r, ring) * kv_dim + kv + d);
                vv4 = load4(values + ring_row(start + r, ring) * kv_dim + kv + d);
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

template [[host_name("attention_tiled_64")]] kernel void attention_tiled<float, 8, 8, 32>(ARGS, KP);
template [[host_name("attention_tiled_128")]] kernel void attention_tiled<float, 8, 16, 32>(ARGS, KP);
template [[host_name("attention_tiled_256")]] kernel void attention_tiled<float, 16, 16, 16>(ARGS, KP);
template [[host_name("attention_tiled_512")]] kernel void attention_tiled<float, 32, 16, 8>(ARGS, KP);
// Over an f16 cache (8-bit mode): the keys and values read as f16, kept as f32 in
// threadgroup memory.
template [[host_name("attention_tiled_h_64")]] kernel void attention_tiled<half, 8, 8, 32>(ARGS, KP);
template [[host_name("attention_tiled_h_128")]] kernel void attention_tiled<half, 8, 16, 32>(ARGS, KP);
template [[host_name("attention_tiled_h_256")]] kernel void attention_tiled<half, 16, 16, 16>(ARGS, KP);
template [[host_name("attention_tiled_h_512")]] kernel void attention_tiled<half, 32, 16, 8>(ARGS, KP);

// A prompt's attention over an f16 cache (the default mode) on the SIMD-group matrix
// units: Metal's counterpart of attention_mma in syrinx.cu. A threadgroup of 128 (4 SIMD
// groups) takes one query head and QT tokens, and the keys and values C positions at a
// time. Q (as f16), the chunk's keys and values, and the scores live in threadgroup
// memory; S = Q Kᵀ and O += P V are 8 × 8 products with f32 sums. The softmax is online,
// as in attention_tiled: each chunk's scores are masked (causal, and the sliding window),
// each row's maximum and sum moved along, and O's rows rescaled by multiplying with a
// diagonal matrix. Each SIMD group holds 16 of O's 8 × 8 blocks (QT × D / 4 numbers).
// Sizes: D = 64 and 128 with 32 tokens, 256 with 16, 512 with 8; C keeps the keys and
// values at 8 KB each.
template <int D, int QT, int C>
kernel void attention_flash(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device const half* keys = ARG_PTR(const half, 1);
    device const half* values = ARG_PTR(const half, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const int tokens = ARG_INT(10);
    constexpr int THREADS = 128, BLOCKS = QT / 8 * (D / 8) / 4;  // O's blocks per SIMD group
    // The softmax's threads per row (neighbours in a SIMD group), at most one per column.
    constexpr int ROW_THREADS = THREADS / QT < C ? THREADS / QT : C, PER_THREAD = C / ROW_THREADS;
    // Q, the keys and the values; O's results go over them at the end (QT × D f32).
    threadgroup half qkv[QT * D + 2 * C * D];
    threadgroup half* sq = qkv;
    threadgroup half* sk = qkv + QT * D;
    threadgroup half* sv = sk + C * D;
    threadgroup float ss[QT * C];   // scores
    threadgroup half sp[QT * C];    // probabilities
    threadgroup float row_max[QT], row_sum[QT];
    threadgroup float diag[QT / 8][64];  // each row block's rescaling, as diagonal matrices
    const int head = gid.x, heads = ngrid.x;
    const int t0 = gid.y * QT, p0 = *pos;
    const int kv = (head / group) * D;
    const int last_row = p0 + min(t0 + QT, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    for (int i = tid; i < QT * D; i += THREADS) {
        const int r = i / D, d = i % D;
        sq[i] = t0 + r < tokens ? half(q[((ulong)(t0 + r) * heads + head) * D + d]) : half(0.0f);
    }
    for (int i = tid; i < QT / 8 * 64; i += THREADS) diag[i / 64][i % 64] = 0.0f;
    if ((int)tid < QT) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    simdgroup_float8x8 o[BLOCKS];
    for (int b = 0; b < BLOCKS; b++) o[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // The thread's row for the softmax, and its columns.
    const int sr = tid / ROW_THREADS, sc0 = (tid % ROW_THREADS) * PER_THREAD;
    const int s_at = p0 + t0 + sr;  // that row's position
    const bool s_live = t0 + sr < tokens;
    const int s_first = window > 0 ? max(0, s_at + 1 - window) : 0;

    for (int start = lo; start <= last_row; start += C) {
        const int count = min(C, last_row + 1 - start);
        BARRIER;  // done with the last chunk's keys, values and probabilities
        for (int i = tid; i < C * D / 4; i += THREADS) {
            const int r = (i * 4) / D, d = (i * 4) % D;
            half4 k4 = half4(0.0h), v4 = half4(0.0h);
            if (r < count) {
                const ulong at = ring_row(start + r, ring) * kv_dim + kv + d;
                k4 = *reinterpret_cast<device const half4*>(keys + at);
                v4 = *reinterpret_cast<device const half4*>(values + at);
            }
            *reinterpret_cast<threadgroup half4*>(sk + r * D + d) = k4;
            *reinterpret_cast<threadgroup half4*>(sv + r * D + d) = v4;
        }
        BARRIER;
        // S = Q Kᵀ, its 8 × 8 tiles shared out among the SIMD groups.
        for (int tile = sg; tile < QT / 8 * (C / 8); tile += 4) {
            const int rb = tile / (C / 8), kb = tile % (C / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d = 0; d < D; d += 8) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, sq + rb * 8 * D + d, D);
                simdgroup_load(b, sk + kb * 8 * D + d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, ss + rb * 8 * C + kb * 8, C);
        }
        BARRIER;
        // The online softmax: ROW_THREADS neighbouring threads to a row (whole SIMD groups
        // sit out when there are more threads than that).
        if ((int)tid < QT * ROW_THREADS) {
            float s[PER_THREAD];
            float mx = -INFINITY;
            for (int j = 0; j < PER_THREAD; j++) {
                const int c = sc0 + j, kpos = start + c;
                const bool seen = s_live && c < count && kpos <= s_at && kpos >= s_first;
                s[j] = seen ? ss[sr * C + c] * scale : -INFINITY;
                mx = max(mx, s[j]);
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o2));
            const float old_max = row_max[sr];
            const float new_max = max(old_max, mx);
            const float correction = new_max == -INFINITY ? 1.0f : exp(old_max - new_max);
            float sum = 0.0f;
            for (int j = 0; j < PER_THREAD; j++) {
                const float p = s[j] == -INFINITY ? 0.0f : exp(s[j] - new_max);
                sp[sr * C + sc0 + j] = half(p);
                sum += p;
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) sum += simd_shuffle_xor(sum, (ushort)o2);
            if (tid % ROW_THREADS == 0) {
                row_max[sr] = new_max;
                row_sum[sr] = row_sum[sr] * correction + sum;
                diag[sr / 8][(sr % 8) * 9] = correction;
            }
        }
        BARRIER;
        // O = diag(correction) O + P V
        for (int b = 0; b < BLOCKS; b++) {
            const int block = sg * BLOCKS + b, rb = block / (D / 8), cb = block % (D / 8);
            simdgroup_float8x8 dg;
            simdgroup_load(dg, &diag[rb][0], 8);
            simdgroup_multiply(o[b], dg, o[b]);
            for (int kc = 0; kc < C; kc += 8) {
                simdgroup_half8x8 p, v;
                simdgroup_load(p, sp + rb * 8 * C + kc, C);
                simdgroup_load(v, sv + kc * D + cb * 8, D);
                simdgroup_multiply_accumulate(o[b], p, v, o[b]);
            }
        }
    }
    BARRIER;  // done with Q, the keys and the values: O goes in their place
    threadgroup float* so = reinterpret_cast<threadgroup float*>(qkv);
    for (int b = 0; b < BLOCKS; b++) {
        const int block = sg * BLOCKS + b, rb = block / (D / 8), cb = block % (D / 8);
        simdgroup_store(o[b], so + rb * 8 * D + cb * 8, D);
    }
    BARRIER;
    for (int i = tid; i < QT * D; i += THREADS) {
        const int r = i / D, d = i % D;
        if (t0 + r < tokens) out[((ulong)(t0 + r) * heads + head) * D + d] = so[i] / row_sum[r];
    }
}

template [[host_name("attention_flash_64")]] kernel void attention_flash<64, 32, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_flash_128")]] kernel void attention_flash<128, 32, 16>(ARGS, KP, uint, uint);
template [[host_name("attention_flash_256")]] kernel void attention_flash<256, 16, 8>(ARGS, KP, uint, uint);
template [[host_name("attention_flash_512")]] kernel void attention_flash<512, 8, 8>(ARGS, KP, uint, uint);

// attention_flash with bigger query tiles: the chunk's keys and then its values share one
// buffer (the scores are worked out before the values come in), wide heads get 8 SIMD
// groups, and O's results are staged over everything at the end, so twice the tokens fit
// a threadgroup and each key and value is read half as often (O is written out in P passes
// of D / P columns, to fit). Same arithmetic.
template <int D, int QT, int C, int SG, int P>
kernel void attention_flash2(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device const half* keys = ARG_PTR(const half, 1);
    device const half* values = ARG_PTR(const half, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const int tokens = ARG_INT(10);
    constexpr int THREADS = 32 * SG, BLOCKS = QT / 8 * (D / 8) / SG;
    constexpr int ROW_THREADS = THREADS / QT < C ? THREADS / QT : C, PER_THREAD = C / ROW_THREADS;
    // [Q | keys or values | scores | probabilities], O's results over all of it at the end.
    constexpr int BYTES_IN = QT * D * 2 + C * D * 2 + QT * C * 4 + QT * C * 2;
    constexpr int DS = D / P, BYTES = BYTES_IN > QT * DS * 4 ? BYTES_IN : QT * DS * 4;
    threadgroup float4 mem[BYTES / 16];
    threadgroup half* sq = reinterpret_cast<threadgroup half*>(mem);
    threadgroup half* skv = sq + QT * D;
    threadgroup float* ss = reinterpret_cast<threadgroup float*>(skv + C * D);
    threadgroup half* sp = reinterpret_cast<threadgroup half*>(ss + QT * C);
    threadgroup float row_max[QT], row_sum[QT];
    threadgroup float diag[QT / 8][64];
    const int head = gid.x, heads = ngrid.x;
    const int t0 = gid.y * QT, p0 = *pos;
    const int kv = (head / group) * D;
    const int last_row = p0 + min(t0 + QT, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    for (int i = tid; i < QT * D; i += THREADS) {
        const int r = i / D, d = i % D;
        sq[i] = t0 + r < tokens ? half(q[((ulong)(t0 + r) * heads + head) * D + d]) : half(0.0f);
    }
    for (int i = tid; i < QT / 8 * 64; i += THREADS) diag[i / 64][i % 64] = 0.0f;
    if ((int)tid < QT) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    simdgroup_float8x8 o[BLOCKS];
    for (int b = 0; b < BLOCKS; b++) o[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int sr = tid / ROW_THREADS, sc0 = (tid % ROW_THREADS) * PER_THREAD;
    const int s_at = p0 + t0 + sr;
    const bool s_live = t0 + sr < tokens;
    const int s_first = window > 0 ? max(0, s_at + 1 - window) : 0;

    for (int start = lo; start <= last_row; start += C) {
        const int count = min(C, last_row + 1 - start);
        BARRIER;  // done with the last chunk's values and probabilities
        for (int i = tid; i < C * D / 4; i += THREADS) {
            const int r = (i * 4) / D, d = (i * 4) % D;
            half4 k4 = half4(0.0h);
            if (r < count) k4 = *reinterpret_cast<device const half4*>(keys + ring_row(start + r, ring) * kv_dim + kv + d);
            *reinterpret_cast<threadgroup half4*>(skv + r * D + d) = k4;
        }
        BARRIER;
        for (int tile = sg; tile < QT / 8 * (C / 8); tile += SG) {
            const int rb = tile / (C / 8), kb = tile % (C / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d = 0; d < D; d += 8) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, sq + rb * 8 * D + d, D);
                simdgroup_load(b, skv + kb * 8 * D + d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, ss + rb * 8 * C + kb * 8, C);
        }
        BARRIER;  // done with the keys: the values go in their place
        for (int i = tid; i < C * D / 4; i += THREADS) {
            const int r = (i * 4) / D, d = (i * 4) % D;
            half4 v4 = half4(0.0h);
            if (r < count) v4 = *reinterpret_cast<device const half4*>(values + ring_row(start + r, ring) * kv_dim + kv + d);
            *reinterpret_cast<threadgroup half4*>(skv + r * D + d) = v4;
        }
        if ((int)tid < QT * ROW_THREADS) {
            float sv[PER_THREAD];
            float mx = -INFINITY;
            for (int j = 0; j < PER_THREAD; j++) {
                const int c = sc0 + j, kpos = start + c;
                const bool seen = s_live && c < count && kpos <= s_at && kpos >= s_first;
                sv[j] = seen ? ss[sr * C + c] * scale : -INFINITY;
                mx = max(mx, sv[j]);
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o2));
            const float old_max = row_max[sr];
            const float new_max = max(old_max, mx);
            const float correction = new_max == -INFINITY ? 1.0f : exp(old_max - new_max);
            float sum = 0.0f;
            for (int j = 0; j < PER_THREAD; j++) {
                const float pr = sv[j] == -INFINITY ? 0.0f : exp(sv[j] - new_max);
                sp[sr * C + sc0 + j] = half(pr);
                sum += pr;
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) sum += simd_shuffle_xor(sum, (ushort)o2);
            if (tid % ROW_THREADS == 0) {
                row_max[sr] = new_max;
                row_sum[sr] = row_sum[sr] * correction + sum;
                diag[sr / 8][(sr % 8) * 9] = correction;
            }
        }
        BARRIER;
        for (int b = 0; b < BLOCKS; b++) {
            const int block = sg * BLOCKS + b, rb = block / (D / 8), cb = block % (D / 8);
            simdgroup_float8x8 dg;
            simdgroup_load(dg, &diag[rb][0], 8);
            simdgroup_multiply(o[b], dg, o[b]);
            for (int kc = 0; kc < C; kc += 8) {
                simdgroup_half8x8 pm, vm;
                simdgroup_load(pm, sp + rb * 8 * C + kc, C);
                simdgroup_load(vm, skv + kc * D + cb * 8, D);
                simdgroup_multiply_accumulate(o[b], pm, vm, o[b]);
            }
        }
    }
    threadgroup float* so = reinterpret_cast<threadgroup float*>(mem);
    for (int pass = 0; pass < P; pass++) {
        BARRIER;  // done with the threadgroup memory (the last pass's results, or the loop's)
        for (int b = 0; b < BLOCKS; b++) {
            const int block = sg * BLOCKS + b, rb = block / (D / 8), cb = block % (D / 8);
            if (cb * 8 / DS == pass) simdgroup_store(o[b], so + rb * 8 * DS + cb * 8 - pass * DS, DS);
        }
        BARRIER;
        for (int i = tid; i < QT * DS; i += THREADS) {
            const int r = i / DS, d = i % DS;
            if (t0 + r < tokens) out[((ulong)(t0 + r) * heads + head) * D + pass * DS + d] = so[i] / row_sum[r];
        }
    }
}

template [[host_name("attention_flash2_64")]] kernel void attention_flash2<64, 64, 32, 4, 1>(ARGS, KP, uint, uint);
template [[host_name("attention_flash2_128")]] kernel void attention_flash2<128, 64, 16, 8, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_flash2_256")]] kernel void attention_flash2<256, 32, 8, 8, 2>(ARGS, KP, uint, uint);
template [[host_name("attention_flash2_512")]] kernel void attention_flash2<512, 16, 8, 8, 2>(ARGS, KP, uint, uint);

// ---- Element-wise steps --------------------------------------------------------------

// ---- A mixture of experts (Gemma 4 26B-A4B) ------------------------------------------
//
// Each token goes through the `used` experts its router scores highest. A slot is one
// token's turn with one expert (slot = token × used + k), and `idx[slot]` is the expert:
// the experts' matrices lie one after another in `w`, `rows` rows each.

// The `used` highest of a token's `experts` scores (the lowest index on a tie), and each
// one's share: a softmax over those scores, times the expert's own scale. One threadgroup
// a token, its threads sharing out the experts: each finds its experts' places among the
// scores (how many beat it), and the first `used` places are the picks. (One thread
// picking them a best at a time took 0.2 ms a token, a third of the GPU's time writing.)
#define MOE_MAX_USED 32
kernel void moe_pick(ARGS, KP) {
    device const float* scores = ARG_PTR(const float, 0);
    device const float* scale = ARG_PTR(const float, 1);
    device int* idx = ARG_PTR(int, 2);
    device float* share = ARG_PTR(float, 3);
    const int experts = ARG_INT(4), used = ARG_INT(5);
    threadgroup int tix[MOE_MAX_USED];
    threadgroup float tsh[MOE_MAX_USED];
    device const float* s = scores + (ulong)gid.x * experts;
    // Scores that can't be ordered (a NaN) would leave places empty: a valid pick first.
    for (int k = tid; k < used; k += tpg.x) {
        tix[k] = k;
        tsh[k] = 0.0f;
    }
    BARRIER;
    for (int e = tid; e < experts; e += tpg.x) {
        const float v = s[e];
        int place = 0;
        for (int j = 0; j < experts; j++) {
            const float o = s[j];
            place += (o > v || (o == v && j < e)) ? 1 : 0;
        }
        if (place < used) {
            tix[place] = e;
            tsh[place] = v;
        }
    }
    BARRIER;
    if (tid != 0) return;
    device int* ix = idx + (ulong)gid.x * used;
    device float* sh = share + (ulong)gid.x * used;
    float e[MOE_MAX_USED];
    const float top = tsh[0];
    float total = 0.0f;
    for (int k = 0; k < used; k++) {
        e[k] = exp(tsh[k] - top);
        total += e[k];
    }
    for (int k = 0; k < used; k++) {
        ix[k] = tix[k];
        sh[k] = e[k] / total * scale[tix[k]];
    }
}

// y = W_e x for a slot's expert e, a threadgroup row (gid.y) a slot: the matrix-vector
// kernels above, with the matrix found through `idx`. `x` is the token's (`x_per_slot`
// 0) or the slot's own (1), `x_stride` numbers apart; `y` is the slot's, `y_stride` apart.
#define MOE_ARGS(ROW_BYTES)                                                             \
    const int rows = ARG_INT(3), cols = ARG_INT(4);                                     \
    device const int* idx = ARG_PTR(const int, 5);                                      \
    const int used = ARG_INT(6), x_stride = ARG_INT(7), x_per_slot = ARG_INT(8);        \
    const int y_stride = ARG_INT(9);                                                    \
    const int slot = gid.y;                                                             \
    const ulong row_bytes = (ROW_BYTES);                                                \
    device const u8* w = ARG_PTR(const u8, 0) + (ulong)idx[slot] * (ulong)rows * row_bytes; \
    device const float* x = ARG_PTR(const float, 1) + (ulong)(x_per_slot ? slot : slot / used) * x_stride; \
    device float* y = ARG_PTR(float, 2) + (ulong)slot * y_stride;

kernel void moe_matvec_q4_0(ARGS, KP) {
    MOE_ARGS(q4_0_row_bytes(ARG_INT(4)))
    MATVEC_ROWS
    const int blocks = cols / 32;
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

kernel void moe_matvec_iq4_nl(ARGS, KP) {
    MOE_ARGS(q4_0_row_bytes(ARG_INT(4)))
    MATVEC_ROWS
    const int blocks = cols / 32;
    const int sub = lane >> 1, j = (lane & 1) * 8;
    for (int b = sub; b < blocks; b += 16) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + b * 16 + j);
            sum[r] += q4_0_scale(row, blocks, b) * (iq4nl_dot8(q, 0, lo0, lo1) + iq4nl_dot8(q, 4, hi0, hi1));
        }
    }
    MATVEC_STORE
}

kernel void moe_matvec_q4_k(ARGS, KP) {
    MOE_ARGS((ulong)(ARG_INT(4) / 256) * 144)
    MATVEC_ROWS
    const int blocks = cols / 256;
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

kernel void moe_matvec_q5_k(ARGS, KP) {
    MOE_ARGS((ulong)(ARG_INT(4) / 256) * 176)
    MATVEC_ROWS
    const int blocks = cols / 256;
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

kernel void moe_matvec_q8_0(ARGS, KP) {
    MOE_ARGS(q8_0_row_bytes(ARG_INT(4)))
    MATVEC_ROWS
    const int blocks = cols / 32;
    const int sub = lane >> 2, j = (lane & 3) * 8;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* row = w + (ulong)(row0 + r) * row_bytes;
            const uint2 q = *reinterpret_cast<device const uint2*>(row + (ulong)b * 32 + j);
            sum[r] += q8_0_scale(row, blocks, b) * q8_0_dot8(q, x0, x1);
        }
    }
    MATVEC_STORE
}

// Fused Q8_0 MoE gate_up matvec + SiTU activation: 4 rows/TG, 1 SIMD group/row, zero barrier.
kernel void ink_moe_gate_up_situ_q8_0(ARGS, KP) {
    const int width = ARG_INT(3), cols = ARG_INT(4);
    device const int* idx = ARG_PTR(const int, 5);
    const int used = ARG_INT(6), x_stride = ARG_INT(7), y_stride = ARG_INT(8);
    const float beta = ARG_F32(9), lb = ARG_F32(10);
    const int slot = gid.y;
    const int blocks = cols / 32;
    const ulong row_bytes = q8_0_row_bytes(cols);
    device const u8* w = ARG_PTR(const u8, 0) + (ulong)idx[slot] * (ulong)(2 * width) * row_bytes;
    device const float* x = ARG_PTR(const float, 1) + (ulong)(slot / used) * x_stride;
    device float* y = ARG_PTR(float, 2) + (ulong)slot * y_stride;
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int row = (int)gid.x * 4 + sgi;
    if (row >= width) return;
    const int sub = lane >> 2, j = (lane & 3) * 8;
    device const u8* wg = w + (ulong)row * row_bytes;
    device const u8* wu = w + (ulong)(width + row) * row_bytes;
    float sg = 0.0f, su = 0.0f;
    for (int b = sub; b < blocks; b += 8) {
        device const float* xb = x + b * 32 + j;
        const float4 x0 = f4(xb), x1 = f4(xb + 4);
        const uint2 qg = *reinterpret_cast<device const uint2*>(wg + (ulong)b * 32 + j);
        const uint2 qu = *reinterpret_cast<device const uint2*>(wu + (ulong)b * 32 + j);
        sg += q8_0_scale(wg, blocks, b) * q8_0_dot8(qg, x0, x1);
        su += q8_0_scale(wu, blocks, b) * q8_0_dot8(qu, x0, x1);
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (lane == 0) {
        y[row] = beta * tanh(sg / beta) / (1.0f + exp(-sg)) * lb * tanh(su / lb);
    }
}

// Single-token fused Q8_0 MoE down projection + weighted expert sum + ffn_post_conv + residual add:
// Each threadgroup handles 2 output rows (`row0..row0+1`) with 8 SIMD groups (1 per active expert,
// `used <= 8`), eliminating both `moe_sum` and `ink_conv` dispatches and all intermediate writes.
#define DOWN8_ROWS 2
kernel void ink_moe_down_sum_conv_q8_0(ARGS, KP) {
    device const u8* w_all = ARG_PTR(const u8, 0);
    device const float* act_all = ARG_PTR(const float, 1);
    device const int* idx = ARG_PTR(const int, 2);
    device const float* share = ARG_PTR(const float, 3);
    device float* history = ARG_PTR(float, 4);
    device const float* conv_w = ARG_PTR(const float, 5);
    device float* x_out = ARG_PTR(float, 6);
    const int dim = ARG_INT(7), width = ARG_INT(8), used = ARG_INT(9);
    const int channels = ARG_INT(10), conv = ARG_INT(11);
    const int row0 = (int)gid.x * DOWN8_ROWS;
    if (row0 >= dim) return;
    const int nrows = min(DOWN8_ROWS, dim - row0);
    const int e = (int)(tid >> 5), lane = (int)(tid & 31);
    threadgroup float parts[8][2];
    if (e < used) {
        const int blocks = width / 32;
        const ulong row_bytes = q8_0_row_bytes(width);
        device const u8* w = w_all + (ulong)idx[e] * (ulong)dim * row_bytes;
        device const float* act = act_all + (ulong)e * width;
        device const u8* r0 = w + (ulong)row0 * row_bytes;
        device const u8* r1 = r0 + row_bytes;
        float s0 = 0.0f, s1 = 0.0f;
        const int sub = lane >> 3, j = (lane & 7) * 4;
        for (int b = sub; b < blocks; b += 4) {
            const float4 xv = f4(act + b * 32 + j);
            const char4 q0 = as_type<char4>(u32_at4(r0 + (ulong)b * 32 + j));
            s0 += q8_0_scale(r0, blocks, b) * dot(float4(q0), xv);
            if (nrows > 1) {
                const char4 q1 = as_type<char4>(u32_at4(r1 + (ulong)b * 32 + j));
                s1 += q8_0_scale(r1, blocks, b) * dot(float4(q1), xv);
            }
        }
        const float sh = share[e];
        s0 = simd_sum(s0) * sh;
        s1 = simd_sum(s1) * sh;
        if (lane == 0) {
            parts[e][0] = s0;
            parts[e][1] = s1;
        }
    }
    BARRIER;
    if ((int)tid < nrows) {
        const int r = row0 + (int)tid;
        float val = 0.0f;
        for (int k = 0; k < used; k++) val += parts[k][tid];
        // Causal post-conv + residual add into x_out[r], shifting history.
        float window[8];
        for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + r];
        window[conv - 1] = val;
        float sum = val;
        device const float* cw = conv_w + (ulong)r * conv;
        for (int k = 0; k < conv; k++) sum += cw[k] * window[k];
        x_out[r] += sum / sqrt((float)(conv + 1));
        for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + r] = window[k + 1];
    }
}

// Fused IQ4_NL MoE gate_up matvec + SiTU activation (4 rows/TG, 1 SIMD group/row, zero barrier).
kernel void ink_moe_gate_up_situ_iq4_nl(ARGS, KP) {
    const int width = ARG_INT(3), cols = ARG_INT(4);
    device const int* idx = ARG_PTR(const int, 5);
    const int used = ARG_INT(6), x_stride = ARG_INT(7), y_stride = ARG_INT(8);
    const float beta = ARG_F32(9), lb = ARG_F32(10);
    const int slot = gid.y;
    const int blocks = cols / 32;
    const ulong row_bytes = q4_0_row_bytes(cols);
    device const u8* w = ARG_PTR(const u8, 0) + (ulong)idx[slot] * (ulong)(2 * width) * row_bytes;
    device const float* x = ARG_PTR(const float, 1) + (ulong)(slot / used) * x_stride;
    device float* y = ARG_PTR(float, 2) + (ulong)slot * y_stride;
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int row = (int)gid.x * 4 + sgi;
    if (row >= width) return;
    const int sub = lane >> 1, j = (lane & 1) * 8;
    device const u8* wg = w + (ulong)row * row_bytes;
    device const u8* wu = w + (ulong)(width + row) * row_bytes;
    float sg = 0.0f, su = 0.0f;
    for (int b = sub; b < blocks; b += 16) {
        device const float* xb = x + b * 32 + j;
        const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
        const uint2 qg = *reinterpret_cast<device const uint2*>(wg + b * 16 + j);
        const uint2 qu = *reinterpret_cast<device const uint2*>(wu + b * 16 + j);
        sg += q4_0_scale(wg, blocks, b) * (iq4nl_dot8(qg, 0, lo0, lo1) + iq4nl_dot8(qg, 4, hi0, hi1));
        su += q4_0_scale(wu, blocks, b) * (iq4nl_dot8(qu, 0, lo0, lo1) + iq4nl_dot8(qu, 4, hi0, hi1));
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (lane == 0) {
        y[row] = beta * tanh(sg / beta) / (1.0f + exp(-sg)) * lb * tanh(su / lb);
    }
}

// Single-token fused IQ4_NL MoE down projection + weighted expert sum + ffn_post_conv + residual add (2 rows/TG).
kernel void ink_moe_down_sum_conv_iq4_nl(ARGS, KP) {
    device const u8* w_all = ARG_PTR(const u8, 0);
    device const float* act_all = ARG_PTR(const float, 1);
    device const int* idx = ARG_PTR(const int, 2);
    device const float* share = ARG_PTR(const float, 3);
    device float* history = ARG_PTR(float, 4);
    device const float* conv_w = ARG_PTR(const float, 5);
    device float* x_out = ARG_PTR(float, 6);
    const int dim = ARG_INT(7), width = ARG_INT(8), used = ARG_INT(9);
    const int channels = ARG_INT(10), conv = ARG_INT(11);
    const int row0 = (int)gid.x * 2;
    if (row0 >= dim) return;
    const int nrows = min(2, dim - row0);
    const int e = (int)(tid >> 5), lane = (int)(tid & 31);
    threadgroup float parts[8][2];
    if (e < used) {
        const int blocks = width / 32;
        const ulong row_bytes = q4_0_row_bytes(width);
        device const u8* w = w_all + (ulong)idx[e] * (ulong)dim * row_bytes;
        device const float* act = act_all + (ulong)e * width;
        float s0 = 0.0f, s1 = 0.0f;
        const int sub = lane >> 1, j = (lane & 1) * 8;
        device const u8* r0 = w + (ulong)row0 * row_bytes;
        device const u8* r1 = r0 + row_bytes;
        for (int b = sub; b < blocks; b += 16) {
            device const float* xb = act + b * 32 + j;
            const float4 lo0 = f4(xb), lo1 = f4(xb + 4), hi0 = f4(xb + 16), hi1 = f4(xb + 20);
            const uint2 q0 = *reinterpret_cast<device const uint2*>(r0 + b * 16 + j);
            s0 += q4_0_scale(r0, blocks, b) * (iq4nl_dot8(q0, 0, lo0, lo1) + iq4nl_dot8(q0, 4, hi0, hi1));
            if (nrows > 1) {
                const uint2 q1 = *reinterpret_cast<device const uint2*>(r1 + b * 16 + j);
                s1 += q4_0_scale(r1, blocks, b) * (iq4nl_dot8(q1, 0, lo0, lo1) + iq4nl_dot8(q1, 4, hi0, hi1));
            }
        }
        const float sh = share[e];
        s0 = simd_sum(s0) * sh;
        s1 = simd_sum(s1) * sh;
        if (lane == 0) {
            parts[e][0] = s0;
            parts[e][1] = s1;
        }
    }
    BARRIER;
    if ((int)tid < nrows) {
        const int r = row0 + (int)tid;
        float val = 0.0f;
        for (int k = 0; k < used; k++) val += parts[k][tid];
        float window[8];
        for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + r];
        window[conv - 1] = val;
        float sum = val;
        device const float* cw = conv_w + (ulong)r * conv;
        for (int k = 0; k < conv; k++) sum += cw[k] * window[k];
        x_out[r] += sum / sqrt((float)(conv + 1));
        for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + r] = window[k + 1];
    }
}

kernel void moe_matvec_q6_k(ARGS, KP) {
    MOE_ARGS((ulong)(ARG_INT(4) / 256) * Q6K_BYTES)
    MATVEC_ROWS
    const int blocks = cols / 256;
    const int sub = lane >> 4, hf = (lane & 15) >> 3, l = (lane & 7) * 4;
    for (int b = sub; b < blocks; b += 2) {
        device const float* xb = x + b * 256 + hf * 128 + l;
        float4 xv[4];
        for (int k = 0; k < 4; k++) xv[k] = f4(xb + k * 32);
        for (int r = 0; r < MV_ROWS; r++) {
            if (r >= nrows) break;
            device const u8* block = w + (row0 + r) * row_bytes + b * Q6K_BYTES;
            const float d = half_at(block + 208);
            const u32 ql0 = u32_at4(block + hf * 64 + l), ql1 = u32_at4(block + hf * 64 + 32 + l);
            const u32 qh = u32_at4(block + 128 + hf * 32 + l);
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

// A slot's 2 × width numbers are its expert's gate rows' products, then its up rows':
// the first half becomes GELU(gate) × up, the expert's activation.
kernel void moe_geglu(ARGS, KP) {
    device float* y = ARG_PTR(float, 0);
    const int width = ARG_INT(1), total = ARG_INT(2);
    const int i = gid.x * tpg.x + tid;
    if (i >= total) return;
    device float* slot = y + (ulong)(i / width) * (2 * width);
    const int j = i % width;
    const float x = slot[j];
    slot[j] = 0.5f * x * (1.0f + tanh(0.7978846f * x * (1.0f + 0.044715f * x * x))) * slot[width + j];
}

// out[token] = Σ over the token's slots of share × the slot's `n` numbers: the slots'
// numbers in slot order, or (`sorted`) in moe_group's order, each slot's at its `place`.
kernel void moe_sum(ARGS, KP) {
    device float* out = ARG_PTR(float, 0);
    device const float* parts = ARG_PTR(const float, 1);
    device const float* share = ARG_PTR(const float, 2);
    const int used = ARG_INT(3), n = ARG_INT(4), total = ARG_INT(5);
    device const int* place = ARG_PTR(const int, 6);
    const int sorted = ARG_INT(7);
    const int i = gid.x * tpg.x + tid;
    if (i >= total) return;
    const int token = i / n, j = i % n;
    float sum = 0.0f;
    for (int k = 0; k < used; k++) {
        const ulong slot = (ulong)token * used + k;
        const ulong at = sorted ? (ulong)place[slot] : slot;
        sum += share[slot] * parts[at * n + j];
    }
    out[i] = sum;
}

// Vectorized moe_sum reading half4 expert outputs (4 elements per thread, n multiple of 4).
kernel void moe_sum_half(ARGS, KP) {
    device float4* out = ARG_PTR(float4, 0);
    device const half4* parts = ARG_PTR(const half4, 1);
    device const float* share = ARG_PTR(const float, 2);
    const int used = ARG_INT(3), n4 = ARG_INT(4), total4 = ARG_INT(5);
    device const int* place = ARG_PTR(const int, 6);
    const int sorted = ARG_INT(7);
    const int i = gid.x * tpg.x + tid;
    if (i >= total4) return;
    const int token = i / n4, j = i % n4;
    float4 sum = float4(0.0f);
    for (int k = 0; k < used; k++) {
        const ulong slot = (ulong)token * used + k;
        const ulong at = sorted ? (ulong)place[slot] : slot;
        sum += share[slot] * float4(parts[at * n4 + j]);
    }
    out[i] = sum;
}

// The slots' inputs in moe_group's order, rounded to f16, for moe_matmul_tc: row `at` of
// `out` (`cols` numbers) is the input of the slot `order[at]`, its token's row of `x`
// (`by_token`), or is row `at` of `x` itself (rows already in that order), `x_stride`
// numbers apart. 4 numbers a thread (`cols` a multiple of 4).
kernel void moe_gather_half(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device half4* out = ARG_PTR(half4, 1);
    device const int* order = ARG_PTR(const int, 2);
    const int used = ARG_INT(3), cols4 = ARG_INT(4) / 4, x_stride = ARG_INT(5), by_token = ARG_INT(6);
    const int total = ARG_INT(7);
    const int i = gid.x * tpg.x + tid;
    if (i >= total) return;
    const int at = i / cols4, j = i % cols4;
    const ulong row = by_token ? (ulong)(order[at] / used) : (ulong)at;
    out[i] = half4(f4(x + row * x_stride + 4 * j));
}

// Fused moe_geglu + f32->f16 conversion for sorted slots into moe_matmul_tc's half
// input buffer: reads [gate | up] per slot as float4 and writes half4 directly.
kernel void moe_geglu_half(ARGS, KP) {
    device const float* gate_up = ARG_PTR(const float, 0);
    device half4* out = ARG_PTR(half4, 1);
    const int width4 = ARG_INT(2) / 4, total4 = ARG_INT(3);
    const int i = gid.x * tpg.x + tid;
    if (i >= total4) return;
    const int at = i / width4, j = i % width4;
    device const float* slot = gate_up + (ulong)at * (ulong)(width4 * 8);
    const float4 x = f4(slot + 4 * j);
    const float4 u = f4(slot + width4 * 4 + 4 * j);
    out[i] = half4(0.5f * x * (1.0f + tanh(0.7978846f * x * (1.0f + 0.044715f * x * x))) * u);
}

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

// Fused SiLU/GELU activation + f32->f16 conversion for matmul_tc: reads gate and up
// (contiguous, n4 = total / 4 float4s) and writes half4 directly into Q8's half buffer,
// skipping both the f32 store to gate and the subsequent to_half pass.
kernel void swiglu_half(ARGS, KP) {
    device const float4* gate = ARG_PTR(const float4, 0);
    device const float4* up = ARG_PTR(const float4, 1);
    device half4* out = ARG_PTR(half4, 2);
    const int n4 = ARG_INT(3);
    const int i = gid.x * tpg.x + tid;
    if (i >= n4) return;
    const float4 g = gate[i];
    const float4 u = up[i];
    out[i] = half4(g / (1.0f + metal::fast::exp2(-g * 1.4426950408889634f)) * u);
}

kernel void geglu_half(ARGS, KP) {
    device const float4* gate = ARG_PTR(const float4, 0);
    device const float4* up = ARG_PTR(const float4, 1);
    device half4* out = ARG_PTR(half4, 2);
    const int n4 = ARG_INT(3);
    const int i = gid.x * tpg.x + tid;
    if (i >= n4) return;
    const float4 x = gate[i];
    const float4 u = up[i];
    out[i] = half4(0.5f * x * (1.0f + tanh(0.7978846f * x * (1.0f + 0.044715f * x * x))) * u);
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

// out[i] *= sigmoid(gate[i]): Muse Glimmer's gate on each head's attention output, from its
// own projection (apply_gate below reads Qwen's, interleaved with the queries).
kernel void sigmoid_mul(ARGS, KP) {
    device float* out = ARG_PTR(float, 0);
    device const float* gate = ARG_PTR(const float, 1);
    const int n = ARG_INT(2);
    PER_NUMBER
    if (i >= n) return;
    out[i] *= 1.0f / (1.0f + exp(-gate[i]));
}

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
// delta_net in syrinx.cu). n a multiple of 8 up to 128.
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

// delta_net with PARTS threads to a column of the state (PARTS × n threads a head): 8
// halves each thread's walk along its column, for one token's latency.
template <int PARTS>
kernel void delta_net_p(ARGS, KP) {
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
    // Optional (0 for none): the state before the batch and after each of its tokens but
    // the last, a whole state a slot, so a draft's rejected guesses can be undone
    // (Qwen35Cuda::rewind).
    device float* steps = ARG_PTR(float, 15);
    threadgroup float scratch[32];
    threadgroup float ks[128];
    threadgroup float qs[128];
    const int h = gid.x, v_heads = ngrid.x, kh = h % k_heads;
    const int j = tid / PARTS, part = tid % PARTS, per = n / PARTS;
    device float* column = state + ((ulong)h * n + j) * n;
    float st[128 / PARTS];
    for (int r = 0; r < 128 / PARTS; r++)
        if (r < per) st[r] = column[part + PARTS * r];
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    const ulong state_size = (ulong)v_heads * n * n, at_column = ((ulong)h * n + j) * n;
    if (NONNULL(steps))
        for (int r = 0; r < 128 / PARTS; r++)
            if (r < per) steps[at_column + part + PARTS * r] = st[r];
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
        for (int r = 0; r < 128 / PARTS; r++) {
            if (r < per) {
                st[r] *= decay;
                dot += st[r] * ks[part + PARTS * r];
            }
        }
        for (ushort o = 1; o < PARTS; o <<= 1) dot += simd_shuffle_xor(dot, o);
        const float delta = (c[2 * key_dim + h * n + j] - dot) * beta;
        float o = 0.0f;
        for (int r = 0; r < 128 / PARTS; r++) {
            if (r < per) {
                st[r] += ks[part + PARTS * r] * delta;
                o += st[r] * qs[part + PARTS * r];
            }
        }
        for (ushort o2 = 1; o2 < PARTS; o2 <<= 1) o += simd_shuffle_xor(o, o2);
        o *= inv_sqrt_n;

        const float sum = block_sum(part == 0 ? o * o : 0.0f, scratch, tid, tpg.x);
        if (part == 0) {
            const float normed = o / sqrt(sum / (float)n + eps) * norm[j];
            const ulong at = (ulong)head_at * n + j;
            const float zj = z[at];
            out[at] = normed * (zj / (1.0f + exp(-zj)));
        }
        if (NONNULL(steps) && t + 1 < tokens) {
            device float* slot = steps + (ulong)(t + 1) * state_size + at_column;
            for (int r = 0; r < 128 / PARTS; r++)
                if (r < per) slot[part + PARTS * r] = st[r];
        }
    }
    for (int r = 0; r < 128 / PARTS; r++)
        if (r < per) column[part + PARTS * r] = st[r];
}

template [[host_name("delta_net_4")]] kernel void delta_net_p<4>(ARGS, KP);
template [[host_name("delta_net_8")]] kernel void delta_net_p<8>(ARGS, KP);

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
// syrinx.cu).
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

// ---- Gated DeltaNet for a prompt in chunks (the WY form) -------------------------------
//
// As delta_chunk_prep and delta_chunk_scan in syrinx.cu (the derivation is there), in
// chunks of DC = 32 tokens instead of 64: with 64, the prep's threadgroup memory is over
// Metal's 32 KB. Scratch per chunk and value head (delta_scratch_floats on the host): W,
// U, Q̂, K̂ (DC × n each), P (DC × DC), γ_DC. n ≤ 128.

#define DC 32
static inline ulong delta_chunk_stride(int n) { return (ulong)4 * DC * n + DC * DC + 4; }

// One threadgroup of 256 per value head (gid.x) and chunk (gid.y).
kernel void delta_chunk_prep(ARGS, KP) {
    device const float* conv = ARG_PTR(const float, 0);
    const int channels = ARG_INT(1), key_dim = ARG_INT(2);
    device const float* beta_raw = ARG_PTR(const float, 3);
    device const float* alpha_raw = ARG_PTR(const float, 4);
    device const float* dt_bias = ARG_PTR(const float, 5);
    device const float* a = ARG_PTR(const float, 6);
    device float* scratch = ARG_PTR(float, 7);
    const int n = ARG_INT(8), k_heads = ARG_INT(9), tokens = ARG_INT(10);
    threadgroup float sm[DC * (128 + 4)];  // the chunk's k (rows of n + 4); then A
    threadgroup float G[DC], bs[DC];
    const int NS = n + 4;
    threadgroup float* ks = sm;
    threadgroup float* As = sm;  // DC × (DC + 1), over ks once A is worked out
    const int h = gid.x, v_heads = ngrid.x, ch = gid.y, kh = h % k_heads;
    const int t0 = ch * DC, len = min(DC, tokens - t0);
    device float* base = scratch + ((ulong)ch * v_heads + h) * delta_chunk_stride(n);
    device float *W = base, *U = W + DC * n, *Qh = U + DC * n, *Kh = Qh + DC * n, *P = Kh + DC * n;
    device const float* q_rows = conv + (ulong)t0 * channels + kh * n;
    device const float* k_rows = conv + (ulong)t0 * channels + key_dim + kh * n;

    for (int idx = tid; idx < DC * n; idx += tpg.x) {
        const int i = idx / n, d = idx % n;
        ks[i * NS + d] = i < len ? k_rows[(ulong)i * channels + d] : 0.0f;
    }
    if ((int)tid < DC) {
        float log_g = 0.0f, beta = 0.0f;  // rows past the end: no decay, no update
        if ((int)tid < len) {
            const int head_at = (t0 + tid) * v_heads + h;
            beta = 1.0f / (1.0f + exp(-beta_raw[head_at]));
            log_g = softplus_(alpha_raw[head_at] + dt_bias[h]) * a[h];
        }
        G[tid] = log_g;
        bs[tid] = beta;
    }
    BARRIER;
    if (tid == 0) {
        float g = 0.0f;
        for (int i = 0; i < DC; i++) G[i] = g += G[i];  // log γ_i
    }
    BARRIER;

    // k·k and q·k in 2 × 2 tiles: thread → rows 2 ti.., columns tj + 16 y.
    const int ti = tid >> 4, tj = tid & 15;
    float kk[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}}, qk[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (int d = 0; d < n; d += 4) {
        float4 ki[2], qi[2], kj[2];
        for (int x = 0; x < 2; x++) {
            const int i = 2 * ti + x;
            ki[x] = *reinterpret_cast<threadgroup const float4*>(&ks[i * NS + d]);
            qi[x] = i < len ? f4(q_rows + (ulong)i * channels + d) : float4(0.0f);
            kj[x] = *reinterpret_cast<threadgroup const float4*>(&ks[(tj + 16 * x) * NS + d]);
        }
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++) {
                kk[x][y] += dot(ki[x], kj[y]);
                qk[x][y] += dot(qi[x], kj[y]);
            }
    }
    const float g_last = G[DC - 1];
    for (int idx = tid; idx < DC * n; idx += tpg.x) {
        const int i = idx / n, d = idx % n;
        Qh[idx] = i < len ? exp(G[i]) * q_rows[(ulong)i * channels + d] : 0.0f;
        Kh[idx] = i < len ? exp(g_last - G[i]) * ks[i * NS + d] : 0.0f;
    }
    if (tid == 0) P[DC * DC] = exp(g_last);
    BARRIER;  // done with ks: A goes in its place
    for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++) {
            const int i = 2 * ti + x, j = tj + 16 * y;
            const float decay = j <= i ? exp(G[i] - G[j]) : 0.0f;
            As[i * (DC + 1) + j] = j < i ? bs[i] * decay * kk[x][y] : 0.0f;
            P[i * DC + j] = j <= i && i < len ? decay * qk[x][y] : 0.0f;
        }
    BARRIER;

    // W and U by forward substitution, a thread per column of [W | U].
    const int col = tid;
    if (col < 2 * n) {
        const bool is_w = col < n;
        const int d = is_w ? col : col - n;
        device float* X = is_w ? W : U;
        float xs[DC];
        for (int i = 0; i < DC; i++) {
            float b = 0.0f;
            if (i < len)
                b = is_w ? bs[i] * exp(G[i]) * k_rows[(ulong)i * channels + d]
                         : bs[i] * conv[(ulong)(t0 + i) * channels + 2 * key_dim + h * n + d];
            float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int m = 0; m < i; m++) part[m & 3] += As[i * (DC + 1) + m] * xs[m];
            b -= (part[0] + part[1]) + (part[2] + part[3]);
            xs[i] = b;
            X[i * n + d] = b;
        }
    }
}

// The chunks in order for columns gid.y × DSC_COLS.. of value head gid.x's state: 64
// threads, the state's columns and the chunk's δ in threadgroup memory. A thread works out
// one row × 8 columns of δ and O, and 2 rows of the new state.
#define DSC_COLS 16
#define DSC_STRIDE (DSC_COLS + 4)
kernel void delta_chunk_scan(ARGS, KP) {
    device const float* scratch = ARG_PTR(const float, 0);
    device float* state = ARG_PTR(float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n = ARG_INT(3), tokens = ARG_INT(4);
    threadgroup float Ss[128][DSC_STRIDE];  // S0[r][c]
    threadgroup float ds[DC][DSC_COLS];     // δ[i][c]
    const int h = gid.x, v_heads = ngrid.x, j0 = gid.y * DSC_COLS;
    const int i = tid >> 1, c0 = (tid & 1) * 8;  // δ and O: row i, 8 columns from c0
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int idx = tid; idx < n * DSC_COLS; idx += tpg.x) {
        const int c = idx / n, r = idx % n;
        Ss[r][c] = state[((ulong)h * n + j0 + c) * n + r];
    }
    const int chunks = (tokens + DC - 1) / DC;
    for (int ch = 0; ch < chunks; ch++) {
        device const float* base = scratch + ((ulong)ch * v_heads + h) * delta_chunk_stride(n);
        device const float *W = base, *U = W + DC * n, *Qh = U + DC * n, *Kh = Qh + DC * n, *P = Kh + DC * n;
        const float gamma = P[DC * DC];
        BARRIER;  // the state is in (or updated)

        // δ = U − W S0
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < n; r += 4) {
            const float4 w4 = f4(&W[i * n + r]);
            for (int e = 0; e < 4; e++) {
                const float wv = w4[e];
                const float4 a = *reinterpret_cast<threadgroup const float4*>(&Ss[r + e][c0]);
                const float4 b = *reinterpret_cast<threadgroup const float4*>(&Ss[r + e][c0 + 4]);
                acc[0] += wv * a.x, acc[1] += wv * a.y, acc[2] += wv * a.z, acc[3] += wv * a.w;
                acc[4] += wv * b.x, acc[5] += wv * b.y, acc[6] += wv * b.z, acc[7] += wv * b.w;
            }
        }
        for (int c = 0; c < 8; c++) ds[i][c0 + c] = U[i * n + j0 + c0 + c] - acc[c];
        BARRIER;

        // O = Q̂ S0 + P δ
        float o[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < n; r += 4) {
            const float4 q4 = f4(&Qh[i * n + r]);
            for (int e = 0; e < 4; e++) {
                const float qv = q4[e];
                const float4 a = *reinterpret_cast<threadgroup const float4*>(&Ss[r + e][c0]);
                const float4 b = *reinterpret_cast<threadgroup const float4*>(&Ss[r + e][c0 + 4]);
                o[0] += qv * a.x, o[1] += qv * a.y, o[2] += qv * a.z, o[3] += qv * a.w;
                o[4] += qv * b.x, o[5] += qv * b.y, o[6] += qv * b.z, o[7] += qv * b.w;
            }
        }
        for (int m = 0; m <= i; m++) {
            const float pv = P[i * DC + m];
            for (int c = 0; c < 8; c++) o[c] += pv * ds[m][c0 + c];
        }
        const int t = ch * DC + i;
        if (t < tokens) {
            device float* dst = out + ((ulong)t * v_heads + h) * n + j0 + c0;
            for (int c = 0; c < 8; c++) dst[c] = o[c] * inv_sqrt_n;
        }
        BARRIER;  // done reading S0

        // S = γ S0 + K̂ᵀ δ, rows r and r + 64
        for (int r = tid; r < n; r += 2 * tpg.x) {
            const int r1 = r + tpg.x;
            const bool two = r1 < n;
            float4 s0[DSC_COLS / 4], s1[DSC_COLS / 4];
            for (int c = 0; c < DSC_COLS / 4; c++) {
                s0[c] = *reinterpret_cast<threadgroup const float4*>(&Ss[r][4 * c]) * gamma;
                s1[c] = two ? *reinterpret_cast<threadgroup const float4*>(&Ss[r1][4 * c]) * gamma : float4(0.0f);
            }
            for (int m = 0; m < DC; m++) {
                const float k0 = Kh[m * n + r], k1 = two ? Kh[m * n + r1] : 0.0f;
                for (int c = 0; c < DSC_COLS / 4; c++) {
                    const float4 d4 = *reinterpret_cast<threadgroup const float4*>(&ds[m][4 * c]);
                    s0[c] += k0 * d4;
                    s1[c] += k1 * d4;
                }
            }
            for (int c = 0; c < DSC_COLS / 4; c++) {
                *reinterpret_cast<threadgroup float4*>(&Ss[r][4 * c]) = s0[c];
                if (two) *reinterpret_cast<threadgroup float4*>(&Ss[r1][4 * c]) = s1[c];
            }
        }
    }
    BARRIER;
    for (int idx = tid; idx < n * DSC_COLS; idx += tpg.x) {
        const int c = idx / n, r = idx % n;
        state[((ulong)h * n + j0 + c) * n + r] = Ss[r][c];
    }
}

// delta_chunk_scan on SIMD-group matrices: the same chunks in order, for COLS columns of
// value head gid.x's state a threadgroup (gid.y), 4 SIMD groups. Each chunk's products
// are 8 × 8 multiplies in f32: δ = U − W S0 (W read from the scratch), O = Q̂ S0 + P δ
// (written out through threadgroup memory, scaled by 1/√n, only the chunk's real
// tokens), S = γ S0 + K̂ᵀ δ (γ S0 as diag(γ) S0). n ≤ 128, a multiple of 8.
template <int COLS>
kernel void delta_chunk_scan_sg(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]]) {
    device const float* scratch = ARG_PTR(const float, 0);
    device float* state = ARG_PTR(float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n = ARG_INT(3), tokens = ARG_INT(4);
    constexpr int CB = COLS / 8, SG = 4;
    threadgroup float Ss[128 * COLS];  // S0[r][c]
    threadgroup float ds[DC * COLS];   // δ, then O
    threadgroup float gdiag[64];
    const int h = gid.x, v_heads = ngrid.x, j0 = gid.y * COLS;
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int idx = tid; idx < n * COLS; idx += tpg.x) {
        const int c = idx / n, r = idx % n;
        Ss[r * COLS + c] = state[((ulong)h * n + j0 + c) * n + r];
    }
    for (int i = tid; i < 64; i += tpg.x) gdiag[i] = 0.0f;
    BARRIER;  // zeroed before any chunk writes its γ on the diagonal
    const int chunks = (tokens + DC - 1) / DC;
    const int rbs = n / 8;
    for (int ch = 0; ch < chunks; ch++) {
        device const float* base = scratch + ((ulong)ch * v_heads + h) * delta_chunk_stride(n);
        device const float *W = base, *U = W + DC * n, *Qh = U + DC * n, *Kh = Qh + DC * n, *P = Kh + DC * n;
        const float gamma = P[DC * DC];
        if (tid < 8) gdiag[tid * 9] = gamma;
        BARRIER;  // the state is in (or updated)

        // W S0 into ds, then δ = U − W S0.
        for (int blk = sg; blk < (DC / 8) * CB; blk += SG) {
            const int ib = blk / CB, cb = blk % CB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = 0; kb < rbs; kb++) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, W + ib * 8 * n + kb * 8, n);
                simdgroup_load(b, Ss + kb * 8 * COLS + cb * 8, COLS);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, ds + ib * 8 * COLS + cb * 8, COLS);
        }
        BARRIER;
        for (int e = tid; e < DC * COLS; e += tpg.x) {
            const int i = e / COLS, c = e % COLS;
            ds[e] = U[i * n + j0 + c] - ds[e];
        }
        BARRIER;
        // O = Q̂ S0 + P δ, kept in registers until δ is no longer needed.
        simdgroup_float8x8 o_blk[(DC / 8) * CB / SG > 0 ? (DC / 8) * CB / SG : 1];
        int nb = 0;
        for (int blk = sg; blk < (DC / 8) * CB; blk += SG, nb++) {
            const int ib = blk / CB, cb = blk % CB;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = 0; kb < rbs; kb++) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, Qh + ib * 8 * n + kb * 8, n);
                simdgroup_load(b, Ss + kb * 8 * COLS + cb * 8, COLS);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            for (int mb = 0; mb <= ib; mb++) {  // P is 0 past the diagonal
                simdgroup_float8x8 a, b;
                simdgroup_load(a, P + ib * 8 * DC + mb * 8, DC);
                simdgroup_load(b, ds + mb * 8 * COLS + cb * 8, COLS);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            o_blk[nb] = acc;
        }
        BARRIER;  // everyone is done reading S0 for O
        // S = γ S0 + K̂ᵀ δ, each SIMD group updating its own blocks.
        simdgroup_float8x8 gm;
        simdgroup_load(gm, gdiag, 8);
        for (int blk = sg; blk < rbs * CB; blk += SG) {
            const int rb = blk / CB, cb = blk % CB;
            simdgroup_float8x8 s0, acc;
            simdgroup_load(s0, Ss + rb * 8 * COLS + cb * 8, COLS);
            simdgroup_multiply(acc, gm, s0);
            for (int mb = 0; mb < DC / 8; mb++) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, Kh + mb * 8 * n + rb * 8, n, ulong2(0, 0), true);
                simdgroup_load(b, ds + mb * 8 * COLS + cb * 8, COLS);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, Ss + rb * 8 * COLS + cb * 8, COLS);
        }
        BARRIER;  // done with δ: O goes in its place
        nb = 0;
        for (int blk = sg; blk < (DC / 8) * CB; blk += SG, nb++) {
            const int ib = blk / CB, cb = blk % CB;
            simdgroup_store(o_blk[nb], ds + ib * 8 * COLS + cb * 8, COLS);
        }
        BARRIER;
        for (int e = tid; e < DC * COLS; e += tpg.x) {
            const int i = e / COLS, c = e % COLS, t = ch * DC + i;
            if (t < tokens) out[((ulong)t * v_heads + h) * n + j0 + c] = ds[e] * inv_sqrt_n;
        }
    }
    BARRIER;
    for (int idx = tid; idx < n * COLS; idx += tpg.x) {
        const int c = idx / n, r = idx % n;
        state[((ulong)h * n + j0 + c) * n + r] = Ss[r * COLS + c];
    }
}

template [[host_name("delta_chunk_scan_sg_8")]] kernel void delta_chunk_scan_sg<8>(ARGS, KP, uint);

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

// ---- GPU-side argmax -----------------------------------------------------------------

// Stage 1: each threadgroup of 256 threads finds the largest value and its smallest index
// across its strided slice of x[0..n], writing one (val, idx) pair per threadgroup.
kernel void argmax_part(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    device const float* x = ARG_PTR(const float, 0);
    device float* vals = ARG_PTR(float, 1);
    device uint* idxs = ARG_PTR(uint, 2);
    const uint n = (uint)ARG_INT(3);
    float best = -INFINITY;
    uint best_i = 0;
    const uint stride = ngrid.x * 256u;
    for (uint i = gid.x * 256u + tid; i < n; i += stride) {
        const float v = x[i];
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    const float sg_max = simd_max(best);
    const uint sg_idx = simd_min(best == sg_max ? best_i : 0xFFFFFFFFu);
    threadgroup float s_val[8];
    threadgroup uint s_idx[8];
    if (lane == 0) {
        s_val[sg] = sg_max;
        s_idx[sg] = sg_idx;
    }
    BARRIER;
    if (tid == 0) {
        float m = s_val[0];
        uint mi = s_idx[0];
        for (int s = 1; s < 8; s++) {
            if (s_val[s] > m || (s_val[s] == m && s_idx[s] < mi)) {
                m = s_val[s];
                mi = s_idx[s];
            }
        }
        vals[gid.x] = m;
        idxs[gid.x] = mi;
    }
}

// Stage 2: single threadgroup of 256 threads reduces the stage-1 partials to out[0].
kernel void argmax_final(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device const float* vals = ARG_PTR(const float, 0);
    device const uint* idxs = ARG_PTR(const uint, 1);
    device uint* out = ARG_PTR(uint, 2);
    const uint count = (uint)ARG_INT(3);
    const float v = tid < count ? vals[tid] : -INFINITY;
    const uint idx = tid < count ? idxs[tid] : 0xFFFFFFFFu;
    const float sg_max = simd_max(v);
    const uint sg_idx = simd_min(v == sg_max ? idx : 0xFFFFFFFFu);
    threadgroup float s_val[8];
    threadgroup uint s_idx[8];
    if (lane == 0) {
        s_val[sg] = sg_max;
        s_idx[sg] = sg_idx;
    }
    BARRIER;
    if (tid == 0) {
        float m = s_val[0];
        uint mi = s_idx[0];
        for (int s = 1; s < 8; s++) {
            if (s_val[s] > m || (s_val[s] == m && s_idx[s] < mi)) {
                m = s_val[s];
                mi = s_idx[s];
            }
        }
        out[0] = mi == 0xFFFFFFFFu ? 0u : mi;
    }
}

// ---------------------------------------------------------------------------------------
// Inkling 2 (models/inkling2_cuda.rs; the CPU reference is models/inkling2.rs), for a
// batch of `tokens` tokens at positions pos0 on (one when generating). Metal only so far:
// the CUDA build has no inkling kernels.

// Attention over depth: for each query q and token t (threadgroup (q, t)), a softmax over
// the token's RMS-normed states (no weight, eps 1e-6) of state·query / sqrt(dim), and the
// weighted sum of the normed states, into row q × stride + t of `out`. The states are the
// `n_blocks` blocks (block s's row for token t at row s × stride + t of `states`) and the
// token's row of `cur`. One pass over the states, one barrier.
#define INK_MAX_STATES 16
kernel void ink_mix(ARGS, KP) {
    device const float* states = ARG_PTR(const float, 0);
    device const float* queries = ARG_PTR(const float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n_blocks = ARG_INT(3), dim = ARG_INT(4), stride = ARG_INT(5);
    device const float* cur = ARG_PTR(const float, 6);
    const int n = n_blocks + 1;
    const int t = gid.y;
    const uint lane = tid & 31, warp = tid >> 5, warps = (tpg.x + 31) >> 5;
    device const float* q = queries + (ulong)gid.x * dim;
    // Each SIMD group's sums for each state, then the states' weights.
    threadgroup float part[32][2 * INK_MAX_STATES];
    threadgroup float weight[INK_MAX_STATES];
    for (int s = 0; s < n; s++) {
        device const float* x = s < n_blocks ? states + ((ulong)s * stride + t) * dim : cur + (ulong)t * dim;
        float sq = 0.0f, dt = 0.0f;
        for (int i = tid; i < dim; i += tpg.x) {
            const float v = x[i];
            sq += v * v;
            dt += v * q[i];
        }
        sq = simd_sum(sq);
        dt = simd_sum(dt);
        if (lane == 0) {
            part[warp][s] = sq;
            part[warp][INK_MAX_STATES + s] = dt;
        }
    }
    BARRIER;
    if (tid == 0) {
        const float root = sqrt((float)dim);
        float w[INK_MAX_STATES], inv[INK_MAX_STATES], top = -INFINITY;
        for (int s = 0; s < n; s++) {
            float a = 0.0f, b = 0.0f;
            for (uint k = 0; k < warps; k++) {
                a += part[k][s];
                b += part[k][INK_MAX_STATES + s];
            }
            inv[s] = 1.0f / sqrt(a / (float)dim + 1e-6f);
            w[s] = b * inv[s] / root;
            top = max(top, w[s]);
        }
        float total = 0.0f;
        for (int s = 0; s < n; s++) {
            w[s] = exp(w[s] - top);
            total += w[s];
        }
        for (int s = 0; s < n; s++) weight[s] = w[s] / total * inv[s];
    }
    BARRIER;
    device float* o = out + ((ulong)gid.x * stride + t) * dim;
    for (int i = tid; i < dim; i += tpg.x) {
        float v = 0.0f;
        for (int s = 0; s < n; s++) {
            device const float* x = s < n_blocks ? states + ((ulong)s * stride + t) * dim : cur + (ulong)t * dim;
            v += weight[s] * x[i];
        }
        o[i] = v;
    }
}

// Fast single-pass float4 ink_mix for dim % 4 == 0, n <= 8, and dim / 4 <= tpg.x:
// each thread holds one float4 per state in registers so states are read ONCE from global
// memory. When `norm_w` is non-null, fuses RMSNorm(out, norm_w, eps) in registers before
// writing out (saving a full rms_norm dispatch and intermediate buffer traffic).
kernel void ink_mix_fast(ARGS, KP) {
    device const float* states = ARG_PTR(const float, 0);
    device const float* queries = ARG_PTR(const float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n_blocks = ARG_INT(3), dim = ARG_INT(4), stride = ARG_INT(5);
    device const float* cur = ARG_PTR(const float, 6);
    device const float* norm_w = ARG_PTR(const float, 7);
    const float eps = ARG_F32(8);
    const int n = n_blocks + 1;
    const int t = gid.y;
    const int dim4 = dim >> 2;
    const uint lane = tid & 31, warp = tid >> 5, warps = (tpg.x + 31) >> 5;
    const bool active = (int)tid < dim4;
    device const float* q = queries + (ulong)gid.x * dim;
    const float4 q4 = active ? f4(q + tid * 4) : float4(0.0f);
    threadgroup float part[32][16];
    threadgroup float weight[8];
    float4 xs[8];
    for (int s = 0; s < n; s++) {
        float sq = 0.0f, dt = 0.0f;
        if (active) {
            device const float* x =
                s < n_blocks ? states + ((ulong)s * stride + t) * dim : cur + (ulong)t * dim;
            const float4 v = f4(x + tid * 4);
            xs[s] = v;
            sq = dot(v, v);
            dt = dot(v, q4);
        }
        sq = simd_sum(sq);
        dt = simd_sum(dt);
        if (lane == 0) {
            part[warp][s] = sq;
            part[warp][8 + s] = dt;
        }
    }
    BARRIER;
    if (warp == 0) {
        // Lane s takes state s (the same sums, in the same order, as one thread did).
        const float root = sqrt((float)dim);
        const int s = (int)lane;
        float inv = 0.0f, w = -INFINITY;
        if (s < n) {
            float a = 0.0f, b = 0.0f;
            for (uint k = 0; k < warps; k++) {
                a += part[k][s];
                b += part[k][8 + s];
            }
            inv = rsqrt(a / (float)dim + 1e-6f);
            w = b * inv / root;
        }
        float top = -INFINITY;
        for (int x = 0; x < n; x++) top = max(top, simd_shuffle(w, (ushort)x));
        const float e = s < n ? exp(w - top) : 0.0f;
        float total = 0.0f;
        for (int x = 0; x < n; x++) total += simd_shuffle(e, (ushort)x);
        if (s < n) weight[s] = e / total * inv;
    }
    BARRIER;
    float4 v = float4(0.0f);
    if (active) {
        for (int s = 0; s < n; s++) v += weight[s] * xs[s];
    }
    if (NONNULL(norm_w)) {
        float sq = simd_sum(active ? dot(v, v) : 0.0f);
        // part[][0] is free again: every thread read its weights after the last barrier.
        BARRIER;
        if (lane == 0) part[warp][0] = sq;
        BARRIER;
        // Every thread adds the parts up itself, in the order one thread would.
        float total = 0.0f;
        for (uint k = 0; k < warps; k++) total += part[k][0];
        const float norm_scale = rsqrt(total / (float)dim + eps);
        if (active) {
            const float4 nw = f4(norm_w + tid * 4);
            v = v * norm_scale * nw;
        }
    }
    if (active) {
        device float* o = out + ((ulong)gid.x * stride + t) * dim;
        *reinterpret_cast<device float4*>(o + tid * 4) = v;
    }
}

// Two independent ink_mix_fast calls in one dispatch: gid.x < 4 mixes `bank` into
// `messages` (4 loop message queries), gid.x >= 4 mixes `stack` into `inputs` (1 or 3
// attention input queries). Eliminates 24 dispatches per token on looped Inkling 2.
kernel void ink_mix_pair(ARGS, KP) {
    const bool is_msg = (int)gid.x < 4;
    const int qi = is_msg ? (int)gid.x : (int)gid.x - 4;
    device const float* states = is_msg ? ARG_PTR(const float, 0) : ARG_PTR(const float, 5);
    device const float* queries = is_msg ? ARG_PTR(const float, 1) : ARG_PTR(const float, 6);
    device float* out = is_msg ? ARG_PTR(float, 2) : ARG_PTR(float, 7);
    const int n_blocks = is_msg ? ARG_INT(3) : ARG_INT(8);
    device const float* cur = is_msg ? ARG_PTR(const float, 4) : ARG_PTR(const float, 9);
    const int dim = ARG_INT(10), stride = ARG_INT(11);
    const int n = n_blocks + 1;
    const int t = gid.y;
    const int dim4 = dim >> 2;
    const uint lane = tid & 31, warp = tid >> 5, warps = (tpg.x + 31) >> 5;
    const bool active = (int)tid < dim4;
    device const float* q = queries + (ulong)qi * dim;
    const float4 q4 = active ? f4(q + tid * 4) : float4(0.0f);
    threadgroup float part[32][16];
    threadgroup float weight[8];
    float4 xs[8];
    for (int s = 0; s < n; s++) {
        float sq = 0.0f, dt = 0.0f;
        if (active) {
            device const float* x =
                s < n_blocks ? states + ((ulong)s * stride + t) * dim : cur + (ulong)t * dim;
            const float4 v = f4(x + tid * 4);
            xs[s] = v;
            sq = dot(v, v);
            dt = dot(v, q4);
        }
        sq = simd_sum(sq);
        dt = simd_sum(dt);
        if (lane == 0) {
            part[warp][s] = sq;
            part[warp][8 + s] = dt;
        }
    }
    BARRIER;
    if (warp == 0) {
        const float root = sqrt((float)dim);
        const int s = (int)lane;
        float inv = 0.0f, w = -INFINITY;
        if (s < n) {
            float a = 0.0f, b = 0.0f;
            for (uint k = 0; k < warps; k++) {
                a += part[k][s];
                b += part[k][8 + s];
            }
            inv = rsqrt(a / (float)dim + 1e-6f);
            w = b * inv / root;
        }
        float top = -INFINITY;
        for (int x = 0; x < n; x++) top = max(top, simd_shuffle(w, (ushort)x));
        const float e = s < n ? exp(w - top) : 0.0f;
        float total = 0.0f;
        for (int x = 0; x < n; x++) total += simd_shuffle(e, (ushort)x);
        if (s < n) weight[s] = e / total * inv;
    }
    BARRIER;
    if (active) {
        float4 v = float4(0.0f);
        for (int s = 0; s < n; s++) v += weight[s] * xs[s];
        device float* o = out + ((ulong)qi * stride + t) * dim;
        *reinterpret_cast<device float4*>(o + tid * 4) = v;
    }
}

// The loop controller's input for each token (a threadgroup each): its row of `cur` and of
// `msg`, each RMS-normed (no weight, eps 1e-6), side by side in a row of 2 × dim.
kernel void ink_pair_norm(ARGS, KP) {
    device const float* cur = ARG_PTR(const float, 0);
    device const float* msg = ARG_PTR(const float, 1);
    device float* out = ARG_PTR(float, 2);
    const int dim = ARG_INT(3);
    threadgroup float scratch[64];
    device const float* c = cur + (ulong)gid.x * dim;
    device const float* m = msg + (ulong)gid.x * dim;
    device float* o = out + (ulong)gid.x * 2 * dim;
    const uint lane = tid & 31, warp = tid >> 5, warps = (tpg.x + 31) >> 5;
    if ((dim & 3) == 0 && dim <= 4096 && tpg.x >= 256) {
        const int n4 = dim >> 2;
        float4 rc[4], rm[4];
        int cnt = 0;
        float sc = 0.0f, sm = 0.0f;
        for (int i = tid; i < n4; i += tpg.x) {
            const float4 cv = f4(c + i * 4), mv = f4(m + i * 4);
            rc[cnt] = cv;
            rm[cnt] = mv;
            cnt++;
            sc += dot(cv, cv);
            sm += dot(mv, mv);
        }
        sc = simd_sum(sc);
        sm = simd_sum(sm);
        if (lane == 0) {
            scratch[warp] = sc;
            scratch[32 + warp] = sm;
        }
        BARRIER;
        const float tsc = simd_sum(lane < warps ? scratch[lane] : 0.0f);
        const float tsm = simd_sum(lane < warps ? scratch[32 + lane] : 0.0f);
        const float ic = rsqrt(tsc / (float)dim + 1e-6f);
        const float im = rsqrt(tsm / (float)dim + 1e-6f);
        int k = 0;
        for (int i = tid; i < n4; i += tpg.x) {
            *reinterpret_cast<device float4*>(o + i * 4) = rc[k] * ic;
            *reinterpret_cast<device float4*>(o + dim + i * 4) = rm[k] * im;
            k++;
        }
    } else {
        float sc = 0.0f, sm = 0.0f;
        for (int i = tid; i < dim; i += tpg.x) {
            sc += c[i] * c[i];
            sm += m[i] * m[i];
        }
        sc = block_sum(sc, scratch, tid, tpg.x);
        sm = block_sum(sm, scratch, tid, tpg.x);
        const float ic = 1.0f / sqrt(sc / (float)dim + 1e-6f), im = 1.0f / sqrt(sm / (float)dim + 1e-6f);
        for (int i = tid; i < dim; i += tpg.x) {
            o[i] = c[i] * ic;
            o[dim + i] = m[i] * im;
        }
    }
}

// A looped layer's controller: out = cur − tanh(forget) × cur + tanh(gain) × message.
kernel void ink_control(ARGS, KP) {
    device const float* cur = ARG_PTR(const float, 0);
    device const float* msg = ARG_PTR(const float, 1);
    device const float* gain = ARG_PTR(const float, 2);
    device const float* forget = ARG_PTR(const float, 3);
    device float* out = ARG_PTR(float, 4);
    const int n = ARG_INT(5);
    PER_NUMBER
    if ((n & 3) == 0) {
        const int n4 = n >> 2;
        if (i >= n4) return;
        const int at = i * 4;
        const float4 c = f4(cur + at);
        const float4 m = f4(msg + at);
        const float4 g = f4(gain + at);
        const float4 f = f4(forget + at);
        *reinterpret_cast<device float4*>(out + at) = c - tanh(f) * c + tanh(g) * m;
    } else {
        if (i >= n) return;
        const float c = cur[i];
        out[i] = c - tanh(forget[i]) * c + tanh(gain[i]) * msg[i];
    }
}

// ink_ctrl_down_f16 for up to 3 controllers at once (gid.y the controller): pointers
// cur 0..2, msg 3..5, w 6..8, low 9..11; dim 12.
kernel void ink_ctrl3_down_f16(ARGS, KP) {
    const int j = gid.y;
    device const float* cur = ARG_PTR(const float, 0 + j);
    device const float* msg = ARG_PTR(const float, 3 + j);
    device const half* w = ARG_PTR(const half, 6 + j);
    device float* low = ARG_PTR(float, 9 + j);
    const int dim = ARG_INT(12);
    const int row = gid.x;
    device const half* wl = w + (ulong)row * (2 * dim);
    device const half* wr = wl + dim;
    float sc = 0.0f, sm = 0.0f, dc = 0.0f, dm = 0.0f;
    for (int i = tid * 4; i < dim; i += tpg.x * 4) {
        const float4 c4 = f4(cur + i);
        const float4 m4 = f4(msg + i);
        const float4 w0 = float4(*reinterpret_cast<device const half4*>(wl + i));
        const float4 w1 = float4(*reinterpret_cast<device const half4*>(wr + i));
        sc += dot(c4, c4);
        sm += dot(m4, m4);
        dc += dot(c4, w0);
        dm += dot(m4, w1);
    }
    const uint lane = tid & 31, warp = tid >> 5, warps = tpg.x >> 5;
    sc = simd_sum(sc);
    sm = simd_sum(sm);
    dc = simd_sum(dc);
    dm = simd_sum(dm);
    threadgroup float parts[32][4];
    if (lane == 0) {
        parts[warp][0] = sc;
        parts[warp][1] = sm;
        parts[warp][2] = dc;
        parts[warp][3] = dm;
    }
    BARRIER;
    if (tid == 0) {
        float tsc = 0.0f, tsm = 0.0f, tdc = 0.0f, tdm = 0.0f;
        for (uint k = 0; k < warps; k++) {
            tsc += parts[k][0];
            tsm += parts[k][1];
            tdc += parts[k][2];
            tdm += parts[k][3];
        }
        const float ic = rsqrt(tsc / (float)dim + 1e-6f);
        const float im = rsqrt(tsm / (float)dim + 1e-6f);
        low[row] = tdc * ic + tdm * im;
    }
}

// ink_ctrl_up_apply_f16 for 3 controllers at once, a threadgroup doing all 3 for its rows
// (every input read before any output is written: an output may be another's input).
// Pointers cur_in 0..2, msg 3..5, low 6..8, gain 9..11, forget 12..14; dim 15, rank 16;
// cur_out 17..19.
kernel void ink_ctrl3_up_apply_f16(ARGS, KP) {
    const int dim = ARG_INT(15), rank = ARG_INT(16);
    threadgroup float s_low[3][128];
    for (int j = 0; j < 3; j++) {
        device const float* low = ARG_PTR(const float, 6 + j);
        for (int i = tid; i < rank; i += tpg.x) s_low[j][i] = low[i];
    }
    BARRIER;
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = (int)gid.x * 8 + (int)sg * 2 + (int)(lane >> 4);
    if (row >= dim) return;
    const int hlane = (int)(lane & 15);
    float out[3];
    for (int j = 0; j < 3; j++) {
        device const half* wg = ARG_PTR(const half, 9 + j) + (ulong)row * rank;
        device const half* wf = ARG_PTR(const half, 12 + j) + (ulong)row * rank;
        float gs = 0.0f, fs = 0.0f;
        for (int k = hlane * 4; k < rank; k += 64) {
            const float4 lv = *reinterpret_cast<threadgroup const float4*>(&s_low[j][k]);
            const float4 g4 = float4(*reinterpret_cast<device const half4*>(wg + k));
            const float4 f4v = float4(*reinterpret_cast<device const half4*>(wf + k));
            gs += dot(g4, lv);
            fs += dot(f4v, lv);
        }
        gs += simd_shuffle_xor(gs, 8);
        gs += simd_shuffle_xor(gs, 4);
        gs += simd_shuffle_xor(gs, 2);
        gs += simd_shuffle_xor(gs, 1);
        fs += simd_shuffle_xor(fs, 8);
        fs += simd_shuffle_xor(fs, 4);
        fs += simd_shuffle_xor(fs, 2);
        fs += simd_shuffle_xor(fs, 1);
        const float c = ARG_PTR(const float, 0 + j)[row];
        out[j] = c - tanh(fs) * c + tanh(gs) * ARG_PTR(const float, 3 + j)[row];
    }
    if (hlane == 0) {
        for (int j = 0; j < 3; j++) ARG_PTR(float, 17 + j)[row] = out[j];
    }
}

// Single-token fused pair-norm + controller down projection (f16 weights):
// low[row] = dot(cur / rms(cur), W[row, 0..dim]) + dot(msg / rms(msg), W[row, dim..2*dim])
// Computed in a single pass using linearity of dot product; 1 threadgroup (256 threads) per row.
kernel void ink_ctrl_down_f16(ARGS, KP) {
    device const float* cur = ARG_PTR(const float, 0);
    device const float* msg = ARG_PTR(const float, 1);
    device const half* w = ARG_PTR(const half, 2);
    device float* low = ARG_PTR(float, 3);
    const int dim = ARG_INT(4);
    const int row = gid.x;
    device const half* wl = w + (ulong)row * (2 * dim);
    device const half* wr = wl + dim;
    float sc = 0.0f, sm = 0.0f, dc = 0.0f, dm = 0.0f;
    for (int i = tid * 4; i < dim; i += tpg.x * 4) {
        const float4 c4 = f4(cur + i);
        const float4 m4 = f4(msg + i);
        const float4 w0 = float4(*reinterpret_cast<device const half4*>(wl + i));
        const float4 w1 = float4(*reinterpret_cast<device const half4*>(wr + i));
        sc += dot(c4, c4);
        sm += dot(m4, m4);
        dc += dot(c4, w0);
        dm += dot(m4, w1);
    }
    const uint lane = tid & 31, warp = tid >> 5, warps = tpg.x >> 5;
    sc = simd_sum(sc);
    sm = simd_sum(sm);
    dc = simd_sum(dc);
    dm = simd_sum(dm);
    threadgroup float parts[32][4];
    if (lane == 0) {
        parts[warp][0] = sc;
        parts[warp][1] = sm;
        parts[warp][2] = dc;
        parts[warp][3] = dm;
    }
    BARRIER;
    if (tid == 0) {
        float tsc = 0.0f, tsm = 0.0f, tdc = 0.0f, tdm = 0.0f;
        for (uint k = 0; k < warps; k++) {
            tsc += parts[k][0];
            tsm += parts[k][1];
            tdc += parts[k][2];
            tdm += parts[k][3];
        }
        const float ic = rsqrt(tsc / (float)dim + 1e-6f);
        const float im = rsqrt(tsm / (float)dim + 1e-6f);
        low[row] = tdc * ic + tdm * im;
    }
}

// Single-token fused controller gain + forget projections (f16 weights) + gated update:
// cur_out[r] = cur_in[r] - tanh(dot(W_forget[r], low)) * cur_in[r] + tanh(dot(W_gain[r], low)) * msg[r].
// 128 threads (4 SIMD groups), 2 rows per SIMD group (16 lanes per row, float4/half4 loads).
kernel void ink_ctrl_up_apply_f16(ARGS, KP) {
    device const float* cur_in = ARG_PTR(const float, 0);
    device const float* msg = ARG_PTR(const float, 1);
    device const float* low = ARG_PTR(const float, 2);
    device const half* w_gain = ARG_PTR(const half, 3);
    device const half* w_forget = ARG_PTR(const half, 4);
    const int dim = ARG_INT(5), rank = ARG_INT(6);
    device float* cur_out = ARG_PTR(float, 7);
    threadgroup float s_low[128];
    for (int i = tid; i < rank; i += tpg.x) s_low[i] = low[i];
    BARRIER;
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = (int)gid.x * 8 + (int)sg * 2 + (int)(lane >> 4);
    if (row >= dim) return;
    const int hlane = (int)(lane & 15);
    device const half* wg = w_gain + (ulong)row * rank;
    device const half* wf = w_forget + (ulong)row * rank;
    float gs = 0.0f, fs = 0.0f;
    for (int k = hlane * 4; k < rank; k += 64) {
        const float4 lv = *reinterpret_cast<threadgroup const float4*>(&s_low[k]);
        const float4 g4 = float4(*reinterpret_cast<device const half4*>(wg + k));
        const float4 f4v = float4(*reinterpret_cast<device const half4*>(wf + k));
        gs += dot(g4, lv);
        fs += dot(f4v, lv);
    }
    gs += simd_shuffle_xor(gs, 8);
    gs += simd_shuffle_xor(gs, 4);
    gs += simd_shuffle_xor(gs, 2);
    gs += simd_shuffle_xor(gs, 1);
    fs += simd_shuffle_xor(fs, 8);
    fs += simd_shuffle_xor(fs, 4);
    fs += simd_shuffle_xor(fs, 2);
    fs += simd_shuffle_xor(fs, 1);
    if (hlane == 0) {
        const float c = cur_in[row];
        cur_out[row] = c - tanh(fs) * c + tanh(gs) * msg[row];
    }
}

// Each row (a threadgroup) of `n`, `per_token` rows a token: times `scale`'s row (row %
// per_token, if given), scaled down to an L2 norm of at most `threshold` (norm + 1e-6, as
// llama.cpp), then, with `log_scale`, times 1 + alpha ln(max(1, (pos + 1) / floor)) for the
// row's token at position pos0 + row / per_token.
kernel void ink_clip(ARGS, KP) {
    device float* x = ARG_PTR(float, 0);
    device const float* scale = ARG_PTR(const float, 1);
    const int n = ARG_INT(2);
    const float threshold = ARG_F32(3);
    const int per_token = ARG_INT(4), pos0 = ARG_INT(5), log_scale = ARG_INT(6);
    const float alpha = ARG_F32(7), floor_ = ARG_F32(8);
    const int rows = ARG_INT(9);
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    const int r = (int)gid.x * 4 + sgi;
    if (r >= rows) return;
    device float* row = x + (ulong)r * n;
    device const float* sr = scale + (ulong)(r % per_token) * n;
    const bool has_scale = NONNULL(scale);
    if (n == 128) {
        const float4 xv = f4(row + lane * 4);
        const float4 v = has_scale ? xv * f4(sr + lane * 4) : xv;
        const float sq = simd_sum(dot(v, v));
        float factor = clamp(threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
        if (log_scale) {
            const float pos = (float)(pos0 + r / per_token);
            factor *= 1.0f + alpha * log(max(1.0f, (pos + 1.0f) / floor_));
        }
        *reinterpret_cast<device float4*>(row + lane * 4) = v * factor;
    } else if (n <= 32) {
        const float v = lane < n ? row[lane] * (has_scale ? sr[lane] : 1.0f) : 0.0f;
        const float sq = simd_sum(v * v);
        if (lane < n) {
            float factor = clamp(threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
            if (log_scale) {
                const float pos = (float)(pos0 + r / per_token);
                factor *= 1.0f + alpha * log(max(1.0f, (pos + 1.0f) / floor_));
            }
            row[lane] = v * factor;
        }
    } else {
        float sq = 0.0f;
        for (int i = lane; i < n; i += 32) {
            const float v = row[i] * (has_scale ? sr[i] : 1.0f);
            sq += v * v;
        }
        sq = simd_sum(sq);
        float factor = clamp(threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
        if (log_scale) {
            const float pos = (float)(pos0 + r / per_token);
            factor *= 1.0f + alpha * log(max(1.0f, (pos + 1.0f) / floor_));
        }
        for (int i = lane; i < n; i += 32) row[i] = row[i] * (has_scale ? sr[i] : 1.0f) * factor;
    }
}

// Inkling 2 after a checked batch of guesses: each slot's convolution history as it was
// after the batch's first `keep` tokens, from the history before the batch (`saved`) and
// the batch's logged inputs (`log`: a slot's 4 convolutions one after another, each
// LOG_ROWS rows of its width). A thread a (slot, channel).
#define INK_LOG_ROWS 8
kernel void ink_history_rebuild(ARGS, KP) {
    device float* hist = ARG_PTR(float, 0);
    device const float* saved = ARG_PTR(const float, 1);
    device const float* log = ARG_PTR(const float, 2);
    const int channels = ARG_INT(3), kv = ARG_INT(4), dim = ARG_INT(5), keep = ARG_INT(6);
    const int total = ARG_INT(7), conv = ARG_INT(8);
    PER_NUMBER
    if (i >= total) return;
    const int slot = i / channels, ch = i % channels;
    int col, width;
    ulong off;
    if (ch < kv) { col = ch; width = kv; off = 0; }
    else if (ch < 2 * kv) { col = ch - kv; width = kv; off = (ulong)INK_LOG_ROWS * kv; }
    else if (ch < 2 * kv + dim) { col = ch - 2 * kv; width = dim; off = (ulong)INK_LOG_ROWS * 2 * kv; }
    else { col = ch - 2 * kv - dim; width = dim; off = (ulong)INK_LOG_ROWS * (2 * kv + dim); }
    const int rows = conv - 1;
    const ulong base = (ulong)slot * rows * channels;
    device const float* lg = log + (ulong)slot * INK_LOG_ROWS * channels + off + col;
    float v[8];
    for (int j = 0; j < rows; j++) {
        const int at = keep + j;  // in [saved rows..., logged rows...]
        v[j] = at < rows ? saved[base + (ulong)at * channels + ch] : lg[(ulong)(at - rows) * width];
    }
    for (int j = 0; j < rows; j++) hist[base + (ulong)j * channels + ch] = v[j];
}

// The causal convolution over the last `conv` inputs with a residual, a thread a channel,
// for `tokens` rows of `x` (rows of `n`) in order: y = (x + Σ_k w[k] in_k + Σ_lag
// taps[lag] in_{conv−1−lag}) / sqrt(m × conv + 1), the inputs oldest first (conv − 1 from
// `history`, rows `channels` apart, then x), `taps` (conv per head of `head_width`
// channels, a row of n / head_width × conv a token) optional, m 2 with them and 1 without.
// y goes to `out` (x itself, or another buffer of the same rows), added to what's there
// with `accumulate`. The history moves along a token at a time.
kernel void ink_conv(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device float* history = ARG_PTR(float, 1);
    device const float* weight = ARG_PTR(const float, 2);
    device const float* taps = ARG_PTR(const float, 3);
    const int n = ARG_INT(4), channels = ARG_INT(5), conv = ARG_INT(6), head_width = ARG_INT(7);
    const int tokens = ARG_INT(8);
    device float* out = ARG_PTR(float, 9);
    const int accumulate = ARG_INT(10);
    PER_NUMBER
    if (i >= n) return;
    float window[8], w[8];
    for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + i];
    for (int k = 0; k < conv; k++) w[k] = weight[(ulong)i * conv + k];
    const bool dynamic = NONNULL(taps);
    const int head = i / head_width, taps_row = n / head_width * conv;
    const float scale = 1.0f / sqrt((float)((dynamic ? 2 : 1) * conv + 1));
    for (int t = 0; t < tokens; t++) {
        const float xi = x[(ulong)t * n + i];
        window[conv - 1] = xi;
        float sum = xi;
        for (int k = 0; k < conv; k++) sum += w[k] * window[k];
        if (dynamic) {
            device const float* tp = taps + (ulong)t * taps_row + head * conv;
            for (int lag = 0; lag < conv; lag++) sum += tp[lag] * window[conv - 1 - lag];
        }
        device float* o = out + (ulong)t * n + i;
        *o = accumulate ? *o + sum * scale : sum * scale;
        for (int k = 0; k < conv - 1; k++) window[k] = window[k + 1];
    }
    for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + i] = window[k];
}

// Attention for heads of 128, split over the positions: threadgroup (head, part, token)
// takes its token's positions first + part × chunk on (chunk: the token's positions over
// the parts), its 4 SIMD groups a position at a time each, the lanes 4 of the head's
// numbers each. Token t is at position pos0 + t, sees positions from first (0, or for a
// sliding window of `window` the last `window`), with scores q·k × scale + r·bias[d] −
// sink, d the distance (row `extent` past the table). Caches are rings of `ring` rows (0:
// a row a position). It writes its largest score, the sum of exp(score − that) and the
// weighted values to `parts` (130 floats a part); ink_attention_merge adds the sink.
#define INK_HD 128
template <typename Cache>
kernel void ink_attention_part_t(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const Cache* keys = ARG_PTR(const Cache, 1);
    device const Cache* values = ARG_PTR(const Cache, 2);
    device float* parts = ARG_PTR(float, 3);
    device const float* r = ARG_PTR(const float, 4);
    device const float* bias = ARG_PTR(const float, 5);
    device const float* sink = ARG_PTR(const float, 6);
    const int pos0 = ARG_INT(7), window = ARG_INT(8), extent = ARG_INT(9), ring = ARG_INT(10);
    const int kv_dim = ARG_INT(11), group = ARG_INT(12), rd = ARG_INT(13);
    const float scale = ARG_F32(14);
    const int heads = ARG_INT(15);
    device float* out = ARG_PTR(float, 16);
    device const float* gate = ARG_PTR(const float, 17);
    const uint lane = tid & 31, warp = tid >> 5, warps = tpg.x >> 5;
    const int h = gid.x, part = gid.y, n_parts = ngrid.y, tok = gid.z;
    const int pos = pos0 + tok;
    const int first = window > 0 ? max(0, pos + 1 - window) : 0;
    const int count = pos + 1 - first;
    const int chunk = (count + n_parts - 1) / n_parts;
    const ulong kv = (ulong)(h / group) * INK_HD;
    const ulong qh = (ulong)tok * heads + h;
    const float4 qv = f4(q + qh * INK_HD + lane * 4);
    const float rv = lane < (uint)rd ? r[qh * rd + lane] : 0.0f;
    const float s0 = sink[qh];
    const float b_far = simd_sum(lane < (uint)rd ? rv * bias[(ulong)extent * rd + lane] : 0.0f) - s0;
    const int begin = part * chunk, end = min(count, begin + chunk);
    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    for (int i = begin + (int)warp; i < end; i += (int)warps) {
        const int t = first + i;
        const ulong row = (ulong)(ring > 0 ? t % ring : t);
        const int d = pos - t;
        const ulong kv_at = row * kv_dim + kv + lane * 4;
        const float4 k = sizeof(Cache) == 2
                             ? float4(*reinterpret_cast<device const half4*>(keys + kv_at))
                             : *reinterpret_cast<device const float4*>(keys + kv_at);
        float s;
        if (d >= extent) {
            s = simd_sum(dot(qv, k)) * scale + b_far;
        } else {
            const float b = lane < (uint)rd ? rv * bias[(ulong)d * rd + lane] : 0.0f;
            s = simd_sum(dot(qv, k) * scale + b) - s0;
        }
        const float m2 = max(m, s);
        const float c = exp(m - m2), p = exp(s - m2);
        l = l * c + p;
        const float4 v = sizeof(Cache) == 2
                             ? float4(*reinterpret_cast<device const half4*>(values + kv_at))
                             : *reinterpret_cast<device const float4*>(values + kv_at);
        acc = acc * c + p * v;
        m = m2;
    }
    threadgroup float tm[16], tl[16];
    threadgroup float4 tacc[16][32];
    if (lane == 0) {
        tm[warp] = m;
        tl[warp] = l;
    }
    tacc[warp][lane] = acc;
    BARRIER;
    if (warp != 0) return;
    float mm = -INFINITY;
    for (uint w = 0; w < warps; w++) mm = max(mm, tm[w]);
    float ll = 0.0f;
    float4 o = 0.0f;
    for (uint w = 0; w < warps; w++) {
        const float c = tm[w] == -INFINITY ? 0.0f : exp(tm[w] - mm);
        ll += tl[w] * c;
        o += tacc[w][lane] * c;
    }
    if (n_parts == 1) {
        // The whole context in this part: with the sink's logit of 0, times the gate.
        const float top = max(mm, 0.0f);
        const float c = mm == -INFINITY ? 0.0f : exp(mm - top);
        const float total = ll * c + exp(-top);
        const ulong at = qh * INK_HD + lane * 4;
        const float4 g = f4(gate + at);
        *reinterpret_cast<device float4*>(out + at) = o * c / total / (1.0f + exp(-g));
        return;
    }
    device float* p = parts + (qh * n_parts + part) * (INK_HD + 2);
    *reinterpret_cast<device float4*>(p + lane * 4) = o;
    if (lane == 0) {
        p[INK_HD] = mm;
        p[INK_HD + 1] = ll;
    }
}
template [[host_name("ink_attention_part")]] kernel void ink_attention_part_t<float>(ARGS, KP);
template [[host_name("ink_attention_part_f16")]] kernel void ink_attention_part_t<half>(ARGS, KP);

// Single-token fused QKVR prep (f16 KV cache): runs K conv+clip -> K cache,
// V conv+rms -> V cache, Q scale+clip+log_scale, and R scale+clip+log_scale in ONE dispatch.
// Grid: (2 * kv_heads + 2 * heads, 1), threads: 128.
kernel void ink_qkvr_prep_f16(ARGS, KP) {
    device const float* k_in = ARG_PTR(const float, 0);
    device const float* v_in = ARG_PTR(const float, 1);
    device float* q = ARG_PTR(float, 2);
    device float* r = ARG_PTR(float, 3);
    device float* history = ARG_PTR(float, 4);
    device const float* k_conv = ARG_PTR(const float, 5);
    device const float* v_conv = ARG_PTR(const float, 6);
    device const float* k_taps = ARG_PTR(const float, 7);
    device const float* v_taps = ARG_PTR(const float, 8);
    device half* k_cache = ARG_PTR(half, 9);
    device half* v_cache = ARG_PTR(half, 10);
    device const float* q_clip_w = ARG_PTR(const float, 11);
    device const float* r_clip_w = ARG_PTR(const float, 12);
    const int kv_heads = ARG_INT(13), heads = ARG_INT(14), rd = ARG_INT(15);
    const int channels = ARG_INT(16), conv = ARG_INT(17), pos = ARG_INT(18), ring = ARG_INT(19);
    const float qk_clip = ARG_F32(20), rel_clip = ARG_F32(21), eps = ARG_F32(22);
    const float log_mult = ARG_F32(23);
    threadgroup float scratch[32];
    const int g = (int)gid.x;
    const int kv_dim = kv_heads * INK_HD;
    const ulong cache_row = (ulong)(ring > 0 ? pos % ring : pos);
    if (g < kv_heads) {
        // K conv + L2 clip -> k_cache
        const int i = g * INK_HD + (int)tid;
        float window[8];
        for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + i];
        const float xi = k_in[i];
        window[conv - 1] = xi;
        float y = xi;
        device const float* wc = k_conv + (ulong)i * conv;
        for (int k = 0; k < conv; k++) y += wc[k] * window[k];
        const bool dyn = NONNULL(k_taps);
        if (dyn) {
            device const float* tp = k_taps + (ulong)g * conv;
            for (int lag = 0; lag < conv; lag++) y += tp[lag] * window[conv - 1 - lag];
        }
        y /= sqrt((float)((dyn ? 2 : 1) * conv + 1));
        const float sq = block_sum(y * y, scratch, tid, tpg.x);
        const float factor = clamp(qk_clip / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
        k_cache[cache_row * kv_dim + i] = half(y * factor);
        for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + i] = window[k + 1];
    } else if (g < 2 * kv_heads) {
        // V conv + RMS norm -> v_cache
        const int vh = g - kv_heads;
        const int i = vh * INK_HD + (int)tid;
        device float* vhist = history + kv_dim;
        float window[8];
        for (int k = 0; k < conv - 1; k++) window[k] = vhist[(ulong)k * channels + i];
        const float xi = v_in[i];
        window[conv - 1] = xi;
        float y = xi;
        device const float* wc = v_conv + (ulong)i * conv;
        for (int k = 0; k < conv; k++) y += wc[k] * window[k];
        const bool dyn = NONNULL(v_taps);
        if (dyn) {
            device const float* tp = v_taps + (ulong)vh * conv;
            for (int lag = 0; lag < conv; lag++) y += tp[lag] * window[conv - 1 - lag];
        }
        y /= sqrt((float)((dyn ? 2 : 1) * conv + 1));
        const float sq = block_sum(y * y, scratch, tid, tpg.x);
        const float factor = rsqrt(sq / (float)INK_HD + eps);
        v_cache[cache_row * kv_dim + i] = half(y * factor);
        for (int k = 0; k < conv - 1; k++) vhist[(ulong)k * channels + i] = window[k + 1];
    } else if (g < 2 * kv_heads + heads) {
        // Q clip + log_scale
        const int h = g - 2 * kv_heads;
        device float* row = q + (ulong)h * INK_HD;
        device const float* sr = q_clip_w + (ulong)h * INK_HD;
        const float v = row[tid] * sr[tid];
        const float sq = block_sum(v * v, scratch, tid, tpg.x);
        const float factor = clamp(qk_clip / (sqrt(sq) + 1e-6f), 0.0f, 1.0f) * log_mult;
        row[tid] = v * factor;
    } else {
        // R clip + log_scale (rd <= 32, 1 warp active)
        const int h = g - 2 * kv_heads - heads;
        if ((int)tid >= 32) return;
        device float* row = r + (ulong)h * rd;
        device const float* sr = r_clip_w + (ulong)h * rd;
        const float v = (int)tid < rd ? row[tid] * sr[tid] : 0.0f;
        const float sq = simd_sum(v * v);
        if ((int)tid < rd) {
            const float factor = clamp(rel_clip / (sqrt(sq) + 1e-6f), 0.0f, 1.0f) * log_mult;
            row[tid] = v * factor;
        }
    }
}

// GQA-grouped single-token decode attention over an f16 KV cache (heads of 128, group <= 4):
// threadgroup (kv_head, part) loads each K/V row once for all `group` query heads sharing it,
// and precomputes r·bias[extent] - sink once before the loop so far positions skip bias reads.
kernel void ink_attention_decode_f16(ARGS, KP) {
    device const float* q = ARG_PTR(const float, 0);
    device const half* keys = ARG_PTR(const half, 1);
    device const half* values = ARG_PTR(const half, 2);
    device float* parts = ARG_PTR(float, 3);
    device const float* r = ARG_PTR(const float, 4);
    device const float* bias = ARG_PTR(const float, 5);
    device const float* sink = ARG_PTR(const float, 6);
    const int pos = ARG_INT(7), window = ARG_INT(8), extent = ARG_INT(9), ring = ARG_INT(10);
    const int kv_dim = ARG_INT(11), group = ARG_INT(12), rd = ARG_INT(13);
    const float scale = ARG_F32(14);
    device float* out = ARG_PTR(float, 16);
    device const float* gate = ARG_PTR(const float, 17);
    const uint lane = tid & 31, warp = tid >> 5, warps = tpg.x >> 5;
    const int kv_head = gid.x, part = gid.y, n_parts = ngrid.y;
    const int first = window > 0 ? max(0, pos + 1 - window) : 0;
    const int count = pos + 1 - first;
    const int chunk = (count + n_parts - 1) / n_parts;
    const ulong kv = (ulong)kv_head * INK_HD;
    const int h0 = kv_head * group;
    const float b_ext = lane < (uint)rd ? bias[(ulong)extent * rd + lane] : 0.0f;
    float4 qv[4], acc[4];
    float rv[4], s0[4], b_far[4], m[4], l[4];
    for (int g = 0; g < group; g++) {
        const ulong h = (ulong)(h0 + g);
        qv[g] = f4(q + h * INK_HD + lane * 4);
        rv[g] = lane < (uint)rd ? r[h * rd + lane] : 0.0f;
        s0[g] = sink[h];
        b_far[g] = simd_sum(rv[g] * b_ext) - s0[g];
        m[g] = -INFINITY;
        l[g] = 0.0f;
        acc[g] = 0.0f;
    }
    const int begin = part * chunk, end = min(count, begin + chunk);
    for (int i = begin + (int)warp; i < end; i += (int)warps) {
        const int t = first + i;
        const ulong row = (ulong)(ring > 0 ? t % ring : t);
        const ulong kv_at = row * kv_dim + kv + lane * 4;
        const float4 k = float4(*reinterpret_cast<device const half4*>(keys + kv_at));
        const float4 v = float4(*reinterpret_cast<device const half4*>(values + kv_at));
        const int d = pos - t;
        if (d >= extent) {
            for (int g = 0; g < group; g++) {
                const float s = simd_sum(dot(qv[g], k)) * scale + b_far[g];
                const float m2 = max(m[g], s);
                const float c = exp(m[g] - m2), p = exp(s - m2);
                l[g] = l[g] * c + p;
                acc[g] = acc[g] * c + p * v;
                m[g] = m2;
            }
        } else {
            const float bd = lane < (uint)rd ? bias[(ulong)d * rd + lane] : 0.0f;
            for (int g = 0; g < group; g++) {
                const float s = simd_sum(dot(qv[g], k) * scale + rv[g] * bd) - s0[g];
                const float m2 = max(m[g], s);
                const float c = exp(m[g] - m2), p = exp(s - m2);
                l[g] = l[g] * c + p;
                acc[g] = acc[g] * c + p * v;
                m[g] = m2;
            }
        }
    }
    threadgroup float tm[16], tl[16];
    threadgroup float4 tacc[16][32];
    for (int g = 0; g < group; g++) {
        if (lane == 0) {
            tm[warp] = m[g];
            tl[warp] = l[g];
        }
        tacc[warp][lane] = acc[g];
        BARRIER;
        if (warp == 0) {
            float mm = -INFINITY;
            for (uint w = 0; w < warps; w++) mm = max(mm, tm[w]);
            float ll = 0.0f;
            float4 o = 0.0f;
            for (uint w = 0; w < warps; w++) {
                const float c = tm[w] == -INFINITY ? 0.0f : exp(tm[w] - mm);
                ll += tl[w] * c;
                o += tacc[w][lane] * c;
            }
            const ulong h = (ulong)(h0 + g);
            if (n_parts == 1) {
                const float top = max(mm, 0.0f);
                const float c = mm == -INFINITY ? 0.0f : exp(mm - top);
                const float total = ll * c + exp(-top);
                const ulong at = h * INK_HD + lane * 4;
                const float4 gv = f4(gate + at);
                *reinterpret_cast<device float4*>(out + at) = o * c / total / (1.0f + exp(-gv));
            } else {
                device float* p = parts + (h * n_parts + part) * (INK_HD + 2);
                *reinterpret_cast<device float4*>(p + lane * 4) = o;
                if (lane == 0) {
                    p[INK_HD] = mm;
                    p[INK_HD + 1] = ll;
                }
            }
        }
        if (g + 1 < group) BARRIER;
    }
}

// ink_attention_decode_f16 for long contexts, laid out as flash decoding: a SIMD group
// takes 32 / LPP positions at once, LPP lanes a position and DPL = 128 / LPP numbers of
// the head a lane (read 8 at a time), so a score needs a sum over LPP lanes, not 32.
// Each group of LPP lanes keeps its own running softmax; they're put together at the
// end, then the SIMD groups as before.
template <int LPP>
kernel void ink_attention_decode_lpp(ARGS, KP) {
    constexpr int DPL = INK_HD / LPP, PPW = 32 / LPP;
    device const float* q = ARG_PTR(const float, 0);
    device const half* keys = ARG_PTR(const half, 1);
    device const half* values = ARG_PTR(const half, 2);
    device float* parts = ARG_PTR(float, 3);
    device const float* r = ARG_PTR(const float, 4);
    device const float* bias = ARG_PTR(const float, 5);
    device const float* sink = ARG_PTR(const float, 6);
    const int pos = ARG_INT(7), window = ARG_INT(8), extent = ARG_INT(9), ring = ARG_INT(10);
    const int kv_dim = ARG_INT(11), group = ARG_INT(12), rd = ARG_INT(13);
    const float scale = ARG_F32(14);
    device float* out = ARG_PTR(float, 16);
    device const float* gate = ARG_PTR(const float, 17);
    const uint lane = tid & 31, warp = tid >> 5, warps = tpg.x >> 5;
    const int pg = (int)lane / LPP, sl = (int)lane % LPP;
    const int kv_head = gid.x, part = gid.y, n_parts = ngrid.y;
    const int first = window > 0 ? max(0, pos + 1 - window) : 0;
    const int count = pos + 1 - first;
    const int chunk = (count + n_parts - 1) / n_parts;
    const ulong kv = (ulong)kv_head * INK_HD + sl * DPL;
    const int h0 = kv_head * group;
    constexpr int RPL = (32 + LPP - 1) / LPP;  // bias numbers a lane, up to 32 in all
    float qv[4][DPL], acc[4][DPL];
    float s0[4], b_far[4], m[4], l[4], rv[4][RPL];
    for (int g = 0; g < group; g++) {
        const ulong h = (ulong)(h0 + g);
        for (int d = 0; d < DPL; d++) {
            qv[g][d] = q[h * INK_HD + sl * DPL + d] * scale;
            acc[g][d] = 0.0f;
        }
        float bf = 0.0f;
        for (int j = 0; j < RPL; j++) {
            const int d = sl + LPP * j;
            rv[g][j] = d < rd ? r[h * rd + d] : 0.0f;
            bf += d < rd ? rv[g][j] * bias[(ulong)extent * rd + d] : 0.0f;
        }
        for (int x = LPP / 2; x > 0; x >>= 1) bf += simd_shuffle_xor(bf, (ushort)x);
        s0[g] = sink[h];
        b_far[g] = bf - s0[g];
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const int begin = part * chunk, end = min(count, begin + chunk);
    for (int i = begin + (int)warp * PPW + pg; i < end; i += (int)warps * PPW) {
        const int t = first + i;
        const ulong row = (ulong)(ring > 0 ? t % ring : t);
        device const uint4* kp = reinterpret_cast<device const uint4*>(keys + row * kv_dim + kv);
        device const uint4* vp = reinterpret_cast<device const uint4*>(values + row * kv_dim + kv);
        float kf[DPL], vf[DPL];
        for (int c = 0; c < DPL / 8; c++) {
            const uint4 kw = kp[c], vw = vp[c];
            const half4 ka = as_type<half4>(kw.xy), kb = as_type<half4>(kw.zw);
            const half4 va = as_type<half4>(vw.xy), vb = as_type<half4>(vw.zw);
            kf[8 * c + 0] = ka.x; kf[8 * c + 1] = ka.y; kf[8 * c + 2] = ka.z; kf[8 * c + 3] = ka.w;
            kf[8 * c + 4] = kb.x; kf[8 * c + 5] = kb.y; kf[8 * c + 6] = kb.z; kf[8 * c + 7] = kb.w;
            vf[8 * c + 0] = va.x; vf[8 * c + 1] = va.y; vf[8 * c + 2] = va.z; vf[8 * c + 3] = va.w;
            vf[8 * c + 4] = vb.x; vf[8 * c + 5] = vb.y; vf[8 * c + 6] = vb.z; vf[8 * c + 7] = vb.w;
        }
        const int dd = pos - t;
        const bool far = dd >= extent;
        for (int g = 0; g < group; g++) {
            float sp = 0.0f;
            for (int d = 0; d < DPL; d++) sp += qv[g][d] * kf[d];
            if (!far) {
                for (int j = 0; j < RPL; j++) {
                    const int d = sl + LPP * j;
                    sp += d < rd ? rv[g][j] * bias[(ulong)dd * rd + d] : 0.0f;
                }
            }
            for (int x = LPP / 2; x > 0; x >>= 1) sp += simd_shuffle_xor(sp, (ushort)x);
            const float sc = far ? sp + b_far[g] : sp - s0[g];
            const float m2 = max(m[g], sc);
            const float c = exp(m[g] - m2), p = exp(sc - m2);
            l[g] = l[g] * c + p;
            for (int d = 0; d < DPL; d++) acc[g][d] = acc[g][d] * c + p * vf[d];
            m[g] = m2;
        }
    }
    // The position groups of each SIMD group together (lanes sl, sl + LPP, ...).
    for (int g = 0; g < group; g++) {
        for (int x = LPP; x < 32; x <<= 1) {
            const float mo = simd_shuffle_xor(m[g], (ushort)x);
            const float lo = simd_shuffle_xor(l[g], (ushort)x);
            const float mm = max(m[g], mo);
            const float ca = m[g] == -INFINITY ? 0.0f : exp(m[g] - mm);
            const float cb = mo == -INFINITY ? 0.0f : exp(mo - mm);
            for (int d = 0; d < DPL; d++) {
                const float ao = simd_shuffle_xor(acc[g][d], (ushort)x);
                acc[g][d] = acc[g][d] * ca + ao * cb;
            }
            l[g] = l[g] * ca + lo * cb;
            m[g] = mm;
        }
    }
    threadgroup float tm[16], tl[16];
    threadgroup float tacc[16][INK_HD];
    for (int g = 0; g < group; g++) {
        if (lane == 0) {
            tm[warp] = m[g];
            tl[warp] = l[g];
        }
        if (pg == 0) {
            for (int d = 0; d < DPL; d++) tacc[warp][sl * DPL + d] = acc[g][d];
        }
        BARRIER;
        if (warp == 0) {
            float mm = -INFINITY;
            for (uint w = 0; w < warps; w++) mm = max(mm, tm[w]);
            float ll = 0.0f;
            float4 o = 0.0f;
            for (uint w = 0; w < warps; w++) {
                const float c = tm[w] == -INFINITY ? 0.0f : exp(tm[w] - mm);
                ll += tl[w] * c;
                o += *reinterpret_cast<threadgroup const float4*>(&tacc[w][lane * 4]) * c;
            }
            const ulong h = (ulong)(h0 + g);
            if (n_parts == 1) {
                const float top = max(mm, 0.0f);
                const float c = mm == -INFINITY ? 0.0f : exp(mm - top);
                const float total = ll * c + exp(-top);
                const ulong at = h * INK_HD + lane * 4;
                const float4 gv = f4(gate + at);
                *reinterpret_cast<device float4*>(out + at) = o * c / total / (1.0f + exp(-gv));
            } else {
                device float* p = parts + (h * n_parts + part) * (INK_HD + 2);
                *reinterpret_cast<device float4*>(p + lane * 4) = o;
                if (lane == 0) {
                    p[INK_HD] = mm;
                    p[INK_HD + 1] = ll;
                }
            }
        }
        if (g + 1 < group) BARRIER;
    }
}
template [[host_name("ink_attention_decode8_f16")]] kernel void ink_attention_decode_lpp<8>(ARGS, KP);

// ink_attention_part's parts of each (head, token) (threadgroup (head, token), 128 threads)
// put together with the sink's logit of 0, times sigmoid(gate), into the token's row of
// heads × 128.
kernel void ink_attention_merge(ARGS, KP) {
    device const float* parts = ARG_PTR(const float, 0);
    device float* out = ARG_PTR(float, 1);
    const int n_parts = ARG_INT(2), heads = ARG_INT(3);
    device const float* gate = ARG_PTR(const float, 4);
    const ulong qh = (ulong)gid.y * heads + gid.x;
    device const float* p = parts + qh * n_parts * (INK_HD + 2);
    float mm = 0.0f;  // the sink
    for (int k = 0; k < n_parts; k++) mm = max(mm, p[k * (INK_HD + 2) + INK_HD]);
    float total = exp(-mm), v = 0.0f;
    for (int k = 0; k < n_parts; k++) {
        device const float* pk = p + k * (INK_HD + 2);
        if (pk[INK_HD] == -INFINITY) continue;
        const float c = exp(pk[INK_HD] - mm);
        total += pk[INK_HD + 1] * c;
        v += pk[tid] * c;
    }
    const ulong at = qh * INK_HD + tid;
    out[at] = v / total / (1.0f + exp(-gate[at]));
}

// One token's router and expert pick in one dispatch (in place of matvec_split_f32 then
// Single-token fused F32 router matvec + expert pick:
// threadgroup `row` computes that expert's score; the last threadgroup picks top-k.
kernel void ink_router_pick(ARGS, KP) {
    device const float* w = ARG_PTR(const float, 0);
    device const float* x = ARG_PTR(const float, 1);
    device float* lg = ARG_PTR(float, 2);
    const int cols = ARG_INT(4);
    device const float* bias = ARG_PTR(const float, 5);
    device int* idx = ARG_PTR(int, 6);
    device float* share = ARG_PTR(float, 7);
    const int experts = ARG_INT(8), used = ARG_INT(9);
    const float scale = ARG_F32(10);
    device atomic_uint* counter = ARG_PTR(atomic_uint, 11);
    const int row = (int)gid.x;
    const int sgi = (int)(tid >> 5), lane = (int)(tid & 31);
    device const float* r = w + (ulong)row * cols;
    float sum = 0.0f;
    for (int c = (sgi * 32 + lane) * 4; c < cols; c += 512) {
        sum += dot(f4(r + c), f4(x + c));
    }
    sum = simd_sum(sum);
    threadgroup float parts[4];
    threadgroup bool last;
    if (lane == 0) parts[sgi] = sum;
    BARRIER;
    if (tid == 0) lg[row] = parts[0] + parts[1] + parts[2] + parts[3];
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    BARRIER;
    if (tid == 0) {
        const uint done = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        last = done == ngrid.x - 1;
        if (last) atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
    BARRIER;
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup int tix[MOE_MAX_USED];
    threadgroup float tp[MOE_MAX_USED];
    threadgroup float sel[512];
    for (int k = tid; k < used; k += tpg.x) {
        tix[k] = k;
        tp[k] = 0.0f;
    }
    for (int e = tid; e < experts; e += tpg.x) sel[e] = 1.0f / (1.0f + exp(-lg[e])) + bias[e];
    BARRIER;
    for (int e = tid; e < experts; e += tpg.x) {
        const float v = sel[e];
        int place = 0;
        for (int j = 0; j < experts; j++) {
            const float o = sel[j];
            place += (o > v || (o == v && j < e)) ? 1 : 0;
        }
        if (place < used) {
            tix[place] = e;
            tp[place] = 1.0f / (1.0f + exp(-lg[e]));
        }
    }
    BARRIER;
    if (tid != 0) return;
    float total = 1e-20f;
    for (int k = 0; k < used; k++) total += tp[k];
    for (int k = 0; k < used; k++) {
        idx[k] = tix[k];
        share[k] = tp[k] / total * scale;
    }
}

// Inkling 2's DSpark drafter: row r of `out` = RMSNorm(Σ_k w_k features[k]) × norm, the four
// feature blocks `block` floats apart (a threadgroup a row).
kernel void dspark_mix(ARGS, KP) {
    device const float* f = ARG_PTR(const float, 0);
    device float* out = ARG_PTR(float, 1);
    device const float* norm = ARG_PTR(const float, 2);
    const int dim = ARG_INT(3), block = ARG_INT(4);
    const float eps = ARG_F32(5);
    const float w0 = ARG_F32(6), w1 = ARG_F32(7), w2 = ARG_F32(8), w3 = ARG_F32(9);
    threadgroup float scratch[32];
    const ulong r = gid.x;
    device const float* f0 = f + r * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += tpg.x) {
        const float v = w0 * f0[i] + w1 * f0[block + i] + w2 * f0[2 * (ulong)block + i] + w3 * f0[3 * (ulong)block + i];
        out[r * dim + i] = v;
        sum += v * v;
    }
    sum = block_sum(sum, scratch, tid, tpg.x);
    const float scale = 1.0f / sqrt(sum / (float)dim + eps);
    for (int i = tid; i < dim; i += tpg.x) out[r * dim + i] = out[r * dim + i] * scale * norm[i];
}

// The DSpark block's queries regrouped for one attention call: `rows` rows of `kv_heads ×
// group` heads ([row][kv][g]) to [kv][row][g] (`back`: the other way), so the rows' heads
// of one key/value head sit together and one call reads each key once.
kernel void dspark_permute(ARGS, KP) {
    device const float* src = ARG_PTR(const float, 0);
    device float* dst = ARG_PTR(float, 1);
    const int rows = ARG_INT(2), kv_heads = ARG_INT(3), group = ARG_INT(4), hd = ARG_INT(5);
    const int back = ARG_INT(6);
    const int total = rows * kv_heads * group * hd;
    PER_NUMBER
    if (i >= total) return;
    const int d = i % hd, head = i / hd;
    const int g = head % group, rest = head / group;
    // `i` indexes the [row][kv][g] layout: rest = row * kv_heads + kv.
    const int kv = rest % kv_heads, row = rest / kv_heads;
    const int j = ((kv * rows + row) * group + g) * hd + d;
    if (back) dst[i] = src[j];
    else dst[j] = src[i];
}

// The DSpark recurrent head's step: from j = joint_proj · [...] + b (3 × rank), state =
// tanh(c) + sigmoid(g) (state − tanh(c)) and latent = tanh(z).
kernel void dspark_head_act(ARGS, KP) {
    device const float* j = ARG_PTR(const float, 0);
    device float* state = ARG_PTR(float, 1);
    device float* latent = ARG_PTR(float, 2);
    const int rank = ARG_INT(3);
    for (int i = tid; i < rank; i += tpg.x) {
        const float gate = 1.0f / (1.0f + exp(-j[i]));
        const float cand = tanh(j[rank + i]);
        state[i] = cand + gate * (state[i] - cand);
        latent[i] = tanh(j[2 * rank + i]);
    }
}

// The experts' pick for each token (a threadgroup each): sigmoid scores, the `used` highest
// after adding `bias` (selection only), their scores normalized to sum 1 (+1e-20) and times
// `scale`.
kernel void ink_moe_pick(ARGS, KP) {
    device const float* logits = ARG_PTR(const float, 0);
    device const float* bias = ARG_PTR(const float, 1);
    device int* idx = ARG_PTR(int, 2);
    device float* share = ARG_PTR(float, 3);
    const int experts = ARG_INT(4), used = ARG_INT(5);
    const float scale = ARG_F32(6);
    threadgroup int tix[MOE_MAX_USED];
    threadgroup float tp[MOE_MAX_USED];
    threadgroup float sel[512];
    device const float* lg = logits + (ulong)gid.x * experts;
    for (int k = tid; k < used; k += tpg.x) {
        tix[k] = k;
        tp[k] = 0.0f;
    }
    for (int e = tid; e < experts; e += tpg.x) sel[e] = 1.0f / (1.0f + exp(-lg[e])) + bias[e];
    BARRIER;
    for (int e = tid; e < experts; e += tpg.x) {
        const float v = sel[e];
        int place = 0;
        for (int j = 0; j < experts; j++) {
            const float o = sel[j];
            place += (o > v || (o == v && j < e)) ? 1 : 0;
        }
        if (place < used) {
            tix[place] = e;
            tp[place] = 1.0f / (1.0f + exp(-lg[e]));
        }
    }
    BARRIER;
    if (tid != 0) return;
    float total = 1e-20f;
    for (int k = 0; k < used; k++) total += tp[k];
    device int* ix = idx + (ulong)gid.x * used;
    device float* sh = share + (ulong)gid.x * used;
    for (int k = 0; k < used; k++) {
        ix[k] = tix[k];
        sh[k] = tp[k] / total * scale;
    }
}

// SiTU for each slot: act = beta tanh(g/beta) sigmoid(g) × lb tanh(u/lb), g and u the two
// halves of the slot's row of `gate_up` (2 × width), into row `slot` of `act` (`stride`
// apart; it may be gate_up itself, stride 2 × width).
kernel void ink_situ(ARGS, KP) {
    device const float* gu = ARG_PTR(const float, 0);
    device float* act = ARG_PTR(float, 1);
    const int width = ARG_INT(2), total = ARG_INT(3);
    const float beta = ARG_F32(4), lb = ARG_F32(5);
    const int stride = ARG_INT(6);
    PER_NUMBER
    if (i >= total) return;
    const int slot = i / width, j = i % width;
    const float g = gu[(ulong)slot * 2 * width + j], u = gu[(ulong)slot * 2 * width + width + j];
    act[(ulong)slot * stride + j] = beta * tanh(g / beta) / (1.0f + exp(-g)) * lb * tanh(u / lb);
}

// The same into f16 rows of `width` (moe_matmul_fast's input, as moe_geglu_half).
kernel void ink_situ_half(ARGS, KP) {
    device const float* gu = ARG_PTR(const float, 0);
    device half* act = ARG_PTR(half, 1);
    const int width = ARG_INT(2), total = ARG_INT(3);
    const float beta = ARG_F32(4), lb = ARG_F32(5);
    PER_NUMBER
    if (i >= total) return;
    const int slot = i / width, j = i % width;
    const float g = gu[(ulong)slot * 2 * width + j], u = gu[(ulong)slot * 2 * width + width + j];
    act[i] = half(beta * tanh(g / beta) / (1.0f + exp(-g)) * lb * tanh(u / lb));
}

// A layer's keys or values for a batch, a threadgroup a head of 128 (a thread a channel):
// each token's projection in `x` (rows of kv_dim) through the causal convolution (as
// ink_conv, with input-dependent `taps` if given), then clipped to an L2 norm of
// `threshold` (`mode` 0: keys) or RMS-normed with `threshold` as eps (1: values), into the
// token's row of the cache (a ring of `ring` rows, or 0 for a row a position). The history
// moves along a token at a time.
template <typename Cache>
kernel void ink_kv_t(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device float* history = ARG_PTR(float, 1);
    device const float* weight = ARG_PTR(const float, 2);
    device const float* taps = ARG_PTR(const float, 3);
    device Cache* cache = ARG_PTR(Cache, 4);
    const int channels = ARG_INT(5), conv = ARG_INT(6), tokens = ARG_INT(7), pos0 = ARG_INT(8);
    const int ring = ARG_INT(9), kv_dim = ARG_INT(10), mode = ARG_INT(11);
    const float threshold = ARG_F32(12);
    threadgroup float scratch[32];
    const int head = gid.x, i = head * INK_HD + tid;
    const int kv_heads = kv_dim / INK_HD;
    float window[8], w[8];
    for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + i];
    for (int k = 0; k < conv; k++) w[k] = weight[(ulong)i * conv + k];
    const bool dynamic = NONNULL(taps);
    const float scale = 1.0f / sqrt((float)((dynamic ? 2 : 1) * conv + 1));
    for (int t = 0; t < tokens; t++) {
        const float xi = x[(ulong)t * kv_dim + i];
        window[conv - 1] = xi;
        float y = xi;
        for (int k = 0; k < conv; k++) y += w[k] * window[k];
        if (dynamic) {
            device const float* tp = taps + ((ulong)t * kv_heads + head) * conv;
            for (int lag = 0; lag < conv; lag++) y += tp[lag] * window[conv - 1 - lag];
        }
        y *= scale;
        const float sq = block_sum(y * y, scratch, tid, tpg.x);
        const float factor = mode == 0 ? clamp(threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f)
                                       : 1.0f / sqrt(sq / (float)INK_HD + threshold);
        const int pos = pos0 + t;
        const ulong row = (ulong)(ring > 0 ? pos % ring : pos);
        cache[row * kv_dim + i] = Cache(y * factor);
        for (int k = 0; k < conv - 1; k++) window[k] = window[k + 1];
    }
    for (int k = 0; k < conv - 1; k++) history[(ulong)k * channels + i] = window[k];
}
template [[host_name("ink_kv")]] kernel void ink_kv_t<float>(ARGS, KP);
template [[host_name("ink_kv_f16")]] kernel void ink_kv_t<half>(ARGS, KP);

// A batch's Inkling 2 attention on the SIMD-group matrix units (the default mode): as
// attention_flash (D = 128, 32 tokens a threadgroup, keys and values 16 at a time, 4 SIMD
// groups), with each score q·k × scale + r·bias[d] − sink (d the distance, row `extent`
// past the table), the sink's logit of 0 as the softmax's starting point (max 0, sum 1,
// O 0), and the output times sigmoid(gate). Threadgroup (head, token block).
template <typename Cache>
kernel void ink_attention_flash_t(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr int D = 128, QT = 32, C = 16, RD = 32;
    device const float* q = ARG_PTR(const float, 0);
    device const Cache* keys = ARG_PTR(const Cache, 1);
    device const Cache* values = ARG_PTR(const Cache, 2);
    device float* out = ARG_PTR(float, 3);
    device const float* r = ARG_PTR(const float, 4);
    device const float* bias = ARG_PTR(const float, 5);
    device const float* sink = ARG_PTR(const float, 6);
    device const float* gate = ARG_PTR(const float, 7);
    const int p0 = ARG_INT(8), window = ARG_INT(9), extent = ARG_INT(10), ring = ARG_INT(11);
    const int kv_dim = ARG_INT(12), group = ARG_INT(13), rd = ARG_INT(14);
    const float scale = ARG_F32(15);
    const int tokens = ARG_INT(16);
    constexpr int THREADS = 128, BLOCKS = QT / 8 * (D / 8) / 4;
    constexpr int ROW_THREADS = THREADS / QT < C ? THREADS / QT : C, PER_THREAD = C / ROW_THREADS;
    threadgroup half qkv[QT * D + 2 * C * D];
    threadgroup half* sq = qkv;
    threadgroup half* sk = qkv + QT * D;
    threadgroup half* sv = sk + C * D;
    threadgroup float ss[QT * C];
    threadgroup half sp[QT * C];
    threadgroup float row_max[QT], row_sum[QT];
    threadgroup float diag[QT / 8][64];
    threadgroup float sr_bias[QT][RD];
    threadgroup float sr_sink[QT];
    threadgroup float sr_far[QT];
    const int head = gid.x, heads = ngrid.x;
    const int t0 = gid.y * QT;
    const int kv = (head / group) * D;
    const int last_row = p0 + min(t0 + QT, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    for (int i = tid; i < QT * D; i += THREADS) {
        const int row = i / D, d = i % D;
        sq[i] = t0 + row < tokens ? half(q[((ulong)(t0 + row) * heads + head) * D + d]) : half(0.0f);
    }
    for (int i = tid; i < QT * RD; i += THREADS) {
        const int row = i / RD, j = i % RD;
        sr_bias[row][j] = t0 + row < tokens && j < rd ? r[((ulong)(t0 + row) * heads + head) * rd + j] : 0.0f;
    }
    for (int i = tid; i < QT / 8 * 64; i += THREADS) diag[i / 64][i % 64] = 0.0f;
    if ((int)tid < QT) {
        // The sink: a logit of 0 with no value.
        row_max[tid] = 0.0f;
        row_sum[tid] = 1.0f;
        const float s0 = t0 + (int)tid < tokens ? sink[(ulong)(t0 + tid) * heads + head] : 0.0f;
        sr_sink[tid] = s0;
        float fb = 0.0f;
        if (t0 + (int)tid < tokens) {
            device const float* rr = r + ((ulong)(t0 + tid) * heads + head) * rd;
            device const float* be = bias + (ulong)extent * rd;
            for (int k = 0; k < rd; k++) fb += rr[k] * be[k];
        }
        sr_far[tid] = fb - s0;
    }
    simdgroup_float8x8 o[BLOCKS];
    for (int b = 0; b < BLOCKS; b++) o[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int srow = tid / ROW_THREADS, sc0 = (tid % ROW_THREADS) * PER_THREAD;
    const int s_at = p0 + t0 + srow;
    const bool s_live = t0 + srow < tokens;
    const int s_first = window > 0 ? max(0, s_at + 1 - window) : 0;

    for (int start = lo; start <= last_row; start += C) {
        const int count = min(C, last_row + 1 - start);
        BARRIER;
        for (int i = tid; i < C * D / 4; i += THREADS) {
            const int row = (i * 4) / D, d = (i * 4) % D;
            half4 k4 = half4(0.0h), v4 = half4(0.0h);
            if (row < count) {
                const ulong at = ring_row(start + row, ring) * kv_dim + kv + d;
                if constexpr (sizeof(Cache) == 2) {
                    k4 = *reinterpret_cast<device const half4*>(keys + at);
                    v4 = *reinterpret_cast<device const half4*>(values + at);
                } else {
                    k4 = half4(*reinterpret_cast<device const float4*>(keys + at));
                    v4 = half4(*reinterpret_cast<device const float4*>(values + at));
                }
            }
            *reinterpret_cast<threadgroup half4*>(sk + row * D + d) = k4;
            *reinterpret_cast<threadgroup half4*>(sv + row * D + d) = v4;
        }
        BARRIER;
        for (int tile = sg; tile < QT / 8 * (C / 8); tile += 4) {
            const int rb = tile / (C / 8), kb = tile % (C / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d = 0; d < D; d += 8) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, sq + rb * 8 * D + d, D);
                simdgroup_load(b, sk + kb * 8 * D + d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, ss + rb * 8 * C + kb * 8, C);
        }
        BARRIER;
        if ((int)tid < QT * ROW_THREADS) {
            float s[PER_THREAD];
            float mx = -INFINITY;
            const float far_off = sr_far[srow];
            for (int j = 0; j < PER_THREAD; j++) {
                const int c = sc0 + j, kpos = start + c;
                const bool seen = s_live && c < count && kpos <= s_at && kpos >= s_first;
                if (!seen) {
                    s[j] = -INFINITY;
                } else {
                    const int dist = s_at - kpos;
                    if (dist >= extent) {
                        s[j] = ss[srow * C + c] * scale + far_off;
                    } else {
                        device const float* br = bias + (ulong)dist * rd;
                        float b = 0.0f;
                        for (int k = 0; k < rd; k++) b += sr_bias[srow][k] * br[k];
                        s[j] = ss[srow * C + c] * scale + b - sr_sink[srow];
                    }
                }
                mx = max(mx, s[j]);
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o2));
            const float old_max = row_max[srow];
            const float new_max = max(old_max, mx);
            const float correction = exp(old_max - new_max);
            float sum = 0.0f;
            for (int j = 0; j < PER_THREAD; j++) {
                const float p = s[j] == -INFINITY ? 0.0f : exp(s[j] - new_max);
                sp[srow * C + sc0 + j] = half(p);
                sum += p;
            }
            for (int o2 = ROW_THREADS / 2; o2 > 0; o2 >>= 1) sum += simd_shuffle_xor(sum, (ushort)o2);
            if (tid % ROW_THREADS == 0) {
                row_max[srow] = new_max;
                row_sum[srow] = row_sum[srow] * correction + sum;
                diag[srow / 8][(srow % 8) * 9] = correction;
            }
        }
        BARRIER;
        for (int b = 0; b < BLOCKS; b++) {
            const int blk = sg * BLOCKS + b, rb = blk / (D / 8), cb = blk % (D / 8);
            simdgroup_float8x8 dg;
            simdgroup_load(dg, &diag[rb][0], 8);
            simdgroup_multiply(o[b], dg, o[b]);
            for (int kc = 0; kc < C; kc += 8) {
                simdgroup_half8x8 p, v;
                simdgroup_load(p, sp + rb * 8 * C + kc, C);
                simdgroup_load(v, sv + kc * D + cb * 8, D);
                simdgroup_multiply_accumulate(o[b], p, v, o[b]);
            }
        }
    }
    BARRIER;
    threadgroup float* so = reinterpret_cast<threadgroup float*>(qkv);
    for (int b = 0; b < BLOCKS; b++) {
        const int blk = sg * BLOCKS + b, rb = blk / (D / 8), cb = blk % (D / 8);
        simdgroup_store(o[b], so + rb * 8 * D + cb * 8, D);
    }
    BARRIER;
    for (int i = tid; i < QT * D; i += THREADS) {
        const int row = i / D, d = i % D;
        if (t0 + row < tokens) {
            const ulong at = ((ulong)(t0 + row) * heads + head) * D + d;
            out[at] = so[i] / row_sum[row] / (1.0f + exp(-gate[at]));
        }
    }
}
template [[host_name("ink_attention_flash")]] kernel void ink_attention_flash_t<float>(ARGS, KP, uint, uint);
template [[host_name("ink_attention_flash_f16")]] kernel void ink_attention_flash_t<half>(ARGS, KP, uint, uint);

// ink_conv for a batch with all tokens at once, a thread a (channel, token): each output
// reads its own last `conv` inputs (from x's rows, or `history` before the batch), so
// `out` must not be x. ink_conv_history moves the history along after.
kernel void ink_conv_parallel(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device const float* history = ARG_PTR(const float, 1);
    device const float* weight = ARG_PTR(const float, 2);
    device const float* taps = ARG_PTR(const float, 3);
    const int n = ARG_INT(4), channels = ARG_INT(5), conv = ARG_INT(6), head_width = ARG_INT(7);
    const int tokens = ARG_INT(8);
    device float* out = ARG_PTR(float, 9);
    const int accumulate = ARG_INT(10);
    const int i = gid.x * tpg.x + tid, t = gid.y;
    if (i >= n || t >= tokens) return;
    const bool dynamic = NONNULL(taps);
    const int head = i / head_width, taps_row = n / head_width * conv;
    // Input k (0 the oldest, conv − 1 this token's).
    auto input = [&](int k) {
        const int at = t - (conv - 1) + k;
        return at >= 0 ? x[(ulong)at * n + i] : history[(ulong)(at + conv - 1) * channels + i];
    };
    const float xi = x[(ulong)t * n + i];
    float sum = xi;
    for (int k = 0; k < conv; k++) sum += weight[(ulong)i * conv + k] * input(k);
    if (dynamic) {
        device const float* tp = taps + (ulong)t * taps_row + head * conv;
        for (int lag = 0; lag < conv; lag++) sum += tp[lag] * input(conv - 1 - lag);
    }
    const float y = sum / sqrt((float)((dynamic ? 2 : 1) * conv + 1));
    device float* o = out + (ulong)t * n + i;
    *o = accumulate ? *o + y : y;
}

// The history after a batch of `tokens` rows of `x` (rows of n): its last conv − 1 inputs.
kernel void ink_conv_history(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device float* history = ARG_PTR(float, 1);
    const int n = ARG_INT(2), channels = ARG_INT(3), conv = ARG_INT(4), tokens = ARG_INT(5);
    PER_NUMBER
    if (i >= n) return;
    float window[8];
    for (int k = 0; k < conv - 1; k++) window[k] = history[(ulong)k * channels + i];
    for (int k = 0; k < conv - 1; k++) {
        const int at = tokens - (conv - 1) + k;  // the input that ends up in place k
        history[(ulong)k * channels + i] = at >= 0 ? x[(ulong)at * n + i] : window[k + tokens];
    }
}

// ink_kv for a batch with all tokens at once: a threadgroup a (head, token), a thread a
// channel. The cache and `x` differ, so each output reads its own inputs; ink_conv_history
// moves the history along after.
template <typename Cache>
kernel void ink_kv_parallel_t(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device const float* history = ARG_PTR(const float, 1);
    device const float* weight = ARG_PTR(const float, 2);
    device const float* taps = ARG_PTR(const float, 3);
    device Cache* cache = ARG_PTR(Cache, 4);
    const int channels = ARG_INT(5), conv = ARG_INT(6), pos0 = ARG_INT(8);
    const int ring = ARG_INT(9), kv_dim = ARG_INT(10), mode = ARG_INT(11);
    const float threshold = ARG_F32(12);
    threadgroup float scratch[32];
    const int head = gid.x, t = gid.y, i = head * INK_HD + tid;
    const int kv_heads = kv_dim / INK_HD;
    auto input = [&](int k) {
        const int at = t - (conv - 1) + k;
        return at >= 0 ? x[(ulong)at * kv_dim + i] : history[(ulong)(at + conv - 1) * channels + i];
    };
    const float xi = x[(ulong)t * kv_dim + i];
    float y = xi;
    for (int k = 0; k < conv; k++) y += weight[(ulong)i * conv + k] * input(k);
    const bool dynamic = NONNULL(taps);
    if (dynamic) {
        device const float* tp = taps + ((ulong)t * kv_heads + head) * conv;
        for (int lag = 0; lag < conv; lag++) y += tp[lag] * input(conv - 1 - lag);
    }
    y /= sqrt((float)((dynamic ? 2 : 1) * conv + 1));
    const float sq = block_sum(y * y, scratch, tid, tpg.x);
    const float factor = mode == 0 ? clamp(threshold / (sqrt(sq) + 1e-6f), 0.0f, 1.0f)
                                   : 1.0f / sqrt(sq / (float)INK_HD + threshold);
    cache[ring_row(pos0 + t, ring) * kv_dim + i] = Cache(y * factor);
}
template [[host_name("ink_kv_parallel")]] kernel void ink_kv_parallel_t<float>(ARGS, KP);
template [[host_name("ink_kv_parallel_f16")]] kernel void ink_kv_parallel_t<half>(ARGS, KP);
