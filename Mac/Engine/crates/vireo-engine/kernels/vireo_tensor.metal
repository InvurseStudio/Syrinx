// Vireo's matrix products on the GPU's neural accelerators (Metal 4's tensor operations:
// M5, A19 and later). Compiled after vireo.metal, into the same library, when the GPU has
// them (src/metal/mod.rs); without them the backend uses matmul_sg in vireo.metal.
//
// The default (fast) mode's matrix products for prompts: each threadgroup decodes a tile
// of TC_ROWS weight rows, TC_K columns at a time, to f16 in threadgroup memory (decode8 in
// vireo.metal), and one matmul2d over its 4 SIMD groups multiplies it with TC_TOKENS
// tokens' activations, rounded to f16 beforehand (to_half, the Mac's counterpart of
// CUDA's quantize_q8), adding up in f32. Exact mode keeps matmul_sg's f32.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// The tile (set by the backend, which sizes the grid to match): weight rows × tokens,
// and the SIMD groups sharing it.
#ifndef TC_ROWS
#define TC_ROWS 32
#endif
#ifndef TC_TOKENS
#define TC_TOKENS 128
#endif
#ifndef TC_SG
#define TC_SG 4
#endif
#define TC_K 64
#define TC_THREADS (32 * TC_SG)

template <int KIND>
kernel void matmul_tc(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device half* x = ARG_PTR(half, 1);  // the activations in f16 (to_half)
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    threadgroup half sa[TC_ROWS * TC_K];  // the weights' tile: [row][k]
    const int row0 = gid.x * TC_ROWS, tok0 = gid.y * TC_TOKENS;
    // Decoding: the tile's runs of 8 numbers (TC_K / 8 a row) shared out among the threads.
    constexpr int RUNS = TC_K / 8, PER = TC_ROWS * RUNS / TC_THREADS;

    // C (tokens × rows) = X (tokens × k) · Wᵀ; extents list the contiguous dimension first.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TC_TOKENS, TC_ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sa, dextents<int32_t, 2>(TC_K, TC_ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, cols}));
    auto acc = mm.get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += TC_K) {
        float v[8];
        for (int part = 0; part < PER; part++) {
            const int run = tid + part * TC_THREADS, lr = run / RUNS, lk = (run % RUNS) * 8;
            if (row0 + lr < rows && k0 + lk < cols) {
                decode8<KIND>(w + (ulong)(row0 + lr) * row_bytes, cols / 32, k0 + lk, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) sa[lr * TC_K + lk + i] = half(v[i]);
        }
        BARRIER;
        const int k_len = min(TC_K, cols - k0);
        auto tile_w = tensor(sa, dextents<int32_t, 2>(k_len, TC_ROWS), array<int, 2>({1, TC_K}));
        auto tile_x = tensor(x + (ulong)tok0 * cols + k0, dextents<int32_t, 2>(k_len, tokens - tok0),
                             array<int, 2>({1, cols}));
        mm.run(tile_x, tile_w, acc);
        BARRIER;
    }
    // y holds a row of `rows` per token: C's rows are y's, the contiguous dimension rows.
    auto t_y = tensor(y + (ulong)tok0 * rows + row0,
                      dextents<int32_t, 2>(min(TC_ROWS, rows - row0), min(TC_TOKENS, tokens - tok0)),
                      array<int, 2>({1, rows}));
    acc.store(t_y);
}

template [[host_name("matmul_tc_f32")]] kernel void matmul_tc<K_F32>(ARGS, KP);
template [[host_name("matmul_tc_f16")]] kernel void matmul_tc<K_F16>(ARGS, KP);
template [[host_name("matmul_tc_q4_0")]] kernel void matmul_tc<K_Q4_0>(ARGS, KP);
template [[host_name("matmul_tc_q8_0")]] kernel void matmul_tc<K_Q8_0>(ARGS, KP);
template [[host_name("matmul_tc_q4_k")]] kernel void matmul_tc<K_Q4_K>(ARGS, KP);
template [[host_name("matmul_tc_q5_k")]] kernel void matmul_tc<K_Q5_K>(ARGS, KP);
template [[host_name("matmul_tc_q6_k")]] kernel void matmul_tc<K_Q6_K>(ARGS, KP);

// The activations rounded to f16 for matmul_tc: 4 numbers a thread (n a multiple of 4).
kernel void to_half(ARGS, KP) {
    device const float4* x = ARG_PTR(const float4, 0);
    device half4* out = ARG_PTR(half4, 1);
    const int n4 = ARG_INT(2) / 4;
    const int i = gid.x * tpg.x + tid;
    if (i < n4) out[i] = half4(x[i]);
}

// A prompt's attention over the f16 cache on the neural accelerators: the counterpart of
// attention_flash (vireo.metal) with S = Q Kᵀ and O += P V as matmul2d operations. A
// threadgroup of 8 SIMD groups takes one query head and AQ tokens; the keys and values
// are read straight from the cache, AC positions at a time. The scores go to threadgroup
// memory for the online softmax (each SIMD group takes AQ / 8 rows, a lane AC / 32 of a
// row's columns; masked causally and by the sliding window from the positions), and O
// stays in a cooperative tensor, its rows rescaled as the maximum moves. The last chunk can
// read up to AC − 1 positions past the context (into the buffer's slack, see
// Context::new_buffer): their probabilities are 0.
template <int D, int AQ>
kernel void attention_tc(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device half* keys = ARG_PTR(half, 1);
    device half* values = ARG_PTR(half, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5), kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const int tokens = ARG_INT(10);
    constexpr int AC = 64, SG = 8, THREADS = 32 * SG, ROWS = AQ / SG, COLS = AC / 32;
    threadgroup half sq[AQ * D];
    threadgroup float ss[AQ * AC];
    threadgroup half sp[AQ * AC];
    threadgroup float sr[AQ];  // each row's rescaling this chunk, and its sum at the end
    const int head = gid.x, heads = ngrid.x;
    const int t0 = gid.y * AQ, p0 = *pos;
    const int kv = (head / group) * D;
    const int last_row = p0 + min(t0 + AQ, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    for (int i = tid; i < AQ * D; i += THREADS) {
        const int r = i / D, d = i % D;
        sq[i] = t0 + r < tokens ? half(q[((ulong)(t0 + r) * heads + head) * D + d]) : half(0.0f);
    }
    auto tq = tensor(sq, dextents<int32_t, 2>(D, AQ));
    auto ts = tensor(ss, dextents<int32_t, 2>(AC, AQ));
    auto tp = tensor(sp, dextents<int32_t, 2>(AC, AQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AC, D, false, true, false,
                                                                   mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<SG>>
        mm_qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(AQ, D, AC, false, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SG>>
        mm_pv;
    auto tv0 = tensor(values + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (uint16_t i = 0; i < co.get_capacity(); i++)
        if (co.is_valid_element(i)) co[i] = 0.0f;
    float row_max[ROWS], row_sum[ROWS];
    for (int jj = 0; jj < ROWS; jj++) {
        row_max[jj] = -INFINITY;
        row_sum[jj] = 0.0f;
    }
    BARRIER;

    for (int start = lo; start <= last_row; start += AC) {
        const int count = min(AC, last_row + 1 - start);
        auto tk = tensor(keys + (ulong)start * kv_dim + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
        mm_qk.run(tq, tk, ts);
        BARRIER;
        for (int jj = 0; jj < ROWS; jj++) {
            const int j = jj * SG + sg, at = p0 + t0 + j;
            const bool live = t0 + j < tokens;
            const int first = window > 0 ? max(0, at + 1 - window) : 0;
            float s[COLS];
            float mx = -INFINITY;
            for (int ii = 0; ii < COLS; ii++) {
                const int c = ii * 32 + lane, kpos = start + c;
                const bool seen = live && c < count && kpos <= at && kpos >= first;
                s[ii] = seen ? ss[j * AC + c] * scale : -INFINITY;
                mx = max(mx, s[ii]);
            }
            mx = simd_max(mx);
            const float new_max = max(row_max[jj], mx);
            const float correction = new_max == -INFINITY ? 1.0f : exp(row_max[jj] - new_max);
            float sum = 0.0f;
            for (int ii = 0; ii < COLS; ii++) {
                const half p = half(s[ii] == -INFINITY ? 0.0f : exp(s[ii] - new_max));
                sp[j * AC + ii * 32 + lane] = p;
                sum += float(p);
            }
            row_sum[jj] = row_sum[jj] * correction + simd_sum(sum);
            row_max[jj] = new_max;
            if (lane == 0) sr[j] = correction;
        }
        BARRIER;
        for (uint16_t i = 0; i < co.get_capacity(); i++)
            if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        auto tv = tensor(values + (ulong)start * kv_dim + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
        mm_pv.run(tp, tv, co);
        BARRIER;
    }
    for (int jj = 0; jj < ROWS; jj++) {
        const int j = jj * SG + sg;
        if (lane == 0) sr[j] = row_sum[jj] == 0.0f ? 0.0f : 1.0f / row_sum[jj];
    }
    BARRIER;
    for (uint16_t i = 0; i < co.get_capacity(); i++)
        if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
    auto td = tensor(out + ((ulong)t0 * heads + head) * D, dextents<int32_t, 2>(D, tokens - t0),
                     array<int, 2>({1, heads * D}));
    co.store(td);
}

template [[host_name("attention_tc_64")]] kernel void attention_tc<64, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_128")]] kernel void attention_tc<128, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_256")]] kernel void attention_tc<256, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_512")]] kernel void attention_tc<512, 16>(ARGS, KP, uint, uint);
