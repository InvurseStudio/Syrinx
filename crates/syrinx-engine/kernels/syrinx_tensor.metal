// Vireo's matrix products on the GPU's neural accelerators (Metal 4's tensor operations:
// M5, A19 and later). Compiled after syrinx.metal, into the same library, when the GPU has
// them (src/metal/mod.rs); without them the backend uses matmul_sg in syrinx.metal.
//
// The default (fast) mode's matrix products for prompts: each threadgroup decodes a tile
// of TC_ROWS weight rows, TC_K columns at a time, to f16 in threadgroup memory (decode8 in
// syrinx.metal), and one matmul2d over its 4 SIMD groups multiplies it with TC_TOKENS
// tokens' activations, rounded to f16 beforehand (to_half, the Mac's counterpart of
// CUDA's quantize_q8), adding up in f32. Exact mode keeps matmul_sg's f32.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// The tile (set by the backend, which sizes the grid to match): weight rows × tokens,
// and the SIMD groups sharing it. Each kernel comes four times: matmul_tc_* with TC_TOKENS
// tokens a tile, matmul_tc_wide_* with TC_WIDE for a prompt's whole batches (a large
// model's weights decoded half as often), and matmul_tc_t8_* to matmul_tc_t192_* for
// short batches (SMALL_TILES in src/metal/mod.rs: no multiplying of padding). ROWS and KK
// are the tile's rows and its columns a step (TC_ROWS and TC_K but for the 8- to 64-token
// tiles, which take 256 columns a step). BUFS is how many tiles of weights the
// threadgroup holds: two, decoded in turn, so a step's decoding doesn't wait for the
// last step's product (128 columns a step: 16 KB); one for the 256-column tiles, where
// two are the whole 32 KB a threadgroup may have and cost far more than they saved (a
// check of 8 tokens on Gemma 4 31B: 107 ms with two, 70 with one).
#ifndef TC_ROWS
#define TC_ROWS 32
#endif
#ifndef TC_TOKENS
#define TC_TOKENS 128
#endif
#ifndef TC_WIDE
#define TC_WIDE 256
#endif
#ifndef TC_SG
#define TC_SG 4
#endif
#ifndef TC_K
#define TC_K 128
#endif
#define TC_THREADS (32 * TC_SG)

template <int KIND, int TOK, int ROWS = TC_ROWS, int KK = TC_K, int BUFS = 2>
kernel void matmul_tc(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device half* x = ARG_PTR(half, 1);  // the activations in f16 (to_half)
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    // Split along the columns (gid.z of `splits`, each a whole number of KK steps): a
    // short batch of a matrix with few rows and long ones (ffn_down, attn_output) has too
    // few threadgroups otherwise. Each split's sums go to its own rows of `parts`, which
    // sum_parts adds up in order.
    const int splits = ARG_INT(6);
    const int chunk = ((cols + KK - 1) / KK + splits - 1) / splits * KK;
    const int k_begin = (int)gid.z * chunk, k_end = min(cols, k_begin + chunk);
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    threadgroup half sab[BUFS * ROWS * KK];  // the weights' tile, [row][k]: with two, a step's decoded while the last's may still be read
    const int row0 = gid.x * ROWS, tok0 = gid.y * TOK;
    // Decoding: the tile's runs of 32 numbers (KK / 32 a row) shared out among the threads.
    constexpr int RUNS = KK / 32, PER = (ROWS * RUNS + TC_THREADS - 1) / TC_THREADS;

    // C (tokens × rows) = X (tokens × k) · Wᵀ; extents list the contiguous dimension first.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sab, dextents<int32_t, 2>(KK, ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, cols}));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    int flip = 0;
    for (int k0 = k_begin; k0 < k_end; k0 += KK, flip ^= 1) {
        threadgroup half* sa = sab + (BUFS > 1 ? flip : 0) * (ROWS * KK);
        for (int part = 0; part < PER; part++) {
            const int run = tid + part * TC_THREADS, lr = run / RUNS, lk = (run % RUNS) * 32;
            if (lr >= ROWS) break;
            threadgroup half* dst = sa + lr * KK + lk;
            if (row0 + lr < rows && k0 + lk < cols) {
                decode32<KIND>(w + (ulong)(row0 + lr) * row_bytes, cols / 32, k0 + lk, dst);
            } else {
                for (int i = 0; i < 32; i++) dst[i] = half(0.0f);
            }
        }
        BARRIER;
        const int k_len = min(KK, cols - k0);
        auto tile_w = tensor(sa, dextents<int32_t, 2>(k_len, ROWS), array<int, 2>({1, KK}));
        auto tile_x = tensor(x + (ulong)tok0 * cols + k0, dextents<int32_t, 2>(k_len, tokens - tok0),
                             array<int, 2>({1, cols}));
        mm.run(tile_x, tile_w, acc);
        if (BUFS == 1) BARRIER;
    }
    // y holds a row of `rows` per token: C's rows are y's, the contiguous dimension rows.
    if (splits > 1) y = ARG_PTR(float, 7) + (ulong)gid.z * rows * tokens;
    auto t_y = tensor(y + (ulong)tok0 * rows + row0,
                      dextents<int32_t, 2>(min(ROWS, rows - row0), min(TOK, tokens - tok0)),
                      array<int, 2>({1, rows}));
    acc.store(t_y);
}

template [[host_name("matmul_tc_f32")]] kernel void matmul_tc<K_F32, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_f16")]] kernel void matmul_tc<K_F16, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_q4_0")]] kernel void matmul_tc<K_Q4_0, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_q8_0")]] kernel void matmul_tc<K_Q8_0, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_q4_k")]] kernel void matmul_tc<K_Q4_K, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_q5_k")]] kernel void matmul_tc<K_Q5_K, TC_TOKENS>(ARGS, KP);
template [[host_name("matmul_tc_q6_k")]] kernel void matmul_tc<K_Q6_K, TC_TOKENS>(ARGS, KP);
// 8 to 64 tokens: 256 columns a step and one tile of weights (see SMALL_TILES in
// src/metal/mod.rs).
template [[host_name("matmul_tc_t8_f32")]] kernel void matmul_tc<K_F32, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_f16")]] kernel void matmul_tc<K_F16, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_q4_0")]] kernel void matmul_tc<K_Q4_0, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_q8_0")]] kernel void matmul_tc<K_Q8_0, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_q4_k")]] kernel void matmul_tc<K_Q4_K, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_q5_k")]] kernel void matmul_tc<K_Q5_K, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t8_q6_k")]] kernel void matmul_tc<K_Q6_K, 8, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_f32")]] kernel void matmul_tc<K_F32, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_f16")]] kernel void matmul_tc<K_F16, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_q4_0")]] kernel void matmul_tc<K_Q4_0, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_q8_0")]] kernel void matmul_tc<K_Q8_0, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_q4_k")]] kernel void matmul_tc<K_Q4_K, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_q5_k")]] kernel void matmul_tc<K_Q5_K, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t16_q6_k")]] kernel void matmul_tc<K_Q6_K, 16, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_f32")]] kernel void matmul_tc<K_F32, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_f16")]] kernel void matmul_tc<K_F16, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_q4_0")]] kernel void matmul_tc<K_Q4_0, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_q8_0")]] kernel void matmul_tc<K_Q8_0, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_q4_k")]] kernel void matmul_tc<K_Q4_K, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_q5_k")]] kernel void matmul_tc<K_Q5_K, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t32_q6_k")]] kernel void matmul_tc<K_Q6_K, 32, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_f32")]] kernel void matmul_tc<K_F32, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_f16")]] kernel void matmul_tc<K_F16, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_q4_0")]] kernel void matmul_tc<K_Q4_0, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_q8_0")]] kernel void matmul_tc<K_Q8_0, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_q4_k")]] kernel void matmul_tc<K_Q4_K, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_q5_k")]] kernel void matmul_tc<K_Q5_K, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t64_q6_k")]] kernel void matmul_tc<K_Q6_K, 64, TC_ROWS, 256, 1>(ARGS, KP);
template [[host_name("matmul_tc_t96_f32")]] kernel void matmul_tc<K_F32, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_f16")]] kernel void matmul_tc<K_F16, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_q4_0")]] kernel void matmul_tc<K_Q4_0, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_q8_0")]] kernel void matmul_tc<K_Q8_0, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_q4_k")]] kernel void matmul_tc<K_Q4_K, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_q5_k")]] kernel void matmul_tc<K_Q5_K, 96>(ARGS, KP);
template [[host_name("matmul_tc_t96_q6_k")]] kernel void matmul_tc<K_Q6_K, 96>(ARGS, KP);
template [[host_name("matmul_tc_t192_f32")]] kernel void matmul_tc<K_F32, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_f16")]] kernel void matmul_tc<K_F16, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_q4_0")]] kernel void matmul_tc<K_Q4_0, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_q8_0")]] kernel void matmul_tc<K_Q8_0, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_q4_k")]] kernel void matmul_tc<K_Q4_K, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_q5_k")]] kernel void matmul_tc<K_Q5_K, 192>(ARGS, KP);
template [[host_name("matmul_tc_t192_q6_k")]] kernel void matmul_tc<K_Q6_K, 192>(ARGS, KP);
template [[host_name("matmul_tc_wide_f32")]] kernel void matmul_tc<K_F32, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_f16")]] kernel void matmul_tc<K_F16, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_q4_0")]] kernel void matmul_tc<K_Q4_0, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_q8_0")]] kernel void matmul_tc<K_Q8_0, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_q4_k")]] kernel void matmul_tc<K_Q4_K, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_q5_k")]] kernel void matmul_tc<K_Q5_K, TC_WIDE>(ARGS, KP);
template [[host_name("matmul_tc_wide_q6_k")]] kernel void matmul_tc<K_Q6_K, TC_WIDE>(ARGS, KP);

// matmul_tc for one tile of moe_group's (syrinx.metal): its expert's rows × its slots, the
// tile a threadgroup row (gid.y). The slots' inputs and products lie in moe_group's order
// (moe_gather_half), so a tile's are one after another: `x` rows of f16 `x_stride` apart,
// `y` rows of f32 `y_stride` apart. Arguments: w, x, y, rows (an expert's), cols, tiles,
// x_stride, y_stride.
#ifndef MOE_TC_TOKENS
#define MOE_TC_TOKENS 64
#endif
template <int KIND, int TOK, int ROWS = TC_ROWS, int KK = TC_K>
kernel void moe_matmul_tc(ARGS, KP) {
    const int rows = ARG_INT(3), cols = ARG_INT(4);
    device const int* tile = ARG_PTR(const int, 5) + 3 * gid.y;
    const int x_stride = ARG_INT(6), y_stride = ARG_INT(7);
    const int tokens = tile[2];
    if (tokens == 0) return;
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    device const u8* w = ARG_PTR(const u8, 0) + (ulong)tile[0] * (ulong)rows * row_bytes;
    device half* x = ARG_PTR(half, 1) + (ulong)tile[1] * x_stride;
    device float* y = ARG_PTR(float, 2) + (ulong)tile[1] * y_stride;
    threadgroup half sab[2 * ROWS * KK];
    const int row0 = gid.x * ROWS;
    constexpr int RUNS = KK / 32, PER = (ROWS * RUNS + TC_THREADS - 1) / TC_THREADS;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sab, dextents<int32_t, 2>(KK, ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, x_stride}));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    int flip = 0;
    for (int k0 = 0; k0 < cols; k0 += KK, flip ^= 1) {
        threadgroup half* sa = sab + flip * (ROWS * KK);
        for (int part = 0; part < PER; part++) {
            const int run = tid + part * TC_THREADS, lr = run / RUNS, lk = (run % RUNS) * 32;
            if (lr >= ROWS) break;
            threadgroup half* dst = sa + lr * KK + lk;
            if (row0 + lr < rows && k0 + lk < cols) {
                decode32<KIND>(w + (ulong)(row0 + lr) * row_bytes, cols / 32, k0 + lk, dst);
            } else {
                for (int i = 0; i < 32; i++) dst[i] = half(0.0f);
            }
        }
        BARRIER;
        const int k_len = min(KK, cols - k0);
        auto tile_w = tensor(sa, dextents<int32_t, 2>(k_len, ROWS), array<int, 2>({1, KK}));
        auto tile_x = tensor(x + k0, dextents<int32_t, 2>(k_len, tokens), array<int, 2>({1, x_stride}));
        mm.run(tile_x, tile_w, acc);
    }
    auto t_y = tensor(y + row0, dextents<int32_t, 2>(min(ROWS, rows - row0), tokens), array<int, 2>({1, y_stride}));
    acc.store(t_y);
}
template [[host_name("moe_matmul_tc_f32")]] kernel void moe_matmul_tc<K_F32, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_f16")]] kernel void moe_matmul_tc<K_F16, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_q4_0")]] kernel void moe_matmul_tc<K_Q4_0, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_q8_0")]] kernel void moe_matmul_tc<K_Q8_0, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_q4_k")]] kernel void moe_matmul_tc<K_Q4_K, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_q5_k")]] kernel void moe_matmul_tc<K_Q5_K, MOE_TC_TOKENS>(ARGS, KP);
template [[host_name("moe_matmul_tc_q6_k")]] kernel void moe_matmul_tc<K_Q6_K, MOE_TC_TOKENS>(ARGS, KP);

// matmul_tc's splits added up, in order: y[i] = parts[0][i] + parts[1][i] + ...
kernel void sum_parts(ARGS, KP) {
    device const float* parts = ARG_PTR(const float, 0);
    device float* y = ARG_PTR(float, 1);
    const int n = ARG_INT(2), splits = ARG_INT(3);
    const int i = gid.x * tpg.x + tid;
    if (i >= n) return;
    float total = parts[i];
    for (int z = 1; z < splits; z++) total += parts[(ulong)z * n + i];
    y[i] = total;
}

// The activations rounded to f16 for matmul_tc: 4 numbers a thread (n a multiple of 4).
kernel void to_half(ARGS, KP) {
    device const float4* x = ARG_PTR(const float4, 0);
    device half4* out = ARG_PTR(half4, 1);
    const int n4 = ARG_INT(2) / 4;
    const int i = gid.x * tpg.x + tid;
    if (i < n4) out[i] = half4(x[i]);
}

// A prompt's attention over the f16 cache on the neural accelerators: the counterpart of
// attention_flash (syrinx.metal) with S = Q Kᵀ and O += P V as matmul2d operations. A
// threadgroup of 8 SIMD groups takes one query head and AQ tokens; the keys and values
// are read straight from the cache, AC positions at a time. The scores go to threadgroup
// memory for the online softmax (each SIMD group takes AQ / 8 rows, a lane AC / 32 of a
// row's columns; masked causally and by the sliding window from the positions), and O
// stays in a cooperative tensor, its rows rescaled as the maximum moves. The last chunk can
// read up to AC − 1 positions past the context (into the buffer's slack, see
// Context::new_buffer): their probabilities are 0. In a ring (a sliding-window layer's
// last positions only, see ring_row in syrinx.metal) the chunks start at multiples of AC, as
// the ring's size is one, so no chunk runs over the ring's end; the positions before the
// window that brings in are masked the same way.
template <int D, int AQ>
kernel void attention_tc(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device const float* q = ARG_PTR(const float, 0);
    device half* keys = ARG_PTR(half, 1);
    device half* values = ARG_PTR(half, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const float scale_log2 = scale * 1.4426950408889634f;
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
    constexpr int CO_MAX = 64;
    short co_row[CO_MAX];
    // A thread's elements past CO_MAX (none for today's sizes) take the slow way: asked
    // their row each time.
    const int co_cap = (int)co.get_capacity(), co_n = min(co_cap, CO_MAX);
    for (int i = 0; i < co_n; i++) {
        const bool valid = co.is_valid_element(i);
        co_row[i] = valid ? (short)co.get_multidimensional_index(i)[1] : (short)-1;
        if (valid) co[i] = 0.0f;
    }
    for (int i = co_n; i < co_cap; i++)
        if (co.is_valid_element(i)) co[i] = 0.0f;
    float row_max[ROWS], row_sum[ROWS];
    for (int jj = 0; jj < ROWS; jj++) {
        row_max[jj] = -INFINITY;
        row_sum[jj] = 0.0f;
    }
    BARRIER;

    for (int start = ring > 0 ? lo - lo % AC : lo; start <= last_row; start += AC) {
        const int count = min(AC, last_row + 1 - start);
        auto tk = tensor(keys + ring_row(start, ring) * kv_dim + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
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
                s[ii] = seen ? ss[j * AC + c] * scale_log2 : -INFINITY;
                mx = max(mx, s[ii]);
            }
            mx = simd_max(mx);
            const float new_max = max(row_max[jj], mx);
            const float correction = new_max == -INFINITY ? 1.0f : metal::fast::exp2(row_max[jj] - new_max);
            float sum = 0.0f;
            for (int ii = 0; ii < COLS; ii++) {
                const half p = half(s[ii] == -INFINITY ? 0.0f : metal::fast::exp2(s[ii] - new_max));
                sp[j * AC + ii * 32 + lane] = p;
                sum += float(p);
            }
            row_sum[jj] = row_sum[jj] * correction + simd_sum(sum);
            row_max[jj] = new_max;
            if (lane == 0) sr[j] = correction;
        }
        BARRIER;
        for (int i = 0; i < co_n; i++) {
            if (co_row[i] < 0) continue;
            const float fix = sr[co_row[i]];
            if (fix != 1.0f) co[i] *= fix;
        }
        for (int i = co_n; i < co_cap; i++)
            if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        auto tv = tensor(values + ring_row(start, ring) * kv_dim + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
        mm_pv.run(tp, tv, co);
        BARRIER;
    }
    for (int jj = 0; jj < ROWS; jj++) {
        const int j = jj * SG + sg;
        if (lane == 0) sr[j] = row_sum[jj] == 0.0f ? 0.0f : 1.0f / row_sum[jj];
    }
    BARRIER;
    for (int i = 0; i < co_n; i++) {
        if (co_row[i] >= 0) co[i] *= sr[co_row[i]];
    }
    for (int i = co_n; i < co_cap; i++)
        if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
    auto td = tensor(out + ((ulong)t0 * heads + head) * D, dextents<int32_t, 2>(D, tokens - t0),
                     array<int, 2>({1, heads * D}));
    co.store(td);
}

template [[host_name("attention_tc_64")]] kernel void attention_tc<64, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_128")]] kernel void attention_tc<128, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_256")]] kernel void attention_tc<256, 32>(ARGS, KP, uint, uint);
template [[host_name("attention_tc_512")]] kernel void attention_tc<512, 16>(ARGS, KP, uint, uint);

// The queries rounded to f16 and regrouped for attention_tcg: from [token][head] to
// [key/value head][token][head in its group], so a key/value head's rows are contiguous
// across tokens. 4 numbers a thread.
kernel void q_group_half(ARGS, KP) {
    device const float4* q = ARG_PTR(const float4, 0);
    device half4* out = ARG_PTR(half4, 1);
    const int heads = ARG_INT(2), group = ARG_INT(3), d4 = ARG_INT(4) / 4, tokens = ARG_INT(5);
    const ulong i = (ulong)gid.x * tpg.x + tid;
    if (i >= (ulong)tokens * heads * d4) return;
    const int d = (int)(i % d4);
    const int row = (int)(i / d4), head = row % heads, t = row / heads;
    const int g = head / group, h = head % group;
    out[(((ulong)g * tokens + t) * group + h) * d4 + d] = half4(q[i]);
}

// attention_tc with a key/value head's query heads together: a threadgroup takes one
// key/value head and R rows of its queries, R / group tokens of the group's heads each,
// from q_group_half's f16 copy in device memory (so the tile isn't bound by threadgroup
// memory). Each key and value chunk is read once for the group, AC positions at a time:
// on the M5 Max the accelerators run long chunks faster (heads of 512 at 30K positions:
// 16 rows × 64 positions a head 21.6 TFLOPS, 16 × 256 a group 32.8). The rows left over
// when the group doesn't divide R (and those past the last token) are dead: they read
// what follows in the buffer and get probabilities of 0. Chunks start at multiples of AC.
// In a ring (a multiple of AC) none runs over its end; otherwise the last is moved back
// to end at the last position, the positions it repeats masked, so nothing is read past
// the context (which must hold AC positions: see the backend).
template <int D, int R, int AC, int SG>
kernel void attention_tcg(ARGS, KP, uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    device half* qh = ARG_PTR(half, 0);
    device half* keys = ARG_PTR(half, 1);
    device half* values = ARG_PTR(half, 2);
    device float* out = ARG_PTR(float, 3);
    device const int* pos = ARG_PTR(const int, 4);
    const int window = ARG_INT(5) & 0xFFFF, ring = ARG_INT(5) >> 16, kv_dim = ARG_INT(7), group = ARG_INT(8);
    const float scale = ARG_F32(9);
    const float scale_log2 = scale * 1.4426950408889634f;
    const int tokens = ARG_INT(10);
    constexpr int ROWS = R / SG, COLS = AC / 32;
    threadgroup float ss[R * AC];
    threadgroup half* sp = (threadgroup half*)ss;
    threadgroup float sr[R];
    threadgroup atomic_uint any_fix;
    const int g = gid.x, heads = ngrid.x * group;
    const int tb = R / group, live_rows = tb * group;
    const int t0 = gid.y * tb, p0 = *pos;
    const int kv = g * D;
    const int last_row = p0 + min(t0 + tb, tokens) - 1;
    const int lo = window > 0 ? max(0, p0 + t0 + 1 - window) : 0;

    auto tq = tensor(qh + ((ulong)g * tokens + t0) * group * D, dextents<int32_t, 2>(D, R));
    auto ts = tensor(ss, dextents<int32_t, 2>(AC, R));
    auto tp = tensor(sp, dextents<int32_t, 2>(AC, R));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(R, AC, D, false, true, false,
                                                                   mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<SG>>
        mm_qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(R, D, AC, false, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SG>>
        mm_pv;
    auto tv0 = tensor(values + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    // Each of this thread's numbers of the output: its row, or -1 for one not in use.
    constexpr int CO_MAX = 64;
    short co_row[CO_MAX];
    // A thread's elements past CO_MAX (none for today's sizes) take the slow way: asked
    // their row each time.
    const int co_cap = (int)co.get_capacity(), co_n = min(co_cap, CO_MAX);
    for (int i = 0; i < co_n; i++) {
        const bool valid = co.is_valid_element(i);
        co_row[i] = valid ? (short)co.get_multidimensional_index(i)[1] : (short)-1;
        if (valid) co[i] = 0.0f;
    }
    for (int i = co_n; i < co_cap; i++)
        if (co.is_valid_element(i)) co[i] = 0.0f;
    float row_max[ROWS], row_sum[ROWS];
    for (int jj = 0; jj < ROWS; jj++) {
        row_max[jj] = -INFINITY;
        row_sum[jj] = 0.0f;
    }

    // A short batch's context in parts (gid.z of ngrid.z, whole chunks each) so every GPU
    // core has work: each part's maxima, sums and sums of values go to `partial`, which
    // attention_tcg_merge adds up.
    device float* partial = ARG_PTR(float, 11);
    const int parts = ngrid.z, first_chunk = lo - lo % AC;
    const int chunks = (last_row - first_chunk) / AC + 1, share = (chunks + parts - 1) / parts;
    const int begin = first_chunk + (int)gid.z * share * AC;
    const int end = parts > 1 ? min(last_row, begin + share * AC - 1) : last_row;
    for (int start = begin; start <= end; start += AC) {
        const int from = ring > 0 ? start : min(start, max(0, last_row + 1 - AC));
        const ulong at_cache = ring_row(from, ring) * kv_dim + kv;
        auto tk = tensor(keys + at_cache, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
        mm_qk.run(tq, tk, ts);
        if (tid == 0) atomic_store_explicit(&any_fix, 0u, memory_order_relaxed);
        BARRIER;
        half p_reg[ROWS][COLS];
        bool local_fix = false;
        for (int jj = 0; jj < ROWS; jj++) {
            const int j = jj * SG + sg, t = t0 + j / group, at = p0 + t;
            const bool live = j < live_rows && t < tokens;
            const int first = max(start, window > 0 ? at + 1 - window : 0);
            float s[COLS];
            float mx = -INFINITY;
            if (live && from >= first && from + AC - 1 <= at) {
                for (int ii = 0; ii < COLS; ii++) {
                    s[ii] = ss[j * AC + ii * 32 + lane] * scale_log2;
                    mx = max(mx, s[ii]);
                }
            } else {
                for (int ii = 0; ii < COLS; ii++) {
                    const int c = ii * 32 + lane, kpos = from + c;
                    const bool seen = live && kpos >= first && kpos <= at;
                    s[ii] = seen ? ss[j * AC + c] * scale_log2 : -INFINITY;
                    mx = max(mx, s[ii]);
                }
            }
            mx = simd_max(mx);
            const float new_max = max(row_max[jj], mx);
            const float correction = new_max == -INFINITY ? 1.0f : metal::fast::exp2(row_max[jj] - new_max);
            float sum = 0.0f;
            if (new_max == -INFINITY) {
                for (int ii = 0; ii < COLS; ii++) p_reg[jj][ii] = half(0.0f);
            } else {
                for (int ii = 0; ii < COLS; ii++) {
                    const half p = half(metal::fast::exp2(s[ii] - new_max));
                    p_reg[jj][ii] = p;
                    sum += float(p);
                }
            }
            row_sum[jj] = row_sum[jj] * correction + simd_sum(sum);
            row_max[jj] = new_max;
            if (lane == 0) {
                sr[j] = correction;
                if (correction != 1.0f) local_fix = true;
            }
        }
        if (local_fix) atomic_store_explicit(&any_fix, 1u, memory_order_relaxed);
        BARRIER;
        for (int jj = 0; jj < ROWS; jj++) {
            const int j = jj * SG + sg;
            for (int ii = 0; ii < COLS; ii++) sp[j * AC + ii * 32 + lane] = p_reg[jj][ii];
        }
        BARRIER;
        // Rescaled only where a row's maximum moved: deep in a context most don't.
        if (atomic_load_explicit(&any_fix, memory_order_relaxed) != 0u) {
            for (int i = 0; i < co_n; i++) {
                if (co_row[i] < 0) continue;
                const float fix = sr[co_row[i]];
                if (fix != 1.0f) co[i] *= fix;
            }
            for (int i = co_n; i < co_cap; i++)
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
        auto tv = tensor(values + at_cache, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
        mm_pv.run(tp, tv, co);
        BARRIER;
    }
    if (parts > 1) {
        device float* mine = partial + (((ulong)g * ngrid.y + gid.y) * parts + gid.z) * R * (2 + D);
        for (int jj = 0; jj < ROWS; jj++) {
            const int j = jj * SG + sg;
            if (lane == 0) {
                mine[j * (2 + D)] = row_max[jj];
                mine[j * (2 + D) + 1] = row_sum[jj];
            }
        }
        for (uint16_t i = 0; i < co.get_capacity(); i++) {
            if (!co.is_valid_element(i)) continue;
            const auto at = co.get_multidimensional_index(i);
            mine[at[1] * (2 + D) + 2 + at[0]] = co[i];
        }
        return;
    }
    for (int jj = 0; jj < ROWS; jj++) {
        const int j = jj * SG + sg;
        if (lane == 0) sr[j] = row_sum[jj] == 0.0f ? 0.0f : 1.0f / row_sum[jj];
    }
    BARRIER;
    for (uint16_t i = 0; i < co.get_capacity(); i++) {
        if (!co.is_valid_element(i)) continue;
        const auto at = co.get_multidimensional_index(i);
        const int d = at[0], j = at[1], t = t0 + j / group;
        if (j < live_rows && t < tokens) out[((ulong)t * heads + g * group + j % group) * D + d] = co[i] * sr[j];
    }
}

// The tiles (TCG_TILES in src/metal/mod.rs), the quickest on the M5 Max of 16-64 rows,
// 64-256 positions and 4 or 8 SIMD groups: each leaves a thread 32 of the output's numbers.
// Heads of 512 (Gemma 4 31B's global layers) keep 16 rows × 256 positions: 32 × 128 was
// 2-30% slower at 1K-30K positions and 7-13% at 32K-256K, with 64 to 1,024 tokens a batch
// (Nov 19, `prompt_attention_speed`, the two alternated with cool-downs between).
template [[host_name("attention_tcg_128")]] kernel void attention_tcg<128, 32, 128, 4>(ARGS, KP, uint, uint);
template [[host_name("attention_tcg_256")]] kernel void attention_tcg<256, 32, 128, 8>(ARGS, KP, uint, uint);
template [[host_name("attention_tcg_512")]] kernel void attention_tcg<512, 16, 256, 8>(ARGS, KP, uint, uint);

// attention_tcg's parts added up: a threadgroup a query row (gid.x: key/value head and
// block of tokens, gid.y: the row in its tile), each part weighted by its maximum against
// the largest. Arguments: partial, out, tokens, heads, group, head size, tile rows, parts.
kernel void attention_tcg_merge(ARGS, KP) {
    device const float* partial = ARG_PTR(const float, 0);
    device float* out = ARG_PTR(float, 1);
    const int tokens = ARG_INT(2), heads = ARG_INT(3), group = ARG_INT(4), D = ARG_INT(5), R = ARG_INT(6);
    const int parts = ARG_INT(7);
    const int kv_heads = heads / group, blocks = (int)ngrid.x / kv_heads;
    const int g = (int)gid.x / blocks, block = (int)gid.x % blocks, j = gid.y;
    const int tb = R / group, t = block * tb + j / group;
    if (j >= tb * group || t >= tokens) return;
    const ulong stride = (ulong)R * (2 + D);
    device const float* mine = partial + (ulong)gid.x * parts * stride + (ulong)j * (2 + D);
    float mx = -INFINITY;
    for (int i = 0; i < parts; i++) mx = max(mx, mine[i * stride]);
    float w[64];  // TCG_MAX_PARTS
    float total = 0.0f;
    for (int i = 0; i < parts; i++) {
        const float m = mine[i * stride];
        w[i] = m == -INFINITY ? 0.0f : metal::fast::exp2(m - mx);
        total += w[i] * mine[i * stride + 1];
    }
    const float scale = total == 0.0f ? 0.0f : 1.0f / total;
    for (int d = tid; d < D; d += tpg.x) {
        float acc = 0.0f;
        for (int i = 0; i < parts; i++) acc += w[i] * mine[i * stride + 2 + d];
        out[((ulong)t * heads + g * group + j % group) * D + d] = acc * scale;
    }
}

// EXPERIMENTAL, off unless VIREO_I8=on (src/metal/mod.rs): faster, but both sides are
// rounded and the model's answers move with it (measured 2026-11-15 against exact mode on
// four small models: the likeliest next token differs at about 5% of positions, 0.1% with
// matmul_tc's f16). Kept to measure and build on, not as a default.
//
// A prompt's matrix products in 8-bit whole numbers on both sides (the accelerators
// multiply these at about twice their f16 rate: 118 against 63 TFLOPS on the M5 Max with
// both read straight from memory). The activations come rounded to 8 bits
// with a scale for every I8_K columns of a token (to_i8); a threadgroup decodes its tile
// of weights I8_K columns at a time to 8 bits with a scale a row (decode32_i8), one
// matmul2d multiplies the two into whole-number sums, and each thread then adds its share
// of them, times the two scales, to its own f32 sums. For Q6_K (a block of 256 columns a
// step) and Q8_0 (8 blocks a step), rows of a whole number of steps; other kinds and
// short batches keep matmul_tc.
typedef signed char i8;
#define I8_K 256

// Thirty-two numbers of a row from column k (a multiple of 32) as 8-bit numbers of the
// scale returned, which is the same for the whole I8_K-column step k is in.
template <int KIND>
static inline float decode32_i8(device const u8* row, int k, threadgroup i8* out);

// Q6_K's numbers are d × scale × (q − 32) with |q − 32| ≤ 32: they are rounded against the
// block's largest scale (kept in the block's padding by Gpu::upload_weights), so the 16
// numbers under it keep 4 steps to each of their own.
template <>
inline float decode32_i8<K_Q6_K>(device const u8* row, int k, threadgroup i8* out) {
    device const u8* block = row + (k / 256) * Q6K_BYTES;
    const int n = k % 256, hf = n / 128, run = (n % 128) / 32;
    const float top = (float)block[210];
    const float mul = top > 0.0f ? 127.0f / (32.0f * top) : 0.0f;
    device const packed_uint4* ql = reinterpret_cast<device const packed_uint4*>(block + hf * 64 + (run & 1) * 32);
    device const packed_uint4* qh = reinterpret_cast<device const packed_uint4*>(block + 128 + hf * 32);
    device const char* sc = reinterpret_cast<device const char*>(block + 192 + hf * 8 + 2 * run);
    const int shift = run < 2 ? 0 : 4;
    threadgroup u32* o = reinterpret_cast<threadgroup u32*>(out);
    for (int h = 0; h < 2; h++) {
        const float s = (float)sc[h] * mul;
        const uint4 lq = ql[h], hq = qh[h];
        const u32 lw[4] = {lq.x, lq.y, lq.z, lq.w}, hw[4] = {hq.x, hq.y, hq.z, hq.w};
        for (int i = 0; i < 4; i++) {
            const u32 qv = ((lw[i] >> shift) & 0x0F0F0F0Fu) | (((hw[i] >> (2 * run)) & 0x03030303u) << 4);
            const float4 v = (float4((float)(qv & 0xFF), (float)((qv >> 8) & 0xFF), (float)((qv >> 16) & 0xFF),
                                     (float)(qv >> 24)) - 32.0f) * s;
            const uint4 r = uint4(int4(rint(v))) & 0xFFu;
            o[4 * h + i] = r.x | (r.y << 8) | (r.z << 16) | (r.w << 24);
        }
    }
    return half_at(block + 208) * (32.0f * top / 127.0f);
}

// Q8_0's are d × q, a d for every 32: rounded against the step's largest |d|.
template <>
inline float decode32_i8<K_Q8_0>(device const u8* row, int k, threadgroup i8* out) {
    device const u8* first = row + (k / I8_K) * (I8_K / 32) * 34;
    float top = 0.0f;
    for (int b = 0; b < I8_K / 32; b++) top = max(top, fabs(half_at(first + b * 34)));
    device const u8* block = row + (k / 32) * 34;
    const float mul = top > 0.0f ? half_at(block) / top : 0.0f;
    for (int i = 0; i < 32; i++) out[i] = (i8)(int)rint((float)(char)block[2 + i] * mul);
    return top;
}

template <int KIND, int TOK, int ROWS>
kernel void matmul_i8(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device i8* x = ARG_PTR(i8, 1);  // the activations in 8 bits (to_i8)
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    device const float* xs = ARG_PTR(const float, 6);  // their scales: [token][step]
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    threadgroup i8 sa[ROWS * I8_K];  // the weights' tile: [row][k]
    threadgroup float sw[ROWS];      // and its rows' scales
    const int row0 = gid.x * ROWS, tok0 = gid.y * TOK, steps = cols / I8_K;
    constexpr int RUNS = I8_K / 32, PER = (ROWS * RUNS + TC_THREADS - 1) / TC_THREADS;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, ROWS, I8_K, false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sa, dextents<int32_t, 2>(I8_K, ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(I8_K, tokens), array<int, 2>({1, cols}));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), int>();
    // This thread's share of the tile's sums: each one's token and row.
    constexpr int MINE = TOK * ROWS / TC_THREADS;
    float sum[MINE];
    ushort at_t[MINE], at_r[MINE];
    const int mine = min((int)acc.get_capacity(), MINE);
    for (int i = 0; i < mine; i++) {
        const auto at = acc.get_multidimensional_index(i);
        at_r[i] = at[0];
        at_t[i] = min(tok0 + (int)at[1], tokens - 1);
        sum[i] = 0.0f;
        acc[i] = 0;
    }

    for (int step = 0; step < steps; step++) {
        const int k0 = step * I8_K;
        for (int part = 0; part < PER; part++) {
            const int run = tid + part * TC_THREADS, lr = run / RUNS, lk = (run % RUNS) * 32;
            if (lr >= ROWS) break;
            threadgroup i8* dst = sa + lr * I8_K + lk;
            float scale = 0.0f;
            if (row0 + lr < rows) {
                scale = decode32_i8<KIND>(w + (ulong)(row0 + lr) * row_bytes, k0 + lk, dst);
            } else {
                for (int i = 0; i < 32; i++) dst[i] = 0;
            }
            if (lk == 0) sw[lr] = scale;
        }
        BARRIER;
        auto tile_x = tensor(x + (ulong)tok0 * cols + k0, dextents<int32_t, 2>(I8_K, tokens - tok0),
                             array<int, 2>({1, cols}));
        mm.run(tile_x, t_w, acc);
        for (int i = 0; i < mine; i++) {
            sum[i] += (float)acc[i] * (xs[(ulong)at_t[i] * steps + step] * sw[at_r[i]]);
            acc[i] = 0;
        }
        BARRIER;
    }
    for (int i = 0; i < mine; i++) {
        const auto at = acc.get_multidimensional_index(i);
        const int t = tok0 + (int)at[1], r = row0 + (int)at[0];
        if (t < tokens && r < rows) y[(ulong)t * rows + r] = sum[i];
    }
}
template [[host_name("matmul_i8_q6_k")]] kernel void matmul_i8<K_Q6_K, 128, 32>(ARGS, KP);
template [[host_name("matmul_i8_q8_0")]] kernel void matmul_i8<K_Q8_0, 128, 32>(ARGS, KP);

// f32 activations to 8 bits for matmul_i8: a threadgroup of 256 threads takes one token's
// I8_K columns, rounds them against the largest of them and notes the scale.
// Arguments: x, the 8-bit copy, the scales, how many numbers.
kernel void to_i8(ARGS, KP) {
    device const float* x = ARG_PTR(const float, 0);
    device i8* out = ARG_PTR(i8, 1);
    device float* scales = ARG_PTR(float, 2);
    threadgroup float most[I8_K / 32];
    const ulong i = (ulong)gid.x * I8_K + tid;
    const float v = x[i];
    const float m = simd_max(fabs(v));
    const uint lane = tid & 31;
    if (lane == 0) most[tid >> 5] = m;
    BARRIER;
    const float top = simd_max(lane < I8_K / 32 ? most[lane] : 0.0f);
    const float scale = top / 127.0f;
    out[i] = top > 0.0f ? (i8)(int)rint(v / scale) : (i8)0;
    if (tid == 0) scales[gid.x] = scale;
}

// delta_chunk_scan on Metal 4 neural accelerators (n = 128, DC = 64, COLS columns of
// value head gid.x's state per threadgroup gid.y, 4 SIMD groups = 128 threads):
//   WS = W · S0,  δ = U − WS
//   O  = Q̂ · S0 + P · δ
//   S  = γ · S0 + K̂ᵀ · δ
// All operands and accumulators stay in f32, as in delta_chunk_scan_sg (the sums are taken
// in another order, so the last bits can differ).
template <int COLS>
kernel void delta_chunk_scan_tc(ARGS, KP) {
    device const float* scratch = ARG_PTR(const float, 0);
    device float* state = ARG_PTR(float, 1);
    device float* out = ARG_PTR(float, 2);
    const int n = ARG_INT(3), tokens = ARG_INT(4);
    constexpr int N = 128, SG = 4;
    threadgroup float Ss[N * COLS];
    threadgroup float ds[DC * COLS];
    const int h = gid.x, v_heads = ngrid.x, j0 = gid.y * COLS;
    const float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (int idx = tid; idx < N * COLS; idx += tpg.x) {
        const int c = idx / N, r = idx % N;
        Ss[r * COLS + c] = state[((ulong)h * N + j0 + c) * N + r];
    }
    BARRIER;

    auto t_s = tensor(Ss, dextents<int32_t, 2>(COLS, N));
    auto t_d = tensor(ds, dextents<int32_t, 2>(COLS, DC));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(DC, COLS, N, false, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<SG>>
        mm_ws;
    // O's two products (inner sizes N and DC) through one matmul, so its cooperative
    // tensor is laid out for both.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(DC, COLS, static_cast<int>(dynamic_extent), false, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SG>>
        mm_o;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(N, COLS, DC, true, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SG>>
        mm_ks;

    const int chunks = (tokens + DC - 1) / DC;
    for (int ch = 0; ch < chunks; ch++) {
        device const float* base = scratch + ((ulong)ch * v_heads + h) * delta_chunk_stride(N);
        device const float *W = base, *U = W + DC * N, *Qh = U + DC * N, *Kh = Qh + DC * N, *P = Kh + DC * N;
        const float gamma = P[DC * DC];

        auto t_w = tensor(const_cast<device float*>(W), dextents<int32_t, 2>(N, DC), array<int, 2>({1, N}));
        auto t_q = tensor(const_cast<device float*>(Qh), dextents<int32_t, 2>(N, DC), array<int, 2>({1, N}));
        auto t_k = tensor(const_cast<device float*>(Kh), dextents<int32_t, 2>(N, DC), array<int, 2>({1, N}));
        auto t_p = tensor(const_cast<device float*>(P), dextents<int32_t, 2>(DC, DC), array<int, 2>({1, DC}));

        // 1) WS = W · S0 -> ds, then δ = U − WS in place.
        mm_ws.run(t_w, t_s, t_d);
        BARRIER;
        for (int e = tid; e < DC * COLS; e += tpg.x) {
            const int i = e / COLS, c = e % COLS;
            ds[e] = U[i * N + j0 + c] - ds[e];
        }
        BARRIER;

        // 2) O = Q̂ · S0 + P · δ in a cooperative tensor while S0 is still intact.
        auto co_o = mm_o.template get_destination_cooperative_tensor<decltype(t_q), decltype(t_s), float>();
        for (uint16_t i = 0; i < co_o.get_capacity(); i++)
            if (co_o.is_valid_element(i)) co_o[i] = 0.0f;
        mm_o.run(t_q, t_s, co_o);
        mm_o.run(t_p, t_d, co_o);
        BARRIER;

        // 3) S = γ · S0 + K̂ᵀ · δ in place in Ss.
        for (int e = tid; e < N * COLS; e += tpg.x) Ss[e] *= gamma;
        BARRIER;
        mm_ks.run(t_k, t_d, t_s);
        BARRIER;

        // 4) Store O through ds (scaled by 1/√n) into out.
        co_o.store(t_d);
        BARRIER;
        for (int e = tid; e < DC * COLS; e += tpg.x) {
            const int i = e / COLS, c = e % COLS, t = ch * DC + i;
            if (t < tokens) out[((ulong)t * v_heads + h) * N + j0 + c] = ds[e] * inv_sqrt_n;
        }
        BARRIER;
    }
    for (int idx = tid; idx < N * COLS; idx += tpg.x) {
        const int c = idx / N, r = idx % N;
        state[((ulong)h * N + j0 + c) * N + r] = Ss[r * COLS + c];
    }
}

template [[host_name("delta_chunk_scan_tc_32")]] kernel void delta_chunk_scan_tc<32>(ARGS, KP);
