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
// and the SIMD groups sharing it. Each kernel comes four times: matmul_tc_* with TC_TOKENS
// tokens a tile, matmul_tc_wide_* with TC_WIDE for a prompt's whole batches (a large
// model's weights decoded half as often), and matmul_tc_t8_* to matmul_tc_t192_* for
// short batches (SMALL_TILES in src/metal/mod.rs: no multiplying of padding). ROWS and KK
// are the tile's rows and its columns a step (TC_ROWS and TC_K but for the 8- and
// 16-token tiles, which take 256 columns a step).
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

template <int KIND, int TOK, int ROWS = TC_ROWS, int KK = TC_K>
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
    threadgroup half sa[ROWS * KK];  // the weights' tile: [row][k]
    const int row0 = gid.x * ROWS, tok0 = gid.y * TOK;
    // Decoding: the tile's runs of 32 numbers (KK / 32 a row) shared out among the threads.
    constexpr int RUNS = KK / 32, PER = (ROWS * RUNS + TC_THREADS - 1) / TC_THREADS;

    // C (tokens × rows) = X (tokens × k) · Wᵀ; extents list the contiguous dimension first.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sa, dextents<int32_t, 2>(KK, ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, cols}));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    for (int k0 = k_begin; k0 < k_end; k0 += KK) {
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
        BARRIER;
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
// 8 and 16 tokens, 256 columns a step (see SMALL_TILES in src/metal/mod.rs).
template [[host_name("matmul_tc_t8_f32")]] kernel void matmul_tc<K_F32, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_f16")]] kernel void matmul_tc<K_F16, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_q4_0")]] kernel void matmul_tc<K_Q4_0, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_q8_0")]] kernel void matmul_tc<K_Q8_0, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_q4_k")]] kernel void matmul_tc<K_Q4_K, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_q5_k")]] kernel void matmul_tc<K_Q5_K, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t8_q6_k")]] kernel void matmul_tc<K_Q6_K, 8, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_f32")]] kernel void matmul_tc<K_F32, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_f16")]] kernel void matmul_tc<K_F16, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_q4_0")]] kernel void matmul_tc<K_Q4_0, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_q8_0")]] kernel void matmul_tc<K_Q8_0, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_q4_k")]] kernel void matmul_tc<K_Q4_K, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_q5_k")]] kernel void matmul_tc<K_Q5_K, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t16_q6_k")]] kernel void matmul_tc<K_Q6_K, 16, TC_ROWS, 256>(ARGS, KP);
template [[host_name("matmul_tc_t32_f32")]] kernel void matmul_tc<K_F32, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_f16")]] kernel void matmul_tc<K_F16, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_q4_0")]] kernel void matmul_tc<K_Q4_0, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_q8_0")]] kernel void matmul_tc<K_Q8_0, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_q4_k")]] kernel void matmul_tc<K_Q4_K, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_q5_k")]] kernel void matmul_tc<K_Q5_K, 32>(ARGS, KP);
template [[host_name("matmul_tc_t32_q6_k")]] kernel void matmul_tc<K_Q6_K, 32>(ARGS, KP);
template [[host_name("matmul_tc_t64_f32")]] kernel void matmul_tc<K_F32, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_f16")]] kernel void matmul_tc<K_F16, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_q4_0")]] kernel void matmul_tc<K_Q4_0, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_q8_0")]] kernel void matmul_tc<K_Q8_0, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_q4_k")]] kernel void matmul_tc<K_Q4_K, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_q5_k")]] kernel void matmul_tc<K_Q5_K, 64>(ARGS, KP);
template [[host_name("matmul_tc_t64_q6_k")]] kernel void matmul_tc<K_Q6_K, 64>(ARGS, KP);
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

// matmul_tc for one tile of moe_group's (vireo.metal): its expert's rows × its slots, the
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
    threadgroup half sa[ROWS * KK];
    const int row0 = gid.x * ROWS;
    constexpr int RUNS = KK / 32, PER = (ROWS * RUNS + TC_THREADS - 1) / TC_THREADS;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<TC_SG>>
        mm;
    auto t_w = tensor(sa, dextents<int32_t, 2>(KK, ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, x_stride}));
    auto acc = mm.template get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += KK) {
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
        BARRIER;
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
// attention_flash (vireo.metal) with S = Q Kᵀ and O += P V as matmul2d operations. A
// threadgroup of 8 SIMD groups takes one query head and AQ tokens; the keys and values
// are read straight from the cache, AC positions at a time. The scores go to threadgroup
// memory for the online softmax (each SIMD group takes AQ / 8 rows, a lane AC / 32 of a
// row's columns; masked causally and by the sliding window from the positions), and O
// stays in a cooperative tensor, its rows rescaled as the maximum moves. The last chunk can
// read up to AC − 1 positions past the context (into the buffer's slack, see
// Context::new_buffer): their probabilities are 0. In a ring (a sliding-window layer's
// last positions only, see ring_row in vireo.metal) the chunks start at multiples of AC, as
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
        auto tv = tensor(values + ring_row(start, ring) * kv_dim + kv, dextents<int32_t, 2>(D, AC), array<int, 2>({1, kv_dim}));
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
    const int tokens = ARG_INT(10);
    constexpr int ROWS = R / SG, COLS = AC / 32;
    threadgroup float ss[R * AC];
    threadgroup half sp[R * AC];
    threadgroup float sr[R];
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
    const int co_n = min((int)co.get_capacity(), CO_MAX);
    for (int i = 0; i < co_n; i++) {
        const bool valid = co.is_valid_element(i);
        co_row[i] = valid ? (short)co.get_multidimensional_index(i)[1] : (short)-1;
        if (valid) co[i] = 0.0f;
    }
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
        BARRIER;
        for (int jj = 0; jj < ROWS; jj++) {
            const int j = jj * SG + sg, t = t0 + j / group, at = p0 + t;
            const bool live = j < live_rows && t < tokens;
            const int first = max(start, window > 0 ? at + 1 - window : 0);
            float s[COLS];
            float mx = -INFINITY;
            for (int ii = 0; ii < COLS; ii++) {
                const int c = ii * 32 + lane, kpos = from + c;
                const bool seen = live && kpos >= first && kpos <= at;
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
        // Rescaled only where a row's maximum moved: deep in a context most don't.
        for (int i = 0; i < co_n; i++) {
            if (co_row[i] < 0) continue;
            const float fix = sr[co_row[i]];
            if (fix != 1.0f) co[i] *= fix;
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
        w[i] = m == -INFINITY ? 0.0f : exp(m - mx);
        total += w[i] * mine[i * stride + 1];
    }
    const float scale = total == 0.0f ? 0.0f : 1.0f / total;
    for (int d = tid; d < D; d += tpg.x) {
        float acc = 0.0f;
        for (int i = 0; i < parts; i++) acc += w[i] * mine[i * stride + 2 + d];
        out[((ulong)t * heads + g * group + j % group) * D + d] = acc * scale;
    }
}
