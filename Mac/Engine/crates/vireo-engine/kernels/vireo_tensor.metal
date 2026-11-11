// Vireo's matrix products on the GPU's neural accelerators (Metal 4's tensor operations:
// M5, A19 and later). Compiled after vireo.metal, into the same library, when the GPU has
// them (src/metal/mod.rs); without them the backend uses matmul_sg in vireo.metal.
//
// The default (fast) mode's matrix products for prompts: each threadgroup decodes a tile
// of TC_ROWS weight rows, TC_K columns at a time, to f16 in threadgroup memory (decode8 in
// vireo.metal), and one matmul2d over its 4 SIMD groups multiplies it with TC_TOKENS
// tokens' activations, read as f32 from GPU memory, adding up in f32. Rounding the
// weights to f16 (with relaxed precision) is this mode's counterpart of the 8-bit
// activations on CUDA; exact mode keeps matmul_sg's f32.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define TC_ROWS 64
#define TC_TOKENS 128
#define TC_K 32

template <int KIND>
kernel void matmul_tc(ARGS, KP) {
    device const u8* w = ARG_PTR(const u8, 0);
    device float* x = ARG_PTR(float, 1);
    device float* y = ARG_PTR(float, 2);
    const int rows = ARG_INT(3), cols = ARG_INT(4), tokens = ARG_INT(5);
    const ulong row_bytes = mm_row_bytes<KIND>(cols);
    threadgroup half sa[TC_ROWS * TC_K];  // the weights' tile: [row][k]
    const int row0 = gid.x * TC_ROWS, tok0 = gid.y * TC_TOKENS;
    // Decoding: thread → row tid / 2, columns (tid % 2) × 16.. of the tile.
    const int lr = tid >> 1, lk = (tid & 1) * 16;
    const bool row_live = row0 + lr < rows;
    device const u8* wrow = w + (ulong)(row0 + lr) * row_bytes;

    // C (tokens × rows) = X (tokens × k) · Wᵀ; extents list the contiguous dimension first.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TC_TOKENS, TC_ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>>
        mm;
    auto t_w = tensor(sa, dextents<int32_t, 2>(TC_K, TC_ROWS));
    auto t_x = tensor(x, dextents<int32_t, 2>(cols, tokens), array<int, 2>({1, cols}));
    auto acc = mm.get_destination_cooperative_tensor<decltype(t_x), decltype(t_w), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += TC_K) {
        float v[8];
        for (int part = 0; part < 2; part++) {
            if (row_live) {
                decode8<KIND>(wrow, cols / 32, k0 + lk + part * 8, v);
            } else {
                for (int i = 0; i < 8; i++) v[i] = 0.0f;
            }
            for (int i = 0; i < 8; i++) sa[lr * TC_K + lk + part * 8 + i] = half(v[i]);
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
