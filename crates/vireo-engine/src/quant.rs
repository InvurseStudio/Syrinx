//! Turns GGUF's quantized blocks back into f32s, and dot products straight from them.
//!
//! This is the CPU reference: plain, exact f32 arithmetic, block by block, to check the
//! GPU kernels against. The block layouts are ggml's (`ggml-common.h`), and the
//! arithmetic follows its `dequantize_row_*` functions.
//!
//! llama.cpp's own CPU and CUDA paths round the *activations* to 8 bits before a
//! quantized dot product, to use integer instructions. This reference doesn't, so its
//! results are slightly more exact than llama.cpp's and won't match them bit for bit.

use crate::half::{bf16_to_f32, read_f16};
use crate::tensor_type::TensorType;

/// The types the CPU reference can read so far.
pub fn supports(kind: TensorType) -> bool {
    matches!(
        kind,
        TensorType::F32
            | TensorType::F16
            | TensorType::BF16
            | TensorType::Q8_0
            | TensorType::Q4_K
            | TensorType::Q6_K
    )
}

/// Decodes one block (`kind.block_len()` numbers) from `src` into `dst`.
///
/// # Panics
/// If `kind` isn't supported (check with [`supports`]) or the slices are too short.
#[inline]
pub fn dequantize_block(kind: TensorType, src: &[u8], dst: &mut [f32]) {
    match kind {
        TensorType::F32 => dst[0] = f32::from_le_bytes(src[..4].try_into().unwrap()),
        TensorType::F16 => dst[0] = read_f16(src),
        TensorType::BF16 => dst[0] = bf16_to_f32(u16::from_le_bytes([src[0], src[1]])),
        TensorType::Q8_0 => q8_0(src, dst),
        TensorType::Q4_K => q4_k(src, dst),
        TensorType::Q6_K => q6_k(src, dst),
        other => panic!("the CPU reference can't read {other} yet"),
    }
}

/// Decodes a whole row: `src` holds `dst.len()` numbers of type `kind`.
pub fn dequantize_row(kind: TensorType, src: &[u8], dst: &mut [f32]) {
    match kind {
        // One number per "block": decode them in a tight loop.
        TensorType::F32 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = f32::from_le_bytes(*s);
            }
        }
        TensorType::F16 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = read_f16(s);
            }
        }
        _ => {
            let (len, bytes) = (kind.block_len() as usize, kind.block_bytes() as usize);
            for (d, s) in dst.chunks_exact_mut(len).zip(src.chunks_exact(bytes)) {
                dequantize_block(kind, s, d);
            }
        }
    }
}

/// The dot product of a row (`x.len()` numbers of type `kind`, in `row`) with `x`.
pub fn dot_row(kind: TensorType, row: &[u8], x: &[f32]) -> f32 {
    match kind {
        TensorType::F32 | TensorType::F16 | TensorType::BF16 => {
            let mut buf = [0f32; 256];
            let mut sum = 0.0;
            let size = kind.block_bytes() as usize;
            for (xs, bytes) in x.chunks(256).zip(row.chunks(256 * size)) {
                dequantize_row(kind, bytes, &mut buf[..xs.len()]);
                sum += dot(&buf[..xs.len()], xs);
            }
            sum
        }
        _ => {
            let len = kind.block_len() as usize;
            let bytes = kind.block_bytes() as usize;
            let mut buf = [0f32; 256];
            let mut sum = 0.0;
            for (xs, block) in x.chunks_exact(len).zip(row.chunks_exact(bytes)) {
                dequantize_block(kind, block, &mut buf[..len]);
                sum += dot(&buf[..len], xs);
            }
            sum
        }
    }
}

/// A dot product with 8 running sums, so the compiler can use vector instructions.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    let mut sums = [0f32; 8];
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}

/// Q8_0, 32 numbers in 34 bytes: an f16 scale `d`, then 32 int8s. x = d × q.
fn q8_0(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src);
    for (y, &q) in dst[..32].iter_mut().zip(&src[2..34]) {
        *y = d * f32::from(q as i8);
    }
}

/// One of Q4_K's 8 sub-block scales and minimums, each 6 bits, packed into 12 bytes:
/// sub-blocks 0–3 in the low 6 bits of bytes 0–3 (scales) and 4–7 (minimums);
/// sub-blocks 4–7 in the nibbles of bytes 8–11, with their top 2 bits in the top
/// 2 bits of bytes 0–7.
#[inline]
fn q4_k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K, 256 numbers in 144 bytes: f16 `d` and `dmin`, 12 bytes of sub-block scales and
/// minimums, then 128 bytes of 4-bit numbers. 8 sub-blocks of 32, each with
/// x = d × scale × q − dmin × min. Each 32 bytes of `qs` hold two sub-blocks: the low
/// nibbles the first, the high nibbles the second.
fn q4_k(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(&src[0..2]);
    let dmin = read_f16(&src[2..4]);
    let scales = &src[4..16];
    let qs = &src[16..144];
    for pair in 0..4 {
        let (sc1, m1) = q4_k_scale_min(2 * pair, scales);
        let (sc2, m2) = q4_k_scale_min(2 * pair + 1, scales);
        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
        let q = &qs[pair * 32..pair * 32 + 32];
        let y = &mut dst[pair * 64..pair * 64 + 64];
        for l in 0..32 {
            y[l] = d1 * f32::from(q[l] & 0xF) - min1;
            y[l + 32] = d2 * f32::from(q[l] >> 4) - min2;
        }
    }
}

/// Q6_K, 256 numbers in 210 bytes: 128 bytes of low 4 bits (`ql`), 64 of high 2 bits
/// (`qh`), 16 int8 scales (one per 16 numbers), then an f16 `d`. x = d × scale × (q − 32).
/// Each half of the block (128 numbers) uses 64 bytes of `ql`, 32 of `qh` and 8 scales.
fn q6_k(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(&src[208..210]);
    for half in 0..2 {
        let ql = &src[half * 64..half * 64 + 64];
        let qh = &src[128 + half * 32..128 + half * 32 + 32];
        let sc = &src[192 + half * 8..192 + half * 8 + 8];
        let y = &mut dst[half * 128..half * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: [u8; 2] = [0x00, 0x3C]; // f16 1.0
    const HALF: [u8; 2] = [0x00, 0x38]; // f16 0.5

    fn decode(kind: TensorType, block: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0; kind.block_len() as usize];
        dequantize_block(kind, block, &mut out);
        out
    }

    #[test]
    fn q8_0_block() {
        let mut b = HALF.to_vec();
        b.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let y = decode(TensorType::Q8_0, &b);
        assert_eq!(y[0], -8.0);
        assert_eq!(y[16], 0.0);
        assert_eq!(y[31], 7.5);
    }

    /// Q4_K's 12 scale bytes for 8 (scale, min) pairs, packed as `q4_k_scale_min` reads them.
    fn pack_q4_k_scales(pairs: [(u8, u8); 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for (j, &(sc, m)) in pairs.iter().enumerate() {
            if j < 4 {
                q[j] |= sc & 63;
                q[j + 4] |= m & 63;
            } else {
                q[j + 4] |= (sc & 0xF) | ((m & 0xF) << 4);
                q[j - 4] |= (sc >> 4) << 6;
                q[j] |= (m >> 4) << 6;
            }
        }
        q
    }

    #[test]
    fn q4_k_scales_unpack() {
        let pairs = [
            (1, 2),
            (63, 0),
            (17, 33),
            (40, 5),
            (63, 63),
            (0, 1),
            (34, 50),
            (7, 60),
        ];
        let packed = pack_q4_k_scales(pairs);
        for (j, &pair) in pairs.iter().enumerate() {
            assert_eq!(q4_k_scale_min(j, &packed), pair, "sub-block {j}");
        }
    }

    #[test]
    fn q4_k_block() {
        // d = 1, dmin = 0.5. Sub-block j has scale j + 1 and minimum 2j.
        let pairs: [(u8, u8); 8] = std::array::from_fn(|j| (j as u8 + 1, 2 * j as u8));
        let mut b = ONE.to_vec();
        b.extend(HALF);
        b.extend(pack_q4_k_scales(pairs));
        // Every byte: low nibble 3, high nibble 10.
        b.extend([0xA3u8; 128]);
        let y = decode(TensorType::Q4_K, &b);
        for j in 0..8 {
            let q = if j % 2 == 0 { 3.0 } else { 10.0 };
            let expected = (j as f32 + 1.0) * q - 0.5 * (2 * j) as f32;
            for i in 0..32 {
                assert_eq!(y[j * 32 + i], expected, "sub-block {j}, number {i}");
            }
        }
    }

    #[test]
    fn q6_k_block() {
        // ql and qh zero: every q is -32. Scales 1, 2, …, 16; d = 0.5.
        let mut b = vec![0u8; 192];
        b.extend((1..=16u8).collect::<Vec<_>>());
        b.extend(HALF);
        let y = decode(TensorType::Q6_K, &b);
        // Number i uses scale (i / 16) + 1.
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(v, 0.5 * ((i / 16) as f32 + 1.0) * -32.0, "number {i}");
        }
        // The highest 6-bit value, 63, in the first number: q = 31.
        b[0] = 0x0F;
        b[128] = 0x03;
        assert_eq!(decode(TensorType::Q6_K, &b)[0], 0.5 * 31.0);
    }

    #[test]
    fn dot_rows_match_decoding() {
        let mut row = HALF.to_vec();
        row.extend((0..32).map(|i| i as u8));
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
        let decoded = decode(TensorType::Q8_0, &row);
        let expected: f32 = decoded.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot_row(TensorType::Q8_0, &row, &x) - expected).abs() < 1e-3);

        let floats: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let self_dot: f32 = x.iter().map(|v| v * v).sum();
        assert!((dot_row(TensorType::F32, &floats, &x) - self_dot).abs() < 1e-3);
    }
}
