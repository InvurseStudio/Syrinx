//! The CPU backend: the reference every other backend is checked against.
//!
//! Plain f32 arithmetic, written for clarity first. It's still multi-threaded and
//! reads quantized weights straight from the memory-mapped file, so a 1B model runs at
//! a usable speed.

mod pool;

pub use pool::Pool;

use crate::quant;
use crate::tensor_type::TensorType;

/// A weight matrix in the model file: `rows` rows of `cols` numbers, row after row.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub kind: TensorType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> Matrix<'a> {
    pub fn row_bytes(&self) -> usize {
        self.cols / self.kind.block_len() as usize * self.kind.block_bytes() as usize
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let n = self.row_bytes();
        &self.data[r * n..(r + 1) * n]
    }

    /// Row `r` as f32s (an embedding lookup).
    pub fn decode_row(&self, r: usize, out: &mut [f32]) {
        quant::dequantize_row(self.kind, self.row(r), &mut out[..self.cols]);
    }
}

/// A pointer that tasks may share because each writes only its own part.
struct Shared<T>(*mut T);
unsafe impl<T> Sync for Shared<T> {}

/// `y = W x`, with the rows split across the pool.
pub fn matvec(pool: &Pool, w: &Matrix, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    // Enough tasks to balance the threads, few enough to keep the overhead small.
    let rows_per_task = (w.rows / (pool.threads() * 8)).max(4);
    let tasks = w.rows.div_ceil(rows_per_task);
    let out = Shared(y.as_mut_ptr());
    pool.run(tasks, &|t| {
        let out = &out;
        let start = t * rows_per_task;
        let end = (start + rows_per_task).min(w.rows);
        for r in start..end {
            let value = quant::dot_row(w.kind, w.row(r), x);
            // Safety: each task writes only rows start..end, and y outlives `run`.
            unsafe { *out.0.add(r) = value };
        }
    });
}

/// RMSNorm: `x / sqrt(mean(x²) + eps) × weight`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

/// Softmax in place, shifted by the maximum so nothing overflows.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU (x × sigmoid(x)) of `gate`, times `up`: the SwiGLU feed-forward's middle.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

/// Rotary position embedding, "normal" style (ggml's `GGML_ROPE_TYPE_NORM`, used by
/// Llama): each head's numbers rotate in adjacent pairs (0, 1), (2, 3), … Pair i turns
/// by `position × base^(−2i / dims) / factors[i]`.
pub fn rope_normal(
    x: &mut [f32],
    head_dim: usize,
    position: usize,
    base: f32,
    factors: Option<&[f32]>,
) {
    let dims = head_dim;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..dims / 2 {
            let mut theta = position as f32 * base.powf(-2.0 * i as f32 / dims as f32);
            if let Some(f) = factors {
                theta /= f[i];
            }
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[2 * i], head[2 * i + 1]);
            head[2 * i] = a * cos - b * sin;
            head[2 * i + 1] = a * sin + b * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_plain_loops() {
        let pool = Pool::new(4);
        let (rows, cols) = (37, 64);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 13) as f32 - 6.0)
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w = Matrix {
            kind: TensorType::F32,
            rows,
            cols,
            data: &data,
        };
        let x: Vec<f32> = (0..cols).map(|i| i as f32 / 10.0).collect();
        let mut y = vec![0.0; rows];
        matvec(&pool, &w, &x, &mut y);
        for r in 0..rows {
            let expected: f32 = (0..cols).map(|c| values[r * cols + c] * x[c]).sum();
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {r}: {} vs {expected}",
                y[r]
            );
        }
    }

    #[test]
    fn rms_norm_and_softmax() {
        let mut out = [0.0; 4];
        rms_norm(
            &[1.0, -1.0, 1.0, -1.0],
            &[1.0, 2.0, 3.0, 4.0],
            0.0,
            &mut out,
        );
        assert_eq!(out, [1.0, -2.0, 3.0, -4.0]);

        let mut p = [1.0, 2.0, 3.0, 1000.0];
        softmax(&mut p);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[3] > 0.999);
    }

    #[test]
    fn rope_rotates_pairs() {
        // Position 0 changes nothing.
        let mut x = [1.0, 2.0, 3.0, 4.0];
        rope_normal(&mut x, 4, 0, 10000.0, None);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
        // Pair 0 turns by `position` radians: (1, 0) at position 1 is (cos 1, sin 1).
        let mut x = [1.0, 0.0, 1.0, 0.0];
        rope_normal(&mut x, 4, 1, 10000.0, None);
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[1] - 1f32.sin()).abs() < 1e-6);
        // Pair 1 turns by 1 / sqrt(10000) = 0.01, halved again by a factor of 2.
        let mut x = [1.0, 0.0, 1.0, 0.0];
        rope_normal(&mut x, 4, 1, 10000.0, Some(&[1.0, 2.0]));
        assert!((x[2] - 0.005f32.cos()).abs() < 1e-6 && (x[3] - 0.005f32.sin()).abs() < 1e-6);
    }
}
