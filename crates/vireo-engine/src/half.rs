//! 16-bit floats, as GGUF stores them: IEEE half precision (f16) and bfloat16.
//!
//! Quantized blocks keep their scales as f16, so this runs for every block of every
//! weight. Written out bit by bit rather than relying on Rust's `f16`, which isn't
//! stable.

/// An IEEE 754 half-precision float's bits, as an f32 (exactly: every f16 is an f32).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exponent = u32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: shift the mantissa up until it has its leading 1, lowering the
            // exponent to match. The value is mantissa × 2^-24.
            let mut e = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13), // infinity and NaN
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// A bfloat16's bits, as an f32: bfloat16 is the top half of an f32.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Reads a little-endian f16 from the start of `bytes`.
#[inline]
pub fn read_f16(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // the largest f16
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14)); // the smallest normal
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // the smallest subnormal
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn every_f16_matches_its_definition() {
        for h in 0..=u16::MAX {
            let exponent = i32::from((h >> 10) & 0x1f);
            let mantissa = f64::from(h & 0x3ff);
            let magnitude = match exponent {
                0 => mantissa * 2f64.powi(-24),
                0x1f => continue,
                _ => (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
            };
            let expected = if h & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            };
            assert_eq!(f64::from(f16_to_f32(h)), expected, "{h:#06x}");
        }
    }

    #[test]
    fn bf16_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
    }
}
