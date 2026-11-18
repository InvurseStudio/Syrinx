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

/// The nearest f16 to `x` (ties to even, as hardware rounds), as its bits. Too large
/// becomes infinity; too small, zero or a subnormal.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // A subnormal f16: the value in units of 2^-24, rounded.
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let (whole, rest, halfway) = (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1));
        let rounded = whole + u32::from(rest > halfway || (rest == halfway && whole & 1 == 1));
        return sign | rounded as u16;
    }
    let mut h = ((e as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is still right, up to infinity
    }
    sign | h as u16
}

/// `x` rounded to the nearest f16, back as an f32.
#[inline]
pub fn round_to_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
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
    fn f32_to_f16_round_trips_and_rounds_to_even() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                assert!(f16_to_f32(f32_to_f16(x)).is_nan());
            } else {
                assert_eq!(f32_to_f16(x), h, "{h:#06x}");
            }
        }
        // Halfway between 1.0 (0x3C00) and the next f16 (0x3C01): ties go to even.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16(1e6), 0x7C00);
        assert_eq!(f32_to_f16(1e-9), 0);
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.5), 0x0001); // rounds up to the smallest subnormal
    }

    #[test]
    fn bf16_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
    }
}
