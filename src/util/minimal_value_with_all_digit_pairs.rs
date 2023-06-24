//! Utilities for constructing various representations of the smallest values that include all digit
//! pairs of a given base that is at least 2.

use {
    bitvec::prelude::*,
    num::{BigUint, Zero},
};

pub const MAX_BASE: usize = 1_usize << u8::BITS;

#[derive(Debug, PartialEq)]
pub enum InvalidBase {
    TooSmall,
    TooLarge,
}

struct Builder {
    sequence: Vec<u8>,
    used_pairs: BitVec<u32>,
    base: usize,
    left_digit: usize,
}

impl Builder {
    fn use_pair(&mut self, index: usize) {
        self.left_digit = index % self.base;
        self.sequence.push(self.left_digit as u8);
        self.used_pairs.set(index, true);
    }

    fn next_pair(&self) -> Option<usize> {
        let start: usize = self.left_digit * self.base;
        self.used_pairs[start..start + self.base]
            .first_zero()
            .map(|slice_index| slice_index + start)
    }
}

pub fn try_sequence(base: usize) -> Result<Vec<u8>, InvalidBase> {
    match base {
        0_usize..=1_usize => Err(InvalidBase::TooSmall),
        2_usize..=MAX_BASE => {
            let base_squared: usize = base * base;
            let base_squared_minus_base: usize = base_squared - base;

            let mut builder: Builder = Builder {
                sequence: Vec::with_capacity(base_squared + 1_usize),
                used_pairs: BitVec::new(),
                base,
                left_digit: 0_usize,
            };

            builder.used_pairs.resize(base_squared, false);

            // "Lie" about pair `base_squared_minus_base` being present so that we don't try to
            // include it after `0, (base - 1)`, which would waste digits.
            builder.use_pair(base_squared_minus_base);

            while let Some(index) = builder.next_pair() {
                builder.use_pair(index);
            }

            // Actually include pair `base_squared_minus_base`
            builder.use_pair(base_squared_minus_base);

            Ok(builder.sequence)
        }
        _ => Err(InvalidBase::TooLarge),
    }
}

pub fn try_string(base: usize) -> Result<String, InvalidBase> {
    if base <= 36_usize {
        try_sequence(base).map(|mut sequence| {
            for digit in sequence.iter_mut() {
                const LETTER_DELTA: u8 = b'A' - 10_u8;

                let digit_val: u8 = *digit;

                *digit = match digit_val {
                    0_u8..=9_u8 => digit_val + b'0',
                    10_u8..=35_u8 => digit_val + LETTER_DELTA,
                    _ => unreachable!(),
                };
            }

            // SAFETY: The only bytes present are either in b'0'..=b'9' or b'A'..=b'Z', which is a
            // valid UTF-8 string
            unsafe { String::from_utf8_unchecked(sequence) }
        })
    } else {
        Err(InvalidBase::TooLarge)
    }
}

pub fn try_value(base: usize) -> Result<u128, InvalidBase> {
    try_sequence(base).and_then(|sequence| {
        let base: u128 = base as u128;
        let mut value: u128 = 0_u128;

        for digit in sequence {
            value = value
                .checked_mul(base)
                .ok_or(InvalidBase::TooLarge)?
                .checked_add(digit as u128)
                .ok_or(InvalidBase::TooLarge)?;
        }

        Ok(value)
    })
}

pub fn try_big_uint(base: usize) -> Result<BigUint, InvalidBase> {
    try_sequence(base).map(|sequence| {
        let mut value: BigUint = BigUint::zero();

        for digit in sequence {
            value *= base;
            value += digit;
        }

        value
    })
}

pub fn try_decimal_string(base: usize) -> Result<String, InvalidBase> {
    try_big_uint(base).map(|big_uint| big_uint.to_string())
}

pub struct Iter(usize);

impl Iterator for Iter {
    type Item = (usize, BigUint);

    fn next(&mut self) -> Option<Self::Item> {
        try_big_uint(self.0).ok().map(|big_uint| {
            let base: usize = self.0;

            self.0 += 1_usize;

            (base, big_uint)
        })
    }
}

pub fn iter() -> Iter {
    Iter(2_usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_sequence() {
        for (base, expected) in [
            Err(InvalidBase::TooSmall),
            Err(InvalidBase::TooSmall),
            Ok(vec![0, 0, 1, 1, 0]),
            Ok(vec![0, 0, 1, 0, 2, 1, 1, 2, 2, 0]),
            Ok(vec![0, 0, 1, 0, 2, 0, 3, 1, 1, 2, 1, 3, 2, 2, 3, 3, 0]),
            Ok(vec![
                0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 1, 2, 1, 3, 1, 4, 2, 2, 3, 2, 4, 3, 3, 4, 4, 0,
            ]),
            Ok(vec![
                0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 1, 2, 1, 3, 1, 4, 1, 5, 2, 2, 3, 2, 4, 2, 5, 3,
                3, 4, 3, 5, 4, 4, 5, 5, 0,
            ]),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(try_sequence(base), expected, "`try_sequence({base}` failed");
        }

        assert!(matches!(try_sequence(MAX_BASE), Ok(_)));
        assert_eq!(try_sequence(MAX_BASE + 1_usize), Err(InvalidBase::TooLarge));
    }

    #[test]
    fn test_try_string() {
        for (base, expected) in [
            Err(InvalidBase::TooSmall),
            Err(InvalidBase::TooSmall),
            Ok("00110".to_owned()),
            Ok("0010211220".into()),
            Ok("00102031121322330".into()),
            Ok("00102030411213142232433440".into()),
            Ok("0010203040511213141522324253343544550".into()),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(try_string(base), expected, "`try_string({base}` failed");
        }

        let string_base_36: String = try_string(36_usize).unwrap();

        const STR_BASE_36_BEGINNING: &str =
            "00102030405060708090A0B0C0D0E0F0G0H0I0J0K0L0M0N0O0P0Q0R0S0T0U0V0W0X0Y0Z1";

        assert_eq!(
            &string_base_36[..STR_BASE_36_BEGINNING.len()],
            STR_BASE_36_BEGINNING
        );

        assert_eq!(try_string(37_usize), Err(InvalidBase::TooLarge));
    }

    #[test]
    fn test_try_value() {
        for (base, expected) in [
            Err(InvalidBase::TooSmall),
            Err(InvalidBase::TooSmall),
            Ok(6),
            Ok(2805),
            Ok(305503932),
            Ok(12935072846764870),
            Ok(303117795566473205184840210),
            Err(InvalidBase::TooLarge),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(try_value(base), expected, "`try_value({base}` failed");
        }
    }

    #[test]
    fn test_decimal_string() {
        for (base, expected) in [
            Err(InvalidBase::TooSmall),
            Err(InvalidBase::TooSmall),
            Ok("6".to_owned()),
            Ok("2805".into()),
            Ok("305503932".into()),
            Ok("12935072846764870".into()),
            Ok("303117795566473205184840210".into()),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                try_decimal_string(base),
                expected,
                "`try_decimal_string({base}` failed"
            );
        }

        assert_eq!(
            // Skip the first 2 zeroes that aren't present in the decimal string
            try_string(10_usize).map(|string| string[2_usize..].into()),
            try_decimal_string(10_usize)
        );
    }

    #[test]
    fn print() {
        for base in 2_usize..=10_usize {
            println!(
                "{base:2}: {:100}, {}",
                try_decimal_string(base).unwrap(),
                try_string(base).unwrap()
            );
        }
    }
}
