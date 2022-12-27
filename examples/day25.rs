use std::fmt::Write;

use {
    aoc_2022::*,
    std::fmt::{Debug, Formatter, Result as FmtResult},
};

#[derive(PartialEq)]
struct Snafu(i64);

impl Snafu {
    const MINUS: u8 = b'-';
    const MINUS_MINUS: u8 = b'=';
}

#[derive(Debug, PartialEq)]
enum ParseSnafuError {
    EmptyStr,
    InvalidInitialByte(u8),
    InvalidByte(u8),
}

impl From<Snafu> for i64 {
    fn from(value: Snafu) -> Self {
        value.0
    }
}

impl Debug for Snafu {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut snafu_digits: [i8; 32_usize] = [0_i8; 32_usize];
        let mut snafu_digit_index: usize = 0_usize;
        let mut remaining_value: i64 = self.0;

        while remaining_value != 0_i64 {
            let mut snafu_digit: i8 = remaining_value.rem_euclid(5_i64) as i8;

            if snafu_digit > 2_i8 {
                snafu_digit -= 5_i8;
                remaining_value += 5_i64;
            }

            snafu_digits[snafu_digit_index] = snafu_digit;
            snafu_digit_index += 1_usize;
            remaining_value = remaining_value.div_euclid(5_i64);
        }

        for snafu_digit in snafu_digits[..snafu_digit_index].iter().rev().copied() {
            f.write_char(match snafu_digit {
                0_i8..=2_i8 => snafu_digit as u8 + b'0',
                -1_i8 => Self::MINUS,
                -2_i8 => Self::MINUS_MINUS,
                _ => unreachable!(),
            } as char)?;
        }

        Ok(())
    }
}

impl TryFrom<&str> for Snafu {
    type Error = ParseSnafuError;

    fn try_from(snafu_str: &str) -> Result<Self, Self::Error> {
        use ParseSnafuError::*;

        if snafu_str.is_empty() {
            Err(EmptyStr)
        } else {
            let mut snafu: Self = Self(0_i64);

            for snafu_byte in snafu_str.as_bytes().iter().copied() {
                snafu.0 = snafu.0 * 5_i64
                    + match snafu_byte {
                        b'0'..=b'2' => {
                            if snafu_byte == b'0' && snafu.0 == 0_i64 {
                                Err(InvalidInitialByte(snafu_byte))
                            } else {
                                Ok((snafu_byte - b'0') as i64)
                            }
                        }
                        Self::MINUS => Ok(-1_i64),
                        Self::MINUS_MINUS => Ok(-2_i64),
                        _ => Err(InvalidByte(snafu_byte)),
                    }?;
            }

            Ok(snafu)
        }
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day25.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                dbg!(Snafu(
                    input
                        .split('\n')
                        .map(Snafu::try_from)
                        .map(Result::unwrap)
                        .map(i64::from)
                        .sum(),
                ));
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DECIMAL_VALUES_A: &[i64] = &[
        1747_i64, 906_i64, 198_i64, 11_i64, 201_i64, 31_i64, 1257_i64, 32_i64, 353_i64, 107_i64,
        7_i64, 3_i64, 37_i64,
    ];
    const DECIMAL_VALUES_B: &[i64] = &[
        1_i64,
        2_i64,
        3_i64,
        4_i64,
        5_i64,
        6_i64,
        7_i64,
        8_i64,
        9_i64,
        10_i64,
        15_i64,
        20_i64,
        2022_i64,
        12345_i64,
        314159265_i64,
    ];
    const SNAFU_STRS_A: &[&str] = &[
        "1=-0-2", "12111", "2=0=", "21", "2=01", "111", "20012", "112", "1=-1=", "1-12", "12",
        "1=", "122",
    ];
    const SNAFU_STRS_B: &[&str] = &[
        "1",
        "2",
        "1=",
        "1-",
        "10",
        "11",
        "12",
        "2=",
        "2-",
        "20",
        "1=0",
        "1-0",
        "1=11-2",
        "1-0---0",
        "1121-1110-1=0",
    ];

    #[test]
    fn test_snafu_to_decimal() {
        for (decimal, snafu_str) in iter_decimal_and_snafu() {
            assert_eq!(
                Snafu::try_from(snafu_str),
                Ok(Snafu(decimal)),
                "\"{snafu_str}\" didn't successfully convert into {decimal}"
            );
        }
    }

    #[test]
    fn test_snafu_sum() {
        let decimal: i64 = SNAFU_STRS_A
            .iter()
            .copied()
            .map(Snafu::try_from)
            .map(Result::unwrap)
            .map(i64::from)
            .sum();

        assert_eq!(
            (decimal, format!("{:#?}", Snafu(decimal))),
            (4_890_i64, "2=-1=0".into())
        );
    }

    fn iter_decimal_and_snafu() -> impl Iterator<Item = (i64, &'static str)> {
        DECIMAL_VALUES_A
            .iter()
            .copied()
            .chain(DECIMAL_VALUES_B.iter().copied())
            .zip(
                SNAFU_STRS_A
                    .iter()
                    .copied()
                    .chain(SNAFU_STRS_B.iter().copied()),
            )
    }
}
