use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{line_ending, not_line_ending, satisfy},
        combinator::iterator,
        combinator::{map, opt, ParserIterator},
        error::{Error, ErrorKind},
        multi::many0,
        sequence::terminated,
        AsChar, Err, IResult,
    },
    std::str::from_utf8_unchecked,
};

struct CalibrationDigit(u8);

impl CalibrationDigit {
    fn match_value<'i>(
        tag_str: &'static str,
        value: u8,
    ) -> impl Fn(&'i str) -> IResult<&'i str, Self> {
        move |input| map(tag(tag_str), |_| Self(value))(input)
    }

    fn parse_string<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::match_value("one", 1_u8),
            Self::match_value("two", 2_u8),
            Self::match_value("three", 3_u8),
            Self::match_value("four", 4_u8),
            Self::match_value("five", 5_u8),
            Self::match_value("six", 6_u8),
            Self::match_value("seven", 7_u8),
            Self::match_value("eight", 8_u8),
            Self::match_value("nine", 9_u8),
        ))(input)
    }

    fn parse_digit<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(char::is_dec_digit), |c| Self(c as u8 - b'0'))(input)
    }
}

impl Parse for CalibrationDigit {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((Self::parse_string, Self::parse_digit))(input)
    }
}

trait ParseDigit<'i>
where
    Self: Copy + Fn(&'i str) -> IResult<&'i str, CalibrationDigit>,
{
}

impl<'i, T: Copy + Fn(&'i str) -> IResult<&'i str, CalibrationDigit>> ParseDigit<'i> for T {}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct CalibrationValue(u8);

impl CalibrationValue {
    fn branch_input<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        if input.is_empty() || !input.is_char_boundary(1_usize) {
            Err(Err::Error(Error::new(input, ErrorKind::Fail)))
        } else {
            Ok((
                // SAFETY: We know this is a valid char boundary due to the check above.
                unsafe { from_utf8_unchecked(&input.as_bytes()[1_usize..]) },
                input,
            ))
        }
    }

    fn parse<'i, F: ParseDigit<'i>>(f: F) -> impl Fn(&'i str) -> IResult<&'i str, Self> {
        move |input| {
            let (remaining_lines, line) = terminated(not_line_ending, opt(line_ending))(input)?;

            let mut digit_iter: ParserIterator<&str, _, _> = iterator(
                line,
                map(Self::branch_input, |input| {
                    f(input)
                        .map(|(_, calibration_digit)| calibration_digit.0)
                        .ok()
                }),
            );

            let mut first_digit: Option<u8> = None;
            let mut last_digit: Option<u8> = None;

            for digit in (&mut digit_iter).filter_map(|x| x) {
                last_digit = Some(digit);

                if first_digit.is_none() {
                    first_digit = last_digit.clone();
                }
            }

            digit_iter.finish()?;

            first_digit
                .zip(last_digit)
                .ok_or(Err::Error(Error::new(input, ErrorKind::Fail)))
                .map(|(first_digit, last_digit)| {
                    (remaining_lines, Self(first_digit * 10_u8 + last_digit))
                })
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct CalibrationValues(Vec<CalibrationValue>);

impl CalibrationValues {
    fn parse<'i, F: ParseDigit<'i>>(f: F) -> impl Fn(&'i str) -> IResult<&'i str, Self> {
        move |input| {
            map(
                many0(terminated(CalibrationValue::parse(f), opt(line_ending))),
                Self,
            )(input)
        }
    }

    fn parse_original<'i>(input: &'i str) -> IResult<&'i str, Self> {
        Self::parse(CalibrationDigit::parse_digit)(input)
    }

    fn parse_with_strings<'i>(input: &'i str) -> IResult<&'i str, Self> {
        Self::parse(CalibrationDigit::parse)(input)
    }

    fn sum_calibration_values(&self) -> u32 {
        self.0
            .iter()
            .map(|calibration_value| calibration_value.0 as u32)
            .sum()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(String);

impl Solution {
    fn try_sum_calibration_values_original(&self) -> Result<u32, Err<Error<&str>>> {
        CalibrationValues::parse_original(&self.0)
            .map(|(_, calibration_values)| calibration_values.sum_calibration_values())
    }

    fn try_sum_calibration_values_with_strings(&self) -> Result<u32, Err<Error<&str>>> {
        CalibrationValues::parse_with_strings(&self.0)
            .map(|(_, calibration_values)| calibration_values.sum_calibration_values())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        Ok((&input[input.len()..], Self(input.into())))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_sum_calibration_values_original().ok());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_sum_calibration_values_with_strings().ok());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &[
        "1abc2\n\
        pqr3stu8vwx\n\
        a1b2c3d4e5f\n\
        treb7uchet\n",
        "two1nine\n\
        eightwothree\n\
        abcone2threexyz\n\
        xtwone3four\n\
        4nineeightseven2\n\
        zoneight234\n\
        7pqrstsixteen\n",
    ];

    fn calibration_values(index: usize) -> &'static CalibrationValues {
        use super::CalibrationValue as CV;

        static ONCE_LOCK: OnceLock<Vec<CalibrationValues>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                CalibrationValues(vec![CV(12), CV(38), CV(15), CV(77)]),
                CalibrationValues(vec![CV(29), CV(83), CV(13), CV(24), CV(42), CV(14), CV(76)]),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(
            Solution::try_from(SOLUTION_STRS[0_usize])
                .ok()
                .and_then(|solution| CalibrationValues::parse_original(&solution.0)
                    .ok()
                    .map(|(_, calibration_values)| calibration_values))
                .as_ref(),
            Some(calibration_values(0_usize))
        );
        assert_eq!(
            Solution::try_from(SOLUTION_STRS[1_usize])
                .ok()
                .and_then(
                    |solution| CalibrationValues::parse_with_strings(&solution.0)
                        .ok()
                        .map(|(_, calibration_values)| calibration_values)
                )
                .as_ref(),
            Some(calibration_values(1_usize))
        );
    }

    #[test]
    fn test_sum_calibration_values() {
        for (index, calibration_values_sum) in [142_u32, 281_u32].into_iter().enumerate() {
            assert_eq!(
                calibration_values(index).sum_calibration_values(),
                calibration_values_sum
            );
        }
    }
}
