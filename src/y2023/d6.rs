use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, not_line_ending, space1},
        combinator::{eof, iterator, ParserIterator},
        error::Error,
        sequence::{preceded, tuple},
        Err, IResult,
    },
    std::ops::Range,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct Record {
    time: u64,
    distance: u64,
}

impl Record {
    const NEGATIVE: f64 = -1.0_f64;
    const POSITIVE: f64 = 1.0_f64;

    /// Compute a winning time extremum for a given record time and distance, and a sign multiplier.
    ///
    /// Using `T` for the record time, the total distance traveled is:
    /// ```
    /// (T - t) * t = -t ^ 2 + T * t
    /// ```
    /// Using `D` for the record distance, we can construct an inequality of winning ranges:
    /// ```
    /// D < -t ^ 2 + T * t
    /// ==
    /// 0 < -t ^ 2 + T * t - D
    /// ```
    /// Plugging this into the quadratic formula, we get the following:
    /// ```
    /// t = (-T +- sqrt(T ^ 2 - 4 * D)) / -2
    /// ==
    /// t = (T -+ sqrt(T ^ 2 - 4 * D)) / 2
    /// ```
    fn winning_time_extremum(time: f64, distance: f64, sign: f64) -> f64 {
        (time + sign * (time * time - 4.0_f64 * distance).sqrt()) * 0.5_f64
    }

    fn winning_time_range(&self) -> Range<u64> {
        let time: f64 = self.time as f64;
        let distance: f64 = self.distance as f64;
        let min: f64 = Self::winning_time_extremum(time, distance, Self::NEGATIVE);
        let start: u64 = min.floor() as u64 + 1_u64;
        let max: f64 = Self::winning_time_extremum(time, distance, Self::POSITIVE);
        let end: u64 = max.ceil() as u64;

        start..end
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution(Vec<Record>);

impl Solution {
    fn iter_winning_time_ranges(&self) -> impl Iterator<Item = Range<u64>> + '_ {
        self.0.iter().map(|record| record.winning_time_range())
    }

    fn concat_digits(left: u64, right: u64) -> u64 {
        left * 10_u64.pow(digits(right as u32) as u32) + right
    }

    fn big_record(&self) -> Record {
        self.0
            .iter()
            .fold(Record::default(), |big_record, record| Record {
                time: Self::concat_digits(big_record.time, record.time),
                distance: Self::concat_digits(big_record.distance, record.distance),
            })
    }

    fn range_len(range: Range<u64>) -> u64 {
        range.end - range.start
    }

    fn winning_time_range_len_product(&self) -> u64 {
        self.iter_winning_time_ranges()
            .map(Self::range_len)
            .product()
    }

    fn big_winning_time_range_len(&self) -> u64 {
        Self::range_len(self.big_record().winning_time_range())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        let (input, (time_input, distance_input)): (&str, (&str, &str)) = tuple((
            preceded(tag("Time:"), not_line_ending),
            preceded(tuple((line_ending, tag("Distance:"))), not_line_ending),
        ))(input)?;

        let mut time_iter: ParserIterator<&str, _, _> =
            iterator(time_input, preceded(space1, parse_integer::<u64>));
        let mut distance_iter: ParserIterator<&str, _, _> =
            iterator(distance_input, preceded(space1, parse_integer::<u64>));

        for (time, distance) in &mut time_iter.zip(&mut distance_iter) {
            solution.0.push(Record { time, distance });
        }

        eof(time_iter.finish()?.0)?;
        eof(distance_iter.finish()?.0)?;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.winning_time_range_len_product());

        if args.verbose {
            dbg!(self.iter_winning_time_ranges().collect::<Vec<Range<u64>>>());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.big_winning_time_range_len());

        if args.verbose {
            dbg!(self.big_record().winning_time_range());
        }
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

    const SOLUTION_STR: &'static str = "\
        Time:      7  15   30\n\
        Distance:  9  40  200\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Record {
                    time: 7_u64,
                    distance: 9_u64,
                },
                Record {
                    time: 15_u64,
                    distance: 40_u64,
                },
                Record {
                    time: 30_u64,
                    distance: 200_u64,
                },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_winning_time_range() {
        assert_eq!(
            solution()
                .iter_winning_time_ranges()
                .collect::<Vec<Range<u64>>>(),
            vec![2_u64..6_u64, 4_u64..12_u64, 11_u64..20_u64,]
        );
    }

    #[test]
    fn test_winning_time_range_len_product() {
        assert_eq!(solution().winning_time_range_len_product(), 288_u64);
    }

    #[test]
    fn test_big_record() {
        assert_eq!(
            solution().big_record(),
            Record {
                time: 71530_u64,
                distance: 940200_u64,
            }
        );
    }

    #[test]
    fn test_big_winning_time_range_len() {
        assert_eq!(solution().big_winning_time_range_len(), 71503_u64);
    }
}
