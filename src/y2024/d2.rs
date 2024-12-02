use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::{many0, many1_count},
        sequence::terminated,
        Err, IResult,
    },
    std::ops::{Range, RangeInclusive},
};

/* --- Day 2: Red-Nosed Reports ---

Fortunately, the first location The Historians want to search isn't a long walk from the Chief Historian's office.

While the Red-Nosed Reindeer nuclear fusion/fission plant appears to contain no sign of the Chief Historian, the engineers there run up to you as soon as they see you. Apparently, they still talk about the time Rudolph was saved through molecular synthesis from a single electron.

They're quick to add that - since you're already here - they'd really appreciate your help analyzing some unusual data from the Red-Nosed reactor. You turn to check if The Historians are waiting for you, but they seem to have already divided into groups that are currently searching every corner of the facility. You offer to help with the unusual data.

The unusual data (your puzzle input) consists of many reports, one report per line. Each report is a list of numbers called levels that are separated by spaces. For example:

7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9

This example data contains six reports each containing five levels.

The engineers are trying to figure out which reports are safe. The Red-Nosed reactor safety systems can only tolerate levels that are either gradually increasing or gradually decreasing. So, a report only counts as safe if both of the following are true:

    The levels are either all increasing or all decreasing.
    Any two adjacent levels differ by at least one and at most three.

In the example above, the reports can be found safe or unsafe by checking those rules:

    7 6 4 2 1: Safe because the levels are all decreasing by 1 or 2.
    1 2 7 8 9: Unsafe because 2 7 is an increase of 5.
    9 7 6 2 1: Unsafe because 6 2 is a decrease of 4.
    1 3 2 4 5: Unsafe because 1 3 is increasing but 3 2 is decreasing.
    8 6 4 4 1: Unsafe because 4 4 is neither an increase or a decrease.
    1 3 6 7 9: Safe because the levels are all increasing by 1, 2, or 3.

So, in this example, 2 reports are safe.

Analyze the unusual data from the engineers. How many reports are safe?

--- Part Two ---

The engineers are surprised by the low number of safe reports until they realize they forgot to tell you about the Problem Dampener.

The Problem Dampener is a reactor-mounted module that lets the reactor safety systems tolerate a single bad level in what would otherwise be a safe report. It's like the bad level never happened!

Now, the same rules apply as before, except if removing a single level from an unsafe report would make it safe, the report instead counts as safe.

More of the above example's reports are now safe:

    7 6 4 2 1: Safe without removing any level.
    1 2 7 8 9: Unsafe regardless of which level is removed.
    9 7 6 2 1: Unsafe regardless of which level is removed.
    1 3 2 4 5: Safe by removing the second level, 3.
    8 6 4 4 1: Safe by removing the third level, 4.
    1 3 6 7 9: Safe without removing any level.

Thanks to the Problem Dampener, 4 reports are actually safe!

Update your analysis by handling situations where the Problem Dampener can remove a single level from unsafe reports. How many reports are now safe? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Report {
    levels_range: Range<u16>,
}

impl Report {
    const SAFE_DELTA_RANGE: RangeInclusive<i32> = 1_i32..=3_i32;

    fn levels<'l>(&self, levels: &'l [u8]) -> &'l [u8] {
        &levels[self.levels_range.as_range_usize()]
    }

    fn is_safe_internal<I: Iterator<Item = u8>>(mut iter: I) -> bool {
        iter.next().map_or(true, |prev_level| {
            let mut prev_level: i32 = prev_level as i32;

            iter.map(|curr_level| {
                let curr_level: i32 = curr_level as i32;
                let delta: i32 = curr_level - prev_level;

                prev_level = curr_level;

                delta
            })
            .try_fold(None, |is_increasing, delta| {
                Self::SAFE_DELTA_RANGE
                    .contains(&delta.abs())
                    .then_some(())
                    .and_then(|_| {
                        is_increasing
                            .map_or(true, |is_increasing| is_increasing == (delta > 0_i32))
                            .then_some(Some(delta > 0_i32))
                    })
            })
            .is_some()
        })
    }

    fn is_safe(&self, levels: &[u8]) -> bool {
        Self::is_safe_internal(self.levels(levels).iter().copied())
    }

    fn is_safe_with_problem_dampener(&self, levels: &[u8]) -> bool {
        self.is_safe(levels) || {
            let levels: &[u8] = self.levels(levels);

            (0_usize..levels.len())
                .into_iter()
                .any(|removed_level_index| {
                    Self::is_safe_internal(
                        levels[..removed_level_index]
                            .iter()
                            .chain(levels[removed_level_index + 1_usize..].iter())
                            .copied(),
                    )
                })
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    reports: Vec<Report>,
    levels: Vec<u8>,
}

impl Solution {
    fn count_safe_reports(&self) -> usize {
        self.reports
            .iter()
            .filter(|report| report.is_safe(&self.levels))
            .count()
    }

    fn count_safe_reports_with_problem_dampener(&self) -> usize {
        self.reports
            .iter()
            .filter(|report| report.is_safe_with_problem_dampener(&self.levels))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut levels: Vec<u8> = Vec::new();

        let (input, reports): (&str, Vec<Report>) = many0(terminated(
            |input: &'i str| {
                let levels_range_start: u16 = levels.len() as u16;
                let input: &str =
                    many1_count(map(terminated(parse_integer, opt(tag(" "))), |level| {
                        levels.push(level);
                    }))(input)?
                    .0;
                let levels_range_end: u16 = levels.len() as u16;

                Ok((
                    input,
                    Report {
                        levels_range: levels_range_start..levels_range_end,
                    },
                ))
            },
            opt(line_ending),
        ))(input)?;

        Ok((input, Self { reports, levels }))
    }
}

impl RunQuestions for Solution {
    /// Again, would've been faster if I just stored a levels `Vec` per report.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_safe_reports());
    }

    /// Rust's iterator trait is so powerful.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_safe_reports_with_problem_dampener());
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        7 6 4 2 1\n\
        1 2 7 8 9\n\
        9 7 6 2 1\n\
        1 3 2 4 5\n\
        8 6 4 4 1\n\
        1 3 6 7 9\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                reports: vec![
                    Report {
                        levels_range: 0_u16..5_u16,
                    },
                    Report {
                        levels_range: 5_u16..10_u16,
                    },
                    Report {
                        levels_range: 10_u16..15_u16,
                    },
                    Report {
                        levels_range: 15_u16..20_u16,
                    },
                    Report {
                        levels_range: 20_u16..25_u16,
                    },
                    Report {
                        levels_range: 25_u16..30_u16,
                    },
                ],
                levels: vec![
                    7_u8, 6_u8, 4_u8, 2_u8, 1_u8, 1_u8, 2_u8, 7_u8, 8_u8, 9_u8, 9_u8, 7_u8, 6_u8,
                    2_u8, 1_u8, 1_u8, 3_u8, 2_u8, 4_u8, 5_u8, 8_u8, 6_u8, 4_u8, 4_u8, 1_u8, 1_u8,
                    3_u8, 6_u8, 7_u8, 9_u8,
                ],
            }]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_is_safe() {
        for (index, is_safe) in [vec![true, false, false, false, false, true]]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);

            assert_eq!(
                solution
                    .reports
                    .iter()
                    .map(|report| report.is_safe(&solution.levels))
                    .collect::<Vec<bool>>(),
                is_safe
            );
        }
    }

    #[test]
    fn test_count_safe_reports() {
        for (index, safe_reports_count) in [2_usize].into_iter().enumerate() {
            assert_eq!(solution(index).count_safe_reports(), safe_reports_count);
        }
    }

    #[test]
    fn test_is_safe_with_problem_dampener() {
        for (index, is_safe_with_problem_dampener) in [vec![true, false, false, true, true, true]]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);

            assert_eq!(
                solution
                    .reports
                    .iter()
                    .map(|report| report.is_safe_with_problem_dampener(&solution.levels))
                    .collect::<Vec<bool>>(),
                is_safe_with_problem_dampener
            );
        }
    }

    #[test]
    fn test_count_safe_reports_with_problem_dampener() {
        for (index, safe_reports_with_problem_dampener_count) in [4_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_safe_reports_with_problem_dampener(),
                safe_reports_with_problem_dampener_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
