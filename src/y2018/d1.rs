use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::delimited,
        Err, IResult,
    },
    std::collections::HashSet,
};

/* --- Day 1: Chronal Calibration ---

"We've detected some temporal anomalies," one of Santa's Elves at the Temporal Anomaly Research and Detection Instrument Station tells you. She sounded pretty worried when she called you down here. "At 500-year intervals into the past, someone has been changing Santa's history!"

"The good news is that the changes won't propagate to our time stream for another 25 days, and we have a device" - she attaches something to your wrist - "that will let you fix the changes with no such propagation delay. It's configured to send you 500 years further into the past every few days; that was the best we could do on such short notice."

"The bad news is that we are detecting roughly fifty anomalies throughout time; the device will indicate fixed anomalies with stars. The other bad news is that we only have one device and you're the best person for the job! Good lu--" She taps a button on the device and you suddenly feel like you're falling. To save Christmas, you need to get all fifty stars by December 25th.

Collect stars by solving puzzles. Two puzzles will be made available on each day in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

After feeling like you've been falling for a few minutes, you look at the device's tiny screen. "Error: Device must be calibrated before first use. Frequency drift detected. Cannot maintain destination lock." Below the message, the device shows a sequence of changes in frequency (your puzzle input). A value like +6 means the current frequency increases by 6; a value like -3 means the current frequency decreases by 3.

For example, if the device displays frequency changes of +1, -2, +3, +1, then starting from a frequency of zero, the following changes would occur:

    Current frequency  0, change of +1; resulting frequency  1.
    Current frequency  1, change of -2; resulting frequency -1.
    Current frequency -1, change of +3; resulting frequency  2.
    Current frequency  2, change of +1; resulting frequency  3.

In this example, the resulting frequency is 3.

Here are other example situations:

    +1, +1, +1 results in  3
    +1, +1, -2 results in  0
    -1, -2, -3 results in -6

Starting with a frequency of zero, what is the resulting frequency after all of the changes in frequency have been applied?

--- Part Two ---

You notice that the device repeats the same frequency change list over and over. To calibrate the device, you need to find the first frequency it reaches twice.

For example, using the same list of changes above, the device would loop as follows:

    Current frequency  0, change of +1; resulting frequency  1.
    Current frequency  1, change of -2; resulting frequency -1.
    Current frequency -1, change of +3; resulting frequency  2.
    Current frequency  2, change of +1; resulting frequency  3.
    (At this point, the device continues from the start of the list.)
    Current frequency  3, change of +1; resulting frequency  4.
    Current frequency  4, change of -2; resulting frequency  2, which has already been seen.

In this example, the first frequency reached twice is 2. Note that your device might need to repeat its list of frequency changes many times before a duplicate frequency is found, and that duplicates might be found while in the middle of processing the list.

Here are other examples:

    +1, -1 first reaches 0 twice.
    +3, +3, +4, -2, -4 first reaches 10 twice.
    -6, +3, +8, +5, -6 first reaches 5 twice.
    +7, +7, -2, -7, -4 first reaches 14 twice.

What is the first frequency your device reaches twice? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<i32>);

impl Solution {
    fn sum_frequency_changes(&self) -> i32 {
        self.0.iter().copied().sum()
    }

    fn iter_frequencies(&self) -> impl Iterator<Item = i32> + '_ {
        let mut sum: i32 = 0_i32;

        [0_i32]
            .into_iter()
            .chain(self.0.iter().cycle().copied())
            .map(move |frequency_change| {
                sum += frequency_change;

                sum
            })
    }

    fn first_repeated_frequency(&self) -> i32 {
        let mut observed_frequencies: HashSet<i32> = HashSet::new();

        self.iter_frequencies()
            .try_fold((), |_, frequency| {
                observed_frequencies
                    .insert(frequency)
                    .then_some(())
                    .ok_or(frequency)
            })
            .err()
            .unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(delimited(
                opt(tag("+")),
                parse_integer,
                opt(alt((tag(", "), line_ending))),
            )),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_frequency_changes());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.first_repeated_frequency());
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
        "+1, -2, +3, +1",
        "+1, +1, +1",
        "+1, +1, -2",
        "-1, -2, -3",
        "+1, -1",
        "+3, +3, +4, -2, -4",
        "-6, +3, +8, +5, -6",
        "+7, +7, -2, -7, -4",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![1_i32, -2_i32, 3_i32, 1_i32]),
                Solution(vec![1_i32, 1_i32, 1_i32]),
                Solution(vec![1_i32, 1_i32, -2_i32]),
                Solution(vec![-1_i32, -2_i32, -3_i32]),
                Solution(vec![1_i32, -1_i32]),
                Solution(vec![3_i32, 3_i32, 4_i32, -2_i32, -4_i32]),
                Solution(vec![-6_i32, 3_i32, 8_i32, 5_i32, -6_i32]),
                Solution(vec![7_i32, 7_i32, -2_i32, -7_i32, -4_i32]),
            ]
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
    fn test_sum_frequency_changes() {
        for (index, frequency_changes_sum) in
            [3_i32, 3_i32, 0_i32, -6_i32, 0_i32, 4_i32, 4_i32, 1_i32]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).sum_frequency_changes(),
                frequency_changes_sum
            );
        }
    }

    #[test]
    fn test_first_repeated_frequency() {
        for (index, opt_first_repeated_frequency) in [
            Some(2_i32),
            None,
            Some(0_i32),
            None,
            Some(0_i32),
            Some(10_i32),
            Some(5_i32),
            Some(14_i32),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(first_repeated_frequency) = opt_first_repeated_frequency {
                assert_eq!(
                    solution(index).first_repeated_frequency(),
                    first_repeated_frequency
                );
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
