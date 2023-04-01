use {
    crate::*,
    nom::{
        character::complete::{digit1, line_ending},
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::terminated,
        Err,
    },
    std::{
        mem::{replace, MaybeUninit},
        str::FromStr,
    },
};

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<u16>);

impl Solution {
    fn count_increases_of_rolling_window<const N: usize>(&self) -> usize {
        // SAFETY: `0_u16` is a valid value for a `u16`.
        let mut window: [u16; N] = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut count: usize = 0;

        self.0
            .iter()
            .copied()
            .enumerate()
            .fold(0_u16, |prev_sum, (index, curr_depth)| {
                let sum: u16 = prev_sum - replace(&mut window[index % N], curr_depth) + curr_depth;

                count += (index >= N && sum > prev_sum) as usize;

                sum
            });

        count
    }

    fn count_increases(&self) -> usize {
        self.count_increases_of_rolling_window::<1_usize>()
    }

    fn count_increases_of_rolling_window_3(&self) -> usize {
        self.count_increases_of_rolling_window::<3_usize>()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_increases());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_increases_of_rolling_window_3());
    }
}

impl<'a> TryFrom<&'a str> for Solution {
    type Error = Err<Error<&'a str>>;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut iter = iterator(
            value,
            terminated(map_res(digit1, u16::from_str), opt(line_ending)),
        );
        let result: Result<Self, Self::Error> = Ok(Self(iter.collect()));

        iter.finish()?;

        result
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const SWEEP_REPORT_STR: &str = "199\n200\n208\n210\n200\n207\n240\n269\n260\n263\n";

    lazy_static! {
        static ref SOLUTION: Solution = new_solutions();
    }

    fn new_solutions() -> Solution {
        Solution(vec![199, 200, 208, 210, 200, 207, 240, 269, 260, 263])
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SWEEP_REPORT_STR), Ok(new_solutions()))
    }

    #[test]
    fn test_count_increases() {
        assert_eq!(SOLUTION.count_increases(), 7);
    }

    #[test]
    fn test_count_increases_of_rolling_window_3() {
        assert_eq!(SOLUTION.count_increases_of_rolling_window_3(), 5);
    }
}
