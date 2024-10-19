use {
    crate::*,
    nom::{
        character::complete::satisfy,
        combinator::{map, verify},
        error::Error,
        multi::many0,
        Err, IResult,
    },
    num::Integer,
};

/* --- Day 1: Inverse Captcha ---

The night before Christmas, one of Santa's Elves calls you in a panic. "The printer's broken! We can't print the Naughty or Nice List!" By the time you make it to sub-basement 17, there are only a few minutes until midnight. "We have a big problem," she says; "there must be almost fifty bugs in this system, but nothing else can print The List. Stand in this square, quick! There's no time to explain; if you can convince them to pay you in stars, you'll be able to--" She pulls a lever and the world goes blurry.

When your eyes can focus again, everything seems a lot more pixelated than before. She must have sent you inside the computer! You check the system clock: 25 milliseconds until midnight. With that much time, you should be able to collect all fifty stars by December 25th.

Collect stars by solving puzzles. Two puzzles will be made available on each day millisecond in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

You're standing in a room with "digitization quarantine" written in LEDs along one wall. The only door is locked, but it includes a small interface. "Restricted Area - Strictly No Digitized Users Allowed."

It goes on to explain that you may only leave by solving a captcha to prove you're not a human. Apparently, you only get one millisecond to solve the captcha: too fast for a normal human, but it feels like hours to you.

The captcha requires you to review a sequence of digits (your puzzle input) and find the sum of all digits that match the next digit in the list. The list is circular, so the digit after the last digit is the first digit in the list.

For example:

    1122 produces a sum of 3 (1 + 2) because the first digit (1) matches the second digit and the third digit (2) matches the fourth digit.
    1111 produces 4 because each digit (all 1) matches the next.
    1234 produces 0 because no digit matches the next.
    91212129 produces 9 because the only digit that matches the next one is the last digit, 9.

What is the solution to your captcha?

--- Part Two ---

You notice a progress bar that jumps to 50% completion. Apparently, the door isn't yet satisfied, but it did emit a star as encouragement. The instructions change:

Now, instead of considering the next digit, it wants you to consider the digit halfway around the circular list. That is, if your list contains 10 items, only include a digit in your sum if the digit 10/2 = 5 steps forward matches it. Fortunately, your list has an even number of elements.

For example:

    1212 produces 6: the list contains 4 items, and all four digits match the digit 2 items ahead.
    1221 produces 0, because every comparison is between a 1 and a 2.
    123425 produces 4, because both 2s match each other, but no other digit has a match.
    123123 produces 12.
    12131415 produces 4.

What is the solution to your new captcha? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<u8>);

impl Solution {
    fn sum_matching_digits<I: Iterator<Item = (u8, u8)>>(iter: I) -> usize {
        iter.filter_map(|(a, b)| (a == b).then_some(a as usize))
            .sum()
    }

    fn sum_digits_where_next_matches(&self) -> usize {
        Self::sum_matching_digits(
            self.0
                .windows(2_usize)
                .map(|bytes| (bytes[0_usize], bytes[1_usize]))
                .chain(self.0.last().copied().zip(self.0.first().copied())),
        )
    }

    fn sum_digits_where_across_matches(&self) -> usize {
        Self::sum_matching_digits(
            self.0.iter().copied().zip(
                self.0[self.0.len() / 2_usize..]
                    .iter()
                    .chain(self.0[..self.0.len() / 2_usize].iter())
                    .copied(),
            ),
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            verify(
                many0(map(satisfy(|c: char| c.is_ascii_digit()), |c: char| {
                    c as u8 - b'0'
                })),
                |digits: &Vec<u8>| digits.len().is_even(),
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Trivial, no comment.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_digits_where_next_matches());
    }

    /// Trivial, no comment.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_digits_where_across_matches());
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
        "1122", "1111", "1234", "91212129", "1212", "1221", "123425", "123123", "12131415",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![1_u8, 1_u8, 2_u8, 2_u8]),
                Solution(vec![1_u8, 1_u8, 1_u8, 1_u8]),
                Solution(vec![1_u8, 2_u8, 3_u8, 4_u8]),
                Solution(vec![9_u8, 1_u8, 2_u8, 1_u8, 2_u8, 1_u8, 2_u8, 9_u8]),
                Solution(vec![1_u8, 2_u8, 1_u8, 2_u8]),
                Solution(vec![1_u8, 2_u8, 2_u8, 1_u8]),
                Solution(vec![1_u8, 2_u8, 3_u8, 4_u8, 2_u8, 5_u8]),
                Solution(vec![1_u8, 2_u8, 3_u8, 1_u8, 2_u8, 3_u8]),
                Solution(vec![1_u8, 2_u8, 1_u8, 3_u8, 1_u8, 4_u8, 1_u8, 5_u8]),
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
    fn test_sum_digits_where_next_matches() {
        for (index, digits_where_next_matches_sum) in [
            3_usize, 4_usize, 0_usize, 9_usize, 0_usize, 3_usize, 0_usize, 0_usize, 0_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).sum_digits_where_next_matches(),
                digits_where_next_matches_sum
            );
        }
    }

    #[test]
    fn test_sum_digits_where_across_matches() {
        for (index, digits_where_across_matches_sum) in [
            0_usize, 4_usize, 0_usize, 6_usize, 6_usize, 0_usize, 4_usize, 12_usize, 4_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).sum_digits_where_across_matches(),
                digits_where_across_matches_sum
            );
        }
    }
}
