use {
    crate::*,
    nom::{
        character::complete::{line_ending, not_line_ending, space0},
        combinator::{all_consuming, map, map_res, opt, verify},
        error::Error,
        multi::many0_count,
        sequence::terminated,
        Err, IResult,
    },
    std::ops::{Range, RangeInclusive},
};

/* --- Day 2: Corruption Checksum ---

As you walk through the door, a glowing humanoid shape yells in your direction. "You there! Your state appears to be idle. Come help us repair the corruption in this spreadsheet - if we take another millisecond, we'll have to display an hourglass cursor!"

The spreadsheet consists of rows of apparently-random numbers. To make sure the recovery process is on the right track, they need you to calculate the spreadsheet's checksum. For each row, determine the difference between the largest value and the smallest value; the checksum is the sum of all of these differences.

For example, given the following spreadsheet:

5 1 9 5
7 5 3
2 4 6 8

    The first row's largest and smallest values are 9 and 1, and their difference is 8.
    The second row's largest and smallest values are 7 and 3, and their difference is 4.
    The third row's difference is 6.

In this example, the spreadsheet's checksum would be 8 + 4 + 6 = 18.

What is the checksum for the spreadsheet in your puzzle input?

--- Part Two ---

"Great work; looks like we're on the right track after all. Here's a star for your effort." However, the program seems a little worried. Can programs be worried?

"Based on what we're seeing, it looks like all the User wanted is some information about the evenly divisible values in the spreadsheet. Unfortunately, none of us are equipped for that kind of calculation - most of us specialize in bitwise operations."

It sounds like the goal is to find the only two numbers in each row where one evenly divides the other - that is, where the result of the division operation is a whole number. They would like you to find those numbers on each line, divide them, and add up each line's result.

For example, given the following spreadsheet:

5 9 2 8
9 4 7 3
3 8 6 5

    In the first row, the only two numbers that evenly divide are 8 and 2; the result of this division is 4.
    In the second row, the two numbers are 9 and 3; the result is 3.
    In the third row, the result is 2.

In this example, the sum of the results would be 4 + 3 + 2 = 9.

What is the sum of each row's result in your puzzle input? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    cells: Vec<u16>,
    row_cell_ranges: Vec<Range<u32>>,
}

impl Solution {
    fn iter_rows(&self) -> impl Iterator<Item = &[u16]> {
        self.row_cell_ranges
            .iter()
            .map(|row_cell_range| &self.cells[row_cell_range.as_range_usize()])
    }

    fn try_row_range(cells: &[u16]) -> Option<RangeInclusive<u16>> {
        (!cells.is_empty())
            .then(|| {
                cells.iter().fold((u16::MAX, u16::MIN), |(min, max), cell| {
                    (min.min(*cell), max.max(*cell))
                })
            })
            .map(|(min, max)| min..=max)
    }

    fn try_row_evenly_divisible_pair(cells: &[u16]) -> Option<(u16, u16)> {
        cells
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(index, cell_a)| {
                cells[index + 1_usize..]
                    .iter()
                    .copied()
                    .map(move |cell_b| (cell_a.max(cell_b), cell_a.min(cell_b)))
            })
            .find(|(cell_a, cell_b)| cell_a % cell_b == 0_u16)
    }

    fn iter_row_ranges(&self) -> impl Iterator<Item = RangeInclusive<u16>> + '_ {
        self.iter_rows().filter_map(Self::try_row_range)
    }

    fn iter_row_evenly_divisible_pairs(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        self.iter_rows()
            .filter_map(Self::try_row_evenly_divisible_pair)
    }

    fn checksum(&self) -> u16 {
        self.iter_row_ranges()
            .map(|row_range| *row_range.end() - *row_range.start())
            .sum()
    }

    fn quotient_sum(&self) -> u16 {
        self.iter_row_evenly_divisible_pairs()
            .map(|(cell_a, cell_b)| cell_a / cell_b)
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut cells: Vec<u16> = Vec::new();
        let mut row_cell_ranges: Vec<Range<u32>> = Vec::new();

        let input: &str = many0_count(map_res(
            terminated(
                verify(not_line_ending, |input: &&str| !input.is_empty()),
                opt(line_ending),
            ),
            |input: &str| {
                let start: u32 = cells.len() as u32;

                all_consuming(many0_count(map(
                    terminated(parse_integer, space0),
                    |cell| {
                        cells.push(cell);
                    },
                )))(input)
                .ok()
                .ok_or(())?;

                let end: u32 = cells.len() as u32;

                row_cell_ranges.push(start..end);

                Result::<(), ()>::Ok(())
            },
        ))(input)?
        .0;

        Ok((
            input,
            Self {
                cells,
                row_cell_ranges,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// I could have laid out data in this struct better, but lists of lists feels wasteful. It does
    /// make parsing a lot cleaner/simpler, though. The problem itself was trivial.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.checksum());
    }

    /// I'm not a huge fan of introducing a new example just for the second problem, where the first
    /// example doesn't meet the new criteria laid out in the second problem. Specifically, the
    /// constraint of having only a single pair of evenly divisible cells per row is not true in the
    /// case of the first example.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.quotient_sum());
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
        "5 1 9 5\n\
        7 5 3\n\
        2 4 6 8\n",
        "5 9 2 8\n\
        9 4 7 3\n\
        3 8 6 5\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    cells: vec![
                        5_u16, 1_u16, 9_u16, 5_u16, 7_u16, 5_u16, 3_u16, 2_u16, 4_u16, 6_u16, 8_u16,
                    ],
                    row_cell_ranges: vec![0_u32..4_u32, 4_u32..7_u32, 7_u32..11_u32],
                },
                Solution {
                    cells: vec![
                        5_u16, 9_u16, 2_u16, 8_u16, 9_u16, 4_u16, 7_u16, 3_u16, 3_u16, 8_u16,
                        6_u16, 5_u16,
                    ],
                    row_cell_ranges: vec![0_u32..4_u32, 4_u32..8_u32, 8_u32..12_u32],
                },
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
    fn test_iter_row_ranges() {
        for (index, row_ranges) in [
            vec![1_u16..=9_u16, 3_u16..=7_u16, 2_u16..=8_u16],
            vec![2_u16..=9_u16, 3_u16..=9_u16, 3_u16..=8_u16],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_row_ranges()
                    .collect::<Vec<RangeInclusive<u16>>>(),
                row_ranges
            );
        }
    }

    #[test]
    fn test_checksum() {
        for (index, checksum) in [18_u16, 18_u16].into_iter().enumerate() {
            assert_eq!(solution(index).checksum(), checksum);
        }
    }

    #[test]
    fn test_iter_row_evenly_divisible_pairs() {
        for (index, row_evenly_divisible_pairs) in [
            vec![(5_u16, 1_u16), (4_u16, 2_u16)],
            vec![(8_u16, 2_u16), (9_u16, 3_u16), (6_u16, 3_u16)],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_row_evenly_divisible_pairs()
                    .collect::<Vec<(u16, u16)>>(),
                row_evenly_divisible_pairs
            );
        }
    }

    #[test]
    fn test_quotient_sum() {
        for (index, quotient_sum) in [7_u16, 9_u16].into_iter().enumerate() {
            assert_eq!(solution(index).quotient_sum(), quotient_sum);
        }
    }
}
