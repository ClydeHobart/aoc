use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many1,
        sequence::terminated,
        Err, IResult,
    },
};

struct Differential {
    values: Vec<i32>,
    base_samples: usize,
    rows: usize,
}

impl Differential {
    fn triangle_base_samples(&self) -> usize {
        triangle_number(self.base_samples)
    }

    fn row_len_and_start(&self, row: usize) -> (usize, usize) {
        let row_len: usize = self.base_samples - row;
        let start: usize = self.triangle_base_samples() - triangle_number(row_len);

        (row_len, start)
    }

    fn row_values(&self, row: usize) -> &[i32] {
        let (row_len, start): (usize, usize) = self.row_len_and_start(row);
        let end: usize = start + row_len;

        &self.values[start..end]
    }

    fn row_values_mut(&mut self, row: usize) -> &mut [i32] {
        let (row_len, start): (usize, usize) = self.row_len_and_start(row);
        let end: usize = start + row_len;

        &mut self.values[start..end]
    }

    fn increase_base_samples(&mut self, base_samples: usize) {
        let old_base_samples: usize = self.base_samples;

        assert!(base_samples >= old_base_samples);

        let base_samples_delta: usize = base_samples - old_base_samples;

        self.base_samples = base_samples;

        self.values.resize(
            self.triangle_base_samples() - triangle_number(base_samples - self.rows),
            0_i32,
        );

        for row in 0_usize..self.rows {
            let (row_len, start): (usize, usize) = self.row_len_and_start(row);

            self.values[start + row_len - base_samples_delta..].rotate_right(base_samples_delta);
        }
    }

    fn increase_rows(&mut self, rows: usize) {
        let old_rows: usize = self.rows;

        assert!(rows >= old_rows);
        assert!(rows <= self.base_samples);

        self.rows = rows;
        self.values.resize(
            self.triangle_base_samples() - triangle_number(self.base_samples - rows),
            0_i32,
        );
    }

    fn shift_right(&mut self, cols: usize) {
        self.increase_base_samples(self.base_samples + cols);

        for row in 0_usize..self.rows {
            self.row_values_mut(row).rotate_right(cols);
        }
    }

    fn extrapolate(&mut self) {
        let mut scratch_row: Vec<i32> = vec![0_i32; self.base_samples];

        self.increase_base_samples(self.base_samples + 1_usize);

        let mut row: usize = 0_usize;

        loop {
            let curr_row_values: &[i32] = self.row_values(row);

            // Subtract 1 since we just increased the base samples: the last right column is empty.
            let curr_row_values_len: usize = curr_row_values.len() - 1_usize;
            let mut curr_row_is_all_zero: bool = curr_row_values[0_usize] == 0_i32;

            for (curr_values_pair, next_value) in curr_row_values[..curr_row_values_len]
                .windows(2_usize)
                .zip(scratch_row.iter_mut())
            {
                let left_value: i32 = curr_values_pair[0_usize];
                let right_value: i32 = curr_values_pair[1_usize];

                *next_value = right_value - left_value;
                curr_row_is_all_zero &= right_value == 0_i32;
            }

            if curr_row_is_all_zero {
                break;
            }

            row += 1_usize;

            if row == self.rows {
                self.increase_rows(row + 1_usize);
            }

            let mut next_row_values: &mut [i32] = self.row_values_mut(row);

            next_row_values = &mut next_row_values[..curr_row_values_len - 1_usize];
            next_row_values.copy_from_slice(&scratch_row[..next_row_values.len()]);

            assert!(row + 1_usize < self.base_samples);
        }

        let mut prev_extrapolated_value: i32 = 0_i32;

        for (rev_row, row) in (0_usize..=row).rev().enumerate() {
            let row_values: &mut [i32] = self.row_values_mut(row);
            let extrapolated_value: i32 = if rev_row == 0_usize {
                0_i32
            } else {
                row_values[row_values.len() - 2_usize] + prev_extrapolated_value
            };

            *row_values.last_mut().unwrap() = extrapolated_value;
            prev_extrapolated_value = extrapolated_value;
        }
    }

    fn rev_extrapolate(&mut self) {
        self.extrapolate();
        self.shift_right(1_usize);

        let mut prev_extrapolated_value: i32 = 0_i32;

        for (rev_row, row) in (0_usize..self.rows).rev().enumerate() {
            let row_values: &mut [i32] = self.row_values_mut(row);
            let extrapolated_value: i32 = if rev_row == 0_usize {
                0_i32
            } else {
                row_values[1_usize] - prev_extrapolated_value
            };

            *row_values.first_mut().unwrap() = extrapolated_value;
            prev_extrapolated_value = extrapolated_value;
        }
    }
}

impl From<&History> for Differential {
    fn from(value: &History) -> Self {
        Self {
            values: value.0.clone(),
            base_samples: value.0.len(),
            rows: 1_usize,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct History(Vec<i32>);

impl History {
    fn extrapolate(&self) -> i32 {
        let mut differential: Differential = self.into();

        differential.extrapolate();

        *differential.row_values(0_usize).last().unwrap()
    }

    fn rev_extrapolate(&self) -> i32 {
        let mut differential: Differential = self.into();

        differential.rev_extrapolate();

        differential.values[0_usize]
    }
}

impl Parse for History {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many1(terminated(parse_integer::<i32>, opt(tag(" ")))), Self)(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<History>);

impl Solution {
    fn iter_extrapolations(&self) -> impl Iterator<Item = i32> + '_ {
        self.0.iter().map(|history| history.extrapolate())
    }

    fn sum_extrapolations(&self) -> i32 {
        self.iter_extrapolations().sum()
    }

    fn iter_rev_extrapolations(&self) -> impl Iterator<Item = i32> + '_ {
        self.0.iter().map(|history| history.rev_extrapolate())
    }

    fn sum_rev_extrapolations(&self) -> i32 {
        self.iter_rev_extrapolations().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many1(terminated(History::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_extrapolations());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_rev_extrapolations());
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
        0 3 6 9 12 15\n\
        1 3 6 10 15 21\n\
        10 13 16 21 30 45\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                History(vec![0, 3, 6, 9, 12, 15]),
                History(vec![1, 3, 6, 10, 15, 21]),
                History(vec![10, 13, 16, 21, 30, 45]),
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_history_extrapolate() {
        for (real_extrapolation, expected_extrapolation) in solution()
            .iter_extrapolations()
            .zip([18_i32, 28_i32, 68_i32])
        {
            assert_eq!(real_extrapolation, expected_extrapolation);
        }
    }

    #[test]
    fn test_history_rev_extrapolate() {
        for (real_extrapolation, expected_extrapolation) in solution()
            .iter_rev_extrapolations()
            .zip([-3_i32, 0_i32, 5_i32])
        {
            assert_eq!(real_extrapolation, expected_extrapolation);
        }
    }
}
