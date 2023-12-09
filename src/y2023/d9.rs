use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many1_count,
        sequence::terminated,
        Err, IResult,
    },
    std::ops::Range,
};

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
#[derive(Default)]
struct History {
    value_range: Range<u16>,
    front_extrapolation: i32,
    back_extrapolation: i32,
}

impl History {
    const NEXT_ROW_POS_DELTA: IVec2 = IVec2::new(-1_i32, 1_i32);

    /// Fills the grid with a (partial) pyramid of differential values until the last row is all 0.
    /// It is assumed the first row of the grid is filled with the corresponding values for this
    /// history.
    ///
    /// Returns the number of filled rows.
    fn fill_grid(&self, grid: &mut Grid2D<i32>) -> i32 {
        let cols: i32 = self.value_range.len() as i32;

        let mut rows: i32 = 1_i32;

        for row_pos in CellIter2D::try_from(IVec2::ZERO..IVec2::Y * cols).unwrap() {
            let mut prev_value: i32 = *grid.get(row_pos).unwrap();
            let mut row_is_all_zero: bool = prev_value == 0_i32;

            for curr_pos in CellIter2D::try_from(row_pos..row_pos + IVec2::X * (cols - row_pos.y))
                .unwrap()
                .skip(1_usize)
            {
                let curr_value: i32 = *grid.get(curr_pos).unwrap();

                *grid.get_mut(curr_pos + Self::NEXT_ROW_POS_DELTA).unwrap() =
                    curr_value - prev_value;
                row_is_all_zero &= curr_value == 0_i32;
                prev_value = curr_value;
            }

            if row_is_all_zero {
                rows = row_pos.y + 1_i32;

                break;
            }
        }

        rows
    }

    fn extrapolation<F: Fn(i32) -> IVec2>(
        &self,
        grid: &Grid2D<i32>,
        rows: i32,
        get_row_pos: F,
        delta_sign: i32,
    ) -> i32 {
        (0_i32..rows)
            .rev()
            .enumerate()
            .fold(0_i32, |prev_extrapolation, (rev_row, row)| {
                if rev_row == 0_usize {
                    0_i32
                } else {
                    *grid.get(get_row_pos(row)).unwrap() + delta_sign * prev_extrapolation
                }
            })
    }

    fn front_extrapolation(&self, grid: &Grid2D<i32>, rows: i32) -> i32 {
        self.extrapolation(grid, rows, |row| IVec2::new(0_i32, row), -1_i32)
    }

    fn back_extrapolation(&self, grid: &Grid2D<i32>, rows: i32) -> i32 {
        self.extrapolation(
            grid,
            rows,
            |row| IVec2::new(self.value_range.len() as i32 - row - 1_i32, row),
            1_i32,
        )
    }

    fn extrapolate(&mut self, grid: &mut Grid2D<i32>) {
        let rows: i32 = self.fill_grid(grid);

        self.front_extrapolation = self.front_extrapolation(grid, rows);
        self.back_extrapolation = self.back_extrapolation(grid, rows);
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    histories: Vec<History>,
    values: Vec<i32>,
}

impl Solution {
    fn parse_internal<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        let input: &str = many1_count(|input: &'i str| {
            let start: u16 = solution.values.len() as u16;

            let input: &str = terminated(
                many1_count(|input: &'i str| {
                    map(terminated(parse_integer::<i32>, opt(tag(" "))), |value| {
                        solution.values.push(value);
                    })(input)
                }),
                opt(line_ending),
            )(input)?
            .0;

            let end: u16 = solution.values.len() as u16;

            solution.histories.push(History {
                value_range: start..end,
                ..History::default()
            });

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, solution))
    }

    fn extrapolate(&mut self) {
        let max_cols: i32 = self
            .histories
            .iter()
            .map(|history| history.value_range.len())
            .max()
            .unwrap_or_default() as i32;

        let mut grid: Grid2D<i32> = Grid2D::default(IVec2::new(max_cols, max_cols));

        for history in self.histories.iter_mut() {
            let values: &[i32] = &self.values[history.value_range.as_range_usize()];

            grid.cells_mut()[..values.len()].copy_from_slice(values);
            history.extrapolate(&mut grid);
        }
    }

    fn iter_back_extrapolations(&self) -> impl Iterator<Item = i32> + '_ {
        self.histories
            .iter()
            .map(|history| history.back_extrapolation)
    }

    fn sum_back_extrapolations(&self) -> i32 {
        self.iter_back_extrapolations().sum()
    }

    fn iter_front_extrapolations(&self) -> impl Iterator<Item = i32> + '_ {
        self.histories
            .iter()
            .map(|history| history.front_extrapolation)
    }

    fn sum_front_extrapolations(&self) -> i32 {
        self.iter_front_extrapolations().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Self::parse_internal, |mut solution| {
            solution.extrapolate();

            solution
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_back_extrapolations());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_front_extrapolations());
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

        ONCE_LOCK.get_or_init(|| Solution {
            histories: vec![
                History {
                    value_range: 0_u16..6_u16,
                    ..History::default()
                },
                History {
                    value_range: 6_u16..12_u16,
                    ..History::default()
                },
                History {
                    value_range: 12_u16..18_u16,
                    ..History::default()
                },
            ],
            values: vec![
                0, 3, 6, 9, 12, 15, 1, 3, 6, 10, 15, 21, 10, 13, 16, 21, 30, 45,
            ],
        })
    }

    #[test]
    fn test_parse_internal() {
        assert_eq!(
            Solution::parse_internal(SOLUTION_STR)
                .map(|(_, solution)| solution)
                .as_ref(),
            Ok(solution())
        );
    }

    #[test]
    fn test_iter_back_extrapolations() {
        let mut solution: Solution = solution().clone();

        solution.extrapolate();

        for (real_extrapolation, expected_extrapolation) in solution
            .iter_back_extrapolations()
            .zip([18_i32, 28_i32, 68_i32])
        {
            assert_eq!(real_extrapolation, expected_extrapolation);
        }
    }

    #[test]
    fn test_iter_front_extrapolations() {
        let mut solution: Solution = solution().clone();

        solution.extrapolate();

        for (real_extrapolation, expected_extrapolation) in solution
            .iter_front_extrapolations()
            .zip([-3_i32, 0_i32, 5_i32])
        {
            assert_eq!(real_extrapolation, expected_extrapolation);
        }
    }
}
