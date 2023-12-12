use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
    },
};

define_cell! {
    #[repr(u8)]
    #[derive(Copy, Clone, Default, PartialEq)]
    enum Cell {
        #[default]
        EmptySpace = EMPTY_SPACE = b'.',
        Galaxy = GALAXY = b'#',
    }
}

impl Debug for Cell
where
    Self: IsValidAscii,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // SAFETY: Guaranteed by `IsValidAscii`
        f.write_str(unsafe { from_utf8_unchecked(&[*self as u8]) })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(Vec<IVec2>);

impl Solution {
    const SMALL_SCALAR: i32 = 2_i32;
    const LARGE_SCALAR: i32 = 1000000_i32;

    fn expand(&self, scalar: i32) -> Self {
        let mut x_values: Vec<i32> = Vec::new();
        let mut y_values: Vec<i32> = Vec::new();

        for pos in self.0.iter() {
            x_values.push(pos.x);
            y_values.push(pos.y);
        }

        x_values.sort();
        x_values.dedup();
        y_values.dedup();

        let mut solution: Self = self.clone();
        let expand = |values: &[i32], value: i32| {
            let index: i32 = values.binary_search(&value).unwrap() as i32;

            scalar * (value - index) + index
        };

        for pos in solution.0.iter_mut() {
            pos.x = expand(&x_values, pos.x);
            pos.y = expand(&y_values, pos.y);
        }

        solution
    }

    fn iter_dists(&self) -> impl Iterator<Item = u64> + '_ {
        self.0
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(index, pos_a)| {
                self.0[index + 1_usize..]
                    .iter()
                    .copied()
                    .map(move |pos_b| manhattan_distance_2d(pos_a, pos_b) as u64)
            })
    }

    fn sum_small_expanded_dists(&self) -> u64 {
        self.expand(Self::SMALL_SCALAR).iter_dists().sum()
    }

    fn sum_large_expanded_dists(&self) -> u64 {
        self.expand(Self::LARGE_SCALAR).iter_dists().sum()
    }
}

impl From<Grid2D<Cell>> for Solution {
    fn from(value: Grid2D<Cell>) -> Self {
        Self(
            value
                .cells()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, cell)| {
                    if cell == Cell::Galaxy {
                        Some(value.pos_from_index(index))
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::<Cell>::parse, Self::from)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_small_expanded_dists());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_large_expanded_dists());
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
        "\
        ...#......\n\
        .......#..\n\
        #.........\n\
        ..........\n\
        ......#...\n\
        .#........\n\
        .........#\n\
        ..........\n\
        .......#..\n\
        #...#.....\n",
        "\
        ....#........\n\
        .........#...\n\
        #............\n\
        .............\n\
        .............\n\
        ........#....\n\
        .#...........\n\
        ............#\n\
        .............\n\
        .............\n\
        .........#...\n\
        #....#.......\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    IVec2::new(3_i32, 0_i32),
                    IVec2::new(7_i32, 1_i32),
                    IVec2::new(0_i32, 2_i32),
                    IVec2::new(6_i32, 4_i32),
                    IVec2::new(1_i32, 5_i32),
                    IVec2::new(9_i32, 6_i32),
                    IVec2::new(7_i32, 8_i32),
                    IVec2::new(0_i32, 9_i32),
                    IVec2::new(4_i32, 9_i32),
                ]),
                Solution(vec![
                    IVec2::new(4_i32, 0_i32),
                    IVec2::new(9_i32, 1_i32),
                    IVec2::new(0_i32, 2_i32),
                    IVec2::new(8_i32, 5_i32),
                    IVec2::new(1_i32, 6_i32),
                    IVec2::new(12_i32, 7_i32),
                    IVec2::new(9_i32, 10_i32),
                    IVec2::new(0_i32, 11_i32),
                    IVec2::new(5_i32, 11_i32),
                ]),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_expand() {
        assert_eq!(
            &solution(0_usize).expand(Solution::SMALL_SCALAR),
            solution(1_usize)
        )
    }

    #[test]
    fn test_sum_small_expanded_dists() {
        assert_eq!(solution(0_usize).sum_small_expanded_dists(), 374_u64);
    }

    #[test]
    fn test_large_expansion() {
        assert_eq!(
            solution(0_usize).expand(10_i32).iter_dists().sum::<u64>(),
            1030_u64
        );
        assert_eq!(
            solution(0_usize).expand(100_i32).iter_dists().sum::<u64>(),
            8410_u64
        );
    }
}
