use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, multi::many0, Err, IResult},
};

/* --- Day 3: Perfectly Spherical Houses in a Vacuum ---

Santa is delivering presents to an infinite two-dimensional grid of houses.

He begins by delivering a present to the house at his starting location, and then an elf at the North Pole calls him via radio and tells him where to move next. Moves are always exactly one house to the north (^), south (v), east (>), or west (<). After each move, he delivers another present to the house at his new location.

However, the elf back at the north pole has had a little too much eggnog, and so his directions are a little off, and Santa ends up visiting some houses more than once. How many houses receive at least one present?

For example:

    > delivers presents to 2 houses: one at the starting location, and one to the east.
    ^>v< delivers presents to 4 houses in a square, including twice to the house at his starting/ending location.
    ^v^v^v^v^v delivers a bunch of presents to some very lucky children at only 2 houses.

--- Part Two ---

The next year, to speed up the process, Santa creates a robot version of himself, Robo-Santa, to deliver presents with him.

Santa and Robo-Santa start at the same location (delivering two presents to the same starting house), then take turns moving based on instructions from the elf, who is eggnoggedly reading from the same script as the previous year.

This year, how many houses receive at least one present?

For example:

    ^v delivers presents to 3 houses, because Santa goes north, and then Robo-Santa goes south.
    ^>v< now delivers presents to 3 houses, and Santa and Robo-Santa end up back where they started.
    ^v^v^v^v^v now delivers presents to 11 houses, with Santa going one direction and Robo-Santa going the other. */

#[repr(C)]
struct PresentCountCell(u8);

impl From<u32> for PresentCountCell {
    fn from(value: u32) -> Self {
        Self(match value {
            0_u32 => b'.',
            1_u32..=9_u32 => value as u8 + b'0',
            _ => b'+',
        })
    }
}

/// SAFETY: See `impl From<u32> for PresentCountCell`
unsafe impl IsValidAscii for PresentCountCell {}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    moves: Vec<Direction>,
}

impl Solution {
    fn present_recipient_count(present_count_grid: &Grid2D<u32>) -> usize {
        present_count_grid
            .cells()
            .iter()
            .copied()
            .filter(|&present_count| present_count > 0_u32)
            .count()
    }

    fn present_count_grid_string(present_count_grid: &Grid2D<u32>) -> String {
        Grid2D::try_from_cells_and_dimensions(
            present_count_grid
                .cells()
                .iter()
                .copied()
                .map(PresentCountCell::from)
                .collect(),
            present_count_grid.dimensions(),
        )
        .unwrap()
        .into()
    }

    fn iter_santa_poses<'s>(&'s self, start: IVec2) -> impl Iterator<Item = IVec2> + 's {
        let mut santa_pos: IVec2 = start;

        [santa_pos]
            .into_iter()
            .chain(self.moves.iter().map(move |dir| {
                santa_pos += dir.vec();

                santa_pos
            }))
    }

    fn iter_santa_and_robo_santa_poses<'s>(
        &'s self,
        start: IVec2,
    ) -> impl Iterator<Item = IVec2> + 's {
        let mut santa_pos: IVec2 = start;
        let mut robo_santa_pos: IVec2 = start;

        [santa_pos, robo_santa_pos]
            .into_iter()
            .chain(self.moves.chunks_exact(2_usize).flat_map(move |dirs| {
                santa_pos += dirs.first().unwrap().vec();
                robo_santa_pos += dirs.last().unwrap().vec();

                [santa_pos, robo_santa_pos]
            }))
    }

    fn present_count_grid_start_and_dimensions<
        's,
        I: Iterator<Item = IVec2> + 's,
        F: Fn(&'s Solution, IVec2) -> I,
    >(
        &'s self,
        iter_poses: F,
    ) -> (IVec2, IVec2) {
        let (min, max): (IVec2, IVec2) = iter_poses(self, IVec2::ZERO)
            .fold((IVec2::MAX, IVec2::MIN), |(min, max), pos| {
                (min.min(pos), max.max(pos))
            });

        (-min, max - min + IVec2::ONE)
    }

    fn present_count_grid<
        's,
        I: Iterator<Item = IVec2> + 's,
        F: Copy + Fn(&'s Solution, IVec2) -> I,
    >(
        &'s self,
        iter_poses: F,
    ) -> Grid2D<u32> {
        let (start, dimensions): (IVec2, IVec2) =
            self.present_count_grid_start_and_dimensions(iter_poses);
        let mut present_count_grid: Grid2D<u32> = Grid2D::default(dimensions);

        for pos in iter_poses(self, start) {
            *present_count_grid.get_mut(pos).unwrap() += 1_u32;
        }

        present_count_grid
    }

    fn santa_present_count_grid(&self) -> Grid2D<u32> {
        self.present_count_grid(Self::iter_santa_poses)
    }

    fn santa_and_robo_santa_present_count_grid(&self) -> Grid2D<u32> {
        self.present_count_grid(Self::iter_santa_and_robo_santa_poses)
    }

    fn santa_present_recipient_count(&self) -> usize {
        Self::present_recipient_count(&self.santa_present_count_grid())
    }

    fn santa_and_robo_santa_present_recipient_count(&self) -> usize {
        Self::present_recipient_count(&self.santa_and_robo_santa_present_count_grid())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(Direction::parse_from_nesw("^>v<")), |moves| Self {
            moves,
        })(input)
    }
}

impl RunQuestions for Solution {
    /// Now I have a better parser for directions!
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.santa_present_recipient_count());
        } else {
            let santa_present_count_grid: Grid2D<u32> = self.santa_present_count_grid();

            dbg!(Self::present_recipient_count(&santa_present_count_grid));

            println!(
                "{}",
                Self::present_count_grid_string(&santa_present_count_grid)
            )
        }
    }

    /// Fun adding the visualization for this.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.santa_and_robo_santa_present_recipient_count());
        } else {
            let santa_and_robo_santa_present_count_grid: Grid2D<u32> =
                self.santa_and_robo_santa_present_count_grid();

            dbg!(Self::present_recipient_count(
                &santa_and_robo_santa_present_count_grid
            ));

            println!(
                "{}",
                Self::present_count_grid_string(&santa_and_robo_santa_present_count_grid)
            )
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

    const SOLUTION_STRS: &'static [&'static str] = &[">", "^v", "^>v<", "^v^v^v^v^v"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Direction::{East as E, North as N, South as S, West as W};

            vec![
                Solution { moves: vec![E] },
                Solution { moves: vec![N, S] },
                Solution {
                    moves: vec![N, E, S, W],
                },
                Solution {
                    moves: vec![N, S, N, S, N, S, N, S, N, S],
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
    fn test_santa_present_count_grid() {
        for (index, santa_present_count_grid) in [
            Grid2D::try_from_cells_and_dimensions(vec![1_u32, 1_u32], (2_i32, 1_i32).into())
                .unwrap(),
            Grid2D::try_from_cells_and_dimensions(vec![1_u32, 2_u32], (1_i32, 2_i32).into())
                .unwrap(),
            Grid2D::try_from_cells_and_dimensions(
                vec![1_u32, 1_u32, 2_u32, 1_u32],
                (2_i32, 2_i32).into(),
            )
            .unwrap(),
            Grid2D::try_from_cells_and_dimensions(vec![5_u32, 6_u32], (1_i32, 2_i32).into())
                .unwrap(),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).santa_present_count_grid(),
                santa_present_count_grid
            );
        }
    }

    #[test]
    fn test_santa_present_recipient_count() {
        for (index, santa_present_recipient_count) in
            [2_usize, 2_usize, 4_usize, 2_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).santa_present_recipient_count(),
                santa_present_recipient_count
            );
        }
    }

    #[test]
    fn test_santa_and_robo_santa_present_count_grid() {
        for (index, santa_and_robo_santa_present_count_grid) in [
            Grid2D::try_from_cells_and_dimensions(vec![2_u32], (1_i32, 1_i32).into()).unwrap(),
            Grid2D::try_from_cells_and_dimensions(vec![1_u32, 2_u32, 1_u32], (1_i32, 3_i32).into())
                .unwrap(),
            Grid2D::try_from_cells_and_dimensions(
                vec![1_u32, 0_u32, 4_u32, 1_u32],
                (2_i32, 2_i32).into(),
            )
            .unwrap(),
            Grid2D::try_from_cells_and_dimensions(
                vec![
                    1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 2_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32,
                ],
                (1_i32, 11_i32).into(),
            )
            .unwrap(),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).santa_and_robo_santa_present_count_grid(),
                santa_and_robo_santa_present_count_grid
            );
        }
    }

    #[test]
    fn test_santa_and_robo_santa_present_recipient_count() {
        for (index, santa_and_robo_santa_present_recipient_count) in
            [1_usize, 3_usize, 3_usize, 11_usize]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).santa_and_robo_santa_present_recipient_count(),
                santa_and_robo_santa_present_recipient_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
