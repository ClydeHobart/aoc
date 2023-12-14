use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
};

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    enum Cell {
        RoundedRock = ROUNDED_ROCK = b'O',
        CubeShapedRock = CUBE_SHAPED_ROCK = b'#',
        #[default]
        EmptySpace = EMPTY_SPACE = b'.',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(Grid2D<Cell>);

impl Solution {
    const DIR: Direction = Direction::North;
    const MANY_SPIN_CYCLES: usize = 1_000_000_000_usize;
    const SPIN_CYCLE_STEP: usize = 100_usize;

    fn add_rounded_rocks_to_row_or_col(
        &mut self,
        start_pos: IVec2,
        end_pos: IVec2,
        rounded_rock_count: usize,
    ) {
        if start_pos != end_pos && rounded_rock_count > 0_usize {
            for (index, rounded_rock_pos) in CellIter2D::try_from(start_pos..end_pos)
                .unwrap()
                .enumerate()
            {
                *self.0.get_mut(rounded_rock_pos).unwrap() = if index < rounded_rock_count {
                    Cell::RoundedRock
                } else {
                    Cell::EmptySpace
                };
            }
        }
    }

    fn tilt(&mut self, dir: Direction) {
        let dimensions: IVec2 = self.0.dimensions();

        // It helps for putting the rounded rocks into place to have the reverse direction
        let dir: Direction = dir.rev();
        let dir_vec: IVec2 = dir.vec();
        let dir_vec_times_dimensions: IVec2 = dir_vec * dimensions;

        for mut curr_pos in CellIter2D::corner(&self.0, dir.prev()) {
            let mut start_pos: IVec2 = curr_pos;
            let mut rounded_rock_count: usize = 0_usize;
            let end_pos: IVec2 = curr_pos + dir_vec_times_dimensions;

            while curr_pos != end_pos {
                let next_pos: IVec2 = curr_pos + dir_vec;
                let cell: Cell = *self.0.get(curr_pos).unwrap();

                match cell {
                    Cell::CubeShapedRock => {
                        *self.0.get_mut(curr_pos).unwrap() = Cell::CubeShapedRock;
                        self.add_rounded_rocks_to_row_or_col(
                            start_pos,
                            curr_pos,
                            rounded_rock_count,
                        );
                        start_pos = next_pos;
                        rounded_rock_count = 0_usize;
                    }
                    Cell::RoundedRock => {
                        rounded_rock_count += 1_usize;
                    }
                    _ => (),
                }

                curr_pos = next_pos;
            }

            self.add_rounded_rocks_to_row_or_col(start_pos, curr_pos, rounded_rock_count);
        }
    }

    fn iter_rounded_rocks(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.0
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| {
                if *cell == Cell::RoundedRock {
                    Some(self.0.pos_from_index(index))
                } else {
                    None
                }
            })
    }

    fn total_load(&self, dir: Direction) -> i32 {
        let dir_vec: IVec2 = dir.vec();
        let dist_is_load_pos: IVec2 = CellIter2D::corner(&self.0, dir).next().unwrap() - dir_vec;

        self.iter_rounded_rocks()
            .map(|pos| manhattan_distance_2d(IVec2::ZERO, dir_vec * (pos - dist_is_load_pos)))
            .sum()
    }

    fn tilted_total_load(&self, dir: Direction) -> i32 {
        let mut solution: Solution = self.clone();

        solution.tilt(dir);

        solution.total_load(dir)
    }

    fn spin_cycle(&mut self) {
        for dir in [
            Direction::North,
            Direction::West,
            Direction::South,
            Direction::East,
        ] {
            self.tilt(dir);
        }
    }

    fn try_find_period(total_load_samples: &[i32]) -> Option<usize> {
        (2_usize..=total_load_samples.len() / 2_usize)
            .into_iter()
            .find(|period| {
                let initial_period_total_load_samples: &[i32] = &total_load_samples[..*period];

                total_load_samples[*period..]
                    .chunks(*period)
                    .all(|period_total_load_samples| {
                        period_total_load_samples
                            == &initial_period_total_load_samples[..period_total_load_samples.len()]
                    })
            })
    }

    fn extrapolate_total_loads_after_many_spin_cycles(&self) -> i32 {
        let mut solution: Self = self.clone();
        let mut total_load_samples: Vec<i32> = Vec::with_capacity(Self::SPIN_CYCLE_STEP);
        let mut spin_cycles: usize = 0_usize;

        let period: usize = loop {
            spin_cycles += Self::SPIN_CYCLE_STEP;
            total_load_samples.clear();
            total_load_samples.extend((0_usize..Self::SPIN_CYCLE_STEP).map(|_| {
                let total_load: i32 = solution.total_load(Self::DIR);

                solution.spin_cycle();

                total_load
            }));

            if let Some(period) = Self::try_find_period(&total_load_samples) {
                break period;
            }
        };
        let period_total_load_samples: &[i32] =
            &total_load_samples[Self::SPIN_CYCLE_STEP - period..];
        let remaining_spin_cycles: usize = (Self::MANY_SPIN_CYCLES - spin_cycles) % period;

        period_total_load_samples[remaining_spin_cycles]
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::<Cell>::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let mut solution: Self = self.clone();

            solution.tilt(Self::DIR);

            let total_tilted_load: i32 = solution.total_load(Self::DIR);

            println!(
                "total_tilted_load == {total_tilted_load}\n\n{}",
                String::from(solution.0)
            );
        } else {
            dbg!(self.tilted_total_load(Self::DIR));
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.extrapolate_total_loads_after_many_spin_cycles());
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
        O....#....\n\
        O.OO#....#\n\
        .....##...\n\
        OO.#O....O\n\
        .O.....O#.\n\
        O.#..O.#.#\n\
        ..O..#O..O\n\
        .......O..\n\
        #....###..\n\
        #OO..#....\n",
        "\
        OOOO.#.O..\n\
        OO..#....#\n\
        OO..O##..O\n\
        O..#.OO...\n\
        ........#.\n\
        ..#....#.#\n\
        ..O..#.O.O\n\
        ..O.......\n\
        #....###..\n\
        #....#....\n",
        "\
        .....#....\n\
        ....#...O#\n\
        ...OO##...\n\
        .OO#......\n\
        .....OOO#.\n\
        .O#...O#.#\n\
        ....O#....\n\
        ......OOOO\n\
        #...O###..\n\
        #..OO#....\n",
        "\
        .....#....\n\
        ....#...O#\n\
        .....##...\n\
        ..O#......\n\
        .....OOO#.\n\
        .O#...O#.#\n\
        ....O#...O\n\
        .......OOO\n\
        #..OO###..\n\
        #.OOO#...O\n",
        "\
        .....#....\n\
        ....#...O#\n\
        .....##...\n\
        ..O#......\n\
        .....OOO#.\n\
        .O#...O#.#\n\
        ....O#...O\n\
        .......OOO\n\
        #...O###.O\n\
        #.OOO#...O\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        use Cell::{CubeShapedRock as C, EmptySpace as E, RoundedRock as R};

        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(
                    Grid2D::try_from_cells_and_width(
                        vec![
                            R, E, E, E, E, C, E, E, E, E, R, E, R, R, C, E, E, E, E, C, E, E, E, E,
                            E, C, C, E, E, E, R, R, E, C, R, E, E, E, E, R, E, R, E, E, E, E, E, R,
                            C, E, R, E, C, E, E, R, E, C, E, C, E, E, R, E, E, C, R, E, E, R, E, E,
                            E, E, E, E, E, R, E, E, C, E, E, E, E, C, C, C, E, E, C, R, R, E, E, C,
                            E, E, E, E,
                        ],
                        10_usize,
                    )
                    .unwrap(),
                ),
                Solution(
                    Grid2D::try_from_cells_and_width(
                        vec![
                            R, R, R, R, E, C, E, R, E, E, R, R, E, E, C, E, E, E, E, C, R, R, E, E,
                            R, C, C, E, E, R, R, E, E, C, E, R, R, E, E, E, E, E, E, E, E, E, E, E,
                            C, E, E, E, C, E, E, E, E, C, E, C, E, E, R, E, E, C, E, R, E, R, E, E,
                            R, E, E, E, E, E, E, E, C, E, E, E, E, C, C, C, E, E, C, E, E, E, E, C,
                            E, E, E, E,
                        ],
                        10_usize,
                    )
                    .unwrap(),
                ),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS[..2_usize].into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_tilt() {
        let mut local_solution: Solution = solution(0_usize).clone();

        local_solution.tilt(Solution::DIR);

        assert_eq!(&local_solution, solution(1_usize));
    }

    #[test]
    fn test_total_load() {
        assert_eq!(solution(1_usize).total_load(Solution::DIR), 136_i32);
    }

    #[test]
    fn test_tilted_total_load() {
        assert_eq!(solution(0_usize).tilted_total_load(Solution::DIR), 136_i32);
    }

    #[test]
    fn test_spin_cycle() {
        const SOLUTION_STR_INDICES: [usize; 4_usize] = [0_usize, 2_usize, 3_usize, 4_usize];

        for solution_str_index_pair in SOLUTION_STR_INDICES.windows(2_usize) {
            let mut solution: Solution = SOLUTION_STRS[solution_str_index_pair[0_usize]]
                .try_into()
                .unwrap();

            solution.spin_cycle();

            assert_eq!(
                solution,
                SOLUTION_STRS[solution_str_index_pair[1_usize]]
                    .try_into()
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_extrapolate_total_loads_after_many_spin_cycles() {
        assert_eq!(
            solution(0_usize).extrapolate_total_loads_after_many_spin_cycles(),
            64_i32
        );
    }
}
