use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    strum::IntoEnumIterator,
};

/* --- Day 20: Race Condition ---

The Historians are quite pixelated again. This time, a massive, black building looms over you - you're right outside the CPU!

While The Historians get to work, a nearby program sees that you're idle and challenges you to a race. Apparently, you've arrived just in time for the frequently-held race condition festival!

The race takes place on a particularly long and twisting code path; programs compete to see who can finish in the fewest picoseconds. The winner even gets their very own mutex!

They hand you a map of the racetrack (your puzzle input). For example:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

The map consists of track (.) - including the start (S) and end (E) positions (both of which also count as track) - and walls (#).

When a program runs through the racetrack, it starts at the start position. Then, it is allowed to move up, down, left, or right; each such move takes 1 picosecond. The goal is to reach the end position as quickly as possible. In this example racetrack, the fastest time is 84 picoseconds.

Because there is only a single path from the start to the end and the programs all go the same speed, the races used to be pretty boring. To make things more interesting, they introduced a new rule to the races: programs are allowed to cheat.

The rules for cheating are very strict. Exactly once during a race, a program may disable collision for up to 2 picoseconds. This allows the program to pass through walls as if they were regular track. At the end of the cheat, the program must be back on normal track again; otherwise, it will receive a segmentation fault and get disqualified.

So, a program could complete the course in 72 picoseconds (saving 12 picoseconds) by cheating for the two moves marked 1 and 2:

###############
#...#...12....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

Or, a program could complete the course in 64 picoseconds (saving 20 picoseconds) by cheating for the two moves marked 1 and 2:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...12..#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

This cheat saves 38 picoseconds:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.####1##.###
#...###.2.#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

This cheat saves 64 picoseconds and takes the program directly to the end:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..21...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

Each cheat has a psinct start position (the position where the cheat is activated, just before the first move that is allowed to go through walls) and end position; cheats are uniquely identified by their start position and end position.

In this example, the total number of cheats (grouped by the amount of time they save) are as follows:

    There are 14 cheats that save 2 picoseconds.
    There are 14 cheats that save 4 picoseconds.
    There are 2 cheats that save 6 picoseconds.
    There are 4 cheats that save 8 picoseconds.
    There are 2 cheats that save 10 picoseconds.
    There are 3 cheats that save 12 picoseconds.
    There is one cheat that saves 20 picoseconds.
    There is one cheat that saves 36 picoseconds.
    There is one cheat that saves 38 picoseconds.
    There is one cheat that saves 40 picoseconds.
    There is one cheat that saves 64 picoseconds.

You aren't sure what the conditions of the racetrack will be like, so to give yourself as many options as possible, you'll need a list of the best cheats. How many cheats would save you at least 100 picoseconds?

--- Part Two ---

The programs seem perplexed by your list of cheats. Apparently, the two-picosecond cheating rule was deprecated several milliseconds ago! The latest version of the cheating rule permits a single cheat that instead lasts at most 20 picoseconds.

Now, in addition to all the cheats that were possible in just two picoseconds, many more cheats are possible. This six-picosecond cheat saves 76 picoseconds:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#1#####.#.#.###
#2#####.#.#...#
#3#####.#.###.#
#456.E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

Because this cheat has the same start and end positions as the one above, it's the same cheat, even though the path taken during the cheat is different:

###############
#...#...#.....#
#.#.#.#.#.###.#
#S12..#.#.#...#
###3###.#.#.###
###4###.#.#...#
###5###.#.###.#
###6.E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############

Cheats don't need to use all 20 picoseconds; cheats can last any amount of time up to and including 20 picoseconds (but can still only end when the program is on normal track). Any cheat time not used is lost; it can't be saved for another cheat later.

You'll still need a list of the best cheats, but now there are even more to choose between. Here are the quantities of cheats in this example that save 50 picoseconds or more:

    There are 32 cheats that save 50 picoseconds.
    There are 31 cheats that save 52 picoseconds.
    There are 29 cheats that save 54 picoseconds.
    There are 39 cheats that save 56 picoseconds.
    There are 25 cheats that save 58 picoseconds.
    There are 23 cheats that save 60 picoseconds.
    There are 20 cheats that save 62 picoseconds.
    There are 19 cheats that save 64 picoseconds.
    There are 12 cheats that save 66 picoseconds.
    There are 14 cheats that save 68 picoseconds.
    There are 12 cheats that save 70 picoseconds.
    There are 22 cheats that save 72 picoseconds.
    There are 4 cheats that save 74 picoseconds.
    There are 3 cheats that save 76 picoseconds.

Find the best cheats using the updated cheating rules. How many cheats would save you at least 100 picoseconds? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Track = TRACK = b'.',
        Wall = WALL = b'#',
        Start = START = b'S',
        End = END = b'E',
    }
}

struct PsGridFiller<'s> {
    solution: &'s mut Solution,
}

impl<'s> BreadthFirstSearch for PsGridFiller<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.solution.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.solution.end
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        Vec::new()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            self.solution
                .cell_grid
                .get(neighbor)
                .map(|&cell| (cell == Cell::Track).then_some(neighbor))
                .flatten()
        }));
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let from_ps: u16 = *self.solution.ps_grid.get(*from).unwrap();

        *self.solution.ps_grid.get_mut(*to).unwrap() = from_ps + 1_u16;
    }

    fn reset(&mut self) {
        self.solution.ps_grid.cells_mut().fill(u16::MAX);
        *self.solution.ps_grid.get_mut(self.solution.start).unwrap() = 0_u16;
    }
}

struct Cheat {
    saved_ps: u16,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    cell_grid: Grid2D<Cell>,
    start: IVec2,
    end: IVec2,
    ps_grid: Grid2D<u16>,
}

impl Solution {
    const MIN_SAVED_PS: u16 = 100_u16;
    const Q1_MAX_CHEAT_PS: u16 = 2_u16;
    const Q2_MAX_CHEAT_PS: u16 = 20_u16;

    fn try_new(mut cell_grid: Grid2D<Cell>, start: IVec2, end: IVec2) -> Option<Self> {
        *cell_grid.get_mut(start).unwrap() = Cell::Track;
        *cell_grid.get_mut(end).unwrap() = Cell::Track;

        let ps_grid: Grid2D<u16> = Grid2D::try_from_cells_and_dimensions(
            vec![u16::MAX; cell_grid.cells().len()],
            cell_grid.dimensions(),
        )
        .unwrap();

        let mut solution: Self = Self {
            cell_grid,
            start,
            end,
            ps_grid,
        };

        solution.fill_ps_grid();

        (*solution.ps_grid.get(end).unwrap() as usize + 1_usize
            == solution
                .cell_grid
                .iter_positions_with_cell(&Cell::Track)
                .count())
        .then_some(solution)
    }

    fn iter_cheats(&self, max_cheat_ps: u16) -> impl Iterator<Item = Cheat> + '_ {
        (max_cheat_ps > 0_u16)
            .then(move || {
                let min_cheat_end_delta: i32 = -(max_cheat_ps as i32);
                let max_cheat_end_delta: i32 = max_cheat_ps as i32;
                self.cell_grid
                    .iter_positions_with_cell(&Cell::Track)
                    .flat_map(move |cheat_start| {
                        let cheat_start_ps: u16 = *self.ps_grid.get(cheat_start).unwrap();

                        (cheat_start.y + min_cheat_end_delta..=cheat_start.y + max_cheat_end_delta)
                            .flat_map(move |cheat_end_y| {
                                (cheat_start.x + min_cheat_end_delta
                                    ..=cheat_start.x + max_cheat_end_delta)
                                    .filter_map(move |cheat_end_x| {
                                        let cheat_end: IVec2 = (cheat_end_x, cheat_end_y).into();

                                        self.ps_grid.get(cheat_end).map(|&cheat_end_ps| {
                                            (cheat_start, cheat_end, cheat_start_ps, cheat_end_ps)
                                        })
                                    })
                            })
                    })
                    .filter_map(
                        move |(cheat_start, cheat_end, cheat_start_ps, cheat_end_ps)| {
                            let cheat_ps: u16 =
                                manhattan_distance_2d(cheat_start, cheat_end) as u16;

                            (cheat_ps <= max_cheat_ps
                                && cheat_end_ps != u16::MAX
                                && cheat_end_ps > cheat_start_ps + cheat_ps)
                                .then(|| Cheat {
                                    saved_ps: cheat_end_ps - cheat_start_ps - cheat_ps,
                                })
                        },
                    )
            })
            .into_iter()
            .flatten()
    }

    fn count_cheats(&self, max_cheat_ps: u16, min_saved_ps: u16) -> usize {
        self.iter_cheats(max_cheat_ps)
            .filter(|cheat| cheat.saved_ps >= min_saved_ps)
            .count()
    }

    fn fill_ps_grid(&mut self) {
        PsGridFiller { solution: self }.run();
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |cell_grid| {
            SmallPos::are_dimensions_valid(cell_grid.dimensions())
                .then(|| {
                    cell_grid
                        .try_find_single_position_with_cell(&Cell::Start)
                        .zip(cell_grid.try_find_single_position_with_cell(&Cell::End))
                })
                .flatten()
                .and_then(|(start, end)| Self::try_new(cell_grid, start, end))
        })(input)
    }
}

impl RunQuestions for Solution {
    /// love rust iterators
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_cheats(Self::Q1_MAX_CHEAT_PS, Self::MIN_SAVED_PS));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_cheats(Self::Q2_MAX_CHEAT_PS, Self::MIN_SAVED_PS));
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
    use {
        super::*,
        std::{collections::HashMap, sync::OnceLock},
    };

    const SOLUTION_STRS: &'static [&'static str] = &["\
        ###############\n\
        #...#...#.....#\n\
        #.#.#.#.#.###.#\n\
        #S#...#.#.#...#\n\
        #######.#.#.###\n\
        #######.#.#...#\n\
        #######.#.###.#\n\
        ###..E#...#...#\n\
        ###.#######.###\n\
        #...###...#...#\n\
        #.#####.#.###.#\n\
        #.#...#.#.#...#\n\
        #.#.#.#.#.#.###\n\
        #...#...#...###\n\
        ###############\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            const M: u16 = u16::MAX;

            use Cell::{Track as T, Wall as W};

            vec![Solution {
                cell_grid: Grid2D::try_from_cells_and_dimensions(
                    vec![
                        W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, T, T, T, W, T, T, T, W, T,
                        T, T, T, T, W, W, T, W, T, W, T, W, T, W, T, W, W, W, T, W, W, T, W, T, T,
                        T, W, T, W, T, W, T, T, T, W, W, W, W, W, W, W, W, T, W, T, W, T, W, W, W,
                        W, W, W, W, W, W, W, T, W, T, W, T, T, T, W, W, W, W, W, W, W, W, T, W, T,
                        W, W, W, T, W, W, W, W, T, T, T, W, T, T, T, W, T, T, T, W, W, W, W, T, W,
                        W, W, W, W, W, W, T, W, W, W, W, T, T, T, W, W, W, T, T, T, W, T, T, T, W,
                        W, T, W, W, W, W, W, T, W, T, W, W, W, T, W, W, T, W, T, T, T, W, T, W, T,
                        W, T, T, T, W, W, T, W, T, W, T, W, T, W, T, W, T, W, W, W, W, T, T, T, W,
                        T, T, T, W, T, T, T, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W,
                    ],
                    15_i32 * IVec2::ONE,
                )
                .unwrap(),
                start: (1_i32, 3_i32).into(),
                end: (5_i32, 7_i32).into(),
                ps_grid: Grid2D::try_from_cells_and_dimensions(
                    vec![
                        M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, 2, 3, 4, M, 10, 11, 12, M,
                        26, 27, 28, 29, 30, M, M, 1, M, 5, M, 9, M, 13, M, 25, M, M, M, 31, M, M,
                        0, M, 6, 7, 8, M, 14, M, 24, M, 34, 33, 32, M, M, M, M, M, M, M, M, 15, M,
                        23, M, 35, M, M, M, M, M, M, M, M, M, M, 16, M, 22, M, 36, 37, 38, M, M, M,
                        M, M, M, M, M, 17, M, 21, M, M, M, 39, M, M, M, M, 82, 83, 84, M, 18, 19,
                        20, M, 42, 41, 40, M, M, M, M, 81, M, M, M, M, M, M, M, 43, M, M, M, M, 78,
                        79, 80, M, M, M, 60, 59, 58, M, 44, 45, 46, M, M, 77, M, M, M, M, M, 61, M,
                        57, M, M, M, 47, M, M, 76, M, 70, 69, 68, M, 62, M, 56, M, 50, 49, 48, M,
                        M, 75, M, 71, M, 67, M, 63, M, 55, M, 51, M, M, M, M, 74, 73, 72, M, 66,
                        65, 64, M, 54, 53, 52, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M,
                        M,
                    ],
                    15_i32 * IVec2::ONE,
                )
                .unwrap(),
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
    fn test_iter_cheats() {
        for (index, cheat_counts_and_ps_saved) in [vec![
            (14_usize, 2_u16),
            (14_usize, 4_u16),
            (2_usize, 6_u16),
            (4_usize, 8_u16),
            (2_usize, 10_u16),
            (3_usize, 12_u16),
            (1_usize, 20_u16),
            (1_usize, 36_u16),
            (1_usize, 38_u16),
            (1_usize, 40_u16),
            (1_usize, 64_u16),
        ]]
        .into_iter()
        .enumerate()
        {
            let mut ps_saved_to_count: HashMap<u16, usize> = HashMap::new();

            for cheat in solution(index).iter_cheats(Solution::Q1_MAX_CHEAT_PS) {
                if let Some(count) = ps_saved_to_count.get_mut(&cheat.saved_ps) {
                    *count += 1_usize;
                } else {
                    ps_saved_to_count.insert(cheat.saved_ps, 1_usize);
                }
            }

            let mut actual_cheat_counts_and_ps_saved: Vec<(usize, u16)> = ps_saved_to_count
                .into_iter()
                .map(|(ps_saved, count)| (count, ps_saved))
                .collect();

            actual_cheat_counts_and_ps_saved.sort_by_key(|&(_, ps_saved)| ps_saved);

            assert_eq!(actual_cheat_counts_and_ps_saved, cheat_counts_and_ps_saved);
        }
    }

    #[test]
    fn test_count_cheats() {
        for (max_cheat_ps, min_saved_ps, cheat_count) in [
            (
                Solution::Q1_MAX_CHEAT_PS,
                0_u16,
                14_usize
                    + 14_usize
                    + 2_usize
                    + 4_usize
                    + 2_usize
                    + 3_usize
                    + 1_usize
                    + 1_usize
                    + 1_usize
                    + 1_usize
                    + 1_usize,
            ),
            (
                Solution::Q2_MAX_CHEAT_PS,
                50_u16,
                32_usize
                    + 31_usize
                    + 29_usize
                    + 39_usize
                    + 25_usize
                    + 23_usize
                    + 20_usize
                    + 19_usize
                    + 12_usize
                    + 14_usize
                    + 12_usize
                    + 22_usize
                    + 4_usize
                    + 3_usize,
            ),
        ] {
            assert_eq!(
                solution(0_usize).count_cheats(max_cheat_ps, min_saved_ps),
                cheat_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
