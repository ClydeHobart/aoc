use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        multi::separated_list0, sequence::separated_pair, Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        iter::from_fn,
    },
    strum::IntoEnumIterator,
};

/* --- Day 18: RAM Run ---

You and The Historians look a lot more pixelated than you remember. You're inside a computer at the North Pole!

Just as you're about to check out your surroundings, a program runs up to you. "This region of memory isn't safe! The User misunderstood what a pushdown automaton is and their algorithm is pushing whole bytes down on top of us! Run!"

The algorithm is fast - it's going to cause a byte to fall into your memory space once every nanosecond! Fortunately, you're faster, and by quickly scanning the algorithm, you create a list of which bytes will fall (your puzzle input) in the order they'll land in your memory space.

Your memory space is a two-dimensional grid with coordinates that range from 0 to 70 both horizontally and vertically. However, for the sake of example, suppose you're on a smaller grid with coordinates that range from 0 to 6 and the following list of incoming byte positions:

5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0

Each byte position is given as an X,Y coordinate, where X is the distance from the left edge of your memory space and Y is the distance from the top edge of your memory space.

You and The Historians are currently in the top left corner of the memory space (at 0,0) and need to reach the exit in the bottom right corner (at 70,70 in your memory space, but at 6,6 in this example). You'll need to simulate the falling bytes to plan out where it will be safe to run; for now, simulate just the first few bytes falling into your memory space.

As bytes fall into your memory space, they make that coordinate corrupted. Corrupted memory coordinates cannot be entered by you or The Historians, so you'll need to plan your route carefully. You also cannot leave the boundaries of the memory space; your only hope is to reach the exit.

In the above example, if you were to draw the memory space after the first 12 bytes have fallen (using . for safe and # for corrupted), it would look like this:

...#...
..#..#.
....#..
...#..#
..#..#.
.#..#..
#.#....

You can take steps up, down, left, or right. After just 12 bytes have corrupted locations in your memory space, the shortest path from the top left corner to the exit would take 22 steps. Here (marked with O) is one such path:

OO.#OOO
.O#OO#O
.OOO#OO
...#OO#
..#OO#.
.#.O#..
#.#OOOO

Simulate the first kilobyte (1024 bytes) falling onto your memory space. Afterward, what is the minimum number of steps needed to reach the exit?

--- Part Two ---

The Historians aren't as used to moving around in this pixelated universe as you are. You're afraid they're not going to be fast enough to make it to the exit before the path is completely blocked.

To determine how fast everyone needs to go, you need to determine the first byte that will cut off the path to the exit.

In the above example, after the byte at 1,1 falls, there is still a path to the exit:

O..#OOO
O##OO#O
O#OO#OO
OOO#OO#
###OO##
.##O###
#.#OOOO

However, after adding the very next byte (at 6,1), there is no longer a path to the exit:

...#...
.##..##
.#..#..
...#..#
###..##
.##.###
#.#....

So, in this example, the coordinates of the first byte that prevents the exit from being reachable are 6,1.

Simulate more of the bytes that are about to corrupt your memory space. What are the coordinates of the first byte that will prevent the exit from being reachable from your starting position? (Provide the answer as two integers separated by a comma with no other characters.) */

struct PrevData {
    prev: SmallPos,
    cost: i32,
}

struct CorruptedLocationPathFinder<'c> {
    corrupted_locations: &'c BitSlice,
    dimensions: IVec2,
    start: IVec2,
    end: IVec2,
    curr_to_prev_data: HashMap<SmallPos, PrevData>,
}

impl<'c> CorruptedLocationPathFinder<'c> {
    /// Asserts if `pos`` isn't valid.
    fn small_pos_from_pos(pos: IVec2) -> SmallPos {
        SmallPos::try_from_pos(pos).unwrap()
    }

    fn try_new(
        corrupted_locations: &'c BitSlice,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<Self> {
        (SmallPos::are_dimensions_valid(dimensions)
            && grid_2d_contains(start, dimensions)
            && grid_2d_contains(end, dimensions))
        .then(|| Self {
            corrupted_locations,
            dimensions,
            start,
            end,
            curr_to_prev_data: HashMap::new(),
        })
    }

    fn iter_start_to_end_path_rev(&self) -> impl Iterator<Item = IVec2> + '_ {
        let mut pos: IVec2 = self.end;

        from_fn(move || {
            (pos != self.start).then(|| {
                let next: IVec2 = pos;

                pos = self.curr_to_prev_data[&Self::small_pos_from_pos(pos)]
                    .prev
                    .get();

                next
            })
        })
        .chain([self.start])
    }

    fn try_start_to_end_path(&self) -> Option<Vec<IVec2>> {
        self.curr_to_prev_data
            .contains_key(&Self::small_pos_from_pos(self.end))
            .then(|| {
                let mut path: VecDeque<IVec2> = VecDeque::new();

                for pos in self.iter_start_to_end_path_rev() {
                    path.push_front(pos);
                }

                path.into()
            })
    }
}

impl<'c> WeightedGraphSearch for CorruptedLocationPathFinder<'c> {
    type Vertex = IVec2;

    type Cost = i32;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.end
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        // We don't want to allocate here since we'll be calling it a bunch trying to find the first
        // exit-preventing corrupted location.
        Vec::new()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.curr_to_prev_data
            .get(&Self::small_pos_from_pos(*vertex))
            .map_or(i32::MAX, |prev_data| prev_data.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        manhattan_distance_2d(*vertex, self.end)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            grid_2d_try_index_from_pos_and_dimensions(neighbor, self.dimensions)
                .map_or(false, |index| !self.corrupted_locations[index])
                .then_some(OpenSetElement(neighbor, 1_i32))
        }));
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.curr_to_prev_data.insert(
            Self::small_pos_from_pos(*to),
            PrevData {
                prev: Self::small_pos_from_pos(*from),
                cost,
            },
        );
    }

    fn reset(&mut self) {
        self.curr_to_prev_data.clear();

        let small_start: SmallPos = Self::small_pos_from_pos(self.start);

        self.curr_to_prev_data.insert(
            small_start,
            PrevData {
                prev: small_start,
                cost: 0_i32,
            },
        );
    }
}

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Default)]
    enum Cell {
        #[default]
        Safe = SAFE = b'.',
        Corrupted = CORRUPTED = b'#',
        Path = PATH = b'O',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<IVec2>);

impl Solution {
    const CORRUPTED_LOCATIONS_COUNT: usize = 1024_usize;
    const DIMENSIONS: IVec2 = IVec2::new(71_i32, 71_i32);
    const START: IVec2 = IVec2::ZERO;
    const END: IVec2 = IVec2::new(Self::DIMENSIONS.x - 1_i32, Self::DIMENSIONS.y - 1_i32);

    fn steps_in_path(path: &[IVec2]) -> usize {
        path.len().checked_sub(1_usize).unwrap_or_default()
    }

    fn first_corrupted_locations(
        &self,
        corrupted_locations_count: usize,
        dimensions: IVec2,
    ) -> BitVec {
        assert!(dimensions.cmpge(IVec2::ZERO).all());

        let mut corrupted_locations: BitVec =
            bitvec![0; dimensions.x as usize * dimensions.y as usize];

        for corrupted_location in &self.0[..corrupted_locations_count.min(self.0.len())] {
            if let Some(index) =
                grid_2d_try_index_from_pos_and_dimensions(*corrupted_location, dimensions)
            {
                corrupted_locations.set(index, true);
            }
        }

        corrupted_locations
    }

    fn try_path(
        &self,
        corrupted_locations: &BitSlice,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<Vec<IVec2>> {
        CorruptedLocationPathFinder::try_new(&corrupted_locations, dimensions, start, end)
            .map(|mut corrupted_location_path_finder| {
                corrupted_location_path_finder.run_a_star();

                corrupted_location_path_finder.try_start_to_end_path()
            })
            .flatten()
    }

    fn grid_string_from_path(
        &self,
        corrupted_locations: &BitSlice,
        dimensions: IVec2,
        path: &[IVec2],
    ) -> String {
        let mut grid: Grid2D<Cell> = Grid2D::default(dimensions);

        for index in corrupted_locations.iter_ones() {
            *grid
                .get_mut(grid_2d_pos_from_index_and_dimensions(index, dimensions))
                .unwrap() = Cell::Corrupted;
        }

        for pos in path.iter().copied() {
            *grid.get_mut(pos).unwrap() = Cell::Path;
        }

        grid.into()
    }

    fn try_min_steps_to_exit(
        &self,
        corrupted_locations_count: usize,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<usize> {
        let corrupted_locations: BitVec =
            self.first_corrupted_locations(corrupted_locations_count, dimensions);

        self.try_path(&corrupted_locations, dimensions, start, end)
            .map(|path| Self::steps_in_path(&path))
    }

    fn try_min_steps_to_exit_and_grid_string(
        &self,
        corrupted_locations_count: usize,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<(usize, String)> {
        let corrupted_locations: BitVec =
            self.first_corrupted_locations(corrupted_locations_count, dimensions);

        self.try_path(&corrupted_locations, dimensions, start, end)
            .map(|path| {
                (
                    Self::steps_in_path(&path),
                    self.grid_string_from_path(&corrupted_locations, dimensions, &path),
                )
            })
    }

    fn try_first_corrupted_location_preventing_exit_index(
        &self,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<usize> {
        let mut corrupted_locations: BitVec =
            bitvec![0; dimensions.x as usize * dimensions.y as usize];

        self.0
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, corrupted_location)| {
                grid_2d_try_index_from_pos_and_dimensions(corrupted_location, dimensions)
                    .map_or(false, |grid_index| {
                        corrupted_locations.set(grid_index, true);

                        CorruptedLocationPathFinder::try_new(
                            &corrupted_locations,
                            dimensions,
                            start,
                            end,
                        )
                        .map_or(
                            false,
                            |mut corrupted_location_path_finder| {
                                corrupted_location_path_finder.run_a_star().is_none()
                            },
                        )
                    })
                    .then_some(index)
            })
            .next()
    }

    fn string_from_corrupted_location_index(&self, index: usize) -> String {
        let pos: IVec2 = self.0[index];

        format!("{},{}", pos.x, pos.y)
    }

    fn try_first_corrupted_location_preventing_exit_string(
        &self,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<String> {
        self.try_first_corrupted_location_preventing_exit_index(dimensions, start, end)
            .map(|index| self.string_from_corrupted_location_index(index))
    }

    fn try_first_corrupted_location_preventing_exit_string_and_previous_path_grid_string(
        &self,
        dimensions: IVec2,
        start: IVec2,
        end: IVec2,
    ) -> Option<(String, String)> {
        self.try_first_corrupted_location_preventing_exit_index(dimensions, start, end)
            .map(|index| {
                (
                    self.string_from_corrupted_location_index(index),
                    self.try_min_steps_to_exit_and_grid_string(index, dimensions, start, end)
                        .unwrap()
                        .1,
                )
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_list0(
                line_ending,
                map(
                    separated_pair(parse_integer, tag(","), parse_integer),
                    |(x, y)| IVec2 { x, y },
                ),
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Super easy. I suspect part two will take into account that not all locations were corrupted
    /// initially, so you need to chart a path that takes advantage of that.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.try_min_steps_to_exit(
                Self::CORRUPTED_LOCATIONS_COUNT,
                Self::DIMENSIONS,
                Self::START,
                Self::END
            ));
        } else if let Some((min_steps_to_exit, grid_string)) = self
            .try_min_steps_to_exit_and_grid_string(
                Self::CORRUPTED_LOCATIONS_COUNT,
                Self::DIMENSIONS,
                Self::START,
                Self::END,
            )
        {
            dbg!(min_steps_to_exit);
            println!("{grid_string}");
        } else {
            eprintln!("Failed to find path to exit.");
        }
    }

    /// Got a bit worried there that my implementation wasn't fast enough and that I'd need to give
    /// some sort of "path hint" using the previous path.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.try_first_corrupted_location_preventing_exit_string(
                Self::DIMENSIONS,
                Self::START,
                Self::END
            ));
        } else if let Some((
            first_corrupted_location_preventing_exit_string,
            previous_path_grid_string,
        )) = self
            .try_first_corrupted_location_preventing_exit_string_and_previous_path_grid_string(
                Self::DIMENSIONS,
                Self::START,
                Self::END,
            )
        {
            dbg!(first_corrupted_location_preventing_exit_string);
            println!("{previous_path_grid_string}");
        } else {
            eprintln!("Failed to find a corrupted location preventing exit.");
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        5,4\n\
        4,2\n\
        4,5\n\
        3,0\n\
        2,1\n\
        6,3\n\
        2,4\n\
        1,5\n\
        0,6\n\
        3,3\n\
        2,6\n\
        5,1\n\
        1,2\n\
        5,5\n\
        2,5\n\
        6,5\n\
        1,4\n\
        0,4\n\
        6,4\n\
        1,1\n\
        6,1\n\
        1,0\n\
        0,5\n\
        1,6\n\
        2,0\n"];
    const CORRUPTED_LOCATIONS_COUNT: usize = 12_usize;
    const DIMENSIONS: IVec2 = IVec2::new(7_i32, 7_i32);
    const START: IVec2 = IVec2::ZERO;
    const END: IVec2 = IVec2::new(DIMENSIONS.x - 1_i32, DIMENSIONS.y - 1_i32);

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                (5_i32, 4_i32).into(),
                (4_i32, 2_i32).into(),
                (4_i32, 5_i32).into(),
                (3_i32, 0_i32).into(),
                (2_i32, 1_i32).into(),
                (6_i32, 3_i32).into(),
                (2_i32, 4_i32).into(),
                (1_i32, 5_i32).into(),
                (0_i32, 6_i32).into(),
                (3_i32, 3_i32).into(),
                (2_i32, 6_i32).into(),
                (5_i32, 1_i32).into(),
                (1_i32, 2_i32).into(),
                (5_i32, 5_i32).into(),
                (2_i32, 5_i32).into(),
                (6_i32, 5_i32).into(),
                (1_i32, 4_i32).into(),
                (0_i32, 4_i32).into(),
                (6_i32, 4_i32).into(),
                (1_i32, 1_i32).into(),
                (6_i32, 1_i32).into(),
                (1_i32, 0_i32).into(),
                (0_i32, 5_i32).into(),
                (1_i32, 6_i32).into(),
                (2_i32, 0_i32).into(),
            ])]
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
    fn test_try_min_steps_to_exit() {
        for (index, min_steps_to_exit) in [Some(22_usize)].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_min_steps_to_exit(
                    CORRUPTED_LOCATIONS_COUNT,
                    DIMENSIONS,
                    START,
                    END
                ),
                min_steps_to_exit
            );
        }
    }

    #[test]
    fn try_first_corrupted_location_preventing_exit_string() {
        for (index, first_corrupted_location_preventing_exit_string) in
            [Some(String::from("6,1"))].into_iter().enumerate()
        {
            assert_eq!(
                solution(index)
                    .try_first_corrupted_location_preventing_exit_string(DIMENSIONS, START, END),
                first_corrupted_location_preventing_exit_string
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
