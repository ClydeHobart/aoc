use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::collections::HashSet,
};

/* --- Day 6: Guard Gallivant ---

The Historians use their fancy device again, this time to whisk you all away to the North Pole prototype suit manufacturing lab... in the year 1518! It turns out that having direct access to history is very convenient for a group of historians.

You still have to be careful of time paradoxes, and so it will be important to avoid anyone from 1518 while The Historians search for the Chief. Unfortunately, a single guard is patrolling this part of the lab.

Maybe you can work out where the guard will go ahead of time so that The Historians can search safely?

You start by making a map (your puzzle input) of the situation. For example:

....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...

The map shows the current position of the guard with ^ (to indicate the guard is currently facing up from the perspective of the map). Any obstructions - crates, desks, alchemical reactors, etc. - are shown as #.

Lab guards in 1518 follow a very strict patrol protocol which involves repeatedly following these steps:

    If there is something directly in front of you, turn right 90 degrees.
    Otherwise, take a step forward.

Following the above protocol, the guard moves up several times until she reaches an obstacle (in this case, a pile of failed suit prototypes):

....#.....
....^....#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#...

Because there is now an obstacle in front of the guard, she turns right before continuing straight in her new facing direction:

....#.....
........>#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#...

Reaching another obstacle (a spool of several very long polymers), she turns right again and continues downward:

....#.....
.........#
..........
..#.......
.......#..
..........
.#......v.
........#.
#.........
......#...

This process continues for a while, but the guard eventually leaves the mapped area (after walking past a tank of universal solvent):

....#.....
.........#
..........
..#.......
.......#..
..........
.#........
........#.
#.........
......#v..

By predicting the guard's route, you can determine which specific positions in the lab will be in the patrol path. Including the guard's starting position, the positions visited by the guard before leaving the area are marked with an X:

....#.....
....XXXXX#
....X...X.
..#.X...X.
..XXXXX#X.
..X.X.X.X.
.#XXXXXXX.
.XXXXXXX#.
#XXXXXXX..
......#X..

In this example, the guard will visit 41 distinct positions on your map.

Predict the path of the guard. How many distinct positions will the guard visit before leaving the mapped area?

--- Part Two ---

While The Historians begin working around the guard's patrol route, you borrow their fancy device and step outside the lab. From the safety of a supply closet, you time travel through the last few months and record the nightly status of the lab's guard post on the walls of the closet.

Returning after what seems like only a few seconds to The Historians, they explain that the guard's patrol area is simply too large for them to safely search the lab without getting caught.

Fortunately, they are pretty sure that adding a single new obstruction won't cause a time paradox. They'd like to place the new obstruction in such a way that the guard will get stuck in a loop, making the rest of the lab safe to search.

To have the lowest chance of creating a time paradox, The Historians would like to know all of the possible positions for such an obstruction. The new obstruction can't be placed at the guard's starting position - the guard is there right now and would notice.

In the above example, there are only 6 different positions where a new obstruction would cause the guard to get stuck in a loop. The diagrams of these six situations use O to mark the new obstruction, | to show a position where the guard moves up/down, - to show a position where the guard moves left/right, and + to show a position where the guard moves both up/down and left/right.

Option one, put a printing press next to the guard's starting position:

....#.....
....+---+#
....|...|.
..#.|...|.
....|..#|.
....|...|.
.#.O^---+.
........#.
#.........
......#...

Option two, put a stack of failed suit prototypes in the bottom right quadrant of the mapped area:

....#.....
....+---+#
....|...|.
..#.|...|.
..+-+-+#|.
..|.|.|.|.
.#+-^-+-+.
......O.#.
#.........
......#...

Option three, put a crate of chimney-squeeze prototype fabric next to the standing desk in the bottom right quadrant:

....#.....
....+---+#
....|...|.
..#.|...|.
..+-+-+#|.
..|.|.|.|.
.#+-^-+-+.
.+----+O#.
#+----+...
......#...

Option four, put an alchemical retroencabulator near the bottom left corner:

....#.....
....+---+#
....|...|.
..#.|...|.
..+-+-+#|.
..|.|.|.|.
.#+-^-+-+.
..|...|.#.
#O+---+...
......#...

Option five, put the alchemical retroencabulator a bit to the right instead:

....#.....
....+---+#
....|...|.
..#.|...|.
..+-+-+#|.
..|.|.|.|.
.#+-^-+-+.
....|.|.#.
#..O+-+...
......#...

Option six, put a tank of sovereign glue right next to the tank of universal solvent:

....#.....
....+---+#
....|...|.
..#.|...|.
..+-+-+#|.
..|.|.|.|.
.#+-^-+-+.
.+----++#.
#+----++..
......#O..

It doesn't really matter what you choose to use as an obstacle so long as you and The Historians can put it into position without the guard noticing. The important thing is having enough options that you can find one that minimizes time paradoxes, and in this example, there are 6 different positions you could choose.

You need to get the guard stuck in a loop by adding a single new obstruction. How many different positions could you choose for this obstruction? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Empty = EMPTY = b'.',
        Obstacle = OBSTACLE = b'#',
        NorthFacingGuard = NORTH_FACING_GUARD =  b'^',
        EastFacingGuard = EAST_FACING_GUARD = b'>',
        SouthFacingGuard = SOUTH_FACING_GUARD =  b'v',
        WestFacingGuard = WEST_FACING_GUARD = b'<',
        PrevGuardPos = PREV_GUARD_POS = b'X',
    }
}

impl Cell {
    fn from_guard_dir(dir: Direction) -> Self {
        match dir {
            Direction::North => Self::NorthFacingGuard,
            Direction::East => Self::EastFacingGuard,
            Direction::South => Self::SouthFacingGuard,
            Direction::West => Self::WestFacingGuard,
        }
    }

    fn try_guard_dir(self) -> Option<Direction> {
        match self {
            Self::NorthFacingGuard => Some(Direction::North),
            Self::EastFacingGuard => Some(Direction::East),
            Self::SouthFacingGuard => Some(Direction::South),
            Self::WestFacingGuard => Some(Direction::West),
            _ => None,
        }
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct GuardPosAndDir {
    x: u8,
    y: u8,
    dir: Direction,
}

impl GuardPosAndDir {
    const MAX_POS: IVec2 = IVec2::new(u8::MAX as i32, u8::MAX as i32);
    const MAX_DIMENSIONS: IVec2 = IVec2::new(Self::MAX_POS.x + 1_i32, Self::MAX_POS.y + 1_i32);

    /// SAFETY: This will panic if either component can't be converted to a `u8`
    unsafe fn from_pos_and_dir_unsafe(pos: IVec2, dir: Direction) -> Self {
        Self {
            x: pos.x as u8,
            y: pos.y as u8,
            dir,
        }
    }

    fn pos(self) -> IVec2 {
        IVec2::new(self.x as i32, self.y as i32)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    grid: Grid2D<Cell>,
    guard_pos: IVec2,
    guard_dir: Direction,
    guard_history: HashSet<GuardPosAndDir>,
}

impl Solution {
    fn count_distinct_visited_poses(grid_string: &str) -> usize {
        grid_string
            .as_bytes()
            .iter()
            .copied()
            .filter(|b| *b == Cell::PREV_GUARD_POS)
            .count()
    }

    fn grid_string(&self, paint_history: bool) -> String {
        let mut grid: Grid2D<Cell> = self.grid.clone();

        if paint_history {
            for prev_guard_pos_and_dir in &self.guard_history {
                *grid.get_mut(prev_guard_pos_and_dir.pos()).unwrap() = Cell::PrevGuardPos;
            }
        }

        if grid.contains(self.guard_pos) {
            *grid.get_mut(self.guard_pos).unwrap() = Cell::from_guard_dir(self.guard_dir);
        }

        grid.into()
    }

    fn is_guard_in_grid(&self) -> bool {
        self.grid.contains(self.guard_pos)
    }

    fn grid_string_after_leaving(&self) -> String {
        let mut solution: Solution = self.clone();

        while solution.is_guard_in_grid() {
            solution.step();
        }

        solution.grid_string(true)
    }

    fn count_distinct_visited_poses_before_leaving(&self) -> usize {
        Self::count_distinct_visited_poses(&self.grid_string_after_leaving())
    }

    /// SAFETY: It is the caller's responsibility to ensure the guard is currently in the grid.
    unsafe fn guard_pos_and_dir_unsafe(&self) -> GuardPosAndDir {
        GuardPosAndDir::from_pos_and_dir_unsafe(self.guard_pos, self.guard_dir)
    }

    fn count_new_obstacle_positions_that_result_in_cycles(&self) -> usize {
        let mut solution: Solution = self.clone();

        self.grid
            .cells()
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, cell)| {
                *cell == Cell::Empty && {
                    let pos: IVec2 = self.grid.pos_from_index(*index);

                    pos != self.guard_pos && {
                        solution.grid.cells_mut().copy_from_slice(self.grid.cells());
                        *solution.grid.get_mut(pos).unwrap() = Cell::Obstacle;
                        solution.guard_pos = self.guard_pos;
                        solution.guard_dir = self.guard_dir;
                        solution.guard_history.clear();
                        solution.guard_history.extend(self.guard_history.iter());

                        solution.results_in_cycle()
                    }
                }
            })
            .count()
    }

    /// Returns whether a cycle was detected.
    fn step(&mut self) -> bool {
        if self.is_guard_in_grid() {
            let next_guard_pos: IVec2 = self.guard_pos + self.guard_dir.vec();

            if let Some(next_cell) = self.grid.get(next_guard_pos) {
                if *next_cell == Cell::Empty {
                    self.guard_pos = next_guard_pos;
                } else {
                    self.guard_dir = self.guard_dir.next();
                }

                // SAFETY: The guard is in the grid.
                !self
                    .guard_history
                    .insert(unsafe { self.guard_pos_and_dir_unsafe() })
            } else {
                // Bye bye
                self.guard_pos = next_guard_pos;

                false
            }
        } else {
            false
        }
    }

    fn results_in_cycle(&mut self) -> bool {
        let mut detected_cycle: bool = false;

        while self.is_guard_in_grid() && {
            detected_cycle = self.step();

            !detected_cycle
        } {}

        detected_cycle
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |mut grid: Grid2D<Cell>| {
            grid.cells()
                .iter()
                .enumerate()
                .try_fold(None, |mut guard_pos_and_dir, (index, cell)| {
                    (*cell != Cell::PrevGuardPos
                        && (cell.try_guard_dir().map_or(true, |guard_dir| {
                            guard_pos_and_dir
                                .replace((grid.pos_from_index(index), guard_dir))
                                .is_none()
                        })))
                    .then_some(guard_pos_and_dir)
                })
                .flatten()
                .filter(|_| {
                    grid.dimensions()
                        .cmple(GuardPosAndDir::MAX_DIMENSIONS)
                        .all()
                })
                .map(|(guard_pos, guard_dir)| {
                    *grid.get_mut(guard_pos).unwrap() = Cell::Empty;

                    Self {
                        grid,
                        guard_pos,
                        guard_dir,
                        // SAFETY: The guard is in the grid.
                        guard_history: [unsafe {
                            GuardPosAndDir::from_pos_and_dir_unsafe(guard_pos, guard_dir)
                        }]
                        .into(),
                    }
                })
        })(input)
    }
}

impl RunQuestions for Solution {
    /// Seems a bit simple.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let grid_string: String = self.grid_string_after_leaving();
            let distinct_visited_poses_before_leaving_count: usize =
                Self::count_distinct_visited_poses(&grid_string);

            dbg!(distinct_visited_poses_before_leaving_count);
            println!("{grid_string}");
        } else {
            dbg!(self.count_distinct_visited_poses_before_leaving());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_new_obstacle_positions_that_result_in_cycles());
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
        ....#.....\n\
        .........#\n\
        ..........\n\
        ..#.......\n\
        .......#..\n\
        ..........\n\
        .#..^.....\n\
        ........#.\n\
        #.........\n\
        ......#...\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{Empty as E, Obstacle as O};

            vec![Solution {
                grid: Grid2D::try_from_cells_and_dimensions(
                    vec![
                        E, E, E, E, O, E, E, E, E, E, E, E, E, E, E, E, E, E, E, O, E, E, E, E, E,
                        E, E, E, E, E, E, E, O, E, E, E, E, E, E, E, E, E, E, E, E, E, E, O, E, E,
                        E, E, E, E, E, E, E, E, E, E, E, O, E, E, E, E, E, E, E, E, E, E, E, E, E,
                        E, E, E, O, E, O, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, O, E, E, E,
                    ],
                    10_i32 * IVec2::ONE,
                )
                .unwrap(),
                guard_pos: IVec2::new(4_i32, 6_i32),
                guard_dir: Direction::North,
                // SAFETY: The position is within the acceptable bounds
                guard_history: [unsafe {
                    GuardPosAndDir::from_pos_and_dir_unsafe(
                        IVec2::new(4_i32, 6_i32),
                        Direction::North,
                    )
                }]
                .into(),
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
    fn test_step() {
        for (index, steps_and_grid_strings) in [vec![
            (
                0_usize,
                "\
                ....#.....\n\
                .........#\n\
                ..........\n\
                ..#.......\n\
                .......#..\n\
                ..........\n\
                .#..^.....\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
            (
                5_usize,
                "\
                ....#.....\n\
                ....^....#\n\
                ....X.....\n\
                ..#.X.....\n\
                ....X..#..\n\
                ....X.....\n\
                .#..X.....\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
            (
                1_usize,
                "\
                ....#.....\n\
                ....>....#\n\
                ....X.....\n\
                ..#.X.....\n\
                ....X..#..\n\
                ....X.....\n\
                .#..X.....\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
            (
                4_usize,
                "\
                ....#.....\n\
                ....XXXX>#\n\
                ....X.....\n\
                ..#.X.....\n\
                ....X..#..\n\
                ....X.....\n\
                .#..X.....\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
            (
                1_usize,
                "\
                ....#.....\n\
                ....XXXXv#\n\
                ....X.....\n\
                ..#.X.....\n\
                ....X..#..\n\
                ....X.....\n\
                .#..X.....\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
            (
                5_usize,
                "\
                ....#.....\n\
                ....XXXXX#\n\
                ....X...X.\n\
                ..#.X...X.\n\
                ....X..#X.\n\
                ....X...X.\n\
                .#..X...v.\n\
                ........#.\n\
                #.........\n\
                ......#...\n",
            ),
        ]]
        .into_iter()
        .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            for (steps, grid_string) in steps_and_grid_strings {
                for _ in 0_usize..steps {
                    solution.step();
                }

                assert_eq!(solution.grid_string(true), grid_string);
            }
        }
    }

    #[test]
    fn test_count_distinct_visited_poses_before_leaving() {
        for (index, distinct_visited_poses_before_leaving_count) in
            [41_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).count_distinct_visited_poses_before_leaving(),
                distinct_visited_poses_before_leaving_count
            );
        }
    }

    #[test]
    fn test_count_new_obstacle_positions_that_result_in_cycles() {
        for (index, new_obstacle_positions_that_result_in_cycles_count) in
            [6_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).count_new_obstacle_positions_that_result_in_cycles(),
                new_obstacle_positions_that_result_in_cycles_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
