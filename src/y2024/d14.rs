use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::map,
        error::Error,
        multi::separated_list0,
        sequence::{separated_pair, tuple},
        Err, IResult,
    },
    num::Integer,
    std::cmp::Ordering,
};

/* --- Day 14: Restroom Redoubt ---

One of The Historians needs to use the bathroom; fortunately, you know there's a bathroom near an unvisited location on their list, and so you're all quickly teleported directly to the lobby of Easter Bunny Headquarters.

Unfortunately, EBHQ seems to have "improved" bathroom security again after your last visit. The area outside the bathroom is swarming with robots!

To get The Historian safely to the bathroom, you'll need a way to predict where the robots will be in the future. Fortunately, they all seem to be moving on the tile floor in predictable straight lines.

You make a list (your puzzle input) of all of the robots' current positions (p) and velocities (v), one robot per line. For example:

p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3

Each robot's position is given as p=x,y where x represents the number of tiles the robot is from the left wall and y represents the number of tiles from the top wall (when viewed from above). So, a position of p=0,0 means the robot is all the way in the top-left corner.

Each robot's velocity is given as v=x,y where x and y are given in tiles per second. Positive x means the robot is moving to the right, and positive y means the robot is moving down. So, a velocity of v=1,-2 means that each second, the robot moves 1 tile to the right and 2 tiles up.

The robots outside the actual bathroom are in a space which is 101 tiles wide and 103 tiles tall (when viewed from above). However, in this example, the robots are in a space which is only 11 tiles wide and 7 tiles tall.

The robots are good at navigating over/under each other (due to a combination of springs, extendable legs, and quadcopters), so they can share the same tile and don't interact with each other. Visually, the number of robots on each tile in this example looks like this:

1.12.......
...........
...........
......11.11
1.1........
.........1.
.......1...

These robots have a unique feature for maximum bathroom security: they can teleport. When a robot would run into an edge of the space they're in, they instead teleport to the other side, effectively wrapping around the edges. Here is what robot p=2,4 v=2,-3 does for the first few seconds:

Initial state:
...........
...........
...........
...........
..1........
...........
...........

After 1 second:
...........
....1......
...........
...........
...........
...........
...........

After 2 seconds:
...........
...........
...........
...........
...........
......1....
...........

After 3 seconds:
...........
...........
........1..
...........
...........
...........
...........

After 4 seconds:
...........
...........
...........
...........
...........
...........
..........1

After 5 seconds:
...........
...........
...........
.1.........
...........
...........
...........

The Historian can't wait much longer, so you don't have to simulate the robots for very long. Where will the robots be after 100 seconds?

In the above example, the number of robots on each tile after 100 seconds has elapsed looks like this:

......2..1.
...........
1..........
.11........
.....1.....
...12......
.1....1....

To determine the safest area, count the number of robots in each quadrant after 100 seconds. Robots that are exactly in the middle (horizontally or vertically) don't count as being in any quadrant, so the only relevant robots are:

..... 2..1.
..... .....
1.... .....

..... .....
...12 .....
.1... 1....

In this example, the quadrants contain 1, 3, 4, and 1 robot. Multiplying these together gives a total safety factor of 12.

Predict the motion of the robots in your list within a space which is 101 tiles wide and 103 tiles tall. What will the safety factor be after exactly 100 seconds have elapsed?

--- Part Two ---

During the bathroom break, someone notices that these robots seem awfully similar to ones built and used at the North Pole. If they're the same type of robots, they should have a hard-coded Easter egg: very rarely, most of the robots should arrange themselves into a picture of a Christmas tree.

What is the fewest number of seconds that must elapse for the robots to display the Easter egg? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Robot {
    pos: IVec2,
    vel: IVec2,
}

impl Robot {
    fn parse_component<'i>(name: &'i str) -> impl FnMut(&'i str) -> IResult<&'i str, IVec2> {
        map(
            tuple((tag(name), tag("="), parse_integer, tag(","), parse_integer)),
            |(_, _, x, _, y)| IVec2 { x, y },
        )
    }
}

impl Parse for Robot {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                Self::parse_component("p"),
                tag(" "),
                Self::parse_component("v"),
            ),
            |(pos, vel)| Self { pos, vel },
        )(input)
    }
}

struct Cell(u8);

impl Cell {
    fn increment(&mut self) {
        self.0 = match self.0 {
            b'.' => b'1',
            b'1'..=b'8' => self.0 + 1_u8,
            b'9' | b'#' => b'#',
            _ => unreachable!(),
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self(b'.')
    }
}

// SAFETY: See `Default` implementation and `increment`
unsafe impl IsValidAscii for Cell {}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(Vec<Robot>);

impl Solution {
    const DIMENSIONS: IVec2 = IVec2::new(101_i32, 103_i32);
    const SECS: usize = 100_usize;
    const ALIGNMENT_DENSITY_THRESHOLD: f32 = 0.25_f32;

    fn grid_string(&self, dimensions: IVec2) -> String {
        let mut grid: Grid2D<Cell> = Grid2D::default(dimensions);

        for robot in &self.0 {
            if let Some(cell) = grid.get_mut(robot.pos) {
                cell.increment();
            }
        }

        grid.into()
    }

    fn safety_factor(&self, dimensions: IVec2) -> usize {
        let mut robot_quadrant_counts: [usize; 4_usize] = [0_usize; 4_usize];
        let half_dimensions: IVec2 = dimensions / 2_i32;
        let x_dimensions_is_even: bool = dimensions.x.is_even();
        let y_dimensions_is_even: bool = dimensions.y.is_even();

        for robot in &self.0 {
            let offset: IVec2 = robot.pos - half_dimensions;
            let sanitize_ordering = |pos: i32, dimension_is_even: bool| {
                if dimension_is_even {
                    pos.cmp(&0_i32).then(Ordering::Greater)
                } else {
                    pos.cmp(&0_i32)
                }
            };

            if let Some(quadrant_index) = match (
                sanitize_ordering(offset.x, x_dimensions_is_even),
                sanitize_ordering(offset.y, y_dimensions_is_even),
            ) {
                (Ordering::Equal, _) => None,
                (_, Ordering::Equal) => None,
                (Ordering::Greater, Ordering::Greater) => Some(0_usize),
                (Ordering::Less, Ordering::Greater) => Some(1_usize),
                (Ordering::Less, Ordering::Less) => Some(2_usize),
                (Ordering::Greater, Ordering::Less) => Some(3_usize),
            } {
                robot_quadrant_counts[quadrant_index] += 1_usize;
            }
        }

        robot_quadrant_counts.into_iter().product()
    }

    fn simulate_immutable(&self, dimensions: IVec2, secs: usize) -> Self {
        let mut solution: Self = self.clone();

        solution.simulate(dimensions, secs);

        solution
    }

    fn safety_factor_after_simulation(&self, dimensions: IVec2, secs: usize) -> usize {
        self.simulate_immutable(dimensions, secs)
            .safety_factor(dimensions)
    }

    fn count_robot<F: Fn(IVec2) -> i32>(pos: IVec2, f: F, counts: &mut [usize]) {
        if let Some(index) = usize::try_from(f(pos))
            .ok()
            .filter(|&index| index < counts.len())
        {
            counts[index] += 1_usize;
        }
    }

    fn try_update_alignment_phase(
        phase: usize,
        counts: &[usize],
        density_factor: f32,
        alignment_phase: &mut Option<usize>,
    ) {
        if alignment_phase.is_none()
            && phase < counts.len()
            && counts
                .iter()
                .map(|&count| count as f32 * density_factor)
                .max_by(|density_a, density_b| {
                    density_a.partial_cmp(density_b).unwrap_or(Ordering::Equal)
                })
                .map_or(false, |max_density| {
                    max_density > Self::ALIGNMENT_DENSITY_THRESHOLD
                })
        {
            *alignment_phase = Some(phase);
        }
    }

    fn try_find_alignment_phases(&self, dimensions: IVec2) -> Option<(usize, usize)> {
        dimensions
            .cmpgt(IVec2::ZERO)
            .all()
            .then(|| {
                let mut solution: Self = self.clone();
                let mut counts: Vec<usize> =
                    vec![0_usize; dimensions.x as usize + dimensions.y as usize];
                let mut x_alignment_phase: Option<usize> = None;
                let mut y_alignment_phase: Option<usize> = None;

                let (x_counts, y_counts): (&mut [usize], &mut [usize]) =
                    counts.split_at_mut(dimensions.x as usize);
                let x_density_factor: f32 = 1.0_f32 / y_counts.len() as f32;
                let y_density_factor: f32 = 1.0_f32 / x_counts.len() as f32;

                (0_usize..(x_counts.len().max(y_counts.len()))).try_for_each(|phase| {
                    x_counts.fill(0_usize);
                    y_counts.fill(1_usize);

                    for robot in &solution.0 {
                        Self::count_robot(robot.pos, |pos| pos.x, x_counts);
                        Self::count_robot(robot.pos, |pos| pos.y, y_counts);
                    }

                    Self::try_update_alignment_phase(
                        phase,
                        x_counts,
                        x_density_factor,
                        &mut x_alignment_phase,
                    );
                    Self::try_update_alignment_phase(
                        phase,
                        y_counts,
                        y_density_factor,
                        &mut y_alignment_phase,
                    );

                    (x_alignment_phase.is_none() || y_alignment_phase.is_none()).then(|| {
                        solution.simulate(dimensions, 1_usize);
                    })
                });

                x_alignment_phase.zip(y_alignment_phase)
            })
            .flatten()
    }

    fn try_find_diagram_secs(&self, dimensions: IVec2) -> Option<usize> {
        self.try_find_alignment_phases(dimensions)
            .map(|(x_alignment_phase, y_alignment_phase)| {
                // x_alignment_phase + dimensions.x * k_x = y_alignment_phase + dimensions.y * k_y
                // where k_x and k_y are both positive integers in [0, dimensions.y) and [0,
                // dimensions.x), respectively.
                let alignment_phase_delta: usize = y_alignment_phase - x_alignment_phase;
                let dimensions_x: usize = dimensions.x as usize;
                let dimensions_y: usize = dimensions.y as usize;

                (0_usize..dimensions.x as usize)
                    .find(|&k_y| {
                        (alignment_phase_delta + dimensions_y * k_y) % dimensions_x == 0_usize
                    })
                    .map(|k_y| y_alignment_phase + dimensions_y * k_y)
            })
            .flatten()
    }

    fn simulate(&mut self, dimensions: IVec2, secs: usize) {
        let secs: i32 = secs as i32;

        for robot in &mut self.0 {
            robot.pos = (robot.pos + secs * robot.vel).rem_euclid(dimensions);
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(separated_list0(line_ending, Robot::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Straight forward. I expect question 2 is "now 100000000000 steps".
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let solution: Self = self.simulate_immutable(Self::DIMENSIONS, Self::SECS);

            dbg!(solution.safety_factor(Self::DIMENSIONS));

            println!("grid:\n\n{}\n", solution.grid_string(Self::DIMENSIONS));
        } else {
            dbg!(self.safety_factor_after_simulation(Self::DIMENSIONS, Self::SECS));
        }
    }

    /// Initially figured it out via pencil and paper, took a bit to put together code that could do
    /// the same automatically.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if let Some(diagram_secs) =
            dbg!(self.try_find_diagram_secs(Self::DIMENSIONS)).filter(|_| args.verbose)
        {
            println!(
                "{}",
                self.simulate_immutable(Self::DIMENSIONS, diagram_secs)
                    .grid_string(Self::DIMENSIONS)
            );
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        "p=2,4 v=2,-3",
        "\
        p=0,4 v=3,-3\n\
        p=6,3 v=-1,-3\n\
        p=10,3 v=-1,2\n\
        p=2,0 v=2,-1\n\
        p=0,0 v=1,3\n\
        p=3,0 v=-2,-2\n\
        p=7,6 v=-1,-3\n\
        p=3,0 v=-1,-2\n\
        p=9,3 v=2,3\n\
        p=7,3 v=-1,2\n\
        p=2,4 v=2,-3\n\
        p=9,5 v=-3,-3\n",
    ];
    const DIMENSIONS: IVec2 = IVec2::new(11_i32, 7_i32);

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![Robot {
                    pos: (2_i32, 4_i32).into(),
                    vel: (2_i32, -3_i32).into(),
                }]),
                Solution(vec![
                    Robot {
                        pos: (0_i32, 4_i32).into(),
                        vel: (3_i32, -3_i32).into(),
                    },
                    Robot {
                        pos: (6_i32, 3_i32).into(),
                        vel: (-1_i32, -3_i32).into(),
                    },
                    Robot {
                        pos: (10_i32, 3_i32).into(),
                        vel: (-1_i32, 2_i32).into(),
                    },
                    Robot {
                        pos: (2_i32, 0_i32).into(),
                        vel: (2_i32, -1_i32).into(),
                    },
                    Robot {
                        pos: (0_i32, 0_i32).into(),
                        vel: (1_i32, 3_i32).into(),
                    },
                    Robot {
                        pos: (3_i32, 0_i32).into(),
                        vel: (-2_i32, -2_i32).into(),
                    },
                    Robot {
                        pos: (7_i32, 6_i32).into(),
                        vel: (-1_i32, -3_i32).into(),
                    },
                    Robot {
                        pos: (3_i32, 0_i32).into(),
                        vel: (-1_i32, -2_i32).into(),
                    },
                    Robot {
                        pos: (9_i32, 3_i32).into(),
                        vel: (2_i32, 3_i32).into(),
                    },
                    Robot {
                        pos: (7_i32, 3_i32).into(),
                        vel: (-1_i32, 2_i32).into(),
                    },
                    Robot {
                        pos: (2_i32, 4_i32).into(),
                        vel: (2_i32, -3_i32).into(),
                    },
                    Robot {
                        pos: (9_i32, 5_i32).into(),
                        vel: (-3_i32, -3_i32).into(),
                    },
                ]),
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
    fn test_grid_string() {
        for (index, grid_string) in [
            "\
            ...........\n\
            ...........\n\
            ...........\n\
            ...........\n\
            ..1........\n\
            ...........\n\
            ...........\n",
            "\
            1.12.......\n\
            ...........\n\
            ...........\n\
            ......11.11\n\
            1.1........\n\
            .........1.\n\
            .......1...\n",
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).grid_string(DIMENSIONS), grid_string);
        }
    }

    #[test]
    fn test_simulate() {
        for (index, secs_and_grid_strings) in [
            vec![
                (
                    1_usize,
                    "\
                    ...........\n\
                    ....1......\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n",
                ),
                (
                    1_usize,
                    "\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ......1....\n\
                    ...........\n",
                ),
                (
                    1_usize,
                    "\
                    ...........\n\
                    ...........\n\
                    ........1..\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n",
                ),
                (
                    1_usize,
                    "\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    ..........1\n",
                ),
                (
                    1_usize,
                    "\
                    ...........\n\
                    ...........\n\
                    ...........\n\
                    .1.........\n\
                    ...........\n\
                    ...........\n\
                    ...........\n",
                ),
            ],
            vec![(
                100_usize,
                "\
                ......2..1.\n\
                ...........\n\
                1..........\n\
                .11........\n\
                .....1.....\n\
                ...12......\n\
                .1....1....\n",
            )],
        ]
        .into_iter()
        .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            for (secs, grid_string) in secs_and_grid_strings {
                solution.simulate(DIMENSIONS, secs);

                assert_eq!(solution.grid_string(DIMENSIONS), grid_string);
            }
        }
    }

    #[test]
    fn test_safety_factor() {
        for (index, (secs, safety_factor)) in [(5_usize, 0_usize), (100_usize, 12_usize)]
            .into_iter()
            .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            solution.simulate(DIMENSIONS, secs);

            assert_eq!(solution.safety_factor(DIMENSIONS), safety_factor);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
