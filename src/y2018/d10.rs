use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated, tuple},
        Err, IResult,
    },
    std::mem::swap,
    std::ops::RangeInclusive,
};

/* --- Day 10: The Stars Align ---

It's no use; your navigation system simply isn't capable of providing walking directions in the arctic circle, and certainly not in 1018.

The Elves suggest an alternative. In times like these, North Pole rescue operations will arrange points of light in the sky to guide missing Elves back to base. Unfortunately, the message is easy to miss: the points move slowly enough that it takes hours to align them, but have so much momentum that they only stay aligned for a second. If you blink at the wrong time, it might be hours before another message appears.

You can see these points of light floating in the distance, and record their position in the sky and their velocity, the relative change in position per second (your puzzle input). The coordinates are all given from your perspective; given enough time, those positions and velocities will move the points into a cohesive message!

Rather than wait, you decide to fast-forward the process and calculate what the points will eventually spell.

For example, suppose you note the following points:

position=< 9,  1> velocity=< 0,  2>
position=< 7,  0> velocity=<-1,  0>
position=< 3, -2> velocity=<-1,  1>
position=< 6, 10> velocity=<-2, -1>
position=< 2, -4> velocity=< 2,  2>
position=<-6, 10> velocity=< 2, -2>
position=< 1,  8> velocity=< 1, -1>
position=< 1,  7> velocity=< 1,  0>
position=<-3, 11> velocity=< 1, -2>
position=< 7,  6> velocity=<-1, -1>
position=<-2,  3> velocity=< 1,  0>
position=<-4,  3> velocity=< 2,  0>
position=<10, -3> velocity=<-1,  1>
position=< 5, 11> velocity=< 1, -2>
position=< 4,  7> velocity=< 0, -1>
position=< 8, -2> velocity=< 0,  1>
position=<15,  0> velocity=<-2,  0>
position=< 1,  6> velocity=< 1,  0>
position=< 8,  9> velocity=< 0, -1>
position=< 3,  3> velocity=<-1,  1>
position=< 0,  5> velocity=< 0, -1>
position=<-2,  2> velocity=< 2,  0>
position=< 5, -2> velocity=< 1,  2>
position=< 1,  4> velocity=< 2,  1>
position=<-2,  7> velocity=< 2, -2>
position=< 3,  6> velocity=<-1, -1>
position=< 5,  0> velocity=< 1,  0>
position=<-6,  0> velocity=< 2,  0>
position=< 5,  9> velocity=< 1, -2>
position=<14,  7> velocity=<-2,  0>
position=<-3,  6> velocity=< 2, -1>

Each line represents one point. Positions are given as <X, Y> pairs: X represents how far left (negative) or right (positive) the point appears, while Y represents how far up (negative) or down (positive) the point appears.

At 0 seconds, each point has the position given. Each second, each point's velocity is added to its position. So, a point with velocity <1, -2> is moving to the right, but is moving upward twice as quickly. If this point's initial position were <3, 9>, after 3 seconds, its position would become <6, 3>.

Over time, the points listed above would move like this:

Initially:
........#.............
................#.....
.........#.#..#.......
......................
#..........#.#.......#
...............#......
....#.................
..#.#....#............
.......#..............
......#...............
...#...#.#...#........
....#..#..#.........#.
.......#..............
...........#..#.......
#...........#.........
...#.......#..........

After 1 second:
......................
......................
..........#....#......
........#.....#.......
..#.........#......#..
......................
......#...............
....##.........#......
......#.#.............
.....##.##..#.........
........#.#...........
........#...#.....#...
..#...........#.......
....#.....#.#.........
......................
......................

After 2 seconds:
......................
......................
......................
..............#.......
....#..#...####..#....
......................
........#....#........
......#.#.............
.......#...#..........
.......#..#..#.#......
....#....#.#..........
.....#...#...##.#.....
........#.............
......................
......................
......................

After 3 seconds:
......................
......................
......................
......................
......#...#..###......
......#...#...#.......
......#...#...#.......
......#####...#.......
......#...#...#.......
......#...#...#.......
......#...#...#.......
......#...#..###......
......................
......................
......................
......................

After 4 seconds:
......................
......................
......................
............#.........
........##...#.#......
......#.....#..#......
.....#..##.##.#.......
.......##.#....#......
...........#....#.....
..............#.......
....#......#...#......
.....#.....##.........
...............#......
...............#......
......................
......................

After 3 seconds, the message appeared briefly: HI. Of course, your message will be much longer and will take many more seconds to appear.

What message will eventually appear in the sky?

--- Part Two ---

Good thing you didn't have to wait, because that would have taken a long time - much longer than the 3 seconds in the example above.

Impressed by your sub-hour communication capabilities, the Elves are curious: exactly how many seconds would they have needed to wait for that message to appear? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Point {
    pos: IVec2,
    vel: IVec2,
}

impl Point {
    fn parse_ivec2<'i>(field_name: &'i str) -> impl FnMut(&'i str) -> IResult<&'i str, IVec2> {
        map(
            tuple((
                tag(field_name),
                tag("=<"),
                space0,
                parse_integer,
                tag(","),
                space0,
                parse_integer,
                tag(">"),
            )),
            |(_, _, _, x, _, _, y, _)| IVec2 { x, y },
        )
    }
}

impl Parse for Point {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                Self::parse_ivec2("position"),
                tag(" "),
                Self::parse_ivec2("velocity"),
            ),
            |(pos, vel)| Self { pos, vel },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Point>);

impl Solution {
    fn compute_bounding_box(poses: &[IVec2]) -> RangeInclusive<IVec2> {
        let mut min: IVec2 = IVec2::MAX;
        let mut max: IVec2 = IVec2::MIN;

        for pos in poses {
            min = min.min(*pos);
            max = max.max(*pos);
        }

        min..=max
    }

    fn compute_bounding_box_area(bounding_box: &RangeInclusive<IVec2>) -> usize {
        let dimensions: IVec2 = *bounding_box.end() - *bounding_box.start() + IVec2::ONE;

        dimensions.x as usize * dimensions.y as usize
    }

    fn message_string(poses: &[IVec2], bounding_box: &RangeInclusive<IVec2>) -> String {
        let min: IVec2 = *bounding_box.start();
        let max: IVec2 = *bounding_box.end();
        let dimensions: IVec2 = max - min + IVec2::ONE;

        let mut grid: Grid2D<Pixel> = Grid2D::default(dimensions);

        for pos in poses {
            *grid.get_mut(*pos - min).unwrap() = Pixel::Light;
        }

        grid.into()
    }

    fn find_message_and_seconds(&self) -> (String, usize) {
        let mut curr_poses: Vec<IVec2> = self.0.iter().map(|point| point.pos).collect();
        let mut next_poses: Vec<IVec2> = curr_poses.clone();
        let mut curr_bounding_box: RangeInclusive<IVec2> = Self::compute_bounding_box(&curr_poses);
        let mut curr_bounding_box_area: usize = Self::compute_bounding_box_area(&curr_bounding_box);
        let mut next_bounding_box: RangeInclusive<IVec2>;
        let mut next_bounding_box_area: usize;
        let mut seconds: usize = 0_usize;

        while {
            for (point, (curr_pos, next_pos)) in self
                .0
                .iter()
                .zip(curr_poses.iter().zip(next_poses.iter_mut()))
            {
                *next_pos = *curr_pos + point.vel;
            }

            next_bounding_box = Self::compute_bounding_box(&next_poses);
            next_bounding_box_area = Self::compute_bounding_box_area(&next_bounding_box);

            next_bounding_box_area < curr_bounding_box_area
        } {
            swap(&mut curr_poses, &mut next_poses);
            curr_bounding_box = next_bounding_box;
            curr_bounding_box_area = next_bounding_box_area;
            seconds += 1_usize;
        }

        (
            Self::message_string(&curr_poses, &curr_bounding_box),
            seconds,
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Point::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I'm thankful for the metric of "it just gets smaller until it doesn't".
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        println!("message:\n\n{}", self.find_message_and_seconds().0);
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_message_and_seconds().1);
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
        position=< 9,  1> velocity=< 0,  2>\n\
        position=< 7,  0> velocity=<-1,  0>\n\
        position=< 3, -2> velocity=<-1,  1>\n\
        position=< 6, 10> velocity=<-2, -1>\n\
        position=< 2, -4> velocity=< 2,  2>\n\
        position=<-6, 10> velocity=< 2, -2>\n\
        position=< 1,  8> velocity=< 1, -1>\n\
        position=< 1,  7> velocity=< 1,  0>\n\
        position=<-3, 11> velocity=< 1, -2>\n\
        position=< 7,  6> velocity=<-1, -1>\n\
        position=<-2,  3> velocity=< 1,  0>\n\
        position=<-4,  3> velocity=< 2,  0>\n\
        position=<10, -3> velocity=<-1,  1>\n\
        position=< 5, 11> velocity=< 1, -2>\n\
        position=< 4,  7> velocity=< 0, -1>\n\
        position=< 8, -2> velocity=< 0,  1>\n\
        position=<15,  0> velocity=<-2,  0>\n\
        position=< 1,  6> velocity=< 1,  0>\n\
        position=< 8,  9> velocity=< 0, -1>\n\
        position=< 3,  3> velocity=<-1,  1>\n\
        position=< 0,  5> velocity=< 0, -1>\n\
        position=<-2,  2> velocity=< 2,  0>\n\
        position=< 5, -2> velocity=< 1,  2>\n\
        position=< 1,  4> velocity=< 2,  1>\n\
        position=<-2,  7> velocity=< 2, -2>\n\
        position=< 3,  6> velocity=<-1, -1>\n\
        position=< 5,  0> velocity=< 1,  0>\n\
        position=<-6,  0> velocity=< 2,  0>\n\
        position=< 5,  9> velocity=< 1, -2>\n\
        position=<14,  7> velocity=<-2,  0>\n\
        position=<-3,  6> velocity=< 2, -1>\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                Point {
                    pos: IVec2::new(9_i32, 1_i32),
                    vel: IVec2::new(0_i32, 2_i32),
                },
                Point {
                    pos: IVec2::new(7_i32, 0_i32),
                    vel: IVec2::new(-1_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(3_i32, -2_i32),
                    vel: IVec2::new(-1_i32, 1_i32),
                },
                Point {
                    pos: IVec2::new(6_i32, 10_i32),
                    vel: IVec2::new(-2_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(2_i32, -4_i32),
                    vel: IVec2::new(2_i32, 2_i32),
                },
                Point {
                    pos: IVec2::new(-6_i32, 10_i32),
                    vel: IVec2::new(2_i32, -2_i32),
                },
                Point {
                    pos: IVec2::new(1_i32, 8_i32),
                    vel: IVec2::new(1_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(1_i32, 7_i32),
                    vel: IVec2::new(1_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(-3_i32, 11_i32),
                    vel: IVec2::new(1_i32, -2_i32),
                },
                Point {
                    pos: IVec2::new(7_i32, 6_i32),
                    vel: IVec2::new(-1_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(-2_i32, 3_i32),
                    vel: IVec2::new(1_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(-4_i32, 3_i32),
                    vel: IVec2::new(2_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(10_i32, -3_i32),
                    vel: IVec2::new(-1_i32, 1_i32),
                },
                Point {
                    pos: IVec2::new(5_i32, 11_i32),
                    vel: IVec2::new(1_i32, -2_i32),
                },
                Point {
                    pos: IVec2::new(4_i32, 7_i32),
                    vel: IVec2::new(0_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(8_i32, -2_i32),
                    vel: IVec2::new(0_i32, 1_i32),
                },
                Point {
                    pos: IVec2::new(15_i32, 0_i32),
                    vel: IVec2::new(-2_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(1_i32, 6_i32),
                    vel: IVec2::new(1_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(8_i32, 9_i32),
                    vel: IVec2::new(0_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(3_i32, 3_i32),
                    vel: IVec2::new(-1_i32, 1_i32),
                },
                Point {
                    pos: IVec2::new(0_i32, 5_i32),
                    vel: IVec2::new(0_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(-2_i32, 2_i32),
                    vel: IVec2::new(2_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(5_i32, -2_i32),
                    vel: IVec2::new(1_i32, 2_i32),
                },
                Point {
                    pos: IVec2::new(1_i32, 4_i32),
                    vel: IVec2::new(2_i32, 1_i32),
                },
                Point {
                    pos: IVec2::new(-2_i32, 7_i32),
                    vel: IVec2::new(2_i32, -2_i32),
                },
                Point {
                    pos: IVec2::new(3_i32, 6_i32),
                    vel: IVec2::new(-1_i32, -1_i32),
                },
                Point {
                    pos: IVec2::new(5_i32, 0_i32),
                    vel: IVec2::new(1_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(-6_i32, 0_i32),
                    vel: IVec2::new(2_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(5_i32, 9_i32),
                    vel: IVec2::new(1_i32, -2_i32),
                },
                Point {
                    pos: IVec2::new(14_i32, 7_i32),
                    vel: IVec2::new(-2_i32, 0_i32),
                },
                Point {
                    pos: IVec2::new(-3_i32, 6_i32),
                    vel: IVec2::new(2_i32, -1_i32),
                },
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
    fn test_find_message_and_seconds() {
        for (index, message_and_seconds) in [(
            "\
            #...#..###\n\
            #...#...#.\n\
            #...#...#.\n\
            #####...#.\n\
            #...#...#.\n\
            #...#...#.\n\
            #...#...#.\n\
            #...#..###\n"
                .to_owned(),
            3_usize,
        )]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).find_message_and_seconds(),
                message_and_seconds
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
