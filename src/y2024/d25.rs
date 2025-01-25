use {
    crate::*,
    glam::IVec2,
    nom::{
        character::complete::line_ending,
        combinator::{map, map_opt},
        error::Error,
        multi::separated_list0,
        Err, IResult,
    },
};

/* --- Day 25: Code Chronicle ---

Out of ideas and time, The Historians agree that they should go back to check the Chief Historian's office one last time, just in case he went back there without you noticing.

When you get there, you are surprised to discover that the door to his office is locked! You can hear someone inside, but knocking yields no response. The locks on this floor are all fancy, expensive, virtual versions of five-pin tumbler locks, so you contact North Pole security to see if they can help open the door.

Unfortunately, they've lost track of which locks are installed and which keys go with them, so the best they can do is send over schematics of every lock and every key for the floor you're on (your puzzle input).

The schematics are in a cryptic file format, but they do contain manufacturer information, so you look up their support number.

"Our Virtual Five-Pin Tumbler product? That's our most expensive model! Way more secure than--" You explain that you need to open a door and don't have a lot of time.

"Well, you can't know whether a key opens a lock without actually trying the key in the lock (due to quantum hidden variables), but you can rule out some of the key/lock combinations."

"The virtual system is complicated, but part of it really is a crude simulation of a five-pin tumbler lock, mostly for marketing reasons. If you look at the schematics, you can figure out whether a key could possibly fit in a lock."

He transmits you some example schematics:

#####
.####
.####
.####
.#.#.
.#...
.....

#####
##.##
.#.##
...##
...#.
...#.
.....

.....
#....
#....
#...#
#.#.#
#.###
#####

.....
.....
#.#..
###..
###.#
###.#
#####

.....
.....
.....
#....
#.#..
#.#.#
#####

"The locks are schematics that have the top row filled (#) and the bottom row empty (.); the keys have the top row empty and the bottom row filled. If you look closely, you'll see that each schematic is actually a set of columns of various heights, either extending downward from the top (for locks) or upward from the bottom (for keys)."

"For locks, those are the pins themselves; you can convert the pins in schematics to a list of heights, one per column. For keys, the columns make up the shape of the key where it aligns with pins; those can also be converted to a list of heights."

"So, you could say the first lock has pin heights 0,5,3,4,3:"

#####
.####
.####
.####
.#.#.
.#...
.....

"Or, that the first key has heights 5,0,2,1,3:"

.....
#....
#....
#...#
#.#.#
#.###
#####

"These seem like they should fit together; in the first four columns, the pins and key don't overlap. However, this key cannot be for this lock: in the rightmost column, the lock's pin overlaps with the key, which you know because in that column the sum of the lock height and key height is more than the available space."

"So anyway, you can narrow down the keys you'd need to try by just testing each key with each lock, which means you would have to check... wait, you have how many locks? But the only installation that size is at the North--" You disconnect the call.

In this example, converting both locks to pin heights produces:

0,5,3,4,3
1,2,0,5,3

Converting all three keys to heights produces:

5,0,2,1,3
4,3,4,0,2
3,0,2,0,1

Then, you can try every key with every lock:

    Lock 0,5,3,4,3 and key 5,0,2,1,3: overlap in the last column.
    Lock 0,5,3,4,3 and key 4,3,4,0,2: overlap in the second column.
    Lock 0,5,3,4,3 and key 3,0,2,0,1: all columns fit!
    Lock 1,2,0,5,3 and key 5,0,2,1,3: overlap in the first column.
    Lock 1,2,0,5,3 and key 4,3,4,0,2: all columns fit!
    Lock 1,2,0,5,3 and key 3,0,2,0,1: all columns fit!

So, in this example, the number of unique lock/key pairs that fit together without overlapping in any column is 3.

Analyze your lock and key schematics. How many unique lock/key pairs fit together without overlapping in any column? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct PinHeights([u8; Schematic::PINS]);

impl PinHeights {
    fn overlaps(&self, other: &Self) -> bool {
        self.0
            .into_iter()
            .zip(other.0.into_iter())
            .any(|(height_a, height_b)| height_a + height_b > Schematic::PINS as u8)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Schematic {
    Lock(PinHeights),
    Key(PinHeights),
}

impl Schematic {
    const PINS: usize = 5_usize;
    const DIMENSIONS: IVec2 = IVec2::new(Self::PINS as i32, Self::PINS as i32 + 2_i32);

    fn try_compute_heights(grid: &Grid2D<Pixel>, dir: Direction) -> Option<PinHeights> {
        CellIter2D::until_boundary(
            grid,
            IVec2::Y
                * match dir {
                    Direction::South => Some(0_i32),
                    Direction::North => Some(Self::DIMENSIONS.y - 1_i32),
                    _ => None,
                }?,
            Direction::East,
        )
        .try_fold(
            PinHeights([0_u8; Self::PINS]),
            |mut pin_heights, col_pos| {
                CellIter2D::until_boundary(grid, col_pos, dir)
                    .try_fold((Pixel::Light, 0_i32), |(expected_pixel, height), pos| {
                        let pixel: Pixel = *grid.get(pos).unwrap();

                        if pixel == expected_pixel {
                            Some((expected_pixel, height + pixel.is_light() as i32))
                        } else if expected_pixel == Pixel::Light {
                            Some((Pixel::Dark, height))
                        } else {
                            None
                        }
                    })
                    .and_then(|(_, height)| {
                        (1_i32..=(1_i32 + Self::PINS as i32))
                            .contains(&height)
                            .then(|| {
                                pin_heights.0[col_pos.x as usize] = height as u8 - 1_u8;

                                pin_heights
                            })
                    })
            },
        )
    }
}

impl Parse for Schematic {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |grid| {
            (grid.dimensions() == Self::DIMENSIONS)
                .then(|| {
                    Self::try_compute_heights(&grid, Direction::South)
                        .map(Self::Lock)
                        .or_else(|| {
                            Self::try_compute_heights(&grid, Direction::North).map(Self::Key)
                        })
                })
                .flatten()
        })(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    locks: Vec<PinHeights>,
    keys: Vec<PinHeights>,
}

impl Solution {
    fn non_overlapping_pair_count(&self) -> usize {
        self.locks
            .iter()
            .flat_map(|lock| self.keys.iter().map(move |key| (lock, key)))
            .filter(|(lock, key)| !lock.overlaps(key))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut locks: Vec<PinHeights> = Vec::new();
        let mut keys: Vec<PinHeights> = Vec::new();

        let input: &str = separated_list0(
            line_ending,
            map(Schematic::parse, |schematic| match schematic {
                Schematic::Lock(lock) => locks.push(lock),
                Schematic::Key(key) => keys.push(key),
            }),
        )(input)?
        .0;

        Ok((input, Self { locks, keys }))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.non_overlapping_pair_count());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {}
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
        #####\n\
        .####\n\
        .####\n\
        .####\n\
        .#.#.\n\
        .#...\n\
        .....\n\
        \n\
        #####\n\
        ##.##\n\
        .#.##\n\
        ...##\n\
        ...#.\n\
        ...#.\n\
        .....\n\
        \n\
        .....\n\
        #....\n\
        #....\n\
        #...#\n\
        #.#.#\n\
        #.###\n\
        #####\n\
        \n\
        .....\n\
        .....\n\
        #.#..\n\
        ###..\n\
        ###.#\n\
        ###.#\n\
        #####\n\
        \n\
        .....\n\
        .....\n\
        .....\n\
        #....\n\
        #.#..\n\
        #.#.#\n\
        #####\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                locks: vec![
                    PinHeights([0_u8, 5_u8, 3_u8, 4_u8, 3_u8]),
                    PinHeights([1_u8, 2_u8, 0_u8, 5_u8, 3_u8]),
                ],
                keys: vec![
                    PinHeights([5_u8, 0_u8, 2_u8, 1_u8, 3_u8]),
                    PinHeights([4_u8, 3_u8, 4_u8, 0_u8, 2_u8]),
                    PinHeights([3_u8, 0_u8, 2_u8, 0_u8, 1_u8]),
                ],
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
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
