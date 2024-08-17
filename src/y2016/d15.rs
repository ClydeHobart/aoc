use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 15: Timing is Everything ---

The halls open into an interior plaza containing a large kinetic sculpture. The sculpture is in a sealed enclosure and seems to involve a set of identical spherical capsules that are carried to the top and allowed to bounce through the maze of spinning pieces.

Part of the sculpture is even interactive! When a button is pressed, a capsule is dropped and tries to fall through slots in a set of rotating discs to finally go through a little hole at the bottom and come out of the sculpture. If any of the slots aren't aligned with the capsule as it passes, the capsule bounces off the disc and soars away. You feel compelled to get one of those capsules.

The discs pause their motion each second and come in different sizes; they seem to each have a fixed number of positions at which they stop. You decide to call the position with the slot 0, and count up for each position it reaches next.

Furthermore, the discs are spaced out so that after you push the button, one second elapses before the first disc is reached, and one second elapses as the capsule passes from one disc to the one below it. So, if you push the button at time=100, then the capsule reaches the top disc at time=101, the second disc at time=102, the third disc at time=103, and so on.

The button will only drop a capsule at an integer time - no fractional seconds allowed.

For example, at time=0, suppose you see the following arrangement:

Disc #1 has 5 positions; at time=0, it is at position 4.
Disc #2 has 2 positions; at time=0, it is at position 1.

If you press the button exactly at time=0, the capsule would start to fall; it would reach the first disc at time=1. Since the first disc was at position 4 at time=0, by time=1 it has ticked one position forward. As a five-position disc, the next position is 0, and the capsule falls through the slot.

Then, at time=2, the capsule reaches the second disc. The second disc has ticked forward two positions at this point: it started at position 1, then continued to position 0, and finally ended up at position 1 again. Because there's only a slot at position 0, the capsule bounces away.

If, however, you wait until time=5 to push the button, then when the capsule reaches each disc, the first disc will have ticked forward 5+1 = 6 times (to position 0), and the second disc will have ticked forward 5+2 = 7 times (also to position 0). In this case, the capsule would fall through the discs and come out of the machine.

However, your situation has more than two discs; you've noted their positions in your puzzle input. What is the first time you can press the button to get a capsule?

--- Part Two ---

After getting the first capsule (it contained a star! what great fortune!), the machine detects your success and begins to rearrange itself.

When it's done, the discs are back in their original configuration as if it were time=0 again, but a new disc with 11 positions and starting at position 0 has appeared exactly one second below the previously-bottom disc.

With this new disc, and counting again starting from time=0 with the configuration in your puzzle input, what is the first time you can press the button to get another capsule? */

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct Disc {
    num: i64,
    pos_count: i64,
    initial_pos: i64,
}

impl Disc {
    const HOLE: Self = Self {
        num: 0_i64,
        pos_count: 1_i64,
        initial_pos: 0_i64,
    };

    fn as_disc_0(&self) -> Self {
        Self {
            num: 0_i64,
            pos_count: self.pos_count,
            initial_pos: (self.initial_pos + self.num) % self.pos_count,
        }
    }
}

impl Parse for Disc {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("Disc #"),
                parse_integer,
                tag(" has "),
                parse_integer,
                tag(" positions; at time=0, it is at position "),
                parse_integer,
                tag("."),
            )),
            |(_, num, _, pos_count, _, initial_pos, _)| Self {
                num,
                pos_count,
                initial_pos,
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Disc>);

impl Solution {
    fn try_disc_iter_as_disc<'a, T: Iterator<Item = &'a Disc>>(iter: T) -> Option<Disc> {
        iter.map(Disc::as_disc_0)
            .try_fold(Disc::HOLE, |accumulator, disc| {
                let output: ExtendedEuclideanAlgorithmOutput =
                    extended_euclidean_algorithm(accumulator.pos_count, disc.pos_count);

                // This uses math from https://en.wikipedia.org/wiki/Chinese_remainder_theorem
                (output.gcd == 1_i64).then(|| {
                    let t: i64 = accumulator.initial_pos * output.y * disc.pos_count
                        + disc.initial_pos * output.x * accumulator.pos_count;
                    let pos_count: i64 = accumulator.pos_count * disc.pos_count;
                    let initial_pos: i64 = t.rem_euclid(pos_count);

                    Disc {
                        num: 0_i64,
                        pos_count,
                        initial_pos,
                    }
                })
            })
    }

    fn try_as_disc(&self) -> Option<Disc> {
        Self::try_disc_iter_as_disc(self.0.iter())
    }

    fn try_first_time(&self) -> Option<i64> {
        self.try_as_disc()
            .map(|disc| disc.pos_count - disc.initial_pos)
    }

    fn try_as_disc_with_extra_disc(&self) -> Option<Disc> {
        let max_num: i64 = self.0.iter().map(|disc| disc.num).max().unwrap_or_default();
        let extra_disc: Disc = Disc {
            num: max_num + 1_i64,
            pos_count: 11_i64,
            initial_pos: 0_i64,
        };

        Self::try_disc_iter_as_disc(self.0.iter().chain([&extra_disc]))
    }

    fn try_first_time_with_extra_disc(&self) -> Option<i64> {
        self.try_as_disc_with_extra_disc()
            .map(|disc| disc.pos_count - disc.initial_pos)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Disc::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(disc) = self.try_as_disc() {
                let first_time: i64 = disc.pos_count - disc.initial_pos;

                dbg!(&disc, first_time);
            } else {
                eprintln!("Failed to convert to disc!");
            }
        } else {
            dbg!(self.try_first_time());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(disc_with_extra_disc) = self.try_as_disc_with_extra_disc() {
                let first_time_with_extra_disc: i64 =
                    disc_with_extra_disc.pos_count - disc_with_extra_disc.initial_pos;

                dbg!(&disc_with_extra_disc, first_time_with_extra_disc);
            } else {
                eprintln!("Failed to convert to disc!");
            }
        } else {
            dbg!(self.try_first_time_with_extra_disc());
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

    const SOLUTION_STR: &'static str = "\
        Disc #1 has 5 positions; at time=0, it is at position 4.\n\
        Disc #2 has 2 positions; at time=0, it is at position 1.\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Disc {
                    num: 1_i64,
                    pos_count: 5_i64,
                    initial_pos: 4_i64,
                },
                Disc {
                    num: 2_i64,
                    pos_count: 2_i64,
                    initial_pos: 1_i64,
                },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_as_disc() {
        assert_eq!(
            solution().try_as_disc(),
            Some(Disc {
                num: 0_i64,
                pos_count: 10_i64,
                initial_pos: 5_i64
            })
        );
    }

    #[test]
    fn test_try_first_time() {
        assert_eq!(solution().try_first_time(), Some(5_i64));
    }

    #[test]
    fn test() {
        let input: String = std::fs::read_to_string("input/y2016/d15.txt").unwrap();
        let solution: Solution = input.as_str().try_into().unwrap();

        solution.try_as_disc();
    }
}
