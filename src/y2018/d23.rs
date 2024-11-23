use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 23: Experimental Emergency Teleportation ---

Using your torch to search the darkness of the rocky cavern, you finally locate the man's friend: a small reindeer.

You're not sure how it got so far in this cave. It looks sick - too sick to walk - and too heavy for you to carry all the way back. Sleighs won't be invented for another 1500 years, of course.

The only option is experimental emergency teleportation.

You hit the "experimental emergency teleportation" button on the device and push I accept the risk on no fewer than 18 different warning messages. Immediately, the device deploys hundreds of tiny nanobots which fly around the cavern, apparently assembling themselves into a very specific formation. The device lists the X,Y,Z position (pos) for each nanobot as well as its signal radius (r) on its tiny screen (your puzzle input).

Each nanobot can transmit signals to any integer coordinate which is a distance away from it less than or equal to its signal radius (as measured by Manhattan distance). Coordinates a distance away of less than or equal to a nanobot's signal radius are said to be in range of that nanobot.

Before you start the teleportation process, you should determine which nanobot is the strongest (that is, which has the largest signal radius) and then, for that nanobot, the total number of nanobots that are in range of it, including itself.

For example, given the following nanobots:

pos=<0,0,0>, r=4
pos=<1,0,0>, r=1
pos=<4,0,0>, r=3
pos=<0,2,0>, r=1
pos=<0,5,0>, r=3
pos=<0,0,3>, r=1
pos=<1,1,1>, r=1
pos=<1,1,2>, r=1
pos=<1,3,1>, r=1

The strongest nanobot is the first one (position 0,0,0) because its signal radius, 4 is the largest. Using that nanobot's location and signal radius, the following nanobots are in or out of range:

    The nanobot at 0,0,0 is distance 0 away, and so it is in range.
    The nanobot at 1,0,0 is distance 1 away, and so it is in range.
    The nanobot at 4,0,0 is distance 4 away, and so it is in range.
    The nanobot at 0,2,0 is distance 2 away, and so it is in range.
    The nanobot at 0,5,0 is distance 5 away, and so it is not in range.
    The nanobot at 0,0,3 is distance 3 away, and so it is in range.
    The nanobot at 1,1,1 is distance 3 away, and so it is in range.
    The nanobot at 1,1,2 is distance 4 away, and so it is in range.
    The nanobot at 1,3,1 is distance 5 away, and so it is not in range.

In this example, in total, 7 nanobots are in range of the nanobot with the largest signal radius.

Find the nanobot with the largest signal radius. How many nanobots are in range of its signals? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution;

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        todo!()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        todo!();
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        todo!();
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

    const SOLUTION_STRS: &'static [&'static str] = &[""];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![])[index]
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
