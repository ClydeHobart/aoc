use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::collections::HashSet,
};

/* --- Day 1: No Time for a Taxicab ---

Santa's sleigh uses a very high-precision clock to guide its movements, and the clock's oscillator is regulated by stars. Unfortunately, the stars have been stolen... by the Easter Bunny. To save Christmas, Santa needs you to retrieve all fifty stars by December 25th.

Collect stars by solving puzzles. Two puzzles will be made available on each day in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

You're airdropped near Easter Bunny Headquarters in a city somewhere. "Near", unfortunately, is as close as you can get - the instructions on the Easter Bunny Recruiting Document the Elves intercepted start here, and nobody had time to work them out further.

The Document indicates that you should start at the given coordinates (where you just landed) and face North. Then, follow the provided sequence: either turn left (L) or right (R) 90 degrees, then walk forward the given number of blocks, ending at a new intersection.

There's no time to follow such ridiculous instructions on foot, though, so you take a moment and work out the destination. Given that you can only walk on the street grid of the city, how far is the shortest path to the destination?

For example:

    Following R2, L3 leaves you 2 blocks East and 3 blocks North, or 5 blocks away.
    R2, R2, R2 leaves you 2 blocks due South of your starting position, which is 2 blocks away.
    R5, L5, R5, R3 leaves you 12 blocks away.

How many blocks away is Easter Bunny HQ?

--- Part Two ---

Then, you notice the instructions continue on the back of the Recruiting Document. Easter Bunny HQ is actually at the first location you visit twice.

For example, if your instructions are R8, R4, R4, R8, the first location you visit twice is 4 blocks away, due East.

How many blocks away is the first location you visit twice? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Step {
    left: bool,
    blocks: i32,
}

impl Step {
    fn process(&self, state: State) -> State {
        let dir: Direction = state.dir.turn(self.left);
        let pos: IVec2 = state.pos + self.blocks * dir.vec();

        State { pos, dir }
    }
}

impl Parse for Step {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                alt((map(tag("L"), |_| true), map(tag("R"), |_| false))),
                parse_integer::<i32>,
            )),
            |(left, blocks)| Self { left, blocks },
        )(input)
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    pos: IVec2,
    dir: Direction,
}

impl Default for State {
    fn default() -> Self {
        Self {
            pos: IVec2::ZERO,
            dir: Direction::North,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Step>);

impl Solution {
    fn iter_states(&self) -> impl Iterator<Item = State> + '_ {
        let mut state: State = State::default();

        self.0.iter().map(move |step| {
            state = step.process(state);

            state
        })
    }

    fn iter_poses(&self) -> impl Iterator<Item = IVec2> + '_ {
        let mut pos: IVec2 = State::default().pos;

        [State::default().pos]
            .into_iter()
            .chain(self.iter_states().flat_map(move |state| {
                let start: IVec2 = pos;

                pos = state.pos;

                CellIter2D::try_from(start..=state.pos)
                    .unwrap()
                    .skip(1_usize)
            }))
    }

    fn end(&self) -> State {
        self.iter_states().last().unwrap_or_default()
    }

    fn blocks_to_pos(pos: IVec2) -> i32 {
        manhattan_magnitude_2d(pos)
    }

    fn blocks_to_state(state: State) -> i32 {
        Self::blocks_to_pos(state.pos)
    }

    fn blocks_to_end(&self) -> i32 {
        Self::blocks_to_state(self.end())
    }

    fn try_get_first_repeat_pos(&self) -> Option<IVec2> {
        let mut poses: HashSet<IVec2> = HashSet::new();

        self.iter_poses()
            .try_for_each(|pos| if poses.insert(pos) { Ok(()) } else { Err(pos) })
            .map(|_| None)
            .unwrap_or_else(Some)
    }

    fn try_get_first_repeat_blocks(&self) -> Option<i32> {
        self.try_get_first_repeat_pos().map(Self::blocks_to_pos)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Step::parse, opt(tag(", ")))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let easter_bunny_hq: State = self.end();
            let blocks_to_easter_bunny_hq: i32 = Self::blocks_to_state(easter_bunny_hq);

            dbg!(easter_bunny_hq, blocks_to_easter_bunny_hq);
        } else {
            dbg!(self.blocks_to_end());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let easter_bunny_hq: Option<IVec2> = self.try_get_first_repeat_pos();
            let blocks_to_easter_bunny_hq: Option<i32> = easter_bunny_hq.map(Self::blocks_to_pos);

            dbg!(easter_bunny_hq, blocks_to_easter_bunny_hq);
        } else {
            dbg!(self.try_get_first_repeat_blocks());
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

    const SOLUTION_STRS: &'static [&'static str] =
        &["R2, L3", "R2, R2, R2", "R5, L5, R5, R3", "R8, R4, R4, R8"];
    const EASTER_BUNNY_HQ_POSES: &'static [IVec2] =
        &[IVec2::new(2_i32, -3_i32), IVec2::new(0_i32, 2_i32)];
    const EASTER_BUNNY_HQ_BLOCKS: &'static [i32] = &[5_i32, 2_i32, 12_i32];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        enum Turn {
            L,
            R,
        }

        use Turn::*;

        macro_rules! solutions {
            [ $( [ $( $turn:expr, $blocks:literal);* ], )* ] => {
                vec![ $(
                    Solution(vec![ $(
                        Step {
                            left: matches!($turn, L),
                            blocks: $blocks
                        },
                    )* ]),
                )* ]
            };
        }

        &ONCE_LOCK.get_or_init(|| {
            solutions![
                [R, 2; L, 3],
                [R, 2; R, 2; R, 2],
                [R, 5; L, 5; R, 5; R, 3],
                [R, 8; R, 4; R, 4; R, 8],
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_easter_bunny_hq_poses() {
        for (index, easter_bunny_hq_pos) in EASTER_BUNNY_HQ_POSES.into_iter().copied().enumerate() {
            assert_eq!(solution(index).end().pos, easter_bunny_hq_pos);
        }
    }

    #[test]
    fn test_blocks_to_easter_bunny_hq() {
        for (index, easter_bunny_hq_blocks) in
            EASTER_BUNNY_HQ_BLOCKS.into_iter().copied().enumerate()
        {
            assert_eq!(solution(index).blocks_to_end(), easter_bunny_hq_blocks);
        }
    }

    #[test]
    fn test_try_get_first_repeat_pos() {
        assert_eq!(
            solution(3_usize).try_get_first_repeat_pos(),
            Some(IVec2::new(4_i32, 0_i32))
        );
    }
}
