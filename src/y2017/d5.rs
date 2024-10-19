use {
    crate::*,
    nom::{
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::terminated,
        Err, IResult,
    },
};

/* --- Day 5: A Maze of Twisty Trampolines, All Alike ---

An urgent interrupt arrives from the CPU: it's trapped in a maze of jump instructions, and it would like assistance from any programs with spare cycles to help find the exit.

The message includes a list of the offsets for each jump. Jumps are relative: -1 moves to the previous instruction, and 2 skips the next one. Start at the first instruction in the list. The goal is to follow the jumps until one leads outside the list.

In addition, these instructions are a little strange; after each jump, the offset of that instruction increases by 1. So, if you come across an offset of 3, you would move three instructions forward, but change it to a 4 for the next time it is encountered.

For example, consider the following list of jump offsets:

0
3
0
1
-3

Positive jumps ("forward") move downward; negative jumps move upward. For legibility in this example, these offset values will be written all on one line, with the current instruction marked in parentheses. The following steps would be taken before an exit is found:

    (0) 3  0  1  -3  - before we have taken any steps.
    (1) 3  0  1  -3  - jump with offset 0 (that is, don't jump at all). Fortunately, the instruction is then incremented to 1.
     2 (3) 0  1  -3  - step forward because of the instruction we just modified. The first instruction is incremented again, now to 2.
     2  4  0  1 (-3) - jump all the way to the end; leave a 4 behind.
     2 (4) 0  1  -2  - go back to where we just were; increment -3 to -2.
     2  5  0  1  -2  - jump 4 steps forward, escaping the maze.

In this example, the exit is reached in 5 steps.

How many steps does it take to reach the exit?

--- Part Two ---

Now, the jumps are even stranger: after each jump, if the offset was three or more, instead decrease it by 1. Otherwise, increase it by 1 as before.

Using this rule with the above example, the process now takes 10 steps, and the offset values after finding the exit are left as 2 3 2 3 -1.

How many steps does it now take to reach the exit? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<i32>);

impl Solution {
    fn steps_until_exit_with_jumps_internal<M: Fn(&mut i32)>(
        &self,
        modify_jump: M,
    ) -> (usize, Vec<i32>) {
        let mut steps: usize = 0_usize;
        let mut index: i32 = 0_i32;
        let mut jumps: Vec<i32> = self.0.clone();

        while let Some(index_usize) = usize::try_from(index)
            .ok()
            .filter(|index| *index < jumps.len())
        {
            let jump: &mut i32 = &mut jumps[index_usize];
            index += *jump;
            modify_jump(jump);
            steps += 1_usize;
        }

        (steps, jumps)
    }

    fn increment_jump(jump: &mut i32) {
        *jump += 1_i32;
    }

    fn steps_until_exit_with_jumps(&self) -> (usize, Vec<i32>) {
        self.steps_until_exit_with_jumps_internal(Self::increment_jump)
    }

    fn steps_until_exit(&self) -> usize {
        self.steps_until_exit_with_jumps().0
    }

    fn strange_jump(jump: &mut i32) {
        *jump += if *jump >= 3_i32 { -1_i32 } else { 1_i32 };
    }

    fn strange_steps_until_exit_with_jumps(&self) -> (usize, Vec<i32>) {
        self.steps_until_exit_with_jumps_internal(Self::strange_jump)
    }

    fn strange_steps_until_exit(&self) -> usize {
        self.strange_steps_until_exit_with_jumps().0
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(parse_integer, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Getting flashbacks from AOC 2016. This one's a lot more mild so far.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.steps_until_exit());
    }

    /// Pretty easy to extend the first one for this one.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.strange_steps_until_exit());
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
        0\n\
        3\n\
        0\n\
        1\n\
        -3\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![Solution(vec![0_i32, 3_i32, 0_i32, 1_i32, -3_i32])])[index]
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
    fn test_steps_until_exit_with_jumps() {
        for (index, steps_until_exit_with_jumps) in
            [(5_usize, vec![2_i32, 5_i32, 0_i32, 1_i32, -2_i32])]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).steps_until_exit_with_jumps(),
                steps_until_exit_with_jumps
            );
        }
    }

    #[test]
    fn test_strange_steps_until_exit_with_jumps() {
        for (index, strange_steps_until_exit_with_jumps) in
            [(10_usize, vec![2_i32, 3_i32, 2_i32, 3_i32, -1_i32])]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).strange_steps_until_exit_with_jumps(),
                strange_steps_until_exit_with_jumps
            );
        }
    }
}
