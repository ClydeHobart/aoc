use {
    crate::*,
    nom::{combinator::map, error::Error, multi::many0, Err, IResult},
};

/* --- Day 1: Not Quite Lisp ---

Santa was hoping for a white Christmas, but his weather machine's "snow" function is powered by stars, and he's fresh out! To save Christmas, he needs you to collect fifty stars by December 25th.

Collect stars by helping Santa solve puzzles. Two puzzles will be made available on each day in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

Here's an easy puzzle to warm you up.

Santa is trying to deliver presents in a large apartment building, but he can't find the right floor - the directions he got are a little confusing. He starts on the ground floor (floor 0) and then follows the instructions one character at a time.

An opening parenthesis, (, means he should go up one floor, and a closing parenthesis, ), means he should go down one floor.

The apartment building is very tall, and the basement is very deep; he will never find the top or bottom floors.

For example:

    (()) and ()() both result in floor 0.
    ((( and (()(()( both result in floor 3.
    ))((((( also results in floor 3.
    ()) and ))( both result in floor -1 (the first basement level).
    ))) and )())()) both result in floor -3.

To what floor do the instructions take Santa?

--- Part Two ---

Now, given the same instructions, find the position of the first character that causes him to enter the basement (floor -1). The first character in the instructions has position 1, the second character has position 2, and so on.

For example:

    ) causes him to enter the basement at character position 1.
    ()()) causes him to enter the basement at character position 5.

What is the position of the character that causes Santa to first enter the basement? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug, PartialEq))]
    #[derive(Clone, Copy)]
    enum Instruction {
        Up = UP = b'(',
        Down = DOWN = b')',
    }
}

impl Instruction {
    fn floor_delta(self) -> i32 {
        match self {
            Self::Up => 1_i32,
            Self::Down => -1_i32,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    fn floor(&self) -> i32 {
        self.0.iter().copied().map(Instruction::floor_delta).sum()
    }

    fn try_first_basement_position(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .try_fold(0_i32, |mut floor, (instruction_index, instruction)| {
                floor += instruction.floor_delta();

                if floor >= 0_i32 {
                    Ok(floor)
                } else {
                    Err(instruction_index + 1_usize)
                }
            })
            .err()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(Instruction::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Refreshing
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.floor());
    }

    /// one-indexing, yuck
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_first_basement_position());
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
        "(())", "()()", "(((", "(()(()(", "))(((((", "())", "))(", ")))", ")())())",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Instruction::{Down as D, Up as U};

            vec![
                Solution(vec![U, U, D, D]),
                Solution(vec![U, D, U, D]),
                Solution(vec![U, U, U]),
                Solution(vec![U, U, D, U, U, D, U]),
                Solution(vec![D, D, U, U, U, U, U]),
                Solution(vec![U, D, D]),
                Solution(vec![D, D, U]),
                Solution(vec![D, D, D]),
                Solution(vec![D, U, D, D, U, D, D]),
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
    fn test_floor() {
        for (index, floor) in [
            0_i32, 0_i32, 3_i32, 3_i32, 3_i32, -1_i32, -1_i32, -3_i32, -3_i32,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).floor(), floor);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
