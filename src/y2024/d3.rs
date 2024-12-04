use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::anychar,
        combinator::{all_consuming, map, peek, success},
        error::Error,
        multi::many0,
        sequence::{preceded, tuple},
        Err, IResult,
    },
};

/* --- Day 3: Mull It Over ---

"Our computers are having issues, so I have no idea if we have any Chief Historians in stock! You're welcome to check the warehouse, though," says the mildly flustered shopkeeper at the North Pole Toboggan Rental Shop. The Historians head out to take a look.

The shopkeeper turns to you. "Any chance you can see why our computers are having issues again?"

The computer appears to be trying to run a program, but its memory (your puzzle input) is corrupted. All of the instructions have been jumbled up!

It seems like the goal of the program is just to multiply some numbers. It does that with instructions like mul(X,Y), where X and Y are each 1-3 digit numbers. For instance, mul(44,46) multiplies 44 by 46 to get a result of 2024. Similarly, mul(123,4) would multiply 123 by 4.

However, because the program's memory has been corrupted, there are also many invalid characters that should be ignored, even if they look like part of a mul instruction. Sequences like mul(4*, mul(6,9!, ?(12,34), or mul ( 2 , 4 ) do nothing.

For example, consider the following section of corrupted memory:

xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))

Only the four highlighted sections are real mul instructions. Adding up the result of each instruction produces 161 (2*4 + 5*5 + 11*8 + 8*5).

Scan the corrupted memory for uncorrupted mul instructions. What do you get if you add up all of the results of the multiplications?

--- Part Two ---

As you scan through the corrupted memory, you notice that some of the conditional statements are also still intact. If you handle some of the uncorrupted conditional statements in the program, you might be able to get an even more accurate result.

There are two new instructions you'll need to handle:

    The do() instruction enables future mul instructions.
    The don't() instruction disables future mul instructions.

Only the most recent do() or don't() instruction applies. At the beginning of the program, mul instructions are enabled.

For example:

xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))

This corrupted memory is similar to the example from before, but this time the mul(5,5) and mul(11,8) instructions are disabled because there is a don't() instruction before them. The other mul instructions function normally, including the one at the end that gets re-enabled by a do() instruction.

This time, the sum of the results is 48 (2*4 + 8*5).

Handle the new instructions; what do you get if you add up all of the results of just the enabled multiplications? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct MultiplyInstruction {
    a: i32,
    b: i32,
}

impl MultiplyInstruction {
    fn multiply(self) -> i32 {
        self.a * self.b
    }
}

impl Parse for MultiplyInstruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("mul("),
                parse_integer,
                tag(","),
                parse_integer,
                tag(")"),
            )),
            |(_, a, _, b, _)| Self { a, b },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Instruction {
    Multiply(MultiplyInstruction),
    Corrupted,
    Do,
    Dont,
}

impl Instruction {
    fn parse_multiply<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(MultiplyInstruction::parse, Self::Multiply)(input)
    }

    fn parse_corrupted<'i>(input: &'i str) -> IResult<&'i str, Self> {
        preceded(
            anychar,
            alt((
                map(all_consuming(success(())), |_| Self::Corrupted),
                map(
                    peek(alt((
                        Self::parse_multiply,
                        Self::parse_do,
                        Self::parse_dont,
                    ))),
                    |_| Self::Corrupted,
                ),
                Self::parse_corrupted,
            )),
        )(input)
    }

    fn parse_do<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(tag("do()"), |_| Self::Do)(input)
    }

    fn parse_dont<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(tag("don't()"), |_| Self::Dont)(input)
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::parse_multiply,
            Self::parse_do,
            Self::parse_dont,
            Self::parse_corrupted,
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    fn all_product_sum(&self) -> i32 {
        self.0
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Multiply(multiply_instruction) => Some(*multiply_instruction),
                _ => None,
            })
            .map(MultiplyInstruction::multiply)
            .sum()
    }

    fn enabled_product_sum(&self) -> i32 {
        let mut is_enabled: bool = true;

        self.0
            .iter()
            .filter_map(|instruction| {
                let mut return_value: Option<MultiplyInstruction> = None;

                match instruction {
                    Instruction::Multiply(multiply_instruction) => {
                        return_value = is_enabled.then_some(*multiply_instruction);
                    }
                    Instruction::Corrupted => (),
                    Instruction::Do => is_enabled = true,
                    Instruction::Dont => is_enabled = false,
                }

                return_value
            })
            .map(MultiplyInstruction::multiply)
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(Instruction::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I feel like parsing this would've been easier in regex.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.all_product_sum());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.enabled_product_sum());
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
        "mul(44,46)",
        "mul(123,4)",
        "mul(4*",
        "mul(6,9!",
        "?(12,34)",
        "mul ( 2 , 4 )",
        "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))",
        "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![Instruction::Multiply(MultiplyInstruction {
                    a: 44_i32,
                    b: 46_i32,
                })]),
                Solution(vec![Instruction::Multiply(MultiplyInstruction {
                    a: 123_i32,
                    b: 4_i32,
                })]),
                Solution(vec![Instruction::Corrupted]),
                Solution(vec![Instruction::Corrupted]),
                Solution(vec![Instruction::Corrupted]),
                Solution(vec![Instruction::Corrupted]),
                Solution(vec![
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction { a: 2_i32, b: 4_i32 }),
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction { a: 5_i32, b: 5_i32 }),
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction {
                        a: 11_i32,
                        b: 8_i32,
                    }),
                    Instruction::Multiply(MultiplyInstruction { a: 8_i32, b: 5_i32 }),
                    Instruction::Corrupted,
                ]),
                Solution(vec![
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction { a: 2_i32, b: 4_i32 }),
                    Instruction::Corrupted,
                    Instruction::Dont,
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction { a: 5_i32, b: 5_i32 }),
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction {
                        a: 11_i32,
                        b: 8_i32,
                    }),
                    Instruction::Corrupted,
                    Instruction::Do,
                    Instruction::Corrupted,
                    Instruction::Multiply(MultiplyInstruction { a: 8_i32, b: 5_i32 }),
                    Instruction::Corrupted,
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
    fn test_all_product_sum() {
        for (index, all_product_sum) in [
            2024_i32, 492_i32, 0_i32, 0_i32, 0_i32, 0_i32, 161_i32, 161_i32,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).all_product_sum(), all_product_sum);
        }
    }

    #[test]
    fn test_enabled_product_sum() {
        for (index, enabled_product_sum) in [
            2024_i32, 492_i32, 0_i32, 0_i32, 0_i32, 0_i32, 161_i32, 48_i32,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).enabled_product_sum(), enabled_product_sum);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
