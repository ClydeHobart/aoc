use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    strum::EnumCount as EnumCountTrait,
    strum_macros::EnumCount,
};

/* --- Day 12: Leonardo's Monorail ---

You finally reach the top floor of this building: a garden with a slanted glass ceiling. Looks like there are no more stars to be had.

While sitting on a nearby bench amidst some tiger lilies, you manage to decrypt some of the files you extracted from the servers downstairs.

According to these documents, Easter Bunny HQ isn't just this building - it's a collection of buildings in the nearby area. They're all connected by a local monorail, and there's another building not far from here! Unfortunately, being night, the monorail is currently not operating.

You remotely connect to the monorail control systems and discover that the boot sequence expects a password. The password-checking logic (your puzzle input) is easy to extract, but the code it uses is strange: it's assembunny code designed for the new computer you just assembled. You'll have to execute the code and get the password.

The assembunny code you've extracted operates on four registers (a, b, c, and d) that start at 0 and can hold any integer. However, it seems to make use of only a few instructions:

    cpy x y copies x (either an integer or the value of a register) into register y.
    inc x increases the value of register x by one.
    dec x decreases the value of register x by one.
    jnz x y jumps to an instruction y away (positive means forward; negative means backward), but only if x is not zero.

The jnz instruction moves relative to itself: an offset of -1 would continue at the previous instruction, while an offset of 2 would skip over the next instruction.

For example:

cpy 41 a
inc a
inc a
dec a
jnz a 2
dec a

The above code would set register a to 41, increase its value by 2, decrease its value by 1, and then skip the last dec a (because a is not zero, so the jnz a 2 skips it), leaving register a at 42. When you move past the last instruction, the program halts.

After executing the assembunny code in your puzzle input, what value is left in register a?

--- Part Two ---

As you head down the fire escape to the monorail, you notice it didn't start; register c needs to be initialized to the position of the ignition key.

If you instead initialize register c to be 1, what value is now left in register a? */

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, EnumCount, PartialEq)]
    pub enum Register {
        A = A_VALUE = b'a',
        B = B_VALUE = b'b',
        C = C_VALUE = b'c',
        D = D_VALUE = b'd',
    }
}

impl Register {
    pub fn index(self) -> usize {
        ((self as u8) - Self::A_VALUE) as usize
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Copy, Clone, Debug)]
pub enum Value {
    Register(Register),
    Constant(i32),
}

impl Value {
    fn evaluate(&self, state: &State) -> i32 {
        match self {
            Value::Register(register) => state.registers[register.index()],
            Value::Constant(constant) => *constant,
        }
    }
}

impl Parse for Value {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Register::parse, Self::Register),
            map(parse_integer, Self::Constant),
        ))(input)
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug)]
pub struct State {
    pub registers: [i32; Register::COUNT],
    is_toggled: BitVec,
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug)]
pub enum Instruction {
    Cpy { x: Value, y: Register },
    Inc { x: Register },
    Dec { x: Register },
    JNZ { x: Value, y: Value },
    Tgl { x: Value },
}

impl Instruction {
    fn process(&self, state: &mut State, index: usize) -> isize {
        if let Some(instruction) = (!state.is_toggled[index])
            .then_some(self.clone())
            .or_else(|| self.try_invert())
        {
            match instruction {
                Self::Cpy { x, y } => {
                    let value: i32 = x.evaluate(state);

                    state.registers[y.index()] = value;

                    1_isize
                }
                Self::Inc { x } => {
                    state.registers[x.index()] += 1_i32;

                    1_isize
                }
                Self::Dec { x } => {
                    state.registers[x.index()] -= 1_i32;

                    1_isize
                }
                Self::JNZ { x, y } => {
                    if x.evaluate(state) != 0_i32 {
                        y.evaluate(state) as isize
                    } else {
                        1_isize
                    }
                }
                Self::Tgl { x } => {
                    if let Some(index) = usize::try_from(index as i32 + x.evaluate(state))
                        .ok()
                        .filter(|index| *index < state.is_toggled.len())
                    {
                        let value: bool = !state.is_toggled[index];

                        state.is_toggled.set(index, value);
                    }

                    1_isize
                }
            }
        } else {
            1_isize
        }
    }

    fn try_invert(&self) -> Option<Self> {
        match self.clone() {
            Self::Cpy { x, y } => Some(Self::JNZ {
                x,
                y: Value::Register(y),
            }),
            Self::Inc { x } => Some(Self::Dec { x }),
            Self::Dec { x } => Some(Self::Inc { x }),
            Self::JNZ { x, y } => match y {
                Value::Register(y) => Some(Self::Cpy { x, y }),
                Value::Constant(_) => None,
            },
            Self::Tgl { x } => match x {
                Value::Register(x) => Some(Self::Inc { x }),
                Value::Constant(_) => None,
            },
        }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                tuple((tag("cpy "), Value::parse, tag(" "), Register::parse)),
                |(_, x, _, y)| Self::Cpy { x, y },
            ),
            map(preceded(tag("inc "), Register::parse), |x| Self::Inc { x }),
            map(preceded(tag("dec "), Register::parse), |x| Self::Dec { x }),
            map(
                tuple((tag("jnz "), Value::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::JNZ { x, y },
            ),
            map(preceded(tag("tgl "), Value::parse), |x| Self::Tgl { x }),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    pub fn parse_instructions<'i>(input: &'i str) -> IResult<&'i str, Vec<Instruction>> {
        many0(terminated(Instruction::parse, opt(line_ending)))(input)
    }

    pub fn process_for_state(instructions: &[Instruction], state: State) -> State {
        let mut state: State = state;
        let mut instruction_index: isize = 0_isize;

        while let Some(instruction) = usize::try_from(instruction_index)
            .ok()
            .map(|instruction_index| instructions.get(instruction_index))
            .flatten()
        {
            instruction_index += instruction.process(&mut state, instruction_index as usize);
        }

        state
    }

    pub fn state_for_instructions(instructions: &[Instruction]) -> State {
        State {
            registers: Default::default(),
            is_toggled: bitvec![0; instructions.len()],
        }
    }

    fn process(&self) -> State {
        Self::process_for_state(&self.0, Self::state_for_instructions(&self.0))
    }

    fn a_after_process(&self) -> i32 {
        self.process().registers[Register::A.index()]
    }

    fn process_with_ignition_key(&self) -> State {
        let mut state: State = Self::state_for_instructions(&self.0);

        state.registers[Register::C.index()] = 1_i32;

        Self::process_for_state(&self.0, state)
    }

    fn a_after_process_with_ignition_key(&self) -> i32 {
        self.process_with_ignition_key().registers[Register::A.index()]
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Self::parse_instructions, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.process());
        } else {
            dbg!(self.a_after_process());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.process_with_ignition_key());
        } else {
            dbg!(self.a_after_process_with_ignition_key());
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
        cpy 41 a\n\
        inc a\n\
        inc a\n\
        dec a\n\
        jnz a 2\n\
        dec a\n";

    fn solution() -> &'static Solution {
        use {
            Instruction::*,
            Register::*,
            Value::{Constant as Con, Register as Reg},
        };

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Cpy { x: Con(41), y: A },
                Inc { x: A },
                Inc { x: A },
                Dec { x: A },
                JNZ {
                    x: Reg(A),
                    y: Con(2),
                },
                Dec { x: A },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_instruction_process() {
        const STATE_AND_PROCESS_OUTPUTS: &'static [(i32, isize)] = &[
            (0_i32, 1_isize),
            (41_i32, 1_isize),
            (42_i32, 1_isize),
            (43_i32, 1_isize),
            (42_i32, 2_isize),
            (42_i32, 0_isize),
        ];

        for (instruction_index, state_and_process_outputs) in
            STATE_AND_PROCESS_OUTPUTS.windows(2_usize).enumerate()
        {
            let (start_a, process_output): (i32, isize) = state_and_process_outputs[0_usize];
            let end_a: i32 = state_and_process_outputs[1_usize].0;
            let mut state: State = State {
                registers: [start_a, 0_i32, 0_i32, 0_i32],
                is_toggled: bitvec![0; solution().0.len()],
            };

            assert_eq!(
                solution().0[instruction_index].process(&mut state, instruction_index),
                process_output
            );
            assert_eq!(state.registers, [end_a, 0_i32, 0_i32, 0_i32]);
        }
    }

    #[test]
    fn test_solution_process() {
        assert_eq!(
            solution().process().registers,
            [42_i32, 0_i32, 0_i32, 0_i32]
        );
    }
}
