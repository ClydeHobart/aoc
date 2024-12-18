use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, verify},
        error::Error,
        multi::{many_m_n, separated_list1},
        sequence::tuple,
        Err, IResult,
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator, VariantNames},
    strum_macros::EnumVariantNames,
};

/* --- Day 17: Chronospatial Computer ---

The Historians push the button on their strange device, but this time, you all just feel like you're falling.

"Situation critical", the device announces in a familiar voice. "Bootstrapping process failed. Initializing debugger...."

The small handheld device suddenly unfolds into an entire computer! The Historians look around nervously before one of them tosses it to you.

This seems to be a 3-bit computer: its program is a list of 3-bit numbers (0 through 7), like 0,1,2,3. The computer also has three registers named A, B, and C, but these registers aren't limited to 3 bits and can instead hold any integer.

The computer knows eight instructions, each identified by a 3-bit number (called the instruction's opcode). Each instruction also reads the 3-bit number after it as an input; this is called its operand.

A number called the instruction pointer identifies the position in the program from which the next opcode will be read; it starts at 0, pointing at the first 3-bit number in the program. Except for jump instructions, the instruction pointer increases by 2 after each instruction is processed (to move past the instruction's opcode and its operand). If the computer tries to read an opcode past the end of the program, it instead halts.

So, the program 0,1,2,3 would run the instruction whose opcode is 0 and pass it the operand 1, then run the instruction having opcode 2 and pass it the operand 3, then halt.

There are two types of operands; each instruction specifies the type of its operand. The value of a literal operand is the operand itself. For example, the value of the literal operand 7 is the number 7. The value of a combo operand can be found as follows:

    Combo operands 0 through 3 represent literal values 0 through 3.
    Combo operand 4 represents the value of register A.
    Combo operand 5 represents the value of register B.
    Combo operand 6 represents the value of register C.
    Combo operand 7 is reserved and will not appear in valid programs.

The eight instructions are as follows:

The adv instruction (opcode 0) performs division. The numerator is the value in the A register. The denominator is found by raising 2 to the power of the instruction's combo operand. (So, an operand of 2 would divide A by 4 (2^2); an operand of 5 would divide A by 2^B.) The result of the division operation is truncated to an integer and then written to the A register.

The bxl instruction (opcode 1) calculates the bitwise XOR of register B and the instruction's literal operand, then stores the result in register B.

The bst instruction (opcode 2) calculates the value of its combo operand modulo 8 (thereby keeping only its lowest 3 bits), then writes that value to the B register.

The jnz instruction (opcode 3) does nothing if the A register is 0. However, if the A register is not zero, it jumps by setting the instruction pointer to the value of its literal operand; if this instruction jumps, the instruction pointer is not increased by 2 after this instruction.

The bxc instruction (opcode 4) calculates the bitwise XOR of register B and register C, then stores the result in register B. (For legacy reasons, this instruction reads an operand but ignores it.)

The out instruction (opcode 5) calculates the value of its combo operand modulo 8, then outputs that value. (If a program outputs multiple values, they are separated by commas.)

The bdv instruction (opcode 6) works exactly like the adv instruction except that the result is stored in the B register. (The numerator is still read from the A register.)

The cdv instruction (opcode 7) works exactly like the adv instruction except that the result is stored in the C register. (The numerator is still read from the A register.)

Here are some examples of instruction operation:

    If register C contains 9, the program 2,6 would set register B to 1.
    If register A contains 10, the program 5,0,5,1,5,4 would output 0,1,2.
    If register A contains 2024, the program 0,1,5,4,3,0 would output 4,2,5,6,7,7,7,7,3,1,0 and leave 0 in register A.
    If register B contains 29, the program 1,7 would set register B to 26.
    If register B contains 2024 and register C contains 43690, the program 4,0 would set register B to 44354.

The Historians' strange device has finished initializing its debugger and is displaying some information about the program it is trying to run (your puzzle input). For example:

Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0

Your first task is to determine what the program is trying to output. To do this, initialize the registers to the given values, then run the given program, collecting any output produced by out instructions. (Always join the values produced by out instructions with commas.) After the above program halts, its final output will be 4,6,3,5,6,3,5,2,1,0.

Using the information provided by the debugger, initialize the registers to the given values, then run the program. Once it halts, what do you get if you use commas to join the values it output into a single string?

--- Part Two ---

Digging deeper in the device's manual, you discover the problem: this program is supposed to output another copy of the program! Unfortunately, the value in register A seems to have been corrupted. You'll need to find a new value to which you can initialize register A so that the program's output instructions produce an exact copy of the program itself.

For example:

Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0

This program outputs a copy of itself if register A is instead initialized to 117440. (The original initial value of register A, 2024, is ignored.)

What is the lowest positive initial value for register A that causes the program to output a copy of itself? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, EnumCount, EnumIter, EnumVariantNames)]
enum Register {
    A,
    B,
    C,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct RegistersState([u32; Register::COUNT]);

impl Parse for RegistersState {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut registers_state: Self = Self(Default::default());
        let mut register_iter: RegisterIter = Register::iter();

        let input: &str = many_m_n(Register::COUNT, Register::COUNT, |input: &'i str| {
            let register: Register = register_iter.next().unwrap();
            let (input, register_state): (&str, u32) = map(
                tuple((
                    tag("Register "),
                    tag(Register::VARIANTS[register as usize]),
                    tag(": "),
                    parse_integer,
                    line_ending,
                )),
                |(_, _, _, register_state, _)| register_state,
            )(input)?;

            registers_state.0[register as usize] = register_state;

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, registers_state))
    }
}

#[derive(Clone, Copy)]
enum ComboOperand {
    Literal(u8),
    Register(Register),
}

impl ComboOperand {
    fn new(value: u8) -> Self {
        assert!(Solution::is_value_valid(value));

        match value {
            0_u8..=3_u8 => Self::Literal(value),
            4_u8 => Self::Register(Register::A),
            5_u8 => Self::Register(Register::B),
            6_u8 => Self::Register(Register::C),
            _ => unreachable!(),
        }
    }

    fn value(self, state: &State) -> u32 {
        match self {
            Self::Literal(literal_operand) => literal_operand as u32,
            Self::Register(register) => state.registers_state.0[register as usize],
        }
    }
}

#[derive(Clone, Copy)]
enum Instruction {
    Div(ComboOperand, Register),
    Bxl(u8),
    Bst(ComboOperand),
    Jnz(u8),
    Bxc,
    Out(ComboOperand),
}

impl Instruction {
    fn new(op_code_value: u8, operand_value: u8) -> Self {
        assert!(Solution::is_value_valid(op_code_value));
        assert!(Solution::is_value_valid(operand_value));

        match op_code_value {
            0_u8 | 6_u8 | 7_u8 => Self::Div(
                ComboOperand::new(operand_value),
                match op_code_value {
                    0_u8 => Register::A,
                    6_u8 => Register::B,
                    7_u8 => Register::C,
                    _ => unreachable!(),
                },
            ),
            1_u8 => Self::Bxl(operand_value),
            2_u8 => Self::Bst(ComboOperand::new(operand_value)),
            3_u8 => Self::Jnz(operand_value),
            4_u8 => Self::Bxc,
            5_u8 => Self::Out(ComboOperand::new(operand_value)),
            _ => unreachable!(),
        }
    }

    fn execute(self, state: &mut State) {
        let mut instruction_pointer_jump: Option<usize> = None;

        match self {
            Self::Div(combo_operand, register) => {
                let value: u32 = state.registers_state.0[Register::A as usize]
                    / (1_u32 << combo_operand.value(state));

                state.registers_state.0[register as usize] = value;
            }
            Self::Bxl(literal_operand) => {
                state.registers_state.0[Register::B as usize] ^= literal_operand as u32;
            }
            Self::Bst(combo_operand) => {
                let value: u32 = combo_operand.value(state) % Solution::VALUE_COUNT as u32;

                state.registers_state.0[Register::B as usize] = value;
            }
            Self::Jnz(literal_operand) => {
                instruction_pointer_jump = (state.registers_state.0[Register::A as usize] != 0_u32)
                    .then_some(literal_operand as usize);
            }
            Self::Bxc => {
                let value: u32 = state.registers_state.0[Register::C as usize];

                state.registers_state.0[Register::B as usize] ^= value;
            }
            Self::Out(combo_operand) => {
                let value: u32 = combo_operand.value(state);

                state
                    .output
                    .push((value % Solution::VALUE_COUNT as u32) as u8);
            }
        }

        state.instruction_pointer =
            instruction_pointer_jump.unwrap_or(state.instruction_pointer + 2_usize);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct State {
    registers_state: RegistersState,
    instruction_pointer: usize,
    output: Vec<u8>,
}

impl State {
    fn execute(&mut self, program: &[u8]) {
        while let Some(instruction_values) =
            program.get(self.instruction_pointer..self.instruction_pointer + 2_usize)
        {
            let instruction: Instruction = Instruction::new(
                *instruction_values.first().unwrap(),
                *instruction_values.last().unwrap(),
            );

            instruction.execute(self);
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    registers_state: RegistersState,
    program: Vec<u8>,
}

impl Solution {
    const BITS: u32 = 3_u32;
    const VALUE_COUNT: u8 = 1_u8 << Self::BITS;

    fn is_value_valid(value: u8) -> bool {
        value < Self::VALUE_COUNT
    }

    fn try_find_minimal_initial_a_internal(&self, program: &[u8], a: u64) -> Option<u64> {
        if let Some(last_program_value_index) = program.len().checked_sub(1_usize) {
            let program_value: u8 = program[last_program_value_index];
            let target: u8 = program_value ^ self.program[11_usize];
            let program: &[u8] = &program[..last_program_value_index];

            for next in 0_u8..Self::VALUE_COUNT {
                let a: u64 = (a << Self::BITS) | next as u64;
                let b: u8 = next ^ self.program[3_usize];

                if (b ^ ((a >> b) % Self::VALUE_COUNT as u64) as u8) == target {
                    if let Some(minimal_initial_a) =
                        self.try_find_minimal_initial_a_internal(program, a)
                    {
                        return Some(minimal_initial_a);
                    }
                }
            }

            None
        } else {
            Some(a)
        }
    }

    fn execute(&self) -> State {
        let mut state: State = State {
            registers_state: self.registers_state,
            instruction_pointer: 0_usize,
            output: Vec::new(),
        };

        state.execute(&self.program);

        state
    }

    fn execution_output_string(&self) -> String {
        self.execute()
            .output
            .into_iter()
            .enumerate()
            .flat_map(|(index, value)| {
                (index > 0_usize)
                    .then_some(',')
                    .into_iter()
                    .chain([(value + b'0') as char])
            })
            .collect()
    }

    fn try_find_minimal_initial_a(&self) -> Option<u64> {
        (self.program.len() == 16_usize
            && self.program[2_usize] == 1_u8
            && self.program[10_usize] == 1_u8)
            .then(|| self.try_find_minimal_initial_a_internal(&self.program, 0_u64))
            .flatten()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                RegistersState::parse,
                line_ending,
                tag("Program: "),
                verify(
                    separated_list1(tag(","), parse_integer),
                    |program: &Vec<u8>| program.iter().all(|&b| Self::is_value_valid(b)),
                ),
            )),
            |(registers_state, _, _, program)| Self {
                registers_state,
                program,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I feel like we'll recursively run on the output...
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.execution_output_string());
    }

    /// This is another one that's difficult to explain without revealing too much about the user
    /// input.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_find_minimal_initial_a());
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
        "\
        Register A: 0\n\
        Register B: 0\n\
        Register C: 9\n\
        \n\
        Program: 2,6\n",
        "\
        Register A: 10\n\
        Register B: 0\n\
        Register C: 0\n\
        \n\
        Program: 5,0,5,1,5,4\n",
        "\
        Register A: 2024\n\
        Register B: 0\n\
        Register C: 0\n\
        \n\
        Program: 0,1,5,4,3,0\n",
        "\
        Register A: 0\n\
        Register B: 29\n\
        Register C: 0\n\
        \n\
        Program: 1,7\n",
        "\
        Register A: 0\n\
        Register B: 2024\n\
        Register C: 43690\n\
        \n\
        Program: 4,0\n",
        "\
        Register A: 729\n\
        Register B: 0\n\
        Register C: 0\n\
        \n\
        Program: 0,1,5,4,3,0\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    registers_state: RegistersState([0_u32, 0_u32, 9_u32]),
                    program: vec![2_u8, 6_u8],
                },
                Solution {
                    registers_state: RegistersState([10_u32, 0_u32, 0_u32]),
                    program: vec![5_u8, 0_u8, 5_u8, 1_u8, 5_u8, 4_u8],
                },
                Solution {
                    registers_state: RegistersState([2024_u32, 0_u32, 0_u32]),
                    program: vec![0_u8, 1_u8, 5_u8, 4_u8, 3_u8, 0_u8],
                },
                Solution {
                    registers_state: RegistersState([0_u32, 29_u32, 0_u32]),
                    program: vec![1_u8, 7_u8],
                },
                Solution {
                    registers_state: RegistersState([0_u32, 2024_u32, 43690_u32]),
                    program: vec![4_u8, 0_u8],
                },
                Solution {
                    registers_state: RegistersState([729_u32, 0_u32, 0_u32]),
                    program: vec![0_u8, 1_u8, 5_u8, 4_u8, 3_u8, 0_u8],
                },
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
    fn test_execute() {
        for (index, state) in [
            // If register C contains 9, the program 2,6 would set register B to 1.
            State {
                registers_state: RegistersState([0_u32, 1_u32, 9_u32]),
                instruction_pointer: 2_usize,
                output: vec![],
            },
            // If register A contains 10, the program 5,0,5,1,5,4 would output 0,1,2.
            State {
                registers_state: RegistersState([10_u32, 0_u32, 0_u32]),
                instruction_pointer: 6_usize,
                output: vec![0_u8, 1_u8, 2_u8],
            },
            // If register A contains 2024, the program 0,1,5,4,3,0 would output
            // 4,2,5,6,7,7,7,7,3,1,0 and leave 0 in register A.
            State {
                registers_state: RegistersState([0_u32, 0_u32, 0_u32]),
                instruction_pointer: 6_usize,
                output: vec![
                    4_u8, 2_u8, 5_u8, 6_u8, 7_u8, 7_u8, 7_u8, 7_u8, 3_u8, 1_u8, 0_u8,
                ],
            },
            // If register B contains 29, the program 1,7 would set register B to 26.
            State {
                registers_state: RegistersState([0_u32, 26_u32, 0_u32]),
                instruction_pointer: 2_usize,
                output: vec![],
            },
            // If register B contains 2024 and register C contains 43690, the program 4,0 would set
            // register B to 44354.
            State {
                registers_state: RegistersState([0_u32, 44354_u32, 43690_u32]),
                instruction_pointer: 2_usize,
                output: vec![],
            },
            // After the above program halts, its final output will be 4,6,3,5,6,3,5,2,1,0.
            State {
                registers_state: RegistersState([0_u32, 0_u32, 0_u32]),
                instruction_pointer: 6_usize,
                output: vec![4_u8, 6_u8, 3_u8, 5_u8, 6_u8, 3_u8, 5_u8, 2_u8, 1_u8, 0_u8],
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).execute(), state);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
