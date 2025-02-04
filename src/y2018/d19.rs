use {
    crate::{
        y2018::d16::{Instruction, RegisterRaw, Registers as D16Registers},
        *,
    },
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        multi::separated_list0, sequence::tuple, Err, IResult,
    },
    num::{NumCast, One},
    std::{
        fmt::{Error as FmtError, Write},
        fs::{create_dir_all, write},
    },
    y2018::d16::OpCode,
};

/* --- Day 19: Go With The Flow ---

With the Elves well on their way constructing the North Pole base, you turn your attention back to understanding the inner workings of programming the device.

You can't help but notice that the device's opcodes don't contain any flow control like jump instructions. The device's manual goes on to explain:

"In programs where flow control is required, the instruction pointer can be bound to a register so that it can be manipulated directly. This way, setr/seti can function as absolute jumps, addr/addi can function as relative jumps, and other opcodes can cause truly fascinating effects."

This mechanism is achieved through a declaration like #ip 1, which would modify register 1 so that accesses to it let the program indirectly access the instruction pointer itself. To compensate for this kind of binding, there are now six registers (numbered 0 through 5); the five not bound to the instruction pointer behave as normal. Otherwise, the same rules apply as the last time you worked with this device.

When the instruction pointer is bound to a register, its value is written to that register just before each instruction is executed, and the value of that register is written back to the instruction pointer immediately after each instruction finishes execution. Afterward, move to the next instruction by adding one to the instruction pointer, even if the value in the instruction pointer was just updated by an instruction. (Because of this, instructions must effectively set the instruction pointer to the instruction before the one they want executed next.)

The instruction pointer is 0 during the first instruction, 1 during the second, and so on. If the instruction pointer ever causes the device to attempt to load an instruction outside the instructions defined in the program, the program instead immediately halts. The instruction pointer starts at 0.

It turns out that this new information is already proving useful: the CPU in the device is not very powerful, and a background process is occupying most of its time. You dump the background process' declarations and instructions to a file (your puzzle input), making sure to use the names of the opcodes rather than the numbers.

For example, suppose you have the following program:

#ip 0
seti 5 0 1
seti 6 0 2
addi 0 1 0
addr 1 2 3
setr 1 0 0
seti 8 0 4
seti 9 0 5

When executed, the following instructions are executed. Each line contains the value of the instruction pointer at the time the instruction started, the values of the six registers before executing the instructions (in square brackets), the instruction itself, and the values of the six registers after executing the instruction (also in square brackets).

ip=0 [0, 0, 0, 0, 0, 0] seti 5 0 1 [0, 5, 0, 0, 0, 0]
ip=1 [1, 5, 0, 0, 0, 0] seti 6 0 2 [1, 5, 6, 0, 0, 0]
ip=2 [2, 5, 6, 0, 0, 0] addi 0 1 0 [3, 5, 6, 0, 0, 0]
ip=4 [4, 5, 6, 0, 0, 0] setr 1 0 0 [5, 5, 6, 0, 0, 0]
ip=6 [6, 5, 6, 0, 0, 0] seti 9 0 5 [6, 5, 6, 0, 0, 9]

In detail, when running this program, the following events occur:

    The first line (#ip 0) indicates that the instruction pointer should be bound to register 0 in this program. This is not an instruction, and so the value of the instruction pointer does not change during the processing of this line.
    The instruction pointer contains 0, and so the first instruction is executed (seti 5 0 1). It updates register 0 to the current instruction pointer value (0), sets register 1 to 5, sets the instruction pointer to the value of register 0 (which has no effect, as the instruction did not modify register 0), and then adds one to the instruction pointer.
    The instruction pointer contains 1, and so the second instruction, seti 6 0 2, is executed. This is very similar to the instruction before it: 6 is stored in register 2, and the instruction pointer is left with the value 2.
    The instruction pointer is 2, which points at the instruction addi 0 1 0. This is like a relative jump: the value of the instruction pointer, 2, is loaded into register 0. Then, addi finds the result of adding the value in register 0 and the value 1, storing the result, 3, back in register 0. Register 0 is then copied back to the instruction pointer, which will cause it to end up 1 larger than it would have otherwise and skip the next instruction (addr 1 2 3) entirely. Finally, 1 is added to the instruction pointer.
    The instruction pointer is 4, so the instruction setr 1 0 0 is run. This is like an absolute jump: it copies the value contained in register 1, 5, into register 0, which causes it to end up in the instruction pointer. The instruction pointer is then incremented, leaving it at 6.
    The instruction pointer is 6, so the instruction seti 9 0 5 stores 9 into register 5. The instruction pointer is incremented, causing it to point outside the program, and so the program ends.

What value is left in register 0 when the background process halts?

--- Part Two ---

A new background process immediately spins up in its place. It appears identical, but on closer inspection, you notice that this time, register 0 started with the value 1.

What value is left in register 0 when this new background process halts? */

const REGISTERS_LEN: usize = 6_usize;

pub type Registers = D16Registers<REGISTERS_LEN>;

struct Constants {
    instruction_20_g_eq_g_mul_b: RegisterRaw,
    instruction_21_h_eq_h_add_b: RegisterRaw,
    instruction_23_h_eq_h_add_b: RegisterRaw,
    instruction_31_h_eq_h_mul_b: RegisterRaw,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Program {
    instruction_pointer: RegisterRaw,
    instructions: Vec<Instruction>,
}

impl Program {
    fn as_result_string<T, E: ToString>(result: Result<T, E>) -> Result<T, String> {
        result.map_err(|e| e.to_string())
    }

    fn try_instruction_pointer_register_index(&self) -> Option<usize> {
        Registers::try_register_index(self.instruction_pointer).ok()
    }

    fn try_get_instruction<'i>(&'i self, registers: &Registers) -> Option<&'i Instruction> {
        self.try_instruction_pointer_register_index()
            .and_then(|register_index| registers.0.get(register_index))
            .and_then(|&instruction_index| <usize as NumCast>::from(instruction_index))
            .and_then(|instruction_index| self.instructions.get(instruction_index))
    }

    fn try_execute_instruction(
        &self,
        instruction: &Instruction,
        registers: &mut Registers,
    ) -> Option<()> {
        instruction.try_execute(registers).ok().map(|_| {
            registers.0[self.try_instruction_pointer_register_index().unwrap()] +=
                RegisterRaw::one();
        })
    }

    pub fn try_run(&self, register_0: RegisterRaw) -> Option<Registers> {
        let mut registers: Registers = Registers::default();

        registers.0[0_usize] = register_0;

        while let Some(instruction) = self.try_get_instruction(&registers) {
            self.try_execute_instruction(instruction, &mut registers)?;
            dbg!(registers);

            std::hint::black_box(())
        }

        Some(registers)
    }

    pub fn try_register_0_after_run(&self, register_0: RegisterRaw) -> Option<RegisterRaw> {
        self.try_run(register_0)
            .map(|registers| registers.0[0_usize])
    }

    pub fn instruction_pointer(&self) -> RegisterRaw {
        self.instruction_pointer
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// If both `a` and `b` are `None`, retrieving `a` is preferred.
    pub fn try_get_constant(
        &self,
        instruction_index: usize,
        op_code: OpCode,
        a: Option<RegisterRaw>,
        b: Option<RegisterRaw>,
        c: RegisterRaw,
    ) -> Option<RegisterRaw> {
        (a.is_none() || b.is_none())
            .then(|| a.or(b))
            .zip(self.instructions.get(instruction_index))
            .and_then(|(a_or_b, instruction)| {
                (instruction.op_code == op_code
                    && (a.is_none() || instruction.a == a_or_b.unwrap())
                    && (b.is_none() || instruction.b == a_or_b.unwrap())
                    && instruction.c == c)
                    .then(|| {
                        if a.is_none() {
                            instruction.a
                        } else {
                            instruction.b
                        }
                    })
            })
    }

    pub fn try_print_simplified_to_file(&self, module_path: &str) -> Result<(), String> {
        let mut register_names: Vec<char> = vec!['d', 'e', 'f', 'g', 'h'];

        register_names.insert(
            self.try_instruction_pointer_register_index()
                .ok_or_else(|| {
                    format!(
                        "{} was not a valid register index",
                        self.instruction_pointer
                    )
                })?,
            'i',
        );

        let mut simplified: String = String::new();

        writeln!(&mut simplified, "// {register_names:?}").map_err(|e| e.to_string())?;

        let mut instruction_string: String = String::new();

        let max_instruction_string_len: usize =
            Self::as_result_string::<usize, FmtError>(self.instructions.iter().try_fold(
                0_usize,
                |max_instruction_string_len, instruction| {
                    instruction_string.clear();

                    instruction.print_simplified(&register_names, &mut instruction_string)?;

                    Ok(max_instruction_string_len.max(instruction_string.len()))
                },
            ))?;
        let instruction_padded_len: usize =
            ((max_instruction_string_len / 4_usize) + 1_usize) * 4_usize;

        for (index, instruction) in self.instructions.iter().enumerate() {
            instruction_string.clear();

            Self::as_result_string(
                instruction.print_simplified(&register_names, &mut instruction_string),
            )?;
            Self::as_result_string(writeln!(
                &mut simplified,
                "{0: <1$}// {2:>02}: {3} {4} {5} {6}",
                instruction_string,
                instruction_padded_len,
                index,
                instruction.op_code.tag_str(),
                instruction.a,
                instruction.b,
                instruction.c,
            ))?;
        }

        println!("{simplified}");

        let args: Args = Args::parse(module_path).map_err(|e| format!("{e:?}"))?.1;
        let path: String = format!("output/y{}/d{}_simplified.txt", args.year, args.day);

        create_dir_all(&path[..path.rfind('/').unwrap()]).map_err(|e| e.to_string())?;

        write(path, simplified.as_bytes()).map_err(|e| e.to_string())
    }
}

impl Parse for Program {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("#ip "),
                parse_integer,
                line_ending,
                separated_list0(line_ending, Instruction::parse),
            )),
            |(_, instruction_pointer, _, instructions)| Self {
                instruction_pointer,
                instructions,
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Program);

impl Solution {
    const Q1_REGISTER_0: RegisterRaw = 0 as RegisterRaw;
    const Q2_REGISTER_0: RegisterRaw = 1 as RegisterRaw;

    fn try_run(&self, register_0: RegisterRaw) -> Option<Registers> {
        self.0.try_run(register_0)
    }

    fn try_register_0_after_run(&self, register_0: RegisterRaw) -> Option<RegisterRaw> {
        self.0.try_register_0_after_run(register_0)
    }

    fn try_get_constant(
        &self,
        instruction_index: usize,
        op_code: OpCode,
        a: RegisterRaw,
        c: RegisterRaw,
    ) -> Option<RegisterRaw> {
        self.0
            .try_get_constant(instruction_index, op_code, Some(a), None, c)
    }

    fn try_get_constants(&self) -> Option<Constants> {
        const G: RegisterRaw = 3 as RegisterRaw;
        const H: RegisterRaw = 4 as RegisterRaw;
        const I: RegisterRaw = 5 as RegisterRaw;

        // This function could be more rigorous, but doing so would expose more details about the
        // user-specific input than I'm comfortable sharing.
        (self.0.instructions.len() == 36_usize && self.0.instruction_pointer == I)
            .then_some(())
            .and_then(|_| {
                self.try_get_constant(20_usize, OpCode::MulI, G, G)
                    .zip(self.try_get_constant(21_usize, OpCode::AddI, H, H))
                    .zip(
                        self.try_get_constant(23_usize, OpCode::AddI, H, H)
                            .zip(self.try_get_constant(31_usize, OpCode::MulI, H, H)),
                    )
                    .map(
                        |(
                            (instruction_20_g_eq_g_mul_b, instruction_21_h_eq_h_add_b),
                            (instruction_23_h_eq_h_add_b, instruction_31_h_eq_h_mul_b),
                        )| Constants {
                            instruction_20_g_eq_g_mul_b,
                            instruction_21_h_eq_h_add_b,
                            instruction_23_h_eq_h_add_b,
                            instruction_31_h_eq_h_mul_b,
                        },
                    )
            })
    }

    fn try_run_simplified(&self, register_0: RegisterRaw) -> Option<Registers> {
        matches!(register_0, 0 | 1)
            .then(|| self.try_get_constants())
            .flatten()
            .map(|constants| {
                const INSTRUCTION_16_I_EQ_I_MUL_I: RegisterRaw = 16 as RegisterRaw;
                const INSTRUCTION_17_G_EQ_G_ADD_B: RegisterRaw = 2 as RegisterRaw;
                const INSTRUCTION_19_G_EQ_I_MUL_G: RegisterRaw = 19 as RegisterRaw;
                const INSTRUCTION_22_H_EQ_H_MUL_I: RegisterRaw = 22 as RegisterRaw;
                const INSTRUCTION_27_H_EQ_I: RegisterRaw = 27 as RegisterRaw;
                const INSTRUCTION_28_H_EQ_H_MUL_I: RegisterRaw = 28 as RegisterRaw;
                const INSTRUCTION_29_H_EQ_I_ADD_H: RegisterRaw = 29 as RegisterRaw;
                const INSTRUCTION_30_H_EQ_I_MUL_H: RegisterRaw = 30 as RegisterRaw;
                const INSTRUCTION_32_H_EQ_H_MUL_I: RegisterRaw = 32 as RegisterRaw;
                let g = INSTRUCTION_17_G_EQ_G_ADD_B
                    * INSTRUCTION_17_G_EQ_G_ADD_B
                    * INSTRUCTION_19_G_EQ_I_MUL_G
                    * constants.instruction_20_g_eq_g_mul_b
                    + (constants.instruction_21_h_eq_h_add_b * INSTRUCTION_22_H_EQ_H_MUL_I
                        + constants.instruction_23_h_eq_h_add_b)
                    + register_0
                        * (INSTRUCTION_27_H_EQ_I * INSTRUCTION_28_H_EQ_H_MUL_I
                            + INSTRUCTION_29_H_EQ_I_ADD_H)
                        * INSTRUCTION_30_H_EQ_I_MUL_H
                        * constants.instruction_31_h_eq_h_mul_b
                        * INSTRUCTION_32_H_EQ_H_MUL_I;

                // The instructions perform this computation:

                // f = 1 as RegisterRaw;
                // while {
                //     e = 1 as RegisterRaw;
                //     while {
                //         if f * e == g as RegisterRaw {
                //             d += f;
                //         }
                //         e += 1 as RegisterRaw;
                //         e <= g
                //     } {}
                //     f += 1 as RegisterRaw;
                //     f <= g
                // } {}

                // This is an algorithm for summing the factors of a number. Unfortunately, running
                // this code as is is wildly inefficient for `g` when `initial_d` is true. Instead,
                // let's compute that number a different way.
                let d = iter_factors(g as usize)
                    .map(|factor| factor as RegisterRaw)
                    .sum();

                D16Registers([
                    d,
                    g + 1 as RegisterRaw,
                    g + 1 as RegisterRaw,
                    g,
                    1 as RegisterRaw,
                    INSTRUCTION_16_I_EQ_I_MUL_I * INSTRUCTION_16_I_EQ_I_MUL_I + 1 as RegisterRaw,
                ])
            })
    }

    fn try_register_0_after_run_simplified(&self, register_0: RegisterRaw) -> Option<RegisterRaw> {
        self.try_run_simplified(register_0)
            .map(|registers| registers.0[0_usize])
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Program::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Please don't make me reverse engineer this program.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.try_run(Self::Q1_REGISTER_0));
            dbg!(self.try_run_simplified(Self::Q1_REGISTER_0));

            if let Err(e) = self.0.try_print_simplified_to_file(module_path!()) {
                eprintln!("{e}");
            }
        } else {
            dbg!(self.try_register_0_after_run(Self::Q1_REGISTER_0));
        }
    }

    /// You did it anyway.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.try_run_simplified(Self::Q2_REGISTER_0));
        } else {
            dbg!(self.try_register_0_after_run_simplified(Self::Q2_REGISTER_0));
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
    use {super::*, crate::y2018::d16::OpCode, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        #ip 0\n\
        seti 5 0 1\n\
        seti 6 0 2\n\
        addi 0 1 0\n\
        addr 1 2 3\n\
        setr 1 0 0\n\
        seti 8 0 4\n\
        seti 9 0 5\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(Program {
                instruction_pointer: 0 as RegisterRaw,
                instructions: vec![
                    Instruction {
                        op_code: OpCode::SetI,
                        a: 5 as RegisterRaw,
                        b: 0 as RegisterRaw,
                        c: 1 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::SetI,
                        a: 6 as RegisterRaw,
                        b: 0 as RegisterRaw,
                        c: 2 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::AddI,
                        a: 0 as RegisterRaw,
                        b: 1 as RegisterRaw,
                        c: 0 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::AddR,
                        a: 1 as RegisterRaw,
                        b: 2 as RegisterRaw,
                        c: 3 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::SetR,
                        a: 1 as RegisterRaw,
                        b: 0 as RegisterRaw,
                        c: 0 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::SetI,
                        a: 8 as RegisterRaw,
                        b: 0 as RegisterRaw,
                        c: 4 as RegisterRaw,
                    },
                    Instruction {
                        op_code: OpCode::SetI,
                        a: 9 as RegisterRaw,
                        b: 0 as RegisterRaw,
                        c: 5 as RegisterRaw,
                    },
                ],
            })]
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
    fn test_try_execute_instruction() {
        for (index, registers_list) in [vec![
            D16Registers([
                1 as RegisterRaw,
                5 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
            ]),
            D16Registers([
                2 as RegisterRaw,
                5 as RegisterRaw,
                6 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
            ]),
            D16Registers([
                4 as RegisterRaw,
                5 as RegisterRaw,
                6 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
            ]),
            D16Registers([
                6 as RegisterRaw,
                5 as RegisterRaw,
                6 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
                0 as RegisterRaw,
            ]),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            let mut curr_registers: Registers = Registers::default();

            for registers in registers_list {
                solution
                    .0
                    .try_execute_instruction(
                        solution.0.try_get_instruction(&curr_registers).unwrap(),
                        &mut curr_registers,
                    )
                    .unwrap();

                assert_eq!(curr_registers, registers);
            }
        }
    }

    #[test]
    fn test_try_register_0_after_run() {
        for (index, register_0_after_run) in [Some(7 as RegisterRaw)].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_register_0_after_run(Solution::Q1_REGISTER_0),
                register_0_after_run
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;
        let mut args: Args = Args::parse(module_path!()).unwrap().1;

        args.question_args.verbose = true;

        Solution::both(&args);
    }
}
