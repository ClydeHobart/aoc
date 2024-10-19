use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    static_assertions::const_assert_eq,
    std::{
        mem::{align_of, size_of},
        slice::{from_raw_parts, from_raw_parts_mut},
    },
    strum::EnumCount,
};

/* --- Day 23: Coprocessor Conflagration ---

You decide to head directly to the CPU and fix the printer from there. As you get close, you find an experimental coprocessor doing so much work that the local programs are afraid it will halt and catch fire. This would cause serious issues for the rest of the computer, so you head in and see what you can do.

The code it's running seems to be a variant of the kind you saw recently on that tablet. The general functionality seems very similar, but some of the instructions are different:

    set X Y sets register X to the value of Y.
    sub X Y decreases register X by the value of Y.
    mul X Y sets register X to the result of multiplying the value contained in register X by the value of Y.
    jnz X Y jumps with an offset of the value of Y, but only if the value of X is not zero. (An offset of 2 skips the next instruction, an offset of -1 jumps to the previous instruction, and so on.)

    Only the instructions listed above are used. The eight registers here, named a through h, all start at 0.

The coprocessor is currently set to some kind of debug mode, which allows for testing, but prevents it from doing any meaningful work.

If you run the program (your puzzle input), how many times is the mul instruction invoked?

--- Part Two ---

Now, it's time to fix the problem.

The debug mode switch is wired directly to register a. You flip the switch, which makes register a now start at 1 when the program is executed.

Immediately, the coprocessor begins to overheat. Whoever wrote this program obviously didn't choose a very efficient implementation. You'll need to optimize the program if it has any hope of completing before Santa needs that printer working.

The coprocessor's ultimate goal is to determine the final value left in register h once the program completes. Technically, if it had that... it wouldn't even need to run the program.

After setting register a to 1, if the program were to run to completion, what value would be left in register h? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, EnumCount, PartialEq)]
enum Register {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

impl Register {
    const BYTES: [u8; Self::COUNT] = Self::bytes();

    const fn bytes() -> [u8; Self::COUNT] {
        let mut bytes: [u8; Self::COUNT] = [0_u8; Self::COUNT];
        let mut index: usize = 0_usize;

        while index < Self::COUNT {
            bytes[index] = b'a' + index as u8;
            index += 1_usize;
        }

        bytes
    }

    fn alt_branch<'i>(self) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        map(
            tag(&Self::BYTES[self as usize..self as usize + 1_usize]),
            move |_| self,
        )
    }
}

impl Parse for Register {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::A.alt_branch(),
            Self::B.alt_branch(),
            Self::C.alt_branch(),
            Self::D.alt_branch(),
            Self::E.alt_branch(),
            Self::F.alt_branch(),
            Self::G.alt_branch(),
            Self::H.alt_branch(),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Value {
    Register(Register),
    Constant(i32),
}

impl Parse for Value {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Register::parse, Self::Register),
            map(parse_integer, Self::Constant),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Instruction {
    Set { x: Register, y: Value },
    Sub { x: Register, y: Value },
    Mul { x: Register, y: Value },
    JNZ { x: Value, y: Value },
}

impl Instruction {
    fn try_get_x_register(&self) -> Option<Register> {
        match self {
            Self::Set { x, .. } => Some(*x),
            Self::Sub { x, .. } => Some(*x),
            Self::Mul { x, .. } => Some(*x),
            Self::JNZ {
                x: Value::Register(x),
                ..
            } => Some(*x),
            _ => None,
        }
    }

    fn try_get_y_constant(&self) -> Option<i32> {
        match self {
            Instruction::Set {
                y: Value::Constant(y),
                ..
            } => Some(*y),
            Instruction::Sub {
                y: Value::Constant(y),
                ..
            } => Some(*y),
            Instruction::Mul {
                y: Value::Constant(y),
                ..
            } => Some(*y),
            Instruction::JNZ {
                y: Value::Constant(y),
                ..
            } => Some(*y),
            _ => None,
        }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                tuple((tag("set "), Register::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::Set { x, y },
            ),
            map(
                tuple((tag("sub "), Register::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::Sub { x, y },
            ),
            map(
                tuple((tag("mul "), Register::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::Mul { x, y },
            ),
            map(
                tuple((tag("jnz "), Value::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::JNZ { x, y },
            ),
        ))(input)
    }
}

#[derive(Default)]
struct Constants {
    b_0: i32,
    b_4: i32,
    b_5: i32,
    c_7: i32,
    b_30: i32,
}

#[allow(dead_code)]
enum Procedure {
    Label00,
    Label01,
    Label02,
    Label03,
    Label04,
    Label05,
    Label06,
    Label07,
    Label08,
    Label09,
    Label10,
    Label11,
    Label12,
}

impl Procedure {}

#[repr(C)]
#[derive(Default)]
struct Registers {
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    f: i32,
    g: i32,
    h: i32,
}

const_assert_eq!(size_of::<Registers>(), size_of::<[i32; Register::COUNT]>());
const_assert_eq!(
    align_of::<Registers>(),
    align_of::<[i32; Register::COUNT]>()
);

impl Registers {
    fn len() -> usize {
        Register::COUNT
    }

    fn data(&self) -> *const i32 {
        self as *const Self as *const i32
    }

    fn as_slice(&self) -> &[i32] {
        // SAFETY: the const asserts before the impl guarantee this is safe.
        unsafe { from_raw_parts(self.data(), Self::len()) }
    }

    fn data_mut(&mut self) -> *mut i32 {
        self as *mut Self as *mut i32
    }

    fn as_slice_mut(&mut self) -> &mut [i32] {
        // SAFETY: the const asserts before the impl guarantee this is safe.
        unsafe { from_raw_parts_mut(self.data_mut(), Self::len()) }
    }

    #[allow(dead_code)]
    fn execute_0(&mut self, constants: &Constants, procedure: Procedure) -> Option<Procedure> {
        Some(match procedure {
            Procedure::Label00 => {
                self.b = constants.b_0;
                self.c = self.b;

                if self.a != 0_i32 {
                    Procedure::Label01
                } else {
                    Procedure::Label02
                }
            }
            Procedure::Label01 => {
                self.b *= constants.b_4;
                self.b -= constants.b_5;
                self.c = self.b;
                self.c -= constants.c_7;

                Procedure::Label02
            }
            Procedure::Label02 => {
                self.f = 1_i32;
                self.d = 2_i32;

                Procedure::Label03
            }
            Procedure::Label03 => {
                self.e = 2_i32;

                Procedure::Label04
            }
            Procedure::Label04 => {
                self.g = self.d;
                self.g *= self.e;
                self.g -= self.b;

                if self.g != 0_i32 {
                    Procedure::Label06
                } else {
                    Procedure::Label05
                }
            }
            Procedure::Label05 => {
                self.f = 0_i32;

                Procedure::Label06
            }
            Procedure::Label06 => {
                self.e -= -1_i32;
                self.g = self.e;
                self.g -= self.b;

                if self.g != 0_i32 {
                    Procedure::Label04
                } else {
                    Procedure::Label07
                }
            }
            Procedure::Label07 => {
                self.d -= -1_i32;
                self.g = self.d;
                self.g -= self.b;

                if self.g != 0_i32 {
                    Procedure::Label03
                } else {
                    Procedure::Label08
                }
            }
            Procedure::Label08 => {
                if self.f != 0_i32 {
                    Procedure::Label10
                } else {
                    Procedure::Label09
                }
            }
            Procedure::Label09 => {
                self.h -= -1_i32;

                Procedure::Label10
            }
            Procedure::Label10 => {
                self.g = self.b;
                self.g -= self.c;

                if self.g != 0_i32 {
                    Procedure::Label12
                } else {
                    Procedure::Label11
                }
            }
            Procedure::Label11 => None?,
            Procedure::Label12 => {
                self.b -= constants.b_30;

                Procedure::Label02
            }
        })
    }

    #[allow(dead_code)]
    fn execute_1(&mut self, constants: &Constants) {
        // Label00, Label01
        if self.a != 0_i32 {
            self.b = constants.b_0 * constants.b_4 - constants.b_5;
            self.c = constants.b_0 * constants.b_4 - constants.b_5 - constants.c_7;
        } else {
            self.b = constants.b_0;
            self.c = constants.b_0;
        }

        while {
            // Label02
            self.f = 1_i32;
            self.d = 2_i32;

            while {
                // Label03
                self.e = 2_i32;

                while {
                    // Label04
                    self.g = self.d;
                    self.g *= self.e;
                    self.g -= self.b;

                    if self.g == 0_i32 {
                        // Label05
                        // 106700 / 2 - 2: f = 0
                        self.f = 0_i32;
                    }

                    // Label06
                    self.e -= -1_i32;
                    self.g = self.e;
                    self.g -= self.b;

                    self.g != 0_i32
                } {}

                // Label07
                self.d -= -1_i32;
                self.g = self.d;
                self.g -= self.b;

                self.g != 0_i32
            } {}

            // Label08
            if self.f == 0_i32 {
                // Label09
                self.h -= -1_i32;
            }

            // Label10
            self.g = self.b;
            self.g -= self.c;

            if self.g != 0_i32 {
                // Label12
                self.b -= constants.b_30;

                true
            } else {
                // Label11
                false
            }
        } {}
    }

    fn try_execute_2(&mut self, constants: &Constants) -> Option<()> {
        (constants.b_30 != 0_i32
            && constants.c_7 % constants.b_30 == 0_i32
            && constants.c_7 / constants.b_30 >= 0_i32
            && constants.b_0 * constants.b_4 - constants.b_5 > 0_i32
            && constants.c_7 < 0_i32)
            .then(|| {
                let [b, c]: [i32; 2_usize] = if self.a != 0_i32 {
                    let b: i32 = constants.b_0 * constants.b_4 - constants.b_5;

                    [b, b - constants.c_7]
                } else {
                    [constants.b_0, constants.b_0]
                };

                self.b = c;
                self.c = c;
                self.d = c;
                self.e = c;
                self.f = is_prime(c as u32) as i32;
                self.g = 0_i32;
                self.h = (b..=c)
                    .filter(|b| (*b - c) % constants.b_30 == 0_i32)
                    .map(|b| is_composite(b as u32) as i32)
                    .sum();
            })
    }
}

#[derive(Default)]
struct State {
    registers: Registers,
    instruction_index: isize,
}

impl State {
    fn can_execute(&self, instructions: &[Instruction]) -> bool {
        usize::try_from(self.instruction_index)
            .ok()
            .filter(|instruction_index| *instruction_index < instructions.len())
            .is_some()
    }

    fn evaluate(&self, value: Value) -> i32 {
        match value {
            Value::Register(register) => self.registers.as_slice()[register as usize],
            Value::Constant(constant) => constant,
        }
    }

    fn execute(&mut self, instructions: &[Instruction]) {
        let mut instruction_index_delta: i32 = 1_i32;

        match &instructions[self.instruction_index as usize] {
            Instruction::Set { x, y } => {
                let value: i32 = self.evaluate(*y);

                self.registers.as_slice_mut()[*x as usize] = value;
            }
            Instruction::Sub { x, y } => {
                let value: i32 = self.evaluate(*y);

                self.registers.as_slice_mut()[*x as usize] -= value;
            }
            Instruction::Mul { x, y } => {
                let value: i32 = self.evaluate(*y);

                self.registers.as_slice_mut()[*x as usize] *= value;
            }
            Instruction::JNZ { x, y } => {
                if self.evaluate(*x) != 0_i32 {
                    instruction_index_delta = self.evaluate(*y);
                }
            }
        }

        self.instruction_index += instruction_index_delta as isize;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    fn mul_invocation_count_in_debug_mode(&self) -> usize {
        let mut state: State = State::default();
        let mut mul_invocation_count: usize = 0_usize;

        while state.can_execute(&self.0) {
            mul_invocation_count += matches!(
                self.0[state.instruction_index as usize],
                Instruction::Mul { .. }
            ) as usize;
            state.execute(&self.0);
        }

        mul_invocation_count
    }

    fn try_constants(&self) -> Option<Constants> {
        let mut constants: Constants = Constants::default();

        for (instruction_index, register, constant) in [
            (0_usize, Register::B, &mut constants.b_0),
            (4_usize, Register::B, &mut constants.b_4),
            (5_usize, Register::B, &mut constants.b_5),
            (7_usize, Register::C, &mut constants.c_7),
            (30_usize, Register::B, &mut constants.b_30),
        ] {
            *constant = self
                .0
                .get(instruction_index)
                .map(|instruction| {
                    instruction
                        .try_get_x_register()
                        .zip(instruction.try_get_y_constant())
                })
                .flatten()
                .map(|(x_register, y_constant)| (x_register == register).then_some(y_constant))
                .flatten()?;
        }

        Some(constants)
    }

    fn try_h_after_execution(&self) -> Option<i32> {
        let constants: Constants = self.try_constants()?;

        let mut registers: Registers = Registers::default();

        registers.a = 1_i32;

        registers.try_execute_2(&constants)?;

        Some(registers.h)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(Instruction::parse, opt(line_ending))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I didn't want to re-use my register state from Day 18 because it was more difficult to
    /// debug.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.mul_invocation_count_in_debug_mode());
    }

    /// Took me a while to work through this one. `Registers::execute_0` is my initial reverse
    /// engineering of the instructions, which then has the control flow simplified in
    /// `Registers::execute_1`. In order to actually determine the answer, though, I had to figure
    /// out what this was achieving mathematically. By simplifying the constants used, and watching
    /// the registers in the debugger, I was able to determine that it was iterating through some
    /// numbers with a set step size, and counting how many of those were composite.
    /// `iter_prime_factors` was a utility that I already had, so throwing together `is_prime` and
    /// `is_composite` functions was easy enough.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_h_after_execution());
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
        let args: Args = Args::parse(module_path!()).unwrap().1;

        Solution::both(&args);
    }
}
