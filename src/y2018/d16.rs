use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::{Error as NomError, ErrorKind},
        multi::separated_list0,
        sequence::{delimited, separated_pair, tuple},
        Err, IResult,
    },
    num::{NumCast, Zero},
    std::{
        fmt::{Display, Error as FmtError, Write},
        mem::transmute,
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

/* --- Day 16: Chronal Classification ---

As you see the Elves defend their hot chocolate successfully, you go back to falling through time. This is going to become a problem.

If you're ever going to return to your own time, you need to understand how this device on your wrist works. You have a little while before you reach your next destination, and with a bit of trial and error, you manage to pull up a programming manual on the device's tiny screen.

According to the manual, the device has four registers (numbered 0 through 3) that can be manipulated by instructions containing one of 16 opcodes. The registers start with the value 0.

Every instruction consists of four values: an opcode, two inputs (named A and B), and an output (named C), in that order. The opcode specifies the behavior of the instruction and how the inputs are interpreted. The output, C, is always treated as a register.

In the opcode descriptions below, if something says "value A", it means to take the number given as A literally. (This is also called an "immediate" value.) If something says "register A", it means to use the number given as A to read from (or write to) the register with that number. So, if the opcode addi adds register A and value B, storing the result in register C, and the instruction addi 0 7 3 is encountered, it would add 7 to the value contained by register 0 and store the sum in register 3, never modifying registers 0, 1, or 2 in the process.

Many opcodes are similar except for how they interpret their arguments. The opcodes fall into seven general categories:

Addition:

    addr (add register) stores into register C the result of adding register A and register B.
    addi (add immediate) stores into register C the result of adding register A and value B.

Multiplication:

    mulr (multiply register) stores into register C the result of multiplying register A and register B.
    muli (multiply immediate) stores into register C the result of multiplying register A and value B.

Bitwise AND:

    banr (bitwise AND register) stores into register C the result of the bitwise AND of register A and register B.
    bani (bitwise AND immediate) stores into register C the result of the bitwise AND of register A and value B.

Bitwise OR:

    borr (bitwise OR register) stores into register C the result of the bitwise OR of register A and register B.
    bori (bitwise OR immediate) stores into register C the result of the bitwise OR of register A and value B.

Assignment:

    setr (set register) copies the contents of register A into register C. (Input B is ignored.)
    seti (set immediate) stores value A into register C. (Input B is ignored.)

Greater-than testing:

    gtir (greater-than immediate/register) sets register C to 1 if value A is greater than register B. Otherwise, register C is set to 0.
    gtri (greater-than register/immediate) sets register C to 1 if register A is greater than value B. Otherwise, register C is set to 0.
    gtrr (greater-than register/register) sets register C to 1 if register A is greater than register B. Otherwise, register C is set to 0.

Equality testing:

    eqir (equal immediate/register) sets register C to 1 if value A is equal to register B. Otherwise, register C is set to 0.
    eqri (equal register/immediate) sets register C to 1 if register A is equal to value B. Otherwise, register C is set to 0.
    eqrr (equal register/register) sets register C to 1 if register A is equal to register B. Otherwise, register C is set to 0.

Unfortunately, while the manual gives the name of each opcode, it doesn't seem to indicate the number. However, you can monitor the CPU to see the contents of the registers before and after instructions are executed to try to work them out. Each opcode has a number from 0 through 15, but the manual doesn't say which is which. For example, suppose you capture the following sample:

Before: [3, 2, 1, 1]
9 2 1 2
After:  [3, 2, 2, 1]

This sample shows the effect of the instruction 9 2 1 2 on the registers. Before the instruction is executed, register 0 has value 3, register 1 has value 2, and registers 2 and 3 have value 1. After the instruction is executed, register 2's value becomes 2.

The instruction itself, 9 2 1 2, means that opcode 9 was executed with A=2, B=1, and C=2. Opcode 9 could be any of the 16 opcodes listed above, but only three of them behave in a way that would cause the result shown in the sample:

    Opcode 9 could be mulr: register 2 (which has a value of 1) times register 1 (which has a value of 2) produces 2, which matches the value stored in the output register, register 2.
    Opcode 9 could be addi: register 2 (which has a value of 1) plus value 1 produces 2, which matches the value stored in the output register, register 2.
    Opcode 9 could be seti: value 2 matches the value stored in the output register, register 2; the number given for B is irrelevant.

None of the other opcodes produce the result captured in the sample. Because of this, the sample above behaves like three opcodes.

You collect many of these samples (the first section of your puzzle input). The manual also includes a small test program (the second section of your puzzle input) - you can ignore it for now.

Ignoring the opcode numbers, how many samples in your puzzle input behave like three or more opcodes?

--- Part Two ---

Using the samples you collected, work out the number of each opcode and execute the test program (the second section of your puzzle input).

What value is contained in register 0 after executing the test program? */

#[allow(dead_code)]
#[derive(Debug)]
pub enum Error {
    InvalidRegisterIndexRegister { register: RegisterRaw },
    InvalidOpCodeUsize { value: usize },
    InvalidOpCodeRegisterRaw { value: RegisterRaw },
    AddOverflow { a: RegisterRaw, b: RegisterRaw },
    MulOverflow { a: RegisterRaw, b: RegisterRaw },
    OpCodeRegisterHasNoPotentialOpCodes { op_code: RegisterRaw },
    OpCodeMappingFinderFailed,
}

pub type RegisterRaw = i64;

const DEFAULT_REGISTERS_LEN: usize = 4_usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Registers<const LEN: usize = DEFAULT_REGISTERS_LEN>(pub [RegisterRaw; LEN]);

impl<const LEN: usize> Default for Registers<LEN> {
    fn default() -> Self {
        Self(LargeArrayDefault::large_array_default())
    }
}

impl<const LEN: usize> Registers<LEN> {
    pub fn try_register_index(register: RegisterRaw) -> Result<usize, Error> {
        <usize as NumCast>::from(register)
            .filter(|&register_index| register_index < LEN)
            .ok_or(Error::InvalidRegisterIndexRegister { register })
    }
}

impl<const LEN: usize> Parse for Registers<LEN> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        delimited(
            tag("["),
            map(parse_separated_array(parse_integer, tag(", ")), Self),
            tag("]"),
        )(input)
    }
}

// These are allowed because they're constructed in `TryFrom::try_from`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, EnumCount, EnumIter, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum OpCode {
    AddR,
    AddI,
    MulR,
    MulI,
    BAnR,
    BAnI,
    BOrR,
    BOrI,
    SetR,
    SetI,
    GTIR,
    GTRI,
    GTRR,
    EqIR,
    EqRI,
    EqRR,
}

impl OpCode {
    fn all() -> OpCodeBitArr {
        let mut all: OpCodeBitArr = OpCodeBitArr::ZERO;

        all[..Self::COUNT].fill(true);

        all
    }

    pub fn a_is_register(self) -> bool {
        !matches!(self, Self::SetI | Self::GTIR | Self::EqIR)
    }

    pub fn b_is_register(self) -> bool {
        matches!(
            self,
            Self::AddR
                | Self::MulR
                | Self::BAnR
                | Self::BOrR
                | Self::GTIR
                | Self::GTRR
                | Self::EqIR
                | Self::EqRR
        )
    }

    pub fn is_set(self) -> bool {
        matches!(self, Self::SetR | Self::SetI)
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::GTIR | Self::GTRI | Self::GTRR | Self::EqIR | Self::EqRI | Self::EqRR
        )
    }

    pub fn try_op_str(self) -> Option<&'static str> {
        match self {
            Self::AddR | Self::AddI => Some("+"),
            Self::MulR | Self::MulI => Some("*"),
            Self::BAnR | Self::BAnI => Some("&"),
            Self::BOrR | Self::BOrI => Some("|"),
            Self::GTIR | Self::GTRI | Self::GTRR => Some(">"),
            Self::EqIR | Self::EqRI | Self::EqRR => Some("=="),
            _ => None,
        }
    }

    pub fn tag_str(self) -> &'static str {
        match self {
            Self::AddR => "addr",
            Self::AddI => "addi",
            Self::MulR => "mulr",
            Self::MulI => "muli",
            Self::BAnR => "banr",
            Self::BAnI => "bani",
            Self::BOrR => "borr",
            Self::BOrI => "bori",
            Self::SetR => "setr",
            Self::SetI => "seti",
            Self::GTIR => "gtir",
            Self::GTRI => "gtri",
            Self::GTRR => "gtrr",
            Self::EqIR => "eqir",
            Self::EqRI => "eqri",
            Self::EqRR => "eqrr",
        }
    }
}

impl Parse for OpCode {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        Self::iter()
            .find_map(|op_code| {
                tag::<&str, &str, NomError<&str>>(op_code.tag_str())(input)
                    .ok()
                    .map(|(remaining, _)| (remaining, op_code))
            })
            .ok_or_else(|| NomErr::Error(NomError::new(input, ErrorKind::Alt)))
    }
}

impl TryFrom<usize> for OpCode {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value >= Self::COUNT {
            Err(Error::InvalidOpCodeUsize { value })
        } else {
            // SAFETY: `value` is valid
            Ok(unsafe { transmute(value as u8) })
        }
    }
}

impl TryFrom<RegisterRaw> for OpCode {
    type Error = Error;

    fn try_from(value: RegisterRaw) -> Result<Self, Self::Error> {
        if value < RegisterRaw::zero() {
            Err(Error::InvalidOpCodeRegisterRaw { value })
        } else {
            (value as usize).try_into()
        }
    }
}

type OpCodeBitArr = BitArr!(for OpCode::COUNT, in u16);

#[derive(Clone, Copy)]
pub enum StatementType {
    CEqA,
    COpEqA,
    COpEqB,
    CEqAOpB,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
pub struct Instruction<O = OpCode> {
    pub op_code: O,
    pub a: RegisterRaw,
    pub b: RegisterRaw,
    pub c: RegisterRaw,
}

impl Instruction<OpCode> {
    pub fn try_execute<const LEN: usize>(
        self,
        registers: &mut Registers<LEN>,
    ) -> Result<(), Error> {
        let a: Result<usize, Error> = Registers::<LEN>::try_register_index(self.a);
        let b: Result<usize, Error> = Registers::<LEN>::try_register_index(self.b);
        let c: RegisterRaw = match self.op_code {
            OpCode::AddR => {
                let a: RegisterRaw = registers.0[a?];
                let b: RegisterRaw = registers.0[b?];

                a.checked_add(b)
                    .ok_or_else(|| Error::AddOverflow { a, b })?
            }
            OpCode::AddI => {
                let a: RegisterRaw = registers.0[a?];
                let b: RegisterRaw = self.b;

                a.checked_add(b)
                    .ok_or_else(|| Error::AddOverflow { a, b })?
            }
            OpCode::MulR => {
                let a: RegisterRaw = registers.0[a?];
                let b: RegisterRaw = registers.0[b?];

                a.checked_mul(b)
                    .ok_or_else(|| Error::MulOverflow { a, b })?
            }
            OpCode::MulI => {
                let a: RegisterRaw = registers.0[a?];
                let b: RegisterRaw = self.b;

                a.checked_mul(b)
                    .ok_or_else(|| Error::MulOverflow { a, b })?
            }
            OpCode::BAnR => registers.0[a?] & registers.0[b?],
            OpCode::BAnI => registers.0[a?] & self.b,
            OpCode::BOrR => registers.0[a?] | registers.0[b?],
            OpCode::BOrI => registers.0[a?] | self.b,
            OpCode::SetR => registers.0[a?],
            OpCode::SetI => self.a,
            OpCode::GTIR => (self.a > registers.0[b?]) as RegisterRaw,
            OpCode::GTRI => (registers.0[a?] > self.b) as RegisterRaw,
            OpCode::GTRR => (registers.0[a?] > registers.0[b?]) as RegisterRaw,
            OpCode::EqIR => (self.a == registers.0[b?]) as RegisterRaw,
            OpCode::EqRI => (registers.0[a?] == self.b) as RegisterRaw,
            OpCode::EqRR => (registers.0[a?] == registers.0[b?]) as RegisterRaw,
        };

        registers.0[Registers::<LEN>::try_register_index(self.c).unwrap()] = c;

        Ok(())
    }

    pub fn statement_type(&self) -> StatementType {
        if self.op_code.is_set() {
            StatementType::CEqA
        } else if self.op_code.is_comparison() {
            StatementType::CEqAOpB
        } else if self.a == self.c {
            StatementType::COpEqB
        } else if self.b == self.c {
            StatementType::COpEqA
        } else {
            StatementType::CEqAOpB
        }
    }

    fn a(&self) -> &RegisterRaw {
        &self.a
    }

    fn b(&self) -> &RegisterRaw {
        &self.b
    }

    fn display_component<'d, F: Fn(OpCode) -> bool, G: Fn(&Self) -> &RegisterRaw>(
        &'d self,
        register_names: &'d [char],
        register_name: &'d mut char,
        component_is_register: F,
        access_component: G,
    ) -> &'d dyn Display {
        if component_is_register(self.op_code) {
            *register_name = register_names[*access_component(self) as usize];

            register_name as &dyn Display
        } else {
            access_component(self) as &dyn Display
        }
    }

    fn display_a<'d>(
        &'d self,
        register_names: &'d [char],
        register_name: &'d mut char,
    ) -> &'d dyn Display {
        self.display_component(
            register_names,
            register_name,
            OpCode::a_is_register,
            Self::a,
        )
    }

    fn display_b<'d>(
        &'d self,
        register_names: &'d [char],
        register_name: &'d mut char,
    ) -> &'d dyn Display {
        self.display_component(
            register_names,
            register_name,
            OpCode::b_is_register,
            Self::b,
        )
    }

    fn display_c(&self, register_names: &[char]) -> char {
        register_names[self.c as usize]
    }

    pub fn print_simplified(
        &self,
        register_names: &[char],
        string: &mut String,
    ) -> Result<(), FmtError> {
        let mut register_a_name: char = ' ';
        let mut register_b_name: char = ' ';

        match self.statement_type() {
            StatementType::CEqA => write!(
                string,
                "{} = {};",
                self.display_c(register_names),
                self.display_a(register_names, &mut register_a_name)
            ),
            StatementType::COpEqA => write!(
                string,
                "{} {}= {};",
                self.display_c(register_names),
                self.op_code.try_op_str().unwrap(),
                self.display_a(register_names, &mut register_a_name)
            ),
            StatementType::COpEqB => write!(
                string,
                "{} {}= {};",
                self.display_c(register_names),
                self.op_code.try_op_str().unwrap(),
                self.display_b(register_names, &mut register_b_name)
            ),
            StatementType::CEqAOpB => write!(
                string,
                "{} = {} {} {};",
                self.display_c(register_names),
                self.display_a(register_names, &mut register_a_name),
                self.op_code.try_op_str().unwrap(),
                self.display_b(register_names, &mut register_b_name)
            ),
        }
    }
}

impl Instruction<RegisterRaw> {
    fn with_op_code(self, op_code: OpCode) -> Instruction {
        let Instruction { a, b, c, .. } = self;

        Instruction { op_code, a, b, c }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                OpCode::parse,
                tag(" "),
                parse_separated_array(parse_integer, tag(" ")),
            )),
            |(op_code, _, [a, b, c])| Self { op_code, a, b, c },
        )(input)
    }
}

impl Parse for Instruction<RegisterRaw> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            parse_separated_array(parse_integer, tag(" ")),
            |[op_code, a, b, c]| {
                (OpCode::try_from(op_code).is_ok()
                    && Registers::<DEFAULT_REGISTERS_LEN>::try_register_index(c).is_ok())
                .then(|| Self { op_code, a, b, c })
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Sample {
    before: Registers,
    instruction: Instruction<RegisterRaw>,
    after: Registers,
}

impl Sample {
    fn potential_op_codes(&self) -> OpCodeBitArr {
        let mut potential_op_codes: OpCodeBitArr = OpCodeBitArr::ZERO;

        for (index, mut is_op_code_potential) in
            potential_op_codes[..OpCode::COUNT].iter_mut().enumerate()
        {
            let mut before: Registers = self.before.clone();

            is_op_code_potential.set(
                self.instruction
                    .with_op_code(index.try_into().unwrap())
                    .try_execute(&mut before)
                    .map_or_else(|_| false, |_| before == self.after),
            );
        }

        potential_op_codes
    }
}

impl Parse for Sample {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("Before: "),
                Registers::parse,
                line_ending,
                Instruction::parse,
                line_ending,
                tag("After:  "),
                Registers::parse,
            )),
            |(_, before, _, instruction, _, _, after)| Self {
                before,
                instruction,
                after,
            },
        )(input)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct OpCodeMapping<O = OpCode>([O; OpCode::COUNT]);

impl OpCodeMapping<Option<OpCode>> {
    fn set_op_codes(&self) -> OpCodeBitArr {
        let mut op_code_bit_arr: OpCodeBitArr = OpCodeBitArr::ZERO;

        for op_code in self.0.into_iter().flatten() {
            op_code_bit_arr.set(op_code as usize, true);
        }

        op_code_bit_arr
    }

    fn set_register_values(&self) -> OpCodeBitArr {
        let mut op_code_bit_arr: OpCodeBitArr = OpCodeBitArr::ZERO;

        for index in self
            .0
            .iter()
            .enumerate()
            .filter_map(|(index, op_code)| op_code.is_some().then_some(index))
        {
            op_code_bit_arr.set(index, true);
        }

        op_code_bit_arr
    }
}

struct OpCodeMappingFinder {
    potential_op_codes_array: [OpCodeBitArr; OpCode::COUNT],
    end: Option<OpCodeMapping>,
}

impl OpCodeMappingFinder {
    fn try_find_op_code_mapping(samples: &[Sample]) -> Result<OpCodeMapping, Error> {
        Self::try_from(samples).and_then(|mut finder| {
            finder
                .run()
                .map_or(Err(Error::OpCodeMappingFinderFailed), |_| {
                    Ok(finder.end.unwrap())
                })
        })
    }
}

impl DepthFirstSearch for OpCodeMappingFinder {
    type Vertex = OpCodeMapping<Option<OpCode>>;

    fn start(&self) -> &Self::Vertex {
        const START: OpCodeMapping<Option<OpCode>> = OpCodeMapping([None; OpCode::COUNT]);

        &START
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.0.iter().all(Option::is_some)
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        // We don't care about the path laid out like this
        Vec::new()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        let clear_op_codes: OpCodeBitArr = !vertex.set_op_codes();
        let clear_register_values: OpCodeBitArr = !vertex.set_register_values();

        neighbors.clear();
        neighbors.extend(
            self.potential_op_codes_array
                .iter()
                .enumerate()
                .filter_map(|(register_value, &potential_op_codes)| {
                    clear_register_values[register_value]
                        .then(|| (register_value, potential_op_codes & clear_op_codes))
                })
                .min_by_key(|(_, potential_op_codes)| potential_op_codes.count_ones())
                .into_iter()
                .flat_map(|(register_value, potential_op_codes)| {
                    potential_op_codes.into_iter().enumerate().filter_map(
                        move |(op_code, is_potential)| {
                            is_potential.then(|| {
                                let mut neighbor: Self::Vertex = vertex.clone();

                                neighbor.0[register_value] = Some(op_code.try_into().unwrap());

                                neighbor
                            })
                        },
                    )
                }),
        );
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        if self.is_end(to) {
            let mut end: OpCodeMapping =
                OpCodeMapping([0_usize.try_into().unwrap(); OpCode::COUNT]);

            for (src_op_code, dst_op_code) in to.0.into_iter().zip(end.0.iter_mut()) {
                *dst_op_code = src_op_code.unwrap();
            }

            self.end = Some(end);
        }
    }

    fn reset(&mut self) {
        self.end = None;
    }
}

impl TryFrom<&[Sample]> for OpCodeMappingFinder {
    type Error = Error;

    fn try_from(samples: &[Sample]) -> Result<Self, Self::Error> {
        let mut potential_op_codes_array: [OpCodeBitArr; OpCode::COUNT] =
            [OpCode::all(); OpCode::COUNT];

        samples
            .iter()
            .try_fold((), |_, sample| {
                let potential_op_codes: &mut OpCodeBitArr =
                    &mut potential_op_codes_array[sample.instruction.op_code as usize];

                *potential_op_codes = *potential_op_codes & sample.potential_op_codes();

                potential_op_codes.any().then_some(()).ok_or(
                    Error::OpCodeRegisterHasNoPotentialOpCodes {
                        op_code: sample.instruction.op_code,
                    },
                )
            })
            .map(|_| Self {
                potential_op_codes_array,
                end: None,
            })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    samples: Vec<Sample>,
    instructions: Vec<Instruction<RegisterRaw>>,
}

impl Solution {
    const MIN_POTENTIAL_OP_CODES: usize = 3_usize;

    fn sample_count_with_min_potential_op_codes(&self, min_potential_op_codes: usize) -> usize {
        self.samples
            .iter()
            .filter(|sample| sample.potential_op_codes().count_ones() >= min_potential_op_codes)
            .count()
    }

    fn try_op_code_mapping(&self) -> Result<OpCodeMapping, Error> {
        OpCodeMappingFinder::try_find_op_code_mapping(&self.samples)
    }

    fn try_final_registers(&self) -> Result<Registers, Error> {
        let mut registers: Registers = Registers::default();

        self.try_op_code_mapping()
            .and_then(|op_code_mapping| {
                self.instructions.iter().try_fold((), |_, instruction| {
                    instruction
                        .with_op_code(op_code_mapping.0[instruction.op_code as usize])
                        .try_execute(&mut registers)
                })
            })
            .map(|_| registers)
    }

    fn try_final_register_0(&self) -> Result<RegisterRaw, Error> {
        self.try_final_registers()
            .map(|registers| registers.0[0_usize])
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                separated_list0(tuple((line_ending, line_ending)), Sample::parse),
                opt(tuple((line_ending, line_ending, line_ending, line_ending))),
                separated_list0(line_ending, Instruction::parse),
            ),
            |(samples, instructions)| Self {
                samples,
                instructions,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Guessing part 2 is gonna be "figure out what's what, run the program, and return what's in
    /// a register".
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sample_count_with_min_potential_op_codes(Self::MIN_POTENTIAL_OP_CODES));
    }

    /// Part 2 took way longer than expected for a question description that was quite expected lol.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_final_register_0()).ok();
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<NomError<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        Before: [3, 2, 1, 1]\n\
        9 2 1 2\n\
        After:  [3, 2, 2, 1]\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                samples: vec![Sample {
                    before: Registers([
                        3 as RegisterRaw,
                        2 as RegisterRaw,
                        1 as RegisterRaw,
                        1 as RegisterRaw,
                    ]),
                    instruction: Instruction {
                        op_code: 9 as RegisterRaw,
                        a: 2 as RegisterRaw,
                        b: 1 as RegisterRaw,
                        c: 2 as RegisterRaw,
                    },
                    after: Registers([
                        3 as RegisterRaw,
                        2 as RegisterRaw,
                        2 as RegisterRaw,
                        1 as RegisterRaw,
                    ]),
                }],
                instructions: Vec::new(),
            }]
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
    fn test_potential_op_codes() {
        for (index, potential_op_codes) in [vec![
            bitarr_typed![OpCodeBitArr; 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        ]]
        .into_iter()
        .enumerate()
        {
            for (sample, potential_op_codes) in
                solution(index).samples.iter().zip(potential_op_codes)
            {
                assert_eq!(sample.potential_op_codes(), potential_op_codes);
            }
        }
    }

    #[test]
    fn test_sample_count_with_min_potential_op_codes() {
        for (index, sample_count_with_min_potential_op_codes) in [1_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index)
                    .sample_count_with_min_potential_op_codes(Solution::MIN_POTENTIAL_OP_CODES),
                sample_count_with_min_potential_op_codes
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
