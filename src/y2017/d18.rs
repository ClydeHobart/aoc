use {
    crate::*,
    futures::executor::block_on,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{line_ending, satisfy},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    pin_utils::pin_mut,
    std::{
        cell::RefCell,
        collections::VecDeque,
        future::{poll_fn, Future},
        ops::Deref,
        rc::Rc,
        task::Poll,
    },
};

/* --- Day 18: Duet ---

You discover a tablet containing some strange assembly code labeled simply "Duet". Rather than bother the sound card with it, you decide to run the code yourself. Unfortunately, you don't see any documentation, so you're left to figure out what the instructions mean on your own.

It seems like the assembly is meant to operate on a set of registers that are each named with a single letter and that can each hold a single integer. You suppose each register should start with a value of 0.

There aren't that many instructions, so it shouldn't be hard to figure out what they do. Here's what you determine:

    snd X plays a sound with a frequency equal to the value of X.
    set X Y sets register X to the value of Y.
    add X Y increases register X by the value of Y.
    mul X Y sets register X to the result of multiplying the value contained in register X by the value of Y.
    mod X Y sets register X to the remainder of dividing the value contained in register X by the value of Y (that is, it sets X to the result of X modulo Y).
    rcv X recovers the frequency of the last sound played, but only when the value of X is not zero. (If it is zero, the command does nothing.)
    jgz X Y jumps with an offset of the value of Y, but only if the value of X is greater than zero. (An offset of 2 skips the next instruction, an offset of -1 jumps to the previous instruction, and so on.)

Many of the instructions can take either a register (a single letter) or a number. The value of a register is the integer it contains; the value of a number is that number.

After each jump instruction, the program continues with the instruction to which the jump jumped. After any other instruction, the program continues with the next instruction. Continuing (or jumping) off either end of the program terminates it.

For example:

set a 1
add a 2
mul a a
mod a 5
snd a
set a 0
rcv a
jgz a -1
set a 1
jgz a -2

    The first four instructions set a to 1, add 2 to it, square it, and then set it to itself modulo 5, resulting in a value of 4.
    Then, a sound with frequency 4 (the value of a) is played.
    After that, a is set to 0, causing the subsequent rcv and jgz instructions to both be skipped (rcv because a is 0, and jgz because a is not greater than 0).
    Finally, a is set to 1, causing the next jgz instruction to activate, jumping back two instructions to another jump, which jumps again to the rcv, which ultimately triggers the recover operation.

At the time the recover operation is executed, the frequency of the last sound played is 4.

What is the value of the recovered frequency (the value of the most recently played sound) the first time a rcv instruction is executed with a non-zero value?

--- Part Two ---

As you congratulate yourself for a job well done, you notice that the documentation has been on the back of the tablet this entire time. While you actually got most of the instructions correct, there are a few key differences. This assembly code isn't about sound at all - it's meant to be run twice at the same time.

Each running copy of the program has its own set of registers and follows the code independently - in fact, the programs don't even necessarily run at the same speed. To coordinate, they use the send (snd) and receive (rcv) instructions:

    snd X sends the value of X to the other program. These values wait in a queue until that program is ready to receive them. Each program has its own message queue, so a program can never receive a message it sent.
    rcv X receives the next value and stores it in register X. If no values are in the queue, the program waits for a value to be sent to it. Programs do not continue to the next instruction until they have received a value. Values are received in the order they are sent.

Each program also has its own program ID (one 0 and the other 1); the register p should begin with this value.

For example:

snd 1
snd 2
snd p
rcv a
rcv b
rcv c
rcv d

Both programs begin by sending three values to the other. Program 0 sends 1, 2, 0; program 1 sends 1, 2, 1. Then, each program receives a value (both 1) and stores it in a, receives another value (both 2) and stores it in b, and then each receives the program ID of the other program (program 0 receives 1; program 1 receives 0) and stores it in c. Each program now sees a different value in its own copy of register c.

Finally, both programs try to rcv a fourth time, but no data is waiting for either of them, and they reach a deadlock. When this happens, both programs terminate.

It should be noted that it would be equally valid for the programs to run at different speeds; for example, program 0 might have sent all three values and then stopped at the first rcv before program 1 executed even its first instruction.

Once both of your programs have terminated (regardless of what caused them to do so), how many times did program 1 send a value? */

type RegisterName = char;
type RegisterIndexRaw = u8;
type RegisterNameList = IdList<RegisterName, RegisterIndexRaw>;
type RegisterIndex = TableIndex<RegisterIndexRaw>;

fn parse_register<'i>(input: &'i str) -> IResult<&'i str, RegisterName> {
    satisfy(|c| c.is_ascii_lowercase())(input)
}

trait State {
    fn get(&self, register: RegisterIndex) -> i64;
    fn has_halted(&self, instructions_len: usize) -> bool;
    fn get_instruction_index(&self) -> isize;
    fn set(&mut self, register: RegisterIndex, value: i64);
    fn offset_instruction_index(&mut self, offset: isize);

    fn snd(&mut self, value: i64);
    async fn rcv(&mut self, register: RegisterIndex);
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct SingleState {
    registers: Vec<i64>,
    snd: Option<i64>,
    rcv: Option<i64>,
    instruction_index: isize,
}

impl SingleState {
    fn new(registers_len: usize) -> Self {
        SingleState {
            registers: vec![0_i64; registers_len],
            snd: None,
            rcv: None,
            instruction_index: 0_isize,
        }
    }

    fn execute(&mut self, instructions: &[Instruction]) {
        while !self.has_halted(instructions.len()) {
            block_on(instructions[self.get_instruction_index() as usize].execute(self));
        }
    }
}

impl State for SingleState {
    fn get(&self, register: RegisterIndex) -> i64 {
        self.registers[register.get()]
    }

    fn has_halted(&self, instructions_len: usize) -> bool {
        self.rcv.is_some()
            || self.instruction_index < 0_isize
            || self.instruction_index as usize >= instructions_len
    }

    fn get_instruction_index(&self) -> isize {
        self.instruction_index
    }

    fn set(&mut self, register: RegisterIndex, value: i64) {
        self.registers[register.get()] = value;
    }

    fn offset_instruction_index(&mut self, offset: isize) {
        self.instruction_index += offset;
    }

    fn snd(&mut self, value: i64) {
        self.snd = Some(value);
    }

    async fn rcv(&mut self, register: RegisterIndex) {
        if self.get(register) != 0_i64 {
            self.rcv = self.snd.clone();
        }
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Default)]
struct ValueChannel(Rc<RefCell<VecDeque<i64>>>);

impl ValueChannel {
    fn snd(&mut self, value: i64) {
        Rc::deref(&self.0).borrow_mut().push_back(value);
    }

    async fn rcv(&mut self) -> i64 {
        poll_fn(|_cx| poll_from_option(Rc::deref(&self.0).borrow_mut().pop_front())).await
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct HalfDoubleState {
    registers: Vec<i64>,
    snd: ValueChannel,
    rcv: ValueChannel,
    snd_count: usize,
    instruction_index: isize,
}

impl HalfDoubleState {
    fn execute<'a>(&'a mut self, instructions: &'a [Instruction]) -> impl Future<Output = ()> + 'a {
        poll_fn(|cx| {
            if self.has_halted(instructions.len()) {
                Poll::Pending
            } else {
                let future = instructions[self.get_instruction_index() as usize].execute(self);

                pin_mut!(future);

                future.poll(cx)
            }
        })
    }
}

impl State for HalfDoubleState {
    fn get(&self, register: RegisterIndex) -> i64 {
        self.registers[register.get()]
    }

    fn has_halted(&self, instructions_len: usize) -> bool {
        self.instruction_index < 0_isize || self.instruction_index as usize >= instructions_len
    }

    fn get_instruction_index(&self) -> isize {
        self.instruction_index
    }

    fn set(&mut self, register: RegisterIndex, value: i64) {
        self.registers[register.get()] = value;
    }

    fn offset_instruction_index(&mut self, offset: isize) {
        self.instruction_index += offset;
    }

    fn snd(&mut self, value: i64) {
        self.snd.snd(value);
        self.snd_count += 1_usize;
    }

    async fn rcv(&mut self, register: RegisterIndex) {
        self.registers[register.get()] = self.rcv.rcv().await;
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct DoubleState {
    halves: [HalfDoubleState; 2_usize],
}

impl DoubleState {
    fn new(registers_len: usize, program_id_register: RegisterIndex) -> Self {
        let registers: Vec<i64> = vec![0_i64; registers_len];
        let snd: ValueChannel = ValueChannel::default();
        let rcv: ValueChannel = ValueChannel::default();
        let snd_count: usize = 0_usize;
        let instruction_index: isize = 0_isize;
        let half_0: HalfDoubleState = HalfDoubleState {
            registers,
            snd,
            rcv,
            snd_count,
            instruction_index,
        };

        let mut registers: Vec<i64> = half_0.registers.clone();

        registers[program_id_register.get()] = 1_i64;

        let snd: ValueChannel = half_0.rcv.clone();
        let rcv: ValueChannel = half_0.snd.clone();
        let half_1: HalfDoubleState = HalfDoubleState {
            registers,
            snd,
            rcv,
            snd_count,
            instruction_index,
        };

        Self {
            halves: [half_0, half_1],
        }
    }

    /// Returns Pending when no more progress can be made, otherwise makes some progress.
    fn execute_one_async<'a>(
        &'a mut self,
        instructions: &'a [Instruction],
    ) -> impl Future<Output = ()> + 'a {
        let [half_0, half_1]: &mut [HalfDoubleState; 2_usize] = &mut self.halves;

        poll_fn(move |cx| {
            let future_0 = half_0.execute(instructions);
            let future_1 = half_1.execute(instructions);

            pin_mut!(future_0, future_1);

            match (future_0.poll(cx), future_1.poll(cx)) {
                (Poll::Ready(_), Poll::Ready(_)) => Poll::Ready(()),
                (Poll::Ready(_), Poll::Pending) => Poll::Ready(()),
                (Poll::Pending, Poll::Ready(_)) => Poll::Ready(()),
                (Poll::Pending, Poll::Pending) => Poll::Pending,
            }
        })
    }

    /// Returns Pending when no more progress can be made, otherwise makes all progress.
    async fn execute_all_async(&mut self, instructions: &[Instruction]) {
        loop {
            self.execute_one_async(instructions).await
        }
    }

    fn execute(&mut self, instructions: &[Instruction]) {
        block_on(poll_fn(|cx| {
            let future = self.execute_all_async(instructions);

            pin_mut!(future);

            let _ = future.poll(cx);

            Poll::Ready(())
        }));
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Value<R = RegisterIndex> {
    Constant(i64),
    Register(R),
}

impl Value {
    fn new(value: Value<RegisterName>, registers: &mut RegisterNameList) -> Self {
        match value {
            Value::Constant(constant) => Self::Constant(constant),
            Value::Register(register) => Self::Register(registers.find_or_add_index(&register)),
        }
    }

    fn evaluate<S: State>(&self, state: &S) -> i64 {
        match self {
            Self::Constant(constant) => *constant,
            Self::Register(register) => state.get(*register),
        }
    }
}

impl Parse for Value<RegisterName> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(parse_integer, Self::Constant),
            map(parse_register, Self::Register),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Instruction<R = RegisterIndex> {
    Snd { x: Value<R> },
    Set { x: R, y: Value<R> },
    Add { x: R, y: Value<R> },
    Mul { x: R, y: Value<R> },
    Mod { x: R, y: Value<R> },
    Rcv { x: R },
    Jgz { x: Value<R>, y: Value<R> },
}

impl Instruction {
    fn new(instruction: Instruction<RegisterName>, registers: &mut RegisterNameList) -> Self {
        match instruction {
            Instruction::Snd { x } => Self::Snd {
                x: Value::new(x, registers),
            },
            Instruction::Set { x, y } => {
                let x: RegisterIndex = registers.find_or_add_index(&x);
                let y: Value = Value::new(y, registers);

                Self::Set { x, y }
            }
            Instruction::Add { x, y } => {
                let x: RegisterIndex = registers.find_or_add_index(&x);
                let y: Value = Value::new(y, registers);

                Self::Add { x, y }
            }
            Instruction::Mul { x, y } => {
                let x: RegisterIndex = registers.find_or_add_index(&x);
                let y: Value = Value::new(y, registers);

                Self::Mul { x, y }
            }
            Instruction::Mod { x, y } => {
                let x: RegisterIndex = registers.find_or_add_index(&x);
                let y: Value = Value::new(y, registers);

                Self::Mod { x, y }
            }
            Instruction::Rcv { x } => Self::Rcv {
                x: registers.find_or_add_index(&x),
            },
            Instruction::Jgz { x, y } => {
                let x: Value = Value::new(x, registers);
                let y: Value = Value::new(y, registers);

                Self::Jgz { x, y }
            }
        }
    }

    async fn execute<S: State>(&self, state: &mut S) {
        let mut offset: isize = 1_isize;

        match self {
            Self::Snd { x } => state.snd(x.evaluate(state)),
            Self::Set { x, y } => state.set(*x, y.evaluate(state)),
            Self::Add { x, y } => state.set(*x, state.get(*x) + y.evaluate(state)),
            Self::Mul { x, y } => state.set(*x, state.get(*x) * y.evaluate(state)),
            Self::Mod { x, y } => state.set(*x, state.get(*x) % y.evaluate(state)),
            Self::Rcv { x } => state.rcv(*x).await,
            Self::Jgz { x, y } => {
                if x.evaluate(state) > 0_i64 {
                    offset = y.evaluate(state) as isize;
                }
            }
        }

        state.offset_instruction_index(offset);
    }
}

impl Parse for Instruction<RegisterName> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        macro_rules! alt_branch {
            ($tag:literal => $variant:ident { x }) => {
                map(preceded(tag($tag), parse_register), |x| Self::$variant {
                    x,
                })
            };
            ($tag:literal => $variant:ident { x, y }) => {
                map(
                    tuple((tag($tag), parse_register, tag(" "), Value::parse)),
                    |(_, x, _, y)| Self::$variant { x, y },
                )
            };
        }

        alt((
            map(preceded(tag("snd "), Value::parse), |x| Self::Snd { x }),
            alt_branch!("set " => Set { x, y }),
            alt_branch!("add " => Add { x, y }),
            alt_branch!("mul " => Mul { x, y }),
            alt_branch!("mod " => Mod { x, y }),
            alt_branch!("rcv " => Rcv { x }),
            map(
                tuple((tag("jgz "), Value::parse, tag(" "), Value::parse)),
                |(_, x, _, y)| Self::Jgz { x, y },
            ),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    registers: RegisterNameList,
    instructions: Vec<Instruction>,
    program_id_register: RegisterIndex,
}

impl Solution {
    fn single_state_execute(&self) -> SingleState {
        let mut state: SingleState = SingleState::new(self.registers.as_slice().len());

        state.execute(&self.instructions);

        state
    }

    fn recovered_frequency_value(&self) -> Option<i64> {
        self.single_state_execute().rcv
    }

    fn double_state_execute(&self) -> DoubleState {
        let mut state: DoubleState =
            DoubleState::new(self.registers.as_slice().len(), self.program_id_register);

        state.execute(&self.instructions);

        state
    }

    fn program_1_snd_count(&self) -> usize {
        self.double_state_execute().halves[1_usize].snd_count
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut registers: RegisterNameList = RegisterNameList::new();

        let (input, instructions): (&str, Vec<Instruction>) = many0(terminated(
            map(Instruction::parse, |instruction| {
                Instruction::new(instruction, &mut registers)
            }),
            opt(line_ending),
        ))(input)?;
        let program_id_register: RegisterIndex = registers.find_or_add_index(&'p');

        Ok((
            input,
            Self {
                registers,
                instructions,
                program_id_register,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// Rust's panic on overflow in debug builds is a blessing. I've done so many of these little
    /// instruction set implementations that my mind has turned to mush.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.single_state_execute());
        } else {
            dbg!(self.recovered_frequency_value());
        }
    }

    /// I'm afraid. Into async-await I go.
    ///
    /// Update: I'm disappointed I wasn't able to sort out an async-await implementation of this. My
    /// solution is hacky and it reveals more of the player-specific input than I'm comfortable
    /// with.
    ///
    /// Update: I figured it out. I ended up needing less of the `futures` crate than I initially
    /// thought, which I see as a plus. I might be able to rework things to use `select` if I really
    /// wanted to, but honestly the syntax around `Stream` objects from `futures` is pretty gross,
    /// so I think I'll leave things where they are now.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.double_state_execute());
        } else {
            dbg!(self.program_1_snd_count());
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
    use {
        super::{
            Instruction::*,
            Value::{Constant as C, Register as R},
            *,
        },
        std::sync::OnceLock,
    };

    const SOLUTION_STRS: &'static [&'static str] = &[
        "\
        set a 1\n\
        add a 2\n\
        mul a a\n\
        mod a 5\n\
        snd a\n\
        set a 0\n\
        rcv a\n\
        jgz a -1\n\
        set a 1\n\
        jgz a -2\n",
        "snd 1\n\
        snd 2\n\
        snd p\n\
        rcv a\n\
        rcv b\n\
        rcv c\n\
        rcv d\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                {
                    let a: RegisterIndex = 0_usize.into();

                    Solution {
                        registers: vec!['a', 'p'].try_into().unwrap(),
                        instructions: vec![
                            Set { x: a, y: C(1_i64) },
                            Add { x: a, y: C(2_i64) },
                            Mul { x: a, y: R(a) },
                            Mod { x: a, y: C(5_i64) },
                            Snd { x: R(a) },
                            Set { x: a, y: C(0_i64) },
                            Rcv { x: a },
                            Jgz {
                                x: R(a),
                                y: C(-1_i64),
                            },
                            Set { x: a, y: C(1_i64) },
                            Jgz {
                                x: R(a),
                                y: C(-2_i64),
                            },
                        ],
                        program_id_register: 1_usize.into(),
                    }
                },
                {
                    let [p, a, b, c, d]: [RegisterIndex; 5_usize] = [
                        0_usize.into(),
                        1_usize.into(),
                        2_usize.into(),
                        3_usize.into(),
                        4_usize.into(),
                    ];

                    Solution {
                        registers: vec!['p', 'a', 'b', 'c', 'd'].try_into().unwrap(),
                        instructions: vec![
                            Snd { x: C(1_i64) },
                            Snd { x: C(2_i64) },
                            Snd { x: R(p) },
                            Rcv { x: a },
                            Rcv { x: b },
                            Rcv { x: c },
                            Rcv { x: d },
                        ],
                        program_id_register: 0_usize.into(),
                    }
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
    fn test_single_state_execute() {
        for (index, state) in [
            SingleState {
                registers: vec![1_i64, 0_i64],
                snd: Some(4_i64),
                rcv: Some(4_i64),
                instruction_index: 7_isize,
            },
            SingleState {
                registers: vec![0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
                snd: Some(0_i64),
                rcv: None,
                instruction_index: 7_isize,
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).single_state_execute(), state);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
