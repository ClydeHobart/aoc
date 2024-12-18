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
};

/* --- Day 8: I Heard You Like Registers ---

You receive a signal directly from the CPU. Because of your recent assistance with jump instructions, it would like you to compute the result of a series of unusual register instructions.

Each instruction consists of several parts: the register to modify, whether to increase or decrease that register's value, the amount by which to increase or decrease it, and a condition. If the condition fails, skip the instruction without modifying the register. The registers all start at 0. The instructions look like this:

b inc 5 if a > 1
a inc 1 if b < 5
c dec -10 if a >= 1
c inc -20 if c == 10

These instructions would be processed as follows:

    Because a starts at 0, it is not greater than 1, and so b is not modified.
    a is increased by 1 (to 1) because b is less than 5 (it is 0).
    c is decreased by -10 (to 10) because a is now greater than or equal to 1 (it is 1).
    c is increased by -20 (to -10) because c is equal to 10.

After this process, the largest value in any register is 1.

You might also encounter <= (less than or equal to) or != (not equal to). However, the CPU doesn't have the bandwidth to tell you what all the registers are named, and leaves that to you to determine.

What is the largest value in any register after completing the instructions in your puzzle input?

--- Part Two ---

To be safe, the CPU also needs to know the highest value held in any register during this process so that it can decide how much memory to allocate to these operations. For example, in the above instructions, the highest value ever held was 10 (in register c after the third instruction was evaluated). */

type RegisterName = StaticString<{ Solution::REGISTER_NAME_LEN }>;
type RegisterIndexRaw = u16;
type RegisterIndex = Index<RegisterIndexRaw>;
type RegisterNameList = IdList<RegisterName, RegisterIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Operator {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Operator {
    fn tag(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    fn alt_branch<'i>(self) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        map(tag(self.tag()), move |_| self)
    }

    fn cmp_func<T: PartialOrd>(self) -> fn(&T, &T) -> bool {
        match self {
            Self::Eq => T::eq,
            Self::Ne => T::ne,
            Self::Lt => T::lt,
            Self::Le => T::le,
            Self::Gt => T::gt,
            Self::Ge => T::ge,
        }
    }
}

impl Parse for Operator {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::Eq.alt_branch(),
            Self::Ne.alt_branch(),
            Self::Le.alt_branch(),
            Self::Lt.alt_branch(),
            Self::Ge.alt_branch(),
            Self::Gt.alt_branch(),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Condition {
    register_index: RegisterIndex,
    operator: Operator,
    comparand: i32,
}

type ConditionParams = (RegisterName, Operator, i32);

impl Condition {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, ConditionParams> {
        map(
            tuple((
                tag("if "),
                Solution::parse_register_name,
                tag(" "),
                Operator::parse,
                tag(" "),
                parse_integer,
            )),
            |(_, register_name, _, operator, _, comparand)| (register_name, operator, comparand),
        )(input)
    }

    fn new(
        (register_name, operator, comparand): ConditionParams,
        registers: &mut RegisterNameList,
    ) -> Self {
        let register_index: RegisterIndex = registers.find_or_add_index(&register_name);

        Self {
            register_index,
            operator,
            comparand,
        }
    }

    fn evaluate(&self, registers: &[i32]) -> bool {
        self.operator.cmp_func()(&registers[self.register_index.get()], &self.comparand)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Instruction {
    register_index: RegisterIndex,
    increment: bool,
    quantity: i32,
    condition: Condition,
}

impl Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, (RegisterName, bool, i32, ConditionParams)> {
        map(
            tuple((
                Solution::parse_register_name,
                tag(" "),
                alt((map(tag("inc"), |_| true), map(tag("dec"), |_| false))),
                tag(" "),
                parse_integer::<i32>,
                tag(" "),
                Condition::parse,
            )),
            |(register_name, _, increment, _, quantity, _, condition)| {
                (register_name, increment, quantity, condition)
            },
        )(input)
    }

    fn new(
        register_name: RegisterName,
        increment: bool,
        quantity: i32,
        condition: ConditionParams,
        registers: &mut RegisterNameList,
    ) -> Self {
        let register_index: RegisterIndex = registers.find_or_add_index(&register_name);
        let condition: Condition = Condition::new(condition, registers);

        Self {
            register_index,
            increment,
            quantity,
            condition,
        }
    }

    fn process(&self, registers: &mut [i32]) -> Option<i32> {
        self.condition.evaluate(registers).then(|| {
            let register: &mut i32 = &mut registers[self.register_index.get()];
            let register_value: i32 =
                *register + if self.increment { 1_i32 } else { -1_i32 } * self.quantity;

            *register = register_value;

            register_value
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    registers: RegisterNameList,
    instructions: Vec<Instruction>,
}

impl Solution {
    const REGISTER_NAME_LEN: usize = 3_usize;

    fn parse_register_name<'i>(input: &'i str) -> IResult<&'i str, RegisterName> {
        RegisterName::parse_char1(1_usize, |c| c.is_ascii_lowercase())(input)
    }

    fn process(&self) -> Option<(i32, Vec<i32>)> {
        let mut registers: Vec<i32> = vec![0_i32; self.registers.as_slice().len()];

        self.instructions
            .iter()
            .filter_map(|instruction| instruction.process(&mut registers))
            .max()
            .map(|max| (max, registers))
    }

    fn largest_value_post_process(&self) -> Option<i32> {
        self.process()
            .map(|(_, registers)| registers.into_iter().max())
            .flatten()
    }

    fn largest_value(&self) -> Option<i32> {
        self.process()
            .map(|(largest_value, _)| largest_value.max(0_i32))
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut registers: RegisterNameList = RegisterNameList::new();

        let (input, instructions): (&str, Vec<Instruction>) = many0(terminated(
            map(
                Instruction::parse,
                |(register_name, increment, quantity, condition)| {
                    Instruction::new(
                        register_name,
                        increment,
                        quantity,
                        condition,
                        &mut registers,
                    )
                },
            ),
            opt(line_ending),
        ))(input)?;

        Ok((
            input,
            Self {
                registers,
                instructions,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// It was fun putting together the table type for this. The parsing functions also look really
    /// clean. I could've simplified the quantity and increment storage to just be a delta, but I
    /// didn't know what part 2 would look like.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.largest_value_post_process());
    }

    /// By only comparing the most recently modified value, we don't need to run a max operation on
    /// the whole list of values each time. That said, this is probably fast enough where it
    /// wouldn't be noticeably slower for the size of our user input on modern hardware.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.largest_value());
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
    use {super::*, std::sync::OnceLock, Operator::*};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        b inc 5 if a > 1\n\
        a inc 1 if b < 5\n\
        c dec -10 if a >= 1\n\
        c inc -20 if c == 10\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! solution {
            {
                [ $( $register_name:expr, )* ],
                [ $( {
                    $register_index:expr,
                    $increment:expr,
                    $quantity:expr,
                    {
                        $condition_register_index:expr,
                        $operator:expr,
                        $comparand:expr,
                    },
                }, )* ]
            } => { Solution {
                registers: RegisterNameList::try_from(
                    vec![ $( $register_name, )* ]
                        .into_iter()
                        .map(|register_name| RegisterName::try_from(register_name).unwrap())
                        .collect::<Vec<_>>())
                    .unwrap(),
                instructions: vec![ $( Instruction {
                    register_index: $register_index.into(),
                    increment: $increment,
                    quantity: $quantity,
                    condition: Condition {
                        register_index: $condition_register_index.into(),
                        operator: $operator,
                        comparand: $comparand,
                    },
                }, )* ]
            } }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![solution! {
                [ "b", "a", "c", ],
                [
                    { 0, true, 5, { 1, Gt, 1, }, },
                    { 1, true, 1, { 0, Lt, 5, }, },
                    { 2, false, -10, { 1, Ge, 1, }, },
                    { 2, true, -20, { 2, Eq, 10, }, },
                ]
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
    fn test_process() {
        for (index, registers_over_time) in [&[
            &[0_i32, 0_i32, 0_i32],
            &[0_i32, 1_i32, 0_i32],
            &[0_i32, 1_i32, 10_i32],
            &[0_i32, 1_i32, -10_i32],
        ]]
        .into_iter()
        .enumerate()
        {
            let mut registers: Vec<i32> = vec![0_i32; registers_over_time[0_usize].len()];

            for (instruction, expected_registers) in
                solution(index).instructions.iter().zip(registers_over_time)
            {
                instruction.process(&mut registers);

                assert_eq!(&registers, expected_registers);
            }
        }
    }

    #[test]
    fn test_largest_value_post_process() {
        for (index, largest_value_post_process) in [Some(1_i32)].into_iter().enumerate() {
            assert_eq!(
                solution(index).largest_value_post_process(),
                largest_value_post_process
            );
        }
    }

    #[test]
    fn test_largest_value() {
        for (index, largest_value) in [Some(10_i32)].into_iter().enumerate() {
            assert_eq!(solution(index).largest_value(), largest_value);
        }
    }
}
