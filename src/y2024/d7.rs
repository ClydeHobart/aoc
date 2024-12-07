use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_res, opt, success},
        error::Error,
        multi::{many0, many_m_n},
        sequence::{preceded, separated_pair, terminated},
        Err, IResult,
    },
    rayon::iter::{IntoParallelRefIterator, ParallelIterator},
    std::ops::Range,
};

#[cfg(test)]
use std::fmt::{Debug, Formatter, Result as FmtResult};

/* --- Day 7: Bridge Repair ---

The Historians take you to a familiar rope bridge over a river in the middle of a jungle. The Chief isn't on this side of the bridge, though; maybe he's on the other side?

When you go to cross the bridge, you notice a group of engineers trying to repair it. (Apparently, it breaks pretty frequently.) You won't be able to cross until it's fixed.

You ask how long it'll take; the engineers tell you that it only needs final calibrations, but some young elephants were playing nearby and stole all the operators from their calibration equations! They could finish the calibrations if only someone could determine which test values could possibly be produced by placing any combination of operators into their calibration equations (your puzzle input).

For example:

190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20

Each line represents a single equation. The test value appears before the colon on each line; it is your job to determine whether the remaining numbers can be combined with operators to produce the test value.

Operators are always evaluated left-to-right, not according to precedence rules. Furthermore, numbers in the equations cannot be rearranged. Glancing into the jungle, you can see elephants holding two different types of operators: add (+) and multiply (*).

Only three of the above equations can be made true by inserting operators:

    190: 10 19 has only one position that accepts an operator: between 10 and 19. Choosing + would give 29, but choosing * would give the test value (10 * 19 = 190).
    3267: 81 40 27 has two positions for operators. Of the four possible configurations of the operators, two cause the right side to match the test value: 81 + 40 * 27 and 81 * 40 + 27 both equal 3267 (when evaluated left-to-right)!
    292: 11 6 16 20 can be solved in exactly one way: 11 + 6 * 16 + 20.

The engineers just need the total calibration result, which is the sum of the test values from just the equations that could possibly be true. In the above example, the sum of the test values for the three equations listed above is 3749.

Determine which equations could possibly be true. What is their total calibration result?

--- Part Two ---

The engineers seem concerned; the total calibration result you gave them is nowhere close to being within safety tolerances. Just then, you spot your mistake: some well-hidden elephants are holding a third type of operator.

The concatenation operator (||) combines the digits from its left and right inputs into a single number. For example, 12 || 345 would become 12345. All operators are still evaluated left-to-right.

Now, apart from the three equations that could be made true using only addition and multiplication, the above example has three more equations that can be made true by inserting operators:

    156: 15 6 can be made true through a single concatenation: 15 || 6 = 156.
    7290: 6 8 6 15 can be made true using 6 * 8 || 6 * 15.
    192: 17 8 14 can be made true using 17 || 8 + 14.

Adding up all six test values (the three that could be made before using only + and * plus the new three that can now be made by also using ||) produces the new total calibration result of 11387.

Using your new knowledge of elephant hiding spots, determine which equations could possibly be true. What is their total calibration result? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, PartialEq)]
enum Operator {
    #[default]
    Add,
    Mul,
    Concat,
}

impl Operator {
    fn try_next(self, enable_concat: bool) -> Option<Self> {
        match self {
            Self::Add => Some(Self::Mul),
            Self::Mul => enable_concat.then_some(Self::Concat),
            Self::Concat => None,
        }
    }
}

type OperatorBitArray = BitArr!(for Equation::MAX_OPERATORS_LEN, in u16);

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Copy, Default)]
struct Operators {
    is_mul: OperatorBitArray,
    is_concat: OperatorBitArray,
}

impl Operators {
    #[cfg(test)]
    fn from_iter<I: IntoIterator<Item = Operator>>(operators_iter: I) -> Self {
        let mut operators: Self = Self::default();

        for (index, operator) in operators_iter.into_iter().enumerate() {
            operators.set(index, operator);
        }

        operators
    }

    fn get(self, index: usize) -> Operator {
        match (self.is_mul[index], self.is_concat[index]) {
            (false, false) => Operator::Add,
            (true, false) => Operator::Mul,
            (false, true) => Operator::Concat,
            (true, true) => unreachable!(),
        }
    }

    fn set(&mut self, index: usize, operator: Operator) {
        self.is_mul.set(index, operator == Operator::Mul);
        self.is_concat.set(index, operator == Operator::Concat);
    }
}

#[cfg(test)]
impl Debug for Operators {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut operators: [Operator; Equation::MAX_OPERATORS_LEN] =
            [Operator::default(); Equation::MAX_OPERATORS_LEN];

        for index in 0_usize..Equation::MAX_OPERATORS_LEN {
            operators[index] = self.get(index);
        }

        f.debug_tuple("Operators").field(&operators).finish()
    }
}

struct IterOperators {
    operators: Operators,
    index: usize,
    len: usize,
    enable_concat: bool,
}

impl IterOperators {
    fn new(len: usize, enable_concat: bool) -> Self {
        Self {
            operators: Operators::default(),
            index: 0_usize,
            len,
            enable_concat,
        }
    }
}

impl Iterator for IterOperators {
    type Item = Operators;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.len).then(|| {
            let next: Self::Item = self.operators;

            let mut carry_index: usize = 0_usize;
            let mut should_continue: bool = true;

            while should_continue && self.index < self.len {
                if let Some(operator) = self.operators.get(self.index).try_next(self.enable_concat)
                {
                    self.operators.set(self.index, operator);
                    should_continue = false;
                } else {
                    self.index += 1_usize;
                    carry_index = self.index
                }
            }

            for index in 0_usize..carry_index {
                self.operators.set(index, Operator::default());
            }

            if self.index < self.len {
                self.index = 0_usize
            }

            next
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Equation {
    test_value: u64,
    operand_range: Range<u16>,
}

impl Equation {
    const MIN_OPERANDS_LEN: usize = 2_usize;
    const MAX_OPERATORS_LEN: usize = u16::BITS as usize;
    const MAX_OPERANDS_LEN: usize = Self::MAX_OPERATORS_LEN + 1_usize;

    fn parse<'o, 'i: 'o>(
        operands: &'o mut Vec<u64>,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Self> + 'o {
        map(
            separated_pair(parse_integer, tag(":"), |input: &'i str| {
                let operand_range_start: u16 =
                    map_res(success(()), |_| operands.len().try_into())(input)?.1;
                let input: &str = many_m_n(
                    Self::MIN_OPERANDS_LEN,
                    Self::MAX_OPERANDS_LEN,
                    preceded(
                        tag(" "),
                        map(parse_integer, |operand| {
                            operands.push(operand);
                        }),
                    ),
                )(input)?
                .0;
                let operand_range_end: u16 =
                    map_res(success(()), |_| operands.len().try_into())(input)?.1;

                Ok((input, operand_range_start..operand_range_end))
            }),
            |(test_value, operand_range)| Self {
                test_value,
                operand_range,
            },
        )
    }

    fn next_largest_power_of_ten(value: u64) -> u64 {
        // There are more, but our input doesn't require more than this.
        const POWERS_OF_TEN: &'static [u64] = &[1_u64, 10_u64, 100_u64, 1000_u64];

        let digits: usize = if value == 0_u64 {
            1_usize
        } else {
            value.ilog10() as usize + 1_usize
        };

        POWERS_OF_TEN[digits]
    }

    fn try_find_operators(&self, operands: &[u64], enable_concat: bool) -> Option<Operators> {
        IterOperators::new(self.operand_range.len() - 1_usize, enable_concat)
            .filter_map(|operators| {
                operands[self.operand_range.as_range_usize()]
                    .iter()
                    .copied()
                    .skip(1_usize)
                    .enumerate()
                    .try_fold(
                        operands[self.operand_range.start as usize] as u64,
                        |value, (index, operand)| {
                            match operators.get(index) {
                                Operator::Add => value.checked_add(operand as u64),
                                Operator::Mul => value.checked_mul(operand as u64),
                                Operator::Concat => value
                                    .checked_mul(Self::next_largest_power_of_ten(operand))
                                    .and_then(|value| value.checked_add(operand as u64)),
                            }
                            .filter(|value| *value <= self.test_value)
                        },
                    )
                    .map_or(false, |value| value == self.test_value)
                    .then_some(operators)
            })
            .next()
    }

    fn is_possible(&self, operands: &[u64], enable_concat: bool) -> bool {
        self.try_find_operators(operands, enable_concat).is_some()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    operands: Vec<u64>,
    equations: Vec<Equation>,
}

impl Solution {
    fn total_calibration_result(&self, enable_concat: bool) -> u64 {
        self.equations
            .par_iter()
            .filter_map(|equation| {
                equation
                    .is_possible(&self.operands, enable_concat)
                    .then_some(equation.test_value)
            })
            .sum()
    }

    fn total_calibration_result_without_concat(&self) -> u64 {
        self.total_calibration_result(false)
    }

    fn total_calibration_result_with_concat(&self) -> u64 {
        self.total_calibration_result(true)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut operands: Vec<u64> = Vec::new();

        let (input, equations): (&str, Vec<Equation>) =
            many0(terminated(Equation::parse(&mut operands), opt(line_ending)))(input)?;

        Ok((
            input,
            Self {
                operands,
                equations,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// Surprised there weren't more kinks to iron out in this one.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_calibration_result_without_concat());
    }

    /// Rayon saved this one.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_calibration_result_with_concat());
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
        190: 10 19\n\
        3267: 81 40 27\n\
        83: 17 5\n\
        156: 15 6\n\
        7290: 6 8 6 15\n\
        161011: 16 10 13\n\
        192: 17 8 14\n\
        21037: 9 7 18 13\n\
        292: 11 6 16 20\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                operands: vec![
                    10_u64, 19_u64, 81_u64, 40_u64, 27_u64, 17_u64, 5_u64, 15_u64, 6_u64, 6_u64,
                    8_u64, 6_u64, 15_u64, 16_u64, 10_u64, 13_u64, 17_u64, 8_u64, 14_u64, 9_u64,
                    7_u64, 18_u64, 13_u64, 11_u64, 6_u64, 16_u64, 20_u64,
                ],
                equations: vec![
                    Equation {
                        test_value: 190_u64,
                        operand_range: 0_u16..2_u16,
                    },
                    Equation {
                        test_value: 3267_u64,
                        operand_range: 2_u16..5_u16,
                    },
                    Equation {
                        test_value: 83_u64,
                        operand_range: 5_u16..7_u16,
                    },
                    Equation {
                        test_value: 156_u64,
                        operand_range: 7_u16..9_u16,
                    },
                    Equation {
                        test_value: 7290_u64,
                        operand_range: 9_u16..13_u16,
                    },
                    Equation {
                        test_value: 161011_u64,
                        operand_range: 13_u16..16_u16,
                    },
                    Equation {
                        test_value: 192_u64,
                        operand_range: 16_u16..19_u16,
                    },
                    Equation {
                        test_value: 21037_u64,
                        operand_range: 19_u16..23_u16,
                    },
                    Equation {
                        test_value: 292_u64,
                        operand_range: 23_u16..27_u16,
                    },
                ],
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
    fn test_try_find_operators_without_concat() {
        use Operator::*;

        for (index, operators_list) in [vec![
            Some(Operators::from_iter([Mul])),
            Some(Operators::from_iter([Mul, Add])),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(Operators::from_iter([Add, Mul, Add])),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (equation, operators) in solution.equations.iter().zip(operators_list) {
                assert_eq!(
                    equation.try_find_operators(&solution.operands, false),
                    operators
                );
            }
        }
    }

    #[test]
    fn test_total_calibration_result_without_concat() {
        for (index, total_calibration_result_without_concat) in [3749_u64].into_iter().enumerate() {
            assert_eq!(
                solution(index).total_calibration_result_without_concat(),
                total_calibration_result_without_concat
            );
        }
    }

    #[test]
    fn test_iter_operators() {
        use Operator::{Add, Concat as Con, Mul};

        assert_eq!(
            IterOperators::new(3_usize, false).count(),
            2_usize.pow(3_u32)
        );

        for (actual_operators, expected_operators) in IterOperators::new(3_usize, false).zip([
            Operators::from_iter([Add, Add, Add]),
            Operators::from_iter([Mul, Add, Add]),
            Operators::from_iter([Add, Mul, Add]),
            Operators::from_iter([Mul, Mul, Add]),
            Operators::from_iter([Add, Add, Mul]),
            Operators::from_iter([Mul, Add, Mul]),
            Operators::from_iter([Add, Mul, Mul]),
            Operators::from_iter([Mul, Mul, Mul]),
        ]) {
            assert_eq!(actual_operators, expected_operators);
        }

        assert_eq!(
            IterOperators::new(3_usize, true).count(),
            3_usize.pow(3_u32)
        );

        for (actual_operators, expected_operators) in IterOperators::new(3_usize, true).zip([
            Operators::from_iter([Add, Add, Add]),
            Operators::from_iter([Mul, Add, Add]),
            Operators::from_iter([Con, Add, Add]),
            Operators::from_iter([Add, Mul, Add]),
            Operators::from_iter([Mul, Mul, Add]),
            Operators::from_iter([Con, Mul, Add]),
            Operators::from_iter([Add, Con, Add]),
            Operators::from_iter([Mul, Con, Add]),
            Operators::from_iter([Con, Con, Add]),
            Operators::from_iter([Add, Add, Mul]),
            Operators::from_iter([Mul, Add, Mul]),
            Operators::from_iter([Con, Add, Mul]),
            Operators::from_iter([Add, Mul, Mul]),
            Operators::from_iter([Mul, Mul, Mul]),
            Operators::from_iter([Con, Mul, Mul]),
            Operators::from_iter([Add, Con, Mul]),
            Operators::from_iter([Mul, Con, Mul]),
            Operators::from_iter([Con, Con, Mul]),
            Operators::from_iter([Add, Add, Con]),
            Operators::from_iter([Mul, Add, Con]),
            Operators::from_iter([Con, Add, Con]),
            Operators::from_iter([Add, Mul, Con]),
            Operators::from_iter([Mul, Mul, Con]),
            Operators::from_iter([Con, Mul, Con]),
            Operators::from_iter([Add, Con, Con]),
            Operators::from_iter([Mul, Con, Con]),
            Operators::from_iter([Con, Con, Con]),
        ]) {
            assert_eq!(actual_operators, expected_operators);
        }
    }

    #[test]
    fn test_try_find_operators_with_concat() {
        use Operator::*;

        for (index, operators_list) in [vec![
            Some(Operators::from_iter([Mul])),
            Some(Operators::from_iter([Mul, Add])),
            None,
            Some(Operators::from_iter([Concat])),
            Some(Operators::from_iter([Mul, Concat, Mul])),
            None,
            Some(Operators::from_iter([Concat, Add])),
            None,
            Some(Operators::from_iter([Add, Mul, Add])),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (equation, operators) in solution.equations.iter().zip(operators_list) {
                assert_eq!(
                    equation.try_find_operators(&solution.operands, true),
                    operators
                );
            }
        }
    }

    #[test]
    fn test_total_calibration_result_with_concat() {
        for (index, total_calibration_result_with_concat) in [11387_u64].into_iter().enumerate() {
            assert_eq!(
                solution(index).total_calibration_result_with_concat(),
                total_calibration_result_with_concat
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
