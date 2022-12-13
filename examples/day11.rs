use {
    aoc_2022::*,
    std::{
        num::ParseIntError,
        str::{FromStr, Split},
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    Old,
    Constant(u64),
}

impl Operand {
    fn evaluate(&self, old: u64) -> u64 {
        match self {
            Self::Old => old,
            Self::Constant(constant) => *constant,
        }
    }
}

#[derive(Debug, PartialEq)]
struct InvalidOperand(ParseIntError);

impl<'s> TryFrom<&'s str> for Operand {
    type Error = InvalidOperand;

    fn try_from(operand_str: &'s str) -> Result<Self, Self::Error> {
        use Operand::*;

        if operand_str == "old" {
            Ok(Old)
        } else {
            u64::from_str(operand_str)
                .map(Constant)
                .map_err(InvalidOperand)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Add,
    Multiply,
}

#[derive(Debug, PartialEq)]
struct InvalidOperation<'s>(&'s str);

impl<'s> TryFrom<&'s str> for Operation {
    type Error = InvalidOperation<'s>;

    fn try_from(operation_str: &'s str) -> Result<Self, Self::Error> {
        use Operation::*;

        Ok(match operation_str {
            "+" => Add,
            "*" => Multiply,
            invalid_operation_str => Err(InvalidOperation(invalid_operation_str))?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Expression {
    lhs: Operand,
    rhs: Operand,
    op: Operation,
}

impl Expression {
    #[cfg(test)]
    fn new(lhs: Operand, op: Operation, rhs: Operand) -> Self {
        Self { lhs, rhs, op }
    }

    fn call(&self, old: u64) -> u64 {
        let lhs: u64 = self.lhs.evaluate(old);
        let rhs: u64 = self.rhs.evaluate(old);

        match self.op {
            Operation::Add => lhs + rhs,
            Operation::Multiply => lhs * rhs,
        }
    }
}

#[derive(Debug, PartialEq)]
enum ExpressionParseError<'s> {
    NoLhsToken,
    FailedToParseLhs(ParseIntError),
    NoOperationToken,
    InvalidOperationToken(&'s str),
    NoRhsToken,
    FailedToParseRhs(ParseIntError),
    ExtraTokenFound(&'s str),
}

impl<'s> TryFrom<&'s str> for Expression {
    type Error = ExpressionParseError<'s>;

    fn try_from(expression_str: &'s str) -> Result<Self, Self::Error> {
        use ExpressionParseError::*;

        let mut expression_token_iter: Split<char> = expression_str.split(' ');

        let lhs: Operand = match expression_token_iter.next() {
            None => Err(NoLhsToken),
            Some(lhs_str) => lhs_str
                .try_into()
                .map_err(|invalid_operand: InvalidOperand| FailedToParseLhs(invalid_operand.0)),
        }?;
        let op: Operation = match expression_token_iter.next() {
            None => Err(NoOperationToken),
            Some(operation_str) => {
                operation_str
                    .try_into()
                    .map_err(|invalid_operation: InvalidOperation| {
                        InvalidOperationToken(invalid_operation.0)
                    })
            }
        }?;
        let rhs: Operand = match expression_token_iter.next() {
            None => Err(NoRhsToken),
            Some(rhs_str) => rhs_str
                .try_into()
                .map_err(|invalid_operand: InvalidOperand| FailedToParseRhs(invalid_operand.0)),
        }?;

        match expression_token_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(Self { lhs, rhs, op }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MonkeyMemo {
    items: Vec<u64>,
    operation: Expression,
    test_divisor: u64,
    if_true: usize,
    if_false: usize,
    inspected: usize,
}

fn inspect_and_throw_internal<F: Fn(&MonkeyMemo, u64) -> u64>(
    monkey_memo: &mut MonkeyMemo,
    items_and_recipients: &mut Vec<(u64, usize)>,
    f: F,
) {
    items_and_recipients.clear();

    for item in monkey_memo.items.iter() {
        let item: u64 = f(monkey_memo, *item);

        items_and_recipients.push((
            item,
            if item % monkey_memo.test_divisor == 0_u64 {
                monkey_memo.if_true
            } else {
                monkey_memo.if_false
            },
        ));
    }

    monkey_memo.items.clear();
    monkey_memo.inspected += items_and_recipients.len();
}

impl MonkeyMemo {
    fn compute_new_worry_with_division(&self, item: u64) -> u64 {
        self.operation.call(item) / 3_u64
    }

    fn inspect_and_throw(&mut self, items_and_recipients: &mut Vec<(u64, usize)>) {
        inspect_and_throw_internal(
            self,
            items_and_recipients,
            Self::compute_new_worry_with_division,
        );
    }

    fn compute_new_worry(&self, item: u64) -> u64 {
        self.operation.call(item)
    }

    fn inspect_and_throw_with_worry(&mut self, items_and_recipients: &mut Vec<(u64, usize)>) {
        inspect_and_throw_internal(self, items_and_recipients, Self::compute_new_worry);
    }

    #[cfg(test)]
    fn items(&self) -> Vec<u64> {
        self.items.clone()
    }

    fn inspected(&self) -> usize {
        self.inspected
    }
}

fn validate_prefix_and_suffix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    suffix: &str,
    f: F,
) -> Result<&'s str, E> {
    if value.len() >= prefix.len() + suffix.len()
        && value.get(..prefix.len()).map_or(false, |p| p == prefix)
        && value
            .get(value.len() - suffix.len()..)
            .map_or(false, |s| s == suffix)
    {
        Ok(&value[prefix.len()..value.len() - suffix.len()])
    } else {
        Err(f(value))
    }
}

fn validate_prefix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    f: F,
) -> Result<&'s str, E> {
    validate_prefix_and_suffix(value, prefix, "", f)
}

#[derive(Debug, PartialEq)]
enum MonkeyMemoParseError<'s> {
    NoStartingItemsToken,
    InvalidStartingItemsToken(&'s str),
    FailedToParseItem(ParseIntError),
    NoOperationToken,
    InvalidOperationToken(&'s str),
    FailedToParseOperation(ExpressionParseError<'s>),
    NoTestToken,
    InvalidTestToken(&'s str),
    FailedToParseDivisor(ParseIntError),
    NoIfTrueToken,
    InvalidIfTrueToken(&'s str),
    FailedToParseIfTrueTarget(ParseIntError),
    NoIfFalseToken,
    InvalidIfFalseToken(&'s str),
    FailedToParseIfFalseTarget(ParseIntError),
    ExtraTokenFound(&'s str),
}

impl<'s> TryFrom<Split<'s, char>> for MonkeyMemo {
    type Error = MonkeyMemoParseError<'s>;

    fn try_from(mut monkey_memo_line_iter: Split<'s, char>) -> Result<Self, Self::Error> {
        use Error::*;
        use MonkeyMemoParseError as Error;

        let items: Vec<u64> = match monkey_memo_line_iter.next() {
            None => Err(NoStartingItemsToken),
            Some(starting_items_str) => {
                let mut items: Vec<u64> = Vec::new();

                for item_token in validate_prefix(
                    starting_items_str,
                    "  Starting items: ",
                    InvalidStartingItemsToken,
                )?
                .split(", ")
                {
                    items.push(u64::from_str(item_token).map_err(FailedToParseItem)?);
                }

                Ok(items)
            }
        }?;
        let operation: Expression = match monkey_memo_line_iter.next() {
            None => Err(NoOperationToken),
            Some(operation_str) => {
                validate_prefix(operation_str, "  Operation: new = ", InvalidOperationToken)?
                    .try_into()
                    .map_err(FailedToParseOperation)
            }
        }?;
        let test_divisor: u64 = match monkey_memo_line_iter.next() {
            None => Err(NoTestToken),
            Some(test_divisor_str) => u64::from_str(validate_prefix(
                test_divisor_str,
                "  Test: divisible by ",
                InvalidTestToken,
            )?)
            .map_err(FailedToParseDivisor),
        }?;
        let if_true: usize = match monkey_memo_line_iter.next() {
            None => Err(NoIfTrueToken),
            Some(if_true_str) => usize::from_str(validate_prefix(
                if_true_str,
                "    If true: throw to monkey ",
                InvalidIfTrueToken,
            )?)
            .map_err(FailedToParseIfTrueTarget),
        }?;
        let if_false: usize = match monkey_memo_line_iter.next() {
            None => Err(NoIfFalseToken),
            Some(if_false_str) => usize::from_str(validate_prefix(
                if_false_str,
                "    If false: throw to monkey ",
                InvalidIfFalseToken,
            )?)
            .map_err(FailedToParseIfFalseTarget),
        }?;

        match monkey_memo_line_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(Self {
                items,
                operation,
                test_divisor,
                if_true,
                if_false,
                inspected: 0_usize,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MonkeyNotes {
    memos: Vec<MonkeyMemo>,
    modulus: u64,
}

impl MonkeyNotes {
    #[inline]
    fn item(&self, item: u64) -> u64 {
        item
    }

    #[inline]
    fn item_with_modulus(&self, item: u64) -> u64 {
        item % self.modulus
    }

    fn run_rounds_internal<
        I: Fn(&mut MonkeyMemo, &mut Vec<(u64, usize)>),
        M: Fn(&MonkeyNotes, u64) -> u64,
    >(
        &mut self,
        rounds: usize,
        inspect_and_throw: I,
        modify_item: M,
    ) {
        let mut items_and_recipients: Vec<(u64, usize)> = Vec::new();

        for _ in 0_usize..rounds {
            for monkey_memo_index in 0_usize..self.memos.len() {
                inspect_and_throw(
                    &mut self.memos[monkey_memo_index],
                    &mut items_and_recipients,
                );

                for (item, monkey_memo_index) in items_and_recipients.iter() {
                    let item: u64 = modify_item(self, *item);

                    self.memos[*monkey_memo_index].items.push(item);
                }

                items_and_recipients.clear();
            }
        }
    }

    fn run_rounds(&mut self, rounds: usize) {
        self.run_rounds_internal(rounds, MonkeyMemo::inspect_and_throw, Self::item);
    }

    fn run_rounds_with_worry_and_modulus(&mut self, rounds: usize) {
        self.run_rounds_internal(
            rounds,
            MonkeyMemo::inspect_and_throw_with_worry,
            Self::item_with_modulus,
        );
    }

    #[cfg(test)]
    fn items(&self) -> Vec<Vec<u64>> {
        self.memos.iter().map(MonkeyMemo::items).collect()
    }

    fn inspected_counts(&self) -> Vec<usize> {
        self.memos.iter().map(MonkeyMemo::inspected).collect()
    }

    fn monkey_business(&self) -> usize {
        match self.memos.len() {
            0_usize => 0_usize,
            1_usize => self.memos[0_usize].inspected,
            _ => {
                let mut inspected_counts: Vec<usize> = self.inspected_counts();

                inspected_counts.sort_by(|a, b| a.cmp(b).reverse());

                inspected_counts[0_usize] * inspected_counts[1_usize]
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum MonkeyNotesParseError<'s> {
    NoMonkeyIndexToken,
    InvalidMonkeyIndexToken(&'s str),
    FailedToParseMonkeyIndex(ParseIntError),
    IncorrectMonkeyIndex { expected: usize, found: usize },
    FailedToParseMonkeyMemo(MonkeyMemoParseError<'s>),
    InvalidIfTrueTarget { monkey_count: usize, target: usize },
    InvalidIfFalseTarget { monkey_count: usize, target: usize },
}

impl<'s> TryFrom<&'s str> for MonkeyNotes {
    type Error = MonkeyNotesParseError<'s>;

    fn try_from(monkey_notes_str: &'s str) -> Result<Self, Self::Error> {
        use Error::*;
        use MonkeyNotesParseError as Error;

        let mut monkey_notes: MonkeyNotes = MonkeyNotes {
            memos: Vec::new(),
            modulus: 1_u64,
        };

        for (expected, monkey_memo_str) in monkey_notes_str.split("\n\n").enumerate() {
            let mut monkey_memo_line_iter: Split<char> = monkey_memo_str.split('\n');

            let found: usize = usize::from_str(validate_prefix_and_suffix(
                monkey_memo_line_iter.next().ok_or(NoMonkeyIndexToken)?,
                "Monkey ",
                ":",
                InvalidMonkeyIndexToken,
            )?)
            .map_err(FailedToParseMonkeyIndex)?;

            if found != expected {
                Err(IncorrectMonkeyIndex { expected, found })?;
            } else {
                monkey_notes.memos.push(
                    monkey_memo_line_iter
                        .try_into()
                        .map_err(FailedToParseMonkeyMemo)?,
                );

                let test_divisor: u64 = monkey_notes.memos.last().unwrap().test_divisor;

                if monkey_notes.modulus % test_divisor != 0_u64 {
                    monkey_notes.modulus *= test_divisor;
                }
            }
        }

        let monkey_count: usize = monkey_notes.memos.len();

        for monkey_memo in monkey_notes.memos.iter() {
            if monkey_memo.if_true >= monkey_count {
                return Err(InvalidIfTrueTarget {
                    monkey_count,
                    target: monkey_memo.if_true,
                });
            }

            if monkey_memo.if_false >= monkey_count {
                return Err(InvalidIfFalseTarget {
                    monkey_count,
                    target: monkey_memo.if_false,
                });
            }
        }

        Ok(monkey_notes)
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day11.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match MonkeyNotes::try_from(input) {
                    Ok(mut monkey_notes) => {
                        let mut monkey_notes_with_worry_and_modulus: MonkeyNotes =
                            monkey_notes.clone();

                        monkey_notes.run_rounds(20_usize);

                        let monkey_business: usize = monkey_notes.monkey_business();

                        monkey_notes_with_worry_and_modulus
                            .run_rounds_with_worry_and_modulus(10_000_usize);

                        let monkey_business_with_worry_and_modulus: usize =
                            monkey_notes_with_worry_and_modulus.monkey_business();

                        println!(
                            "monkey_business == {monkey_business}\n\
                            monkey_business_with_worry_and_modulus == \
                            {monkey_business_with_worry_and_modulus}"
                        );
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                }
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MONKEY_NOTES_STR: &str = concat!(
        "Monkey 0:\n",
        "  Starting items: 79, 98\n",
        "  Operation: new = old * 19\n",
        "  Test: divisible by 23\n",
        "    If true: throw to monkey 2\n",
        "    If false: throw to monkey 3\n",
        "\n",
        "Monkey 1:\n",
        "  Starting items: 54, 65, 75, 74\n",
        "  Operation: new = old + 6\n",
        "  Test: divisible by 19\n",
        "    If true: throw to monkey 2\n",
        "    If false: throw to monkey 0\n",
        "\n",
        "Monkey 2:\n",
        "  Starting items: 79, 60, 97\n",
        "  Operation: new = old * old\n",
        "  Test: divisible by 13\n",
        "    If true: throw to monkey 1\n",
        "    If false: throw to monkey 3\n",
        "\n",
        "Monkey 3:\n",
        "  Starting items: 74\n",
        "  Operation: new = old + 3\n",
        "  Test: divisible by 17\n",
        "    If true: throw to monkey 0\n",
        "    If false: throw to monkey 1"
    );

    #[test]
    fn test_monkey_notes_try_from_str() {
        assert_eq!(MONKEY_NOTES_STR.try_into(), Ok(example_monkey_notes()));
    }

    #[test]
    fn test_run_rounds() {
        let mut monkey_notes: MonkeyNotes = example_monkey_notes();

        assert_eq!(
            (0_usize..10_usize)
                .into_iter()
                .map(|_| -> Vec<Vec<u64>> {
                    monkey_notes.run_rounds(1_usize);

                    monkey_notes.items()
                })
                .collect::<Vec<Vec<Vec<u64>>>>(),
            vec![
                vec![
                    vec![20, 23, 27, 26],
                    vec![2080, 25, 167, 207, 401, 1046],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![695, 10, 71, 135, 350],
                    vec![43, 49, 58, 55, 362],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![16, 18, 21, 20, 122],
                    vec![1468, 22, 150, 286, 739],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![491, 9, 52, 97, 248, 34],
                    vec![39, 45, 43, 258],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![15, 17, 16, 88, 1037],
                    vec![20, 110, 205, 524, 72],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![8, 70, 176, 26, 34],
                    vec![481, 32, 36, 186, 2190],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![162, 12, 14, 64, 732, 17],
                    vec![148, 372, 55, 72],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![51, 126, 20, 26, 136],
                    vec![343, 26, 30, 1546, 36],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![116, 10, 12, 517, 14],
                    vec![108, 267, 43, 55, 288],
                    vec![],
                    vec![],
                ],
                vec![
                    vec![91, 16, 20, 98],
                    vec![481, 245, 22, 26, 1092, 30],
                    vec![],
                    vec![],
                ],
            ]
        );

        monkey_notes.run_rounds(5_usize);

        assert_eq!(
            monkey_notes.items(),
            vec![
                vec![83, 44, 8, 184, 9, 20, 26, 102],
                vec![110, 36],
                vec![],
                vec![]
            ]
        );

        monkey_notes.run_rounds(5_usize);

        assert_eq!(
            monkey_notes.items(),
            vec![
                vec![10, 12, 14, 26, 34],
                vec![245, 93, 53, 199, 115],
                vec![],
                vec![]
            ]
        );
    }

    #[test]
    fn test_monkey_business() {
        let mut monkey_notes: MonkeyNotes = example_monkey_notes();

        monkey_notes.run_rounds(20_usize);

        assert_eq!(monkey_notes.monkey_business(), 10605_usize);
    }

    #[test]
    fn test_run_rounds_with_worry_and_modulus() {
        let mut monkey_notes: MonkeyNotes = example_monkey_notes();

        monkey_notes.run_rounds_with_worry_and_modulus(1_usize);

        assert_eq!(monkey_notes.inspected_counts(), vec![2, 4, 3, 6]);

        monkey_notes.run_rounds_with_worry_and_modulus(19_usize);

        assert_eq!(monkey_notes.inspected_counts(), vec![99, 97, 8, 103]);

        let mut monkey_notes: MonkeyNotes = example_monkey_notes();

        assert_eq!(
            (0_usize..10_usize)
                .into_iter()
                .map(|_| -> Vec<usize> {
                    monkey_notes.run_rounds_with_worry_and_modulus(1_000_usize);

                    monkey_notes.inspected_counts()
                })
                .collect::<Vec<Vec<usize>>>(),
            vec![
                vec![5204, 4792, 199, 5192],
                vec![10419, 9577, 392, 10391],
                vec![15638, 14358, 587, 15593],
                vec![20858, 19138, 780, 20797],
                vec![26075, 23921, 974, 26000],
                vec![31294, 28702, 1165, 31204],
                vec![36508, 33488, 1360, 36400],
                vec![41728, 38268, 1553, 41606],
                vec![46945, 43051, 1746, 46807],
                vec![52166, 47830, 1938, 52013]
            ]
        );
    }

    fn example_monkey_notes() -> MonkeyNotes {
        use {Operand::*, Operation::*};

        MonkeyNotes {
            memos: vec![
                MonkeyMemo {
                    items: vec![79, 98],
                    operation: Expression::new(Old, Multiply, Constant(19)),
                    test_divisor: 23,
                    if_true: 2,
                    if_false: 3,
                    inspected: 0,
                },
                MonkeyMemo {
                    items: vec![54, 65, 75, 74],
                    operation: Expression::new(Old, Add, Constant(6)),
                    test_divisor: 19,
                    if_true: 2,
                    if_false: 0,
                    inspected: 0,
                },
                MonkeyMemo {
                    items: vec![79, 60, 97],
                    operation: Expression::new(Old, Multiply, Old),
                    test_divisor: 13,
                    if_true: 1,
                    if_false: 3,
                    inspected: 0,
                },
                MonkeyMemo {
                    items: vec![74],
                    operation: Expression::new(Old, Add, Constant(3)),
                    test_divisor: 17,
                    if_true: 0,
                    if_false: 1,
                    inspected: 0,
                },
            ],
            modulus: 96577,
        }
    }
}
