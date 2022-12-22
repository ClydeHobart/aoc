use {
    aoc_2022::*,
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        hash::Hash,
        mem::transmute,
        num::ParseIntError,
        str::{from_utf8_unchecked, FromStr, Split},
    },
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

/// A monkey's name, which is expected to be 4 ASCII lowercase characters in string form
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Tag(u32);

macro_rules! tag {
    ($utf8_bytes:literal) => {
        Tag::from_utf8_bytes(*$utf8_bytes)
    };
}

impl Tag {
    const ROOT: Self = tag!(b"root");
    const HUMN: Self = tag!(b"humn");

    const fn from_utf8_bytes(bytes: [u8; 4_usize]) -> Self {
        Self(u32::from_ne_bytes(bytes))
    }

    const fn as_utf8_bytes(&self) -> &[u8; 4_usize] {
        // SAFETY: We're converting a reference to a 4-byte `Tag` to a reference to a 4-byte array
        unsafe { transmute(self) }
    }

    const fn as_str(&self) -> &str {
        // SAFETY: Any constructed `Tag` is assumed to contain 4 ASCII lowercase characters
        unsafe { from_utf8_unchecked(self.as_utf8_bytes()) }
    }
}

impl Debug for Tag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, PartialEq)]
enum ParseTagError<'s> {
    InvalidLength(&'s str),
    IsNotAsciiLowercase(char),
}

impl<'s> TryFrom<&'s str> for Tag {
    type Error = ParseTagError<'s>;

    fn try_from(tag_str: &'s str) -> Result<Self, Self::Error> {
        use ParseTagError::*;

        if tag_str.len() != 4_usize {
            Err(InvalidLength(tag_str))
        } else {
            let mut bytes: [u8; 4_usize] = [0_u8; 4_usize];

            for (byte, c) in bytes.iter_mut().zip(tag_str.chars()) {
                if !c.is_ascii_lowercase() {
                    return Err(IsNotAsciiLowercase(c));
                }

                *byte = c as u8;
            }

            Ok(Self::from_utf8_bytes(bytes))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, PartialEq)]
enum ParseOperationError<'s> {
    InvalidPattern(&'s str),
}

impl<'s> TryFrom<&'s str> for Operation {
    type Error = ParseOperationError<'s>;

    fn try_from(operation_str: &'s str) -> Result<Self, Self::Error> {
        use {Operation::*, ParseOperationError::*};

        match operation_str {
            "+" => Ok(Add),
            "-" => Ok(Subtract),
            "*" => Ok(Multiply),
            "/" => Ok(Divide),
            invalid_pattern => Err(InvalidPattern(invalid_pattern)),
        }
    }
}

type ExpressionNumericType = i64;

#[derive(Clone, Debug, PartialEq)]
enum Expression {
    Constant(ExpressionNumericType),
    Operation {
        operand_a: Tag,
        operand_b: Tag,
        operation: Operation,
    },
}

impl Expression {
    fn values_for_tag(
        &self,
        tag: Tag,
        stack: &mut Vec<Tag>,
        values: &mut HashMap<Tag, ExpressionNumericType>,
    ) {
        match self {
            Expression::Constant(value) => {
                values.insert(tag, *value);
                stack.pop();
            }
            Expression::Operation {
                operand_a,
                operand_b,
                operation,
            } => {
                if let Some(value_a) = values.get(operand_a).copied() {
                    if let Some(value_b) = values.get(operand_b).copied() {
                        use Operation::*;

                        values.insert(
                            tag,
                            match operation {
                                Add => value_a + value_b,
                                Subtract => value_a - value_b,
                                Multiply => value_a * value_b,
                                Divide => value_a / value_b,
                            },
                        );
                        stack.pop();
                    } else {
                        stack.push(*operand_b);
                    }
                } else {
                    stack.push(*operand_a);
                }
            }
        }
    }

    fn find_path_to_variable(&self, stack: &mut Vec<Tag>, a_is_variable: &mut Vec<bool>) {
        match self {
            Expression::Constant(_) => {
                stack.pop();
            }
            Expression::Operation {
                operand_a,
                operand_b,
                ..
            } => {
                if stack.len() > a_is_variable.len() {
                    a_is_variable.push(true);
                    stack.push(*operand_a);
                } else {
                    let last_a_is_variable: &mut bool = a_is_variable.last_mut().unwrap();

                    if *last_a_is_variable {
                        *last_a_is_variable = false;
                        stack.push(*operand_b);
                    } else {
                        a_is_variable.pop();
                        stack.pop();
                    }
                }
            }
        }
    }

    fn get_operand(&self, get_a: bool) -> Option<Tag> {
        match self {
            Expression::Constant(_) => None,
            Expression::Operation {
                operand_a,
                operand_b,
                ..
            } => Some(if get_a { *operand_a } else { *operand_b }),
        }
    }

    fn update_constrained_value_and_operation_tag(
        &self,
        constrained_value: ExpressionNumericType,
        a_is_variable: bool,
        expressions: &Expressions,
        stack: &mut Vec<Tag>,
        values: &mut HashMap<Tag, ExpressionNumericType>,
    ) -> (ExpressionNumericType, Tag) {
        use Operation::*;

        match self {
            Expression::Constant(_) => panic!(
                "`Expression::update_constrained_value_and_operation_tag` called on a non-\
                    `Operation`-type `Expression`"
            ),
            Expression::Operation {
                operand_a,
                operand_b,
                operation,
            } => {
                if a_is_variable {
                    let value_b: ExpressionNumericType =
                        expressions.values_for_tag(*operand_b, stack, values);

                    (
                        match operation {
                            Add => constrained_value - value_b,
                            Subtract => constrained_value + value_b,
                            Multiply => constrained_value / value_b,
                            Divide => constrained_value * value_b,
                        },
                        *operand_a,
                    )
                } else {
                    let value_a: ExpressionNumericType =
                        expressions.values_for_tag(*operand_a, stack, values);

                    (
                        match operation {
                            Add => constrained_value - value_a,
                            Subtract => value_a - constrained_value,
                            Multiply => constrained_value / value_a,
                            Divide => value_a / constrained_value,
                        },
                        *operand_b,
                    )
                }
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum ParseExpressionError<'s> {
    NoInitialChar,
    FailedToParseConstant(ParseIntError),
    NoOperationOperandAToken,
    FailedToParseOperationOperandA(ParseTagError<'s>),
    NoOperationToken,
    FailedToParseOperation(ParseOperationError<'s>),
    NoOperationOperandBToken,
    FailedToParseOperationOperandB(ParseTagError<'s>),
    ExtraOperationTokenFound(&'s str),
}

impl<'s> TryFrom<&'s str> for Expression {
    type Error = ParseExpressionError<'s>;

    fn try_from(expression_str: &'s str) -> Result<Self, Self::Error> {
        use ParseExpressionError::*;

        if let Some(initial_char) = expression_str.chars().next() {
            if initial_char.is_ascii_digit() {
                ExpressionNumericType::from_str(expression_str)
                    .map(Expression::Constant)
                    .map_err(FailedToParseConstant)
            } else {
                let mut operation_token_iter: Split<char> = expression_str.split(' ');

                let operand_a: Tag = operation_token_iter
                    .next()
                    .ok_or(NoOperationOperandAToken)?
                    .try_into()
                    .map_err(FailedToParseOperationOperandA)?;
                let operation: Operation = operation_token_iter
                    .next()
                    .ok_or(NoOperationToken)?
                    .try_into()
                    .map_err(FailedToParseOperation)?;
                let operand_b: Tag = operation_token_iter
                    .next()
                    .ok_or(NoOperationOperandBToken)?
                    .try_into()
                    .map_err(FailedToParseOperationOperandB)?;

                match operation_token_iter.next() {
                    Some(extra_token) => Err(ExtraOperationTokenFound(extra_token)),
                    None => Ok(Expression::Operation {
                        operand_a,
                        operand_b,
                        operation,
                    }),
                }
            }
        } else {
            Err(NoInitialChar)
        }
    }
}

#[derive(Debug, PartialEq)]
struct Expressions(HashMap<Tag, Expression>);

#[derive(Debug, PartialEq)]
enum ComputeConstrainedValueError {
    VariableIsFulcrum,
    FulcrumDoesNotDependOnVariable,
}

impl Expressions {
    fn values_for_tag(
        &self,
        tag: Tag,
        stack: &mut Vec<Tag>,
        values: &mut HashMap<Tag, ExpressionNumericType>,
    ) -> ExpressionNumericType {
        stack.clear();
        stack.push(tag);

        while let Some(tag) = stack.last().copied() {
            if values.contains_key(&tag) {
                stack.pop();
            } else {
                self.0[&tag].values_for_tag(tag, stack, values);
            }
        }

        values[&tag]
    }

    fn value_for_tag(&self, tag: Tag) -> ExpressionNumericType {
        let mut stack: Vec<Tag> = Vec::new();
        let mut values: HashMap<Tag, ExpressionNumericType> = HashMap::new();

        self.values_for_tag(tag, &mut stack, &mut values)
    }

    fn compute_constrained_value(
        &self,
        fulcrum: Tag,
        variable: Tag,
    ) -> Result<ExpressionNumericType, ComputeConstrainedValueError> {
        use ComputeConstrainedValueError::*;

        if variable == fulcrum {
            return Err(VariableIsFulcrum);
        }

        // 1. Find the path to the variable
        let mut stack: Vec<Tag> = vec![fulcrum];
        let mut a_is_variable: Vec<bool> = Vec::new();

        while let Some(tag) = stack.last().copied() {
            if tag == variable {
                break;
            }

            self.0[&tag].find_path_to_variable(&mut stack, &mut a_is_variable);
        }

        if stack.is_empty() {
            return Err(FulcrumDoesNotDependOnVariable);
        }

        // At this point, following the breadcrumps in `a_is_variable` will only lead
        // to `Operation`-type `Expression`s, including `fulcrum`

        // 2. Find the value the initial constrained value
        let fulcrum_expression: &Expression = &self.0[&fulcrum];
        let first_a_is_variable: bool = a_is_variable[0_usize];

        let mut values: HashMap<Tag, ExpressionNumericType> = HashMap::new();
        let (mut constrained_value, mut operation_tag) = (
            self.values_for_tag(
                fulcrum_expression
                    .get_operand(!first_a_is_variable)
                    .unwrap(),
                &mut stack,
                &mut values,
            ),
            fulcrum_expression.get_operand(first_a_is_variable).unwrap(),
        );

        // 3. Use the constraints and the operations until we find the value of `variable`
        for a_is_variable in a_is_variable[1_usize..].iter().copied() {
            (constrained_value, operation_tag) = self.0[&operation_tag]
                .update_constrained_value_and_operation_tag(
                    constrained_value,
                    a_is_variable,
                    self,
                    &mut stack,
                    &mut values,
                );
        }

        assert_eq!(
            operation_tag, variable,
            "`Expressions::compute_variable_value()` \
            ended on tag {operation_tag:?} instead of tag {variable:?}"
        );

        Ok(constrained_value)
    }
}

#[derive(Debug, PartialEq)]
enum ParseExpressionsError<'s> {
    NoMonkeyTagToken,
    FailedToParseMonkeyTag(ParseTagError<'s>),
    NoExpressionToken,
    FailedToParseExpression(ParseExpressionError<'s>),
    ExtraTokenFound(&'s str),
    PreExistingValueForMonkeyTag {
        monkey: Tag,
        old_value: Expression,
        new_value: Expression,
    },
    NoRootTag,
    RootIsNotOperation,
    NoHumnTag,
}

impl<'s> TryFrom<&'s str> for Expressions {
    type Error = ParseExpressionsError<'s>;

    fn try_from(expressions_str: &'s str) -> Result<Self, Self::Error> {
        use ParseExpressionsError::*;

        let mut expressions: Expressions = Expressions(HashMap::new());

        for tag_and_expression_str in expressions_str.split('\n') {
            let mut token_iter: Split<&str> = tag_and_expression_str.split(": ");

            let monkey: Tag = token_iter
                .next()
                .ok_or(NoMonkeyTagToken)?
                .try_into()
                .map_err(FailedToParseMonkeyTag)?;
            let new_value: Expression = token_iter
                .next()
                .ok_or(NoExpressionToken)?
                .try_into()
                .map_err(FailedToParseExpression)?;

            if let Some(extra_token) = token_iter.next() {
                return Err(ExtraTokenFound(extra_token));
            }

            if let Some(old_value) = expressions.0.insert(monkey, new_value.clone()) {
                return Err(PreExistingValueForMonkeyTag {
                    monkey,
                    old_value,
                    new_value,
                });
            }
        }

        match expressions.0.get(&Tag::ROOT) {
            None => Err(NoRootTag),
            Some(Expression::Constant(_)) => Err(RootIsNotOperation),
            Some(_) => {
                if !expressions.0.contains_key(&Tag::HUMN) {
                    Err(NoHumnTag)
                } else {
                    Ok(expressions)
                }
            }
        }
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day21.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match Expressions::try_from(input) {
                    Ok(expressions) => {
                        dbg!(expressions.value_for_tag(Tag::ROOT));
                        dbg!(expressions.compute_constrained_value(Tag::ROOT, Tag::HUMN)).ok();
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

    const EXPRESSIONS_STR: &str = concat!(
        "root: pppw + sjmn\n",
        "dbpl: 5\n",
        "cczh: sllz + lgvd\n",
        "zczc: 2\n",
        "ptdq: humn - dvpt\n",
        "dvpt: 3\n",
        "lfqf: 4\n",
        "humn: 5\n",
        "ljgn: 2\n",
        "sjmn: drzm * dbpl\n",
        "sllz: 4\n",
        "pppw: cczh / lfqf\n",
        "lgvd: ljgn * ptdq\n",
        "drzm: hmdt - zczc\n",
        "hmdt: 32",
    );
    const DRZM: Tag = tag!(b"drzm");
    const HMDT: Tag = tag!(b"hmdt");
    const ZCZC: Tag = tag!(b"zczc");
    const SJMN: Tag = tag!(b"sjmn");
    const DBPL: Tag = tag!(b"dbpl");
    const ROOT: Tag = Tag::ROOT;
    const HUMN: Tag = Tag::HUMN;

    const DRZM_VALUE: ExpressionNumericType = 32;
    const HMDT_VALUE: ExpressionNumericType = 2;
    const ZCZC_VALUE: ExpressionNumericType = 30;
    const SJMN_VALUE: ExpressionNumericType = 5;
    const DBPL_VALUE: ExpressionNumericType = 150;
    const ROOT_VALUE: ExpressionNumericType = 152;
    const HUMN_VALUE: ExpressionNumericType = 301;

    lazy_static! {
        static ref EXPRESSIONS: Expressions = expressions();
    }

    #[test]
    fn test_expressions_try_from_str() {
        assert_eq!(EXPRESSIONS_STR.try_into().as_ref(), Ok(&*EXPRESSIONS))
    }

    #[test]
    fn test_expressions_values_for_tag() {
        let mut stack: Vec<Tag> = Vec::new();
        let mut values: HashMap<Tag, ExpressionNumericType> = HashMap::new();

        EXPRESSIONS.values_for_tag(Tag::ROOT, &mut stack, &mut values);

        assert_eq!(values.get(&HMDT), Some(&DRZM_VALUE));
        assert_eq!(values.get(&ZCZC), Some(&HMDT_VALUE));
        assert_eq!(values.get(&DRZM), Some(&ZCZC_VALUE));
        assert_eq!(values.get(&DBPL), Some(&SJMN_VALUE));
        assert_eq!(values.get(&SJMN), Some(&DBPL_VALUE));
        assert_eq!(values.get(&ROOT), Some(&ROOT_VALUE));
    }

    #[test]
    fn test_expressions_value_for_tag() {
        assert_eq!(EXPRESSIONS.value_for_tag(HMDT), DRZM_VALUE);
        assert_eq!(EXPRESSIONS.value_for_tag(ZCZC), HMDT_VALUE);
        assert_eq!(EXPRESSIONS.value_for_tag(DRZM), ZCZC_VALUE);
        assert_eq!(EXPRESSIONS.value_for_tag(DBPL), SJMN_VALUE);
        assert_eq!(EXPRESSIONS.value_for_tag(SJMN), DBPL_VALUE);
        assert_eq!(EXPRESSIONS.value_for_tag(ROOT), ROOT_VALUE);
    }

    #[test]
    fn test_expressions_compute_variable_value() {
        assert_eq!(
            EXPRESSIONS.compute_constrained_value(ROOT, HUMN),
            Ok(HUMN_VALUE)
        );
    }

    fn expressions() -> Expressions {
        use Expression::{Constant as C, Operation as O};

        macro_rules! expr {
            ($op_a:literal, $op:ident, $op_b:literal) => {
                O {
                    operand_a: tag!($op_a),
                    operand_b: tag!($op_b),
                    operation: Operation::$op,
                }
            };
            ($constant:literal) => {
                C($constant)
            };
        }

        macro_rules! expressions {
            [ $( ( $monkey:literal: $( $token:tt ),+ ), )* ] => {
                Expressions(vec![ $(
                    (tag!($monkey), expr!( $( $token ),+ )),
                )* ].into_iter().collect::<HashMap<Tag, Expression>>())
            };
        }

        expressions![
            (b"root": b"pppw", Add, b"sjmn"),
            (b"dbpl": 5),
            (b"cczh": b"sllz", Add, b"lgvd"),
            (b"zczc": 2),
            (b"ptdq": b"humn", Subtract, b"dvpt"),
            (b"dvpt": 3),
            (b"lfqf": 4),
            (b"humn": 5),
            (b"ljgn": 2),
            (b"sjmn": b"drzm", Multiply, b"dbpl"),
            (b"sllz": 4),
            (b"pppw": b"cczh", Divide, b"lfqf"),
            (b"lgvd": b"ljgn", Multiply, b"ptdq"),
            (b"drzm": b"hmdt", Subtract, b"zczc"),
            (b"hmdt": 32),
        ]
    }
}
