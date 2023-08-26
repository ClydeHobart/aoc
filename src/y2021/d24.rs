use {
    crate::*,
    glam::I64Vec4,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, separated_pair, terminated},
        Err, IResult, Parser,
    },
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        ops::{Range, RangeInclusive},
    },
};

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum Variable {
    X,
    Y,
    Z,
    W,
}

impl Variable {
    fn evaluate(&self, variables: &I64Vec4) -> i64 {
        variables[*self as usize]
    }

    fn branch<'i>(
        tag_str: &'static str,
        variable: Self,
    ) -> impl Parser<&'i str, Self, Error<&'i str>> {
        map(tag(tag_str), move |_| variable)
    }
}

impl Parse for Variable {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::branch("x", Self::X),
            Self::branch("y", Self::Y),
            Self::branch("z", Self::Z),
            Self::branch("w", Self::W),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(PartialEq)]
enum Param {
    Variable(Variable),
    Constant(i64),
}

impl Param {
    fn evaluate(&self, variables: &I64Vec4) -> i64 {
        match self {
            Param::Variable(variable) => variable.evaluate(variables),
            Param::Constant(constant) => *constant,
        }
    }

    fn constant(&self) -> Option<i64> {
        match self {
            Self::Constant(constant) => Some(*constant),
            _ => None,
        }
    }
}

impl Parse for Param {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Variable::parse, Self::Variable),
            map(parse_integer, Self::Constant),
        ))(input)
    }
}

trait Process {
    fn process(&self, variables: &mut I64Vec4, input: &mut &[u8]) -> bool;
}

#[cfg_attr(test, derive(Debug))]
#[derive(PartialEq)]
enum Instruction {
    Inp { a: Variable },
    Add { a: Variable, b: Param },
    Mul { a: Variable, b: Param },
    Div { a: Variable, b: Param },
    Mod { a: Variable, b: Param },
    Eql { a: Variable, b: Param },
}

impl Instruction {
    fn is_inp(&self) -> bool {
        matches!(self, Self::Inp { .. })
    }

    fn b(&self) -> Option<&Param> {
        match self {
            Self::Inp { .. } => None,
            Self::Add { b, .. } => Some(b),
            Self::Mul { b, .. } => Some(b),
            Self::Div { b, .. } => Some(b),
            Self::Mod { b, .. } => Some(b),
            Self::Eql { b, .. } => Some(b),
        }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(preceded(tag("inp "), Variable::parse), |a| Self::Inp { a }),
            map(
                preceded(
                    tag("add "),
                    separated_pair(Variable::parse, tag(" "), Param::parse),
                ),
                |(a, b)| Self::Add { a, b },
            ),
            map(
                preceded(
                    tag("mul "),
                    separated_pair(Variable::parse, tag(" "), Param::parse),
                ),
                |(a, b)| Self::Mul { a, b },
            ),
            map(
                preceded(
                    tag("div "),
                    separated_pair(Variable::parse, tag(" "), Param::parse),
                ),
                |(a, b)| Self::Div { a, b },
            ),
            map(
                preceded(
                    tag("mod "),
                    separated_pair(Variable::parse, tag(" "), Param::parse),
                ),
                |(a, b)| Self::Mod { a, b },
            ),
            map(
                preceded(
                    tag("eql "),
                    separated_pair(Variable::parse, tag(" "), Param::parse),
                ),
                |(a, b)| Self::Eql { a, b },
            ),
        ))(input)
    }
}

impl Process for Instruction {
    fn process(&self, variables: &mut I64Vec4, input: &mut &[u8]) -> bool {
        match self {
            Instruction::Inp { a } => {
                if input.is_empty() {
                    false
                } else {
                    variables[*a as usize] = input[0_usize] as i64;
                    *input = &input[1_usize..];

                    true
                }
            }
            Instruction::Add { a, b } => {
                variables[*a as usize] = a.evaluate(variables) + b.evaluate(variables);

                true
            }
            Instruction::Mul { a, b } => {
                variables[*a as usize] = a.evaluate(variables) * b.evaluate(variables);

                true
            }
            Instruction::Div { a, b } => {
                let b: i64 = b.evaluate(variables);

                if b == 0_i64 {
                    false
                } else {
                    variables[*a as usize] = a.evaluate(variables) / b;

                    true
                }
            }
            Instruction::Mod { a, b } => {
                let a_val: i64 = a.evaluate(variables);
                let b: i64 = b.evaluate(variables);

                if a_val < 0_i64 || b <= 0_i64 {
                    false
                } else {
                    variables[*a as usize] = a_val % b;

                    true
                }
            }
            Instruction::Eql { a, b } => {
                variables[*a as usize] = if a.evaluate(variables) == b.evaluate(variables) {
                    1_i64
                } else {
                    0_i64
                };

                true
            }
        }
    }
}

impl Process for [Instruction] {
    fn process(&self, variables: &mut I64Vec4, input: &mut &[u8]) -> bool {
        self.iter()
            .try_fold((), |_, instruction| {
                if instruction.process(variables, input) {
                    Some(())
                } else {
                    None
                }
            })
            .is_some()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ModelNumber([u8; ModelNumber::DIGITS]);

impl ModelNumber {
    const DIGITS: usize = 14_usize;
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum ModelNumberError {
    TooFewDigits,
    TooManyDigits,
    ZeroPresent,
}

impl From<ModelNumber> for u64 {
    fn from(value: ModelNumber) -> Self {
        value
            .0
            .into_iter()
            .rfold((1_u64, 0_u64), |(factor, sum), digit| {
                (factor * 10_u64, sum + factor * digit as u64)
            })
            .1
    }
}

impl TryFrom<u64> for ModelNumber {
    type Error = ModelNumberError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        (0_usize..Self::DIGITS)
            .try_rfold(
                (value, Self(Default::default())),
                |(value, mut model_number), digit_index| {
                    if value == 0_u64 {
                        Err(ModelNumberError::TooFewDigits)
                    } else {
                        let digit: u8 = (value % 10_u64) as u8;

                        if digit == 0_u8 {
                            Err(ModelNumberError::ZeroPresent)
                        } else {
                            model_number.0[digit_index] = digit;

                            Ok((value / 10_u64, model_number))
                        }
                    }
                },
            )
            .and_then(|(value, model_number)| {
                if value != 0_u64 {
                    Err(ModelNumberError::TooManyDigits)
                } else {
                    Ok(model_number)
                }
            })
    }
}

#[derive(Clone, Copy)]
struct ZString {
    bytes: [u8; ZString::BYTES_LEN],
    len: u8,
}

impl ZString {
    const ZERO: u8 = b'A';
    const BYTES_LEN: usize = ModelNumber::DIGITS / 2_usize;

    fn last(self) -> i8 {
        if self.len == 0_u8 {
            0_i8
        } else {
            (self.bytes[(self.len - 1_u8) as usize] - Self::ZERO) as i8
        }
    }

    fn push(&mut self, value: i8) {
        self.bytes[self.len as usize] = (value as u8) + Self::ZERO;
        self.len += 1_u8;
    }

    fn pop(&mut self) {
        self.len = self.len.saturating_sub(1_u8);
    }

    fn add(&mut self, value: i8) {
        let mut value: u8 = value as u8;

        for byte in self.bytes[..self.len as usize].iter_mut().rev() {
            *byte += value;

            if *byte > b'Z' {
                *byte -= 26_u8;
                value = 1_u8;
            } else {
                value = 0_u8;

                break;
            }
        }

        if value > 0_u8 {
            self.bytes[self.len as usize] = value + Self::ZERO;
            self.len += 1_u8;
            self.bytes[..self.len as usize].rotate_right(1_usize);
        }
    }
}

impl Debug for ZString {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ZString")
            .field("bytes", &self.bytes)
            .field("len", &self.len)
            .field("string", &String::from(*self))
            .finish()
    }
}

impl Default for ZString {
    fn default() -> Self {
        ZString {
            bytes: [b'A'; Self::BYTES_LEN],
            len: 0_u8,
        }
    }
}

impl From<ZString> for String {
    fn from(value: ZString) -> Self {
        let len: usize = value.len.max(1_u8) as usize;
        let mut vec: Vec<u8> = vec![ZString::ZERO; len];

        for (byte_src, byte_dest) in value.bytes[..len].iter().copied().zip(vec.iter_mut()) {
            *byte_dest = byte_src;
        }

        // SAFETY: Any `u8` in `value.bytes` is in the range `b'A'..=b'Z'`
        unsafe { String::from_utf8_unchecked(vec) }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CycleConstants {
    i: bool,
    j: i8,
    k: i8,
}

impl CycleConstants {
    fn process(self, z_string: &mut ZString, w: i8) {
        match (z_string.last() + self.j == w, self.i) {
            (false, false) => z_string.push(w + self.k),
            (false, true) => z_string.add(w + self.k),
            (true, false) => (),
            (true, true) => z_string.pop(),
        }
    }
}

/// An alternative representation of a set of instructions that simplifies what's going on behind
/// the scenes.
///
/// Each set of instructions following an `inp w` instruction looks like the following:
///
/// ```
/// "mul x 0"  => x = 0;
/// "add x z"  => x = z_0;
/// "mod x 26" => x = z_0 % 26;
/// "div z i"  => z = z_0 / i;
/// "add x j"  => x = z_0 % 26 + j;
/// "eql x w"  => x = (z_0 % 26 + j == w);
/// "eql x 0"  => x = (z_0 % 26 + j != w);
/// "mul y 0"  => y = 0;
/// "add y 25" => y = 25;
/// "mul y x"  => y = 25 * (z_0 % 26 + j != w);
/// "add y 1"  => y = 25 * (z_0 % 26 + j != w) + 1;
/// "mul z y"  => z = (z_0 / a) * (25 * (z_0 % 26 + j != w) + 1);
/// "mul y 0"  => y = 0;
/// "add y w"  => y = w;
/// "add y k"  => y = w + k;
/// "mul y x"  => y = (w + k) * (z_0 % 26 + j != w);
/// "add z y"  => z = (z_0 / i) * (25 * (z_0 % 26 + j != w) + 1) + (w + k) * (z_0 % 26 + j != w);
/// ```
///
/// Also worth noting:
/// * `i` is either 1 or 26.
/// * `i` is 26 seven times and 1 seven times.
/// * Proceeding through the cycles of instructions (where one cycle consists of `inp w` followed by
///   lines following the pattern above), the count of observed cycles where `i` is 26 is always
///   less than or equal to the count of observed cycles where the `i` is 1.
///
///
/// Think about z as a string of letters, or a base 26 number where the possible digits are just
/// a..=z. In effect, each cycle just takes the existing `z` value (written as `z_0` above) and does
/// the following operation to it:
///
/// ```
/// if z.last() + j == w {
///     z / i
/// } else {
///     z * 26 / i + (w + k)
/// }
/// ```
///
/// If instead you interpret `i` as a bool, where `i == 26` is `true` and `i == 1` is `false`, this
/// becomes the contents of `CycleConstants::process`.
#[derive(Debug)]
struct Monad([CycleConstants; ModelNumber::DIGITS]);

impl Monad {
    fn optimal_valid_model_number(&self, largest: bool) -> Option<ModelNumber> {
        const DIGITS_ADD_1: usize = ModelNumber::DIGITS + 1_usize;

        let mut model_number: ModelNumber = ModelNumber(Default::default());
        let mut z_strings: [Option<ZString>; DIGITS_ADD_1] = [None; DIGITS_ADD_1];
        let mut z_string_lens: [u8; DIGITS_ADD_1] = [0_u8; DIGITS_ADD_1];
        let mut z_string_len: u8 = 0_u8;

        for (cycle_constants, z_string_len_dest) in
            self.0.iter().copied().zip(z_string_lens.iter_mut())
        {
            *z_string_len_dest = z_string_len;

            if cycle_constants.i {
                z_string_len = z_string_len.checked_sub(1_u8)?;
            } else {
                z_string_len += 1_u8;
            }
        }

        if z_string_len != 0_u8 {
            None
        } else {
            z_strings[0_usize] = Some(Default::default());

            const W_RANGE: RangeInclusive<i8> = 1_i8..=9_i8;

            self.0.iter().copied().enumerate().try_fold(
                (),
                |_, (index, curr_cycle_constants)| {
                    let mut z_string: ZString = z_strings[index]?;

                    if z_string.len == z_string_lens[index] {
                        if curr_cycle_constants.i {
                            let w: i8 = z_string.last() + curr_cycle_constants.j;

                            if W_RANGE.contains(&w) {
                                model_number.0[index] = w as u8;
                                curr_cycle_constants.process(&mut z_string, w);
                                z_strings[index + 1_usize] = Some(z_string);

                                Some(())
                            } else {
                                None
                            }
                        } else {
                            let next_cycle_constants: CycleConstants = (index..ModelNumber::DIGITS)
                                .find_map(|index| {
                                    let cycle_constants: CycleConstants = self.0[index];

                                    if cycle_constants.i
                                        && (z_string_lens[index] == z_string.len + 1_u8)
                                    {
                                        Some(cycle_constants)
                                    } else {
                                        None
                                    }
                                })?;
                            let neg_curr_k_sub_next_j: i8 =
                                -curr_cycle_constants.k - next_cycle_constants.j;
                            let w: i8 = if largest {
                                *W_RANGE.end() + neg_curr_k_sub_next_j.min(0_i8)
                            } else {
                                *W_RANGE.start() + neg_curr_k_sub_next_j.max(0_i8)
                            };

                            if W_RANGE.contains(&w) {
                                model_number.0[index] = w as u8;
                                curr_cycle_constants.process(&mut z_string, w);
                                z_strings[index + 1_usize] = Some(z_string);

                                Some(())
                            } else {
                                None
                            }
                        }
                    } else {
                        None
                    }
                },
            )?;

            let final_z_string: ZString = z_strings[ModelNumber::DIGITS]?;

            if final_z_string.len == 0_u8 {
                Some(model_number)
            } else {
                None
            }
        }
    }

    fn largest_valid_model_number(&self) -> Option<ModelNumber> {
        self.optimal_valid_model_number(true)
    }

    fn smallest_valid_model_number(&self) -> Option<ModelNumber> {
        self.optimal_valid_model_number(false)
    }
}

impl TryFrom<&Solution> for Monad {
    type Error = ();

    fn try_from(value: &Solution) -> Result<Self, Self::Error> {
        use {
            Instruction::*,
            Param::{Constant as C, Variable as V},
            Variable::*,
        };

        const CYCLE_LEN: usize = 18_usize;
        const INSTRUCTIONS: [Instruction; CYCLE_LEN] = [
            Inp { a: W },           // 0
            Mul { a: X, b: C(0) },  // 1
            Add { a: X, b: V(Z) },  // 2
            Mod { a: X, b: C(26) }, // 3
            Div {
                a: Z,
                b: C(i64::MAX),
            }, // 4
            Add {
                a: X,
                b: C(i64::MAX),
            }, // 5
            Eql { a: X, b: V(W) },  // 6
            Eql { a: X, b: C(0) },  // 7
            Mul { a: Y, b: C(0) },  // 8
            Add { a: Y, b: C(25) }, // 9
            Mul { a: Y, b: V(X) },  // 10
            Add { a: Y, b: C(1) },  // 11
            Mul { a: Z, b: V(Y) },  // 12
            Mul { a: Y, b: C(0) },  // 13
            Add { a: Y, b: V(W) },  // 14
            Add {
                a: Y,
                b: C(i64::MAX),
            }, // 15
            Mul { a: Y, b: V(X) },  // 16
            Add { a: Z, b: V(Y) },  // 17
        ];

        let instruction_ranges: Vec<Range<usize>> = value.instruction_ranges();

        if instruction_ranges.iter().cloned().all(|instruction_range| {
            instruction_range.len() == CYCLE_LEN
                && value.0[instruction_range]
                    .iter()
                    .zip(INSTRUCTIONS)
                    .enumerate()
                    .all(|(index, (instruction_a, instruction_b))| match index {
                        4_usize => matches!(instruction_a, Div { a: Z, b: C(_) }),
                        5_usize => matches!(instruction_a, Add { a: X, b: C(_) }),
                        15_usize => matches!(instruction_a, Add { a: Y, b: C(_) }),
                        _ => *instruction_a == instruction_b,
                    })
        }) {
            let mut monad: Self = Self(Default::default());

            if instruction_ranges
                .into_iter()
                .enumerate()
                .map(|(index, instruction_range)| {
                    let instructions: &[Instruction] = &value.0[instruction_range];

                    (
                        index,
                        instructions[4_usize].b().unwrap().constant().unwrap(),
                        instructions[5_usize].b().unwrap().constant().unwrap(),
                        instructions[15_usize].b().unwrap().constant().unwrap(),
                    )
                })
                .try_fold(0_u8, |not_i_count, (index, i, j, k)| {
                    i8::try_from(j)
                        .ok()
                        .zip(i8::try_from(k).ok())
                        .and_then(|(j, k)| {
                            if (i == 1_i64 && j > 0_i8 || i == 26_i64 && j < 0_i8)
                                && (0_i8..=16_i8).contains(&k)
                            {
                                let i: bool = i == 26_i64;

                                monad.0[index] = CycleConstants { i, j, k };

                                if i {
                                    not_i_count.checked_sub(1_u8)
                                } else {
                                    Some(not_i_count + 1_u8)
                                }
                            } else {
                                None
                            }
                        })
                })
                == Some(0_u8)
            {
                Ok(monad)
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    fn instruction_ranges(&self) -> Vec<Range<usize>> {
        let inp_indices: Vec<usize> = (0_usize..self.0.len())
            .filter(|index| self.0[*index].is_inp())
            .collect();

        inp_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(inp_indices_index, inp_index)| {
                inp_index
                    ..inp_indices
                        .get(inp_indices_index + 1_usize)
                        .copied()
                        .unwrap_or(self.0.len())
            })
            .collect()
    }

    #[allow(dead_code)]
    fn is_model_number_valid(&self, model_number: &ModelNumber) -> bool {
        let mut variables: I64Vec4 = I64Vec4::ZERO;
        let mut input: &[u8] = &model_number.0;

        self.0.process(&mut variables, &mut input) && variables.z == 0_i64
    }

    #[allow(dead_code)]
    fn largest_valid_model_number_brute_force(&self) -> Option<u64> {
        (11111111111111_u64..99999999999999_u64)
            .rev()
            .filter_map(|model_number| ModelNumber::try_from(model_number).ok())
            .find(|model_number| self.is_model_number_valid(model_number))
            .map(u64::from)
    }

    #[allow(dead_code)]
    fn largest_valid_model_number_dfs(&self) -> Option<u64> {
        let instruction_ranges: Vec<Range<usize>> = self.instruction_ranges();

        if instruction_ranges.len() == ModelNumber::DIGITS {
            let mut variables: I64Vec4 = I64Vec4::ZERO;
            let first_inp: usize = instruction_ranges.first().unwrap().start;

            if first_inp != 0_usize {
                // By the definition of `first_inp`, the `input` slice can start empty for this
                let mut input: &[u8] = &[];

                assert!(self.0[..first_inp].process(&mut variables, &mut input));
            }

            struct State {
                variables: I64Vec4,
                instruction_range_index: usize,
                model_number: u64,
            }

            let mut frontier: Vec<State> = vec![State {
                variables,
                instruction_range_index: 0_usize,
                model_number: 0_u64,
            }];

            while let Some(state) = frontier.pop() {
                if state.instruction_range_index != ModelNumber::DIGITS {
                    let instructions: &[Instruction] =
                        &self.0[instruction_ranges[state.instruction_range_index].clone()];
                    let instruction_range_index: usize = state.instruction_range_index + 1_usize;
                    let model_number: u64 = state.model_number * 10_u64;

                    frontier.extend((1_u8..=9_u8).map(|digit| {
                        let mut variables: I64Vec4 = state.variables;
                        let mut input: &[u8] = &[digit];

                        assert!(instructions.process(&mut variables, &mut input));

                        State {
                            variables,
                            instruction_range_index,
                            model_number: model_number + digit as u64,
                        }
                    }));
                } else if state.variables.z == 0_i64 {
                    return Some(state.model_number);
                }
            }
        }

        None
    }

    #[allow(dead_code)]
    fn largest_valid_model_number_by_digit(&self) -> Option<u64> {
        let instruction_ranges: Vec<Range<usize>> = self.instruction_ranges();

        if instruction_ranges.len() == ModelNumber::DIGITS {
            let mut variables: I64Vec4 = I64Vec4::ZERO;
            let first_inp: usize = instruction_ranges.first().unwrap().start;

            if first_inp != 0_usize {
                // By the definition of `first_inp`, the `input` slice can start empty for this
                let mut input: &[u8] = &[];

                assert!(self.0[..first_inp].process(&mut variables, &mut input));
            }

            let mut model_number: ModelNumber = ModelNumber(Default::default());

            for (digit_index, instructions_range) in instruction_ranges.into_iter().enumerate() {
                let instructions: &[Instruction] = &self.0[instructions_range];

                for digit in (1_u8..=9_u8).rev() {
                    let mut local_variables: I64Vec4 = variables;
                    let mut input: &[u8] = &[digit];

                    assert!(instructions.process(&mut local_variables, &mut input));

                    if local_variables.z == 0_i64 {
                        variables = local_variables;
                        model_number.0[digit_index] = digit;

                        break;
                    }
                }

                if model_number.0[digit_index] == 0_u8 {
                    return None;
                }
            }

            Some(model_number.into())
        } else {
            None
        }
    }

    fn optimal_valid_model_number<F: Fn(&Monad) -> Option<ModelNumber>>(
        &self,
        f: F,
    ) -> Option<u64> {
        Monad::try_from(self)
            .ok()
            .as_ref()
            .and_then(f)
            .map(u64::from)
    }

    fn largest_valid_model_number_monad(&self) -> Option<u64> {
        self.optimal_valid_model_number(Monad::largest_valid_model_number)
    }

    fn largest_valid_model_number(&self) -> Option<u64> {
        self.largest_valid_model_number_monad()
    }

    fn smallest_valid_model_number_monad(&self) -> Option<u64> {
        self.optimal_valid_model_number(Monad::smallest_valid_model_number)
    }

    fn smallest_valid_model_number(&self) -> Option<u64> {
        self.smallest_valid_model_number_monad()
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
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.largest_valid_model_number());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.smallest_valid_model_number());
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
    use {super::*, glam::Vec4Swizzles, std::sync::OnceLock};

    const SOLUTION_STR: &str = "\
        inp w\n\
        add z w\n\
        mod z 2\n\
        div w 2\n\
        add y w\n\
        mod y 2\n\
        div w 2\n\
        add x w\n\
        mod x 2\n\
        div w 2\n\
        mod w 2\n";

    fn solution() -> &'static Solution {
        use {
            Instruction::*,
            Param::{Constant as C, Variable as V},
            Variable::*,
        };

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Inp { a: W },
                Add { a: Z, b: V(W) },
                Mod { a: Z, b: C(2) },
                Div { a: W, b: C(2) },
                Add { a: Y, b: V(W) },
                Mod { a: Y, b: C(2) },
                Div { a: W, b: C(2) },
                Add { a: X, b: V(W) },
                Mod { a: X, b: C(2) },
                Div { a: W, b: C(2) },
                Mod { a: W, b: C(2) },
            ])
        })
    }

    #[test]
    fn test_solution_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_model_number_try_from_u64() {
        assert_eq!(
            ModelNumber::try_from(13579246899999_u64),
            Ok(ModelNumber([1, 3, 5, 7, 9, 2, 4, 6, 8, 9, 9, 9, 9, 9]))
        );
    }

    #[test]
    fn test_instruction_process() {
        for w in 0_u8..10_u8 {
            let mut variables: I64Vec4 = I64Vec4::ZERO;
            let mut input: &[u8] = &[w];

            fn variables_to_u8(variables: I64Vec4) -> u8 {
                variables.zyxw().to_array().into_iter().enumerate().fold(
                    0_u8,
                    |byte, (bit_index, bit)| {
                        if bit != 0_i64 {
                            byte | (1_u8 << bit_index)
                        } else {
                            byte
                        }
                    },
                )
            }

            assert!(solution().0.process(&mut variables, &mut input));
            assert_eq!(variables_to_u8(variables), w);
        }
    }
}
