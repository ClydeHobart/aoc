use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::satisfy,
        combinator::{map, verify},
        error::Error,
        multi::separated_list0,
        sequence::{delimited, tuple},
        Err, IResult,
    },
};

/* --- Day 12: JSAbacusFramework.io ---

Santa's Accounting-Elves need help balancing the books after a recent order. Unfortunately, their accounting software uses a peculiar storage format. That's where you come in.

They have a JSON document which contains a variety of things: arrays ([1,2,3]), objects ({"a":1, "b":2}), numbers, and strings. Your first job is to simply find all of the numbers throughout the document and add them together.

For example:

    [1,2,3] and {"a":2,"b":4} both have a sum of 6.
    [[[3]]] and {"a":{"b":4},"c":-1} both have a sum of 3.
    {"a":[-1,1]} and [-1,{"a":1}] both have a sum of 0.
    [] and {} both have a sum of 0.

You will not encounter any strings containing numbers.

What is the sum of all numbers in the document?

--- Part Two ---

Uh oh - the Accounting-Elves have realized that they double-counted everything red.

Ignore any object (and all of its children) which has any property with the value "red". Do this only for objects ({...}), not arrays ([...]).

    [1,2,3] still has a sum of 6.
    [1,{"c":"red","b":2},3] now has a sum of 4, because the middle object is ignored.
    {"d":"red","e":[1,2,3,4],"f":5} now has a sum of 0, because the entire structure is ignored.
    [1,"red",5] has a sum of 6, because "red" in an array has no effect. */

const MAX_STRING_LEN: usize = 7_usize;

type JsonString = StaticString<MAX_STRING_LEN>;

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Json {
    Array(Vec<Json>),
    Object(Vec<(char, Json)>),
    Number(i32),
    String(JsonString),
}

impl Json {
    fn visit_numbers<P: Fn(&[(char, Json)]) -> bool, F: FnMut(i32)>(
        &self,
        object_fields_predicate: &P,
        visit_number: &mut F,
    ) {
        match self {
            Self::Array(elements) => {
                for element in elements {
                    element.visit_numbers(object_fields_predicate, visit_number);
                }
            }
            Self::Object(fields) => {
                if object_fields_predicate(fields) {
                    for (_, field_value) in fields {
                        field_value.visit_numbers(object_fields_predicate, visit_number);
                    }
                }
            }
            Self::Number(number) => visit_number(*number),
            Self::String(_) => (),
        }
    }
}

impl Parse for Json {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                delimited(tag("["), separated_list0(tag(","), Self::parse), tag("]")),
                Self::Array,
            ),
            map(
                verify(
                    delimited(
                        tag("{"),
                        separated_list0(
                            tag(","),
                            map(
                                tuple((
                                    tag("\""),
                                    satisfy(|c| c.is_ascii_lowercase()),
                                    tag("\":"),
                                    Self::parse,
                                )),
                                |(_, field_name, _, field_value)| (field_name, field_value),
                            ),
                        ),
                        tag("}"),
                    ),
                    |fields: &Vec<(char, Json)>| {
                        LetterCounts::from(fields.iter().map(|(field_name, _)| *field_name as u8)).0
                            [0_usize]
                            .count
                            <= 1_u8
                    },
                ),
                Self::Object,
            ),
            map(parse_integer, Self::Number),
            map(
                delimited(
                    tag("\""),
                    JsonString::parse_char1(1_usize, |c| c.is_ascii_lowercase()),
                    tag("\""),
                ),
                Self::String,
            ),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Json);

impl Solution {
    fn sum_numbers(&self) -> i32 {
        let mut sum: i32 = 0_i32;

        self.0.visit_numbers(&|_| true, &mut |number| {
            sum += number;
        });

        sum
    }

    fn sum_numbers_skipping_red(&self) -> i32 {
        let mut sum: i32 = 0_i32;

        self.0.visit_numbers(
            &|fields| {
                fields.iter().all(|(_, field_value)| match field_value {
                    Json::String(string) => string.as_str() != "red",
                    _ => true,
                })
            },
            &mut |number| {
                sum += number;
            },
        );

        sum
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Json::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Simpler to parse than anticipated.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_numbers());
    }

    /// No biggie.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_numbers_skipping_red());
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
        "[1,2,3]",
        r#"{"a":2,"b":4}"#,
        "[[[3]]]",
        r#"{"a":{"b":4},"c":-1}"#,
        r#"{"a":[-1,1]}"#,
        r#"[-1,{"a":1}]"#,
        "[]",
        "{}",
        r#"[1,{"c":"red","b":2},3]"#,
        r#"{"d":"red","e":[1,2,3,4],"f":5}"#,
        r#"[1,"red",5]"#,
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(Json::Array(vec![
                    Json::Number(1_i32),
                    Json::Number(2_i32),
                    Json::Number(3_i32),
                ])),
                Solution(Json::Object(vec![
                    ('a', Json::Number(2_i32)),
                    ('b', Json::Number(4_i32)),
                ])),
                Solution(Json::Array(vec![Json::Array(vec![Json::Array(vec![
                    Json::Number(3_i32),
                ])])])),
                Solution(Json::Object(vec![
                    ('a', Json::Object(vec![('b', Json::Number(4_i32))])),
                    ('c', Json::Number(-1_i32)),
                ])),
                Solution(Json::Object(vec![(
                    'a',
                    Json::Array(vec![Json::Number(-1_i32), Json::Number(1_i32)]),
                )])),
                Solution(Json::Array(vec![
                    Json::Number(-1_i32),
                    Json::Object(vec![('a', Json::Number(1_i32))]),
                ])),
                Solution(Json::Array(Vec::new())),
                Solution(Json::Object(Vec::new())),
                Solution(Json::Array(vec![
                    Json::Number(1_i32),
                    Json::Object(vec![
                        ('c', Json::String("red".try_into().unwrap())),
                        ('b', Json::Number(2_i32)),
                    ]),
                    Json::Number(3_i32),
                ])),
                Solution(Json::Object(vec![
                    ('d', Json::String("red".try_into().unwrap())),
                    (
                        'e',
                        Json::Array(vec![
                            Json::Number(1_i32),
                            Json::Number(2_i32),
                            Json::Number(3_i32),
                            Json::Number(4_i32),
                        ]),
                    ),
                    ('f', Json::Number(5_i32)),
                ])),
                Solution(Json::Array(vec![
                    Json::Number(1_i32),
                    Json::String("red".try_into().unwrap()),
                    Json::Number(5_i32),
                ])),
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
    fn test_sum_numbers() {
        for (index, numbers_sum) in [
            6_i32, 6_i32, 3_i32, 3_i32, 0_i32, 0_i32, 0_i32, 0_i32, 6_i32, 15_i32, 6_i32,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).sum_numbers(), numbers_sum);
        }
    }

    #[test]
    fn test_sum_numbers_skipping_red() {
        for (index, numbers_sum_skipping_red) in [
            6_i32, 6_i32, 3_i32, 3_i32, 0_i32, 0_i32, 0_i32, 0_i32, 4_i32, 0_i32, 6_i32,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).sum_numbers_skipping_red(),
                numbers_sum_skipping_red
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
