use {
    crate::*,
    nom::{
        bytes::complete::take_while1,
        character::complete::satisfy,
        combinator::{map, map_opt},
        error::Error,
        multi::many0,
        Err, IResult,
    },
    std::{
        fmt::{Display, Formatter, Result as FmtResult, Write},
        mem::swap,
    },
};

/* --- Day 10: Elves Look, Elves Say ---

Today, the Elves are playing a game called look-and-say. They take turns making sequences by reading aloud the previous sequence and using that reading as the next sequence. For example, 211 is read as "one two, two ones", which becomes 1221 (1 2, 2 1s).

Look-and-say sequences are generated iteratively, using the previous value as input for the next step. For each step, take the previous value, and replace each run of digits (like 111) with the number of digits (3) followed by the digit itself (1).

For example:

    1 becomes 11 (1 copy of digit 1).
    11 becomes 21 (2 copies of digit 1).
    21 becomes 1211 (one 2 followed by one 1).
    1211 becomes 111221 (one 1, one 2, and two 1s).
    111221 becomes 312211 (three 1s, two 2s, and one 1).

Starting with the digits in your puzzle input, apply this process 40 times. What is the length of the result?

--- Part Two ---

Neat, right? You might also enjoy hearing John Conway talking about this sequence (that's Conway of Conway's Game of Life fame).

Now, starting again with the digits in your puzzle input, apply this process 50 times. What is the length of the new result? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Run {
    length: u8,
    value: u8,
}

impl Run {
    const MAX_LENGTH_USIZE: usize = u8::MAX as usize;
}

impl Display for Run {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let c: char = (self.value + b'0') as char;

        for _ in 0_u8..self.length {
            f.write_char(c)?;
        }

        Ok(())
    }
}

impl Parse for Run {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let digit: char = satisfy(|c| c.is_ascii_digit())(input)?.1;
        let value: u8 = digit as u8 - b'0';
        let (input, length): (&str, u8) = map_opt(take_while1(|c| c == digit), |run_str: &str| {
            let length: usize = run_str.len();

            (length <= Self::MAX_LENGTH_USIZE).then(|| length as u8)
        })(input)?;

        Ok((input, Self { length, value }))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
pub struct Solution(Vec<Run>);

impl Solution {
    const Q1_ITERATIONS: usize = 40_usize;
    const Q2_ITERATIONS: usize = 50_usize;

    fn iterate(&self, iterations: usize) -> Self {
        let mut current: Self = self.clone();
        let mut next: Self = Default::default();

        for _ in 0_usize..iterations {
            for run in current.0.iter() {
                next.append_digit(run.length);
                next.append_digit(run.value);
            }

            swap(&mut current, &mut next);
            next.0.clear();
        }

        current
    }

    fn length(&self) -> usize {
        self.0.iter().map(|run| run.length as usize).sum()
    }

    fn append_digit(&mut self, value: u8) {
        if let Some(last_run_length) = self
            .0
            .last_mut()
            .and_then(|last_run| (last_run.value == value).then_some(&mut last_run.length))
        {
            *last_run_length += 1_u8;
        } else {
            self.0.push(Run {
                length: 1_u8,
                value,
            })
        }
    }
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for run in &self.0 {
            run.fmt(f)?
        }

        Ok(())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(Run::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I expect Q2 is "now do it for 50 million" but I'm not sure how to disentangle a sequence
    /// from coupling with its neighbors.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.iterate(Self::Q1_ITERATIONS).length());
    }

    /// What a pleasant surprise!
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.iterate(Self::Q2_ITERATIONS).length());
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

    const SOLUTION_STRS: &'static [&'static str] = &["211", "1", "11", "21", "1211", "111221"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    Run {
                        length: 1_u8,
                        value: 2_u8,
                    },
                    Run {
                        length: 2_u8,
                        value: 1_u8,
                    },
                ]),
                Solution(vec![Run {
                    length: 1_u8,
                    value: 1_u8,
                }]),
                Solution(vec![Run {
                    length: 2_u8,
                    value: 1_u8,
                }]),
                Solution(vec![
                    Run {
                        length: 1_u8,
                        value: 2_u8,
                    },
                    Run {
                        length: 1_u8,
                        value: 1_u8,
                    },
                ]),
                Solution(vec![
                    Run {
                        length: 1_u8,
                        value: 1_u8,
                    },
                    Run {
                        length: 1_u8,
                        value: 2_u8,
                    },
                    Run {
                        length: 2_u8,
                        value: 1_u8,
                    },
                ]),
                Solution(vec![
                    Run {
                        length: 3_u8,
                        value: 1_u8,
                    },
                    Run {
                        length: 2_u8,
                        value: 2_u8,
                    },
                    Run {
                        length: 1_u8,
                        value: 1_u8,
                    },
                ]),
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
    fn test_to_string() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(solution(index).to_string(), solution_str);
        }
    }

    #[test]
    fn test_iterate() {
        for (index, iterated_solution_str) in ["1221", "11", "21", "1211", "111221", "312211"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).iterate(1_usize).to_string(),
                iterated_solution_str
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
