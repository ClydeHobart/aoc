use {
    crate::*,
    nom::{
        bytes::complete::take_while1,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::terminated,
        Err,
    },
    std::ops::Range,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Status {
    Valid,
    Incomplete(u16),
    Corrupted(u16),
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Line {
    range: Range<u16>,
    status: Status,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    lines: Vec<Line>,
    string: String,
}

impl Solution {
    fn map_left_to_right(byte: u8) -> Option<u8> {
        Some(match byte {
            b'(' => b')',
            b'[' => b']',
            b'{' => b'}',
            b'<' => b'>',
            _ => None?,
        })
    }

    fn get_line(&self, index: usize) -> Option<(&Line, &str)> {
        self.lines.get(index).and_then(|line| {
            Some((
                line,
                self.string
                    .get(line.range.start as usize..line.range.end as usize)?,
            ))
        })
    }

    fn iter_lines(&self) -> impl Iterator<Item = (&Line, &str)> + '_ {
        (0_usize..self.lines.len()).filter_map(|index| self.get_line(index))
    }

    fn iter_corrupted_lines(&self) -> impl Iterator<Item = (&Line, &str)> + '_ {
        self.iter_lines()
            .filter(|(line, _)| matches!(line.status, Status::Corrupted(_)))
    }

    fn iter_incomplete_lines(&self) -> impl Iterator<Item = (&Line, &str)> + '_ {
        self.iter_lines()
            .filter(|(line, _)| matches!(line.status, Status::Incomplete(_)))
    }

    fn score_corrupted_line((line, line_str): (&Line, &str)) -> u32 {
        const ROUND: u32 = 3_u32;
        const SQUARE: u32 = 57_u32;
        const CURLY: u32 = 1_197_u32;
        const ANGLE: u32 = 25_137_u32;

        match line.status {
            Status::Corrupted(index) => line_str
                .as_bytes()
                .get(index as usize)
                .map(|byte| match *byte {
                    b')' => ROUND,
                    b']' => SQUARE,
                    b'}' => CURLY,
                    b'>' => ANGLE,
                    _ => 0_u32,
                })
                .unwrap_or_default(),
            _ => 0_u32,
        }
    }

    fn score_incomplete_line(&self, line: &Line) -> u64 {
        const ROUND: u64 = 1_u64;
        const SQUARE: u64 = 2_u64;
        const CURLY: u64 = 3_u64;
        const ANGLE: u64 = 4_u64;
        const RATIO: u64 = 5_u64;

        match line.status {
            Status::Incomplete(len) => self
                .string
                .get(line.range.end as usize..(line.range.end + len) as usize)
                .unwrap_or("")
                .as_bytes()
                .iter()
                .fold(0_u64, |score, byte| {
                    score * RATIO
                        + match *byte {
                            b')' => ROUND,
                            b']' => SQUARE,
                            b'}' => CURLY,
                            b'>' => ANGLE,
                            _ => 0_u64,
                        }
                }),
            _ => 0_u64,
        }
    }

    fn total_syntax_error_score(&self) -> u32 {
        self.iter_corrupted_lines()
            .map(Self::score_corrupted_line)
            .sum()
    }

    fn middle_completion_score(&self) -> u64 {
        let mut completion_scores: Vec<u64> = self
            .iter_incomplete_lines()
            .map(|(line, _)| self.score_incomplete_line(line))
            .collect();

        completion_scores.sort();

        completion_scores
            .get(completion_scores.len() / 2_usize)
            .cloned()
            .unwrap_or_default()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_syntax_error_score());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.middle_completion_score());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        let mut string: String = String::with_capacity(input.len());
        let mut stack: Vec<u8> = Vec::new();

        let lines: Vec<Line> = many0(terminated(
            map(
                take_while1(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>')),
                |line: &'i str| {
                    stack.clear();

                    let start: u16 = string.len() as u16;

                    string.push_str(line);

                    let range: Range<u16> = start..start + line.len() as u16;
                    let status: Status = 'status: {
                        const INVALID_STACK_MESSAGE: &str =
                            "`stack` should only contain the specified bytes";
                        for (index, byte) in line.as_bytes().iter().copied().enumerate() {
                            if matches!(byte, b'(' | b'[' | b'{' | b'<') {
                                stack.push(byte);
                            } else if stack.pop().map_or(true, |last| {
                                Self::map_left_to_right(last).expect(INVALID_STACK_MESSAGE) != byte
                            }) {
                                break 'status Status::Corrupted(index as u16);
                            }
                        }

                        if stack.is_empty() {
                            Status::Valid
                        } else {
                            let completion_len: u16 = stack.len() as u16;

                            for byte in stack.drain(..).rev() {
                                string.push(
                                    Self::map_left_to_right(byte).expect(INVALID_STACK_MESSAGE)
                                        as char,
                                );
                            }

                            Status::Incomplete(completion_len)
                        }
                    };

                    Line { range, status }
                },
            ),
            opt(line_ending),
        ))(input)?
        .1;

        Ok(Self { lines, string })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const SOLUTION_STR: &str = concat!(
        "[({(<(())[]>[[{[]{<()<>>\n",
        "[(()[<>])]({[<{<<[]>>(\n",
        "{([(<{}[<>[]}>{[]{[(<()>\n",
        "(((({<>}<{<{<>}{[]{[]{}\n",
        "[[<[([]))<([[{}[[()]]]\n",
        "[{[{({}]{}}([{[{{{}}([]\n",
        "{<[[]]>}<{[{[{[]{()[[[]\n",
        "[<(<(<(<{}))><([]([]()\n",
        "<{([([[(<>()){}]>(<<{{\n",
        "<{([{{}}[<[[[<>{}]]]>[]]\n",
    );

    lazy_static! {
        static ref SOLUTION: Solution = solution();
    }

    fn solution() -> Solution {
        use Status::{Corrupted as C, Incomplete as I};

        macro_rules! lines {
            [ $( { $range:expr, $status:expr, }, )* ] => {
                vec![ $( Line { range: $range, status: $status, }, )* ]
            };
        }

        Solution {
            lines: lines![
                // [({([[{{
                { 0..24, I(8), },
                // "({[<{("
                { 32..54, I(6), },
                { 60..84, C(12), },
                // "((((<{<{{",
                { 84..107, I(9), },
                { 116..138, C(8), },
                { 138..161, C(7), },
                // "<{[{[{{[["
                { 161..184, I(9), },
                { 193..215, C(10), },
                { 215..237, C(16), },
                // "<{(["
                { 237..261, I(4), },
            ],
            string: concat!(
                "[({(<(())[]>[[{[]{<()<>>",
                "}}]])})]",
                "[(()[<>])]({[<{<<[]>>(",
                ")}>]})",
                "{([(<{}[<>[]}>{[]{[(<()>",
                "(((({<>}<{<{<>}{[]{[]{}",
                "}}>}>))))",
                "[[<[([]))<([[{}[[()]]]",
                "[{[{({}]{}}([{[{{{}}([]",
                "{<[[]]>}<{[{[{[]{()[[[]",
                "]]}}]}]}>",
                "[<(<(<(<{}))><([]([]()",
                "<{([([[(<>()){}]>(<<{{",
                "<{([{{}}[<[[[<>{}]]]>[]]",
                "])}>",
            )
            .into(),
        }
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR), Ok(solution()))
    }

    #[test]
    fn test_total_syntax_error_score() {
        assert_eq!(SOLUTION.total_syntax_error_score(), 26_397_u32);
    }

    #[test]
    fn test_middle_completion_score() {
        assert_eq!(SOLUTION.middle_completion_score(), 288_957_u64);
    }
}
