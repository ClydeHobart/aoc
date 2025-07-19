use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0_count,
        sequence::preceded,
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 8: Matchsticks ---

Space on the sleigh is limited this year, and so Santa will be bringing his list as a digital copy. He needs to know how much space it will take up when stored.

It is common in many programming languages to provide a way to escape special characters in strings. For example, C, JavaScript, Perl, Python, and even PHP handle special characters in very similar ways.

However, it is important to realize the difference between the number of characters in the code representation of the string literal and the number of characters in the in-memory string itself.

For example:

    "" is 2 characters of code (the two double quotes), but the string contains zero characters.
    "abc" is 5 characters of code, but 3 characters in the string data.
    "aaa\"aaa" is 10 characters of code, but the string itself contains six "a" characters and a single, escaped quote character, for a total of 7 characters in the string data.
    "\x27" is 6 characters of code, but the string itself contains just one - an apostrophe ('), escaped using hexadecimal notation.

Santa's list is a file that contains many double-quoted string literals, one on each line. The only escape sequences used are \\ (which represents a single backslash), \" (which represents a lone double-quote character), and \x plus two hexadecimal characters (which represents a single character with that ASCII code).

Disregarding the whitespace in the file, what is the number of characters of code for string literals minus the number of characters in memory for the values of the strings in total for the entire file?

For example, given the four strings above, the total number of characters of string code (2 + 5 + 10 + 6 = 23) minus the total number of characters in memory for string values (0 + 3 + 7 + 1 = 11) is 23 - 11 = 12.

--- Part Two ---

Now, let's go the other way. In addition to finding the number of characters of code, you should now encode each code representation as a new string and find the number of characters of the new encoded representation, including the surrounding double quotes.

For example:

    "" encodes to "\"\"", an increase from 2 characters to 6.
    "abc" encodes to "\"abc\"", an increase from 5 characters to 9.
    "aaa\"aaa" encodes to "\"aaa\\\"aaa\"", an increase from 10 characters to 16.
    "\x27" encodes to "\"\\x27\"", an increase from 6 characters to 11.

Your task is to find the total number of characters to represent the newly encoded strings minus the number of characters of code in each original string literal. For example, for the strings above, the total encoded length (6 + 9 + 16 + 11 = 42) minus the characters in the original code representation (23, just like in the first part of this puzzle) is 42 - 23 = 19. */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct StringLiteral {
    code: Range<u16>,
    decoded: Range<u16>,
    encoded: Range<u16>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    code: String,
    decoded: Vec<u8>,
    encoded: String,
    literals: Vec<StringLiteral>,
}

impl Solution {
    fn code_byte_count(&self) -> usize {
        self.literals.iter().map(|literal| literal.code.len()).sum()
    }

    fn decoded_byte_count(&self) -> usize {
        self.literals
            .iter()
            .map(|literal| literal.decoded.len())
            .sum()
    }

    fn encoded_byte_count(&self) -> usize {
        self.literals
            .iter()
            .map(|literal| literal.encoded.len())
            .sum()
    }

    fn extra_decoded_byte_count(&self) -> usize {
        self.code_byte_count() - self.decoded_byte_count()
    }

    fn extra_encoded_byte_count(&self) -> usize {
        self.encoded_byte_count() - self.code_byte_count()
    }

    fn print_encoded_code_decoded(&self) {
        for literal in &self.literals {
            println!(
                "{:70}{:50}{:50}",
                &self.encoded[literal.encoded.as_range_usize()],
                &self.code[literal.code.as_range_usize()],
                (&self.decoded[literal.decoded.as_range_usize()])
                    .iter()
                    .copied()
                    .map(char::from)
                    .collect::<String>()
            );
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let code: String = input.into();
        let mut decoded: Vec<u8> = Vec::with_capacity(code.len());
        let mut encoded: String = String::with_capacity(code.len() * 3_usize / 2_usize);
        let mut literals: Vec<StringLiteral> = Vec::new();
        let mut code_len: u16 = 0_u16;

        let input: &str = many0_count(|input| {
            let code_start: u16 = code_len;
            let decoded_start: u16 = decoded.len() as u16;
            let encoded_start: u16 = encoded.len() as u16;
            let input: &str = tag("\"")(input)?.0;

            code_len += 1_u16;
            encoded.push_str("\"\\\"");

            let input: &str = many0_count(map(
                alt((
                    preceded(
                        tag("\\"),
                        alt((
                            map(tag("\\"), |_| (2_u16, b'\\', "\\\\\\\\", "")),
                            map(tag("\""), |_| (2_u16, b'"', "\\\\\\\"", "")),
                            map(
                                preceded(
                                    tag("x"),
                                    take_while_m_n(2_usize, 2_usize, |c: char| {
                                        c.is_ascii_hexdigit()
                                    }),
                                ),
                                |hex_str| {
                                    (
                                        4_u16,
                                        u8::from_str_radix(hex_str, 16_u32).unwrap(),
                                        "\\\\x",
                                        hex_str,
                                    )
                                },
                            ),
                        )),
                    ),
                    map(
                        take_while_m_n(1_usize, 1_usize, |c| c != '\"'),
                        |byte_str: &str| (1_u16, byte_str.as_bytes()[0_usize], "", byte_str),
                    ),
                )),
                |(additional_code_len, decoded_byte, encoded_str_a, encoded_str_b)| {
                    code_len += additional_code_len;
                    decoded.push(decoded_byte);
                    encoded.push_str(encoded_str_a);
                    encoded.push_str(encoded_str_b);
                },
            ))(input)?
            .0;
            let input: &str = tag("\"")(input)?.0;

            code_len += 1_u16;
            encoded.push_str("\\\"\"");

            let code_end: u16 = code_len;
            let decoded_end: u16 = decoded.len() as u16;
            let encoded_end: u16 = encoded.len() as u16;

            literals.push(StringLiteral {
                code: code_start..code_end,
                decoded: decoded_start..decoded_end,
                encoded: encoded_start..encoded_end,
            });

            let (input, line_ending_str): (&str, Option<&str>) = opt(line_ending)(input)?;

            if let Some(line_ending_str) = line_ending_str {
                code_len += line_ending_str.len() as u16;
            }

            Ok((input, ()))
        })(input)?
        .0;

        Ok((
            input,
            Self {
                code,
                decoded,
                encoded,
                literals,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// In 2015, Eric Wastl did not know that only bytes in the range 0x00-0x7F are valid ASCII. I
    /// will never forgive him for this transgression.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.extra_decoded_byte_count());
    }

    /// Still pissed about part 1
    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.extra_encoded_byte_count());

        if args.verbose {
            self.print_encoded_code_decoded();
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
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        \"\"\n\
        \"abc\"\n\
        \"aaa\\\"aaa\"\n\
        \"\\x27\"\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                code: "\"\"\n\"abc\"\n\"aaa\\\"aaa\"\n\"\\x27\"\n".into(),
                decoded: b"abcaaa\"aaa'".into(),
                encoded: "\
                    \"\\\"\\\"\"\
                    \"\\\"abc\\\"\"\
                    \"\\\"aaa\\\\\\\"aaa\\\"\"\
                    \"\\\"\\\\x27\\\"\""
                    .into(),
                literals: vec![
                    StringLiteral {
                        code: 0_u16..2_u16,
                        decoded: 0_u16..0_u16,
                        encoded: 0_u16..6_u16,
                    },
                    StringLiteral {
                        code: 3_u16..8_u16,
                        decoded: 0_u16..3_u16,
                        encoded: 6_u16..15_u16,
                    },
                    StringLiteral {
                        code: 9_u16..19_u16,
                        decoded: 3_u16..10_u16,
                        encoded: 15_u16..31_u16,
                    },
                    StringLiteral {
                        code: 20_u16..26_u16,
                        decoded: 10_u16..11_u16,
                        encoded: 31_u16..42_u16,
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
    fn test_code_byte_count() {
        for (index, code_byte_count) in [23_usize].into_iter().enumerate() {
            assert_eq!(solution(index).code_byte_count(), code_byte_count);
        }
    }

    #[test]
    fn test_decoded_byte_count() {
        for (index, decoded_byte_count) in [11_usize].into_iter().enumerate() {
            assert_eq!(solution(index).decoded_byte_count(), decoded_byte_count);
        }
    }

    #[test]
    fn test_extra_decoded_byte_count() {
        for (index, extra_decoded_character_count) in [12_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).extra_decoded_byte_count(),
                extra_decoded_character_count
            );
        }
    }

    #[test]
    fn test_encoded_byte_count() {
        for (index, encoded_byte_count) in [42_usize].into_iter().enumerate() {
            assert_eq!(solution(index).encoded_byte_count(), encoded_byte_count);
        }
    }

    #[test]
    fn test_extra_encoded_byte_count() {
        for (index, extra_encoded_character_count) in [19_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).extra_encoded_byte_count(),
                extra_encoded_character_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
