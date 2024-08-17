use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::{tag, take, take_till1, take_while_m_n},
        character::complete::not_line_ending,
        combinator::{map, verify},
        error::Error,
        multi::many0,
        sequence::tuple,
        Err, IResult,
    },
    std::{iter::repeat, ops::Range},
};

/* --- Day 9: Explosives in Cyberspace ---

Wandering around a secure area, you come across a datalink port to a new part of the network. After briefly scanning it for interesting files, you find one file in particular that catches your attention. It's compressed with an experimental format, but fortunately, the documentation for the format is nearby.

The format compresses a sequence of characters. Whitespace is ignored. To indicate that some sequence should be repeated, a marker is added to the file, like (10x2). To decompress this marker, take the subsequent 10 characters and repeat them 2 times. Then, continue reading the file after the repeated data. The marker itself is not included in the decompressed output.

If parentheses or other characters appear within the data referenced by a marker, that's okay - treat it like normal data, not a marker, and then resume looking for markers after the decompressed section.

For example:

    ADVENT contains no markers and decompresses to itself with no changes, resulting in a decompressed length of 6.
    A(1x5)BC repeats only the B a total of 5 times, becoming ABBBBBC for a decompressed length of 7.
    (3x3)XYZ becomes XYZXYZXYZ for a decompressed length of 9.
    A(2x2)BCD(2x2)EFG doubles the BC and EF, becoming ABCBCDEFEFG for a decompressed length of 11.
    (6x1)(1x3)A simply becomes (1x3)A - the (1x3) looks like a marker, but because it's within a data section of another marker, it is not treated any differently from the A that comes after it. It has a decompressed length of 6.
    X(8x2)(3x3)ABCY becomes X(3x3)ABC(3x3)ABCY (for a decompressed length of 18), because the decompressed data from the (8x2) marker (the (3x3)ABC) is skipped and not processed further.

What is the decompressed length of the file (your puzzle input)? Don't count whitespace. */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Marker {
    range: Range<u32>,
    repetitions: u32,
}

impl Marker {
    fn parse_len_and_repetitions<'i>(input: &'i str) -> IResult<&'i str, (usize, usize)> {
        map(
            tuple((tag("("), parse_integer, tag("x"), parse_integer, tag(")"))),
            |(_, len, _, repetitions, _)| (len, repetitions),
        )(input)
    }
}

impl Parse for Marker {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, (len, repetitions)) = Self::parse_len_and_repetitions(input)?;
        let len: u32 = len as u32;
        let repetitions: u32 = repetitions as u32;
        let (input, _) = take(len)(input)?;

        Ok((
            input,
            Self {
                range: 0_u32..len,
                repetitions,
            },
        ))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Str {
    range: Range<u32>,
}

impl Str {
    fn parse_until_marker<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        take_till1(|c| c == '(' || c == '\n' || c == '\r')(input)
    }
}

impl Parse for Str {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (remaining, _) = Self::parse_until_marker(input)?;

        Ok((
            remaining,
            Self {
                range: 0_u32..(input.len() - remaining.len()) as u32,
            },
        ))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Token {
    Marker(Marker),
    Str(Str),
}

impl Token {
    fn repetitions(&self) -> usize {
        match self {
            Token::Marker(marker) => marker.repetitions as usize,
            Token::Str(_) => 1_usize,
        }
    }

    fn range(&self) -> Range<usize> {
        match self {
            Token::Marker(marker) => &marker.range,
            Token::Str(s) => &s.range,
        }
        .as_range_usize()
    }

    fn range_mut(&mut self) -> &mut Range<u32> {
        match self {
            Token::Marker(marker) => &mut marker.range,
            Token::Str(s) => &mut s.range,
        }
    }

    fn iter_strs<'s>(&self, string: &'s str) -> impl Iterator<Item = &'s str> {
        let string: &str = &string[self.range()];

        repeat(()).take(self.repetitions()).map(move |_| string)
    }
}

impl Parse for Token {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((map(Marker::parse, Self::Marker), map(Str::parse, Self::Str)))(input)
    }
}

enum Node {
    String(usize),
    Marker {
        nodes: Vec<Node>,
        repetitions: usize,
    },
}

impl Node {
    fn parse_string<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map(Str::parse_until_marker, str::len), Self::String)(input)
    }

    fn parse_marker<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, (len, repetitions)) = Marker::parse_len_and_repetitions(input)?;
        let (input, nodes_input) = take(len)(input)?;
        let (remaining_nodes_input, nodes) = many0(Self::parse)(nodes_input)?;
        let (_, _) = verify(
            take_while_m_n(0_usize, 1_usize, |_: char| true),
            str::is_empty,
        )(remaining_nodes_input)?;

        Ok((input, Self::Marker { nodes, repetitions }))
    }

    fn decompressed_len(&self) -> usize {
        match self {
            Node::String(len) => *len,
            Node::Marker { nodes, repetitions } => {
                *repetitions * nodes.iter().map(Self::decompressed_len).sum::<usize>()
            }
        }
    }
}

impl Parse for Node {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((Self::parse_string, Self::parse_marker))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    string: String,
    tokens: Vec<Token>,
}

impl Solution {
    fn iter_strs(&self) -> impl Iterator<Item = &str> {
        self.tokens
            .iter()
            .flat_map(|token| token.iter_strs(&self.string))
    }

    fn decompress(&self) -> String {
        let mut payload: String = String::new();

        for string in self.iter_strs() {
            payload.push_str(string);
        }

        payload
    }

    fn decompressed_len(&self) -> usize {
        self.iter_strs().map(str::len).sum()
    }

    fn version_two_decompressed_len_for_str(input: &str) -> usize {
        many0(Node::parse)(input)
            .unwrap_or_default()
            .1
            .iter()
            .map(Node::decompressed_len)
            .sum()
    }

    fn version_two_decompressed_len(&self) -> usize {
        Self::version_two_decompressed_len_for_str(&self.string)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (remaining, string) = map(not_line_ending, String::from)(input)?;

        let mut input: &str = input;
        let mut tokens: Vec<Token> = Vec::new();
        let mut offset: u32 = 0_u32;

        loop {
            match Token::parse(input) {
                Ok((next_input, mut token)) => {
                    let parsed_len: u32 = (input.len() - next_input.len()) as u32;
                    let range: &mut Range<u32> = token.range_mut();

                    range.start = offset + parsed_len - range.end;
                    range.end += range.start;
                    offset += parsed_len;
                    tokens.push(token);
                    input = next_input;
                }
                Result::Err(err) => match err {
                    Err::Incomplete(_) => unimplemented!(),
                    Err::Error(_) => break,
                    Err::Failure(e) => return Err(Err::Failure(e)),
                },
            }
        }

        Ok((remaining, Self { string, tokens }))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let payload: String = self.decompress();

            dbg!(payload.len());
            println!("{payload}");
        } else {
            dbg!(self.decompressed_len());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.version_two_decompressed_len());
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
        "ADVENT",
        "A(1x5)BC",
        "(3x3)XYZ",
        "A(2x2)BCD(2x2)EFG",
        "(6x1)(1x3)A",
        "X(8x2)(3x3)ABCY",
    ];
    const DECOMPRESSED_STRS: &'static [&'static str] = &[
        "ADVENT",
        "ABBBBBC",
        "XYZXYZXYZ",
        "ABCBCDEFEFG",
        "(1x3)A",
        "X(3x3)ABC(3x3)ABCY",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! token {
            ($range:expr) => {
                Token::Str(Str { range: $range })
            };
            ($range:expr, $repetitions:expr) => {
                Token::Marker(Marker {
                    range: $range,
                    repetitions: $repetitions,
                })
            };
        }

        macro_rules! tokens {
            [ $( ( $range:expr $(, $repetitions:expr)? ), )* ] => {
                vec![ $( token!( $range $(, $repetitions)? ), )* ]
            }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    string: SOLUTION_STRS[0_usize].into(),
                    tokens: tokens![(0..6),],
                },
                Solution {
                    string: SOLUTION_STRS[1_usize].into(),
                    tokens: tokens![(0..1), (6..7, 5), (7..8),],
                },
                Solution {
                    string: SOLUTION_STRS[2_usize].into(),
                    tokens: tokens![(5..8, 3),],
                },
                Solution {
                    string: SOLUTION_STRS[3_usize].into(),
                    tokens: tokens![(0..1), (6..8, 2), (8..9), (14..16, 2), (16..17),],
                },
                Solution {
                    string: SOLUTION_STRS[4_usize].into(),
                    tokens: tokens![(5..11, 1),],
                },
                Solution {
                    string: SOLUTION_STRS[5_usize].into(),
                    tokens: tokens![(0..1), (6..14, 2), (14..15),],
                },
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_decompress() {
        for (index, expected) in DECOMPRESSED_STRS.into_iter().copied().enumerate() {
            assert_eq!(solution(index).decompress(), expected);
        }
    }

    #[test]
    fn test_decompressed_len() {
        for (index, expected) in DECOMPRESSED_STRS.into_iter().copied().enumerate() {
            assert_eq!(solution(index).decompressed_len(), expected.len());
        }
    }

    #[test]
    fn test_version_two_decompressed_len() {
        for (string, len) in [
            ("(3x3)XYZ", 9_usize),
            ("X(8x2)(3x3)ABCY", 20_usize),
            ("(27x12)(20x12)(13x14)(7x10)(1x12)A", 241920_usize),
            (
                "(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN",
                445_usize,
            ),
        ] {
            assert_eq!(Solution::version_two_decompressed_len_for_str(string), len);
        }
    }
}
