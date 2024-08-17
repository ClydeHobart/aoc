use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{line_ending, satisfy},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::mem::swap,
};

/* --- Day 21: Scrambled Letters and Hash ---

The computer system you're breaking into uses a weird scrambling function to store its passwords. It shouldn't be much trouble to create your own scrambled password so you can add it to the system; you just have to implement the scrambler.

The scrambling function is a series of operations (the exact list is provided in your puzzle input). Starting with the password to be scrambled, apply each operation in succession to the string. The individual operations behave as follows:

    swap position X with position Y means that the letters at indexes X and Y (counting from 0) should be swapped.
    swap letter X with letter Y means that the letters X and Y should be swapped (regardless of where they appear in the string).
    rotate left/right X steps means that the whole string should be rotated; for example, one right rotation would turn abcd into dabc.
    rotate based on position of letter X means that the whole string should be rotated to the right based on the index of letter X (counting from 0) as determined before this instruction does any rotations. Once the index is determined, rotate the string to the right one time, plus a number of times equal to that index, plus one additional time if the index was at least 4.
    reverse positions X through Y means that the span of letters at indexes X through Y (including the letters at X and Y) should be reversed in order.
    move position X to position Y means that the letter which is at index X should be removed from the string, then inserted such that it ends up at index Y.

For example, suppose you start with abcde and perform the following operations:

    swap position 4 with position 0 swaps the first and last letters, producing the input for the next step, ebcda.
    swap letter d with letter b swaps the positions of d and b: edcba.
    reverse positions 0 through 4 causes the entire string to be reversed, producing abcde.
    rotate left 1 step shifts all letters left one position, causing the first letter to wrap to the end of the string: bcdea.
    move position 1 to position 4 removes the letter at position 1 (c), then inserts it at position 4 (the end of the string): bdeac.
    move position 3 to position 0 removes the letter at position 3 (a), then inserts it at position 0 (the front of the string): abdec.
    rotate based on position of letter b finds the index of letter b (1), then rotates the string right once plus a number of times equal to that index (2): ecabd.
    rotate based on position of letter d finds the index of letter d (4), then rotates the string right once, plus a number of times equal to that index, plus an additional time because the index was at least 4, for a total of 6 right rotations: decab.

After these steps, the resulting scrambled password is decab.

Now, you just need to generate a new scrambled password and you can access the system. Given the list of scrambling operations in your puzzle input, what is the result of scrambling abcdefgh?

--- Part Two ---

You scrambled the password correctly, but you discover that you can't actually modify the password file on the system. You'll need to un-scramble one of the existing passwords by reversing the scrambling process.

What is the un-scrambled version of the scrambled password fbgdceah?
*/

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Operation {
    SwapPosition { x: u8, y: u8 },
    SwapLetter { x: u8, y: u8 },
    Rotate { left: bool, steps: u8 },
    RotatePosition { x: u8 },
    ReversePositions { x: u8, y: u8 },
    MovePosition { x: u8, y: u8 },
    AntiRotatePosition { x: u8 },
}

impl Operation {
    const ANTI_ROTATE_POSITION_MAP: [u8; 8_usize] =
        [1_u8, 1_u8, 6_u8, 2_u8, 7_u8, 3_u8, 0_u8, 4_u8];

    fn parse_letter<'i>(input: &'i str) -> IResult<&'i str, u8> {
        map(satisfy(|c: char| c.is_ascii_alphabetic()), |c| c as u8)(input)
    }

    fn swap_positions(min: usize, max: usize, bytes: &mut [u8]) {
        let (min_bytes, max_bytes): (&mut [u8], &mut [u8]) = bytes.split_at_mut(max);

        swap(&mut min_bytes[min], &mut max_bytes[0_usize]);
    }

    fn try_execute(self, bytes: &mut [u8]) -> Option<()> {
        match self {
            Self::SwapPosition { x, y } => (x == y).then_some(()).or_else(|| {
                let (min, max): (usize, usize) = min_and_max(x as usize, y as usize);

                (max < bytes.len()).then_some((min, max)).map(|(min, max)| {
                    Self::swap_positions(min, max, bytes);
                })
            }),
            Self::SwapLetter { x, y } => (x == y).then_some(()).or_else(|| {
                bytes
                    .iter()
                    .position(|b| *b == x)
                    .zip(bytes.iter().position(|b| *b == y))
                    .map(|(x, y)| {
                        let (min, max): (usize, usize) = min_and_max(x, y);

                        Self::swap_positions(min, max, bytes);
                    })
            }),
            Self::Rotate { left, steps } => Some(if left {
                bytes.rotate_left(steps as usize);
            } else {
                bytes.rotate_right(steps as usize);
            }),
            Self::RotatePosition { x } => bytes.iter().position(|b| *b == x).map(|index| {
                bytes.rotate_right((1_usize + index + (index >= 4_usize) as usize) % bytes.len());
            }),
            Self::ReversePositions { x, y } => {
                let (min, max): (usize, usize) = min_and_max(x as usize, y as usize);

                bytes.get_mut(min..=max).map(|bytes| bytes.reverse())
            }
            Self::MovePosition { x, y } => (x == y).then_some(()).or_else(|| {
                let (min, max): (usize, usize) = min_and_max(x as usize, y as usize);

                bytes.get_mut(min..=max).map(|bytes| {
                    if x < y {
                        bytes.rotate_left(1_usize);
                    } else {
                        bytes.rotate_right(1_usize);
                    }
                })
            }),
            Self::AntiRotatePosition { x } => (bytes.len() == Self::ANTI_ROTATE_POSITION_MAP.len())
                .then(|| bytes.iter().position(|b| *b == x))
                .flatten()
                .map(|index| {
                    bytes.rotate_left(Self::ANTI_ROTATE_POSITION_MAP[index] as usize);
                }),
        }
    }

    fn try_invert(self, bytes_len: usize) -> Option<Self> {
        match self {
            Self::SwapPosition { x: _, y: _ } => Some(self),
            Self::SwapLetter { x: _, y: _ } => Some(self),
            Self::Rotate { left, steps } => Some(Self::Rotate { left: !left, steps }),
            // This is not a good question. It's very contrived to the parameters of the question
            // itself and not the general cases that they describe. Even the example password
            // itself, "abcde", is fundamentally incompatible with the concept of inverting this
            // operation type, since letters originally in indices 2 and 4 both map to being in
            // index 0, meaning given a 5-letter password, if we're trying to unscramble and we come
            // across a point where we're trying to reverse an operation of this type, and the
            // letter whose position we care about is in position 0, there's no way to know for
            // certain what the password state was prior to that operation.
            Self::RotatePosition { x } => (bytes_len == Self::ANTI_ROTATE_POSITION_MAP.len())
                .then_some(Self::AntiRotatePosition { x }),
            Self::ReversePositions { x: _, y: _ } => Some(self),
            Self::MovePosition { x, y } => Some(Self::MovePosition { x: y, y: x }),
            Self::AntiRotatePosition { x: _ } => None,
        }
    }
}

impl Parse for Operation {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                tuple((
                    tag("swap position "),
                    parse_integer,
                    tag(" with position "),
                    parse_integer,
                )),
                |(_, x, _, y)| Self::SwapPosition { x, y },
            ),
            map(
                tuple((
                    tag("swap letter "),
                    Self::parse_letter,
                    tag(" with letter "),
                    Self::parse_letter,
                )),
                |(_, x, _, y)| Self::SwapLetter { x, y },
            ),
            map(
                tuple((
                    tag("rotate "),
                    alt((map(tag("left "), |_| true), map(tag("right "), |_| false))),
                    parse_integer,
                    tag(" step"),
                    opt(tag("s")),
                )),
                |(_, left, steps, _, _)| Self::Rotate { left, steps },
            ),
            map(
                preceded(
                    tag("rotate based on position of letter "),
                    Self::parse_letter,
                ),
                |x| Self::RotatePosition { x },
            ),
            map(
                tuple((
                    tag("reverse positions "),
                    parse_integer,
                    tag(" through "),
                    parse_integer,
                )),
                |(_, x, _, y)| Self::ReversePositions { x, y },
            ),
            map(
                tuple((
                    tag("move position "),
                    parse_integer,
                    tag(" to position "),
                    parse_integer,
                )),
                |(_, x, _, y)| Self::MovePosition { x, y },
            ),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Operation>);

impl Solution {
    const PASSWORD: &'static str = "abcdefgh";
    const SCRAMBLED_PASSWORD: &'static str = "fbgdceah";

    fn try_scramble(&self, password: &str) -> Option<String> {
        let mut password: Vec<u8> = password.as_bytes().into();

        self.0
            .iter()
            .try_fold((), |_, operation| operation.try_execute(&mut password))
            .map(|_| String::from_utf8(password).ok())
            .flatten()
    }

    fn try_unscramble(&self, scrambled_password: &str) -> Option<String> {
        let mut scrambled_password: Vec<u8> = scrambled_password.as_bytes().into();

        self.0
            .iter()
            .rev()
            .try_fold((), |_, operation| {
                operation
                    .try_invert(scrambled_password.len())
                    .and_then(|operation| operation.try_execute(&mut scrambled_password))
            })
            .map(|_| String::from_utf8(scrambled_password).ok())
            .flatten()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Operation::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(scrambled_password) = self.try_scramble(Self::PASSWORD) {
                dbg!(
                    &scrambled_password,
                    self.try_unscramble(&scrambled_password)
                );
            } else {
                eprintln!("Failed to scramble password.");
            }
        } else {
            dbg!(self.try_scramble(Self::PASSWORD));
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_unscramble(Self::SCRAMBLED_PASSWORD));
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
    use {
        super::*,
        std::{str::from_utf8, sync::OnceLock},
    };

    const SOLUTION_STR: &'static str = "\
        swap position 4 with position 0\n\
        swap letter d with letter b\n\
        reverse positions 0 through 4\n\
        rotate left 1 step\n\
        move position 1 to position 4\n\
        move position 3 to position 0\n\
        rotate based on position of letter b\n\
        rotate based on position of letter d\n";
    const EXPECTED_BYTES: &'static [&'static [u8; 5_usize]] = &[
        b"ebcda", b"edcba", b"abcde", b"bcdea", b"bdeac", b"abdec", b"ecabd", b"decab",
    ];

    fn solution() -> &'static Solution {
        use Operation::*;

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                SwapPosition { x: 4_u8, y: 0_u8 },
                SwapLetter { x: b'd', y: b'b' },
                ReversePositions { x: 0_u8, y: 4_u8 },
                Rotate {
                    left: true,
                    steps: 1_u8,
                },
                MovePosition { x: 1_u8, y: 4_u8 },
                MovePosition { x: 3_u8, y: 0_u8 },
                RotatePosition { x: b'b' },
                RotatePosition { x: b'd' },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_execute() {
        let mut bytes: [u8; 5_usize] = [b'a', b'b', b'c', b'd', b'e'];

        assert_eq!(solution().0.len(), EXPECTED_BYTES.len());

        for (operation, expected_bytes) in solution().0.iter().zip(EXPECTED_BYTES.iter().copied()) {
            assert_eq!(operation.try_execute(&mut bytes), Some(()));
            assert_eq!(
                &bytes,
                expected_bytes,
                "bytes as str: {:?},\nexpected bytes as str: {:?}",
                from_utf8(&bytes),
                from_utf8(expected_bytes)
            );
        }
    }

    #[test]
    fn test_try_scramble() {
        assert_eq!(solution().try_scramble("abcde"), Some("decab".into()));
    }
}
