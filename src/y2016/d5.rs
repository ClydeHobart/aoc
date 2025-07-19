use {
    crate::*,
    md5::compute,
    nom::{character::complete::not_line_ending, combinator::map, error::Error, Err, IResult},
    std::{
        fmt::{Debug, Write},
        iter::repeat,
        mem::size_of,
    },
};

/* --- Day 5: How About a Nice Game of Chess? ---

You are faced with a security door designed by Easter Bunny engineers that seem to have acquired most of their security knowledge by watching hacking movies.

The eight-character password for the door is generated one character at a time by finding the MD5 hash of some Door ID (your puzzle input) and an increasing integer index (starting with 0).

A hash indicates the next character in the password if its hexadecimal representation starts with five zeroes. If it does, the sixth character in the hash is the next character of the password.

For example, if the Door ID is abc:

    The first index which produces a hash that starts with five zeroes is 3231929, which we find by hashing abc3231929; the sixth character of the hash, and thus the first character of the password, is 1.
    5017308 produces the next interesting hash, which starts with 000008f82..., so the second character of the password is 8.
    The third time a hash starts with five zeroes is for abc5278568, discovering the character f.

In this example, after continuing this search a total of eight times, the password is 18f47a30.

Given the actual Door ID, what is the password?

--- Part Two ---

As the door slides open, you are presented with a second door that uses a slightly more inspired security mechanism. Clearly unimpressed by the last version (in what movie is the password decrypted in order?!), the Easter Bunny engineers have worked out a better solution.

Instead of simply filling in the password from left to right, the hash now also indicates the position within the password to fill. You still look for hashes that begin with five zeroes; however, now, the sixth character represents the position (0-7), and the seventh character is the character to put in that position.

A hash result of 000001f means that f is the second character in the password. Use only the first result for each position, and ignore invalid positions.

For example, if the Door ID is abc:

    The first interesting hash is from abc3231929, which produces 0000015...; so, 5 goes in position 1: _5______.
    In the previous method, 5017308 produced an interesting hash; however, it is ignored, because it specifies an invalid position (8).
    The second interesting hash is at index 5357525, which produces 000004e...; so, e goes in position 4: _5__e___.

You almost choke on your popcorn as the final character falls into place, producing the password 05ace8e3.

Given the actual Door ID and this new method, what is the password? Be extra proud of your solution if it uses a cinematic "decrypting" animation. */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(String);

impl Solution {
    const ZERO_MASK: u32 = 0xFFFFF000_u32;
    const NIBBLE_MASK: u32 = 0x000000F0_u32;
    const PASSWORD_LEN: usize = 8_usize;

    fn byte_to_hexadecimal_char(byte: u8) -> char {
        (if byte < 10_u8 {
            byte + b'0'
        } else {
            byte + 87u8 // b'a' - 10_u8
        }) as char
    }

    fn get_initial_md5_word(string: &mut String, index: u32) -> u32 {
        write!(string, "{index}").ok();

        let mut bytes: [u8; size_of::<u32>()] = Default::default();

        let src: &[u8] = &compute(string.as_bytes()).0[..size_of::<u32>()];

        bytes.copy_from_slice(src);

        u32::from_be_bytes(bytes)
    }

    fn try_get_char(string: &mut String, index: u32) -> Option<char> {
        let initial_md5_word: u32 = Self::get_initial_md5_word(string, index);

        (initial_md5_word & Self::ZERO_MASK == 0_u32)
            .then(|| Self::byte_to_hexadecimal_char((initial_md5_word >> 8_u32) as u8))
    }

    fn try_get_positional_char(string: &mut String, index: u32) -> Option<(u8, char)> {
        let initial_md5_word: u32 = Self::get_initial_md5_word(string, index);

        (initial_md5_word & Self::ZERO_MASK == 0_u32)
            .then(|| {
                let pos_byte: u8 = (initial_md5_word >> 8_u32) as u8;

                (pos_byte < Self::PASSWORD_LEN as u8).then(|| {
                    (
                        pos_byte,
                        Self::byte_to_hexadecimal_char(
                            ((initial_md5_word & Self::NIBBLE_MASK) >> 4_u32) as u8,
                        ),
                    )
                })
            })
            .flatten()
    }

    fn iter_chars<'f, T: Debug, F: Fn(&mut String, u32) -> Option<T> + 'f>(
        &'f self,
        f: F,
        verbose: bool,
    ) -> impl Iterator<Item = T> + 'f {
        let mut string: String = self.0.clone();
        let mut index: u32 = 0_u32;

        repeat(()).filter_map(move |_| {
            let c: Option<T> = f(&mut string, index);

            string.truncate(self.0.len());
            index += 1_u32;

            if verbose {
                if let Some(c) = c.as_ref() {
                    dbg!(index, c);
                }
            }

            c
        })
    }

    fn get_password(&self, verbose: bool) -> String {
        self.iter_chars(Self::try_get_char, verbose)
            .take(Self::PASSWORD_LEN)
            .collect()
    }

    fn get_positional_password(&self, verbose: bool) -> String {
        let mut password: [char; Self::PASSWORD_LEN] = ['_'; Self::PASSWORD_LEN];
        let mut computed_characters: usize = 0_usize;

        for (pos_byte, c) in self.iter_chars(Self::try_get_positional_char, verbose) {
            let pos: usize = pos_byte as usize;

            if password[pos] == '_' {
                password[pos] = c;

                computed_characters += 1_usize;

                if computed_characters == Self::PASSWORD_LEN {
                    break;
                }
            }

            if verbose {
                let password: String = password.into_iter().collect();

                dbg!(password);
            }
        }

        password.into_iter().collect()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map(not_line_ending, String::from), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.get_password(args.verbose));
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.get_positional_password(args.verbose));
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

    const SOLUTION_STR: &'static str = "abc";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution("abc".into()))
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_get_char() {
        let mut string: String = solution().0.clone();

        assert_eq!(Solution::try_get_char(&mut string, 3231929_u32), Some('1'));
    }

    #[test]
    fn test_get_password() {
        assert_eq!(solution().get_password(true), "18f47a30".to_owned());
    }

    #[test]
    fn test_get_positional_password() {
        assert_eq!(
            solution().get_positional_password(true),
            "05ace8e3".to_owned()
        );
    }
}
