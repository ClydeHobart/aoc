use {
    crate::*,
    nom::{
        bytes::complete::take_while1,
        character::complete::{line_ending, not_line_ending, space0},
        combinator::{all_consuming, map, map_res, opt, verify},
        error::Error,
        multi::many0_count,
        sequence::terminated,
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 4: High-Entropy Passphrases ---

A new system policy has been put in place that requires all accounts to use a passphrase instead of simply a password. A passphrase consists of a series of words (lowercase letters) separated by spaces.

To ensure security, a valid passphrase must contain no duplicate words.

For example:

    aa bb cc dd ee is valid.
    aa bb cc dd aa is not valid - the word aa appears more than once.
    aa bb cc dd aaa is valid - aa and aaa count as different words.

The system's full passphrase list is available as your puzzle input. How many passphrases are valid?

--- Part Two ---

For added security, yet another system policy has been put in place. Now, a valid passphrase must contain no two words that are anagrams of each other - that is, a passphrase is invalid if any word's letters can be rearranged to form any other word in the passphrase.

For example:

    abcde fghij is a valid passphrase.
    abcde xyz ecdab is not valid - the letters from the third word can be rearranged to form the first word.
    a ab abc abd abf abj is a valid passphrase, because all letters need to be used when forming another word.
    iiii oiii ooii oooi oooo is valid.
    oiii ioii iioi iiio is not valid - any of these words can be rearranged to form any other word.

Under this new system policy, how many passphrases are valid? */

const MAX_WORD_LEN: usize = 7_usize;

type WordString = StaticString<MAX_WORD_LEN>;

#[cfg_attr(test, derive(Debug))]
#[derive(PartialEq)]
struct Word {
    string: WordString,
    letter_counts_hash: u64,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    words: Vec<Word>,
    passphrases: Vec<Range<u32>>,
}

impl Solution {
    fn iter_passphrases(&self) -> impl Iterator<Item = &[Word]> {
        self.passphrases
            .iter()
            .map(|passphrase_range| &self.words[passphrase_range.as_range_usize()])
    }

    fn is_passphrase_valid(passphrase: &[Word]) -> bool {
        passphrase
            .iter()
            .enumerate()
            .all(|(index, word)| !passphrase[index + 1_usize..].contains(word))
    }

    fn iter_letter_counts_hashes_in_passphrase(
        passphrase: &[Word],
    ) -> impl Iterator<Item = u64> + '_ {
        passphrase.iter().map(|word| word.letter_counts_hash)
    }

    fn is_passphrase_doubly_valid(passphrase: &[Word]) -> bool {
        Self::iter_letter_counts_hashes_in_passphrase(passphrase)
            .enumerate()
            .all(|(index, letter_counts_hash_a)| {
                Self::iter_letter_counts_hashes_in_passphrase(&passphrase[index + 1_usize..])
                    .find(|letter_counts_hash_b| *letter_counts_hash_b == letter_counts_hash_a)
                    .is_none()
            })
    }

    fn valid_passphrase_count(&self) -> usize {
        self.iter_passphrases()
            .filter(|passphrase| Self::is_passphrase_valid(passphrase))
            .count()
    }

    fn doubly_valid_passphrase_count(&self) -> usize {
        self.iter_passphrases()
            .filter(|passphrase| Self::is_passphrase_doubly_valid(passphrase))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut words: Vec<Word> = Vec::new();
        let mut passphrases: Vec<Range<u32>> = Vec::new();

        let input: &str = many0_count(map_res(
            terminated(
                verify(not_line_ending, |input: &&str| !input.is_empty()),
                opt(line_ending),
            ),
            |input: &str| {
                let start: u32 = words.len() as u32;

                all_consuming(many0_count(map(
                    terminated(
                        map_res(
                            take_while1(|c: char| c.is_ascii_lowercase()),
                            WordString::try_from,
                        ),
                        space0::<&str, NomError<&str>>,
                    ),
                    |string| {
                        let letter_counts_hash: u64 =
                            LetterCounts::from_str(string.as_str()).compute_hash();

                        words.push(Word {
                            string,
                            letter_counts_hash,
                        });
                    },
                )))(input)
                .ok()
                .ok_or(())?;

                let end: u32 = words.len() as u32;

                passphrases.push(start..end);

                Result::<(), ()>::Ok(())
            },
        ))(input)?
        .0;

        Ok((input, Self { words, passphrases }))
    }
}

impl RunQuestions for Solution {
    /// Not too bad. Now I have a utility type for small strings.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.valid_passphrase_count());
    }

    /// Already having `LetterCounts` in `util` helps here.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.doubly_valid_passphrase_count());
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
        "aa bb cc dd ee\n\
        aa bb cc dd aa\n\
        aa bb cc dd aaa\n",
        "abcde fghij\n\
        abcde xyz ecdab\n\
        a ab abc abd abf abj\n\
        iiii oiii ooii oooi oooo\n\
        oiii ioii iioi iiio\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! words {
            [ $( $word:literal, )* ] => { vec![ $( {
                let string: WordString = WordString::try_from($word).unwrap();
                let letter_counts_hash: u64 = LetterCounts::from_str(string.as_str()).compute_hash();

                Word {
                    string,
                    letter_counts_hash,
                }
            }, )* ] }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    words: words![
                        "aa", "bb", "cc", "dd", "ee", "aa", "bb", "cc", "dd", "aa", "aa", "bb",
                        "cc", "dd", "aaa",
                    ],
                    passphrases: vec![0_u32..5_u32, 5_u32..10_u32, 10_u32..15_u32],
                },
                Solution {
                    words: words![
                        "abcde", "fghij", "abcde", "xyz", "ecdab", "a", "ab", "abc", "abd", "abf",
                        "abj", "iiii", "oiii", "ooii", "oooi", "oooo", "oiii", "ioii", "iioi",
                        "iiio",
                    ],
                    passphrases: vec![
                        0_u32..2_u32,
                        2_u32..5_u32,
                        5_u32..11_u32,
                        11_u32..16_u32,
                        16_u32..20_u32,
                    ],
                },
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
    fn test_is_passphrase_valid() {
        for (index, is_passphrase_valid_vec) in
            [vec![true, false, true], vec![true, true, true, true, true]]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_passphrases()
                    .map(Solution::is_passphrase_valid)
                    .collect::<Vec<bool>>(),
                is_passphrase_valid_vec
            );
        }
    }

    #[test]
    fn test_valid_passphrase_count() {
        for (index, valid_passphrase_count) in [2_usize, 5_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).valid_passphrase_count(),
                valid_passphrase_count
            );
        }
    }

    #[test]
    fn test_is_passphrase_doubly_valid() {
        for (index, is_passphrase_doubly_valid_vec) in [
            vec![true, false, true],
            vec![true, false, true, true, false],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_passphrases()
                    .map(Solution::is_passphrase_doubly_valid)
                    .collect::<Vec<bool>>(),
                is_passphrase_doubly_valid_vec
            );
        }
    }

    #[test]
    fn test_doubly_valid_passphrase_count() {
        for (index, doubly_valid_passphrase_count) in [2_usize, 3_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).doubly_valid_passphrase_count(),
                doubly_valid_passphrase_count
            );
        }
    }
}
