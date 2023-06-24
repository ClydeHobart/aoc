use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_res, opt},
        error::{Error, ErrorKind},
        multi::{fold_many_m_n, many0},
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::{array::IntoIter, cmp::Ordering, mem::transmute, ops::Range},
};

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct SignalPattern(u8);

impl SignalPattern {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self, Error<&'i str>> {
        terminated(
            map_res(
                take_while_m_n(
                    Solution::MIN_SEGMENT_COUNT,
                    Solution::MAX_SEGMENT_COUNT,
                    Solution::is_valid_char,
                ),
                Self::try_from,
            ),
            opt(tag(" ")),
        )(input)
    }

    fn compare_one_counts(a: &Self, b: &Self) -> Ordering {
        a.0.count_ones().cmp(&b.0.count_ones())
    }

    fn is_segment_count_unique(&self) -> bool {
        (Solution::IS_SEGMENT_COUNT_UNIQUE & 1_u8 << self.0.count_ones()) != 0_u8
    }

    fn decipher(self, cipher: Cipher) -> Option<u8> {
        let segments: u8 = self
            .0
            .view_bits::<Lsb0>()
            .iter_ones()
            .fold(0_u8, |segments, index| segments | cipher.0[index]);

        Solution::DIGITS_SORTED_BY_SEGMENTS
            .binary_search_by(|digit| Solution::SEGMENTS[*digit as usize].cmp(&segments))
            .ok()
            .map(|index| Solution::DIGITS_SORTED_BY_SEGMENTS[index])
    }
}

impl TryFrom<&str> for SignalPattern {
    type Error = ();

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        if (Solution::MIN_SEGMENT_COUNT..=Solution::MAX_SEGMENT_COUNT).contains(&input.len())
            && input.chars().all(Solution::is_valid_char)
        {
            let mut signal_pattern: Self = Self::default();

            for byte in input.bytes() {
                signal_pattern.0 |= 1_u8 << (byte - b'a');
            }

            Ok(signal_pattern)
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct UniqueSignalPatterns([SignalPattern; Solution::DIGITS]);

impl UniqueSignalPatterns {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self, Error<&'i str>> {
        let mut unique_signal_patterns: Self = Self([SignalPattern::default(); Solution::DIGITS]);
        let mut segment_count_counts: [u8; 8_usize] = [0_u8; 8_usize];
        let (input, _) = fold_many_m_n(
            Solution::DIGITS,
            Solution::DIGITS,
            SignalPattern::parse,
            || 0_usize,
            |index, signal_pattern| {
                unique_signal_patterns.0[index] = signal_pattern;
                segment_count_counts[signal_pattern.0.count_ones() as usize] += 1_u8;

                index + 1_usize
            },
        )(input)?;

        if segment_count_counts == Solution::SEGMENT_COUNT_COUNTS {
            unique_signal_patterns
                .0
                .sort_by(SignalPattern::compare_one_counts);

            Ok((input, unique_signal_patterns))
        } else {
            Err(Err::Failure(Error::new(input, ErrorKind::Fail)))
        }
    }

    fn cipher(&self) -> Cipher {
        const INDEX_1: usize = Solution::start_range_with_segment_count(2_usize);
        const INDEX_7: usize = Solution::start_range_with_segment_count(3_usize);
        const INDEX_4: usize = Solution::start_range_with_segment_count(4_usize);
        const RANGE_5_SEGMENTS: Range<usize> = Solution::range_with_segment_count(5_usize);
        const RANGE_6_SEGMENTS: Range<usize> = Solution::range_with_segment_count(6_usize);

        let and_product_of_range = |range: Range<usize>, mask: u8| -> u8 {
            self.0[range]
                .iter()
                .copied()
                .fold(mask, |and_product, sp| and_product & sp.0)
        };
        let sp_1: u8 = self.0[INDEX_1].0;
        let sp_7: u8 = self.0[INDEX_7].0;
        let sp_4: u8 = self.0[INDEX_4].0;
        let cf: u8 = sp_1 & sp_7;
        let bd: u8 = sp_4 & !sp_7;
        let eg: u8 = Cipher::MASK & !(sp_4 | sp_7);

        #[derive(Default)]
        struct Wires {
            a: u8,
            b: u8,
            c: u8,
            d: u8,
            e: u8,
            f: u8,
            g: u8,
        }

        impl IntoIterator for Wires {
            type IntoIter = IntoIter<u8, { Solution::MAX_SEGMENT_COUNT }>;
            type Item = u8;

            fn into_iter(self) -> Self::IntoIter {
                // SAFETY: Wires and [u8; Solution::MAX_SEGMENT_COUNT] have the same size and
                // alignment, and it's safe to reinterpret struct `u8`s as an array of `u8`s
                unsafe { transmute::<Self, [u8; Solution::MAX_SEGMENT_COUNT]>(self) }.into_iter()
            }
        }

        let mut wires: Wires = Wires::default();

        wires.a = sp_1 ^ sp_7;
        wires.f = and_product_of_range(RANGE_6_SEGMENTS, cf);
        wires.c = cf ^ wires.f;
        wires.d = and_product_of_range(RANGE_5_SEGMENTS, bd);
        wires.b = bd ^ wires.d;
        wires.g = and_product_of_range(RANGE_6_SEGMENTS, eg);
        wires.e = eg ^ wires.g;

        let mut cipher: Cipher = Cipher::default();

        for (index, wire) in wires.into_iter().enumerate() {
            cipher.0[wire.trailing_zeros() as usize] = 1_u8 << index;
        }

        cipher
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Cipher([u8; Solution::MAX_SEGMENT_COUNT]);

impl Cipher {
    const MASK: u8 = (1_u8 << Solution::MAX_SEGMENT_COUNT) - 1_u8;
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct FourDigitOutputValue([SignalPattern; FourDigitOutputValue::DIGITS]);

impl FourDigitOutputValue {
    const DIGITS: usize = 4_usize;

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self, Error<&'i str>> {
        let mut four_digit_output_value: Self = Self([SignalPattern::default(); Self::DIGITS]);

        let (input, _) = fold_many_m_n(
            Self::DIGITS,
            Self::DIGITS,
            SignalPattern::parse,
            || 0_usize,
            |index, signal_pattern| {
                four_digit_output_value.0[index] = signal_pattern;

                index + 1_usize
            },
        )(input)?;

        Ok((input, four_digit_output_value))
    }

    fn decipher(self, cipher: Cipher) -> Option<u16> {
        self.0.into_iter().try_fold(0_u16, |value, sp| {
            Some(value * 10_u16 + sp.decipher(cipher)? as u16)
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Entry {
    unique_signal_patterns: UniqueSignalPatterns,
    four_digit_output_value: FourDigitOutputValue,
}

impl Entry {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self, Error<&'i str>> {
        terminated(
            map(
                separated_pair(
                    UniqueSignalPatterns::parse,
                    tag("| "),
                    FourDigitOutputValue::parse,
                ),
                Self::from,
            ),
            opt(line_ending),
        )(input)
    }

    fn decipher(&self) -> Option<u16> {
        self.four_digit_output_value
            .decipher(self.unique_signal_patterns.cipher())
    }
}

impl From<(UniqueSignalPatterns, FourDigitOutputValue)> for Entry {
    fn from(
        (unique_signal_patterns, four_digit_output_value): (
            UniqueSignalPatterns,
            FourDigitOutputValue,
        ),
    ) -> Self {
        Self {
            unique_signal_patterns,
            four_digit_output_value,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Entry>);

impl Solution {
    const DIGITS: usize = 10_usize;
    const SEGMENTS: [u8; Solution::DIGITS] = [
        0b1110111_u8,
        0b0100100_u8,
        0b1011101_u8,
        0b1101101_u8,
        0b0101110_u8,
        0b1101011_u8,
        0b1111011_u8,
        0b0100101_u8,
        0b1111111_u8,
        0b1101111_u8,
    ];
    const DIGITS_SORTED_BY_SEGMENTS: [u8; Solution::DIGITS] = Solution::digits_sorted_by_segments();
    const SEGMENT_COUNT_COUNTS: [u8; 8_usize] = Solution::segment_count_counts();
    const IS_SEGMENT_COUNT_UNIQUE: u8 = Solution::is_segment_count_unique();
    const MIN_SEGMENT_COUNT: usize = Solution::min_segment_count();
    const MAX_SEGMENT_COUNT: usize = Solution::max_segment_count();

    const fn segment_count_counts() -> [u8; 8_usize] {
        let mut segment_count_counts: [u8; 8_usize] = [0_u8; 8_usize];
        let mut i: usize = 0_usize;

        while i < Solution::SEGMENTS.len() {
            segment_count_counts[Solution::SEGMENTS[i].count_ones() as usize] += 1_u8;
            i += 1_usize;
        }

        segment_count_counts
    }

    const fn is_segment_count_unique() -> u8 {
        let mut is_segment_count_unique: u8 = 0_u8;
        let mut segment_count: usize = 0_usize;

        while segment_count < Solution::SEGMENT_COUNT_COUNTS.len() {
            if Solution::SEGMENT_COUNT_COUNTS[segment_count] == 1_u8 {
                is_segment_count_unique |= 1_u8 << segment_count;
            }

            segment_count += 1_usize;
        }

        is_segment_count_unique
    }

    const fn min_segment_count() -> usize {
        let mut index: usize = 0_usize;

        while index < Solution::SEGMENT_COUNT_COUNTS.len() {
            if Solution::SEGMENT_COUNT_COUNTS[index] != 0_u8 {
                return index;
            }

            index += 1_usize
        }

        0_usize
    }

    const fn max_segment_count() -> usize {
        let mut index: usize = Solution::SEGMENT_COUNT_COUNTS.len();

        while index > 0_usize {
            index -= 1_usize;

            if Solution::SEGMENT_COUNT_COUNTS[index] != 0_u8 {
                return index;
            }
        }

        0_usize
    }

    const fn digits_sorted_by_segments() -> [u8; Solution::DIGITS] {
        let mut digits: [u8; Solution::DIGITS] = [0_u8; Solution::DIGITS];

        {
            let mut index: usize = 0_usize;

            while index < digits.len() {
                digits[index] = index as u8;
                index += 1_usize;
            }
        }

        {
            let mut index_1: usize = 0_usize;

            // Bubble sort, since this is small and operations in a const context are limited
            while index_1 < digits.len() - 1_usize {
                let mut min_index: usize = index_1;
                let mut index_2: usize = index_1 + 1_usize;

                while index_2 < digits.len() {
                    if Solution::SEGMENTS[digits[index_2] as usize]
                        < Solution::SEGMENTS[digits[min_index] as usize]
                    {
                        min_index = index_2;
                    }

                    index_2 += 1_usize;
                }

                if min_index != index_1 {
                    let digit: u8 = digits[index_1];

                    digits[index_1] = digits[min_index];
                    digits[min_index] = digit;
                }

                index_1 += 1_usize;
            }
        }

        digits
    }

    const fn start_range_with_segment_count(segment_count: usize) -> usize {
        let mut index: usize = 0_usize;
        let mut sum: u8 = 0_u8;

        while index < segment_count {
            sum += Solution::SEGMENT_COUNT_COUNTS[index];
            index += 1_usize;
        }

        sum as usize
    }

    const fn end_range_with_segment_count(segment_count: usize) -> usize {
        let mut index: usize = Solution::SEGMENT_COUNT_COUNTS.len();
        let mut sum: u8 = 0_u8;

        while index > segment_count + 1_usize {
            index -= 1_usize;
            sum += Solution::SEGMENT_COUNT_COUNTS[index];
        }

        Solution::DIGITS - sum as usize
    }

    const fn range_with_segment_count(segment_count: usize) -> Range<usize> {
        Self::start_range_with_segment_count(segment_count)
            ..Self::end_range_with_segment_count(segment_count)
    }

    fn is_valid_char(c: char) -> bool {
        ('a'..='g').contains(&c)
    }

    fn count_unique_segment_counts(&self) -> usize {
        self.0
            .iter()
            .flat_map(|entry| entry.four_digit_output_value.0.iter().copied())
            .filter(SignalPattern::is_segment_count_unique)
            .count()
    }

    fn sum_values(&self) -> Option<u32> {
        self.0
            .iter()
            .try_fold(0_u32, |sum, entry| Some(sum + entry.decipher()? as u32))
    }
}

impl From<Vec<Entry>> for Solution {
    fn from(entries: Vec<Entry>) -> Self {
        Self(entries)
    }
}
impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_unique_segment_counts());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_values());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(map(many0(Entry::parse), Self::from)(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const ENTRY_STR: &str =
        "acedgfb cdfbe gcdfa fbcad dab cefabd cdfgeb eafb cagedb ab | cdfeb fcadb cdfeb cdbaf\n";
    const ENTRIES_STR: &str = concat!(
        "be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe\n",
        "edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc\n",
        "fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg\n",
        "fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb\n",
        "aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea\n",
        "fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb\n",
        "dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe\n",
        "bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef\n",
        "egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb\n",
        "gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce\n",
    );

    fn entry() -> &'static Entry {
        static ONCE_LOCK: OnceLock<Entry> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            macro_rules! signal_patterns {
            [ $( $sp:expr ),* $(,)? ] => {
                [ $( SignalPattern($sp) ),* ]
            };
        }

            let sps = signal_patterns![
                0x03, 0x0B, 0x33, 0x3E, 0x6D, 0x2F, 0x3F, 0x7E, 0x5F, 0x7F, 0x3E, 0x2F, 0x3E, 0x2F
            ];

            let mut entry: Entry = Entry {
                unique_signal_patterns: UniqueSignalPatterns(Default::default()),
                four_digit_output_value: FourDigitOutputValue(Default::default()),
            };

            for index in 0_usize..Solution::DIGITS {
                entry.unique_signal_patterns.0[index] = sps[index];
            }

            for index in 0_usize..FourDigitOutputValue::DIGITS {
                entry.four_digit_output_value.0[index] = sps[index + Solution::DIGITS];
            }

            entry
        })
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution::try_from(ENTRIES_STR).unwrap_or(Solution(Vec::new())))
    }

    #[test]
    fn test_entry_try_from_str() {
        assert_eq!(
            Entry::parse(ENTRY_STR).map(|(_, entry)| entry).as_ref(),
            Ok(entry())
        );
    }

    #[test]
    fn test_solution_try_from_str() {
        macro_rules! solution_ones {
            [ $( ( $( $usp:expr ),* ; $( $dov:expr ),* ), )* ] => {
                vec![ $( $( $usp, )* $( $dov, )* )* ]
            };
        }

        assert_eq!(
            solution()
                .0
                .iter()
                .flat_map(|entry| {
                    entry
                        .unique_signal_patterns
                        .0
                        .iter()
                        .chain(entry.four_digit_output_value.0.iter())
                })
                .map(|sp| sp.0.count_ones())
                .collect::<Vec<u32>>(),
            solution_ones![
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 7, 5, 6, 4),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 6, 3, 7, 2),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 2, 2, 6, 3),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 6, 5, 6, 2),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 4, 7, 3, 5),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 7, 4, 2, 7),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 4, 5, 4, 7),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 2, 6, 5, 5),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 7, 3, 2, 3),
                (2, 3, 4, 5, 5, 5, 6, 6, 6, 7; 4, 5, 2, 5),
            ]
        )
    }

    #[test]
    fn test_count_unique_segment_counts() {
        assert_eq!(solution().count_unique_segment_counts(), 26_usize);
    }

    #[test]
    fn test_unique_signal_patterns_cipher() {
        assert_eq!(
            entry().unique_signal_patterns.cipher(),
            Cipher([
                0b0000100_u8,
                0b0100000_u8,
                0b1000000_u8,
                0b0000001_u8,
                0b0000010_u8,
                0b0001000_u8,
                0b0010000_u8
            ])
        )
    }

    #[test]
    fn test_entry_decipher() {
        macro_rules! somes { [ $( $value:expr ),* ] => { vec![ $( Some($value), )* ] }; }

        assert_eq!(
            solution()
                .0
                .iter()
                .map(|entry| entry.decipher())
                .collect::<Vec<Option<u16>>>(),
            somes![8394, 9781, 1197, 9361, 4873, 8418, 4548, 1625, 8717, 4315]
        )
    }

    #[test]
    fn test_sum_values() {
        assert_eq!(solution().sum_values(), Some(61229));
    }
}
