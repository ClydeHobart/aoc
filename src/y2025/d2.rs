use {
    crate::*,
    arrayvec::{ArrayString, CapacityError},
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, map_res, opt},
        error::Error,
        multi::separated_list0,
        sequence::{separated_pair, tuple},
        Err, IResult,
    },
    num::Integer,
    static_assertions::const_assert,
    std::{fmt::Write, iter::from_fn, ops::RangeInclusive},
};

/* --- Day 2: Gift Shop ---

You get inside and take the elevator to its only other stop: the gift shop. "Thank you for visiting the North Pole!" gleefully exclaims a nearby sign. You aren't sure who is even allowed to visit the North Pole, but you know you can access the lobby through here, and from there you can access the rest of the North Pole base.

As you make your way through the surprisingly extensive selection, one of the clerks recognizes you and asks for your help.

As it turns out, one of the younger Elves was playing on a gift shop computer and managed to add a whole bunch of invalid product IDs to their gift shop database! Surely, it would be no trouble for you to identify the invalid product IDs for them, right?

They've even checked most of the product ID ranges already; they only have a few product ID ranges (your puzzle input) that you'll need to check. For example:

11-22,95-115,998-1012,1188511880-1188511890,222220-222224,
1698522-1698528,446443-446449,38593856-38593862,565653-565659,
824824821-824824827,2121212118-2121212124

(The ID ranges are wrapped here for legibility; in your input, they appear on a single long line.)

The ranges are separated by commas (,); each range gives its first ID and last ID separated by a dash (-).

Since the young Elf was just doing silly patterns, you can find the invalid IDs by looking for any ID which is made only of some sequence of digits repeated twice. So, 55 (5 twice), 6464 (64 twice), and 123123 (123 twice) would all be invalid IDs.

None of the numbers have leading zeroes; 0101 isn't an ID at all. (101 is a valid ID that you would ignore.)

Your job is to find all of the invalid IDs that appear in the given ranges. In the above example:

    11-22 has two invalid IDs, 11 and 22.
    95-115 has one invalid ID, 99.
    998-1012 has one invalid ID, 1010.
    1188511880-1188511890 has one invalid ID, 1188511885.
    222220-222224 has one invalid ID, 222222.
    1698522-1698528 contains no invalid IDs.
    446443-446449 has one invalid ID, 446446.
    38593856-38593862 has one invalid ID, 38593859.
    The rest of the ranges contain no invalid IDs.

Adding up all the invalid IDs in this example produces 1227775554.

What do you get if you add up all of the invalid IDs?

--- Part Two ---

The clerk quickly discovers that there are still invalid IDs in the ranges in your list. Maybe the young Elf was doing other silly patterns as well?

Now, an ID is invalid if it is made only of some sequence of digits repeated at least twice. So, 12341234 (1234 two times), 123123123 (123 three times), 1212121212 (12 five times), and 1111111 (1 seven times) are all invalid IDs.

From the same example as before:

    11-22 still has two invalid IDs, 11 and 22.
    95-115 now has two invalid IDs, 99 and 111.
    998-1012 now has two invalid IDs, 999 and 1010.
    1188511880-1188511890 still has one invalid ID, 1188511885.
    222220-222224 still has one invalid ID, 222222.
    1698522-1698528 still contains no invalid IDs.
    446443-446449 still has one invalid ID, 446446.
    38593856-38593862 still has one invalid ID, 38593859.
    565653-565659 now has one invalid ID, 565656.
    824824821-824824827 now has one invalid ID, 824824824.
    2121212118-2121212124 now has one invalid ID, 2121212121.

Adding up all the invalid IDs in this example produces 4174379265.

What do you get if you add up all of the invalid IDs using these new rules? */

const DECIMAL_RADIX: u32 = 10_u32;
const ID_STRING_LEN: usize = 10_usize;
const POSSIBLE_ID_STRING_LEN_COUNT: usize = ID_STRING_LEN + 1_usize;

const fn compute_repeating_sequence_length_factors_array() -> [u16; POSSIBLE_ID_STRING_LEN_COUNT] {
    let mut repeating_sequence_length_factors_array: [u16; POSSIBLE_ID_STRING_LEN_COUNT] =
        [0_u16; POSSIBLE_ID_STRING_LEN_COUNT];

    const_assert!(ID_STRING_LEN <= u16::BITS as usize);

    let mut id_string_len: usize = 2_usize;

    while id_string_len < repeating_sequence_length_factors_array.len() {
        let mut repeating_sequence_length_factors: u16 = 0_u16;
        let mut factor_candidate: usize = 1_usize;

        while factor_candidate <= id_string_len / 2_usize {
            if id_string_len % factor_candidate == 0_usize {
                repeating_sequence_length_factors |= 1_u16 << factor_candidate;
            }

            factor_candidate += 1_usize;
        }

        repeating_sequence_length_factors_array[id_string_len] = repeating_sequence_length_factors;
        id_string_len += 1_usize;
    }

    repeating_sequence_length_factors_array
}

const REPEATING_SEQUENCE_LENGTH_FACTORS_ARRAY: [u16; POSSIBLE_ID_STRING_LEN_COUNT] =
    compute_repeating_sequence_length_factors_array();

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
struct IdString(ArrayString<ID_STRING_LEN>);

impl IdString {
    fn id(&self) -> usize {
        usize::from_str_radix(&self.0, DECIMAL_RADIX).unwrap()
    }

    fn increment(&mut self) {
        if unsafe { self.0.as_bytes_mut() }
            .iter_mut()
            .rev()
            .try_fold((), |_, ascii_byte| {
                if *ascii_byte == b'9' {
                    *ascii_byte = b'0';

                    Some(())
                } else {
                    *ascii_byte += 1_u8;

                    None
                }
            })
            .is_some()
        {
            if self.0.is_full() {
                *self = 0_usize.try_into().unwrap();
            } else {
                *self = (DECIMAL_RADIX.pow(self.0.len() as u32) as usize)
                    .try_into()
                    .unwrap();
            }
        }
    }

    fn is_simply_valid(&self) -> bool {
        let len: usize = self.0.len();

        len.is_odd() || {
            let half_len: usize = len / 2_usize;
            let str_slice: &str = self.0.as_str();

            str_slice[..half_len] != str_slice[half_len..]
        }
    }

    fn is_complexly_valid(&self) -> bool {
        REPEATING_SEQUENCE_LENGTH_FACTORS_ARRAY[self.0.len()]
            .view_bits::<Lsb0>()
            .iter_ones()
            .all(|repeating_sequence_length_factor| {
                let byte_slice: &[u8] = self.0.as_bytes();
                let repeating_sequence: &[u8] = &byte_slice[..repeating_sequence_length_factor];

                byte_slice[repeating_sequence_length_factor..]
                    .chunks_exact(repeating_sequence_length_factor)
                    .any(|next_sequence| next_sequence != repeating_sequence)
            })
    }
}

impl TryFrom<usize> for IdString {
    type Error = CapacityError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if digits(value) > ID_STRING_LEN {
            Err(CapacityError::new(()))
        } else {
            let mut id_string: Self = Self(ArrayString::new());

            write!(&mut id_string.0, "{}", value).unwrap();

            Ok(id_string)
        }
    }
}

impl Parse for IdString {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_res(parse_integer::<usize>, Self::try_from)(input)
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct IdRange(RangeInclusive<IdString>);

impl IdRange {
    fn iter_id_strings(&self) -> impl Iterator<Item = IdString> + '_ {
        let mut id_string: IdString = self.0.start().clone();
        let mut is_exhausted: bool = false;

        from_fn(move || {
            (!is_exhausted).then(|| {
                let next: IdString = id_string.clone();

                is_exhausted = id_string == *self.0.end();
                id_string.increment();

                next
            })
        })
    }

    fn iter_simply_invalid_id_strings(&self) -> impl Iterator<Item = IdString> + '_ {
        self.iter_id_strings()
            .filter(|id_string| !id_string.is_simply_valid())
    }

    fn iter_complexly_invalid_id_strings(&self) -> impl Iterator<Item = IdString> + '_ {
        self.iter_id_strings()
            .filter(|id_string| !id_string.is_complexly_valid())
    }
}

impl Parse for IdRange {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            separated_pair(IdString::parse, tag("-"), IdString::parse),
            |(start, end)| (start.id() <= end.id()).then(|| Self(start..=end)),
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<IdRange>);

impl Solution {
    fn compute_invalid_id_sum<
        'r,
        's: 'r,
        I: Iterator<Item = IdString> + 'r,
        F: Fn(&'r IdRange) -> I,
    >(
        &'s self,
        iter_invalid_id_strings: F,
    ) -> usize {
        self.0
            .iter()
            .flat_map(|id_range| iter_invalid_id_strings(id_range))
            .map(|id_string| id_string.id())
            .sum()
    }

    fn compute_simply_invalid_id_sum(&self) -> usize {
        self.compute_invalid_id_sum(IdRange::iter_simply_invalid_id_strings)
    }

    fn compute_complexly_invalid_id_sum(&self) -> usize {
        self.compute_invalid_id_sum(IdRange::iter_complexly_invalid_id_strings)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_list0(tuple((tag(","), opt(line_ending))), IdRange::parse),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_simply_invalid_id_sum());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_complexly_invalid_id_sum());
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
    use {super::*, bitvec::view::BitView, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        11-22,95-115,998-1012,1188511880-1188511890,222220-222224,\n\
        1698522-1698528,446443-446449,38593856-38593862,565653-565659,\n\
        824824821-824824827,2121212118-2121212124"];

    macro_rules! id {
        ($id:expr) => {
            IdString::try_from($id).unwrap()
        };
    }

    macro_rules! ids { ( $($id:expr),* $(,)? ) => { vec![ $( id!($id), )* ] } }

    struct IdRangeData {
        start: IdString,
        simply_invalid_id_strings: Vec<IdString>,
        complexly_invalid_id_strings: Vec<IdString>,
        end: IdString,
    }

    struct SolutionData(Vec<IdRangeData>);

    fn solution_data(index: usize) -> &'static SolutionData {
        static ONCE_LOCK: OnceLock<Vec<SolutionData>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![SolutionData(vec![
                IdRangeData {
                    start: id!(11),
                    simply_invalid_id_strings: ids![11, 22],
                    complexly_invalid_id_strings: ids![11, 22],
                    end: id!(22),
                },
                IdRangeData {
                    start: id!(95),
                    simply_invalid_id_strings: ids![99],
                    complexly_invalid_id_strings: ids![99, 111],
                    end: id!(115),
                },
                IdRangeData {
                    start: id!(998),
                    simply_invalid_id_strings: ids![1010],
                    complexly_invalid_id_strings: ids![999, 1010],
                    end: id!(1012),
                },
                IdRangeData {
                    start: id!(1188511880),
                    simply_invalid_id_strings: ids![1188511885],
                    complexly_invalid_id_strings: ids![1188511885],
                    end: id!(1188511890),
                },
                IdRangeData {
                    start: id!(222220),
                    simply_invalid_id_strings: ids![222222],
                    complexly_invalid_id_strings: ids![222222],
                    end: id!(222224),
                },
                IdRangeData {
                    start: id!(1698522),
                    simply_invalid_id_strings: ids![],
                    complexly_invalid_id_strings: ids![],
                    end: id!(1698528),
                },
                IdRangeData {
                    start: id!(446443),
                    simply_invalid_id_strings: ids![446446],
                    complexly_invalid_id_strings: ids![446446],
                    end: id!(446449),
                },
                IdRangeData {
                    start: id!(38593856),
                    simply_invalid_id_strings: ids![38593859],
                    complexly_invalid_id_strings: ids![38593859],
                    end: id!(38593862),
                },
                IdRangeData {
                    start: id!(565653),
                    simply_invalid_id_strings: ids![],
                    complexly_invalid_id_strings: ids![565656],
                    end: id!(565659),
                },
                IdRangeData {
                    start: id!(824824821),
                    simply_invalid_id_strings: ids![],
                    complexly_invalid_id_strings: ids![824824824],
                    end: id!(824824827),
                },
                IdRangeData {
                    start: id!(2121212118),
                    simply_invalid_id_strings: ids![],
                    complexly_invalid_id_strings: ids![2121212121],
                    end: id!(2121212124),
                },
            ])]
        })[index]
    }

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                solution_data(0_usize)
                    .0
                    .iter()
                    .map(|id_range_data| {
                        IdRange(id_range_data.start.clone()..=id_range_data.end.clone())
                    })
                    .collect(),
            )]
        })[index]
    }

    #[test]
    fn test_repeating_sequence_length_factors_array() {
        assert_eq!(REPEATING_SEQUENCE_LENGTH_FACTORS_ARRAY[0_usize], 0_u16);

        for (id_string_len, mut repeating_sequence_length_factors) in
            REPEATING_SEQUENCE_LENGTH_FACTORS_ARRAY
                .into_iter()
                .enumerate()
                .skip(1_usize)
        {
            for factor in iter_factors(id_string_len) {
                assert_eq!(
                    factor != id_string_len,
                    repeating_sequence_length_factors.view_bits::<Lsb0>()[factor]
                );

                repeating_sequence_length_factors
                    .view_bits_mut::<Lsb0>()
                    .set(factor, false);
            }

            assert_eq!(repeating_sequence_length_factors, 0_u16);
        }
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

    fn test_is_valid<F: Fn(&IdString) -> bool, G: Fn(&IdRangeData) -> &[IdString]>(
        is_valid: F,
        get_invalid_id_strings: G,
    ) {
        for id_range_data in solution_data(0_usize).0.iter() {
            let invalid_id_strings: &[IdString] = get_invalid_id_strings(id_range_data);

            assert_eq!(
                is_valid(&id_range_data.start),
                !invalid_id_strings.contains(&id_range_data.start)
            );

            for invalid_id_string in invalid_id_strings {
                assert!(!is_valid(invalid_id_string));
            }

            assert_eq!(
                is_valid(&id_range_data.end),
                !invalid_id_strings.contains(&id_range_data.end)
            );
        }
    }

    #[test]
    fn test_is_simply_valid() {
        test_is_valid(IdString::is_simply_valid, |id_range_data| {
            &id_range_data.simply_invalid_id_strings
        });
    }

    #[test]
    fn test_is_complexly_valid() {
        test_is_valid(IdString::is_complexly_valid, |id_range_data| {
            &id_range_data.complexly_invalid_id_strings
        });
    }

    fn iter_id_range_with_data() -> impl Iterator<Item = (&'static IdRange, &'static IdRangeData)> {
        solution(0_usize)
            .0
            .iter()
            .zip(solution_data(0_usize).0.iter())
    }

    #[test]
    fn test_iter_id_strings() {
        for (id_range, id_range_data) in iter_id_range_with_data() {
            assert_eq!(
                id_range.iter_id_strings().collect::<Vec<IdString>>(),
                (id_range_data.start.id()..=id_range_data.end.id())
                    .map(|id| id!(id))
                    .collect::<Vec<IdString>>()
            );
        }
    }

    #[test]
    fn test_iter_simply_invalid_id_strings() {
        for (id_range, id_range_data) in iter_id_range_with_data() {
            assert_eq!(
                id_range
                    .iter_simply_invalid_id_strings()
                    .collect::<Vec<IdString>>(),
                id_range_data.simply_invalid_id_strings
            );
        }
    }

    #[test]
    fn test_iter_complexly_invalid_id_strings() {
        for (id_range, id_range_data) in iter_id_range_with_data() {
            assert_eq!(
                id_range
                    .iter_complexly_invalid_id_strings()
                    .collect::<Vec<IdString>>(),
                id_range_data.complexly_invalid_id_strings
            );
        }
    }

    #[test]
    fn test_compute_simply_invalid_id_sum() {
        assert_eq!(
            solution(0_usize).compute_simply_invalid_id_sum(),
            1227775554_usize
        );
    }

    #[test]
    fn test_compute_complexly_invalid_id_sum() {
        assert_eq!(
            solution(0_usize).compute_complexly_invalid_id_sum(),
            4174379265_usize
        );
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
