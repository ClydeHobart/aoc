use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        character::complete::one_of,
        combinator::{iterator, map},
        error::Error,
        Err, IResult,
    },
    num::Integer,
    std::mem::swap,
};

/* --- Day 16: Dragon Checksum ---

You're done scanning this part of the network, but you've left traces of your presence. You need to overwrite some disks with random-looking data to cover your tracks and update the local security system with a new checksum for those disks.

For the data to not be suspicious, it needs to have certain properties; purely random data will be detected as tampering. To generate appropriate random data, you'll need to use a modified dragon curve.

Start with an appropriate initial state (your puzzle input). Then, so long as you don't have enough data yet to fill the disk, repeat the following steps:

    Call the data you have at this point "a".
    Make a copy of "a"; call this copy "b".
    Reverse the order of the characters in "b".
    In "b", replace all instances of 0 with 1 and all 1s with 0.
    The resulting data is "a", then a single 0, then "b".

For example, after a single step of this process,

    1 becomes 100.
    0 becomes 001.
    11111 becomes 11111000000.
    111100001010 becomes 1111000010100101011110000.

Repeat these steps until you have enough data to fill the desired disk.

Once the data has been generated, you also need to create a checksum of that data. Calculate the checksum only for the data that fits on the disk, even if you generated more data than that in the previous step.

The checksum for some given data is created by considering each non-overlapping pair of characters in the input data. If the two characters match (00 or 11), the next checksum character is a 1. If the characters do not match (01 or 10), the next checksum character is a 0. This should produce a new string which is exactly half as long as the original. If the length of the checksum is even, repeat the process until you end up with a checksum with an odd length.

For example, suppose we want to fill a disk of length 12, and when we finally generate a string of at least length 12, the first 12 characters are 110010110100. To generate its checksum:

    Consider each pair: 11, 00, 10, 11, 01, 00.
    These are same, same, different, same, different, same, producing 110101.
    The resulting string has length 6, which is even, so we repeat the process.
    The pairs are 11 (same), 01 (different), 01 (different).
    This produces the checksum 100, which has an odd length, so we stop.

Therefore, the checksum for 110010110100 is 100.

Combining all of these steps together, suppose you want to fill a disk of length 20 using an initial state of 10000:

    Because 10000 is too short, we first use the modified dragon curve to make it longer.
    After one round, it becomes 10000011110 (11 characters), still too short.
    After two rounds, it becomes 10000011110010000111110 (23 characters), which is enough.
    Since we only need 20, but we have 23, we get rid of all but the first 20 characters: 10000011110010000111.
    Next, we start calculating the checksum; after one round, we have 0111110101, which 10 characters long (even), so we continue.
    After two rounds, we have 01100, which is 5 characters long (odd), so we are done.

In this example, the correct checksum would therefore be 01100.

The first disk you have to fill has length 272. Using the initial state in your puzzle input, what is the correct checksum?

--- Part Two ---

The second disk you have to fill has length 35651584. Again using the initial state in your puzzle input, what is the correct checksum for this disk? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(BitVec);

impl Solution {
    const FIRST_DISC_LEN: usize = 272_usize;
    const SECOND_DISC_LEN: usize = 35651584_usize;

    fn bitslice_as_string(bitslice: &BitSlice) -> String {
        bitslice
            .iter()
            .by_vals()
            .map(|b| (b as u8 + b'0') as char)
            .collect()
    }

    /// Computes the checksum of a given disc. If the given disc doesn't have a positive, even
    /// length, this will panic.
    fn checksum_for_disc(disc: &BitSlice) -> BitVec {
        assert!(!disc.is_empty() && disc.len().is_even());

        let [mut curr_checksum, mut next_checksum]: [BitVec; 2_usize] =
            [disc.into(), BitVec::with_capacity(disc.len() / 2_usize)];

        while curr_checksum.len().is_even() {
            next_checksum.clear();
            next_checksum.extend(
                curr_checksum
                    .chunks_exact(2_usize)
                    .map(|pair| pair[0_usize] == pair[1_usize]),
            );
            swap(&mut curr_checksum, &mut next_checksum);
        }

        curr_checksum
    }

    /// Fills a disc of a specified length. If the solution's bit vector is empty, this will panic.
    fn fill_disc(&self, disc_len: usize) -> BitVec {
        assert!(!self.0.is_empty());

        let [mut curr_disc, mut next_disc]: [BitVec; 2_usize] =
            [self.0.clone(), BitVec::with_capacity(disc_len)];

        if let Some(additional) = disc_len.checked_sub(curr_disc.len()) {
            curr_disc.reserve(additional);
        }

        while curr_disc.len() < disc_len {
            next_disc.clear();
            next_disc.extend_from_bitslice(&curr_disc);
            next_disc.push(false);
            next_disc.extend(curr_disc.iter().by_vals().rev().map(|b| !b));
            swap(&mut curr_disc, &mut next_disc);
        }

        curr_disc
    }

    fn checksum_for_disc_len(&self, disc_len: usize) -> String {
        let disc: BitVec = self.fill_disc(disc_len);

        let checksum: BitVec = Self::checksum_for_disc(&disc[..disc_len]);

        Self::bitslice_as_string(&checksum)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut bool_iter = iterator(input, map(one_of("01"), |c: char| c == '1'));

        let solution: Self = Self(BitVec::from_iter((&mut bool_iter).into_iter()));

        let input: &str = bool_iter.finish()?.0;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let disc: BitVec = self.fill_disc(Self::FIRST_DISC_LEN);
            let checksum: BitVec = Self::checksum_for_disc(&disc[..Self::FIRST_DISC_LEN]);

            println!(
                "disc: \"{}\"\nchecksum: \"{}\"",
                Self::bitslice_as_string(&disc),
                Self::bitslice_as_string(&checksum)
            );
        } else {
            dbg!(self.checksum_for_disc_len(Self::FIRST_DISC_LEN));
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let disc: BitVec = self.fill_disc(Self::SECOND_DISC_LEN);
            let checksum: BitVec = Self::checksum_for_disc(&disc[..Self::SECOND_DISC_LEN]);

            println!(
                "disc: \"{}\"\nchecksum: \"{}\"",
                Self::bitslice_as_string(&disc),
                Self::bitslice_as_string(&checksum)
            );
        } else {
            dbg!(self.checksum_for_disc_len(Self::SECOND_DISC_LEN));
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

    const SOLUTION_STR: &'static str = "1111000010100101011110000";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(BitVec::from_iter(
                [
                    true, true, true, true, false, false, false, false, true, false, true, false,
                    false, true, false, true, false, true, true, true, true, false, false, false,
                    false,
                ]
                .into_iter(),
            ))
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_bitslice_as_string() {
        for string in [
            "0",
            "1",
            "00",
            "01",
            "10",
            "11",
            "1010",
            "0101",
            "0000",
            "1111",
            "01000110111010010100001111",
        ] {
            assert_eq!(
                Solution::bitslice_as_string(&Solution::try_from(string).unwrap().0),
                string
            );
        }
    }

    #[test]
    fn test_fill_disc() {
        for (start, end, len) in [
            ("1", "100", None),
            ("0", "001", None),
            ("11111", "11111000000", None),
            ("111100001010", "1111000010100101011110000", None),
            ("10000", "10000011110010000111110", Some(20_usize)),
        ] {
            let solution: Solution = Solution::try_from(start).unwrap();
            let disc: BitVec = solution.fill_disc(len.unwrap_or(end.len()));

            assert_eq!(Solution::bitslice_as_string(&disc), end);
        }
    }

    #[test]
    fn test_checksum_for_disc() {
        for (disc, checksum) in [("110010110100", "100"), ("10000011110010000111", "01100")] {
            let disc: BitVec = Solution::try_from(disc).unwrap().0;

            assert_eq!(
                Solution::bitslice_as_string(&Solution::checksum_for_disc(&disc)),
                checksum
            );
        }
    }
}
