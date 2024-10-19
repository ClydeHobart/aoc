use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::{not_line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::{ops::Range, str::from_utf8_unchecked},
};

/* --- Day 10: Knot Hash ---

You come across some programs that are trying to implement a software emulation of a hash based on knot-tying. The hash these programs are implementing isn't very strong, but you decide to help them anyway. You make a mental note to remind the Elves later not to invent their own cryptographic functions.

This hash function simulates tying a knot in a circle of string with 256 marks on it. Based on the input to be hashed, the function repeatedly selects a span of string, brings the ends together, and gives the span a half-twist to reverse the order of the marks within it. After doing this many times, the order of the marks is used to build the resulting hash.

  4--5   pinch   4  5           4   1
 /    \  5,0,1  / \/ \  twist  / \ / \
3      0  -->  3      0  -->  3   X   0
 \    /         \ /\ /         \ / \ /
  2--1           2  1           2   5

To achieve this, begin with a list of numbers from 0 to 255, a current position which begins at 0 (the first element in the list), a skip size (which starts at 0), and a sequence of lengths (your puzzle input). Then, for each length:

    Reverse the order of that length of elements in the list, starting with the element at the current position.
    Move the current position forward by that length plus the skip size.
    Increase the skip size by one.

The list is circular; if the current position and the length try to reverse elements beyond the end of the list, the operation reverses using as many extra elements as it needs from the front of the list. If the current position moves past the end of the list, it wraps around to the front. Lengths larger than the size of the list are invalid.

Here's an example using a smaller list:

Suppose we instead only had a circular list containing five elements, 0, 1, 2, 3, 4, and were given input lengths of 3, 4, 1, 5.

    The list begins as [0] 1 2 3 4 (where square brackets indicate the current position).
    The first length, 3, selects ([0] 1 2) 3 4 (where parentheses indicate the sublist to be reversed).
    After reversing that section (0 1 2 into 2 1 0), we get ([2] 1 0) 3 4.
    Then, the current position moves forward by the length, 3, plus the skip size, 0: 2 1 0 [3] 4. Finally, the skip size increases to 1.

    The second length, 4, selects a section which wraps: 2 1) 0 ([3] 4.
    The sublist 3 4 2 1 is reversed to form 1 2 4 3: 4 3) 0 ([1] 2.
    The current position moves forward by the length plus the skip size, a total of 5, causing it not to move because it wraps around: 4 3 0 [1] 2. The skip size increases to 2.

    The third length, 1, selects a sublist of a single element, and so reversing it has no effect.
    The current position moves forward by the length (1) plus the skip size (2): 4 [3] 0 1 2. The skip size increases to 3.

    The fourth length, 5, selects every element starting with the second: 4) ([3] 0 1 2. Reversing this sublist (3 0 1 2 4 into 4 2 1 0 3) produces: 3) ([4] 2 1 0.
    Finally, the current position moves forward by 8: 3 4 2 1 [0]. The skip size increases to 4.

In this example, the first two numbers in the list end up being 3 and 4; to check the process, you can multiply them together to produce 12.

However, you should instead use the standard list size of 256 (with values 0 to 255) and the sequence of lengths in your puzzle input. Once this process is complete, what is the result of multiplying the first two numbers in the list?

--- Part Two ---

The logic you've constructed forms a single round of the Knot Hash algorithm; running the full thing requires many of these rounds. Some input and output processing is also required.

First, from now on, your input should be taken not as a list of numbers, but as a string of bytes instead. Unless otherwise specified, convert characters to bytes using their ASCII codes. This will allow you to handle arbitrary ASCII strings, and it also ensures that your input lengths are never larger than 255. For example, if you are given 1,2,3, you should convert it to the ASCII codes for each character: 49,44,50,44,51.

Once you have determined the sequence of lengths to use, add the following lengths to the end of the sequence: 17, 31, 73, 47, 23. For example, if you are given 1,2,3, your final sequence of lengths should be 49,44,50,44,51,17,31,73,47,23 (the ASCII codes from the input string combined with the standard length suffix values).

Second, instead of merely running one round like you did above, run a total of 64 rounds, using the same length sequence in each round. The current position and skip size should be preserved between rounds. For example, if the previous example was your first round, you would start your second round with the same length sequence (3, 4, 1, 5, 17, 31, 73, 47, 23, now assuming they came from ASCII codes and include the suffix), but start with the previous round's current position (4) and skip size (4).

Once the rounds are complete, you will be left with the numbers from 0 to 255 in some order, called the sparse hash. Your next task is to reduce these to a list of only 16 numbers called the dense hash. To do this, use numeric bitwise XOR to combine each consecutive block of 16 numbers in the sparse hash (there are 16 such blocks in a list of 256 numbers). So, the first element in the dense hash is the first sixteen elements of the sparse hash XOR'd together, the second element in the dense hash is the second sixteen elements of the sparse hash XOR'd together, etc.

For example, if the first sixteen elements of your sparse hash are as shown below, and the XOR operator is ^, you would calculate the first output number like this:

65 ^ 27 ^ 9 ^ 1 ^ 4 ^ 3 ^ 40 ^ 50 ^ 91 ^ 7 ^ 6 ^ 0 ^ 2 ^ 5 ^ 68 ^ 22 = 64

Perform this operation on each of the sixteen blocks of sixteen numbers in your sparse hash to determine the sixteen numbers in your dense hash.

Finally, the standard way to represent a Knot Hash is as a single hexadecimal string; the final output is the dense hash in hexadecimal notation. Because each number in your dense hash will be between 0 and 255 (inclusive), always represent each number as two hexadecimal digits (including a leading zero as necessary). So, if your first three numbers are 64, 7, 255, they correspond to the hexadecimal numbers 40, 07, ff, and so the first six characters of the hash would be 4007ff. Because every Knot Hash is sixteen such numbers, the hexadecimal representation is always 32 hexadecimal digits (0-f) long.

Here are some example hashes:

    The empty string becomes a2582a3a0e66e6e86e3812dcb672a272.
    AoC 2017 becomes 33efeb34ea91902bb2f59c9920caa6cd.
    1,2,3 becomes 3efbe78a8d82f29979031a4aa0b16a9d.
    1,2,4 becomes 63960835bcdc130f0b66d7ff4f6a5a8e.

Treating your puzzle input as a string of ASCII characters, what is the Knot Hash of your puzzle input? Ignore any leading or trailing whitespace you might encounter. */

type SparseHash = [u8; KnotHashState::SPARSE_HASH_LEN];
pub type DenseHash = [u8; KnotHashState::DENSE_HASH_LEN];

/// Invariant: This always holds a valid UTF-8 bytes.
struct DenseHashString([u8; Self::LEN]);

impl DenseHashString {
    const LEN: usize = KnotHashState::DENSE_HASH_LEN * 2_usize;
    const ALPHA_OFFSET: u8 = b'a' - 10_u8;
    const HIGH_NIBBLE_RANGE: Range<usize> = u8::BITS as usize / 2_usize..u8::BITS as usize;
    const LOW_NIBBLE_RANGE: Range<usize> = 0_usize..Self::HIGH_NIBBLE_RANGE.start;

    fn nibble_value_to_ascii(nibble: u8) -> u8 {
        if nibble < 10_u8 {
            nibble + b'0'
        } else {
            nibble + Self::ALPHA_OFFSET
        }
    }

    fn as_str(&self) -> &str {
        // SAFETY: This is guaranteed by the invariant.
        unsafe { from_utf8_unchecked(&self.0) }
    }
}

impl From<DenseHash> for DenseHashString {
    fn from(value: DenseHash) -> Self {
        let mut dense_hash_string: Self = Self(LargeArrayDefault::large_array_default());

        for (hexadecimal_nibble_pair, dense_byte) in dense_hash_string
            .0
            .chunks_exact_mut(Self::LEN / KnotHashState::DENSE_HASH_LEN)
            .zip(value)
        {
            hexadecimal_nibble_pair[0_usize] = Self::nibble_value_to_ascii(
                dense_byte.view_bits::<Lsb0>()[Self::HIGH_NIBBLE_RANGE].load(),
            );
            hexadecimal_nibble_pair[1_usize] = Self::nibble_value_to_ascii(
                dense_byte.view_bits::<Lsb0>()[Self::LOW_NIBBLE_RANGE].load(),
            );
        }

        dense_hash_string
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct KnotHashState {
    string: SparseHash,
    len: usize,
    curr_pos: usize,
    skip_size: usize,
}

impl KnotHashState {
    const SPARSE_HASH_LEN: usize = 256_usize;
    const DENSE_HASH_LEN: usize = 16_usize;

    fn new(len: usize) -> Self {
        assert!(len <= Self::SPARSE_HASH_LEN);

        let mut string: SparseHash = SparseHash::large_array_default();

        for (index, number) in string[..len].iter_mut().enumerate() {
            *number = index as u8;
        }

        string[len..].fill(0_u8);

        Self {
            string,
            len,
            curr_pos: 0_usize,
            skip_size: 0_usize,
        }
    }

    fn knot_len(&mut self, len: usize) {
        assert!(len <= self.len);

        self.string[..len].reverse();

        let curr_pos_delta: usize = (len + self.skip_size) % self.len;

        self.string[..self.len].rotate_left(curr_pos_delta);
        self.curr_pos = (self.curr_pos + curr_pos_delta) % self.len;
        self.skip_size += 1_usize;
    }

    fn knot_lens(&mut self, lens: &[usize]) {
        for len in lens.iter().copied() {
            self.knot_len(len);
        }
    }

    fn initial_pair_product(&self) -> u32 {
        let first_index: usize = self.len - self.curr_pos;
        let second_index: usize = (first_index + 1_usize) % self.len;

        self.string[first_index] as u32 * self.string[second_index] as u32
    }

    fn dense_hash(&self) -> DenseHash {
        let mut dense_hash: DenseHash = DenseHash::default();
        let mut sparse_hash: SparseHash = self.string;

        sparse_hash.rotate_left(self.len - self.curr_pos);

        for (dense_byte, sparse_slice) in dense_hash
            .iter_mut()
            .zip(sparse_hash.chunks_exact(Self::SPARSE_HASH_LEN / Self::DENSE_HASH_LEN))
        {
            *dense_byte = sparse_slice
                .iter()
                .copied()
                .fold(0_u8, |acc, sparse_byte| acc ^ sparse_byte);
        }

        dense_hash
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    input: String,
    simple_lens: Vec<usize>,
}

impl Solution {
    const LEN: usize = 256_usize;
    const SUFFIX_LENS: &'static [usize] = &[17_usize, 31_usize, 73_usize, 47_usize, 23_usize];
    const ROUNDS: usize = 64_usize;

    fn initial_pair_product_after_knotting_for_len(&self, len: usize) -> u32 {
        let mut knot_hash_state: KnotHashState = KnotHashState::new(len);

        knot_hash_state.knot_lens(&self.simple_lens);

        knot_hash_state.initial_pair_product()
    }

    fn initial_pair_product_after_knotting(&self) -> u32 {
        self.initial_pair_product_after_knotting_for_len(Self::LEN)
    }

    pub fn knot_hash_for_str(string: &str) -> DenseHash {
        let mut knot_hash_state: KnotHashState = KnotHashState::new(Self::LEN);

        for _ in 0_usize..Self::ROUNDS {
            for len in string
                .as_bytes()
                .iter()
                .copied()
                .map(usize::from)
                .chain(Self::SUFFIX_LENS.iter().copied())
            {
                knot_hash_state.knot_len(len);
            }
        }

        knot_hash_state.dense_hash()
    }

    fn knot_hash_string_for_str(string: &str) -> DenseHashString {
        Self::knot_hash_for_str(string).into()
    }

    fn knot_hash(&self) -> DenseHashString {
        Self::knot_hash_string_for_str(&self.input)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(parse_integer, tuple((opt(tag(",")), space0)))),
            |simple_lens| Self {
                input: not_line_ending::<&'i str, NomError<&'i str>>(input)
                    .unwrap()
                    .1
                    .into(),
                simple_lens,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// *Knot* too bad, but Part Two has a huge description... eek.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.initial_pair_product_after_knotting());
    }

    /// Not as bad as I had initially thought, but still a big pivot from part one, or at least a
    /// substantial amount more. Only at the very last minute did I realize that it's probably a good
    /// idea to include a `not_line_ending` filter on the input, as otherwise the terminated `\n`
    /// would throw off the hash.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.knot_hash().as_str());
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

    const SOLUTION_STRS: &'static [&'static str] = &["3, 4, 1, 5"];
    const LEN: usize = 5_usize;

    macro_rules! string {
        [ $( $number:expr, )* ] => { {
            let mut string: SparseHash = SparseHash::large_array_default();
            let string_slice: &[u8] = &[ $( $number, )* ];

            string[..string_slice.len()].copy_from_slice(string_slice);

            string
        } }
    }

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                input: SOLUTION_STRS[0_usize].into(),
                simple_lens: vec![3_usize, 4_usize, 1_usize, 5_usize],
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
    fn test_knot_hash_state_knot_len() {
        for (index, knot_hash_states) in [&[
            KnotHashState {
                string: string![3_u8, 4_u8, 2_u8, 1_u8, 0_u8,],
                len: LEN,
                curr_pos: 3_usize,
                skip_size: 1_usize,
            },
            KnotHashState {
                string: string![1_u8, 2_u8, 4_u8, 3_u8, 0_u8,],
                len: LEN,
                curr_pos: 3_usize,
                skip_size: 2_usize,
            },
            KnotHashState {
                string: string![3_u8, 0_u8, 1_u8, 2_u8, 4_u8,],
                len: LEN,
                curr_pos: 1_usize,
                skip_size: 3_usize,
            },
            KnotHashState {
                string: string![0_u8, 3_u8, 4_u8, 2_u8, 1_u8,],
                len: LEN,
                curr_pos: 4_usize,
                skip_size: 4_usize,
            },
        ]]
        .into_iter()
        .enumerate()
        {
            let mut knot_hash_state: KnotHashState = KnotHashState::new(LEN);

            for (len, expected_knot_hash_state) in solution(index)
                .simple_lens
                .iter()
                .copied()
                .zip(knot_hash_states)
            {
                knot_hash_state.knot_len(len);

                assert_eq!(knot_hash_state, *expected_knot_hash_state);
            }
        }
    }

    #[test]
    fn test_initial_pair_product_after_knotting_for_len() {
        for (index, initial_pair_product_after_knotting_for_len) in [12_u32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).initial_pair_product_after_knotting_for_len(LEN),
                initial_pair_product_after_knotting_for_len
            );
        }
    }

    #[test]
    fn test_knot_hash_for_str() {
        for (string, knot_hash_string_for_str) in [
            ("", "a2582a3a0e66e6e86e3812dcb672a272"),
            ("AoC 2017", "33efeb34ea91902bb2f59c9920caa6cd"),
            ("1,2,3", "3efbe78a8d82f29979031a4aa0b16a9d"),
            ("1,2,4", "63960835bcdc130f0b66d7ff4f6a5a8e"),
        ]
        .into_iter()
        {
            assert_eq!(
                Solution::knot_hash_string_for_str(string).as_str(),
                knot_hash_string_for_str
            );
        }
    }
}
