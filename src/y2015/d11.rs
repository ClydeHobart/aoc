use {
    crate::*,
    bitvec::prelude::*,
    nom::{bytes::complete::take_while_m_n, combinator::map, error::Error, Err, IResult},
    static_assertions::{const_assert, const_assert_eq},
    std::{
        collections::HashSet,
        ops::{Index, IndexMut, Range},
        str::from_utf8_unchecked,
        sync::OnceLock,
    },
};

#[cfg(test)]
use arrayvec::ArrayVec;

/* --- Day 11: Corporate Policy ---

Santa's previous password expired, and he needs help choosing a new one.

To help him remember his new password after the old one expires, Santa has devised a method of coming up with a password based on the previous one. Corporate policy dictates that passwords must be exactly eight lowercase letters (for security reasons), so he finds his new password by incrementing his old password string repeatedly until it is valid.

Incrementing is just like counting with numbers: xx, xy, xz, ya, yb, and so on. Increase the rightmost letter one step; if it was z, it wraps around to a, and repeat with the next letter to the left until one doesn't wrap around.

Unfortunately for Santa, a new Security-Elf recently started, and he has imposed some additional password requirements:

    Passwords must include one increasing straight of at least three letters, like abc, bcd, cde, and so on, up to xyz. They cannot skip letters; abd doesn't count.
    Passwords may not contain the letters i, o, or l, as these letters can be mistaken for other characters and are therefore confusing.
    Passwords must contain at least two different, non-overlapping pairs of letters, like aa, bb, or zz.

For example:

    hijklmmn meets the first requirement (because it contains the straight hij) but fails the second requirement requirement (because it contains i and l).
    abbceffg meets the third requirement (because it repeats bb and ff) but fails the first requirement.
    abbcegjk fails the third requirement, because it only has one double letter (bb).
    The next password after abcdefgh is abcdffaa.
    The next password after ghijklmn is ghjaabcc, because you eventually skip all the passwords that start with ghi..., since i is not allowed.

Given Santa's current password (your puzzle input), what should his next password be?

--- Part Two ---

Santa's password expired again. What's the next one? */

const_assert_eq!(RequirementData::SOLO_VALUE_LEN, u32::BITS as usize);

type SoloBitArray = BitArr!(for RequirementData::SOLO_VALUE_LEN, in u32);

struct RequirementData {
    is_solo_valid: SoloBitArray,
    is_pair_valid: BitVec,
    is_trio_valid: BitVec,
}

impl RequirementData {
    const SOLO_LETTER_LEN: usize = 1_usize;
    const PAIR_LETTER_LEN: usize = 2_usize;
    const TRIO_LETTER_LEN: usize = 3_usize;
    const SOLO_BIT_LEN: usize = Self::SOLO_LETTER_LEN * Solution::LETTER_BIT_LEN;
    const PAIR_BIT_LEN: usize = Self::PAIR_LETTER_LEN * Solution::LETTER_BIT_LEN;
    const TRIO_BIT_LEN: usize = Self::TRIO_LETTER_LEN * Solution::LETTER_BIT_LEN;
    const SOLO_VALUE_LEN: usize = 1_usize << Self::SOLO_BIT_LEN;
    const PAIR_VALUE_LEN: usize = 1_usize << Self::PAIR_BIT_LEN;
    const TRIO_VALUE_LEN: usize = 1_usize << Self::TRIO_BIT_LEN;

    const VALID_PAIR_REQUIREMENT_COUNT: usize = 2_usize;
    const INVALID_LETTERS: [u8; 3_usize] = *b"iol";
    const RESET_INDEX: usize = Self::reset_index();

    const fn is_letter_valid(letter: u8) -> bool {
        let mut is_letter_valid: bool = true;
        let mut invalid_letter_index: usize = 0_usize;

        while invalid_letter_index < Self::INVALID_LETTERS.len() && is_letter_valid {
            is_letter_valid = letter != Self::INVALID_LETTERS[invalid_letter_index];

            invalid_letter_index += 1_usize;
        }

        is_letter_valid
    }

    // This is used in a `const_assert!`.
    #[allow(dead_code)]
    const fn there_are_consecutive_invalid_letters() -> bool {
        let mut there_are_consecutive_invalid_letters: bool = false;
        let mut letter: u8 = MIN_ASCII_LOWERCASE_LETTER;

        while letter < MAX_ASCII_LOWERCASE_LETTER && !there_are_consecutive_invalid_letters {
            there_are_consecutive_invalid_letters =
                !Self::is_letter_valid(letter) && !Self::is_letter_valid(letter + 1_u8);

            letter += 1_u8;
        }

        there_are_consecutive_invalid_letters
    }

    const fn reset_index() -> usize {
        let mut reset_index: usize = 0_usize;

        while !Self::is_letter_valid(ascii_lowercase_letter_from_index(reset_index)) {
            reset_index += 1_usize;
        }

        reset_index
    }

    fn get() -> &'static Self {
        static ONCE_LOCK: OnceLock<RequirementData> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            let mut requirement_data: RequirementData = RequirementData {
                is_solo_valid: SoloBitArray::ZERO,
                is_pair_valid: bitvec![0; Self::PAIR_VALUE_LEN],
                is_trio_valid: bitvec![0; Self::TRIO_VALUE_LEN],
            };

            requirement_data.is_solo_valid[..LETTER_COUNT].fill(true);

            for letter in Self::INVALID_LETTERS {
                requirement_data
                    .is_solo_valid
                    .set(index_from_ascii_lowercase_letter(letter), false);
            }

            for letter_0 in 0_usize..LETTER_COUNT {
                let mut solution: Solution = Solution(0_u64);

                let is_solo_valid: bool = requirement_data.is_solo_valid[letter_0];

                solution[0_usize].store_le(letter_0);

                for letter_1 in 0_usize..LETTER_COUNT {
                    let is_pair_valid: bool =
                        is_solo_valid && requirement_data.is_solo_valid[letter_1];

                    solution[1_usize].store_le(letter_1);
                    requirement_data.is_pair_valid.set(
                        solution[0_usize..Self::PAIR_LETTER_LEN].load_le::<usize>(),
                        is_pair_valid && letter_0 == letter_1,
                    );

                    for letter_2 in 0_usize..LETTER_COUNT {
                        let is_trio_valid: bool =
                            is_pair_valid && requirement_data.is_solo_valid[letter_2];

                        solution[2_usize].store_le(letter_2);
                        requirement_data.is_trio_valid.set(
                            solution[0_usize..Self::TRIO_LETTER_LEN].load_le::<usize>(),
                            is_trio_valid
                                && letter_2 + 1_usize == letter_1
                                && letter_1 + 1_usize == letter_0,
                        );
                    }
                }
            }

            requirement_data
        })
    }

    #[cfg(test)]
    fn print_internal<T: BitStore, const N: usize>(is_valid: &BitSlice<T>, group_str: &str)
    where
        [u8; N]: Default,
    {
        println!("valid {group_str}s:");

        let mut solution: Solution = Solution(0_u64);
        let mut bytes: [u8; N] = Default::default();
        let mut letter_indices: ArrayVec<usize, N> = ArrayVec::new();

        letter_indices.push(0_usize);

        while let Some(letter_index) = letter_indices.pop() {
            if letter_index < LETTER_COUNT {
                bytes[letter_indices.len()] = ascii_lowercase_letter_from_index(letter_index);
                solution[letter_indices.len()].store_le(letter_index);
                letter_indices.push(letter_index + 1_usize);

                if !letter_indices.is_full() {
                    letter_indices.push(0_usize);
                } else if is_valid[solution[0_usize..letter_indices.len()].load_le::<usize>()] {
                    let mut bytes: [u8; N] = bytes;

                    bytes.reverse();

                    // SAFETY: `bytes` is full of lowercase ASCII letters.
                    println!("    {}", unsafe { from_utf8_unchecked(&bytes) });
                }
            }
        }
    }

    #[allow(dead_code)]
    #[cfg(test)]
    fn print() {
        let requirement_data: &RequirementData = Self::get();

        Self::print_internal::<_, { Self::SOLO_LETTER_LEN }>(
            &requirement_data.is_solo_valid,
            "solo",
        );
        Self::print_internal::<_, { Self::PAIR_LETTER_LEN }>(
            &requirement_data.is_pair_valid,
            "pair",
        );
        Self::print_internal::<_, { Self::TRIO_LETTER_LEN }>(
            &requirement_data.is_trio_valid,
            "trio",
        );
    }
}

struct NextSolutionFinder {
    start: Solution,
    end: Solution,
    visited: HashSet<Solution>,
}

impl WeightedGraphSearch for NextSolutionFinder {
    type Vertex = Solution;
    type Cost = u64;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.is_valid()
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        Vec::new()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        if self.visited.contains(vertex) {
            vertex.0.wrapping_sub(self.start.0)
        } else {
            u64::MAX
        }
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        zero_heuristic(self, vertex)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let requirement_data: &RequirementData = RequirementData::get();

        match vertex.evaluate_requirements(true, true, true) {
            (Some(first_invalid_index), _, _) => {
                let neighbor: Solution = vertex.next_valid_solos(first_invalid_index);

                neighbors.push(OpenSetElement(neighbor, neighbor.0.wrapping_sub(vertex.0)));
            }
            (None, valid_pair_letters, any_trio_is_valid) => {
                if valid_pair_letters.count_ones() < RequirementData::VALID_PAIR_REQUIREMENT_COUNT {
                    neighbors.extend(
                        (0_usize
                            ..=Solution::PASSWORD_LETTER_LEN - RequirementData::PAIR_LETTER_LEN)
                            .flat_map(|index| {
                                let pair_range: Range<usize> =
                                    index..index + RequirementData::PAIR_LETTER_LEN;
                                let pair_value: usize = vertex
                                    [index..index + RequirementData::PAIR_LETTER_LEN]
                                    .load_le::<usize>();

                                (!requirement_data.is_pair_valid[pair_value])
                                    .then(|| {
                                        let remaining_pair_values: &BitSlice =
                                            &requirement_data.is_pair_valid[pair_value..];
                                        let pair_values_to_skip: usize =
                                            remaining_pair_values.leading_zeros();

                                        (pair_values_to_skip < remaining_pair_values.len()).then(
                                            || {
                                                let mut neighbor_a: Solution = *vertex;

                                                neighbor_a[pair_range.clone()]
                                                    .store_le(pair_value + pair_values_to_skip);
                                                neighbor_a.reset_start(index);

                                                let mut neighbor_b: Solution = neighbor_a;

                                                neighbor_b.reset_start(pair_range.end - 1_usize);

                                                [
                                                    Some(OpenSetElement(
                                                        neighbor_a,
                                                        neighbor_a.0 - vertex.0,
                                                    )),
                                                    neighbor_b.0.checked_sub(vertex.0).map(
                                                        |cost| OpenSetElement(neighbor_b, cost),
                                                    ),
                                                ]
                                            },
                                        )
                                    })
                                    .flatten()
                                    .into_iter()
                                    .flatten()
                                    .flatten()
                            }),
                    );
                }

                if !any_trio_is_valid {
                    neighbors.extend(
                        (0_usize
                            ..=Solution::PASSWORD_LETTER_LEN - RequirementData::TRIO_LETTER_LEN)
                            .flat_map(|index| {
                                let trio_range: Range<usize> =
                                    index..index + RequirementData::TRIO_LETTER_LEN;
                                let trio_value: usize = vertex
                                    [index..index + RequirementData::TRIO_LETTER_LEN]
                                    .load_le::<usize>();

                                (!requirement_data.is_trio_valid[trio_value])
                                    .then(|| {
                                        let remaining_trio_values: &BitSlice =
                                            &requirement_data.is_trio_valid[trio_value..];
                                        let trio_values_to_skip: usize =
                                            remaining_trio_values.leading_zeros();

                                        (trio_values_to_skip < remaining_trio_values.len()).then(
                                            || {
                                                let mut neighbor_a: Solution = *vertex;

                                                neighbor_a[trio_range.clone()]
                                                    .store_le(trio_value + trio_values_to_skip);
                                                neighbor_a.reset_start(index);

                                                let mut neighbor_b: Solution = neighbor_a;

                                                neighbor_b.reset_start(trio_range.end - 1_usize);

                                                [
                                                    Some(OpenSetElement(
                                                        neighbor_a,
                                                        neighbor_a.0 - vertex.0,
                                                    )),
                                                    neighbor_b.0.checked_sub(vertex.0).map(
                                                        |cost| OpenSetElement(neighbor_b, cost),
                                                    ),
                                                ]
                                            },
                                        )
                                    })
                                    .flatten()
                                    .into_iter()
                                    .flatten()
                                    .flatten()
                            }),
                    );
                }

                if neighbors.is_empty() {
                    let mut neighbor: Solution = Solution(0_u64);

                    neighbor.reset_start(Solution::PASSWORD_LETTER_LEN);
                    neighbors.push(OpenSetElement(neighbor, neighbor.0.wrapping_sub(vertex.0)));
                }
            }
        }
    }

    fn update_vertex(
        &mut self,
        _from: &Self::Vertex,
        &to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.visited.insert(to);

        if self.is_end(&to) && (self.end == self.start || self.cost_from_start(&self.end) > cost) {
            self.end = to;
        }
    }

    fn reset(&mut self) {
        self.visited.clear();
        self.visited.insert(self.start);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Solution(u64);

type SolutionStaticString = StaticString<{ Solution::PASSWORD_LETTER_LEN }>;

impl Solution {
    const PASSWORD_LETTER_LEN: usize = 8_usize;
    const LETTER_BIT_LEN: usize = (LETTER_COUNT - 1_usize).ilog2() as usize + 1_usize;
    const PASSWORD_BIT_MASK: u64 =
        (1_u64 << (Self::PASSWORD_LETTER_LEN * Self::LETTER_BIT_LEN)) - 1_u64;

    const fn bit_index_from_letter_index(letter_index: usize) -> usize {
        letter_index * Self::LETTER_BIT_LEN
    }

    const fn bit_range_from_letter_range(letter_range: Range<usize>) -> Range<usize> {
        Self::bit_index_from_letter_index(letter_range.start)
            ..Self::bit_index_from_letter_index(letter_range.end)
    }

    const fn letter_range_from_letter_index(letter_index: usize) -> Range<usize> {
        letter_index..letter_index + 1_usize
    }

    fn next_valid_solos(self, first_invalid_index: usize) -> Self {
        let requirement_data: &RequirementData = RequirementData::get();
        let first_invalid_value: usize = self[first_invalid_index].load::<usize>();

        let remaining_solo_values: &BitSlice<u32> =
            &requirement_data.is_solo_valid[first_invalid_value..];
        let solo_values_to_skip: usize = remaining_solo_values.leading_zeros();

        if solo_values_to_skip < remaining_solo_values.len() {
            let mut next_valid_solos: Self = self;

            next_valid_solos[first_invalid_index]
                .store_le(first_invalid_value + solo_values_to_skip);
            next_valid_solos.reset_start(first_invalid_index);

            next_valid_solos
        } else {
            let next_index: usize = first_invalid_index + 1_usize;

            if next_index < Self::PASSWORD_LETTER_LEN {
                let next_value: usize = self[next_index].load::<usize>() + 1_usize;

                if requirement_data.is_solo_valid[next_value] {
                    let mut next_valid_solos: Self = self;

                    next_valid_solos[next_index].store_le(next_value);
                    next_valid_solos.reset_start(next_index);

                    next_valid_solos
                } else {
                    let mut invalid_solos: Self = self;

                    invalid_solos[next_index].store_le(next_value);
                    invalid_solos.reset_start(next_index);

                    invalid_solos.next_valid_solos(next_index)
                }
            } else {
                let mut next_valid_solos: Self = Self(0_u64);

                next_valid_solos.reset_start(Self::PASSWORD_LETTER_LEN);

                next_valid_solos
            }
        }
    }

    fn next(self) -> Self {
        let mut start: Self = Self((self.0 + 1_u64) & Self::PASSWORD_BIT_MASK);

        if let (Some(first_invalid_index), _, _) = start.evaluate_requirements(true, false, false) {
            start = start.next_valid_solos(first_invalid_index);
        }

        let mut next_solution_finder: NextSolutionFinder = NextSolutionFinder {
            start,
            end: start,
            visited: HashSet::new(),
        };

        next_solution_finder.run_dijkstra().unwrap();

        next_solution_finder.end
    }

    fn evaluate_requirements(
        self,
        evaluate_solos: bool,
        evaluate_pairs: bool,
        evaluate_trios: bool,
    ) -> (Option<usize>, SoloBitArray, bool) {
        let requirement_data: &RequirementData = RequirementData::get();

        (0_usize..Self::PASSWORD_LETTER_LEN).rev().fold(
            (None, SoloBitArray::ZERO, false),
            |(mut first_invalid_index, mut valid_pair_letters, mut any_trio_is_valid), index| {
                let solo_index: usize =
                    self[index..index + RequirementData::SOLO_LETTER_LEN].load_le::<usize>();

                first_invalid_index = first_invalid_index.or_else(|| {
                    (evaluate_solos && !requirement_data.is_solo_valid[solo_index]).then_some(index)
                });

                if evaluate_pairs
                    && index + RequirementData::PAIR_LETTER_LEN <= Self::PASSWORD_LETTER_LEN
                    && requirement_data.is_pair_valid
                        [self[index..index + RequirementData::PAIR_LETTER_LEN].load_le::<usize>()]
                {
                    valid_pair_letters.set(solo_index, true);
                }

                any_trio_is_valid = any_trio_is_valid
                    || (evaluate_trios
                        && index + RequirementData::TRIO_LETTER_LEN <= Self::PASSWORD_LETTER_LEN
                        && requirement_data.is_trio_valid[self
                            [index..index + RequirementData::TRIO_LETTER_LEN]
                            .load_le::<usize>()]);

                (first_invalid_index, valid_pair_letters, any_trio_is_valid)
            },
        )
    }

    fn is_valid(self) -> bool {
        let (first_invalid_index, valid_letter_pairs, any_trio_is_valid): (
            Option<usize>,
            SoloBitArray,
            bool,
        ) = self.evaluate_requirements(true, true, true);

        first_invalid_index.is_none()
            && valid_letter_pairs.count_ones() >= RequirementData::VALID_PAIR_REQUIREMENT_COUNT
            && any_trio_is_valid
    }

    fn as_string(self) -> SolutionStaticString {
        self.into()
    }

    /// This is useful for debugging
    #[allow(dead_code)]
    fn as_debug_string(self) -> String {
        let vertex: Solution = self;

        use std::fmt::Write;

        let vertex_string: SolutionStaticString = vertex.as_string();

        if vertex_string
            .as_str()
            .as_bytes()
            .iter()
            .all(|byte| ASCII_LOWERCASE_LETTER_RANGE.contains(byte))
        {
            vertex_string.as_str().into()
        } else {
            let mut segments: Vec<String> = Vec::with_capacity(8_usize);

            segments.resize(8_usize, String::new());

            for (letter_index, string) in
                (0_usize..Solution::PASSWORD_LETTER_LEN).zip(segments.iter_mut().rev())
            {
                let value: usize = if letter_index == Solution::PASSWORD_LETTER_LEN - 1_usize {
                    vertex.0.view_bits::<Lsb0>()[letter_index * Solution::LETTER_BIT_LEN..]
                        .load::<usize>()
                } else {
                    vertex[letter_index].load::<usize>()
                };

                if value < LETTER_COUNT {
                    write!(
                        string,
                        "{}",
                        ascii_lowercase_letter_from_index(value) as char
                    )
                    .ok();
                } else {
                    write!(string, "{value}").ok();
                }
            }

            let mut segments_string: String = String::new();

            write!(&mut segments_string, "[").ok();

            for (index, segment) in segments.into_iter().enumerate() {
                write!(
                    &mut segments_string,
                    "{}{segment}",
                    if index > 0_usize { "," } else { "" }
                )
                .ok();
            }

            write!(&mut segments_string, "]").ok();

            segments_string
        }
    }

    fn reset_start(&mut self, len_to_reset: usize) {
        for solo_letter_index in 0_usize..len_to_reset {
            self[solo_letter_index].store_le(RequirementData::RESET_INDEX);
        }
    }
}

const_assert!(
    (Solution::PASSWORD_LETTER_LEN + 2_usize) * Solution::LETTER_BIT_LEN <= usize::BITS as usize
);

impl From<Solution> for SolutionStaticString {
    fn from(value: Solution) -> Self {
        let mut bytes: [u8; Solution::PASSWORD_LETTER_LEN] = Default::default();

        for (index, letter) in bytes.iter_mut().rev().enumerate() {
            *letter = ascii_lowercase_letter_from_index(value[index].load_le::<usize>());
        }

        // SAFETY: all bytes are ASCII lowercase letters
        unsafe { from_utf8_unchecked(&bytes) }.try_into().unwrap()
    }
}

impl Index<Range<usize>> for Solution {
    type Output = BitSlice<u64, Lsb0>;

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.0.view_bits()[Self::bit_range_from_letter_range(index)]
    }
}

impl Index<usize> for Solution {
    type Output = BitSlice<u64, Lsb0>;

    fn index(&self, index: usize) -> &Self::Output {
        &self[Self::letter_range_from_letter_index(index)]
    }
}

impl IndexMut<Range<usize>> for Solution {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.0.view_bits_mut()[Self::bit_range_from_letter_range(index)]
    }
}

impl IndexMut<usize> for Solution {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self[Self::letter_range_from_letter_index(index)]
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            take_while_m_n(
                Self::PASSWORD_LETTER_LEN,
                Self::PASSWORD_LETTER_LEN,
                |c: char| c.is_ascii_lowercase(),
            ),
            |password: &str| {
                let mut solution: Self = Self(Default::default());

                for (index, letter) in password.as_bytes().iter().rev().copied().enumerate() {
                    solution[index].store_le(index_from_ascii_lowercase_letter(letter));
                }

                solution
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Took longer than expected on this one.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.next().as_string());
    }

    /// Also took longer than expected on this one because it found a bug.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.next().next().as_string());
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
    use super::*;

    const SOLUTION_STRS: &'static [&'static str] =
        &["hijklmmn", "abbceffg", "abbcegjk", "abcdefgh", "ghijklmn"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(0b_00111_01000_01001_01010_01011_01100_01100_01101_u64),
                Solution(0b_00000_00001_00001_00010_00100_00101_00101_00110_u64),
                Solution(0b_00000_00001_00001_00010_00100_00110_01001_01010_u64),
                Solution(0b_00000_00001_00010_00011_00100_00101_00110_00111_u64),
                Solution(0b_00110_00111_01000_01001_01010_01011_01100_01101_u64),
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
    fn test_as_string() {
        for (index, string) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(solution(index).as_string().as_str(), string);
        }
    }

    #[test]
    fn test_next() {
        for (index, next_string) in ["hjaaabcc", "abbcefgg", "abbcffgh", "abcdffaa", "ghjaabcc"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).next().as_string().as_str(),
                next_string,
                "original: {}",
                solution(index).as_string().as_str()
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
