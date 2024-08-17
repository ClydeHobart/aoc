use {
    crate::*,
    bitvec::prelude::*,
    md5::{compute, Digest},
    nom::{character::complete::alpha1, combinator::map, error::Error, Err, IResult},
    rayon::iter::{IntoParallelRefMutIterator, ParallelIterator},
    static_assertions::const_assert_eq,
    std::{collections::VecDeque, fmt::Write, marker::PhantomData, mem::swap, ops::Range},
};

/* --- Day 14: One-Time Pad ---

In order to communicate securely with Santa while you're on this mission, you've been using a one-time pad that you generate using a pre-agreed algorithm. Unfortunately, you've run out of keys in your one-time pad, and so you need to generate some more.

To generate keys, you first get a stream of random data by taking the MD5 of a pre-arranged salt (your puzzle input) and an increasing integer index (starting with 0, and represented in decimal); the resulting MD5 hash should be represented as a string of lowercase hexadecimal digits.

However, not all of these MD5 hashes are keys, and you need 64 new keys for your one-time pad. A hash is a key only if:

    It contains three of the same character in a row, like 777. Only consider the first such triplet in a hash.
    One of the next 1000 hashes in the stream contains that same character five times in a row, like 77777.

Considering future hashes for five-of-a-kind sequences does not cause those hashes to be skipped; instead, regardless of whether the current hash is a key, always resume testing for keys starting with the very next hash.

For example, if the pre-arranged salt is abc:

    The first index which produces a triple is 18, because the MD5 hash of abc18 contains ...cc38887a5.... However, index 18 does not count as a key for your one-time pad, because none of the next thousand hashes (index 19 through index 1018) contain 88888.
    The next index which produces a triple is 39; the hash of abc39 contains eee. It is also the first key: one of the next thousand hashes (the one at index 816) contains eeeee.
    None of the next six triples are keys, but the one after that, at index 92, is: it contains 999 and index 200 contains 99999.
    Eventually, index 22728 meets all of the criteria to generate the 64th key.

So, using our example salt of abc, index 22728 produces the 64th key.

Given the actual salt in your puzzle input, what index produces your 64th one-time pad key?

--- Part Two ---

Of course, in order to make this process even more secure, you've also implemented key stretching.

Key stretching forces attackers to spend more time generating hashes. Unfortunately, it forces everyone else to spend more time, too.

To implement key stretching, whenever you generate a hash, before you use it, you first find the MD5 hash of that hash, then the MD5 hash of that hash, and so on, a total of 2016 additional hashings. Always use lowercase hexadecimal representations of hashes.

For example, to find the stretched hash for index 0 and salt abc:

    Find the MD5 hash of abc0: 577571be4de9dcce85a041ba0410f29f.
    Then, find the MD5 hash of that hash: eec80a0c92dc8a0777c619d9bb51e910.
    Then, find the MD5 hash of that hash: 16062ce768787384c81fe17a7a60c7e3.
    ...repeat many times...
    Then, find the MD5 hash of that hash: a107ff634856bb300138cac6568c0f24.

So, the stretched hash for index 0 in this situation is a107ff.... In the end, you find the original hash (one use of MD5), then find the hash-of-the-previous-hash 2016 times, for a total of 2017 uses of MD5.

The rest of the process remains the same, but now the keys are entirely different. Again for salt abc:

    The first triple (222, at index 5) has no matching 22222 in the next thousand hashes.
    The second triple (eee, at index 10) hash a matching eeeee at index 89, and so it is the first key.
    Eventually, index 22551 produces the 64th key (triple fff with matching fffff at index 22859.

Given the actual salt in your puzzle input and using 2016 extra MD5 calls of key stretching, what index now produces your 64th one-time pad key? */

#[derive(Default)]
struct Nibbles([u8; 32_usize]);

impl Nibbles {
    const ALHPA_OFFSET: u8 = b'a' - 10_u8;

    fn initialize(&mut self, data: &[u8]) {
        let digest: Digest = compute(data);

        for (nibble_dst, nibble) in self.0.iter_mut().zip(
            digest
                .0
                .as_bits::<Msb0>()
                .chunks_exact(4_usize)
                .map(|nibble_slice| nibble_slice.load::<u8>()),
        ) {
            *nibble_dst = nibble;
        }
    }

    fn make_printable(&mut self) {
        for nibble in self.0.iter_mut() {
            let nibble_value: u8 = *nibble;

            *nibble = nibble_value
                + if nibble_value <= 9_u8 {
                    b'0'
                } else {
                    Self::ALHPA_OFFSET
                };
        }
    }
}

type RunBitArr = BitArr!(for 16_usize, in u16);

#[derive(Clone, Copy)]
struct HashRuns {
    index: u32,
    first_three_len_run: Option<u8>,
    five_len_runs: RunBitArr,
}

impl HashRuns {
    fn is_slice_all_value(slice: &[u8], value: u8) -> bool {
        slice.iter().copied().all(|slice_byte| slice_byte == value)
    }

    fn from_nibbles_and_index(nibbles: &Nibbles, index: u32) -> Self {
        let mut first_three_len_run: Option<u8> = None;
        let mut five_len_runs: RunBitArr = RunBitArr::ZERO;

        for (slice_index, three_nibbles_slice) in nibbles.0.windows(3_usize).enumerate() {
            let nibble: u8 = three_nibbles_slice[0_usize];

            if Self::is_slice_all_value(three_nibbles_slice, nibble) {
                if first_three_len_run.is_none() {
                    first_three_len_run = Some(nibble);
                }

                let bit_index: usize = nibble as usize;

                if !five_len_runs[bit_index] {
                    five_len_runs.set(
                        bit_index,
                        nibbles.0.get(slice_index..slice_index + 5_usize).map_or(
                            false,
                            |five_nibbles_slice| {
                                Self::is_slice_all_value(five_nibbles_slice, nibble)
                            },
                        ),
                    );
                }
            }
        }

        HashRuns {
            index,
            first_three_len_run,
            five_len_runs,
        }
    }

    fn is_interesting(self) -> bool {
        self.first_three_len_run.is_some() || self.five_len_runs.any()
    }
}

trait Hash {
    fn hash(data: &[u8], index: u32) -> HashRuns;
}

struct StandardHash;

impl Hash for StandardHash {
    fn hash(data: &[u8], index: u32) -> HashRuns {
        let mut nibbles: Nibbles = Nibbles::default();

        nibbles.initialize(data);

        HashRuns::from_nibbles_and_index(&nibbles, index)
    }
}

struct StretchedHash;

impl StretchedHash {
    const ITERATIONS: usize = 2016_usize;
}

impl Hash for StretchedHash {
    fn hash(data: &[u8], index: u32) -> HashRuns {
        let [mut nibbles_a, mut nibbles_b]: [Nibbles; 2_usize] = Default::default();
        let [mut curr_nibbles, mut prev_nibbles]: [&mut Nibbles; 2_usize] =
            [&mut nibbles_a, &mut nibbles_b];

        curr_nibbles.initialize(data);

        for _ in 0_usize..Self::ITERATIONS {
            swap(&mut curr_nibbles, &mut prev_nibbles);
            prev_nibbles.make_printable();
            curr_nibbles.initialize(&prev_nibbles.0);
        }

        HashRuns::from_nibbles_and_index(curr_nibbles, index)
    }
}

const KEY_ITER_THREAD_COUNT: usize = 8_usize;
const KEY_ITER_FIVE_LEN_RUN_LEN: usize = 1000_usize;

const_assert_eq!(KEY_ITER_FIVE_LEN_RUN_LEN % KEY_ITER_THREAD_COUNT, 0_usize);

const KEY_ITER_HASHES_PER_THREAD: usize = KEY_ITER_FIVE_LEN_RUN_LEN / KEY_ITER_THREAD_COUNT;

#[derive(Default)]
struct ThreadData {
    index_range: Range<u32>,
    string: String,
    interesting_hash_runs: VecDeque<HashRuns>,
}

struct KeyIter<H: Hash> {
    hash_index: u32,
    key_index: u32,
    salt_len: usize,
    // context: String,
    thread_datas: [ThreadData; KEY_ITER_THREAD_COUNT],
    interesting_hash_runs: VecDeque<HashRuns>,
    hash: PhantomData<H>,
}

impl<H: Hash> KeyIter<H> {
    fn new(salt: &str) -> Self {
        let mut thread_datas: [ThreadData; KEY_ITER_THREAD_COUNT] =
            LargeArrayDefault::large_array_default();

        for thread_data in thread_datas.iter_mut() {
            thread_data.string = salt.into();
        }

        Self {
            hash_index: 0_u32,
            key_index: 0_u32,
            salt_len: salt.len(),
            thread_datas,
            interesting_hash_runs: VecDeque::new(),
            hash: PhantomData::default(),
        }
    }

    fn compute_hash_runs(string: &mut String, salt_len: usize, index: u32) -> HashRuns {
        string.truncate(salt_len);

        write!(string, "{}", index).ok();

        H::hash(string.as_bytes(), index)
    }

    fn compute_interesting_indices(&mut self) {
        // while self.hash_index <= self.key_index + 1000_u32 {
        if self.hash_index <= self.key_index + KEY_ITER_FIVE_LEN_RUN_LEN as u32 {
            for (index, thread_data) in self.thread_datas.iter_mut().enumerate() {
                let index_range_start: u32 =
                    self.hash_index + (index * KEY_ITER_HASHES_PER_THREAD) as u32;

                thread_data.index_range =
                    index_range_start..index_range_start + KEY_ITER_HASHES_PER_THREAD as u32;
            }

            // We really want 1 + 1000 different hashes computed when self.hash_index and
            // self.key_index start off at the same value. Throw that extra one at the end.
            self.thread_datas.last_mut().unwrap().index_range.end += 1_u32;
            self.thread_datas.par_iter_mut().for_each(|thread_data| {
                thread_data.interesting_hash_runs.extend(
                    thread_data.index_range.clone().filter_map(|index| {
                        let hash_runs: HashRuns =
                            Self::compute_hash_runs(&mut thread_data.string, self.salt_len, index);

                        hash_runs.is_interesting().then_some(hash_runs)
                    }),
                );
            });
            self.interesting_hash_runs.extend(
                self.thread_datas
                    .iter_mut()
                    .flat_map(|thread_data| thread_data.interesting_hash_runs.drain(..)),
            );
            self.hash_index += KEY_ITER_FIVE_LEN_RUN_LEN as u32 + 1_u32;

            // let hash_runs: HashRuns = self.compute_hash_runs();

            // if hash_runs.is_interesting() {
            //     self.interesting_hash_runs.push_back(hash_runs);
            // }

            // self.hash_index += 1_u32;
        }
    }
}

impl<H: Hash> Iterator for KeyIter<H> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.compute_interesting_indices();

            if let Some(hash_runs) = self.interesting_hash_runs.front() {
                if hash_runs.index == self.key_index {
                    let hash_runs = self.interesting_hash_runs.pop_front().unwrap();
                    let max_hash_run_index: u32 = self.key_index + KEY_ITER_FIVE_LEN_RUN_LEN as u32;
                    let result: Option<u32> = hash_runs
                        .first_three_len_run
                        .map_or(false, |nibble| {
                            let nibble_index: usize = nibble as usize;

                            self.interesting_hash_runs
                                .iter()
                                .take_while(|hash_runs| hash_runs.index <= max_hash_run_index)
                                .any(|hash_runs| hash_runs.five_len_runs[nibble_index])
                        })
                        .then_some(self.key_index);

                    self.key_index += 1_u32;

                    if result.is_some() {
                        return result;
                    }
                } else {
                    self.key_index = hash_runs.index;
                }
            } else {
                self.key_index = self.hash_index;
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(String);

impl Solution {
    fn iter_standard_keys(&self) -> KeyIter<StandardHash> {
        KeyIter::new(&self.0)
    }

    fn iter_stretched_keys(&self) -> KeyIter<StretchedHash> {
        KeyIter::new(&self.0)
    }

    fn key_64<H: Hash>(key_iter: KeyIter<H>, verbose: bool) -> u32 {
        key_iter
            .enumerate()
            .map(|(index, key)| {
                if verbose {
                    dbg!(index);
                }

                key
            })
            .skip(63_usize)
            .next()
            .unwrap()
    }

    fn standard_key_64(&self, verbose: bool) -> u32 {
        Self::key_64(self.iter_standard_keys(), verbose)
    }

    fn stretched_key_64(&self, verbose: bool) -> u32 {
        Self::key_64(self.iter_stretched_keys(), verbose)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map(alpha1, String::from), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.standard_key_64(args.verbose));
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.stretched_key_64(args.verbose));
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

    const SOLUTION_STR: &'static str = "abc\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution("abc".into()))
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_compute_hash_runs() {
        const SALT_LEN: usize = 3_usize;

        type StandardKeyIter = KeyIter<StandardHash>;
        type StretchedKeyIter = KeyIter<StretchedHash>;

        let mut string: String = "abc".into();

        let string: &mut String = &mut string;

        assert_eq!(
            StandardKeyIter::compute_hash_runs(string, SALT_LEN, 18_u32).first_three_len_run,
            Some(8_u8)
        );
        assert_eq!(
            StandardKeyIter::compute_hash_runs(string, SALT_LEN, 39_u32).first_three_len_run,
            Some(0xE_u8)
        );
        assert!(
            StandardKeyIter::compute_hash_runs(string, SALT_LEN, 816_u32).five_len_runs[0xE_usize]
        );
        assert_eq!(
            StandardKeyIter::compute_hash_runs(string, SALT_LEN, 92_u32).first_three_len_run,
            Some(9_u8)
        );
        assert!(
            StandardKeyIter::compute_hash_runs(string, SALT_LEN, 200_u32).five_len_runs[9_usize]
        );
        assert_eq!(
            StretchedKeyIter::compute_hash_runs(string, SALT_LEN, 5_u32).first_three_len_run,
            Some(2_u8)
        );
        assert_eq!(
            StretchedKeyIter::compute_hash_runs(string, SALT_LEN, 10_u32).first_three_len_run,
            Some(0xE_u8)
        );
        assert!(
            StretchedKeyIter::compute_hash_runs(string, SALT_LEN, 89_u32).five_len_runs[0xE_usize]
        );
        assert_eq!(
            StretchedKeyIter::compute_hash_runs(string, SALT_LEN, 22551_u32).first_three_len_run,
            Some(0xF_u8)
        );
        assert!(
            StretchedKeyIter::compute_hash_runs(string, SALT_LEN, 22859_u32).five_len_runs
                [0xF_usize]
        );
        // let mut key_iter: KeyIter<_> = solution().iter_standard_keys();

        // key_iter.hash_index = 18_u32;

        // assert_eq!(key_iter.compute_hash_runs().first_three_len_run, Some(8_u8));

        // key_iter.hash_index = 39_u32;

        // assert_eq!(
        //     key_iter.compute_hash_runs().first_three_len_run,
        //     Some(0xE_u8)
        // );

        // key_iter.hash_index = 816_u32;

        // assert!(key_iter.compute_hash_runs().five_len_runs[0xE_usize]);

        // key_iter.hash_index = 92_u32;

        // assert_eq!(key_iter.compute_hash_runs().first_three_len_run, Some(9_u8));

        // key_iter.hash_index = 200_u32;

        // assert!(key_iter.compute_hash_runs().five_len_runs[9_usize]);

        // let mut key_iter: KeyIter<_> = solution().iter_stretched_keys();

        // key_iter.hash_index = 5_u32;

        // assert_eq!(key_iter.compute_hash_runs().first_three_len_run, Some(2_u8));

        // key_iter.hash_index = 10_u32;

        // assert_eq!(
        //     key_iter.compute_hash_runs().first_three_len_run,
        //     Some(0xE_u8)
        // );

        // key_iter.hash_index = 89_u32;

        // assert!(key_iter.compute_hash_runs().five_len_runs[0xE_usize]);

        // key_iter.hash_index = 22551_u32;

        // assert_eq!(
        //     key_iter.compute_hash_runs().first_three_len_run,
        //     Some(0xF_u8)
        // );

        // key_iter.hash_index = 22859_u32;

        // assert!(key_iter.compute_hash_runs().five_len_runs[0xF_usize]);
    }

    #[test]
    fn test_iter_standard_keys() {
        assert_eq!(
            solution()
                .iter_standard_keys()
                .take(2_usize)
                .collect::<Vec<u32>>(),
            vec![39_u32, 92_u32]
        );

        assert_eq!(
            solution().iter_standard_keys().skip(63_usize).next(),
            Some(22728_u32)
        );
    }

    #[test]
    fn test_iter_stretched_keys() {
        assert_eq!(solution().iter_stretched_keys().next(), Some(10_u32));

        assert_eq!(
            solution().iter_stretched_keys().skip(63_usize).next(),
            Some(22551_u32)
        );
    }

    #[test]
    fn test_key_64() {
        assert_eq!(solution().standard_key_64(true), 22728);
    }
}
