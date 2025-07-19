use {
    crate::*,
    bitvec::prelude::*,
    md5::compute,
    nom::{bytes::complete::take_while, combinator::map, error::Error, Err, IResult},
    std::fmt::Write,
};

/* --- Day 4: The Ideal Stocking Stuffer ---

Santa needs help mining some AdventCoins (very similar to bitcoins) to use as gifts for all the economically forward-thinking little girls and boys.

To do this, he needs to find MD5 hashes which, in hexadecimal, start with at least five zeroes. The input to the MD5 hash is some secret key (your puzzle input, given below) followed by a number in decimal. To mine AdventCoins, you must find Santa the lowest positive number (no leading zeroes: 1, 2, 3, ...) that produces such a hash.

For example:

    If your secret key is abcdef, the answer is 609043, because the MD5 hash of abcdef609043 starts with five zeroes (000001dbbfa...), and it is the lowest such number to do so.
    If your secret key is pqrstuv, the lowest number it combines with to make an MD5 hash starting with five zeroes is 1048970; that is, the MD5 hash of pqrstuv1048970 looks like 000006136ef.... */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(String);

impl Solution {
    const NIBBLE_BITS: usize = u8::BITS as usize / 2_usize;
    const Q1_PREFIX_NIBBLE_LEN: usize = 5_usize;
    const Q2_PREFIX_NIBBLE_LEN: usize = 6_usize;

    fn yields_zero_prefixed_hash(&mut self, value: u32, prefix_bit_len: usize) -> bool {
        write!(&mut self.0, "{value}").ok();

        compute(&self.0).0.as_bits::<Msb0>()[..prefix_bit_len].load::<usize>() == 0_usize
    }

    fn min_positive_number_yielding_zero_suffixed_hash(
        &self,
        prefix_nibble_len: usize,
    ) -> Option<u32> {
        let len: usize = self.0.len();
        let prefix_bit_len: usize = prefix_nibble_len * Self::NIBBLE_BITS;
        let mut solution: Self = self.clone();

        (1_u32..u32::MAX).find(|&value| {
            let yields_5_zero_suffixed_hash: bool =
                solution.yields_zero_prefixed_hash(value, prefix_bit_len);

            solution.0.truncate(len);

            yields_5_zero_suffixed_hash
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            map(
                take_while(|c: char| c.is_ascii_alphanumeric()),
                String::from,
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Previous advent problems from the future prepared me for this well lol.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.min_positive_number_yielding_zero_suffixed_hash(Self::Q1_PREFIX_NIBBLE_LEN));
    }

    /// I could improve this to use the answer from q1 to jump-start the search, but I prefer
    /// keeping the questions non-mutative.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.min_positive_number_yielding_zero_suffixed_hash(Self::Q2_PREFIX_NIBBLE_LEN));
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

    const SOLUTION_STRS: &'static [&'static str] = &["abcdef", "pqrstuv"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![Solution("abcdef".into()), Solution("pqrstuv".into())])
            [index]
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
    fn test_min_positive_number_yielding_5_zero_suffixed_hash() {
        for (index, min_positive_number_yielding_5_zero_suffixed_hash) in
            [Some(609043_u32), Some(1048970_u32)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).min_positive_number_yielding_zero_suffixed_hash(
                    Solution::Q1_PREFIX_NIBBLE_LEN
                ),
                min_positive_number_yielding_5_zero_suffixed_hash
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
