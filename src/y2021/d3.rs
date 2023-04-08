use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        character::complete::{digit1, line_ending},
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::terminated,
        Err,
    },
    std::{cmp::Ordering, mem::transmute},
};

// Allow dead code, since these values are constructed via `transmute` in `<MostCommonBit as
// From<Ordering>>::from`
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(i8)]
enum MostCommonBit {
    Zero = -1,
    EquallyCommon = 0,
    One = 1,
}

impl MostCommonBit {
    fn try_as_bool(self) -> Option<bool> {
        match self {
            MostCommonBit::Zero => Some(false),
            MostCommonBit::EquallyCommon => None,
            MostCommonBit::One => Some(true),
        }
    }
}

impl From<Ordering> for MostCommonBit {
    fn from(value: Ordering) -> Self {
        // SAFETY: `Ordering` is also `repr(i8)` with the same set of 3 values
        unsafe { transmute(value) }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    nums: Vec<u16>,
    bits: u32,
}

impl Solution {
    fn compute_most_commont_bit<I: Iterator<Item = u16>>(iter: I, bit: u16) -> MostCommonBit {
        iter.fold(0_i32, |delta, num| {
            delta + if num & bit != 0_u16 { 1_i32 } else { -1_i32 }
        })
        .cmp(&0_i32)
        .into()
    }

    fn compute_gamma_rate(&self) -> u16 {
        (0_u32..self.bits)
            .fold((1_u16, 0_u16), |(bit, gamma_rate), _| {
                (
                    bit << 1_u32,
                    match Self::compute_most_commont_bit(self.nums.iter().copied(), bit) {
                        MostCommonBit::Zero => gamma_rate,
                        MostCommonBit::EquallyCommon => {
                            unimplemented!("The desired outcome in this case is ill-defined")
                        }
                        MostCommonBit::One => gamma_rate | bit,
                    },
                )
            })
            .1
    }

    fn compute_gamma_and_epsilon_rates(&self) -> (u16, u16) {
        let gamma_rate: u16 = self.compute_gamma_rate();

        (gamma_rate, gamma_rate ^ ((1_u16 << self.bits) - 1_u16))
    }

    fn compute_power_consumption(&self) -> u32 {
        let (gamma_rate, epsilon_rate): (u16, u16) = self.compute_gamma_and_epsilon_rates();

        gamma_rate as u32 * epsilon_rate as u32
    }

    fn oxygen_generator_rating_bit_criteria(mcb: MostCommonBit) -> bool {
        mcb.try_as_bool().unwrap_or(true)
    }

    fn co2_scrubber_rating_bit_critera(mcb: MostCommonBit) -> bool {
        !Self::oxygen_generator_rating_bit_criteria(mcb)
    }

    fn compute_rating<B: Fn(MostCommonBit) -> bool>(&self, bit_criteria: B) -> u16 {
        let mut filtered_out_nums: BitVec<u32, Lsb0> = bitvec![u32, Lsb0; 0; self.nums.len()];
        let mut to_be_filtered_out_nums: Vec<usize> = Vec::new();
        let mut valid_nums: usize = self.nums.len();
        let mut bit: u16 = 1_u16 << self.bits;

        while valid_nums > 1_usize {
            assert_ne!(
                bit, 1_u16,
                "The desired outcome in this case is ill-defined"
            );
            bit >>= 1_u32;

            let expected_bit: bool = bit_criteria(Self::compute_most_commont_bit(
                filtered_out_nums.iter_zeros().map(|index| self.nums[index]),
                bit,
            ));

            for index in filtered_out_nums
                .iter_zeros()
                .filter(|index| (self.nums[*index] & bit != 0_u16) != expected_bit)
            {
                to_be_filtered_out_nums.push(index);
            }

            valid_nums -= to_be_filtered_out_nums.len();

            for index in to_be_filtered_out_nums.drain(..) {
                filtered_out_nums.set(index, true);
            }
        }

        assert_eq!(
            valid_nums, 1_usize,
            "The desired outcome in this case is ill-defined"
        );

        self.nums[filtered_out_nums.iter_zeros().next().unwrap()]
    }

    fn compute_oxygen_generator_rating(&self) -> u16 {
        self.compute_rating(Self::oxygen_generator_rating_bit_criteria)
    }

    fn compute_co2_scrubber_rating(&self) -> u16 {
        self.compute_rating(Self::co2_scrubber_rating_bit_critera)
    }

    fn compute_life_support_rating(&self) -> u32 {
        self.compute_oxygen_generator_rating() as u32 * self.compute_co2_scrubber_rating() as u32
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_power_consumption());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_life_support_rating());
    }
}

impl<'a> TryFrom<&'a str> for Solution {
    type Error = Err<Error<&'a str>>;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let bits: u32 = digit1(value)?.1.len() as u32;
        let mut iter = iterator(
            value,
            terminated(
                map_res(digit1, |src| u16::from_str_radix(src, 2)),
                opt(line_ending),
            ),
        );
        let nums: Vec<u16> = iter.collect();

        iter.finish()?;

        Ok(Self { nums, bits })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const DIAGNOSTIC_REPORT_STR: &str =
        "00100\n11110\n10110\n10111\n10101\n01111\n00111\n11100\n10000\n11001\n00010\n01010";

    lazy_static! {
        static ref SOLUTION: Solution = solutions();
    }

    fn solutions() -> Solution {
        Solution {
            nums: vec![
                0b00100_u16,
                0b11110_u16,
                0b10110_u16,
                0b10111_u16,
                0b10101_u16,
                0b01111_u16,
                0b00111_u16,
                0b11100_u16,
                0b10000_u16,
                0b11001_u16,
                0b00010_u16,
                0b01010_u16,
            ],
            bits: 5_u32,
        }
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(DIAGNOSTIC_REPORT_STR), Ok(solutions()));
    }

    #[test]
    fn test_compute_gamma_and_epsilon_rates() {
        assert_eq!(SOLUTION.compute_gamma_and_epsilon_rates(), (22_u16, 9_u16));
    }

    #[test]
    fn test_compute_power_consumption() {
        assert_eq!(SOLUTION.compute_power_consumption(), 198_u32);
    }

    #[test]
    fn test_compute_oxygen_generator_rating() {
        assert_eq!(SOLUTION.compute_oxygen_generator_rating(), 23_u16);
    }

    #[test]
    fn test_compute_co2_scrubber_rating() {
        assert_eq!(SOLUTION.compute_co2_scrubber_rating(), 10_u16);
    }

    #[test]
    fn test_compute_life_support_rating() {
        assert_eq!(SOLUTION.compute_life_support_rating(), 230_u32);
    }
}
