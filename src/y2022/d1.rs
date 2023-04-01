use {
    crate::*,
    std::{mem::MaybeUninit, str::FromStr},
};

pub struct Solution(String);

impl Solution {
    fn iter_elf_calories(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.split("\n\n").map(|elf_calories_str: &str| -> u32 {
            elf_calories_str
                .split('\n')
                .map(|calories_str: &str| -> u32 {
                    match u32::from_str(calories_str) {
                        Ok(calories) => calories,
                        Err(err) => {
                            eprintln!(
                                "Encountered ParseIntError {} while parsing \"{}\"",
                                err, calories_str
                            );

                            0_u32
                        }
                    }
                })
                .sum()
        })
    }

    fn calories_sum_of_max_calories_elf(&self) -> u32 {
        match self.iter_elf_calories().max() {
            Some(calories_sum_of_max_calories_elf) => calories_sum_of_max_calories_elf,
            None => {
                eprintln!("Iterator yielded no maximum. Were there any lines?");

                0
            }
        }
    }

    fn calories_sum_of_top_n_calories_elves<const N: usize>(&self) -> u32 {
        // SAFETY: `0_u32` is 4 consecutive `0_u8` bytes in memory, and `MaybeUninit` guarantees
        // alignment
        let mut top_n_calories: [u32; N] = unsafe { MaybeUninit::zeroed().assume_init() };
        let n: usize = top_n_calories.len();

        if n == 0 {
            eprintln!(
                "calories_sum_of_top_n_calories_elves() called with N == 0, terminating early"
            );

            return 0;
        }

        let n_minus_1 = n - 1_usize;

        for elf_calories in self.iter_elf_calories() {
            for index in 0_usize..n {
                if elf_calories > top_n_calories[index] {
                    top_n_calories[n_minus_1] = elf_calories;
                    top_n_calories[index..].rotate_right(1_usize);

                    break;
                }
            }
        }

        top_n_calories.into_iter().sum()
    }

    fn calories_sum_of_top_3_calories_elves(&self) -> u32 {
        self.calories_sum_of_top_n_calories_elves::<3_usize>()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.calories_sum_of_max_calories_elf());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.calories_sum_of_top_3_calories_elves());
    }
}

impl TryFrom<&str> for Solution {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(Solution(value.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALORIES_STR: &str = concat!(
        "1000\n", "2000\n", "3000\n", "\n", "4000\n", "\n", "5000\n", "6000\n", "\n", "7000\n",
        "8000\n", "9000\n", "\n", "10000"
    );

    #[test]
    fn test_calories_sum_of_max_calories_elf() {
        assert_eq!(
            Solution::try_from(CALORIES_STR)
                .unwrap()
                .calories_sum_of_max_calories_elf(),
            24000
        );
    }

    #[test]
    fn test_calories_sum_of_top_3_calories_elves() {
        assert_eq!(
            Solution::try_from(CALORIES_STR)
                .unwrap()
                .calories_sum_of_top_3_calories_elves(),
            45000
        );
    }
}
