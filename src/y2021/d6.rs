use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::digit1,
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::terminated,
        Err,
    },
    std::{num::ParseIntError, str::FromStr},
};

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct State {
    counts: [usize; State::COUNTS_LEN],
    offset: usize,
    total: usize,
}

impl State {
    const COUNTS_LEN: usize = 9_usize;
    const CYCLE_LEN: usize = 7_usize;

    fn step(&mut self) {
        let new: usize = self.counts[self.offset];

        self.total += new;
        self.counts[(self.offset + Self::CYCLE_LEN) % Self::COUNTS_LEN] += new;
        self.offset = (self.offset + 1_usize) % Self::COUNTS_LEN;
    }

    #[cfg(test)]
    fn build_timers(&self) -> Vec<u8> {
        let mut timers: Vec<u8> = Vec::with_capacity(self.total);

        for timer in 0_u8..Self::COUNTS_LEN as u8 {
            for _ in 0_usize..self.counts[(self.offset + timer as usize) % Self::COUNTS_LEN] {
                timers.push(timer);
            }
        }

        timers
    }
}

impl<'i> TryFrom<&'i str> for State {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        enum ParseTimerError {
            ParseIntError(ParseIntError),
            OutOfRange,
        }

        let mut counts: [usize; State::COUNTS_LEN] = [0_usize; State::COUNTS_LEN];
        let mut total: usize = 0_usize;
        let mut iter = iterator(
            input,
            terminated(
                map_res(digit1, |input: &str| {
                    let timer: u8 = u8::from_str(input).map_err(ParseTimerError::ParseIntError)?;

                    if timer < Self::COUNTS_LEN as u8 {
                        Ok(timer)
                    } else {
                        Err(ParseTimerError::OutOfRange)
                    }
                }),
                opt(tag(",")),
            ),
        );

        for index in iter.into_iter() {
            counts[index as usize] += 1_usize;
            total += 1_usize;
        }

        iter.finish()?;

        Ok(Self {
            counts,
            offset: 0_usize,
            total,
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(State);

impl Solution {
    fn count_lanternfish(&self, days: usize) -> usize {
        let mut state: State = self.0.clone();

        for _ in 0_usize..days {
            state.step();
        }

        state.total
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_lanternfish(80_usize));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_lanternfish(256_usize));
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(State::try_from(input)?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const LANTERNFISH_STR: &str = "3,4,3,1,2";

    lazy_static! {
        static ref SOLUTION: Solution = solution();
    }

    fn solution() -> Solution {
        Solution(State {
            counts: [
                0_usize, 1_usize, 1_usize, 2_usize, 1_usize, 0_usize, 0_usize, 0_usize, 0_usize,
            ],
            offset: 0_usize,
            total: 5_usize,
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(LANTERNFISH_STR), Ok(solution()));
    }

    #[test]
    fn test_build_timers() {
        assert_eq!(SOLUTION.0.build_timers(), vec![1, 2, 3, 3, 4]);
    }

    #[test]
    fn test_step() {
        let mut all_timers: Vec<Vec<u8>> = vec![
            vec![2, 3, 2, 0, 1],
            vec![1, 2, 1, 6, 0, 8],
            vec![0, 1, 0, 5, 6, 7, 8],
            vec![6, 0, 6, 4, 5, 6, 7, 8, 8],
            vec![5, 6, 5, 3, 4, 5, 6, 7, 7, 8],
            vec![4, 5, 4, 2, 3, 4, 5, 6, 6, 7],
            vec![3, 4, 3, 1, 2, 3, 4, 5, 5, 6],
            vec![2, 3, 2, 0, 1, 2, 3, 4, 4, 5],
            vec![1, 2, 1, 6, 0, 1, 2, 3, 3, 4, 8],
            vec![0, 1, 0, 5, 6, 0, 1, 2, 2, 3, 7, 8],
            vec![6, 0, 6, 4, 5, 6, 0, 1, 1, 2, 6, 7, 8, 8, 8],
            vec![5, 6, 5, 3, 4, 5, 6, 0, 0, 1, 5, 6, 7, 7, 7, 8, 8],
            vec![4, 5, 4, 2, 3, 4, 5, 6, 6, 0, 4, 5, 6, 6, 6, 7, 7, 8, 8],
            vec![3, 4, 3, 1, 2, 3, 4, 5, 5, 6, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8],
            vec![2, 3, 2, 0, 1, 2, 3, 4, 4, 5, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7],
            vec![
                1, 2, 1, 6, 0, 1, 2, 3, 3, 4, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 8,
            ],
            vec![
                0, 1, 0, 5, 6, 0, 1, 2, 2, 3, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 7, 8,
            ],
            vec![
                6, 0, 6, 4, 5, 6, 0, 1, 1, 2, 6, 0, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 8, 8, 8,
            ],
        ];
        let mut state = SOLUTION.0.clone();

        for timers in all_timers.iter_mut() {
            timers.sort();
            state.step();

            assert_eq!(*timers, state.build_timers());
        }
    }

    #[test]
    fn test_count_lanternfish() {
        assert_eq!(SOLUTION.count_lanternfish(18_usize), 26_usize);
        assert_eq!(SOLUTION.count_lanternfish(80_usize), 5934_usize);
        assert_eq!(SOLUTION.count_lanternfish(256_usize), 26_984_457_539_usize);
    }
}
