use {
    crate::*,
    nom::{
        character::complete::space1,
        combinator::{map, opt},
        error::Error,
        multi::many_m_n,
        sequence::terminated,
        Err, IResult,
    },
    std::collections::HashMap,
};

#[cfg(test)]
use std::fmt::{Debug, Formatter, Result as FmtResult};

/* --- Day 6: Memory Reallocation ---

A debugger program here is having an issue: it is trying to repair a memory reallocation routine, but it keeps getting stuck in an infinite loop.

In this area, there are sixteen memory banks; each memory bank can hold any number of blocks. The goal of the reallocation routine is to balance the blocks between the memory banks.

The reallocation routine operates in cycles. In each cycle, it finds the memory bank with the most blocks (ties won by the lowest-numbered memory bank) and redistributes those blocks among the banks. To do this, it removes all of the blocks from the selected bank, then moves to the next (by index) memory bank and inserts one of the blocks. It continues doing this until it runs out of blocks; if it reaches the last memory bank, it wraps around to the first one.

The debugger would like to know how many redistributions can be done before a blocks-in-banks configuration is produced that has been seen before.

For example, imagine a scenario with only four memory banks:

    The banks start with 0, 2, 7, and 0 blocks. The third bank has the most blocks, so it is chosen for redistribution.
    Starting with the next bank (the fourth bank) and then continuing to the first bank, the second bank, and so on, the 7 blocks are spread out over the memory banks. The fourth, first, and second banks get two blocks each, and the third bank gets one back. The final result looks like this: 2 4 1 2.
    Next, the second bank is chosen because it contains the most blocks (four). Because there are four memory banks, each gets one block. The result is: 3 1 2 3.
    Now, there is a tie between the first and fourth memory banks, both of which have three blocks. The first bank wins the tie, and its three blocks are distributed evenly over the other three banks, leaving it with none: 0 2 3 4.
    The fourth bank is chosen, and its four blocks are distributed such that each of the four banks receives one: 1 3 4 1.
    The third bank is chosen, and the same thing happens: 2 4 1 2.

At this point, we've reached a state we've seen before: 2 4 1 2 was already seen. The infinite loop is detected after the fifth block redistribution cycle, and so the answer in this example is 5.

Given the initial block counts in your puzzle input, how many redistribution cycles must be completed before a configuration is produced that has been seen before?

--- Part Two ---

Out of curiosity, the debugger would also like to know the size of the loop: starting from a state that has already been seen, how many block redistribution cycles must be performed before that same state is seen again?

In the example above, 2 4 1 2 is seen again after four cycles, and so the answer in that example would be 4.

How many cycles are in the infinite loop that arises from the configuration in your puzzle input? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
struct MemoryBanksState([u8; Self::LEN]);

impl MemoryBanksState {
    const LEN: usize = 16_usize;

    fn cycle(&mut self, len: usize) {
        let max_index: usize = self.0[..len]
            .iter()
            .enumerate()
            .rev()
            .max_by_key(|(_, blocks)| **blocks)
            .unwrap()
            .0;
        let blocks: &mut u8 = &mut self.0[max_index];
        let blocks_value: u8 = *blocks;

        *blocks = 0_u8;

        let start_blocks_index: usize = (max_index + 1_usize) % len;
        let additional_blocks: u8 = blocks_value / len as u8;
        let end_extra_redistribution_index: usize = blocks_value as usize % len;

        for redistribution_index in 0_usize..len {
            let blocks_index: usize = (start_blocks_index + redistribution_index) % len;

            self.0[blocks_index] +=
                additional_blocks + (redistribution_index < end_extra_redistribution_index) as u8;
        }
    }
}

#[cfg_attr(test, derive(PartialEq))]
pub struct Solution {
    state: MemoryBanksState,
    len: usize,
}

struct CycleData {
    steps_until_cycle_detection: usize,
    cycle_period: usize,
}

impl Solution {
    fn detect_cycle(&self) -> CycleData {
        let mut visited_states: HashMap<MemoryBanksState, usize> = HashMap::new();
        let mut state: MemoryBanksState = self.state;

        while visited_states.get(&state).is_none() {
            visited_states.insert(state, visited_states.len());
            state.cycle(self.len);
        }

        let prev_steps: usize = *visited_states.get(&state).unwrap();
        let curr_steps: usize = visited_states.len();

        CycleData {
            steps_until_cycle_detection: curr_steps,
            cycle_period: curr_steps - prev_steps,
        }
    }

    fn steps_until_cycle_detection(&self) -> usize {
        self.detect_cycle().steps_until_cycle_detection
    }

    fn cycle_period(&self) -> usize {
        self.detect_cycle().cycle_period
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut state: MemoryBanksState = MemoryBanksState::default();
        let mut len: usize = 0_usize;

        let input: &str = many_m_n(
            1_usize,
            MemoryBanksState::LEN,
            map(terminated(parse_integer, opt(space1)), |blocks| {
                state.0[len] = blocks;
                len += 1_usize;
            }),
        )(input)?
        .0;

        Ok((input, Self { state, len }))
    }
}

impl RunQuestions for Solution {
    /// Naming is a bit weird here. They call the individual steps "cycles", but what we're
    /// ultimately searching for is a cycle in the graph theory sense. Having 16 as a max size means
    /// it's not that bad to just keep track of an array of `u8`s for each block count, which is
    /// nice. If `max_by_key` didn't grab the last equally maximal and instead grabbed the first, I
    /// wouldn't need to use the `rev`, but it's not the end of the world.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.steps_until_cycle_detection());
    }

    /// I originally called the first function `cycle_cycle_period` because I erroneously assumed
    /// that it found its way back to the initial state. Adjusting things wasn't that tough, just
    /// using a hash map instead of a hash set.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.cycle_period());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
impl Debug for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_tuple("Solution")
            .field(&&self.state.0[..self.len])
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["0 2 7 0"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                state: MemoryBanksState([
                    0_u8, 2_u8, 7_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8,
                    0_u8, 0_u8, 0_u8,
                ]),
                len: 4_usize,
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
    fn test_steps_until_cycle_detection() {
        for (index, steps_until_cycle_detection) in [5_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).steps_until_cycle_detection(),
                steps_until_cycle_detection
            );
        }
    }

    #[test]
    fn test_cycle_period() {
        for (index, cycle_period) in [4_usize].into_iter().enumerate() {
            assert_eq!(solution(index).cycle_period(), cycle_period);
        }
    }
}
