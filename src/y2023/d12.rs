use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many1_count,
        sequence::terminated,
        Err, IResult,
    },
    std::{
        collections::{hash_map::DefaultHasher, HashMap},
        hash::{Hash, Hasher},
        ops::Range,
    },
};

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    enum SpringState {
        Operational = OPERATIONAL = b'.',
        Damaged = DAMAGED = b'#',
        Unknown = UNKNOWN = b'?',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct RowRanges {
    spring_state_range: Range<u32>,
    damaged_run_len_range: Range<u32>,
}

#[derive(Clone, Copy, Debug)]
struct SliceWithHash<'r, T> {
    slice: &'r [T],
    hash: u64,
}

impl<'r, T: Hash> SliceWithHash<'r, T> {
    fn new(slice: &'r [T]) -> Self {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        slice.hash(&mut hasher);

        Self {
            slice,
            hash: hasher.finish(),
        }
    }
}

#[derive(Debug)]
struct Row<'r> {
    spring_states: SliceWithHash<'r, SpringState>,
    damaged_run_lens: SliceWithHash<'r, u8>,
    committed_damaged_run_len: i8,
}

impl<'r> Row<'r> {
    fn new(
        spring_states: SliceWithHash<'r, SpringState>,
        damaged_run_lens: SliceWithHash<'r, u8>,
        committed_damaged_run_len: i8,
    ) -> Self {
        Self {
            spring_states,
            damaged_run_lens,
            committed_damaged_run_len,
        }
    }

    fn hash(&self) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        self.spring_states.hash.hash(&mut hasher);
        self.damaged_run_lens.hash.hash(&mut hasher);
        self.committed_damaged_run_len.hash(&mut hasher);

        hasher.finish()
    }

    fn next_index_relative_to_spring_state<F: Fn(&SpringState, &SpringState) -> bool>(
        &self,
        target_spring_state: SpringState,
        max_len: u8,
        f: F,
    ) -> usize {
        self.spring_states
            .slice
            .iter()
            .position(|spring_state| f(spring_state, &target_spring_state))
            .unwrap_or((max_len as usize).min(self.spring_states.slice.len()))
    }

    fn next_index_that_is_spring_state(
        &self,
        target_spring_state: SpringState,
        max_len: u8,
    ) -> usize {
        self.next_index_relative_to_spring_state(target_spring_state, max_len, SpringState::eq)
    }

    fn next_index_that_is_not_spring_state(
        &self,
        target_spring_state: SpringState,
        max_len: u8,
    ) -> usize {
        self.next_index_relative_to_spring_state(target_spring_state, max_len, SpringState::ne)
    }

    fn different_arrangement_count(&self, map: &mut HashMap<u64, usize>) -> usize {
        let hash: u64 = self.hash();

        if let Some(different_arrangement_count) = map.get(&hash) {
            *different_arrangement_count
        } else {
            let different_arrangement_count: usize = match (
                self.spring_states.slice.is_empty(),
                self.damaged_run_lens.slice.is_empty(),
            ) {
                (false, false) => match self.spring_states.slice.first().copied().unwrap() {
                    SpringState::Operational => {
                        if self.committed_damaged_run_len <= 0_i8 {
                            let spring_state_start: usize = self
                                .next_index_that_is_not_spring_state(
                                    SpringState::Operational,
                                    u8::MAX,
                                );

                            Self::new(
                                SliceWithHash::new(&self.spring_states.slice[spring_state_start..]),
                                self.damaged_run_lens,
                                0_i8,
                            )
                            .different_arrangement_count(map)
                        } else {
                            // This branch was expecting more damaged springs
                            0_usize
                        }
                    }
                    SpringState::Damaged => {
                        if self.committed_damaged_run_len < 0_i8 {
                            // This branch was expecting a break from damaged springs
                            0_usize
                        } else {
                            let expected_damaged_run_len: u8 =
                                if self.committed_damaged_run_len > 0_i8 {
                                    self.committed_damaged_run_len as u8
                                } else {
                                    self.damaged_run_lens.slice.first().copied().unwrap()
                                };
                            let real_damaged_run_len: u8 = self.next_index_that_is_not_spring_state(
                                SpringState::Damaged,
                                expected_damaged_run_len + 1_u8,
                            ) as u8;

                            if real_damaged_run_len > expected_damaged_run_len {
                                // The current run doesn't match what's expected
                                0_usize
                            } else {
                                if real_damaged_run_len == expected_damaged_run_len {
                                    // The expected damaged run is now complete
                                    Self::new(
                                        SliceWithHash::new(
                                            &self.spring_states.slice
                                                [real_damaged_run_len as usize..],
                                        ),
                                        SliceWithHash::new(&self.damaged_run_lens.slice[1_usize..]),
                                        -1_i8,
                                    )
                                } else {
                                    // Commit to a (potentially ongoing) damaged run
                                    Self::new(
                                        SliceWithHash::new(
                                            &self.spring_states.slice
                                                [real_damaged_run_len as usize..],
                                        ),
                                        self.damaged_run_lens,
                                        (expected_damaged_run_len - real_damaged_run_len) as i8,
                                    )
                                }
                                .different_arrangement_count(map)
                            }
                        }
                    }
                    SpringState::Unknown => {
                        if self.committed_damaged_run_len < 0_i8 {
                            // This branch is expecting a break from damaged springs
                            Self::new(
                                SliceWithHash::new(&self.spring_states.slice[1_usize..]),
                                self.damaged_run_lens,
                                0_i8,
                            )
                            .different_arrangement_count(map)
                        } else {
                            let expected_damaged_run_len: u8 =
                                if self.committed_damaged_run_len > 0_i8 {
                                    self.committed_damaged_run_len as u8
                                } else {
                                    self.damaged_run_lens.slice.first().copied().unwrap()
                                };
                            let unknown_run_len: u8 = self.next_index_that_is_not_spring_state(
                                SpringState::Unknown,
                                expected_damaged_run_len + 1_u8,
                            ) as u8;
                            let different_arrangement_count_from_acting: usize =
                                if expected_damaged_run_len <= unknown_run_len {
                                    // We can complete a run right here
                                    Self::new(
                                        SliceWithHash::new(
                                            &self.spring_states.slice
                                                [expected_damaged_run_len as usize..],
                                        ),
                                        SliceWithHash::new(&self.damaged_run_lens.slice[1_usize..]),
                                        -1_i8,
                                    )
                                } else {
                                    // Commit to a new damaged run
                                    Self::new(
                                        SliceWithHash::new(
                                            &self.spring_states.slice[unknown_run_len as usize..],
                                        ),
                                        self.damaged_run_lens,
                                        (expected_damaged_run_len - unknown_run_len) as i8,
                                    )
                                }
                                .different_arrangement_count(map);

                            let different_arrangement_count_from_waiting: usize =
                                if self.committed_damaged_run_len == 0_i8 {
                                    Self::new(
                                        SliceWithHash::new(&self.spring_states.slice[1_usize..]),
                                        self.damaged_run_lens,
                                        0_i8,
                                    )
                                    .different_arrangement_count(map)
                                } else {
                                    0_usize
                                };

                            different_arrangement_count_from_acting
                                + different_arrangement_count_from_waiting
                        }
                    }
                },
                (false, true) => {
                    if self.next_index_that_is_spring_state(SpringState::Damaged, u8::MAX)
                        == self.spring_states.slice.len()
                    {
                        // Interpret all Unknown as Operational
                        1_usize
                    } else {
                        // The remaining run cannot be expressed
                        0_usize
                    }
                }
                (true, false) => {
                    // The remaining damaged run length cannot be expressed
                    0_usize
                }
                (true, true) => {
                    // 0 choose 0 is 1
                    1_usize
                }
            };

            map.insert(hash, different_arrangement_count);

            different_arrangement_count
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    row_ranges: Vec<RowRanges>,
    spring_states: Vec<SpringState>,
    damaged_run_lens: Vec<u8>,
}

impl Solution {
    const UNFOLD_REPETITIONS: usize = 5_usize;

    fn iter_rows<'r>(&'r self) -> impl Iterator<Item = Row<'r>> {
        self.row_ranges.iter().map(|row_ranges| {
            Row::new(
                SliceWithHash::new(
                    &self.spring_states[row_ranges.spring_state_range.as_range_usize()],
                ),
                SliceWithHash::new(
                    &self.damaged_run_lens[row_ranges.damaged_run_len_range.as_range_usize()],
                ),
                0_i8,
            )
        })
    }

    fn iter_different_arrangement_counts(&self, verbose: bool) -> impl Iterator<Item = usize> + '_ {
        let mut map: HashMap<u64, usize> = HashMap::new();

        self.iter_rows().enumerate().map(move |(index, row)| {
            if verbose {
                dbg!(index);
            }

            row.different_arrangement_count(&mut map)
        })
    }

    fn different_arrangement_count_sum(&self, verbose: bool) -> usize {
        self.iter_different_arrangement_counts(verbose).sum()
    }

    fn unfold(&self) -> Self {
        let mut solution: Self = Self::default();

        for row in self.iter_rows() {
            let spring_states_start: u32 = solution.spring_states.len() as u32;

            solution.spring_states.extend(
                (0_usize..Self::UNFOLD_REPETITIONS).into_iter().flat_map(
                    |repetition_index: usize| {
                        if repetition_index != 0_usize {
                            Some(SpringState::Unknown)
                        } else {
                            None
                        }
                        .into_iter()
                        .chain(row.spring_states.slice.iter().copied())
                    },
                ),
            );

            let spring_states_end: u32 = solution.spring_states.len() as u32;
            let damaged_run_lens_start: u32 = solution.damaged_run_lens.len() as u32;

            solution.damaged_run_lens.extend(
                (0_usize..Self::UNFOLD_REPETITIONS)
                    .into_iter()
                    .flat_map(|_| row.damaged_run_lens.slice.iter().copied()),
            );

            let damaged_run_lens_end: u32 = solution.damaged_run_lens.len() as u32;

            solution.row_ranges.push(RowRanges {
                spring_state_range: spring_states_start..spring_states_end,
                damaged_run_len_range: damaged_run_lens_start..damaged_run_lens_end,
            });
        }

        solution
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        let input: &str = many1_count(|input: &'i str| {
            let spring_states_start: u32 = solution.spring_states.len() as u32;

            let input: &str = terminated(
                many1_count(map(SpringState::parse, |spring_state: SpringState| {
                    solution.spring_states.push(spring_state);
                })),
                tag(" "),
            )(input)?
            .0;

            let spring_states_end: u32 = solution.spring_states.len() as u32;
            let damaged_run_len_start: u32 = solution.damaged_run_lens.len() as u32;

            let input: &str = terminated(
                many1_count(terminated(
                    map(parse_integer::<u8>, |damaged_run_len: u8| {
                        solution.damaged_run_lens.push(damaged_run_len);
                    }),
                    opt(tag(",")),
                )),
                opt(line_ending),
            )(input)?
            .0;

            let damaged_run_len_end: u32 = solution.damaged_run_lens.len() as u32;

            solution.row_ranges.push(RowRanges {
                spring_state_range: spring_states_start..spring_states_end,
                damaged_run_len_range: damaged_run_len_start..damaged_run_len_end,
            });

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.different_arrangement_count_sum(args.verbose));
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.unfold().different_arrangement_count_sum(args.verbose));
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

    const SOLUTION_STR: &'static str = "\
        ???.### 1,1,3\n\
        .??..??...?##. 1,1,3\n\
        ?#?#?#?#?#?#?#? 1,3,1,6\n\
        ????.#...#... 4,1,1\n\
        ????.######..#####. 1,6,5\n\
        ?###???????? 3,2,1\n";

    fn solution() -> &'static Solution {
        use SpringState::{Damaged as D, Operational as O, Unknown as U};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            row_ranges: vec![
                RowRanges {
                    spring_state_range: 0..7,
                    damaged_run_len_range: 0..3,
                },
                RowRanges {
                    spring_state_range: 7..21,
                    damaged_run_len_range: 3..6,
                },
                RowRanges {
                    spring_state_range: 21..36,
                    damaged_run_len_range: 6..10,
                },
                RowRanges {
                    spring_state_range: 36..49,
                    damaged_run_len_range: 10..13,
                },
                RowRanges {
                    spring_state_range: 49..68,
                    damaged_run_len_range: 13..16,
                },
                RowRanges {
                    spring_state_range: 68..80,
                    damaged_run_len_range: 16..19,
                },
            ],
            spring_states: vec![
                U, U, U, O, D, D, D, // Row 0
                O, U, U, O, O, U, U, O, O, O, U, D, D, O, // Row 1
                U, D, U, D, U, D, U, D, U, D, U, D, U, D, U, // Row 2
                U, U, U, U, O, D, O, O, O, D, O, O, O, // Row 3
                U, U, U, U, O, D, D, D, D, D, D, O, O, D, D, D, D, D, O, // Row 4
                U, D, D, D, U, U, U, U, U, U, U, U, // Row 5
            ],
            damaged_run_lens: vec![
                1, 1, 3, // Run 0
                1, 1, 3, // Run 1
                1, 3, 1, 6, // Run 2
                4, 1, 1, // Run 3
                1, 6, 5, // Run 4
                3, 2, 1, // Run 5
            ],
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_different_arrangement_count() {
        assert_eq!(
            solution()
                .iter_different_arrangement_counts(false)
                .collect::<Vec<usize>>(),
            vec![1_usize, 4_usize, 1_usize, 1_usize, 4_usize, 10_usize]
        );
    }

    #[test]
    fn test_different_arrangement_count_sum() {
        assert_eq!(solution().different_arrangement_count_sum(false), 21_usize);
    }

    #[test]
    fn test_unfold() {
        assert_eq!(
            solution()
                .unfold()
                .iter_different_arrangement_counts(false)
                .collect::<Vec<usize>>(),
            vec![
                1_usize,
                16384_usize,
                1_usize,
                16_usize,
                2500_usize,
                506250_usize
            ]
        );
        assert_eq!(
            solution().unfold().different_arrangement_count_sum(false),
            525152_usize
        );
    }
}
