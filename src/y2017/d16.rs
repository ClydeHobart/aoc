use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::one_of,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::{mem::swap, str::from_utf8},
};

#[cfg(test)]
use rand::prelude::*;

/* --- Day 16: Permutation Promenade ---

You come upon a very unusual sight; a group of programs here appear to be dancing.

There are sixteen programs in total, named a through p. They start by standing in a line: a stands in position 0, b stands in position 1, and so on until p, which stands in position 15.

The programs' dance consists of a sequence of dance moves:

    Spin, written sX, makes X programs move from the end to the front, but maintain their order otherwise. (For example, s3 on abcde produces cdeab).
    Exchange, written xA/B, makes the programs at positions A and B swap places.
    Partner, written pA/B, makes the programs named A and B swap places.

For example, with only five programs standing in a line (abcde), they could do the following dance:

    s1, a spin of size 1: eabcd.
    x3/4, swapping the last two programs: eabdc.
    pe/b, swapping programs e and b: baedc.

After finishing their dance, the programs end up in order baedc.

You watch the dance for a while and record their dance moves (your puzzle input). In what order are the programs standing after their dance? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum DanceMove {
    Spin { len: u8 },
    Exchange { index_a: u8, index_b: u8 },
    Partner { program_a: u8, program_b: u8 },
}

impl DanceMove {
    fn parse_program<'i>(input: &'i str) -> IResult<&'i str, u8> {
        map(one_of(LetterProgramArray::PROGRAMS), |c| c as u8)(input)
    }

    fn try_swap_indices(index_a: usize, index_b: usize, programs: &mut [u8]) -> bool {
        let [min_index, max_index]: [usize; 2_usize] = [index_a.min(index_b), index_a.max(index_b)];

        min_index != max_index && max_index < programs.len() && {
            let (min_slice, max_slice): (&mut [u8], &mut [u8]) = programs.split_at_mut(max_index);

            swap(&mut min_slice[min_index], &mut max_slice[0_usize]);

            true
        }
    }

    fn try_perform(self, programs: &mut [u8]) -> bool {
        match self {
            Self::Spin { len } => {
                let len: usize = len as usize;

                programs.len() >= len && {
                    programs.rotate_right(len);

                    true
                }
            }
            Self::Exchange { index_a, index_b } => {
                Self::try_swap_indices(index_a as usize, index_b as usize, programs)
            }
            Self::Partner {
                program_a,
                program_b,
            } => programs
                .iter()
                .position(|program| *program == program_a)
                .zip(programs.iter().position(|program| *program == program_b))
                .map_or(false, |(index_a, index_b)| {
                    Self::try_swap_indices(index_a, index_b, programs)
                }),
        }
    }

    fn is_partner(self) -> bool {
        matches!(self, Self::Partner { .. })
    }

    #[cfg(test)]
    fn random_index(thread_rng: &mut ThreadRng) -> u8 {
        thread_rng.gen_range(0_u8..Solution::MAX_PROGRAMS_LEN as u8)
    }

    #[cfg(test)]
    fn random_index_pair(thread_rng: &mut ThreadRng) -> (u8, u8) {
        let index_a: u8 = Self::random_index(thread_rng);
        let mut index_b: u8 = thread_rng.gen_range(0_u8..Solution::MAX_PROGRAMS_LEN as u8 - 1_u8);

        index_b += (index_b == index_a) as u8;

        (index_a, index_b)
    }

    #[cfg(test)]
    fn random(thread_rng: &mut ThreadRng) -> Self {
        match thread_rng.gen_range(0_u32..3_u32) {
            0_u32 => Self::Spin {
                len: Self::random_index(thread_rng),
            },
            1_u32 => {
                let (index_a, index_b): (u8, u8) = Self::random_index_pair(thread_rng);

                Self::Exchange { index_a, index_b }
            }
            2_u32 => {
                let (index_a, index_b): (u8, u8) = Self::random_index_pair(thread_rng);
                let programs: &[u8] = LetterProgramArray::PROGRAMS.as_bytes();

                Self::Partner {
                    program_a: programs[index_a as usize],
                    program_b: programs[index_b as usize],
                }
            }
            _ => unreachable!(),
        }
    }
}

impl Parse for DanceMove {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(preceded(tag("s"), parse_integer), |len| Self::Spin { len }),
            map(
                tuple((tag("x"), parse_integer, tag("/"), parse_integer)),
                |(_, index_a, _, index_b)| Self::Exchange { index_a, index_b },
            ),
            map(
                tuple((tag("p"), Self::parse_program, tag("/"), Self::parse_program)),
                |(_, program_a, _, program_b)| Self::Partner {
                    program_a,
                    program_b,
                },
            ),
        ))(input)
    }
}

type ProgramString = StaticString<{ Solution::MAX_PROGRAMS_LEN }>;
type ProgramArray = [u8; Solution::MAX_PROGRAMS_LEN];

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct LetterProgramArray(ProgramArray);

impl LetterProgramArray {
    const PROGRAMS: &'static str = "abcdefghijklmnop";

    fn try_program_string(&self) -> Option<ProgramString> {
        ProgramString::try_from(from_utf8(&self.0).ok()?).ok()
    }
}

impl Default for LetterProgramArray {
    fn default() -> Self {
        let mut programs: ProgramArray = ProgramArray::default();

        programs.copy_from_slice(Self::PROGRAMS.as_bytes());

        Self(programs)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct IndexProgramArray(ProgramArray);

impl IndexProgramArray {
    const PROGRAMS: ProgramArray = Self::programs();

    const fn programs() -> ProgramArray {
        let mut programs: ProgramArray = [0_u8; Solution::MAX_PROGRAMS_LEN];
        let mut index: usize = 0_usize;

        while index < programs.len() {
            programs[index] = index as u8;
            index += 1_usize;
        }

        programs
    }

    fn apply_position_transformation(&self, state: &mut Self) {
        let mut next_state: Self = Self(ProgramArray::default());

        for (next_state_index, state_index) in self.0.into_iter().enumerate() {
            next_state.0[next_state_index] = state.0[state_index as usize];
        }

        *state = next_state;
    }

    fn apply_program_transformation(&self, state: &mut Self) {
        let mut next_state: Self = Self(ProgramArray::default());

        for (next_state_index, self_index) in state.0.into_iter().enumerate() {
            next_state.0[next_state_index] = self.0[self_index as usize];
        }

        *state = next_state;
    }
}

impl Default for IndexProgramArray {
    fn default() -> Self {
        Self(Self::PROGRAMS)
    }
}

impl From<IndexProgramArray> for LetterProgramArray {
    fn from(value: IndexProgramArray) -> Self {
        let mut programs: ProgramArray = value.0;

        for program in &mut programs {
            *program += b'a';
        }

        Self(programs)
    }
}

impl From<LetterProgramArray> for IndexProgramArray {
    fn from(value: LetterProgramArray) -> Self {
        let mut programs: ProgramArray = value.0;

        for program in &mut programs {
            *program -= b'a';
        }

        Self(programs)
    }
}

#[derive(Clone)]
struct DecomposedTransformations {
    position_transformation: IndexProgramArray,
    program_transformation: IndexProgramArray,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<DanceMove>);

impl Solution {
    const MAX_PROGRAMS_LEN: usize = 16_usize;

    fn try_perform_dance(&self, repetitions: usize) -> Option<LetterProgramArray> {
        let mut letter_programs: LetterProgramArray = LetterProgramArray::default();

        for _ in 0_usize..repetitions {
            for dance_move in &self.0 {
                dance_move
                    .try_perform(&mut letter_programs.0)
                    .then_some(())?;
            }
        }

        Some(letter_programs)
    }

    fn try_one_dance_string(&self) -> Option<ProgramString> {
        self.try_perform_dance(1_usize)?.try_program_string()
    }

    fn try_decompose(&self) -> Option<DecomposedTransformations> {
        let mut position_transformation: IndexProgramArray = IndexProgramArray::default();
        let mut program_transformation: LetterProgramArray = LetterProgramArray::default();

        for dance_move in &self.0 {
            dance_move
                .try_perform(if dance_move.is_partner() {
                    &mut program_transformation.0
                } else {
                    &mut position_transformation.0
                })
                .then_some(())?;
        }

        Some(DecomposedTransformations {
            position_transformation,
            program_transformation: program_transformation.into(),
        })
    }

    fn apply_transformation<A: Copy + Fn(&IndexProgramArray, &mut IndexProgramArray)>(
        transformation: &IndexProgramArray,
        state: &mut IndexProgramArray,
        repetitions: usize,
        apply_transformation: A,
    ) {
        let mut compound_transformation: IndexProgramArray = IndexProgramArray::default();
        let mut applied_repetitions: usize = 0_usize;

        while applied_repetitions < repetitions {
            apply_transformation(transformation, &mut compound_transformation);

            applied_repetitions += 1_usize;

            if compound_transformation == IndexProgramArray::default() {
                // We found a cycle. Skip forward to the last incomplete cycle.
                applied_repetitions +=
                    (repetitions - applied_repetitions) / applied_repetitions * applied_repetitions;
            }
        }

        apply_transformation(&compound_transformation, state);
    }

    fn repeat_transformation<A: Copy + Fn(&IndexProgramArray, &mut IndexProgramArray)>(
        transformation: &mut IndexProgramArray,
        repetitions: usize,
        apply_transformation: A,
    ) {
        let mut state: IndexProgramArray = IndexProgramArray::default();

        Self::apply_transformation(
            transformation,
            &mut state,
            repetitions,
            apply_transformation,
        );

        *transformation = state;
    }

    fn repeat_decomposed_transformations(
        decomposed_transformations: &mut DecomposedTransformations,
        repetitions: usize,
    ) {
        Self::repeat_transformation(
            &mut decomposed_transformations.position_transformation,
            repetitions,
            IndexProgramArray::apply_position_transformation,
        );
        Self::repeat_transformation(
            &mut decomposed_transformations.program_transformation,
            repetitions,
            IndexProgramArray::apply_program_transformation,
        );
    }

    fn try_one_billion_dances_string(&self) -> Option<ProgramString> {
        let mut decomposed_transformations: DecomposedTransformations = self.try_decompose()?;

        Self::repeat_decomposed_transformations(
            &mut decomposed_transformations,
            1_000_000_000_usize,
        );

        let mut index_programs: IndexProgramArray = IndexProgramArray::default();

        Solution::apply_transformation(
            &decomposed_transformations.position_transformation,
            &mut index_programs,
            1_usize,
            IndexProgramArray::apply_position_transformation,
        );
        Solution::apply_transformation(
            &decomposed_transformations.program_transformation,
            &mut index_programs,
            1_usize,
            IndexProgramArray::apply_program_transformation,
        );

        LetterProgramArray::from(index_programs).try_program_string()
    }

    #[cfg(test)]
    fn init_letter_programs(letter_programs: &mut [u8]) {
        let letter_programs_len: usize = letter_programs.len();

        assert!(letter_programs_len <= LetterProgramArray::PROGRAMS.len());

        letter_programs
            .copy_from_slice(&LetterProgramArray::PROGRAMS.as_bytes()[..letter_programs_len])
    }

    #[cfg(test)]
    fn randomize(&mut self, dance_moves_len: usize, thread_rng: &mut ThreadRng) {
        self.0.clear();
        self.0
            .extend((0_usize..dance_moves_len).map(|_| DanceMove::random(thread_rng)))
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(DanceMove::parse, opt(tag(",")))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Smells of https://adventofcode.com/2016/day/21
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self
            .try_one_dance_string()
            .as_ref()
            .map(StaticString::as_str));
    }

    /// I spent way too much time on this. My initial thought was run it 10x to get a 10x
    /// transformation, then run the 10x transformation 10x to get a 100x transformation, but that
    /// didn't work initially, because the partner dance move isn't position-based. I then broke it
    /// down into position transformation and program transformation (just the partner moves), and
    /// when fleshing this out, I also attempted to incorporate support for arbitrarily large
    /// numbers by breaking things down into prime factorization, thinking the 10x method would work
    /// part-wise for prime factors, but then I realized that it's really closer to addition than
    /// multiplication: transformation x applied to identity state is just x. Applying x again is
    /// 2x, applying again is 3x. So combining factor transformations was not doing what I wanted:
    /// 6x = (2 * 3)x, but I was doing 2x + 3x. I then realized I was overthiking this, and that
    /// cycle detection was likely the way to go, since it's a closed loop (no, I don't have a proof
    /// for this, but intuition tells me it's true).
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self
            .try_one_billion_dances_string()
            .as_ref()
            .map(StaticString::as_str));
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

    const SOLUTION_STRS: &'static [&'static str] = &["s1,x3/4,pe/b"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                DanceMove::Spin { len: 1_u8 },
                DanceMove::Exchange {
                    index_a: 3_u8,
                    index_b: 4_u8,
                },
                DanceMove::Partner {
                    program_a: b'e',
                    program_b: b'b',
                },
            ])]
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
    fn test_try_perform() {
        let mut programs: [u8; 5_usize] = Default::default();

        Solution::init_letter_programs(&mut programs);

        assert_eq!(programs, *b"abcde");

        for (dance_move, expected_programs) in solution(0_usize)
            .0
            .iter()
            .zip([b"eabcd", b"eabdc", b"baedc"])
        {
            assert!(dance_move.try_perform(&mut programs));
            assert_eq!(programs, *expected_programs);
        }
    }

    #[test]
    fn test_decomposed_transformations() {
        let mut local_solution: Solution = Solution(Vec::new());
        let mut thread_rng: ThreadRng = thread_rng();

        for repetitions in
            (1_usize..=10_usize).chain([60_usize, 100_usize, 210_usize, 900_usize, 1000_usize])
        {
            for _ in 0_usize..10_usize {
                if repetitions == 1_usize || repetitions == 6_usize {
                    local_solution.0.clear();
                    local_solution.0.extend(solution(0_usize).0.iter().copied());
                } else {
                    local_solution.randomize(10000_usize / repetitions, &mut thread_rng);
                }

                let expected_index_programs: IndexProgramArray = local_solution
                    .try_perform_dance(repetitions)
                    .unwrap()
                    .into();
                let mut decomposed_transformations: DecomposedTransformations =
                    local_solution.try_decompose().unwrap();

                Solution::repeat_decomposed_transformations(
                    &mut decomposed_transformations,
                    repetitions,
                );

                {
                    let mut index_programs: IndexProgramArray = IndexProgramArray::default();

                    Solution::apply_transformation(
                        &decomposed_transformations.position_transformation,
                        &mut index_programs,
                        1_usize,
                        IndexProgramArray::apply_position_transformation,
                    );
                    Solution::apply_transformation(
                        &decomposed_transformations.program_transformation,
                        &mut index_programs,
                        1_usize,
                        IndexProgramArray::apply_program_transformation,
                    );

                    assert_eq!(index_programs, expected_index_programs);
                }

                {
                    let mut index_programs: IndexProgramArray = IndexProgramArray::default();

                    Solution::apply_transformation(
                        &decomposed_transformations.program_transformation,
                        &mut index_programs,
                        1_usize,
                        IndexProgramArray::apply_program_transformation,
                    );
                    Solution::apply_transformation(
                        &decomposed_transformations.position_transformation,
                        &mut index_programs,
                        1_usize,
                        IndexProgramArray::apply_position_transformation,
                    );

                    assert_eq!(index_programs, expected_index_programs);
                }
            }
        }
    }
}
