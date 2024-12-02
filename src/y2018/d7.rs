use {
    crate::*,
    bitvec::prelude::*,
    derive_deref::{Deref, DerefMut},
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, satisfy},
        combinator::{map, map_opt, opt, success},
        error::Error,
        multi::many0_count,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    static_assertions::const_assert,
    std::{
        collections::VecDeque,
        fmt::{Debug, Formatter, Result as FmtResult},
        io::Write,
        ops::RangeInclusive,
    },
};

/* --- Day 7: The Sum of Its Parts ---

You find yourself standing on a snow-covered coastline; apparently, you landed a little off course. The region is too hilly to see the North Pole from here, but you do spot some Elves that seem to be trying to unpack something that washed ashore. It's quite cold out, so you decide to risk creating a paradox by asking them for directions.

"Oh, are you the search party?" Somehow, you can understand whatever Elves from the year 1018 speak; you assume it's Ancient Nordic Elvish. Could the device on your wrist also be a translator? "Those clothes don't look very warm; take this." They hand you a heavy coat.

"We do need to find our way back to the North Pole, but we have higher priorities at the moment. You see, believe it or not, this box contains something that will solve all of Santa's transportation problems - at least, that's what it looks like from the pictures in the instructions." It doesn't seem like they can read whatever language it's in, but you can: "Sleigh kit. Some assembly required."

"'Sleigh'? What a wonderful name! You must help us assemble this 'sleigh' at once!" They start excitedly pulling more parts out of the box.

The instructions specify a series of steps and requirements about which steps must be finished before others can begin (your puzzle input). Each step is designated by a single letter. For example, suppose you have the following instructions:

Step C must be finished before step A can begin.
Step C must be finished before step F can begin.
Step A must be finished before step B can begin.
Step A must be finished before step D can begin.
Step B must be finished before step E can begin.
Step D must be finished before step E can begin.
Step F must be finished before step E can begin.

Visually, these requirements look like this:

  -->A--->B--
 /    \      \
C      -->D----->E
 \           /
  ---->F-----

Your first goal is to determine the order in which the steps should be completed. If more than one step is ready, choose the step which is first alphabetically. In this example, the steps would be completed as follows:

    Only C is available, and so it is done first.
    Next, both A and F are available. A is first alphabetically, so it is done next.
    Then, even though F was available earlier, steps B and D are now also available, and B is the first alphabetically of the three.
    After that, only D and F are available. E is not available because only some of its prerequisites are complete. Therefore, D is completed next.
    F is the only choice, so it is done next.
    Finally, E is completed.

So, in this example, the correct order is CABDFE.

In what order should the steps in your instructions be completed? */

type StepIndexRawRaw = u8;
type StepIndexRaw = TableIndex<StepIndexRawRaw>;

#[derive(Clone, Copy, Deref, DerefMut, Default, Eq, Ord, PartialEq, PartialOrd)]
struct StepIndex(StepIndexRaw);

impl Debug for StepIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.opt()
            .map(|step_index| (step_index.get() as u8 + b'A') as char)
            .fmt(f)
    }
}

impl From<usize> for StepIndex {
    fn from(value: usize) -> Self {
        StepIndex(value.into())
    }
}

type StepId = char;

const STEP_ID_RANGE_START: StepId = 'A';
const STEP_ID_RANGE_END: StepId = 'Z';
const STEP_ID_RANGE: RangeInclusive<StepId> = STEP_ID_RANGE_START..=STEP_ID_RANGE_END;
const STEP_ID_RANGE_LEN: usize =
    STEP_ID_RANGE_END as usize - STEP_ID_RANGE_START as usize + 1_usize;

const_assert!(STEP_ID_RANGE_LEN <= u32::BITS as usize);

type StepBitArrayRaw = BitArr!(for STEP_ID_RANGE_LEN, in u32);

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Copy, Deref, DerefMut, Default)]
struct StepBitArray(StepBitArrayRaw);

impl Debug for StepBitArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut debug_list = f.debug_list();

        for step_index in self.0.iter_ones() {
            debug_list.entry(&((step_index as u8 + b'A') as char));
        }

        debug_list.finish()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Default)]
struct StepData {
    steps_before: StepBitArray,
    steps_after: StepBitArray,
}

type Step = TableElement<StepId, StepData>;
type StepTable = Table<StepId, StepData, StepIndexRawRaw>;

struct Requirement<S = StepIndexRaw> {
    step_before: S,
    step_after: S,
}

impl Requirement<StepId> {
    fn parse_step_id<'i>(input: &'i str) -> IResult<&'i str, StepId> {
        satisfy(|c| STEP_ID_RANGE.contains(&c))(input)
    }
}

impl Parse for Requirement<StepId> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("Step "),
                Self::parse_step_id,
                tag(" must be finished before step "),
                Self::parse_step_id,
                tag(" can begin."),
            )),
            |(_, step_before, _, step_after, _)| Self {
                step_before,
                step_after,
            },
        )(input)
    }
}

struct StepOrderComputer<'s> {
    solution: &'s Solution,
    disabled_step_data_list: Vec<StepData>,
}

impl<'s> StepOrderComputer<'s> {
    fn new(solution: &'s Solution) -> Self {
        let disabled_step_data_list: Vec<StepData> =
            vec![StepData::default(); solution.0.as_slice().len()];

        Self {
            solution,
            disabled_step_data_list,
        }
    }

    fn iter_step_indices(&self) -> impl Iterator<Item = StepIndex> {
        (0_usize..self.disabled_step_data_list.len()).map(StepIndex::from)
    }

    fn enabled_step_bit_array(
        base_step_bit_array: StepBitArray,
        disabled_step_bit_array: StepBitArray,
    ) -> StepBitArray {
        StepBitArray(base_step_bit_array.0 & !disabled_step_bit_array.0)
    }

    fn get_enabled_step_data(&self, step_index: StepIndex) -> StepData {
        let base_step_data: StepData = self.solution.0.as_slice()[step_index.get()].data;
        let disabled_step_data: StepData = self.disabled_step_data_list[step_index.get()];

        StepData {
            steps_before: Self::enabled_step_bit_array(
                base_step_data.steps_before,
                disabled_step_data.steps_before,
            ),
            steps_after: Self::enabled_step_bit_array(
                base_step_data.steps_after,
                disabled_step_data.steps_after,
            ),
        }
    }
}

impl<'s> Kahn for StepOrderComputer<'s> {
    type Vertex = StepIndex;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>) {
        initial_set.clear();
        initial_set.extend(
            self.iter_step_indices()
                .filter(|step_index| !self.has_in_neighbors(&step_index)),
        );
    }

    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(
            self.get_enabled_step_data(*vertex)
                .steps_after
                .iter_ones()
                .map(StepIndex::from),
        )
    }

    fn remove_edge(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.disabled_step_data_list[from.get()]
            .steps_after
            .set(to.get(), true);
        self.disabled_step_data_list[to.get()]
            .steps_before
            .set(from.get(), true);
    }

    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool {
        self.get_enabled_step_data(*vertex).steps_before.any()
    }

    fn any_edges_exist(&self) -> bool {
        self.iter_step_indices()
            .any(|step_index| self.has_in_neighbors(&step_index))
    }

    fn reset(&mut self) {
        self.disabled_step_data_list.fill(StepData::default());
    }

    fn order_set(&self, set: &mut VecDeque<Self::Vertex>) {
        set.make_contiguous().sort();
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct WorkerState {
    step: StepIndex,
    secs_remaining: u8,
}

impl WorkerState {
    fn new(step: StepIndex, use_sec_discount: bool) -> Self {
        Self {
            step,
            secs_remaining: (!use_sec_discount) as u8 * 60_u8 + 1_u8 + step.get() as u8,
        }
    }

    fn has_work(self) -> bool {
        self.step.is_valid()
    }

    fn tick(&mut self, secs: u8) -> Option<StepIndex> {
        self.has_work()
            .then(|| {
                self.secs_remaining -= secs;

                (self.secs_remaining == 0_u8).then(|| {
                    let finished_step: StepIndex = self.step;

                    *self = Self::default();

                    finished_step
                })
            })
            .flatten()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(StepTable);

impl Solution {
    const WORKER_COUNT: usize = 5_usize;
    const USE_SEC_DISCOUNT: bool = false;

    fn parse_requirement_line<'i>(input: &'i str) -> IResult<&'i str, Requirement<StepId>> {
        terminated(Requirement::parse, opt(line_ending))(input)
    }

    fn try_compute_step_order(&self) -> Option<String> {
        let mut step_order_computer: StepOrderComputer = StepOrderComputer::new(self);

        step_order_computer.run().map(|step_indices| {
            step_indices
                .into_iter()
                .map(|step_index| self.0.as_slice()[step_index.get()].id)
                .collect()
        })
    }

    fn try_next_step(
        &self,
        finished_steps: StepBitArray,
        in_progress_steps: StepBitArray,
    ) -> Option<StepIndex> {
        let is_invalid_next_step: StepBitArray =
            StepBitArray(finished_steps.0 | in_progress_steps.0);

        (0_usize..self.0.as_slice().len())
            .filter(|&step_index| {
                !is_invalid_next_step[step_index] && {
                    let steps_before: StepBitArray =
                        self.0.as_slice()[step_index].data.steps_before;

                    steps_before.0 & finished_steps.0 == steps_before.0
                }
            })
            .next()
            .map(StepIndex::from)
    }

    fn try_compute_multi_worker_step_order_and_time<const W: usize>(
        &self,
        use_sec_discount: bool,
    ) -> Option<(String, u32)> {
        let mut worker_states: [WorkerState; W] = LargeArrayDefault::large_array_default();
        let mut finished_steps: StepBitArray = StepBitArray(StepBitArrayRaw::ZERO);
        let mut in_progress_steps: StepBitArray = StepBitArray(StepBitArrayRaw::ZERO);
        let mut step_order: String = String::with_capacity(self.0.as_slice().len());
        let mut time: u32 = 0_u32;

        while {
            worker_states.iter_mut().try_for_each(|worker_state| {
                if worker_state.has_work() {
                    Some(())
                } else {
                    self.try_next_step(finished_steps, in_progress_steps)
                        .map(|next_step| {
                            *worker_state = WorkerState::new(next_step, use_sec_discount);
                            in_progress_steps.set(next_step.get(), true);
                        })
                }
            });

            worker_states.into_iter().any(WorkerState::has_work)
        } {
            let secs: u8 = worker_states
                .into_iter()
                .filter_map(|worker_state| {
                    worker_state
                        .has_work()
                        .then_some(worker_state.secs_remaining)
                })
                .min()
                .unwrap();

            for worker_state in worker_states.iter_mut() {
                if let Some(finished_step) = worker_state.tick(secs) {
                    in_progress_steps.set(finished_step.get(), false);
                    finished_steps.set(finished_step.get(), true);
                    step_order.push(self.0.as_slice()[finished_step.get()].id);
                }
            }

            time += secs as u32;
        }

        (finished_steps.count_ones() == self.0.as_slice().len()).then_some((step_order, time))
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut step_table: StepTable = StepTable::new();

        many0_count(map(Self::parse_requirement_line, |requirement| {
            step_table.find_or_add_index(&requirement.step_before);
            step_table.find_or_add_index(&requirement.step_after);
        }))(input)?;

        map_opt(success(()), |_| {
            step_table.as_slice_mut().sort_by_key(|step| step.id);

            step_table
                .as_slice()
                .first()
                .filter(|first_step| {
                    first_step.id == STEP_ID_RANGE_START
                        && step_table.as_slice().last().unwrap().id as usize
                            - STEP_ID_RANGE_START as usize
                            + 1_usize
                            == step_table.as_slice().len()
                })
                .is_some()
                .then_some(())
        })(input)?;

        let input: &str = many0_count(map(Self::parse_requirement_line, |requirement| {
            let step_before: StepIndexRaw = step_table.find_index(&requirement.step_before);
            let step_after: StepIndexRaw = step_table.find_index(&requirement.step_after);

            assert!(step_before.is_valid());
            assert!(step_after.is_valid());

            let steps: &mut [Step] = step_table.as_slice_mut();

            steps[step_before.get()]
                .data
                .steps_after
                .set(step_after.get(), true);
            steps[step_after.get()]
                .data
                .steps_before
                .set(step_before.get(), true);
        }))(input)?
        .0;

        Ok((input, Self(step_table)))
    }
}

impl RunQuestions for Solution {
    /// Keeping traits of graph algorithms has proven really handy.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_step_order());
    }

    /// I tried to debug this with some custom natvis, but setting that up is harder than I
    /// expected. Custom `Debug` implementations suffice somewhat, but they require having log
    /// commands in.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(
            self.try_compute_multi_worker_step_order_and_time::<{ Self::WORKER_COUNT }>(
                Self::USE_SEC_DISCOUNT
            )
        );

        let mut file: std::fs::File = std::fs::File::create("output/y2018/d7.txt").unwrap();

        for step in self.0.as_slice() {
            writeln!(&mut file, "{}", step.id).ok();
        }

        for step in self.0.as_slice() {
            for step_after in step.data.steps_after.iter_ones() {
                writeln!(
                    &mut file,
                    "{} {}",
                    step.id,
                    self.0.as_slice()[step_after].id
                )
                .ok();
            }
        }
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        Step C must be finished before step A can begin.\n\
        Step C must be finished before step F can begin.\n\
        Step A must be finished before step B can begin.\n\
        Step A must be finished before step D can begin.\n\
        Step B must be finished before step E can begin.\n\
        Step D must be finished before step E can begin.\n\
        Step F must be finished before step E can begin.\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                vec![
                    Step {
                        id: 'A',
                        data: StepData {
                            steps_before: StepBitArray(bitarr![u32, Lsb0; 0, 0, 1]),
                            steps_after: StepBitArray(bitarr![u32, Lsb0; 0, 1, 0, 1]),
                        },
                    },
                    Step {
                        id: 'B',
                        data: StepData {
                            steps_before: StepBitArray(bitarr![u32, Lsb0; 1]),
                            steps_after: StepBitArray(bitarr![u32, Lsb0; 0, 0, 0, 0, 1]),
                        },
                    },
                    Step {
                        id: 'C',
                        data: StepData {
                            steps_before: StepBitArray(BitArray::ZERO),
                            steps_after: StepBitArray(bitarr![u32, Lsb0; 1, 0, 0, 0, 0, 1]),
                        },
                    },
                    Step {
                        id: 'D',
                        data: StepData {
                            steps_before: StepBitArray(bitarr![u32, Lsb0; 1]),
                            steps_after: StepBitArray(bitarr![u32, Lsb0; 0, 0, 0, 0, 1]),
                        },
                    },
                    Step {
                        id: 'E',
                        data: StepData {
                            steps_before: StepBitArray(bitarr![u32, Lsb0; 0, 1, 0, 1, 0, 1]),
                            steps_after: StepBitArray(BitArray::ZERO),
                        },
                    },
                    Step {
                        id: 'F',
                        data: StepData {
                            steps_before: StepBitArray(bitarr![u32, Lsb0; 0, 0, 1]),
                            steps_after: StepBitArray(bitarr![u32, Lsb0; 0, 0, 0, 0, 1]),
                        },
                    },
                ]
                .try_into()
                .unwrap(),
            )]
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
    fn test_try_compute_step_order() {
        for (index, computed_step_order) in [Some("CABDFE".into())].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_compute_step_order(),
                computed_step_order
            );
        }
    }

    #[test]
    fn try_compute_multi_worker_step_order_and_time() {
        const WORKER_COUNT: usize = 2_usize;
        const USE_SEC_DISCOUNT: bool = true;

        for (index, computed_multi_worker_step_order_and_time) in
            [Some(("CABFDE".into(), 15_u32))].into_iter().enumerate()
        {
            assert_eq!(
                solution(index)
                    .try_compute_multi_worker_step_order_and_time::<WORKER_COUNT>(USE_SEC_DISCOUNT),
                computed_multi_worker_step_order_and_time
            );
        }
    }

    #[test]
    fn test_input() {
        // rebuild pls
        let args: Args = Args::parse(module_path!()).unwrap().1;

        Solution::both(&args);
    }
}
