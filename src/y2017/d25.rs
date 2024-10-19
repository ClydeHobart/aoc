use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{anychar, line_ending, one_of},
        combinator::{cond, map, map_opt, map_res},
        error::Error,
        multi::many0,
        sequence::{delimited, separated_pair, terminated, tuple},
        Err, IResult,
    },
    std::{
        collections::VecDeque,
        fmt::{Debug, Formatter, Result as FmtResult},
    },
};

/* --- Day 25: The Halting Problem ---

Following the twisty passageways deeper and deeper into the CPU, you finally reach the core of the computer. Here, in the expansive central chamber, you find a grand apparatus that fills the entire room, suspended nanometers above your head.

You had always imagined CPUs to be noisy, chaotic places, bustling with activity. Instead, the room is quiet, motionless, and dark.

Suddenly, you and the CPU's garbage collector startle each other. "It's not often we get many visitors here!", he says. You inquire about the stopped machinery.

"It stopped milliseconds ago; not sure why. I'm a garbage collector, not a doctor." You ask what the machine is for.

"Programs these days, don't know their origins. That's the Turing machine! It's what makes the whole computer work." You try to explain that Turing machines are merely models of computation, but he cuts you off. "No, see, that's just what they want you to think. Ultimately, inside every CPU, there's a Turing machine driving the whole thing! Too bad this one's broken. We're doomed!"

You ask how you can help. "Well, unfortunately, the only way to get the computer running again would be to create a whole new Turing machine from scratch, but there's no way you can-" He notices the look on your face, gives you a curious glance, shrugs, and goes back to sweeping the floor.

You find the Turing machine blueprints (your puzzle input) on a tablet in a nearby pile of debris. Looking back up at the broken Turing machine above, you can start to identify its parts:

    A tape which contains 0 repeated infinitely to the left and right.
    A cursor, which can move left or right along the tape and read or write values at its current position.
    A set of states, each containing rules about what to do based on the current value under the cursor.

Each slot on the tape has two possible values: 0 (the starting value for all slots) and 1. Based on whether the cursor is pointing at a 0 or a 1, the current state says what value to write at the current position of the cursor, whether to move the cursor left or right one slot, and which state to use next.

For example, suppose you found the following blueprint:

Begin in state A.
Perform a diagnostic checksum after 6 steps.

In state A:
  If the current value is 0:
    - Write the value 1.
    - Move one slot to the right.
    - Continue with state B.
  If the current value is 1:
    - Write the value 0.
    - Move one slot to the left.
    - Continue with state B.

In state B:
  If the current value is 0:
    - Write the value 1.
    - Move one slot to the left.
    - Continue with state A.
  If the current value is 1:
    - Write the value 1.
    - Move one slot to the right.
    - Continue with state A.

Running it until the number of steps required to take the listed diagnostic checksum would result in the following tape configurations (with the cursor marked in square brackets):

... 0  0  0 [0] 0  0 ... (before any steps; about to run state A)
... 0  0  0  1 [0] 0 ... (after 1 step;     about to run state B)
... 0  0  0 [1] 1  0 ... (after 2 steps;    about to run state A)
... 0  0 [0] 0  1  0 ... (after 3 steps;    about to run state B)
... 0 [0] 1  0  1  0 ... (after 4 steps;    about to run state A)
... 0  1 [1] 0  1  0 ... (after 5 steps;    about to run state B)
... 0  1  1 [0] 1  0 ... (after 6 steps;    about to run state A)

The CPU can confirm that the Turing machine is working by taking a diagnostic checksum after a specific number of steps (given in the blueprint). Once the specified number of steps have been executed, the Turing machine should pause; once it does, count the number of times 1 appears on the tape. In the above example, the diagnostic checksum is 3.

Recreate the Turing machine and save the computer! What is the diagnostic checksum it produces once it's working again? */

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct StateId(u8);

impl StateId {
    const OFFSET: u8 = b'A';
    const INVALID: Self = Self(u8::MAX);
}

impl Debug for StateId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        char::from(*self).fmt(f)
    }
}

impl Default for StateId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl From<StateId> for char {
    fn from(value: StateId) -> Self {
        if value == StateId::INVALID {
            '*'
        } else {
            (value.0 + StateId::OFFSET) as char
        }
    }
}

impl TryFrom<char> for StateId {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        value
            .is_ascii_uppercase()
            .then(|| Self(value as u8 - Self::OFFSET))
            .ok_or(())
    }
}

impl Parse for StateId {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_res(anychar, Self::try_from)(input)
    }
}

fn parse_bool<'i>(input: &'i str) -> IResult<&'i str, bool> {
    map(one_of("01"), |c| c == '1')(input)
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Default)]
struct StateCase<S = StateIndex> {
    write_value: bool,
    move_right: bool,
    next_state: S,
}

impl StateCase {
    fn try_from_state_id_state_case(
        StateCase {
            write_value,
            move_right,
            next_state,
        }: StateCase<StateId>,
        states: &StateTable,
    ) -> Option<Self> {
        Some(Self {
            write_value,
            move_right,
            next_state: states.find_index(&next_state).opt()?,
        })
    }
}

impl<S: Parse> StateCase<S> {
    fn parse_curr_value<'i>(input: &'i str) -> IResult<&'i str, bool> {
        delimited(tag("  If the current value is "), parse_bool, tag(":"))(input)
    }

    fn parse_write_value<'i>(input: &'i str) -> IResult<&'i str, bool> {
        delimited(tag("    - Write the value "), parse_bool, tag("."))(input)
    }

    fn parse_move_right<'i>(input: &'i str) -> IResult<&'i str, bool> {
        delimited(
            tag("    - Move one slot to the "),
            alt((map(tag("left"), |_| false), map(tag("right"), |_| true))),
            tag("."),
        )(input)
    }

    fn parse_next_state<'i>(input: &'i str) -> IResult<&'i str, S> {
        delimited(tag("    - Continue with state "), S::parse, tag("."))(input)
    }

    fn parse_curr_value_and_state_case<'i>(input: &'i str) -> IResult<&'i str, (bool, Self)> {
        separated_pair(Self::parse_curr_value, line_ending, Self::parse)(input)
    }
}

impl<S: Parse> Parse for StateCase<S> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                Self::parse_write_value,
                line_ending,
                Self::parse_move_right,
                line_ending,
                Self::parse_next_state,
            )),
            |(write_value, _, move_right, _, next_state)| Self {
                write_value,
                move_right,
                next_state,
            },
        )(input)
    }
}

const STATE_CASES_LEN: usize = 2_usize;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct State<S = StateIndex> {
    cases: [StateCase<S>; STATE_CASES_LEN],
}

impl State {
    fn try_from_state_id_state(
        State {
            cases: [case_0, case_1],
        }: State<StateId>,
        states: &StateTable,
    ) -> Option<Self> {
        Some(Self {
            cases: [
                StateCase::try_from_state_id_state_case(case_0, states)?,
                StateCase::try_from_state_id_state_case(case_1, states)?,
            ],
        })
    }
}

impl State<StateId> {
    fn parse_id<'i>(input: &'i str) -> IResult<&'i str, StateId> {
        delimited(tag("In state "), StateId::parse, tag(":"))(input)
    }

    fn parse_id_and_state<'i>(input: &'i str) -> IResult<&'i str, (StateId, Self)> {
        separated_pair(Self::parse_id, line_ending, Self::parse)(input)
    }
}

impl<S: Parse + Default> Parse for State<S> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut input: &str = input;
        let mut state: Self = Self::default();

        for case_index in 0_usize..STATE_CASES_LEN {
            let (next_input, state_case): (&str, StateCase<S>) = map_opt(
                terminated(
                    StateCase::parse_curr_value_and_state_case,
                    cond(case_index == 0_usize, line_ending),
                ),
                |(curr_value, state_case)| {
                    (curr_value as usize == case_index).then_some(state_case)
                },
            )(input)?;

            input = next_input;
            state.cases[case_index] = state_case;
        }

        Ok((input, state))
    }
}

type StateIndexRaw = u8;
type StateIndex = TableIndex<StateIndexRaw>;
type StateTable = Table<StateId, State, StateIndexRaw>;

type RuntimeStateBitsRaw = usize;

struct RuntimeState {
    state: StateIndex,
    slot: usize,
    bits: VecDeque<RuntimeStateBitsRaw>,
}

impl RuntimeState {
    const ZERO_RUNTIME_STATE_BITS_RAW: RuntimeStateBitsRaw = 0 as RuntimeStateBitsRaw;
    const RUNTIME_STATE_BITS_RAW_BITS: usize = RuntimeStateBitsRaw::BITS as usize;

    fn new(state: StateIndex) -> Self {
        Self {
            state,
            slot: 0_usize,
            bits: vec![0 as RuntimeStateBitsRaw].into(),
        }
    }

    fn read(&self) -> bool {
        self.bits[self.slot / Self::RUNTIME_STATE_BITS_RAW_BITS].view_bits::<Lsb0>()
            [self.slot % Self::RUNTIME_STATE_BITS_RAW_BITS]
    }

    fn diagnostic_checksum(&self) -> usize {
        let (slice_a, slice_b): (&[usize], &[usize]) = self.bits.as_slices();

        slice_a.view_bits::<Lsb0>().count_ones() + slice_b.view_bits::<Lsb0>().count_ones()
    }

    fn write(&mut self, value: bool) {
        self.bits[self.slot / Self::RUNTIME_STATE_BITS_RAW_BITS]
            .view_bits_mut::<Lsb0>()
            .set(self.slot % Self::RUNTIME_STATE_BITS_RAW_BITS, value);
    }

    fn move_slot(&mut self, right: bool) {
        if right {
            self.slot += 1_usize;

            if self.slot == self.bits.len() * Self::RUNTIME_STATE_BITS_RAW_BITS {
                self.bits.push_back(Self::ZERO_RUNTIME_STATE_BITS_RAW);
            }
        } else {
            if self.slot == 0_usize {
                self.bits.push_front(Self::ZERO_RUNTIME_STATE_BITS_RAW);
                self.slot = Self::RUNTIME_STATE_BITS_RAW_BITS - 1_usize;
            } else {
                self.slot -= 1_usize;
            }
        }
    }

    fn step(&mut self, states: &StateTable) {
        assert!(self.state.is_valid());
        assert!(self.state.get() < states.as_slice().len());

        let case: StateCase = states.as_slice()[self.state.get()].data.cases[self.read() as usize];

        self.write(case.write_value);
        self.move_slot(case.move_right);
        self.state = case.next_state;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    start_state: StateIndex,
    diagnostic_checksum_steps: usize,
    states: StateTable,
}

impl Solution {
    fn parse_start_state<'i>(input: &'i str) -> IResult<&'i str, StateId> {
        delimited(tag("Begin in state "), StateId::parse, tag("."))(input)
    }

    fn parse_diagnostic_checksum_steps<'i>(input: &'i str) -> IResult<&'i str, usize> {
        delimited(
            tag("Perform a diagnostic checksum after "),
            parse_integer,
            tag(" steps."),
        )(input)
    }

    fn diagnostic_checksum_after_steps(&self) -> usize {
        let mut runtime_state: RuntimeState = RuntimeState::new(self.start_state);

        for _ in 0_usize..self.diagnostic_checksum_steps {
            runtime_state.step(&self.states);
        }

        runtime_state.diagnostic_checksum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let original_input: &str = input;
        let mut input: &'i str = input;
        let mut states: StateTable = StateTable::new();

        let (next_input, (_, diagnostic_checksum_steps)): (&str, (StateId, usize)) =
            separated_pair(
                Self::parse_start_state,
                line_ending,
                Self::parse_diagnostic_checksum_steps,
            )(input)?;

        input = next_input;

        many0(map(
            tuple((line_ending, line_ending, State::parse_id_and_state)),
            |(_, _, (id, _))| {
                states.find_or_add_index(&id);
            },
        ))(input)?;

        for state_index in 0_usize..states.as_slice().len() {
            let (next_input, state): (&str, State) = map_opt(
                tuple((line_ending, line_ending, State::parse_id_and_state)),
                |(_, _, (_, state))| State::try_from_state_id_state(state, &states),
            )(input)?;

            input = next_input;
            states.as_slice_mut()[state_index].data = state;
        }

        let start_state = map_opt(Self::parse_start_state, |start_state| {
            states.find_index(&start_state).opt()
        })(original_input)?
        .1;

        Ok((
            input,
            Self {
                start_state,
                diagnostic_checksum_steps,
                states,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// I initially set out to create a `BitVecDeque` for this, but determined it wasn't worth the
    /// effort to stand up for this.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.diagnostic_checksum_after_steps());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {}
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
        Begin in state A.\n\
        Perform a diagnostic checksum after 6 steps.\n\
        \n\
        In state A:\n  \
        If the current value is 0:\n    \
            - Write the value 1.\n    \
            - Move one slot to the right.\n    \
            - Continue with state B.\n  \
        If the current value is 1:\n    \
            - Write the value 0.\n    \
            - Move one slot to the left.\n    \
            - Continue with state B.\n\
        \n\
        In state B:\n  \
        If the current value is 0:\n    \
            - Write the value 1.\n    \
            - Move one slot to the left.\n    \
            - Continue with state A.\n  \
        If the current value is 1:\n    \
            - Write the value 1.\n    \
            - Move one slot to the right.\n    \
            - Continue with state A.\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                start_state: 0_usize.into(),
                diagnostic_checksum_steps: 6_usize,
                states: vec![
                    TableElement {
                        id: 'A'.try_into().unwrap(),
                        data: State {
                            cases: [
                                StateCase {
                                    write_value: true,
                                    move_right: true,
                                    next_state: 1_usize.into(),
                                },
                                StateCase {
                                    write_value: false,
                                    move_right: false,
                                    next_state: 1_usize.into(),
                                },
                            ],
                        },
                    },
                    TableElement {
                        id: 'B'.try_into().unwrap(),
                        data: State {
                            cases: [
                                StateCase {
                                    write_value: true,
                                    move_right: false,
                                    next_state: 0_usize.into(),
                                },
                                StateCase {
                                    write_value: true,
                                    move_right: true,
                                    next_state: 0_usize.into(),
                                },
                            ],
                        },
                    },
                ]
                .try_into()
                .unwrap(),
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
    fn test_diagnostic_checksum_after_steps() {
        for (index, diagnostic_checksum_after_steps) in [3_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).diagnostic_checksum_after_steps(),
                diagnostic_checksum_after_steps
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
