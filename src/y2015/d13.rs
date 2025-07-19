use {
    crate::*,
    arrayvec::ArrayVec,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, success, verify},
        error::Error,
        multi::separated_list1,
        sequence::tuple,
        Err, IResult,
    },
    std::iter::repeat,
};

/* --- Day 13: Knights of the Dinner Table ---

In years past, the holiday feast with your family hasn't gone so well. Not everyone gets along! This year, you resolve, will be different. You're going to find the optimal seating arrangement and avoid all those awkward conversations.

You start by writing up a list of everyone invited and the amount their happiness would increase or decrease if they were to find themselves sitting next to each other person. You have a circular table that will be just big enough to fit everyone comfortably, and so each person will have exactly two neighbors.

For example, suppose you have only four attendees planned, and you calculate their potential happiness as follows:

Alice would gain 54 happiness units by sitting next to Bob.
Alice would lose 79 happiness units by sitting next to Carol.
Alice would lose 2 happiness units by sitting next to David.
Bob would gain 83 happiness units by sitting next to Alice.
Bob would lose 7 happiness units by sitting next to Carol.
Bob would lose 63 happiness units by sitting next to David.
Carol would lose 62 happiness units by sitting next to Alice.
Carol would gain 60 happiness units by sitting next to Bob.
Carol would gain 55 happiness units by sitting next to David.
David would gain 46 happiness units by sitting next to Alice.
David would lose 7 happiness units by sitting next to Bob.
David would gain 41 happiness units by sitting next to Carol.

Then, if you seat Alice next to David, Alice would lose 2 happiness units (because David talks so much), but David would gain 46 happiness units (because Alice is such a good listener), for a total change of 44.

If you continue around the table, you could then seat Bob next to Alice (Bob gains 83, Alice gains 54). Finally, seat Carol, who sits next to Bob (Carol gains 60, Bob loses 7) and David (Carol gains 55, David gains 41). The arrangement looks like this:

     +41 +46
+55   David    -2
Carol       Alice
+60    Bob    +54
     -7  +83

After trying every other seating arrangement in this hypothetical scenario, you find that this one is the most optimal, with a total change in happiness of 330.

What is the total change in happiness for the optimal seating arrangement of the actual guest list?

--- Part Two ---

In all the commotion, you realize that you forgot to seat yourself. At this point, you're pretty apathetic toward the whole thing, and your happiness wouldn't really go up or down regardless of who you sit next to. You assume everyone else would be just as ambivalent about sitting next to you, too.

So, add yourself to the list, and give all happiness relationships that involve you a score of 0.

What is the total change in happiness for the optimal seating arrangement that actually includes yourself? */

const MIN_ATTENDEE_ID_LEN: usize = 1_usize;
const MAX_ATTENDEE_ID_LEN: usize = 7_usize;
const MAX_ATTENDEES: usize = MAX_BEADS;

type AttendeeId = StaticString<MAX_ATTENDEE_ID_LEN>;

fn parse_attendee_id<'i>(input: &'i str) -> IResult<&'i str, AttendeeId> {
    AttendeeId::parse_char1(MIN_ATTENDEE_ID_LEN, |c| c.is_ascii_alphabetic())(input)
}

type AttendeeIndexRaw = u8;
type AttendeeIndex = Index<AttendeeIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct AttendeeData {
    happiness_deltas: ArrayVec<i8, MAX_ATTENDEES>,
}

#[cfg(test)]
type Attendee = TableElement<AttendeeId, AttendeeData>;
type AttendeeTable = Table<AttendeeId, AttendeeData, AttendeeIndexRaw>;

struct LineData {
    happiness_target_attendee: AttendeeId,
    happiness_delta: i8,
    adjacent_attendee: AttendeeId,
}

impl Parse for LineData {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                parse_attendee_id,
                tag(" would "),
                alt((map(tag("gain"), |_| 1_i8), map(tag("lose"), |_| -1_i8))),
                tag(" "),
                parse_integer::<i8>,
                tag(" happiness units by sitting next to "),
                parse_attendee_id,
                tag("."),
            )),
            |(
                happiness_target_attendee,
                _,
                sign,
                _,
                happiness_magnitude,
                _,
                adjacent_attendee,
                _,
            )| {
                Self {
                    happiness_target_attendee,
                    happiness_delta: sign * happiness_magnitude,
                    adjacent_attendee,
                }
            },
        )(input)
    }
}

struct AttendeeBitConstants {
    attendees: usize,
    bits_per_attendee: usize,
    attendee_mask: u64,
    attendee_pair_mask: u64,
}

impl AttendeeBitConstants {
    fn new(attendees: usize) -> Self {
        let bits_per_attendee: usize = Solution::bits_per_attendee(attendees);
        let bits_per_attendee_pair: usize = bits_per_attendee * 2_usize;
        let attendee_mask: u64 = (1_u64 << bits_per_attendee) - 1_u64;
        let attendee_pair_mask: u64 = (1_u64 << bits_per_attendee_pair) - 1_u64;

        Self {
            attendees,
            bits_per_attendee,
            attendee_mask,
            attendee_pair_mask,
        }
    }

    fn total_happiness_delta(&self, solution: &Solution, attendee_bit_field: BitFieldArray) -> i32 {
        let BitFieldArray(BitArray {
            _ord,
            data: [mut attendee_bits],
        }) = attendee_bit_field;

        (0_usize..self.attendees)
            .map(|_| {
                let attendee_pair: u64 = attendee_bits & self.attendee_pair_mask;

                attendee_bits >>= self.bits_per_attendee;

                let attendee_a: AttendeeIndex =
                    ((attendee_pair & self.attendee_mask) as usize).into();
                let attendee_b: AttendeeIndex =
                    ((attendee_pair >> self.bits_per_attendee) as usize).into();

                solution.happiness_delta_for_pair(attendee_a, attendee_b)
            })
            .sum()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution(AttendeeTable);

impl Solution {
    fn bits_per_attendee(attendees: usize) -> usize {
        bits_per_field(attendees)
    }

    fn attendees(&self) -> usize {
        self.0.as_slice().len()
    }

    fn happiness_delta(&self, attendee_a: AttendeeIndex, attendee_b: AttendeeIndex) -> i32 {
        self.0
            .as_slice()
            .get(attendee_a.get())
            .map(|attendee_a| {
                attendee_a
                    .data
                    .happiness_deltas
                    .get(attendee_b.get())
                    .cloned()
            })
            .flatten()
            .unwrap_or_default() as i32
    }

    fn happiness_delta_for_pair(
        &self,
        attendee_a: AttendeeIndex,
        attendee_b: AttendeeIndex,
    ) -> i32 {
        self.happiness_delta(attendee_a, attendee_b) + self.happiness_delta(attendee_b, attendee_a)
    }

    fn attendee_id_list(
        &self,
        attendee_bit_field: BitFieldArray,
        attendees: usize,
    ) -> ArrayVec<AttendeeId, MAX_ATTENDEES> {
        beads_from_necklace(
            &attendee_bit_field,
            attendees,
            Self::bits_per_attendee(attendees),
        )
        .into_iter()
        .map(|attendee_index| {
            self.0.as_slice().get(attendee_index as usize).map_or_else(
                || "Self".try_into().unwrap(),
                |attendee| attendee.id.clone(),
            )
        })
        .collect()
    }

    fn optimal_seating_arrangement_and_total_happiness_delta_internal(
        &self,
        attendees: usize,
    ) -> (BitFieldArray, i32) {
        let attendee_bit_constants: AttendeeBitConstants = AttendeeBitConstants::new(attendees);

        iter_distinct_bead_necklaces(attendees, true)
            .map(|attendee_bit_field| {
                let total_happiness_delta: i32 =
                    attendee_bit_constants.total_happiness_delta(self, attendee_bit_field.clone());

                (attendee_bit_field, total_happiness_delta)
            })
            .max_by_key(|(_, total_happiness_delta)| *total_happiness_delta)
            .unwrap()
    }

    fn optimal_seating_arrangement_total_happiness_delta(&self) -> i32 {
        self.optimal_seating_arrangement_and_total_happiness_delta_internal(self.attendees())
            .1
    }

    fn optimal_seating_arrangement_total_happiness_delta_with_self(&self) -> i32 {
        self.optimal_seating_arrangement_and_total_happiness_delta_internal(
            self.attendees() + 1_usize,
        )
        .1
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        separated_list1(
            line_ending,
            map(LineData::parse, |line_data| {
                solution
                    .0
                    .find_or_add_index(&line_data.happiness_target_attendee);
                solution.0.find_or_add_index(&line_data.adjacent_attendee);
            }),
        )(input)?;

        solution.0.sort_by_id();

        let attendees: usize = solution.0.as_slice().len();

        verify(success(()), |_| attendees <= MAX_ATTENDEES)(input)?;

        for attendee in solution.0.as_slice_mut() {
            attendee
                .data
                .happiness_deltas
                .extend(repeat(0_i8).take(attendees));
        }

        let remaining_input: &str = separated_list1(
            line_ending,
            map(LineData::parse, |line_data| {
                let happiness_target_attendee: AttendeeIndex = solution
                    .0
                    .find_index_binary_search(&line_data.happiness_target_attendee);
                let adjacent_attendee: AttendeeIndex = solution
                    .0
                    .find_index_binary_search(&line_data.adjacent_attendee);

                solution.0.as_slice_mut()[happiness_target_attendee.get()]
                    .data
                    .happiness_deltas[adjacent_attendee.get()] = line_data.happiness_delta;
            }),
        )(input)?
        .0;

        verify(success(()), |_| {
            solution
                .0
                .as_slice()
                .iter()
                .enumerate()
                .all(|(attendee_index_a, attendee_a)| {
                    attendee_a
                        .data
                        .happiness_deltas
                        .iter()
                        .copied()
                        .enumerate()
                        .all(|(attendee_index_b, happiness_delta)| {
                            happiness_delta != 0_i8 || attendee_index_a == attendee_index_b
                        })
                })
        })(input)?;

        Ok((remaining_input, solution))
    }
}

impl RunQuestions for Solution {
    /// Generalized necklace iterator that only iterates over a necklace once per equivalence unit
    /// is hard.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let attendees: usize = self.attendees();
            let (optimal_seating_arrangement, total_happiness_delta): (BitFieldArray, i32) =
                self.optimal_seating_arrangement_and_total_happiness_delta_internal(attendees);

            dbg!(
                self.attendee_id_list(optimal_seating_arrangement, attendees),
                total_happiness_delta
            );
        } else {
            dbg!(self.optimal_seating_arrangement_total_happiness_delta());
        }
    }

    /// I'm thankful I didn't lean too hard into const generic parameters for the number of
    /// attendees and bits per attendees.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let attendees: usize = self.attendees() + 1_usize;
            let (optimal_seating_arrangement, total_happiness_delta): (BitFieldArray, i32) =
                self.optimal_seating_arrangement_and_total_happiness_delta_internal(attendees);

            dbg!(
                self.attendee_id_list(optimal_seating_arrangement, attendees),
                total_happiness_delta
            );
        } else {
            dbg!(self.optimal_seating_arrangement_total_happiness_delta_with_self());
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
        Alice would gain 54 happiness units by sitting next to Bob.\n\
        Alice would lose 79 happiness units by sitting next to Carol.\n\
        Alice would lose 2 happiness units by sitting next to David.\n\
        Bob would gain 83 happiness units by sitting next to Alice.\n\
        Bob would lose 7 happiness units by sitting next to Carol.\n\
        Bob would lose 63 happiness units by sitting next to David.\n\
        Carol would lose 62 happiness units by sitting next to Alice.\n\
        Carol would gain 60 happiness units by sitting next to Bob.\n\
        Carol would gain 55 happiness units by sitting next to David.\n\
        David would gain 46 happiness units by sitting next to Alice.\n\
        David would lose 7 happiness units by sitting next to Bob.\n\
        David would gain 41 happiness units by sitting next to Carol.\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! attendees {
            [ $( $id:expr => [ $( $happiness_delta:expr ),* ], )* ] => { Solution(vec![ $(
                Attendee {
                    id: $id.try_into().unwrap(),
                    data: AttendeeData {
                        happiness_deltas: [ $( $happiness_delta, )* ].into_iter().collect(),
                    },
                },
            )* ].try_into().unwrap()) }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![attendees!(
                "Alice" => [  0_i8, 54_i8, -79_i8,  -2_i8],
                "Bob" =>   [ 83_i8,  0_i8,  -7_i8, -63_i8],
                "Carol" => [-62_i8, 60_i8,   0_i8,  55_i8],
                "David" => [ 46_i8, -7_i8,  41_i8,   0_i8],
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
    fn test_optimal_seating_arrangement_and_total_happiness_delta_internal() {
        for (index, optimal_seating_arrangement_and_total_happiness_delta_internal) in
            [(BitFieldArray(BitArray::new([0b_11_10_01_00_u64])), 330_i32)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).optimal_seating_arrangement_and_total_happiness_delta_internal(
                    solution(index).attendees()
                ),
                optimal_seating_arrangement_and_total_happiness_delta_internal
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
