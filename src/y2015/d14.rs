use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_res},
        error::Error,
        multi::separated_list1,
        sequence::tuple,
        Err, IResult,
    },
};

/* --- Day 14: Reindeer Olympics ---

This year is the Reindeer Olympics! Reindeer can fly at high speeds, but must rest occasionally to recover their energy. Santa would like to know which of his reindeer is fastest, and so he has them race.

Reindeer can only either be flying (always at their top speed) or resting (not moving at all), and always spend whole seconds in either state.

For example, suppose you have the following Reindeer:

    Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.
    Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds.

After one second, Comet has gone 14 km, while Dancer has gone 16 km. After ten seconds, Comet has gone 140 km, while Dancer has gone 160 km. On the eleventh second, Comet begins resting (staying at 140 km), and Dancer continues on for a total distance of 176 km. On the 12th second, both reindeer are resting. They continue to rest until the 138th second, when Comet flies for another ten seconds. On the 174th second, Dancer flies for another 11 seconds.

In this example, after the 1000th second, both reindeer are resting, and Comet is in the lead at 1120 km (poor Dancer has only gotten 1056 km by that point). So, in this situation, Comet would win (if the race ended at 1000 seconds).

Given the descriptions of each reindeer (in your puzzle input), after exactly 2503 seconds, what distance has the winning reindeer traveled?

--- Part Two ---

Seeing how reindeer move in bursts, Santa decides he's not pleased with the old scoring system.

Instead, at the end of each second, he awards one point to the reindeer currently in the lead. (If there are multiple reindeer tied for the lead, they each get one point.) He keeps the traditional 2503 second time limit, of course, as doing otherwise would be entirely ridiculous.

Given the example reindeer from above, after the first second, Dancer is in the lead and gets one point. He stays in the lead until several seconds into Comet's second burst: after the 140th second, Comet pulls into the lead and gets his first point. Of course, since Dancer had been in the lead for the 139 seconds before that, he has accumulated 139 points by the 140th second.

After the 1000th second, Dancer has accumulated 689 points, while poor Comet, our old champion, only has 312. So, with the new scoring system, Dancer would win (if the race ended at 1000 seconds).

Again given the descriptions of each reindeer (in your puzzle input), after exactly 2503 seconds, how many points does the winning reindeer have? */

type ReindeerIndexRaw = u8;

const MIN_REINDEER_ID_LEN: usize = 1_usize;
const MAX_REINDEER_ID_LEN: usize = 7_usize;

type ReindeerId = StaticString<MAX_REINDEER_ID_LEN>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct ReindeerData {
    /// Flight speed in *km/s*.
    flight_speed: u8,

    /// Flight duration in *s*.
    flight_duration: u8,

    /// Rest duration in *s*.
    rest_duration: u8,
}

impl ReindeerData {
    fn cycle_period(self) -> u16 {
        self.flight_duration as u16 + self.rest_duration as u16
    }

    fn distance(self, seconds: u16) -> u16 {
        let period: u16 = self.cycle_period();
        let full_cycles: u16 = seconds / period;
        let partial_cycle_seconds: u16 = seconds - full_cycles * period;
        let flight_speed: u16 = self.flight_speed as u16;
        let flight_duration: u16 = self.flight_duration as u16;

        full_cycles * flight_speed * flight_duration
            + flight_speed * partial_cycle_seconds.min(flight_duration)
    }
}

impl Parse for ReindeerData {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag(" can fly "),
                parse_integer,
                tag(" km/s for "),
                parse_integer,
                tag(" seconds, but then must rest for "),
                parse_integer,
                tag(" seconds."),
            )),
            |(_, flight_speed, _, flight_duration, _, rest_duration, _)| Self {
                flight_speed,
                flight_duration,
                rest_duration,
            },
        )(input)
    }
}

type Reindeer = TableElement<ReindeerId, ReindeerData>;

impl Parse for Reindeer {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                ReindeerId::parse_char1(MIN_REINDEER_ID_LEN, |c| c.is_ascii_alphabetic()),
                ReindeerData::parse,
            )),
            |(id, data)| Self { id, data },
        )(input)
    }
}

type ReindeerTable = Table<ReindeerId, ReindeerData, ReindeerIndexRaw>;

#[derive(Default)]
struct ReindeerState {
    remaining_flight_duration: u8,
    remaining_rest_duration: u8,
    distance: u16,
    points: u16,
}

impl ReindeerState {
    fn new(flight_duration: u8) -> Self {
        Self {
            remaining_flight_duration: flight_duration,
            ..Default::default()
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(ReindeerTable);

impl Solution {
    const SECONDS: u16 = 2503_u16;

    fn winning_reindeer_id_and_distance(&self, seconds: u16) -> (ReindeerId, u16) {
        self.0
            .as_slice()
            .iter()
            .map(|reindeer| (reindeer.id.clone(), reindeer.data.distance(seconds)))
            .max_by_key(|(_, distance)| *distance)
            .unwrap()
    }

    fn winning_reindeer_distance(&self, seconds: u16) -> u16 {
        self.winning_reindeer_id_and_distance(seconds).1
    }

    fn reindeer_states(&self) -> Vec<ReindeerState> {
        self.0
            .as_slice()
            .iter()
            .map(|reindeer| ReindeerState::new(reindeer.data.flight_duration))
            .collect()
    }

    fn winning_reindeer_id_and_points(&self, seconds: u16) -> (ReindeerId, u16) {
        let mut reindeer_states: Vec<ReindeerState> = self.reindeer_states();

        for _ in 0_u16..seconds {
            let max_distance: u16 = self
                .0
                .as_slice()
                .iter()
                .zip(reindeer_states.iter_mut())
                .fold(0_u16, |max_distance, (reindeer, reindeer_state)| {
                    if reindeer_state.remaining_flight_duration > 0_u8 {
                        reindeer_state.distance += reindeer.data.flight_speed as u16;
                        reindeer_state.remaining_flight_duration -= 1_u8;

                        if reindeer_state.remaining_flight_duration == 0_u8 {
                            reindeer_state.remaining_rest_duration = reindeer.data.rest_duration;
                        }
                    } else {
                        assert!(reindeer_state.remaining_rest_duration > 0_u8);

                        reindeer_state.remaining_rest_duration -= 1_u8;

                        if reindeer_state.remaining_rest_duration == 0_u8 {
                            reindeer_state.remaining_flight_duration =
                                reindeer.data.flight_duration;
                        }
                    }

                    max_distance.max(reindeer_state.distance)
                });

            for reindeer_state in &mut reindeer_states {
                if reindeer_state.distance == max_distance {
                    reindeer_state.points += 1_u16;
                }
            }
        }

        self.0
            .as_slice()
            .iter()
            .zip(reindeer_states.into_iter())
            .map(|(reindeer, reindeer_state)| (reindeer.id.clone(), reindeer_state.points))
            .max_by_key(|(_, points)| *points)
            .unwrap()
    }

    fn winning_reindeer_points(&self, seconds: u16) -> u16 {
        self.winning_reindeer_id_and_points(seconds).1
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            map_res(
                separated_list1(line_ending, Reindeer::parse),
                ReindeerTable::try_from,
            ),
            |mut reindeer_table| {
                reindeer_table.sort_by_id();

                Self(reindeer_table)
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// prediction: q2 is just super high second count.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.winning_reindeer_id_and_distance(Self::SECONDS));
        } else {
            dbg!(self.winning_reindeer_distance(Self::SECONDS));
        }
    }

    /// curve ball!
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.winning_reindeer_id_and_points(Self::SECONDS));
        } else {
            dbg!(self.winning_reindeer_points(Self::SECONDS));
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
        Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.\n\
        Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds.\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                vec![
                    Reindeer {
                        id: "Comet".try_into().unwrap(),
                        data: ReindeerData {
                            flight_speed: 14_u8,
                            flight_duration: 10_u8,
                            rest_duration: 127_u8,
                        },
                    },
                    Reindeer {
                        id: "Dancer".try_into().unwrap(),
                        data: ReindeerData {
                            flight_speed: 16_u8,
                            flight_duration: 11_u8,
                            rest_duration: 162_u8,
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
    fn test_distance() {
        for (index, seconds_and_distances) in [vec![
            (1_u16, vec![14_u16, 16_u16]),
            (10_u16, vec![140_u16, 160_u16]),
            (11_u16, vec![140_u16, 176_u16]),
            (12_u16, vec![140_u16, 176_u16]),
            (1000_u16, vec![1120_u16, 1056_u16]),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (seconds, distances) in seconds_and_distances {
                assert_eq!(
                    solution
                        .0
                        .as_slice()
                        .iter()
                        .map(|reindeer| reindeer.data.distance(seconds))
                        .collect::<Vec<u16>>(),
                    distances
                );
            }
        }
    }

    #[test]
    fn test_winning_reindeer_id_and_points() {
        for (index, winning_reindeer_id_and_points) in
            [(ReindeerId::try_from("Dancer").unwrap(), 689_u16)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).winning_reindeer_id_and_points(1000_u16),
                winning_reindeer_id_and_points
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
