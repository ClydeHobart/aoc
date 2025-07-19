use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 4: Repose Record ---

You've sneaked into another supply closet - this time, it's across from the prototype suit manufacturing lab. You need to sneak inside and fix the issues with the suit, but there's a guard stationed outside the lab, so this is as close as you can safely get.

As you search the closet for anything that might help, you discover that you're not the first person to want to sneak in. Covering the walls, someone has spent an hour starting every midnight for the past few months secretly observing this guard post! They've been writing down the ID of the one guard on duty that night - the Elves seem to have decided that one guard was enough for the overnight shift - as well as when they fall asleep or wake up while at their post (your puzzle input).

For example, consider the following records, which have already been organized into chronological order:

[1518-11-01 00:00] Guard #10 begins shift
[1518-11-01 00:05] falls asleep
[1518-11-01 00:25] wakes up
[1518-11-01 00:30] falls asleep
[1518-11-01 00:55] wakes up
[1518-11-01 23:58] Guard #99 begins shift
[1518-11-02 00:40] falls asleep
[1518-11-02 00:50] wakes up
[1518-11-03 00:05] Guard #10 begins shift
[1518-11-03 00:24] falls asleep
[1518-11-03 00:29] wakes up
[1518-11-04 00:02] Guard #99 begins shift
[1518-11-04 00:36] falls asleep
[1518-11-04 00:46] wakes up
[1518-11-05 00:03] Guard #99 begins shift
[1518-11-05 00:45] falls asleep
[1518-11-05 00:55] wakes up

Timestamps are written using year-month-day hour:minute format. The guard falling asleep or waking up is always the one whose shift most recently started. Because all asleep/awake times are during the midnight hour (00:00 - 00:59), only the minute portion (00 - 59) is relevant for those events.

Visually, these records show that the guards are asleep at these times:

Date   ID   Minute
            000000000011111111112222222222333333333344444444445555555555
            012345678901234567890123456789012345678901234567890123456789
11-01  #10  .....####################.....#########################.....
11-02  #99  ........................................##########..........
11-03  #10  ........................#####...............................
11-04  #99  ....................................##########..............
11-05  #99  .............................................##########.....

The columns are Date, which shows the month-day portion of the relevant day; ID, which shows the guard on duty that day; and Minute, which shows the minutes during which the guard was asleep within the midnight hour. (The Minute column's header shows the minute's ten's digit in the first row and the one's digit in the second row.) Awake is shown as ., and asleep is shown as #.

Note that guards count as asleep on the minute they fall asleep, and they count as awake on the minute they wake up. For example, because Guard #10 wakes up at 00:25 on 1518-11-01, minute 25 is marked as awake.

If you can figure out the guard most likely to be asleep at a specific time, you might be able to trick that guard into working tonight so you can have the best chance of sneaking in. You have two strategies for choosing the best guard/minute combination.

Strategy 1: Find the guard that has the most minutes asleep. What minute does that guard spend asleep the most?

In the example above, Guard #10 spent the most minutes asleep, a total of 50 minutes (20+25+5), while Guard #99 only slept for a total of 30 minutes (10+10+10). Guard #10 was asleep most during minute 24 (on two days, whereas any other minute the guard was asleep was only seen on one day).

While this example listed the entries in chronological order, your entries are in the order you found them. You'll need to organize them before they can be analyzed.

What is the ID of the guard you chose multiplied by the minute you chose? (In the above example, the answer would be 10 * 24 = 240.)

--- Part Two ---

Strategy 2: Of all guards, which guard is most frequently asleep on the same minute?

In the example above, Guard #99 spent minute 45 asleep more than any other guard or minute - three times in total. (In all other cases, any guard spent any minute asleep at most twice.)

What is the ID of the guard you chose multiplied by the minute you chose? (In the above example, the answer would be 99 * 45 = 4455.) */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct Date {
    month: u8,
    day: u8,
}

impl Date {
    const MONTHS_PER_YEAR: u8 = 12_u8;
    const DAYS_PER_MONTH: [u8; Self::MONTHS_PER_YEAR as usize] = [
        31_u8, // January
        28_u8, // February (not a complete picture, but I don't care)
        31_u8, // March
        30_u8, // April
        31_u8, // May
        30_u8, // June
        31_u8, // July
        31_u8, // August
        30_u8, // September
        31_u8, // October
        30_u8, // November
        31_u8, // December
    ];

    fn is_last_day_of_month(self) -> bool {
        self.day == Self::DAYS_PER_MONTH[self.month as usize - 1_usize]
    }

    fn next(self) -> Self {
        if self.is_last_day_of_month() {
            Self {
                month: self.month % Self::MONTHS_PER_YEAR + 1_u8,
                day: 1_u8,
            }
        } else {
            Self {
                day: self.day + 1_u8,
                ..self
            }
        }
    }
}

impl Parse for Date {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                tag("1518-"),
                parse_integer_n_bytes(2_usize),
                tag("-"),
                parse_integer_n_bytes(2_usize),
            )),
            |(_, month, _, day)| {
                ((1_u8..=Self::MONTHS_PER_YEAR).contains(&month)
                    && (1_u8..=Self::DAYS_PER_MONTH[month as usize - 1_usize]).contains(&day))
                .then_some(Date { month, day })
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct Time {
    hour: u8,
    minute: u8,
}

impl Time {
    const HOURS_PER_DAY: u8 = 24_u8;
    const MINS_PER_HOUR: u8 = 60_u8;
    const MINS_PER_DAY: u32 = Self::MINS_PER_HOUR as u32 * Self::HOURS_PER_DAY as u32;

    fn portion_of_day(self) -> f32 {
        (self.minute as f32 / Self::MINS_PER_HOUR as f32 + self.hour as f32)
            / Self::HOURS_PER_DAY as f32
    }

    fn minute_of_day(self) -> u32 {
        self.hour as u32 * Self::MINS_PER_HOUR as u32 + self.minute as u32
    }
}

impl Parse for Time {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                parse_integer_n_bytes(2_usize),
                tag(":"),
                parse_integer_n_bytes(2_usize),
            )),
            |(hour, _, minute)| {
                (hour < Self::HOURS_PER_DAY && minute < Self::MINS_PER_HOUR)
                    .then_some(Self { hour, minute })
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct DateTime {
    date: Date,
    time: Time,
}

impl DateTime {
    fn effective_date(self) -> Date {
        if self.time.portion_of_day().round() == 0.0_f32 {
            self.date
        } else {
            self.date.next()
        }
    }
}

impl Parse for DateTime {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((tag("["), Date::parse, tag(" "), Time::parse, tag("]"))),
            |(_, date, _, time, _)| Self { date, time },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
enum UpdateRecordType {
    GuardBeginsShift { id: u16 },
    FallsAsleep,
    WakesUp,
}

impl UpdateRecordType {
    fn is_awake(self) -> bool {
        self != Self::FallsAsleep
    }
}

impl Parse for UpdateRecordType {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                tuple((tag("Guard #"), parse_integer, tag(" begins shift"))),
                |(_, id, _)| Self::GuardBeginsShift { id },
            ),
            map(tag("falls asleep"), |_| Self::FallsAsleep),
            map(tag("wakes up"), |_| Self::WakesUp),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct UpdateRecord {
    date_time: DateTime,
    update_record_type: UpdateRecordType,
}

impl Parse for UpdateRecord {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((DateTime::parse, tag(" "), UpdateRecordType::parse)),
            |(date_time, _, update_record_type)| Self {
                date_time,
                update_record_type,
            },
        )(input)
    }
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct GuardIdDate {
    guard_id: u16,
    date: Date,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct DateRecord {
    guard_id: u16,
    date: Date,
    update_record_range: Range<u16>,
}

impl DateRecord {
    fn minutes_asleep(self, solution: &Solution) -> u32 {
        let mut minutes_asleep: u32 = 0_u32;
        let mut previous_time: u32 = 0_u32;

        for update_record in &solution.update_records[self.update_record_range.as_range_usize()] {
            let current_time: u32 = update_record.date_time.time.minute_of_day();

            if update_record.update_record_type == UpdateRecordType::WakesUp {
                minutes_asleep += current_time - previous_time;
            }

            previous_time = current_time;
        }

        minutes_asleep
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct DaysAsleep(u16);

impl RegionTreeValue for DaysAsleep {
    fn insert_value_into_leaf_with_matching_range(&mut self, other: &Self) {
        self.0 += other.0;
    }

    fn should_convert_leaf_to_parent(&self, other: &Self) -> bool {
        other.0 != 0_u16
    }

    fn get_leaf<const D: usize, I: RangeIntTrait>(
        &self,
        _range: &RangeD<I, D>,
        _child_range: &RangeD<I, D>,
    ) -> Self {
        *self
    }

    fn try_convert_parent_to_leaf<'a, I>(mut iter: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a,
    {
        iter.try_fold(None, |expectation, days_asleep| {
            expectation
                .map(|expectation| expectation == *days_asleep)
                .unwrap_or(true)
                .then_some(Some(*days_asleep))
        })
        .flatten()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct GuardRecord {
    guard_id: u16,
    date_record_range: Range<u16>,
}

impl GuardRecord {
    const MINUTE_OF_DAY_RANGE_END: u16 = 1_u16 << bits_to_store(Time::MINS_PER_DAY as usize);

    fn minutes_asleep(self, solution: &Solution) -> u32 {
        solution.date_records[self.date_record_range.as_range_usize()]
            .iter()
            .map(|date_record| date_record.clone().minutes_asleep(solution))
            .sum()
    }

    fn days_asleep_binary_tree(self, solution: &Solution) -> BinaryTree<DaysAsleep, u16> {
        let mut days_asleep_binary_tree: BinaryTree<DaysAsleep, u16> = BinaryTree::new(
            ([0_u16]..[Self::MINUTE_OF_DAY_RANGE_END])
                .try_into()
                .unwrap(),
            DaysAsleep::default(),
        );

        for date_record in &solution.date_records[self.date_record_range.as_range_usize()] {
            let mut previous_time: u16 = 0_u16;

            for update_record in
                &solution.update_records[date_record.update_record_range.as_range_usize()]
            {
                let current_time: u16 = update_record.date_time.time.minute_of_day() as u16;

                if update_record.update_record_type == UpdateRecordType::WakesUp {
                    for range_1 in
                        Range1::<u16>::iter_from_start_and_end_1d(previous_time..current_time)
                    {
                        days_asleep_binary_tree.insert(&range_1, &DaysAsleep(1_u16));
                    }
                }

                previous_time = current_time;
            }
        }

        days_asleep_binary_tree
    }

    fn try_most_asleep_minute_and_days_asleep(self, solution: &Solution) -> Option<(u16, u16)> {
        let days_asleep_binary_tree: BinaryTree<DaysAsleep, u16> =
            self.days_asleep_binary_tree(solution);
        let mut most_asleep_minute_and_days_asleep: Option<(u16, u16)> = None;

        days_asleep_binary_tree.visit_all_leaves(
            |_| true,
            |range_1, days_asleep| {
                if let Some((_, most_days_asleep)) = &most_asleep_minute_and_days_asleep {
                    if *most_days_asleep < days_asleep.0 {
                        most_asleep_minute_and_days_asleep =
                            Some((range_1.start[0_usize], days_asleep.0));
                    }
                } else if days_asleep.0 > 0_u16 {
                    // Just take the first minute of the range.
                    most_asleep_minute_and_days_asleep =
                        Some((range_1.start[0_usize], days_asleep.0));
                }

                //
                true
            },
        );

        most_asleep_minute_and_days_asleep
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    update_records: Vec<UpdateRecord>,
    date_records: Vec<DateRecord>,
    guard_records: Vec<GuardRecord>,
}

impl Solution {
    fn try_date_records_from_update_records(
        update_records: &[UpdateRecord],
    ) -> Option<Vec<DateRecord>> {
        let mut date_records: Vec<DateRecord> = Vec::new();
        let mut current_date_record_option: Option<DateRecord> = None;

        update_records
            .iter()
            .enumerate()
            .try_for_each(|(index, update_record)| {
                let mut iter_option: Option<()> = None;

                match &mut current_date_record_option {
                    Some(current_date_record) => {
                        if update_record.date_time.effective_date() == current_date_record.date {
                            match update_record.update_record_type {
                                UpdateRecordType::GuardBeginsShift { .. } => {
                                    // Only one guard per day, this is an error.
                                }
                                UpdateRecordType::FallsAsleep => {
                                    // We can only fall asleep if we're currently awake
                                    if update_records[index - 1_usize]
                                        .update_record_type
                                        .is_awake()
                                    {
                                        current_date_record.update_record_range.end += 1_u16;

                                        iter_option = Some(());
                                    }
                                }
                                UpdateRecordType::WakesUp => {
                                    // We can only wake up if we're currently asleep
                                    if !update_records[index - 1_usize]
                                        .update_record_type
                                        .is_awake()
                                    {
                                        current_date_record.update_record_range.end += 1_u16;

                                        iter_option = Some(());
                                    }
                                }
                            }
                        // Verify that the last day ended awake.
                        } else if update_records[index - 1_usize]
                            .update_record_type
                            .is_awake()
                        {
                            // We need to start the day with a guard beginning their shift.
                            if let UpdateRecordType::GuardBeginsShift { id } =
                                update_record.update_record_type
                            {
                                date_records.push(current_date_record_option.take().unwrap());
                                current_date_record_option = Some(DateRecord {
                                    date: update_record.date_time.effective_date(),
                                    guard_id: id,
                                    update_record_range: index as u16..index as u16 + 1_u16,
                                });

                                iter_option = Some(())
                            }
                        }
                    }
                    None => {
                        // We need to start the day with a guard beginning their shift.
                        if let UpdateRecordType::GuardBeginsShift { id } =
                            update_record.update_record_type
                        {
                            current_date_record_option = Some(DateRecord {
                                date: update_record.date_time.effective_date(),
                                guard_id: id,
                                update_record_range: index as u16..index as u16 + 1_u16,
                            });

                            iter_option = Some(())
                        }
                    }
                }

                iter_option
            })
            .map(|_| {
                date_records.extend(current_date_record_option);
                date_records.sort_by_key(|date_record| GuardIdDate {
                    guard_id: date_record.guard_id,
                    date: date_record.date,
                });

                date_records
            })
    }

    fn guard_records_from_date_records(date_records: &[DateRecord]) -> Vec<GuardRecord> {
        let mut guard_records: Vec<GuardRecord> = Vec::new();
        let mut current_guard_record_option: Option<GuardRecord> = None;

        for (index, date_record) in date_records.iter().enumerate() {
            if let Some(current_guard_record) = &mut current_guard_record_option {
                if date_record.guard_id == current_guard_record.guard_id {
                    current_guard_record.date_record_range.end += 1_u16;
                } else {
                    guard_records.push(current_guard_record_option.take().unwrap());

                    current_guard_record_option = Some(GuardRecord {
                        guard_id: date_record.guard_id,
                        date_record_range: index as u16..index as u16 + 1_u16,
                    });
                }
            } else {
                current_guard_record_option = Some(GuardRecord {
                    guard_id: date_record.guard_id,
                    date_record_range: index as u16..index as u16 + 1_u16,
                });
            }
        }

        guard_records.extend(current_guard_record_option);

        guard_records
    }

    fn try_from_update_records(update_records: Vec<UpdateRecord>) -> Option<Self> {
        Self::try_date_records_from_update_records(&update_records).map(|date_records| {
            let guard_records: Vec<GuardRecord> =
                Self::guard_records_from_date_records(&date_records);

            Self {
                update_records,
                date_records,
                guard_records,
            }
        })
    }

    fn try_most_minutes_asleep_guard_index(&self) -> Option<usize> {
        (0_usize..self.guard_records.len()).max_by_key(|guard_record_index| {
            self.guard_records[*guard_record_index]
                .clone()
                .minutes_asleep(self)
        })
    }

    fn try_strategy_1_most_asleep_guard_id_minute_and_days_asleep(
        &self,
    ) -> Option<(u16, u16, u16)> {
        self.try_most_minutes_asleep_guard_index()
            .and_then(|most_asleep_guard_index| {
                let most_asleep_guard_record: &GuardRecord =
                    &self.guard_records[most_asleep_guard_index];

                most_asleep_guard_record
                    .clone()
                    .try_most_asleep_minute_and_days_asleep(self)
                    .map(|(most_asleep_minute, most_days_asleep)| {
                        (
                            most_asleep_guard_record.guard_id,
                            most_asleep_minute,
                            most_days_asleep,
                        )
                    })
            })
    }

    fn try_strategy_1_most_asleep_guard_id_and_minute_product(&self) -> Option<u32> {
        self.try_strategy_1_most_asleep_guard_id_minute_and_days_asleep()
            .map(|(most_asleep_guard_id, most_asleep_minute, _)| {
                most_asleep_guard_id as u32 * most_asleep_minute as u32
            })
    }

    fn try_strategy_2_most_asleep_guard_id_minute_and_days_asleep(
        &self,
    ) -> Option<(u16, u16, u16)> {
        self.guard_records
            .iter()
            .filter_map(|guard_record| {
                guard_record
                    .clone()
                    .try_most_asleep_minute_and_days_asleep(self)
                    .map(|(most_asleep_minute, most_days_asleep)| {
                        (guard_record.guard_id, most_asleep_minute, most_days_asleep)
                    })
            })
            .max_by_key(|(_, _, most_days_asleep)| *most_days_asleep)
    }

    fn try_strategy_2_most_asleep_guard_id_and_minute_product(&self) -> Option<u32> {
        self.try_strategy_2_most_asleep_guard_id_minute_and_days_asleep()
            .map(|(most_asleep_guard_id, most_asleep_minute, _)| {
                most_asleep_guard_id as u32 * most_asleep_minute as u32
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            map(
                many0(terminated(UpdateRecord::parse, opt(line_ending))),
                |mut update_records| {
                    update_records.sort_by_key(|update_record| update_record.date_time);
                    update_records
                },
            ),
            Self::try_from_update_records,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Year 2018 has hands! Funky data structures in this one to keep dynamic allocation down.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((most_asleep_guard_id, most_asleep_minute, most_days_asleep)) =
                self.try_strategy_1_most_asleep_guard_id_minute_and_days_asleep()
            {
                dbg!(most_asleep_guard_id, most_asleep_minute, most_days_asleep);
                dbg!(most_asleep_guard_id as u32 * most_asleep_minute as u32);
            } else {
                eprintln!("Failed to compute most asleep guard ID, minute, and days asleep.");
            }
        } else {
            dbg!(self.try_strategy_1_most_asleep_guard_id_and_minute_product());
        }
    }

    /// Well set up for this one following q1.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((most_asleep_guard_id, most_asleep_minute, most_days_asleep)) =
                self.try_strategy_2_most_asleep_guard_id_minute_and_days_asleep()
            {
                dbg!(most_asleep_guard_id, most_asleep_minute, most_days_asleep);
                dbg!(most_asleep_guard_id as u32 * most_asleep_minute as u32);
            } else {
                eprintln!("Failed to compute most asleep guard ID, minute, and days asleep.");
            }
        } else {
            dbg!(self.try_strategy_2_most_asleep_guard_id_and_minute_product());
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
        [1518-11-05 00:55] wakes up\n\
        [1518-11-05 00:45] falls asleep\n\
        [1518-11-05 00:03] Guard #99 begins shift\n\
        [1518-11-04 00:46] wakes up\n\
        [1518-11-04 00:36] falls asleep\n\
        [1518-11-04 00:02] Guard #99 begins shift\n\
        [1518-11-03 00:29] wakes up\n\
        [1518-11-03 00:24] falls asleep\n\
        [1518-11-03 00:05] Guard #10 begins shift\n\
        [1518-11-02 00:50] wakes up\n\
        [1518-11-02 00:40] falls asleep\n\
        [1518-11-01 23:58] Guard #99 begins shift\n\
        [1518-11-01 00:55] wakes up\n\
        [1518-11-01 00:30] falls asleep\n\
        [1518-11-01 00:25] wakes up\n\
        [1518-11-01 00:05] falls asleep\n\
        [1518-11-01 00:00] Guard #10 begins shift\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                update_records: vec![
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 00_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::GuardBeginsShift { id: 10_u16 },
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 05_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 25_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 30_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 55_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 01_u8,
                            },
                            time: Time {
                                hour: 23_u8,
                                minute: 58_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::GuardBeginsShift { id: 99_u16 },
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 02_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 40_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 02_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 50_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 03_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 05_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::GuardBeginsShift { id: 10_u16 },
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 03_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 24_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 03_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 29_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 04_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 02_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::GuardBeginsShift { id: 99_u16 },
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 04_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 36_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 04_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 46_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 05_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 03_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::GuardBeginsShift { id: 99_u16 },
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 05_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 45_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::FallsAsleep,
                    },
                    UpdateRecord {
                        date_time: DateTime {
                            date: Date {
                                month: 11_u8,
                                day: 05_u8,
                            },
                            time: Time {
                                hour: 00_u8,
                                minute: 55_u8,
                            },
                        },
                        update_record_type: UpdateRecordType::WakesUp,
                    },
                ],
                date_records: vec![
                    DateRecord {
                        date: Date {
                            month: 11_u8,
                            day: 01_u8,
                        },
                        guard_id: 10_u16,
                        update_record_range: 0_u16..5_u16,
                    },
                    DateRecord {
                        date: Date {
                            month: 11_u8,
                            day: 03_u8,
                        },
                        guard_id: 10_u16,
                        update_record_range: 8_u16..11_u16,
                    },
                    DateRecord {
                        date: Date {
                            month: 11_u8,
                            day: 02_u8,
                        },
                        guard_id: 99_u16,
                        update_record_range: 5_u16..8_u16,
                    },
                    DateRecord {
                        date: Date {
                            month: 11_u8,
                            day: 04_u8,
                        },
                        guard_id: 99_u16,
                        update_record_range: 11_u16..14_u16,
                    },
                    DateRecord {
                        date: Date {
                            month: 11_u8,
                            day: 05_u8,
                        },
                        guard_id: 99_u16,
                        update_record_range: 14_u16..17_u16,
                    },
                ],
                guard_records: vec![
                    GuardRecord {
                        guard_id: 10_u16,
                        date_record_range: 0_u16..2_u16,
                    },
                    GuardRecord {
                        guard_id: 99_u16,
                        date_record_range: 2_u16..5_u16,
                    },
                ],
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
    fn test_try_most_minutes_asleep_guard_index() {
        for (index, most_minutes_asleep_guard_index) in [Some(0_usize)].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_most_minutes_asleep_guard_index(),
                most_minutes_asleep_guard_index
            );
        }
    }

    #[test]
    fn test_try_most_asleep_minute_and_days_asleep() {
        for (solution_index, most_asleep_minutes) in
            [[Some((24_u16, 2_u16)), Some((45_u16, 3_u16))]]
                .into_iter()
                .enumerate()
        {
            let solution: &Solution = solution(solution_index);

            for (guard_index, most_asleep_minute_and_days_asleep) in
                most_asleep_minutes.into_iter().enumerate()
            {
                assert_eq!(
                    solution.guard_records[guard_index]
                        .clone()
                        .try_most_asleep_minute_and_days_asleep(solution),
                    most_asleep_minute_and_days_asleep,
                    "solution_index: {solution_index}, guard_index: {guard_index}"
                );
            }
        }
    }

    #[test]
    fn test_try_strategy_1_most_asleep_guard_id_minute_and_days_asleep() {
        for (index, strategy_1_most_asleep_guard_id_minute_and_days_asleep) in
            [Some((10_u16, 24_u16, 2_u16))].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_strategy_1_most_asleep_guard_id_minute_and_days_asleep(),
                strategy_1_most_asleep_guard_id_minute_and_days_asleep
            );
        }
    }

    #[test]
    fn test_try_strategy_2_most_asleep_guard_id_minute_and_days_asleep() {
        for (index, strategy_2_most_asleep_guard_id_minute_and_days_asleep) in
            [Some((99_u16, 45_u16, 3_u16))].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_strategy_2_most_asleep_guard_id_minute_and_days_asleep(),
                strategy_2_most_asleep_guard_id_minute_and_days_asleep
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
