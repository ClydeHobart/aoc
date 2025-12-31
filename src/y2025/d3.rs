use {
    self::{
        bank::prelude::*, battery::prelude::*, joltage::prelude::*, joltage_computer::prelude::*,
    },
    crate::*,
    nom::{character::complete::line_ending, error::Error, multi::separated_list1, Err, IResult},
};

/* --- Day 3: Lobby ---

You descend a short staircase, enter the surprisingly vast lobby, and are quickly cleared by the security checkpoint. When you get to the main elevators, however, you discover that each one has a red light above it: they're all offline.

"Sorry about that," an Elf apologizes as she tinkers with a nearby control panel. "Some kind of electrical surge seems to have fried them. I'll try to get them online soon."

You explain your need to get further underground. "Well, you could at least take the escalator down to the printing department, not that you'd get much further than that without the elevators working. That is, you could if the escalator weren't also offline."

"But, don't worry! It's not fried; it just needs power. Maybe you can get it running while I keep working on the elevators."

There are batteries nearby that can supply emergency power to the escalator for just such an occasion. The batteries are each labeled with their joltage rating, a value from 1 to 9. You make a note of their joltage ratings (your puzzle input). For example:

987654321111111
811111111111119
234234234234278
818181911112111

The batteries are arranged into banks; each line of digits in your input corresponds to a single bank of batteries. Within each bank, you need to turn on exactly two batteries; the joltage that the bank produces is equal to the number formed by the digits on the batteries you've turned on. For example, if you have a bank like 12345 and you turn on batteries 2 and 4, the bank would produce 24 jolts. (You cannot rearrange batteries.)

You'll need to find the largest possible joltage each bank can produce. In the above example:

    In 987654321111111, you can make the largest joltage possible, 98, by turning on the first two batteries.
    In 811111111111119, you can make the largest joltage possible by turning on the batteries labeled 8 and 9, producing 89 jolts.
    In 234234234234278, you can make 78 by turning on the last two batteries (marked 7 and 8).
    In 818181911112111, the largest joltage you can produce is 92.

The total output joltage is the sum of the maximum joltage from each bank, so in this example, the total output joltage is 98 + 89 + 78 + 92 = 357.

There are many batteries in front of you. Find the maximum joltage possible from each bank; what is the total output joltage?

--- Part Two ---

The escalator doesn't move. The Elf explains that it probably needs more joltage to overcome the static friction of the system and hits the big red "joltage limit safety override" button. You lose count of the number of times she needs to confirm "yes, I'm sure" and decorate the lobby a bit while you wait.

Now, you need to make the largest joltage by turning on exactly twelve batteries within each bank.

The joltage output for the bank is still the number formed by the digits of the batteries you've turned on; the only difference is that now there will be 12 digits in each bank's joltage output instead of two.

Consider again the example from before:

987654321111111
811111111111119
234234234234278
818181911112111

Now, the joltages are much larger:

    In 987654321111111, the largest joltage can be found by turning on everything except some 1s at the end to produce 987654321111.
    In the digit sequence 811111111111119, the largest joltage can be found by turning on everything except some 1s, producing 811111111119.
    In 234234234234278, the largest joltage can be found by turning on everything except a 2 battery, a 3 battery, and another 2 battery near the start to produce 434234234278.
    In 818181911112111, the joltage 888911112111 is produced by turning on everything except some 1s near the front.

The total output joltage is now much larger: 987654321111 + 811111111119 + 434234234278 + 888911112111 = 3121910778619.

What is the new total output joltage? */

mod joltage {
    pub mod prelude {
        pub use super::Joltage;
    }

    pub type JoltageRaw = u64;

    #[derive(Clone, Copy, Debug, Default)]
    #[cfg_attr(test, derive(PartialEq))]
    pub struct Joltage(pub JoltageRaw);

    impl Joltage {
        pub const ZERO: Self = Self(0 as JoltageRaw);
    }
}

mod battery {
    use {
        super::*,
        nom::{character::complete::satisfy, combinator::map, IResult},
    };

    pub mod prelude {
        pub use super::{Battery, BatteryIndex};
    }

    pub type BatteryIndexRaw = u16;

    pub type BatteryIndex = Index<BatteryIndexRaw>;

    pub type BatteryRaw = u8;

    #[derive(Clone, Copy)]
    #[cfg_attr(test, derive(Debug, PartialEq))]
    pub struct Battery(pub BatteryRaw);

    impl Battery {
        fn is_char_valid(value: char) -> bool {
            value.is_ascii_digit()
        }

        unsafe fn from_char_unsafe(value: char) -> Self {
            Self(value as BatteryRaw - b'0' as BatteryRaw)
        }

        pub fn joltage(self) -> Joltage {
            Joltage(self.0 as joltage::JoltageRaw)
        }
    }

    impl Parse for Battery {
        fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
            map(satisfy(Self::is_char_valid), |value| unsafe {
                Self::from_char_unsafe(value)
            })(input)
        }
    }

    impl TryFrom<char> for Battery {
        type Error = ();

        fn try_from(value: char) -> Result<Self, Self::Error> {
            Self::is_char_valid(value)
                .then(|| unsafe { Self::from_char_unsafe(value) })
                .ok_or(())
        }
    }
}

mod joltage_computer {
    use {
        super::*,
        glam::IVec2,
        std::{
            iter::{from_fn, repeat},
            mem::take,
        },
    };

    #[cfg(test)]
    use bitvec::prelude::*;

    pub mod prelude {
        pub use super::JoltageComputer;
    }

    pub const MAX_ENABLED_BATTERY_COUNT: usize = digits(joltage::JoltageRaw::MAX as usize);

    pub const MAX_BANK_BATTERIES_LEN: usize = 128_usize;

    #[cfg(test)]
    type BatteryBitArray = BitArr!(for MAX_BANK_BATTERIES_LEN, in u64);

    #[cfg_attr(test, derive(Debug, PartialEq))]
    #[derive(Clone, Copy)]
    pub struct JoltageComputerCell {
        pub joltage: Joltage,

        #[cfg(test)]
        enabled_batteries: BatteryBitArray,
    }

    impl JoltageComputerCell {
        const DEFAULT: Self = Self {
            joltage: Joltage::ZERO,

            #[cfg(test)]
            enabled_batteries: BatteryBitArray::ZERO,
        };
    }

    #[cfg_attr(test, derive(Debug, PartialEq))]
    #[derive(Clone, Copy)]
    struct JoltageComputerDimensions {
        bank_batteries_len: usize,
        enabled_battery_count: usize,
    }

    impl JoltageComputerDimensions {
        fn from_ivec2(value: IVec2) -> Self {
            Self {
                bank_batteries_len: value.x as usize,
                enabled_battery_count: value.y as usize,
            }
        }

        fn into_ivec2(self) -> IVec2 {
            (
                self.bank_batteries_len as i32,
                self.enabled_battery_count as i32,
            )
                .into()
        }
    }

    impl From<IVec2> for JoltageComputerDimensions {
        fn from(value: IVec2) -> Self {
            Self::from_ivec2(value)
        }
    }

    impl From<JoltageComputerDimensions> for IVec2 {
        fn from(value: JoltageComputerDimensions) -> Self {
            value.into_ivec2()
        }
    }

    pub struct JoltageComputer(Grid2D<JoltageComputerCell>);

    impl JoltageComputer {
        pub fn new() -> Self {
            Self(Grid2D::empty(IVec2::ZERO))
        }

        pub fn try_compute_max_joltage(
            &mut self,
            bank_batteries: &[Battery],
            enabled_battery_count: usize,
        ) -> Result<JoltageComputerCell, &'static str> {
            take(self)
                .try_initialize(bank_batteries, enabled_battery_count)
                .map(|joltage_computer| {
                    let (state, max_joltage): (Self, JoltageComputerCell) =
                        joltage_computer.compute_max_joltage();

                    *self = state;

                    max_joltage
                })
                .map_err(|(state, err)| {
                    *self = state;

                    err
                })
        }

        fn get_dimensions(&self) -> JoltageComputerDimensions {
            self.0.max_dimensions().into()
        }

        fn try_initialize<'b>(
            self,
            bank_batteries: &'b [Battery],
            enabled_battery_count: usize,
        ) -> Result<JoltageComputerWithBatteries<'b>, (Self, &'static str)> {
            let bank_batteries_len: usize = bank_batteries.len();

            if bank_batteries_len > MAX_BANK_BATTERIES_LEN {
                Err((
                    self,
                    "bank_batteries len was greater than MAX_BANK_BATTERIES_LEN",
                ))
            } else if enabled_battery_count > MAX_ENABLED_BATTERY_COUNT {
                Err((
                    self,
                    "enabled_battery_count was greater than MAX_BANK_BATTERIES_LEN",
                ))
            } else if enabled_battery_count > bank_batteries_len {
                Err((
                    self,
                    "enabled_battery_count was greater than bank_batteries_len",
                ))
            } else {
                let mut state: Self = self;

                state.0.clear_and_resize(
                    JoltageComputerDimensions {
                        bank_batteries_len,
                        enabled_battery_count,
                    }
                    .into_ivec2()
                        + IVec2::ONE,
                    JoltageComputerCell::DEFAULT,
                );

                Ok(JoltageComputerWithBatteries {
                    state,
                    bank_batteries,
                })
            }
        }
    }

    impl Default for JoltageComputer {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg_attr(test, derive(Debug, PartialEq))]
    struct JoltageComputerCellInputs {
        dimensions: JoltageComputerDimensions,
        joltage_factor: Joltage,
        battery: Battery,

        #[cfg(test)]
        battery_index: BatteryIndex,
    }

    struct JoltageComputerWithBatteries<'b> {
        state: JoltageComputer,
        bank_batteries: &'b [Battery],
    }

    impl<'b> JoltageComputerWithBatteries<'b> {
        fn iter_joltage_factors(initial_joltage_factor: Joltage) -> impl Iterator<Item = Joltage> {
            let mut joltage_factor: Joltage = initial_joltage_factor;

            from_fn(move || {
                let next: Option<Joltage> = Some(joltage_factor);

                joltage_factor.0 *= 10_u64;

                next
            })
        }

        fn iter_enumerated_batteries(
            bank_batteries: &'b [Battery],
            enabled_battery_count: usize,
        ) -> impl Iterator<Item = (BatteryIndex, Battery)> + 'b {
            bank_batteries[..bank_batteries.len() - enabled_battery_count + 1_usize]
                .iter()
                .copied()
                .enumerate()
                .rev()
                .map(|(battery_index, battery)| (battery_index.into(), battery))
        }

        fn get_dimensions(&self) -> JoltageComputerDimensions {
            self.state.get_dimensions()
        }

        fn iter_cell_inputs(&self) -> impl Iterator<Item = JoltageComputerCellInputs> + 'b {
            let dimensions: JoltageComputerDimensions = self.get_dimensions();
            let bank_batteries: &'b [Battery] = self.bank_batteries;

            (1_usize..=dimensions.enabled_battery_count)
                .zip(Self::iter_joltage_factors(Joltage(1_u64)))
                .flat_map(move |(enabled_battery_count, joltage_factor)| {
                    repeat((enabled_battery_count, joltage_factor)).zip(
                        (enabled_battery_count..=dimensions.bank_batteries_len).zip(
                            Self::iter_enumerated_batteries(bank_batteries, enabled_battery_count),
                        ),
                    )
                })
                .map(
                    |(
                        (enabled_battery_count, joltage_factor),
                        (bank_batteries_len, (_battery_index, battery)),
                    )| {
                        JoltageComputerCellInputs {
                            dimensions: JoltageComputerDimensions {
                                bank_batteries_len,
                                enabled_battery_count,
                            },
                            joltage_factor,
                            battery,

                            #[cfg(test)]
                            battery_index: _battery_index,
                        }
                    },
                )
        }

        fn compute_max_joltage_for_cell(&mut self, cell_inputs: &JoltageComputerCellInputs) {
            let dimensions: IVec2 = cell_inputs.dimensions.into();
            let mut cell_a: JoltageComputerCell =
                *self.state.0.get(dimensions - IVec2::ONE).unwrap();

            cell_a.joltage.0 += cell_inputs.joltage_factor.0 * cell_inputs.battery.joltage().0;

            #[cfg(test)]
            cell_a
                .enabled_batteries
                .set(cell_inputs.battery_index.get(), true);

            let cell_b: JoltageComputerCell = *self.state.0.get(dimensions - IVec2::X).unwrap();

            let max_joltage_cell: JoltageComputerCell = if cell_a
                .joltage
                .0
                .cmp(&cell_b.joltage.0)
                .then_with(|| {
                    cell_inputs
                        .dimensions
                        .enabled_battery_count
                        .cmp(&2_usize)
                        .reverse()
                })
                .is_gt()
            {
                cell_a
            } else {
                cell_b
            };

            *self.state.0.get_mut(dimensions).unwrap() = max_joltage_cell;
        }

        fn compute_max_joltage(mut self) -> (JoltageComputer, JoltageComputerCell) {
            for cell_inputs in self.iter_cell_inputs() {
                self.compute_max_joltage_for_cell(&cell_inputs);
            }

            let max_joltage: JoltageComputerCell = self
                .state
                .0
                .get(self.get_dimensions().into())
                .unwrap()
                .clone();

            (self.state, max_joltage)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{joltage::JoltageRaw, Battery as B, Joltage as J, *};

        const SIMPLE_BANK_BATTERIES: &'static [Battery] =
            &[B(0), B(1), B(2), B(3), B(4), B(5), B(6), B(7)];
        const COMPLEX_BANK_BATTERIES: &'static [Battery] =
            &[B(7), B(1), B(5), B(3), B(4), B(2), B(6), B(0)];

        #[test]
        fn test_iter_joltage_factors() {
            for (initial_joltage_factor, joltage_factors) in [
                (J(1), vec![J(1), J(10), J(100), J(1000), J(10000)]),
                (
                    J(100),
                    vec![J(100), J(1000), J(10000), J(100000), J(1000000)],
                ),
                (
                    J(10000),
                    vec![J(10000), J(100000), J(1000000), J(10000000), J(100000000)],
                ),
            ] {
                assert_eq!(
                    JoltageComputerWithBatteries::iter_joltage_factors(initial_joltage_factor)
                        .take(joltage_factors.len())
                        .collect::<Vec<Joltage>>(),
                    joltage_factors
                );
            }
        }

        #[test]
        fn test_iter_enumerated_batteries() {
            macro_rules! enumerated_batteries {
                [ $( $battery:expr ),* $(,)? ] => {
                    vec![ $( (BatteryIndex::from($battery as usize), B($battery)), )* ]
                };
            }

            for (enabled_battery_count, enumerated_batteries) in [
                (1_usize, enumerated_batteries![7, 6, 5, 4, 3, 2, 1, 0]),
                (2_usize, enumerated_batteries![6, 5, 4, 3, 2, 1, 0]),
                (3_usize, enumerated_batteries![5, 4, 3, 2, 1, 0]),
                (6_usize, enumerated_batteries![2, 1, 0]),
                (7_usize, enumerated_batteries![1, 0]),
                (8_usize, enumerated_batteries![0]),
            ] {
                assert_eq!(
                    JoltageComputerWithBatteries::iter_enumerated_batteries(
                        SIMPLE_BANK_BATTERIES,
                        enabled_battery_count
                    )
                    .collect::<Vec<(BatteryIndex, Battery)>>(),
                    enumerated_batteries
                );
            }
        }

        #[test]
        fn test_iter_cell_inputs() {
            let joltage_computer_with_batteries: JoltageComputerWithBatteries<'static> =
                JoltageComputer::new()
                    .try_initialize(COMPLEX_BANK_BATTERIES, 8_usize)
                    .ok()
                    .unwrap();

            macro_rules! cell_inputs {
                [ $( (
                    $bank_batteries_len:expr,
                    $enabled_battery_count:expr,
                    $battery:expr,
                    $battery_index:expr $(,)?
                ) ),* $(,)? ] => { vec![ $( JoltageComputerCellInputs {
                    dimensions: JoltageComputerDimensions {
                        bank_batteries_len: $bank_batteries_len,
                        enabled_battery_count: $enabled_battery_count,
                    },
                    joltage_factor: J((10 as JoltageRaw).pow($enabled_battery_count - 1)),
                    battery: B($battery),
                    battery_index: $battery_index.into()
                }, )* ] };
            }

            assert_eq!(
                joltage_computer_with_batteries
                    .iter_cell_inputs()
                    .collect::<Vec<JoltageComputerCellInputs>>(),
                cell_inputs![
                    (1, 1, 0, 7),
                    (2, 1, 6, 6),
                    (3, 1, 2, 5),
                    (4, 1, 4, 4),
                    (5, 1, 3, 3),
                    (6, 1, 5, 2),
                    (7, 1, 1, 1),
                    (8, 1, 7, 0),
                    (2, 2, 6, 6),
                    (3, 2, 2, 5),
                    (4, 2, 4, 4),
                    (5, 2, 3, 3),
                    (6, 2, 5, 2),
                    (7, 2, 1, 1),
                    (8, 2, 7, 0),
                    (3, 3, 2, 5),
                    (4, 3, 4, 4),
                    (5, 3, 3, 3),
                    (6, 3, 5, 2),
                    (7, 3, 1, 1),
                    (8, 3, 7, 0),
                    (4, 4, 4, 4),
                    (5, 4, 3, 3),
                    (6, 4, 5, 2),
                    (7, 4, 1, 1),
                    (8, 4, 7, 0),
                    (5, 5, 3, 3),
                    (6, 5, 5, 2),
                    (7, 5, 1, 1),
                    (8, 5, 7, 0),
                    (6, 6, 5, 2),
                    (7, 6, 1, 1),
                    (8, 6, 7, 0),
                    (7, 7, 1, 1),
                    (8, 7, 7, 0),
                    (8, 8, 7, 0),
                ]
            );
        }

        #[test]
        fn test_compute_max_joltage_for_cell() {
            let (joltage_computer, max_joltage): (JoltageComputer, JoltageComputerCell) =
                JoltageComputer::new()
                    .try_initialize(COMPLEX_BANK_BATTERIES, 3_usize)
                    .ok()
                    .unwrap()
                    .compute_max_joltage();

            assert_eq!(
                joltage_computer.get_dimensions(),
                JoltageComputerDimensions {
                    bank_batteries_len: 8_usize,
                    enabled_battery_count: 3_usize
                }
            );

            macro_rules! cell {
                ($joltage:expr, $enabled_batteries:expr) => {
                    JoltageComputerCell {
                        joltage: J($joltage),
                        enabled_batteries: BatteryBitArray::new([$enabled_batteries, 0_u64]),
                    }
                };
            }

            macro_rules! cells {
                [ $( ( $joltage:expr, $enabled_batteries:expr ) ),* $( , )? ] => { vec![ $( cell!(
                    $joltage,
                    $enabled_batteries
                ), )* ] };
            }

            slice_assert_eq_break(
                &joltage_computer
                    .0
                    .cells()
                    .iter()
                    .copied()
                    .collect::<Vec<JoltageComputerCell>>(),
                &cells![
                    // enabled_battery_count == 0
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    // enabled_battery_count == 1
                    (0, 0b00000000),
                    (0, 0b10000000),
                    (6, 0b01000000),
                    (6, 0b01000000),
                    (6, 0b01000000),
                    (6, 0b01000000),
                    (6, 0b01000000),
                    (6, 0b01000000),
                    (7, 0b00000001),
                    // enabled_battery_count == 2
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (60, 0b11000000),
                    (60, 0b11000000),
                    (60, 0b11000000),
                    (60, 0b11000000),
                    (60, 0b11000000),
                    (60, 0b11000000),
                    (76, 0b01000001),
                    // enabled_battery_count == 3
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (0, 0b00000000),
                    (260, 0b11100000),
                    (460, 0b11010000),
                    (460, 0b11010000),
                    (560, 0b11000100),
                    (560, 0b11000100),
                    (760, 0b11000001),
                ],
            );

            assert_eq!(max_joltage, cell!(760, 0b11000001));
        }
    }
}

mod bank {
    use {
        super::*,
        nom::{
            combinator::{map, map_opt, success},
            multi::many1_count,
            IResult,
        },
        std::ops::Range,
    };

    pub mod prelude {
        pub use super::Bank;
    }

    #[cfg_attr(test, derive(Debug, PartialEq))]
    pub struct Bank(pub Range<BatteryIndex>);

    impl Bank {
        fn try_get_batteries_len_as_battery_index(batteries: &[Battery]) -> Option<BatteryIndex> {
            BatteryIndex::try_from(batteries.len()).ok()
        }

        fn parse_batteries_len_as_battery_index<'i>(
            batteries: &[Battery],
        ) -> impl Fn(&'i str) -> IResult<&'i str, BatteryIndex> + '_ {
            |input: &'i str| {
                map_opt(success(()), |_| {
                    Self::try_get_batteries_len_as_battery_index(batteries)
                })(input)
            }
        }

        fn get_range(&self) -> Range<usize> {
            self.0.start.get()..self.0.end.get()
        }

        fn get_bank_batteries<'b>(&self, batteries: &'b [Battery]) -> &'b [Battery] {
            &batteries[self.get_range()]
        }

        pub fn parse_with_batteries<'i>(
            batteries: &mut Vec<Battery>,
        ) -> impl FnMut(&'i str) -> IResult<&'i str, Self> + '_ {
            |input: &'i str| {
                let start: BatteryIndex =
                    Self::parse_batteries_len_as_battery_index(batteries)(input)?.1;
                let input: &str = many1_count(|input| {
                    map(Battery::parse, |battery| batteries.push(battery))(input)
                })(input)?
                .0;
                let end: BatteryIndex =
                    Self::parse_batteries_len_as_battery_index(batteries)(input)?.1;

                Ok((input, Self(start..end)))
            }
        }

        pub fn try_compute_max_joltage(
            &self,
            batteries: &[Battery],
            enabled_battery_count: usize,
            joltage_computer: &mut JoltageComputer,
        ) -> Result<Joltage, &'static str> {
            joltage_computer
                .try_compute_max_joltage(self.get_bank_batteries(batteries), enabled_battery_count)
                .map(|joltage_computer_cell| joltage_computer_cell.joltage)
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    batteries: Vec<Battery>,
    banks: Vec<Bank>,
}

impl Solution {
    fn try_compute_max_battery_joltage_sum(&self, enabled_battery_count: usize) -> Option<Joltage> {
        let mut joltage_computer: JoltageComputer = JoltageComputer::new();

        self.banks.iter().try_fold(Joltage(0), |sum, bank| {
            bank.try_compute_max_joltage(
                &self.batteries,
                enabled_battery_count,
                &mut joltage_computer,
            )
            .ok()
            .map(|joltage| Joltage(sum.0 + joltage.0))
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut batteries: Vec<Battery> = Vec::new();

        let (input, banks): (&str, Vec<Bank>) =
            separated_list1(line_ending, Bank::parse_with_batteries(&mut batteries))(input)?;

        Ok((input, Self { batteries, banks }))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_max_battery_joltage_sum(2_usize));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_max_battery_joltage_sum(12_usize));
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
        987654321111111\n\
        811111111111119\n\
        234234234234278\n\
        818181911112111\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                batteries: vec![
                    Battery(9),
                    Battery(8),
                    Battery(7),
                    Battery(6),
                    Battery(5),
                    Battery(4),
                    Battery(3),
                    Battery(2),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(8),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(9),
                    Battery(2),
                    Battery(3),
                    Battery(4),
                    Battery(2),
                    Battery(3),
                    Battery(4),
                    Battery(2),
                    Battery(3),
                    Battery(4),
                    Battery(2),
                    Battery(3),
                    Battery(4),
                    Battery(2),
                    Battery(7),
                    Battery(8),
                    Battery(8),
                    Battery(1),
                    Battery(8),
                    Battery(1),
                    Battery(8),
                    Battery(1),
                    Battery(9),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                    Battery(2),
                    Battery(1),
                    Battery(1),
                    Battery(1),
                ],
                banks: vec![
                    Bank(0_usize.into()..15_usize.into()),
                    Bank(15_usize.into()..30_usize.into()),
                    Bank(30_usize.into()..45_usize.into()),
                    Bank(45_usize.into()..60_usize.into()),
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
    fn test_try_compute_max_battery_joltage_sum() {
        for (index, max_two_battery_joltage_sum) in [Joltage(357)].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_compute_max_battery_joltage_sum(2_usize),
                Some(max_two_battery_joltage_sum)
            );
        }

        for (index, max_twelve_battery_joltage_sum) in
            [Joltage(3121910778619)].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_compute_max_battery_joltage_sum(12_usize),
                Some(max_twelve_battery_joltage_sum)
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
