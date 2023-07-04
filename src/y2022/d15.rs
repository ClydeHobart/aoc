use {
    self::quad_tree::*,
    crate::*,
    glam::IVec2,
    std::{
        iter::Peekable,
        num::ParseIntError,
        ops::Range,
        str::{FromStr, Split},
    },
};

#[cfg(test)]
use std::ops::RangeInclusive;

#[derive(Debug, Default, PartialEq)]
struct Corners([IVec2; 4_usize]);

impl Corners {
    fn all_covered_by_reading(&self, reading: &SensorReading) -> bool {
        reading.contains(self.0[0_usize])
            && reading.contains(self.0[1_usize])
            && reading.contains(self.0[2_usize])
            && reading.contains(self.0[3_usize])
    }

    fn any_covered_by_reading(&self, reading: &SensorReading) -> bool {
        reading.contains(self.0[0_usize])
            || reading.contains(self.0[1_usize])
            || reading.contains(self.0[2_usize])
            || reading.contains(self.0[3_usize])
    }
}

#[derive(Debug, PartialEq)]
struct SensorReading {
    sensor: IVec2,
    beacon: IVec2,
    corners: Corners,
    manhattan_distance: i32,
}

#[derive(Debug, PartialEq)]
pub struct InvalidPrefixError<'s> {
    actual: &'s str,
    prefix: &'static str,
}

#[derive(Debug, PartialEq)]
pub enum ParseComponentError<'s> {
    InvalidPrefix(InvalidPrefixError<'s>),
    FailedToParse(ParseIntError),
}

#[derive(Debug, PartialEq)]
pub enum ParsePositionError<'s> {
    InvalidPrefix(InvalidPrefixError<'s>),
    NoXToken,
    FailedToParseX(ParseComponentError<'s>),
    NoYToken,
    FailedToParseY(ParseComponentError<'s>),
    ExtraTokenFound(&'s str),
}

#[derive(Debug, PartialEq)]
pub enum ParseSensorReadingError<'s> {
    NoSensorToken,
    FailedToParseSensor(ParsePositionError<'s>),
    NoBeaconToken,
    FailedToParseBeacon(ParsePositionError<'s>),
    ExtraTokenFound(&'s str),
}

fn manhattan_distance(a: IVec2, b: IVec2) -> i32 {
    let abs: IVec2 = (a - b).abs();

    abs.x + abs.y
}

impl SensorReading {
    fn new(sensor: IVec2, beacon: IVec2) -> Self {
        let manhattan_distance: i32 = manhattan_distance(sensor, beacon);
        let mut manhattan_distance_spoke: IVec2 = manhattan_distance * IVec2::X;
        let mut corners: Corners = Default::default();

        for corner in corners.0.iter_mut() {
            *corner = sensor + manhattan_distance_spoke;
            manhattan_distance_spoke = manhattan_distance_spoke.perp();
        }

        Self {
            sensor,
            beacon,
            corners,
            manhattan_distance,
        }
    }

    fn parse_component<'s>(
        component_str: &'s str,
        prefix: &'static str,
    ) -> Result<i32, ParseComponentError<'s>> {
        use ParseComponentError::*;

        i32::from_str(validate_prefix(component_str, prefix, |actual| {
            InvalidPrefix(InvalidPrefixError { actual, prefix })
        })?)
        .map_err(FailedToParse)
    }

    fn parse_position<'s>(
        position_str: &'s str,
        prefix: &'static str,
    ) -> Result<IVec2, ParsePositionError<'s>> {
        use ParsePositionError::*;

        let mut component_iter: Split<char> = validate_prefix(position_str, prefix, |actual| {
            InvalidPrefix(InvalidPrefixError { actual, prefix })
        })?
        .split(',');

        let x: i32 = Self::parse_component(component_iter.next().ok_or(NoXToken)?, " x=")
            .map_err(FailedToParseX)?;
        let y: i32 = Self::parse_component(component_iter.next().ok_or(NoYToken)?, " y=")
            .map_err(FailedToParseY)?;

        match component_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(IVec2 { x, y }),
        }
    }

    #[inline(always)]
    fn contains(&self, pos: IVec2) -> bool {
        manhattan_distance(self.sensor, pos) <= self.manhattan_distance
    }
}

impl<'s> TryFrom<&'s str> for SensorReading {
    type Error = ParseSensorReadingError<'s>;

    fn try_from(sensor_reading_str: &'s str) -> Result<Self, Self::Error> {
        use ParseSensorReadingError::*;

        let mut sensor_reading_iter: Split<char> = sensor_reading_str.split(':');

        let sensor: IVec2 = Self::parse_position(
            sensor_reading_iter.next().ok_or(NoSensorToken)?,
            "Sensor at",
        )
        .map_err(FailedToParseSensor)?;
        let beacon: IVec2 = Self::parse_position(
            sensor_reading_iter.next().ok_or(NoBeaconToken)?,
            " closest beacon is at",
        )
        .map_err(FailedToParseBeacon)?;

        match sensor_reading_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(Self::new(sensor, beacon)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(u8)]
enum PoiEntityType {
    Sensor = 1_u8,
    Beacon = 2_u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum PoiInfluenceEdgeType {
    Rising,
    Falling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum PoiType {
    Entity(PoiEntityType),
    InfluenceEdge(PoiInfluenceEdgeType),
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
struct PointOfInterest {
    position: i32,
    poi_type: PoiType,
}

struct PoiIter<'p> {
    pois: &'p [PointOfInterest],
    index: usize,
}

impl<'p> Iterator for PoiIter<'p> {
    type Item = &'p [PointOfInterest];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.pois.len() {
            let current_poi: &PointOfInterest = &self.pois[self.index];

            let start: usize = self.index;
            let end: usize = start
                + self.pois[start..]
                    .iter()
                    .position(|upcoming_poi| *upcoming_poi != *current_poi)
                    .unwrap_or(self.pois.len() - start);

            self.index = end;

            Some(&self.pois[start..end])
        } else {
            None
        }
    }
}

#[derive(Default)]
struct PositionState {
    #[cfg(test)]
    position: i32,
    active_influence: u8,
    rising_edge: u8,
    falling_edge: u8,
    entity: Option<PoiEntityType>,
}

struct PositionIter<'p> {
    poi_iter: Peekable<PoiIter<'p>>,
    position: i32,
    active_influence: u8,
}

impl<'p> Iterator for PositionIter<'p> {
    type Item = PositionState;

    fn next(&mut self) -> Option<Self::Item> {
        use {PoiInfluenceEdgeType::*, PoiType::*};

        if self.poi_iter.peek().is_some() {
            let mut position_state: PositionState = PositionState {
                #[cfg(test)]
                position: self.position,
                active_influence: self.active_influence,
                ..Default::default()
            };

            while let Some(identical_pois) = self.poi_iter.peek() {
                if self.position != identical_pois[0_usize].position {
                    break;
                }

                let identical_pois: &[PointOfInterest] = self.poi_iter.next().unwrap();

                match identical_pois[0_usize].poi_type {
                    Entity(poi_entity_type) => {
                        position_state.entity = Some(poi_entity_type);
                    }
                    InfluenceEdge(Rising) => {
                        position_state.rising_edge = identical_pois.len() as u8;
                        position_state.active_influence += position_state.rising_edge;
                        self.active_influence += position_state.rising_edge;
                    }
                    InfluenceEdge(Falling) => {
                        position_state.falling_edge = identical_pois.len() as u8;
                        self.active_influence -= position_state.falling_edge;
                    }
                }
            }

            self.position += 1_i32;

            Some(position_state)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq)]
struct SensorReadings {
    readings: Vec<SensorReading>,
    pois: Vec<PointOfInterest>,
}

impl SensorReadings {
    fn from_readings(readings: Vec<SensorReading>) -> Self {
        let pois: Vec<PointOfInterest> = Vec::with_capacity(4_usize * readings.len());

        SensorReadings { readings, pois }
    }

    fn get_x(v: IVec2) -> i32 {
        v.x
    }

    fn get_y(v: IVec2) -> i32 {
        v.y
    }

    fn build_pois(&mut self, query_u: i32, u_is_x: bool) {
        use {PoiEntityType::*, PoiInfluenceEdgeType::*, PoiType::*};

        let [get_u, get_v]: [fn(IVec2) -> i32; 2_usize] = if u_is_x {
            [Self::get_x, Self::get_y]
        } else {
            [Self::get_y, Self::get_x]
        };

        self.pois.clear();

        for reading in self.readings.iter() {
            let manhattan_distance: i32 = reading.manhattan_distance;
            let sensor_u: i32 = get_u(reading.sensor);
            let sensor_v: i32 = get_v(reading.sensor);
            let beacon_u: i32 = get_u(reading.beacon);
            let beacon_v: i32 = get_v(reading.beacon);
            let sensor_to_query_distance: i32 = (sensor_u - query_u).abs();
            let remaining_distance: i32 = manhattan_distance - sensor_to_query_distance;

            if remaining_distance >= 0 {
                self.pois.push(PointOfInterest {
                    position: sensor_v - remaining_distance,
                    poi_type: InfluenceEdge(Rising),
                });
                self.pois.push(PointOfInterest {
                    position: sensor_v + remaining_distance,
                    poi_type: InfluenceEdge(Falling),
                });
            }

            if sensor_u == query_u {
                self.pois.push(PointOfInterest {
                    position: sensor_v,
                    poi_type: Entity(Sensor),
                });
            }

            if beacon_u == query_u {
                self.pois.push(PointOfInterest {
                    position: beacon_v,
                    poi_type: Entity(Beacon),
                });
            }
        }

        self.pois.sort_unstable();
    }

    fn position_iter(&self) -> PositionIter {
        let mut poi_iter: Peekable<PoiIter> = PoiIter {
            pois: &self.pois,
            index: 0_usize,
        }
        .peekable();
        let position: i32 = poi_iter
            .peek()
            .map(|identical_pois| identical_pois[0_usize].position)
            .unwrap_or_default();

        PositionIter {
            poi_iter,
            position,
            active_influence: 0_u8,
        }
    }

    fn count_positions_that_cannot_contain_beacon(&mut self, query_u: i32, u_is_x: bool) -> usize {
        use PoiEntityType::Beacon;

        self.build_pois(query_u, u_is_x);

        let mut count: usize = 0_usize;

        for position_state in self.position_iter() {
            count += (position_state.active_influence > 0_u8
                && position_state.entity != Some(Beacon)) as usize;
        }

        count
    }

    fn count_positions_that_cannot_contain_beacon_in_row(&mut self, row: i32) -> usize {
        self.count_positions_that_cannot_contain_beacon(row, false)
    }

    #[cfg(test)]
    fn find_undetected_beacon_in_range_brute_force(
        &mut self,
        x_range: RangeInclusive<i32>,
        y_range: RangeInclusive<i32>,
        mut print_status_updates: bool,
    ) -> Option<IVec2> {
        let mut percent: u8 = 0_u8;
        let mut x_iteration: usize = 0_usize;
        let x_iteration_divisor: usize = (*x_range.end() - *x_range.start()) as usize / 100_usize;

        print_status_updates &= x_iteration_divisor != 0_usize;

        for x in x_range {
            self.build_pois(x, true);

            for position_state in self.position_iter() {
                let y: i32 = position_state.position;

                if y_range.contains(&y) && position_state.active_influence == 0_u8 {
                    return Some(IVec2 { x, y });
                }
            }

            if print_status_updates && x_iteration % x_iteration_divisor == 0_usize {
                println!("{percent}% searched");

                percent += 1_u8;
            }

            x_iteration += 1_usize;
        }

        None
    }

    fn tuning_frequency(pos: IVec2) -> i64 {
        pos.x as i64 * 4_000_000_i64 + pos.y as i64
    }
}

impl<'s> TryFrom<&'s str> for SensorReadings {
    type Error = ParseSensorReadingError<'s>;

    fn try_from(sensor_readings_str: &'s str) -> Result<Self, Self::Error> {
        let mut readings: Vec<SensorReading> = Vec::new();

        for sensor_reading_str in sensor_readings_str.split('\n') {
            readings.push(sensor_reading_str.try_into()?);
        }

        Ok(Self::from_readings(readings))
    }
}

mod quad_tree {
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq)]
    struct IRange2 {
        x: Range<i32>,
        y: Range<i32>,
    }

    impl IRange2 {
        #[inline(always)]
        fn new(x: Range<i32>, y: Range<i32>) -> Self {
            IRange2 { x, y }
        }

        #[inline(always)]
        fn is_empty(&self) -> bool {
            self.x.is_empty() || self.y.is_empty()
        }

        fn subdivide(&self) -> [Self; 4_usize] {
            let x_mid: i32 = self.x.start + (self.x.len() as i32 / 2_i32);
            let y_mid: i32 = self.y.start + (self.y.len() as i32 / 2_i32);

            [
                IRange2::new(self.x.start..x_mid, self.y.start..y_mid),
                IRange2::new(self.x.start..x_mid, y_mid..self.y.end),
                IRange2::new(x_mid..self.x.end, self.y.start..y_mid),
                IRange2::new(x_mid..self.x.end, y_mid..self.y.end),
            ]
        }

        fn corners(&self) -> Corners {
            let x_end: i32 = self.x.end - 1_i32;
            let y_end: i32 = self.y.end - 1_i32;

            Corners([
                IVec2::new(self.x.start, self.y.start),
                IVec2::new(self.x.start, y_end),
                IVec2::new(x_end, self.y.start),
                IVec2::new(x_end, y_end),
            ])
        }

        fn overlaps_reading(&self, reading: &SensorReading) -> bool {
            self.contains(reading.sensor) || self.contains_any_corner(&reading.corners)
        }

        #[inline(always)]
        fn contains_any_corner(&self, corners: &Corners) -> bool {
            self.contains(corners.0[0_usize])
                || self.contains(corners.0[1_usize])
                || self.contains(corners.0[2_usize])
                || self.contains(corners.0[3_usize])
        }

        #[inline(always)]
        fn contains(&self, pos: IVec2) -> bool {
            self.x.contains(&pos.x) && self.y.contains(&pos.y)
        }
    }

    /// Data structure used to keep track efficiently of regions that are fully covered or fully not
    /// covered
    ///
    /// # Invariants
    ///
    /// * `children` is `None` *iff* either `present_sensor_influences == 0_u64` (meaning no sensor
    ///   covers this region) **XOR** collectively the sensors indicated by `present_sensor_influences`
    ///   fully cover the region
    #[derive(Debug, Default)]
    pub(super) struct BeaconPresenceQuadTree {
        range: IRange2,
        corners: Corners,
        present_sensor_influences: u64,
        children: Option<Box<[BeaconPresenceQuadTree; 4_usize]>>,
    }

    #[derive(Debug)]
    pub(super) enum BpqtError {
        EmptyXRange,
        EmptyYRange,
        TooManyReadings,
    }

    impl BeaconPresenceQuadTree {
        pub(super) fn from_readings_and_ranges(
            readings: &[SensorReading],
            x: Range<i32>,
            y: Range<i32>,
        ) -> Result<Self, BpqtError> {
            if x.is_empty() {
                Err(BpqtError::EmptyXRange)
            } else if y.is_empty() {
                Err(BpqtError::EmptyYRange)
            } else if readings.len() > 64_usize {
                Err(BpqtError::TooManyReadings)
            } else {
                let mut root: Self = Default::default();

                root.set_range(IRange2 { x, y });

                for (index, reading) in readings.iter().enumerate() {
                    root.add_reading(index, reading);
                }

                Ok(root)
            }
        }

        fn is_fully_covered(&self) -> bool {
            self.present_sensor_influences != 0_u64 && self.children.is_none()
        }

        fn set_range(&mut self, range: IRange2) {
            self.corners = range.corners();
            self.range = range;

            if self.range.is_empty() {
                self.present_sensor_influences = u64::MAX;
                self.children = None;
            }
        }

        fn add_reading(&mut self, index: usize, reading: &SensorReading) {
            if !self.is_fully_covered()
                && (self.range.overlaps_reading(reading)
                    || self.corners.any_covered_by_reading(reading))
            {
                self.present_sensor_influences |= 1_u64 << index as u32;

                if self.corners.all_covered_by_reading(reading) {
                    self.children = None
                } else {
                    // At least one child overlaps
                    self.init_children();

                    let mut all_children_are_fully_covered: bool = true;

                    for child in self.children.as_mut().unwrap().iter_mut() {
                        child.add_reading(index, reading);

                        all_children_are_fully_covered =
                            all_children_are_fully_covered && child.is_fully_covered()
                    }

                    if all_children_are_fully_covered {
                        self.children = None;
                    }
                }
            }
        }

        fn init_children(&mut self) {
            if self.children.is_none() {
                self.children = Some(Default::default());

                for (range, child) in self
                    .range
                    .subdivide()
                    .into_iter()
                    .zip(self.children.as_mut().unwrap().iter_mut())
                {
                    child.set_range(range);
                }
            }
        }

        pub(super) fn find_gap(&self) -> Option<IVec2> {
            if self.is_fully_covered() {
                None
            } else if let Some(children) = self.children.as_ref() {
                children.iter().find_map(Self::find_gap)
            } else {
                Some(self.corners.0[0_usize])
            }
        }
    }
}

pub struct Solution(SensorReadings);

impl Solution {
    fn count_positions_that_cannot_contain_beacon_in_row(&mut self, row: i32) -> usize {
        self.0
            .count_positions_that_cannot_contain_beacon_in_row(row)
    }

    fn isolate_distress_beacon_tuning_frequency_in_range(
        &self,
        x_range: Range<i32>,
        y_range: Range<i32>,
    ) -> Option<i64> {
        BeaconPresenceQuadTree::from_readings_and_ranges(&self.0.readings, x_range, y_range)
            .ok()
            .and_then(|bpqt| bpqt.find_gap().map(SensorReadings::tuning_frequency))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_positions_that_cannot_contain_beacon_in_row(2_000_000_i32));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self
            .isolate_distress_beacon_tuning_frequency_in_range(0..4_000_001_i32, 0..4_000_001_i32));
    }
}

impl<'s> TryFrom<&'s str> for Solution {
    type Error = ParseSensorReadingError<'s>;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::ops::RangeInclusive};

    const SENSOR_READING_STR: &str = concat!(
        "Sensor at x=2, y=18: closest beacon is at x=-2, y=15\n",
        "Sensor at x=9, y=16: closest beacon is at x=10, y=16\n",
        "Sensor at x=13, y=2: closest beacon is at x=15, y=3\n",
        "Sensor at x=12, y=14: closest beacon is at x=10, y=16\n",
        "Sensor at x=10, y=20: closest beacon is at x=10, y=16\n",
        "Sensor at x=14, y=17: closest beacon is at x=10, y=16\n",
        "Sensor at x=8, y=7: closest beacon is at x=2, y=10\n",
        "Sensor at x=2, y=0: closest beacon is at x=2, y=10\n",
        "Sensor at x=0, y=11: closest beacon is at x=2, y=10\n",
        "Sensor at x=20, y=14: closest beacon is at x=25, y=17\n",
        "Sensor at x=17, y=20: closest beacon is at x=21, y=22\n",
        "Sensor at x=16, y=7: closest beacon is at x=15, y=3\n",
        "Sensor at x=14, y=3: closest beacon is at x=15, y=3\n",
        "Sensor at x=20, y=1: closest beacon is at x=15, y=3",
    );
    const UNDETECTED_BEACON: IVec2 = IVec2::new(14_i32, 11_i32);
    const RANGE_INCLUSIVE: RangeInclusive<i32> = 0_i32..=20_i32;
    const RANGE: Range<i32> = 0_i32..21_i32;
    const TUNING_FREQUENCY: i64 = 56_000_011_i64;

    #[test]
    fn test_sensor_readings_try_from_str() {
        assert_eq!(SENSOR_READING_STR.try_into(), Ok(example_sensor_readings()));
    }

    #[test]
    fn test_count_positions_that_cannot_contain_beacon_in_row() {
        assert_eq!(
            example_sensor_readings().count_positions_that_cannot_contain_beacon_in_row(10),
            26
        );
    }

    #[test]
    fn test_find_undetected_beacon_in_range_brute_force() {
        assert_eq!(
            example_sensor_readings().find_undetected_beacon_in_range_brute_force(
                RANGE_INCLUSIVE,
                RANGE_INCLUSIVE,
                false
            ),
            Some(UNDETECTED_BEACON)
        )
    }

    #[test]
    fn test_tuning_frequency() {
        assert_eq!(
            SensorReadings::tuning_frequency(UNDETECTED_BEACON),
            TUNING_FREQUENCY
        );
    }

    #[test]
    fn test_quad_tree_find_gap() {
        let quad_tree: BeaconPresenceQuadTree = BeaconPresenceQuadTree::from_readings_and_ranges(
            &example_sensor_readings().readings,
            RANGE,
            RANGE,
        )
        .unwrap();

        assert_eq!(quad_tree.find_gap(), Some(UNDETECTED_BEACON))
    }

    #[test]
    fn test_full() {
        assert_eq!(
            SensorReadings::try_from(SENSOR_READING_STR)
                .ok()
                .and_then(|mut sensor_readings| sensor_readings
                    .find_undetected_beacon_in_range_brute_force(
                        RANGE_INCLUSIVE,
                        RANGE_INCLUSIVE,
                        false
                    ))
                .map(SensorReadings::tuning_frequency),
            Some(TUNING_FREQUENCY)
        );
    }

    fn example_sensor_readings() -> SensorReadings {
        macro_rules! readings {
            ($({ s: ($sx:expr, $sy:expr), b: ($bx:expr, $by:expr) }, )*) => {
                vec![
                    $(
                        SensorReading::new(
                            IVec2::new($sx, $sy),
                            IVec2::new($bx, $by),
                        ),
                    )*
                ]
            };
        }

        SensorReadings {
            readings: readings![
                { s: (2, 18), b: (-2, 15) }, // 7
                { s: (9, 16), b: (10, 16) }, // 1
                { s: (13, 2), b: (15, 3) }, // 3
                { s: (12, 14), b: (10, 16) }, // 4
                { s: (10, 20), b: (10, 16) }, // 4
                { s: (14, 17), b: (10, 16) }, // 5
                { s: (8, 7), b: (2, 10) }, // 9
                { s: (2, 0), b: (2, 10) }, // 10
                { s: (0, 11), b: (2, 10) }, // 3
                { s: (20, 14), b: (25, 17) }, // 8
                { s: (17, 20), b: (21, 22) }, // 6
                { s: (16, 7), b: (15, 3) }, // 5
                { s: (14, 3), b: (15, 3) }, // 1
                { s: (20, 1), b: (15, 3) }, // 7
            ],
            pois: Vec::with_capacity(4_usize * 14_usize),
        }
    }
}
