use {
    aoc_2022::*,
    glam::IVec2,
    std::{
        num::ParseIntError,
        str::{FromStr, Split},
    },
};

#[derive(Debug, PartialEq)]
struct SensorReading {
    sensor: IVec2,
    beacon: IVec2,
}

#[derive(Debug, PartialEq)]
struct InvalidPrefixError<'s> {
    actual: &'s str,
    prefix: &'static str,
}

#[derive(Debug, PartialEq)]
enum ParseComponentError<'s> {
    InvalidPrefix(InvalidPrefixError<'s>),
    FailedToParse(ParseIntError),
}

#[derive(Debug, PartialEq)]
enum ParsePositionError<'s> {
    InvalidPrefix(InvalidPrefixError<'s>),
    NoXToken,
    FailedToParseX(ParseComponentError<'s>),
    NoYToken,
    FailedToParseY(ParseComponentError<'s>),
    ExtraTokenFound(&'s str),
}

#[derive(Debug, PartialEq)]
enum ParseSensorReadingError<'s> {
    NoSensorToken,
    FailedToParseSensor(ParsePositionError<'s>),
    NoBeaconToken,
    FailedToParseBeacon(ParsePositionError<'s>),
    ExtraTokenFound(&'s str),
}

impl SensorReading {
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

    fn manhattan_distance(&self) -> i32 {
        let abs: IVec2 = (self.beacon - self.sensor).abs();

        abs.x + abs.y
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
            None => Ok(Self { sensor, beacon }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum PoiType {
    Sensor,
    Beacon,
    InfluenceRisingEdge,
    InfluenceFallingEdge,
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

    fn count_positions_that_cannot_contain_beacon<U: Fn(IVec2) -> i32, V: Fn(IVec2) -> i32>(
        &mut self,
        query_u: i32,
        get_u: U,
        get_v: V,
    ) -> usize {
        use PoiType::*;

        let Self { readings, pois } = self;

        pois.clear();

        for reading in readings.iter() {
            let manhattan_distance: i32 = reading.manhattan_distance();
            let sensor_u: i32 = get_u(reading.sensor);
            let sensor_v: i32 = get_v(reading.sensor);
            let beacon_u: i32 = get_u(reading.beacon);
            let beacon_v: i32 = get_v(reading.beacon);
            let sensor_to_query_distance: i32 = (sensor_u - query_u).abs();
            let remaining_distance: i32 = manhattan_distance - sensor_to_query_distance;

            if remaining_distance >= 0 {
                pois.push(PointOfInterest {
                    position: sensor_v - remaining_distance,
                    poi_type: InfluenceRisingEdge,
                });
                pois.push(PointOfInterest {
                    position: sensor_v + remaining_distance,
                    poi_type: InfluenceFallingEdge,
                });
            }

            if sensor_u == query_u {
                pois.push(PointOfInterest {
                    position: sensor_v,
                    poi_type: Sensor,
                });
            }

            if beacon_u == query_u {
                pois.push(PointOfInterest {
                    position: beacon_v,
                    poi_type: Beacon,
                });
            }
        }

        pois.sort_unstable();

        let mut count: usize = 0_usize;
        let mut beacon_count: usize = 0_usize;
        let mut active_influences: usize = 0_usize;
        let mut previous_poi: Option<PointOfInterest> = None;

        for identical_pois in (PoiIter {
            pois: &pois,
            index: 0_usize,
        }) {
            let poi: &PointOfInterest = &identical_pois[0_usize];

            if let Some(previous_poi) = previous_poi {
                let position_difference: i32 = poi.position - previous_poi.position;

                if position_difference > 0_i32 {
                    if active_influences > 0_usize {
                        count += position_difference as usize;
                    }
                }
            }

            match poi.poi_type {
                Sensor => {
                    // A sensor will only be present if corresponding rising and falling edges
                    // are also present, and its presence doesn't affect the count.
                    assert_eq!(
                        identical_pois.len(),
                        1_usize,
                        "There is more than 1 sensor at position {}",
                        poi.position
                    );
                }
                Beacon => {
                    // A beacon will only be present if corresponding rising and falling edges
                    // are also present, but its presence does affect the count.
                    assert_ne!(
                        active_influences, 0_usize,
                        "No sensors have active influence at position {}, where there's a beacon",
                        poi.position
                    );
                    beacon_count += 1_usize;
                }
                InfluenceRisingEdge => {
                    active_influences += identical_pois.len();
                }
                InfluenceFallingEdge => {
                    active_influences -= identical_pois.len();

                    if active_influences == 0_usize {
                        // An influence was active for this position, but the next pass/follow-up
                        // logic won't catch that. Account for it now.
                        count += 1_usize;
                    }
                }
            }

            previous_poi = Some(poi.clone());
        }

        if let Some(previous_poi) = previous_poi {
            if !matches!(previous_poi.poi_type, InfluenceFallingEdge) {
                count += 1_usize;
            }
        }

        count - beacon_count
    }

    fn count_positions_that_cannot_contain_beacon_in_row(&mut self, row: i32) -> usize {
        self.count_positions_that_cannot_contain_beacon(row, |v| v.y, |v| v.x)
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

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day15.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match SensorReadings::try_from(input) {
                    Ok(mut sensor_readings) => {
                        println!(
                            "sensor_readings\
                            .count_positions_that_cannot_contain_beacon_in_row(2_000_000_i32) == \
                            {}",
                            sensor_readings
                                .count_positions_that_cannot_contain_beacon_in_row(2_000_000_i32)
                        );
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                },
            )
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn example_sensor_readings() -> SensorReadings {
        macro_rules! readings {
            ($({ s: ($sx:expr, $sy:expr), b: ($bx:expr, $by:expr) }, )*) => {
                vec![
                    $(
                        SensorReading {
                            sensor: IVec2::new($sx, $sy),
                            beacon: IVec2::new($bx, $by),
                        },
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
