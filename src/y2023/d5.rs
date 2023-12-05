use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space0, space1},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{delimited, terminated, tuple},
        Err, IResult,
    },
    std::{cmp::Ordering, collections::VecDeque, iter::Iterator, mem::transmute, ops::Range},
};

#[cfg_attr(test, derive(Debug, PartialEq))]
struct MapEntry {
    destination_start: u32,
    source_start: u32,
    range_len: u32,
}

impl MapEntry {
    fn cmp(&self, source_value: u32) -> Ordering {
        if source_value < self.source_start {
            Ordering::Greater
        } else if source_value - self.source_start < self.range_len {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }

    fn map(&self, source_value: u32) -> u32 {
        source_value - self.source_start + self.destination_start
    }
}

impl Parse for MapEntry {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                parse_integer::<u32>,
                space1,
                parse_integer::<u32>,
                space1,
                parse_integer::<u32>,
            )),
            |(destination_start, _, source_start, _, range_len)| MapEntry {
                destination_start,
                source_start,
                range_len,
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct Map(Vec<MapEntry>);

impl Map {
    fn search_for_map_entry(&self, source_value: u32) -> Result<usize, usize> {
        self.0
            .binary_search_by(|map_entry| map_entry.cmp(source_value))
    }

    fn map(&self, source_value: u32) -> u32 {
        self.search_for_map_entry(source_value)
            .map_or(source_value, |map_entry_index| {
                self.0[map_entry_index].map(source_value)
            })
    }
}

impl Parse for Map {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(MapEntry::parse, opt(line_ending))),
            |mut map_entries| {
                map_entries.sort_by_key(|map_entry| map_entry.source_start);

                Self(map_entries)
            },
        )(input)
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(usize)]
enum MapType {
    SeedToSoil,
    SoilToFertilizer,
    FertilizerToWater,
    WaterToLight,
    LightToTemperature,
    TemperatureToHumidity,
    HumidityToLocation,
    Count,
}

impl MapType {
    const COUNT: usize = MapType::Count as usize;

    const fn is_count(self) -> bool {
        matches!(self, Self::Count)
    }

    const fn tag_str(self) -> &'static str {
        match self {
            MapType::SeedToSoil => "seed-to-soil map:",
            MapType::SoilToFertilizer => "soil-to-fertilizer map:",
            MapType::FertilizerToWater => "fertilizer-to-water map:",
            MapType::WaterToLight => "water-to-light map:",
            MapType::LightToTemperature => "light-to-temperature map:",
            MapType::TemperatureToHumidity => "temperature-to-humidity map:",
            MapType::HumidityToLocation => "humidity-to-location map:",
            MapType::Count => "",
        }
    }

    fn iter() -> Self {
        // SAFETY: The first variant has value 0.
        unsafe { transmute::<usize, Self>(0_usize) }
    }

    fn tag<'i>(self) -> impl Fn(&'i str) -> IResult<&'i str, &'i str> {
        tag(self.tag_str())
    }

    fn next_value(self) -> Option<Self> {
        if self.is_count() {
            None
        } else {
            // SAFETY: `Count` is the highest value variant.
            Some(unsafe { transmute::<usize, Self>(self as usize + 1_usize) })
        }
    }
}

impl Iterator for MapType {
    type Item = Self;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_count() {
            None
        } else {
            let next: Option<Self> = Some(*self);

            // SAFETY: `Count` is the highest value variant.
            *self = self.next_value().unwrap();

            next
        }
    }
}

struct QueueRange {
    start: u32,
    len: u32,
    map_type: MapType,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    seeds: Vec<u32>,
    maps: [Map; MapType::COUNT],
}

impl Solution {
    fn map(&self, source_value: u32, map_range: Range<MapType>) -> u32 {
        let mut value: u32 = source_value;

        for map_type in map_range.start {
            if map_type as usize >= map_range.end as usize {
                break;
            }

            value = self.maps[map_type as usize].map(value);
        }

        value
    }

    fn map_seed_to_location(&self, seed: u32) -> u32 {
        self.map(seed, MapType::iter()..MapType::Count)
    }

    fn iter_locations(&self) -> impl Iterator<Item = u32> + '_ {
        self.seeds
            .iter()
            .copied()
            .map(|seed| self.map_seed_to_location(seed))
    }

    fn min_location(&self) -> u32 {
        self.iter_locations().min().unwrap_or_default()
    }

    fn min_location_for_seed_ranges(&self) -> u32 {
        let mut queue: VecDeque<QueueRange> = self
            .seeds
            .chunks_exact(2_usize)
            .map(|start_and_len| QueueRange {
                start: start_and_len[0_usize],
                len: start_and_len[1_usize],
                map_type: MapType::iter(),
            })
            .collect();

        while queue
            .front()
            .map_or(false, |queue_range| !queue_range.map_type.is_count())
        {
            let mut queue_range = queue.pop_front().unwrap();
            let map: &Map = &self.maps[queue_range.map_type as usize];
            let map_type: MapType = queue_range.map_type.next_value().unwrap();

            while queue_range.len > 0_u32 {
                match map.search_for_map_entry(queue_range.start) {
                    Ok(map_entry_index) => {
                        // The range starts in the map entry
                        let map_entry: &MapEntry = &map.0[map_entry_index];
                        let start: u32 = map_entry.map(queue_range.start);

                        if queue_range.start - map_entry.source_start + queue_range.len
                            <= map_entry.range_len
                        {
                            // The range ends in the map entry
                            queue.push_back(QueueRange {
                                start,
                                len: queue_range.len,
                                map_type,
                            });
                            queue_range.len = 0_u32;
                        } else {
                            // The range extends past the map entry
                            let len: u32 =
                                map_entry.source_start + map_entry.range_len - queue_range.start;

                            queue.push_back(QueueRange {
                                start,
                                len,
                                map_type,
                            });
                            queue_range.start += len;
                            queue_range.len -= len;
                        }
                    }
                    Err(map_entry_index) => {
                        // The range doesn't start in the map entry
                        let start: u32 = queue_range.start;
                        let mut full_range_gets_directly_mapped = true;

                        if map_entry_index < map.0.len() {
                            // There is another entry to compare against
                            let map_entry: &MapEntry = &map.0[map_entry_index];
                            let len: u32 = map_entry.source_start - queue_range.start;

                            if len < queue_range.len {
                                full_range_gets_directly_mapped = false;

                                queue.push_back(QueueRange {
                                    start,
                                    len,
                                    map_type,
                                });
                                queue_range.start += len;
                                queue_range.len -= len;
                            }
                        }

                        if full_range_gets_directly_mapped {
                            queue.push_back(QueueRange {
                                start,
                                len: queue_range.len,
                                map_type,
                            });
                            queue_range.len = 0_u32;
                        }
                    }
                }
            }
        }

        queue
            .into_iter()
            .map(|queue_range| queue_range.start)
            .min()
            .unwrap_or_default()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Solution = Solution::default();

        let (mut input, seeds): (&str, Vec<u32>) = delimited(
            tag("seeds: "),
            many0(terminated(parse_integer::<u32>, space0)),
            tuple((line_ending, line_ending)),
        )(input)?;

        solution.seeds = seeds;

        for map_type in MapType::iter() {
            let (next_input, map): (&str, Map) = delimited(
                tuple((map_type.tag(), line_ending)),
                Map::parse,
                opt(line_ending),
            )(input)?;

            input = next_input;
            solution.maps[map_type as usize] = map;
        }

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.min_location());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.min_location_for_seed_ranges());
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

    macro_rules! map_entry {
        ($destination_start:expr, $source_start:expr, $range_len:expr) => {
            MapEntry {
                destination_start: $destination_start,
                source_start: $source_start,
                range_len: $range_len,
            }
        };
    }

    macro_rules! map {
        [ $( $destination_start:expr, $source_start:expr, $range_len:expr; )* ] => {
            Map(vec![ $( map_entry!($destination_start, $source_start, $range_len), )* ])
        }
    }

    const SOLUTION_STR: &'static str = "\
        seeds: 79 14 55 13\n\
        \n\
        seed-to-soil map:\n\
        50 98 2\n\
        52 50 48\n\
        \n\
        soil-to-fertilizer map:\n\
        0 15 37\n\
        37 52 2\n\
        39 0 15\n\
        \n\
        fertilizer-to-water map:\n\
        49 53 8\n\
        0 11 42\n\
        42 0 7\n\
        57 7 4\n\
        \n\
        water-to-light map:\n\
        88 18 7\n\
        18 25 70\n\
        \n\
        light-to-temperature map:\n\
        45 77 23\n\
        81 45 19\n\
        68 64 13\n\
        \n\
        temperature-to-humidity map:\n\
        0 69 1\n\
        1 0 69\n\
        \n\
        humidity-to-location map:\n\
        60 56 37\n\
        56 93 4\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            seeds: vec![79, 14, 55, 13],
            maps: [
                map![
                    52, 50, 48;
                    50, 98, 2;
                ],
                map![
                    39, 0, 15;
                    0, 15, 37;
                    37, 52, 2;
                ],
                map![
                    42, 0, 7;
                    57, 7, 4;
                    0, 11, 42;
                    49, 53, 8;
                ],
                map![
                    88, 18, 7;
                    18, 25, 70;
                ],
                map![
                    81, 45, 19;
                    68, 64, 13;
                    45, 77, 23;
                ],
                map![
                    1, 0, 69;
                    0, 69, 1;
                ],
                map![
                    60, 56, 37;
                    56, 93, 4;
                ],
            ],
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_map_map() {
        let solution: &Solution = solution();
        let seeds: &[u32] = &solution.seeds;
        let map: &Map = &solution.maps[MapType::SeedToSoil as usize];

        assert_eq!(
            seeds
                .iter()
                .copied()
                .map(|seed| map.map(seed))
                .collect::<Vec<u32>>(),
            vec![81_u32, 14_u32, 57_u32, 13_u32]
        );
    }

    #[test]
    fn test_map_seed_to_location() {
        assert_eq!(
            solution().iter_locations().collect::<Vec<u32>>(),
            vec![82_u32, 43_u32, 86_u32, 35_u32]
        );
    }

    #[test]
    fn test_min_location() {
        assert_eq!(solution().min_location(), 35_u32);
    }

    #[test]
    fn test_min_location_for_seed_ranges() {
        assert_eq!(solution().min_location_for_seed_ranges(), 46_u32);
    }
}
