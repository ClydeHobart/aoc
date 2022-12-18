use {
    aoc_2022::*,
    glam::IVec2,
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        mem::MaybeUninit,
        num::ParseIntError,
        ops::Range,
        str::FromStr,
    },
};

///! # Restrictions
///
/// * A valve tag is two capital ASCII characters
/// * Flow rate is at most `u16::MAX`
/// * Any valve has at most 5 neighboring valves
/// * There are at most 64 valves
/// * The sum of all flow rates multiplied by the number of time steps is at most `u16::MAX`

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

const MAX_TIME_STEPS: u16 = 30_u16;

#[derive(Clone, Debug, Default, PartialEq)]
struct ScanLine {
    tag: u16,
    flow_rate: u16,
    neighbors: [u16; 5_usize],
    neighbor_count: u8,
}

impl ScanLine {
    fn parse_tag<'s, E, F: Fn(&'s str) -> E>(tag_str: &'s str, f: F) -> Result<u16, E> {
        if tag_str.len() == 2_usize && tag_str.chars().all(|c| c.is_ascii_uppercase()) {
            let mut bytes: [u8; 2_usize] = Default::default();

            bytes.copy_from_slice(tag_str.as_bytes());

            Ok(u16::from_ne_bytes(bytes))
        } else {
            Err(f(tag_str))
        }
    }

    fn parse_tag_keep_remaining<'s, E, F: Fn(&'s str) -> E>(
        tag_str: &'s str,
        f: F,
    ) -> Result<(u16, &'s str), E> {
        if tag_str.is_char_boundary(2_usize) {
            Self::parse_tag(&tag_str[..2_usize], f).map(|tag| (tag, &tag_str[2_usize..]))
        } else {
            Err(f(tag_str))
        }
    }

    fn parse_neighbor_tags<'s>(
        neighbor_tags_str: &'s str,
        single_expected: bool,
    ) -> Result<([u16; 5_usize], u8), ParseScanLineError<'s>> {
        use ParseScanLineError::*;

        let mut neighbor_list: ([u16; 5_usize], u8) = Default::default();

        for neighbor_tag_str in neighbor_tags_str.split(", ") {
            if neighbor_list.1 as usize == neighbor_list.0.len() {
                return Err(TooManyNeighbors(neighbor_tag_str));
            }

            neighbor_list.0[neighbor_list.1 as usize] =
                Self::parse_tag(neighbor_tag_str, InvalidNeighbor)?;
            neighbor_list.1 += 1_u8;
        }

        if single_expected != (neighbor_list.1 == 1_u8) {
            Err(PluralityMismatch(neighbor_tags_str))
        } else {
            Ok(neighbor_list)
        }
    }

    fn neighbors(&self) -> &[u16] {
        &self.neighbors[..self.neighbor_count as usize]
    }

    #[cfg(test)]
    fn new<const N: usize>(tag: u16, flow_rate: u16, neighbors: [u16; N]) -> Option<Self> {
        if neighbors.len() > 5_usize {
            None
        } else {
            let mut scan_line: Self = Self {
                tag,
                flow_rate,
                neighbor_count: neighbors.len() as u8,
                ..Default::default()
            };

            scan_line.neighbors[..neighbors.len()].copy_from_slice(&neighbors);

            Some(scan_line)
        }
    }
}

#[derive(Debug, PartialEq)]
struct InvalidText<'s> {
    actual: &'s str,
    expected: &'static [&'static str],
}

#[derive(Debug, PartialEq)]
enum ParseScanLineError<'s> {
    InvalidValveText(InvalidText<'s>),
    InvalidValveTag(&'s str),
    InvalidFlowRateText(InvalidText<'s>),
    InvalidFlowRate(ParseIntError),
    InvalidNeighborText(InvalidText<'s>),
    InvalidNeighbor(&'s str),
    TooManyNeighbors(&'s str),
    PluralityMismatch(&'s str),
}

impl<'s> TryFrom<&'s str> for ScanLine {
    type Error = ParseScanLineError<'s>;

    fn try_from(mut scan_line_str: &'s str) -> Result<Self, Self::Error> {
        use ParseScanLineError::*;

        const VALVE_TEXT: &str = "Valve ";
        const FLOW_RATE_TEXT: &str = " has flow rate=";
        const SINGLE_NEIGHBOR: &str = " tunnel leads to valve ";
        const MULTIPLE_NEIGHBORS: &str = " tunnels lead to valves ";

        scan_line_str = validate_prefix(scan_line_str, VALVE_TEXT, |actual| {
            InvalidValveText(InvalidText {
                actual,
                expected: &[VALVE_TEXT],
            })
        })?;

        let (tag, mut scan_line_str) =
            Self::parse_tag_keep_remaining(scan_line_str, InvalidValveTag)?;

        scan_line_str = validate_prefix(scan_line_str, FLOW_RATE_TEXT, |actual| {
            InvalidFlowRateText(InvalidText {
                actual,
                expected: &[FLOW_RATE_TEXT],
            })
        })?;

        let (flow_rate_str, scan_line_str) =
            scan_line_str
                .split_once(';')
                .ok_or(InvalidFlowRateText(InvalidText {
                    actual: scan_line_str,
                    expected: &[";"],
                }))?;

        let flow_rate: u16 = u16::from_str(flow_rate_str).map_err(InvalidFlowRate)?;

        let (neighbors, neighbor_count) =
            match validate_prefix(scan_line_str, SINGLE_NEIGHBOR, |s| s) {
                Ok(neighbor_tags_str) => Self::parse_neighbor_tags(neighbor_tags_str, true),
                Err(scan_line_str) => Self::parse_neighbor_tags(
                    validate_prefix(scan_line_str, MULTIPLE_NEIGHBORS, |actual| {
                        InvalidNeighborText(InvalidText {
                            actual,
                            expected: &[SINGLE_NEIGHBOR, MULTIPLE_NEIGHBORS],
                        })
                    })?,
                    false,
                ),
            }?;

        Ok(Self {
            tag,
            flow_rate,
            neighbors,
            neighbor_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ScanLines {
    lines: Vec<ScanLine>,
    vert_to_edges: Vec<u64>,
}

impl ScanLines {
    fn new(lines: Vec<ScanLine>) -> Self {
        let vert_to_edges: Vec<u64> = Self::vert_to_edges(&lines);

        Self {
            lines,
            vert_to_edges,
        }
    }

    fn vert_to_edges(lines: &Vec<ScanLine>) -> Vec<u64> {
        let mut vert_to_edges: Vec<u64> = Vec::with_capacity(lines.len());

        for scan_line in lines.iter() {
            let mut edges: u64 = 0_u64;

            for neighbor_tag in scan_line.neighbors().iter().copied() {
                if let Some(neighbor_index) = lines
                    .iter()
                    .position(|scan_line| scan_line.tag == neighbor_tag)
                {
                    edges |= 1_u64 << neighbor_index as u32;
                }
            }

            vert_to_edges.push(edges);
        }

        vert_to_edges
    }

    fn useful_trips(&self) -> Result<UsefulTrips, UsefulTripsError> {
        ScanLinesUsefulTripsFinder::useful_trips(self)
    }

    fn unpack_verts(mut verts: u64, verts_vec: &mut Vec<u32>) {
        verts_vec.clear();

        loop {
            let vert: u32 = verts.trailing_zeros();

            if vert >= u64::BITS {
                break;
            }

            verts_vec.push(vert);
            verts &= !(1_u64 << vert);
        }
    }

    fn valve_count(&self) -> usize {
        self.lines.len()
    }
}

#[derive(Debug, PartialEq)]
enum ParseScanLinesError<'s> {
    InvalidScanLine(ParseScanLineError<'s>),
    TooManyValves,
    FlowRateSumTooLarge,
}

impl<'s> TryFrom<&'s str> for ScanLines {
    type Error = ParseScanLinesError<'s>;

    fn try_from(scan_lines_str: &'s str) -> Result<Self, Self::Error> {
        use ParseScanLinesError::*;

        let mut lines: Vec<ScanLine> = Vec::new();

        for scan_line_str in scan_lines_str.split('\n') {
            lines.push(scan_line_str.try_into().map_err(InvalidScanLine)?);
        }

        if lines.len() > 64_usize {
            Err(TooManyValves)
        } else if lines
            .iter()
            .map(|scan_line| (scan_line.flow_rate * MAX_TIME_STEPS) as u64)
            .sum::<u64>()
            > u16::MAX as u64
        {
            Err(FlowRateSumTooLarge)
        } else {
            Ok(ScanLines::new(lines))
        }
    }
}

#[derive(Debug, PartialEq)]
struct ScanLinesUsefulTripsFinder<'s> {
    scan_lines: &'s ScanLines,
    sources: &'s mut Vec<u32>,
    start: u32,
}

impl<'s> ScanLinesUsefulTripsFinder<'s> {
    fn reset_sources(&mut self) {
        self.sources.fill(u32::MAX);
    }

    fn useful_trips(scan_lines: &ScanLines) -> Result<UsefulTrips, UsefulTripsError> {
        use UsefulTripsError::*;

        let valve_count: usize = scan_lines.valve_count();

        let mut useful_trips: UsefulTrips = Default::default();
        let mut sources: Vec<u32> = Vec::with_capacity(valve_count);
        let mut all_dests: u64 = 0_u64;

        sources.resize(valve_count, u32::MAX);

        for start in 0_u32..valve_count as u32 {
            let mut trips_finder: ScanLinesUsefulTripsFinder = ScanLinesUsefulTripsFinder {
                scan_lines,
                sources: &mut sources,
                start,
            };

            trips_finder.reset_sources();
            trips_finder.run().ok_or(()).unwrap_err();

            let mut range: Range<usize> = useful_trips.trips.len()..useful_trips.trips.len();
            let mut start_dests: u64 = 0_u64;

            for dest in 0_u32..valve_count as u32 {
                let dest_index: usize = dest as usize;
                let mut source: u32 = sources[dest_index];

                if dest != start && scan_lines.lines[dest_index].flow_rate != 0_u16 {
                    if source != u32::MAX {
                        let mut verts_between: u64 = 0_u64;

                        // Start with 1 to account for opening the valve, since only after opening
                        // does the valve start releasing pressrue
                        let mut cost: i32 = 1_i32;

                        loop {
                            cost += 1_i32;

                            if source == start {
                                break;
                            } else if source == u32::MAX {
                                return Err(InvalidPathBackToStart { start, valve: dest });
                            }

                            verts_between |= 1_u64 << source;
                            source = sources[source as usize];
                        }

                        useful_trips.trips.push(UsefulTrip {
                            verts_between,
                            dest,
                            cost,
                        });
                        start_dests |= 1_u64 << dest;
                    } else {
                        return Err(InvalidPathBackToStart { start, valve: dest });
                    }
                }
            }

            range.end = useful_trips.trips.len();
            useful_trips.headers.push(UsefulTripHeader {
                dests: start_dests,
                range,
            });
            all_dests |= start_dests;
        }

        useful_trips.dests = all_dests;

        Ok(useful_trips)
    }
}

impl<'s> BreadthFirstSearch for ScanLinesUsefulTripsFinder<'s> {
    type Vertex = u32;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    /// We just want to find a shortest-path tree
    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    /// This is only called when `is_end` returns true
    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!();
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        ScanLines::unpack_verts(self.scan_lines.vert_to_edges[*vertex as usize], neighbors);
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.sources[*to as usize] = *from;
    }
}

/// A trip that provides utility to the goal of releasing as much pressure as possible
///
/// # Run-time Invariants
///
/// * `verts_between.count_ones() + 2_u32 == self.cost`
#[derive(PartialEq)]
struct UsefulTrip {
    verts_between: u64,
    dest: u32,
    cost: i32,
}

impl Debug for UsefulTrip {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut verts_between: Vec<u32> = Vec::with_capacity(self.cost as usize - 2_usize);

        ScanLines::unpack_verts(self.verts_between, &mut verts_between);

        f.debug_struct("UsefulTrip")
            .field("verts_between", &verts_between)
            .field("dest", &self.dest)
            .field("cost", &self.cost)
            .finish()
    }
}

#[derive(PartialEq)]
struct UsefulTripHeader {
    range: Range<usize>,
    dests: u64,
}

impl Debug for UsefulTripHeader {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut dests: Vec<u32> = Vec::with_capacity(self.dests.count_ones() as usize);

        ScanLines::unpack_verts(self.dests, &mut dests);

        f.debug_struct("UsefulTripHeader")
            .field("range", &self.range)
            .field("dests", &dests)
            .finish()
    }
}

#[derive(Debug, Default, PartialEq)]
struct UsefulTrips {
    headers: Vec<UsefulTripHeader>,
    trips: Vec<UsefulTrip>,
    dests: u64,
}

impl UsefulTrips {
    fn get(&self, start: u32) -> &[UsefulTrip] {
        &self.trips[self.headers[start as usize].range.clone()]
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
enum UsefulTripsError {
    InvalidPathBackToStart { start: u32, valve: u32 },
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct PressureReleaseState {
    valves_open: u64,
    pressure_released: i32,

    /// This is a destination index, since only destinations are worth visiting
    next: u32,
}

impl Debug for PressureReleaseState {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut valves_open: Vec<u32> = Vec::with_capacity(self.valves_open.count_ones() as usize);

        ScanLines::unpack_verts(self.valves_open, &mut valves_open);

        f.debug_struct("PressureReleaseState")
            .field("valves_open", &valves_open)
            .field("pressure_released", &self.pressure_released)
            .field("next", &self.next)
            .finish()
    }
}

impl Default for PressureReleaseState {
    fn default() -> Self {
        Self {
            valves_open: 0_u64,
            pressure_released: 0_i32,
            next: u32::MAX,
        }
    }
}

#[derive(Debug, PartialEq)]
struct PrsRange(Range<usize>);

impl PrsRange {
    const INVALID: Self = Self(usize::MAX..usize::MAX);

    fn start(states: &Vec<PressureReleaseState>) -> Self {
        Self(states.len()..states.len())
    }

    fn end(&mut self, states: &Vec<PressureReleaseState>) {
        self.0.end = states.len();
    }
}

impl Default for PrsRange {
    fn default() -> Self {
        Self::INVALID
    }
}

struct PressureReleaseExplorer<'a> {
    scan_lines: &'a [ScanLine],
    useful_trips: &'a UsefulTrips,
    states: Vec<PressureReleaseState>,
    state_ranges: Grid<PrsRange>,
    dests: Vec<u32>,
    index_to_dest_index: Vec<usize>,
    explored_time_steps: usize,
}

impl<'a> PressureReleaseExplorer<'a> {
    fn new(scan_lines: &'a ScanLines, useful_trips: &'a UsefulTrips) -> Self {
        let dest_count: usize = useful_trips.dests.count_ones() as usize;
        let valve_count: usize = scan_lines.valve_count();

        let states: Vec<PressureReleaseState> = Vec::new();
        let state_ranges: Grid<PrsRange> = Grid::default(IVec2::new(dest_count as i32, 0_i32));

        let mut dests: Vec<u32> = Vec::with_capacity(dest_count);

        ScanLines::unpack_verts(useful_trips.dests, &mut dests);

        let mut index_to_dest_index: Vec<usize> = Vec::with_capacity(valve_count);

        index_to_dest_index.resize(valve_count, usize::MAX);

        for (dest_index, dest) in dests.iter().enumerate() {
            index_to_dest_index[*dest as usize] = dest_index;
        }

        Self {
            scan_lines: &scan_lines.lines,
            useful_trips,
            states,
            state_ranges,
            dests,
            index_to_dest_index,
            explored_time_steps: 0_usize,
        }
    }

    fn explore_time_steps(&mut self, time_steps: usize) {
        if time_steps < self.explored_time_steps {
            return;
        }

        self.state_ranges
            .resize_rows(time_steps + 1_usize, PrsRange::default);

        let mut new_states: Vec<PressureReleaseState> = Vec::new();

        for time in self.explored_time_steps..time_steps {
            for state_range_iter in CellIter::until_boundary(
                &self.state_ranges,
                IVec2::new(0_i32, time as i32),
                Direction::East,
            ) {
                self.build_states(
                    self.dests[state_range_iter.x as usize],
                    time,
                    &mut new_states,
                );

                let mut range: PrsRange = PrsRange::start(&self.states);

                self.states.append(&mut new_states);

                range.end(&self.states);

                *self.state_ranges.get_mut(state_range_iter).unwrap() = range;
            }
        }
    }

    fn build_states(&self, start: u32, time: usize, new_states: &mut Vec<PressureReleaseState>) {
        new_states.clear();

        let time: i32 = time as i32;
        let start_valve: u64 = 1_u64 << start;
        let start_valve_pressure_released: i32 =
            self.scan_lines[start as usize].flow_rate as i32 * time;

        // Add a terminal state
        new_states.push(PressureReleaseState {
            valves_open: start_valve,
            pressure_released: start_valve_pressure_released,
            next: u32::MAX,
        });

        for useful_trip in self.useful_trips.get(start) {
            // Make sure 1) there's enough time to travel here and 2) there would be time to stay
            // there, otherwise it wouldn't contribute to the pressure released
            let time_at_dest: i32 = time - useful_trip.cost;

            if time_at_dest > 0_i32 {
                let dest: u32 = useful_trip.dest;
                let dest_usize: usize = dest as usize;

                for dest_state in &self.states[self
                    .state_ranges
                    .get(IVec2::new(
                        self.index_to_dest_index[dest_usize] as i32,
                        time_at_dest,
                    ))
                    .unwrap()
                    .0
                    .clone()]
                {
                    // Make sure that the `dest_state` timeline doesn't activate the `start` valve
                    // later on down the line, since then there would be double-dipping in terms of
                    // accounted-for pressure released
                    if dest_state.valves_open & start_valve == 0_u64 {
                        new_states.push(PressureReleaseState {
                            valves_open: dest_state.valves_open | start_valve,
                            pressure_released: dest_state.pressure_released
                                + start_valve_pressure_released,
                            next: dest,
                        });
                    }
                }
            }
        }

        // Reverse it, so that states will still be grouped together by `valves_open` first, then
        // sorted in descending order of `pressure_released`. The order of `valves_open` doesn't
        // matter
        new_states.sort_unstable_by(|prs_a, prs_b| prs_a.cmp(prs_b).reverse());
        new_states.dedup_by_key(|prs| prs.valves_open);
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day16.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match ScanLines::try_from(input) {
                    Ok(scan_lines) => {
                        let useful_trips: UsefulTrips = scan_lines.useful_trips().unwrap();

                        let mut pressure_release_explorer: PressureReleaseExplorer =
                            PressureReleaseExplorer::new(&scan_lines, &useful_trips);

                        pressure_release_explorer.explore_time_steps(30_usize);

                        let mut states: Vec<PressureReleaseState> = Vec::new();

                        let aa_tag: u16 = u16::from_ne_bytes([b'A', b'A']);
                        let start: u32 = scan_lines
                            .lines
                            .iter()
                            .position(|scan_line| scan_line.tag == aa_tag)
                            .unwrap() as u32;

                        pressure_release_explorer.build_states(start, 30_usize, &mut states);

                        states.sort_by_key(|prs| prs.pressure_released);

                        println!(
                            "states.len() == {}, states.last() == {:#?}",
                            states.len(),
                            states.last()
                        );
                        // optimal pressure_released is not 2107
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                }
            })
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

    const SCAN_LINES_STR: &str = concat!(
        "Valve AA has flow rate=0; tunnels lead to valves DD, II, BB\n",
        "Valve BB has flow rate=13; tunnels lead to valves CC, AA\n",
        "Valve CC has flow rate=2; tunnels lead to valves DD, BB\n",
        "Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE\n",
        "Valve EE has flow rate=3; tunnels lead to valves FF, DD\n",
        "Valve FF has flow rate=0; tunnels lead to valves EE, GG\n",
        "Valve GG has flow rate=0; tunnels lead to valves FF, HH\n",
        "Valve HH has flow rate=22; tunnel leads to valve GG\n",
        "Valve II has flow rate=0; tunnels lead to valves AA, JJ\n",
        "Valve JJ has flow rate=21; tunnel leads to valve II",
    );

    const AA: u16 = u16::from_ne_bytes([b'A', b'A']);
    const BB: u16 = u16::from_ne_bytes([b'B', b'B']);
    const CC: u16 = u16::from_ne_bytes([b'C', b'C']);
    const DD: u16 = u16::from_ne_bytes([b'D', b'D']);
    const EE: u16 = u16::from_ne_bytes([b'E', b'E']);
    const FF: u16 = u16::from_ne_bytes([b'F', b'F']);
    const GG: u16 = u16::from_ne_bytes([b'G', b'G']);
    const HH: u16 = u16::from_ne_bytes([b'H', b'H']);
    const II: u16 = u16::from_ne_bytes([b'I', b'I']);
    const JJ: u16 = u16::from_ne_bytes([b'J', b'J']);

    lazy_static! {
        static ref SCAN_LINES: ScanLines = example_scan_lines();
        static ref USEFUL_TRIPS: UsefulTrips = example_useful_trips();
    }

    #[test]
    fn test_scan_lines_try_from_str() {
        assert_eq!(SCAN_LINES_STR.try_into().as_ref(), Ok(&*SCAN_LINES));
    }

    #[test]
    fn test_scan_lines_to_useful_trips() {
        assert_eq!(SCAN_LINES.useful_trips().as_ref(), Ok(&*USEFUL_TRIPS));
    }

    #[test]
    fn test_pressure_release_explorer() {
        let mut pressure_release_explorer: PressureReleaseExplorer =
            PressureReleaseExplorer::new(&*SCAN_LINES, &*USEFUL_TRIPS);

        pressure_release_explorer.explore_time_steps(30_usize);

        let mut states: Vec<PressureReleaseState> = Vec::new();

        pressure_release_explorer.build_states(0_u32, 30_usize, &mut states);

        states.sort_by_key(|prs| prs.pressure_released);

        assert_eq!(states.last().unwrap().pressure_released, 1651);
    }

    fn example_scan_lines() -> ScanLines {
        macro_rules! scan_lines {
            [ $( $tag:expr , $flow_rate:expr => $( $neighbor:expr ),* ;)* ] => {
                ScanLines::new(vec![
                    $( ScanLine::new($tag, $flow_rate, [ $( $neighbor ),* ]).unwrap(), )*
                ])
            };
        }

        scan_lines!(
            AA, 0 => DD, II, BB;
            BB, 13 => CC, AA;
            CC, 2 => DD, BB;
            DD, 20 => CC, AA, EE;
            EE, 3 => FF, DD;
            FF, 0 => EE, GG;
            GG, 0 => FF, HH;
            HH, 22 => GG;
            II, 0 => AA, JJ;
            JJ, 21 => II;
        )
    }

    fn example_useful_trips() -> UsefulTrips {
        let tag_to_index_and_mask: HashMap<u16, (u32, u64)> =
            vec![AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]
                .into_iter()
                .enumerate()
                .map(|(index, tag)| (tag, (index as u32, 1_u64 << index as u32)))
                .collect();

        let pack_tags = |tags: &[u16]| -> u64 {
            let mut packed: u64 = 0_u64;

            for tag in tags {
                packed |= tag_to_index_and_mask[tag].1;
            }

            packed
        };

        macro_rules! useful_trips {
            [ $( $present_dest_tags:expr => [ $( $verts_between_tags:expr => ($dest_tag:expr, $cost:expr) ,)* ] ,)* ] => {
                vec![ $(
                    (
                        pack_tags(&$present_dest_tags),

                        vec![ $(
                            UsefulTrip {
                                verts_between: pack_tags(&$verts_between_tags),
                                dest: tag_to_index_and_mask[&$dest_tag].0,
                                cost: $cost
                            },
                        )* ]
                    ),
                )* ]
            };
        }

        let vert_to_trips: Vec<(u64, Vec<UsefulTrip>)> = useful_trips![
            // AA
            [BB, CC, DD, EE, HH, JJ] => [
                [] => (BB, 2),
                [BB] => (CC, 3),
                [] => (DD, 2),
                [DD] => (EE, 3),
                [DD, EE, FF, GG] => (HH, 6),
                [II] => (JJ, 3),
            ],
            // BB
            [CC, DD, EE, HH, JJ] => [
                [] => (CC, 2),
                [AA] => (DD, 3),
                [AA, DD] => (EE, 4),
                [AA, DD, EE, FF, GG] => (HH, 7),
                [AA, II] => (JJ, 4),
            ],
            // CC
            [BB, DD, EE, HH, JJ] => [
                [] => (BB, 2),
                [] => (DD, 2),
                [DD] => (EE, 3),
                [DD, EE, FF, GG] => (HH, 6),
                [BB, AA, II] => (JJ, 5),
            ],
            // DD
            [BB, CC, EE, HH, JJ] => [
                [AA] => (BB, 3),
                [] => (CC, 2),
                [] => (EE, 2),
                [EE, FF, GG] => (HH, 5),
                [AA, II] => (JJ, 4),
            ],
            // EE
            [BB, CC, DD, HH, JJ] => [
                [DD, AA] => (BB, 4),
                [DD] => (CC, 3),
                [] => (DD, 2),
                [FF, GG] => (HH, 4),
                [DD, AA, II] => (JJ, 5),
            ],
            // FF
            [BB, CC, DD, EE, HH, JJ] => [
                [EE, DD, AA] => (BB, 5),
                [EE, DD] => (CC, 4),
                [EE] => (DD, 3),
                [] => (EE, 2),
                [GG] => (HH, 3),
                [EE, DD, AA, II] => (JJ, 6),
            ],
            // GG
            [BB, CC, DD, EE, HH, JJ] => [
                [FF, EE, DD, AA] => (BB, 6),
                [FF, EE, DD] => (CC, 5),
                [FF, EE] => (DD, 4),
                [FF] => (EE, 3),
                [] => (HH, 2),
                [FF, EE, DD, AA, II] => (JJ, 7),
            ],
            // HH
            [BB, CC, DD, EE, JJ] => [
                [GG, FF, EE, DD, AA] => (BB, 7),
                [GG, FF, EE, DD] => (CC, 6),
                [GG, FF, EE] => (DD, 5),
                [GG, FF] => (EE, 4),
                [GG, FF, EE, DD, AA, II] => (JJ, 8),
            ],
            // II
            [BB, CC, DD, EE, HH, JJ] => [
                [AA] => (BB, 3),
                [AA, BB] => (CC, 4),
                [AA] => (DD, 3),
                [AA, DD] => (EE, 4),
                [AA, DD, EE, FF, GG] => (HH, 7),
                [] => (JJ, 2),
            ],
            // JJ
            [BB, CC, DD, EE, HH] => [
                [II, AA] => (BB, 4),
                [II, AA, BB] => (CC, 5),
                [II, AA] => (DD, 4),
                [II, AA, DD] => (EE, 5),
                [II, AA, DD, EE, FF, GG] => (HH, 8),
            ],
        ];

        let mut useful_trips: UsefulTrips = Default::default();
        let mut all_dests: u64 = 0_u64;

        for (start_dests, mut trips) in vert_to_trips {
            let mut range: Range<usize> = useful_trips.trips.len()..useful_trips.trips.len();

            useful_trips.trips.append(&mut trips);
            range.end = useful_trips.trips.len();
            useful_trips.headers.push(UsefulTripHeader {
                range,
                dests: start_dests,
            });
            all_dests |= start_dests;
        }

        useful_trips.dests = all_dests;

        useful_trips
    }
}
