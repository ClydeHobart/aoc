use {
    aoc::*,
    glam::{IVec2, IVec3, Vec3Swizzles},
    std::{
        cmp::Ordering,
        collections::VecDeque,
        fmt::{Debug, Formatter, Result as FmtResult},
        iter::Peekable,
        num::ParseIntError,
        ops::Range,
        slice::Iter,
        str::FromStr,
    },
};

///! # Restrictions
///
/// * A valve tag is two capital ASCII characters
/// * Flow rate is at most `u16::MAX`
/// * Any valve has at most 5 neighboring valves
/// * There are at most 64 valves
///
/// # Notes
///
/// This was a really taxing problem. My first solution to the second question worked properly for
/// the test input, but took too long and too much memory to produce an answer for my input. After
/// doing some investigation online, I heard that some input is fine to just strip out all the
/// destinations that the first actor visited, then re-run the search again for the second actor.
/// My implementation of this approach does not work with the test input, but it does for my input.
/// This doesn't feel amazing, but I don't have the time right now to implement a more correct
/// solution.

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

        // Build up the useful trips from a start
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
                            index: dest,
                            cost,
                        });
                        start_dests |= 1_u64 << dest;
                    } else {
                        return Err(InvalidPathBackToStart { start, valve: dest });
                    }
                }
            }

            range.end = useful_trips.trips.len();
            useful_trips.headers_from_start.push(UsefulTripHeader {
                range,
                indices: start_dests,
            });
            all_dests |= start_dests;
        }

        useful_trips.dests = all_dests;

        // Build up the useful trips to a destination
        useful_trips.init_headers_to_dest();

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
#[derive(Clone, PartialEq)]
struct UsefulTrip {
    verts_between: u64,
    index: u32,
    cost: i32,
}

impl Debug for UsefulTrip {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut verts_between: Vec<u32> = Vec::with_capacity(self.cost as usize - 2_usize);

        ScanLines::unpack_verts(self.verts_between, &mut verts_between);

        f.debug_struct("UsefulTrip")
            .field("verts_between", &verts_between)
            .field("index", &self.index)
            .field("cost", &self.cost)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
struct UsefulTripHeader {
    range: Range<usize>,
    indices: u64,
}

impl Debug for UsefulTripHeader {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut dests: Vec<u32> = Vec::with_capacity(self.indices.count_ones() as usize);

        ScanLines::unpack_verts(self.indices, &mut dests);

        f.debug_struct("UsefulTripHeader")
            .field("range", &self.range)
            .field("dests", &dests)
            .finish()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct UsefulTrips {
    /// `index` is the start, `headers_from_start[index].index` is the destination
    headers_from_start: Vec<UsefulTripHeader>,

    /// `index` is the destination, `headers_to_dest[index].index` is the start
    headers_to_dest: Vec<UsefulTripHeader>,
    trips: Vec<UsefulTrip>,
    dests: u64,
}

impl UsefulTrips {
    fn get_from_start<'a>(&'a self, start: u32) -> impl Iterator<Item = &'a UsefulTrip> {
        self.get(&self.headers_from_start[start as usize])
    }

    fn get_to_dest<'a>(&'a self, dest: u32) -> impl Iterator<Item = &'a UsefulTrip> {
        self.get(&self.headers_to_dest[dest as usize])
    }

    fn get<'a>(&'a self, header: &'a UsefulTripHeader) -> impl Iterator<Item = &'a UsefulTrip> {
        self.trips[header.range.clone()]
            .iter()
            .filter(|useful_trip| header.indices & (1_u64 << useful_trip.index) != 0_u64)
    }

    fn init_headers_to_dest(&mut self) {
        for dest in 0_u32..self.headers_from_start.len() as u32 {
            let dest_mask: u64 = 1_u64 << dest;
            let mut range: Range<usize> = self.trips.len()..self.trips.len();
            let mut dest_starts: u64 = 0_u64;

            for (start, start_to_header) in self.headers_from_start.iter().enumerate() {
                if start_to_header.indices & dest_mask != 0_u64 {
                    dest_starts |= 1_u64 << start as u32;

                    let mut useful_trip: UsefulTrip = self.trips[start_to_header.range.clone()]
                        .iter()
                        .find(|useful_trip| useful_trip.index == dest)
                        .unwrap()
                        .clone();

                    useful_trip.index = start as u32;
                    self.trips.push(useful_trip);
                }
            }

            range.end = self.trips.len();
            self.headers_to_dest.push(UsefulTripHeader {
                range,
                indices: dest_starts,
            });
        }
    }

    fn trim(&mut self, path: &Vec<u32>) {
        let mut path_iter: Peekable<Iter<u32>> = path.iter().peekable();
        let mut blocked_starts: u64 = 0_u64;
        let mut blocked_dests: u64 = 0_u64;

        while let Some(start) = path_iter.next() {
            let start: u32 = *start;

            if let Some(dest) = path_iter.peek() {
                let dest: u32 = **dest;

                blocked_starts |= 1_u64 << start;
                blocked_dests |= 1_u64 << dest;
            }
        }

        let allowed_starts: u64 = !blocked_starts;
        let allowed_dests: u64 = !blocked_dests;

        for (header_from_start, header_to_dest) in self
            .headers_from_start
            .iter_mut()
            .zip(self.headers_to_dest.iter_mut())
        {
            header_from_start.indices &= allowed_dests;
            header_to_dest.indices &= allowed_starts;
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
enum UsefulTripsError {
    InvalidPathBackToStart { start: u32, valve: u32 },
}

struct DestInfo {
    dests: Vec<u32>,
    index_to_dest_index: Vec<usize>,
}

impl DestInfo {
    fn new(valve_count: usize, dests_packed: u64) -> Self {
        let mut dests: Vec<u32> = Vec::with_capacity(dests_packed.count_ones() as usize);

        ScanLines::unpack_verts(dests_packed, &mut dests);

        let mut index_to_dest_index: Vec<usize> = Vec::with_capacity(valve_count);

        index_to_dest_index.resize(valve_count, usize::MAX);

        for (dest_index, dest) in dests.iter().enumerate() {
            index_to_dest_index[*dest as usize] = dest_index;
        }

        Self {
            dests,
            index_to_dest_index,
        }
    }
}

mod question_1 {
    use super::*;

    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) struct PressureReleaseState {
        pub valves_open: u64,
        pub pressure_released: i32,

        /// This is a destination index, since only destinations are worth visiting
        pub next: u32,
    }

    impl Debug for PressureReleaseState {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            let mut valves_open: Vec<u32> =
                Vec::with_capacity(self.valves_open.count_ones() as usize);

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

    pub(super) struct PressureReleaseExplorer<'a> {
        scan_lines: &'a [ScanLine],
        useful_trips: &'a UsefulTrips,
        states: Vec<PressureReleaseState>,
        state_ranges: Grid2D<Range<usize>>,
        dest_info: DestInfo,
        explored_time_steps: usize,
    }

    impl<'a> PressureReleaseExplorer<'a> {
        pub(super) fn new(scan_lines: &'a ScanLines, useful_trips: &'a UsefulTrips) -> Self {
            let dest_count: usize = useful_trips.dests.count_ones() as usize;
            let valve_count: usize = scan_lines.valve_count();

            let states: Vec<PressureReleaseState> = Vec::new();
            let state_ranges: Grid2D<Range<usize>> =
                Grid2D::default(IVec2::new(dest_count as i32, 0_i32));

            let dest_info: DestInfo = DestInfo::new(valve_count, useful_trips.dests);

            Self {
                scan_lines: &scan_lines.lines,
                useful_trips,
                states,
                state_ranges,
                dest_info,
                explored_time_steps: 0_usize,
            }
        }

        pub(super) fn explore_time_steps(&mut self, time_steps: usize) {
            if time_steps < self.explored_time_steps {
                return;
            }

            self.state_ranges
                .resize_rows(time_steps + 1_usize, Default::default);

            let mut new_states: Vec<PressureReleaseState> = Vec::new();

            for time in self.explored_time_steps..time_steps {
                for state_range_iter in CellIter2D::until_boundary(
                    &self.state_ranges,
                    IVec2::new(0_i32, time as i32),
                    Direction::East,
                ) {
                    self.build_states(
                        self.dest_info.dests[state_range_iter.x as usize],
                        time,
                        &mut new_states,
                    );

                    let mut range: Range<usize> = self.states.len()..self.states.len();

                    self.states.append(&mut new_states);

                    range.end = self.states.len();

                    *self.state_ranges.get_mut(state_range_iter).unwrap() = range;
                }
            }

            self.explored_time_steps = time_steps;
        }

        fn build_states(
            &self,
            start: u32,
            time: usize,
            new_states: &mut Vec<PressureReleaseState>,
        ) {
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

            for useful_trip in self.useful_trips.get_from_start(start) {
                // Make sure 1) there's enough time to travel here and 2) there would be time to stay
                // there, otherwise it wouldn't contribute to the pressure released
                let time_at_dest: i32 = time - useful_trip.cost;

                if time_at_dest > 0_i32 {
                    let dest: u32 = useful_trip.index;
                    let dest_usize: usize = dest as usize;

                    for dest_state in &self.states[self
                        .state_ranges
                        .get(IVec2::new(
                            self.dest_info.index_to_dest_index[dest_usize] as i32,
                            time_at_dest,
                        ))
                        .unwrap()
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

        pub(super) fn find_optimal_path(
            &self,
            start: u32,
            time: usize,
        ) -> Vec<PressureReleaseState> {
            let mut states: Vec<PressureReleaseState> = Vec::new();

            self.build_states(start, time, &mut states);

            states.sort_by_key(|prs| prs.pressure_released);

            states
                .last()
                .cloned()
                .map(|prs| self.get_stateful_path(start, time, prs))
                .unwrap_or_default()
        }

        fn get_stateful_path(
            &self,
            mut start: u32,
            mut time: usize,
            mut pressure_release_state: PressureReleaseState,
        ) -> Vec<PressureReleaseState> {
            let mut path: Vec<PressureReleaseState> = Vec::new();

            loop {
                path.push(pressure_release_state.clone());

                if pressure_release_state.next == u32::MAX {
                    break;
                }

                let useful_trip_option: Option<&UsefulTrip> = self
                    .useful_trips
                    .get_from_start(start)
                    .find(|useful_trip| useful_trip.index == pressure_release_state.next);

                assert!(
                    useful_trip_option.is_some(),
                    "Could not find useful trip from {start} to {}\n\
                    self.useful_trips.get_from_start(start): {:#?}\n\
                    path: {path:#?}",
                    pressure_release_state.next,
                    self.useful_trips
                        .get_from_start(start)
                        .collect::<Vec<&UsefulTrip>>()
                );

                let useful_trip: &UsefulTrip = useful_trip_option.unwrap();
                let pressure_released: i32 = pressure_release_state.pressure_released
                    - self.scan_lines[start as usize].flow_rate as i32 * time as i32;
                let valves_open: u64 = pressure_release_state.valves_open
                    & !(1_u64 << start)
                    & self.useful_trips.dests;

                time -= useful_trip.cost as usize;
                start = pressure_release_state.next;

                let dest: i32 = self.dest_info.index_to_dest_index[start as usize] as i32;

                let mut found_match: bool = false;
                for state in self.states[self
                    .state_ranges
                    .get(IVec2::new(dest, time as i32))
                    .unwrap()
                    .clone()]
                .iter()
                {
                    if state.pressure_released == pressure_released
                        && state.valves_open == valves_open
                    {
                        pressure_release_state = state.clone();
                        found_match = true;

                        break;
                    }
                }

                if !found_match {
                    let mut valves_open_list: Vec<u32> =
                        Vec::with_capacity(valves_open.count_ones() as usize);

                    ScanLines::unpack_verts(valves_open, &mut valves_open_list);

                    let candidates: &[PressureReleaseState] = &self.states[self
                        .state_ranges
                        .get(IVec2::new(dest, time as i32))
                        .unwrap()
                        .clone()];

                    assert!(
                        found_match,
                        "Couldn't find state from index {start} (destination index {dest}) at \
                        time {time} with pressure released {pressure_released} and valves open \
                        {valves_open_list:#?}\n\
                        path: {path:#?}\n\
                        candidates (count {}): {candidates:#?}",
                        candidates.len()
                    );
                }
            }

            path
        }

        pub(super) fn get_stateless_path(
            start: u32,
            stateful_path: &Vec<PressureReleaseState>,
        ) -> Vec<u32> {
            [start]
                .into_iter()
                .chain(stateful_path.iter().map(|prs| prs.next))
                .take(stateful_path.len())
                .collect()
        }
    }
}

mod question_2_attempt_1 {
    #![allow(dead_code, unused_variables)]

    use super::*;

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) struct ActorState {
        index: u8,
        travel_remaining: u8,
    }

    impl ActorState {
        #[inline(always)]
        fn is_in_transit(self) -> bool {
            self.travel_remaining != 0_u8
        }

        #[inline(always)]
        fn effectively_eq(self, other: Self) -> bool {
            self.travel_remaining == 0_u8 && other.travel_remaining == 0_u8
                || self.index == other.index && self.travel_remaining == other.travel_remaining
        }
    }

    impl Default for ActorState {
        fn default() -> Self {
            Self {
                index: u8::MAX,
                travel_remaining: 0_u8,
            }
        }
    }

    #[derive(Clone, Default, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) struct BiActorState {
        pub valves_open: u64,
        pub pressure_released: u16,
        pub actor_a: ActorState,
        pub actor_b: ActorState,
    }

    impl BiActorState {
        #[inline(always)]
        fn effectively_eq(&self, other: &Self) -> bool {
            self.valves_open == other.valves_open
                && (self.actor_a.effectively_eq(other.actor_a)
                    && self.actor_b.effectively_eq(other.actor_b)
                    || self.actor_a.effectively_eq(other.actor_b)
                        && self.actor_b.effectively_eq(other.actor_a))
        }
    }

    impl Debug for BiActorState {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            let mut valves_open: Vec<u32> =
                Vec::with_capacity(self.valves_open.count_ones() as usize);

            ScanLines::unpack_verts(self.valves_open, &mut valves_open);

            f.debug_struct("BiActorState")
                .field("valves_open", &valves_open)
                .field("pressure_released", &self.pressure_released)
                .field("actor_a", &self.actor_a)
                .field("actor_b", &self.actor_b)
                .finish()
        }
    }

    pub(super) struct BiActorExplorer<'a> {
        scan_lines: &'a [ScanLine],
        useful_trips: &'a UsefulTrips,
        states: VecDeque<BiActorState>,
        state_ranges: Grid3D<Range<usize>>,
        dest_info: DestInfo,
        explored_time_steps: usize,
    }

    /// A representation of an optimal timeline. Since this isn't necessarily for indices that are
    /// both destionations, this attempts to hijack an existing timeline.
    #[allow(dead_code)]
    #[derive(Clone, Debug, Default)]
    pub(super) struct OptimalTimeline {
        /// The state indicating where each actor will travel to. Note that `state.valves_open` will
        /// not contain the bits for the two start states. Support for that can be added upon
        /// demand ;)
        state: BiActorState,

        /// The position in `state_ranges` that this timeline was found in
        chain_entrance: IVec3,

        /// The state within `states[state_ranges.get(&chain_entrance).unwrap().clone()]` that
        /// corresponds to the timeline
        state_index: usize,
    }

    impl OptimalTimeline {
        pub fn is_valid(&self) -> bool {
            self.state != BiActorState::default()
        }

        pub fn get_state<'b, 'a: 'b>(
            &self,
            bi_actor_explorer: &'b BiActorExplorer<'a>,
        ) -> Option<&'b BiActorState> {
            if self.is_valid() {
                Some(
                    &bi_actor_explorer.states[bi_actor_explorer
                        .get_range(&self.chain_entrance)
                        .start
                        + self.state_index],
                )
            } else {
                None
            }
        }
    }

    pub(super) struct Frame {
        dest_index_a_dest_index_b_time: IVec3,
        state_index: usize,
        state: BiActorState,
    }

    impl Debug for Frame {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            f.debug_struct("Frame")
                .field("dest_index_a", &self.dest_index_a_dest_index_b_time.x)
                .field("dest_index_a", &self.dest_index_a_dest_index_b_time.y)
                .field("time", &self.dest_index_a_dest_index_b_time.z)
                .field("state_index", &self.state_index)
                .field("state", &self.state)
                .finish()
        }
    }

    impl<'a> BiActorExplorer<'a> {
        pub(super) fn new(scan_lines: &'a ScanLines, useful_trips: &'a UsefulTrips) -> Self {
            let dests_packed: u64 = useful_trips.dests;
            let dest_count: usize = dests_packed.count_ones() as usize;
            let valve_count: usize = scan_lines.valve_count();

            let states: VecDeque<BiActorState> = VecDeque::new();
            let state_ranges: Grid3D<Range<usize>> =
                Grid3D::default(IVec3::new(dest_count as i32, dest_count as i32, 1_i32));

            let dest_info: DestInfo = DestInfo::new(valve_count, dests_packed);

            let mut bi_actor_explorer: BiActorExplorer = Self {
                scan_lines: &scan_lines.lines,
                useful_trips,
                states,
                state_ranges,
                dest_info,
                explored_time_steps: 0_usize,
            };

            bi_actor_explorer.init_terminal_states();

            bi_actor_explorer
        }

        /// See `OptimalTimeline` comments
        pub(super) fn find_optimal_timeline(
            &mut self,
            start_a: u32,
            start_b: u32,
            time_steps: usize,
            status_updates: bool,
        ) -> OptimalTimeline {
            self.explore_time_steps(time_steps, status_updates);

            let time: i32 = time_steps as i32;

            let mut candidates: Vec<OptimalTimeline> = Vec::new();

            for useful_trip_a in self.useful_trips.get_from_start(start_a) {
                let dest_a: u32 = useful_trip_a.index;
                let dest_index_a: i32 = self.dest_info.index_to_dest_index[dest_a as usize] as i32;
                let cost_a: i32 = useful_trip_a.cost;
                let actor_a: ActorState = ActorState {
                    index: dest_a as u8,
                    travel_remaining: 0_u8,
                };

                for useful_trip_b in self.useful_trips.get_from_start(start_b) {
                    let dest_b: u32 = useful_trip_b.index;

                    if dest_a != dest_b {
                        let dest_index_b: i32 =
                            self.dest_info.index_to_dest_index[dest_b as usize] as i32;
                        let cost_b: i32 = useful_trip_b.cost;
                        let actor_b: ActorState = ActorState {
                            index: dest_b as u8,
                            travel_remaining: 0_u8,
                        };

                        match cost_a.cmp(&cost_b) {
                            Ordering::Equal => {
                                // Find states at time `time - cost_a`
                                let chain_entrance: IVec3 =
                                    IVec3::new(dest_index_a, dest_index_b, time - cost_a);

                                for (state_index, state) in self
                                    .states
                                    .range(self.get_range(&chain_entrance))
                                    .enumerate()
                                {
                                    if !state.actor_a.is_in_transit()
                                        && !state.actor_b.is_in_transit()
                                    {
                                        candidates.push(OptimalTimeline {
                                            state: BiActorState {
                                                actor_a,
                                                actor_b,
                                                ..state.clone()
                                            },
                                            chain_entrance,
                                            state_index,
                                        });
                                    }
                                }
                            }
                            Ordering::Less => {
                                // Hijack states at depth `time - cost_b`
                                let chain_entrance: IVec3 =
                                    IVec3::new(dest_index_a, dest_index_b, time - cost_b);
                                let travel_remaining_a: u8 = (cost_b - cost_a) as u8;

                                for (state_index, state) in self
                                    .states
                                    .range(self.get_range(&chain_entrance))
                                    .enumerate()
                                {
                                    if !state.actor_b.is_in_transit()
                                        && state.actor_a.travel_remaining == travel_remaining_a
                                    {
                                        candidates.push(OptimalTimeline {
                                            state: BiActorState {
                                                actor_a,
                                                actor_b,
                                                ..state.clone()
                                            },
                                            chain_entrance,
                                            state_index,
                                        });
                                    }
                                }
                            }
                            Ordering::Greater => {
                                // Hijack a state at depth `time - cost_a`
                                let chain_entrance: IVec3 =
                                    IVec3::new(dest_index_a, dest_index_b, time - cost_a);
                                let travel_remaining_b: u8 = (cost_a - cost_b) as u8;

                                for (state_index, state) in self
                                    .states
                                    .range(self.get_range(&chain_entrance))
                                    .enumerate()
                                {
                                    if !state.actor_a.is_in_transit()
                                        && state.actor_b.travel_remaining == travel_remaining_b
                                    {
                                        candidates.push(OptimalTimeline {
                                            state: BiActorState {
                                                actor_a,
                                                actor_b,
                                                ..state.clone()
                                            },
                                            chain_entrance,
                                            state_index,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            candidates.sort_unstable_by(|ot_a, ot_b| {
                ot_b.state
                    .pressure_released
                    .cmp(&ot_a.state.pressure_released)
            });

            candidates.first().cloned().unwrap_or_default()
        }

        fn get_range(&self, pos: &IVec3) -> Range<usize> {
            self.state_ranges.get(&pos).unwrap().clone()
        }

        /// This function doesn't (yet) work, but it might not be needed
        #[deprecated]
        pub(super) fn get_timeline(&self, optimal_timeline: &OptimalTimeline) -> Vec<Frame> {
            let mut timeline: Vec<Frame> =
                Vec::with_capacity(optimal_timeline.chain_entrance.z as usize + 1_usize);

            if optimal_timeline.is_valid() {
                let mut dest_index_a_dest_index_b_time: IVec3 = optimal_timeline.chain_entrance;
                let mut state_index: usize = optimal_timeline.state_index;

                while state_index != usize::MAX {
                    let state: &BiActorState = &self.states
                        [self.get_range(&dest_index_a_dest_index_b_time).start + state_index];

                    timeline.push(Frame {
                        dest_index_a_dest_index_b_time,
                        state_index,
                        state: state.clone(),
                    });

                    state_index = usize::MAX;

                    let [dest_index_a, dest_index_b, time]: [i32; 3_usize] =
                        dest_index_a_dest_index_b_time.to_array();
                    let next_actor_state_dest_index_and_pressure_released_delta =
                        |mut actor_state: ActorState, dest_index: i32| -> (ActorState, i32, u16) {
                            let index: u32 = self.dest_info.dests[dest_index as usize];
                            let flow_rate: u16 = self.scan_lines[index as usize].flow_rate;

                            if actor_state.index == u8::MAX {
                                (actor_state, dest_index, flow_rate)
                            } else {
                                let cost: Option<i32> = self
                                    .useful_trips
                                    .get_from_start(index)
                                    .find(|useful_trip| {
                                        useful_trip.index == actor_state.index as u32
                                    })
                                    .map(|useful_trip| useful_trip.cost);

                                assert!(
                                    cost.is_some(),
                                    "expected a cost for index {index} and actor_state.index {}\n\
                                    actor_state: {actor_state:#?}\n\
                                    dest_index: {dest_index}\n\
                                    optimal_timeline: {optimal_timeline:#?}\n\
                                    timeline: {timeline:#?}",
                                    actor_state.index
                                );

                                if actor_state.travel_remaining < cost.unwrap() as u8 - 1_u8 {
                                    actor_state.travel_remaining += 1_u8;

                                    (actor_state, dest_index, 0_u16)
                                } else {
                                    (
                                        ActorState {
                                            travel_remaining: u8::MAX,
                                            ..Default::default()
                                        },
                                        self.dest_info.index_to_dest_index
                                            [actor_state.index as usize]
                                            as i32,
                                        flow_rate * cost.unwrap() as u16,
                                    )
                                }
                            }
                        };

                    let (expected_actor_state_a, next_dest_index_a, pressure_released_delta_a) =
                        next_actor_state_dest_index_and_pressure_released_delta(
                            state.actor_a,
                            dest_index_a,
                        );
                    let (expected_actor_state_b, next_dest_index_b, pressure_released_delta_b) =
                        next_actor_state_dest_index_and_pressure_released_delta(
                            state.actor_b,
                            dest_index_b,
                        );

                    dest_index_a_dest_index_b_time =
                        IVec3::new(next_dest_index_a, next_dest_index_b, time - 1_i32);

                    if let Some(range) = self.state_ranges.get(&dest_index_a_dest_index_b_time) {
                        let actor_states_match =
                            |expected_actor_state: ActorState, next_actor_state: ActorState| {
                                expected_actor_state.travel_remaining == u8::MAX
                                    || expected_actor_state == next_actor_state
                            };

                        // for (new_state_index, new_state) in
                        //     self.states[range.clone()].iter().enumerate()
                        // {
                        //     if actor_states_match(expected_actor_state_a, new_state.actor_a)
                        //         && actor_states_match(expected_actor_state_b, new_state.actor_b)
                        //         && new_state.pressure_released
                        //             + pressure_released_delta_a
                        //             + pressure_released_delta_b
                        //             == state.pressure_released
                        //     {
                        //         state_index = new_state_index;

                        //         break;
                        //     }
                        // }

                        panic!();
                    }
                }
            }

            timeline
        }

        fn init_terminal_states(&mut self) {
            let dimensions: IVec3 = *self.state_ranges.dimensions();

            for zero_dest_index_b_zero in
                CellIter3D::until_boundary_from_dimensions(&dimensions, IVec3::ZERO, IVec3::Y)
            {
                let valve_b: u64 =
                    1_u64 << self.dest_info.dests[zero_dest_index_b_zero[1_usize] as usize];

                for dest_index_a_dest_index_b_zero in CellIter3D::until_boundary_from_dimensions(
                    &dimensions,
                    zero_dest_index_b_zero,
                    IVec3::X,
                ) {
                    let valve_a: u64 = 1_u64
                        << self.dest_info.dests[dest_index_a_dest_index_b_zero[0_usize] as usize];
                    let start: usize = self.states.len();

                    self.states.push_back(BiActorState {
                        valves_open: valve_a | valve_b,
                        ..Default::default()
                    });
                    *self
                        .state_ranges
                        .get_mut(&dest_index_a_dest_index_b_zero)
                        .unwrap() = start..start + 1_usize;
                }
            }
        }

        fn iterate_over_cells_in_layer(&self, time: usize) -> impl Iterator<Item = IVec3> {
            let dimensions: IVec3 = *self.state_ranges.dimensions();

            CellIter3D::until_boundary_from_dimensions(
                &dimensions,
                IVec3::new(0_i32, 0_i32, time as i32),
                IVec3::Y,
            )
            .map(move |row_start| {
                CellIter3D::until_boundary_from_dimensions(&dimensions, row_start, IVec3::X)
            })
            .flatten()
        }

        fn build_source_states(
            &self,
            dest_a_dest_b_time: IVec3,
            new_states: &mut Vec<(IVec2, BiActorState)>,
        ) {
            let [dest_index_a, dest_index_b, time]: [i32; 3_usize] = dest_a_dest_b_time.to_array();

            if dest_index_a == dest_index_b {
                // No point in both actors visiting the same destination
                return;
            }

            let dest_a: u32 = self.dest_info.dests[dest_index_a as usize];
            let dest_b: u32 = self.dest_info.dests[dest_index_b as usize];
            let flow_rate_a: u16 = self.scan_lines[dest_a as usize].flow_rate;
            let flow_rate_b: u16 = self.scan_lines[dest_b as usize].flow_rate;

            struct UsefulTripInfo {
                valve: u64,
                index: u32,
                dest_index: i32,
                pressure_released: u16,
                travel_remaining: u8,
            }

            impl UsefulTripInfo {
                #[inline(always)]
                fn actor_state(&self, index: u32) -> ActorState {
                    ActorState {
                        index: index as u8,
                        travel_remaining: self.travel_remaining,
                    }
                }
            }

            let get_useful_trip_info = |useful_trip: &UsefulTrip| {
                let index: u32 = useful_trip.index;

                UsefulTripInfo {
                    valve: 1_u64 << useful_trip.index,
                    index,
                    dest_index: self.dest_info.index_to_dest_index[index as usize] as i32,
                    pressure_released: self.scan_lines[useful_trip.index as usize].flow_rate
                        * (time + useful_trip.cost) as u16,
                    // Subtract one so that the range of seen values is `0_u8..useful_trip.cost`,
                    // which has length `useful_trip.cost as usize`
                    travel_remaining: useful_trip.cost as u8 - 1_u8,
                }
            };

            for next_state in self.states.range(self.get_range(&dest_a_dest_b_time)) {
                let iter_useful_trips_to_dest = |dest: u32, avoid_start: u32| {
                    self.useful_trips
                        .get_to_dest(dest)
                        .filter(move |useful_trip_to_dest| {
                            let candidate_start: u32 = useful_trip_to_dest.index;
                            // Make sure the previous state isn't a valve that the timeline of
                            // the next state will also open
                            (next_state.valves_open & (1_u64 << candidate_start))
                                == 0_u64
                                // Make sure it isn't the start we want to avoid
                                && candidate_start != avoid_start
                                // And make sure either we don't care about useless starts, or
                                // this start isn't useless
                                && self.scan_lines[useful_trip_to_dest.index as usize]
                                        .flow_rate
                                        != 0_u16
                        })
                        .map(get_useful_trip_info)
                };

                // While it's likely possible for an input from which the optimal solution
                // includes one actor starting a move after the other has finished opening their
                // final valve, I'll assume that's not the case for my input until I need to
                // assume otherwise, since my current best plan for handling that would
                // drastically increse the state count
                match (
                    next_state.actor_a.travel_remaining,
                    next_state.actor_b.travel_remaining,
                ) {
                    (0_u8, 0_u8) => {
                        let both_are_terminal: bool = next_state.actor_a.index == u8::MAX
                            && next_state.actor_b.index == u8::MAX;

                        // Both A and B are free to start moving here from their starts
                        for info_a in iter_useful_trips_to_dest(dest_a, u32::MAX) {
                            let valves_open: u64 = next_state.valves_open | info_a.valve;
                            let pressure_released: u16 =
                                next_state.pressure_released + info_a.pressure_released;
                            let actor_a: ActorState = info_a.actor_state(dest_a);

                            for info_b in iter_useful_trips_to_dest(dest_b, info_a.index) {
                                new_states.push((
                                    IVec2::new(info_a.dest_index, info_b.dest_index),
                                    BiActorState {
                                        valves_open: valves_open | info_b.valve,
                                        pressure_released: pressure_released
                                            + info_b.pressure_released,
                                        actor_a,
                                        actor_b: info_b.actor_state(dest_b),
                                    },
                                ));

                                // // Insert state at the new B ActorState, but same A state
                                // if both_are_terminal {
                                //     new_states.push((
                                //         IVec2::new(dest_index_a, info_b.dest_index),
                                //         BiActorState {
                                //             valves_open: next_state.valves_open | info_b.valve,
                                //             pressure_released: next_state.pressure_released
                                //                 + flow_rate_a
                                //                 + info_b.pressure_released,
                                //             actor_b: info_b.actor_state(dest_b),
                                //             ..next_state.clone()
                                //         },
                                //     ));
                                // }
                            }

                            // Insert state at the new A ActorState, but same B state
                            if both_are_terminal {
                                new_states.push((
                                    IVec2::new(info_a.dest_index, dest_index_b),
                                    BiActorState {
                                        valves_open,
                                        pressure_released: pressure_released + flow_rate_b,
                                        actor_a,
                                        ..next_state.clone()
                                    },
                                ));
                            }
                        }

                        // This is a terminal state: propagate it up
                        if next_state.actor_a.index == u8::MAX
                            && next_state.actor_b.index == u8::MAX
                        {
                            new_states.push((
                                IVec2::new(dest_index_a, dest_index_b),
                                BiActorState {
                                    pressure_released: next_state.pressure_released
                                        + flow_rate_a
                                        + flow_rate_b,
                                    ..next_state.clone()
                                },
                            ));
                        }
                    }
                    (0_u8, travel_remaining_b) => {
                        let actor_b: ActorState = ActorState {
                            travel_remaining: travel_remaining_b - 1_u8,
                            ..next_state.actor_b
                        };

                        // Only A is free to start moving from here
                        for info_a in iter_useful_trips_to_dest(dest_a, u32::MAX) {
                            new_states.push((
                                IVec2::new(info_a.dest_index, dest_index_b),
                                BiActorState {
                                    valves_open: next_state.valves_open | info_a.valve,
                                    pressure_released: next_state.pressure_released
                                        + info_a.pressure_released,
                                    actor_a: info_a.actor_state(dest_a),
                                    actor_b,
                                },
                            ));
                        }

                        // // This is a terminal state: propagate it up
                        // if next_state.actor_a.index == u8::MAX {
                        //     new_states.push((
                        //         IVec2::new(dest_index_a, dest_index_b),
                        //         BiActorState {
                        //             pressure_released: next_state.pressure_released + flow_rate_a,
                        //             actor_b,
                        //             ..next_state.clone()
                        //         },
                        //     ));
                        // }
                    }
                    (travel_remaining_a, 0_u8) => {
                        let actor_a: ActorState = ActorState {
                            travel_remaining: travel_remaining_a - 1_u8,
                            ..next_state.actor_a
                        };

                        // Only B is free to start moving from here
                        for info_b in iter_useful_trips_to_dest(dest_b, u32::MAX) {
                            new_states.push((
                                IVec2::new(dest_index_a, info_b.dest_index),
                                BiActorState {
                                    valves_open: next_state.valves_open | info_b.valve,
                                    pressure_released: next_state.pressure_released
                                        + info_b.pressure_released,
                                    actor_a,
                                    actor_b: info_b.actor_state(dest_b),
                                },
                            ));
                        }

                        // This is a terminal state: propagate it up
                        if next_state.actor_b.index == u8::MAX {
                            new_states.push((
                                IVec2::new(dest_index_a, dest_index_b),
                                BiActorState {
                                    pressure_released: next_state.pressure_released + flow_rate_b,
                                    actor_a,
                                    ..next_state.clone()
                                },
                            ));
                        }
                    }
                    (travel_remaining_a, travel_remaining_b) => {
                        // Both are traveling
                        new_states.push((
                            IVec2::new(dest_index_a, dest_index_b),
                            BiActorState {
                                actor_a: ActorState {
                                    travel_remaining: travel_remaining_a - 1_u8,
                                    ..next_state.actor_a
                                },
                                actor_b: ActorState {
                                    travel_remaining: travel_remaining_b - 1_u8,
                                    ..next_state.actor_b
                                },
                                ..next_state.clone()
                            },
                        ))
                    }
                }
            }
        }

        pub(super) fn explore_time_steps(&mut self, time_steps: usize, status_updates: bool) {
            if time_steps < self.explored_time_steps {
                return;
            }

            self.state_ranges
                .resize_layers(time_steps + 1_usize, Default::default);

            let mut new_states: Vec<(IVec2, BiActorState)> = Vec::new();

            for time in self.explored_time_steps..time_steps {
                new_states.clear();

                let new_states_start: usize = self.states.len();

                for dest_a_dest_b_time in self.iterate_over_cells_in_layer(time) {
                    self.build_source_states(dest_a_dest_b_time, &mut new_states);
                }

                new_states.sort_unstable_by(|(pos_a, state_a), (pos_b, state_b)| {
                    pos_a
                        .y
                        .cmp(&pos_b.y)
                        .then_with(|| pos_a.x.cmp(&pos_b.x).then_with(|| state_b.cmp(state_a)))
                });

                let mut new_states_slice: &[(IVec2, BiActorState)] = &new_states;

                for dest_a_dest_b_time in self.iterate_over_cells_in_layer(time + 1_usize) {
                    let dest_a_dest_b: IVec2 = dest_a_dest_b_time.xy();
                    let mut range: Range<usize> = self.states.len()..0_usize;
                    let next_new_states_slice_start: usize = new_states_slice
                        .iter()
                        .position(|(pos, _)| *pos != dest_a_dest_b)
                        .unwrap_or(new_states_slice.len());
                    let mut previous_state: BiActorState = BiActorState::default();

                    for (_, state) in new_states_slice[..next_new_states_slice_start].iter() {
                        if !state.effectively_eq(&previous_state) {
                            self.states.push_back(state.clone());
                            previous_state = state.clone();
                        }
                    }

                    new_states_slice = &new_states_slice[next_new_states_slice_start..];

                    range.end = self.states.len();
                    *self.state_ranges.get_mut(&dest_a_dest_b_time).unwrap() = range;
                }

                if status_updates {
                    println!(
                        "Finished {} states (trimmed down from {}) for time {}",
                        self.states.len() - new_states_start,
                        new_states.len(),
                        time + 1_usize
                    );
                }
            }

            self.explored_time_steps = time_steps;
        }
    }
}

fn get_two_stateful_paths(
    scan_lines: &ScanLines,
    start: u32,
    time: usize,
) -> [Vec<question_1::PressureReleaseState>; 2_usize] {
    use question_1::*;

    let mut useful_trips: UsefulTrips = scan_lines.useful_trips().unwrap();
    let mut explorer_1: PressureReleaseExplorer =
        PressureReleaseExplorer::new(scan_lines, &useful_trips);

    explorer_1.explore_time_steps(time);

    let stateful_path_1: Vec<PressureReleaseState> = explorer_1.find_optimal_path(start, time);
    let stateless_path_1: Vec<u32> =
        PressureReleaseExplorer::get_stateless_path(start, &stateful_path_1);

    useful_trips.trim(&stateless_path_1);

    let mut explorer_2: PressureReleaseExplorer =
        PressureReleaseExplorer::new(scan_lines, &useful_trips);

    explorer_2.explore_time_steps(time);

    let stateful_path_2: Vec<PressureReleaseState> = explorer_2.find_optimal_path(start, time);

    [stateful_path_1, stateful_path_2]
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
                        use question_1::*;

                        let useful_trips: UsefulTrips = scan_lines.useful_trips().unwrap();
                        let aa_tag: u16 = u16::from_ne_bytes([b'A', b'A']);
                        let start: u32 = scan_lines
                            .lines
                            .iter()
                            .position(|scan_line| scan_line.tag == aa_tag)
                            .unwrap() as u32;

                        let mut pressure_release_explorer: PressureReleaseExplorer =
                            PressureReleaseExplorer::new(&scan_lines, &useful_trips);

                        pressure_release_explorer.explore_time_steps(30_usize);

                        let path: Vec<PressureReleaseState> =
                            pressure_release_explorer.find_optimal_path(start, 30_usize);

                        println!("path.first() == {:#?}\npath == {:#?}", path.first(), path);

                        let [path_a, path_b] = get_two_stateful_paths(&scan_lines, start, 26_usize);

                        println!(
                            "path_a.first() == {:#?}\n\
                            path_b.first() == {:#?}\n\
                            path_a.first().cloned().unwrap_or_default().pressure_released + \n\
                            path_b.first().cloned().unwrap_or_default().pressure_released = {}",
                            path_a.first(),
                            path_b.first(),
                            path_a
                                .first()
                                .cloned()
                                .unwrap_or_default()
                                .pressure_released
                                + path_b
                                    .first()
                                    .cloned()
                                    .unwrap_or_default()
                                    .pressure_released
                        );

                        // let mut bi_actor_explorer: BiActorExplorer =
                        //     BiActorExplorer::new(&scan_lines, &useful_trips);

                        // let optimal_timeline: OptimalTimeline =
                        //     bi_actor_explorer.find_optimal_timeline(start, start, 26_usize, true);

                        // println!("optimal_timeline: {optimal_timeline:#?}");
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
    use {super::*, std::collections::HashMap};

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
    const AA_INDEX: u32 = 0_u32;

    lazy_static! {
        static ref SCAN_LINES: ScanLines = example_scan_lines();
        static ref USEFUL_TRIPS: UsefulTrips = example_useful_trips();
        static ref TAG_TO_INDEX_AND_MASK: HashMap<u16, (u32, u64)> =
            example_tag_to_index_and_mask();
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
        use question_1::*;

        let mut pressure_release_explorer: PressureReleaseExplorer =
            PressureReleaseExplorer::new(&*SCAN_LINES, &*USEFUL_TRIPS);

        pressure_release_explorer.explore_time_steps(30_usize);

        let path: Vec<PressureReleaseState> =
            pressure_release_explorer.find_optimal_path(AA_INDEX, 30_usize);

        assert_eq!(path.first().unwrap().pressure_released, 1651);

        let path: Vec<u32> = PressureReleaseExplorer::get_stateless_path(AA_INDEX, &path);

        assert_eq!(
            path,
            [AA, DD, BB, JJ, HH, EE, CC]
                .iter()
                .map(|tag| TAG_TO_INDEX_AND_MASK[tag].0)
                .collect::<Vec<u32>>()
        );
    }

    #[test]
    fn test_with_elephant() {
        use question_1::*;

        let [stateful_path_1, stateful_path_2]: [Vec<PressureReleaseState>; 2_usize] =
            get_two_stateful_paths(&*SCAN_LINES, AA_INDEX, 26_usize);

        assert_eq!(
            stateful_path_1
                .first()
                .zip(stateful_path_2.first())
                .map(|(prs_1, prs_2)| prs_1.pressure_released + prs_2.pressure_released),
            Some(1707_i32)
        );
    }

    #[test]
    fn test_bi_actor_explorer() {
        use question_2_attempt_1::*;

        let mut bi_actor_explorer: BiActorExplorer =
            BiActorExplorer::new(&*SCAN_LINES, &*USEFUL_TRIPS);

        let optimal_timeline: OptimalTimeline =
            bi_actor_explorer.find_optimal_timeline(AA_INDEX, AA_INDEX, 26_usize, true);

        println!("optimal_timeline: {optimal_timeline:#?}");
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
        let pack_tags = |tags: &[u16]| -> u64 {
            let mut packed: u64 = 0_u64;

            for tag in tags {
                packed |= TAG_TO_INDEX_AND_MASK[tag].1;
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
                                index: TAG_TO_INDEX_AND_MASK[&$dest_tag].0,
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
            useful_trips.headers_from_start.push(UsefulTripHeader {
                range,
                indices: start_dests,
            });
            all_dests |= start_dests;
        }

        useful_trips.dests = all_dests;
        useful_trips.init_headers_to_dest();

        useful_trips
    }

    fn example_tag_to_index_and_mask() -> HashMap<u16, (u32, u64)> {
        vec![AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]
            .into_iter()
            .enumerate()
            .map(|(index, tag)| (tag, (index as u32, 1_u64 << index as u32)))
            .collect()
    }
}
