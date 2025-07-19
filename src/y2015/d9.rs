use {
    crate::*,
    arrayvec::ArrayVec,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, success, verify},
        error::Error,
        multi::separated_list1,
        sequence::tuple,
        Err, IResult,
    },
    std::{cmp::Ordering, collections::HashMap, iter::repeat, mem::size_of, num::NonZeroU16},
    strum::EnumCount,
};

/* --- Day 9: All in a Single Night ---

Every year, Santa manages to deliver all of his presents in a single night.

This year, however, he has some new locations to visit; his elves have provided him the distances between every pair of locations. He can start and end at any two (different) locations he wants, but he must visit each location exactly once. What is the shortest distance he can travel to achieve this?

For example, given the following distances:

London to Dublin = 464
London to Belfast = 518
Dublin to Belfast = 141

The possible routes are therefore:

Dublin -> London -> Belfast = 982
London -> Dublin -> Belfast = 605
London -> Belfast -> Dublin = 659
Dublin -> Belfast -> London = 659
Belfast -> Dublin -> London = 605
Belfast -> London -> Dublin = 982

The shortest of these is London -> Dublin -> Belfast = 605, and so the answer is 605 in this example.

What is the distance of the shortest route?

--- Part Two ---

The next year, just to show off, Santa decides to take the route with the longest distance instead.

He can still start and end at any two (different) locations he wants, and he still must visit each location exactly once.

For example, given the distances above, the longest route would be 982 via (for example) Dublin -> London -> Belfast.

What is the distance of the longest route? */

const MAX_LOCATION_ID_LEN: usize = 15_usize;

type LocationIndexRaw = u8;
type LocationIndex = Index<LocationIndexRaw>;
type LocationId = StaticString<MAX_LOCATION_ID_LEN>;
type LocationIdList = IdList<LocationId, LocationIndexRaw>;

type Distance = u16;
type NonZeroDistance = NonZeroU16;

// From `arrayvec`
type LenUInt = u32;

const MAX_LOCATION_PATH_LEN: usize = 3_usize * size_of::<LenUInt>() / size_of::<LocationIndex>();

// We need space for all locations, plus one repeat
const MAX_LOCATION_COUNT: usize = MAX_LOCATION_PATH_LEN - 1_usize;

// 1 bit for each location, plus an extra last bit for re-visiting the start.
type LocationBitArray = BitArr!(for MAX_LOCATION_PATH_LEN, in u8);

type LocationPath = ArrayVec<LocationIndex, MAX_LOCATION_PATH_LEN>;
type LocationIdPath = ArrayVec<LocationId, MAX_LOCATION_PATH_LEN>;

#[derive(Clone, Copy, EnumCount, PartialEq)]
enum LocationPathType {
    HamiltonianPath,
    HamiltonianCycle,
}

#[derive(Clone, Copy)]
enum DistanceScoringType {
    Shortest,
    Longest,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct LocationPathAndDistance {
    path: LocationPath,
    distance: Distance,
}

type LocationPathAndDistanceArray = [LocationPathAndDistance; LocationPathType::COUNT];
type LocationPathVertexArray = [Vertex; LocationPathType::COUNT];

#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
struct Vertex {
    current_location: LocationIndex,
    visited_locations: LocationBitArray,
}

#[derive(Default)]
struct VertexData {
    previous_location: Vertex,
    distance: Distance,
}

struct LocationPathFinder<'s> {
    solution: &'s Solution,
    vertex_data_map: HashMap<Vertex, VertexData>,
    start: Vertex,
    optimal_ends: LocationPathVertexArray,
    distance_scoring_type: DistanceScoringType,
}

impl<'s> LocationPathFinder<'s> {
    fn update_optimal_end(
        vertex_data_map: &HashMap<Vertex, VertexData>,
        end: Vertex,
        distance: Distance,
        optimal_end: &mut Vertex,
    ) {
        if !optimal_end.current_location.is_valid()
            // || vertex_data_map[&*optimal_end].distance > distance
            || vertex_data_map[&*optimal_end].distance > distance
        {
            *optimal_end = end;
        }
    }

    fn try_location_path_type(&self, vertex: Vertex) -> Option<LocationPathType> {
        self.solution
            .try_location_path_type(vertex.visited_locations)
    }

    fn try_location_path(&self, vertex: Vertex) -> Option<LocationPath> {
        (vertex.current_location.is_valid() && vertex.visited_locations.any()).then(|| {
            let path_len: usize = vertex.visited_locations.count_ones();

            let mut location_path: LocationPath = LocationPath::new();
            let mut vertex: Vertex = vertex;

            location_path.extend(repeat(Default::default()).take(path_len));

            for location_index in location_path[0_usize..path_len].iter_mut().rev() {
                *location_index = vertex.current_location;
                vertex = self.vertex_data_map[&vertex].previous_location;
            }

            location_path
        })
    }

    fn try_initial_location_index(&self, vertex: Vertex) -> Option<LocationIndex> {
        self.try_location_path(vertex).map(|path| path[0_usize])
    }
}

impl<'s> WeightedGraphSearch for LocationPathFinder<'s> {
    type Vertex = Vertex;
    type Cost = Distance;

    fn start(&self) -> &Vertex {
        &self.start
    }

    fn is_end(&self, &vertex: &Vertex) -> bool {
        self.try_location_path_type(vertex) == Some(LocationPathType::HamiltonianCycle)
    }

    fn path_to(&self, _vertex: &Vertex) -> Vec<Vertex> {
        Vec::new()
    }

    fn cost_from_start(&self, vertex: &Vertex) -> Distance {
        self.vertex_data_map
            .get(vertex)
            .map_or(u16::MAX, |location_data| location_data.distance)
    }

    fn heuristic(&self, vertex: &Vertex) -> Distance {
        zero_heuristic(self, vertex)
    }

    fn neighbors(&self, &vertex: &Vertex, neighbors: &mut Vec<OpenSetElement<Vertex, Distance>>) {
        neighbors.clear();

        let location_count: usize = self.solution.location_count();

        if vertex.current_location.is_valid() {
            match self.try_location_path_type(vertex) {
                None => {
                    neighbors.extend(
                        vertex.visited_locations[..location_count]
                            .iter_zeros()
                            .filter_map(|neighbor_location| {
                                self.solution
                                    .try_distance_between_locations(
                                        vertex.current_location,
                                        neighbor_location.into(),
                                        self.distance_scoring_type,
                                    )
                                    .map(|distance| {
                                        let mut neighbor: Vertex = vertex;
    
                                        neighbor.current_location = neighbor_location.into();
                                        neighbor.visited_locations.set(neighbor_location, true);
    
                                        OpenSetElement(neighbor, distance)
                                    })
                            }),
                    );
                }
                Some(LocationPathType::HamiltonianPath) => {
                    neighbors.extend(
                    self.try_initial_location_index(vertex).and_then(|start_location| {
                        self.solution.try_distance_between_locations(
                            vertex.current_location,
                            start_location,
                            self.distance_scoring_type).map(|distance| {
                                let mut neighbor: Vertex = Default::default();
    
                                neighbor.current_location = start_location;
                                neighbor.visited_locations[..location_count + 1_usize].fill(true);
    
                                OpenSetElement(neighbor, distance)
                            })
                    }));
                }
                Some(LocationPathType::HamiltonianCycle) => (),
            }
        } else {
            neighbors.extend(
                (0_usize..location_count).map(|location_index| {
                    let mut neighbor: Vertex = Default::default();

                    neighbor.current_location = location_index.into();
                    neighbor.visited_locations.set(location_index, true);

                    OpenSetElement(neighbor, 0_u16)
                })
            )
        }
    }

    fn update_vertex(
        &mut self,
        &previous_location: &Vertex,
        &current_location: &Vertex,
        distance: Distance,
        _heuristic: Distance,
    ) {
        self.vertex_data_map.insert(
            current_location,
            VertexData {
                previous_location,
                distance,
            },
        );

        if let Some(path_type) = self.try_location_path_type(current_location) {
            Self::update_optimal_end(
                &self.vertex_data_map,
                current_location,
                distance,
                &mut self.optimal_ends[path_type as usize],
            );
        }
    }

    fn reset(&mut self) {
        self.vertex_data_map.clear();
        self.vertex_data_map.insert(self.start, Default::default());
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    location_ids: LocationIdList,
    distances: Grid2D<Option<NonZeroDistance>>,
    max_distance: NonZeroDistance,
}

impl Solution {
    fn parse_location_id<'i>(input: &'i str) -> IResult<&'i str, LocationId> {
        LocationId::parse_char1(1_usize, |c| c.is_ascii_alphabetic())(input)
    }

    fn parse_line<'i>(input: &'i str) -> IResult<&'i str, (LocationId, LocationId, u16)> {
        map(
            tuple((
                Self::parse_location_id,
                tag(" to "),
                Self::parse_location_id,
                tag(" = "),
                parse_integer,
            )),
            |(from, _, to, _, distance)| (from, to, distance),
        )(input)
    }

    fn location_count(&self) -> usize {
        self.location_ids.as_slice().len()
    }

    fn try_location_path_type(&self, locations: LocationBitArray) -> Option<LocationPathType> {
        match locations.count_ones().cmp(&self.location_count()) {
            Ordering::Less => None,
            Ordering::Equal => Some(LocationPathType::HamiltonianPath),
            Ordering::Greater => Some(LocationPathType::HamiltonianCycle),
        }
    }

    fn try_distance_between_locations(
        &self,
        from: LocationIndex,
        to: LocationIndex,
        distance_scoring_type: DistanceScoringType,
    ) -> Option<Distance> {
        from.opt()
            .zip(to.opt())
            .and_then(|(from, to)| {
                self.distances
                    .get((from.get() as i32, to.get() as i32).into())
            })
            .cloned()
            .flatten()
            .map(|distance| match distance_scoring_type {
                DistanceScoringType::Shortest => distance.get(),
                DistanceScoringType::Longest => self.max_distance.get() - distance.get() + 1_u16,
            })
    }

    fn path_distance(&self, path: &LocationPath) -> Distance {
        path.windows(2_usize)
            .map(|location_index_slice| {
                self.try_distance_between_locations(
                    location_index_slice[0_usize],
                    location_index_slice[1_usize],
                    DistanceScoringType::Shortest,
                )
                .unwrap()
            })
            .sum()
    }

    fn compute_optimal_paths(
        &self,
        distance_scoring_type: DistanceScoringType,
    ) -> LocationPathAndDistanceArray {
        let mut location_path_finder: LocationPathFinder = LocationPathFinder {
            solution: self,
            start: Default::default(),
            vertex_data_map: Default::default(),
            optimal_ends: Default::default(),
            distance_scoring_type,
        };

        location_path_finder.run_dijkstra();

        let mut optimal_paths: LocationPathAndDistanceArray = Default::default();

        for (optimal_end, optimal_path) in location_path_finder
            .optimal_ends
            .iter()
            .copied()
            .zip(optimal_paths.iter_mut())
        {
            if let Some(path) = location_path_finder.try_location_path(optimal_end) {
                optimal_path.distance = self.path_distance(&path);
                optimal_path.path = path;
            }
        }

        optimal_paths
    }

    fn compute_shortest_hamiltonian_path_distance(&self) -> Distance {
        self.compute_optimal_paths(DistanceScoringType::Shortest)
            [LocationPathType::HamiltonianPath as usize]
            .distance
    }

    fn compute_longest_hamiltonian_path_distance(&self) -> Distance {
        self.compute_optimal_paths(DistanceScoringType::Longest)
            [LocationPathType::HamiltonianPath as usize]
            .distance
    }

    fn location_index_path_to_location_id_path(
        &self,
        location_index_path: &LocationPath,
    ) -> LocationIdPath {
        location_index_path
            .into_iter()
            .map(|location_index| self.location_ids.as_id_slice()[location_index.get()].clone())
            .collect()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut location_ids: LocationIdList = LocationIdList::new();

        separated_list1(
            line_ending,
            map(Self::parse_line, |(from, to, _)| {
                location_ids.find_or_add_index(&from);
                location_ids.find_or_add_index(&to);
            }),
        )(input)?;

        verify(success(()), |_| {
            location_ids.as_slice().len() <= MAX_LOCATION_COUNT
        })("more locations were provided than allowed")?;

        location_ids.sort_by_id();

        let location_ids_len: i32 = location_ids.as_slice().len() as i32;

        let mut distances: Grid2D<Option<NonZeroDistance>> =
            Grid2D::default((location_ids_len, location_ids_len).into());

        let input: &str = separated_list1(
            line_ending,
            map(Self::parse_line, |(from, to, distance)| {
                let from: i32 = location_ids.find_index_binary_search(&from).get() as i32;
                let to: i32 = location_ids.find_index_binary_search(&to).get() as i32;

                *distances.get_mut((from, to).into()).unwrap() = NonZeroDistance::new(distance);
                *distances.get_mut((to, from).into()).unwrap() = NonZeroDistance::new(distance);
            }),
        )(input)?
        .0;

        let max_distance: NonZeroDistance =
            distances.cells().iter().copied().flatten().max().unwrap();

        let solution: Self = Self {
            location_ids,
            distances,
            max_distance,
        };

        verify(success(()), |_| {
            (0_i32..solution.location_count() as i32).all(|index| {
                solution
                    .distances
                    .get((index, index).into())
                    .unwrap()
                    .is_none()
            })
        })("a location cannot have a distance to itself")?;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// Totally had the wrong idea for what question 2 would be
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let [
                shortest_hamiltonian_path,
                shortest_hamiltonian_cycle
            ]: LocationPathAndDistanceArray = 
                self.compute_optimal_paths(DistanceScoringType::Shortest);

            dbg!(shortest_hamiltonian_path.distance);

            let shortest_hamiltonian_path_ids: LocationIdPath =
                self.location_index_path_to_location_id_path(&shortest_hamiltonian_path.path);

            dbg!(shortest_hamiltonian_path_ids);
            dbg!(shortest_hamiltonian_cycle.distance);

            let shortest_hamiltonian_cycle_ids: LocationIdPath =
                self.location_index_path_to_location_id_path(&shortest_hamiltonian_cycle.path);

            dbg!(shortest_hamiltonian_cycle_ids);
        } else {
            dbg!(self.compute_shortest_hamiltonian_path_distance());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let [
                longest_hamiltonian_path,
                longest_hamiltonian_cycle
            ]: LocationPathAndDistanceArray = 
                self.compute_optimal_paths(DistanceScoringType::Longest);

            dbg!(longest_hamiltonian_path.distance);

            let longest_hamiltonian_path_ids: LocationIdPath =
                self.location_index_path_to_location_id_path(&longest_hamiltonian_path.path);

            dbg!(longest_hamiltonian_path_ids);
            dbg!(longest_hamiltonian_cycle.distance);

            let longest_hamiltonian_cycle_ids: LocationIdPath =
                self.location_index_path_to_location_id_path(&longest_hamiltonian_cycle.path);

            dbg!(longest_hamiltonian_cycle_ids);
        } else {
            dbg!(self.compute_longest_hamiltonian_path_distance());
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
        London to Dublin = 464\n\
        London to Belfast = 518\n\
        Dublin to Belfast = 141\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                location_ids: vec![
                    LocationId::try_from("Belfast").unwrap(),
                    "Dublin".try_into().unwrap(),
                    "London".try_into().unwrap(),
                ]
                .try_into()
                .unwrap(),
                distances: Grid2D::try_from_cells_and_dimensions(
                    [0, 141, 518, 141, 0, 464, 518, 464, 0]
                        .into_iter()
                        .map(NonZeroDistance::new)
                        .collect(),
                    (3_i32, 3_i32).into(),
                )
                .unwrap(),
                max_distance: NonZeroDistance::new(518).unwrap(),
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
    fn test_compute_shortest_paths() {
        for (index, shortest_paths) in [[
            LocationPathAndDistance {
                path: [0_usize.into(), 1_usize.into(), 2_usize.into()]
                    .into_iter()
                    .collect(),
                distance: 605_u16,
            },
            LocationPathAndDistance {
                path: [
                    0_usize.into(),
                    1_usize.into(),
                    2_usize.into(),
                    0_usize.into(),
                ]
                .into_iter()
                .collect(),
                distance: 1123_u16,
            },
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).compute_optimal_paths(DistanceScoringType::Shortest),
                shortest_paths
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
