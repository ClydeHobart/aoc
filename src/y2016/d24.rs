use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        combinator::{success, verify},
        error::Error,
        Err, IResult,
    },
    num::Integer,
    std::{
        alloc::{alloc_zeroed, Layout},
        collections::VecDeque,
        fmt::{Debug, Formatter, Result as FmtResult},
        mem::take,
        ops::Range,
    },
    strum::IntoEnumIterator,
};

/* --- Day 24: Air Duct Spelunking ---

You've finally met your match; the doors that provide access to the roof are locked tight, and all of the controls and related electronics are inaccessible. You simply can't reach them.

The robot that cleans the air ducts, however, can.

It's not a very fast little robot, but you reconfigure it to be able to interface with some of the exposed wires that have been routed through the HVAC system. If you can direct it to each of those locations, you should be able to bypass the security controls.

You extract the duct layout for this area from some blueprints you acquired and create a map with the relevant locations marked (your puzzle input). 0 is your current location, from which the cleaning robot embarks; the other numbers are (in no particular order) the locations the robot needs to visit at least once each. Walls are marked as #, and open passages are marked as .. Numbers behave like open passages.

For example, suppose you have a map like the following:

###########
#0.1.....2#
#.#######.#
#4.......3#
###########

To reach all of the points of interest as quickly as possible, you would have the robot take the following path:

    0 to 4 (2 steps)
    4 to 1 (4 steps; it can't move diagonally)
    1 to 2 (6 steps)
    2 to 3 (2 steps)

Since the robot isn't very fast, you need to find it the shortest route. This path is the fewest steps (in the above example, a total of 14) required to start at 0 and then visit every other location at least once.

Given your actual map, and starting from location 0, what is the fewest number of steps required to visit every non-0 number marked on the map at least once? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Wall = WALL = b'#',
        Open = OPEN = b'.',
        Zero = ZERO = b'0',
        One = ONE = b'1',
        Two = TWO = b'2',
        Three = THREE = b'3',
        Four = FOUR = b'4',
        Five = FIVE = b'5',
        Six = SIX = b'6',
        Seven = SEVEN = b'7',
        Eight = EIGHT = b'8',
        Nine = NINE = b'9',
        Path = PATH = b'*',
    }
}

impl Cell {
    fn is_digit(self) -> bool {
        !matches!(self, Self::Wall | Self::Open)
    }

    fn poi(self) -> Option<u8> {
        self.is_digit().then(|| self as u8 - b'0')
    }
}

#[cfg_attr(test, derive(Debug))]
struct Neighbor {
    poi: u8,
    dist: i32,
    corners: Vec<IVec2>,
}

impl Neighbor {
    #[cfg(test)]
    fn corners_dist(&self) -> i32 {
        self.corners
            .windows(2_usize)
            .map(|corners| manhattan_distance_2d(corners[0_usize], corners[1_usize]))
            .sum()
    }
}

#[cfg(test)]
impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.poi == other.poi
            && self.dist == other.dist
            && self.corners_dist() == other.corners_dist()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct PointOfInterest {
    neighbors: Vec<Neighbor>,
}

struct NeighborFinder<'g> {
    grid: &'g Grid2D<Cell>,
    start: IVec2,
    neighbors: Vec<Neighbor>,
    parents: Grid2D<Option<Direction>>,
    end_neighbors_len: usize,
    path: VecDeque<IVec2>,
}

impl<'g> NeighborFinder<'g> {
    fn build_path_to_pos(&mut self, pos: IVec2) {
        self.path.clear();

        let mut pos: IVec2 = pos;

        while pos != self.start {
            self.path.push_front(pos);
            pos += self.parents.get(pos).unwrap().unwrap().vec();
        }

        self.path.push_front(pos);
    }

    fn neighbor(&self) -> Neighbor {
        let poi: u8 = self
            .grid
            .get(*self.path.back().unwrap())
            .unwrap()
            .poi()
            .unwrap();
        let dist: i32 = self.path.len() as i32 - 1_i32;

        let mut last_corner: IVec2 = *self.path.front().unwrap();
        let mut last_pos: IVec2 = last_corner;
        let mut corners: Vec<IVec2> = vec![last_corner];

        for pos in self.path.iter().skip(1_usize) {
            let delta: IVec2 = *pos - last_corner;

            if delta.abs().cmpgt(IVec2::ZERO).all() {
                last_corner = last_pos;
                corners.push(last_corner);
            }

            last_pos = *pos;
        }

        corners.push(last_pos);

        Neighbor { poi, dist, corners }
    }

    fn add_neighbor(&mut self, neighbor: Neighbor) {
        let index: usize = self
            .neighbors
            .partition_point(|old_neighbor| old_neighbor.poi < neighbor.poi);

        self.neighbors.insert(index, neighbor)
    }
}

impl<'g> BreadthFirstSearch for NeighborFinder<'g> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        self.neighbors.len() == self.end_neighbors_len
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        Vec::new()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let pos: IVec2 = *vertex + dir.vec();

            self.grid
                .get(pos)
                .map_or(false, |cell| cell.is_digit() || *cell == Cell::Open)
                .then_some(pos)
        }));
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let dir: Direction = (*from - *to).try_into().unwrap();

        *self.parents.get_mut(*to).unwrap() = Some(dir);

        if self.grid.get(*to).unwrap().is_digit() {
            self.build_path_to_pos(*to);

            let neighbor: Neighbor = self.neighbor();

            self.add_neighbor(neighbor);
        }
    }

    fn reset(&mut self) {
        self.neighbors.clear();
        self.parents.cells_mut().fill(None);
        self.path.clear();
    }
}

/// Compact static vector for node indices. The 3 least significant bits are a length, followed by
/// `3 * len` bits of `len` indices (each getting 3).
#[derive(Clone, Copy)]
struct HamiltonianPath(u32);

impl HamiltonianPath {
    const ELEMENT_BITS: usize = u8::BITS.ilog2() as usize;
    const LEN_RANGE: Range<usize> = 0_usize..4_usize;
    const MAX_LEN: usize = (u32::BITS as usize - Self::LEN_RANGE.end) / Self::ELEMENT_BITS;
    const ELEMENTS_RANGE: Range<usize> =
        Self::LEN_RANGE.end..Self::LEN_RANGE.end + Self::MAX_LEN * Self::ELEMENT_BITS;

    fn element_range(index: usize) -> Range<usize> {
        let start: usize = index * Self::ELEMENT_BITS;

        start..start + Self::ELEMENT_BITS
    }

    fn get_bits(&self) -> &BitSlice<u32> {
        self.0.view_bits::<Lsb0>()
    }

    fn get_elements(&self) -> &BitSlice<u32> {
        &self.get_bits()[Self::ELEMENTS_RANGE]
    }

    fn elements_array(&self) -> [u8; Self::MAX_LEN] {
        let mut elements: [u8; Self::MAX_LEN] = [0_u8; Self::MAX_LEN];

        for (index, element) in self.iter_elements().enumerate() {
            elements[index] = element;
        }

        elements
    }

    fn iter_elements(&self) -> impl Iterator<Item = u8> + '_ {
        self.get_elements()
            .chunks_exact(Self::ELEMENT_BITS)
            .map(|element| element.load())
    }

    fn get_len(self) -> usize {
        self.get_bits()[Self::LEN_RANGE].load()
    }

    fn get_bits_mut(&mut self) -> &mut BitSlice<u32> {
        self.0.view_bits_mut::<Lsb0>()
    }

    fn set_len(&mut self, len: usize) {
        self.get_bits_mut()[Self::LEN_RANGE].store(len);
    }

    fn get_elements_mut(&mut self) -> &mut BitSlice<u32> {
        &mut self.get_bits_mut()[Self::ELEMENTS_RANGE]
    }

    fn get_element_mut(&mut self, index: usize) -> &mut BitSlice<u32> {
        &mut self.get_elements_mut()[Self::element_range(index)]
    }

    fn push(&mut self, value: u8) {
        let index: usize = self.get_len();

        assert!(index < Self::MAX_LEN);

        self.get_element_mut(index).store(value);
        self.set_len(index + 1_usize);
    }
}

impl Debug for HamiltonianPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.elements_array()[..self.get_len()].fmt(f)
    }
}

#[derive(Clone, Copy, Debug)]
struct HamiltonianPathDist {
    hamiltonian_path: HamiltonianPath,
    dist: i32,
}

#[derive(Clone, Copy, Default)]
struct NodeSet(u8);

impl NodeSet {
    fn try_index(self) -> Option<usize> {
        self.0.is_even().then_some(self.0 as usize >> 1_usize)
    }

    fn get_bits(&self) -> &BitSlice<u8> {
        self.0.view_bits()
    }

    fn set(&mut self, index: usize, value: bool) {
        self.0.view_bits_mut::<Lsb0>().set(index, value);
    }
}

struct HeldKarpState {
    dist_from_node_to_node: [[i32; Self::MAX_NODES]; Self::MAX_NODES],
    dist_through_nodes_to_node: [[HamiltonianPathDist; Self::MAX_NODES]; Self::NODE_SETS],
    nodes: usize,
    hamiltonian_cycle_dist: HamiltonianPathDist,
    has_ran: bool,
}

impl HeldKarpState {
    const MAX_NODES: usize = u8::BITS as usize;
    const NODE_SETS: usize = 1_usize << (Self::MAX_NODES - 1_usize);

    fn new() -> Box<Self> {
        // SAFETY: This holds all POD, so it can be zeroed
        unsafe { Box::from_raw(alloc_zeroed(Layout::new::<Self>()) as *mut Self) }
    }

    fn all_non_start_nodes(&self) -> NodeSet {
        NodeSet((((1_usize << (self.nodes - 1_usize)) - 1_usize) << 1_usize) as u8)
    }

    fn set_path<F: Fn(&mut HeldKarpState) -> &mut HamiltonianPathDist>(
        &mut self,
        prev_node_set_index: usize,
        prev_node: usize,
        get_curr_hamiltonian_path_dist_mut: F,
        curr_node: usize,
    ) {
        let prev_hamiltonian_path_dist: HamiltonianPathDist =
            self.dist_through_nodes_to_node[prev_node_set_index][prev_node];
        let dist_delta: i32 = self.dist_from_node_to_node[prev_node][curr_node];

        let curr_hamiltonian_path_dist: &mut HamiltonianPathDist =
            get_curr_hamiltonian_path_dist_mut(self);

        *curr_hamiltonian_path_dist = prev_hamiltonian_path_dist;
        curr_hamiltonian_path_dist
            .hamiltonian_path
            .push(curr_node as u8);
        curr_hamiltonian_path_dist.dist += dist_delta;
    }

    fn try_set_min_path<F: Fn(&mut HeldKarpState) -> &mut HamiltonianPathDist>(
        &mut self,
        prev_node_set: NodeSet,
        get_curr_hamiltonian_path_dist_mut: F,
        curr_node: usize,
    ) -> Option<()> {
        prev_node_set.try_index().map(|prev_node_set_index| {
            let prev_node: usize = prev_node_set
                .get_bits()
                .iter_ones()
                .min_by_key(|prev_node| {
                    self.dist_through_nodes_to_node[prev_node_set_index][*prev_node].dist
                        + self.dist_from_node_to_node[*prev_node][curr_node]
                })
                .unwrap();

            self.set_path(
                prev_node_set_index,
                prev_node,
                get_curr_hamiltonian_path_dist_mut,
                curr_node,
            );
        })
    }

    fn run(&mut self) {
        if self.has_ran {
            return;
        }

        let mut node_sets: [NodeSet; Self::NODE_SETS] = LargeArrayDefault::large_array_default();
        let node_sets_slice: &mut [NodeSet] = &mut node_sets[..1_usize << (self.nodes - 1_usize)];

        // how do we make sure there's no empty node set in here?
        for (index, node_set) in node_sets_slice.iter_mut().enumerate() {
            *node_set = NodeSet((index as u8) << 1_u8);
        }

        node_sets_slice.sort_by_key(|node_set| {
            (node_set.get_bits().count_ones() << u8::BITS) | node_set.try_index().unwrap()
        });

        for curr_node_set in node_sets_slice.iter().copied() {
            match curr_node_set.get_bits().count_ones() {
                0_usize => (),
                1_usize => {
                    let node: usize = curr_node_set.get_bits().leading_zeros() as usize;
                    let hamiltonian_path_dist: &mut HamiltonianPathDist = &mut self
                        .dist_through_nodes_to_node[curr_node_set.try_index().unwrap()][node];

                    hamiltonian_path_dist.dist = self.dist_from_node_to_node[0_usize][node];
                    hamiltonian_path_dist.hamiltonian_path.push(0_u8);
                    hamiltonian_path_dist.hamiltonian_path.push(node as u8);
                }
                _ => {
                    let curr_node_set_index: usize = curr_node_set.try_index().unwrap();

                    for curr_node in curr_node_set.get_bits().iter_ones() {
                        let mut prev_node_set: NodeSet = curr_node_set;

                        prev_node_set.set(curr_node, false);

                        self.try_set_min_path(
                            prev_node_set,
                            |held_karp_state| {
                                &mut held_karp_state.dist_through_nodes_to_node[curr_node_set_index]
                                    [curr_node]
                            },
                            curr_node,
                        );
                    }
                }
            }
        }

        self.try_set_min_path(
            self.all_non_start_nodes(),
            |held_karp_state| &mut held_karp_state.hamiltonian_cycle_dist,
            0_usize,
        );

        self.has_ran = true;
    }

    fn min_dist_hamiltonian_path(&self) -> HamiltonianPathDist {
        assert!(self.has_ran);

        let all_non_start_nodes: NodeSet = self.all_non_start_nodes();
        let all_non_start_nodes_index: usize = all_non_start_nodes.try_index().unwrap();

        all_non_start_nodes
            .get_bits()
            .iter_ones()
            .map(|node| self.dist_through_nodes_to_node[all_non_start_nodes_index][node])
            .min_by_key(|hamiltonian_path_dist| hamiltonian_path_dist.dist)
            .unwrap()
    }

    fn min_dist_hamiltonian_cycle(&self) -> HamiltonianPathDist {
        assert!(self.has_ran);

        self.hamiltonian_cycle_dist
    }
}

pub struct Solution {
    grid: Grid2D<Cell>,
    pois: Vec<PointOfInterest>,
    held_karp_state: Option<Box<HeldKarpState>>,
}

impl Solution {
    const DIGITS: usize = 10_usize;

    fn pois(grid: &Grid2D<Cell>, poses: &[IVec2]) -> Vec<PointOfInterest> {
        let mut neighbor_finder: NeighborFinder = NeighborFinder {
            grid: &grid,
            start: IVec2::ZERO,
            neighbors: Vec::new(),
            parents: Grid2D::default(grid.dimensions()),
            end_neighbors_len: poses.len() - 1_usize,
            path: VecDeque::new(),
        };
        let mut pois: Vec<PointOfInterest> = Vec::new();

        for pos in poses.iter().copied() {
            neighbor_finder.start = pos;
            neighbor_finder.run();
            pois.push(PointOfInterest {
                neighbors: take(&mut neighbor_finder.neighbors),
            });
        }

        pois
    }

    fn try_held_karp_state(pois: &[PointOfInterest]) -> Option<Box<HeldKarpState>> {
        (pois.len() <= HeldKarpState::MAX_NODES).then(|| {
            let mut held_karp_state: Box<HeldKarpState> = HeldKarpState::new();

            held_karp_state.nodes = pois.len();

            for (poi, dists_to_nodes) in pois
                .iter()
                .zip(held_karp_state.dist_from_node_to_node.iter_mut())
            {
                for neighbor in &poi.neighbors {
                    dists_to_nodes[neighbor.poi as usize] = neighbor.dist;
                }
            }

            held_karp_state.run();

            held_karp_state
        })
    }

    fn try_min_dist_hamiltonian_path_dist(&self) -> Option<HamiltonianPathDist> {
        self.held_karp_state
            .as_ref()
            .map(|held_karp_state| held_karp_state.min_dist_hamiltonian_path())
    }

    fn try_min_hamiltonian_path_dist(&self) -> Option<i32> {
        self.try_min_dist_hamiltonian_path_dist()
            .map(|hamiltonian_path_dist| hamiltonian_path_dist.dist)
    }

    fn try_min_dist_hamiltonian_cycle_dist(&self) -> Option<HamiltonianPathDist> {
        self.held_karp_state
            .as_ref()
            .map(|held_karp_state| held_karp_state.min_dist_hamiltonian_cycle())
    }

    fn try_min_hamiltonian_cycle_dist(&self) -> Option<i32> {
        self.try_min_dist_hamiltonian_cycle_dist()
            .map(|hamiltonian_path_dist| hamiltonian_path_dist.dist)
    }

    fn path_grid_string(&self, hamiltonian_path: HamiltonianPath) -> String {
        let mut grid: Grid2D<Cell> = self.grid.clone();
        let elements_array: [u8; HamiltonianPath::MAX_LEN] = hamiltonian_path.elements_array();
        let elements: &[u8] = &elements_array[..hamiltonian_path.get_len()];

        for pos in elements
            .windows(2_usize)
            .flat_map(|element_pair| {
                let poi: u8 = element_pair[1_usize];

                self.pois[element_pair[0_usize] as usize]
                    .neighbors
                    .iter()
                    .find(|neighbor| neighbor.poi == poi)
                    .unwrap()
                    .corners
                    .windows(2_usize)
            })
            .flat_map(|corner_pair| {
                CellIter2D::try_from(corner_pair[0_usize]..=corner_pair[1_usize]).unwrap()
            })
        {
            let cell: &mut Cell = grid.get_mut(pos).unwrap();

            if *cell == Cell::Open {
                *cell = Cell::Path;
            }
        }

        grid.into()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (remaining_input, grid): (&str, Grid2D<Cell>) = Grid2D::parse(input)?;
        let mut poses: [IVec2; Self::DIGITS] = [IVec2::NEG_ONE; Self::DIGITS];

        for (index, poi) in grid
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| cell.poi().map(|poi| (index, poi as usize)))
        {
            verify(success(()), |_| poses[poi] == IVec2::NEG_ONE)(input)?;

            poses[poi] = grid.pos_from_index(index);
        }

        verify(success(()), |_| poses[0_usize] != IVec2::NEG_ONE)(input)?;

        let mut has_pos: u16 = 0_u16;

        for (index, pos) in poses.iter().enumerate() {
            has_pos
                .view_bits_mut::<Lsb0>()
                .set(index, *pos != IVec2::NEG_ONE);
        }

        let poses_end: usize = u16::BITS as usize - has_pos.leading_zeros() as usize;

        verify(success(()), |_| {
            (has_pos & ((1_u16 << poses_end) - 1_u16)) == has_pos
        })(input)?;

        let pois: Vec<PointOfInterest> = Self::pois(&grid, &poses[..poses_end]);
        let held_karp_state: Option<Box<HeldKarpState>> = Self::try_held_karp_state(&pois);

        Ok((
            remaining_input,
            Self {
                grid,
                pois,
                held_karp_state,
            },
        ))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(hamiltonian_path_dist) = self.try_min_dist_hamiltonian_path_dist() {
                dbg!(hamiltonian_path_dist);
                println!(
                    "{}",
                    self.path_grid_string(hamiltonian_path_dist.hamiltonian_path)
                );
            } else {
                eprintln!("Failed to get minimum distance HamiltonianPathDist!");
            }
        } else {
            dbg!(self.try_min_hamiltonian_path_dist());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(hamiltonian_cycle_dist) = self.try_min_dist_hamiltonian_cycle_dist() {
                dbg!(hamiltonian_cycle_dist);
                println!(
                    "{}",
                    self.path_grid_string(hamiltonian_cycle_dist.hamiltonian_path)
                );
            } else {
                eprintln!("Failed to get minimum distance HamiltonianPathDist!");
            }
        } else {
            dbg!(self.try_min_hamiltonian_cycle_dist());
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

    const SOLUTION_STR: &'static str = "\
        ###########\n\
        #0.1.....2#\n\
        #.#######.#\n\
        #4.......3#\n\
        ###########\n";

    fn solution() -> &'static Solution {
        use Cell::{Four as E, One as B, Open as O, Three as D, Two as C, Wall as W, Zero as A};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            let grid: Grid2D<Cell> = Grid2D::try_from_cells_and_width(
                vec![
                    W, W, W, W, W, W, W, W, W, W, W, W, A, O, B, O, O, O, O, O, C, W, W, O, W, W,
                    W, W, W, W, W, O, W, W, E, O, O, O, O, O, O, O, D, W, W, W, W, W, W, W, W, W,
                    W, W, W,
                ],
                11_usize,
            )
            .unwrap();
            let pois: Vec<PointOfInterest> = vec![
                PointOfInterest {
                    neighbors: vec![
                        Neighbor {
                            poi: 1_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(1, 1), IVec2::new(3, 1)],
                        },
                        Neighbor {
                            poi: 2_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(1, 1), IVec2::new(9, 1)],
                        },
                        Neighbor {
                            poi: 3_u8,
                            dist: 10_i32,
                            corners: vec![IVec2::new(1, 1), IVec2::new(9, 1), IVec2::new(9, 3)],
                        },
                        Neighbor {
                            poi: 4_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(1, 1), IVec2::new(1, 3)],
                        },
                    ],
                },
                PointOfInterest {
                    neighbors: vec![
                        Neighbor {
                            poi: 0_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(3, 1), IVec2::new(1, 1)],
                        },
                        Neighbor {
                            poi: 2_u8,
                            dist: 6_i32,
                            corners: vec![IVec2::new(3, 1), IVec2::new(9, 1)],
                        },
                        Neighbor {
                            poi: 3_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(3, 1), IVec2::new(9, 1), IVec2::new(9, 3)],
                        },
                        Neighbor {
                            poi: 4_u8,
                            dist: 4_i32,
                            corners: vec![IVec2::new(3, 1), IVec2::new(1, 1), IVec2::new(1, 3)],
                        },
                    ],
                },
                PointOfInterest {
                    neighbors: vec![
                        Neighbor {
                            poi: 0_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(9, 1), IVec2::new(1, 1)],
                        },
                        Neighbor {
                            poi: 1_u8,
                            dist: 6_i32,
                            corners: vec![IVec2::new(9, 1), IVec2::new(3, 1)],
                        },
                        Neighbor {
                            poi: 3_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(9, 1), IVec2::new(9, 3)],
                        },
                        Neighbor {
                            poi: 4_u8,
                            dist: 10_i32,
                            corners: vec![IVec2::new(9, 1), IVec2::new(9, 3), IVec2::new(1, 3)],
                        },
                    ],
                },
                PointOfInterest {
                    neighbors: vec![
                        Neighbor {
                            poi: 0_u8,
                            dist: 10_i32,
                            corners: vec![IVec2::new(9, 3), IVec2::new(9, 1), IVec2::new(1, 1)],
                        },
                        Neighbor {
                            poi: 1_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(9, 3), IVec2::new(9, 1), IVec2::new(3, 1)],
                        },
                        Neighbor {
                            poi: 2_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(9, 3), IVec2::new(9, 1)],
                        },
                        Neighbor {
                            poi: 4_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(9, 3), IVec2::new(1, 3)],
                        },
                    ],
                },
                PointOfInterest {
                    neighbors: vec![
                        Neighbor {
                            poi: 0_u8,
                            dist: 2_i32,
                            corners: vec![IVec2::new(1, 3), IVec2::new(1, 1)],
                        },
                        Neighbor {
                            poi: 1_u8,
                            dist: 4_i32,
                            corners: vec![IVec2::new(1, 3), IVec2::new(1, 1), IVec2::new(3, 1)],
                        },
                        Neighbor {
                            poi: 2_u8,
                            dist: 10_i32,
                            corners: vec![IVec2::new(1, 3), IVec2::new(1, 1), IVec2::new(9, 1)],
                        },
                        Neighbor {
                            poi: 3_u8,
                            dist: 8_i32,
                            corners: vec![IVec2::new(1, 3), IVec2::new(9, 3)],
                        },
                    ],
                },
            ];
            let held_karp_state: Option<Box<HeldKarpState>> = Solution::try_held_karp_state(&pois);

            Solution {
                grid,
                pois,
                held_karp_state,
            }
        })
    }

    #[test]
    fn test_try_from_str() {
        let test_solution: Solution = Solution::try_from(SOLUTION_STR).unwrap();

        assert_eq!(test_solution.grid, solution().grid);
        assert_eq!(test_solution.pois, solution().pois);
    }

    #[test]
    fn test_try_min_hamiltonian_path_dist() {
        assert_eq!(solution().try_min_hamiltonian_path_dist(), Some(14_i32));
    }
}
