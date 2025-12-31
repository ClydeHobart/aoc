use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::{
        collections::{HashMap, HashSet, VecDeque},
        mem::take,
    },
    strum::IntoEnumIterator,
};

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Path = PATH = b'.',
        Forest = FOREST = b'#',
        NorthSlope = NORTH_SLOPE = b'^',
        EastSlope = EAST_SLOPE = b'>',
        SouthSlope = SOUTH_SLOPE = b'v',
        WestSlope = WEST_SLOPE = b'<',
        Start = START = b'S',
        HikePath = HIKE_PATH = b'O',
    }
}

impl Cell {
    fn slope_dir(self) -> Option<Direction> {
        match self {
            Cell::NorthSlope => Some(Direction::North),
            Cell::EastSlope => Some(Direction::East),
            Cell::SouthSlope => Some(Direction::South),
            Cell::WestSlope => Some(Direction::West),
            _ => None,
        }
    }

    fn is_valid_path(self) -> bool {
        matches!(
            self,
            Self::Path
                | Self::NorthSlope
                | Self::EastSlope
                | Self::SouthSlope
                | Self::WestSlope
                | Self::HikePath
        )
    }
}

#[derive(Clone, Copy)]
struct HikeFinderCell {
    cost: i16,
    prev: Option<Direction>,
}

impl HikeFinderCell {
    const MAX_COST: Self = Self {
        cost: i16::MAX,
        prev: None,
    };
}

struct OldLongestHikeFinder<'s> {
    solution: &'s Solution,
    grid: Grid2D<HikeFinderCell>,
}

impl<'s> OldLongestHikeFinder<'s> {
    fn start(&self) -> &IVec2 {
        &self.solution.start
    }

    fn path_to(&self, vertex: &IVec2) -> Vec<IVec2> {
        let mut vertex: IVec2 = *vertex;
        let mut path: VecDeque<IVec2> = VecDeque::new();

        while vertex != self.solution.start {
            path.push_front(vertex);
            vertex = vertex + self.grid.get(vertex).unwrap().prev.unwrap().vec();
        }

        path.push_front(self.solution.start);

        path.into()
    }
}

impl<'s> BreadthFirstSearch for OldLongestHikeFinder<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        OldLongestHikeFinder::start(self)
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        OldLongestHikeFinder::path_to(self, vertex)
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        if let Some(dir) = self.solution.grid.get(*vertex).unwrap().slope_dir() {
            neighbors.extend(
                [dir]
                    .into_iter()
                    .filter_map(self.solution.filter_map_dir_to_valid_neighbors(*vertex)),
            )
        } else {
            neighbors.extend(
                Direction::iter()
                    .filter_map(self.solution.filter_map_dir_to_valid_neighbors(*vertex)),
            )
        }
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let cost: i16 = self.grid.get(*from).unwrap().cost + 1_i16;

        *self.grid.get_mut(*to).unwrap() = HikeFinderCell {
            cost,
            prev: Direction::try_from(*to..*from).ok(),
        }
    }

    fn reset(&mut self) {
        self.grid.cells_mut().fill(HikeFinderCell::MAX_COST);
        self.grid.get_mut(self.solution.start).unwrap().cost = 0_i16;
    }
}

struct NewLongestHikeFinder<'s> {
    old_longest_hike_finder: OldLongestHikeFinder<'s>,
    in_edges: HashMap<IVec2, [bool; 4_usize]>,
}

impl<'s> NewLongestHikeFinder<'s> {
    fn add_edge(&mut self, from: IVec2, to: IVec2) {
        let dir: Direction = Direction::try_from(from - to).unwrap();

        // If the longest hike finder found that `from`` actually came from `to`, ignore this edge.
        if self.old_longest_hike_finder.grid.get(from).unwrap().prev != Some(dir.rev()) {
            if !self.in_edges.contains_key(&to) {
                self.in_edges.insert(to, Default::default());
            }

            self.in_edges.get_mut(&to).unwrap()[dir as usize] = true;
        }
    }

    fn longest_hike(&mut self) -> Option<Vec<IVec2>> {
        let topological_ordering: Vec<IVec2> = self.run()?;

        // Reset the edge set
        self.reset();

        let mut max_path_length: HashMap<IVec2, HikeFinderCell> = HashMap::new();

        for pos in topological_ordering {
            max_path_length.insert(
                pos,
                self.in_edges
                    .get(&pos)
                    .into_iter()
                    .flat_map(|is_dir_in_neighbor| {
                        is_dir_in_neighbor.into_iter().enumerate().filter_map(
                            |(index, is_dir_in_neighbor)| {
                                is_dir_in_neighbor.then(|| {
                                    let dir: Direction = Direction::from_u8(index as u8);

                                    HikeFinderCell {
                                        cost: max_path_length.get(&(pos + dir.vec())).unwrap().cost
                                            + 1_i16,
                                        prev: Some(dir),
                                    }
                                })
                            },
                        )
                    })
                    .max_by_key(|hike_finder_cell| hike_finder_cell.cost)
                    .unwrap_or(HikeFinderCell {
                        cost: 0_i16,
                        prev: None,
                    }),
            );
        }

        let mut longest_hike: VecDeque<IVec2> = VecDeque::new();
        let mut vertex: IVec2 = self.old_longest_hike_finder.solution.end;

        while vertex != self.old_longest_hike_finder.solution.start {
            longest_hike.push_front(vertex);
            vertex = vertex + max_path_length.get(&vertex).unwrap().prev.unwrap().vec();
        }

        longest_hike.push_front(vertex);

        Some(longest_hike.into())
    }
}

impl<'s> Kahn for NewLongestHikeFinder<'s> {
    type Vertex = IVec2;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>) {
        initial_set.clear();
        initial_set.push_back(self.old_longest_hike_finder.solution.start);
    }

    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            self.in_edges
                .get(&neighbor)
                .map_or(false, |is_dir_in_neighbor| {
                    is_dir_in_neighbor[dir.rev() as usize]
                })
                .then_some(neighbor)
        }));
    }

    fn remove_edge(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let is_dir_in_neighbor: &mut [bool; 4_usize] = self.in_edges.get_mut(to).unwrap();

        is_dir_in_neighbor[Direction::try_from(*from - *to).unwrap() as usize] = false;

        if !is_dir_in_neighbor.iter().any(|x| *x) {
            self.in_edges.remove(to);
        }
    }

    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool {
        self.in_edges
            .get(vertex)
            .map_or(false, |is_dir_in_neighbor| {
                is_dir_in_neighbor.iter().any(|x| *x)
            })
    }

    fn any_edges_exist(&self) -> bool {
        !self.in_edges.is_empty()
    }

    fn reset(&mut self) {
        self.in_edges.clear();
        self.old_longest_hike_finder.run();

        for pos in self
            .old_longest_hike_finder
            .solution
            .grid
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| {
                cell.is_valid_path().then(|| {
                    self.old_longest_hike_finder
                        .solution
                        .grid
                        .pos_from_index(index)
                })
            })
        {
            if let Some(dir) = self
                .old_longest_hike_finder
                .solution
                .grid
                .get(pos)
                .unwrap()
                .slope_dir()
            {
                for neighbor in [dir].into_iter().filter_map(
                    self.old_longest_hike_finder
                        .solution
                        .filter_map_dir_to_valid_neighbors(pos),
                ) {
                    self.add_edge(pos, neighbor);
                }
            } else {
                for neighbor in Direction::iter().filter_map(
                    self.old_longest_hike_finder
                        .solution
                        .filter_map_dir_to_valid_neighbors(pos),
                ) {
                    self.add_edge(pos, neighbor);
                }
            }
        }
    }

    fn order_set(&self, _set: &mut VecDeque<Self::Vertex>) {}
}

struct ReducedNeighbor {
    pos: IVec2,
    dir: Direction,
    dist: i16,
}

#[cfg_attr(test, derive(Debug))]
#[derive(Default)]
struct MapValue {
    neighbors: [IVec2; Self::ARRAY_LEN],
    dirs: [Option<Direction>; Self::ARRAY_LEN],
    dists: [i16; Self::ARRAY_LEN],
    len: u32,
}

impl MapValue {
    const ARRAY_LEN: usize = 4_usize;

    fn contains_neighbor(&self, target_neighbor: IVec2) -> bool {
        self.neighbors
            .iter()
            .any(|neighbor| *neighbor == target_neighbor)
    }

    fn push_neighbor(&mut self, neighbor: IVec2, dir: Direction, dist: i16) {
        if !self.contains_neighbor(neighbor) {
            assert!((self.len as usize) < Self::ARRAY_LEN);

            self.neighbors[self.len as usize] = neighbor;
            self.dirs[self.len as usize] = Some(dir);
            self.dists[self.len as usize] = dist;
            self.len += 1_u32;
        }
    }

    fn get_neighbor(&self, index: usize) -> Option<ReducedNeighbor> {
        (index < self.len as usize).then(|| ReducedNeighbor {
            pos: self.neighbors[index],
            dir: self.dirs[index].unwrap(),
            dist: self.dists[index],
        })
    }

    fn iter_neighbors(&self) -> impl Iterator<Item = ReducedNeighbor> + '_ {
        (0_usize..self.len as usize).map(|index| self.get_neighbor(index).unwrap())
    }
}

type ReducedGraph = HashMap<IVec2, MapValue>;

#[derive(Default)]
struct LongestHikeOnDryTrailsFinder {
    longest_known_hike: Option<(Vec<IVec2>, i16)>,
    stack: Vec<(IVec2, usize, i16)>,
    visited: HashSet<IVec2>,
}

impl LongestHikeOnDryTrailsFinder {
    fn find_longest_hike(
        &mut self,
        solution: &Solution,
        reduced_graph: &ReducedGraph,
    ) -> Option<Vec<IVec2>> {
        self.longest_known_hike = None;
        self.stack.clear();
        self.visited.clear();
        self.stack.push((solution.start, 0_usize, 0_i16));
        self.visited.insert(solution.start);

        while let Some((curr_pos, neighbor_index, cost)) = self.stack.last().copied() {
            if curr_pos == solution.end {
                if self.longest_known_hike.is_none() {
                    self.longest_known_hike = Some((Vec::new(), 0_i16));
                }

                let (longest_known_hike, longest_known_cost): &mut (Vec<IVec2>, i16) =
                    self.longest_known_hike.as_mut().unwrap();

                if cost > *longest_known_cost {
                    *longest_known_cost = cost;
                    longest_known_hike.clear();
                    longest_known_hike.extend(self.stack.iter().map(|(pos, _, _)| *pos));
                }

                // Don't explore any children of the end
                self.stack.last_mut().unwrap().1 = MapValue::ARRAY_LEN;
            }

            if let Some(next_reduced_neighbor) = reduced_graph
                .get(&curr_pos)
                .unwrap()
                .get_neighbor(neighbor_index)
            {
                let next: (IVec2, usize, i16) = (
                    next_reduced_neighbor.pos,
                    0_usize,
                    cost + next_reduced_neighbor.dist,
                );

                self.stack.last_mut().unwrap().1 += 1_usize;

                if !self.visited.contains(&next.0) {
                    self.visited.insert(next.0);
                    self.stack.push(next);
                }
            } else {
                self.visited.remove(&curr_pos);
                self.stack.pop();
            }
        }

        take(&mut self.longest_known_hike).map(|(longest_known_hike, _)| longest_known_hike)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Cell>,
    start: IVec2,
    end: IVec2,
}

impl Solution {
    fn filter_map_dir_to_valid_neighbors(
        &self,
        pos: IVec2,
    ) -> impl Fn(Direction) -> Option<IVec2> + '_ {
        move |dir| {
            let pos: IVec2 = pos + dir.vec();

            self.grid.get(pos).and_then(|cell| {
                match cell {
                    Cell::Path => true,
                    Cell::NorthSlope | Cell::EastSlope | Cell::SouthSlope | Cell::WestSlope => {
                        cell.slope_dir() != Some(dir.rev())
                    }
                    _ => false,
                }
                .then_some(pos)
            })
        }
    }

    fn old_longest_hike_finder<'s>(&'s self) -> OldLongestHikeFinder<'s> {
        OldLongestHikeFinder {
            solution: self,
            grid: Grid2D::try_from_cells_and_dimensions(
                vec![HikeFinderCell::MAX_COST; self.grid.cells().len()],
                self.grid.dimensions(),
            )
            .unwrap(),
        }
    }

    fn longest_hike(&self) -> Option<Vec<IVec2>> {
        let mut longest_hike_finder: NewLongestHikeFinder = NewLongestHikeFinder {
            old_longest_hike_finder: self.old_longest_hike_finder(),
            in_edges: HashMap::new(),
        };

        longest_hike_finder.longest_hike()
    }

    fn map_longest_hike_to_len(longest_hike: Vec<IVec2>) -> usize {
        longest_hike.len() - 1_usize
    }

    fn longest_hike_len(&self) -> Option<usize> {
        self.longest_hike().map(Self::map_longest_hike_to_len)
    }

    fn map_longest_hike_to_len_and_string(&self) -> impl Fn(Vec<IVec2>) -> (usize, String) + '_ {
        |longest_hike| {
            let mut grid: Grid2D<Cell> = self.grid.clone();

            for pos in longest_hike.iter().copied() {
                *grid.get_mut(pos).unwrap() = Cell::HikePath;
            }

            *grid.get_mut(self.start).unwrap() = Cell::Start;

            (Self::map_longest_hike_to_len(longest_hike), grid.into())
        }
    }

    fn longest_hike_len_and_string(&self) -> Option<(usize, String)> {
        self.longest_hike()
            .map(self.map_longest_hike_to_len_and_string())
    }

    fn iter_valid_paths(&self, pos: IVec2) -> impl Iterator<Item = IVec2> + '_ {
        Direction::iter().filter_map(move |dir| {
            let pos: IVec2 = pos + dir.vec();

            self.grid
                .get(pos)
                .copied()
                .map(Cell::is_valid_path)
                .unwrap_or_default()
                .then_some(pos)
        })
    }

    fn find_reduced_neighbor(
        &self,
        reduced_graph: &HashMap<IVec2, MapValue>,
        pos: IVec2,
        neighbor_pos: IVec2,
    ) -> Option<(IVec2, i16)> {
        let mut prev_pos: IVec2 = pos;
        let mut curr_pos: IVec2 = neighbor_pos;
        let mut dist: i16 = 1_i16;

        while !reduced_graph.contains_key(&curr_pos) {
            let next_pos: IVec2 = self
                .iter_valid_paths(curr_pos)
                .filter(|next_pos| *next_pos != prev_pos)
                .next()?;

            prev_pos = curr_pos;
            curr_pos = next_pos;
            dist += 1_i16;
        }

        Some((curr_pos, dist))
    }

    fn reduced_graph(&self) -> ReducedGraph {
        let reduced_graph_vertices: Vec<IVec2> = self
            .grid
            .cells()
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, cell)| {
                let pos: IVec2 = self.grid.pos_from_index(index);

                (cell.is_valid_path() && self.iter_valid_paths(pos).count() > 2_usize)
                    .then_some(pos)
            })
            .chain([self.start, self.end])
            .collect();

        let mut reduced_graph: ReducedGraph = reduced_graph_vertices
            .iter()
            .map(|pos| (*pos, MapValue::default()))
            .collect();

        for from_pos in reduced_graph_vertices {
            for from_neighbor_pos in self.iter_valid_paths(from_pos) {
                if let Some((to_pos, dist)) =
                    self.find_reduced_neighbor(&reduced_graph, from_pos, from_neighbor_pos)
                {
                    reduced_graph.get_mut(&from_pos).unwrap().push_neighbor(
                        to_pos,
                        Direction::try_from(from_neighbor_pos - from_pos).unwrap(),
                        dist,
                    );
                }
            }
        }

        reduced_graph
    }

    fn longest_hike_on_dry_trails(&self) -> Option<Vec<IVec2>> {
        let reduced_graph: ReducedGraph = self.reduced_graph();
        let reduced_longest_hike_on_dry_trails: Vec<IVec2> =
            LongestHikeOnDryTrailsFinder::default().find_longest_hike(self, &reduced_graph)?;

        let mut longest_hike_on_dry_trails: Vec<IVec2> = Vec::new();

        for consecutive_reduced_vertices in reduced_longest_hike_on_dry_trails.windows(2_usize) {
            let curr_reduced_pos: IVec2 = consecutive_reduced_vertices[0_usize];
            let next_reduced_pos: IVec2 = consecutive_reduced_vertices[1_usize];

            longest_hike_on_dry_trails.push(curr_reduced_pos);

            let mut prev_pos: IVec2 = curr_reduced_pos;
            let mut curr_pos: IVec2 = curr_reduced_pos
                + reduced_graph
                    .get(&curr_reduced_pos)
                    .unwrap()
                    .iter_neighbors()
                    .find(|reduced_neighbor| reduced_neighbor.pos == next_reduced_pos)
                    .unwrap()
                    .dir
                    .vec();

            while !reduced_graph.contains_key(&curr_pos) {
                longest_hike_on_dry_trails.push(curr_pos);

                // This is safe to unwrap, since longest_hike_on_dry_trails doesn't point to any
                // dead ends.
                let next_pos: IVec2 = self
                    .iter_valid_paths(curr_pos)
                    .filter(|next_pos| *next_pos != prev_pos)
                    .next()
                    .unwrap();

                prev_pos = curr_pos;
                curr_pos = next_pos;
            }
        }

        longest_hike_on_dry_trails.push(self.end);

        Some(longest_hike_on_dry_trails)
    }

    fn longest_hike_on_dry_trails_len(&self) -> Option<usize> {
        self.longest_hike_on_dry_trails()
            .map(Self::map_longest_hike_to_len)
    }

    fn longest_hike_on_dry_trails_len_and_string(&self) -> Option<(usize, String)> {
        self.longest_hike_on_dry_trails()
            .map(self.map_longest_hike_to_len_and_string())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |grid| {
            let pos_is_path = |pos: &IVec2| grid.get(*pos) == Some(&Cell::Path);

            CellIter2D::corner(&grid, Direction::East)
                .find(pos_is_path)
                .and_then(|start| {
                    CellIter2D::corner(&grid, Direction::West)
                        .find(pos_is_path)
                        .map(|end| (start, end))
                })
                .map(|(start, end)| Self { grid, start, end })
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((longest_hike_len, longest_hike_string)) =
                self.longest_hike_len_and_string()
            {
                dbg!(longest_hike_len);
                println!("\n\n{longest_hike_string}\n");
            } else {
                eprintln!("Failed to find new longest hike.");
            }
        } else {
            dbg!(self.longest_hike_len());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((longest_hike_on_dry_trails_len, longest_hike_on_dry_trails_string)) =
                self.longest_hike_on_dry_trails_len_and_string()
            {
                dbg!(longest_hike_on_dry_trails_len);
                println!("\n\n{longest_hike_on_dry_trails_string}\n");
            } else {
                eprintln!("Failed to find longest hike on dry trails.");
            }
        } else {
            dbg!(self.longest_hike_on_dry_trails_len());
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

    const SOLUTION_STR: &'static str = concat!(
        "#.#####################\n", // 0
        "#.......#########...###\n", // 1
        "#######.#########.#.###\n", // 2
        "###.....#.>.>.###.#.###\n", // 3
        "###v#####.#v#.###.#.###\n", // 4
        "###.>...#.#.#.....#...#\n", // 5
        "###v###.#.#.#########.#\n", // 6
        "###...#.#.#.......#...#\n", // 7
        "#####.#.#.#######.#.###\n", // 8
        "#.....#.#.#.......#...#\n", // 9
        "#.#####.#.#.#########v#\n", // 10
        "#.#...#...#...###...>.#\n", // 11
        "#.#.#v#######v###.###v#\n", // 12
        "#...#.>.#...>.>.#.###.#\n", // 13
        "#####v#.#.###v#.#.###.#\n", // 14
        "#.....#...#...#.#.#...#\n", // 15
        "#.#########.###.#.#.###\n", // 16
        "#...###...#...#...#.###\n", // 17
        "###.###.#.###v#####v###\n", // 18
        "#...#...#.#.>.>.#.>.###\n", // 19
        "#.###.###.#.###.#.#v###\n", // 20
        "#.....###...###...#...#\n", // 21
        "#####################.#\n", // 22
    );
    const LONGEST_HIKE_STR: &'static str = "\
        #S#####################\n\
        #OOOOOOO#########...###\n\
        #######O#########.#.###\n\
        ###OOOOO#OOO>.###.#.###\n\
        ###O#####O#O#.###.#.###\n\
        ###OOOOO#O#O#.....#...#\n\
        ###v###O#O#O#########.#\n\
        ###...#O#O#OOOOOOO#...#\n\
        #####.#O#O#######O#.###\n\
        #.....#O#O#OOOOOOO#...#\n\
        #.#####O#O#O#########v#\n\
        #.#...#OOO#OOO###OOOOO#\n\
        #.#.#v#######O###O###O#\n\
        #...#.>.#...>OOO#O###O#\n\
        #####v#.#.###v#O#O###O#\n\
        #.....#...#...#O#O#OOO#\n\
        #.#########.###O#O#O###\n\
        #...###...#...#OOO#O###\n\
        ###.###.#.###v#####O###\n\
        #...#...#.#.>.>.#.>O###\n\
        #.###.###.#.###.#.#O###\n\
        #.....###...###...#OOO#\n\
        #####################O#\n";
    const LONGEST_HIKE_ON_DRY_TRAILS_STR: &'static str = "\
        #S#####################\n\
        #OOOOOOO#########OOO###\n\
        #######O#########O#O###\n\
        ###OOOOO#.>OOO###O#O###\n\
        ###O#####.#O#O###O#O###\n\
        ###O>...#.#O#OOOOO#OOO#\n\
        ###O###.#.#O#########O#\n\
        ###OOO#.#.#OOOOOOO#OOO#\n\
        #####O#.#.#######O#O###\n\
        #OOOOO#.#.#OOOOOOO#OOO#\n\
        #O#####.#.#O#########O#\n\
        #O#OOO#...#OOO###...>O#\n\
        #O#O#O#######O###.###O#\n\
        #OOO#O>.#...>O>.#.###O#\n\
        #####O#.#.###O#.#.###O#\n\
        #OOOOO#...#OOO#.#.#OOO#\n\
        #O#########O###.#.#O###\n\
        #OOO###OOO#OOO#...#O###\n\
        ###O###O#O###O#####O###\n\
        #OOO#OOO#O#OOO>.#.>O###\n\
        #O###O###O#O###.#.#O###\n\
        #OOOOO###OOO###...#OOO#\n\
        #####################O#\n";

    fn solution() -> &'static Solution {
        use Cell::{EastSlope as E, Forest as F, Path as P, SouthSlope as S};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            grid: Grid2D::try_from_cells_and_width(
                vec![
                    F, P, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, P, P,
                    P, P, P, P, P, F, F, F, F, F, F, F, F, F, P, P, P, F, F, F, F, F, F, F, F, F,
                    F, P, F, F, F, F, F, F, F, F, F, P, F, P, F, F, F, F, F, F, P, P, P, P, P, F,
                    P, E, P, E, P, F, F, F, P, F, P, F, F, F, F, F, F, S, F, F, F, F, F, P, F, S,
                    F, P, F, F, F, P, F, P, F, F, F, F, F, F, P, E, P, P, P, F, P, F, P, F, P, P,
                    P, P, P, F, P, P, P, F, F, F, F, S, F, F, F, P, F, P, F, P, F, F, F, F, F, F,
                    F, F, F, P, F, F, F, F, P, P, P, F, P, F, P, F, P, P, P, P, P, P, P, F, P, P,
                    P, F, F, F, F, F, F, P, F, P, F, P, F, F, F, F, F, F, F, P, F, P, F, F, F, F,
                    P, P, P, P, P, F, P, F, P, F, P, P, P, P, P, P, P, F, P, P, P, F, F, P, F, F,
                    F, F, F, P, F, P, F, P, F, F, F, F, F, F, F, F, F, S, F, F, P, F, P, P, P, F,
                    P, P, P, F, P, P, P, F, F, F, P, P, P, E, P, F, F, P, F, P, F, S, F, F, F, F,
                    F, F, F, S, F, F, F, P, F, F, F, S, F, F, P, P, P, F, P, E, P, F, P, P, P, E,
                    P, E, P, F, P, F, F, F, P, F, F, F, F, F, F, S, F, P, F, P, F, F, F, S, F, P,
                    F, P, F, F, F, P, F, F, P, P, P, P, P, F, P, P, P, F, P, P, P, F, P, F, P, F,
                    P, P, P, F, F, P, F, F, F, F, F, F, F, F, F, P, F, F, F, P, F, P, F, P, F, F,
                    F, F, P, P, P, F, F, F, P, P, P, F, P, P, P, F, P, P, P, F, P, F, F, F, F, F,
                    F, P, F, F, F, P, F, P, F, F, F, S, F, F, F, F, F, S, F, F, F, F, P, P, P, F,
                    P, P, P, F, P, F, P, E, P, E, P, F, P, E, P, F, F, F, F, P, F, F, F, P, F, F,
                    F, P, F, P, F, F, F, P, F, P, F, S, F, F, F, F, P, P, P, P, P, F, F, F, P, P,
                    P, F, F, F, P, P, P, F, P, P, P, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F,
                    F, F, F, F, F, F, F, P, F,
                ],
                23_usize,
            )
            .unwrap(),
            start: IVec2::new(1_i32, 0_i32),
            end: IVec2::new(21_i32, 22_i32),
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn try_longest_hike() {
        assert_eq!(
            solution().longest_hike_len_and_string(),
            Some((94_usize, LONGEST_HIKE_STR.into()))
        );
    }

    #[test]
    fn test_longest_hike_on_dry_trails() {
        assert_eq!(
            solution().longest_hike_on_dry_trails_len_and_string(),
            Some((154_usize, LONGEST_HIKE_ON_DRY_TRAILS_STR.into()))
        );
    }
}
