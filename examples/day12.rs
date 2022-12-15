use {
    aoc_2022::*,
    glam::IVec2,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet, VecDeque},
        fmt::{Debug, Formatter, Result as FmtResult},
        hash::Hash,
        mem::take,
        ops::Add,
    },
    strum::IntoEnumIterator,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeightCell(u8);

impl HeightCell {
    const START: u8 = 'S' as u8;
    const END: u8 = 'E' as u8;
}

#[derive(Debug, PartialEq)]
struct InvalidHeightCellChar(char);

impl Default for HeightCell {
    fn default() -> Self {
        Self(u8::MAX)
    }
}

impl TryFrom<char> for HeightCell {
    type Error = InvalidHeightCellChar;

    fn try_from(height_cell_char: char) -> Result<Self, Self::Error> {
        if height_cell_char.is_ascii_lowercase()
            || height_cell_char == Self::START as char
            || height_cell_char == Self::END as char
        {
            Ok(Self(height_cell_char as u8))
        } else {
            Err(InvalidHeightCellChar(height_cell_char))
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct IsVisitable(u8);

impl IsVisitable {
    const FROM_OFFSET: u32 = 4_u32;

    #[inline(always)]
    fn shift(byte: u8, dir: Direction, to: bool) -> u8 {
        byte << if to {
            dir as u32
        } else {
            dir as u32 + Self::FROM_OFFSET
        }
    }

    #[inline(always)]
    fn mask(dir: Direction, to: bool) -> u8 {
        Self::shift(1_u8, dir, to)
    }

    fn get(self, dir: Direction, to: bool) -> bool {
        (self.0 & Self::mask(dir, to)) != 0_u8
    }

    fn set(&mut self, dir: Direction, to: bool, value: bool) {
        self.0 = (self.0 & !Self::mask(dir, to)) | Self::shift(value as u8, dir, to)
    }

    fn array(&self) -> [(Direction, bool); 8_usize] {
        let mut array: [(Direction, bool); 8_usize] = [(Direction::North, false); 8_usize];

        for dir in Direction::iter() {
            array[dir as usize] = (dir, self.get(dir, true));
            array[dir as usize + Self::FROM_OFFSET as usize] = (dir, self.get(dir, false));
        }

        array
    }
}

impl Debug for IsVisitable {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let array: [(Direction, bool); 8_usize] = self.array();

        f.debug_struct("IsVisitable")
            .field("to", &&array[..Self::FROM_OFFSET as usize])
            .field("from", &&array[..Self::FROM_OFFSET as usize])
            .finish()
    }
}

struct IsVisitableGridVisitor(u8);

impl Default for IsVisitableGridVisitor {
    fn default() -> Self {
        Self(u8::MAX)
    }
}

impl GridVisitor for IsVisitableGridVisitor {
    type Old = HeightCell;
    type New = IsVisitable;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        old_grid: &Grid<Self::Old>,
        rev_dir: Direction,
        pos: IVec2,
    ) {
        new.set(rev_dir, true, old.0 >= self.0 - 1_u8);
        new.set(
            rev_dir,
            false,
            old_grid.contains(pos + rev_dir.vec()) && self.0 >= old.0 - 1_u8,
        );
        self.0 = old.0;
    }
}

struct ShortestPath {
    score: usize,
    predecessor: Option<Direction>,
}

impl Default for ShortestPath {
    fn default() -> Self {
        Self {
            score: usize::MAX,
            predecessor: None,
        }
    }
}

#[derive(Debug, PartialEq)]
struct HeightGrid {
    heights: Grid<HeightCell>,
    is_visitable: Grid<IsVisitable>,
    start: IVec2,
    end: IVec2,
}

#[derive(Debug, PartialEq)]
enum HeightGridParseError<'s> {
    FailedToParseGrid(GridParseError<'s, InvalidHeightCellChar>),
    GridContainsNoStartPosition,
    GridContainsNoEndPosition,
}

impl<'s> TryFrom<&'s str> for HeightGrid {
    type Error = HeightGridParseError<'s>;

    fn try_from(height_grid_str: &'s str) -> Result<Self, Self::Error> {
        use HeightGridParseError::*;

        let mut heights: Grid<HeightCell> =
            height_grid_str.try_into().map_err(FailedToParseGrid)?;

        let start: IVec2 = heights.pos_from_index(
            heights
                .cells()
                .iter()
                .position(|height_cell| height_cell.0 == HeightCell::START)
                .ok_or(GridContainsNoStartPosition)?,
        );
        let end: IVec2 = heights.pos_from_index(
            heights
                .cells()
                .iter()
                .position(|height_cell| height_cell.0 == HeightCell::END)
                .ok_or(GridContainsNoEndPosition)?,
        );

        heights.get_mut(start).unwrap().0 = b'a';
        heights.get_mut(end).unwrap().0 = b'z';

        let is_visitable: Grid<IsVisitable> = IsVisitableGridVisitor::visit_grid(&heights);

        Ok(HeightGrid {
            heights,
            is_visitable,
            start,
            end,
        })
    }
}

struct OpenSetElement<V: Clone + PartialEq, C: Clone + Ord>(V, C);

impl<V: Clone + PartialEq, C: Clone + Ord> PartialEq for OpenSetElement<V, C> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> PartialOrd for OpenSetElement<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse the order so that cost is minimized when popping from the heap
        Some(other.1.cmp(&self.1))
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> Eq for OpenSetElement<V, C> {}

impl<V: Clone + PartialEq, C: Clone + Ord> Ord for OpenSetElement<V, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the order so that cost is minimized when popping from the heap
        other.1.cmp(&self.1)
    }
}

/// An implementation of https://en.wikipedia.org/wiki/A*_search_algorithm
trait AStar: Sized {
    type Vertex: Clone + Eq + Hash;
    type Cost: Add<Self::Cost, Output = Self::Cost> + Clone + Ord + Sized;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn cost_between_neighbors(&self, from: &Self::Vertex, to: &Self::Vertex) -> Self::Cost;
    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_score(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        heuristic: Self::Cost,
    );

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
        let start: Self::Vertex = self.start().clone();

        let mut open_set_heap: BinaryHeap<OpenSetElement<Self::Vertex, Self::Cost>> =
            BinaryHeap::new();
        let mut open_set_set: HashSet<Self::Vertex> = HashSet::new();

        open_set_heap.push(OpenSetElement(start.clone(), self.cost_from_start(&start)));
        open_set_set.insert(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        // A pair, where the first field is the new cost for the neighbor, already passed into
        // `update_score`, and a bool representing whether the neighbor was previously in
        // `open_set_set`, meaning `open_set_heap` requires special attention to update its score
        let mut neighbor_updates: Vec<Option<(Self::Cost, bool)>> = Vec::new();
        let mut any_update_was_in_open_set_set: bool = false;

        while let Some(OpenSetElement(current, _)) = open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            let start_to_current: Self::Cost = self.cost_from_start(&current);

            open_set_set.remove(&current);
            self.neighbors(&current, &mut neighbors);
            neighbor_updates.reserve(neighbors.len());

            for neighbor in neighbors.iter() {
                let start_to_neighbor: Self::Cost =
                    start_to_current.clone() + self.cost_between_neighbors(&current, &neighbor);

                if start_to_neighbor < self.cost_from_start(&neighbor) {
                    let neighbor_heuristic: Self::Cost = self.heuristic(&neighbor);

                    self.update_score(
                        &current,
                        &neighbor,
                        start_to_neighbor.clone(),
                        neighbor_heuristic.clone(),
                    );

                    let was_in_open_set_set: bool = !open_set_set.insert(neighbor.clone());

                    neighbor_updates.push(Some((
                        start_to_neighbor + neighbor_heuristic,
                        was_in_open_set_set,
                    )));
                    any_update_was_in_open_set_set |= was_in_open_set_set;
                } else {
                    neighbor_updates.push(None);
                }
            }

            if any_update_was_in_open_set_set {
                // Convert to a vec first, add the new elements, then convert back, so that we don't
                // waste time during `push` operations only to have that effort ignored when
                // converting back to a heap
                let mut open_set_elements: Vec<OpenSetElement<Self::Vertex, Self::Cost>> =
                    take(&mut open_set_heap).into_vec();

                let old_element_count: usize = open_set_elements.len();

                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, is_in_open_set_elements)) = neighbor_update {
                        if *is_in_open_set_elements {
                            if let Some(index) = open_set_elements[..old_element_count]
                                .iter()
                                .position(|OpenSetElement(vertex, _)| *vertex == *neighbor)
                            {
                                open_set_elements[index].1 = cost.clone();
                            }
                        } else {
                            open_set_elements.push(OpenSetElement(neighbor.clone(), cost.clone()));
                        }
                    }
                }

                open_set_heap = open_set_elements.into();
            } else {
                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, _)) = neighbor_update {
                        open_set_heap.push(OpenSetElement(neighbor.clone(), cost.clone()));
                    }
                }
            }

            neighbors.clear();
            neighbor_updates.clear();
            any_update_was_in_open_set_set = false;
        }

        None
    }
}

struct HeightGridAStarAscent<'h> {
    height_grid: &'h HeightGrid,
    shortest_paths: Grid<ShortestPath>,
}

impl<'h> HeightGridAStarAscent<'h> {
    fn new(height_grid: &'h HeightGrid) -> Self {
        let mut shortest_paths: Grid<ShortestPath> =
            Grid::default(height_grid.heights.dimensions());

        shortest_paths.get_mut(height_grid.start).unwrap().score = 0_usize;

        Self {
            height_grid,
            shortest_paths,
        }
    }
}

impl<'h> AStar for HeightGridAStarAscent<'h> {
    type Vertex = IVec2;
    type Cost = usize;

    fn start(&self) -> &Self::Vertex {
        &self.height_grid.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.height_grid.end
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut vertex: IVec2 = *vertex;

        let shortest_path: &ShortestPath = self.shortest_paths.get(vertex).unwrap();

        let mut predecessor_option: Option<Direction> = shortest_path.predecessor.clone();
        let mut path: Vec<IVec2> = Vec::with_capacity(shortest_path.score + 1_usize);

        while let Some(predecessor) = predecessor_option {
            path.push(vertex);
            vertex += predecessor.vec();
            predecessor_option = self.shortest_paths.get(vertex).unwrap().predecessor.clone();
        }

        path.push(self.height_grid.start);
        path.reverse();

        path
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.shortest_paths.get(*vertex).unwrap().score
    }

    fn cost_between_neighbors(&self, _: &Self::Vertex, _: &Self::Vertex) -> Self::Cost {
        1_usize
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        let abs: IVec2 = (self.height_grid.end - *vertex).abs();

        (abs.x + abs.y) as usize
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        let is_visitable: IsVisitable = *self.height_grid.is_visitable.get(*vertex).unwrap();

        for dir in Direction::iter() {
            if is_visitable.get(dir, true) {
                neighbors.push(*vertex + dir.vec());
            }
        }
    }

    fn update_score(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _: Self::Cost,
    ) {
        let shortest_path: &mut ShortestPath = self.shortest_paths.get_mut(*to).unwrap();

        shortest_path.predecessor = Some((*from - *to).try_into().unwrap());
        shortest_path.score = cost;
    }
}

/// An implementation of https://en.wikipedia.org/wiki/Breadth-first_search
trait BreadthFirstSearch: Sized {
    type Vertex: Clone + Debug + Eq + Hash;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex);

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
        let mut queue: VecDeque<Self::Vertex> = VecDeque::new();
        let mut explored: HashSet<Self::Vertex> = HashSet::new();

        let start: Self::Vertex = self.start().clone();
        explored.insert(start.clone());
        queue.push_back(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        while let Some(current) = queue.pop_front() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            self.neighbors(&current, &mut neighbors);

            for neighbor in neighbors.drain(..) {
                if explored.insert(neighbor.clone()) {
                    self.update_parent(&current, &neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}

struct HeightGridBreadthFirstSearchDescent<'h> {
    height_grid: &'h HeightGrid,
    predecessors: Grid<Option<Direction>>,
}

impl<'h> HeightGridBreadthFirstSearchDescent<'h> {
    fn new(height_grid: &'h HeightGrid) -> Self {
        Self {
            height_grid,
            predecessors: Grid::default(height_grid.heights.dimensions()),
        }
    }
}

impl<'h> BreadthFirstSearch for HeightGridBreadthFirstSearchDescent<'h> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.height_grid.end
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        self.height_grid.heights.get(*vertex).unwrap().0 <= b'a'
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        // We'll cheat a bit here. Technically, we're supposed to return a path from `self.start()`
        // to whereever made `self.is_end` return true, but that's the opposite order from what
        // we want in the context of this trait implementation. It's easier to collect in the
        // opposite order and just return that instead of flip it once within this function, and
        // then once more after receiving the `Vec` in the calling context 🤫
        let mut path: Vec<IVec2> = Vec::new();
        let mut vertex: IVec2 = *vertex;

        path.push(vertex);

        while let Some(predecessor) = self.predecessors.get(vertex).unwrap() {
            vertex += predecessor.vec();
            path.push(vertex);
        }

        path
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        let is_visitable: IsVisitable = *self.height_grid.is_visitable.get(*vertex).unwrap();

        for dir in Direction::iter() {
            if is_visitable.get(dir, false) {
                neighbors.push(*vertex + dir.vec());
            }
        }
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        *self.predecessors.get_mut(*to).unwrap() = Some((*from - *to).try_into().unwrap());
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day12.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match HeightGrid::try_from(input) {
                    Ok(height_grid) => {
                        let a_star_path: Vec<IVec2> = HeightGridAStarAscent::new(&height_grid)
                            .run()
                            .expect("Failed to obtain a path from A*");
                        let bfs_path: Vec<IVec2> =
                            HeightGridBreadthFirstSearchDescent::new(&height_grid)
                                .run()
                                .expect("Failed to obtain a path from BFS");

                        println!(
                            "a_star_path.len() == {} (steps == {})\n\
                            bfs_path.len() == {} (steps == {})\n\n\
                            a_star_path == {a_star_path:#?}\n\n
                            bfs_path == {bfs_path:#?}",
                            a_star_path.len(),
                            a_star_path.len() - 1_usize,
                            bfs_path.len(),
                            bfs_path.len() - 1_usize
                        );
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

    const HEIGHT_GRID_STR: &str = concat!(
        "Sabqponm\n",
        "abcryxxl\n",
        "accszExk\n",
        "acctuvwj\n",
        "abdefghi",
    );
    const DIMENSIONS: IVec2 = IVec2::new(8_i32, 5_i32);
    const START: IVec2 = IVec2::ZERO;
    const END: IVec2 = IVec2::new(5_i32, 2_i32);

    #[test]
    fn test_height_grid_try_from_str() {
        match HeightGrid::try_from(HEIGHT_GRID_STR) {
            Ok(height_grid) => compare_height_grids(&height_grid, &example_height_grid()),
            Err(error) => panic!("{error:#?}"),
        }
    }

    fn compare_height_grids(a: &HeightGrid, b: &HeightGrid) {
        if *a != *b {
            if a.heights != b.heights {
                for (index, (a, b)) in a
                    .heights
                    .cells()
                    .iter()
                    .zip(b.heights.cells().iter())
                    .enumerate()
                {
                    assert_eq!(*a, *b, "`HeightCell`s at index {index} do not match");
                }
            }

            if a.is_visitable != b.is_visitable {
                for (index, (a, b)) in a
                    .is_visitable
                    .cells()
                    .iter()
                    .zip(b.is_visitable.cells().iter())
                    .enumerate()
                {
                    assert_eq!(*a, *b, "`IsVisitable`s at index {index} do not match");
                }
            }

            assert_eq!(a.start, b.start);
            assert_eq!(a.end, b.end);
        }
    }

    #[test]
    fn test_a_star_run() {
        let height_grid: HeightGrid = example_height_grid();
        let path: Vec<IVec2> = HeightGridAStarAscent::new(&height_grid).run().unwrap();

        // 31 total steps, but 32 elements including the start
        assert_eq!(path.len(), 32_usize);
    }

    #[test]
    fn test_breadth_first_search_run() {
        let height_grid: HeightGrid = example_height_grid();
        let path: Vec<IVec2> = HeightGridBreadthFirstSearchDescent::new(&height_grid)
            .run()
            .unwrap();

        // 29 total steps, but 30 elements including the start
        assert_eq!(path.len(), 30_usize);
    }

    fn example_height_grid() -> HeightGrid {
        let mut heights: Grid<HeightCell> = Grid::default(DIMENSIONS);
        let mut is_visitable: Grid<IsVisitable> = Grid::default(DIMENSIONS);

        for ((height_cell_char, is_visitable_i32), (height_cell, is_visitable)) in vec![
            // Row 0
            ('a', 0x66),
            ('a', 0xEE),
            ('b', 0xEC),
            ('q', 0x6E),
            ('p', 0xEA),
            ('o', 0xEA),
            ('n', 0xEA),
            ('m', 0xCC),
            // Row 1
            ('a', 0x77),
            ('b', 0xFF),
            ('c', 0xFD),
            ('r', 0x7D),
            ('y', 0x6F),
            ('x', 0xEB),
            ('x', 0xCF),
            ('l', 0xD5),
            // Row 2
            ('a', 0x75),
            ('c', 0x7F),
            ('c', 0xFD),
            ('s', 0x7D),
            ('z', 0x3F),
            ('z', 0x8F),
            ('x', 0xD7),
            ('k', 0xD5),
            // Row 3
            ('a', 0x75),
            ('c', 0x7F),
            ('c', 0xFD),
            ('t', 0x3F),
            ('u', 0xBE),
            ('v', 0xBE),
            ('w', 0x9F),
            ('j', 0xD5),
            // Row 4
            ('a', 0x33),
            ('b', 0xB9),
            ('d', 0x3B),
            ('e', 0xBA),
            ('f', 0xBA),
            ('g', 0xBA),
            ('h', 0xBA),
            ('i', 0x99),
        ]
        .iter()
        .zip(
            heights
                .cells_mut()
                .iter_mut()
                .zip(is_visitable.cells_mut().iter_mut()),
        ) {
            *height_cell = HeightCell(*height_cell_char as u8);
            *is_visitable = IsVisitable(*is_visitable_i32 as u8);
        }

        HeightGrid {
            heights,
            is_visitable,
            start: START,
            end: END,
        }
    }
}
