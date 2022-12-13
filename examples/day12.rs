use {
    aoc_2022::*,
    clap::Parser,
    glam::IVec2,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet},
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
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
    #[inline]
    fn mask(dir: Direction) -> u8 {
        1_u8 << dir as u32
    }

    fn get(self, dir: Direction) -> bool {
        (self.0 & Self::mask(dir)) != 0_u8
    }

    fn set(&mut self, dir: Direction, value: bool) {
        self.0 = (self.0 & !Self::mask(dir)) | ((value as u8) << dir as u32)
    }
}

impl Debug for IsVisitable {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("IsVisitable")?;

        let mut debug_list: DebugList = f.debug_list();

        for dir in Direction::iter() {
            debug_list.entry(&(dir, self.get(dir)));
        }

        debug_list.finish()
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
        _old_grid: &Grid<Self::Old>,
        rev_dir: Direction,
        _pos: IVec2,
    ) {
        new.set(rev_dir, old.0 >= self.0 - 1_u8);
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

trait AStar: Sized {
    type Vertex: Clone + Debug + Eq + Hash;
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

        while let Some(open_set_element) = open_set_heap.pop() {
            let current: Self::Vertex = open_set_element.0;

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

struct HeightGridAStar<'h> {
    height_grid: &'h HeightGrid,
    shortest_paths: Grid<ShortestPath>,
}

impl<'h> HeightGridAStar<'h> {
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

impl<'h> AStar for HeightGridAStar<'h> {
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
            if is_visitable.get(dir) {
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
                        let path: Vec<IVec2> = HeightGridAStar::new(&height_grid)
                            .run()
                            .expect("Failed to obtain a path from A*");

                        println!("path.len() == {}\npath == {path:#?}", path.len());
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
        let path: Vec<IVec2> = HeightGridAStar::new(&height_grid).run().unwrap();

        // 31 total steps, but 32 elements including the start
        assert_eq!(path.len(), 32_usize);
    }

    fn example_height_grid() -> HeightGrid {
        let mut heights: Grid<HeightCell> = Grid::default(DIMENSIONS);
        let mut is_visitable: Grid<IsVisitable> = Grid::default(DIMENSIONS);

        for ((height_cell_char, is_visitable_i32), (height_cell, is_visitable)) in vec![
            // Row 0
            ('a', 0x6),
            ('a', 0xE),
            ('b', 0xC),
            ('q', 0xE),
            ('p', 0xA),
            ('o', 0xA),
            ('n', 0xA),
            ('m', 0xC),
            // Row 1
            ('a', 0x7),
            ('b', 0xF),
            ('c', 0xD),
            ('r', 0xD),
            ('y', 0xF),
            ('x', 0xB),
            ('x', 0xF),
            ('l', 0x5),
            // Row 2
            ('a', 0x5),
            ('c', 0xF),
            ('c', 0xD),
            ('s', 0xD),
            ('z', 0xF),
            ('z', 0xF),
            ('x', 0x7),
            ('k', 0x5),
            // Row 3
            ('a', 0x5),
            ('c', 0xF),
            ('c', 0xD),
            ('t', 0xF),
            ('u', 0xE),
            ('v', 0xE),
            ('w', 0xF),
            ('j', 0x5),
            // Row 4
            ('a', 0x3),
            ('b', 0x9),
            ('d', 0xB),
            ('e', 0xA),
            ('f', 0xA),
            ('g', 0xA),
            ('h', 0xA),
            ('i', 0x9),
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
