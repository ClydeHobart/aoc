use {
    crate::*,
    glam::IVec2,
    std::fmt::{Debug, Formatter, Result as FmtResult},
    strum::IntoEnumIterator,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeightCell(u8);

impl HeightCell {
    const START: u8 = 'S' as u8;
    const END: u8 = 'E' as u8;
}

#[derive(Debug, PartialEq)]
pub struct InvalidHeightCellChar(char);

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
        old_grid: &Grid2D<Self::Old>,
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
    heights: Grid2D<HeightCell>,
    is_visitable: Grid2D<IsVisitable>,
    start: IVec2,
    end: IVec2,
}

#[derive(Debug, PartialEq)]
pub enum HeightGridParseError<'s> {
    FailedToParseGrid(GridParseError<'s, InvalidHeightCellChar>),
    GridContainsNoStartPosition,
    GridContainsNoEndPosition,
}

impl<'s> TryFrom<&'s str> for HeightGrid {
    type Error = HeightGridParseError<'s>;

    fn try_from(height_grid_str: &'s str) -> Result<Self, Self::Error> {
        use HeightGridParseError::*;

        let mut heights: Grid2D<HeightCell> =
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

        let is_visitable: Grid2D<IsVisitable> = IsVisitableGridVisitor::visit_grid(&heights);

        Ok(HeightGrid {
            heights,
            is_visitable,
            start,
            end,
        })
    }
}

struct HeightGridAStarAscent<'h> {
    height_grid: &'h HeightGrid,
    shortest_paths: Grid2D<ShortestPath>,
}

impl<'h> HeightGridAStarAscent<'h> {
    fn new(height_grid: &'h HeightGrid) -> Self {
        Self {
            height_grid,
            shortest_paths: Grid2D::default(height_grid.heights.dimensions()),
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

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        let abs: IVec2 = (self.height_grid.end - *vertex).abs();

        (abs.x + abs.y) as usize
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let is_visitable: IsVisitable = *self.height_grid.is_visitable.get(*vertex).unwrap();

        neighbors.extend(Direction::iter().filter_map(|dir| {
            if is_visitable.get(dir, true) {
                Some(OpenSetElement(*vertex + dir.vec(), 1_usize))
            } else {
                None
            }
        }));
    }

    fn update_vertex(
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

    fn reset(&mut self) {
        self.shortest_paths.cells_mut().fill_with(Default::default);
        self.shortest_paths
            .get_mut(self.height_grid.start)
            .unwrap()
            .score = 0_usize;
    }
}

struct HeightGridBreadthFirstSearchDescent<'h> {
    height_grid: &'h HeightGrid,
    predecessors: Grid2D<Option<Direction>>,
}

impl<'h> HeightGridBreadthFirstSearchDescent<'h> {
    fn new(height_grid: &'h HeightGrid) -> Self {
        Self {
            height_grid,
            predecessors: Grid2D::default(height_grid.heights.dimensions()),
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

    fn reset(&mut self) {
        self.predecessors.cells_mut().fill(None);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(HeightGrid);

impl Solution {
    fn a_star_ascent(&self) -> Vec<IVec2> {
        HeightGridAStarAscent::new(&self.0)
            .run()
            .unwrap_or_default()
    }

    fn bfs_descent(&self) -> Vec<IVec2> {
        HeightGridBreadthFirstSearchDescent::new(&self.0)
            .run()
            .unwrap_or_default()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        let a_star_ascent: Vec<IVec2> = self.a_star_ascent();

        dbg!(a_star_ascent.len());

        if args.verbose {
            dbg!(a_star_ascent);
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        let bfs_descent: Vec<IVec2> = self.bfs_descent();

        dbg!(bfs_descent.len());

        if args.verbose {
            dbg!(bfs_descent);
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = HeightGridParseError<'i>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

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

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            let mut heights: Grid2D<HeightCell> = Grid2D::default(DIMENSIONS);
            let mut is_visitable: Grid2D<IsVisitable> = Grid2D::default(DIMENSIONS);

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

            Solution(HeightGrid {
                heights,
                is_visitable,
                start: START,
                end: END,
            })
        })
    }

    #[test]
    fn test_solution_try_from_str() {
        compare_height_grids(
            &Solution::try_from(HEIGHT_GRID_STR).unwrap().0,
            &solution().0,
        )
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
    fn test_a_star_ascent() {
        assert_eq!(solution().a_star_ascent().len(), 32_usize);
    }

    #[test]
    fn test_bfs_descent() {
        assert_eq!(solution().bfs_descent().len(), 30_usize);
    }
}
