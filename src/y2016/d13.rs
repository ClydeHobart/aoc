use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
    num::Integer,
    std::{
        cell::{RefCell, RefMut},
        collections::{HashMap, VecDeque},
    },
    strum::IntoEnumIterator,
};

#[cfg(test)]
use std::cell::Ref;

/* --- Day 13: A Maze of Twisty Little Cubicles ---

You arrive at the first floor of this new building to discover a much less welcoming environment than the shiny atrium of the last one. Instead, you are in a maze of twisty little cubicles, all alike.

Every location in this area is addressed by a pair of non-negative integers (x,y). Each such coordinate is either a wall or an open space. You can't move diagonally. The cube maze starts at 0,0 and seems to extend infinitely toward positive x and y; negative values are invalid, as they represent a location outside the building. You are in a small waiting area at 1,1.

While it seems chaotic, a nearby morale-boosting poster explains, the layout is actually quite logical. You can determine whether a given x,y coordinate will be a wall or an open space using a simple system:

    Find x*x + 3*x + 2*x*y + y + y*y.
    Add the office designer's favorite number (your puzzle input).
    Find the binary representation of that sum; count the number of bits that are 1.
        If the number of bits that are 1 is even, it's an open space.
        If the number of bits that are 1 is odd, it's a wall.

For example, if the office designer's favorite number were 10, drawing walls as # and open spaces as ., the corner of the building containing 0,0 would look like this:

  0123456789
0 .#.####.##
1 ..#..#...#
2 #....##...
3 ###.#.###.
4 .##..#..#.
5 ..##....#.
6 #...##.###

Now, suppose you wanted to reach 7,4. The shortest route you could take is marked as O:

  0123456789
0 .#.####.##
1 .O#..#...#
2 #OOO.##...
3 ###O#.###.
4 .##OO#OO#.
5 ..##OOO.#.
6 #...##.###

Thus, reaching 7,4 would take a minimum of 11 steps (starting from your current location, 1,1).

What is the fewest number of steps required for you to reach 31,39?

--- Part Two ---

How many locations (distinct x,y coordinates, including your starting location) can you reach in at most 50 steps? */

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Cell {
        OpenSpace = OPEN_SPACE = b'.',
        Wall = WALL = b'#',
        Path = PATH = b'O',
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::OpenSpace
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct GridState {
    grid: Grid2D<Cell>,
    resizing_scratch: Vec<u32>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct State {
    grid_state: RefCell<GridState>,
    favorite_number: u32,
}

impl State {
    fn double_grid(&self) {
        let mut grid_state: RefMut<GridState> = self.grid_state.borrow_mut();

        let GridState {
            grid,
            resizing_scratch,
        } = &mut (*grid_state);

        let old_side_len: usize = grid.dimensions().x as usize;
        let new_side_len: usize = old_side_len * 2_usize;

        resizing_scratch.resize(new_side_len * 2_usize, 0_u32);

        let (old_resizing_scratch, new_resizing_scratch): (&mut [u32], &mut [u32]) =
            resizing_scratch.split_at_mut(new_side_len);

        // Copy over the y-values that are still valid.
        new_resizing_scratch[..old_side_len].copy_from_slice(&old_resizing_scratch[old_side_len..]);

        let x_resizing_scratch = old_resizing_scratch;
        let y_resizing_scratch = new_resizing_scratch;

        for (x_index, x_component) in x_resizing_scratch.iter_mut().enumerate().skip(old_side_len) {
            let x: u32 = x_index as u32;

            *x_component = (x + 3_u32) * x;
        }

        for (y_index, y_component) in y_resizing_scratch.iter_mut().enumerate().skip(old_side_len) {
            let y: u32 = y_index as u32;

            *y_component = (y + 1_u32) * y + self.favorite_number;
        }

        grid.double_dimensions(Cell::OpenSpace);

        for y_index in 0_usize..new_side_len {
            let y_component: u32 = y_resizing_scratch[y_index];
            let y: u32 = y_index as u32;
            let y_cell_index: usize = y_index * new_side_len;

            for (x_index, cell) in grid.cells_mut()[y_cell_index..y_cell_index + new_side_len]
                .iter_mut()
                .enumerate()
                .skip(if y_index < old_side_len {
                    old_side_len
                } else {
                    0_usize
                })
            {
                let x_component: u32 = x_resizing_scratch[x_index];

                *cell = if (x_component + y_component + (2_u32 * x_index as u32 * y))
                    .count_ones()
                    .is_odd()
                {
                    Cell::Wall
                } else {
                    Cell::OpenSpace
                };
            }
        }
    }

    fn is_open_space(&self, pos: IVec2) -> bool {
        while !self.grid_state.borrow().grid.contains(pos) {
            self.double_grid();
        }

        *self.grid_state.borrow().grid.get(pos).unwrap() == Cell::OpenSpace
    }

    fn new(favorite_number: u32) -> Self {
        // (x + 3) * x == (0 + 3) * 0 == 0
        // (y + 1) * y + favorite_number == (0 + 1) * 0 + favorite_number == 0 + favorite_number
        let resizing_scratch: Vec<u32> = vec![0_u32, favorite_number];
        let grid: Grid2D<Cell> = Grid2D::try_from_cells_and_dimensions(
            vec![if favorite_number.count_ones().is_odd() {
                Cell::Wall
            } else {
                Cell::OpenSpace
            }],
            IVec2::ONE,
        )
        .unwrap();

        Self {
            grid_state: RefCell::new(GridState {
                grid,
                resizing_scratch,
            }),
            favorite_number,
        }
    }

    fn iter_neighbors(&self, pos: IVec2) -> impl Iterator<Item = IVec2> + '_ {
        Direction::iter().filter_map(move |dir| {
            let neighbor: IVec2 = pos + dir.vec();

            (neighbor.cmpge(IVec2::ZERO).all() && self.is_open_space(neighbor)).then_some(neighbor)
        })
    }
}

struct PrevState {
    prev: Option<Direction>,
    cost: i32,
}

struct PathFinder<'s> {
    state: &'s State,
    start: IVec2,
    end: IVec2,
    prev_map: HashMap<IVec2, PrevState>,
}

impl<'s> AStar for PathFinder<'s> {
    type Vertex = IVec2;
    type Cost = i32;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.end
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<IVec2> = VecDeque::new();
        let mut pos: IVec2 = *vertex;

        while pos != self.start {
            path.push_front(pos);
            pos = pos + self.prev_map.get(&pos).unwrap().prev.unwrap().vec();
        }

        path.push_front(pos);

        path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.prev_map
            .get(vertex)
            .map_or(i32::MAX, |prev_state| prev_state.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        manhattan_distance_2d(*vertex, self.end)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();
        neighbors.extend(
            self.state
                .iter_neighbors(*vertex)
                .map(|neighbor| OpenSetElement(neighbor, 1_i32)),
        );
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.prev_map.insert(
            *to,
            PrevState {
                prev: Some((*from - *to).try_into().unwrap()),
                cost,
            },
        );
    }

    fn reset(&mut self) {
        self.prev_map.clear();
        self.prev_map.insert(
            self.start,
            PrevState {
                prev: None,
                cost: 0_i32,
            },
        );
    }
}

struct ReachableFinder<'s> {
    state: &'s State,
    start: IVec2,
    max_dist: i32,
    dist_map: HashMap<IVec2, i32>,
}

impl<'s> BreadthFirstSearch for ReachableFinder<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!();
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        if *self.dist_map.get(vertex).unwrap() < self.max_dist {
            neighbors.extend(self.state.iter_neighbors(*vertex));
        }
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let dist: i32 = *self.dist_map.get(from).unwrap() + 1_i32;

        self.dist_map.insert(*to, dist);
    }

    fn reset(&mut self) {
        self.dist_map.clear();
        self.dist_map.insert(self.start, 0_i32);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(State);

impl Solution {
    const START: IVec2 = IVec2::ONE;
    const END: IVec2 = IVec2::new(31_i32, 39_i32);
    const MAX_DIST: i32 = 50_i32;

    fn try_path_for_points(&self, start: IVec2, end: IVec2) -> Option<Vec<IVec2>> {
        let mut path_finder: PathFinder = PathFinder {
            state: &self.0,
            start,
            end,
            prev_map: HashMap::new(),
        };

        path_finder.run()
    }

    fn try_path(&self) -> Option<Vec<IVec2>> {
        self.try_path_for_points(Self::START, Self::END)
    }

    fn try_steps(&self) -> Option<usize> {
        self.try_path().map(|path| path.len() - 1_usize)
    }

    fn impose_path_on_grid(path: &[IVec2], grid: &mut Grid2D<Cell>) {
        for pos in path.iter().copied() {
            if let Some(cell) = grid.get_mut(pos) {
                *cell = Cell::Path;
            }
        }
    }

    fn reachable_locations(&self) -> Vec<IVec2> {
        let mut reachable_finder: ReachableFinder = ReachableFinder {
            state: &self.0,
            start: Self::START,
            max_dist: Self::MAX_DIST,
            dist_map: HashMap::new(),
        };

        reachable_finder.run();

        reachable_finder.dist_map.into_keys().collect()
    }

    fn reachable_locations_len(&self) -> usize {
        self.reachable_locations().len()
    }

    #[cfg(test)]
    fn grid_for_dimensions(&self, dimensions: IVec2) -> Grid2D<Cell> {
        let mut dst_grid: Grid2D<Cell> = Grid2D::default(dimensions);

        self.0.is_open_space(dimensions - IVec2::ONE);

        let grid_state: Ref<GridState> = self.0.grid_state.borrow();
        let src_grid: &Grid2D<Cell> = &grid_state.grid;
        let row_len: usize = dimensions.x as usize;

        for (dst_row, src_row) in dst_grid.cells_mut().chunks_exact_mut(row_len).zip(
            src_grid
                .cells()
                .chunks_exact(src_grid.dimensions().x as usize),
        ) {
            dst_row.copy_from_slice(&src_row[..row_len]);
        }

        dst_grid
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, |favorite_number| {
            Self(State::new(favorite_number))
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(path) = self.try_path() {
                dbg!(path.len() - 1_usize);

                let mut grid: Grid2D<Cell> = self.0.grid_state.borrow().grid.clone();

                Self::impose_path_on_grid(&path, &mut grid);

                println!("{}", String::from(grid));
            } else {
                eprintln!(
                    "Failed to compute path from {} to {}!",
                    Self::START,
                    Self::END
                );
            }
        } else {
            dbg!(self.try_steps());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let reachable_locations: Vec<IVec2> = self.reachable_locations();

            dbg!(reachable_locations.len());

            let mut grid: Grid2D<Cell> = self.0.grid_state.borrow().grid.clone();

            Self::impose_path_on_grid(&reachable_locations, &mut grid);

            println!("{}", String::from(grid));
        } else {
            dbg!(self.reachable_locations_len());
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
    use super::*;

    const SOLUTION_STR: &'static str = "10";

    fn solution() -> Solution {
        Solution(State {
            grid_state: RefCell::new(GridState {
                grid: Grid2D::try_from_cells_and_dimensions(vec![Cell::OpenSpace], IVec2::ONE)
                    .unwrap(),
                resizing_scratch: vec![0_u32, 10_u32],
            }),
            favorite_number: 10_u32,
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR), Ok(solution()));
    }

    #[test]
    fn test_try_path() {
        let solution: Solution = solution();

        let path: Option<Vec<IVec2>> =
            solution.try_path_for_points(IVec2::ONE, IVec2::new(7_i32, 4_i32));

        assert!(path.is_some());

        let path: Vec<IVec2> = path.unwrap();

        for (reality, expectation) in path.iter().zip([
            Some((1, 1)),
            Some((1, 2)),
            Some((2, 2)),
            Some((3, 2)),
            Some((3, 3)),
            Some((3, 4)),
            Some((4, 4)),
            Some((4, 5)),
            Some((5, 5)),
            Some((6, 5)),
            None,
            Some((7, 4)),
        ]) {
            if let Some((x, y)) = expectation {
                assert_eq!(reality, &IVec2::new(x, y));
            }
        }

        let mut grid: Grid2D<Cell> = solution.grid_for_dimensions(IVec2::new(10_i32, 7_i32));

        const GRID_STR: &'static str = "\
            .#.####.##\n\
            ..#..#...#\n\
            #....##...\n\
            ###.#.###.\n\
            .##..#..#.\n\
            ..##....#.\n\
            #...##.###\n";

        assert_eq!(String::from(grid.clone()), GRID_STR);

        const GRID_WITH_PATH_STR: &'static str = "\
            .#.####.##\n\
            .O#..#...#\n\
            #OOO.##...\n\
            ###O#.###.\n\
            .##OO#.O#.\n\
            ..##OOOO#.\n\
            #...##.###\n";

        Solution::impose_path_on_grid(&path, &mut grid);

        assert_eq!(String::from(grid), GRID_WITH_PATH_STR);
    }
}
