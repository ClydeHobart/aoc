use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::{
        collections::{HashMap, HashSet},
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
    },
    strum::{EnumCount, IntoEnumIterator},
};

define_cell! {
    #[repr(u8)]
    #[derive(Copy, Clone, PartialEq)]
    enum Cell {
        Vertical = VERTICAL = b'|',
        Horizontal = HORIZONTAL = b'-',
        NorthEast = NORTH_EAST = b'L',
        NorthWest = NORTH_WEST = b'J',
        SouthWest = SOUTH_WEST = b'7',
        SouthEast = SOUTH_EAST = b'F',
        Ground = GROUND = b'.',
        StartingPosition = STARTING_POS = b'S',
        Inside = INSIDE = b'I',
        Outside = OUTSIDE = b'O',
    }
}

impl Cell {
    fn is_starting_pos(&self) -> bool {
        *self == Self::StartingPosition
    }

    fn neighbor_dirs(&self) -> &'static [Direction] {
        const DIRECTIONS: [Direction; Direction::COUNT + 1_usize] = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
            Direction::North,
        ];

        match self {
            Cell::Vertical => &[Direction::North, Direction::South],
            Cell::Horizontal => &[Direction::East, Direction::West],
            Cell::NorthEast => &DIRECTIONS[0_usize..2_usize],
            Cell::NorthWest => &DIRECTIONS[3_usize..5_usize],
            Cell::SouthWest => &DIRECTIONS[2_usize..4_usize],
            Cell::SouthEast => &DIRECTIONS[1_usize..3_usize],
            _ => &[],
        }
    }
}

impl Debug for Cell
where
    Self: IsValidAscii,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // SAFETY: Guaranteed by `IsValidAscii`
        f.write_str(unsafe { from_utf8_unchecked(&[*self as u8]) })
    }
}

#[derive(Default)]
struct LoopData {
    cells: HashMap<IVec2, u32>,
    max_dist_pos: IVec2,
    max_dist: u32,
}

struct LoopFinder<'s> {
    solution: &'s Solution,
    loop_data: LoopData,
}

impl<'s> BreadthFirstSearch for LoopFinder<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.solution.starting_pos
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        neighbors.extend(
            self.solution
                .grid
                .get(*vertex)
                .unwrap()
                .neighbor_dirs()
                .iter()
                .map(|dir| *vertex + dir.vec()),
        )
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let dist: u32 = if from == self.start() {
            1_u32
        } else {
            self.loop_data.cells.get(from).unwrap() + 1_u32
        };

        self.loop_data.cells.insert(*to, dist);

        if self.loop_data.max_dist < dist {
            self.loop_data.max_dist = dist;
            self.loop_data.max_dist_pos = *to;
        }
    }

    fn reset(&mut self) {
        self.loop_data.cells.clear();
        self.loop_data.max_dist = 0_u32;
        self.loop_data.max_dist_pos = self.solution.starting_pos;
    }
}

struct OutsideFinder<'w, 'o> {
    wall_grid: &'w Grid2D<bool>,
    outside_poses: &'o mut HashSet<IVec2>,
    start: IVec2,
}

impl<'w, 'o> BreadthFirstSearch for OutsideFinder<'w, 'o> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        const NEIGHBOR_DELTAS: [IVec2; 8_usize] = [
            IVec2::X,
            IVec2::new(1_i32, -1_i32),
            IVec2::NEG_Y,
            IVec2::NEG_ONE,
            IVec2::NEG_X,
            IVec2::new(-1_i32, 1_i32),
            IVec2::Y,
            IVec2::ONE,
        ];

        neighbors.clear();

        neighbors.extend(NEIGHBOR_DELTAS.into_iter().filter_map(|neighbor_delta| {
            let pos: IVec2 = *vertex + neighbor_delta;

            if self
                .wall_grid
                .get(pos)
                .copied()
                .map_or(false, |is_wall| !is_wall)
            {
                Some(pos)
            } else {
                None
            }
        }));
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        if (to.x | to.y) % 2_i32 == 0_i32 {
            self.outside_poses.insert(*to / 2_i32);
        }
    }

    fn reset(&mut self) {}
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    grid: Grid2D<Cell>,
    starting_pos: IVec2,
}

impl Solution {
    fn correct_starting_pos(&mut self) -> bool {
        let mut is_neighbor_connected: [bool; 4_usize] = Default::default();

        for dir in Direction::iter().filter_map(|dir| {
            self.grid
                .get(self.starting_pos + dir.vec())
                .filter(|neighbor| neighbor.neighbor_dirs().contains(&dir.rev()))
                .map(|_| dir)
        }) {
            is_neighbor_connected[dir as usize] = true;
        }

        match is_neighbor_connected {
            // N,   E      S     W
            [true, false, true, false] => Some(Cell::Vertical),
            [false, true, false, true] => Some(Cell::Horizontal),
            [true, true, false, false] => Some(Cell::NorthEast),
            [true, false, false, true] => Some(Cell::NorthWest),
            [false, false, true, true] => Some(Cell::SouthWest),
            [false, true, true, false] => Some(Cell::SouthEast),
            _ => None,
        }
        .map_or(false, |starting_pos_cell| {
            *self.grid.get_mut(self.starting_pos).unwrap() = starting_pos_cell;

            true
        })
    }

    fn loop_data(&self) -> LoopData {
        let mut loop_finder: LoopFinder = LoopFinder {
            solution: self,
            loop_data: LoopData::default(),
        };

        loop_finder.run();

        loop_finder.loop_data
    }

    fn max_dist(&self) -> u32 {
        self.loop_data().max_dist
    }

    fn trim(&self, loop_data: Option<&LoopData>) -> Self {
        let mut local_loop_data: Option<LoopData> = None;
        let loop_data: &LoopData = loop_data.unwrap_or_else(|| {
            local_loop_data = Some(self.loop_data());

            local_loop_data.as_ref().unwrap()
        });

        let mut solution: Self = self.clone();

        for (index, cell) in solution.grid.cells_mut().iter_mut().enumerate() {
            let pos: IVec2 = self.grid.pos_from_index(index);

            if pos != self.starting_pos && loop_data.cells.get(&pos).is_none() {
                *cell = Cell::Ground;
            }
        }

        solution
    }

    fn wall_grid(&self) -> Grid2D<bool> {
        let mut wall_grid: Grid2D<bool> =
            Grid2D::default(2_i32 * self.grid.dimensions() - IVec2::ONE);

        for wall_pos in
            self.grid
                .cells()
                .iter()
                .copied()
                .enumerate()
                .flat_map(|(cell_index, cell)| {
                    let cell_pos: IVec2 = self.grid.pos_from_index(cell_index);
                    let wall_pos: IVec2 = 2_i32 * cell_pos;

                    if cell != Cell::Ground {
                        Some(wall_pos)
                    } else {
                        None
                    }
                    .into_iter()
                    .chain(
                        cell.neighbor_dirs()
                            .into_iter()
                            .map(move |dir| wall_pos + dir.vec()),
                    )
                })
        {
            *wall_grid.get_mut(wall_pos).unwrap() = true;
        }

        wall_grid
    }

    fn outside_poses(&self) -> HashSet<IVec2> {
        let wall_grid: Grid2D<bool> = self.wall_grid();
        let nw: IVec2 = IVec2::ZERO;
        let se: IVec2 = self.grid.max_dimensions();
        let ne: IVec2 = IVec2::new(se.x, nw.y);
        let sw: IVec2 = IVec2::new(nw.x, se.y);

        let mut outside_poses: HashSet<IVec2> = HashSet::default();

        for cell_pos in [ne, nw, sw, se, ne]
            .windows(2_usize)
            .flat_map(|corners| CellIter2D::try_from(corners[0_usize]..corners[1_usize]).unwrap())
            .filter(|cell_pos| *self.grid.get(*cell_pos).unwrap() == Cell::Ground)
        {
            if outside_poses.insert(cell_pos) {
                let mut outside_finder: OutsideFinder = OutsideFinder {
                    wall_grid: &wall_grid,
                    outside_poses: &mut outside_poses,
                    start: 2_i32 * cell_pos,
                };

                outside_finder.run();
            }
        }

        outside_poses
    }

    fn inside_pos_count(&self, outside_poses: Option<&HashSet<IVec2>>) -> usize {
        let mut local_solution: Option<Self> = None;
        let mut local_outside_poses: Option<HashSet<IVec2>> = None;
        let mut solution: &Self = self;
        let outside_poses: &HashSet<IVec2> = outside_poses.unwrap_or_else(|| {
            local_solution = Some(self.trim(None));
            solution = local_solution.as_ref().unwrap();
            local_outside_poses = Some(solution.outside_poses());

            local_outside_poses.as_ref().unwrap()
        });

        solution
            .grid
            .cells()
            .iter()
            .copied()
            .filter(|cell| *cell == Cell::Ground)
            .count()
            - outside_poses.len()
    }

    /// Replace outside ground with `O` and inside ground with `I`. This doesn't do any trimming.
    fn paint_inside_and_outside(&self, outside_poses: Option<&HashSet<IVec2>>) -> Self {
        let mut local_outside_poses: Option<HashSet<IVec2>> = None;
        let mut solution: Solution = self.clone();
        let outside_poses: &HashSet<IVec2> = outside_poses.unwrap_or_else(|| {
            local_outside_poses = Some(self.outside_poses());

            local_outside_poses.as_ref().unwrap()
        });

        for outside_pos in outside_poses.into_iter().cloned() {
            *solution.grid.get_mut(outside_pos).unwrap() = Cell::Outside;
        }

        for cell in solution
            .grid
            .cells_mut()
            .iter_mut()
            .filter(|cell| **cell == Cell::Ground)
        {
            *cell = Cell::Inside
        }

        solution
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::<Cell>::parse, |grid| {
            grid.cells()
                .iter()
                .position(Cell::is_starting_pos)
                .and_then(|starting_pos_index| {
                    if grid
                        .cells()
                        .iter()
                        .copied()
                        .filter(Cell::is_starting_pos)
                        .count()
                        == 1_usize
                    {
                        let starting_pos: IVec2 = grid.pos_from_index(starting_pos_index);

                        let mut solution: Self = Self { grid, starting_pos };

                        if solution.correct_starting_pos() {
                            Some(solution)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let loop_data: LoopData = self.loop_data();
            let grid_string: String = self.trim(Some(&loop_data)).grid.into();

            println!("max_dist == {}\n\n{grid_string}", loop_data.max_dist);
        } else {
            dbg!(self.max_dist());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let trimmed: Self = self.trim(None);
            let outside_poses: HashSet<IVec2> = trimmed.outside_poses();
            let grid_string: String = trimmed
                .paint_inside_and_outside(Some(&outside_poses))
                .grid
                .into();

            println!(
                "inside_pos_count == {}\n\n{grid_string}",
                trimmed.inside_pos_count(Some(&outside_poses))
            );
        } else {
            dbg!(self.inside_pos_count(None));
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        // 0_usize..2_usize: Original
        "\
        -L|F7\n\
        7S-7|\n\
        L|7||\n\
        -L-J|\n\
        L|-JF\n",
        "\
        7-F7-\n\
        .FJ|7\n\
        SJLL7\n\
        |F--J\n\
        LJ.LJ\n",
        // 2_usize..4_usize: Trimmed
        "\
        .....\n\
        .S-7.\n\
        .|.|.\n\
        .L-J.\n\
        .....\n",
        "\
        ..F7.\n\
        .FJ|.\n\
        SJ.L7\n\
        |F--J\n\
        LJ...\n",
        // 4_usize..8_usize: Inside-Outside Unpainted
        "\
        ...........\n\
        .S-------7.\n\
        .|F-----7|.\n\
        .||.....||.\n\
        .||.....||.\n\
        .|L-7.F-J|.\n\
        .|..|.|..|.\n\
        .L--J.L--J.\n\
        ...........\n",
        "\
        ..........\n\
        .S------7.\n\
        .|F----7|.\n\
        .||....||.\n\
        .||....||.\n\
        .|L-7F-J|.\n\
        .|..||..|.\n\
        .L--JL--J.\n\
        ..........\n",
        "\
        .F----7F7F7F7F-7....\n\
        .|F--7||||||||FJ....\n\
        .||.FJ||||||||L7....\n\
        FJL7L7LJLJ||LJ.L-7..\n\
        L--J.L7...LJS7F-7L7.\n\
        ....F-J..F7FJ|L7L7L7\n\
        ....L7.F7||L7|.L7L7|\n\
        .....|FJLJ|FJ|F7|.LJ\n\
        ....FJL-7.||.||||...\n\
        ....L---J.LJ.LJLJ...\n",
        "\
        FF7FSF7F7F7F7F7F---7\n\
        L|LJ||||||||||||F--J\n\
        FL-7LJLJ||||||LJL-77\n\
        F--JF--7||LJLJ7F7FJ-\n\
        L---JF-JLJ.||-FJLJJ7\n\
        |F|F-JF---7F7-L7L|7|\n\
        |FFJF7L7F-JF7|JL---7\n\
        7-L-JL7||F7|L7F-7F7|\n\
        L.L7LFJ|||||FJL7||LJ\n\
        L7JLJL-JLJLJL--JLJ.L\n",
        // 8_usize..11_usize: Inside-Outside Painted
        "\
        OOOOOOOOOOO\n\
        OS-------7O\n\
        O|F-----7|O\n\
        O||OOOOO||O\n\
        O||OOOOO||O\n\
        O|L-7OF-J|O\n\
        O|II|O|II|O\n\
        OL--JOL--JO\n\
        OOOOOOOOOOO\n",
        "\
        OOOOOOOOOO\n\
        OS------7O\n\
        O|F----7|O\n\
        O||OOOO||O\n\
        O||OOOO||O\n\
        O|L-7F-J|O\n\
        O|II||II|O\n\
        OL--JL--JO\n\
        OOOOOOOOOO\n",
        "\
        OF----7F7F7F7F-7OOOO\n\
        O|F--7||||||||FJOOOO\n\
        O||OFJ||||||||L7OOOO\n\
        FJL7L7LJLJ||LJIL-7OO\n\
        L--JOL7IIILJS7F-7L7O\n\
        OOOOF-JIIF7FJ|L7L7L7\n\
        OOOOL7IF7||L7|IL7L7|\n\
        OOOOO|FJLJ|FJ|F7|OLJ\n\
        OOOOFJL-7O||O||||OOO\n\
        OOOOL---JOLJOLJLJOOO\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        use Cell::{
            Ground as TG, Horizontal as TH, NorthEast as NE, NorthWest as NW, SouthEast as SE,
            SouthWest as SW, Vertical as TV,
        };

        macro_rules! solution {
            [ $( [ $( $cell:expr, )* ], )* ($x:expr, $y:expr), ] => {
                Solution {
                    grid: Grid2D::try_from_cells_and_width(
                        vec![ $( $( $cell, )* )* ],
                        5_usize
                    ).unwrap(),
                    starting_pos: IVec2::new($x, $y),
                }
            };
        }

        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                solution![
                    [TH, NE, TV, SE, SW,],
                    [SW, SE, TH, SW, TV,],
                    [NE, TV, SW, TV, TV,],
                    [TH, NE, TH, NW, TV,],
                    [NE, TV, TH, NW, SE,],
                    (1_i32, 1_i32),
                ],
                solution![
                    [SW, TH, SE, SW, TH,],
                    [TG, SE, NW, TV, SW,],
                    [SE, NW, NE, NE, SW,],
                    [TV, SE, TH, TH, NW,],
                    [NE, NW, TG, NE, NW,],
                    (0_i32, 2_i32),
                ],
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS[..2_usize].iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_loop_data() {
        let loop_data: LoopData = solution(0_usize).loop_data();

        for (real_dist, expected_dist) in [
            (IVec2::new(1_i32, 2_i32), 1_u32),
            (IVec2::new(2_i32, 1_i32), 1_u32),
            (IVec2::new(1_i32, 3_i32), 2_u32),
            (IVec2::new(3_i32, 1_i32), 2_u32),
            (IVec2::new(2_i32, 3_i32), 3_u32),
            (IVec2::new(3_i32, 2_i32), 3_u32),
            (IVec2::new(3_i32, 3_i32), 4_u32),
        ]
        .map(|(pos, expected_dist)| (*loop_data.cells.get(&pos).unwrap(), expected_dist))
        {
            assert_eq!(real_dist, expected_dist);
        }

        let loop_data: LoopData = solution(1_usize).loop_data();

        assert_eq!(loop_data.max_dist, 8_u32);
        assert_eq!(loop_data.max_dist_pos, IVec2::new(4_i32, 2_i32));
    }

    #[test]
    fn test_trim() {
        for index in 0_usize..2_usize {
            assert_eq!(
                solution(index).trim(None),
                Solution::try_from(SOLUTION_STRS[index + 2_usize]).unwrap()
            );
        }
    }

    #[test]
    fn test_inside_pos_count() {
        for (real_inside_pos_count, expected_inside_pos_count) in
            [4_usize, 4_usize, 8_usize, 10_usize]
                .into_iter()
                .enumerate()
                .map(|(index, expected_inside_pos_count)| {
                    (
                        Solution::try_from(SOLUTION_STRS[index + 4_usize])
                            .unwrap()
                            .inside_pos_count(None),
                        expected_inside_pos_count,
                    )
                })
        {
            assert_eq!(real_inside_pos_count, expected_inside_pos_count);
        }
    }

    #[test]
    fn test_paint_inside_and_outside() {
        for unpainted_index in 4_usize..7_usize {
            assert_eq!(
                Solution::try_from(SOLUTION_STRS[unpainted_index])
                    .unwrap()
                    .trim(None)
                    .paint_inside_and_outside(None),
                Solution::try_from(SOLUTION_STRS[unpainted_index + 4_usize]).unwrap()
            );
        }
    }
}
