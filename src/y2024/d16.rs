use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::collections::{HashMap, HashSet},
};

/* --- Day 16: Reindeer Maze ---

It's time again for the Reindeer Olympics! This year, the big event is the Reindeer Maze, where the Reindeer compete for the lowest score.

You and The Historians arrive to search for the Chief right as the event is about to start. It wouldn't hurt to watch a little, right?

The Reindeer start on the Start Tile (marked S) facing East and need to reach the End Tile (marked E). They can move forward one tile at a time (increasing their score by 1 point), but never into a wall (#). They can also rotate clockwise or counterclockwise 90 degrees at a time (increasing their score by 1000 points).

To figure out the best place to sit, you start by grabbing a map (your puzzle input) from a nearby kiosk. For example:

###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############

There are many paths through this maze, but taking any of the best paths would incur a score of only 7036. This can be achieved by taking a total of 36 steps forward and turning 90 degrees a total of 7 times:


###############
#.......#....E#
#.#.###.#.###^#
#.....#.#...#^#
#.###.#####.#^#
#.#.#.......#^#
#.#.#####.###^#
#..>>>>>>>>v#^#
###^#.#####v#^#
#>>^#.....#v#^#
#^#.#.###.#v#^#
#^....#...#v#^#
#^###.#.#.#v#^#
#S..#.....#>>^#
###############

Here's a second example:

#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################

In this maze, the best paths cost 11048 points; following one such path would look like this:

#################
#...#...#...#..E#
#.#.#.#.#.#.#.#^#
#.#.#.#...#...#^#
#.#.#.#.###.#.#^#
#>>v#.#.#.....#^#
#^#v#.#.#.#####^#
#^#v..#.#.#>>>>^#
#^#v#####.#^###.#
#^#v#..>>>>^#...#
#^#v###^#####.###
#^#v#>>^#.....#.#
#^#v#^#####.###.#
#^#v#^........#.#
#^#v#^#########.#
#S#>>^..........#
#################

Note that the path shown above includes one 90 degree turn as the very first move, rotating the Reindeer from facing East to facing North.

Analyze your map carefully. What is the lowest score a Reindeer could possibly get?

--- Part Two ---

Now that you know what the best paths look like, you can figure out the best spot to sit.

Every non-wall tile (S, ., or E) is equipped with places to sit along the edges of the tile. While determining which of these tiles would be the best spot to sit depends on a whole bunch of factors (how comfortable the seats are, how far away the bathrooms are, whether there's a pillar blocking your view, etc.), the most important factor is whether the tile is on one of the best paths through the maze. If you sit somewhere else, you'd miss all the action!

So, you'll need to determine which tiles are part of any best path through the maze, including the S and E tiles.

In the first example, there are 45 tiles (marked O) that are part of at least one of the various best paths through the maze:

###############
#.......#....O#
#.#.###.#.###O#
#.....#.#...#O#
#.###.#####.#O#
#.#.#.......#O#
#.#.#####.###O#
#..OOOOOOOOO#O#
###O#O#####O#O#
#OOO#O....#O#O#
#O#O#O###.#O#O#
#OOOOO#...#O#O#
#O###.#.#.#O#O#
#O..#.....#OOO#
###############

In the second example, there are 64 tiles that are part of at least one of the best paths:

#################
#...#...#...#..O#
#.#.#.#.#.#.#.#O#
#.#.#.#...#...#O#
#.#.#.#.###.#.#O#
#OOO#.#.#.....#O#
#O#O#.#.#.#####O#
#O#O..#.#.#OOOOO#
#O#O#####.#O###O#
#O#O#..OOOOO#OOO#
#O#O###O#####O###
#O#O#OOO#..OOO#.#
#O#O#O#####O###.#
#O#O#OOOOOOO..#.#
#O#O#O#########.#
#O#OOO..........#
#################

Analyze your map further. How many tiles are part of at least one of the best paths through the maze? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, Default, PartialEq)]
    enum Cell {
        #[default]
        Empty = EMPTY = b'.',
        Wall = WALL = b'#',
        Start = START = b'S',
        End = END = b'E',
        North = NORTH = b'^',
        East = EAST = b'>',
        South = SOUTH = b'v',
        West = WEST = b'<',
        BestSittingSpot = BEST_SITTING_SPOT = b'O',
    }
}

impl Cell {
    fn try_as_direction(self) -> Option<Direction> {
        match self {
            Self::North => Some(Direction::North),
            Self::East => Some(Direction::East),
            Self::South => Some(Direction::South),
            Self::West => Some(Direction::West),
            _ => None,
        }
    }

    fn is_direction(self) -> bool {
        self.try_as_direction().is_some()
    }
}

impl From<Direction> for Cell {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => Self::North,
            Direction::East => Self::East,
            Direction::South => Self::South,
            Direction::West => Self::West,
        }
    }
}

impl TryFrom<Cell> for Direction {
    type Error = ();

    fn try_from(value: Cell) -> Result<Self, Self::Error> {
        value.try_as_direction().ok_or(())
    }
}

struct ParentData {
    parent: SmallPosAndDir,
    cost: u32,
}

struct ReindeerPathFinder<'s> {
    solution: &'s Solution,
    child_to_parent_data: HashMap<SmallPosAndDir, ParentData>,
    end: Option<SmallPosAndDir>,
}

impl<'s> ReindeerPathFinder<'s> {
    fn new(solution: &'s Solution) -> Self {
        Self {
            solution,
            child_to_parent_data: HashMap::new(),
            end: None,
        }
    }
}

impl<'s> WeightedGraphSearch for ReindeerPathFinder<'s> {
    type Vertex = SmallPosAndDir;

    type Cost = u32;

    fn start(&self) -> &Self::Vertex {
        &self.solution.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.child_to_parent_data
            .get(vertex)
            .map_or(u32::MAX, |parent_data| parent_data.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        zero_heuristic::<Self>(self, vertex)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let forward: IVec2 = vertex.pos.get() + vertex.dir.vec();

        neighbors.extend(
            self.solution
                .grid
                .get(forward)
                .filter(|&&cell| cell != Cell::Wall)
                .map(|_| {
                    OpenSetElement(
                        // SAFETY: `forward` is valid.
                        unsafe { SmallPosAndDir::from_pos_and_dir_unsafe(forward, vertex.dir) },
                        1_u32,
                    )
                })
                .into_iter()
                .chain([
                    OpenSetElement(
                        SmallPosAndDir {
                            dir: vertex.dir.prev(),
                            ..*vertex
                        },
                        Solution::TURN_COST,
                    ),
                    OpenSetElement(
                        SmallPosAndDir {
                            dir: vertex.dir.next(),
                            ..*vertex
                        },
                        Solution::TURN_COST,
                    ),
                ]),
        );
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.child_to_parent_data.insert(
            *to,
            ParentData {
                parent: *from,
                cost,
            },
        );

        if to.pos == self.solution.end
            && self
                .end
                .map_or(true, |end| self.child_to_parent_data[&end].cost > cost)
        {
            self.end = Some(*to);
        }
    }

    fn reset(&mut self) {
        self.child_to_parent_data.clear();
        self.child_to_parent_data.insert(
            self.solution.start,
            ParentData {
                parent: self.solution.start,
                cost: 0_u32,
            },
        );
        self.end = None;
    }
}

struct BestSittingSpotFinder<'r, 's: 'r> {
    reindeer_path_finder: &'r ReindeerPathFinder<'s>,
    best_sitting_spots: HashSet<SmallPosAndDir>,
}

impl<'r, 's: 'r> BestSittingSpotFinder<'r, 's> {
    fn new(reindeer_path_finder: &'r ReindeerPathFinder<'s>) -> Self {
        Self {
            reindeer_path_finder,
            best_sitting_spots: HashSet::new(),
        }
    }

    fn best_sitting_spots(reindeer_path_finder: &'r ReindeerPathFinder<'s>) -> HashSet<SmallPos> {
        let mut best_sitting_spot_finder: Self = Self::new(reindeer_path_finder);

        best_sitting_spot_finder.run();

        best_sitting_spot_finder
            .best_sitting_spots
            .into_iter()
            .map(|small_pos_and_dir| small_pos_and_dir.pos)
            .collect()
    }
}

impl<'r, 's: 'r> BreadthFirstSearch for BestSittingSpotFinder<'r, 's> {
    type Vertex = SmallPosAndDir;

    fn start(&self) -> &Self::Vertex {
        self.reindeer_path_finder.end.as_ref().unwrap()
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        let backward: IVec2 = vertex.pos.get() + vertex.dir.rev().vec();
        let start_to_vertex_cost: u32 = self.reindeer_path_finder.child_to_parent_data[vertex].cost;

        neighbors.extend(
            self.reindeer_path_finder
                .solution
                .grid
                .get(backward)
                .filter(|&&cell| cell != Cell::Wall)
                .map(|_| {
                    OpenSetElement(
                        // SAFETY: `backward` is valid.
                        unsafe { SmallPosAndDir::from_pos_and_dir_unsafe(backward, vertex.dir) },
                        1_u32,
                    )
                })
                .into_iter()
                .chain([
                    OpenSetElement(
                        SmallPosAndDir {
                            dir: vertex.dir.prev(),
                            ..*vertex
                        },
                        Solution::TURN_COST,
                    ),
                    OpenSetElement(
                        SmallPosAndDir {
                            dir: vertex.dir.next(),
                            ..*vertex
                        },
                        Solution::TURN_COST,
                    ),
                ])
                .filter_map(|OpenSetElement(neighbor, vertex_to_neighbor_cost)| {
                    (self.reindeer_path_finder.child_to_parent_data[&neighbor].cost
                        + vertex_to_neighbor_cost
                        == start_to_vertex_cost)
                        .then_some(neighbor)
                }),
        );
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        self.best_sitting_spots.insert(*to);
    }

    fn reset(&mut self) {
        self.best_sitting_spots.clear();
        self.best_sitting_spots
            .insert(self.reindeer_path_finder.end.unwrap());
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Cell>,
    start: SmallPosAndDir,
    end: SmallPos,
}

impl Solution {
    const START_DIR: Direction = Direction::East;
    const TURN_COST: u32 = 1000_u32;

    fn run_reindeer_path_finder(&self) -> ReindeerPathFinder {
        let mut reindeer_path_finder: ReindeerPathFinder = ReindeerPathFinder::new(self);

        reindeer_path_finder.run_dijkstra();

        reindeer_path_finder
    }

    fn try_min_score_to_end(&self) -> Option<u32> {
        let reindeer_path_finder: ReindeerPathFinder = self.run_reindeer_path_finder();

        reindeer_path_finder
            .end
            .map(|end| reindeer_path_finder.child_to_parent_data[&end].cost)
    }

    fn try_min_score_to_end_and_string(&self) -> Option<(u32, String)> {
        let reindeer_path_finder: ReindeerPathFinder = self.run_reindeer_path_finder();

        reindeer_path_finder.end.map(|end| {
            let score: u32 = reindeer_path_finder.child_to_parent_data[&end].cost;

            let mut grid: Grid2D<Cell> = self.grid.clone();
            let mut curr_vertex: SmallPosAndDir = end;

            while curr_vertex != self.start {
                *grid.get_mut(curr_vertex.pos.get()).unwrap() = curr_vertex.dir.into();
                curr_vertex = reindeer_path_finder.child_to_parent_data[&curr_vertex].parent;
            }

            *grid.get_mut(self.start.pos.get()).unwrap() = Cell::Start;
            *grid.get_mut(self.end.get()).unwrap() = Cell::End;

            (score, grid.into())
        })
    }

    fn try_best_sitting_spots(&self) -> Option<HashSet<SmallPos>> {
        let reindeer_path_finder: ReindeerPathFinder = self.run_reindeer_path_finder();

        reindeer_path_finder
            .end
            .is_some()
            .then(|| BestSittingSpotFinder::best_sitting_spots(&reindeer_path_finder))
    }

    fn try_best_sitting_spot_count(&self) -> Option<usize> {
        self.try_best_sitting_spots()
            .map(|best_sitting_spots| best_sitting_spots.len())
    }

    fn try_best_sitting_spot_count_and_string(&self) -> Option<(usize, String)> {
        self.try_best_sitting_spots().map(|best_sitting_spots| {
            let count: usize = best_sitting_spots.len();

            let mut grid: Grid2D<Cell> = self.grid.clone();

            for best_sitting_spot in best_sitting_spots {
                *grid.get_mut(best_sitting_spot.get()).unwrap() = Cell::BestSittingSpot;
            }

            (count, grid.into())
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |mut grid| {
            (SmallPos::are_dimensions_valid(grid.dimensions())
                && grid
                    .iter_filtered_positions(|cell: &Cell| {
                        cell.is_direction() || *cell == Cell::BestSittingSpot
                    })
                    .count()
                    == 0_usize)
                .then(|| {
                    grid.try_find_single_position_with_cell(&Cell::Start)
                        .zip(grid.try_find_single_position_with_cell(&Cell::End))
                })
                .flatten()
                .map(|(start, end)| {
                    *grid.get_mut(start).unwrap() = Cell::Empty;
                    *grid.get_mut(end).unwrap() = Cell::Empty;

                    // SAFETY: `start` and `end` are valid.
                    let (start, end): (SmallPos, SmallPos) = unsafe {
                        (
                            SmallPos::from_pos_unsafe(start),
                            SmallPos::from_pos_unsafe(end),
                        )
                    };

                    Self {
                        grid,
                        start: SmallPosAndDir {
                            pos: start,
                            dir: Self::START_DIR,
                        },
                        end,
                    }
                })
        })(input)
    }
}

impl RunQuestions for Solution {
    /// Not bad.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.try_min_score_to_end());
        } else if let Some((min_score_to_end, grid_string)) = self.try_min_score_to_end_and_string()
        {
            dbg!(min_score_to_end);
            println!("{grid_string}");
        } else {
            eprintln!("Failed to find path to end.");
        }
    }

    /// Had to switch from A* to Dijkstra, then it took a bit to find a bug where I was updating my
    /// recorded end when I shouldn't've.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.try_best_sitting_spot_count());
        } else if let Some((best_sitting_spot_count, grid_string)) =
            self.try_best_sitting_spot_count_and_string()
        {
            dbg!(best_sitting_spot_count);
            println!("{grid_string}");
        } else {
            eprintln!("Failed to find path to end.");
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
        "\
        ###############\n\
        #.......#....E#\n\
        #.#.###.#.###.#\n\
        #.....#.#...#.#\n\
        #.###.#####.#.#\n\
        #.#.#.......#.#\n\
        #.#.#####.###.#\n\
        #...........#.#\n\
        ###.#.#####.#.#\n\
        #...#.....#.#.#\n\
        #.#.#.###.#.#.#\n\
        #.....#...#.#.#\n\
        #.###.#.#.#.#.#\n\
        #S..#.....#...#\n\
        ###############\n",
        "\
        #################\n\
        #...#...#...#..E#\n\
        #.#.#.#.#.#.#.#.#\n\
        #.#.#.#...#...#.#\n\
        #.#.#.#.###.#.#.#\n\
        #...#.#.#.....#.#\n\
        #.#.#.#.#.#####.#\n\
        #.#...#.#.#.....#\n\
        #.#.#####.#.###.#\n\
        #.#.#.......#...#\n\
        #.#.###.#####.###\n\
        #.#.#...#.....#.#\n\
        #.#.#.#####.###.#\n\
        #.#.#.........#.#\n\
        #.#.#.#########.#\n\
        #S#.............#\n\
        #################\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{Empty as E, Wall as W};

            vec![
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, E, E, E, E, E, E, E, W,
                            E, E, E, E, E, W, W, E, W, E, W, W, W, E, W, E, W, W, W, E, W, W, E, E,
                            E, E, E, W, E, W, E, E, E, W, E, W, W, E, W, W, W, E, W, W, W, W, W, E,
                            W, E, W, W, E, W, E, W, E, E, E, E, E, E, E, W, E, W, W, E, W, E, W, W,
                            W, W, W, E, W, W, W, E, W, W, E, E, E, E, E, E, E, E, E, E, E, W, E, W,
                            W, W, W, E, W, E, W, W, W, W, W, E, W, E, W, W, E, E, E, W, E, E, E, E,
                            E, W, E, W, E, W, W, E, W, E, W, E, W, W, W, E, W, E, W, E, W, W, E, E,
                            E, E, E, W, E, E, E, W, E, W, E, W, W, E, W, W, W, E, W, E, W, E, W, E,
                            W, E, W, W, E, E, E, W, E, E, E, E, E, W, E, E, E, W, W, W, W, W, W, W,
                            W, W, W, W, W, W, W, W, W,
                        ],
                        15_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    start: SmallPosAndDir {
                        pos: SmallPos { x: 1_u8, y: 13_u8 },
                        dir: Direction::East,
                    },
                    end: SmallPos { x: 13_u8, y: 1_u8 },
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, E, E, E, W, E, E,
                            E, W, E, E, E, W, E, E, E, W, W, E, W, E, W, E, W, E, W, E, W, E, W, E,
                            W, E, W, W, E, W, E, W, E, W, E, E, E, W, E, E, E, W, E, W, W, E, W, E,
                            W, E, W, E, W, W, W, E, W, E, W, E, W, W, E, E, E, W, E, W, E, W, E, E,
                            E, E, E, W, E, W, W, E, W, E, W, E, W, E, W, E, W, W, W, W, W, E, W, W,
                            E, W, E, E, E, W, E, W, E, W, E, E, E, E, E, W, W, E, W, E, W, W, W, W,
                            W, E, W, E, W, W, W, E, W, W, E, W, E, W, E, E, E, E, E, E, E, W, E, E,
                            E, W, W, E, W, E, W, W, W, E, W, W, W, W, W, E, W, W, W, W, E, W, E, W,
                            E, E, E, W, E, E, E, E, E, W, E, W, W, E, W, E, W, E, W, W, W, W, W, E,
                            W, W, W, E, W, W, E, W, E, W, E, E, E, E, E, E, E, E, E, W, E, W, W, E,
                            W, E, W, E, W, W, W, W, W, W, W, W, W, E, W, W, E, W, E, E, E, E, E, E,
                            E, E, E, E, E, E, E, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W,
                            W,
                        ],
                        17_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    start: SmallPosAndDir {
                        pos: SmallPos { x: 1_u8, y: 15_u8 },
                        dir: Direction::East,
                    },
                    end: SmallPos { x: 15_u8, y: 1_u8 },
                },
            ]
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
    fn test_try_min_score_to_end() {
        for (index, min_score_to_end) in [Some(7036_u32), Some(11048_u32)].into_iter().enumerate() {
            assert_eq!(solution(index).try_min_score_to_end(), min_score_to_end);
        }
    }

    #[test]
    fn test_try_best_sitting_spot_count() {
        for (index, best_sitting_spot_count) in
            [Some(45_usize), Some(64_usize)].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_best_sitting_spot_count(),
                best_sitting_spot_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
