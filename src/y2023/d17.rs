use {
    crate::*,
    glam::IVec2,
    nom::{character::complete::satisfy, combinator::map, error::Error, AsChar, Err, IResult},
    std::{
        collections::{HashMap, VecDeque},
        ops::Range,
    },
    strum::IntoEnumIterator,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct HeatLoss(u8);

impl Parse for HeatLoss {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(char::is_dec_digit), |c| Self(c as u8 - b'0'))(input)
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct Vertex {
    pos: IVec2,
    prev_corner: IVec2,
}

impl Vertex {
    fn current_dir(&self) -> Option<Direction> {
        Direction::try_from(self.prev_corner..self.pos).ok()
    }

    fn straight_line_len(&self) -> i32 {
        manhattan_distance_2d(self.prev_corner, self.pos)
    }
}

struct VertexData {
    parent: Vertex,
    cost: u32,
}

struct MinimalHeatLossPathFinder<'s> {
    solution: &'s Solution,
    vertex_to_vertex_data: HashMap<Vertex, VertexData>,
    start: Vertex,
    end_pos: IVec2,
    straight_line_len_range: Range<i32>,
}

impl<'s> AStar for MinimalHeatLossPathFinder<'s> {
    type Vertex = Vertex;
    type Cost = u32;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.pos == self.end_pos
            && vertex.straight_line_len() >= self.straight_line_len_range.start
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<Vertex> = VecDeque::new();
        let mut vertex: Vertex = vertex.clone();

        while vertex != self.start {
            path.push_front(vertex.clone());
            vertex = self
                .vertex_to_vertex_data
                .get(&vertex)
                .unwrap()
                .parent
                .clone();
        }

        path.push_front(vertex);

        path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.vertex_to_vertex_data
            .get(vertex)
            .map_or(u32::MAX, |vertex_data| vertex_data.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        manhattan_distance_2d(vertex.pos, self.end_pos) as u32
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let current_dir: Option<Direction> = vertex.current_dir();
        let straight_line_len: i32 = vertex.straight_line_len();

        neighbors.extend(
            Direction::iter()
                .filter_map(|dir| {
                    let pos: IVec2 = vertex.pos + dir.vec();

                    self.solution.0.get(pos).and_then(|_| {
                        if let Some(current_dir) = current_dir.as_ref() {
                            if dir == *current_dir {
                                if straight_line_len < self.straight_line_len_range.end {
                                    Some(Vertex {
                                        pos,
                                        prev_corner: vertex.prev_corner,
                                    })
                                } else {
                                    None
                                }
                            } else if dir == current_dir.rev() {
                                None
                            } else {
                                if straight_line_len < self.straight_line_len_range.start {
                                    None
                                } else {
                                    Some(Vertex {
                                        pos,
                                        prev_corner: vertex.pos,
                                    })
                                }
                            }
                        } else {
                            Some(Vertex {
                                pos,
                                prev_corner: vertex.pos,
                            })
                        }
                    })
                })
                .map(|neighbor| {
                    let cost: u32 = self.solution.0.get(neighbor.pos).unwrap().0 as u32;

                    OpenSetElement(neighbor, cost)
                }),
        );
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.vertex_to_vertex_data.insert(
            to.clone(),
            VertexData {
                parent: from.clone(),
                cost,
            },
        );
    }

    fn reset(&mut self) {
        self.vertex_to_vertex_data.clear();
        self.vertex_to_vertex_data.insert(
            self.start.clone(),
            VertexData {
                parent: self.start.clone(),
                cost: 0_u32,
            },
        );
    }
}

struct PathGridCell(u8);

impl Default for PathGridCell {
    fn default() -> Self {
        Self::try_from(0_u8).unwrap()
    }
}

impl From<Direction> for PathGridCell {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => Self(b'^'),
            Direction::East => Self(b'>'),
            Direction::South => Self(b'v'),
            Direction::West => Self(b'<'),
        }
    }
}

// SAFETY: `PathGridCell` can only be constructed from valid ASCII bytes.
unsafe impl IsValidAscii for PathGridCell {}

impl TryFrom<u8> for PathGridCell {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, ()> {
        if (0_u8..=9_u8).contains(&value) {
            Ok(Self(value + b'0'))
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<HeatLoss>);

impl Solution {
    const STANDARD_CRUCIBLE_STRAIGHT_LINE_LEN_RANGE: Range<i32> = 0_i32..3_i32;
    const ULTRA_CRUCIBLE_STRAIGHT_LINE_LEN_RANGE: Range<i32> = 4_i32..10_i32;

    fn regular_crucible_path_finder(&self) -> MinimalHeatLossPathFinder {
        MinimalHeatLossPathFinder {
            solution: self,
            vertex_to_vertex_data: HashMap::new(),
            start: Vertex {
                pos: IVec2::ZERO,
                prev_corner: IVec2::ZERO,
            },
            end_pos: self.0.max_dimensions(),
            straight_line_len_range: Self::STANDARD_CRUCIBLE_STRAIGHT_LINE_LEN_RANGE,
        }
    }

    fn ultra_crucible_path_finder(&self) -> MinimalHeatLossPathFinder {
        MinimalHeatLossPathFinder {
            straight_line_len_range: Self::ULTRA_CRUCIBLE_STRAIGHT_LINE_LEN_RANGE,
            ..self.regular_crucible_path_finder()
        }
    }

    fn minimal_heat_loss_grid_and_cost<
        F: for<'a> Fn(&'a Solution) -> MinimalHeatLossPathFinder<'a>,
    >(
        &self,
        path_finder: F,
    ) -> Option<(Grid2D<PathGridCell>, u32)> {
        let mut minimal_heat_loss_path_finder: MinimalHeatLossPathFinder = path_finder(self);

        let path: Option<Vec<Vertex>> = minimal_heat_loss_path_finder.run();

        path.map(|path| {
            let cost: u32 = minimal_heat_loss_path_finder
                .vertex_to_vertex_data
                .get(path.last().unwrap())
                .unwrap()
                .cost;
            let mut grid: Grid2D<PathGridCell> = Grid2D::try_from_cells_and_dimensions(
                self.0
                    .cells()
                    .iter()
                    .map(|heat_loss| PathGridCell::try_from(heat_loss.0).unwrap())
                    .collect(),
                self.0.dimensions(),
            )
            .unwrap();

            for vertex in path {
                if let Some(current_dir) = vertex.current_dir() {
                    *grid.get_mut(vertex.pos).unwrap() = current_dir.into();
                }
            }

            (grid, cost)
        })
    }

    fn regular_crucible_minimal_heat_loss_grid_and_cost(
        &self,
    ) -> Option<(Grid2D<PathGridCell>, u32)> {
        self.minimal_heat_loss_grid_and_cost(Self::regular_crucible_path_finder)
    }

    fn ultra_crucible_minimal_heat_loss_grid_and_cost(
        &self,
    ) -> Option<(Grid2D<PathGridCell>, u32)> {
        self.minimal_heat_loss_grid_and_cost(Self::ultra_crucible_path_finder)
    }

    fn minimal_heat_loss<F: for<'a> Fn(&'a Solution) -> MinimalHeatLossPathFinder<'a>>(
        &self,
        path_finder: F,
    ) -> Option<u32> {
        let mut minimal_heat_loss_path_finder: MinimalHeatLossPathFinder = path_finder(self);

        let path: Option<Vec<Vertex>> = minimal_heat_loss_path_finder.run();

        path.map(|path| {
            minimal_heat_loss_path_finder
                .vertex_to_vertex_data
                .get(path.last().unwrap())
                .unwrap()
                .cost
        })
    }

    fn regular_crucible_minimal_heat_loss(&self) -> Option<u32> {
        self.minimal_heat_loss(Self::regular_crucible_path_finder)
    }

    fn ultra_crucible_minimal_heat_loss(&self) -> Option<u32> {
        self.minimal_heat_loss(Self::ultra_crucible_path_finder)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::<HeatLoss>::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.regular_crucible_minimal_heat_loss());
        } else if let Some((grid, regular_crucible_minimal_heat_loss)) =
            self.regular_crucible_minimal_heat_loss_grid_and_cost()
        {
            dbg!(regular_crucible_minimal_heat_loss);

            println!("\n{}\n", String::from(grid));
        } else {
            eprintln!("failed to find regular crucible minimal heat loss path");
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.ultra_crucible_minimal_heat_loss());
        } else if let Some((grid, ultra_crucible_minimal_heat_loss)) =
            self.ultra_crucible_minimal_heat_loss_grid_and_cost()
        {
            dbg!(ultra_crucible_minimal_heat_loss);

            println!("\n{}\n", String::from(grid));
        } else {
            eprintln!("failed to find ultra crucible minimal heat loss path");
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
        2413432311323\n\
        3215453535623\n\
        3255245654254\n\
        3446585845452\n\
        4546657867536\n\
        1438598798454\n\
        4457876987766\n\
        3637877979653\n\
        4654967986887\n\
        4564679986453\n\
        1224686865563\n\
        2546548887735\n\
        4322674655533\n",
        "\
        111111111111\n\
        999999999991\n\
        999999999991\n\
        999999999991\n\
        999999999991\n",
    ];

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(
                Grid2D::try_from_cells_and_width(
                    [
                        2, 4, 1, 3, 4, 3, 2, 3, 1, 1, 3, 2, 3, 3, 2, 1, 5, 4, 5, 3, 5, 3, 5, 6, 2,
                        3, 3, 2, 5, 5, 2, 4, 5, 6, 5, 4, 2, 5, 4, 3, 4, 4, 6, 5, 8, 5, 8, 4, 5, 4,
                        5, 2, 4, 5, 4, 6, 6, 5, 7, 8, 6, 7, 5, 3, 6, 1, 4, 3, 8, 5, 9, 8, 7, 9, 8,
                        4, 5, 4, 4, 4, 5, 7, 8, 7, 6, 9, 8, 7, 7, 6, 6, 3, 6, 3, 7, 8, 7, 7, 9, 7,
                        9, 6, 5, 3, 4, 6, 5, 4, 9, 6, 7, 9, 8, 6, 8, 8, 7, 4, 5, 6, 4, 6, 7, 9, 9,
                        8, 6, 4, 5, 3, 1, 2, 2, 4, 6, 8, 6, 8, 6, 5, 5, 6, 3, 2, 5, 4, 6, 5, 4, 8,
                        8, 8, 7, 7, 3, 5, 4, 3, 2, 2, 6, 7, 4, 6, 5, 5, 5, 3, 3,
                    ]
                    .into_iter()
                    .map(|x| HeatLoss(x as u8))
                    .collect(),
                    13_usize,
                )
                .unwrap(),
            )
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(
            Solution::try_from(SOLUTION_STRS[0_usize]).as_ref(),
            Ok(solution())
        );
    }

    #[test]
    fn test_regular_crucible_minimal_heat_loss() {
        assert_eq!(
            solution().regular_crucible_minimal_heat_loss(),
            Some(102_u32)
        );
    }

    #[test]
    fn test_ultra_crucible_minimal_heat_loss() {
        assert_eq!(solution().ultra_crucible_minimal_heat_loss(), Some(94_u32));
        assert_eq!(
            Solution::try_from(SOLUTION_STRS[1_usize])
                .unwrap()
                .ultra_crucible_minimal_heat_loss(),
            Some(71_u32)
        );
    }
}
