use {
    crate::*,
    glam::{BVec2, IVec2},
    nom::{
        combinator::{map, verify},
        error::Error,
        Err, IResult,
    },
    num::integer::Roots,
};

/* --- Day 3: Spiral Memory ---

You come across an experimental new kind of memory stored on an infinite two-dimensional grid.

Each square on the grid is allocated in a spiral pattern starting at a location marked 1 and then counting up while spiraling outward. For example, the first few squares are allocated like this:

17  16  15  14  13
18   5   4   3  12
19   6   1   2  11
20   7   8   9  10
21  22  23---> ...

While this is very space-efficient (no squares are skipped), requested data must be carried back to square 1 (the location of the only access port for this memory system) by programs that can only move up, down, left, or right. They always take the shortest path: the Manhattan Distance between the location of the data and square 1.

For example:

    Data from square 1 is carried 0 steps, since it's at the access port.
    Data from square 12 is carried 3 steps, such as: down, left, left.
    Data from square 23 is carried only 2 steps: up twice.
    Data from square 1024 must be carried 31 steps.

How many steps are required to carry the data from the square identified in your puzzle input all the way to the access port?

--- Part Two ---

As a stress test on the system, the programs here clear the grid and then store the value 1 in square 1. Then, in the same allocation order as shown above, they store the sum of the values in all adjacent squares, including diagonals.

So, the first few squares' values are chosen as follows:

    Square 1 starts with the value 1.
    Square 2 has only one adjacent filled square (with value 1), so it also stores 1.
    Square 3 has both of the above squares as neighbors and stores the sum of their values, 2.
    Square 4 has all three of the aforementioned squares as neighbors and stores the sum of their values, 4.
    Square 5 only has the first and fourth squares as neighbors, so it gets the value 5.

Once a square is written, its value does not change. Therefore, the first few squares would receive the following values:

147  142  133  122   59
304    5    4    2   57
330   10    1    1   54
351   11   23   25   26
362  747  806--->   ...

What is the first value written that is larger than your puzzle input? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct GridIndex(u32);

struct GridIndexIVec2ConversionValues {
    max_grid_index_in_ring: GridIndex,
    divisor: u32,
    half_ring_size: i32,
}

impl GridIndexIVec2ConversionValues {
    fn new(ring_size: u32) -> Self {
        Self {
            max_grid_index_in_ring: GridIndex(ring_size * ring_size),
            divisor: (ring_size - 1_u32).max(1_u32),

            // ring_size is always odd, so this is always under the true value. That is intentional.
            half_ring_size: ring_size as i32 / 2_i32,
        }
    }

    fn ring_size(grid_index: GridIndex) -> u32 {
        let sqrt: u32 = grid_index.0.sqrt();
        let complete_rings: u32 = (sqrt - 1_u32) / 2_u32 * 2_u32 + 1_u32;

        complete_rings + 2_u32 * (complete_rings * complete_rings < grid_index.0) as u32
    }

    fn corner(&self, dir: Direction) -> IVec2 {
        dir.prev().vec() * self.half_ring_size + dir.rev().vec() * self.half_ring_size
    }
}

impl From<GridIndex> for IVec2 {
    fn from(value: GridIndex) -> Self {
        let ring_size: u32 = GridIndexIVec2ConversionValues::ring_size(value);
        let conversion_values: GridIndexIVec2ConversionValues =
            GridIndexIVec2ConversionValues::new(ring_size);
        let grid_index_difference: u32 = conversion_values.max_grid_index_in_ring.0 - value.0;
        let side_index: u8 = (grid_index_difference / conversion_values.divisor) as u8;
        let dir: Direction = (side_index + 3_u8).into();

        conversion_values.corner(dir)
            + dir.vec() * (grid_index_difference % conversion_values.divisor) as i32
    }
}

impl From<IVec2> for GridIndex {
    fn from(value: IVec2) -> Self {
        let abs: IVec2 = value.abs();
        let max_element: i32 = abs.max_element();
        let min_element: i32 = abs.min_element();
        let ring_size: u32 = max_element as u32 * 2_u32 + 1_u32;
        let conversion_values: GridIndexIVec2ConversionValues =
            GridIndexIVec2ConversionValues::new(ring_size);
        let dir: Direction = if max_element == min_element {
            // This is along one of the diagonals
            match value.cmpge(IVec2::ZERO) {
                BVec2 { x: true, y: true } => Direction::West,
                BVec2 { x: false, y: true } => Direction::North,
                BVec2 { x: false, y: false } => Direction::East,
                BVec2 { x: true, y: false } => Direction::South,
            }
        } else {
            // This is along one of the sides
            match (abs.x == max_element, value.x > 0_i32, value.y > 0_i32) {
                (false, _, false) => Direction::East,
                (false, _, true) => Direction::West,
                (true, false, _) => Direction::North,
                (true, true, _) => Direction::South,
            }
        };

        let side_index: u8 = (dir as u8 + 1_u8) % 4_u8;
        let grid_index_difference_at_corner: u32 = side_index as u32 * conversion_values.divisor;
        let grid_index_difference: u32 = grid_index_difference_at_corner
            + manhattan_distance_2d(value, conversion_values.corner(dir)) as u32;

        GridIndex(conversion_values.max_grid_index_in_ring.0 - grid_index_difference)
    }
}

#[derive(Clone, Copy)]
struct LinearIndex(usize);

impl From<GridIndex> for LinearIndex {
    fn from(value: GridIndex) -> Self {
        Self((value.0 - 1_u32) as usize)
    }
}

impl From<LinearIndex> for GridIndex {
    fn from(value: LinearIndex) -> Self {
        Self((value.0 + 1_usize) as u32)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
pub struct Solution(u32);

impl Solution {
    fn manhattan_distance_to_access_port(&self) -> i32 {
        manhattan_magnitude_2d(GridIndex(self.0).into())
    }

    fn first_value_larger_than_self(&self) -> u32 {
        let mut values: Vec<u32> = vec![1_u32];

        while *values.last().unwrap() <= self.0 {
            let pos: IVec2 = GridIndex::from(LinearIndex(values.len())).into();
            let value: u32 = (-1_i32..=1_i32)
                .flat_map(|x| (-1_i32..=1_i32).map(move |y| IVec2::new(x, y)))
                .filter(|delta| *delta != IVec2::ZERO)
                .filter_map(|delta| {
                    values
                        .get(LinearIndex::from(GridIndex::from(pos + delta)).0)
                        .copied()
                })
                .sum();

            values.push(value);
        }

        *values.last().unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(verify(parse_integer, |i| *i > 0_u32), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I definitely spent more time on this than I should've. It would've been way faster to just
    /// do it with a grid or something.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.manhattan_distance_to_access_port());
    }

    /// Similarly, it would've been a lot faster to do this with a grid than to figure out the
    /// mapping in the other direction.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.first_value_larger_than_self());
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

    const SOLUTION_STRS: &'static [&'static str] = &["1", "12", "23", "1024"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(1_u32),
                Solution(12_u32),
                Solution(23_u32),
                Solution(1024_u32),
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
    fn test_ring_size() {
        for (index, ring_size) in [1_u32, 5_u32, 5_u32, 33_u32].into_iter().enumerate() {
            assert_eq!(
                GridIndexIVec2ConversionValues::ring_size(GridIndex(solution(index).0)),
                ring_size
            );
        }
    }

    #[test]
    fn test_ivec2_from_grid_index() {
        for (index, pos) in [
            IVec2::ZERO,
            IVec2::new(2_i32, -1_i32),
            IVec2::new(0_i32, 2_i32),
            IVec2::new(-15_i32, -16_i32),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(IVec2::from(GridIndex(solution(index).0)), pos);
        }
    }

    #[test]
    fn test_manhattan_distance_to_access_port() {
        for (index, manhattan_distance_to_access_port) in
            [0_i32, 3_i32, 2_i32, 31_i32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).manhattan_distance_to_access_port(),
                manhattan_distance_to_access_port
            );
        }
    }

    #[test]
    fn test_grid_index_from_ivec2() {
        for index in 0_usize..SOLUTION_STRS.len() {
            let grid_index: GridIndex = GridIndex(solution(index).0);

            assert_eq!(GridIndex::from(IVec2::from(grid_index)), grid_index);
        }
    }

    #[test]
    fn test_first_value_larger_than_self() {
        Solution(1862).first_value_larger_than_self();

        for (index, first_value_larger_than_self) in
            [2_u32, 23_u32, 25_u32, 1968_u32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).first_value_larger_than_self(),
                first_value_larger_than_self
            );
        }
    }
}
