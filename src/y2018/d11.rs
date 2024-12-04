use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
    rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    std::fmt::{Debug, Formatter, Result as FmtResult},
};

/* --- Day 11: Chronal Charge ---

You watch the Elves and their sleigh fade into the distance as they head toward the North Pole.

Actually, you're the one fading. The falling sensation returns.

The low fuel warning light is illuminated on your wrist-mounted device. Tapping it once causes it to project a hologram of the situation: a 300x300 grid of fuel cells and their current power levels, some negative. You're not sure what negative power means in the context of time travel, but it can't be good.

Each fuel cell has a coordinate ranging from 1 to 300 in both the X (horizontal) and Y (vertical) direction. In X,Y notation, the top-left cell is 1,1, and the top-right cell is 300,1.

The interface lets you select any 3x3 square of fuel cells. To increase your chances of getting to your destination, you decide to choose the 3x3 square with the largest total power.

The power level in a given fuel cell can be found through the following process:

    Find the fuel cell's rack ID, which is its X coordinate plus 10.
    Begin with a power level of the rack ID times the Y coordinate.
    Increase the power level by the value of the grid serial number (your puzzle input).
    Set the power level to itself multiplied by the rack ID.
    Keep only the hundreds digit of the power level (so 12345 becomes 3; numbers with no hundreds digit become 0).
    Subtract 5 from the power level.

For example, to find the power level of the fuel cell at 3,5 in a grid with serial number 8:

    The rack ID is 3 + 10 = 13.
    The power level starts at 13 * 5 = 65.
    Adding the serial number produces 65 + 8 = 73.
    Multiplying by the rack ID produces 73 * 13 = 949.
    The hundreds digit of 949 is 9.
    Subtracting 5 produces 9 - 5 = 4.

So, the power level of this fuel cell is 4.

Here are some more example power levels:

    Fuel cell at  122,79, grid serial number 57: power level -5.
    Fuel cell at 217,196, grid serial number 39: power level  0.
    Fuel cell at 101,153, grid serial number 71: power level  4.

Your goal is to find the 3x3 square which has the largest total power. The square must be entirely within the 300x300 grid. Identify this square using the X,Y coordinate of its top-left fuel cell. For example:

For grid serial number 18, the largest total 3x3 square has a top-left corner of 33,45 (with a total power of 29); these fuel cells appear in the middle of this 5x5 region:

-2  -4   4   4   4
-4   4   4   4  -5
 4   3   3   4  -4
 1   1   2   4  -3
-1   0   2  -5  -2

For grid serial number 42, the largest 3x3 square's top-left is 21,61 (with a total power of 30); they are in the middle of this region:

-3   4   2   2   2
-4   4   3   3   4
-5   3   3   4  -4
 4   3   3   4  -3
 3   3   3  -5  -1

What is the X,Y coordinate of the top-left fuel cell of the 3x3 square with the largest total power?

--- Part Two ---

You discover a dial on the side of the device; it seems to let you select a square of any size, not just 3x3. Sizes from 1x1 to 300x300 are supported.

Realizing this, you now must find the square of any size with the largest total power. Identify this square by including its size as a third parameter after the top-left coordinate: a 9x9 square with a top-left corner of 3,5 is identified as 3,5,9.

For example:

    For grid serial number 18, the largest total square (with a total power of 113) is 16x16 and has a top-left corner of 90,269, so its identifier is 90,269,16.
    For grid serial number 42, the largest total square (with a total power of 119) is 12x12 and has a top-left corner of 232,251, so its identifier is 232,251,12.

What is the X,Y,size identifier of the square with the largest total power? */

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Copy)]
struct FuelCellCoordinate2(IVec2);

impl FuelCellCoordinate2 {
    const RACK_ID_OFFSET: i32 = 10_i32;
    const POWER_LEVEL_KEPT_DIGIT_EXPONENT: u32 = 2_u32;
    const POWER_LEVEL_KEPT_DIGIT_DIVISOR: i32 = 10_i32.pow(Self::POWER_LEVEL_KEPT_DIGIT_EXPONENT);
    const POWER_LEVEL_FINAL_OFFSET: i32 = -5_i32;
    const FROM_IVEC2_OFFSET: IVec2 = IVec2::ONE;

    fn rack_id(self) -> i32 {
        self.0.x + Self::RACK_ID_OFFSET
    }

    fn power_level(self, grid_serial_number: i32) -> i32 {
        let rack_id: i32 = self.rack_id();

        let mut power_level: i32 = rack_id * self.0.y;

        power_level += grid_serial_number;
        power_level *= rack_id;
        power_level = (power_level / Self::POWER_LEVEL_KEPT_DIGIT_DIVISOR) % 10_i32;
        power_level += Self::POWER_LEVEL_FINAL_OFFSET;

        power_level as i32
    }
}

impl Debug for FuelCellCoordinate2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{},{}", self.0.x, self.0.y)
    }
}

impl From<IVec2> for FuelCellCoordinate2 {
    fn from(value: IVec2) -> Self {
        Self(value + Self::FROM_IVEC2_OFFSET)
    }
}

impl From<FuelCellCoordinate2> for IVec2 {
    fn from(value: FuelCellCoordinate2) -> Self {
        value.0 - FuelCellCoordinate2::FROM_IVEC2_OFFSET
    }
}

#[cfg_attr(test, derive(PartialEq))]
struct FuelCellCoordinate3 {
    fuel_cell_coordinate_2: FuelCellCoordinate2,
    square_size: i32,
}

impl Debug for FuelCellCoordinate3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?},{}", &self.fuel_cell_coordinate_2, self.square_size)
    }
}

struct PowerLevelPyramid {
    power_level_grids: Vec<Grid2D<i32>>,
}

impl PowerLevelPyramid {
    const GRID_SIDE_LEN: i32 = 300_i32;
    const GRID_DIMENSIONS: IVec2 = IVec2::new(Self::GRID_SIDE_LEN, Self::GRID_SIDE_LEN);

    fn new(grid_serial_number: i32) -> Self {
        let mut power_level_pyramid: Self = Self {
            power_level_grids: Vec::with_capacity(Self::GRID_SIDE_LEN as usize),
        };

        power_level_pyramid.build_base_layer(grid_serial_number);
        power_level_pyramid.build_remaining_layers();

        power_level_pyramid
    }

    fn find_largest_total_power_with_square_size(
        &self,
        square_size: i32,
    ) -> (FuelCellCoordinate2, i32) {
        let power_level_grid: &Grid2D<i32> =
            &self.power_level_grids[square_size as usize - 1_usize];

        power_level_grid
            .cells()
            .par_iter()
            .enumerate()
            .map(|(index, &power_level)| {
                (
                    FuelCellCoordinate2::from(power_level_grid.pos_from_index(index)),
                    power_level,
                )
            })
            .max_by_key(|(_, power_level)| *power_level)
            .unwrap()
    }

    fn find_largest_total_power(&self) -> (FuelCellCoordinate3, i32) {
        (1_i32..=Self::GRID_SIDE_LEN)
            .into_par_iter()
            .map(|square_size| {
                let (fuel_cell_coordinate_2, power_level): (FuelCellCoordinate2, i32) =
                    self.find_largest_total_power_with_square_size(square_size);

                (
                    FuelCellCoordinate3 {
                        fuel_cell_coordinate_2,
                        square_size,
                    },
                    power_level,
                )
            })
            .max_by_key(|(_, power_level)| *power_level)
            .unwrap()
    }

    fn build_base_layer(&mut self, grid_serial_number: i32) {
        let mut power_level_grid: Grid2D<i32> = Grid2D::default(Self::GRID_DIMENSIONS);

        power_level_grid
            .cells_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, power_level)| {
                *power_level = FuelCellCoordinate2::from(grid_2d_pos_from_index_and_dimensions(
                    index,
                    Self::GRID_DIMENSIONS,
                ))
                .power_level(grid_serial_number);
            });

        self.power_level_grids.push(power_level_grid);
    }

    fn build_next_layer(&mut self) {
        assert!(!self.power_level_grids.is_empty());

        let square_side_len_minus_one: i32 = self.power_level_grids.len() as i32;
        let square_side_len: i32 = square_side_len_minus_one + 1_i32;
        let dimensions: IVec2 = Self::GRID_DIMENSIONS - square_side_len_minus_one * IVec2::ONE;
        let base_power_level_grid: &Grid2D<i32> = self.power_level_grids.first().unwrap();
        let prev_power_level_grid: &Grid2D<i32> = self.power_level_grids.last().unwrap();

        let mut curr_power_level_grid: Grid2D<i32> = Grid2D::default(dimensions);

        curr_power_level_grid
            .cells_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, curr_power_level)| {
                let pos: IVec2 = grid_2d_pos_from_index_and_dimensions(index, dimensions);

                *curr_power_level = CellIter2D::try_from(
                    IVec2::new(0_i32, square_side_len_minus_one)
                        ..IVec2::new(square_side_len, square_side_len_minus_one),
                )
                .unwrap()
                .chain(
                    CellIter2D::try_from(
                        IVec2::new(square_side_len_minus_one, 0_i32)
                            ..IVec2::new(square_side_len_minus_one, square_side_len_minus_one),
                    )
                    .unwrap(),
                )
                .map(|base_pos_delta| *base_power_level_grid.get(pos + base_pos_delta).unwrap())
                .chain([*prev_power_level_grid.get(pos).unwrap()])
                .sum()
            });

        self.power_level_grids.push(curr_power_level_grid);
    }

    fn build_remaining_layers(&mut self) {
        assert!(!self.power_level_grids.is_empty());

        while self.power_level_grids.len() < Self::GRID_SIDE_LEN as usize {
            self.build_next_layer();
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(i32);

impl Solution {
    const SQUARE_SIZE: i32 = 3_i32;

    fn find_largest_total_power_with_square_size(
        &self,
        square_size: i32,
    ) -> (FuelCellCoordinate2, i32) {
        PowerLevelPyramid::new(self.0).find_largest_total_power_with_square_size(square_size)
    }

    fn find_largest_total_power(&self) -> (FuelCellCoordinate3, i32) {
        PowerLevelPyramid::new(self.0).find_largest_total_power()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// one-based indexing is one of humanity's greatest sins.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_largest_total_power_with_square_size(Self::SQUARE_SIZE));
    }

    /// This is still slower than I'd like it, even w/ some Rayon sauce applied.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_largest_total_power());
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

    const SOLUTION_STRS: &'static [&'static str] = &["8", "57", "39", "71", "18", "42"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(8_i32),
                Solution(57_i32),
                Solution(39_i32),
                Solution(71_i32),
                Solution(18_i32),
                Solution(42_i32),
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
    fn test_power_level() {
        for (index, (fuel_cell_coordinate, power_level)) in [
            (FuelCellCoordinate2(IVec2::new(3_i32, 5_i32)), 4_i32),
            (FuelCellCoordinate2(IVec2::new(122_i32, 79_i32)), -5_i32),
            (FuelCellCoordinate2(IVec2::new(217_i32, 196_i32)), 0_i32),
            (FuelCellCoordinate2(IVec2::new(101_i32, 153_i32)), 4_i32),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                fuel_cell_coordinate.power_level(solution(index).0),
                power_level
            );
        }
    }

    #[test]
    fn test_find_largest_total_power_with_square_size() {
        for (index, largest_total_power_with_square_size) in [
            None,
            None,
            None,
            None,
            Some((FuelCellCoordinate2(IVec2::new(33_i32, 45_i32)), 29_i32)),
            Some((FuelCellCoordinate2(IVec2::new(21_i32, 61_i32)), 30_i32)),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some((fuel_cell_coordinate, largest_total_power)) =
                largest_total_power_with_square_size
            {
                assert_eq!(
                    solution(index)
                        .find_largest_total_power_with_square_size(Solution::SQUARE_SIZE),
                    (fuel_cell_coordinate, largest_total_power)
                );
            }
        }
    }

    #[test]
    fn test_find_largest_total_power() {
        for (index, largest_total_power) in [
            None,
            None,
            None,
            None,
            Some((
                FuelCellCoordinate3 {
                    fuel_cell_coordinate_2: FuelCellCoordinate2(IVec2::new(90_i32, 269_i32)),
                    square_size: 16_i32,
                },
                113_i32,
            )),
            Some((
                FuelCellCoordinate3 {
                    fuel_cell_coordinate_2: FuelCellCoordinate2(IVec2::new(232_i32, 251_i32)),
                    square_size: 12_i32,
                },
                119_i32,
            )),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(largest_total_power) = largest_total_power {
                assert_eq!(
                    solution(index).find_largest_total_power(),
                    largest_total_power
                );
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
