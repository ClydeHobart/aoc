use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many1,
        sequence::tuple,
        Err, IResult,
    },
    std::cmp::Ordering,
};

#[cfg(test)]
use std::ops::RangeInclusive;

/* --- Day 6: Chronal Coordinates ---

The device on your wrist beeps several times, and once again you feel like you're falling.

"Situation critical," the device announces. "Destination indeterminate. Chronal interference detected. Please specify new target coordinates."

The device then produces a list of coordinates (your puzzle input). Are they places it thinks are safe or dangerous? It recommends you check manual page 729. The Elves did not give you a manual.

If they're dangerous, maybe you can minimize the danger by finding the coordinate that gives the largest distance from the other points.

Using only the Manhattan distance, determine the area around each coordinate by counting the number of integer X,Y locations that are closest to that coordinate (and aren't tied in distance to any other coordinate).

Your goal is to find the size of the largest area that isn't infinite. For example, consider the following list of coordinates:

1, 1
1, 6
8, 3
3, 4
5, 5
8, 9

If we name these coordinates A through F, we can draw them on a grid, putting 0,0 at the top left:

..........
.A........
..........
........C.
...D......
.....E....
.B........
..........
..........
........F.

This view is partial - the actual grid extends infinitely in all directions. Using the Manhattan distance, each location's closest coordinate can be determined, shown here in lowercase:

aaaaa.cccc
aAaaa.cccc
aaaddecccc
aadddeccCc
..dDdeeccc
bb.deEeecc
bBb.eeee..
bbb.eeefff
bbb.eeffff
bbb.ffffFf

Locations shown as . are equally far from two or more coordinates, and so they don't count as being closest to any.

In this example, the areas of coordinates A, B, C, and F are infinite - while not shown here, their areas extend forever outside the visible grid. However, the areas of coordinates D and E are finite: D is closest to 9 locations, and E is closest to 17 (both including the coordinate's location itself). Therefore, in this example, the size of the largest area is 17.

What is the size of the largest area that isn't infinite?

--- Part Two ---

On the other hand, if the coordinates are safe, maybe the best you can do is try to find a region near as many coordinates as possible.

For example, suppose you want the sum of the Manhattan distance to all of the coordinates to be less than 32. For each location, add up the distances to all of the given coordinates; if the total of those distances is less than 32, that location is within the desired region. Using the same coordinates as above, the resulting region looks like this:

..........
.A........
..........
...###..C.
..#D###...
..###E#...
.B.###....
..........
..........
........F.

In particular, consider the highlighted location 4,3 located at the top middle of the region. Its calculation is as follows, where abs() is the absolute value function:

    Distance to coordinate A: abs(4-1) + abs(3-1) =  5
    Distance to coordinate B: abs(4-1) + abs(3-6) =  6
    Distance to coordinate C: abs(4-8) + abs(3-3) =  4
    Distance to coordinate D: abs(4-3) + abs(3-4) =  2
    Distance to coordinate E: abs(4-5) + abs(3-5) =  3
    Distance to coordinate F: abs(4-8) + abs(3-9) = 10
    Total distance: 5 + 6 + 4 + 2 + 3 + 10 = 30

Because the total distance to all coordinates (30) is less than 32, the location is within the region.

This region, which also includes coordinates D and E, has a total size of 16.

Your actual region will need to be much larger than this example, though, instead including all locations with a total distance of less than 10000.

What is the size of the region containing all locations which have a total distance to all given coordinates of less than 10000? */

type CoordinateIndexRaw = u8;
type CoordinateIndex = Index<CoordinateIndexRaw>;

#[derive(Clone, Copy, Default, PartialEq)]
struct CoordinateIndexPixel {
    a: CoordinateIndex,
    b: CoordinateIndex,
}

impl CoordinateIndexPixel {
    fn new(a: CoordinateIndex) -> Self {
        Self {
            a,
            b: CoordinateIndex::invalid(),
        }
    }

    fn is_edge(self) -> bool {
        self.a.is_valid() && self.b.is_valid()
    }

    fn get_edge_is_valid(self) -> CoordinateIndex {
        self.a
    }

    fn get_edge_is_invalid(self) -> CoordinateIndex {
        if self.is_edge() {
            CoordinateIndex::invalid()
        } else {
            self.a
        }
    }

    fn set(&mut self, value: CoordinateIndex, strictly_closer: bool) {
        if strictly_closer {
            *self = Self::new(value);
        } else if self.a.is_valid() {
            self.b = value;
        } else {
            self.a = value;
        }
    }
}

#[allow(unused)]
#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
struct CoordinateIndexCell(u8);

#[allow(unused)]
#[cfg(test)]
impl CoordinateIndexCell {
    const INVALID_BYTE: u8 = b'.';
    const INVALID: Self = Self(Self::INVALID_BYTE);
    const LOWERCASE_BYTE_RANGE_START: u8 = b'a';
    const LOWERCASE_BYTE_RANGE_END: u8 = b'z';
    const UPPERCASE_BYTE_RANGE_START: u8 = b'A';
    const UPPERCASE_BYTE_RANGE_END: u8 = b'Z';
    const LOWERCASE_BYTE_RANGE: RangeInclusive<u8> =
        Self::LOWERCASE_BYTE_RANGE_START..=Self::LOWERCASE_BYTE_RANGE_END;
    const UPPERCASE_BYTE_RANGE: RangeInclusive<u8> =
        Self::UPPERCASE_BYTE_RANGE_START..=Self::UPPERCASE_BYTE_RANGE_END;
    const LOWERCASE_BYTE_RANGE_LEN: usize = Self::byte_range_len(Self::LOWERCASE_BYTE_RANGE);
    const UPPERCASE_BYTE_RANGE_LEN: usize = Self::byte_range_len(Self::UPPERCASE_BYTE_RANGE);
    const COORDINATE_INDEX_COUNT: usize =
        Self::LOWERCASE_BYTE_RANGE_LEN + Self::UPPERCASE_BYTE_RANGE_LEN;
    const COORDINATE_INDEX_LOWERCASE_BYTE_START: u8 = 0_u8;
    const COORDINATE_INDEX_UPPERCASE_BYTE_START: u8 = Self::LOWERCASE_BYTE_RANGE_LEN as u8;
    const COORDINATE_INDEX_TO_LOWERCASE_BYTE_OFFSET: u8 =
        Self::LOWERCASE_BYTE_RANGE_START - Self::COORDINATE_INDEX_LOWERCASE_BYTE_START;
    const COORDINATE_INDEX_TO_UPPERCASE_BYTE_OFFSET: u8 =
        Self::UPPERCASE_BYTE_RANGE_START - Self::COORDINATE_INDEX_UPPERCASE_BYTE_START;

    const fn byte_range_len(byte_range: RangeInclusive<u8>) -> usize {
        *byte_range.end() as usize - *byte_range.start() as usize + 1_usize
    }

    fn is_valid(self) -> bool {
        self != Self::INVALID
    }

    fn try_invert(self) -> Option<Self> {
        self.is_valid().then(|| Self(self.0 ^ ASCII_CASE_MASK))
    }
}

#[cfg(test)]
impl Default for CoordinateIndexCell {
    fn default() -> Self {
        Self::INVALID
    }
}

// SAFETY: See `TryFrom` implementation below.
#[cfg(test)]
unsafe impl IsValidAscii for CoordinateIndexCell {}

#[cfg(test)]
impl TryFrom<CoordinateIndex> for CoordinateIndexCell {
    type Error = ();

    fn try_from(value: CoordinateIndex) -> Result<Self, Self::Error> {
        if !value.is_valid() {
            Ok(Self::INVALID)
        } else if value.get() < Self::LOWERCASE_BYTE_RANGE_LEN {
            Ok(Self(
                value.get() as u8 + Self::COORDINATE_INDEX_TO_LOWERCASE_BYTE_OFFSET,
            ))
        } else if value.get() < Self::COORDINATE_INDEX_COUNT {
            Ok(Self(
                value.get() as u8 + Self::COORDINATE_INDEX_TO_UPPERCASE_BYTE_OFFSET,
            ))
        } else {
            Err(())
        }
    }
}

struct LeastDangerousCoordinateFinder<'s> {
    solution: &'s Solution,
    grid: Grid2D<CoordinateIndexPixel>,
}

impl<'s> LeastDangerousCoordinateFinder<'s> {
    fn try_largest_finite_area(&mut self) -> Option<usize> {
        self.run_jfa_squared();

        Solution::try_largest_finite_area_from_coordinates_len_and_grid(
            self.solution.coordinates.len(),
            &self.grid,
            |cell| *cell,
        )
    }

    #[allow(unused)]
    #[cfg(test)]
    fn try_grid_string(&self) -> Option<String> {
        let mut grid: Grid2D<CoordinateIndexCell> = Grid2D::default(self.grid.dimensions());

        self.grid
            .cells()
            .iter()
            .zip(grid.cells_mut().iter_mut())
            .try_for_each(|(coordinate_index_pixel, coordinate_index_cell)| {
                CoordinateIndexCell::try_from(coordinate_index_pixel.get_edge_is_invalid())
                    .ok()
                    .map(|coordinate_index_cell_source| {
                        *coordinate_index_cell = coordinate_index_cell_source;
                    })
            })
            .map(|_| {
                for coordinate in &self.solution.coordinates {
                    let coordinate_index_cell: &mut CoordinateIndexCell =
                        grid.get_mut(*coordinate).unwrap();

                    if let Some(inverse) = coordinate_index_cell.try_invert() {
                        *coordinate_index_cell = inverse;
                    }
                }

                grid.into()
            })
    }
}

impl<'s> JumpFloodingAlgorithm for LeastDangerousCoordinateFinder<'s> {
    type Pixel = CoordinateIndex;

    type Dist = i32;

    fn dist(a: IVec2, b: IVec2) -> Self::Dist {
        manhattan_distance_2d(a, b)
    }

    fn n(&self) -> i32 {
        self.solution.n
    }

    fn is_pos_valid(&self, pos: IVec2) -> bool {
        self.grid.contains(pos)
    }

    fn try_get_p_pixel(&self, pos: IVec2) -> Option<Self::Pixel> {
        // Consider an edge valid so that a later evaluation doesn't break a tie without reason.
        self.grid.get(pos).unwrap().get_edge_is_valid().opt()
    }

    fn try_get_q_pixel(&self, pos: IVec2) -> Option<Self::Pixel> {
        // Don't consider an edge to be valid, so that we don't propagate from an edge.
        self.grid.get(pos).unwrap().get_edge_is_invalid().opt()
    }

    fn get_seed(&self, pixel: Self::Pixel) -> IVec2 {
        self.solution.coordinates[pixel.get()]
    }

    fn update_pixel(&mut self, pos: IVec2, pixel: Self::Pixel, strictly_closer: bool) {
        self.grid.get_mut(pos).unwrap().set(pixel, strictly_closer);
    }

    fn reset(&mut self) {
        self.grid.cells_mut().fill(CoordinateIndexPixel::default());

        for (index, coordinate) in self.solution.coordinates.iter().enumerate() {
            *self.grid.get_mut(*coordinate).unwrap() = CoordinateIndexPixel::new(index.into());
        }
    }

    fn on_iteration_complate(&mut self) {}
}

struct CoordinateDistCell {
    min_dist_coordinate: CoordinateIndexPixel,
    min_dist: u16,
    total_dist: u16,
}

impl Default for CoordinateDistCell {
    fn default() -> Self {
        Self {
            min_dist_coordinate: Default::default(),
            min_dist: u16::MAX,
            total_dist: Default::default(),
        }
    }
}

pub struct Solution {
    coordinates: Vec<IVec2>,
    n: i32,
    grid: Grid2D<CoordinateDistCell>,
}

impl Solution {
    const THRESHOLD_TOTAL_DIST: u16 = 10_000_u16;

    fn grid_from_coordinates_and_n(coordinates: &[IVec2], n: i32) -> Grid2D<CoordinateDistCell> {
        let mut grid: Grid2D<CoordinateDistCell> = Grid2D::default(n * IVec2::ONE);

        for (coordinate_index, coordinate) in coordinates.iter().enumerate() {
            let coordinate_index: CoordinateIndex = coordinate_index.into();
            let mut pos: IVec2 = IVec2::ZERO;
            let mut y_dist: u16 = (pos.y - coordinate.y).abs() as u16;

            for cell in grid.cells_mut() {
                let dist: u16 = y_dist + (pos.x - coordinate.x).abs() as u16;

                cell.total_dist += dist;

                let ordering: Ordering = cell.min_dist.cmp(&dist);

                if ordering.is_ge() {
                    cell.min_dist_coordinate
                        .set(coordinate_index, ordering.is_gt());
                    cell.min_dist = dist;
                }

                pos.x = (pos.x + 1_i32) % n;

                if pos.x == 0_i32 {
                    pos.y += 1_i32;
                    y_dist = (pos.y - coordinate.y).abs() as u16;
                }
            }
        }

        grid
    }

    fn try_largest_finite_area_from_coordinates_len_and_grid<
        C,
        F: Fn(&C) -> CoordinateIndexPixel,
    >(
        coordinates_len: usize,
        grid: &Grid2D<C>,
        f: F,
    ) -> Option<usize> {
        const INFINITE_AREA: usize = usize::MAX;

        let mut areas: Vec<usize> = vec![0_usize; coordinates_len];

        for pos in CellIter2D::iter_edges_for_dimensions(grid.dimensions()) {
            if let Some(coordinate_index) = f(grid.get(pos).unwrap()).get_edge_is_invalid().opt() {
                areas[coordinate_index.get()] = INFINITE_AREA;
            }
        }

        for cell in grid.cells() {
            if let Some(coordinate_index) = f(cell).get_edge_is_invalid().opt() {
                let area: &mut usize = &mut areas[coordinate_index.get()];

                if *area < INFINITE_AREA {
                    *area += 1_usize;
                }
            }
        }

        areas.into_iter().filter(|area| *area < INFINITE_AREA).max()
    }

    fn least_dangerous_coordinate_finder(&self) -> LeastDangerousCoordinateFinder {
        let solution: &Solution = self;
        let grid: Grid2D<CoordinateIndexPixel> = Grid2D::default(solution.n * IVec2::ONE);

        LeastDangerousCoordinateFinder { solution, grid }
    }

    fn try_largest_finite_area_slow(&self) -> Option<usize> {
        self.least_dangerous_coordinate_finder()
            .try_largest_finite_area()
    }

    fn try_largest_finite_area(&self) -> Option<usize> {
        Self::try_largest_finite_area_from_coordinates_len_and_grid(
            self.coordinates.len(),
            &self.grid,
            |cell| cell.min_dist_coordinate,
        )
    }

    fn count_cells_with_total_dist_less_than_threshold(&self, threshold_total_dist: u16) -> usize {
        self.grid
            .cells()
            .iter()
            .filter(|cell| cell.total_dist < threshold_total_dist)
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many1(map(
                tuple((parse_integer, tag(", "), parse_integer, opt(line_ending))),
                |(x, _, y, _)| IVec2 { x, y },
            )),
            |mut coordinates| {
                let mut min: IVec2 = i32::MAX * IVec2::ONE;
                let mut max: IVec2 = i32::MIN * IVec2::ONE;

                for coordinate in &coordinates {
                    min = min.min(*coordinate);
                    max = max.max(*coordinate);
                }

                let dimensions: IVec2 = max - min + IVec2::ONE;
                let max_dimension: i32 = dimensions.max_element();
                let n: i32 = if max_dimension.count_ones() == 1_u32 {
                    max_dimension
                } else {
                    1_i32 << (max_dimension.ilog2() + 1_u32)
                };
                let center: IVec2 = (max + min) / 2_i32;
                let delta: IVec2 = (n / 2_i32) * IVec2::ONE - center;

                for coordinate in &mut coordinates {
                    *coordinate += delta;
                }

                let grid: Grid2D<CoordinateDistCell> =
                    Self::grid_from_coordinates_and_n(&coordinates, n);

                Self {
                    coordinates,
                    n,
                    grid,
                }
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// My JFA implementation is finicky, compounded by it being "an approximation of a Voronoi
    /// diagram". It took lots of fiddling to get to where we are now, and even now I'm not
    /// confident that this would succeed with 100% of user inputs.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.try_largest_finite_area());

        if args.verbose {
            dbg!(self.try_largest_finite_area_slow());
        }
    }

    // JFA definitely wasn't the way to go for this one. Not only was it way slower than brute
    // forcing the problem, it didn't set me up at all for the second question.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_cells_with_total_dist_less_than_threshold(Self::THRESHOLD_TOTAL_DIST));
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        1, 1\n\
        1, 6\n\
        8, 3\n\
        3, 4\n\
        5, 5\n\
        8, 9\n"];

    fn solution_full(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![Solution::try_from(SOLUTION_STRS[0_usize]).unwrap()])[index]
    }

    fn solution_partial(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                coordinates: vec![
                    IVec2::new(5_i32, 4_i32),
                    IVec2::new(5_i32, 9_i32),
                    IVec2::new(12_i32, 6_i32),
                    IVec2::new(7_i32, 7_i32),
                    IVec2::new(9_i32, 8_i32),
                    IVec2::new(12_i32, 12_i32),
                ],
                n: 16_i32,
                grid: Grid2D::default(16_i32 * IVec2::ONE),
            }]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for index in 0_usize..SOLUTION_STRS.len() {
            let solution_full: &Solution = solution_full(index);
            let solution_partial: &Solution = solution_partial(index);

            assert_eq!(solution_full.coordinates, solution_partial.coordinates);
            assert_eq!(solution_full.n, solution_partial.n);
        }
    }

    #[test]
    fn test_try_largest_finite_area() {
        for (index, largest_finite_area) in [Some(17_usize)].into_iter().enumerate() {
            assert_eq!(
                solution_full(index).try_largest_finite_area(),
                largest_finite_area
            );
            assert_eq!(
                solution_full(index).try_largest_finite_area_slow(),
                largest_finite_area
            );
        }
    }

    #[test]
    fn test_count_cells_with_total_dist_less_than_threshold() {
        for (index, cells_with_total_dist_less_than_threshold_count) in
            [16_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution_full(index).count_cells_with_total_dist_less_than_threshold(32_u16),
                cells_with_total_dist_less_than_threshold_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
