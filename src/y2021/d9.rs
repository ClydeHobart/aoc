use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    std::{mem::MaybeUninit, ops::Range},
    strum::{EnumCount, IntoEnumIterator},
};

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct CharOutOfBounds(char);

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Height(u8);

impl Height {
    const MAX: u8 = 9_u8;
    const HEIGHT: Range<usize> = 0_usize..Direction::COUNT;
    const LOW_POINT: Range<usize> = Direction::COUNT..u8::BITS as usize;

    fn as_bitslice(&self) -> &BitSlice<u8> {
        self.0.view_bits()
    }

    fn as_bitslice_mut(&mut self) -> &mut BitSlice<u8> {
        self.0.view_bits_mut()
    }

    fn get_height(self) -> u8 {
        self.as_bitslice()[Self::HEIGHT].load()
    }

    fn set_height(&mut self, height: u8) {
        self.as_bitslice_mut()[Self::HEIGHT].store(if height > Self::MAX {
            eprintln!("height {height} is greater than max height {}", Self::MAX);

            Self::MAX
        } else {
            height
        });
    }

    fn set_is_lower(&mut self, dir: Direction, is_lower: bool) {
        self.as_bitslice_mut()[Self::LOW_POINT].set(dir as usize, is_lower);
    }

    fn is_low_point(self) -> bool {
        self.as_bitslice()[Self::LOW_POINT].all()
    }

    fn risk_level(self) -> u8 {
        self.get_height() + 1_u8
    }
}

impl TryFrom<char> for Height {
    type Error = CharOutOfBounds;

    fn try_from(height: char) -> Result<Self, Self::Error> {
        match height {
            '0'..='9' => Ok(Height(height as u8 - b'0')),
            _ => Err(CharOutOfBounds(height)),
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct HeightOnlyGrid(Grid2D<Height>);

impl<'i> TryFrom<&'i str> for HeightOnlyGrid {
    type Error = GridParseError<'i, CharOutOfBounds>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(Grid2D::try_from(input)?))
    }
}

struct HeightVisitor(u8);

impl Default for HeightVisitor {
    fn default() -> Self {
        Self(Height::MAX + 1_u8)
    }
}

impl GridVisitor for HeightVisitor {
    type Old = Height;
    type New = Height;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        _old_grid: &Grid2D<Self::Old>,
        rev_dir: Direction,
        _pos: glam::IVec2,
    ) {
        let height: u8 = old.get_height();

        new.set_height(height);
        new.set_is_lower(rev_dir, height < self.0);
        self.0 = height;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Basin {
    start: IVec2,
    size: usize,

    #[cfg(test)]
    positions: Vec<IVec2>,
}

struct BasinSizeFinder<'s> {
    solution: &'s Solution,
    state: &'s mut Basin,
}

impl<'s> BreadthFirstSearch for BasinSizeFinder<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.state.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!("No vertex is an end, so there should be no need to generate a path");
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        for dir in Direction::iter() {
            let neighbor: IVec2 = *vertex + dir.vec();

            if self
                .solution
                .0
                .get(neighbor)
                .map_or(false, |height| height.get_height() != Height::MAX)
            {
                neighbors.push(neighbor);
            }
        }
    }

    fn update_parent(&mut self, _from: &Self::Vertex, _to: &Self::Vertex) {
        self.state.size += 1_usize;

        #[cfg(test)]
        self.state.positions.push(*_to);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Height>);

impl Solution {
    fn low_points(&self) -> impl Iterator<Item = (IVec2, Height)> + '_ {
        self.0
            .cells()
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, height)| height.is_low_point())
            .map(|(index, height)| (self.0.pos_from_index(index), height))
    }

    fn low_point_risk_levels(&self) -> impl Iterator<Item = (IVec2, u8)> + '_ {
        self.low_points()
            .map(|(pos, height)| (pos, height.risk_level()))
    }

    fn low_point_risk_levels_sum(&self) -> u32 {
        self.low_point_risk_levels()
            .map(|(_, risk_level)| risk_level as u32)
            .sum()
    }

    fn basin(&self, low_point: IVec2) -> Basin {
        Basin {
            start: low_point,
            size: 1_usize,

            #[cfg(test)]
            positions: vec![low_point],
        }
    }

    fn basins(&self) -> impl Iterator<Item = Basin> + '_ {
        self.low_points().map(move |(low_point, _)| {
            let mut state: Basin = self.basin(low_point);

            BasinSizeFinder {
                solution: self,
                state: &mut state,
            }
            .run();

            #[cfg(test)]
            state.positions.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));

            state
        })
    }

    fn largest_basin_sizes_product<const N: usize>(&self) -> usize {
        // SAFETY: Zeroed bytes for all of `largest_sizes` is just `[0_usize; N]` (but that
        // expression isn't valid since it won't compile for all `N`).
        let mut largest_sizes: [usize; N] = unsafe { MaybeUninit::zeroed().assume_init() };

        for basin in self.basins() {
            let partition_point: usize = largest_sizes.partition_point(|size| *size >= basin.size);

            if partition_point < largest_sizes.len() {
                largest_sizes[partition_point..].rotate_right(1_usize);
                largest_sizes[partition_point] = basin.size;
            }
        }

        largest_sizes.into_iter().product()
    }

    fn largest_3_basin_sizes_product(&self) -> usize {
        self.largest_basin_sizes_product::<3_usize>()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.low_point_risk_levels_sum());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.largest_3_basin_sizes_product());
    }
}

impl From<&HeightOnlyGrid> for Solution {
    fn from(height_only_grid: &HeightOnlyGrid) -> Self {
        Self(HeightVisitor::visit_grid(&height_only_grid.0))
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = GridParseError<'i, CharOutOfBounds>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok((&HeightOnlyGrid::try_from(input)?).into())
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STR: &str = concat!(
        "2199943210\n",
        "3987894921\n",
        "9856789892\n",
        "8767896789\n",
        "9899965678\n",
    );
    const DIMENSIONS: IVec2 = IVec2::new(10_i32, 5_i32);

    macro_rules! heights {
        [ $( [ $( $height:expr ),* ], )*] => { vec![ $( $( Height($height), )* )* ] };
    }

    fn height_only_grid() -> &'static HeightOnlyGrid {
        static ONCE_LOCK: OnceLock<HeightOnlyGrid> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            HeightOnlyGrid(
                Grid2D::try_from_cells_and_dimensions(
                    heights![
                        [2, 1, 9, 9, 9, 4, 3, 2, 1, 0],
                        [3, 9, 8, 7, 8, 9, 4, 9, 2, 1],
                        [9, 8, 5, 6, 7, 8, 9, 8, 9, 2],
                        [8, 7, 6, 7, 8, 9, 6, 7, 8, 9],
                        [9, 8, 9, 9, 9, 6, 5, 6, 7, 8],
                    ],
                    DIMENSIONS,
                )
                .unwrap(),
            )
        })
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(
                Grid2D::try_from_cells_and_dimensions(
                    heights![
                        [0xD2, 0xF1, 0x19, 0x19, 0x19, 0xD4, 0xD3, 0xD2, 0xD1, 0xF0],
                        [0xE3, 0x09, 0x98, 0xB7, 0x38, 0x09, 0xE4, 0x09, 0xC2, 0xE1],
                        [0x89, 0x98, 0xF5, 0x76, 0x77, 0x78, 0x09, 0xB8, 0x09, 0xE2],
                        [0xD8, 0xD7, 0xE6, 0x67, 0x68, 0x09, 0xB6, 0x37, 0x38, 0x29],
                        [0xC9, 0xE8, 0x49, 0x49, 0x49, 0xD6, 0xF5, 0x76, 0x77, 0x78],
                    ],
                    DIMENSIONS,
                )
                .unwrap_or_else(|| Grid2D::empty(DIMENSIONS)),
            )
        })
    }

    #[test]
    fn test_height_only_grid_try_from_str() {
        assert_eq!(
            HeightOnlyGrid::try_from(SOLUTION_STR).as_ref(),
            Ok(height_only_grid())
        );
    }

    #[test]
    fn test_solution_from_height_only_grid() {
        assert_eq!(&Solution::from(height_only_grid()), solution())
    }

    #[test]
    fn test_low_points() {
        macro_rules! low_points {
            [ $( (($x:expr, $y:expr), $height:expr) ),* ] => {
                vec![$( (IVec2::new($x, $y), Height(0xF0_u8 | $height)), )*]
            };
        }

        assert_eq!(
            solution().low_points().collect::<Vec<(IVec2, Height)>>(),
            low_points![((1, 0), 1), ((9, 0), 0), ((2, 2), 5), ((6, 4), 5)]
        )
    }

    #[test]
    fn test_low_point_risk_levels() {
        macro_rules! low_point_risk_levels {
            [ $( (($x:expr, $y:expr), $height:expr) ),* ] => {
                vec![$( (IVec2::new($x, $y), $height), )*]
            };
        }

        assert_eq!(
            solution()
                .low_point_risk_levels()
                .collect::<Vec<(IVec2, u8)>>(),
            low_point_risk_levels![((1, 0), 2), ((9, 0), 1), ((2, 2), 6), ((6, 4), 6)]
        )
    }

    #[test]
    fn test_low_point_risk_levels_sum() {
        assert_eq!(solution().low_point_risk_levels_sum(), 15_u32);
    }

    #[test]
    fn test_basins() {
        macro_rules! basins {
            [ $( { ($sx:expr, $sy:expr), $size:expr, [ $( ($px:expr, $py:expr), )* ], }, )* ] => {
                vec![ $( Basin {
                    start: IVec2::new($sx, $sy),
                    size: $size,
                    positions: vec![ $( IVec2::new($px, $py), )* ]
                }, )* ]
            };
        }

        assert_eq!(
            solution().basins().collect::<Vec<Basin>>(),
            basins![
                {
                    (1, 0),
                    3,
                    [
                        (0, 0), (0, 1),
                        (1, 0),
                    ],
                },
                {
                    (9, 0),
                    9,
                    [
                        (5, 0),
                        (6, 0), (6, 1),
                        (7, 0),
                        (8, 0), (8, 1),
                        (9, 0), (9, 1), (9, 2),],
                },
                {
                    (2, 2),
                    14,
                    [
                        (0, 3),
                        (1, 2), (1, 3), (1, 4),
                        (2, 1), (2, 2), (2, 3),
                        (3, 1), (3, 2), (3, 3),
                        (4, 1), (4, 2), (4, 3),
                        (5, 2),
                    ],
                },
                {
                    (6, 4),
                    9,
                    [
                        (5, 4),
                        (6, 3), (6, 4),
                        (7, 2), (7, 3), (7, 4),
                        (8, 3), (8, 4),
                        (9, 4),
                    ],
                },
            ]
        )
    }

    #[test]
    fn test_largest_3_basin_sizes_product() {
        assert_eq!(solution().largest_3_basin_sizes_product(), 1134_usize);
    }
}
