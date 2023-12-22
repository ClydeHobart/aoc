use {
    crate::*,
    glam::{IVec2, IVec3, Vec3Swizzles},
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated, tuple},
        Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        ops::Range,
    },
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct Brick(Range<IVec3>);

impl Brick {
    fn parse_extremum<'i>(input: &'i str) -> IResult<&'i str, IVec3> {
        map(
            tuple((
                parse_integer::<i32>,
                tag(","),
                parse_integer::<i32>,
                tag(","),
                parse_integer::<i32>,
            )),
            |(x, _, y, _, z)| IVec3::new(x, y, z),
        )(input)
    }

    fn iter_poses(&self) -> impl Iterator<Item = IVec2> + '_ {
        CellIter2D::try_from(self.0.start.xy()..IVec2::new(self.0.end.x, self.0.start.y))
            .unwrap()
            .flat_map(|left_row| {
                CellIter2D::try_from(left_row..IVec2::new(left_row.x, self.0.end.y)).unwrap()
            })
    }
}

impl Parse for Brick {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(Self::parse_extremum, tag("~"), Self::parse_extremum),
            |(ext_a, ext_b)| {
                let min: IVec3 = ext_a.min(ext_b);
                let max: IVec3 = ext_a.max(ext_b);

                Self(min..max + IVec3::ONE)
            },
        )(input)
    }
}

#[derive(Clone, Copy)]
struct GridCell {
    height: u16,
    brick_index: u16,
}

impl GridCell {
    fn new(height: i32, brick_index: usize) -> Self {
        Self {
            height: height as u16,
            brick_index: brick_index as u16,
        }
    }

    fn height(self) -> Option<i32> {
        (self.height != u16::MAX).then_some(self.height as i32)
    }

    fn brick_index(self) -> Option<usize> {
        (self.brick_index != u16::MAX).then_some(self.brick_index as usize)
    }
}

impl Default for GridCell {
    fn default() -> Self {
        Self {
            height: u16::MAX,
            brick_index: u16::MAX,
        }
    }
}

#[derive(Default)]
struct SupportMapEntry {
    neighbors: VecDeque<u16>,
    start_supported_by: usize,
}

impl SupportMapEntry {
    fn push_supporting(&mut self, brick_index: usize) {
        self.neighbors.push_front(brick_index as u16);
        self.start_supported_by += 1_usize;
    }

    fn push_supported_by(&mut self, brick_index: usize) {
        self.neighbors.push_back(brick_index as u16);
    }

    fn iter_supporting(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors
            .range(..self.start_supported_by)
            .map(|brick_index| *brick_index as usize)
    }

    fn iter_supported_by(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors
            .range(self.start_supported_by..)
            .map(|brick_index| *brick_index as usize)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    bricks: Vec<Brick>,
    dimensions: IVec2,
}

impl Solution {
    fn sort_by_ascending_start(&mut self) {
        self.bricks.sort_by_key(|brick| brick.0.start.z);
    }

    fn settle(&mut self) -> usize {
        self.sort_by_ascending_start();

        let mut grid: Grid2D<GridCell> = Grid2D::default(self.dimensions);

        self.bricks
            .iter_mut()
            .enumerate()
            .filter_map(|(index, brick)| {
                let start_height: i32 = brick
                    .iter_poses()
                    .fold(None, |max_height: Option<i32>, grid_pos| {
                        match (max_height, grid.get(grid_pos).unwrap().height()) {
                            (None, None) => None,
                            (None, Some(max_height)) => Some(max_height),
                            (Some(max_height), None) => Some(max_height),
                            (Some(max_height_a), Some(max_height_b)) => {
                                Some(max_height_a.max(max_height_b))
                            }
                        }
                    })
                    .unwrap_or(1_i32);

                let result: Option<()> = if brick.0.start.z != start_height {
                    let height: i32 = brick.0.end.z - brick.0.start.z;

                    brick.0.start.z = start_height;
                    brick.0.end.z = start_height + height;

                    Some(())
                } else {
                    None
                };

                for pos in brick.iter_poses() {
                    *grid.get_mut(pos).unwrap() = GridCell::new(brick.0.end.z, index);
                }

                result
            })
            .count()
    }

    fn supports_map(&self) -> HashMap<usize, SupportMapEntry> {
        let mut supports_map: HashMap<usize, SupportMapEntry> = (0_usize..self.bricks.len())
            .map(|index| (index, SupportMapEntry::default()))
            .collect();
        let mut grid: Grid2D<GridCell> = Grid2D::default(self.dimensions);

        for (supported_index, brick) in self.bricks.iter().enumerate() {
            let brick_grid_cell: GridCell = GridCell::new(brick.0.end.z, supported_index);

            for pos in brick.iter_poses() {
                let grid_cell: &mut GridCell = grid.get_mut(pos).unwrap();

                if let Some(max_height) = grid_cell.height() {
                    if max_height == brick.0.start.z {
                        let supporting_index: usize = grid_cell.brick_index().unwrap();

                        supports_map
                            .get_mut(&supporting_index)
                            .unwrap()
                            .push_supporting(supported_index);

                        supports_map
                            .get_mut(&supported_index)
                            .unwrap()
                            .push_supported_by(supporting_index);
                    }
                }

                *grid_cell = brick_grid_cell;
            }
        }

        supports_map
    }

    fn iter_brick_indices_and_disintegration_safety(&self) -> impl Iterator<Item = (usize, bool)> {
        let supports_map: HashMap<usize, SupportMapEntry> = self.supports_map();

        (0_usize..self.bricks.len()).map(move |index| {
            (
                index,
                supports_map
                    .get(&index)
                    .unwrap()
                    .iter_supporting()
                    .all(|supported_index| {
                        supports_map
                            .get(&supported_index)
                            .unwrap()
                            .iter_supported_by()
                            .filter(|supporting_index| *supporting_index != index)
                            .count()
                            > 0_usize
                    }),
            )
        })
    }

    /// Iterate over bricks that can safely be disintegrated. This function assumes that `self` is
    /// currently settled. This leaves bricks in
    fn iter_safe_to_disintegrate_brick_indices(&self) -> impl Iterator<Item = usize> {
        self.iter_brick_indices_and_disintegration_safety()
            .filter_map(|(index, safety)| safety.then_some(index))
    }

    fn count_safe_to_disintegrate_bricks_after_settling(&self) -> usize {
        let mut solution: Solution = self.clone();

        solution.settle();

        solution.iter_safe_to_disintegrate_brick_indices().count()
    }

    fn iter_fallen_bricks(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter_brick_indices_and_disintegration_safety()
            .filter_map(|(index, safety)| {
                (!safety).then(|| {
                    let mut solution: Solution = self.clone();

                    solution.bricks.remove(index);

                    solution.settle()
                })
            })
    }

    fn sum_fallen_bricks(&self) -> usize {
        let mut solution: Solution = self.clone();

        solution.settle();

        solution.iter_fallen_bricks().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(Brick::parse, opt(line_ending))),
            |mut bricks| {
                let range: Range<IVec2> =
                    bricks.iter().fold(IVec2::MAX..IVec2::MIN, |range, brick| {
                        range.start.min(brick.0.start.xy())..range.end.max(brick.0.end.xy())
                    });

                let dimensions: IVec2 = range.end - range.start;

                if range.start != IVec2::ZERO {
                    let offset: IVec3 = IVec3::new(range.start.x, range.start.y, 0_i32);

                    for brick in bricks.iter_mut() {
                        brick.0.start -= offset;
                        brick.0.end -= offset;
                    }
                }

                let mut solution: Self = Self { bricks, dimensions };

                solution.sort_by_ascending_start();

                solution
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_safe_to_disintegrate_bricks_after_settling());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_fallen_bricks());
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

    const SOLUTION_STR: &'static str = "\
        1,0,1~1,2,1\n\
        0,0,2~2,0,2\n\
        0,2,3~2,2,3\n\
        0,0,4~0,2,4\n\
        2,0,5~2,2,5\n\
        0,1,6~2,1,6\n\
        1,1,8~1,1,9\n";

    macro_rules! bricks {
        [ $( $x1:expr, $y1:expr, $z1:expr => $x2:expr, $y2:expr, $z2:expr; )* ] => { vec![ $(
            Brick(IVec3::new($x1, $y1, $z1)..IVec3::new($x2, $y2, $z2) + IVec3::ONE),
        )* ] }
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            bricks: bricks![
                1, 0, 1 => 1, 2, 1;
                0, 0, 2 => 2, 0, 2;
                0, 2, 3 => 2, 2, 3;
                0, 0, 4 => 0, 2, 4;
                2, 0, 5 => 2, 2, 5;
                0, 1, 6 => 2, 1, 6;
                1, 1, 8 => 1, 1, 9;
            ],
            dimensions: 3_i32 * IVec2::ONE,
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_settle() {
        let mut solution: Solution = solution().clone();

        assert_eq!(solution.settle(), 5_usize);

        assert_eq!(
            solution,
            Solution {
                bricks: bricks![
                    1, 0, 1 => 1, 2, 1;
                    0, 0, 2 => 2, 0, 2;
                    0, 2, 2 => 2, 2, 2;
                    0, 0, 3 => 0, 2, 3;
                    2, 0, 3 => 2, 2, 3;
                    0, 1, 4 => 2, 1, 4;
                    1, 1, 5 => 1, 1, 6;
                ],
                dimensions: 3_i32 * IVec2::ONE,
            }
        );
    }

    #[test]
    fn test_iter_safe_to_disintegrate_bricks() {
        let mut solution: Solution = solution().clone();

        solution.settle();

        assert_eq!(
            solution
                .iter_safe_to_disintegrate_brick_indices()
                .collect::<Vec<usize>>(),
            vec![
                // 0_usize, A cannot be disintegrated safely
                1_usize, // B can be disintegrated
                2_usize, // C can be distintegrated
                3_usize, // D can be distintegrated
                4_usize, // E can be distintegrated
                // 5_usize, F cannot be disintegrated safely
                6_usize, // G can be distintegrated
            ]
        )
    }

    #[test]
    fn test_iter_fallen_bricks() {
        let mut solution: Solution = solution().clone();

        solution.settle();

        assert_eq!(
            solution.iter_fallen_bricks().collect::<Vec<usize>>(),
            vec![6_usize, 1_usize]
        );
    }
}
