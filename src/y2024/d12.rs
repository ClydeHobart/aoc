use {
    crate::*,
    glam::IVec2,
    nom::{
        character::complete::satisfy,
        combinator::{map, verify},
        error::Error,
        Err, IResult,
    },
    std::{cmp::Ordering, ops::Range},
    strum::IntoEnumIterator,
};

/* --- Day 12: Garden Groups ---

Why not search for the Chief Historian near the gardener and his massive farm? There's plenty of food, so The Historians grab something to eat while they search.

You're about to settle near a complex arrangement of garden plots when some Elves ask if you can lend a hand. They'd like to set up fences around each region of garden plots, but they can't figure out how much fence they need to order or how much it will cost. They hand you a map (your puzzle input) of the garden plots.

Each garden plot grows only a single type of plant and is indicated by a single letter on your map. When multiple garden plots are growing the same type of plant and are touching (horizontally or vertically), they form a region. For example:

AAAA
BBCD
BBCC
EEEC

This 4x4 arrangement includes garden plots growing five different types of plants (labeled A, B, C, D, and E), each grouped into their own region.

In order to accurately calculate the cost of the fence around a single region, you need to know that region's area and perimeter.

The area of a region is simply the number of garden plots the region contains. The above map's type A, B, and C plants are each in a region of area 4. The type E plants are in a region of area 3; the type D plants are in a region of area 1.

Each garden plot is a square and so has four sides. The perimeter of a region is the number of sides of garden plots in the region that do not touch another garden plot in the same region. The type A and C plants are each in a region with perimeter 10. The type B and E plants are each in a region with perimeter 8. The lone D plot forms its own region with perimeter 4.

Visually indicating the sides of plots in each region that contribute to the perimeter using - and |, the above map's regions' perimeters are measured as follows:

+-+-+-+-+
|A A A A|
+-+-+-+-+     +-+
              |D|
+-+-+   +-+   +-+
|B B|   |C|
+   +   + +-+
|B B|   |C C|
+-+-+   +-+ +
          |C|
+-+-+-+   +-+
|E E E|
+-+-+-+

Plants of the same type can appear in multiple separate regions, and regions can even appear within other regions. For example:

OOOOO
OXOXO
OOOOO
OXOXO
OOOOO

The above map contains five regions, one containing all of the O garden plots, and the other four each containing a single X plot.

The four X regions each have area 1 and perimeter 4. The region containing 21 type O plants is more complicated; in addition to its outer edge contributing a perimeter of 20, its boundary with each X region contributes an additional 4 to its perimeter, for a total perimeter of 36.

Due to "modern" business practices, the price of fence required for a region is found by multiplying that region's area by its perimeter. The total price of fencing all regions on a map is found by adding together the price of fence for every region on the map.

In the first example, region A has price 4 * 10 = 40, region B has price 4 * 8 = 32, region C has price 4 * 10 = 40, region D has price 1 * 4 = 4, and region E has price 3 * 8 = 24. So, the total price for the first example is 140.

In the second example, the region with all of the O plants has price 21 * 36 = 756, and each of the four smaller X regions has price 1 * 4 = 4, for a total price of 772 (756 + 4 + 4 + 4 + 4).

Here's a larger example:

RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE

It contains:

    A region of R plants with price 12 * 18 = 216.
    A region of I plants with price 4 * 8 = 32.
    A region of C plants with price 14 * 28 = 392.
    A region of F plants with price 10 * 18 = 180.
    A region of V plants with price 13 * 20 = 260.
    A region of J plants with price 11 * 20 = 220.
    A region of C plants with price 1 * 4 = 4.
    A region of E plants with price 13 * 18 = 234.
    A region of I plants with price 14 * 22 = 308.
    A region of M plants with price 5 * 12 = 60.
    A region of S plants with price 3 * 8 = 24.

So, it has a total price of 1930.

What is the total price of fencing all regions on your map?

--- Part Two ---

Fortunately, the Elves are trying to order so much fence that they qualify for a bulk discount!

Under the bulk discount, instead of using the perimeter to calculate the price, you need to use the number of sides each region has. Each straight section of fence counts as a side, regardless of how long it is.

Consider this example again:

AAAA
BBCD
BBCC
EEEC

The region containing type A plants has 4 sides, as does each of the regions containing plants of type B, D, and E. However, the more complex region containing the plants of type C has 8 sides!

Using the new method of calculating the per-region price by multiplying the region's area by its number of sides, regions A through E have prices 16, 16, 32, 4, and 12, respectively, for a total price of 80.

The second example above (full of type X and O plants) would have a total price of 436.

Here's a map that includes an E-shaped region full of type E plants:

EEEEE
EXXXX
EEEEE
EXXXX
EEEEE

The E-shaped region has an area of 17 and 12 sides for a price of 204. Including the two regions full of type X plants, this map has a total price of 236.

This map has a total price of 368:

AAAAAA
AAABBA
AAABBA
ABBAAA
ABBAAA
AAAAAA

It includes two regions full of type B plants (each with 4 sides) and a single region full of type A plants (with 4 sides on the outside and 8 more sides on the inside, a total of 12 sides). Be especially careful when counting the fence around regions like the one full of type A plants; in particular, each section of fence has an in-side and an out-side, so the fence does not connect across the middle of the region (where the two B regions touch diagonally). (The Elves would have used the Möbius Fencing Company instead, but their contract terms were too one-sided.)

The larger example from before now has the following updated prices:

    A region of R plants with price 12 * 10 = 120.
    A region of I plants with price 4 * 4 = 16.
    A region of C plants with price 14 * 22 = 308.
    A region of F plants with price 10 * 12 = 120.
    A region of V plants with price 13 * 10 = 130.
    A region of J plants with price 11 * 12 = 132.
    A region of C plants with price 1 * 4 = 4.
    A region of E plants with price 13 * 8 = 104.
    A region of I plants with price 14 * 16 = 224.
    A region of M plants with price 5 * 6 = 30.
    A region of S plants with price 3 * 6 = 18.

Adding these together produces its new total price of 1206.

What is the new total price of fencing all regions on your map? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct Plot(u8);

impl Plot {
    const OFFSET: u8 = b'A';
    const DEFAULT: Self = Self(Self::OFFSET);
}

impl Default for Plot {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Parse for Plot {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(|c| c.is_ascii_uppercase()), |c| Self(c as u8))(input)
    }
}

/// SAFETY: See `Parse` implementation.
unsafe impl IsValidAscii for Plot {}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Eq, PartialEq)]
struct Region {
    plot: Plot,
    pos_range: Range<u16>,
}

impl Region {
    fn iter_plot_sides<'s>(
        &self,
        solution: &'s Solution,
    ) -> impl Iterator<Item = SmallPosAndDir> + 's {
        let plot: Plot = self.plot;

        solution.poses[self.pos_range.as_range_usize()]
            .iter()
            .copied()
            .flat_map(move |small_pos| {
                let pos: IVec2 = small_pos.get();

                Direction::iter().filter_map(move |dir| {
                    solution
                        .grid
                        .get(pos + dir.vec())
                        .map_or(true, |neighbor_plot| *neighbor_plot != plot)
                        .then(|| SmallPosAndDir {
                            pos: small_pos,
                            dir,
                        })
                })
            })
    }

    fn side_count(&self, solution: &Solution, plot_sides: &mut Vec<SmallPosAndDir>) -> usize {
        plot_sides.clear();
        plot_sides.extend(self.iter_plot_sides(solution));
        plot_sides.sort_by(|plot_side_a, plot_side_b| {
            (plot_side_a.dir as u8)
                .cmp(&(plot_side_b.dir as u8))
                .then_with(|| {
                    if plot_side_a.dir.is_north_or_south() {
                        plot_side_a
                            .pos
                            .y
                            .cmp(&plot_side_b.pos.y)
                            .then_with(|| plot_side_a.pos.x.cmp(&plot_side_b.pos.x))
                    } else {
                        plot_side_a
                            .pos
                            .x
                            .cmp(&plot_side_b.pos.x)
                            .then_with(|| plot_side_a.pos.y.cmp(&plot_side_b.pos.y))
                    }
                })
        });

        let mut prev_plot_side: Option<SmallPosAndDir> = None;

        plot_sides
            .drain(..)
            .filter(|curr_plot_side| {
                prev_plot_side
                    .replace(curr_plot_side.clone())
                    .map_or(true, |prev_plot_side| {
                        curr_plot_side.dir != prev_plot_side.dir
                            || manhattan_distance_2d(
                                prev_plot_side.pos.get(),
                                curr_plot_side.pos.get(),
                            ) != 1_i32
                    })
            })
            .count()
    }

    fn perimeter(&self, solution: &Solution) -> usize {
        self.iter_plot_sides(solution).count()
    }
}

impl Ord for Region {
    fn cmp(&self, other: &Self) -> Ordering {
        self.plot.cmp(&other.plot).then_with(|| {
            self.pos_range
                .start
                .cmp(&other.pos_range.start)
                .then_with(|| self.pos_range.end.cmp(&other.pos_range.end))
        })
    }
}

impl PartialOrd for Region {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct RegionFinder<'s> {
    solution: &'s mut Solution,
    start: IVec2,
    plot: Plot,
}

impl<'s> BreadthFirstSearch for RegionFinder<'s> {
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
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            self.solution
                .grid
                .get(neighbor)
                .map_or(false, |neighbor_plot| *neighbor_plot == self.plot)
                .then_some(neighbor)
        }))
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        // SAFETY: `to` is valid.
        self.solution
            .poses
            .push(unsafe { SmallPos::from_pos_unsafe(*to) });
    }

    fn reset(&mut self) {}
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Plot>,
    poses: Vec<SmallPos>,
    regions: Vec<Region>,
}

impl Solution {
    fn iter_regions_and_perimeters(&self) -> impl Iterator<Item = (Region, usize)> + '_ {
        self.regions
            .iter()
            .map(|region| (region.clone(), region.perimeter(self)))
    }

    fn iter_regions_and_side_counts(&self) -> impl Iterator<Item = (Region, usize)> + '_ {
        let mut plot_sides: Vec<SmallPosAndDir> = Vec::new();

        self.regions
            .iter()
            .map(move |region| (region.clone(), region.side_count(self, &mut plot_sides)))
    }

    fn total_price(&self) -> usize {
        self.iter_regions_and_perimeters()
            .map(|(region, perimeter)| region.pos_range.len() * perimeter)
            .sum()
    }

    fn total_discount_price(&self) -> usize {
        self.iter_regions_and_side_counts()
            .map(|(region, side_count)| region.pos_range.len() * side_count)
            .sum()
    }

    fn init_poses_and_regions(&mut self) {
        let mut has_visited_pos: SmallPosBitArr = SmallPosBitArr::ZERO;
        let grid_cells_len: usize = self.grid.cells().len();

        for index in 0_usize..grid_cells_len {
            let start: IVec2 = self.grid.pos_from_index(index);

            // SAFETY: `start` is valid.
            let small_start: SmallPos = unsafe { SmallPos::from_pos_unsafe(start) };

            if !has_visited_pos[small_start.sortable_index() as usize] {
                let plot: Plot = self.grid.cells()[index];

                let pos_range_start: u16 = self.poses.len() as u16;

                self.poses.push(small_start);

                RegionFinder {
                    solution: self,
                    start,
                    plot,
                }
                .run();

                let pos_range_end: u16 = self.poses.len() as u16;
                let pos_range: Range<u16> = pos_range_start..pos_range_end;

                self.poses[pos_range.as_range_usize()]
                    .sort_by_key(|small_pos| small_pos.sortable_index());

                for small_pos in &self.poses[pos_range.as_range_usize()] {
                    has_visited_pos.set(small_pos.sortable_index() as usize, true);
                }

                self.regions.push(Region { plot, pos_range })
            }
        }

        self.regions.sort();
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            verify(Grid2D::parse, |grid| {
                SmallPos::are_dimensions_valid(grid.dimensions())
            }),
            |grid| {
                let mut solution: Self = Self {
                    grid,
                    poses: Vec::new(),
                    regions: Vec::new(),
                };

                solution.init_poses_and_regions();

                solution
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Misread it initially, thought that regions could contain disjoint sub-regions.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_price());
    }

    /// Baffled at first, but then I figured out a way that's not too bad.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_discount_price());
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
        AAAA\n\
        BBCD\n\
        BBCC\n\
        EEEC\n",
        "\
        OOOOO\n\
        OXOXO\n\
        OOOOO\n\
        OXOXO\n\
        OOOOO\n",
        "\
        RRRRIICCFF\n\
        RRRRIICCCF\n\
        VVRRRCCFFF\n\
        VVRCCCJFFF\n\
        VVVVCJJCFE\n\
        VVIVCCJJEE\n\
        VVIIICJJEE\n\
        MIIIIIJJEE\n\
        MIIISIJEEE\n\
        MMMISSJEEE\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            macro_rules! small_poses {
                [ $( ($x:expr, $y:expr), )* ] => { vec![ $( SmallPos { x: $x, y: $y }, )* ] }
            }

            vec![
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            Plot(b'A'),
                            Plot(b'A'),
                            Plot(b'A'),
                            Plot(b'A'),
                            Plot(b'B'),
                            Plot(b'B'),
                            Plot(b'C'),
                            Plot(b'D'),
                            Plot(b'B'),
                            Plot(b'B'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'C'),
                        ],
                        4_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    poses: small_poses![
                        (0, 0),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (0, 1),
                        (1, 1),
                        (0, 2),
                        (1, 2),
                        (2, 1),
                        (2, 2),
                        (3, 2),
                        (3, 3),
                        (3, 1),
                        (0, 3),
                        (1, 3),
                        (2, 3),
                    ],
                    regions: vec![
                        Region {
                            plot: Plot(b'A'),
                            pos_range: 0_u16..4_u16,
                        },
                        Region {
                            plot: Plot(b'B'),
                            pos_range: 4_u16..8_u16,
                        },
                        Region {
                            plot: Plot(b'C'),
                            pos_range: 8_u16..12_u16,
                        },
                        Region {
                            plot: Plot(b'D'),
                            pos_range: 12_u16..13_u16,
                        },
                        Region {
                            plot: Plot(b'E'),
                            pos_range: 13_u16..16_u16,
                        },
                    ],
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'X'),
                            Plot(b'O'),
                            Plot(b'X'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'X'),
                            Plot(b'O'),
                            Plot(b'X'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                            Plot(b'O'),
                        ],
                        5_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    poses: small_poses![
                        (0, 0),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (4, 0),
                        (0, 1),
                        (2, 1),
                        (4, 1),
                        (0, 2),
                        (1, 2),
                        (2, 2),
                        (3, 2),
                        (4, 2),
                        (0, 3),
                        (2, 3),
                        (4, 3),
                        (0, 4),
                        (1, 4),
                        (2, 4),
                        (3, 4),
                        (4, 4),
                        (1, 1),
                        (3, 1),
                        (1, 3),
                        (3, 3),
                    ],
                    regions: vec![
                        Region {
                            plot: Plot(b'O'),
                            pos_range: 0_u16..21_u16,
                        },
                        Region {
                            plot: Plot(b'X'),
                            pos_range: 21_u16..22_u16,
                        },
                        Region {
                            plot: Plot(b'X'),
                            pos_range: 22_u16..23_u16,
                        },
                        Region {
                            plot: Plot(b'X'),
                            pos_range: 23_u16..24_u16,
                        },
                        Region {
                            plot: Plot(b'X'),
                            pos_range: 24_u16..25_u16,
                        },
                    ],
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'F'),
                            Plot(b'F'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'F'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'R'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'F'),
                            Plot(b'F'),
                            Plot(b'F'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'R'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'J'),
                            Plot(b'F'),
                            Plot(b'F'),
                            Plot(b'F'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'C'),
                            Plot(b'J'),
                            Plot(b'J'),
                            Plot(b'C'),
                            Plot(b'F'),
                            Plot(b'E'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'I'),
                            Plot(b'V'),
                            Plot(b'C'),
                            Plot(b'C'),
                            Plot(b'J'),
                            Plot(b'J'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'V'),
                            Plot(b'V'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'C'),
                            Plot(b'J'),
                            Plot(b'J'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'M'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'J'),
                            Plot(b'J'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'M'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'I'),
                            Plot(b'S'),
                            Plot(b'I'),
                            Plot(b'J'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'M'),
                            Plot(b'M'),
                            Plot(b'M'),
                            Plot(b'I'),
                            Plot(b'S'),
                            Plot(b'S'),
                            Plot(b'J'),
                            Plot(b'E'),
                            Plot(b'E'),
                            Plot(b'E'),
                        ],
                        10_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    poses: small_poses![
                        // R
                        (0, 0),
                        (1, 0),
                        (2, 0),
                        (3, 0),
                        (0, 1),
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (2, 2),
                        (3, 2),
                        (4, 2),
                        (2, 3),
                        // I
                        (4, 0),
                        (5, 0),
                        (4, 1),
                        (5, 1),
                        // C
                        (6, 0),
                        (7, 0),
                        (6, 1),
                        (7, 1),
                        (8, 1),
                        (5, 2),
                        (6, 2),
                        (3, 3),
                        (4, 3),
                        (5, 3),
                        (4, 4),
                        (4, 5),
                        (5, 5),
                        (5, 6),
                        // F
                        (8, 0),
                        (9, 0),
                        (9, 1),
                        (7, 2),
                        (8, 2),
                        (9, 2),
                        (7, 3),
                        (8, 3),
                        (9, 3),
                        (8, 4),
                        // V
                        (0, 2),
                        (1, 2),
                        (0, 3),
                        (1, 3),
                        (0, 4),
                        (1, 4),
                        (2, 4),
                        (3, 4),
                        (0, 5),
                        (1, 5),
                        (3, 5),
                        (0, 6),
                        (1, 6),
                        // J
                        (6, 3),
                        (5, 4),
                        (6, 4),
                        (6, 5),
                        (7, 5),
                        (6, 6),
                        (7, 6),
                        (6, 7),
                        (7, 7),
                        (6, 8),
                        (6, 9),
                        // C
                        (7, 4),
                        // E
                        (9, 4),
                        (8, 5),
                        (9, 5),
                        (8, 6),
                        (9, 6),
                        (8, 7),
                        (9, 7),
                        (7, 8),
                        (8, 8),
                        (9, 8),
                        (7, 9),
                        (8, 9),
                        (9, 9),
                        // I
                        (2, 5),
                        (2, 6),
                        (3, 6),
                        (4, 6),
                        (1, 7),
                        (2, 7),
                        (3, 7),
                        (4, 7),
                        (5, 7),
                        (1, 8),
                        (2, 8),
                        (3, 8),
                        (5, 8),
                        (3, 9),
                        // M
                        (0, 7),
                        (0, 8),
                        (0, 9),
                        (1, 9),
                        (2, 9),
                        // S
                        (4, 8),
                        (4, 9),
                        (5, 9),
                    ],
                    regions: vec![
                        Region {
                            plot: Plot(b'C'),
                            pos_range: 16_u16..30_u16,
                        },
                        Region {
                            plot: Plot(b'C'),
                            pos_range: 64_u16..65_u16,
                        },
                        Region {
                            plot: Plot(b'E'),
                            pos_range: 65_u16..78_u16,
                        },
                        Region {
                            plot: Plot(b'F'),
                            pos_range: 30_u16..40_u16,
                        },
                        Region {
                            plot: Plot(b'I'),
                            pos_range: 12_u16..16_u16,
                        },
                        Region {
                            plot: Plot(b'I'),
                            pos_range: 78_u16..92_u16,
                        },
                        Region {
                            plot: Plot(b'J'),
                            pos_range: 53_u16..64_u16,
                        },
                        Region {
                            plot: Plot(b'M'),
                            pos_range: 92_u16..97_u16,
                        },
                        Region {
                            plot: Plot(b'R'),
                            pos_range: 0_u16..12_u16,
                        },
                        Region {
                            plot: Plot(b'S'),
                            pos_range: 97_u16..100_u16,
                        },
                        Region {
                            plot: Plot(b'V'),
                            pos_range: 40_u16..53_u16,
                        },
                    ],
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
    fn test_perimeter() {
        for (index, perimeters) in [
            vec![10_usize, 8_usize, 10_usize, 4_usize, 8_usize],
            vec![36_usize, 4_usize, 4_usize, 4_usize, 4_usize],
            vec![
                28_usize, 4_usize, 18_usize, 18_usize, 8_usize, 22_usize, 20_usize, 12_usize,
                18_usize, 8_usize, 20_usize,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_regions_and_perimeters()
                    .map(|(_, perimeter)| perimeter)
                    .collect::<Vec<usize>>(),
                perimeters
            );
        }
    }

    #[test]
    fn test_total_price() {
        for (index, total_price) in [140_usize, 772_usize, 1930_usize].into_iter().enumerate() {
            assert_eq!(solution(index).total_price(), total_price);
        }
    }

    #[test]
    fn test_side_counts() {
        for (index, side_counts) in [
            vec![4_usize, 4_usize, 8_usize, 4_usize, 4_usize],
            vec![20_usize, 4_usize, 4_usize, 4_usize, 4_usize],
            vec![
                22_usize, 4_usize, 8_usize, 12_usize, 4_usize, 16_usize, 12_usize, 6_usize,
                10_usize, 6_usize, 10_usize,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_regions_and_side_counts()
                    .map(|(_, side_count)| side_count)
                    .collect::<Vec<usize>>(),
                side_counts
            );
        }
    }

    #[test]
    fn test_total_discount_price() {
        for (index, total_discount_price) in
            [80_usize, 436_usize, 1206_usize].into_iter().enumerate()
        {
            assert_eq!(solution(index).total_discount_price(), total_discount_price);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
