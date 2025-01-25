use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt},
        error::Error,
        multi::separated_list0,
        sequence::{separated_pair, tuple},
        Err, IResult,
    },
    std::{collections::HashSet, mem::take},
};

/* --- Day 17: Reservoir Research ---

You arrive in the year 18. If it weren't for the coat you got in 1018, you would be very cold: the North Pole base hasn't even been constructed.

Rather, it hasn't been constructed yet. The Elves are making a little progress, but there's not a lot of liquid water in this climate, so they're getting very dehydrated. Maybe there's more underground?

You scan a two-dimensional vertical slice of the ground nearby and discover that it is mostly sand with veins of clay. The scan only provides data with a granularity of square meters, but it should be good enough to determine how much water is trapped there. In the scan, x represents the distance to the right, and y represents the distance down. There is also a spring of water near the surface at x=500, y=0. The scan identifies which square meters are clay (your puzzle input).

For example, suppose your scan shows the following veins of clay:

x=495, y=2..7
y=7, x=495..501
x=501, y=3..7
x=498, y=2..4
x=506, y=1..2
x=498, y=10..13
x=504, y=10..13
y=13, x=498..504

Rendering clay as #, sand as ., and the water spring as +, and with x increasing to the right and y increasing downward, this becomes:

   44444455555555
   99999900000000
   45678901234567
 0 ......+.......
 1 ............#.
 2 .#..#.......#.
 3 .#..#..#......
 4 .#..#..#......
 5 .#.....#......
 6 .#.....#......
 7 .#######......
 8 ..............
 9 ..............
10 ....#.....#...
11 ....#.....#...
12 ....#.....#...
13 ....#######...

The spring of water will produce water forever. Water can move through sand, but is blocked by clay. Water always moves down when possible, and spreads to the left and right otherwise, filling space that has clay on both sides and falling out otherwise.

For example, if five squares of water are created, they will flow downward until they reach the clay and settle there. Water that has come to rest is shown here as ~, while sand through which water has passed (but which is now dry again) is shown as |:

......+.......
......|.....#.
.#..#.|.....#.
.#..#.|#......
.#..#.|#......
.#....|#......
.#~~~~~#......
.#######......
..............
..............
....#.....#...
....#.....#...
....#.....#...
....#######...

Two squares of water can't occupy the same location. If another five squares of water are created, they will settle on the first five, filling the clay reservoir a little more:

......+.......
......|.....#.
.#..#.|.....#.
.#..#.|#......
.#..#.|#......
.#~~~~~#......
.#~~~~~#......
.#######......
..............
..............
....#.....#...
....#.....#...
....#.....#...
....#######...

Water pressure does not apply in this scenario. If another four squares of water are created, they will stay on the right side of the barrier, and no water will reach the left side:

......+.......
......|.....#.
.#..#.|.....#.
.#..#~~#......
.#..#~~#......
.#~~~~~#......
.#~~~~~#......
.#######......
..............
..............
....#.....#...
....#.....#...
....#.....#...
....#######...

At this point, the top reservoir overflows. While water can reach the tiles above the surface of the water, it cannot settle there, and so the next five squares of water settle like this:

......+.......
......|.....#.
.#..#||||...#.
.#..#~~#|.....
.#..#~~#|.....
.#~~~~~#|.....
.#~~~~~#|.....
.#######|.....
........|.....
........|.....
....#...|.#...
....#...|.#...
....#~~~~~#...
....#######...

Note especially the leftmost |: the new squares of water can reach this tile, but cannot stop there. Instead, eventually, they all fall to the right and settle in the reservoir below.

After 10 more squares of water, the bottom reservoir is also full:

......+.......
......|.....#.
.#..#||||...#.
.#..#~~#|.....
.#..#~~#|.....
.#~~~~~#|.....
.#~~~~~#|.....
.#######|.....
........|.....
........|.....
....#~~~~~#...
....#~~~~~#...
....#~~~~~#...
....#######...

Finally, while there is nowhere left for the water to settle, it can reach a few more tiles before overflowing beyond the bottom of the scanned data:

......+.......    (line not counted: above minimum y value)
......|.....#.
.#..#||||...#.
.#..#~~#|.....
.#..#~~#|.....
.#~~~~~#|.....
.#~~~~~#|.....
.#######|.....
........|.....
...|||||||||..
...|#~~~~~#|..
...|#~~~~~#|..
...|#~~~~~#|..
...|#######|..
...|.......|..    (line not counted: below maximum y value)
...|.......|..    (line not counted: below maximum y value)
...|.......|..    (line not counted: below maximum y value)

How many tiles can be reached by the water? To prevent counting forever, ignore tiles with a y coordinate smaller than the smallest y coordinate in your scan data or larger than the largest one. Any x coordinate is valid. In this example, the lowest y coordinate given is 1, and the highest is 13, causing the water spring (in row 0) and the water falling off the bottom of the render (in rows 14 through infinity) to be ignored.

So, in the example above, counting both water at rest (~) and other sand tiles the water can hypothetically reach (|), the total number of tiles the water can reach is 57.

How many tiles can the water reach within the range of y values in your scan?

--- Part Two ---

After a very long time, the water spring will run dry. How much water will be retained?

In the example above, water that won't eventually drain out is shown as ~, a total of 29 tiles.

How many water tiles are left after the water spring stops producing water and all remaining water not at rest has drained? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ClayVein(SmallRangeInclusive<IVec2>);

impl ClayVein {
    fn parse_range<'i>(input: &'i str) -> IResult<&'i str, SmallRangeInclusive<i32>> {
        map_opt(
            separated_pair(parse_integer, tag(".."), parse_integer),
            |(start, end)| (start < end).then(|| (start..=end).into()),
        )(input)
    }
}

impl Parse for ClayVein {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(
                tuple((tag("x="), parse_integer, tag(", y="), Self::parse_range)),
                |(_, x, _, y)| Self((IVec2::new(x, y.start)..=IVec2::new(x, y.end)).into()),
            ),
            map(
                tuple((tag("y="), parse_integer, tag(", x="), Self::parse_range)),
                |(_, y, _, x)| Self((IVec2::new(x.start, y)..=IVec2::new(x.end, y)).into()),
            ),
        ))(input)
    }
}

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Default, PartialEq)]
    enum ScanCell {
        #[default]
        Sand = SAND = b'.',
        Clay = CLAY = b'#',
        Spring = SPRING = b'+',
        WaterInSand = WATER_IN_SAND = b'|',
        WaterAtRestFromLeft = WATER_AT_REST_FROM_LEFT = b'>',
        WaterAtRestFromRight = WATER_AT_REST_FROM_RIGHT = b'<',
        WaterAtRest = WATER_AT_REST = b'~',
    }
}

impl ScanCell {
    fn becomes_water_in_sand(self) -> bool {
        self == Self::Sand
    }

    fn supports_water_at_rest(self) -> bool {
        matches!(self, Self::Clay | Self::WaterAtRest)
    }

    fn puts_left_neighbor_at_rest(self) -> bool {
        matches!(
            self,
            Self::Clay | Self::Spring | Self::WaterAtRestFromRight | Self::WaterAtRest
        )
    }

    fn puts_right_neighbor_at_rest(self) -> bool {
        matches!(
            self,
            Self::Clay | Self::Spring | Self::WaterAtRestFromLeft | Self::WaterAtRest
        )
    }

    fn accepts_left_neighbor_at_rest(self) -> bool {
        matches!(self, Self::WaterInSand | Self::WaterAtRestFromRight)
    }

    fn accepts_right_neighbor_at_rest(self) -> bool {
        matches!(self, Self::WaterInSand | Self::WaterAtRestFromLeft)
    }

    fn spreads_water(self) -> bool {
        matches!(self, Self::Spring | Self::WaterInSand)
    }

    fn is_water_reachable(self) -> bool {
        matches!(
            self,
            Self::WaterInSand
                | Self::WaterAtRestFromLeft
                | Self::WaterAtRestFromRight
                | Self::WaterAtRest
        )
    }
}

struct LeftAndRightResult {
    left_pos: IVec2,
    right_pos: IVec2,
    left: Option<ScanCell>,
    right: Option<ScanCell>,
}

enum SpreadWaterResult {
    NoScanCellDown,
    NoWaterAtRestSupportingDown {
        #[allow(dead_code)]
        down: ScanCell,
    },
    WaterAtRestSupportingDown {
        left_and_right_result: LeftAndRightResult,
        #[allow(dead_code)]
        down: ScanCell,
    },
}

struct Scan {
    grid: Grid2D<ScanCell>,
    #[allow(dead_code)]
    offset: IVec2,
    min_clay_y: i32,
    curr_moving_water: HashSet<IVec2>,
    next_moving_water: HashSet<IVec2>,
}

impl Scan {
    fn up(pos: IVec2) -> IVec2 {
        pos + Direction::UP.vec()
    }

    fn down(pos: IVec2) -> IVec2 {
        pos + Direction::DOWN.vec()
    }

    fn left(pos: IVec2) -> IVec2 {
        pos + Direction::LEFT.vec()
    }

    fn right(pos: IVec2) -> IVec2 {
        pos + Direction::RIGHT.vec()
    }

    fn left_and_right(&self, pos: IVec2) -> LeftAndRightResult {
        let left_pos: IVec2 = Self::left(pos);
        let right_pos: IVec2 = Self::right(pos);

        LeftAndRightResult {
            left_pos,
            right_pos,
            left: self.grid.get(left_pos).copied(),
            right: self.grid.get(right_pos).copied(),
        }
    }

    fn grid_string(&self) -> String {
        self.grid.clone().into()
    }

    fn scan_cells_vertically_within_clay_bounds(&self) -> &[ScanCell] {
        &self.grid.cells()[self.min_clay_y as usize * self.grid.dimensions().x as usize..]
    }

    fn count_water_reachable_tiles(&self) -> usize {
        self.scan_cells_vertically_within_clay_bounds()
            .iter()
            .filter(|scan_cell| scan_cell.is_water_reachable())
            .count()
    }

    fn count_water_at_rest(&self) -> usize {
        self.scan_cells_vertically_within_clay_bounds()
            .iter()
            .filter(|&&scan_cell| scan_cell == ScanCell::WaterAtRest)
            .count()
    }

    /// Returns what's at `pos` after this call.
    fn replace_sand_with_water_in_sand(&mut self, pos: IVec2) -> Option<ScanCell> {
        self.grid
            .get_mut(pos)
            .map(|scan_cell| {
                if scan_cell.becomes_water_in_sand() {
                    *scan_cell = ScanCell::WaterInSand;

                    (ScanCell::WaterInSand, true)
                } else {
                    (*scan_cell, false)
                }
            })
            .map(|(scan_cell, should_insert_pos)| {
                if should_insert_pos {
                    self.next_moving_water.insert(pos);
                }

                scan_cell
            })
    }

    fn spread_water(&mut self, pos: IVec2) -> SpreadWaterResult {
        self.replace_sand_with_water_in_sand(Self::down(pos))
            .map_or(SpreadWaterResult::NoScanCellDown, |down| {
                if down.supports_water_at_rest() {
                    let left_pos: IVec2 = Self::left(pos);
                    let right_pos: IVec2 = Self::right(pos);

                    SpreadWaterResult::WaterAtRestSupportingDown {
                        left_and_right_result: LeftAndRightResult {
                            left_pos,
                            right_pos,
                            left: self.replace_sand_with_water_in_sand(left_pos),
                            right: self.replace_sand_with_water_in_sand(right_pos),
                        },
                        down,
                    }
                } else {
                    SpreadWaterResult::NoWaterAtRestSupportingDown { down }
                }
            })
    }

    fn try_buffer_water_spread_above(&mut self, pos: IVec2) {
        let up: IVec2 = Self::up(pos);

        if self
            .grid
            .get(up)
            .copied()
            .map_or(false, ScanCell::spreads_water)
        {
            self.next_moving_water.insert(up);
        }
    }

    fn try_accept_left_neighbor_at_rest(&mut self, pos: IVec2, scan_cell: Option<ScanCell>) {
        if scan_cell.map_or(false, ScanCell::accepts_left_neighbor_at_rest)
            && self
                .grid
                .get(Self::down(pos))
                .copied()
                .map_or(false, ScanCell::supports_water_at_rest)
        {
            self.next_moving_water.insert(pos);
        }
    }

    fn try_accept_right_neighbor_at_rest(&mut self, pos: IVec2, scan_cell: Option<ScanCell>) {
        if scan_cell.map_or(false, ScanCell::accepts_right_neighbor_at_rest)
            && self
                .grid
                .get(Self::down(pos))
                .copied()
                .map_or(false, ScanCell::supports_water_at_rest)
        {
            self.next_moving_water.insert(pos);
        }
    }

    fn process_left_and_right_result(
        &mut self,
        &LeftAndRightResult {
            left_pos,
            right_pos,
            left,
            right,
        }: &LeftAndRightResult,
        pos: IVec2,
    ) {
        if let Some(water_at_rest) = match (
            left.map_or(false, ScanCell::puts_right_neighbor_at_rest),
            right.map_or(false, ScanCell::puts_left_neighbor_at_rest),
        ) {
            (true, true) => {
                self.try_buffer_water_spread_above(pos);
                self.try_accept_right_neighbor_at_rest(left_pos, left);
                self.try_accept_left_neighbor_at_rest(right_pos, right);

                Some(ScanCell::WaterAtRest)
            }
            (true, false) => {
                self.try_accept_left_neighbor_at_rest(right_pos, right);

                Some(ScanCell::WaterAtRestFromLeft)
            }
            (false, true) => {
                self.try_accept_right_neighbor_at_rest(left_pos, left);

                Some(ScanCell::WaterAtRestFromRight)
            }
            (false, false) => None,
        } {
            *self.grid.get_mut(pos).unwrap() = water_at_rest;
        }
    }

    fn update(&mut self) {
        self.next_moving_water.clear();

        let mut curr_moving_water: HashSet<IVec2> = take(&mut self.curr_moving_water);

        for pos in curr_moving_water.drain() {
            match *self.grid.get(pos).unwrap() {
                ScanCell::Spring => {
                    self.spread_water(pos);
                }
                ScanCell::WaterInSand => {
                    if let SpreadWaterResult::WaterAtRestSupportingDown {
                        left_and_right_result,
                        ..
                    } = self.spread_water(pos)
                    {
                        self.process_left_and_right_result(&left_and_right_result, pos);
                    }
                }
                ScanCell::WaterAtRestFromLeft | ScanCell::WaterAtRestFromRight => {
                    self.process_left_and_right_result(&self.left_and_right(pos), pos);
                }
                ScanCell::WaterAtRest => self.try_buffer_water_spread_above(pos),
                _ => unreachable!(),
            }
        }

        self.curr_moving_water = take(&mut self.next_moving_water);

        // Keep the allocation.
        self.next_moving_water = curr_moving_water;
    }

    fn run(&mut self) {
        while !self.curr_moving_water.is_empty() {
            self.update();
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<ClayVein>);

impl Solution {
    const SPRING_POS: IVec2 = IVec2::new(500_i32, 0_i32);

    fn min_and_max(&self) -> SmallRangeInclusive<IVec2> {
        let left: IVec2 = Direction::LEFT.vec();
        let right: IVec2 = Direction::RIGHT.vec();

        let (min, max): (IVec2, IVec2) =
            self.0
                .iter()
                .fold((IVec2::MAX, IVec2::MIN), |(min, max), clay_vein| {
                    (
                        min.min(clay_vein.0.start + left),
                        max.max(clay_vein.0.end + right),
                    )
                });

        (min..=max).into()
    }

    fn scan(&self) -> Scan {
        let min_and_max: SmallRangeInclusive<IVec2> = self.min_and_max();
        let min: IVec2 = min_and_max.start.min(Self::SPRING_POS);
        let max: IVec2 = min_and_max.end.max(Self::SPRING_POS);
        let dimensions: IVec2 = max - min + IVec2::ONE;
        let offset: IVec2 = -min;
        let min_clay_y: i32 = min_and_max.start.y + offset.y;

        let mut grid: Grid2D<ScanCell> = Grid2D::default(dimensions);

        for (pos, scan_cell) in self.0.iter().flat_map(|clay_vein| {
            CellIter2D::try_from(clay_vein.0.start..=clay_vein.0.end)
                .ok()
                .into_iter()
                .flatten()
                .chain((clay_vein.0.start == clay_vein.0.end).then_some(clay_vein.0.start))
                .map(|pos| (pos, ScanCell::Clay))
                .chain([(Self::SPRING_POS, ScanCell::Spring)])
        }) {
            *grid.get_mut(pos + offset).unwrap() = scan_cell;
        }

        let curr_moving_water: HashSet<IVec2> = [Self::SPRING_POS + offset].into_iter().collect();
        let next_moving_water: HashSet<IVec2> = HashSet::new();

        Scan {
            grid,
            offset,
            min_clay_y,
            curr_moving_water,
            next_moving_water,
        }
    }

    fn count_water_reachable_tiles(&self) -> usize {
        let mut scan: Scan = self.scan();

        scan.run();

        scan.count_water_reachable_tiles()
    }

    fn count_water_at_rest(&self) -> usize {
        let mut scan: Scan = self.scan();

        scan.run();

        scan.count_water_at_rest()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(separated_list0(line_ending, ClayVein::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Caught up for a bit on excluding water vertically between the spring and the tallest clay.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let mut scan: Scan = self.scan();

            scan.run();

            dbg!(scan.count_water_reachable_tiles());

            println!("{}", scan.grid_string());
        } else {
            dbg!(self.count_water_reachable_tiles());
        }
    }

    /// Apparently a lot of people on the subreddit used recursive functions for this, which
    /// confuses me a bit. This seems very clearly to just be an iterative sumulation problem.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_water_at_rest());
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
        x=495, y=2..7\n\
        y=7, x=495..501\n\
        x=501, y=3..7\n\
        x=498, y=2..4\n\
        x=506, y=1..2\n\
        x=498, y=10..13\n\
        x=504, y=10..13\n\
        y=13, x=498..504\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                ClayVein(((495_i32, 2_i32).into()..=(495_i32, 7_i32).into()).into()),
                ClayVein(((495_i32, 7_i32).into()..=(501_i32, 7_i32).into()).into()),
                ClayVein(((501_i32, 3_i32).into()..=(501_i32, 7_i32).into()).into()),
                ClayVein(((498_i32, 2_i32).into()..=(498_i32, 4_i32).into()).into()),
                ClayVein(((506_i32, 1_i32).into()..=(506_i32, 2_i32).into()).into()),
                ClayVein(((498_i32, 10_i32).into()..=(498_i32, 13_i32).into()).into()),
                ClayVein(((504_i32, 10_i32).into()..=(504_i32, 13_i32).into()).into()),
                ClayVein(((498_i32, 13_i32).into()..=(504_i32, 13_i32).into()).into()),
            ])]
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
    fn test_scan_update() {
        for (index, updates_and_grid_strings) in [vec![
            (
                0_usize,
                "\
                ......+.......\n\
                ............#.\n\
                .#..#.......#.\n\
                .#..#..#......\n\
                .#..#..#......\n\
                .#.....#......\n\
                .#.....#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                6_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#....|#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#...|<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#..|<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#.|<<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#|<<<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#~<<<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#~~<<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#~~~<<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#~~~~<#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                1_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#....|#......\n\
                .#~~~~~#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                9_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#.|#......\n\
                .#~~~~~#......\n\
                .#~~~~~#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                3_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#.|#......\n\
                .#..#~~#......\n\
                .#~~~~~#......\n\
                .#~~~~~#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                3_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#.|.....#.\n\
                .#..#~~#......\n\
                .#..#~~#......\n\
                .#~~~~~#......\n\
                .#~~~~~#......\n\
                .#######......\n\
                ..............\n\
                ..............\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#.....#...\n\
                ....#######...\n",
            ),
            (
                12_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#>>>|...#.\n\
                .#..#~~#|.....\n\
                .#..#~~#|.....\n\
                .#~~~~~#|.....\n\
                .#~~~~~#|.....\n\
                .#######|.....\n\
                ........|.....\n\
                ........|.....\n\
                ....#...|.#...\n\
                ....#...|.#...\n\
                ....#...|.#...\n\
                ....#######...\n",
            ),
            (
                8_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#>>>|...#.\n\
                .#..#~~#|.....\n\
                .#..#~~#|.....\n\
                .#~~~~~#|.....\n\
                .#~~~~~#|.....\n\
                .#######|.....\n\
                ........|.....\n\
                ........|.....\n\
                ....#...|.#...\n\
                ....#..|||#...\n\
                ....#~~~~~#...\n\
                ....#######...\n",
            ),
            (
                7_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#>>>|...#.\n\
                .#..#~~#|.....\n\
                .#..#~~#|.....\n\
                .#~~~~~#|.....\n\
                .#~~~~~#|.....\n\
                .#######|.....\n\
                ........|.....\n\
                ........|.....\n\
                ....#..|||#...\n\
                ....#~~~~~#...\n\
                ....#~~~~~#...\n\
                ....#######...\n",
            ),
            (
                7_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#>>>|...#.\n\
                .#..#~~#|.....\n\
                .#..#~~#|.....\n\
                .#~~~~~#|.....\n\
                .#~~~~~#|.....\n\
                .#######|.....\n\
                ........|.....\n\
                .......|||....\n\
                ....#~~~~~#...\n\
                ....#~~~~~#...\n\
                ....#~~~~~#...\n\
                ....#######...\n",
            ),
            (
                8_usize,
                "\
                ......+.......\n\
                ......|.....#.\n\
                .#..#>>>|...#.\n\
                .#..#~~#|.....\n\
                .#..#~~#|.....\n\
                .#~~~~~#|.....\n\
                .#~~~~~#|.....\n\
                .#######|.....\n\
                ........|.....\n\
                ...|||||||||..\n\
                ...|#~~~~~#|..\n\
                ...|#~~~~~#|..\n\
                ...|#~~~~~#|..\n\
                ...|#######|..\n",
            ),
        ]]
        .into_iter()
        .enumerate()
        {
            let mut scan: Scan = solution(index).scan();

            for (updates, grid_string) in updates_and_grid_strings {
                for _ in 0_usize..updates {
                    scan.update();
                }

                assert_eq!(scan.grid_string(), grid_string);
            }
        }
    }

    #[test]
    fn test_run() {
        for (index, grid_string) in ["\
            ......+.......\n\
            ......|.....#.\n\
            .#..#>>>|...#.\n\
            .#..#~~#|.....\n\
            .#..#~~#|.....\n\
            .#~~~~~#|.....\n\
            .#~~~~~#|.....\n\
            .#######|.....\n\
            ........|.....\n\
            ...|||||||||..\n\
            ...|#~~~~~#|..\n\
            ...|#~~~~~#|..\n\
            ...|#~~~~~#|..\n\
            ...|#######|..\n"]
        .into_iter()
        .enumerate()
        {
            let mut scan: Scan = solution(index).scan();

            scan.run();

            assert_eq!(scan.grid_string(), grid_string);
        }
    }

    #[test]
    fn test_count_water_reachable_tiles() {
        for (index, water_reachable_tiles_count) in [57_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_water_reachable_tiles(),
                water_reachable_tiles_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
