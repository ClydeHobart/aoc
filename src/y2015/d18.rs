use {
    crate::*,
    arrayvec::ArrayVec,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        character::complete::line_ending,
        combinator::{map, map_opt, success, verify},
        error::Error,
        Err, IResult,
    },
    std::{marker::PhantomData, mem::swap, ops::Range},
};

/* --- Day 18: Like a GIF For Your Yard ---

After the million lights incident, the fire code has gotten stricter: now, at most ten thousand lights are allowed. You arrange them in a 100x100 grid.

Never one to let you down, Santa again mails you instructions on the ideal lighting configuration. With so few lights, he says, you'll have to resort to animation.

Start by setting your lights to the included initial configuration (your puzzle input). A # means "on", and a . means "off".

Then, animate your grid in steps, where each step decides the next configuration based on the current one. Each light's next state (either on or off) depends on its current state and the current states of the eight lights adjacent to it (including diagonals). Lights on the edge of the grid might have fewer than eight neighbors; the missing ones always count as "off".

For example, in a simplified 6x6 grid, the light marked A has the neighbors numbered 1 through 8, and the light marked B, which is on an edge, only has the neighbors marked 1 through 5:

1B5...
234...
......
..123.
..8A4.
..765.

The state a light should have next is based on its current state (on or off) plus the number of neighbors that are on:

    A light which is on stays on when 2 or 3 neighbors are on, and turns off otherwise.
    A light which is off turns on if exactly 3 neighbors are on, and stays off otherwise.

All of the lights update simultaneously; they all consider the same current state before moving to the next.

Here's a few steps from an example configuration of another 6x6 grid:

Initial state:
.#.#.#
...##.
#....#
..#...
#.#..#
####..

After 1 step:
..##..
..##.#
...##.
......
#.....
#.##..

After 2 steps:
..###.
......
..###.
......
.#....
.#....

After 3 steps:
...#..
......
...#..
..##..
......
......

After 4 steps:
......
......
..##..
..##..
......
......

After 4 steps, this example has four lights on.

In your grid of 100x100 lights, given your initial configuration, how many lights are on after 100 steps?

--- Part Two ---

You flip the instructions over; Santa goes on to point out that this is all just an implementation of Conway's Game of Life. At least, it was, until you notice that something's wrong with the grid of lights you bought: four lights, one in each corner, are stuck on and can't be turned off. The example above will actually run like this:

Initial state:
##.#.#
...##.
#....#
..#...
#.#..#
####.#

After 1 step:
#.##.#
####.#
...##.
......
#...#.
#.####

After 2 steps:
#..#.#
#....#
.#.##.
...##.
.#..##
##.###

After 3 steps:
#...##
####.#
..##.#
......
##....
####.#

After 4 steps:
#.####
#....#
...#..
.##...
#.....
#.#..#

After 5 steps:
##.###
.##..#
.##...
.##...
#.#...
##...#

After 5 steps, this example now has 17 lights on.

In your grid of 100x100 lights, given your initial configuration, but with the four corners always in the on state, how many lights are on after 100 steps? */

type BitRow = BitArr!(for BoundBitGrid::MAX_PADDED_SIDE_LEN);
type BitGrid = ArrayVec<BitRow, { BoundBitGrid::MAX_PADDED_SIDE_LEN }>;

#[derive(Clone, Copy)]
struct CellState {
    is_alive: bool,
    alive_neighbors: u8,
}

impl CellState {
    const WILL_BE_ALIVE: [BitArray<[u16; 1_usize]>; 2_usize] = [
        BitArray {
            _ord: PhantomData,
            data: [0b_0_00001000_u16],
        },
        BitArray {
            _ord: PhantomData,
            data: [0b_0_00001100_u16],
        },
    ];

    fn will_be_alive(self) -> bool {
        Self::WILL_BE_ALIVE[self.is_alive as usize][self.alive_neighbors as usize]
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct BoundBitGrid {
    bit_grid: BitGrid,
    dimensions: IVec2,
}

impl BoundBitGrid {
    const MAX_SIDE_LEN: usize = 100_usize;
    const MAX_PADDED_SIDE_LEN: usize = Self::MAX_SIDE_LEN + 2_usize;
    const CELL_INFLUENCE_SIDE_LEN: usize = 3_usize;
    const MIDDLE_ROW_MASK: u32 = 0b010_u32;

    fn cell_influence_range_1d(coordinate: i32) -> Range<usize> {
        let start: usize = coordinate as usize;

        start..start + Self::CELL_INFLUENCE_SIDE_LEN
    }

    fn parse_bit_row_and_len<'i>(input: &'i str) -> IResult<&'i str, (BitRow, i32)> {
        map_opt(
            parse_array_vec::<{ Self::MAX_SIDE_LEN }, Pixel, _>(Pixel::parse),
            |pixels| {
                (!pixels.is_empty()).then(|| {
                    let bit_row_len: i32 = pixels.len() as i32;
                    let mut bit_row: BitRow = BitRow::ZERO;

                    for (pixel, bit_ref) in pixels.into_iter().zip(&mut bit_row[1_usize..]) {
                        bit_ref.commit(pixel.is_light())
                    }

                    (bit_row, bit_row_len)
                })
            },
        )(input)
    }

    fn string(&self) -> String {
        let mut grid: Grid2D<Pixel> = Grid2D::default(self.dimensions);

        for (bit, pixel) in self
            .bit_grid
            .iter()
            .skip(1_usize)
            .flat_map(|bit_row| {
                bit_row
                    .iter()
                    .by_vals()
                    .skip(1_usize)
                    .take(self.dimensions.x as usize)
            })
            .zip(grid.cells_mut())
        {
            *pixel = bit.into();
        }

        grid.into()
    }

    fn cell_state(&self, pos: IVec2) -> CellState {
        assert!(pos.cmpge(IVec2::ZERO).all());
        assert!(pos.cmplt(self.dimensions).all());

        let row_range: Range<usize> = Self::cell_influence_range_1d(pos.y);
        let col_range: Range<usize> = Self::cell_influence_range_1d(pos.x);
        let [top_row, middle_row, bottom_row] = self.bit_grid[row_range]
            .iter()
            .map(|row| row[col_range.clone()].load_le::<u32>())
            .collect::<ArrayVec<u32, { Self::CELL_INFLUENCE_SIDE_LEN }>>()
            .into_inner()
            .unwrap();
        let is_alive: bool = (middle_row & Self::MIDDLE_ROW_MASK) != 0_u32;
        let alive_neighbors: u8 = (top_row.count_ones()
            + (middle_row & !Self::MIDDLE_ROW_MASK).count_ones()
            + bottom_row.count_ones()) as u8;

        CellState {
            is_alive,
            alive_neighbors,
        }
    }

    fn next<F: Fn(&mut Self)>(&self, next: &mut Self, on_finish: F) {
        next.bit_grid.fill(BitRow::ZERO);

        let next_rows: &mut [BitRow] = &mut next.bit_grid[1_usize..];

        let rows_len: usize = self.dimensions.y as usize;
        let cols_len: usize = self.dimensions.x as usize;

        for y in 0_usize..rows_len {
            let next_row: &mut BitSlice = &mut next_rows[y][1_usize..];

            for x in 0_usize..cols_len {
                next_row.set(
                    x,
                    self.cell_state((x as i32, y as i32).into()).will_be_alive(),
                );
            }
        }

        on_finish(next);
    }

    fn step<F: Fn(&mut Self)>(&self, steps: usize, on_finish: F) -> Self {
        let mut a: Self = self.clone();
        let mut b: Self = self.clone();
        let mut curr: &mut Self = &mut a;
        let mut next: &mut Self = &mut b;

        for _ in 0_usize..steps {
            curr.next(next, |bound_bit_grid| on_finish(bound_bit_grid));
            swap(&mut curr, &mut next);
        }

        curr.clone()
    }

    fn count_lights(&self) -> usize {
        self.bit_grid
            .iter()
            .map(|bit_row| bit_row.count_ones())
            .sum()
    }

    fn set_four_corners(&mut self) {
        // This is in absolute terms, taking padding into consideration.
        let min_pos: IVec2 = IVec2::ONE;
        let max_pos: IVec2 = self.dimensions;

        for pos in [
            min_pos,
            (max_pos.x, min_pos.y).into(),
            (min_pos.x, max_pos.y).into(),
            max_pos,
        ] {
            self.bit_grid[pos.y as usize].set(pos.x as usize, true);
        }
    }
}

impl Parse for BoundBitGrid {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (output, bit_rows_and_lens): (&str, ArrayVec<(BitRow, i32), { Self::MAX_SIDE_LEN }>) =
            parse_separated_array_vec(Self::parse_bit_row_and_len, line_ending)(input)?;

        let mut bit_grid: BitGrid = BitGrid::new();

        bit_grid.push(BitRow::ZERO);

        let dimensions: Option<Option<IVec2>> = bit_rows_and_lens.into_iter().try_fold(
            None,
            |dimensions: Option<IVec2>, (bit_row, len)| {
                bit_grid.push(bit_row);

                match dimensions {
                    Some(dimensions) => {
                        (dimensions.x == len).then_some(Some((len, dimensions.y + 1_i32).into()))
                    }
                    None => Some(Some((len, 1_i32).into())),
                }
            },
        );

        verify(success(()), |_| dimensions.is_some())(input)?;

        let dimensions: IVec2 = dimensions.unwrap().unwrap_or_default();

        bit_grid.push(BitRow::ZERO);

        Ok((
            output,
            Self {
                bit_grid,
                dimensions,
            },
        ))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(BoundBitGrid);

impl Solution {
    const STEPS: usize = 100_usize;

    fn step(&self, steps: usize) -> BoundBitGrid {
        self.0.step(steps, |_| ())
    }

    fn count_lights_after_steps(&self, steps: usize) -> usize {
        self.step(steps).count_lights()
    }

    fn step_with_four_corners(&self, steps: usize) -> BoundBitGrid {
        let mut bound_bit_grid: BoundBitGrid = self.0.clone();

        bound_bit_grid.set_four_corners();

        bound_bit_grid.step(steps, BoundBitGrid::set_four_corners)
    }

    fn count_lights_after_steps_with_four_corners(&self, steps: usize) -> usize {
        self.step_with_four_corners(steps).count_lights()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(BoundBitGrid::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I thought I'd done GOL before...
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let bound_bit_grid: BoundBitGrid = self.step(Self::STEPS);

            dbg!(bound_bit_grid.count_lights());
            println!("{}", bound_bit_grid.string());
        } else {
            dbg!(self.count_lights_after_steps(Self::STEPS));
        }
    }

    /// Not much to say here, it's basically just "stick your fingers in the pie a bit".
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let bound_bit_grid: BoundBitGrid = self.step_with_four_corners(Self::STEPS);

            dbg!(bound_bit_grid.count_lights());
            println!("{}", bound_bit_grid.string());
        } else {
            dbg!(self.count_lights_after_steps_with_four_corners(Self::STEPS));
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        .#.#.#\n\
        ...##.\n\
        #....#\n\
        ..#...\n\
        #.#..#\n\
        ####..\n"];
    const STEPS: &'static [&'static [&'static str]] = &[&[
        "\
        ..##..\n\
        ..##.#\n\
        ...##.\n\
        ......\n\
        #.....\n\
        #.##..\n",
        "\
        ..###.\n\
        ......\n\
        ..###.\n\
        ......\n\
        .#....\n\
        .#....\n",
        "\
        ...#..\n\
        ......\n\
        ...#..\n\
        ..##..\n\
        ......\n\
        ......\n",
        "\
        ......\n\
        ......\n\
        ..##..\n\
        ..##..\n\
        ......\n\
        ......\n",
    ]];
    const STEPS_WITH_FOUR_CORNERS: &'static [&'static [&'static str]] = &[&[
        "\
        ##.#.#\n\
        ...##.\n\
        #....#\n\
        ..#...\n\
        #.#..#\n\
        ####.#\n",
        "\
        #.##.#\n\
        ####.#\n\
        ...##.\n\
        ......\n\
        #...#.\n\
        #.####\n",
        "\
        #..#.#\n\
        #....#\n\
        .#.##.\n\
        ...##.\n\
        .#..##\n\
        ##.###\n",
        "\
        #...##\n\
        ####.#\n\
        ..##.#\n\
        ......\n\
        ##....\n\
        ####.#\n",
        "\
        #.####\n\
        #....#\n\
        ...#..\n\
        .##...\n\
        #.....\n\
        #.#..#\n",
        "\
        ##.###\n\
        .##..#\n\
        .##...\n\
        .##...\n\
        #.#...\n\
        ##...#\n",
    ]];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(BoundBitGrid {
                bit_grid: [
                    BitRow::ZERO,
                    bitarr_typed!(BitRow; 0, 0, 1, 0, 1, 0, 1),
                    bitarr_typed!(BitRow; 0, 0, 0, 0, 1, 1, 0),
                    bitarr_typed!(BitRow; 0, 1, 0, 0, 0, 0, 1),
                    bitarr_typed!(BitRow; 0, 0, 0, 1, 0, 0, 0),
                    bitarr_typed!(BitRow; 0, 1, 0, 1, 0, 0, 1),
                    bitarr_typed!(BitRow; 0, 1, 1, 1, 1, 0, 0),
                    BitRow::ZERO,
                ]
                .into_iter()
                .collect(),
                dimensions: (6_i32, 6_i32).into(),
            })]
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
    fn test_string() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(solution(index).0.string(), solution_str);
        }
    }

    #[test]
    fn test_next() {
        for (index, steps) in STEPS.iter().copied().enumerate() {
            let mut curr: BoundBitGrid = solution(index).0.clone();
            let mut next: BoundBitGrid = curr.clone();

            for step in steps.iter().copied() {
                curr.next(&mut next, |_| ());
                curr = next.clone();

                assert_eq!(next.string(), step);
            }
        }
    }

    #[test]
    fn test_steps() {
        for (solution_index, steps) in STEPS.iter().copied().enumerate() {
            let grid: &BoundBitGrid = &solution(solution_index).0;

            for (step_index, step) in steps.iter().copied().enumerate() {
                assert_eq!(grid.step(step_index + 1_usize, |_| ()).string(), step);
            }
        }
    }

    #[test]
    fn test_set_four_corners() {
        for (index, steps_with_four_corners) in STEPS_WITH_FOUR_CORNERS.iter().copied().enumerate()
        {
            let mut bound_bit_grid: BoundBitGrid = solution(index).0.clone();

            bound_bit_grid.set_four_corners();

            assert_eq!(bound_bit_grid.string(), steps_with_four_corners[0_usize]);
        }
    }

    #[test]
    fn test_step_with_four_corners() {
        for (solution_index, steps_with_four_corners) in
            STEPS_WITH_FOUR_CORNERS.iter().copied().enumerate()
        {
            let solution: &Solution = solution(solution_index);

            for (step_index, step) in steps_with_four_corners.iter().copied().enumerate() {
                assert_eq!(solution.step_with_four_corners(step_index).string(), step);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
