use {
    crate::*,
    bitvec::prelude::*,
    nom::{combinator::map, error::Error, multi::many0_count, Err, IResult},
    std::ops::Range,
};

/* --- Day 18: Like a Rogue ---

As you enter this room, you hear a loud click! Some of the tiles in the floor here seem to be pressure plates for traps, and the trap you just triggered has run out of... whatever it tried to do to you. You doubt you'll be so lucky next time.

Upon closer examination, the traps and safe tiles in this room seem to follow a pattern. The tiles are arranged into rows that are all the same width; you take note of the safe tiles (.) and traps (^) in the first row (your puzzle input).

The type of tile (trapped or safe) in each row is based on the types of the tiles in the same position, and to either side of that position, in the previous row. (If either side is off either end of the row, it counts as "safe" because there isn't a trap embedded in the wall.)

For example, suppose you know the first row (with tiles marked by letters) and want to determine the next row (with tiles marked by numbers):

ABCDE
12345

The type of tile 2 is based on the types of tiles A, B, and C; the type of tile 5 is based on tiles D, E, and an imaginary "safe" tile. Let's call these three tiles from the previous row the left, center, and right tiles, respectively. Then, a new tile is a trap only in one of the following situations:

    Its left and center tiles are traps, but its right tile is not.
    Its center and right tiles are traps, but its left tile is not.
    Only its left tile is a trap.
    Only its right tile is a trap.

In any other situation, the new tile is safe.

Then, starting with the row ..^^., you can determine the next row by applying those rules to each new tile:

    The leftmost character on the next row considers the left (nonexistent, so we assume "safe"), center (the first ., which means "safe"), and right (the second ., also "safe") tiles on the previous row. Because all of the trap rules require a trap in at least one of the previous three tiles, the first tile on this new row is also safe, ..
    The second character on the next row considers its left (.), center (.), and right (^) tiles from the previous row. This matches the fourth rule: only the right tile is a trap. Therefore, the next tile in this new row is a trap, ^.
    The third character considers .^^, which matches the second trap rule: its center and right tiles are traps, but its left tile is not. Therefore, this tile is also a trap, ^.
    The last two characters in this new row match the first and third rules, respectively, and so they are both also traps, ^.

After these steps, we now know the next row of tiles in the room: .^^^^. Then, we continue on to the next row, using the same rules, and get ^^..^. After determining two new rows, our map looks like this:

..^^.
.^^^^
^^..^

Here's a larger example with ten tiles per row and ten rows:

.^^.^.^^^^
^^^...^..^
^.^^.^.^^.
..^^...^^^
.^^^^.^^.^
^^..^.^^..
^^^^..^^^.
^..^^^^.^^
.^^^..^.^^
^^.^^^..^^

In ten rows, this larger example has 38 safe tiles.

Starting with the map in your puzzle input, in a total of 40 rows (including the starting row), how many safe tiles are there?

--- Part Two ---

How many safe tiles are there in a total of 400000 rows? */

define_cell! {
    #[repr(u8)]
    #[derive(PartialEq)]
    enum Tile {
        Safe = SAFE = b'.',
        Trap = TRAP = b'^',
    }
}

impl From<Tile> for bool {
    fn from(value: Tile) -> Self {
        value == Tile::Trap
    }
}

impl From<bool> for Tile {
    fn from(value: bool) -> Self {
        if value {
            Self::Trap
        } else {
            Self::Safe
        }
    }
}

struct TileGrid {
    bitvec: BitVec,
    padded_width: usize,
}

impl TileGrid {
    const IS_TRAP: [u8; 1_usize] = Self::is_trap();

    const fn is_trap() -> [u8; 1_usize] {
        [(1_u8 << 0b011_u32) | (1_u8 << 0b110_u32) | (1_u8 << 0b001_u32) | (1_u8 << 0b100_u32)]
    }

    fn unpadded_row_range(&self) -> Range<usize> {
        1_usize..self.padded_width - 1_usize
    }

    fn bitslice_starting_at_row(&self, row: usize) -> &BitSlice {
        &self.bitvec[self.padded_width * row..]
    }

    fn iter_rows_starting_at_row(&self, row: usize) -> impl Iterator<Item = &BitSlice> {
        let unpadded_row_range: Range<usize> = self.unpadded_row_range();

        self.bitslice_starting_at_row(row)
            .chunks_exact(self.padded_width)
            .map(move |chunk| &chunk[unpadded_row_range.clone()])
    }

    fn iter_rows(&self) -> impl Iterator<Item = &BitSlice> {
        self.iter_rows_starting_at_row(0_usize)
    }

    fn safe_tile_count(&self) -> usize {
        self.iter_rows().flat_map(BitSlice::iter_zeros).count()
    }

    fn as_grid2d(&self) -> Grid2D<Tile> {
        Grid2D::try_from_cells_and_width(
            self.iter_rows()
                .flat_map(|row| row.iter().by_vals())
                .map(Tile::from)
                .collect(),
            self.padded_width - 2_usize,
        )
        .unwrap()
    }

    fn as_string(&self) -> String {
        self.as_grid2d().into()
    }

    fn grow_to_height(&mut self, height: usize) {
        let mut next_row: BitVec = BitVec::with_capacity(self.padded_width);

        let final_bitvec_len: usize = height * self.padded_width;

        while self.bitvec.len() < final_bitvec_len {
            let last_row: &BitSlice = &self.bitvec[self.bitvec.len() - self.padded_width..];

            next_row.extend(
                last_row
                    .windows(3_usize)
                    .map(|window| Self::IS_TRAP.as_bits::<Lsb0>()[window.load::<usize>()]),
            );

            self.bitvec.push(false);
            self.bitvec.extend(next_row.drain(..));
            self.bitvec.push(false);
        }
    }
}

struct SafeTileCountCruncher {
    row: u128,
    mask: u128,
    width: usize,
    height: usize,
    safe_tile_count: usize,
}

impl SafeTileCountCruncher {
    fn try_new(bitslice: &BitSlice) -> Option<Self> {
        (bitslice.len() <= u128::BITS as usize).then(|| {
            let width: usize = bitslice.len() - 2_usize;

            let mut safe_tile_count_cruncher: Self = Self {
                row: bitslice.load(),
                mask: (1_u128 << (width + 1_usize)) - 2_u128,
                width,
                height: 1_usize,
                safe_tile_count: 0_usize,
            };

            safe_tile_count_cruncher.update_safe_tile_count();

            safe_tile_count_cruncher
        })
    }

    fn grow_to_height(&mut self, height: usize) {
        while self.height < height {
            self.row = ((self.row >> 1_u32) ^ (self.row << 1_u32)) & self.mask;
            self.update_safe_tile_count();
            self.height += 1_usize;
        }
    }

    fn update_safe_tile_count(&mut self) {
        self.safe_tile_count += self.width - self.row.count_ones() as usize;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(BitVec);

impl Solution {
    const HEIGHT: usize = 40_usize;
    const LARGE_HEIGHT: usize = 400000_usize;

    fn as_tile_grid(&self) -> TileGrid {
        TileGrid {
            bitvec: self.0.clone(),
            padded_width: self.0.len(),
        }
    }

    fn safe_tile_count_for_height(&self, height: usize) -> usize {
        let mut tile_grid: TileGrid = self.as_tile_grid();

        tile_grid.grow_to_height(height);

        tile_grid.safe_tile_count()
    }

    fn try_safe_tile_count_for_large_height(&self, large_height: usize) -> Option<usize> {
        SafeTileCountCruncher::try_new(&self.0).map(|mut safe_tile_count_cruncher| {
            safe_tile_count_cruncher.grow_to_height(large_height);

            safe_tile_count_cruncher.safe_tile_count
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut bitvec: BitVec = BitVec::new();

        bitvec.push(false);

        let input: &str = many0_count(map(Tile::parse, |tile| {
            bitvec.push(tile.into());
        }))(input)?
        .0;

        bitvec.push(false);

        Ok((input, Solution(bitvec)))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let mut tile_grid: TileGrid = self.as_tile_grid();

            tile_grid.grow_to_height(Self::HEIGHT);

            dbg!(tile_grid.safe_tile_count());
            println!("{}", tile_grid.as_string());
        } else {
            dbg!(self.safe_tile_count_for_height(Self::HEIGHT));
        }
    }

    // Not 15618906, too low
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_safe_tile_count_for_large_height(Self::LARGE_HEIGHT));
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

    const SOLUTION_STRS: &'static [&'static str] = &["..^^.\n", ".^^.^.^^^^\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(
                    [false, false, false, true, true, false, false]
                        .into_iter()
                        .collect(),
                ),
                Solution(
                    [
                        false, false, true, true, false, true, false, true, true, true, true, false,
                    ]
                    .into_iter()
                    .collect(),
                ),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_as_string() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(solution(index).as_tile_grid().as_string(), solution_str);
        }
    }

    #[test]
    fn test_grow_to_height() {
        for (index, (height, string)) in [
            (
                3_usize,
                "\
                ..^^.\n\
                .^^^^\n\
                ^^..^\n",
            ),
            (
                10_usize,
                "\
                .^^.^.^^^^\n\
                ^^^...^..^\n\
                ^.^^.^.^^.\n\
                ..^^...^^^\n\
                .^^^^.^^.^\n\
                ^^..^.^^..\n\
                ^^^^..^^^.\n\
                ^..^^^^.^^\n\
                .^^^..^.^^\n\
                ^^.^^^..^^\n",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mut tile_grid: TileGrid = solution(index).as_tile_grid();

            tile_grid.grow_to_height(height);

            assert_eq!(tile_grid.as_string(), string);
        }
    }

    #[test]
    fn test_safe_tile_count_for_height() {
        for (index, (height, safe_tile_count)) in [(3_usize, 6_usize), (10_usize, 38_usize)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).safe_tile_count_for_height(height),
                safe_tile_count
            );
        }
    }

    #[test]
    fn test_try_safe_tile_count_for_large_height() {
        for (index, (height, safe_tile_count)) in [(3_usize, 6_usize), (10_usize, 38_usize)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).try_safe_tile_count_for_large_height(height),
                Some(safe_tile_count)
            );
        }
    }
}
