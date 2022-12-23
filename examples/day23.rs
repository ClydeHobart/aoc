use {
    aoc_2022::*,
    glam::IVec2,
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        iter::Peekable,
        mem::{size_of, transmute},
        ops::Range,
        str::Split,
    },
    strum::EnumCount,
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate static_assertions;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
enum ElfCell {
    #[default]
    Empty = b'.',
    Elf = b'#',
}

impl ElfCell {
    const EMPTY_U8: u8 = Self::Empty as u8;
    const ELF_U8: u8 = Self::Elf as u8;
}

#[derive(Clone)]
struct IntermediateElfGrid(Grid2D<ElfCell>);

impl From<IntermediateElfGrid> for Grid2DString {
    fn from(intermediate: IntermediateElfGrid) -> Self {
        const_assert_eq!(size_of::<ElfCell>(), size_of::<u8>());
        const_assert_eq!(size_of::<IntermediateElfGrid>(), size_of::<Grid2DString>());

        // SAFETY: Currently, both `IntermediateElfGrid` and `Grid2DString` are just new-type
        // pattern structs around `Grid2D` structs of 1-Byte-sized elements. The const asserts above
        // will hopefully catch any issues, should that not be the case at some point
        unsafe { transmute(intermediate) }
    }
}

type Proposals = HashMap<IVec2, IVec2>;
type Blocks = [u64; 3_usize];

#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    masks: Blocks,
    dir: Direction,
}

type Candidates = [Candidate; Direction::COUNT];

struct Masks {
    blocks: Blocks,
    prev_or_next: u64,
    bit: u64,
    neighbors: u64,
    candidates: Candidates,
}

impl Masks {
    fn shift(&mut self, bits: u32) {
        self.bit <<= bits;
        self.neighbors <<= bits;

        for candidate in self.candidates.iter_mut() {
            for mask in candidate.masks.iter_mut() {
                *mask <<= bits;
            }
        }
    }

    #[inline(always)]
    fn is_bit_set(&self) -> bool {
        self.blocks[1_usize] & self.bit != 0_u64
    }

    #[inline(always)]
    fn are_any_neighbors_set(&self) -> bool {
        (self.prev_or_next | (self.blocks[1_usize] & !self.bit)) & self.neighbors != 0_u64
    }

    fn try_find_proposal(&self) -> Option<Direction> {
        self.candidates
            .iter()
            .find(|candidate| {
                self.blocks
                    .iter()
                    .zip(candidate.masks.iter())
                    .all(|(block, candidate_mask)| *block & *candidate_mask == 0_u64)
            })
            .map(|candidate| candidate.dir)
    }
}

/// A representation of the Elves in 2D space
///
/// To make checking with the masks easier, the outer bits of the `u64`s overlap between adjacent
/// cells. Consider the following scenario for a given row consisting of two `u64`s (the numbers
/// indicate the x position that that bit represents):
///
/// `[-1, 0, 1, 2, ..., 59, 60, 61, 62], [61, 62, 63, 64, ..., 121, 122, 123, 124]`
#[derive(Clone, PartialEq)]
struct ElfGrid {
    grid: Grid2D<u64>,
    min: IVec2,
    max: IVec2,
    candidates: Candidates,
}

impl ElfGrid {
    const NEIGHBORS: u64 = 0b111_u64;
    const NORTH_CANDIDATE: Candidate = Candidate {
        masks: [0b111_u64, 0_u64, 0_u64],
        dir: Direction::North,
    };
    const SOUTH_CANDIDATE: Candidate = Candidate {
        masks: [0_u64, 0_u64, 0b111_u64],
        dir: Direction::South,
    };
    const WEST_CANDIDATE: Candidate = Candidate {
        masks: [0b1_u64, 0b1_u64, 0b1_u64],
        dir: Direction::West,
    };
    const EAST_CANDIDATE: Candidate = Candidate {
        masks: [0b100_u64, 0b100_u64, 0b100_u64],
        dir: Direction::East,
    };
    const CANDIDATES: [Candidate; Direction::COUNT] = [
        Self::NORTH_CANDIDATE,
        Self::SOUTH_CANDIDATE,
        Self::WEST_CANDIDATE,
        Self::EAST_CANDIDATE,
    ];
    const CELL_BITS_U32: u32 = u64::BITS - 2_u32;
    const CELL_BITS_MINUS_1: i32 = Self::CELL_BITS_I32 - 1_i32;
    const CELL_BITS_I32: i32 = Self::CELL_BITS_U32 as i32;
    const INVALID_POS: IVec2 = IVec2::new(i32::MIN, i32::MIN);

    fn from_intermediate(intermediate: &IntermediateElfGrid) -> Self {
        // Compute the intermediate min and max. These are likely just `IVec2::ZERO` and
        // `intermediate.0.max_dimensions()`, but there's no imposed restriction for that
        let (im_min, im_max): (IVec2, IVec2) =
            CellIter2D::until_boundary(&intermediate.0, IVec2::ZERO, Direction::South)
                .map(|row_iter| {
                    CellIter2D::until_boundary(&intermediate.0, row_iter, Direction::East)
                })
                .flatten()
                .fold(
                    (i32::MAX * IVec2::ONE, i32::MIN * IVec2::ONE),
                    |(min, max), pos| {
                        if *intermediate.0.get(pos).unwrap() == ElfCell::Elf {
                            (min.min(pos), max.max(pos))
                        } else {
                            (min, max)
                        }
                    },
                );

        // The intermediate dimensions, scaled up to allow space for the elves to explore
        let im_dimensions: IVec2 = 3_i32 * (im_max - im_min + IVec2::ONE);

        // Amount to add to a position in the intermediate space to translate it to the new space.
        // This is half the `intermediate_dimensions` minus the average of `old_min` and `old_max`,
        // which simplifies to the following:
        let offset: IVec2 = (im_dimensions - im_max - im_min) / 2_i32;

        let mut dimensions: IVec2 = im_dimensions;

        // Update the x to be in terms of the minimum number of `u64`s needed to hold the previous
        // `dimensions.x` value count of bits
        dimensions.x = (dimensions.x - 1_i32) / Self::CELL_BITS_I32 + 1_i32;

        let grid: Grid2D<u64> = Grid2D::default(dimensions);
        let min: IVec2 = im_min + offset;
        let max: IVec2 = im_max + offset;
        let candidates: Candidates = Self::CANDIDATES;

        let mut elf_grid: Self = Self {
            grid,
            min,
            max,
            candidates,
        };

        for pos in CellIter2D::until_boundary(&intermediate.0, IVec2::ZERO, Direction::South)
            .map(|row_iter| CellIter2D::until_boundary(&intermediate.0, row_iter, Direction::East))
            .flatten()
        {
            if *intermediate.0.get(pos).unwrap() == ElfCell::Elf {
                elf_grid.set_bit_for_pos(pos + offset);
            }
        }

        elf_grid
    }

    fn to_intermediate(&self) -> IntermediateElfGrid {
        let im_dimensions: IVec2 = self.max - self.min + IVec2::ONE;

        let mut intermediate: IntermediateElfGrid =
            IntermediateElfGrid(Grid2D::default(im_dimensions));

        for elf_pos in self.iter_elves() {
            *intermediate.0.get_mut(elf_pos - self.min).unwrap() = ElfCell::Elf;
        }

        intermediate
    }

    fn iter_elves(&self) -> impl Iterator<Item = IVec2> + '_ {
        (self.min.y..=self.max.y)
            .map(|y| self.x_block_range().map(move |block_x| (block_x, y)))
            .flatten()
            .map(|(block_x, y)| {
                let cell_x_for_block: i32 = block_x * Self::CELL_BITS_I32;
                let block: u64 = *self.grid.get(IVec2::new(block_x, y)).unwrap();

                let mut mask: u64 = 0b10_u64;

                (0_i32..Self::CELL_BITS_I32).filter_map(move |bit_x| {
                    if block & mask != 0_u64 {
                        mask <<= 1_u32;
                        Some(IVec2::new(cell_x_for_block + bit_x, y))
                    } else {
                        mask <<= 1_u32;
                        None
                    }
                })
            })
            .flatten()
    }

    fn try_as_string(&self) -> Grid2DStringResult {
        Grid2DString::from(self.to_intermediate()).try_as_string()
    }

    fn empty_ground_tiles(&self) -> usize {
        let dimensions: IVec2 = self.max - self.min + IVec2::ONE;

        dimensions.x as usize * dimensions.y as usize - self.iter_elves().count()
    }

    fn run(&mut self, rounds: usize) -> &mut Self {
        self.run_internal(rounds);

        self
    }

    fn run_until_static(&mut self) -> usize {
        self.run_internal(usize::MAX)
    }

    fn run_internal(&mut self, rounds: usize) -> usize {
        let mut proposals: Proposals = Proposals::new();

        for round in 0_usize..rounds {
            for y in self.min.y..=self.max.y {
                let x_block_range: Range<i32> = self.x_block_range();
                let x_bit_range: Range<i32> = self.x_bit_range();

                if x_block_range.len() == 1_usize {
                    // The x positions are fully contained within the same block, meaning the `bits`
                    // argument to `collect_proposals_for_block` doesn't need special attention
                    self.collect_proposals_for_block(
                        IVec2::new(x_block_range.start, y),
                        x_bit_range,
                        &mut proposals,
                    );
                } else {
                    // Collect proposals for the first partial block
                    self.collect_proposals_for_block(
                        IVec2::new(x_block_range.start, y),
                        x_bit_range.start..Self::CELL_BITS_I32,
                        &mut proposals,
                    );

                    let x_block_range_end_minus_1: i32 = x_block_range.end - 1_i32;

                    // Collect proposals for any full blocks
                    for x_block in x_block_range.start + 1_i32..x_block_range_end_minus_1 {
                        self.collect_proposals_for_block(
                            IVec2::new(x_block, y),
                            0_i32..Self::CELL_BITS_I32,
                            &mut proposals,
                        );
                    }

                    // Collect proposals for the last partial block
                    self.collect_proposals_for_block(
                        IVec2::new(x_block_range_end_minus_1, y),
                        0_i32..x_bit_range.end,
                        &mut proposals,
                    );
                }
            }

            let moved_elf_count: usize = proposals
                .drain()
                .filter(|(_, requester)| *requester != Self::INVALID_POS)
                .fold(0_usize, |count, (pos, requester)| {
                    self.clear_bit_for_pos(requester);
                    self.set_bit_for_pos(pos);
                    self.min = self.min.min(pos);
                    self.max = self.max.max(pos);

                    count + 1_usize
                });

            self.candidates.rotate_left(1_usize);

            if moved_elf_count == 0_usize && rounds == usize::MAX {
                self.recalibrate_min_and_max();

                return round + 1_usize;
            }
        }

        self.recalibrate_min_and_max();

        rounds
    }

    fn recalibrate_min_and_max(&mut self) -> &mut Self {
        self.min = IVec2::ZERO;
        self.max = self.grid.dimensions() * IVec2::new(Self::CELL_BITS_I32, 1_i32) - IVec2::ONE;

        let mut min: IVec2 = i32::MAX * IVec2::ONE;
        let mut max: IVec2 = i32::MIN * IVec2::ONE;

        for elf_pos in self.iter_elves() {
            min = min.min(elf_pos);
            max = max.max(elf_pos);
        }

        self.min = min;
        self.max = max;

        self
    }

    fn collect_proposals_for_block(
        &self,
        block_pos: IVec2,
        bits: Range<i32>,
        proposals: &mut Proposals,
    ) {
        let mut masks: Masks = self.masks(block_pos);
        let mut pos: IVec2 =
            IVec2::new(block_pos.x * Self::CELL_BITS_I32 + bits.start, block_pos.y);

        masks.shift(bits.start as u32);

        for _ in bits {
            if masks.is_bit_set() && masks.are_any_neighbors_set() {
                if let Some(dir) = masks.try_find_proposal() {
                    let proposal: IVec2 = pos + dir.vec();

                    if let Some(previous_requester) = proposals.get_mut(&proposal) {
                        // Oof, someone else wants this position.
                        *previous_requester = Self::INVALID_POS;
                    } else {
                        proposals.insert(proposal, pos);
                    }
                }
            }

            masks.shift(1_u32);
            pos.x += 1_i32;
        }
    }

    fn masks(&self, block_pos: IVec2) -> Masks {
        let blocks: Blocks = [
            self.grid
                .get(block_pos - IVec2::Y)
                .cloned()
                .unwrap_or_default(),
            *self.grid.get(block_pos).unwrap(),
            self.grid
                .get(block_pos + IVec2::Y)
                .cloned()
                .unwrap_or_default(),
        ];
        let prev_or_next: u64 = blocks[0_usize] | blocks[2_usize];

        Masks {
            blocks,
            prev_or_next,
            bit: 0b10_u64,
            neighbors: Self::NEIGHBORS,
            candidates: self.candidates.clone(),
        }
    }

    fn x_block_range(&self) -> Range<i32> {
        self.min.x / Self::CELL_BITS_I32..self.max.x / Self::CELL_BITS_I32 + 1_i32
    }

    fn x_bit_range(&self) -> Range<i32> {
        self.min.x % Self::CELL_BITS_I32..self.max.x % Self::CELL_BITS_I32 + 1_i32
    }

    fn set_bit_for_pos_generic<F: Fn(&mut u64, i32)>(&mut self, mut pos: IVec2, f: F) {
        let bit_x: i32 = pos.x % Self::CELL_BITS_I32;

        pos.x /= Self::CELL_BITS_I32;

        f(self.grid.get_mut(pos).unwrap(), bit_x);

        if bit_x == 0_i32 {
            // Check if there's a neighboring `u64` to update
            if let Some(block) = self.grid.get_mut(pos - IVec2::X) {
                f(block, Self::CELL_BITS_I32);
            }
        } else if bit_x == Self::CELL_BITS_MINUS_1 {
            // Check if there's a neighboring `u64` to update
            if let Some(block) = self.grid.get_mut(pos + IVec2::X) {
                f(block, -1_i32);
            }
        }
    }

    #[inline(always)]
    fn set_bit_for_pos(&mut self, pos: IVec2) {
        self.set_bit_for_pos_generic(pos, Self::set_bit);
    }

    #[inline(always)]
    fn clear_bit_for_pos(&mut self, pos: IVec2) {
        self.set_bit_for_pos_generic(pos, Self::clear_bit);
    }

    #[inline(always)]
    fn set_bit(block: &mut u64, index: i32) {
        // Add one to account for the overlap bit
        *block |= 1_u64 << (index + 1_i32) as u32;
    }

    #[inline(always)]
    fn clear_bit(block: &mut u64, index: i32) {
        // Add one to account for the overlap bit
        *block &= !(1_u64 << (index + 1_i32) as u32);
    }
}

impl Debug for ElfGrid {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let result: Grid2DStringResult = Grid2DString::from(self.to_intermediate()).try_as_string();

        match result {
            Ok(string) => f.write_fmt(format_args!("\n{string}")),
            Err(err) => f.write_fmt(format_args!("{err:#?}")),
        }
    }
}

impl Default for ElfGrid {
    fn default() -> Self {
        Self {
            grid: Grid2D::default(IVec2::ZERO),
            min: IVec2::ZERO,
            max: IVec2::ZERO,
            candidates: Self::CANDIDATES,
        }
    }
}

#[derive(Debug, PartialEq)]
enum ParseElfGridError {
    RowLengthsDoNotMatch,
    InvalidElfCellByte(u8),
}

impl TryFrom<&str> for ElfGrid {
    type Error = ParseElfGridError;

    fn try_from(elf_grid_str: &str) -> Result<Self, Self::Error> {
        use ParseElfGridError::*;

        let mut elf_row_iter: Peekable<Split<char>> = elf_grid_str.split('\n').peekable();

        match elf_row_iter.peek() {
            Some(elf_row_str) => {
                let width: usize = elf_row_str.len();

                let mut height: usize = 0_usize;
                let mut elf_cells: Vec<ElfCell> = Vec::new();

                for elf_row_str in elf_row_iter {
                    if elf_row_str.len() != width {
                        return Err(RowLengthsDoNotMatch);
                    }

                    for elf_cell in elf_row_str.as_bytes().iter().copied() {
                        elf_cells.push(match elf_cell {
                            ElfCell::EMPTY_U8 => Ok(ElfCell::Empty),
                            ElfCell::ELF_U8 => Ok(ElfCell::Elf),
                            invalid_byte => Err(InvalidElfCellByte(invalid_byte)),
                        }?);
                    }

                    height += 1_usize;
                }

                let mut intermediate: IntermediateElfGrid =
                    IntermediateElfGrid(Grid2D::default(IVec2::new(width as i32, height as i32)));

                intermediate.0.cells_mut().copy_from_slice(&elf_cells);

                Ok(Self::from_intermediate(&intermediate))
            }
            None => Ok(Default::default()),
        }
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day23.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match ElfGrid::try_from(input) {
                    Ok(mut elf_grid_1) => {
                        let mut elf_grid_2: ElfGrid = elf_grid_1.clone();

                        dbg!(elf_grid_1.run(10_usize).empty_ground_tiles());
                        dbg!(elf_grid_2.run_until_static());

                        if args.verbose {
                            println!(
                                "elf_grid_1.try_as_string():\n\
                                \n\
                                {}\n\
                                elf_grid_2.try_as_string():\n\
                                \n\
                                {}",
                                elf_grid_1
                                    .try_as_string()
                                    .unwrap_or_else(|err| format!("{err:#?}")),
                                elf_grid_2
                                    .try_as_string()
                                    .unwrap_or_else(|err| format!("{err:#?}")),
                            );
                        }
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                }
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELF_GRID_STR: &str = concat!(
        "....#..\n",
        "..###.#\n",
        "#...#.#\n",
        ".#...##\n",
        "#.###..\n",
        "##.#.##\n",
        ".#..#..\n",
    );
    const ELF_GRID_10_ROUNDS: &str = concat!(
        "......#.....\n",
        "..........#.\n",
        ".#.#..#.....\n",
        ".....#......\n",
        "..#.....#..#\n",
        "#......##...\n",
        "....##......\n",
        ".#........#.\n",
        "...#.#..#...\n",
        "............\n",
        "...#..#..#..\n",
    );
    const ELF_GRID_20_ROUNDS: &str = concat!(
        ".......#......\n",
        "....#......#..\n",
        "..#.....#.....\n",
        "......#.......\n",
        "...#....#.#..#\n",
        "#.............\n",
        "....#.....#...\n",
        "..#.....#.....\n",
        "....#.#....#..\n",
        ".........#....\n",
        "....#......#..\n",
        ".......#......\n",
    );
    const ELF_GRID_SMALL_STR: &str = "##\n#.\n..\n##\n";
    const ELF_GRID_SMALL_1_ROUND: &str = "##\n..\n#.\n.#\n#.\n";
    const ELF_GRID_SMALL_2_ROUNDS: &str = ".##.\n#...\n...#\n....\n.#..\n";
    const ELF_GRID_SMALL_3_ROUNDS: &str = "..#..\n....#\n#....\n....#\n.....\n..#..\n";

    lazy_static! {
        static ref ELF_GRID: ElfGrid = elf_grid();
        static ref ELF_GRID_SMALL: ElfGrid = elf_grid_small();
    }

    #[test]
    fn test_elf_grid_try_from_str() {
        assert_eq!(
            // Skip the last bit, since that's only for the output form
            ElfGrid::try_from(&ELF_GRID_STR[..ELF_GRID_STR.len() - 1_usize])
                .as_ref()
                .map(ElfGrid::try_as_string),
            Ok(Ok(ELF_GRID_STR.into()))
        );
    }

    #[test]
    fn test_elf_grid_run() {
        let mut elf_grid_small: ElfGrid = ELF_GRID_SMALL.clone();

        assert_eq!(
            elf_grid_small.run(1_usize).try_as_string(),
            Ok(ELF_GRID_SMALL_1_ROUND.into())
        );
        assert_eq!(
            elf_grid_small.run(1_usize).try_as_string(),
            Ok(ELF_GRID_SMALL_2_ROUNDS.into())
        );
        assert_eq!(
            elf_grid_small.run(1_usize).try_as_string(),
            Ok(ELF_GRID_SMALL_3_ROUNDS.into())
        );

        let mut elf_grid: ElfGrid = ELF_GRID.clone();

        assert_eq!(
            elf_grid.run(10_usize).try_as_string(),
            Ok(ELF_GRID_10_ROUNDS.into())
        );
        assert_eq!(
            elf_grid.run(10_usize).try_as_string(),
            Ok(ELF_GRID_20_ROUNDS.into())
        );
    }

    #[test]
    fn test_elf_grid_empty_ground_tiles() {
        assert_eq!(
            ELF_GRID.clone().run(10_usize).empty_ground_tiles(),
            110_usize
        );
    }

    #[test]
    fn test_elf_grid_run_until_static() {
        assert_eq!(ELF_GRID.clone().run_until_static(), 20_usize);
    }

    fn elf_grid() -> ElfGrid {
        ElfGrid::try_from(&ELF_GRID_STR[..ELF_GRID_STR.len() - 1_usize]).unwrap()
    }

    fn elf_grid_small() -> ElfGrid {
        ElfGrid::try_from(&ELF_GRID_SMALL_STR[..ELF_GRID_SMALL_STR.len() - 1_usize]).unwrap()
    }
}
