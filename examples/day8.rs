use {
    self::{cell_iter::*, direction::*, neighbor_heights::*},
    aoc_2022::*,
    clap::Parser,
    glam::IVec2,
    static_assertions::const_assert,
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        iter::Peekable,
        mem::transmute,
        ops::{Index, IndexMut},
        str::Split,
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

mod direction {
    use super::*;

    macro_rules! define_direction {
        {
            $(#[$meta:meta])*
            $vis:vis enum $direction:ident {
                $( $variant:ident, )*
            }
        } => {
            $(#[$meta])*
            $vis enum $direction {
                $( $variant, )*
            }

            const OFFSETS: [u32; $direction::COUNT] = [
                $( $direction::$variant.offset_internal(), )*
            ];

            const MASKS: [u16; $direction::COUNT] = [
                $( $direction::$variant.mask_internal(), )*
            ];

            const VECS: [IVec2; $direction::COUNT] = [
                $( $direction::$variant.vec_internal(), )*
            ];
        };
    }

    pub const BITS: u32 = u16::BITS / Direction::COUNT as u32;
    // pub const MASK: u16 = (1_u16 << BITS) - 1_u16;
    pub const MIN_INVALID: u8 = 10_u8;

    define_direction! {
        #[derive(Copy, Clone, Debug, EnumCount, EnumIter)]
        #[repr(usize)]
        pub enum Direction {
            North,
            East,
            South,
            West,
        }
    }

    // This guarantees we can safely convert from `usize` to `Direction` by masking the smallest 2
    // bits
    const_assert!(Direction::COUNT == 4_usize);

    impl Direction {
        #[inline]
        pub const fn offset(self) -> u32 {
            OFFSETS[self as usize]
        }

        #[inline]
        pub const fn mask(self) -> u16 {
            MASKS[self as usize]
        }

        #[inline]
        pub const fn vec(self) -> IVec2 {
            VECS[self as usize]
        }

        #[inline]
        pub const fn from_usize(value: usize) -> Self {
            unsafe { transmute(value & (Self::COUNT - 1_usize)) }
        }

        #[inline]
        pub const fn next(self) -> Self {
            Self::from_usize(self as usize + 1_usize)
        }

        const fn offset_internal(self) -> u32 {
            self as u32 * BITS
        }

        const fn mask_internal(self) -> u16 {
            let start: u32 = self.offset();
            let end: u32 = start + BITS;

            match 1_u16.checked_shl(end) {
                Some(shift_result) => shift_result,
                None => 0_u16,
            }
            .wrapping_sub(1_u16)
                & !((1_u16 << start) - 1_u16)
        }

        const fn vec_internal(self) -> IVec2 {
            match self {
                Self::North => IVec2::NEG_Y,
                Self::East => IVec2::X,
                Self::South => IVec2::Y,
                Self::West => IVec2::NEG_X,
            }
        }
    }

    impl From<Direction> for IVec2 {
        fn from(value: Direction) -> Self {
            value.vec()
        }
    }

    impl From<usize> for Direction {
        fn from(value: usize) -> Self {
            Self::from_usize(value)
        }
    }
}

mod cell_iter {
    use super::*;

    pub struct CellIter {
        dir: Direction,
        curr: IVec2,
        end: IVec2,
    }

    impl CellIter {
        pub(super) fn corner(height_grid: &HeightGrid, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let curr: IVec2 = (-height_grid.dimensions * (dir_vec + dir_vec.perp()))
                .clamp(IVec2::ZERO, height_grid.max());

            Self::until(height_grid, dir, curr)
        }

        #[inline(always)]
        pub(super) fn until(height_grid: &HeightGrid, dir: Direction, curr: IVec2) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let end: IVec2 = (curr + dir_vec * height_grid.dimensions)
                .clamp(IVec2::ZERO, height_grid.max())
                + dir_vec;

            Self { dir, curr, end }
        }
    }

    impl Iterator for CellIter {
        type Item = IVec2;

        fn next(&mut self) -> Option<Self::Item> {
            if self.curr != self.end {
                let prev: IVec2 = self.curr;

                self.curr += self.dir.vec();

                Some(prev)
            } else {
                None
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_corner() {
            let height_grid: HeightGrid = HeightGrid::new(5_usize);

            assert_eq!(
                Direction::iter()
                    .map(|dir: Direction| -> CellIter { CellIter::corner(&height_grid, dir) })
                    .flatten()
                    .map(|pos: IVec2| -> usize { height_grid.index_from_pos(pos) })
                    .collect::<Vec<usize>>(),
                vec![
                    20, 15, 10, 5, 0, // North
                    0, 1, 2, 3, 4, // East
                    4, 9, 14, 19, 24, // South
                    24, 23, 22, 21, 20 // West
                ]
            );
        }
    }
}

mod neighbor_heights {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct NeighborHeights(u16);

    pub const ZERO: NeighborHeights = NeighborHeights(0_u16);

    impl NeighborHeights {
        pub const fn new() -> Self {
            Self(u16::MAX)
        }

        #[inline]
        pub fn get_neighbor(self, dir: Direction) -> u8 {
            ((self.0 & dir.mask()) >> dir.offset()) as u8
        }

        pub fn set_neighbor(&mut self, dir: Direction, neighbor_height: u8) {
            let neighbor_height: u16 = (neighbor_height as u16) << dir.offset();
            let mask: u16 = dir.mask();

            self.0 = (self.0 & !mask) | neighbor_height;
        }
    }

    impl Debug for NeighborHeights {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            f.write_str("NeighborHeights")?;
            f.debug_list()
                .entries(
                    Direction::iter()
                        .map(|dir: Direction| -> (Direction, u8) { (dir, self.get_neighbor(dir)) }),
                )
                .finish()
        }
    }

    impl Default for NeighborHeights {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(align(4))]
struct HeightCell {
    height: u8,
    min_neighbor: u8,
    neighbor_heights: NeighborHeights,
}

const ZERO: HeightCell = HeightCell {
    height: 0_u8,
    min_neighbor: 0_u8,
    neighbor_heights: neighbor_heights::ZERO,
};

impl HeightCell {
    const fn new(height: u8) -> Self {
        Self {
            height,
            min_neighbor: u8::MAX,
            neighbor_heights: NeighborHeights::new(),
        }
    }

    #[inline(always)]
    pub fn get_neighbor(self, dir: Direction) -> u8 {
        self.neighbor_heights.get_neighbor(dir)
    }

    pub fn set_neighbor(&mut self, dir: Direction, neighbor_height: u8) {
        self.min_neighbor = self.min_neighbor.min(neighbor_height);
        self.neighbor_heights.set_neighbor(dir, neighbor_height);
    }
}

struct HeightGrid {
    cells: Vec<HeightCell>,

    /// Should only contain unsigned values, but is signed for ease of use for iterating
    dimensions: IVec2,
}

#[allow(dead_code)]
#[derive(Debug)]
enum HeightGridComputeError {
    NeighborCellHadInvalidHeight {
        vec: IVec2,
        cell: HeightCell,
        dir: Direction,
    },
}

impl HeightGrid {
    fn new(side_len: usize) -> Self {
        Self {
            cells: Vec::with_capacity(side_len * side_len),
            dimensions: IVec2::new(side_len as i32, side_len as i32),
        }
    }

    #[inline]
    fn is_in_grid(&self, pos: IVec2) -> bool {
        pos.cmpge(IVec2::ZERO).all() && pos.cmplt(self.dimensions).all()
    }

    fn is_border(&self, pos: IVec2) -> bool {
        self.is_in_grid(pos) && (pos.cmpeq(IVec2::ZERO).any() || pos.cmpeq(self.max()).any())
    }

    #[inline]
    fn pos_from_index(&self, index: usize) -> IVec2 {
        let x: usize = self.dimensions.x as usize;

        IVec2::new((index % x) as i32, (index / x) as i32)
    }

    #[inline]
    fn index_from_pos(&self, pos: IVec2) -> usize {
        pos.y as usize * self.dimensions.x as usize + pos.x as usize
    }

    fn try_index_from_pos(&self, pos: IVec2) -> Option<usize> {
        if self.is_in_grid(pos) {
            Some(self.index_from_pos(pos))
        } else {
            None
        }
    }

    #[inline(always)]
    fn max(&self) -> IVec2 {
        self.dimensions - IVec2::ONE
    }

    fn compute_min_neighbors(&mut self) -> Result<(), HeightGridComputeError> {
        for dir in Direction::iter() {
            let cell_dir: Direction = dir.next();
            let neighbor_dir: Direction = cell_dir.next();
            let neighbor_vec: IVec2 = neighbor_dir.vec();

            for row_iter in CellIter::corner(self, dir) {
                for cell_iter in CellIter::until(self, cell_dir, row_iter) {
                    let neighbor: HeightCell = self[cell_iter + neighbor_vec];
                    let neighbor_neighbor_height: u8 = neighbor.get_neighbor(neighbor_dir);

                    if neighbor_neighbor_height >= MIN_INVALID {
                        return Err(HeightGridComputeError::NeighborCellHadInvalidHeight {
                            vec: cell_iter + neighbor_vec,
                            cell: neighbor,
                            dir: neighbor_dir,
                        });
                    }

                    self[cell_iter]
                        .set_neighbor(neighbor_dir, neighbor.height.max(neighbor_neighbor_height));
                }
            }
        }

        Ok(())
    }
}

impl Debug for HeightGrid {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("HeightGrid")?;
        let mut y_list: DebugList = f.debug_list();

        for y in 0_i32..self.dimensions.y {
            let start: usize = (y * self.dimensions.x) as usize;

            y_list.entry(&&self.cells[start..(start + self.dimensions.x as usize)]);
        }

        y_list.finish()
    }
}

impl Index<IVec2> for HeightGrid {
    type Output = HeightCell;

    fn index(&self, pos: IVec2) -> &Self::Output {
        match self.try_index_from_pos(pos) {
            Some(index) => &self.cells[index],
            None => &ZERO,
        }
    }
}

impl IndexMut<IVec2> for HeightGrid {
    fn index_mut(&mut self, pos: IVec2) -> &mut Self::Output {
        match self.try_index_from_pos(pos) {
            Some(index) => &mut self.cells[index],
            None => panic!(
                "`HeightGrid::index_mut` called on grid with dimensions {:?} for position {:?}",
                self.dimensions, pos
            ),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum HeightGridParseError<'s> {
    NoInitialToken,
    IsNotAscii(&'s str),
    InvalidLength { line: &'s str, expected_len: usize },
    IsNotAsciiDigit(char),
    ComputationFailed(HeightGridComputeError),
}

impl<'s> TryFrom<&'s str> for HeightGrid {
    type Error = HeightGridParseError<'s>;

    fn try_from(height_grid_str: &'s str) -> Result<Self, Self::Error> {
        use HeightGridParseError as Error;

        let mut height_grid_line_iter: Peekable<Split<char>> =
            height_grid_str.split('\n').peekable();

        let side_len: usize = height_grid_line_iter
            .peek()
            .ok_or(Error::NoInitialToken)?
            .len();

        let mut height_grid: HeightGrid = HeightGrid::new(side_len);
        let mut lines: usize = 0_usize;

        for height_grid_line_str in height_grid_line_iter {
            if !height_grid_line_str.is_ascii() {
                return Err(Error::IsNotAscii(height_grid_line_str));
            }

            if height_grid_line_str.len() != side_len {
                return Err(Error::InvalidLength {
                    line: height_grid_line_str,
                    expected_len: side_len,
                });
            }

            for height_cell_char in height_grid_line_str.chars() {
                if !height_cell_char.is_ascii_digit() {
                    return Err(Error::IsNotAsciiDigit(height_cell_char));
                }

                height_grid
                    .cells
                    .push(HeightCell::new(height_cell_char as u8 - ZERO_OFFSET));
            }

            lines += 1_usize;
        }

        if lines != side_len {
            height_grid.dimensions.y = lines as i32;
        }

        height_grid
            .compute_min_neighbors()
            .map_err(Error::ComputationFailed)?;

        Ok(height_grid)
    }
}

fn count_visible_cells(height_grid: &HeightGrid) -> usize {
    height_grid
        .cells
        .iter()
        .enumerate()
        .filter(|&(index, &height_cell): &(usize, &HeightCell)| -> bool {
            height_grid.is_border(height_grid.pos_from_index(index))
                || height_cell.height > height_cell.min_neighbor
        })
        .count()
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day8.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match HeightGrid::try_from(input) {
                    Ok(height_grid) => {
                        println!(
                            "count_visible_cells == {}",
                            count_visible_cells(&height_grid)
                        );
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

#[test]
fn test() {
    const HEIGHT_GRID_STR: &str = "\
        30373\n\
        25512\n\
        65332\n\
        33549\n\
        35390";

    match HeightGrid::try_from(HEIGHT_GRID_STR) {
        Ok(height_grid) => assert_eq!(
            count_visible_cells(&height_grid),
            21_usize,
            "{height_grid:#?}"
        ),
        Err(error) => panic!("{error:#?}"),
    }
}
