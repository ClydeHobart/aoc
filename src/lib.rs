use {
    glam::IVec2,
    memmap::Mmap,
    static_assertions::const_assert,
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        fs::File,
        io::{Error, ErrorKind, Result as IoResult},
        iter::Peekable,
        mem::transmute,
        ops::{Deref, DerefMut, Range, RangeInclusive},
        str::{from_utf8, Split, Utf8Error},
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

pub use {
    self::{cell_iter::*, direction::*},
    clap::Parser,
};

/// Arguments for program execution
///
/// Currently, this is just an input file path, but it may be more later. The default is
/// intentionally left empty such that multiple example programs can use the same struct without
/// needing to re-define it with a different default path.
#[derive(Parser)]
pub struct Args {
    /// Input file path
    #[arg(short, long, default_value_t)]
    input_file_path: String,
}

impl Args {
    /// Returns the input file path, or a provided default if the field is empty
    ///
    /// # Arguments
    ///
    /// * `default` - A default input file path string slice to use if `self.input_file_path` is
    ///   empty
    pub fn input_file_path<'a>(&'a self, default: &'a str) -> &'a str {
        if self.input_file_path.is_empty() {
            default
        } else {
            &self.input_file_path
        }
    }
}

/// Opens a memory-mapped UTF-8 file at a specified path, and passes in a `&str` over the file to a
/// provided callback function
///
/// # Arguments
///
/// * `file_path` - A string slice file path to open as a read-only file
/// * `f` - A callback function to invoke on the contents of the file as a string slice
///
/// # Errors
///
/// This function returns a `Result::Err`-wrapped `std::io::Error` if an error has occurred.
/// Possible causes are:
///
/// * `std::fs::File::open` was unable to open a read-only file at `file_path`
/// * `memmap::Mmap::map` fails to create an `Mmap` instance for the opened file
/// * `std::str::from_utf8` determines the file is not in valid UTF-8 format
///
/// `f` is only executed *iff* an error is not encountered.
///
/// # Safety
///
/// This function uses `Mmap::map`, which is an unsafe function. There is no guarantee that an
/// external process won't modify the file after it is opened as read-only.
///
/// # Undefined Behavior
///
/// Related to the **Safety** section above, it is UB if the opened file is modified by an external
/// process while this function is referring to it as an immutable string slice. For more info on
/// this, see:
///
/// * https://www.reddit.com/r/rust/comments/wyq3ih/why_are_memorymapped_files_unsafe/
/// * https://users.rust-lang.org/t/how-unsafe-is-mmap/19635
/// * https://users.rust-lang.org/t/is-there-no-safe-way-to-use-mmap-in-rust/70338
pub unsafe fn open_utf8_file<F: FnOnce(&str)>(file_path: &str, f: F) -> IoResult<()> {
    let file: File = File::open(file_path)?;

    // SAFETY: This operation is unsafe
    let mmap: Mmap = Mmap::map(&file)?;
    let bytes: &[u8] = &mmap;
    let utf8_str: &str = from_utf8(bytes).map_err(|utf8_error: Utf8Error| -> Error {
        Error::new(ErrorKind::InvalidData, utf8_error)
    })?;

    f(utf8_str);

    Ok(())
}

pub struct TokenStream<'i, 't, I: Iterator<Item = &'t str>>(&'i mut I);

impl<'i, 't, I: Iterator<Item = &'t str>> TokenStream<'i, 't, I> {
    pub fn new(iter: &'i mut I) -> Self {
        Self(iter)
    }
}

impl<'i, 't, I: Iterator<Item = &'t str>> Deref for TokenStream<'i, 't, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'i, 't, I: Iterator<Item = &'t str>> DerefMut for TokenStream<'i, 't, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

pub fn unreachable_any<T, U>(_: T) -> U {
    unreachable!();
}

pub const LOWERCASE_A_OFFSET: u8 = b'a';
pub const UPPERCASE_A_OFFSET: u8 = b'A';
pub const ZERO_OFFSET: u8 = b'0';

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

            const VECS: [IVec2; $direction::COUNT] = [
                $( $direction::$variant.vec_internal(), )*
            ];
        };
    }

    define_direction! {
        #[derive(Copy, Clone, Debug, EnumCount, EnumIter, PartialEq)]
        #[repr(u8)]
        pub enum Direction {
            North,
            East,
            South,
            West,
        }
    }

    // This guarantees we can safely convert from `u8` to `Direction` by masking the smallest 2
    // bits, which is the same as masking by `U8_MASK`
    const_assert!(Direction::COUNT == 4_usize);

    impl Direction {
        const U8_MASK: u8 = Self::COUNT as u8 - 1_u8;

        #[inline]
        pub const fn vec(self) -> IVec2 {
            VECS[self as usize]
        }

        #[inline]
        pub const fn from_u8(value: u8) -> Self {
            // SAFETY: See `const_assert` above
            unsafe { transmute(value & Self::U8_MASK) }
        }

        #[inline]
        pub const fn next(self) -> Self {
            Self::from_u8(self as u8 + 1_u8)
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

    impl From<u8> for Direction {
        fn from(value: u8) -> Self {
            Self::from_u8(value)
        }
    }

    impl TryFrom<IVec2> for Direction {
        type Error = ();

        fn try_from(value: IVec2) -> Result<Self, Self::Error> {
            VECS.iter()
                .position(|vec| *vec == value)
                .map(|index| (index as u8).into())
                .ok_or(())
        }
    }
}

pub struct SideLen(pub usize);

impl From<SideLen> for IVec2 {
    fn from(side_len: SideLen) -> Self {
        IVec2::new(side_len.0 as i32, side_len.0 as i32)
    }
}

pub struct Grid<T> {
    cells: Vec<T>,

    /// Should only contain unsigned values, but is signed for ease of use for iterating
    dimensions: IVec2,
}

impl<T> Grid<T> {
    pub fn try_from_cells_and_width(cells: Vec<T>, width: usize) -> Option<Self> {
        let cells_len: usize = cells.len();

        if cells_len % width != 0_usize {
            None
        } else {
            Some(Self {
                cells,
                dimensions: IVec2::new(width as i32, (cells_len / width) as i32),
            })
        }
    }

    #[cfg(test)]
    pub fn empty(dimensions: IVec2) -> Self {
        Self {
            cells: Vec::new(),
            dimensions,
        }
    }

    pub fn allocate(dimensions: IVec2) -> Self {
        Self {
            cells: Vec::with_capacity((dimensions.x * dimensions.y) as usize),
            dimensions,
        }
    }

    #[inline]
    pub fn cells(&self) -> &[T] {
        &self.cells
    }

    #[inline]
    pub fn cells_mut(&mut self) -> &mut [T] {
        &mut self.cells
    }

    #[inline]
    pub fn dimensions(&self) -> IVec2 {
        self.dimensions
    }

    #[inline]
    pub fn contains(&self, pos: IVec2) -> bool {
        pos.cmpge(IVec2::ZERO).all() && pos.cmplt(self.dimensions).all()
    }

    pub fn is_border(&self, pos: IVec2) -> bool {
        self.contains(pos)
            && (pos.cmpeq(IVec2::ZERO).any() || pos.cmpeq(self.max_dimensions()).any())
    }

    #[inline]
    pub fn index_from_pos(&self, pos: IVec2) -> usize {
        pos.y as usize * self.dimensions.x as usize + pos.x as usize
    }

    pub fn try_index_from_pos(&self, pos: IVec2) -> Option<usize> {
        if self.contains(pos) {
            Some(self.index_from_pos(pos))
        } else {
            None
        }
    }

    pub fn pos_from_index(&self, index: usize) -> IVec2 {
        let x: usize = self.dimensions.x as usize;

        IVec2::new((index % x) as i32, (index / x) as i32)
    }

    #[inline(always)]
    pub fn max_dimensions(&self) -> IVec2 {
        self.dimensions - IVec2::ONE
    }

    pub fn get(&self, pos: IVec2) -> Option<&T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &self.cells[index])
    }

    pub fn get_mut(&mut self, pos: IVec2) -> Option<&mut T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &mut self.cells[index])
    }
}

impl<T: Debug> Debug for Grid<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("Grid")?;
        let mut y_list: DebugList = f.debug_list();

        for y in 0_i32..self.dimensions.y {
            let start: usize = (y * self.dimensions.x) as usize;

            y_list.entry(&&self.cells[start..(start + self.dimensions.x as usize)]);
        }

        y_list.finish()
    }
}

impl<T: Default> Grid<T> {
    pub fn default(dimensions: IVec2) -> Self {
        let capacity: usize = (dimensions.x * dimensions.y) as usize;
        let mut cells: Vec<T> = Vec::with_capacity(capacity);

        cells.resize_with(capacity, T::default);

        Self { cells, dimensions }
    }
}

impl<T: PartialEq> PartialEq for Grid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.cells == other.cells
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum GridParseError<'s, E> {
    NoInitialToken,
    IsNotAscii(&'s str),
    InvalidLength { line: &'s str, expected_len: usize },
    CellParseError(E),
}

impl<'s, E, T: TryFrom<char, Error = E>> TryFrom<&'s str> for Grid<T> {
    type Error = GridParseError<'s, E>;

    fn try_from(grid_str: &'s str) -> Result<Self, Self::Error> {
        use GridParseError as Error;

        let mut grid_line_iter: Peekable<Split<char>> = grid_str.split('\n').peekable();

        let side_len: usize = grid_line_iter.peek().ok_or(Error::NoInitialToken)?.len();

        let mut grid: Grid<T> = Grid::allocate(SideLen(side_len).into());
        let mut lines: usize = 0_usize;

        for grid_line_str in grid_line_iter {
            if !grid_line_str.is_ascii() {
                return Err(Error::IsNotAscii(grid_line_str));
            }

            if grid_line_str.len() != side_len {
                return Err(Error::InvalidLength {
                    line: grid_line_str,
                    expected_len: side_len,
                });
            }

            for cell_char in grid_line_str.chars() {
                grid.cells
                    .push(cell_char.try_into().map_err(Error::CellParseError)?);
            }

            lines += 1_usize;
        }

        if lines != side_len {
            grid.dimensions.y = lines as i32;
        }

        Ok(grid)
    }
}

mod cell_iter {
    use super::*;

    pub struct CellIter {
        curr: IVec2,
        end: IVec2,
        dir: Direction,
    }

    impl CellIter {
        pub fn corner<T>(grid: &Grid<T>, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let curr: IVec2 = (-grid.dimensions() * (dir_vec + dir_vec.perp()))
                .clamp(IVec2::ZERO, grid.max_dimensions());

            Self::until_boundary(grid, curr, dir)
        }

        pub fn until_boundary<T>(grid: &Grid<T>, curr: IVec2, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let end: IVec2 = (curr + dir_vec * grid.dimensions())
                .clamp(IVec2::ZERO, grid.max_dimensions())
                + dir_vec;

            Self { curr, end, dir }
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

    #[derive(Debug)]
    pub enum CellIterFromRangeError {
        PositionsIdentical,
        PositionsNotAligned,
    }

    impl TryFrom<Range<IVec2>> for CellIter {
        type Error = CellIterFromRangeError;

        fn try_from(Range { start, end }: Range<IVec2>) -> Result<Self, Self::Error> {
            use CellIterFromRangeError::*;

            let delta: IVec2 = end - start;

            if delta == IVec2::ZERO {
                Err(PositionsIdentical)
            } else if delta.x != 0_i32 && delta.y != 0_i32 {
                Err(PositionsNotAligned)
            } else {
                let abs: IVec2 = delta.abs();
                let dir: Direction = (delta / (abs.x + abs.y)).try_into().unwrap();

                Ok(Self {
                    curr: start,
                    end,
                    dir,
                })
            }
        }
    }

    impl TryFrom<RangeInclusive<IVec2>> for CellIter {
        type Error = CellIterFromRangeError;

        fn try_from(range_inclusive: RangeInclusive<IVec2>) -> Result<Self, Self::Error> {
            use CellIterFromRangeError::*;

            let curr: IVec2 = *range_inclusive.start();
            let end: IVec2 = *range_inclusive.end();
            let delta: IVec2 = end - curr;

            if delta == IVec2::ZERO {
                Err(PositionsIdentical)
            } else if delta.x != 0_i32 && delta.y != 0_i32 {
                Err(PositionsNotAligned)
            } else {
                let abs: IVec2 = delta.abs();
                let dir: Direction = (delta / (abs.x + abs.y)).try_into().unwrap();

                Ok(Self {
                    curr,
                    end: end + dir.vec(),
                    dir,
                })
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_corner() {
            let grid: Grid<()> = Grid::empty(SideLen(5_usize).into());

            assert_eq!(
                Direction::iter()
                    .map(|dir: Direction| -> CellIter { CellIter::corner(&grid, dir) })
                    .flatten()
                    .map(|pos: IVec2| -> usize { grid.index_from_pos(pos) })
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

pub trait GridVisitor: Default + Sized {
    type Old;
    type New: Default;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        old_grid: &Grid<Self::Old>,
        rev_dir: Direction,
        pos: IVec2,
    );

    fn visit_grid(old_grid: &Grid<Self::Old>) -> Grid<Self::New> {
        let mut new_grid: Grid<Self::New> = Grid::default(old_grid.dimensions());

        for dir in Direction::iter() {
            let row_dir: Direction = dir.next();

            // Look back the way we came to make the most use of the local `GridVisitor`
            let rev_dir: Direction = (row_dir as u8 + 2_u8).into();

            for row_pos in CellIter::corner(old_grid, dir) {
                let mut grid_visitor: Self = Self::default();

                for pos in CellIter::until_boundary(old_grid, row_pos, row_dir) {
                    grid_visitor.visit_cell(
                        new_grid.get_mut(pos).unwrap(),
                        old_grid.get(pos).unwrap(),
                        old_grid,
                        rev_dir,
                        pos,
                    );
                }
            }
        }

        new_grid
    }
}

pub fn validate_prefix_and_suffix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    suffix: &str,
    f: F,
) -> Result<&'s str, E> {
    if value.len() >= prefix.len() + suffix.len()
        && value.get(..prefix.len()).map_or(false, |p| p == prefix)
        && value
            .get(value.len() - suffix.len()..)
            .map_or(false, |s| s == suffix)
    {
        Ok(&value[prefix.len()..value.len() - suffix.len()])
    } else {
        Err(f(value))
    }
}

pub fn validate_prefix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    f: F,
) -> Result<&'s str, E> {
    validate_prefix_and_suffix(value, prefix, "", f)
}
