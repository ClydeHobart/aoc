use {
    clap::Parser,
    glam::IVec2,
    memmap::Mmap,
    static_assertions::const_assert,
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        fs::File,
        io::{Error, ErrorKind, Result as IoResult},
        iter::Peekable,
        mem::transmute,
        ops::{Deref, DerefMut},
        str::{from_utf8, Split, Utf8Error},
    },
    strum::{EnumCount, EnumIter},
};

pub use direction::*;

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
#[derive(Debug)]
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
