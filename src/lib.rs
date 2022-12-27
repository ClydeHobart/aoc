use {
    glam::{IVec2, IVec3},
    memmap::Mmap,
    static_assertions::const_assert,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet, VecDeque},
        fmt::{Debug, DebugList, Error as FmtError, Formatter, Result as FmtResult, Write},
        fs::File,
        hash::Hash,
        io::{Error as IoError, ErrorKind, Result as IoResult},
        iter::Peekable,
        mem::{take, transmute},
        ops::{Add, Deref, DerefMut, Range, RangeInclusive},
        str::{from_utf8, Split, Utf8Error},
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

pub use {
    self::{direction::*, grid_2d::*, grid_3d::*},
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

    /// Print extra information, if there is any
    #[arg(short, long, default_value_t)]
    pub verbose: bool,
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
    let utf8_str: &str = from_utf8(bytes).map_err(|utf8_error: Utf8Error| -> IoError {
        IoError::new(ErrorKind::InvalidData, utf8_error)
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
        pub const U8_MASK: u8 = Self::COUNT as u8 - 1_u8;

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

        #[inline]
        pub const fn rev(self) -> Self {
            Self::from_u8(self as u8 + 2_u8)
        }

        #[inline]
        pub const fn prev(self) -> Self {
            Self::from_u8(self as u8 + 3_u8)
        }

        pub const fn turn(self, left: bool) -> Self {
            if left {
                self.prev()
            } else {
                self.next()
            }
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

#[derive(Debug)]
pub enum CellIterFromRangeError {
    PositionsIdentical,
    PositionsNotAligned,
}

mod grid_2d {
    use super::*;

    pub fn abs_sum_2d(pos: IVec2) -> i32 {
        let abs: IVec2 = pos.abs();

        abs.x + abs.y
    }

    pub struct Grid2D<T> {
        cells: Vec<T>,

        /// Should only contain unsigned values, but is signed for ease of use for iterating
        dimensions: IVec2,
    }

    impl<T> Grid2D<T> {
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

        pub fn resize_rows<F: FnMut() -> T>(&mut self, new_row_len: usize, f: F) {
            self.dimensions.y = new_row_len as i32;
            self.cells
                .resize_with((self.dimensions.x * self.dimensions.y) as usize, f);
        }

        pub fn reserve_rows(&mut self, additional_rows: usize) {
            self.cells
                .reserve(self.dimensions.x as usize * additional_rows);
        }
    }

    impl<T: Clone> Clone for Grid2D<T> {
        fn clone(&self) -> Self {
            Self {
                cells: self.cells.clone(),
                dimensions: self.dimensions,
            }
        }
    }

    impl<T: Debug> Debug for Grid2D<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            f.write_str("Grid2D")?;
            let mut y_list: DebugList = f.debug_list();

            for y in 0_i32..self.dimensions.y {
                let start: usize = (y * self.dimensions.x) as usize;

                y_list.entry(&&self.cells[start..(start + self.dimensions.x as usize)]);
            }

            y_list.finish()
        }
    }

    impl<T: Default> Grid2D<T> {
        pub fn default(dimensions: IVec2) -> Self {
            let capacity: usize = (dimensions.x * dimensions.y) as usize;
            let mut cells: Vec<T> = Vec::with_capacity(capacity);

            cells.resize_with(capacity, T::default);

            Self { cells, dimensions }
        }
    }

    impl<T: PartialEq> PartialEq for Grid2D<T> {
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

    impl<'s, E, T: TryFrom<char, Error = E>> TryFrom<&'s str> for Grid2D<T> {
        type Error = GridParseError<'s, E>;

        fn try_from(grid_str: &'s str) -> Result<Self, Self::Error> {
            use GridParseError as Error;

            let mut grid_line_iter: Peekable<Split<char>> = grid_str.split('\n').peekable();

            let side_len: usize = grid_line_iter.peek().ok_or(Error::NoInitialToken)?.len();

            let mut grid: Grid2D<T> = Grid2D::allocate(SideLen(side_len).into());
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

    pub struct CellIter2D {
        curr: IVec2,
        end: IVec2,
        dir: Direction,
    }

    impl CellIter2D {
        pub fn corner<T>(grid: &Grid2D<T>, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let curr: IVec2 = (-grid.dimensions() * (dir_vec + dir_vec.perp()))
                .clamp(IVec2::ZERO, grid.max_dimensions());

            Self::until_boundary(grid, curr, dir)
        }

        pub fn until_boundary<T>(grid: &Grid2D<T>, curr: IVec2, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let end: IVec2 = (curr + dir_vec * grid.dimensions())
                .clamp(IVec2::ZERO, grid.max_dimensions())
                + dir_vec;

            Self { curr, end, dir }
        }
    }

    impl Iterator for CellIter2D {
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

    impl TryFrom<Range<IVec2>> for CellIter2D {
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

    impl TryFrom<RangeInclusive<IVec2>> for CellIter2D {
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

    pub struct Grid2DString(Grid2D<u8>);

    impl Grid2DString {
        pub fn grid(&self) -> &Grid2D<u8> {
            &self.0
        }

        pub fn grid_mut(&mut self) -> &mut Grid2D<u8> {
            &mut self.0
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum Grid2DStringError {
        Utf8Error(Utf8Error),
        FmtError(FmtError),
    }

    pub type Grid2DStringResult = Result<String, Grid2DStringError>;

    impl Grid2DString {
        pub fn try_as_string(&self) -> Grid2DStringResult {
            use Grid2DStringError as Error;

            let dimensions: IVec2 = self.0.dimensions;
            let width: usize = dimensions.x as usize;
            let height: usize = dimensions.y as usize;
            let bytes: &[u8] = &self.0.cells;

            let mut string: String = String::with_capacity((width + 1_usize) * height);

            for y in 0_usize..height {
                let start: usize = y * width;
                let end: usize = start + width;

                write!(
                    &mut string,
                    "{}\n",
                    from_utf8(&bytes[start..end]).map_err(Error::Utf8Error)?
                )
                .map_err(Error::FmtError)?;
            }

            Ok(string)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_corner() {
            let grid: Grid2D<()> = Grid2D::empty(SideLen(5_usize).into());

            assert_eq!(
                Direction::iter()
                    .map(|dir: Direction| -> CellIter2D { CellIter2D::corner(&grid, dir) })
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

mod grid_3d {
    use super::*;

    pub struct Grid3D<T> {
        cells: Vec<T>,

        /// Should only contain unsigned values, but is signed for ease of use for iterating
        dimensions: IVec3,
    }

    pub fn manhattan_distance_3d(a: &IVec3, b: &IVec3) -> i32 {
        let abs_diff: IVec3 = (*b - *a).abs();

        abs_diff.x + abs_diff.y + abs_diff.z
    }

    pub fn sanitize_dir_3d(dir: &mut IVec3) {
        let abs: IVec3 = dir.abs();

        if abs.x + abs.y + abs.z != 1_i32 {
            *dir = if abs == IVec3::ZERO {
                IVec3::X
            } else {
                const POS_AND_NEG_AXES: [IVec3; 6_usize] = [
                    IVec3::X,
                    IVec3::NEG_X,
                    IVec3::Y,
                    IVec3::NEG_Y,
                    IVec3::Z,
                    IVec3::NEG_Z,
                ];

                *POS_AND_NEG_AXES
                    .iter()
                    .min_by_key(|axis| manhattan_distance_3d(*axis, dir))
                    .unwrap()
            }
        }
    }

    impl<T> Grid3D<T> {
        pub fn empty(dimensions: IVec3) -> Self {
            Self {
                cells: Vec::new(),
                dimensions,
            }
        }

        pub fn allocate(dimensions: IVec3) -> Self {
            Self {
                cells: Vec::with_capacity((dimensions.x * dimensions.y * dimensions.z) as usize),
                dimensions,
            }
        }

        #[inline(always)]
        pub fn cells(&self) -> &[T] {
            &self.cells
        }

        #[inline(always)]
        pub fn cells_mut(&mut self) -> &mut [T] {
            &mut self.cells
        }

        #[inline(always)]
        pub fn dimensions(&self) -> &IVec3 {
            &self.dimensions
        }

        #[inline(always)]
        pub fn contains(&self, pos: &IVec3) -> bool {
            pos.cmpge(IVec3::ZERO).all() && pos.cmplt(self.dimensions).all()
        }

        pub fn is_border(&self, pos: &IVec3) -> bool {
            self.contains(pos)
                && (pos.cmpeq(IVec3::ZERO).any() || pos.cmpeq(self.max_dimensions()).any())
        }

        pub fn index_from_pos(&self, pos: &IVec3) -> usize {
            let [width, height, _] = self.width_height_depth();

            pos.z as usize * width * height + pos.y as usize * width + pos.x as usize
        }

        pub fn try_index_from_pos(&self, pos: &IVec3) -> Option<usize> {
            if self.contains(pos) {
                Some(self.index_from_pos(pos))
            } else {
                None
            }
        }

        pub fn pos_from_index(&self, mut index: usize) -> IVec3 {
            let [width, height, _] = self.width_height_depth();
            let width_height_product: usize = width * height;
            let z: i32 = (index / width_height_product) as i32;

            index %= width_height_product;

            let y: i32 = (index / width) as i32;

            index %= width;

            let x: i32 = index as i32;

            IVec3 { x, y, z }
        }

        #[inline(always)]
        pub fn max_dimensions(&self) -> IVec3 {
            self.dimensions - IVec3::ONE
        }

        pub fn get(&self, pos: &IVec3) -> Option<&T> {
            self.try_index_from_pos(pos)
                .map(|index: usize| &self.cells[index])
        }

        pub fn get_mut(&mut self, pos: &IVec3) -> Option<&mut T> {
            self.try_index_from_pos(pos)
                .map(|index: usize| &mut self.cells[index])
        }

        pub fn rem_euclid(&self, pos: &IVec3) -> IVec3 {
            IVec3::new(
                pos.x.rem_euclid(self.dimensions.x),
                pos.y.rem_euclid(self.dimensions.y),
                pos.z.rem_euclid(self.dimensions.z),
            )
        }

        pub fn resize_layers<F: FnMut() -> T>(&mut self, new_layer_len: usize, f: F) {
            self.dimensions.z = new_layer_len as i32;
            self.cells.resize_with(
                (self.dimensions.x * self.dimensions.y * self.dimensions.z) as usize,
                f,
            );
        }

        pub fn reserve_layers(&mut self, additional_layers: usize) {
            self.cells
                .reserve((self.dimensions.x * self.dimensions.y) as usize * additional_layers);
        }

        #[inline(always)]
        fn width_height_depth(&self) -> [usize; 3_usize] {
            [
                self.dimensions.x as usize,
                self.dimensions.y as usize,
                self.dimensions.z as usize,
            ]
        }
    }

    impl<T: Clone> Clone for Grid3D<T> {
        fn clone(&self) -> Self {
            Self {
                cells: self.cells.clone(),
                dimensions: self.dimensions,
            }
        }
    }

    impl<T: Debug> Debug for Grid3D<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            let y_layers: Vec<&[T]> = self
                .cells
                .chunks_exact(self.dimensions.x as usize)
                .collect();
            let z_layers: Vec<&[&[T]]> =
                y_layers.chunks_exact(self.dimensions.y as usize).collect();

            f.write_str("Grid3D")?;
            f.debug_list().entries(z_layers.iter()).finish()
        }
    }

    impl<T: Default> Grid3D<T> {
        pub fn default(dimensions: IVec3) -> Self {
            let capacity: usize = (dimensions.x * dimensions.y * dimensions.z) as usize;
            let mut cells: Vec<T> = Vec::with_capacity(capacity);

            cells.resize_with(capacity, T::default);

            Self { cells, dimensions }
        }
    }

    impl<T> Default for Grid3D<T> {
        fn default() -> Self {
            Self::empty(IVec3::ZERO)
        }
    }

    impl<T: PartialEq> PartialEq for Grid3D<T> {
        fn eq(&self, other: &Self) -> bool {
            self.dimensions == other.dimensions && self.cells == other.cells
        }
    }

    pub struct CellIter3D {
        curr: IVec3,
        end: IVec3,
        dir: IVec3,
    }

    impl CellIter3D {
        pub fn until_boundary<T>(grid: &Grid3D<T>, curr: IVec3, mut dir: IVec3) -> Self {
            sanitize_dir_3d(&mut dir);

            let end: IVec3 =
                (curr + dir * *grid.dimensions()).clamp(IVec3::ZERO, grid.max_dimensions()) + dir;

            Self { curr, end, dir }
        }

        pub fn until_boundary_from_dimensions(
            dimensions: &IVec3,
            curr: IVec3,
            mut dir: IVec3,
        ) -> Self {
            sanitize_dir_3d(&mut dir);

            let end: IVec3 =
                (curr + dir * *dimensions).clamp(IVec3::ZERO, *dimensions - IVec3::ONE) + dir;

            Self { curr, end, dir }
        }
    }

    impl Iterator for CellIter3D {
        type Item = IVec3;

        fn next(&mut self) -> Option<Self::Item> {
            if self.curr != self.end {
                let prev: IVec3 = self.curr;

                self.curr += self.dir;

                Some(prev)
            } else {
                None
            }
        }
    }

    impl TryFrom<Range<IVec3>> for CellIter3D {
        type Error = CellIterFromRangeError;

        fn try_from(Range { start, end }: Range<IVec3>) -> Result<Self, Self::Error> {
            use CellIterFromRangeError::*;

            let delta: IVec3 = end - start;

            if delta == IVec3::ZERO {
                Err(PositionsIdentical)
            } else if delta.cmpeq(IVec3::ZERO).bitmask().count_ones() != 2_u32 {
                Err(PositionsNotAligned)
            } else {
                let dir: IVec3 = delta / manhattan_distance_3d(&start, &end);

                Ok(Self {
                    curr: start,
                    end,
                    dir,
                })
            }
        }
    }

    impl TryFrom<RangeInclusive<IVec3>> for CellIter3D {
        type Error = CellIterFromRangeError;

        fn try_from(range_inclusive: RangeInclusive<IVec3>) -> Result<Self, Self::Error> {
            use CellIterFromRangeError::*;

            let curr: IVec3 = *range_inclusive.start();
            let end: IVec3 = *range_inclusive.end();
            let delta: IVec3 = end - curr;

            if delta == IVec3::ZERO {
                Err(PositionsIdentical)
            } else if delta.cmpeq(IVec3::ZERO).bitmask().count_ones() != 2_u32 {
                Err(PositionsNotAligned)
            } else {
                let dir: IVec3 = delta / manhattan_distance_3d(&curr, &end);

                Ok(Self {
                    curr,
                    end: end + dir,
                    dir,
                })
            }
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
        old_grid: &Grid2D<Self::Old>,
        rev_dir: Direction,
        pos: IVec2,
    );

    fn visit_grid(old_grid: &Grid2D<Self::Old>) -> Grid2D<Self::New> {
        let mut new_grid: Grid2D<Self::New> = Grid2D::default(old_grid.dimensions());

        for dir in Direction::iter() {
            let row_dir: Direction = dir.next();

            // Look back the way we came to make the most use of the local `GridVisitor`
            let rev_dir: Direction = (row_dir as u8 + 2_u8).into();

            for row_pos in CellIter2D::corner(old_grid, dir) {
                let mut grid_visitor: Self = Self::default();

                for pos in CellIter2D::until_boundary(old_grid, row_pos, row_dir) {
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

struct OpenSetElement<V: Clone + PartialEq, C: Clone + Ord>(V, C);

impl<V: Clone + PartialEq, C: Clone + Ord> PartialEq for OpenSetElement<V, C> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> PartialOrd for OpenSetElement<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse the order so that cost is minimized when popping from the heap
        Some(other.1.cmp(&self.1))
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> Eq for OpenSetElement<V, C> {}

impl<V: Clone + PartialEq, C: Clone + Ord> Ord for OpenSetElement<V, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the order so that cost is minimized when popping from the heap
        other.1.cmp(&self.1)
    }
}

/// An implementation of https://en.wikipedia.org/wiki/A*_search_algorithm
pub trait AStar: Sized {
    type Vertex: Clone + Eq + Hash;
    type Cost: Add<Self::Cost, Output = Self::Cost> + Clone + Ord + Sized;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn cost_between_neighbors(&self, from: &Self::Vertex, to: &Self::Vertex) -> Self::Cost;
    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_score(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        heuristic: Self::Cost,
    );

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
        let start: Self::Vertex = self.start().clone();

        let mut open_set_heap: BinaryHeap<OpenSetElement<Self::Vertex, Self::Cost>> =
            BinaryHeap::new();
        let mut open_set_set: HashSet<Self::Vertex> = HashSet::new();

        open_set_heap.push(OpenSetElement(start.clone(), self.cost_from_start(&start)));
        open_set_set.insert(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        // A pair, where the first field is the new cost for the neighbor, already passed into
        // `update_score`, and a bool representing whether the neighbor was previously in
        // `open_set_set`, meaning `open_set_heap` requires special attention to update its score
        let mut neighbor_updates: Vec<Option<(Self::Cost, bool)>> = Vec::new();
        let mut any_update_was_in_open_set_set: bool = false;

        while let Some(OpenSetElement(current, _)) = open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            let start_to_current: Self::Cost = self.cost_from_start(&current);

            open_set_set.remove(&current);
            self.neighbors(&current, &mut neighbors);
            neighbor_updates.reserve(neighbors.len());

            for neighbor in neighbors.iter() {
                let start_to_neighbor: Self::Cost =
                    start_to_current.clone() + self.cost_between_neighbors(&current, &neighbor);

                if start_to_neighbor < self.cost_from_start(&neighbor) {
                    let neighbor_heuristic: Self::Cost = self.heuristic(&neighbor);

                    self.update_score(
                        &current,
                        &neighbor,
                        start_to_neighbor.clone(),
                        neighbor_heuristic.clone(),
                    );

                    let was_in_open_set_set: bool = !open_set_set.insert(neighbor.clone());

                    neighbor_updates.push(Some((
                        start_to_neighbor + neighbor_heuristic,
                        was_in_open_set_set,
                    )));
                    any_update_was_in_open_set_set |= was_in_open_set_set;
                } else {
                    neighbor_updates.push(None);
                }
            }

            if any_update_was_in_open_set_set {
                // Convert to a vec first, add the new elements, then convert back, so that we don't
                // waste time during `push` operations only to have that effort ignored when
                // converting back to a heap
                let mut open_set_elements: Vec<OpenSetElement<Self::Vertex, Self::Cost>> =
                    take(&mut open_set_heap).into_vec();

                let old_element_count: usize = open_set_elements.len();

                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, is_in_open_set_elements)) = neighbor_update {
                        if *is_in_open_set_elements {
                            if let Some(index) = open_set_elements[..old_element_count]
                                .iter()
                                .position(|OpenSetElement(vertex, _)| *vertex == *neighbor)
                            {
                                open_set_elements[index].1 = cost.clone();
                            }
                        } else {
                            open_set_elements.push(OpenSetElement(neighbor.clone(), cost.clone()));
                        }
                    }
                }

                open_set_heap = open_set_elements.into();
            } else {
                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, _)) = neighbor_update {
                        open_set_heap.push(OpenSetElement(neighbor.clone(), cost.clone()));
                    }
                }
            }

            neighbors.clear();
            neighbor_updates.clear();
            any_update_was_in_open_set_set = false;
        }

        None
    }
}

/// An implementation of https://en.wikipedia.org/wiki/Breadth-first_search
pub trait BreadthFirstSearch: Sized {
    type Vertex: Clone + Debug + Eq + Hash;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex);

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
        let mut queue: VecDeque<Self::Vertex> = VecDeque::new();
        let mut explored: HashSet<Self::Vertex> = HashSet::new();

        let start: Self::Vertex = self.start().clone();
        explored.insert(start.clone());
        queue.push_back(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        while let Some(current) = queue.pop_front() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            self.neighbors(&current, &mut neighbors);

            for neighbor in neighbors.drain(..) {
                if explored.insert(neighbor.clone()) {
                    self.update_parent(&current, &neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}
