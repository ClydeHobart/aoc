pub use direction::*;

use {
    super::*,
    glam::IVec2,
    std::{
        fmt::{Debug, DebugList, Error as FmtError, Formatter, Result as FmtResult, Write},
        iter::Peekable,
        ops::{Range, RangeInclusive},
        str::{from_utf8, Split, Utf8Error},
    },
    strum::IntoEnumIterator,
};

mod direction {
    use {
        super::*,
        static_assertions::const_assert,
        std::mem::transmute,
        strum::{EnumCount, EnumIter},
    };

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