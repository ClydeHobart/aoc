pub use {direction::*, hex::*};

use {
    super::*,
    bitvec::prelude::*,
    glam::{BVec2, IVec2},
    nom::{
        character::complete::line_ending,
        combinator::opt,
        error::{Error as NomError, ErrorKind as NomErrorKind},
        multi::many1_count,
        sequence::tuple,
        Err,
    },
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult, Write},
        iter::Peekable,
        mem::transmute,
        ops::{Add, Range, RangeInclusive},
        str::{from_utf8, Lines},
    },
    strum::IntoEnumIterator,
};

macro_rules! define_direction {
    {
        $( #[$meta:meta] )*
        $vis:vis enum $direction:ident {
            $(
                $( #[$variant_meta:meta] )?
                $variant:ident,
            )*
        }
    } => {
        $(#[$meta])*
        $vis enum $direction {
            $(
                $( #[$variant_meta] )?
                $variant,
            )*
        }

        const VECS: [IVec2; $direction::COUNT] = [
            $( $direction::$variant.vec_internal(), )*
        ];
    };
}

mod direction {
    use {
        super::*,
        static_assertions::const_assert,
        std::mem::transmute,
        strum::{EnumCount, EnumIter},
    };

    define_direction! {
        #[derive(Copy, Clone, Debug, Default, EnumCount, EnumIter, Eq, Hash, PartialEq)]
        #[repr(u8)]
        pub enum Direction {
            #[default]
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
        pub const COUNT_U8: u8 = Self::COUNT as u8;
        pub const MASK: u8 = Self::COUNT_U8 - 1_u8;
        pub const HALF_COUNT: u8 = Self::COUNT_U8 / 2_u8;
        pub const PREV_DELTA: u8 = Self::COUNT_U8 - 1_u8;

        #[inline]
        pub const fn vec(self) -> IVec2 {
            VECS[self as usize]
        }

        #[inline]
        pub const fn from_u8(value: u8) -> Self {
            // SAFETY: See `const_assert` above
            unsafe { transmute(value & Self::MASK) }
        }

        #[inline]
        pub const fn next(self) -> Self {
            Self::from_u8(self as u8 + 1_u8)
        }

        #[inline]
        pub const fn rev(self) -> Self {
            Self::from_u8(self as u8 + Self::HALF_COUNT)
        }

        #[inline]
        pub const fn prev(self) -> Self {
            Self::from_u8(self as u8 + Self::PREV_DELTA)
        }

        pub const fn turn(self, left: bool) -> Self {
            if left {
                self.prev()
            } else {
                self.next()
            }
        }

        pub const fn is_north_or_south(self) -> bool {
            (self as u8 & 1_u8) == 0_u8
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

    impl Add<Turn> for Direction {
        type Output = Self;

        fn add(self, rhs: Turn) -> Self::Output {
            (self as u8 + rhs as u8).into()
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

    impl TryFrom<Range<IVec2>> for Direction {
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

                Ok((delta / (abs.x + abs.y)).try_into().unwrap())
            }
        }
    }

    impl TryFrom<RangeInclusive<IVec2>> for Direction {
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

                Ok((delta / (abs.x + abs.y)).try_into().unwrap())
            }
        }
    }

    #[derive(Copy, Clone, Debug, Default, EnumCount, EnumIter, Eq, Hash, PartialEq)]
    #[repr(u8)]
    pub enum Turn {
        Left = Direction::West as u8,
        #[default]
        Straight = Direction::North as u8,
        Right = Direction::East as u8,
    }

    impl Turn {
        pub fn next(self) -> Self {
            match self {
                Self::Left => Self::Straight,
                Self::Straight => Self::Right,
                Self::Right => Self::Left,
            }
        }

        pub fn prev(self) -> Self {
            match self {
                Self::Left => Self::Right,
                Self::Straight => Self::Left,
                Self::Right => Self::Straight,
            }
        }

        pub fn non_straight_opt(self) -> Option<Self> {
            (self != Self::Straight).then_some(self)
        }
    }

    impl From<Turn> for Direction {
        fn from(value: Turn) -> Self {
            match value {
                Turn::Left => Self::West,
                Turn::Straight => Self::North,
                Turn::Right => Self::East,
            }
        }
    }

    impl TryFrom<Direction> for Turn {
        type Error = ();

        fn try_from(value: Direction) -> Result<Self, Self::Error> {
            match value {
                Direction::North => Ok(Self::Straight),
                Direction::East => Ok(Self::Right),
                Direction::South => Err(()),
                Direction::West => Ok(Self::Left),
            }
        }
    }
}

mod hex {
    use {
        super::*,
        static_assertions::const_assert,
        std::mem::transmute,
        strum::{EnumCount, EnumIter},
    };

    define_direction! {
        #[derive(Copy, Clone, Debug, EnumCount, EnumIter, Eq, Hash, PartialEq)]
        #[repr(u8)]
        pub enum HexDirection {
            North,
            NorthEast,
            SouthEast,
            South,
            SouthWest,
            NorthWest,
        }
    }

    const_assert!(HexDirection::COUNT == 6_usize);

    impl HexDirection {
        pub const COUNT_U8: u8 = Self::COUNT as u8;
        pub const HALF_COUNT: u8 = Self::COUNT_U8 / 2_u8;
        pub const PREV_DELTA: u8 = Self::COUNT_U8 - 1_u8;

        #[inline]
        pub const fn vec(self) -> IVec2 {
            VECS[self as usize]
        }

        #[inline]
        pub const fn from_u8(value: u8) -> Self {
            // SAFETY: This is set to the count, but as a `u8`.
            unsafe { transmute(value % Self::COUNT_U8) }
        }

        #[inline]
        pub const fn next(self) -> Self {
            Self::from_u8(self as u8 + 1_u8)
        }

        #[inline]
        pub const fn rev(self) -> Self {
            Self::from_u8(self as u8 + Self::HALF_COUNT)
        }

        #[inline]
        pub const fn prev(self) -> Self {
            Self::from_u8(self as u8 + Self::PREV_DELTA)
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
                Self::NorthEast => IVec2::X,
                Self::SouthEast => IVec2::ONE,
                Self::South => IVec2::Y,
                Self::SouthWest => IVec2::NEG_X,
                Self::NorthWest => IVec2::NEG_ONE,
            }
        }
    }

    impl From<Direction> for HexDirection {
        fn from(value: Direction) -> Self {
            match value {
                Direction::North => Self::North,
                Direction::East => Self::NorthEast,
                Direction::South => Self::South,
                Direction::West => Self::SouthWest,
            }
        }
    }

    impl TryFrom<HexDirection> for Direction {
        type Error = ();

        fn try_from(value: HexDirection) -> Result<Self, Self::Error> {
            match value {
                HexDirection::North => Ok(Self::North),
                HexDirection::NorthEast => Ok(Self::East),
                HexDirection::SouthEast => Err(()),
                HexDirection::South => Ok(Self::South),
                HexDirection::SouthWest => Ok(Self::West),
                HexDirection::NorthWest => Err(()),
            }
        }
    }

    impl From<HexDirection> for IVec2 {
        fn from(value: HexDirection) -> Self {
            value.vec()
        }
    }

    impl From<u8> for HexDirection {
        fn from(value: u8) -> Self {
            Self::from_u8(value)
        }
    }

    impl TryFrom<IVec2> for HexDirection {
        type Error = ();

        fn try_from(value: IVec2) -> Result<Self, Self::Error> {
            VECS.iter()
                .position(|vec| *vec == value)
                .map(|index| (index as u8).into())
                .ok_or(())
        }
    }

    pub fn hex_manhattan_magnitude(pos: IVec2) -> i32 {
        if pos.cmpeq(IVec2::ZERO).any() || pos.x * pos.y < 0_i32 {
            // If either of the components is 0, just travel directly to `pos`. Otherwise, if the
            // product of the components is negative, it's in a quadrant with no diagonal routes
            // that accelerate travel, leaving only orthogonal segments.
            manhattan_magnitude_2d(pos)
        } else {
            // Travel diagonally as much as possible (`pos.abs().min_element()`), then orthogonally
            // the rest of the way (`pos.abs().max_element() - pos.abs().min_element()`).
            pos.abs().max_element()
        }
    }

    pub fn hex_manhattan_distance(a: IVec2, b: IVec2) -> i32 {
        hex_manhattan_magnitude(a - b)
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

pub fn grid_2d_contains(pos: IVec2, dimensions: IVec2) -> bool {
    (pos.cmpge(IVec2::ZERO) & pos.cmplt(dimensions)).all()
}

pub fn grid_2d_pos_from_index_and_dimensions(index: usize, dimensions: IVec2) -> IVec2 {
    let x: usize = dimensions.x as usize;

    IVec2::new((index % x) as i32, (index / x) as i32)
}

pub fn grid_2d_try_index_from_pos_and_dimensions(pos: IVec2, dimensions: IVec2) -> Option<usize> {
    grid_2d_contains(pos, dimensions)
        .then(|| pos.y as usize * dimensions.x as usize + pos.x as usize)
}

pub struct Grid2D<T> {
    cells: Vec<T>,

    /// Should only contain unsigned values, but is signed for ease of use for iterating
    dimensions: IVec2,
}

impl<T> Grid2D<T> {
    pub fn try_from_cells_and_dimensions(cells: Vec<T>, dimensions: IVec2) -> Option<Self> {
        if dimensions.cmpge(IVec2::ZERO) == BVec2::TRUE
            && cells.len() == dimensions.x as usize * dimensions.y as usize
        {
            Some(Self { cells, dimensions })
        } else {
            None
        }
    }

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
    pub fn dimensions(&self) -> IVec2 {
        self.dimensions
    }

    #[inline]
    pub fn area(&self) -> usize {
        (self.dimensions.x * self.dimensions.y) as usize
    }

    #[inline]
    pub fn contains(&self, pos: IVec2) -> bool {
        grid_2d_contains(pos, self.dimensions)
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
        grid_2d_try_index_from_pos_and_dimensions(pos, self.dimensions)
    }

    pub fn pos_from_index(&self, index: usize) -> IVec2 {
        grid_2d_pos_from_index_and_dimensions(index, self.dimensions)
    }

    #[inline(always)]
    pub fn max_dimensions(&self) -> IVec2 {
        self.dimensions - IVec2::ONE
    }

    pub fn get(&self, pos: IVec2) -> Option<&T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &self.cells[index])
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = IVec2> {
        let dimensions: IVec2 = self.dimensions;

        CellIter2D::try_from(IVec2::ZERO..IVec2::new(0_i32, dimensions.y))
            .unwrap()
            .flat_map(move |pos| {
                CellIter2D::try_from(pos..IVec2::new(dimensions.x, pos.y)).unwrap()
            })
    }

    pub fn iter_filtered_positions<'a, P: Fn(&T) -> bool + 'a>(
        &'a self,
        predicate: P,
    ) -> impl Iterator<Item = IVec2> + 'a {
        self.cells
            .iter()
            .enumerate()
            .filter_map(move |(index, cell)| predicate(cell).then(|| self.pos_from_index(index)))
    }

    pub fn iter_positions_with_cell<'a>(&'a self, target: &'a T) -> impl Iterator<Item = IVec2> + 'a
    where
        T: PartialEq,
    {
        self.iter_filtered_positions(|cell| *cell == *target)
    }

    pub fn try_find_single_position_with_cell(&self, target: &T) -> Option<IVec2>
    where
        T: PartialEq,
    {
        self.iter_positions_with_cell(target)
            .try_fold(None, |prev_pos, curr_pos| {
                prev_pos.is_none().then_some(Some(curr_pos))
            })
            .flatten()
    }

    #[inline]
    pub fn cells_mut(&mut self) -> &mut [T] {
        &mut self.cells
    }

    pub fn get_mut(&mut self, pos: IVec2) -> Option<&mut T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &mut self.cells[index])
    }

    pub fn resize_rows(&mut self, new_row_len: usize, value: T)
    where
        T: Clone,
    {
        self.resize_rows_with(new_row_len, || value.clone());
    }

    pub fn resize_rows_with<F: FnMut() -> T>(&mut self, new_row_len: usize, f: F) {
        self.dimensions.y = new_row_len as i32;
        self.cells.resize_with(self.area(), f);
    }

    pub fn clear_and_resize(&mut self, dimensions: IVec2, value: T)
    where
        T: Clone,
    {
        if self.dimensions.x == dimensions.x {
            self.resize_rows(dimensions.y as usize, value)
        } else {
            let old_area: usize = self.area();

            self.dimensions = dimensions;
            self.cells.resize(self.area(), value.clone());

            let old_len: usize = old_area.min(self.area());

            self.cells[..old_len].fill(value);
        }
    }

    pub fn double_dimensions(&mut self, value: T)
    where
        T: Clone,
    {
        let row_len: usize = self.dimensions.x as usize;
        let new_dimensions: IVec2 = 2_i32 * self.dimensions;

        // This is just used for `index_from_pos`.
        let new_self: Self = Self {
            cells: Vec::new(),
            dimensions: new_dimensions,
        };

        self.cells.resize(
            new_dimensions.x as usize * new_dimensions.y as usize,
            value.clone(),
        );

        // Move the old rows into their new spots.
        for y in (1_i32..self.dimensions.y).rev() {
            let row_pos: IVec2 = y * IVec2::Y;
            let old_row_start: usize = self.index_from_pos(row_pos);
            let new_row_start: usize = new_self.index_from_pos(row_pos);
            let (old_row_slice, new_row_slice): (&mut [T], &mut [T]) =
                self.cells.split_at_mut(new_row_start);

            new_row_slice[..row_len]
                .clone_from_slice(&old_row_slice[old_row_start..old_row_start + row_len]);
        }

        // Initialize the new columns within the bounds of the old rows.
        for y in 0_i32..self.dimensions.y {
            let new_row_start: usize = new_self.index_from_pos(IVec2::new(self.dimensions.x, y));

            self.cells[new_row_start..new_row_start + row_len].fill(value.clone());
        }

        self.dimensions = new_dimensions;
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

impl<T: Parse> Parse for Grid2D<T> {
    fn parse(input: &str) -> IResult<&str, Self> {
        let mut width: Option<usize> = None;
        let mut cells: Vec<T> = Vec::new();
        let (input, _) = many1_count(map_res(
            tuple((T::parse, opt(line_ending))),
            |(cell, opt_line_ending)| -> Result<(), ()> {
                cells.push(cell);

                if opt_line_ending.is_some() {
                    match width {
                        Some(width) => {
                            if cells.len() % width != 0_usize {
                                Err(())?;
                            }
                        }
                        None => {
                            width = Some(cells.len());
                        }
                    }
                }

                Ok(())
            },
        ))(input)?;

        if let Some(width) = width {
            if cells.len() % width != 0_usize {
                Err(Err::Failure(NomError::new(input, NomErrorKind::ManyMN)))
            } else {
                Ok((
                    input,
                    Grid2D::try_from_cells_and_width(cells, width).unwrap(),
                ))
            }
        } else {
            let width: usize = cells.len();

            Ok((
                input,
                Grid2D::try_from_cells_and_width(cells, width).unwrap(),
            ))
        }
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

        let mut grid_line_iter: Peekable<Lines> = grid_str.lines().peekable();

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
    pub fn corner_for_dimensions(dimensions: IVec2, dir: Direction) -> Self {
        let dir_vec: IVec2 = dir.vec();
        let curr: IVec2 =
            (-dimensions * (dir_vec + dir_vec.perp())).clamp(IVec2::ZERO, dimensions - IVec2::ONE);

        Self::until_boundary_for_dimensions(dimensions, curr, dir)
    }

    pub fn corner<T>(grid: &Grid2D<T>, dir: Direction) -> Self {
        Self::corner_for_dimensions(grid.dimensions(), dir)
    }

    pub fn until_boundary_for_dimensions(dimensions: IVec2, curr: IVec2, dir: Direction) -> Self {
        let dir_vec: IVec2 = dir.vec();
        let end: IVec2 =
            (curr + dir_vec * dimensions).clamp(IVec2::ZERO, dimensions - IVec2::ONE) + dir_vec;

        Self { curr, end, dir }
    }

    pub fn until_boundary<T>(grid: &Grid2D<T>, curr: IVec2, dir: Direction) -> Self {
        Self::until_boundary_for_dimensions(grid.dimensions(), curr, dir)
    }

    pub fn iter_edges_for_dimensions(dimensions: IVec2) -> impl Iterator<Item = IVec2> {
        let take_dimensions: IVec2 = dimensions - IVec2::ONE;

        Direction::iter().flat_map(move |dir| {
            // Use `dir.next()` so that it starts at (0, 0) and wraps around clockwise from there
            let dir: Direction = dir.next();

            Self::corner_for_dimensions(dimensions, dir)
                .take(manhattan_magnitude_2d(dir.vec() * take_dimensions) as usize)
        })
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

    fn try_from(range: Range<IVec2>) -> Result<Self, Self::Error> {
        let curr: IVec2 = range.start;
        let end: IVec2 = range.end;

        Direction::try_from(range).map(|dir| Self { curr, end, dir })
    }
}

impl TryFrom<RangeInclusive<IVec2>> for CellIter2D {
    type Error = CellIterFromRangeError;

    fn try_from(range_inclusive: RangeInclusive<IVec2>) -> Result<Self, Self::Error> {
        let curr: IVec2 = *range_inclusive.start();
        let end: IVec2 = *range_inclusive.end();

        Direction::try_from(range_inclusive).map(|dir| Self {
            curr,
            end: end + dir.vec(),
            dir,
        })
    }
}

/// A marker trait to indicate that a type is a single byte, and any possible value is a valid ASCII
/// byte.
///
/// # Safety
///
/// Only implement this on a trait that meets the following criteria:
///
/// * `std::mem::size_of::<Self>() == 1_usize`
/// * `std::str::from_utf8(std::mem::transmute::<[Self], [u8]>(value)).is_ok()` for any `value:
/// [Self]`.
pub unsafe trait IsValidAscii {}

impl<T: IsValidAscii> From<Grid2D<T>> for String {
    fn from(value: Grid2D<T>) -> Self {
        let dimensions: IVec2 = value.dimensions;
        let width: usize = dimensions.x as usize;
        let height: usize = dimensions.y as usize;

        // SAFETY: Guaranteed by `T` implementing `IsValidAscii`
        let bytes: &[u8] = unsafe { transmute(value.cells()) };

        let mut string: String = String::with_capacity((width + 1_usize) * height);

        for y in 0_usize..height {
            let start: usize = y * width;
            let end: usize = start + width;
            let row_str: &str = from_utf8(&bytes[start..end]).unwrap_or_else(|e| {
                panic!("A `Grid2DString` contained an invalid UTF-8 slice: {e:?}");
            });

            writeln!(&mut string, "{row_str}").unwrap_or_else(|e| {
                panic!(
                    "`String::write_fmt` returned an `Err`, despite both its `write_str` and 
                    `write_char` definitions returning an `Ok`: {e:?}"
                );
            });
        }

        string
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

pub fn manhattan_magnitude_2d(pos: IVec2) -> i32 {
    let abs: IVec2 = pos.abs();

    abs.x + abs.y
}

pub fn manhattan_distance_2d(a: IVec2, b: IVec2) -> i32 {
    manhattan_magnitude_2d(a - b)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SmallPos {
    pub x: u8,
    pub y: u8,
}

impl SmallPos {
    pub const MIN_POS: IVec2 = IVec2::ZERO;
    pub const MAX_POS: IVec2 = IVec2::new(u8::MAX as i32, u8::MAX as i32);
    pub const MAX_DIMENSIONS: IVec2 = IVec2::new(Self::MAX_POS.x + 1_i32, Self::MAX_POS.y + 1_i32);

    /// SAFETY: This will panic if either component can't be converted to a `u8`
    pub unsafe fn from_pos_unsafe(pos: IVec2) -> Self {
        Self {
            x: pos.x as u8,
            y: pos.y as u8,
        }
    }

    pub fn is_pos_valid(pos: IVec2) -> bool {
        grid_2d_contains(pos, Self::MAX_DIMENSIONS)
    }

    pub fn try_from_pos(pos: IVec2) -> Option<Self> {
        // SAFETY: `pos` has been verified.
        Self::is_pos_valid(pos).then(|| unsafe { Self::from_pos_unsafe(pos) })
    }

    pub fn from_sortable_index(sortable_index: u16) -> Self {
        Self {
            x: (sortable_index & 0xFF_u16) as u8,
            y: (sortable_index >> u8::BITS) as u8,
        }
    }

    pub fn get(self) -> IVec2 {
        IVec2::new(self.x as i32, self.y as i32)
    }

    pub fn sortable_index_from_components(x: u8, y: u8) -> u16 {
        ((y as u16) << u8::BITS) | x as u16
    }

    pub fn sortable_index(self) -> u16 {
        Self::sortable_index_from_components(self.x, self.y)
    }

    pub unsafe fn sortable_index_from_pos_unsafe(pos: IVec2) -> u16 {
        Self::sortable_index_from_components(pos.x as u8, pos.y as u8)
    }

    pub fn try_set(&mut self, pos: IVec2) -> bool {
        if Self::is_pos_valid(pos) {
            // SAFETY: `pos` is valid.
            *self = unsafe { Self::from_pos_unsafe(pos) };

            true
        } else {
            false
        }
    }
}

pub type SmallPosBitArr =
    BitArr!(for (SmallPos::MAX_DIMENSIONS.x * SmallPos::MAX_DIMENSIONS.y) as usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SmallPosAndDir {
    pub pos: SmallPos,
    pub dir: Direction,
}

impl SmallPosAndDir {
    /// SAFETY: This will panic if either component can't be converted to a `u8`
    pub unsafe fn from_pos_and_dir_unsafe(pos: IVec2, dir: Direction) -> Self {
        Self {
            pos: SmallPos::from_pos_unsafe(pos),
            dir,
        }
    }

    pub fn try_from_pos_and_dir(pos: IVec2, dir: Direction) -> Option<Self> {
        // SAFETY: `pos` has been verified.
        SmallPos::is_pos_valid(pos).then(|| unsafe { Self::from_pos_and_dir_unsafe(pos, dir) })
    }

    pub fn from_sortable_index_and_dir(sortable_index: u16, dir: Direction) -> Self {
        Self {
            pos: SmallPos::from_sortable_index(sortable_index),
            dir,
        }
    }
}

pub fn sortable_index_from_pos(pos: IVec2) -> u64 {
    const TOGGLE_MASK: u64 = (1_u64 << (u64::BITS - 1_u32)) | (1_u64 << (u32::BITS - 1_u32));

    // SAFETY: Trivial.
    ((unsafe { transmute::<i32, u32>(pos.y) } as u64) << u32::BITS
        | unsafe { transmute::<i32, u32>(pos.x) } as u64)
        ^ TOGGLE_MASK
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
