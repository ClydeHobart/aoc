use {
    self::{cell_iter::*, direction::*},
    aoc_2022::*,
    clap::Parser,
    glam::IVec2,
    static_assertions::const_assert,
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        iter::Peekable,
        mem::transmute,
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

            const VECS: [IVec2; $direction::COUNT] = [
                $( $direction::$variant.vec_internal(), )*
            ];
        };
    }

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

mod cell_iter {
    use super::*;

    pub struct CellIter {
        dir: Direction,
        curr: IVec2,
        end: IVec2,
    }

    impl CellIter {
        pub(super) fn corner<T>(grid: &Grid<T>, dir: Direction) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let curr: IVec2 = (-grid.dimensions * (dir_vec + dir_vec.perp()))
                .clamp(IVec2::ZERO, grid.max_dimensions());

            Self::until(grid, dir, curr)
        }

        #[inline(always)]
        pub(super) fn until<T>(grid: &Grid<T>, dir: Direction, curr: IVec2) -> Self {
            let dir_vec: IVec2 = dir.vec();
            let end: IVec2 = (curr + dir_vec * grid.dimensions)
                .clamp(IVec2::ZERO, grid.max_dimensions())
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

struct SideLen(usize);

impl From<SideLen> for IVec2 {
    fn from(side_len: SideLen) -> Self {
        IVec2::new(side_len.0 as i32, side_len.0 as i32)
    }
}

struct Grid<T> {
    cells: Vec<T>,

    /// Should only contain unsigned values, but is signed for ease of use for iterating
    dimensions: IVec2,
}

impl<T> Grid<T> {
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
    fn max_dimensions(&self) -> IVec2 {
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
    fn default(dimensions: IVec2) -> Self {
        let capacity: usize = (dimensions.x * dimensions.y) as usize;
        let mut cells: Vec<T> = Vec::with_capacity(capacity);

        cells.resize_with(capacity, T::default);

        Self { cells, dimensions }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum GridParseError<'s, E> {
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

trait GridVisitor: Default + Sized {
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
        let mut new_grid: Grid<Self::New> = Grid::default(old_grid.dimensions);

        for dir in Direction::iter() {
            let row_dir: Direction = dir.next();

            // Look back the way we came to make the most use of the local `GridVisitor`
            let rev_dir: Direction = (row_dir as usize + 2_usize).into();

            for row_pos in CellIter::corner(old_grid, dir) {
                let mut grid_visitor: Self = Self::default();

                for pos in CellIter::until(old_grid, row_dir, row_pos) {
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

#[derive(Debug)]
struct Height(u8);

#[derive(Debug)]
struct CharIsNotAsciiDigit(char);

impl TryFrom<char> for Height {
    type Error = CharIsNotAsciiDigit;

    fn try_from(height_char: char) -> Result<Self, Self::Error> {
        if height_char.is_ascii_digit() {
            Ok(Height(height_char as u8 - ZERO_OFFSET))
        } else {
            Err(CharIsNotAsciiDigit(height_char))
        }
    }
}

#[derive(Debug, Default)]
struct IsVisible(bool);

#[derive(Default)]
struct ComputeIsVisible {
    max_row_height: u8,
}

impl GridVisitor for ComputeIsVisible {
    type Old = Height;
    type New = IsVisible;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        old_grid: &Grid<Self::Old>,
        _rev_dir: Direction,
        pos: IVec2,
    ) {
        if !new.0 {
            new.0 = old_grid.is_border(pos) || old.0 > self.max_row_height
        }

        self.max_row_height = self.max_row_height.max(old.0);
    }
}

impl Grid<IsVisible> {
    fn count(&self) -> usize {
        self.cells
            .iter()
            .filter(|is_visible: &&IsVisible| is_visible.0)
            .count()
    }
}

#[derive(Debug)]
struct ScenicScore(u32);

impl Default for ScenicScore {
    fn default() -> Self {
        Self(1_u32)
    }
}

#[derive(Default)]
struct ComputeScenicScore {
    height_to_viewing_distance: [u32; 10_usize],
}

impl GridVisitor for ComputeScenicScore {
    type Old = Height;
    type New = ScenicScore;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        _old_grid: &Grid<Self::Old>,
        _rev_dir: Direction,
        _pos: IVec2,
    ) {
        let height_index: usize = old.0 as usize;

        if new.0 != 0_u32 {
            new.0 *= self.height_to_viewing_distance[height_index];
        }

        // All cells not taller than this cell can now only see this one
        self.height_to_viewing_distance[..=height_index].fill(1_u32);

        if let Some(taller_slice) = self
            .height_to_viewing_distance
            .get_mut(height_index + 1_usize..)
        {
            for taller_viewing_distance in taller_slice {
                *taller_viewing_distance += 1_u32;
            }
        }
    }
}

impl Grid<ScenicScore> {
    fn max(&self) -> u32 {
        self.cells
            .iter()
            .map(|scenic_score: &ScenicScore| scenic_score.0)
            .max()
            .unwrap_or_default()
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day8.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match Grid::<Height>::try_from(input) {
                    Ok(height_grid) => {
                        println!(
                            "ComputeIsVisible::visit_grid(&height_grid).count() == {}\n\
                            ComputeScenicScore::visit_grid(&height_grid).max() == {}",
                            ComputeIsVisible::visit_grid(&height_grid).count(),
                            ComputeScenicScore::visit_grid(&height_grid).max()
                        );
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                },
            )
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

    match Grid::<Height>::try_from(HEIGHT_GRID_STR) {
        Ok(height_grid) => {
            let is_visible_grid: Grid<IsVisible> = ComputeIsVisible::visit_grid(&height_grid);

            assert_eq!(
                is_visible_grid.count(),
                21_usize,
                "height_grid: {height_grid:#?}\n\nis_visible_grid: {is_visible_grid:#?}"
            );

            let scenic_score_grid: Grid<ScenicScore> = ComputeScenicScore::visit_grid(&height_grid);

            assert_eq!(
                scenic_score_grid.max(),
                8_u32,
                "height_grid: {height_grid:#?}\n\ncenic_score_grid: {scenic_score_grid:#?}"
            );
        }
        Err(error) => panic!("{error:#?}"),
    }
}
