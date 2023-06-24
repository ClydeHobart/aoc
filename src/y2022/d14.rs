use {
    crate::*,
    glam::IVec2,
    static_assertions::const_assert_eq,
    std::{
        fmt::{Error as FmtError, Write},
        mem::{align_of, size_of, transmute},
        num::ParseIntError,
        str::{from_utf8_unchecked, FromStr, Split},
    },
};

const SAND_SOURCE: IVec2 = IVec2::new(500_i32, 0_i32);

#[derive(Debug, PartialEq)]
struct Scan {
    points: Vec<IVec2>,
    path_boundaries: Vec<usize>,
    min: IVec2,
    max: IVec2,
}

impl Scan {
    fn add_point(&mut self, point: IVec2) {
        self.max = self.max.max(point);
        self.min = self.min.min(point);
        self.points.push(point);
    }

    fn add_path_boundary(&mut self) {
        self.path_boundaries.push(self.points.len());
    }

    fn add_floor(&mut self) {
        let floor_center: IVec2 = IVec2::new(SAND_SOURCE.x, self.max.y + 2_i32);
        let floor_start: IVec2 = floor_center + (floor_center - SAND_SOURCE).perp();

        self.add_point(floor_start);
        self.add_point(2_i32 * floor_center - floor_start);
        self.add_path_boundary();
    }

    fn iter_lines(&self) -> impl Iterator<Item = &[IVec2]> {
        (0_usize..self.path_boundaries.len() - 1_usize)
            .into_iter()
            .map(|path_boundary_index| {
                (self.path_boundaries[path_boundary_index]
                    ..self.path_boundaries[path_boundary_index + 1_usize] - 1_usize)
                    .into_iter()
                    .map(|point_index| &self.points[point_index..point_index + 2_usize])
            })
            .flatten()
    }

    fn scan_grid(&self) -> Result<ScanGrid, ScanGridFromScanError> {
        use {ScanCell::*, ScanGridFromScanError::*};

        let mut scan_grid: ScanGrid = ScanGrid {
            grid: Grid2D::default(self.max - self.min + IVec2::ONE),
            source: SAND_SOURCE - self.min,
        };

        for line in self.iter_lines() {
            for pos in CellIter2D::try_from(line[0_usize]..=line[1_usize]).unwrap() {
                *scan_grid.grid.get_mut(pos - self.min).unwrap() = Rock;
            }
        }

        let source: &mut ScanCell = scan_grid.grid.get_mut(scan_grid.source).unwrap();

        match source {
            Air => {
                *source = Source;

                Ok(scan_grid)
            }
            Rock => Err(RockPresentAtSource),
            _ => unreachable!(),
        }
    }
}

impl Default for Scan {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            path_boundaries: Vec::new(),
            min: SAND_SOURCE,
            max: SAND_SOURCE,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ScanParseError<'s> {
    NoXToken(&'s str),
    FailedToParseX(ParseIntError),
    NoYToken(&'s str),
    FailedToParseY(ParseIntError),
    ExtraTokenFound(&'s str),
    PathIsTooShort { path: &'s str, len: usize },
    LineIsNotHorizontalNorVertical { start: IVec2, end: IVec2 },
}

impl<'s> TryFrom<&'s str> for Scan {
    type Error = ScanParseError<'s>;

    fn try_from(scan_str: &'s str) -> Result<Self, Self::Error> {
        use ScanParseError::*;

        let mut scan: Scan = Scan::default();

        for path in scan_str.split('\n') {
            scan.add_path_boundary();

            for point_str in path.split(" -> ") {
                let mut component_token_iter: Split<char> = point_str.split(',');

                let x: i32 = match component_token_iter.next() {
                    None => Err(NoXToken(point_str)),
                    Some(x_str) => i32::from_str(x_str).map_err(FailedToParseX),
                }?;
                let y: i32 = match component_token_iter.next() {
                    None => Err(NoYToken(point_str)),
                    Some(y_str) => i32::from_str(y_str).map_err(FailedToParseY),
                }?;
                let point: IVec2 = match component_token_iter.next() {
                    Some(extra_token) => Err(ExtraTokenFound(extra_token)),
                    None => Ok(IVec2::new(x, y)),
                }?;

                scan.add_point(point);
            }

            let len: usize = scan.points.len() - *scan.path_boundaries.last().unwrap();

            if len < 2_usize {
                return Err(PathIsTooShort { path, len });
            }
        }

        scan.add_path_boundary();

        for line in scan.iter_lines() {
            let start: IVec2 = line[0_usize];
            let end: IVec2 = line[1_usize];
            let delta: IVec2 = end - start;

            if delta.x != 0_i32 && delta.y != 0_i32 {
                return Err(LineIsNotHorizontalNorVertical { start, end });
            }
        }

        Ok(scan)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
enum ScanCell {
    Source = b'+',
    #[default]
    Air = b'.',
    Rock = b'#',
    Sand = b'o',
}

#[cfg_attr(test, derive(Clone))]
#[derive(Debug, PartialEq)]
struct ScanGrid {
    grid: Grid2D<ScanCell>,
    source: IVec2,
}

#[derive(Debug, PartialEq)]
enum ScanGridFromScanError {
    RockPresentAtSource,
}

#[cfg(test)]
#[derive(Debug, PartialEq)]
enum ScanGridParseError<'s> {
    FailedToParseScan(ScanParseError<'s>),
    FailedToConvertFromScan(ScanGridFromScanError),
}

#[derive(Debug, PartialEq)]
enum AddSandError {
    FellOutOfBounds,
    BlockedSource,
}

impl ScanGrid {
    #[cfg(test)]
    fn try_from_str(scan_str: &str, with_floor: bool) -> Result<Self, ScanGridParseError> {
        use ScanGridParseError::*;

        let mut scan: Scan = Scan::try_from(scan_str).map_err(FailedToParseScan)?;

        if with_floor {
            scan.add_floor();
        }

        scan.scan_grid().map_err(FailedToConvertFromScan)
    }

    fn string(&self) -> Result<String, FmtError> {
        let dimensions: IVec2 = self.grid.dimensions();
        let width: usize = dimensions.x as usize;
        let height: usize = dimensions.y as usize;

        const_assert_eq!(size_of::<ScanCell>(), size_of::<u8>());
        const_assert_eq!(align_of::<ScanCell>(), align_of::<u8>());

        // SAFETY: The `const_assert_eq`s above guarantee size and alignment
        let bytes: &[u8] = unsafe { transmute(self.grid.cells()) };

        let mut string: String = String::with_capacity((width + 1_usize) * height);

        for y in 0_usize..height {
            let start: usize = y * width;
            let end: usize = start + width;

            write!(
                &mut string,
                "{}\n",
                // SAFETY: bytes comes from a slice of `ScanCell` elements, all of which have a
                // valid ASCII byte as their internal value
                unsafe { from_utf8_unchecked(&bytes[start..end]) }
            )?;
        }

        Ok(string)
    }

    fn step(&self, sand: IVec2) -> Option<IVec2> {
        const DELTA_CANDIDATES: [IVec2; 3_usize] =
            [IVec2::Y, IVec2::new(-1_i32, 1_i32), IVec2::ONE];

        DELTA_CANDIDATES
            .iter()
            .map(|delta_candidate| sand + *delta_candidate)
            .find(|sand_candidate| {
                self.grid
                    .get(*sand_candidate)
                    .copied()
                    .unwrap_or(ScanCell::Air)
                    == ScanCell::Air
            })
    }

    fn add_sand_unit(&mut self) -> Result<(), AddSandError> {
        use {AddSandError::*, ScanCell::*};

        let mut sand: IVec2 = self.source;

        while let Some(new_sand) = self.step(sand) {
            if self.grid.get(new_sand).is_none() {
                return Err(FellOutOfBounds);
            }

            sand = new_sand;
        }

        let scan_cell: &mut ScanCell = self.grid.get_mut(sand).unwrap();

        match scan_cell {
            Air => Ok(*scan_cell = Sand),
            Source => Err(BlockedSource),
            other => unreachable!(
                "Encountered unexpected `ScanCell::{other:?}` \
                while adding sand"
            ),
        }
    }

    fn add_sand_units(&mut self, units: usize) -> Result<(), (usize, AddSandError)> {
        for unit in 0_usize..units {
            self.add_sand_unit()
                .map_err(|add_sand_error| (unit, add_sand_error))?;
        }

        Ok(())
    }

    fn add_all_sand_units(&mut self) -> (usize, AddSandError) {
        self.add_sand_units(usize::MAX).unwrap_err()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Scan);

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        let mut floorless_scan_grid: Result<ScanGrid, ScanGridFromScanError> = self.0.scan_grid();

        dbg!(floorless_scan_grid
            .as_mut()
            .map(ScanGrid::add_all_sand_units))
        .ok();

        if args.verbose {
            let floorless_scan_grid_string: String = floorless_scan_grid
                .as_ref()
                .map(ScanGrid::string)
                .map(Result::unwrap_or_default)
                .unwrap_or_default();

            println!("floorless_scan_grid_string:\n\"\"\"\n{floorless_scan_grid_string}\n\"\"\"");
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        self.0.add_floor();

        let mut floorful_scan_grid_string: Result<ScanGrid, ScanGridFromScanError> =
            self.0.scan_grid();

        dbg!(floorful_scan_grid_string
            .as_mut()
            .map(ScanGrid::add_all_sand_units))
        .ok();

        if args.verbose {
            let floorful_scan_grid_string: String = floorful_scan_grid_string
                .as_ref()
                .map(ScanGrid::string)
                .map(Result::unwrap_or_default)
                .unwrap_or_default();

            println!("floorless_scan_grid_string:\n\"\"\"\n{floorful_scan_grid_string}\n\"\"\"");
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = ScanParseError<'i>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SCAN_STR: &str = concat!(
        "498,4 -> 498,6 -> 496,6\n",
        "503,4 -> 502,4 -> 502,9 -> 494,9",
    );
    const SCAN_GRID_0_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "..........\n",
        "..........\n",
        "....#...##\n",
        "....#...#.\n",
        "..###...#.\n",
        "........#.\n",
        "........#.\n",
        "#########.\n",
    );
    const SCAN_GRID_1_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "..........\n",
        "..........\n",
        "....#...##\n",
        "....#...#.\n",
        "..###...#.\n",
        "........#.\n",
        "......o.#.\n",
        "#########.\n",
    );
    const SCAN_GRID_2_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "..........\n",
        "..........\n",
        "....#...##\n",
        "....#...#.\n",
        "..###...#.\n",
        "........#.\n",
        ".....oo.#.\n",
        "#########.\n",
    );
    const SCAN_GRID_5_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "..........\n",
        "..........\n",
        "....#...##\n",
        "....#...#.\n",
        "..###...#.\n",
        "......o.#.\n",
        "....oooo#.\n",
        "#########.\n",
    );
    const SCAN_GRID_22_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "......o...\n",
        ".....ooo..\n",
        "....#ooo##\n",
        "....#ooo#.\n",
        "..###ooo#.\n",
        "....oooo#.\n",
        "...ooooo#.\n",
        "#########.\n",
    );
    const SCAN_GRID_24_STR: &str = concat!(
        "......+...\n",
        "..........\n",
        "......o...\n",
        ".....ooo..\n",
        "....#ooo##\n",
        "...o#ooo#.\n",
        "..###ooo#.\n",
        "....oooo#.\n",
        ".o.ooooo#.\n",
        "#########.\n",
    );
    const SCAN_GRID_DIMENSIONS: IVec2 = IVec2::new(10_i32, 10_i32);
    const WIDTH: usize = SCAN_GRID_DIMENSIONS.x as usize;
    const HEIGHT: usize = SCAN_GRID_DIMENSIONS.y as usize;

    #[test]
    fn test_solution_try_from_str() {
        assert_eq!(SCAN_STR.try_into().as_ref(), Ok(solution()));
    }

    #[test]
    fn test_scan_scan_grid() {
        assert_eq!(solution().0.scan_grid().as_ref(), Ok(scan_grid()));
    }

    #[test]
    fn test_scan_grid_try_from_str() {
        assert_eq!(
            ScanGrid::try_from_str(SCAN_STR, false).as_ref(),
            Ok(scan_grid())
        );
    }

    #[test]
    fn test_scan_grid_string() {
        assert_eq!(scan_grid().string(), Ok(SCAN_GRID_0_STR.into()))
    }

    #[test]
    fn test_scan_grid_add_sand() {
        use AddSandError::*;

        let mut scan_grid: ScanGrid = scan_grid().clone();

        assert_eq!(scan_grid.add_sand_unit(), Ok(()));
        assert_eq!(scan_grid.string(), Ok(SCAN_GRID_1_STR.into()));
        assert_eq!(scan_grid.add_sand_unit(), Ok(()));
        assert_eq!(scan_grid.string(), Ok(SCAN_GRID_2_STR.into()));
        assert_eq!(scan_grid.add_sand_units(3_usize), Ok(()));
        assert_eq!(scan_grid.string(), Ok(SCAN_GRID_5_STR.into()));
        assert_eq!(scan_grid.add_sand_units(17_usize), Ok(()));
        assert_eq!(scan_grid.string(), Ok(SCAN_GRID_22_STR.into()));
        assert_eq!(scan_grid.add_sand_units(2_usize), Ok(()));
        assert_eq!(scan_grid.string(), Ok(SCAN_GRID_24_STR.into()));
        assert_eq!(scan_grid.add_sand_unit(), Err(FellOutOfBounds));
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
        macro_rules! points { [$($x:expr, $y:expr)=>*] => { vec![ $( IVec2::new($x, $y) ),*] }; }

        Solution(Scan {
            points: [
                points![498,4 => 498,6 => 496,6],
                points![503,4 => 502,4 => 502,9 => 494,9],
            ]
            .into_iter()
            .map(Vec::<IVec2>::into_iter)
            .flatten()
            .collect(),
            path_boundaries: vec![0_usize, 3_usize, 7_usize],
            min: IVec2::new(494_i32, 0_i32),
            max: IVec2::new(503_i32, 9_i32),
        })})
    }

    fn scan_grid() -> &'static ScanGrid {
        static ONCE_LOCK: OnceLock<ScanGrid> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            use ScanCell::{Air as A, Rock as R, Source as X};

            let cells: [[ScanCell; WIDTH]; HEIGHT] = [
                [A, A, A, A, A, A, X, A, A, A],
                [A, A, A, A, A, A, A, A, A, A],
                [A, A, A, A, A, A, A, A, A, A],
                [A, A, A, A, A, A, A, A, A, A],
                [A, A, A, A, R, A, A, A, R, R],
                [A, A, A, A, R, A, A, A, R, A],
                [A, A, R, R, R, A, A, A, R, A],
                [A, A, A, A, A, A, A, A, R, A],
                [A, A, A, A, A, A, A, A, R, A],
                [R, R, R, R, R, R, R, R, R, A],
            ];

            let mut scan_grid: ScanGrid = ScanGrid {
                grid: Grid2D::default(SCAN_GRID_DIMENSIONS),
                source: IVec2::new(6_i32, 0_i32),
            };

            for (src, dest) in cells
                .iter()
                .map(|row| row.iter())
                .flatten()
                .zip(scan_grid.grid.cells_mut().iter_mut())
            {
                *dest = *src;
            }

            scan_grid
        })
    }
}
