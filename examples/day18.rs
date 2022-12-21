use {
    aoc_2022::*,
    glam::{BVec4, IVec3},
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        mem::{size_of, transmute},
        num::ParseIntError,
        str::{from_utf8_unchecked, FromStr, Split},
    },
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C, align(1))]
struct LavaDropletScanCell {
    exposed_x: bool,
    exposed_y: bool,
    exposed_z: bool,
    occupied: bool,
}

impl LavaDropletScanCell {
    const SIZE: usize = size_of::<LavaDropletScanCell>();
    const BYTES: [u8; LavaDropletScanCell::SIZE] = *b"XYZO";

    fn as_utf8_bytes(self) -> [u8; LavaDropletScanCell::SIZE] {
        let mut bytes: [u8; Self::SIZE] = [b' '; Self::SIZE];

        for (index, b) in self.as_bools().into_iter().enumerate() {
            if b {
                bytes[index] = Self::BYTES[index];
            }
        }

        bytes
    }

    const fn as_bvec4(self) -> BVec4 {
        // SAFETY: Both `LavaDropletScanCell` and `BVec4` have `#[repr(C, align(1))]`, with only 4
        // `bool`s as fields
        unsafe { transmute(self) }
    }

    const fn as_bools(self) -> [bool; LavaDropletScanCell::SIZE] {
        // SAFETY: `LavaDropletScanCell` has `#[repr(C)]`, and it's composed of `SIZE` `bool`s
        unsafe { transmute(self) }
    }

    const fn as_bools_ref(&self) -> &[bool; LavaDropletScanCell::SIZE] {
        // SAFETY: `LavaDropletScanCell` has `#[repr(C)]`, and it's composed of `SIZE` `bool`s
        unsafe { transmute(self) }
    }

    fn as_bools_mut(&mut self) -> &mut [bool; LavaDropletScanCell::SIZE] {
        // SAFETY: `LavaDropletScanCell` has `#[repr(C)]`, and it's composed of `SIZE` `bool`s
        unsafe { transmute(self) }
    }

    const fn from_bools(bools: [bool; LavaDropletScanCell::SIZE]) -> Self {
        // SAFETY: `LavaDropletScanCell` has `#[repr(C)]`, and it's composed of `SIZE` `bool`s
        unsafe { transmute(bools) }
    }

    const fn from_bvec4(bvec4: BVec4) -> Self {
        // SAFETY: Both `LavaDropletScanCell` and `BVec4` have `#[repr(C, align(1))]`, with only 4
        // `bool`s as fields
        unsafe { transmute(bvec4) }
    }

    fn from_str(s: &str) -> Self {
        let mut bools: [bool; Self::SIZE] = Default::default();

        for (index, byte) in s.as_bytes().iter().copied().enumerate() {
            bools[index] = byte == Self::BYTES[index];
        }

        Self::from_bools(bools)
    }

    fn get_bool(&self, index: usize) -> Option<&bool> {
        self.as_bools_ref().get(index)
    }

    fn get_bool_mut(&mut self, index: usize) -> Option<&mut bool> {
        self.as_bools_mut().get_mut(index)
    }
}

impl Debug for LavaDropletScanCell {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!(
            "\"{}\"",
            // SAFETY: `LavaDropletScanCell` only outputs ASCII bytes
            unsafe { from_utf8_unchecked(&self.as_utf8_bytes()) }
        ))
    }
}

impl<'s> From<&'s str> for LavaDropletScanCell {
    fn from(s: &'s str) -> Self {
        Self::from_str(s)
    }
}

impl From<LavaDropletScanCell> for [u8; LavaDropletScanCell::SIZE] {
    fn from(cell: LavaDropletScanCell) -> Self {
        cell.as_utf8_bytes()
    }
}

impl From<[bool; LavaDropletScanCell::SIZE]> for LavaDropletScanCell {
    fn from(bools: [bool; LavaDropletScanCell::SIZE]) -> Self {
        Self::from_bools(bools)
    }
}

impl From<LavaDropletScanCell> for [bool; LavaDropletScanCell::SIZE] {
    fn from(cell: LavaDropletScanCell) -> Self {
        cell.as_bools()
    }
}

#[derive(Debug, PartialEq)]
enum ParseIVec3Error<'s> {
    NoXToken,
    FailedToParseX(ParseIntError),
    NoYToken,
    FailedToParseY(ParseIntError),
    NoZToken,
    FailedToParseZ(ParseIntError),
    ExtraTokenFound(&'s str),
}

#[derive(Debug, Default, PartialEq)]
struct LavaDropletScan {
    cubes: Vec<IVec3>,
    grid: Grid3D<LavaDropletScanCell>,
    offset: IVec3,
}

impl LavaDropletScan {
    fn parse_cube_cooridnates(lava_droplet_cube_str: &str) -> Result<IVec3, ParseIVec3Error> {
        use ParseIVec3Error::*;

        let mut component_iter: Split<char> = lava_droplet_cube_str.split(',');

        let x: i32 =
            i32::from_str(component_iter.next().ok_or(NoXToken)?).map_err(FailedToParseX)?;
        let y: i32 =
            i32::from_str(component_iter.next().ok_or(NoYToken)?).map_err(FailedToParseY)?;
        let z: i32 =
            i32::from_str(component_iter.next().ok_or(NoZToken)?).map_err(FailedToParseZ)?;

        match component_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(IVec3 { x, y, z }),
        }
    }

    fn build_grid(cubes: &Vec<IVec3>, dimensions: IVec3) -> Grid3D<LavaDropletScanCell> {
        let mut grid: Grid3D<LavaDropletScanCell> = Grid3D::default(dimensions);

        for pos in cubes.iter() {
            grid.get_mut(pos).unwrap().occupied = true;
        }

        Self::initialize_grid_exposed(&mut grid);

        grid
    }

    fn initialize_grid_exposed(grid: &mut Grid3D<LavaDropletScanCell>) {
        for z_iter in CellIter3D::until_boundary(grid, IVec3::ZERO, IVec3::Z) {
            for y_iter in CellIter3D::until_boundary(grid, z_iter, IVec3::Y) {
                for pos in CellIter3D::until_boundary(grid, y_iter, IVec3::X) {
                    let cell: &mut LavaDropletScanCell = grid.get_mut(&pos).unwrap();

                    if cell.occupied {
                        for exposed in cell.as_bools_mut()[..IVec3::AXES.len()].iter_mut() {
                            *exposed ^= true;
                        }

                        for (index, axis) in IVec3::AXES.iter().enumerate() {
                            *grid
                                .get_mut(&(pos - *axis))
                                .unwrap()
                                .get_bool_mut(index)
                                .unwrap() ^= true;
                        }
                    }
                }
            }
        }
    }

    fn count_exposed(grid: &Grid3D<LavaDropletScanCell>) -> usize {
        const ONLY_EXPOSED: BVec4 = BVec4::new(true, true, true, false);

        grid.cells()
            .iter()
            .map(|cell| {
                LavaDropletScanCell::from_bvec4(cell.as_bvec4() & ONLY_EXPOSED)
                    .as_bools()
                    .into_iter()
            })
            .flatten()
            .filter(|b| *b)
            .count()
    }

    fn surface_area(&self) -> usize {
        Self::count_exposed(&self.grid)
    }

    fn surface_area_external_only(&self) -> usize {
        let not_externally_exposed: Grid3D<LavaDropletScanCell> =
            LavaDropletScanNotExternallyExposedBuilder::build(self);

        Self::count_exposed(&not_externally_exposed)
    }
}

impl<'s> TryFrom<&'s str> for LavaDropletScan {
    type Error = ParseIVec3Error<'s>;

    fn try_from(lava_droplet_scan_str: &'s str) -> Result<Self, Self::Error> {
        let mut cubes: Vec<IVec3> = Vec::new();
        let mut min: IVec3 = i32::MAX * IVec3::ONE;
        let mut max: IVec3 = i32::MIN * IVec3::ONE;

        for lava_droplet_cube_str in lava_droplet_scan_str.split('\n') {
            let cube: IVec3 = Self::parse_cube_cooridnates(lava_droplet_cube_str)?;

            min = min.min(cube);
            max = max.max(cube);
            cubes.push(cube);
        }

        Ok(if cubes.is_empty() {
            LavaDropletScan::default()
        } else {
            let offset: IVec3 = min - IVec3::ONE;

            for cube in cubes.iter_mut() {
                *cube -= offset;
            }

            let dimensions: IVec3 = max - min + 2_i32 * IVec3::ONE;
            let grid: Grid3D<LavaDropletScanCell> = Self::build_grid(&cubes, dimensions);

            LavaDropletScan {
                cubes,
                grid,
                offset,
            }
        })
    }
}

struct LavaDropletScanNotExternallyExposedBuilder<'l> {
    lava_droplet_scan: &'l LavaDropletScan,

    /// The result of the builder, where `occupied` for a cell indicates whether or not the cell is
    /// *not* externally exposed
    not_externally_exposed: &'l mut Grid3D<LavaDropletScanCell>,
}

impl<'l> LavaDropletScanNotExternallyExposedBuilder<'l> {
    fn wrap_position(&self, pos: IVec3) -> IVec3 {
        let dimensions: IVec3 = *self.lava_droplet_scan.grid.dimensions();

        IVec3::new(
            pos.x.rem_euclid(dimensions.x),
            pos.y.rem_euclid(dimensions.y),
            pos.z.rem_euclid(dimensions.z),
        )
    }

    fn build(lava_droplet_scan: &LavaDropletScan) -> Grid3D<LavaDropletScanCell> {
        let dimensions: IVec3 = *lava_droplet_scan.grid.dimensions();

        let mut not_externally_exposed: Grid3D<LavaDropletScanCell> = Grid3D::allocate(dimensions);

        not_externally_exposed.resize_layers(dimensions.z as usize, || LavaDropletScanCell {
            occupied: true,
            ..Default::default()
        });
        not_externally_exposed
            .get_mut(&IVec3::ZERO)
            .unwrap()
            .occupied = false;
        LavaDropletScanNotExternallyExposedBuilder {
            lava_droplet_scan,
            not_externally_exposed: &mut not_externally_exposed,
        }
        .run();
        LavaDropletScan::initialize_grid_exposed(&mut not_externally_exposed);

        not_externally_exposed
    }
}

impl<'l> BreadthFirstSearch for LavaDropletScanNotExternallyExposedBuilder<'l> {
    type Vertex = IVec3;

    fn start(&self) -> &Self::Vertex {
        &IVec3::ZERO
    }

    // This is just used to simulate water filling the space, so we don't want to terminate early
    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    // This is only called if `is_end` returns `true`
    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!();
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        let cell: &LavaDropletScanCell = self.lava_droplet_scan.grid.get(vertex).unwrap();

        for (index, axis) in IVec3::AXES.iter().enumerate() {
            if !*cell.get_bool(index).unwrap() {
                neighbors.push(self.wrap_position(*vertex + *axis));
            }

            let neighbor: IVec3 = self.wrap_position(*vertex - *axis);

            if !*self
                .lava_droplet_scan
                .grid
                .get(&neighbor)
                .unwrap()
                .get_bool(index)
                .unwrap()
            {
                neighbors.push(neighbor);
            }
        }
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        self.not_externally_exposed.get_mut(to).unwrap().occupied = false;
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day18.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match LavaDropletScan::try_from(input) {
                    Ok(lava_droplet_scan) => {
                        dbg!(lava_droplet_scan.surface_area());
                        dbg!(lava_droplet_scan.surface_area_external_only());
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

#[cfg(test)]
mod tests {
    use super::*;

    const LAVA_DROPLET_SCAN_STR: &str =
        "2,2,2\n1,2,2\n3,2,2\n2,1,2\n2,3,2\n2,2,1\n2,2,3\n2,2,4\n2,2,6\n1,2,5\n3,2,5\n2,1,5\n2,3,5";

    lazy_static! {
        static ref LAVA_DROPLET_SCAN: LavaDropletScan = lava_droplet_scan();
    }

    #[test]
    fn test_lava_droplet_scan_try_from_str() {
        assert_eq!(
            LAVA_DROPLET_SCAN_STR.try_into().as_ref(),
            Ok(&*LAVA_DROPLET_SCAN)
        );
    }

    #[test]
    fn test_lava_droplet_scan_surface_area() {
        assert_eq!(LAVA_DROPLET_SCAN.surface_area(), 64_usize);
    }

    #[test]
    fn test_lava_droplet_scan_surface_area_external_only() {
        assert_eq!(LAVA_DROPLET_SCAN.surface_area_external_only(), 58_usize);
    }

    fn lava_droplet_scan() -> LavaDropletScan {
        const OFFSET: IVec3 = IVec3::ZERO;
        const DIMENSIONS: IVec3 = IVec3::new(4_i32, 4_i32, 7_i32);

        macro_rules! cubes {
            ($( ( $x:expr, $y:expr, $z:expr ), )*) => {
                vec![ $( IVec3::new($x, $y, $z) - OFFSET, )* ]
            };
        }

        macro_rules! grid {
            [ $( [ $( [ $( $cell:literal ),* ], )* ], )*] => { {
                let cells: Vec<LavaDropletScanCell> = vec![
                    $( $( $( $cell.into(), )* )* )*
                ];

                let mut grid: Grid3D<LavaDropletScanCell> = Grid3D::default(DIMENSIONS);

                grid.cells_mut().copy_from_slice(&cells);

                grid
            } };
        }

        LavaDropletScan {
            cubes: cubes![
                (2, 2, 2),
                (1, 2, 2),
                (3, 2, 2),
                (2, 1, 2),
                (2, 3, 2),
                (2, 2, 1),
                (2, 2, 3),
                (2, 2, 4),
                (2, 2, 6),
                (1, 2, 5),
                (3, 2, 5),
                (2, 1, 5),
                (2, 3, 5),
            ],
            grid: grid![
                // z == 0
                [
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", "  Z ", "    "],
                    ["    ", "    ", "    ", "    "],
                ],
                // z == 1
                [
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", " YZ ", "    "],
                    ["    ", "X Z ", "XY O", "  Z "],
                    ["    ", "    ", "  Z ", "    "],
                ],
                // z == 2
                [
                    ["    ", "    ", " Y  ", "    "],
                    ["    ", "XY  ", "X ZO", " Y  "],
                    ["X   ", " YZO", "   O", "XYZO"],
                    ["    ", "X   ", "XYZO", "    "],
                ],
                // z == 3
                [
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", " Y  ", "    "],
                    ["    ", "X   ", "XY O", "    "],
                    ["    ", "    ", "    ", "    "],
                ],
                // z == 4
                [
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", " YZ ", "    "],
                    ["    ", "X Z ", "XYZO", "  Z "],
                    ["    ", "    ", "  Z ", "    "],
                ],
                // z == 5
                [
                    ["    ", "    ", " Y  ", "    "],
                    ["    ", "XY  ", "XYZO", " Y  "],
                    ["X   ", "XYZO", "XYZ ", "XYZO"],
                    ["    ", "X   ", "XYZO", "    "],
                ],
                // z == 6
                [
                    ["    ", "    ", "    ", "    "],
                    ["    ", "    ", " Y  ", "    "],
                    ["    ", "X   ", "XYZO", "    "],
                    ["    ", "    ", "    ", "    "],
                ],
            ],
            offset: OFFSET,
        }
    }
}
