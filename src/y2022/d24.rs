use {
    crate::*,
    glam::{IVec2, IVec3, Vec3Swizzles},
    std::{collections::HashMap, iter::Peekable, mem::transmute, str::Split},
    strum::{EnumCount, IntoEnumIterator},
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
pub enum BlizzardCellByte {
    Wall = b'#',
    #[default]
    ClearGround = b'.',
    North = b'^',
    East = b'>',
    South = b'v',
    West = b'<',
    Two = b'2',
    Three = b'3',
    Four = b'4',
}

impl BlizzardCellByte {
    const WALL_U8: u8 = Self::Wall as u8;
    const CLEAR_GROUND_U8: u8 = Self::ClearGround as u8;
    const NORTH_U8: u8 = Self::North as u8;
    const EAST_U8: u8 = Self::East as u8;
    const SOUTH_U8: u8 = Self::South as u8;
    const WEST_U8: u8 = Self::West as u8;
    const TWO_U8: u8 = Self::Two as u8;
    const THREE_U8: u8 = Self::Three as u8;
    const FOUR_U8: u8 = Self::Four as u8;

    fn try_add(a: Self, b: Self) -> Option<Self> {
        match (a, b) {
            (Self::Wall, Self::Wall) => None,
            (Self::ClearGround, _) => Some(b),
            (_, Self::ClearGround) => Some(a),
            _ => {
                let blizzard_sum: u8 = a.blizzards() + b.blizzards();

                match blizzard_sum {
                    0_u8..=1_u8 => unreachable!(),
                    2_u8..=4_u8 => Some(
                        // SAFETY: A `BlizzardCellByte` is just one byte, and this will map directly
                        // to a variant in the `Self::Two..=Self::Four` range
                        unsafe { transmute(b'2' + blizzard_sum - 2_u8) },
                    ),
                    _ => None,
                }
            }
        }
    }

    const fn blizzards(self) -> u8 {
        match self {
            Self::Wall => u8::MAX,
            Self::ClearGround => 0_u8,
            Self::Two | Self::Three | Self::Four => self as u8 - Self::Two as u8 + 2_u8,
            _ => 1_u8,
        }
    }
}

// SAFETY: Trivial
unsafe impl IsValidAscii for BlizzardCellByte {}

#[derive(Debug, PartialEq)]
pub struct ParseBlizzardCellByteError(u8);

impl TryFrom<u8> for BlizzardCellByte {
    type Error = ParseBlizzardCellByteError;

    fn try_from(blizzard_cell_byte: u8) -> Result<Self, Self::Error> {
        match blizzard_cell_byte {
            Self::WALL_U8 => Ok(Self::Wall),
            Self::CLEAR_GROUND_U8 => Ok(Self::ClearGround),
            Self::NORTH_U8 => Ok(Self::North),
            Self::EAST_U8 => Ok(Self::East),
            Self::SOUTH_U8 => Ok(Self::South),
            Self::WEST_U8 => Ok(Self::West),
            Self::TWO_U8 => Ok(Self::Two),
            Self::THREE_U8 => Ok(Self::Three),
            Self::FOUR_U8 => Ok(Self::Four),
            _ => Err(ParseBlizzardCellByteError(blizzard_cell_byte)),
        }
    }
}

struct BlizzardGrid2D(Grid2D<BlizzardCellByte>);

impl BlizzardGrid2D {
    fn iter_walls(&self) -> impl Iterator<Item = IVec2> + '_ {
        let door_gap: IVec2 = 2_i32 * IVec2::X;
        let max_dimensions: IVec2 = self.0.max_dimensions();

        [
            (door_gap, Direction::East),
            (max_dimensions * IVec2::X, Direction::South),
            (max_dimensions - door_gap, Direction::West),
            (max_dimensions * IVec2::Y, Direction::North),
        ]
        .into_iter()
        .map(|(start, dir)| CellIter2D::until_boundary(&self.0, start, dir))
        .flatten()
    }

    fn validate(&self) -> Result<(), InvalidBlizzardGrid2DState> {
        use InvalidBlizzardGrid2DState::*;

        self.validate_walls().map_err(GapInWall)?;

        let doors_and_dirs: [(IVec2, Direction); 2_usize] = [
            (IVec2::X, Direction::South),
            (self.0.max_dimensions() - IVec2::X, Direction::North),
        ];

        for (door, _) in doors_and_dirs.iter() {
            self.validate_door(*door).map_err(InvalidDoor)?;
        }

        for (door, dir) in doors_and_dirs {
            self.validate_door_column(door, dir)
                .map_err(BlizzardHeadedForDoor)?;
        }

        self.validate_valley().map_err(WallInValley)?;

        Ok(())
    }

    fn validate_walls(&self) -> Result<(), IVec2> {
        if let Some(gap_in_wall) = self
            .iter_walls()
            .find(|wall_pos| *self.0.get(*wall_pos).unwrap() != BlizzardCellByte::Wall)
        {
            Err(gap_in_wall)
        } else {
            Ok(())
        }
    }

    fn validate_door(&self, door: IVec2) -> Result<(), IVec2> {
        if *self.0.get(door).unwrap() != BlizzardCellByte::ClearGround {
            Err(door)
        } else {
            Ok(())
        }
    }

    fn validate_door_column(&self, door: IVec2, dir: Direction) -> Result<(), IVec2> {
        let dir_vec: IVec2 = dir.vec();
        let max_dimensions: IVec2 = self.0.max_dimensions();

        if let Some(blizzard_headed_for_door) =
            CellIter2D::try_from(door + dir_vec..door + max_dimensions.y * dir_vec)
                .unwrap()
                .find(|column_pos| {
                    let blizzard_cell_byte: BlizzardCellByte = *self.0.get(*column_pos).unwrap();

                    blizzard_cell_byte == BlizzardCellByte::North
                        || blizzard_cell_byte == BlizzardCellByte::South
                })
        {
            Err(blizzard_headed_for_door)
        } else {
            Ok(())
        }
    }

    fn validate_valley(&self) -> Result<(), IVec2> {
        if let Some(wall_in_valley) =
            CellIter2D::until_boundary(&self.0, IVec2::ZERO, Direction::South)
                .filter(|row_iter| self.0.is_border(*row_iter))
                .map(|row_iter| CellIter2D::until_boundary(&self.0, row_iter, Direction::East))
                .flatten()
                .find(|valley_pos| {
                    !self.0.is_border(*valley_pos)
                        && *self.0.get(*valley_pos).unwrap() == BlizzardCellByte::Wall
                })
        {
            Err(wall_in_valley)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum InvalidBlizzardGrid2DState {
    GapInWall(IVec2),
    InvalidDoor(IVec2),
    BlizzardHeadedForDoor(IVec2),
    WallInValley(IVec2),
}

#[derive(Debug, PartialEq)]
pub enum ParseBlizzardGrid2DError {
    InvalidLineLength,
    FailedToParseBlizzardCellByte(ParseBlizzardCellByteError),
    InvalidState(InvalidBlizzardGrid2DState),
}

impl TryFrom<&str> for BlizzardGrid2D {
    type Error = ParseBlizzardGrid2DError;

    fn try_from(blizzard_grid_2d_str: &str) -> Result<Self, Self::Error> {
        use ParseBlizzardGrid2DError::*;

        let mut blizzard_grid_2d_line_iter: Peekable<Split<char>> =
            blizzard_grid_2d_str.split('\n').peekable();

        let width: i32 = blizzard_grid_2d_line_iter
            .peek()
            .map_or(0_i32, |first_blizzard_grid_2d_line| {
                first_blizzard_grid_2d_line.len() as i32
            });

        if width < 3_i32 {
            return Err(InvalidLineLength);
        }

        let mut blizzard_cell_bytes: Vec<BlizzardCellByte> = Vec::new();

        let mut height: i32 = 0_i32;

        for blizzard_grid_2d_line in blizzard_grid_2d_line_iter {
            if blizzard_grid_2d_line.len() as i32 != width {
                return Err(InvalidLineLength);
            }

            for blizzard_cell_byte in blizzard_grid_2d_line.as_bytes().iter().copied() {
                blizzard_cell_bytes.push(
                    blizzard_cell_byte
                        .try_into()
                        .map_err(FailedToParseBlizzardCellByte)?,
                );
            }

            height += 1_i32;
        }

        let mut blizzard_grid_2d: Self = Self(Grid2D::default(IVec2::new(width, height)));

        blizzard_grid_2d
            .0
            .cells_mut()
            .copy_from_slice(&blizzard_cell_bytes);
        blizzard_grid_2d.validate().map_err(InvalidState)?;

        Ok(blizzard_grid_2d)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct BlizzardGrid3D {
    north_south: Grid3D<BlizzardCellByte>,
    east_west: Grid3D<BlizzardCellByte>,
}

impl BlizzardGrid3D {
    const DELTAS: [(IVec3, BlizzardCellByte); Direction::COUNT] = {
        use {BlizzardCellByte as Bcb, BlizzardGrid3D as Bg3D, Direction as Dir};

        [
            (Bg3D::dir_to_delta(Dir::North), Bcb::North),
            (Bg3D::dir_to_delta(Dir::South), Bcb::South),
            (Bg3D::dir_to_delta(Dir::East), Bcb::East),
            (Bg3D::dir_to_delta(Dir::West), Bcb::West),
        ]
    };

    fn is_clear(&self, pos: IVec3) -> bool {
        let mut ns_pos: IVec3 = pos;
        let mut ew_pos: IVec3 = pos;

        ns_pos.z %= self.north_south.dimensions().z;
        ew_pos.z %= self.east_west.dimensions().z;

        self.north_south
            .get(&ns_pos)
            .map_or(false, |cell| *cell == BlizzardCellByte::ClearGround)
            && self
                .east_west
                .get(&ew_pos)
                .map_or(false, |cell| *cell == BlizzardCellByte::ClearGround)
    }

    const fn start() -> IVec2 {
        IVec2::new(0_i32, -1_i32)
    }

    fn end(&self) -> IVec2 {
        self.north_south.max_dimensions().xy() + IVec2::Y
    }

    fn path_to_end(&self) -> Option<Vec<IVec3>> {
        self.path((Self::start(), 0_i32).into(), self.end())
    }

    fn path_to_end_then_start_then_end(&self) -> Option<Vec<IVec3>> {
        let mut path: Vec<IVec3> = self.path_to_end()?;

        for pos in self
            .path(*path.last().unwrap(), Self::start())?
            .into_iter()
            .skip(1_usize)
        {
            path.push(pos);
        }

        for pos in self
            .path(*path.last().unwrap(), self.end())?
            .into_iter()
            .skip(1_usize)
        {
            path.push(pos);
        }

        Some(path)
    }

    fn path(&self, start: IVec3, end: IVec2) -> Option<Vec<IVec3>> {
        ExpeditionPathSearch {
            blizzard_grid_3d: self,
            search: HashMap::new(),
            start,
            end,
        }
        .run_a_star()
    }

    fn fill_out_period(&mut self) {
        // Because we're storing the north-south and east-west blizzards in different grids, it
        // doesn't matter if the LCM of their periods is quite large, since all we need to store
        // is a horizontal period's worth, and a vertical period's worth
        Self::fill_out_period_for_grid(&mut self.north_south, true, Self::north_south_deltas);
        Self::fill_out_period_for_grid(&mut self.east_west, false, Self::east_west_deltas);
    }

    const fn dir_to_delta(dir: Direction) -> IVec3 {
        let dir_vec: IVec2 = dir.vec();

        IVec3::new(dir_vec.x, dir_vec.y, 1_i32)
    }

    fn north_south_deltas(cell: BlizzardCellByte) -> &'static [(IVec3, BlizzardCellByte)] {
        match cell {
            BlizzardCellByte::North => &Self::DELTAS[..1_usize],
            BlizzardCellByte::South => &Self::DELTAS[1_usize..2_usize],
            BlizzardCellByte::Two => &Self::DELTAS[..2_usize],
            BlizzardCellByte::ClearGround => &[],
            _ => panic!("Cannot convert cell {:?} into a north-south delta", cell),
        }
    }

    fn east_west_deltas(cell: BlizzardCellByte) -> &'static [(IVec3, BlizzardCellByte)] {
        match cell {
            BlizzardCellByte::East => &Self::DELTAS[2_usize..3_usize],
            BlizzardCellByte::West => &Self::DELTAS[3_usize..],
            BlizzardCellByte::Two => &Self::DELTAS[2_usize..],
            BlizzardCellByte::ClearGround => &[],
            _ => panic!("Cannot convert cell {:?} into a east-west delta", cell),
        }
    }

    fn fill_out_period_for_grid<D: Fn(BlizzardCellByte) -> &'static [(IVec3, BlizzardCellByte)]>(
        grid: &mut Grid3D<BlizzardCellByte>,
        north_south: bool,
        deltas: D,
    ) {
        let dimensions: IVec3 = *grid.dimensions();
        let period: i32 = if north_south {
            dimensions.y
        } else {
            dimensions.x
        };

        grid.resize_layers(period as usize, Default::default);

        let dimensions: IVec3 = *grid.dimensions();

        for pos in (0_i32..period - 1_i32)
            .map(|time| {
                CellIter3D::until_boundary_from_dimensions(
                    &dimensions,
                    (IVec2::ZERO, time).into(),
                    IVec3::Y,
                )
                .map(|row_iter| {
                    CellIter3D::until_boundary_from_dimensions(&dimensions, row_iter, IVec3::X)
                })
                .flatten()
            })
            .flatten()
        {
            for (delta, old_cell) in deltas(*grid.get(&pos).unwrap()).iter().copied() {
                let new_cell: &mut BlizzardCellByte =
                    grid.get_mut(&grid.rem_euclid(&(pos + delta))).unwrap();

                *new_cell = BlizzardCellByte::try_add(*new_cell, old_cell)
                    .expect("Could not add old cell into accumulator new cell");
            }
        }
    }

    #[cfg(test)]
    fn get_blizzard_grid_2d_at_time(&self, time: usize) -> BlizzardGrid2D {
        let ns_dimensions: IVec3 = *self.north_south.dimensions();
        let ns_time: usize = time % ns_dimensions.z as usize;
        let ew_time: usize = time % self.east_west.dimensions().z as usize;
        let blizzard_grid_2d_empty: BlizzardGrid2D =
            BlizzardGrid2D(Grid2D::empty(ns_dimensions.xy() + 2_i32 * IVec2::ONE));

        let mut blizzard_grid_2d: BlizzardGrid2D =
            BlizzardGrid2D(Grid2D::default(blizzard_grid_2d_empty.0.dimensions()));

        for wall_pos in blizzard_grid_2d_empty.iter_walls() {
            *blizzard_grid_2d.0.get_mut(wall_pos).unwrap() = BlizzardCellByte::Wall;
        }

        let cells_in_2d_grid: usize = ns_dimensions.x as usize * ns_dimensions.y as usize;
        let ns_start: usize = cells_in_2d_grid * ns_time;
        let ns_end: usize = ns_start + cells_in_2d_grid;
        let ew_start: usize = cells_in_2d_grid * ew_time;
        let ew_end: usize = ew_start + cells_in_2d_grid;

        for (blizzard_cell_byte, (ns_cell, ew_cell)) in blizzard_grid_2d
            .0
            .cells_mut()
            .iter_mut()
            .enumerate()
            .filter_map(|(index_2d, cell)| {
                if !blizzard_grid_2d_empty
                    .0
                    .is_border(blizzard_grid_2d_empty.0.pos_from_index(index_2d))
                {
                    Some(cell)
                } else {
                    None
                }
            })
            .zip(
                self.north_south.cells()[ns_start..ns_end]
                    .iter()
                    .copied()
                    .zip(self.east_west.cells()[ew_start..ew_end].iter().copied()),
            )
        {
            *blizzard_cell_byte = BlizzardCellByte::try_add(ns_cell, ew_cell)
                .expect("North-south cell and east-west cell could not be added");
        }

        blizzard_grid_2d
    }

    #[cfg(test)]
    fn as_string_at_time(&self, time: usize) -> String {
        self.get_blizzard_grid_2d_at_time(time).0.into()
    }
}

#[derive(Debug, PartialEq)]
pub enum ConvertBlizzardGrid3DError {
    FromStr(ParseBlizzardGrid2DError),
    AmbiguousBlizzardCellByte(BlizzardCellByte),
}

impl TryFrom<BlizzardGrid2D> for BlizzardGrid3D {
    type Error = ConvertBlizzardGrid3DError;

    fn try_from(blizzard_grid_2d: BlizzardGrid2D) -> Result<Self, Self::Error> {
        use ConvertBlizzardGrid3DError::*;

        let dimensions: IVec3 =
            (blizzard_grid_2d.0.dimensions() - 2_i32 * IVec2::ONE, 1_i32).into();

        let mut blizzard_grid_3d: Self = Self {
            north_south: Grid3D::default(dimensions),
            east_west: Grid3D::default(dimensions),
        };

        for ((ns_cell, ew_cell), cell) in
            blizzard_grid_3d
                .north_south
                .cells_mut()
                .iter_mut()
                .zip(blizzard_grid_3d.east_west.cells_mut())
                .zip(blizzard_grid_2d.0.cells().iter().enumerate().filter_map(
                    |(index_2d, cell)| {
                        if !blizzard_grid_2d
                            .0
                            .is_border(blizzard_grid_2d.0.pos_from_index(index_2d))
                        {
                            Some(*cell)
                        } else {
                            None
                        }
                    },
                ))
        {
            match cell {
                BlizzardCellByte::North | BlizzardCellByte::South => {
                    *ns_cell = cell;
                }
                BlizzardCellByte::East | BlizzardCellByte::West => {
                    *ew_cell = cell;
                }
                BlizzardCellByte::ClearGround => {}
                _ => return Err(AmbiguousBlizzardCellByte(cell)),
            }
        }

        blizzard_grid_3d.fill_out_period();

        Ok(blizzard_grid_3d)
    }
}

impl TryFrom<&str> for BlizzardGrid3D {
    type Error = ConvertBlizzardGrid3DError;

    fn try_from(blizzard_grid_2d_str: &str) -> Result<Self, Self::Error> {
        use ConvertBlizzardGrid3DError::*;

        BlizzardGrid2D::try_from(blizzard_grid_2d_str)
            .map_err(FromStr)?
            .try_into()
    }
}

#[derive(Clone, Copy)]
struct PathElement {
    cost: i32,
    prev: Option<Direction>,
}

impl Default for PathElement {
    fn default() -> Self {
        Self {
            cost: i32::MAX,
            prev: None,
        }
    }
}

struct ExpeditionPathSearch<'b> {
    blizzard_grid_3d: &'b BlizzardGrid3D,
    search: HashMap<IVec3, PathElement>,
    start: IVec3,
    end: IVec2,
}

impl<'b> WeightedGraphSearch for ExpeditionPathSearch<'b> {
    type Vertex = IVec3;
    type Cost = i32;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.xy() == self.end
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: Vec<IVec3> = Vec::with_capacity(self.search[vertex].cost as usize + 1_usize);
        let mut vertex: IVec3 = *vertex;

        while vertex != self.start {
            path.push(vertex);

            if let Some(dir) = self.search[&vertex].prev {
                vertex += IVec3::from((dir.vec(), -1_i32));
            } else {
                vertex.z -= 1_i32
            }
        }

        path.push(self.start);
        path.reverse();

        path
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.search
            .get(vertex)
            .map_or(i32::MAX, |path_element| path_element.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        abs_sum_2d(self.end - vertex.xy())
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let vertex_plus_z: IVec3 = *vertex + IVec3::Z;

        if vertex_plus_z.xy() == self.start.xy() || self.blizzard_grid_3d.is_clear(vertex_plus_z) {
            neighbors.push(OpenSetElement(vertex_plus_z, 1_i32));
        }

        for dir in Direction::iter() {
            let pos: IVec3 = vertex_plus_z + IVec3::from((dir.vec(), 0_i32));

            if self.is_end(&pos) || self.blizzard_grid_3d.is_clear(pos) {
                neighbors.push(OpenSetElement(pos, 1_i32));
            }
        }
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.search.insert(
            *to,
            PathElement {
                cost,
                prev: Direction::try_from(from.xy() - to.xy()).ok(),
            },
        );
    }

    fn reset(&mut self) {
        self.search.clear();
        self.search.insert(
            self.start,
            PathElement {
                cost: 0_i32,
                ..Default::default()
            },
        );
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(BlizzardGrid3D);

impl Solution {
    fn fewest_minutes_to_end(&self) -> usize {
        self.0
            .path_to_end()
            .map_or(0_usize, |path| path.len() - 1_usize)
    }

    fn fewest_minutes_to_end_then_start_then_end(&self) -> usize {
        self.0
            .path_to_end_then_start_then_end()
            .map_or(0_usize, |path| path.len() - 1_usize)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.fewest_minutes_to_end());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.fewest_minutes_to_end_then_start_then_end());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = ConvertBlizzardGrid3DError;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const BLIZZARD_GRID_2D_SIMPLE_TIME_0: &str = concat!(
        "#.#####\n",
        "#.....#\n",
        "#>....#\n",
        "#.....#\n",
        "#...v.#\n",
        "#.....#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D_SIMPLE_TIME_1: &str = concat!(
        "#.#####\n",
        "#.....#\n",
        "#.>...#\n",
        "#.....#\n",
        "#.....#\n",
        "#...v.#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D_SIMPLE_TIME_2: &str = concat!(
        "#.#####\n",
        "#...v.#\n",
        "#..>..#\n",
        "#.....#\n",
        "#.....#\n",
        "#.....#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D_SIMPLE_TIME_3: &str = concat!(
        "#.#####\n",
        "#.....#\n",
        "#...2.#\n",
        "#.....#\n",
        "#.....#\n",
        "#.....#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D_SIMPLE_TIME_4: &str = concat!(
        "#.#####\n",
        "#.....#\n",
        "#....>#\n",
        "#...v.#\n",
        "#.....#\n",
        "#.....#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D_SIMPLE_TIME_5: &str = concat!(
        "#.#####\n",
        "#.....#\n",
        "#>....#\n",
        "#.....#\n",
        "#...v.#\n",
        "#.....#\n",
        "#####.#\n",
    );
    const BLIZZARD_GRID_2D: &str = concat!(
        "#.######\n",
        "#>>.<^<#\n",
        "#.<..<<#\n",
        "#>v.><>#\n",
        "#<^v^^>#\n",
        "######.#",
    );

    fn blizzard_grid_3d_simple() -> &'static BlizzardGrid3D {
        static ONCE_LOCK: OnceLock<BlizzardGrid3D> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            BlizzardGrid3D::try_from(
                &BLIZZARD_GRID_2D_SIMPLE_TIME_0[..BLIZZARD_GRID_2D_SIMPLE_TIME_0.len() - 1_usize],
            )
            .unwrap()
        })
    }

    fn blizzard_grid_3d() -> &'static BlizzardGrid3D {
        static ONCE_LOCK: OnceLock<BlizzardGrid3D> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| BLIZZARD_GRID_2D.try_into().unwrap())
    }

    #[test]
    fn test_blizzard_grid_3d_try_from_str() {
        assert_eq!(
            BlizzardGrid3D::try_from(
                &BLIZZARD_GRID_2D_SIMPLE_TIME_0[..BLIZZARD_GRID_2D_SIMPLE_TIME_0.len() - 1_usize]
            )
            .map(|blizzard_grid_3d| blizzard_grid_3d.as_string_at_time(0_usize)),
            Ok(BLIZZARD_GRID_2D_SIMPLE_TIME_0.to_owned())
        );
    }

    #[test]
    fn test_blizzard_grid_3d_fill_out_period() {
        assert_eq!(
            blizzard_grid_3d_simple().as_string_at_time(1_usize),
            BLIZZARD_GRID_2D_SIMPLE_TIME_1.to_owned()
        );
        assert_eq!(
            blizzard_grid_3d_simple().as_string_at_time(2_usize),
            BLIZZARD_GRID_2D_SIMPLE_TIME_2.to_owned()
        );
        assert_eq!(
            blizzard_grid_3d_simple().as_string_at_time(3_usize),
            BLIZZARD_GRID_2D_SIMPLE_TIME_3.to_owned()
        );
        assert_eq!(
            blizzard_grid_3d_simple().as_string_at_time(4_usize),
            BLIZZARD_GRID_2D_SIMPLE_TIME_4.to_owned()
        );
        assert_eq!(
            blizzard_grid_3d_simple().as_string_at_time(5_usize),
            BLIZZARD_GRID_2D_SIMPLE_TIME_5.to_owned()
        );
    }

    #[test]
    fn test_blizzard_grid_3d_path_to_end() {
        assert_eq!(
            blizzard_grid_3d()
                .path_to_end()
                .map(|path| path.len() - 1_usize),
            Some(18_usize)
        );
    }

    #[test]
    fn test_blizzard_grid_3d_path_to_end_then_start_then_end() {
        assert_eq!(
            blizzard_grid_3d()
                .path_to_end_then_start_then_end()
                .map(|path| path.len() - 1_usize),
            Some(54_usize)
        );
    }
}
