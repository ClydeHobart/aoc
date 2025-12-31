use {
    crate::*,
    glam::IVec2,
    std::{
        num::ParseIntError,
        slice::{Iter, SliceIndex},
        str::{FromStr, Split},
    },
    strum::EnumCount,
};

define_cell! {
    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    enum MapCell {
        #[default]
        Void = VOID = b' ',
        Open = OPEN = b'.',
        Wall = WALL = b'#',
        North = NORTH = b'^',
        East = EAST = b'>',
        South = SOUTH = b'v',
        West = WEST = b'<',
    }
}

impl MapCell {
    fn is_valid_str_input(&self) -> bool {
        matches!(self, Self::Void | Self::Open | Self::Wall)
    }
}

impl From<Direction> for MapCell {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => Self::North,
            Direction::East => Self::East,
            Direction::South => Self::South,
            Direction::West => Self::West,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct Map(Grid2D<MapCell>);

impl TryFrom<&str> for Map {
    type Error = ();

    fn try_from(map_str: &str) -> Result<Self, Self::Error> {
        let (width, height): (i32, i32) =
            map_str
                .split('\n')
                .fold((0_i32, 0_i32), |(max_width, height), map_row_str| {
                    (max_width.max(map_row_str.len() as i32), height + 1_i32)
                });

        let mut map: Self = Self(Grid2D::default(IVec2::new(width, height)));

        for (row_pos_iter, map_row_str) in
            CellIter2D::until_boundary(&map.0, IVec2::ZERO, Direction::South)
                .zip(map_str.split('\n'))
        {
            for (pos, map_cell_byte) in
                CellIter2D::until_boundary(&map.0, row_pos_iter, Direction::East)
                    .zip(map_row_str.as_bytes().iter().copied())
            {
                *map.0.get_mut(pos).unwrap() = map_cell_byte
                    .try_into()
                    .ok()
                    .filter(MapCell::is_valid_str_input)
                    .ok_or(())?;
            }
        }

        Ok(map)
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug)]
struct CubeNeighbor {
    to_pos: IVec2,
    from_dir: Direction,
    to_dir: Direction,
}

impl CubeNeighbor {
    #[inline(always)]
    fn is_initialized(&self) -> bool {
        self.to_pos != IVec2::NEG_ONE
    }

    #[inline(always)]
    fn is_valid(&self, dir: Direction) -> bool {
        self.is_initialized() && self.from_dir == dir
    }

    fn try_next(&self, dir: Direction, cube: bool) -> Option<TraceState> {
        if cube && self.is_valid(dir) {
            Some(TraceState {
                pos: self.to_pos,
                dir: self.to_dir,
            })
        } else {
            None
        }
    }
}

impl Default for CubeNeighbor {
    fn default() -> Self {
        Self {
            to_pos: IVec2::NEG_ONE,
            from_dir: Direction::North,
            to_dir: Direction::North,
        }
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Default)]
struct Neighbors {
    flat: [u8; 4_usize],
    cube_neighbor_a: CubeNeighbor,
    cube_neighbor_b: CubeNeighbor,
}

impl Neighbors {
    fn next(&self, trace_state: &TraceState, cube: bool) -> TraceState {
        self.cube_neighbor_a
            .try_next(trace_state.dir, cube)
            .or_else(|| self.cube_neighbor_b.try_next(trace_state.dir, cube))
            .unwrap_or_else(|| {
                let axis_a: IVec2 = trace_state.dir.vec().abs();
                let axis_b: IVec2 = IVec2::ONE - axis_a;

                TraceState {
                    pos: axis_b * trace_state.pos
                        + self.flat[trace_state.dir as usize] as i32 * axis_a,
                    dir: trace_state.dir,
                }
            })
    }
}

#[derive(Default)]
struct MapCellToNeighborsGridVisitor {
    first_position: Option<i32>,
}

impl GridVisitor for MapCellToNeighborsGridVisitor {
    type Old = MapCell;
    type New = Neighbors;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        old_grid: &Grid2D<Self::Old>,
        rev_dir: Direction,
        pos: IVec2,
    ) {
        if *old != MapCell::Void {
            let dir: Direction = rev_dir.rev();
            let dir_vec: IVec2 = dir.vec();
            let next: IVec2 = pos + dir_vec;

            if self.first_position.is_none() {
                self.first_position = Some(abs_sum_2d(pos * dir_vec));
            }

            if old_grid
                .get(next)
                .map(|old_next| *old_next != MapCell::Void)
                .unwrap_or_default()
            {
                new.flat[dir as usize] = abs_sum_2d(next * dir_vec) as u8;
            } else {
                new.flat[dir as usize] = self.first_position.unwrap() as u8;
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct NeighborsGrid(Grid2D<Neighbors>);

#[derive(Debug, PartialEq)]
pub enum ConstructNeighborsGridError {
    NonVoidMapCellCountIsNotSixTimesASquare(i32),
    DimensionsAreNotMultiplesOfCubeSideLen((IVec2, i32)),
    MapCellDoesNotFitCubeMap(IVec2),
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NetCell {
    is_not_void: bool,
    is_corner: bool,
}

impl NeighborsGrid {
    /// I'm sure there's some proof out there for why this is the case, but all cube maps have a
    /// perimeter of 14 units, and so they also consist of 14 vertices
    const CUBE_NET_PERIMETER: usize = 14_usize;
    const COINCIDENT_EDGE_COUNT: usize = Self::CUBE_NET_PERIMETER / 2_usize;

    fn try_new(map: &Map) -> Result<Self, ConstructNeighborsGridError> {
        use ConstructNeighborsGridError::*;

        let mut neighbors_grid: Self = Self(MapCellToNeighborsGridVisitor::visit_grid(&map.0));

        let cube_side_len: i32 =
            Self::cube_side_len(&map).map_err(NonVoidMapCellCountIsNotSixTimesASquare)?;

        Self::validate_dimensions(&map, cube_side_len)
            .map_err(DimensionsAreNotMultiplesOfCubeSideLen)?;

        let cube_face_corner_grid: Grid2D<NetCell> =
            Self::cube_face_corner_grid(&map, cube_side_len).map_err(MapCellDoesNotFitCubeMap)?;

        let mut perimeter_dirs: [Direction; Self::CUBE_NET_PERIMETER] =
            [Direction::North; Self::CUBE_NET_PERIMETER];

        let mut perimeter_corners: [IVec2; Self::CUBE_NET_PERIMETER] = Default::default();

        Self::perimeter(
            &cube_face_corner_grid,
            &mut perimeter_dirs,
            &mut perimeter_corners,
        );

        let mut coincident_corners: [u16; Self::CUBE_NET_PERIMETER] = Default::default();

        Self::coincident_corners(&perimeter_dirs, &mut coincident_corners);

        let mut coincident_edges: [(u8, u8); Self::COINCIDENT_EDGE_COUNT] = Default::default();

        Self::coincident_edges(&coincident_corners, &mut coincident_edges);

        for (index_a, index_b) in coincident_edges {
            let mut init_cube_neighbors = |index_a: usize, index_b: usize| {
                let edge_dir_a: Direction = perimeter_dirs[index_a];
                let edge_dir_b: Direction = perimeter_dirs[index_b];
                let from_dir: Direction = edge_dir_a.turn(true);
                let to_dir: Direction = edge_dir_b.turn(false);
                let edge_start = |index: usize, edge_dir: Direction| -> IVec2 {
                    // Only `East`-pointing edges are properly aligned in the corner where both
                    // coordinate components are multiples of `cube_side_len`. The other directions
                    // need adjustment
                    let mut edge_start: IVec2 = cube_side_len * perimeter_corners[index];
                    let mut adjuster: IVec2 = IVec2::NEG_X;

                    for _ in 0_u8..(edge_dir as u8 + Direction::COUNT as u8 - Direction::East as u8)
                        & Direction::MASK
                    {
                        edge_start += adjuster;
                        adjuster = adjuster.perp();
                    }

                    edge_start
                };
                let edge_start_a: IVec2 = edge_start(index_a, edge_dir_a);
                let edge_start_b: IVec2 =
                    // Reverse the direction and position of edge B, so that they're zipping in the
                    // same direction
                    edge_start(index_b, edge_dir_b) + (cube_side_len - 1_i32) * edge_dir_b.vec();
                for (from_pos, to_pos) in CellIter2D::try_from(
                    edge_start_a..edge_start_a + cube_side_len * edge_dir_a.vec(),
                )
                .unwrap()
                .zip(
                    CellIter2D::try_from(
                        edge_start_b..edge_start_b + cube_side_len * edge_dir_b.rev().vec(),
                    )
                    .unwrap(),
                ) {
                    let cube_neighbor: CubeNeighbor = CubeNeighbor {
                        to_pos,
                        from_dir,
                        to_dir,
                    };
                    let neighbors: &mut Neighbors = neighbors_grid.0.get_mut(from_pos).unwrap();

                    if !neighbors.cube_neighbor_a.is_initialized() {
                        neighbors.cube_neighbor_a = cube_neighbor
                    } else {
                        neighbors.cube_neighbor_b = cube_neighbor;
                    }
                }
            };

            init_cube_neighbors(index_a as usize, index_b as usize);
            init_cube_neighbors(index_b as usize, index_a as usize);
        }

        Ok(neighbors_grid)
    }

    fn cube_side_len(map: &Map) -> Result<i32, i32> {
        let non_void_count: i32 = map
            .0
            .cells()
            .iter()
            .copied()
            .filter(|map_cell| *map_cell != MapCell::Void)
            .count() as i32;

        let sqrt_non_void_count_div_6: f32 = (non_void_count as f32 / 6.0_f32).sqrt();

        if sqrt_non_void_count_div_6 % 1.0_f32 != 0.0_f32 {
            Err(non_void_count)
        } else {
            Ok(sqrt_non_void_count_div_6 as i32)
        }
    }

    fn validate_dimensions(map: &Map, cube_side_len: i32) -> Result<(), (IVec2, i32)> {
        let dimensions: IVec2 = map.0.dimensions();

        if dimensions % cube_side_len != IVec2::ZERO {
            Err((dimensions, cube_side_len))
        } else {
            Ok(())
        }
    }

    fn cube_face_corner_grid(map: &Map, cube_side_len: i32) -> Result<Grid2D<NetCell>, IVec2> {
        let dimensions_div_cube_side_len: IVec2 = map.0.dimensions() / cube_side_len;
        let mut cube_face_corner_grid: Grid2D<NetCell> =
            Grid2D::default(dimensions_div_cube_side_len + IVec2::ONE);

        for row_iter in
            CellIter2D::try_from(IVec2::ZERO..dimensions_div_cube_side_len * IVec2::Y).unwrap()
        {
            for pos in CellIter2D::try_from(
                row_iter..IVec2::new(dimensions_div_cube_side_len.x, row_iter.y),
            )
            .unwrap()
            {
                if *map.0.get(cube_side_len * pos).unwrap() != MapCell::Void {
                    *cube_face_corner_grid.get_mut(pos).unwrap() = NetCell {
                        is_not_void: true,
                        is_corner: true,
                    };
                    cube_face_corner_grid
                        .get_mut(pos + IVec2::X)
                        .unwrap()
                        .is_corner = true;
                    cube_face_corner_grid
                        .get_mut(pos + IVec2::Y)
                        .unwrap()
                        .is_corner = true;
                    cube_face_corner_grid
                        .get_mut(pos + IVec2::ONE)
                        .unwrap()
                        .is_corner = true;
                }
            }
        }

        for row_iter in CellIter2D::until_boundary(&map.0, IVec2::ZERO, Direction::South) {
            for pos in CellIter2D::until_boundary(&map.0, row_iter, Direction::East) {
                if (*map.0.get(pos).unwrap() != MapCell::Void)
                    != cube_face_corner_grid
                        .get(pos / cube_side_len)
                        .unwrap()
                        .is_not_void
                {
                    return Err(pos);
                }
            }
        }

        Ok(cube_face_corner_grid)
    }

    /// Find the perimeter of the cube face corner grid, winding in a clock-wise fashion
    fn perimeter(
        cube_face_corner_grid: &Grid2D<NetCell>,
        perimeter_dirs: &mut [Direction; Self::CUBE_NET_PERIMETER],
        perimeter_corners: &mut [IVec2; Self::CUBE_NET_PERIMETER],
    ) {
        let (mut corner, mut dir): (IVec2, Direction) = (
            CellIter2D::until_boundary(cube_face_corner_grid, IVec2::ZERO, Direction::East)
                .find(|pos| cube_face_corner_grid.get(*pos).unwrap().is_corner)
                .unwrap()
                // There will be at least two in the top row, so start at the second
                + IVec2::X,
            Direction::East,
        );
        let mut index: usize = 1_usize;

        perimeter_corners[0_usize] = corner;
        perimeter_dirs[0_usize] = dir;

        while index < perimeter_dirs.len() {
            let left_dir: Direction = dir.turn(true);
            let left_pos: IVec2 = corner + left_dir.vec();

            (corner, dir) = if cube_face_corner_grid
                .get(left_pos)
                .copied()
                .unwrap_or_default()
                .is_corner
            {
                (left_pos, left_dir)
            } else {
                let fwd_pos: IVec2 = corner + dir.vec();

                if cube_face_corner_grid
                    .get(fwd_pos)
                    .copied()
                    .unwrap_or_default()
                    .is_corner
                {
                    (fwd_pos, dir)
                } else {
                    let right_dir: Direction = dir.turn(false);

                    (corner + right_dir.vec(), right_dir)
                }
            };

            perimeter_corners[index] = corner;
            perimeter_dirs[index] = dir;
            index += 1_usize;
        }

        perimeter_corners.rotate_right(1_usize);
    }

    #[inline(always)]
    const fn sanitize_perimeter_index(index: usize) -> usize {
        index % Self::CUBE_NET_PERIMETER
    }

    #[inline(always)]
    const fn next_perimeter_index(index: usize) -> usize {
        (index + 1_usize) % Self::CUBE_NET_PERIMETER
    }

    #[inline(always)]
    const fn prev_perimeter_index(index: usize) -> usize {
        const MINUS_1: usize = NeighborsGrid::CUBE_NET_PERIMETER - 1_usize;

        (index + MINUS_1) % Self::CUBE_NET_PERIMETER
    }

    #[inline(always)]
    fn get_bit(bitset: u16, index: usize) -> bool {
        bitset & 1_u16 << index as u32 != 0_u16
    }

    #[inline(always)]
    fn set_bit(bitset: &mut u16, index: usize) {
        *bitset |= 1_u16 << index as u32;
    }

    #[inline(always)]
    fn clear_bit(bitset: &mut u16, index: usize) {
        *bitset &= !(1_u16 << index as u32);
    }

    #[cfg(test)]
    fn unpack_indices_single(indices: u16) -> Vec<u32> {
        (0_u32..u16::BITS)
            .filter(|index| indices & 1_u16 << *index != 0_u16)
            .collect()
    }

    #[cfg(test)]
    fn unpack_indices(indices_slice: &[u16; Self::CUBE_NET_PERIMETER]) -> Vec<Vec<u32>> {
        indices_slice
            .iter()
            .map(|indices| Self::unpack_indices_single(*indices))
            .collect()
    }

    fn coincident_corners(
        perimeter_dirs: &[Direction; Self::CUBE_NET_PERIMETER],
        coincident_corners: &mut [u16; Self::CUBE_NET_PERIMETER],
    ) {
        const MAX_CONCAVE_CORNERS: usize = 4_usize;
        const MAX_ZIPPING_ITERATIONS: usize = 6_usize;

        let mut exposed_corners: u16 = 0_u16;

        for (index, coincident_corners) in coincident_corners.iter_mut().enumerate() {
            Self::set_bit(coincident_corners, index);
            Self::set_bit(&mut exposed_corners, index);
        }

        // First find the concave corner seeds;
        let mut coincident_indices: [[(usize, usize); MAX_CONCAVE_CORNERS];
            MAX_ZIPPING_ITERATIONS] =
            [[(usize::MAX, usize::MAX); MAX_CONCAVE_CORNERS]; MAX_ZIPPING_ITERATIONS];
        let mut concave_corner_count: usize = 0_usize;
        let mut concave_corners: u16 = 0_u16;

        for (curr_index, curr_dir) in perimeter_dirs.iter().enumerate() {
            let prev_index: usize = Self::prev_perimeter_index(curr_index);

            if perimeter_dirs[prev_index].turn(true) == *curr_dir {
                coincident_indices[0_usize][concave_corner_count] = (curr_index, curr_index);
                concave_corner_count += 1_usize;
                Self::set_bit(&mut concave_corners, curr_index);
            }
        }

        exposed_corners &= !concave_corners;

        let mut zipping_iteration: usize = 1_usize;

        while exposed_corners != 0_u16 && zipping_iteration < MAX_ZIPPING_ITERATIONS {
            let mut next_coincident_corners: [u16; Self::CUBE_NET_PERIMETER] =
                coincident_corners.clone();
            let mut next_exposed_corners: u16 = exposed_corners;

            let mut run_zipping_iteration = |can_be_partially_not_exposed: bool| {
                let mut any_lineage_found_pair: bool = false;

                for concave_corner_index in 0_usize..concave_corner_count {
                    let (parent_prev_index, parent_next_index) =
                        coincident_indices[zipping_iteration - 1_usize][concave_corner_index];

                    if parent_prev_index != usize::MAX && parent_next_index != usize::MAX {
                        let prev_index: usize = Self::prev_perimeter_index(parent_prev_index);
                        let next_index: usize = Self::next_perimeter_index(parent_next_index);

                        if !Self::get_bit(concave_corners, prev_index)
                            && !Self::get_bit(concave_corners, next_index)
                            && if can_be_partially_not_exposed {
                                Self::get_bit(exposed_corners, prev_index)
                                    || Self::get_bit(exposed_corners, next_index)
                            } else {
                                Self::get_bit(exposed_corners, prev_index)
                                    && Self::get_bit(exposed_corners, next_index)
                            }
                        {
                            coincident_indices[zipping_iteration][concave_corner_index] =
                                (prev_index, next_index);

                            let coincident_corners: u16 = next_coincident_corners[prev_index]
                                | next_coincident_corners[next_index];

                            for index in 0_usize..Self::CUBE_NET_PERIMETER {
                                if Self::get_bit(coincident_corners, index) {
                                    next_coincident_corners[index] |= coincident_corners;
                                }
                            }

                            Self::clear_bit(&mut next_exposed_corners, prev_index);
                            Self::clear_bit(&mut next_exposed_corners, next_index);
                            any_lineage_found_pair = true;
                        }
                    }
                }

                any_lineage_found_pair
            };

            if !run_zipping_iteration(false) {
                run_zipping_iteration(true);
            }

            *coincident_corners = next_coincident_corners;
            exposed_corners = next_exposed_corners;
            zipping_iteration += 1_usize;
        }
    }

    fn coincident_edges(
        coincident_corners: &[u16; Self::CUBE_NET_PERIMETER],
        coincident_edges: &mut [(u8, u8); Self::COINCIDENT_EDGE_COUNT],
    ) {
        let mut edge_found_for_index: u16 = 0_u16;
        let mut edge_pair_index: usize = 0_usize;

        while edge_pair_index < coincident_edges.len() {
            let index_a: usize = edge_found_for_index.count_ones() as usize;
            let index_a_coincident_corners: u16 = coincident_corners[index_a];
            let index_b: usize = 'index_b: {
                let next_index_a: usize = Self::next_perimeter_index(index_a);
                let next_index_a_coincident_corners: u16 = coincident_corners[next_index_a];

                for index_b_dirty in next_index_a..next_index_a + Self::CUBE_NET_PERIMETER {
                    let index_b: usize = Self::sanitize_perimeter_index(index_b_dirty);

                    // If `index_b` is coincident with `next_index_a`...
                    if Self::get_bit(next_index_a_coincident_corners, index_b) {
                        let next_index_b: usize = Self::next_perimeter_index(index_b);

                        // and `next_index_b` is coincident with `index_a`
                        if Self::get_bit(index_a_coincident_corners, next_index_b) {
                            break 'index_b index_b;
                        }
                    }
                }

                panic!(
                    "Failed to find index B for index A {index_a}!\n\
                    index A coincident corners: {:014b}\n\
                    next index A coincident corners: {:014b}",
                    index_a_coincident_corners, next_index_a_coincident_corners
                );
            };

            Self::set_bit(&mut edge_found_for_index, index_a);
            Self::set_bit(&mut edge_found_for_index, index_b);
            coincident_edges[edge_pair_index] = (index_a as u8, index_b as u8);
            edge_pair_index += 1_usize;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    Move(u8),
    Turn { left: bool },
}

#[derive(Debug, PartialEq)]
pub enum ParseInstructionError {
    FailedToParseMove(ParseIntError),
    InvalidByte(u8),
}

impl Instruction {
    fn parse(s: &str) -> Option<Result<(Self, &str), ParseInstructionError>> {
        use ParseInstructionError::*;

        if s.is_empty() {
            None
        } else {
            let next_non_digit_index: usize = s
                .chars()
                .position(|c| !c.is_ascii_digit())
                .unwrap_or(s.len());

            Some(if next_non_digit_index != 0_usize {
                u8::from_str(&s[..next_non_digit_index])
                    .map(|tiles| (Self::Move(tiles), &s[next_non_digit_index..]))
                    .map_err(FailedToParseMove)
            } else {
                match s.as_bytes()[0_usize] {
                    b'L' => Ok((Self::Turn { left: true }, &s[1_usize..])),
                    b'R' => Ok((Self::Turn { left: false }, &s[1_usize..])),
                    invalid_byte => Err(InvalidByte(invalid_byte)),
                }
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Instructions(Vec<Instruction>);

impl TryFrom<&str> for Instructions {
    type Error = ParseInstructionError;

    fn try_from(mut instructions_str: &str) -> Result<Self, Self::Error> {
        let mut instructions: Self = Self(Vec::new());

        while let Some(result) = Instruction::parse(instructions_str) {
            let (instruction, new_instructions_str) = result?;

            instructions.0.push(instruction);
            instructions_str = new_instructions_str;
        }

        Ok(instructions)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct TraceState {
    pos: IVec2,
    dir: Direction,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct PasswordTracer {
    map: Map,
    neighbors_grid: NeighborsGrid,
    instructions: Instructions,
    states: Vec<TraceState>,
}

struct TraceStateIter<'a> {
    map: &'a Map,
    neighbors_grid: &'a NeighborsGrid,
    instruction_iter: Iter<'a, Instruction>,
    state: TraceState,
    tiles_remaining: u8,
    cube: bool,
}

impl<'a> Iterator for TraceStateIter<'a> {
    type Item = TraceState;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.tiles_remaining > 0_u8 {
                let state: TraceState = self
                    .neighbors_grid
                    .0
                    .get(self.state.pos)
                    .unwrap()
                    .next(&self.state, self.cube);

                if *self.map.0.get(state.pos).unwrap() == MapCell::Open {
                    self.tiles_remaining -= 1_u8;
                    self.state = state;

                    return Some(self.state.clone());
                }

                self.tiles_remaining = 0_u8;
            }

            match self.instruction_iter.next() {
                None => return None,
                Some(Instruction::Move(tiles_remaining)) => {
                    self.tiles_remaining = *tiles_remaining;
                }
                Some(Instruction::Turn { left }) => {
                    self.state.dir = self.state.dir.turn(*left);

                    return Some(self.state.clone());
                }
            }
        }
    }
}

impl PasswordTracer {
    fn try_from_map_and_instructions(
        map: Map,
        instructions: Instructions,
    ) -> Result<Self, ConstructNeighborsGridError> {
        let neighbors_grid: NeighborsGrid = NeighborsGrid::try_new(&map)?;
        let states: Vec<TraceState> = Vec::new();

        let mut password_tracer: Self = Self {
            map,
            neighbors_grid,
            instructions,
            states,
        };

        password_tracer.reset();

        Ok(password_tracer)
    }

    fn reset(&mut self) {
        self.states.clear();

        let pos: IVec2 = CellIter2D::until_boundary(&self.map.0, IVec2::ZERO, Direction::East)
            .find(|pos| *self.map.0.get(*pos).unwrap() == MapCell::Open)
            .unwrap();

        self.states.push(TraceState {
            pos,
            dir: Direction::East,
        });
    }

    fn run(&mut self, cube: bool) -> &mut Self {
        if self.states.len() != 1_usize {
            self.reset();
        }

        let (trace_state_iter, states) = self.iter(cube);

        for state in trace_state_iter {
            states.push(state);
        }

        self
    }

    fn iter<'a>(&'a mut self, cube: bool) -> (TraceStateIter<'a>, &'a mut Vec<TraceState>) {
        let Self {
            map,
            neighbors_grid,
            instructions,
            states,
        } = self;

        (
            TraceStateIter {
                map,
                neighbors_grid,
                instruction_iter: instructions.0.iter(),
                state: states[0_usize].clone(),
                tiles_remaining: 0_u8,
                cube,
            },
            states,
        )
    }

    fn as_string_for_state_range<R: SliceIndex<[TraceState], Output = [TraceState]>>(
        &self,
        range: R,
    ) -> String {
        let mut map_cell_grid: Grid2D<MapCell> = self.map.0.clone();

        for trace_state in self.states[range].iter() {
            *map_cell_grid.get_mut(trace_state.pos).unwrap() = trace_state.dir.into();
        }

        map_cell_grid.into()
    }

    fn as_string(&self) -> String {
        self.as_string_for_state_range(..)
    }

    fn final_password(&self) -> i32 {
        let last_state: TraceState = self.states.last().cloned().unwrap();
        let pos: IVec2 = last_state.pos + IVec2::ONE;

        1_000_i32 * pos.y + 4 * pos.x + last_state.dir.prev() as i32
    }
}

#[derive(Debug, PartialEq)]
pub enum ParsePasswordTracerError {
    NoMapToken,
    FailedToParseMap(()),
    NoInstructionsToken,
    FailedToParseInstructions(ParseInstructionError),
    ExtraTokenFound,
    FailedToConstructNeighbors(ConstructNeighborsGridError),
}

impl TryFrom<&str> for PasswordTracer {
    type Error = ParsePasswordTracerError;

    fn try_from(password_tracer_str: &str) -> Result<Self, Self::Error> {
        use ParsePasswordTracerError::*;

        let mut token_iter: Split<&str> = password_tracer_str.split("\n\n");

        let map: Map = token_iter
            .next()
            .ok_or(NoMapToken)?
            .try_into()
            .map_err(FailedToParseMap)?;
        let instructions: Instructions = token_iter
            .next()
            .ok_or(NoInstructionsToken)?
            .try_into()
            .map_err(FailedToParseInstructions)?;

        if token_iter.next().is_some() {
            Err(ExtraTokenFound)
        } else {
            PasswordTracer::try_from_map_and_instructions(map, instructions)
                .map_err(FailedToConstructNeighbors)
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(PasswordTracer);

impl Solution {
    fn final_password_2d(&mut self) -> i32 {
        self.0.run(false).final_password()
    }

    fn final_password_3d(&mut self) -> i32 {
        self.0.run(true).final_password()
    }

    fn as_string(&self) -> String {
        self.0.as_string()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.final_password_2d());

        if args.verbose {
            println!("self.as_string():\n\n{}", self.as_string());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.final_password_3d());

        if args.verbose {
            println!("self.as_string():\n\n{}", self.as_string());
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = ParsePasswordTracerError;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const MAP_STR: &str = concat!(
        "        ...#\n",
        "        .#..\n",
        "        #...\n",
        "        ....\n",
        "...#.......#\n",
        "........#...\n",
        "..#....#....\n",
        "..........#.\n",
        "        ...#....\n",
        "        .....#..\n",
        "        .#......\n",
        "        ......#.",
    );
    const INSTRUCTIONS_STR: &str = "10R5L5R10L4R5L5";
    const MAP_GRID_2D_STRING_STRING: &str = concat!(
        "        ...#    \n",
        "        .#..    \n",
        "        #...    \n",
        "        ....    \n",
        "...#.......#    \n",
        "........#...    \n",
        "..#....#....    \n",
        "..........#.    \n",
        "        ...#....\n",
        "        .....#..\n",
        "        .#......\n",
        "        ......#.\n",
    );
    const PASSWORD_TRACER_STR: &str = concat!(
        "        ...#\n",
        "        .#..\n",
        "        #...\n",
        "        ....\n",
        "...#.......#\n",
        "........#...\n",
        "..#....#....\n",
        "..........#.\n",
        "        ...#....\n",
        "        .....#..\n",
        "        .#......\n",
        "        ......#.\n",
        "\n",
        "10R5L5R10L4R5L5",
    );
    const PASSWORD_TRACER_STRING: &str = concat!(
        "        >>v#    \n",
        "        .#v.    \n",
        "        #.v.    \n",
        "        ..v.    \n",
        "...#...v..v#    \n",
        ">>>v...>#.>>    \n",
        "..#v...#....    \n",
        "...>>>>v..#.    \n",
        "        ...#....\n",
        "        .....#..\n",
        "        .#......\n",
        "        ......#.\n",
    );
    const PASSWORD_TRACER_CUBE_STRING: &str = concat!(
        "        >>v#    \n",
        "        .#v.    \n",
        "        #.v.    \n",
        "        ..v.    \n",
        "...#..^...v#    \n",
        ".>>>>>^.#.>>    \n",
        ".^#....#....    \n",
        ".^........#.    \n",
        "        ...#..v.\n",
        "        .....#v.\n",
        "        .#v<<<<.\n",
        "        ..v...#.\n",
    );

    fn instructions() -> &'static Instructions {
        static ONCE_LOCK: OnceLock<Instructions> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            use Instruction::*;

            Instructions(vec![
                Move(10_u8),
                Turn { left: false },
                Move(5_u8),
                Turn { left: true },
                Move(5_u8),
                Turn { left: false },
                Move(10_u8),
                Turn { left: true },
                Move(4_u8),
                Turn { left: false },
                Move(5_u8),
                Turn { left: true },
                Move(5_u8),
            ])
        })
    }

    fn password_tracer() -> &'static PasswordTracer {
        static ONCE_LOCK: OnceLock<PasswordTracer> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| PasswordTracer::try_from(PASSWORD_TRACER_STR).unwrap())
    }

    #[test]
    fn test_map_try_from_str() {
        assert_eq!(
            Map::try_from(MAP_STR).map(|map| String::from(map.0)),
            Ok(MAP_GRID_2D_STRING_STRING.into())
        );
    }

    #[test]
    fn test_instructions_try_from_str() {
        assert_eq!(INSTRUCTIONS_STR.try_into().as_ref(), Ok(instructions()));
    }

    #[test]
    fn test_password_tracer_run() {
        assert_eq!(
            password_tracer().clone().run(false).as_string(),
            PASSWORD_TRACER_STRING.to_owned()
        );
    }

    #[test]
    fn test_password_tracer_final_password() {
        assert_eq!(
            password_tracer().clone().run(false).final_password(),
            6_032_i32
        );
    }

    #[test]
    fn test_password_tracer_run_cube() {
        assert_eq!(
            password_tracer().clone().run(true).as_string(),
            PASSWORD_TRACER_CUBE_STRING.to_owned()
        );
    }

    #[test]
    fn test_password_tracer_final_password_cube() {
        assert_eq!(
            password_tracer().clone().run(true).final_password(),
            5_031_i32
        );
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(
            (0_i32..50_i32 * 50_i32)
                .map(|i| (i as f32).sqrt())
                .filter(|f| (*f % 1.0_f32) == 0.0_f32)
                .map(|f| f as i32)
                .collect::<Vec<i32>>(),
            (0_i32..50_i32).collect::<Vec<i32>>()
        );
    }

    /// Tests related to cube nets
    ///
    /// # Nets
    ///
    /// ```
    /// // 0:
    /// //
    /// // ###
    /// //  #
    /// //  #
    /// //  #
    ///
    /// // 1:
    /// //
    /// // ##
    /// //  ##
    /// //  #
    /// //  #
    ///
    /// // 2:
    /// //
    /// // ##
    /// //  #
    /// //  ##
    /// //  #
    ///
    /// // 3:
    /// //
    /// // ##
    /// //  #
    /// //  #
    /// //  ##
    ///
    /// // 4.
    /// //
    /// //  #
    /// // ###
    /// //  #
    /// //  #
    ///
    /// // 5.
    /// //
    /// //  #
    /// // ##
    /// //  ##
    /// //  #
    ///
    /// // 6.
    /// //
    /// // ##
    /// //  #
    /// //  ##
    /// //   #
    ///
    /// // 7.
    /// //
    /// //   #
    /// // ###
    /// //  #
    /// //  #
    ///
    /// // 8.
    /// //
    /// // #
    /// // #
    /// // ##
    /// //  #
    /// //  #
    ///
    /// // 9.
    /// //
    /// // #
    /// // ##
    /// //  ##
    /// //  #
    ///
    /// // 10.
    /// //
    /// // #
    /// // ##
    /// //  ##
    /// //   #
    /// ```
    mod cube_net {
        use self::coincident_corners::COINCIDENT_CORNERS;

        use {super::*, coincident_corners::*, cube_face_corner_grid_fns::*, perimeters::*};

        const CUBE_NET_PERIMETER: usize = NeighborsGrid::CUBE_NET_PERIMETER;
        const CUBE_NET_COUNT: usize = 11_usize;

        fn cube_face_corner_grids() -> &'static Vec<Grid2D<NetCell>> {
            static ONCE_LOCK: OnceLock<Vec<Grid2D<NetCell>>> = OnceLock::new();

            ONCE_LOCK.get_or_init(new_cube_face_corner_grids)
        }

        #[test]
        fn test_password_tracer_perimeter() {
            for (index, (cube_face_corner_grid, expected_perimeter)) in cube_face_corner_grids()
                .iter()
                .zip(PERIMETERS.iter())
                .enumerate()
            {
                let mut actual_perimeter: Perimeter = [Direction::North; CUBE_NET_PERIMETER];
                let mut perimeter_corners: [IVec2; CUBE_NET_PERIMETER] = Default::default();

                NeighborsGrid::perimeter(
                    cube_face_corner_grid,
                    &mut actual_perimeter,
                    &mut perimeter_corners,
                );

                assert_eq!(
                    actual_perimeter, *expected_perimeter,
                    "Perimeter {index} did not match expected"
                );
            }
        }

        #[test]
        fn test_password_tracer_coincident_corners() {
            for (index, (perimeter, expected_coincident_corners)) in
                PERIMETERS.iter().zip(COINCIDENT_CORNERS.iter()).enumerate()
            {
                let mut actual_coincident_corners: CoincidentCorners = Default::default();

                NeighborsGrid::coincident_corners(perimeter, &mut actual_coincident_corners);

                if actual_coincident_corners != *expected_coincident_corners {
                    panic!(
                        "Coincident corners {index} did not match expected\n\
                        \n\
                        actual:\n\
                        \n\
                        {:#?}\n\
                        \n\
                        expected:\n\
                        \n\
                        {:#?}",
                        NeighborsGrid::unpack_indices(&actual_coincident_corners),
                        NeighborsGrid::unpack_indices(expected_coincident_corners)
                    );
                }
            }
        }

        mod cube_face_corner_grid_fns {
            use super::*;

            macro_rules! grid_vec {
                [ $( [ $( $b:literal ),+ ], )+ ] => {
                    [ $( $(
                        NetCell {
                            is_not_void: $b & 2 != 0,
                            is_corner: $b & 1 != 0,
                        },
                    )+ )+ ].into_iter().collect::<Vec<NetCell>>()
                };
            }

            pub(crate) fn new_cube_face_corner_grids() -> Vec<Grid2D<NetCell>> {
                vec![
                    cube_face_corner_grid_0(),
                    cube_face_corner_grid_1(),
                    cube_face_corner_grid_2(),
                    cube_face_corner_grid_3(),
                    cube_face_corner_grid_4(),
                    cube_face_corner_grid_5(),
                    cube_face_corner_grid_6(),
                    cube_face_corner_grid_7(),
                    cube_face_corner_grid_8(),
                    cube_face_corner_grid_9(),
                    cube_face_corner_grid_10(),
                ]
            }

            fn cube_face_corner_grid_0() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 3, 3, 1],
                        [1, 3, 1, 1],
                        [0, 3, 1, 0],
                        [0, 3, 1, 0],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_1() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 3, 1, 0],
                        [1, 3, 3, 1],
                        [0, 3, 1, 1],
                        [0, 3, 1, 0],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_2() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 3, 1, 0],
                        [1, 3, 1, 0],
                        [0, 3, 3, 1],
                        [0, 3, 1, 1],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_3() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 3, 1, 0],
                        [1, 3, 1, 0],
                        [0, 3, 1, 0],
                        [0, 3, 3, 1],
                        [0, 1, 1, 1],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_4() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [0, 3, 1, 0],
                        [3, 3, 3, 1],
                        [1, 3, 1, 1],
                        [0, 3, 1, 0],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_5() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [0, 3, 1, 0],
                        [3, 3, 1, 0],
                        [1, 3, 3, 1],
                        [0, 3, 1, 1],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_6() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 3, 1, 0],
                        [1, 3, 1, 0],
                        [0, 3, 3, 1],
                        [0, 1, 3, 1],
                        [0, 0, 1, 1],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_7() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [0, 0, 3, 1],
                        [3, 3, 3, 1],
                        [1, 3, 1, 1],
                        [0, 3, 1, 0],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_8() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 1, 0],
                        [3, 1, 0],
                        [3, 3, 1],
                        [1, 3, 1],
                        [0, 3, 1],
                        [0, 1, 1],
                    ],
                    IVec2::new(3_i32, 6_i32),
                )
            }

            fn cube_face_corner_grid_9() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 1, 0, 0],
                        [3, 3, 1, 0],
                        [1, 3, 3, 1],
                        [0, 3, 1, 1],
                        [0, 1, 1, 0],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_10() -> Grid2D<NetCell> {
                cube_face_corner_grid_from_vec_and_dimensions(
                    grid_vec![
                        [3, 1, 0, 0],
                        [3, 3, 1, 0],
                        [1, 3, 3, 1],
                        [0, 1, 3, 1],
                        [0, 0, 1, 1],
                    ],
                    IVec2::new(4_i32, 5_i32),
                )
            }

            fn cube_face_corner_grid_from_vec_and_dimensions(
                v: Vec<NetCell>,
                dimensions: IVec2,
            ) -> Grid2D<NetCell> {
                let mut grid: Grid2D<NetCell> = Grid2D::default(dimensions);

                grid.cells_mut().copy_from_slice(&v);

                grid
            }
        }

        mod perimeters {
            use super::*;
            use Direction::{East as E, North as N, South as S, West as W};

            pub type Perimeter = [Direction; CUBE_NET_PERIMETER];

            pub const PERIMETERS: [Perimeter; CUBE_NET_COUNT] = [
                PERIMITER_0,
                PERIMITER_1,
                PERIMITER_2,
                PERIMITER_3,
                PERIMITER_4,
                PERIMITER_5,
                PERIMITER_6,
                PERIMETER_7,
                PERIMETER_8,
                PERIMETER_9,
                PERIMETER_10,
            ];

            const PERIMITER_0: Perimeter = [E, E, E, S, W, S, S, S, W, N, N, N, W, N];
            const PERIMITER_1: Perimeter = [E, E, S, E, S, W, S, S, W, N, N, N, W, N];
            const PERIMITER_2: Perimeter = [E, E, S, S, E, S, W, S, W, N, N, N, W, N];
            const PERIMITER_3: Perimeter = [E, E, S, S, S, E, S, W, W, N, N, N, W, N];
            const PERIMITER_4: Perimeter = [E, S, E, S, W, S, S, W, N, N, W, N, E, N];
            const PERIMITER_5: Perimeter = [E, S, S, E, S, W, S, W, N, N, W, N, E, N];
            const PERIMITER_6: Perimeter = [E, E, S, S, E, S, S, W, N, W, N, N, W, N];
            const PERIMETER_7: Perimeter = [E, S, S, W, S, S, W, N, N, W, N, E, E, N];
            const PERIMETER_8: Perimeter = [E, S, S, E, S, S, S, W, N, N, W, N, N, N];
            const PERIMETER_9: Perimeter = [E, S, E, S, E, S, W, S, W, N, N, W, N, N];
            const PERIMETER_10: Perimeter = [E, S, E, S, E, S, S, W, N, W, N, W, N, N];
        }

        mod coincident_corners {
            use super::*;

            pub type CoincidentCorners = [u16; CUBE_NET_PERIMETER];

            pub const COINCIDENT_CORNERS: &[CoincidentCorners] = &[
                COINCIDENT_CORNERS_0,
                COINCIDENT_CORNERS_1,
                COINCIDENT_CORNERS_2,
                COINCIDENT_CORNERS_3,
                COINCIDENT_CORNERS_4,
            ];

            #[allow(dead_code)]
            const COINCIDENT_CORNERS_: CoincidentCorners = [
                0b00000000000001_u16,
                0b00000000000010_u16,
                0b00000000000100_u16,
                0b00000000001000_u16,
                0b00000000010000_u16,
                0b00000000100000_u16,
                0b00000001000000_u16,
                0b00000010000000_u16,
                0b00000100000000_u16,
                0b00001000000000_u16,
                0b00010000000000_u16,
                0b00100000000000_u16,
                0b01000000000000_u16,
                0b10000000000000_u16,
            ];

            const COINCIDENT_CORNERS_0: CoincidentCorners = [
                0b00010000000001_u16,
                0b00001000000010_u16,
                0b00000100000100_u16,
                0b00000010001000_u16,
                0b00000001010000_u16,
                0b00000000100000_u16, // 5
                0b00000001010000_u16,
                0b00000010001000_u16,
                0b00000100000100_u16,
                0b00001000000010_u16,
                0b00010000000001_u16,
                0b10100000000000_u16,
                0b01000000000000_u16, // 12
                0b10100000000000_u16,
            ];
            const COINCIDENT_CORNERS_1: CoincidentCorners = [
                0b00010000000001_u16,
                0b00001000000010_u16,
                0b00000100010100_u16,
                0b00000000001000_u16, // 3
                0b00000100010100_u16,
                0b00000010100000_u16,
                0b00000001000000_u16, // 6
                0b00000010100000_u16,
                0b00000100010100_u16,
                0b00001000000010_u16,
                0b00010000000001_u16,
                0b10100000000000_u16,
                0b01000000000000_u16, // 12
                0b10100000000000_u16,
            ];
            const COINCIDENT_CORNERS_2: CoincidentCorners = [
                0b00010000000001_u16,
                0b00001000000010_u16,
                0b00000101000100_u16,
                0b00000000101000_u16,
                0b00000000010000_u16, // 4
                0b00000000101000_u16,
                0b00000101000100_u16,
                0b00000010000000_u16,
                0b00000101000100_u16,
                0b00001000000010_u16,
                0b00010000000001_u16,
                0b10100000000000_u16,
                0b01000000000000_u16, // 12
                0b10100000000000_u16,
            ];
            const COINCIDENT_CORNERS_3: CoincidentCorners = [
                0b00010000000001_u16,
                0b00001000000010_u16,
                0b00000100000100_u16,
                0b00000010001000_u16,
                0b00000001010000_u16,
                0b00000000100000_u16, // 5
                0b00000001010000_u16,
                0b00000010001000_u16,
                0b00000100000100_u16,
                0b00001000000010_u16,
                0b00010000000001_u16,
                0b10100000000000_u16,
                0b01000000000000_u16, // 12
                0b10100000000000_u16,
            ];
            const COINCIDENT_CORNERS_4: CoincidentCorners = [
                0b01000100000001_u16,
                0b00000010001010_u16,
                0b00000000000100_u16, // 2
                0b00000010001010_u16,
                0b00000001010000_u16,
                0b00000000100000_u16, // 5
                0b00000001010000_u16,
                0b00000010001010_u16,
                0b01000100000001_u16,
                0b00101000000000_u16,
                0b00010000000000_u16, // 10
                0b00101000000000_u16,
                0b01000100000001_u16,
                0b10000000000000_u16, // 13
            ];
        }
    }
}
