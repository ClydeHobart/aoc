use std::slice::SliceIndex;

use {
    aoc_2022::*,
    glam::IVec2,
    std::{
        mem::{size_of, transmute},
        num::ParseIntError,
        slice::Iter,
        str::{FromStr, Split},
    },
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate static_assertions;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(u8)]
enum MapCell {
    #[default]
    Void = b' ',
    Open = b'.',
    Wall = b'#',
}

impl MapCell {
    const VOID_U8: u8 = Self::Void as u8;
    const OPEN_U8: u8 = Self::Open as u8;
    const WALL_U8: u8 = Self::Wall as u8;
}

#[derive(Debug, PartialEq)]
struct InvalidMapCellByte(u8);

impl TryFrom<u8> for MapCell {
    type Error = InvalidMapCellByte;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            Self::VOID_U8 => Ok(Self::Void),
            Self::OPEN_U8 => Ok(Self::Open),
            Self::WALL_U8 => Ok(Self::Wall),
            invalid_byte => Err(InvalidMapCellByte(invalid_byte)),
        }
    }
}

#[derive(Clone)]
struct Map(Grid2D<MapCell>);

impl Map {}

impl From<Map> for Grid2DString {
    fn from(map: Map) -> Self {
        const_assert_eq!(size_of::<MapCell>(), size_of::<u8>());
        const_assert_eq!(size_of::<Map>(), size_of::<Grid2DString>());

        // SAFETY: Currently, both `Map` and `MapString` are just new-type pattern structs around
        // `Grid2D` structs of 1-Byte-sized elements. The const asserts above will hopefully catch
        // any issues, should that not be the case at some point
        unsafe { transmute(map) }
    }
}

impl TryFrom<&str> for Map {
    type Error = InvalidMapCellByte;

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
                *map.0.get_mut(pos).unwrap() = map_cell_byte.try_into()?;
            }
        }

        Ok(map)
    }
}

#[derive(Clone, Copy, Default)]
struct NeighborPositions([u8; 4_usize]);

impl NeighborPositions {
    fn next(self, trace_state: &TraceState) -> IVec2 {
        let axis_a: IVec2 = trace_state.dir.vec().abs();
        let axis_b: IVec2 = IVec2::ONE - axis_a;

        axis_b * trace_state.pos + self.0[trace_state.dir as usize] as i32 * axis_a
    }
}

#[derive(Default)]
struct MapCellToNeighborPositionsGridVisitor {
    first_position: Option<i32>,
}

impl GridVisitor for MapCellToNeighborPositionsGridVisitor {
    type Old = MapCell;
    type New = NeighborPositions;

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
                new.0[dir as usize] = abs_sum_2d(next * dir_vec) as u8;
            } else {
                new.0[dir as usize] = self.first_position.unwrap() as u8;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    Move(u8),
    Turn { left: bool },
}

#[derive(Debug, PartialEq)]
enum ParseInstructionError {
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

#[derive(Clone)]
struct TraceState {
    pos: IVec2,
    dir: Direction,
}

impl TraceState {
    fn byte(&self) -> u8 {
        use Direction::*;

        match self.dir {
            North => b'^',
            East => b'>',
            South => b'v',
            West => b'<',
        }
    }
}

#[derive(Clone)]
struct PasswordTracer {
    map: Map,
    neighbor_positions: Grid2D<NeighborPositions>,
    instructions: Instructions,
    states: Vec<TraceState>,
}

struct TraceStateIter<'a> {
    map: &'a Map,
    neighbor_positions: &'a Grid2D<NeighborPositions>,
    instruction_iter: Iter<'a, Instruction>,
    state: TraceState,
    tiles_remaining: u8,
}

impl<'a> Iterator for TraceStateIter<'a> {
    type Item = TraceState;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.tiles_remaining > 0_u8 {
                let pos: IVec2 = self
                    .neighbor_positions
                    .get(self.state.pos)
                    .unwrap()
                    .next(&self.state);

                if *self.map.0.get(pos).unwrap() == MapCell::Open {
                    self.tiles_remaining -= 1_u8;
                    self.state.pos = pos;

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

    fn run(&mut self) -> &mut Self {
        if self.states.len() != 1_usize {
            self.reset();
        }

        let (trace_state_iter, states) = self.iter();

        for state in trace_state_iter {
            states.push(state);
        }

        self
    }

    fn iter(&mut self) -> (TraceStateIter, &mut Vec<TraceState>) {
        let Self {
            map,
            neighbor_positions,
            instructions,
            states,
        } = self;

        (
            TraceStateIter {
                map,
                neighbor_positions,
                instruction_iter: instructions.0.iter(),
                state: states[0_usize].clone(),
                tiles_remaining: 0_u8,
            },
            states,
        )
    }

    fn try_as_string_for_state_range<R: SliceIndex<[TraceState], Output = [TraceState]>>(
        &self,
        range: R,
    ) -> Grid2DStringResult {
        let mut grid_2d_string: Grid2DString = self.map.clone().into();

        let grid: &mut Grid2D<u8> = grid_2d_string.grid_mut();

        for trace_state in self.states[range].iter() {
            *grid.get_mut(trace_state.pos).unwrap() = trace_state.byte();
        }

        grid_2d_string.try_as_string()
    }

    fn try_as_string(&self) -> Grid2DStringResult {
        self.try_as_string_for_state_range(..)
    }

    fn final_password(&self) -> i32 {
        let last_state: TraceState = self.states.last().cloned().unwrap();
        let pos: IVec2 = last_state.pos + IVec2::ONE;

        1_000_i32 * pos.y + 4 * pos.x + last_state.dir.prev() as i32
    }
}

#[derive(Debug, PartialEq)]
enum ParsePasswordTracerError {
    NoMapToken,
    FailedToParseMap(InvalidMapCellByte),
    NoInstructionsToken,
    FailedToParseInstructions(ParseInstructionError),
    ExtraTokenFound,
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
            let neighbor_positions: Grid2D<NeighborPositions> =
                MapCellToNeighborPositionsGridVisitor::visit_grid(&map.0);
            let states: Vec<TraceState> = Vec::new();

            let mut password_tracer: Self = Self {
                map,
                neighbor_positions,
                instructions,
                states,
            };

            password_tracer.reset();

            Ok(password_tracer)
        }
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day22.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match PasswordTracer::try_from(input) {
                    Ok(mut password_tracer) => {
                        password_tracer.run();

                        dbg!(password_tracer.final_password());

                        if args.verbose {
                            println!(
                                "password_tracer.try_as_string():\n\n{}",
                                password_tracer
                                    .try_as_string()
                                    .unwrap_or_else(|err| format!("{err:#?}"))
                            );
                        }
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
    const MAP_GRID2DSTRING_STRING: &str = concat!(
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

    lazy_static! {
        static ref INSTRUCTIONS: Instructions = instructions();
        static ref PASSWORD_TRACER: PasswordTracer = password_tracer();
    }

    #[test]
    fn test_map_try_from_str() {
        assert_eq!(
            Map::try_from(MAP_STR)
                .map(Grid2DString::from)
                .map(|grid_2d_string| grid_2d_string.try_as_string()),
            Ok(Ok(MAP_GRID2DSTRING_STRING.into()))
        );
    }

    #[test]
    fn test_instructions_try_from_str() {
        assert_eq!(INSTRUCTIONS_STR.try_into().as_ref(), Ok(&*INSTRUCTIONS));
    }

    #[test]
    fn test_password_tracer_run() {
        assert_eq!(
            PASSWORD_TRACER.clone().run().try_as_string(),
            Ok(PASSWORD_TRACER_STRING.into())
        );
    }

    #[test]
    fn test_password_tracer_final_password() {
        assert_eq!(PASSWORD_TRACER.clone().run().final_password(), 6_032_i32);
    }

    fn instructions() -> Instructions {
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
    }

    fn password_tracer() -> PasswordTracer {
        PasswordTracer::try_from(PASSWORD_TRACER_STR).unwrap()
    }
}
