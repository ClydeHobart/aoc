use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::map_opt,
        error::Error,
        multi::many_m_n,
        sequence::{delimited, terminated, tuple},
        Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        mem::{transmute, MaybeUninit},
        ops::{BitAnd, Deref, DerefMut, Range, RangeInclusive},
        sync::OnceLock,
    },
};

#[derive(PartialEq)]
struct Params {
    hallway_start: IVec2,
    hallway_len: usize,
    side_room_len: usize,
    amphipod_types: usize,
}

impl Params {
    const fn new(side_room_len: usize) -> Self {
        Self {
            hallway_start: IVec2::ONE,
            hallway_len: 11_usize,
            side_room_len,
            amphipod_types: Cell::AMPHIPOD_TYPES,
        }
    }

    const fn hallway_len_u8(&self) -> u8 {
        self.hallway_len as u8
    }

    const fn side_room_len_u8(&self) -> u8 {
        self.side_room_len as u8
    }

    const fn amphipod_types_u8(&self) -> u8 {
        self.amphipod_types as u8
    }

    const fn cells(&self) -> usize {
        self.hallway_len + self.amphipod_types * self.side_room_len
    }

    const fn invalid_cell_index_start(&self) -> usize {
        (self.hallway_len / 2_usize) - self.amphipod_types + 1_usize
    }

    const fn burrow_dimensions(&self) -> IVec2 {
        IVec2::new(
            self.hallway_len as i32 + 2_i32,
            self.side_room_len as i32 + 3_i32,
        )
    }

    const fn side_room_start(&self) -> IVec2 {
        IVec2::new(
            self.hallway_start.x + self.invalid_cell_index_start() as i32,
            self.hallway_start.y + 1_i32,
        )
    }
}

/// SAFETY: Type must be a `bitvec::array::BitArray`
unsafe trait BitArrayTrait:
    BitAnd<Self, Output = Self> + Copy + Deref<Target = BitSlice<u32, Lsb0>> + DerefMut + Sized
{
}

/// SAFETY: Trivial
unsafe impl<const SIZE: usize> BitArrayTrait for BitArray<[u32; SIZE], Lsb0> {}

trait ParamsTrait: Sized {
    type CellIndexBitArray: BitArrayTrait;
    const PARAMS: Params;

    fn cell_index_bit_array_zero() -> Self::CellIndexBitArray {
        // SAFETY: If a type is meeting the safety requirements to implement `BitArrayTrait`,
        // this operation is safe
        unsafe { MaybeUninit::zeroed().assume_init() }
    }

    fn invalid_cell_indices() -> Self::CellIndexBitArray {
        let mut invalid_cell_indices: Self::CellIndexBitArray = Self::cell_index_bit_array_zero();
        let mut invalid_cell_index: usize = Self::PARAMS.invalid_cell_index_start();

        for _ in 0_usize..Self::PARAMS.amphipod_types {
            invalid_cell_indices.set(invalid_cell_index, true);
            invalid_cell_index += 2_usize;
        }

        invalid_cell_indices
    }

    fn cell_positions() -> Vec<IVec2> {
        let side_room_start: IVec2 = Self::PARAMS.side_room_start();

        (0_usize..Self::PARAMS.hallway_len)
            .into_iter()
            .map(|x_delta| Self::PARAMS.hallway_start + x_delta as i32 * IVec2::X)
            .chain(
                (0_usize..Self::PARAMS.side_room_len)
                    .into_iter()
                    .flat_map(|y_delta| {
                        (0_usize..Self::PARAMS.amphipod_types)
                            .into_iter()
                            .map(move |x_delta| {
                                side_room_start + IVec2::new(2_i32 * x_delta as i32, y_delta as i32)
                            })
                    }),
            )
            .collect()
    }

    fn neighbor_map_once_lock() -> &'static OnceLock<NeighborMap<Self>> {
        match Self::PARAMS.side_room_len {
            2_usize => {
                static ONCE_LOCK: OnceLock<NeighborMap<SmallPositionState>> = OnceLock::new();

                assert!(Self::PARAMS == SmallPositionState::PARAMS);

                // SAFETY: Guaranteed by the above
                unsafe { transmute(&ONCE_LOCK) }
            }
            4_usize => {
                static ONCE_LOCK: OnceLock<NeighborMap<LargePositionState>> = OnceLock::new();

                assert!(Self::PARAMS == LargePositionState::PARAMS);

                // SAFETY: Guaranteed by the above
                unsafe { transmute(&ONCE_LOCK) }
            }
            _ => unimplemented!(),
        }
    }
}

struct Neighbor<P: ParamsTrait> {
    cell_index: u8,
    distance: u8,
    vacant_cell_indices: <P as ParamsTrait>::CellIndexBitArray,
}

trait NeighborMapTrait {
    type OnceLock;
}

struct NeighborMap<P: ParamsTrait> {
    neighbors: Vec<Neighbor<P>>,
    neighbor_ranges: Vec<Range<u16>>,
}

impl<P: ParamsTrait + 'static> NeighborMap<P> {
    fn get() -> &'static Self {
        P::neighbor_map_once_lock().get_or_init(|| {
            let hallway_y: i32 = P::PARAMS.hallway_start.y;
            let cell_positions: Vec<IVec2> = P::cell_positions();
            let cell_indices_map: HashMap<IVec2, u8> = cell_positions
                .iter()
                .copied()
                .enumerate()
                .map(|(cell_index, cell_position)| (cell_position, cell_index as u8))
                .collect();
            let get_cell_index = |cell_position: IVec2| -> u8 {
                cell_indices_map.get(&cell_position).copied().unwrap()
            };
            let try_new_neighbor = |from_cell_index: u8,
                                    to_cell_index: u8|
             -> Option<Neighbor<P>> {
                let mut vacant_cell_indices: P::CellIndexBitArray = P::cell_index_bit_array_zero();

                let mut set_cell_position = |cell_position: IVec2, vacant: bool| {
                    vacant_cell_indices.set(get_cell_index(cell_position).into(), vacant);
                };

                fn iter_cell_positions(
                    ranges: &[RangeInclusive<IVec2>],
                ) -> impl Iterator<Item = IVec2> + '_ {
                    ranges
                        .iter()
                        .cloned()
                        .filter(|range| *range.start() != *range.end())
                        .map(CellIter2D::try_from)
                        .flat_map(Result::unwrap)
                }

                fn sum_distance(ranges: &[RangeInclusive<IVec2>]) -> u8 {
                    ranges
                        .iter()
                        .map(|range| abs_sum_2d(*range.start() - *range.end()))
                        .sum::<i32>() as u8
                }

                let from: IVec2 = cell_positions[from_cell_index as usize];
                let to: IVec2 = cell_positions[to_cell_index as usize];
                let from_is_in_hallway: bool = from.y == hallway_y;
                let to_is_in_hallway: bool = to.y == hallway_y;
                let from_hallway_position: IVec2 = IVec2::new(from.x, hallway_y);
                let to_hallway_position: IVec2 = IVec2::new(to.x, hallway_y);

                match (from_is_in_hallway, to_is_in_hallway) {
                    // Amphipods are not allowed to move from a hallway position to another hallway
                    // position
                    (true, true) => None,
                    (false, false) => {
                        if from.x == to.x {
                            // There's no reason for an amphipod to move from one cell in a side
                            // room to a different cell in the same side room.
                            None
                        } else {
                            let ranges: [RangeInclusive<IVec2>; 3_usize] = [
                                // Add any positions in the same side room as `from`
                                from..=from_hallway_position,
                                // Add any positions in the hallway
                                from_hallway_position..=to_hallway_position,
                                // Add any positions in the same side room as `to`
                                to_hallway_position..=to,
                            ];

                            for cell_position in iter_cell_positions(&ranges) {
                                set_cell_position(cell_position, true);
                            }

                            // Clear `from`
                            set_cell_position(from, false);

                            let cell_index: u8 = to_cell_index;
                            let distance: u8 = sum_distance(&ranges);

                            Some(Neighbor {
                                cell_index,
                                distance,
                                vacant_cell_indices,
                            })
                        }
                    }
                    _ => {
                        let hallway_position: IVec2 = if from_is_in_hallway {
                            to_hallway_position
                        } else {
                            from_hallway_position
                        };
                        let ranges: [RangeInclusive<IVec2>; 2_usize] =
                            [from..=hallway_position, hallway_position..=to];

                        for cell_position in iter_cell_positions(&ranges) {
                            set_cell_position(cell_position, true);
                        }

                        // Clear `from`
                        set_cell_position(from, false);

                        let cell_index: u8 = to_cell_index;
                        let distance: u8 = sum_distance(&ranges);

                        Some(Neighbor {
                            cell_index,
                            distance,
                            vacant_cell_indices,
                        })
                    }
                }
            };

            let invalid_cell_indices: P::CellIndexBitArray = P::invalid_cell_indices();
            let cells: usize = P::PARAMS.cells();
            let cells_u8: u8 = cells as u8;
            let mut neighbors: Vec<Neighbor<P>> = Vec::new();
            let mut neighbor_ranges: Vec<Range<u16>> = Vec::with_capacity(cells);

            for from_cell_index in 0_u8..cells_u8 {
                let start: u16 = neighbors.len() as u16;

                if !invalid_cell_indices[from_cell_index as usize] {
                    for to_cell_index in 0_u8..cells_u8 {
                        if !invalid_cell_indices[to_cell_index as usize] {
                            if let Some(neighbor) = try_new_neighbor(from_cell_index, to_cell_index)
                            {
                                neighbors.push(neighbor);
                            }
                        }
                    }
                }

                let end: u16 = neighbors.len() as u16;

                neighbor_ranges.push(start..end);
            }

            Self {
                neighbors,
                neighbor_ranges,
            }
        })
    }

    pub fn neighbors(cell_index: u8) -> &'static [Neighbor<P>] {
        let neighbor_map: &Self = Self::get();

        &neighbor_map.neighbors[neighbor_map.neighbor_ranges[cell_index as usize].as_range_usize()]
    }

    pub fn distance(from_cell_index: u8, to_cell_index: u8) -> Option<u8> {
        let neighbors: &[Neighbor<P>] = Self::neighbors(from_cell_index);

        neighbors
            .binary_search_by_key(&to_cell_index, |neighbor| neighbor.cell_index)
            .ok()
            .map(|neighbor_index| neighbors[neighbor_index].distance)
    }
}

#[derive(Copy, Clone, PartialEq)]
enum IsCellIndexValidResult {
    Invalid,
    ValidUnlocked,
    ValidLocked,
}

pub type AmphipodArrays<const SIDE_ROOM_LEN: usize> = [[u8; SIDE_ROOM_LEN]; Cell::AMPHIPOD_TYPES];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(C, align(8))]
struct PositionState<const SIDE_ROOM_LEN: usize>(AmphipodArrays<SIDE_ROOM_LEN>);

impl<const SIDE_ROOM_LEN: usize> PositionState<SIDE_ROOM_LEN>
where
    Self: ParamsTrait,
{
    const INVALID: Self = Self(Self::invalid_amphipod_arrays());
    const ORGANIZED: Self = Self::organized();

    fn empty_cell_grid() -> Grid2D<Cell> {
        fn empty_cell_grid(params: &Params) -> Grid2D<Cell> {
            let burrow_dimensions: IVec2 = params.burrow_dimensions();
            let mut cells: Vec<Cell> =
                Vec::with_capacity((burrow_dimensions.x * burrow_dimensions.y) as usize);

            for _ in 0_i32..burrow_dimensions.x {
                cells.push(Cell::Wall);
            }

            cells.push(Cell::Wall);

            for _ in 0_usize..params.hallway_len {
                cells.push(Cell::Vacant);
            }

            cells.push(Cell::Wall);

            let invalid_cell_index_start: usize = params.invalid_cell_index_start();
            let mut push_row = |shoulder: Cell, side_room: Cell| {
                for _ in 0_usize..invalid_cell_index_start {
                    cells.push(shoulder);
                }

                for _ in 0_usize..params.amphipod_types {
                    cells.push(Cell::Wall);
                    cells.push(side_room);
                }

                cells.push(Cell::Wall);

                for _ in 0_usize..invalid_cell_index_start {
                    cells.push(shoulder);
                }
            };

            for side_room_row_index in 0_usize..params.side_room_len {
                push_row(
                    if side_room_row_index == 0_usize {
                        Cell::Wall
                    } else {
                        Cell::Void
                    },
                    Cell::Vacant,
                );
            }

            push_row(
                if params.side_room_len == 0_usize {
                    Cell::Wall
                } else {
                    Cell::Void
                },
                Cell::Wall,
            );

            Grid2D::try_from_cells_and_dimensions(cells, burrow_dimensions).unwrap()
        }

        empty_cell_grid(&Self::PARAMS)
    }

    fn as_cell_grid(&self) -> Grid2D<Cell> {
        let hallway_len_u8: u8 = Self::PARAMS.hallway_len_u8();
        let amphipod_types: i32 = Self::PARAMS.amphipod_types as i32;
        let mut cell_grid: Grid2D<Cell> = Self::empty_cell_grid();

        for (amphipod_index, cell_indices) in self.0.iter().copied().enumerate() {
            let cell: Cell = Cell::try_from(amphipod_index as u8 + b'A').unwrap();

            for cell_index in cell_indices {
                *cell_grid
                    .get_mut(
                        if let Some(side_room_cell_index) =
                            cell_index.checked_sub(hallway_len_u8).map(i32::from)
                        {
                            IVec2::new(
                                2_i32 * (side_room_cell_index % amphipod_types as i32) + 3_i32,
                                (side_room_cell_index / amphipod_types as i32) + 2_i32,
                            )
                        } else {
                            IVec2::new(cell_index as i32, 0_i32) + Self::PARAMS.hallway_start
                        },
                    )
                    .unwrap() = cell;
            }
        }

        cell_grid
    }

    fn as_string_safe(&self) -> String {
        Grid2DString::from(self.as_cell_grid())
            .try_as_string()
            .unwrap_or_else(|_| "[invalid]".into())
    }

    fn present_cell_indices(self) -> <Self as ParamsTrait>::CellIndexBitArray {
        let mut present_cell_indices: <Self as ParamsTrait>::CellIndexBitArray =
            Self::cell_index_bit_array_zero();

        for cell_index in self.0.into_iter().flat_map(IntoIterator::into_iter) {
            present_cell_indices.set(cell_index as usize, true);
        }

        present_cell_indices
    }

    #[allow(dead_code)]
    fn try_organize_amphipods_dfs(self) -> Option<(Vec<Self>, u32)> {
        let mut result: OrganizeAmphipodsResult<SIDE_ROOM_LEN> = Default::default();

        OrganizeAmphipods {
            start: self,
            result: &mut result,
        }
        .depth_first_search()
    }

    fn try_organize_amphipods_astar(self) -> Option<(Vec<Self>, u32)> {
        let mut result: OrganizeAmphipodsResult<SIDE_ROOM_LEN> = Default::default();

        AStar::run(&mut OrganizeAmphipods {
            start: self,
            result: &mut result,
        })
        .and_then(|path| result.energy_to_organize().map(|energy| (path, energy)))
    }

    #[allow(dead_code)]
    fn try_organize_amphipods_dijkstra(self) -> Option<(Vec<Self>, u32)> {
        let mut result: OrganizeAmphipodsResult<SIDE_ROOM_LEN> = Default::default();

        Dijkstra::run(&mut OrganizeAmphipods {
            start: self,
            result: &mut result,
        })
        .and_then(|path| result.energy_to_organize().map(|energy| (path, energy)))
    }

    fn try_organize_amphipods(self) -> Option<(Vec<Self>, u32)> {
        self.try_organize_amphipods_astar()
    }

    fn previous_entry(self) -> PreviousEntry<SIDE_ROOM_LEN> {
        PreviousEntry {
            previous: Self::INVALID,
            energy: 0_u32,
        }
    }

    fn sort(&mut self) {
        for side_room in self.0.iter_mut() {
            side_room.sort();
        }
    }

    fn heuristic(self) -> u32 {
        let hallway_len_u8: u8 = Self::PARAMS.hallway_len_u8();
        let amphipod_types_u8: u8 = Self::PARAMS.amphipod_types_u8();

        self.0
            .into_iter()
            .enumerate()
            .flat_map(|(amphipod_index, amphipod_cell_indices)| {
                amphipod_cell_indices
                    .into_iter()
                    .map(move |amphipod_cell_index| {
                        (
                            amphipod_index as u8,
                            Cell::energy_per_cell_for_amphipod_index(amphipod_index),
                            amphipod_cell_index,
                        )
                    })
            })
            .filter(|(amphipod_index, _, amphipod_cell_index)| {
                amphipod_cell_index.checked_sub(hallway_len_u8).map_or(
                    true,
                    |side_room_cell_index| {
                        (side_room_cell_index % amphipod_types_u8) != *amphipod_index
                    },
                )
            })
            .map(|(amphipod_index, energy_per_cell, amphipod_cell_index)| {
                NeighborMap::<Self>::distance(amphipod_cell_index, amphipod_index + hallway_len_u8)
                    .unwrap() as u32
                    * energy_per_cell
            })
            .sum()
    }

    const fn invalid_amphipod_arrays() -> AmphipodArrays<SIDE_ROOM_LEN> {
        [[u8::MAX; SIDE_ROOM_LEN]; Cell::AMPHIPOD_TYPES]
    }

    const fn organized() -> Self {
        let mut organized: Self = Self::INVALID;
        let mut cell_index: u8 = Self::PARAMS.hallway_len_u8();
        let mut side_room_row_index: usize = 0_usize;

        while side_room_row_index < Self::PARAMS.side_room_len {
            let mut amphipod_index: usize = 0_usize;

            while amphipod_index < Self::PARAMS.amphipod_types {
                organized.0[amphipod_index][side_room_row_index] = cell_index;

                cell_index += 1_u8;
                amphipod_index += 1_usize;
            }

            side_room_row_index += 1_usize;
        }

        organized
    }

    #[inline(always)]
    fn is_side_room_cell_index_locked(
        side_room_cell_index: u8,
        amphipod_index: u8,
        locked_amphipods: u8,
    ) -> bool {
        let amphipod_types_u8: u8 = Self::PARAMS.amphipod_types_u8();

        (side_room_cell_index % amphipod_types_u8 == amphipod_index)
            && (Self::PARAMS.side_room_len_u8() - (side_room_cell_index / amphipod_types_u8)
                <= locked_amphipods)
    }

    #[inline(always)]
    fn is_cell_index_locked(cell_index: u8, amphipod_index: u8, locked_amphipods: u8) -> bool {
        cell_index
            .checked_sub(Self::PARAMS.hallway_len_u8())
            .map_or(false, |side_room_cell_index| {
                Self::is_side_room_cell_index_locked(
                    side_room_cell_index,
                    amphipod_index,
                    locked_amphipods,
                )
            })
    }

    #[inline(always)]
    fn is_cell_index_valid(
        from_cell_index: u8,
        to_cell_index: u8,
        amphipod_index: u8,
        locked_amphipods: u8,
    ) -> IsCellIndexValidResult {
        let hallway_len_u8: u8 = Self::PARAMS.hallway_len_u8();

        match (
            from_cell_index >= hallway_len_u8,
            to_cell_index.checked_sub(hallway_len_u8),
        ) {
            // Bad case that shouldn't be iterated over as a neighbor
            (false, None) => IsCellIndexValidResult::Invalid,
            // The amphipod is either in the wrong side room or there are amphipods of other types
            // behind it. Any hallway location is valid.
            (true, None) => IsCellIndexValidResult::ValidUnlocked,
            // An amphipod can only move into a side room if it'll be locked
            (_, Some(to_side_room_cell_index)) => {
                if Self::is_side_room_cell_index_locked(
                    to_side_room_cell_index,
                    amphipod_index,
                    locked_amphipods,
                ) {
                    IsCellIndexValidResult::ValidLocked
                } else {
                    IsCellIndexValidResult::Invalid
                }
            }
        }
    }
}

type SmallPositionState = PositionState<2_usize>;
type LargePositionState = PositionState<4_usize>;

impl SmallPositionState {
    const BURROW_STR: &str = concat!(
        "#############",
        "#...........#",
        "###.#.#.#.###",
        "  #.#.#.#.#  ",
        "  #########  ",
    );

    const fn burrow_str_index(pos: IVec2) -> usize {
        (pos.y * Self::PARAMS.burrow_dimensions().x + pos.x) as usize
    }

    fn burrow_str_slice(range: Range<IVec2>) -> &'static str {
        &Self::BURROW_STR[Self::burrow_str_index(range.start)..Self::burrow_str_index(range.end)]
    }
}

impl<const SIDE_ROOM_LEN: usize> Default for PositionState<SIDE_ROOM_LEN>
where
    Self: ParamsTrait,
{
    fn default() -> Self {
        Self::INVALID
    }
}

impl ParamsTrait for SmallPositionState {
    type CellIndexBitArray = BitArr!(for Self::PARAMS.cells(), in u32);
    const PARAMS: Params = Params::new(2_usize);
}

impl ParamsTrait for LargePositionState {
    type CellIndexBitArray = BitArr!(for Self::PARAMS.cells(), in u32);
    const PARAMS: Params = Params::new(4_usize);
}

impl From<SmallPositionState> for LargePositionState {
    fn from(value: SmallPositionState) -> Self {
        const INSERTION: AmphipodArrays<2_usize> = [
            [18_u8, 21_u8],
            [17_u8, 20_u8],
            [16_u8, 22_u8],
            [15_u8, 19_u8],
        ];
        const BOTTOM_ROW_CELL_INDEX: u8 = LargePositionState::PARAMS.hallway_len_u8()
            + LargePositionState::PARAMS.amphipod_types_u8();
        const BOTTOM_ROW_OFFSET: u8 = 2_u8 * LargePositionState::PARAMS.amphipod_types_u8();

        let mut large_position_state: LargePositionState = LargePositionState::INVALID;

        for (
            (small_amphipod_cell_indices, insertion_amphipod_cell_indices),
            large_amphipod_cell_indices,
        ) in value
            .0
            .into_iter()
            .zip(INSERTION)
            .zip(large_position_state.0.iter_mut())
        {
            for (source_amphipod_cell_index, dest_amphipod_cell_index) in
                small_amphipod_cell_indices
                    .into_iter()
                    .map(|ampipod_cell_index| {
                        if ampipod_cell_index >= BOTTOM_ROW_CELL_INDEX {
                            ampipod_cell_index + BOTTOM_ROW_OFFSET
                        } else {
                            ampipod_cell_index
                        }
                    })
                    .chain(insertion_amphipod_cell_indices)
                    .zip(large_amphipod_cell_indices.iter_mut())
            {
                *dest_amphipod_cell_index = source_amphipod_cell_index;
            }
        }

        large_position_state
    }
}

impl<const SIDE_ROOM_LEN: usize> Parse for PositionState<SIDE_ROOM_LEN>
where
    Self: ParamsTrait,
{
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let burrow_dimensions: IVec2 = SmallPositionState::PARAMS.burrow_dimensions();

        map_opt(
            tuple((
                delimited(
                    tuple((
                        tag(SmallPositionState::burrow_str_slice(
                            IVec2::ZERO..IVec2::new(burrow_dimensions.x, 0_i32),
                        )),
                        line_ending,
                        tag("#"),
                    )),
                    |input| {
                        let mut state: Self = Default::default();

                        (0_u8..Self::PARAMS.hallway_len_u8())
                            .into_iter()
                            .try_fold(input, |input, cell_index| {
                                Ok(map_opt(Cell::parse, |cell| {
                                    if let Some(amphipod_index) = cell.amphipod_index() {
                                        *state.0[amphipod_index]
                                            .iter_mut()
                                            .find(|cell_index| **cell_index == u8::MAX)? =
                                            cell_index;
                                    }

                                    Some(())
                                })(input)?
                                .0)
                            })
                            .map(|input| {
                                state.sort();

                                (input, state)
                            })
                    },
                    tuple((tag("#"), line_ending)),
                ),
                terminated(
                    move |input| {
                        let mut state: Self = Default::default();
                        let mut cell_index: u8 = Self::PARAMS.hallway_len_u8();

                        (2_i32..(2_i32 + Self::PARAMS.side_room_len as i32))
                            .into_iter()
                            .try_fold(input, |input, hallway_row_y| {
                                let hallway_row_y: i32 = hallway_row_y.min(3_i32);

                                Ok(tuple((
                                    tag(SmallPositionState::burrow_str_slice(
                                        IVec2::new(0_i32, hallway_row_y)
                                            ..IVec2::new(3_i32, hallway_row_y),
                                    )),
                                    many_m_n(
                                        Cell::AMPHIPOD_TYPES,
                                        Cell::AMPHIPOD_TYPES,
                                        map_opt(terminated(Cell::parse, tag("#")), |cell| {
                                            if let Some(amphipod_index) = cell.amphipod_index() {
                                                *state.0[amphipod_index]
                                                    .iter_mut()
                                                    .find(|cell_index| **cell_index == u8::MAX)? =
                                                    cell_index;
                                            }

                                            cell_index += 1_u8;

                                            Some(())
                                        }),
                                    ),
                                    tuple((
                                        tag(SmallPositionState::burrow_str_slice(
                                            IVec2::new(burrow_dimensions.x - 2_i32, hallway_row_y)
                                                ..IVec2::new(burrow_dimensions.x, hallway_row_y),
                                        )
                                        .trim()),
                                        line_ending,
                                    )),
                                ))(input)?
                                .0)
                            })
                            .map(|input| {
                                state.sort();

                                (input, state)
                            })
                    },
                    tuple((
                        tag(SmallPositionState::burrow_str_slice(
                            IVec2::new(0_i32, burrow_dimensions.y - 1_i32)
                                ..(burrow_dimensions - IVec2::Y),
                        )
                        .trim_end()),
                        line_ending,
                    )),
                ),
            )),
            |(hallway_state, side_room_state)| {
                let mut state: Self = hallway_state;

                for (source_cell_indices, dest_cell_indices) in
                    side_room_state.0.into_iter().zip(state.0.iter_mut())
                {
                    fn count(cell_indices: &[u8], valid: bool) -> usize {
                        cell_indices
                            .iter()
                            .filter(|cell_index| (**cell_index != u8::MAX) == valid)
                            .count()
                    }

                    if count(&source_cell_indices, true) != count(dest_cell_indices, false) {
                        return None;
                    }

                    for (source_cell_index, dest_cell_index) in source_cell_indices
                        .into_iter()
                        .filter(|cell_index| *cell_index != u8::MAX)
                        .zip(
                            dest_cell_indices
                                .iter_mut()
                                .filter(|cell_index| **cell_index == u8::MAX),
                        )
                    {
                        *dest_cell_index = source_cell_index;
                    }
                }

                Some(state)
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct SideRoomState<const SIDE_ROOM_LEN: usize>(AmphipodArrays<SIDE_ROOM_LEN>);

impl<const SIDE_ROOM_LEN: usize> SideRoomState<SIDE_ROOM_LEN>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    const INVALID: SideRoomState<SIDE_ROOM_LEN> =
        Self(PositionState::<SIDE_ROOM_LEN>::invalid_amphipod_arrays());

    #[inline]
    fn new(position_state: PositionState<SIDE_ROOM_LEN>) -> Self {
        let hallway_len_u8: u8 = PositionState::<SIDE_ROOM_LEN>::PARAMS.hallway_len_u8();
        let amphipod_types: usize = PositionState::<SIDE_ROOM_LEN>::PARAMS.amphipod_types;
        let mut side_room_state: Self = SideRoomState::INVALID;

        for (amphipod_index, amphipod_cell_indices) in position_state.0.into_iter().enumerate() {
            for amphipod_cell_index in amphipod_cell_indices {
                if let Some(side_room_cell_index) = amphipod_cell_index
                    .checked_sub(hallway_len_u8)
                    .map(usize::from)
                {
                    side_room_state.0[side_room_cell_index % amphipod_types]
                        [side_room_cell_index / amphipod_types] = amphipod_index as u8;
                }
            }
        }

        side_room_state
    }

    #[inline]
    fn locked_amphipods(self) -> [u8; Cell::AMPHIPOD_TYPES] {
        let side_room_len: usize = PositionState::<SIDE_ROOM_LEN>::PARAMS.side_room_len;
        let mut locked_amphipods: [u8; Cell::AMPHIPOD_TYPES] = [0_u8; Cell::AMPHIPOD_TYPES];

        for (organized_amphipod_index, (current_amphipod_indices, locked_amphipods)) in self
            .0
            .into_iter()
            .zip(locked_amphipods.iter_mut())
            .enumerate()
        {
            let organized_amphipod_index: u8 = organized_amphipod_index as u8;

            for current_amphipod_indices_index in (0_usize..side_room_len).rev() {
                if current_amphipod_indices[current_amphipod_indices_index]
                    == organized_amphipod_index
                {
                    *locked_amphipods += 1_u8;
                } else {
                    break;
                }
            }
        }

        locked_amphipods
    }
}

struct PreviousEntry<const SIDE_ROOM_LEN: usize> {
    previous: PositionState<SIDE_ROOM_LEN>,
    energy: u32,
}

#[derive(Default)]
struct OrganizeAmphipodsResult<const SIDE_ROOM_LEN: usize> {
    previous_map: HashMap<PositionState<SIDE_ROOM_LEN>, PreviousEntry<SIDE_ROOM_LEN>>,
}

impl<const SIDE_ROOM_LEN: usize> OrganizeAmphipodsResult<SIDE_ROOM_LEN>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    fn energy_to_organize(&self) -> Option<u32> {
        self.previous_map
            .get(&PositionState::<SIDE_ROOM_LEN>::ORGANIZED)
            .map(|previous_entry| previous_entry.energy)
    }

    fn path(&self) -> Vec<PositionState<SIDE_ROOM_LEN>> {
        let mut path: VecDeque<PositionState<SIDE_ROOM_LEN>> = VecDeque::new();
        let mut vertex: PositionState<SIDE_ROOM_LEN> = PositionState::ORGANIZED;

        while let Some(previous_entry) = self.previous_map.get(&vertex) {
            path.push_front(vertex);
            vertex = previous_entry.previous;
        }

        path.into()
    }
}

struct OrganizeAmphipods<'r, const SIDE_ROOM_LEN: usize>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    start: PositionState<SIDE_ROOM_LEN>,
    result: &'r mut OrganizeAmphipodsResult<SIDE_ROOM_LEN>,
}

impl<'r, const SIDE_ROOM_LEN: usize> OrganizeAmphipods<'r, SIDE_ROOM_LEN>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    fn start(&self) -> &PositionState<SIDE_ROOM_LEN> {
        &self.start
    }

    fn is_end(&self, vertex: &PositionState<SIDE_ROOM_LEN>) -> bool {
        *vertex == PositionState::ORGANIZED
    }

    fn path_to(&self, vertex: &PositionState<SIDE_ROOM_LEN>) -> Vec<PositionState<SIDE_ROOM_LEN>> {
        assert!(self.is_end(vertex));

        self.result.path()
    }

    fn cost_from_start(&self, vertex: &PositionState<SIDE_ROOM_LEN>) -> u32 {
        self.result
            .previous_map
            .get(vertex)
            .map_or(u32::MAX, |previous_entry| previous_entry.energy)
    }

    fn heuristic(&self, vertex: &PositionState<SIDE_ROOM_LEN>) -> u32 {
        vertex.heuristic()
    }

    fn neighbors(
        &self,
        vertex: &PositionState<SIDE_ROOM_LEN>,
        neighbors: &mut Vec<OpenSetElement<PositionState<SIDE_ROOM_LEN>, u32>>,
    ) {
        neighbors.clear();

        let present_cell_indices: <PositionState<SIDE_ROOM_LEN> as ParamsTrait>::CellIndexBitArray =
            vertex.present_cell_indices();
        let side_room_state: SideRoomState<SIDE_ROOM_LEN> = SideRoomState::new(*vertex);
        let locked_amphipods: [u8; Cell::AMPHIPOD_TYPES] = side_room_state.locked_amphipods();

        for (amphipod_index, amphipod_cell_indices) in vertex.0.into_iter().enumerate().rev() {
            let locked_amphipods: u8 = locked_amphipods[amphipod_index];
            let energy_per_cell: u32 = Cell::energy_per_cell_for_amphipod_index(amphipod_index);
            let amphipod_index_u8: u8 = amphipod_index as u8;

            for (side_room_cell_index, amphipod_cell_index) in
                amphipod_cell_indices.into_iter().enumerate()
            {
                if !PositionState::is_cell_index_locked(
                    amphipod_cell_index,
                    amphipod_index_u8,
                    locked_amphipods,
                ) {
                    for neighbor in
                        NeighborMap::<PositionState<SIDE_ROOM_LEN>>::neighbors(amphipod_cell_index)
                    {
                        if !(neighbor.vacant_cell_indices & present_cell_indices)
                            .deref()
                            .any()
                        {
                            let is_cell_index_valid_result: IsCellIndexValidResult =
                                PositionState::is_cell_index_valid(
                                    amphipod_cell_index,
                                    neighbor.cell_index,
                                    amphipod_index_u8,
                                    locked_amphipods + 1_u8,
                                );

                            if is_cell_index_valid_result != IsCellIndexValidResult::Invalid {
                                let mut neighbor_state: PositionState<SIDE_ROOM_LEN> = *vertex;

                                neighbor_state.0[amphipod_index][side_room_cell_index] =
                                    neighbor.cell_index;
                                neighbor_state.sort();

                                let is_best_neighbor: bool = is_cell_index_valid_result
                                    == IsCellIndexValidResult::ValidLocked;
                                let energy_cost: u32 = neighbor.distance as u32 * energy_per_cell;
                                let open_set_element: OpenSetElement<
                                    PositionState<SIDE_ROOM_LEN>,
                                    u32,
                                > = OpenSetElement(neighbor_state, energy_cost);

                                if is_best_neighbor {
                                    neighbors.clear();
                                    neighbors.push(open_set_element);

                                    return;
                                }

                                neighbors.push(open_set_element);
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_vertex(
        &mut self,
        from: &PositionState<SIDE_ROOM_LEN>,
        to: &PositionState<SIDE_ROOM_LEN>,
        cost: u32,
    ) {
        self.result.previous_map.insert(
            *to,
            PreviousEntry {
                previous: *from,
                energy: cost,
            },
        );
    }

    fn reset(&mut self) {
        self.result.previous_map.clear();
        self.result
            .previous_map
            .insert(self.start, self.start.previous_entry());
    }

    #[allow(dead_code)]
    fn depth_first_search(&mut self) -> Option<(Vec<PositionState<SIDE_ROOM_LEN>>, u32)> {
        self.reset();

        struct NeighborData<const SIDE_ROOM_LEN: usize> {
            neighbor: PositionState<SIDE_ROOM_LEN>,
            start_to_neighbor_cost: u32,
            vertex_to_organized_heuristic: u32,
        }

        let mut neighbors: Vec<OpenSetElement<PositionState<SIDE_ROOM_LEN>, u32>> = Vec::new();
        let mut neighbor_datas: Vec<NeighborData<SIDE_ROOM_LEN>> = Vec::new();
        let mut frontier: Vec<PositionState<SIDE_ROOM_LEN>> = vec![self.start];

        while let Some(vertex) = frontier.pop() {
            if vertex != PositionState::<SIDE_ROOM_LEN>::ORGANIZED {
                let start_to_vertex_cost: u32 =
                    self.result.previous_map.get(&vertex).unwrap().energy;

                self.neighbors(&vertex, &mut neighbors);
                neighbor_datas.extend(
                    neighbors
                        .drain(..)
                        .map(
                            |OpenSetElement(neighbor, vertex_to_neighbor_cost)| NeighborData {
                                neighbor,
                                start_to_neighbor_cost: start_to_vertex_cost
                                    + vertex_to_neighbor_cost,
                                vertex_to_organized_heuristic: vertex_to_neighbor_cost
                                    + neighbor.heuristic(),
                            },
                        )
                        .filter(|neighbor_data| {
                            let previous_map: &mut HashMap<
                                PositionState<SIDE_ROOM_LEN>,
                                PreviousEntry<SIDE_ROOM_LEN>,
                            > = &mut self.result.previous_map;

                            if let Some(previous_entry) =
                                previous_map.get_mut(&neighbor_data.neighbor)
                            {
                                if previous_entry.energy > neighbor_data.start_to_neighbor_cost {
                                    previous_entry.previous = vertex;
                                    previous_entry.energy = neighbor_data.start_to_neighbor_cost;

                                    true
                                } else {
                                    false
                                }
                            } else {
                                previous_map.insert(
                                    neighbor_data.neighbor,
                                    PreviousEntry {
                                        previous: vertex,
                                        energy: neighbor_data.start_to_neighbor_cost,
                                    },
                                );

                                true
                            }
                        }),
                );
                neighbor_datas
                    .sort_by_key(|neighbor_data| neighbor_data.vertex_to_organized_heuristic);
                frontier.extend(
                    neighbor_datas
                        .drain(..)
                        .map(|neighbor_data| neighbor_data.neighbor),
                );
            }
        }

        self.result
            .energy_to_organize()
            .map(|energy| (self.result.path(), energy))
    }
}

impl<'r, const SIDE_ROOM_LEN: usize> AStar for OrganizeAmphipods<'r, SIDE_ROOM_LEN>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    type Vertex = PositionState<SIDE_ROOM_LEN>;
    type Cost = u32;

    fn start(&self) -> &Self::Vertex {
        self.start()
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        self.is_end(vertex)
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        self.path_to(vertex)
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.cost_from_start(vertex)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.heuristic(vertex)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        self.neighbors(vertex, neighbors)
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.update_vertex(from, to, cost);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl<'r, const SIDE_ROOM_LEN: usize> Dijkstra for OrganizeAmphipods<'r, SIDE_ROOM_LEN>
where
    PositionState<SIDE_ROOM_LEN>: ParamsTrait,
{
    type Vertex = PositionState<SIDE_ROOM_LEN>;

    type Cost = u32;

    fn start(&self) -> &Self::Vertex {
        self.start()
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        self.is_end(vertex)
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        self.path_to(vertex)
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.cost_from_start(vertex)
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        self.neighbors(vertex, neighbors)
    }

    fn update_vertex(&mut self, from: &Self::Vertex, to: &Self::Vertex, cost: Self::Cost) {
        self.update_vertex(from, to, cost);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Void = VOID = b' ',
        Wall = WALL = b'#',
        Vacant = VACANT = b'.',
        Amber = AMBER = b'A',
        Bronze = BRONZE = b'B',
        Copper = COPPER = b'C',
        Desert = DESERT = b'D',
    }
}

impl Cell {
    const AMPHIPOD_TYPES: usize = (Cell::DESERT + 1_u8 - Cell::AMBER) as usize;

    fn amphipod_index(self) -> Option<usize> {
        match self {
            Self::Void | Self::Wall | Self::Vacant => None,
            _ => Some((self as u8 - Self::AMBER) as usize),
        }
    }

    #[inline(always)]
    const fn energy_per_cell_for_amphipod_index(amphipod_index: usize) -> u32 {
        10_u32.pow(amphipod_index as u32)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(SmallPositionState);

impl Solution {
    fn try_organize_amphipods(&self) -> Option<(Vec<SmallPositionState>, u32)> {
        self.0.try_organize_amphipods()
    }

    fn try_compute_organize_amphipods_energy(&self) -> Option<u32> {
        self.try_organize_amphipods().map(|(_, energy)| energy)
    }

    fn try_large_organize_amphipods(&self) -> Option<(Vec<LargePositionState>, u32)> {
        LargePositionState::from(self.0).try_organize_amphipods()
    }

    fn try_large_compute_organize_amphipods_energy(&self) -> Option<u32> {
        self.try_large_organize_amphipods()
            .map(|(_, energy)| energy)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(PositionState::parse, |state| {
            if state
                .0
                .into_iter()
                .flatten()
                .any(|cell_index| cell_index >= SmallPositionState::PARAMS.hallway_len_u8())
            {
                Some(Self(state))
            } else {
                None
            }
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((path, energy)) = self.try_organize_amphipods() {
                dbg!(energy);

                for (index, state) in path.into_iter().enumerate() {
                    println!("State {index}:\n{}", state.as_string_safe());
                }
            } else {
                println!("self.try_organize_amphipods().is_none()");
            }
        } else {
            dbg!(self.try_compute_organize_amphipods_energy());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((path, energy)) = self.try_large_organize_amphipods() {
                dbg!(energy);

                for (index, state) in path.into_iter().enumerate() {
                    println!("State {index}:\n{}", state.as_string_safe());
                }
            } else {
                println!("self.try_large_organize_amphipods().is_none()");
            }
        } else {
            dbg!(self.try_large_compute_organize_amphipods_energy());
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_POSITION_STATE_STRS: &[&str] = &[
        concat!(
            "#############\n",
            "#...........#\n",
            "###B#C#B#D###\n",
            "  #A#D#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#...B.......#\n",
            "###B#C#.#D###\n",
            "  #A#D#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#...B.......#\n",
            "###B#.#C#D###\n",
            "  #A#D#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#...B.D.....#\n",
            "###B#.#C#D###\n",
            "  #A#.#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.....D.....#\n",
            "###B#.#C#D###\n",
            "  #A#B#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.....D.....#\n",
            "###.#B#C#D###\n",
            "  #A#B#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.....D.D...#\n",
            "###.#B#C#.###\n",
            "  #A#B#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.....D.D.A.#\n",
            "###.#B#C#.###\n",
            "  #A#B#C#.#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.....D...A.#\n",
            "###.#B#C#.###\n",
            "  #A#B#C#D#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#.........A.#\n",
            "###.#B#C#D###\n",
            "  #A#B#C#D#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#...........#\n",
            "###A#B#C#D###\n",
            "  #A#B#C#D#\n",
            "  #########\n",
        ),
    ];

    const LARGE_POSITION_STATE_STRS: &[&str] = &[
        concat!(
            "#############\n",
            "#A......B.BB#\n",
            "###.#C#.#D###\n",
            "  #D#C#.#A#\n",
            "  #D#B#.#C#\n",
            "  #A#D#C#A#\n",
            "  #########\n",
        ),
        concat!(
            "#############\n",
            "#A......B.BB#\n",
            "###.#.#.#D###\n",
            "  #D#C#.#A#\n",
            "  #D#B#C#C#\n",
            "  #A#D#C#A#\n",
            "  #########\n",
        ),
    ];
    const SOLUTION: Solution = Solution(PositionState([
        [15_u8, 18_u8],
        [11_u8, 13_u8],
        [12_u8, 17_u8],
        [14_u8, 16_u8],
    ]));

    fn small_position_state(index: usize) -> SmallPositionState {
        static ONCE_LOCK: OnceLock<Vec<SmallPositionState>> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            SMALL_POSITION_STATE_STRS
                .iter()
                .copied()
                .map(|state_str| PositionState::parse(state_str).unwrap().1)
                .collect()
        })[index]
    }

    fn large_position_state(index: usize) -> LargePositionState {
        static ONCE_LOCK: OnceLock<Vec<LargePositionState>> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            LARGE_POSITION_STATE_STRS
                .iter()
                .copied()
                .map(|state_str| PositionState::parse(state_str).unwrap().1)
                .collect()
        })[index]
    }

    #[test]
    fn test_position_state_parse() {
        small_position_state(0_usize);
    }

    #[test]
    fn test_solution_try_from_str() {
        assert_eq!(
            Solution::try_from(SMALL_POSITION_STATE_STRS[0_usize]),
            Ok(SOLUTION)
        );
    }

    #[test]
    fn test_solution_try_as_string() {
        assert_eq!(
            small_position_state(0_usize).as_string_safe(),
            concat!(
                "#############\n",
                "#...........#\n",
                "###B#C#B#D###\n",
                "  #A#D#C#A#  \n",
                "  #########  \n",
            )
            .to_owned()
        )
    }

    #[test]
    fn test_side_room_state_new() {
        assert_eq!(
            SideRoomState::new(small_position_state(0_usize)),
            SideRoomState([[1_u8, 0_u8], [2_u8, 3_u8], [1_u8, 2_u8], [3_u8, 0_u8]])
        );
    }

    #[test]
    fn test_side_room_state_locked_amphipods() {
        assert_eq!(
            SideRoomState::new(small_position_state(0_usize)).locked_amphipods(),
            [1_u8, 0_u8, 1_u8, 0_u8]
        );
    }

    #[test]
    fn test_organize_amphipods_neighbors_short_circuit() {
        fn test_organize_amphipods_neighbors_short_circuit<
            const SIDE_ROOM_LEN: usize,
            F: Fn(usize) -> PositionState<SIDE_ROOM_LEN>,
        >(
            from_indices: &[usize],
            f: F,
        ) where
            PositionState<SIDE_ROOM_LEN>: ParamsTrait,
        {
            let mut neighbors: Vec<OpenSetElement<PositionState<SIDE_ROOM_LEN>, u32>> = Vec::new();

            for from_index in from_indices.iter().copied() {
                let from_state: PositionState<SIDE_ROOM_LEN> = f(from_index);
                let mut result: OrganizeAmphipodsResult<SIDE_ROOM_LEN> = Default::default();
                let organize_amphipods: OrganizeAmphipods<SIDE_ROOM_LEN> = OrganizeAmphipods {
                    start: from_state,
                    result: &mut result,
                };

                organize_amphipods.neighbors(&from_state, &mut neighbors);

                let to_state: PositionState<SIDE_ROOM_LEN> = f(from_index + 1_usize);
                let neighbors_first: Option<&PositionState<SIDE_ROOM_LEN>> = neighbors
                    .first()
                    .map(|OpenSetElement(neighbor, _)| neighbor);

                assert!(
                    (neighbors.len() == 1_usize) && (neighbors_first == Some(&to_state)),
                    "neighbors.len() == {}\n\
                    from_state:\n{}\
                    neighbors.first():\n{}\
                    to_state:\n{}",
                    neighbors.len(),
                    from_state.as_string_safe(),
                    neighbors_first.map_or_else(|| "empty".into(), PositionState::as_string_safe),
                    to_state.as_string_safe()
                );
            }
        }

        test_organize_amphipods_neighbors_short_circuit(
            &[1_usize, 3_usize, 4_usize, 7_usize, 8_usize, 9_usize],
            small_position_state,
        );
        test_organize_amphipods_neighbors_short_circuit(&[0_usize], large_position_state);
    }

    #[test]
    fn test_solution_try_compute_organize_amphipods_energy() {
        assert_eq!(
            SOLUTION.try_compute_organize_amphipods_energy(),
            Some(12521_u32)
        );
    }

    #[test]
    fn test_solution_try_large_compute_organize_amphipods_energy() {
        assert_eq!(
            SOLUTION.try_large_compute_organize_amphipods_energy(),
            Some(44169_u32)
        );
    }
}
