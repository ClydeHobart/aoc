use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::{
        fmt::{Display, Write},
        mem::{take, transmute},
        str::from_utf8_unchecked,
    },
    strum::IntoEnumIterator,
};

/* --- Day 15: Beverage Bandits ---

Having perfected their hot chocolate, the Elves have a new problem: the Goblins that live in these caves will do anything to steal it. Looks like they're here for a fight.

You scan the area, generating a map of the walls (#), open cavern (.), and starting position of every Goblin (G) and Elf (E) (your puzzle input).

Combat proceeds in rounds; in each round, each unit that is still alive takes a turn, resolving all of its actions before the next unit's turn begins. On each unit's turn, it tries to move into range of an enemy (if it isn't already) and then attack (if it is in range).

All units are very disciplined and always follow very strict combat rules. Units never move or attack diagonally, as doing so would be dishonorable. When multiple choices are equally valid, ties are broken in reading order: top-to-bottom, then left-to-right. For instance, the order in which units take their turns within a round is the reading order of their starting positions in that round, regardless of the type of unit or whether other units have moved after the round started. For example:

                 would take their
These units:   turns in this order:
  #######           #######
  #.G.E.#           #.1.2.#
  #E.G.E#           #3.4.5#
  #.G.E.#           #.6.7.#
  #######           #######

Each unit begins its turn by identifying all possible targets (enemy units). If no targets remain, combat ends.

Then, the unit identifies all of the open squares (.) that are in range of each target; these are the squares which are adjacent (immediately up, down, left, or right) to any target and which aren't already occupied by a wall or another unit. Alternatively, the unit might already be in range of a target. If the unit is not already in range of a target, and there are no open squares which are in range of a target, the unit ends its turn.

If the unit is already in range of a target, it does not move, but continues its turn with an attack. Otherwise, since it is not in range of a target, it moves.

To move, the unit first considers the squares that are in range and determines which of those squares it could reach in the fewest steps. A step is a single movement to any adjacent (immediately up, down, left, or right) open (.) square. Units cannot move into walls or other units. The unit does this while considering the current positions of units and does not do any prediction about where units will be later. If the unit cannot reach (find an open path to) any of the squares that are in range, it ends its turn. If multiple squares are in range and tied for being reachable in the fewest steps, the square which is first in reading order is chosen. For example:

Targets:      In range:     Reachable:    Nearest:      Chosen:
#######       #######       #######       #######       #######
#E..G.#       #E.?G?#       #E.@G.#       #E.!G.#       #E.+G.#
#...#.#  -->  #.?.#?#  -->  #.@.#.#  -->  #.!.#.#  -->  #...#.#
#.G.#G#       #?G?#G#       #@G@#G#       #!G.#G#       #.G.#G#
#######       #######       #######       #######       #######

In the above scenario, the Elf has three targets (the three Goblins):

    Each of the Goblins has open, adjacent squares which are in range (marked with a ? on the map).
    Of those squares, four are reachable (marked @); the other two (on the right) would require moving through a wall or unit to reach.
    Three of these reachable squares are nearest, requiring the fewest steps (only 2) to reach (marked !).
    Of those, the square which is first in reading order is chosen (+).

The unit then takes a single step toward the chosen square along the shortest path to that square. If multiple steps would put the unit equally closer to its destination, the unit chooses the step which is first in reading order. (This requires knowing when there is more than one shortest path so that you can consider the first step of each such path.) For example:

In range:     Nearest:      Chosen:       Distance:     Step:
#######       #######       #######       #######       #######
#.E...#       #.E...#       #.E...#       #4E212#       #..E..#
#...?.#  -->  #...!.#  -->  #...+.#  -->  #32101#  -->  #.....#
#..?G?#       #..!G.#       #...G.#       #432G2#       #...G.#
#######       #######       #######       #######       #######

The Elf sees three squares in range of a target (?), two of which are nearest (!), and so the first in reading order is chosen (+). Under "Distance", each open square is marked with its distance from the destination square; the two squares to which the Elf could move on this turn (down and to the right) are both equally good moves and would leave the Elf 2 steps from being in range of the Goblin. Because the step which is first in reading order is chosen, the Elf moves right one square.

Here's a larger example of movement:

Initially:
#########
#G..G..G#
#.......#
#.......#
#G..E..G#
#.......#
#.......#
#G..G..G#
#########

After 1 round:
#########
#.G...G.#
#...G...#
#...E..G#
#.G.....#
#.......#
#G..G..G#
#.......#
#########

After 2 rounds:
#########
#..G.G..#
#...G...#
#.G.E.G.#
#.......#
#G..G..G#
#.......#
#.......#
#########

After 3 rounds:
#########
#.......#
#..GGG..#
#..GEG..#
#G..G...#
#......G#
#.......#
#.......#
#########

Once the Goblins and Elf reach the positions above, they all are either in range of a target or cannot find any square in range of a target, and so none of the units can move until a unit dies.

After moving (or if the unit began its turn in range of a target), the unit attacks.

To attack, the unit first determines all of the targets that are in range of it by being immediately adjacent to it. If there are no such targets, the unit ends its turn. Otherwise, the adjacent target with the fewest hit points is selected; in a tie, the adjacent target with the fewest hit points which is first in reading order is selected.

The unit deals damage equal to its attack power to the selected target, reducing its hit points by that amount. If this reduces its hit points to 0 or fewer, the selected target dies: its square becomes . and it takes no further turns.

Each unit, either Goblin or Elf, has 3 attack power and starts with 200 hit points.

For example, suppose the only Elf is about to attack:

       HP:            HP:
G....  9       G....  9
..G..  4       ..G..  4
..EG.  2  -->  ..E..
..G..  2       ..G..  2
...G.  1       ...G.  1

The "HP" column shows the hit points of the Goblin to the left in the corresponding row. The Elf is in range of three targets: the Goblin above it (with 4 hit points), the Goblin to its right (with 2 hit points), and the Goblin below it (also with 2 hit points). Because three targets are in range, the ones with the lowest hit points are selected: the two Goblins with 2 hit points each (one to the right of the Elf and one below the Elf). Of those, the Goblin first in reading order (the one to the right of the Elf) is selected. The selected Goblin's hit points (2) are reduced by the Elf's attack power (3), reducing its hit points to -1, killing it.

After attacking, the unit's turn ends. Regardless of how the unit's turn ends, the next unit in the round takes its turn. If all units have taken turns in this round, the round ends, and a new round begins.

The Elves look quite outnumbered. You need to determine the outcome of the battle: the number of full rounds that were completed (not counting the round in which combat ends) multiplied by the sum of the hit points of all remaining units at the moment combat ends. (Combat only ends when a unit finds no targets during its turn.)

Below is an entire sample combat. Next to each map, each row's units' hit points are listed from left to right.

Initially:
#######
#.G...#   G(200)
#...EG#   E(200), G(200)
#.#.#G#   G(200)
#..G#E#   G(200), E(200)
#.....#
#######

After 1 round:
#######
#..G..#   G(200)
#...EG#   E(197), G(197)
#.#G#G#   G(200), G(197)
#...#E#   E(197)
#.....#
#######

After 2 rounds:
#######
#...G.#   G(200)
#..GEG#   G(200), E(188), G(194)
#.#.#G#   G(194)
#...#E#   E(194)
#.....#
#######

Combat ensues; eventually, the top Elf dies:

After 23 rounds:
#######
#...G.#   G(200)
#..G.G#   G(200), G(131)
#.#.#G#   G(131)
#...#E#   E(131)
#.....#
#######

After 24 rounds:
#######
#..G..#   G(200)
#...G.#   G(131)
#.#G#G#   G(200), G(128)
#...#E#   E(128)
#.....#
#######

After 25 rounds:
#######
#.G...#   G(200)
#..G..#   G(131)
#.#.#G#   G(125)
#..G#E#   G(200), E(125)
#.....#
#######

After 26 rounds:
#######
#G....#   G(200)
#.G...#   G(131)
#.#.#G#   G(122)
#...#E#   E(122)
#..G..#   G(200)
#######

After 27 rounds:
#######
#G....#   G(200)
#.G...#   G(131)
#.#.#G#   G(119)
#...#E#   E(119)
#...G.#   G(200)
#######

After 28 rounds:
#######
#G....#   G(200)
#.G...#   G(131)
#.#.#G#   G(116)
#...#E#   E(113)
#....G#   G(200)
#######

More combat ensues; eventually, the bottom Elf dies:

After 47 rounds:
#######
#G....#   G(200)
#.G...#   G(131)
#.#.#G#   G(59)
#...#.#
#....G#   G(200)
#######

Before the 48th round can finish, the top-left Goblin finds that there are no targets remaining, and so combat ends. So, the number of full rounds that were completed is 47, and the sum of the hit points of all remaining units is 200+131+59+200 = 590. From these, the outcome of the battle is 47 * 590 = 27730.

Here are a few example summarized combats:

#######       #######
#G..#E#       #...#E#   E(200)
#E#E.E#       #E#...#   E(197)
#G.##.#  -->  #.E##.#   E(185)
#...#E#       #E..#E#   E(200), E(200)
#...E.#       #.....#
#######       #######

Combat ends after 37 full rounds
Elves win with 982 total hit points left
Outcome: 37 * 982 = 36334

#######       #######
#E..EG#       #.E.E.#   E(164), E(197)
#.#G.E#       #.#E..#   E(200)
#E.##E#  -->  #E.##.#   E(98)
#G..#.#       #.E.#.#   E(200)
#..E#.#       #...#.#
#######       #######

Combat ends after 46 full rounds
Elves win with 859 total hit points left
Outcome: 46 * 859 = 39514

#######       #######
#E.G#.#       #G.G#.#   G(200), G(98)
#.#G..#       #.#G..#   G(200)
#G.#.G#  -->  #..#..#
#G..#.#       #...#G#   G(95)
#...E.#       #...G.#   G(200)
#######       #######

Combat ends after 35 full rounds
Goblins win with 793 total hit points left
Outcome: 35 * 793 = 27755

#######       #######
#.E...#       #.....#
#.#..G#       #.#G..#   G(200)
#.###.#  -->  #.###.#
#E#G#G#       #.#.#.#
#...#G#       #G.G#G#   G(98), G(38), G(200)
#######       #######

Combat ends after 54 full rounds
Goblins win with 536 total hit points left
Outcome: 54 * 536 = 28944

#########       #########
#G......#       #.G.....#   G(137)
#.E.#...#       #G.G#...#   G(200), G(200)
#..##..G#       #.G##...#   G(200)
#...##..#  -->  #...##..#
#...#...#       #.G.#...#   G(200)
#.G...G.#       #.......#
#.....G.#       #.......#
#########       #########

Combat ends after 20 full rounds
Goblins win with 937 total hit points left
Outcome: 20 * 937 = 18740

What is the outcome of the combat described in your puzzle input?

--- Part Two ---

According to your calculations, the Elves are going to lose badly. Surely, you won't mess up the timeline too much if you give them just a little advanced technology, right?

You need to make sure the Elves not only win, but also suffer no losses: even the death of a single Elf is unacceptable.

However, you can't go too far: larger changes will be more likely to permanently alter spacetime.

So, you need to find the outcome of the battle in which the Elves have the lowest integer attack power (at least 4) that allows them to win without a single death. The Goblins always have an attack power of 3.

In the first summarized example above, the lowest attack power the Elves need to win without losses is 15:

#######       #######
#.G...#       #..E..#   E(158)
#...EG#       #...E.#   E(14)
#.#.#G#  -->  #.#.#.#
#..G#E#       #...#.#
#.....#       #.....#
#######       #######

Combat ends after 29 full rounds
Elves win with 172 total hit points left
Outcome: 29 * 172 = 4988

In the second example above, the Elves need only 4 attack power:

#######       #######
#E..EG#       #.E.E.#   E(200), E(23)
#.#G.E#       #.#E..#   E(200)
#E.##E#  -->  #E.##E#   E(125), E(200)
#G..#.#       #.E.#.#   E(200)
#..E#.#       #...#.#
#######       #######

Combat ends after 33 full rounds
Elves win with 948 total hit points left
Outcome: 33 * 948 = 31284

In the third example above, the Elves need 15 attack power:

#######       #######
#E.G#.#       #.E.#.#   E(8)
#.#G..#       #.#E..#   E(86)
#G.#.G#  -->  #..#..#
#G..#.#       #...#.#
#...E.#       #.....#
#######       #######

Combat ends after 37 full rounds
Elves win with 94 total hit points left
Outcome: 37 * 94 = 3478

In the fourth example above, the Elves need 12 attack power:

#######       #######
#.E...#       #...E.#   E(14)
#.#..G#       #.#..E#   E(152)
#.###.#  -->  #.###.#
#E#G#G#       #.#.#.#
#...#G#       #...#.#
#######       #######

Combat ends after 39 full rounds
Elves win with 166 total hit points left
Outcome: 39 * 166 = 6474

In the last example above, the lone Elf needs 34 attack power:

#########       #########
#G......#       #.......#
#.E.#...#       #.E.#...#   E(38)
#..##..G#       #..##...#
#...##..#  -->  #...##..#
#...#...#       #...#...#
#.G...G.#       #.......#
#.....G.#       #.......#
#########       #########

Combat ends after 30 full rounds
Elves win with 38 total hit points left
Outcome: 30 * 38 = 1140

After increasing the Elves' attack power until it is just barely enough for them to win without any Elves dying, what is the outcome of the combat described in your puzzle input? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd,)]
    enum Cell {
        #[default]
        Open = OPEN = b'.',
        Wall = WALL = b'#',
        Goblin = GOBLIN = b'G',
        Elf = ELF = b'E',
    }
}

impl Cell {
    fn is_unit(self) -> bool {
        matches!(self, Self::Goblin | Self::Elf)
    }

    fn try_enemy(self) -> Option<Self> {
        match self {
            Self::Goblin => Some(Self::Elf),
            Self::Elf => Some(Self::Goblin),
            _ => None,
        }
    }
}

type UnitIndexRaw = u8;
type UnitIndex = Index<UnitIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct UnitData {
    hit_points: u8,
}

impl UnitData {
    const INITIAL_HIT_POINTS: u8 = 200_u8;

    const fn new() -> Self {
        Self {
            hit_points: Self::INITIAL_HIT_POINTS,
        }
    }

    fn is_dead(self) -> bool {
        self.hit_points == 0_u8
    }

    fn receive_damage(&mut self, attack_power: u8) {
        self.hit_points = self.hit_points.saturating_sub(attack_power);
    }
}

impl Default for UnitData {
    fn default() -> Self {
        Self::new()
    }
}

type Unit = TableElement<SmallPos, UnitData>;
type UnitTable = Table<SmallPos, UnitData, UnitIndexRaw>;

type UnitTypeIndexRaw = u8;
type UnitTypeIndex = Index<UnitTypeIndexRaw>;

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
struct UnitTypeData {
    unit_count: u16,
    attack_power: u8,
}

impl UnitTypeData {
    const ATTACK_POWER: u8 = 3_u8;
}

impl Default for UnitTypeData {
    fn default() -> Self {
        Self {
            unit_count: 0_u16,
            attack_power: Self::ATTACK_POWER,
        }
    }
}

type UnitType = TableElement<Cell, UnitTypeData>;
type UnitTypeTable = Table<Cell, UnitTypeData, UnitTypeIndexRaw>;

struct RunState {
    unit_small_poses: Vec<SmallPos>,
    bfs_state: BreadthFirstSearchState<IVec2>,
    unit_pos: IVec2,
    dist_from_unit: Grid2D<u16>,
    selected_in_range_pos: IVec2,
    is_along_fastest_path: BitVec,
}

impl RunState {
    fn new(solution: &Solution) -> Self {
        RunState {
            unit_small_poses: Vec::new(),
            bfs_state: BreadthFirstSearchState::default(),
            unit_pos: IVec2::ZERO,
            dist_from_unit: Grid2D::try_from_cells_and_dimensions(
                vec![u16::MAX; solution.grid.cells().len()],
                solution.grid.dimensions(),
            )
            .unwrap(),
            selected_in_range_pos: IVec2::ZERO,
            is_along_fastest_path: bitvec![0; solution.grid.cells().len()],
        }
    }
}

struct DistFromUnitPopulator<'s> {
    solution: &'s Solution,
    state: &'s mut RunState,
}

impl<'s> BreadthFirstSearch for DistFromUnitPopulator<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.state.unit_pos
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            self.solution
                .grid
                .get(neighbor)
                .and_then(|&neighbor_cell| (neighbor_cell == Cell::Open).then_some(neighbor))
        }));
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let from_dist: u16 = *self.state.dist_from_unit.get(*from).unwrap();

        *self.state.dist_from_unit.get_mut(*to).unwrap() = from_dist + 1_u16;
    }

    fn reset(&mut self) {
        self.state.dist_from_unit.cells_mut().fill(u16::MAX);
        *self
            .state
            .dist_from_unit
            .get_mut(self.state.unit_pos)
            .unwrap() = 0_u16;
    }
}

struct IsAlongFastestPathDeterminer<'s> {
    solution: &'s Solution,
    state: &'s mut RunState,
}

impl<'s> BreadthFirstSearch for IsAlongFastestPathDeterminer<'s> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.state.selected_in_range_pos
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        let vertex_dist: u16 = *self.state.dist_from_unit.get(*vertex).unwrap();

        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let neighbor: IVec2 = *vertex + dir.vec();

            self.state
                .dist_from_unit
                .get(neighbor)
                .and_then(|&neighbor_dist| {
                    (neighbor_dist != u16::MAX && neighbor_dist + 1_u16 == vertex_dist)
                        .then_some(neighbor)
                })
        }))
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        self.state
            .is_along_fastest_path
            .set(self.solution.grid.index_from_pos(*to), true);
    }

    fn reset(&mut self) {
        self.state.is_along_fastest_path.fill(false);
        self.state.is_along_fastest_path.set(
            self.solution
                .grid
                .index_from_pos(self.state.selected_in_range_pos),
            true,
        );
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    grid: Grid2D<Cell>,
    units: UnitTable,
    unit_types: UnitTypeTable,
    rounds: usize,
}

impl Solution {
    fn append_dyn_display_to_report_string(dyn_display: &dyn Display, report_string: &mut String) {
        write!(report_string, "{dyn_display}").ok();
    }

    fn try_choose_movement_pos(
        &self,
        unit_pos: IVec2,
        run_state: &mut RunState,
    ) -> Option<SmallPos> {
        let enemy_cell: Cell = self.grid.get(unit_pos)?.try_enemy()?;

        run_state.unit_pos = unit_pos;

        let mut bfs_state: BreadthFirstSearchState<IVec2> = take(&mut run_state.bfs_state);

        DistFromUnitPopulator {
            solution: self,
            state: run_state,
        }
        .run_internal(&mut bfs_state);

        let selected_in_range_pos: IVec2 = {
            let unit_move_selection_state: &RunState = run_state;

            self.units
                .as_slice()
                .iter()
                .filter_map(|unit| {
                    let target_pos: IVec2 = unit.id.get();

                    self.grid
                        .get(target_pos)
                        .and_then(|&unit_cell| (unit_cell == enemy_cell).then_some(target_pos))
                })
                .flat_map(|target_pos| {
                    Direction::iter().filter_map(move |dir| {
                        let in_range_pos: IVec2 = target_pos + dir.vec();

                        unit_move_selection_state
                            .dist_from_unit
                            .get(in_range_pos)
                            .and_then(|&dist| {
                                (dist != u16::MAX).then_some((
                                    in_range_pos,
                                    self.grid.index_from_pos(in_range_pos),
                                    dist,
                                ))
                            })
                    })
                })
                .min_by(|(_, index_a, dist_a), (_, index_b, dist_b)| {
                    dist_a.cmp(dist_b).then_with(|| index_a.cmp(index_b))
                })
                .map(|(in_range_pos, _, _)| in_range_pos)?
        };

        run_state.selected_in_range_pos = selected_in_range_pos;

        IsAlongFastestPathDeterminer {
            solution: self,
            state: run_state,
        }
        .run_internal(&mut bfs_state);

        run_state.bfs_state = bfs_state;

        Direction::iter()
            .filter_map(|dir| {
                let movement_pos: IVec2 = unit_pos + dir.vec();

                self.grid
                    .try_index_from_pos(movement_pos)
                    .and_then(|index| {
                        run_state.is_along_fastest_path[index].then_some((movement_pos, index))
                    })
            })
            .min_by_key(|&(_, index)| index)
            .and_then(|(movement_pos, _)| (movement_pos != unit_pos).then_some(movement_pos))
            .and_then(SmallPos::try_from_pos)
    }

    fn is_combat_active(&self) -> bool {
        self.unit_types.as_slice().len() > 1_usize
    }

    fn product_of_rounds_and_hit_points_sum(&self) -> usize {
        self.rounds
            * self
                .units
                .as_slice()
                .iter()
                .map(|unit| unit.data.hit_points as usize)
                .sum::<usize>()
    }

    fn outcome_and_solution(&self) -> (usize, Self) {
        let mut solution: Solution = self.clone();

        solution.run();

        (solution.product_of_rounds_and_hit_points_sum(), solution)
    }

    fn outcome(&self) -> usize {
        self.outcome_and_solution().0
    }

    fn report_string(&self) -> String {
        let mut report_string: String = String::new();
        let width: usize = self.grid.dimensions().x as usize;

        for (row_index, row_cells) in self.grid.cells().chunks_exact(width).enumerate() {
            Self::append_dyn_display_to_report_string(
                unsafe { &from_utf8_unchecked(transmute(row_cells)) } as &dyn Display,
                &mut report_string,
            );

            for (unit_row_index, (unit_char, hit_points)) in row_cells
                .iter()
                .enumerate()
                .filter_map(|(col_index, cell)| {
                    cell.is_unit().then(|| {
                        let unit_pos: IVec2 =
                            self.grid.pos_from_index(row_index * width + col_index);

                        // SAFETY: `unit_pos` is valid
                        let unit_small_pos: SmallPos =
                            unsafe { SmallPos::from_pos_unsafe(unit_pos) };

                        (
                            *cell as u8 as char,
                            self.units.as_slice()
                                [self.units.find_index_binary_search(&unit_small_pos).get()]
                            .data
                            .hit_points,
                        )
                    })
                })
                .enumerate()
            {
                Self::append_dyn_display_to_report_string(
                    &if unit_row_index == 0_usize {
                        "   "
                    } else {
                        ", "
                    } as &dyn Display,
                    &mut report_string,
                );
                Self::append_dyn_display_to_report_string(
                    &unit_char as &dyn Display,
                    &mut report_string,
                );
                Self::append_dyn_display_to_report_string(&"(" as &dyn Display, &mut report_string);
                Self::append_dyn_display_to_report_string(
                    &hit_points as &dyn Display,
                    &mut report_string,
                );
                Self::append_dyn_display_to_report_string(&")" as &dyn Display, &mut report_string);
            }

            Self::append_dyn_display_to_report_string(&"\n" as &dyn Display, &mut report_string);
        }

        report_string
    }

    fn elf_win_outcome_and_solution(&self) -> (usize, Self) {
        let mut unit_types: [UnitType; 2_usize] = [
            TableElement {
                id: Default::default(),
                data: Default::default(),
            },
            TableElement {
                id: Default::default(),
                data: Default::default(),
            },
        ];

        unit_types.clone_from_slice(self.unit_types.as_slice());

        let elf_unit_type_index: usize = self.unit_types.find_index(&Cell::Elf).get();

        let mut solution: Solution = self.clone();
        let mut run_state: RunState = RunState::new(self);

        while {
            let initial_elf_unit_type: UnitType =
                solution.unit_types.as_slice()[elf_unit_type_index].clone();

            while solution.run_round(&mut run_state) {}

            solution.unit_types.as_slice()[0_usize] != initial_elf_unit_type
        } {
            solution.grid.cells_mut().copy_from_slice(self.grid.cells());
            solution.units.clear();

            for unit in self.units.as_slice() {
                solution.units.insert_binary_search(unit.id, unit.data);
            }

            unit_types[elf_unit_type_index].data.attack_power += 1_u8;
            solution.unit_types.clear();

            for unit_type in &unit_types {
                solution.unit_types.insert(unit_type.id, unit_type.data);
            }

            solution.rounds = 0_usize;
        }

        (solution.product_of_rounds_and_hit_points_sum(), solution)
    }

    fn elf_win_outcome(&self) -> usize {
        self.elf_win_outcome_and_solution().0
    }

    fn run_round(&mut self, run_state: &mut RunState) -> bool {
        let mut unit_small_poses: Vec<SmallPos> = take(&mut run_state.unit_small_poses);

        unit_small_poses.clear();
        unit_small_poses.extend(self.units.as_slice().iter().map(|unit| unit.id));

        let is_combat_active: bool = unit_small_poses
            .drain(..)
            .try_fold((), |_, mut unit_small_pos| {
                let mut unit_pos: IVec2 = unit_small_pos.get();
                let unit_cell: Cell = *self.grid.get(unit_pos).unwrap();

                if !unit_cell.is_unit() {
                    Some(())
                } else if !self.is_combat_active() {
                    None
                } else {
                    if let Some(movement_small_pos) =
                        self.try_choose_movement_pos(unit_pos, run_state)
                    {
                        let unit_index: UnitIndex =
                            self.units.find_index_binary_search(&unit_small_pos);

                        assert!(unit_index.is_valid());

                        *self.grid.get_mut(unit_pos).unwrap() = Cell::Open;

                        unit_small_pos = movement_small_pos;
                        unit_pos = unit_small_pos.get();

                        *self.grid.get_mut(unit_pos).unwrap() = unit_cell;
                        self.units.as_slice_mut()[unit_index.get()].id = unit_small_pos;
                        self.units.sort_by_id();
                    }

                    if let Some((target_unit_cell, target_unit_index)) =
                        unit_cell.try_enemy().and_then(|target_unit_cell| {
                            Direction::iter()
                                .filter_map(|dir| {
                                    let target_pos: IVec2 = unit_pos + dir.vec();

                                    self.grid.get(target_pos).and_then(|&cell| {
                                        (cell == target_unit_cell).then(|| {
                                            // SAFETY: `in_range_pos` is valid.
                                            let target_small_pos: SmallPos =
                                                unsafe { SmallPos::from_pos_unsafe(target_pos) };
                                            let target_unit_index: UnitIndex = self
                                                .units
                                                .find_index_binary_search(&target_small_pos);

                                            (
                                                target_unit_index,
                                                self.units.as_slice()[target_unit_index.get()]
                                                    .data
                                                    .hit_points,
                                                self.grid.index_from_pos(target_pos),
                                            )
                                        })
                                    })
                                })
                                .min_by_key(|&(_, hit_points, index)| (hit_points, index))
                                .map(|(target_unit_index, _, _)| {
                                    (target_unit_cell, target_unit_index)
                                })
                        })
                    {
                        let target_unit: &mut Unit =
                            &mut self.units.as_slice_mut()[target_unit_index.get()];

                        let attack_power: u8 = self.unit_types.as_slice()
                            [self.unit_types.find_index(&unit_cell).get()]
                        .data
                        .attack_power;

                        target_unit.data.receive_damage(attack_power);

                        if target_unit.data.is_dead() {
                            *self
                                .grid
                                .get_mut(self.units.remove_by_index(target_unit_index).id.get())
                                .unwrap() = Cell::Open;

                            let target_unit_type_index: UnitTypeIndex =
                                self.unit_types.find_index(&target_unit_cell);
                            let target_unit_type_unit_count: &mut u16 =
                                &mut self.unit_types.as_slice_mut()[target_unit_type_index.get()]
                                    .data
                                    .unit_count;

                            *target_unit_type_unit_count -= 1_u16;

                            if *target_unit_type_unit_count == 0_u16 {
                                self.unit_types.remove_by_index(target_unit_type_index);
                            }
                        }
                    }

                    Some(())
                }
            })
            .is_some();

        run_state.unit_small_poses = unit_small_poses;

        self.rounds += is_combat_active as usize;

        is_combat_active
    }

    fn run(&mut self) {
        let mut run_state: RunState = RunState::new(self);

        while self.run_round(&mut run_state) {}
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |grid: Grid2D<Cell>| {
            (SmallPos::are_dimensions_valid(grid.dimensions())
                && grid.iter_filtered_positions(|cell| cell.is_unit()).count()
                    < UnitIndexRaw::MAX as usize)
                .then(|| {
                    let units: UnitTable = grid
                        .cells()
                        .iter()
                        .enumerate()
                        .filter_map(|(index, cell)| {
                            cell.is_unit().then(|| Unit {
                                id: unsafe {
                                    SmallPos::from_pos_unsafe(grid.pos_from_index(index))
                                },
                                data: UnitData::new(),
                            })
                        })
                        .collect::<Vec<Unit>>()
                        .try_into()
                        .unwrap();

                    let mut unit_types: UnitTypeTable = UnitTypeTable::new();

                    for unit in units
                        .as_slice()
                        .iter()
                        .map(|unit| grid.get(unit.id.get()).unwrap())
                    {
                        let unit_type_index: UnitTypeIndex = unit_types.find_or_add_index(unit);

                        unit_types.as_slice_mut()[unit_type_index.get()]
                            .data
                            .unit_count += 1_u16;
                    }

                    unit_types.sort_by_id();

                    let rounds: usize = 0_usize;

                    Self {
                        grid,
                        units,
                        unit_types,
                        rounds,
                    }
                })
        })(input)
    }
}

impl RunQuestions for Solution {
    /// Are you not entertained?
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let (outcome, solution): (usize, Self) = self.outcome_and_solution();

            dbg!(outcome);

            println!("{}", solution.report_string());
        } else {
            dbg!(self.outcome());
        }
    }

    /// I think there's still room for optimization here. It takes a bit on this one. Binary search,
    /// first going up, then once an upper and lower bound do normal binary search?
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let (outcome, solution): (usize, Self) = self.elf_win_outcome_and_solution();

            dbg!(
                outcome,
                solution.unit_types.as_slice()[0_usize].data.attack_power
            );

            println!("{}", solution.report_string());
        } else {
            dbg!(self.elf_win_outcome());
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
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &[
        "\
        #######\n\
        #.G...#\n\
        #...EG#\n\
        #.#.#G#\n\
        #..G#E#\n\
        #.....#\n\
        #######\n",
        "\
        #######\n\
        #G..#E#\n\
        #E#E.E#\n\
        #G.##.#\n\
        #...#E#\n\
        #...E.#\n\
        #######\n",
        "\
        #######\n\
        #E..EG#\n\
        #.#G.E#\n\
        #E.##E#\n\
        #G..#.#\n\
        #..E#.#\n\
        #######\n",
        "\
        #######\n\
        #E.G#.#\n\
        #.#G..#\n\
        #G.#.G#\n\
        #G..#.#\n\
        #...E.#\n\
        #######\n",
        "\
        #######\n\
        #.E...#\n\
        #.#..G#\n\
        #.###.#\n\
        #E#G#G#\n\
        #...#G#\n\
        #######\n",
        "\
        #########\n\
        #G......#\n\
        #.E.#...#\n\
        #..##..G#\n\
        #...##..#\n\
        #...#...#\n\
        #.G...G.#\n\
        #.....G.#\n\
        #########\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{Elf as E, Goblin as G, Open as O, Wall as W};

            macro_rules! units {
                [ $( ($x:expr, $y:expr), )* ] => { vec![ $(
                    TableElement { id: SmallPos { x: $x, y: $y }, data: UnitData::new() },
                )* ].try_into().unwrap() }
            }

            macro_rules! unit_counts {
                [ $( ( $cell:expr, $count:expr ), )* ] => { vec![ $(
                    TableElement {
                        id: $cell,
                        data: UnitTypeData {
                            unit_count: $count,
                            attack_power: UnitTypeData::ATTACK_POWER
                        },
                    },
                )* ].try_into().unwrap() }
            }

            vec![
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, O, G, O, O, O, W, W, O, O, O, E, G, W, W, O, W,
                            O, W, G, W, W, O, O, G, W, E, W, W, O, O, O, O, O, W, W, W, W, W, W, W,
                            W,
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![(2, 1), (4, 2), (5, 2), (5, 3), (3, 4), (5, 4),],
                    unit_types: unit_counts![(E, 2_u16), (G, 4_u16),],
                    rounds: 0_usize,
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, G, O, O, W, E, W, W, E, W, E, O, E, W, W, G, O,
                            W, W, O, W, W, O, O, O, W, E, W, W, O, O, O, E, O, W, W, W, W, W, W, W,
                            W,
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![
                        (1, 1),
                        (5, 1),
                        (1, 2),
                        (3, 2),
                        (5, 2),
                        (1, 3),
                        (5, 4),
                        (4, 5),
                    ],
                    unit_types: unit_counts![(E, 6_u16), (G, 2_u16),],
                    rounds: 0_usize,
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, E, O, O, E, G, W, W, O, W, G, O, E, W, W, E, O,
                            W, W, E, W, W, G, O, O, W, O, W, W, O, O, E, W, O, W, W, W, W, W, W, W,
                            W,
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![
                        (1, 1),
                        (4, 1),
                        (5, 1),
                        (3, 2),
                        (5, 2),
                        (1, 3),
                        (5, 3),
                        (1, 4),
                        (3, 5),
                    ],
                    unit_types: unit_counts![(E, 6_u16), (G, 3_u16),],
                    rounds: 0_usize,
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, E, O, G, W, O, W, W, O, W, G, O, O, W, W, G, O,
                            W, O, G, W, W, G, O, O, W, O, W, W, O, O, O, E, O, W, W, W, W, W, W, W,
                            W,
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![(1, 1), (3, 1), (3, 2), (1, 3), (5, 3), (1, 4), (4, 5),],
                    unit_types: unit_counts![(E, 2_u16), (G, 5_u16),],
                    rounds: 0_usize,
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, O, E, O, O, O, W, W, O, W, O, O, G, W, W, O, W,
                            W, W, O, W, W, E, W, G, W, G, W, W, O, O, O, W, G, W, W, W, W, W, W, W,
                            W,
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![(2, 1), (5, 2), (1, 4), (3, 4), (5, 4), (5, 5),],
                    unit_types: unit_counts![(E, 2_u16), (G, 4_u16),],
                    rounds: 0_usize,
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, W, W, G, O, O, O, O, O, O, W, W, O, E, O, W, O,
                            O, O, W, W, O, O, W, W, O, O, G, W, W, O, O, O, W, W, O, O, W, W, O, O,
                            O, W, O, O, O, W, W, O, G, O, O, O, G, O, W, W, O, O, O, O, O, G, O, W,
                            W, W, W, W, W, W, W, W, W,
                        ],
                        9_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    units: units![(1, 1), (2, 2), (7, 3), (2, 6), (6, 6), (6, 7),],
                    unit_types: unit_counts![(E, 1_u16), (G, 5_u16),],
                    rounds: 0_usize,
                },
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_try_choose_movement_pos() {
        let solution: Solution = Solution::try_from(
            "\
            #######\n\
            #E..G.#\n\
            #...#.#\n\
            #.G.#G#\n\
            #######\n",
        )
        .unwrap();

        assert_eq!(
            solution.try_choose_movement_pos((1_i32, 1_i32).into(), &mut RunState::new(&solution)),
            Some(SmallPos { x: 2_u8, y: 1_u8 })
        );

        let solution: Solution = Solution::try_from(
            "\
            #######\n\
            #.E...#\n\
            #.....#\n\
            #...G.#\n\
            #######\n",
        )
        .unwrap();

        assert_eq!(
            solution.try_choose_movement_pos((2_i32, 1_i32).into(), &mut RunState::new(&solution)),
            Some(SmallPos { x: 3_u8, y: 1_u8 })
        );
    }

    #[test]
    fn test_run_round() {
        let mut solution: Solution = Solution::try_from(
            "\
            #########\n\
            #G..G..G#\n\
            #.......#\n\
            #.......#\n\
            #G..E..G#\n\
            #.......#\n\
            #.......#\n\
            #G..G..G#\n\
            #########\n",
        )
        .unwrap();
        let mut run_state: RunState = RunState::new(&solution);

        for grid_string in [
            "\
            #########\n\
            #.G...G.#\n\
            #...G...#\n\
            #...E..G#\n\
            #.G.....#\n\
            #.......#\n\
            #G..G..G#\n\
            #.......#\n\
            #########\n",
            "\
            #########\n\
            #..G.G..#\n\
            #...G...#\n\
            #.G.E.G.#\n\
            #.......#\n\
            #G..G..G#\n\
            #.......#\n\
            #.......#\n\
            #########\n",
            "\
            #########\n\
            #.......#\n\
            #..GGG..#\n\
            #..GEG..#\n\
            #G..G...#\n\
            #......G#\n\
            #.......#\n\
            #.......#\n\
            #########\n",
        ] {
            solution.run_round(&mut run_state);

            assert_eq!(String::from(solution.grid.clone()), grid_string);
        }
    }

    #[test]
    fn test_run() {
        for (index, grid_string) in [
            "\
            #######\n\
            #G....#\n\
            #.G...#\n\
            #.#.#G#\n\
            #...#.#\n\
            #....G#\n\
            #######\n",
            "\
            #######\n\
            #...#E#\n\
            #E#...#\n\
            #.E##.#\n\
            #E..#E#\n\
            #.....#\n\
            #######\n",
            "\
            #######\n\
            #.E.E.#\n\
            #.#E..#\n\
            #E.##.#\n\
            #.E.#.#\n\
            #...#.#\n\
            #######\n",
            "\
            #######\n\
            #G.G#.#\n\
            #.#G..#\n\
            #..#..#\n\
            #...#G#\n\
            #...G.#\n\
            #######\n",
            "\
            #######\n\
            #.....#\n\
            #.#G..#\n\
            #.###.#\n\
            #.#.#.#\n\
            #G.G#G#\n\
            #######\n",
            "\
            #########\n\
            #.G.....#\n\
            #G.G#...#\n\
            #.G##...#\n\
            #...##..#\n\
            #.G.#...#\n\
            #.......#\n\
            #.......#\n\
            #########\n",
        ]
        .into_iter()
        .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            solution.run();

            assert_eq!(String::from(solution.grid), grid_string);
        }
    }

    #[test]
    fn test_outcome() {
        for (index, outcome) in [
            27730_usize,
            36334_usize,
            39514_usize,
            27755_usize,
            28944_usize,
            18740_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).outcome(), outcome);
        }
    }

    #[test]
    fn test_elf_win_outcome_and_solution() {
        for (index, elf_win_outcome_grid_string_and_attack_power) in [
            Some((
                4988_usize,
                "\
                #######\n\
                #..E..#\n\
                #...E.#\n\
                #.#.#.#\n\
                #...#.#\n\
                #.....#\n\
                #######\n",
                15_u8,
            )),
            None,
            None,
            Some((
                3478_usize,
                "\
                #######\n\
                #.E.#.#\n\
                #.#E..#\n\
                #..#..#\n\
                #...#.#\n\
                #.....#\n\
                #######\n",
                15_u8,
            )),
            Some((
                6474_usize,
                "\
                #######\n\
                #...E.#\n\
                #.#..E#\n\
                #.###.#\n\
                #.#.#.#\n\
                #...#.#\n\
                #######\n",
                12_u8,
            )),
            Some((
                1140_usize,
                "\
                #########\n\
                #.......#\n\
                #.E.#...#\n\
                #..##...#\n\
                #...##..#\n\
                #...#...#\n\
                #.......#\n\
                #.......#\n\
                #########\n",
                34_u8,
            )),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some((outcome, grid_string, attack_power)) =
                elf_win_outcome_grid_string_and_attack_power
            {
                let elf_win_outcome_and_solution = solution(index).elf_win_outcome_and_solution();

                assert_eq!(elf_win_outcome_and_solution.0, outcome);
                assert_eq!(
                    String::from(elf_win_outcome_and_solution.1.grid),
                    grid_string
                );
                assert_eq!(
                    elf_win_outcome_and_solution.1.unit_types.as_slice()[0_usize]
                        .data
                        .attack_power,
                    attack_power
                );
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
