use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::collections::VecDeque,
};

/* --- Day 15: Warehouse Woes ---

You appear back inside your own mini submarine! Each Historian drives their mini submarine in a different direction; maybe the Chief has his own submarine down here somewhere as well?

You look up to see a vast school of lanternfish swimming past you. On closer inspection, they seem quite anxious, so you drive your mini submarine over to see if you can help.

Because lanternfish populations grow rapidly, they need a lot of food, and that food needs to be stored somewhere. That's why these lanternfish have built elaborate warehouse complexes operated by robots!

These lanternfish seem so anxious because they have lost control of the robot that operates one of their most important warehouses! It is currently running amok, pushing around boxes in the warehouse with no regard for lanternfish logistics or lanternfish inventory management strategies.

Right now, none of the lanternfish are brave enough to swim up to an unpredictable robot so they could shut it off. However, if you could anticipate the robot's movements, maybe they could find a safe option.

The lanternfish already have a map of the warehouse and a list of movements the robot will attempt to make (your puzzle input). The problem is that the movements will sometimes fail as boxes are shifted around, making the actual movements of the robot difficult to predict.

For example:

##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^

As the robot (@) attempts to move, if there are any boxes (O) in the way, the robot will also attempt to push those boxes. However, if this action would cause the robot or a box to move into a wall (#), nothing moves instead, including the robot. The initial positions of these are shown on the map at the top of the document the lanternfish gave you.

The rest of the document describes the moves (^ for up, v for down, < for left, > for right) that the robot will attempt to make, in order. (The moves form a single giant sequence; they are broken into multiple lines just to make copy-pasting easier. Newlines within the move sequence should be ignored.)

Here is a smaller example to get started:

########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

<^^>>>vv<v>>v<<

Were the robot to attempt the given sequence of moves, it would push around the boxes as follows:

Initial state:
########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move <:
########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move ^:
########
#.@O.O.#
##..O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move ^:
########
#.@O.O.#
##..O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move >:
########
#..@OO.#
##..O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move >:
########
#...@OO#
##..O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move >:
########
#...@OO#
##..O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

Move v:
########
#....OO#
##..@..#
#...O..#
#.#.O..#
#...O..#
#...O..#
########

Move v:
########
#....OO#
##..@..#
#...O..#
#.#.O..#
#...O..#
#...O..#
########

Move <:
########
#....OO#
##.@...#
#...O..#
#.#.O..#
#...O..#
#...O..#
########

Move v:
########
#....OO#
##.....#
#..@O..#
#.#.O..#
#...O..#
#...O..#
########

Move >:
########
#....OO#
##.....#
#...@O.#
#.#.O..#
#...O..#
#...O..#
########

Move >:
########
#....OO#
##.....#
#....@O#
#.#.O..#
#...O..#
#...O..#
########

Move v:
########
#....OO#
##.....#
#.....O#
#.#.O@.#
#...O..#
#...O..#
########

Move <:
########
#....OO#
##.....#
#.....O#
#.#O@..#
#...O..#
#...O..#
########

Move <:
########
#....OO#
##.....#
#.....O#
#.#O@..#
#...O..#
#...O..#
########

The larger example has many more moves; after the robot has finished those moves, the warehouse would look like this:

##########
#.O.O.OOO#
#........#
#OO......#
#OO@.....#
#O#.....O#
#O.....OO#
#O.....OO#
#OO....OO#
##########

The lanternfish use their own custom Goods Positioning System (GPS for short) to track the locations of the boxes. The GPS coordinate of a box is equal to 100 times its distance from the top edge of the map plus its distance from the left edge of the map. (This process does not stop at wall tiles; measure all the way to the edges of the map.)

So, the box shown below has a distance of 1 from the top edge of the map and 4 from the left edge of the map, resulting in a GPS coordinate of 100 * 1 + 4 = 104.

#######
#...O..
#......

The lanternfish would like to know the sum of all boxes' GPS coordinates after the robot finishes moving. In the larger example, the sum of all boxes' GPS coordinates is 10092. In the smaller example, the sum is 2028.

Predict the motion of the robot and boxes in the warehouse. After the robot is finished moving, what is the sum of all boxes' GPS coordinates?

--- Part Two ---

The lanternfish use your information to find a safe moment to swim in and turn off the malfunctioning robot! Just as they start preparing a festival in your honor, reports start coming in that a second warehouse's robot is also malfunctioning.

This warehouse's layout is surprisingly similar to the one you just helped. There is one key difference: everything except the robot is twice as wide! The robot's list of movements doesn't change.

To get the wider warehouse's map, start with your original map and, for each tile, make the following changes:

    If the tile is #, the new map contains ## instead.
    If the tile is O, the new map contains [] instead.
    If the tile is ., the new map contains .. instead.
    If the tile is @, the new map contains @. instead.

This will produce a new warehouse map which is twice as wide and with wide boxes that are represented by []. (The robot does not change size.)

The larger example from before would now look like this:

####################
##....[]....[]..[]##
##............[]..##
##..[][]....[]..[]##
##....[]@.....[]..##
##[]##....[]......##
##[]....[]....[]..##
##..[][]..[]..[][]##
##........[]......##
####################

Because boxes are now twice as wide but the robot is still the same size and speed, boxes can be aligned such that they directly push two other boxes at once. For example, consider this situation:

#######
#...#.#
#.....#
#..OO@#
#..O..#
#.....#
#######

<vv<<^^<<^^

After appropriately resizing this map, the robot would push around these boxes as follows:

Initial state:
##############
##......##..##
##..........##
##....[][]@.##
##....[]....##
##..........##
##############

Move <:
##############
##......##..##
##..........##
##...[][]@..##
##....[]....##
##..........##
##############

Move v:
##############
##......##..##
##..........##
##...[][]...##
##....[].@..##
##..........##
##############

Move v:
##############
##......##..##
##..........##
##...[][]...##
##....[]....##
##.......@..##
##############

Move <:
##############
##......##..##
##..........##
##...[][]...##
##....[]....##
##......@...##
##############

Move <:
##############
##......##..##
##..........##
##...[][]...##
##....[]....##
##.....@....##
##############

Move ^:
##############
##......##..##
##...[][]...##
##....[]....##
##.....@....##
##..........##
##############

Move ^:
##############
##......##..##
##...[][]...##
##....[]....##
##.....@....##
##..........##
##############

Move <:
##############
##......##..##
##...[][]...##
##....[]....##
##....@.....##
##..........##
##############

Move <:
##############
##......##..##
##...[][]...##
##....[]....##
##...@......##
##..........##
##############

Move ^:
##############
##......##..##
##...[][]...##
##...@[]....##
##..........##
##..........##
##############

Move ^:
##############
##...[].##..##
##...@.[]...##
##....[]....##
##..........##
##..........##
##############

This warehouse also uses GPS to locate the boxes. For these larger boxes, distances are measured from the edge of the map to the closest edge of the box in question. So, the box shown below has a distance of 1 from the top edge of the map and 5 from the left edge of the map, resulting in a GPS coordinate of 100 * 1 + 5 = 105.

##########
##...[]...
##........

In the scaled-up version of the larger example from above, after the robot has finished all of its moves, the warehouse would look like this:

####################
##[].......[].[][]##
##[]...........[].##
##[]........[][][]##
##[]......[]....[]##
##..##......[]....##
##..[]............##
##..@......[].[][]##
##......[][]..[]..##
####################

The sum of these boxes' GPS coordinates is 9021.

Predict the motion of the robot and boxes in this new, scaled-up warehouse. What is the sum of all boxes' final GPS coordinates? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, Default, PartialEq)]
    enum Cell {
        #[default]
        Empty = EMPTY = b'.',
        Wall = WALL = b'#',
        Robot = ROBOT = b'@',
        Box = BOX = b'O',
        LeftBox = LEFT_BOX = b'[',
        RightBox = RIGHT_BOX = b']',
    }
}

impl Cell {
    fn is_any_box(self) -> bool {
        matches!(self, Self::Box | Self::LeftBox | Self::RightBox)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    grid: Grid2D<Cell>,
    robot: IVec2,
    moves: VecDeque<Direction>,
}

impl Solution {
    const SECOND_WAREHOUSE_SCALE: IVec2 = IVec2::new(2_i32, 1_i32);

    fn grid_string(&self) -> String {
        self.grid.clone().into()
    }

    fn iter_boxes(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.grid
            .cells()
            .iter()
            .enumerate()
            .filter_map(|(index, &cell)| {
                (cell == Cell::Box || cell == Cell::LeftBox)
                    .then(|| self.grid.pos_from_index(index))
            })
    }

    fn gps_coordinate_sum(&self) -> i32 {
        self.iter_boxes()
            .map(|box_pos| box_pos.x + box_pos.y * 100_i32)
            .sum()
    }

    fn self_after_all_moves(&self) -> Self {
        let mut solution: Self = self.clone();

        solution.execute_all_moves();

        solution
    }

    fn gps_coordinate_sum_after_all_moves(&self) -> i32 {
        self.self_after_all_moves().gps_coordinate_sum()
    }

    fn second_warehouse(&self) -> Self {
        let mut grid: Grid2D<Cell> =
            Grid2D::default(self.grid.dimensions() * Self::SECOND_WAREHOUSE_SCALE);

        for (src_cell, dst_cells) in self
            .grid
            .cells()
            .iter()
            .zip(grid.cells_mut().chunks_exact_mut(2_usize))
        {
            match *src_cell {
                Cell::Empty => dst_cells.fill(Cell::Empty),
                Cell::Wall => dst_cells.fill(Cell::Wall),
                Cell::Robot => *dst_cells.first_mut().unwrap() = Cell::Robot,
                Cell::Box => {
                    *dst_cells.first_mut().unwrap() = Cell::LeftBox;
                    *dst_cells.last_mut().unwrap() = Cell::RightBox;
                }
                _ => unimplemented!(),
            }
        }

        Self {
            grid,
            robot: self.robot * Self::SECOND_WAREHOUSE_SCALE,
            moves: self.moves.clone(),
        }
    }

    fn second_warehouse_self_after_all_moves(&self) -> Self {
        let mut solution: Self = self.second_warehouse();

        solution.second_warehouse_execute_all_moves();

        solution
    }

    fn second_warehouse_gps_coordinate_sum_after_all_moves(&self) -> i32 {
        self.second_warehouse_self_after_all_moves()
            .gps_coordinate_sum()
    }

    fn move_entity(&mut self, pos: IVec2, dir: Direction) {
        let cell: &mut Cell = self.grid.get_mut(pos).unwrap();
        let cell_value: Cell = *cell;
        let dir_vec: IVec2 = dir.vec();

        *cell = Cell::Empty;
        *self.grid.get_mut(pos + dir_vec).unwrap() = cell_value;

        if cell_value == Cell::LeftBox && dir.is_north_or_south() {
            self.move_entity(pos + Direction::East.vec(), dir);
        }
    }

    fn try_to_move(&mut self) {
        if let Some(dir) = self.moves.pop_front() {
            if let Some(empty_pos) = CellIter2D::until_boundary(&self.grid, self.robot, dir)
                .skip(1_usize)
                .skip_while(|&pos| self.grid.get(pos).unwrap().is_any_box())
                .next()
                .filter(|&pos| *self.grid.get(pos).unwrap() == Cell::Empty)
            {
                for pos in CellIter2D::try_from(empty_pos..=self.robot)
                    .unwrap()
                    .skip(1_usize)
                {
                    self.move_entity(pos, dir);
                }

                self.robot += dir.vec();
            }
        }
    }

    fn execute_all_moves(&mut self) {
        while !self.moves.is_empty() {
            self.try_to_move()
        }
    }

    fn try_to_move_entity(
        pos: IVec2,
        dir: Direction,
        positions_to_clear: &mut VecDeque<IVec2>,
        entities_to_move: &mut Vec<IVec2>,
    ) {
        if !entities_to_move.contains(&pos) {
            let next_pos: IVec2 = pos + dir.vec();

            positions_to_clear.push_back(next_pos);
            positions_to_clear.push_back(next_pos + Direction::East.vec());
            entities_to_move.push(pos);
        }
    }

    fn second_warehouse_try_to_move(
        &mut self,
        positions_to_clear: &mut VecDeque<IVec2>,
        entities_to_move: &mut Vec<IVec2>,
    ) {
        if let Some(dir) = self.moves.pop_front() {
            if dir.is_north_or_south() {
                positions_to_clear.clear();
                entities_to_move.clear();
                positions_to_clear.push_back(self.robot + dir.vec());
                entities_to_move.push(self.robot);

                let mut pushed_against_wall: bool = false;

                while let Some(position_to_clear) = positions_to_clear
                    .pop_front()
                    .filter(|_| !pushed_against_wall)
                {
                    match self.grid.get(position_to_clear).unwrap() {
                        Cell::Empty => (),
                        Cell::Wall => pushed_against_wall = true,
                        Cell::LeftBox => {
                            Self::try_to_move_entity(
                                position_to_clear,
                                dir,
                                positions_to_clear,
                                entities_to_move,
                            );
                        }
                        Cell::RightBox => {
                            Self::try_to_move_entity(
                                position_to_clear + Direction::West.vec(),
                                dir,
                                positions_to_clear,
                                entities_to_move,
                            );
                        }
                        _ => unimplemented!(),
                    }
                }

                if !pushed_against_wall {
                    for pos in entities_to_move.drain(..).rev() {
                        self.move_entity(pos, dir);
                    }

                    self.robot += dir.vec();
                }
            } else {
                self.moves.push_front(dir);

                self.try_to_move();
            }
        }
    }

    fn second_warehouse_execute_all_moves(&mut self) {
        let mut positions_to_clear: VecDeque<IVec2> = VecDeque::new();
        let mut entities_to_move: Vec<IVec2> = Vec::new();

        while !self.moves.is_empty() {
            self.second_warehouse_try_to_move(&mut positions_to_clear, &mut entities_to_move);
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                map_opt(Grid2D::parse, |grid| {
                    (grid
                        .cells()
                        .iter()
                        .all(|&cell| cell != Cell::LeftBox && cell != Cell::RightBox)
                        && CellIter2D::iter_edges_for_dimensions(grid.dimensions())
                            .all(|pos| grid.get(pos).map_or(false, |&cell| cell == Cell::Wall)))
                    .then(|| {
                        grid.cells()
                            .iter()
                            .enumerate()
                            .filter_map(|(index, &cell)| {
                                (cell == Cell::Robot).then(|| grid.pos_from_index(index))
                            })
                            .next()
                    })
                    .flatten()
                    .map(|robot| (grid, robot))
                }),
                line_ending,
                map(
                    many0(terminated(
                        alt((
                            map(tag("^"), |_| Direction::North),
                            map(tag(">"), |_| Direction::East),
                            map(tag("v"), |_| Direction::South),
                            map(tag("<"), |_| Direction::West),
                        )),
                        opt(line_ending),
                    )),
                    VecDeque::from,
                ),
            ),
            |((grid, robot), moves)| Self { grid, robot, moves },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// No issues.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let solution: Self = self.self_after_all_moves();

            dbg!(solution.gps_coordinate_sum());

            println!("{}", solution.grid_string());
        } else {
            dbg!(self.gps_coordinate_sum_after_all_moves());
        }
    }

    /// Very daunting initially, but I'm pretty content with my approach.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let solution: Self = self.second_warehouse_self_after_all_moves();

            dbg!(solution.gps_coordinate_sum());

            println!("{}", solution.grid_string());
        } else {
            dbg!(self.second_warehouse_gps_coordinate_sum_after_all_moves());
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
        ########\n\
        #..O.O.#\n\
        ##@.O..#\n\
        #...O..#\n\
        #.#.O..#\n\
        #...O..#\n\
        #......#\n\
        ########\n\
        \n\
        <^^>>>vv<v>>v<<\n",
        "\
        ##########\n\
        #..O..O.O#\n\
        #......O.#\n\
        #.OO..O.O#\n\
        #..O@..O.#\n\
        #O#..O...#\n\
        #O..O..O.#\n\
        #.OO.O.OO#\n\
        #....O...#\n\
        ##########\n\
        \n\
        <vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^\n\
        vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v\n\
        ><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<\n\
        <<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^\n\
        ^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><\n\
        ^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^\n\
        >^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^\n\
        <><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>\n\
        ^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>\n\
        v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use {
                Cell::{Box as B, Empty as E, Robot as R, Wall as W},
                Direction::*,
            };

            vec![
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, W, E, E, B, E, B, E, W, W, W, R, E, B, E, E, W,
                            W, E, E, E, B, E, E, W, W, E, W, E, B, E, E, W, W, E, E, E, B, E, E, W,
                            W, E, E, E, E, E, E, W, W, W, W, W, W, W, W, W,
                        ],
                        8_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    robot: (2_i32, 2_i32).into(),
                    moves: vec![
                        West, North, North, East, East, East, South, South, West, South, East,
                        East, South, West, West,
                    ]
                    .into(),
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            W, W, W, W, W, W, W, W, W, W, W, E, E, B, E, E, B, E, B, W, W, E, E, E,
                            E, E, E, B, E, W, W, E, B, B, E, E, B, E, B, W, W, E, E, B, R, E, E, B,
                            E, W, W, B, W, E, E, B, E, E, E, W, W, B, E, E, B, E, E, B, E, W, W, E,
                            B, B, E, B, E, B, B, W, W, E, E, E, E, B, E, E, E, W, W, W, W, W, W, W,
                            W, W, W, W,
                        ],
                        10_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                    robot: (4_i32, 4_i32).into(),
                    moves: vec![
                        West, South, South, East, North, West, South, North, East, South, East,
                        North, South, South, North, South, East, South, West, East, South, North,
                        South, West, South, West, North, South, South, West, West, West, North,
                        East, West, West, East, West, East, East, South, West, South, South, South,
                        West, East, North, South, North, East, North, West, West, West, East, West,
                        West, South, West, West, West, South, North, South, South, North, South,
                        East, North, South, South, South, West, West, North, East, North, South,
                        North, North, East, West, West, East, East, East, West, East, North, West,
                        West, East, West, North, South, South, North, North, West, East, South,
                        South, South, West, East, East, West, North, North, South, East, North,
                        East, South, South, West, East, South, West, West, West, West, South, West,
                        North, South, East, North, West, North, North, East, East, East, North,
                        West, South, West, South, East, West, East, South, South, East, South,
                        North, South, North, West, East, East, West, East, East, East, East, West,
                        North, North, East, South, South, East, South, West, North, North, North,
                        East, East, South, North, South, North, West, North, North, East, South,
                        North, North, East, South, North, West, North, South, East, South, West,
                        East, East, South, North, South, North, West, South, East, South, North,
                        North, West, North, North, South, South, West, West, West, South, West,
                        North, East, East, North, North, North, North, East, East, East, South,
                        North, West, East, South, South, South, North, East, West, South, West,
                        West, West, East, North, North, North, South, South, North, West, South,
                        South, South, East, North, East, South, West, North, North, North, North,
                        South, West, East, North, East, South, South, South, South, East, West,
                        East, East, South, North, West, West, North, North, North, North, North,
                        North, East, West, North, East, West, East, East, East, West, East, North,
                        North, West, West, North, North, South, East, East, East, West, North,
                        West, South, East, North, West, South, South, East, East, South, East,
                        East, East, North, South, East, West, East, North, South, East, West, West,
                        West, West, South, East, East, South, West, South, West, South, East,
                        South, South, South, East, North, West, East, West, West, East, North,
                        East, West, North, East, East, West, East, North, South, West, East, West,
                        North, South, South, South, West, North, North, West, East, West, South,
                        West, West, West, West, West, East, West, North, South, West, West, West,
                        East, West, West, West, North, North, West, South, West, North, North,
                        North, East, West, North, East, East, North, West, South, North, East,
                        West, West, West, North, East, East, North, South, West, South, North,
                        South, West, South, North, East, North, East, East, North, South, East,
                        South, South, East, North, West, West, North, South, West, East, East,
                        West, West, East, West, West, South, West, West, South, East, West, East,
                        South, West, North, South, South, West, West, West, East, North, North,
                        South, North, East, North, North, East, East, East, West, West, North,
                        South, East, East, South, North, South, East, West, North, North, East,
                        East, North, West, East, South, South, North, West, East, West, North,
                        North, East, North, North, North, West, East, West, South, South, South,
                        South, South, North, South, West, South, West, West, East, North, South,
                        West, South, East, South, West, West, North, East, West, West, East, West,
                        West, East, West, West, West, North, North, West, West, West, North, West,
                        West, East, East, West, West, East, West, North, North, North, East, North,
                        North, West, East, North, East, South, West, East, North, North, East,
                        South, South, West, North, South, North, South, West, South, South, East,
                        North, West, East, West, South, West, North, South, East, North, North,
                        North, East, East, East, North, North, South, South, South, North, East,
                        South, South, South, West, East, East, East, North, West, North, East,
                        East, East, East, East, North, West, West, North, South, East, North,
                        South, South, South, West, East, North, West, East, West, West, South,
                        East, South, North, North, East, East, East, West, West, North, North,
                        West, East, East, North, South, North, West, South, North, South, South,
                        West, East, South, North, West, West, East, North, West, North, South,
                        North, South, East, West, North, West, West, West, East, West, West, North,
                        West, South, East, West, South, West, East, South, South, East, East,
                        South, East, West, South, North, West, South, South, West, East, South,
                        North, West, West, North,
                    ]
                    .into(),
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
    fn test_grid_string() {
        for (index, grid_string) in [
            "\
            ########\n\
            #..O.O.#\n\
            ##@.O..#\n\
            #...O..#\n\
            #.#.O..#\n\
            #...O..#\n\
            #......#\n\
            ########\n",
            "\
            ##########\n\
            #..O..O.O#\n\
            #......O.#\n\
            #.OO..O.O#\n\
            #..O@..O.#\n\
            #O#..O...#\n\
            #O..O..O.#\n\
            #.OO.O.OO#\n\
            #....O...#\n\
            ##########\n",
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).grid_string(), grid_string);
        }
    }

    #[test]
    fn test_try_to_move() {
        for (index, grid_strings) in [
            Some(vec![
                "\
                ########\n\
                #..O.O.#\n\
                ##@.O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #.@O.O.#\n\
                ##..O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #.@O.O.#\n\
                ##..O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #..@OO.#\n\
                ##..O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #...@OO#\n\
                ##..O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #...@OO#\n\
                ##..O..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #......#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##..@..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##..@..#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.@...#\n\
                #...O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #..@O..#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #...@O.#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #....@O#\n\
                #.#.O..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #.....O#\n\
                #.#.O@.#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #.....O#\n\
                #.#O@..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
                "\
                ########\n\
                #....OO#\n\
                ##.....#\n\
                #.....O#\n\
                #.#O@..#\n\
                #...O..#\n\
                #...O..#\n\
                ########\n",
            ]),
            None,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(grid_strings) = grid_strings {
                let mut solution: Solution = solution(index).clone();

                for grid_string in grid_strings {
                    solution.try_to_move();

                    assert_eq!(solution.grid_string(), grid_string);
                }
            }
        }
    }

    #[test]
    fn test_execute_all_moves() {
        for (index, grid_string) in [
            "\
            ########\n\
            #....OO#\n\
            ##.....#\n\
            #.....O#\n\
            #.#O@..#\n\
            #...O..#\n\
            #...O..#\n\
            ########\n",
            "\
            ##########\n\
            #.O.O.OOO#\n\
            #........#\n\
            #OO......#\n\
            #OO@.....#\n\
            #O#.....O#\n\
            #O.....OO#\n\
            #O.....OO#\n\
            #OO....OO#\n\
            ##########\n",
        ]
        .into_iter()
        .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            solution.execute_all_moves();

            assert_eq!(solution.grid_string(), grid_string);
        }
    }

    #[test]
    fn test_gps_coordinate_sum() {
        for (index, gps_coordinate_sum) in [2028_i32, 10092_i32].into_iter().enumerate() {
            let mut solution: Solution = solution(index).clone();

            solution.execute_all_moves();

            assert_eq!(solution.gps_coordinate_sum(), gps_coordinate_sum);
        }
    }

    #[test]
    fn test_second_warehouse() {
        for (index, grid_string) in [
            None,
            Some(
                "\
                ####################\n\
                ##....[]....[]..[]##\n\
                ##............[]..##\n\
                ##..[][]....[]..[]##\n\
                ##....[]@.....[]..##\n\
                ##[]##....[]......##\n\
                ##[]....[]....[]..##\n\
                ##..[][]..[]..[][]##\n\
                ##........[]......##\n\
                ####################\n",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(grid_string) = grid_string {
                assert_eq!(
                    solution(index).second_warehouse().grid_string(),
                    grid_string
                );
            }
        }
    }

    #[test]
    fn test_second_warehouse_try_to_move() {
        let mut solution: Solution = Solution::try_from(
            "\
            #######\n\
            #...#.#\n\
            #.....#\n\
            #..OO@#\n\
            #..O..#\n\
            #.....#\n\
            #######\n\
            \n\
            <vv<<^^<<^^\n",
        )
        .unwrap()
        .second_warehouse();

        for grid_string in [
            "\
            ##############\n\
            ##......##..##\n\
            ##..........##\n\
            ##...[][]@..##\n\
            ##....[]....##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##..........##\n\
            ##...[][]...##\n\
            ##....[].@..##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##..........##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##.......@..##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##..........##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##......@...##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##..........##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##.....@....##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##.....@....##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##.....@....##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##....@.....##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##...[][]...##\n\
            ##....[]....##\n\
            ##...@......##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##......##..##\n\
            ##...[][]...##\n\
            ##...@[]....##\n\
            ##..........##\n\
            ##..........##\n\
            ##############\n",
            "\
            ##############\n\
            ##...[].##..##\n\
            ##...@.[]...##\n\
            ##....[]....##\n\
            ##..........##\n\
            ##..........##\n\
            ##############\n",
        ] {
            solution.second_warehouse_try_to_move(&mut VecDeque::new(), &mut Vec::new());

            assert_eq!(solution.grid_string(), grid_string);
        }
    }

    #[test]
    fn test_second_warehouse_execute_all_moves() {
        for (index, grid_string) in [
            None,
            Some(
                "\
                ####################\n\
                ##[].......[].[][]##\n\
                ##[]...........[].##\n\
                ##[]........[][][]##\n\
                ##[]......[]....[]##\n\
                ##..##......[]....##\n\
                ##..[]............##\n\
                ##..@......[].[][]##\n\
                ##......[][]..[]..##\n\
                ####################\n",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(grid_string) = grid_string {
                let mut solution: Solution = solution(index).second_warehouse();

                solution.second_warehouse_execute_all_moves();

                assert_eq!(solution.grid_string(), grid_string);
            }
        }
    }

    #[test]
    fn test_second_warehouse_gps_coordinate_sum_after_all_moves() {
        for (index, second_warehouse_gps_coordinate_sum_after_all_moves) in
            [None, Some(9021_i32)].into_iter().enumerate()
        {
            if let Some(second_warehouse_gps_coordinate_sum_after_all_moves) =
                second_warehouse_gps_coordinate_sum_after_all_moves
            {
                assert_eq!(
                    solution(index).second_warehouse_gps_coordinate_sum_after_all_moves(),
                    second_warehouse_gps_coordinate_sum_after_all_moves
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
