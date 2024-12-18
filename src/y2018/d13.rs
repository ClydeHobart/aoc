use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    std::collections::HashMap,
};

/* --- Day 13: Mine Cart Madness ---

A crop of this size requires significant logistics to transport produce, soil, fertilizer, and so on. The Elves are very busy pushing things around in carts on some kind of rudimentary system of tracks they've come up with.

Seeing as how cart-and-track systems don't appear in recorded history for another 1000 years, the Elves seem to be making this up as they go along. They haven't even figured out how to avoid collisions yet.

You map out the tracks (your puzzle input) and see where you can help.

Tracks consist of straight paths (| and -), curves (/ and \), and intersections (+). Curves connect exactly two perpendicular pieces of track; for example, this is a closed loop:

/----\
|    |
|    |
\----/

Intersections occur when two perpendicular paths cross. At an intersection, a cart is capable of turning left, turning right, or continuing straight. Here are two loops connected by two intersections:

/-----\
|     |
|  /--+--\
|  |  |  |
\--+--/  |
   |     |
   \-----/

Several carts are also on the tracks. Carts always face either up (^), down (v), left (<), or right (>). (On your initial map, the track under each cart is a straight path matching the direction the cart is facing.)

Each time a cart has the option to turn (by arriving at any intersection), it turns left the first time, goes straight the second time, turns right the third time, and then repeats those directions starting again with left the fourth time, straight the fifth time, and so on. This process is independent of the particular intersection at which the cart has arrived - that is, the cart has no per-intersection memory.

Carts all move at the same speed; they take turns moving a single step at a time. They do this based on their current location: carts on the top row move first (acting from left to right), then carts on the second row move (again from left to right), then carts on the third row, and so on. Once each cart has moved one step, the process repeats; each of these loops is called a tick.

For example, suppose there are two carts on a straight track:

|  |  |  |  |
v  |  |  |  |
|  v  v  |  |
|  |  |  v  X
|  |  ^  ^  |
^  ^  |  |  |
|  |  |  |  |

First, the top cart moves. It is facing down (v), so it moves down one square. Second, the bottom cart moves. It is facing up (^), so it moves up one square. Because all carts have moved, the first tick ends. Then, the process repeats, starting with the first cart. The first cart moves down, then the second cart moves up - right into the first cart, colliding with it! (The location of the crash is marked with an X.) This ends the second and last tick.

Here is a longer example:

/->-\
|   |  /----\
| /-+--+-\  |
| | |  | v  |
\-+-/  \-+--/
  \------/

/-->\
|   |  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \->--/
  \------/

/---v
|   |  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-+>-/
  \------/

/---\
|   v  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-+->/
  \------/

/---\
|   |  /----\
| /->--+-\  |
| | |  | |  |
\-+-/  \-+--^
  \------/

/---\
|   |  /----\
| /-+>-+-\  |
| | |  | |  ^
\-+-/  \-+--/
  \------/

/---\
|   |  /----\
| /-+->+-\  ^
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /----<
| /-+-->-\  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /---<\
| /-+--+>\  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /--<-\
| /-+--+-v  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /-<--\
| /-+--+-\  |
| | |  | v  |
\-+-/  \-+--/
  \------/

/---\
|   |  /<---\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-<--/
  \------/

/---\
|   |  v----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \<+--/
  \------/

/---\
|   |  /----\
| /-+--v-\  |
| | |  | |  |
\-+-/  ^-+--/
  \------/

/---\
|   |  /----\
| /-+--+-\  |
| | |  X |  |
\-+-/  \-+--/
  \------/

After following their respective paths for a while, the carts eventually crash. To help prevent crashes, you'd like to know the location of the first crash. Locations are given in X,Y coordinates, where the furthest left column is X=0 and the furthest top row is Y=0:

           111
 0123456789012
0/---\
1|   |  /----\
2| /-+--+-\  |
3| | |  X |  |
4\-+-/  \-+--/
5  \------/

In this example, the location of the first crash is 7,3.

--- Part Two ---

There isn't much you can do to prevent crashes in this ridiculous system. However, by predicting the crashes, the Elves know where to be in advance and instantly remove the two crashing carts the moment any crash occurs.

They can proceed like this for a while, but eventually, they're going to run out of carts. It could be useful to figure out where the last cart that hasn't crashed will end up.

For example:

/>-<\
|   |
| /<+-\
| | | v
\>+</ |
  |   ^
  \<->/

/---\
|   |
| v-+-\
| | | |
\-+-/ |
  |   |
  ^---^

/---\
|   |
| /-+-\
| v | |
\-+-/ |
  ^   ^
  \---/

/---\
|   |
| /-+-\
| | | |
\-+-/ ^
  |   |
  \---/

After four very expensive crashes, a tick ends with only one cart remaining; its final location is 6,4.

What is the location of the last cart at the end of the first tick where it is the only cart left? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Empty = EMPTY = b' ',
        TrackHorizontal = TRACK_HORIZONTAL = b'-',
        TrackVertical = TRACK_VERTICAL = b'|',
        TrackCurved1 = TRACK_CURVED_1 = b'\\',
        TrackCurved2 = TRACK_CURVED_2 = b'/',
        TrackIntersection = INTERSECTION = b'+',
        CartFacingNorth = CART_FACING_NORTH = b'^',
        CartFacingEast = CART_FACING_EAST = b'>',
        CartFacingSouth = CART_FACING_SOUTH = b'v',
        CartFacingWest = CART_FACING_WEST = b'<',
    }
}

impl Cell {
    fn track_from_cart_dir(dir: Direction) -> Self {
        if dir.is_north_or_south() {
            Self::TrackVertical
        } else {
            Self::TrackHorizontal
        }
    }

    fn try_cart_dir(self) -> Option<Direction> {
        match self {
            Cell::CartFacingNorth => Some(Direction::North),
            Cell::CartFacingEast => Some(Direction::East),
            Cell::CartFacingSouth => Some(Direction::South),
            Cell::CartFacingWest => Some(Direction::West),
            _ => None,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Cart {
    dir: Direction,
    turn: Turn,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    grid: Grid2D<Cell>,
    carts: HashMap<u16, Cart>,
    sorted_carts: Vec<u16>,
}

impl Solution {
    fn string_for_pos(pos: IVec2) -> String {
        format!("{},{}", pos.x, pos.y)
    }

    fn first_crash_location(&self) -> String {
        let mut solution: Self = self.clone();
        let mut first_crash: Option<IVec2> = None;

        while first_crash.is_none() {
            first_crash = solution.tick();
        }

        Self::string_for_pos(first_crash.unwrap())
    }

    fn try_last_cart_location(&self) -> Option<String> {
        let mut solution: Self = self.clone();

        while solution.carts.len() > 1_usize {
            solution.tick();
        }

        solution.carts.into_keys().last().map(|sortable_index| {
            Self::string_for_pos(SmallPos::from_sortable_index(sortable_index).get())
        })
    }

    fn tick(&mut self) -> Option<IVec2> {
        let mut first_crash: Option<IVec2> = None;

        self.sorted_carts.clear();
        self.sorted_carts
            .extend(self.carts.iter().map(|(&sortable_index, _)| sortable_index));
        self.sorted_carts.sort();

        for curr_sortable_index in self.sorted_carts.drain(..) {
            if let Some(mut cart) = self.carts.get(&curr_sortable_index).copied() {
                self.carts.remove(&curr_sortable_index);

                let curr_dir: Direction = cart.dir;
                let curr_pos_and_dir: SmallPosAndDir =
                    SmallPosAndDir::from_sortable_index_and_dir(curr_sortable_index, curr_dir);
                let curr_pos: IVec2 = curr_pos_and_dir.pos.get();
                let next_pos: IVec2 = curr_pos + curr_dir.vec();

                if let Some(next_cell) = self.grid.get(next_pos) {
                    let next_dir: Direction = match *next_cell {
                        Cell::TrackCurved1 => curr_dir.turn(curr_dir.is_north_or_south()),
                        Cell::TrackCurved2 => curr_dir.turn(!curr_dir.is_north_or_south()),
                        Cell::TrackIntersection => {
                            let turn: Turn = cart.turn;

                            cart.turn = turn.next();

                            curr_dir + turn
                        }
                        Cell::TrackHorizontal | Cell::TrackVertical => curr_dir,
                        _ => unreachable!(),
                    };

                    // SAFETY: `next_pos` is valid.
                    let next_pos_and_dir: SmallPosAndDir =
                        unsafe { SmallPosAndDir::from_pos_and_dir_unsafe(next_pos, next_dir) };
                    let next_sortable_index: u16 = next_pos_and_dir.pos.sortable_index();

                    if self
                        .carts
                        .insert(
                            next_sortable_index,
                            Cart {
                                dir: next_dir,
                                turn: cart.turn,
                            },
                        )
                        .is_some()
                    {
                        self.carts.remove(&next_sortable_index);

                        if first_crash.is_none() {
                            first_crash = Some(next_pos);
                        }
                    }
                }
            }
        }

        first_crash
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |mut grid: Grid2D<Cell>| {
            grid.dimensions()
                .cmple(SmallPos::MAX_DIMENSIONS)
                .all()
                .then(|| {
                    let dimensions: IVec2 = grid.dimensions();

                    let carts: HashMap<u16, Cart> = grid
                        .cells_mut()
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(index, cell)| {
                            let pos: IVec2 =
                                grid_2d_pos_from_index_and_dimensions(index, dimensions);

                            cell.try_cart_dir().map(|dir| {
                                *cell = Cell::track_from_cart_dir(dir);

                                // SAFETY: `pos` is valid.
                                (
                                    unsafe { SmallPosAndDir::from_pos_and_dir_unsafe(pos, dir) }
                                        .pos
                                        .sortable_index(),
                                    Cart {
                                        dir,
                                        turn: Turn::Left,
                                    },
                                )
                            })
                        })
                        .collect();

                    Self {
                        grid,
                        carts,
                        sorted_carts: Vec::new(),
                    }
                })
        })(input)
    }
}

impl RunQuestions for Solution {
    /// The children yearn for the mines.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.first_crash_location());
    }

    /// Had to refactor how I was keeping track of carts to solve this one. My initial try was too
    /// brittle and incorrect.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_last_cart_location());
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
        concat!(
            r"/->-\        ",
            "\n",
            r"|   |  /----\",
            "\n",
            r"| /-+--+-\  |",
            "\n",
            r"| | |  | v  |",
            "\n",
            r"\-+-/  \-+--/",
            "\n",
            r"  \------/   ",
            "\n",
        ),
        concat!(
            r"/>-<\  ", "\n", r"|   |  ", "\n", r"| /<+-\", "\n", r"| | | v", "\n", r"\>+</ |",
            "\n", r"  |   ^", "\n", r"  \<->/", "\n",
        ),
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{
                Empty as E, TrackCurved1 as A, TrackCurved2 as B, TrackHorizontal as H,
                TrackIntersection as I, TrackVertical as V,
            };

            vec![
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            B, H, H, H, A, E, E, E, E, E, E, E, E, V, E, E, E, V, E, E, B, H, H, H,
                            H, A, V, E, B, H, I, H, H, I, H, A, E, E, V, V, E, V, E, V, E, E, V, E,
                            V, E, E, V, A, H, I, H, B, E, E, A, H, I, H, H, B, E, E, A, H, H, H, H,
                            H, H, B, E, E, E,
                        ],
                        IVec2::new(13_i32, 6_i32),
                    )
                    .unwrap(),
                    carts: [
                        (
                            0x03_09_u16,
                            Cart {
                                dir: Direction::South,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x00_02_u16,
                            Cart {
                                dir: Direction::East,
                                turn: Turn::Left,
                            },
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    sorted_carts: Vec::new(),
                },
                Solution {
                    grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            B, H, H, H, A, E, E, V, E, E, E, V, E, E, V, E, B, H, I, H, A, V, E, V,
                            E, V, E, V, A, H, I, H, B, E, V, E, E, V, E, E, E, V, E, E, A, H, H, H,
                            B,
                        ],
                        IVec2::new(7_i32, 7_i32),
                    )
                    .unwrap(),
                    carts: [
                        (
                            0x06_05_u16,
                            Cart {
                                dir: Direction::East,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x06_03_u16,
                            Cart {
                                dir: Direction::West,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x05_06_u16,
                            Cart {
                                dir: Direction::North,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x04_03_u16,
                            Cart {
                                dir: Direction::West,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x04_01_u16,
                            Cart {
                                dir: Direction::East,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x03_06_u16,
                            Cart {
                                dir: Direction::South,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x02_03_u16,
                            Cart {
                                dir: Direction::West,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x00_03_u16,
                            Cart {
                                dir: Direction::West,
                                turn: Turn::Left,
                            },
                        ),
                        (
                            0x00_01_u16,
                            Cart {
                                dir: Direction::East,
                                turn: Turn::Left,
                            },
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    sorted_carts: Vec::new(),
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
    fn test_first_crash_location() {
        for (index, first_crash_location) in ["7,3", "2,0"].into_iter().enumerate() {
            assert_eq!(solution(index).first_crash_location(), first_crash_location);
        }
    }

    #[test]
    fn test_try_last_cart_location() {
        for (index, last_cart_location) in [None, Some(String::from("6,4"))].into_iter().enumerate()
        {
            assert_eq!(solution(index).try_last_cart_location(), last_cart_location);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
