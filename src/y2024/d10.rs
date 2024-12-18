use {
    crate::*,
    glam::IVec2,
    nom::{character::complete::satisfy, combinator::map, error::Error, Err, IResult},
    strum::IntoEnumIterator,
};

/* --- Day 10: Hoof It ---

You all arrive at a Lava Production Facility on a floating island in the sky. As the others begin to search the massive industrial complex, you feel a small nose boop your leg and look down to discover a reindeer wearing a hard hat.

The reindeer is holding a book titled "Lava Island Hiking Guide". However, when you open the book, you discover that most of it seems to have been scorched by lava! As you're about to ask how you can help, the reindeer brings you a blank topographic map of the surrounding area (your puzzle input) and looks up at you excitedly.

Perhaps you can help fill in the missing hiking trails?

The topographic map indicates the height at each position using a scale from 0 (lowest) to 9 (highest). For example:

0123
1234
8765
9876

Based on un-scorched scraps of the book, you determine that a good hiking trail is as long as possible and has an even, gradual, uphill slope. For all practical purposes, this means that a hiking trail is any path that starts at height 0, ends at height 9, and always increases by a height of exactly 1 at each step. Hiking trails never include diagonal steps - only up, down, left, or right (from the perspective of the map).

You look up from the map and notice that the reindeer has helpfully begun to construct a small pile of pencils, markers, rulers, compasses, stickers, and other equipment you might need to update the map with hiking trails.

A trailhead is any position that starts one or more hiking trails - here, these positions will always have height 0. Assembling more fragments of pages, you establish that a trailhead's score is the number of 9-height positions reachable from that trailhead via a hiking trail. In the above example, the single trailhead in the top left corner has a score of 1 because it can reach a single 9 (the one in the bottom left).

This trailhead has a score of 2:

...0...
...1...
...2...
6543456
7.....7
8.....8
9.....9

(The positions marked . are impassable tiles to simplify these examples; they do not appear on your actual topographic map.)

This trailhead has a score of 4 because every 9 is reachable via a hiking trail except the one immediately to the left of the trailhead:

..90..9
...1.98
...2..7
6543456
765.987
876....
987....

This topographic map contains two trailheads; the trailhead at the top has a score of 1, while the trailhead at the bottom has a score of 2:

10..9..
2...8..
3...7..
4567654
...8..3
...9..2
.....01

Here's a larger example:

89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732

This larger example has 9 trailheads. Considering the trailheads in reading order, they have scores of 5, 6, 5, 3, 1, 3, 5, 3, and 5. Adding these scores together, the sum of the scores of all trailheads is 36.

The reindeer gleefully carries over a protractor and adds it to the pile. What is the sum of the scores of all trailheads on your topographic map?

--- Part Two ---

The reindeer spends a few minutes reviewing your hiking trail map before realizing something, disappearing for a few minutes, and finally returning with yet another slightly-charred piece of paper.

The paper describes a second way to measure a trailhead called its rating. A trailhead's rating is the number of distinct hiking trails which begin at that trailhead. For example:

.....0.
..4321.
..5..2.
..6543.
..7..4.
..8765.
..9....

The above map has a single trailhead; its rating is 3 because there are exactly three distinct hiking trails which begin at that position:

.....0.   .....0.   .....0.
..4321.   .....1.   .....1.
..5....   .....2.   .....2.
..6....   ..6543.   .....3.
..7....   ..7....   .....4.
..8....   ..8....   ..8765.
..9....   ..9....   ..9....

Here is a map containing a single trailhead with rating 13:

..90..9
...1.98
...2..7
6543456
765.987
876....
987....

This map contains a single trailhead with rating 227 (because there are 121 distinct hiking trails that lead to the 9 on the right edge and 106 that lead to the 9 on the bottom edge):

012345
123456
234567
345678
4.6789
56789.

Here's the larger example from before:

89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732

Considering its trailheads in reading order, they have ratings of 20, 24, 10, 4, 1, 4, 5, 8, and 5. The sum of all trailhead ratings in this larger example topographic map is 81.

You're not sure how, but the reindeer seems to have crafted some tiny flags out of toothpicks and bits of paper and is using them to mark trailheads on your topographic map. What is the sum of the ratings of all trailheads? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
struct Cell(u8);

impl Cell {
    const ZERO: Self = Self(b'0');
    const NINE: Self = Self(b'9');
    const OFFSET: u8 = b'0';

    const fn new(digit: u8) -> Self {
        Self(digit + Self::OFFSET)
    }
}

/// SAFETY: See `Parse` implementation.
unsafe impl IsValidAscii for Cell {}

impl Parse for Cell {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(|c| c.is_ascii_digit() || c == '.'), |c| {
            Self(c as u8)
        })(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Default)]
struct Trailhead {
    pos: IVec2,
    score: u32,
    rating: u32,
}

struct TrailheadScorer<'g> {
    grid: &'g Grid2D<Cell>,
    trailhead: Trailhead,
}

impl<'g> BreadthFirstSearch for TrailheadScorer<'g> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.trailhead.pos
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        let cell: Cell = self.grid.get(*vertex).copied().unwrap();

        neighbors.extend(
            (cell != Cell::NINE)
                .then(|| {
                    Direction::iter().filter_map(|dir| {
                        let neighbor: IVec2 = *vertex + dir.vec();

                        Solution::is_valid_neighbor(&self.grid, cell, neighbor).then_some(neighbor)
                    })
                })
                .into_iter()
                .flatten(),
        );
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        self.trailhead.score += (self.grid.get(*to).copied().unwrap() == Cell::NINE) as u32;
    }

    fn reset(&mut self) {
        self.trailhead.score = 0_u32;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Cell>);

impl Solution {
    fn is_valid_neighbor(grid: &Grid2D<Cell>, cell: Cell, neighbor: IVec2) -> bool {
        grid.get(neighbor)
            .copied()
            .map_or(false, |neighbor_cell| neighbor_cell.0 == cell.0 + 1_u8)
    }

    fn iter_trailheads(&self) -> impl Iterator<Item = Trailhead> + '_ {
        let mut trailhead_scorer: TrailheadScorer = TrailheadScorer {
            grid: &self.0,
            trailhead: Trailhead::default(),
        };
        let mut ratings: Grid2D<u32> = Grid2D::default(self.0.dimensions());

        for digit in (0_u8..=9_u8).rev() {
            for (index, cell) in self.0.cells().iter().copied().enumerate() {
                if cell == Cell::new(digit) {
                    if cell == Cell::NINE {
                        ratings.cells_mut()[index] = 1_u32;
                    } else {
                        let pos: IVec2 = self.0.pos_from_index(index);
                        let rating: u32 = Direction::iter()
                            .filter_map(|dir| {
                                let neighbor: IVec2 = pos + dir.vec();

                                Self::is_valid_neighbor(&self.0, cell, neighbor)
                                    .then(|| ratings.get(neighbor).copied().unwrap())
                            })
                            .sum();

                        ratings.cells_mut()[index] = rating;
                    }
                }
            }
        }

        self.0
            .cells()
            .iter()
            .copied()
            .enumerate()
            .filter_map(move |(index, cell)| {
                (cell == Cell::ZERO)
                    .then(|| {
                        trailhead_scorer.trailhead.pos = self.0.pos_from_index(index);
                        trailhead_scorer.trailhead.score = 0_u32;
                        trailhead_scorer.trailhead.rating = ratings.cells()[index];

                        if trailhead_scorer.trailhead.rating > 0_u32 {
                            trailhead_scorer.run();
                        }

                        trailhead_scorer.trailhead
                    })
                    .filter(|trailhead| trailhead.score > 0_u32)
            })
    }

    fn trailhead_score_sum(&self) -> u32 {
        self.iter_trailheads()
            .map(|trailhead| trailhead.score)
            .sum()
    }

    fn trailhead_rating_sum(&self) -> u32 {
        self.iter_trailheads()
            .map(|trailhead| trailhead.rating)
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Guessing part 2 is going to be "count the number of distinct trails from each trailhead".
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.trailhead_score_sum());
    }

    /// Not sure if there's a more well-known/applicable algorithm for this, or if this actually has
    /// a name, but it seemed more sensible than something that tried to make A* or Dijkstra work.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.trailhead_rating_sum());
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
        0123\n\
        1234\n\
        8765\n\
        9876\n",
        "\
        ...0...\n\
        ...1...\n\
        ...2...\n\
        6543456\n\
        7.....7\n\
        8.....8\n\
        9.....9\n",
        "\
        ..90..9\n\
        ...1.98\n\
        ...2..7\n\
        6543456\n\
        765.987\n\
        876....\n\
        987....\n",
        "\
        10..9..\n\
        2...8..\n\
        3...7..\n\
        4567654\n\
        ...8..3\n\
        ...9..2\n\
        .....01\n",
        "\
        89010123\n\
        78121874\n\
        87430965\n\
        96549874\n\
        45678903\n\
        32019012\n\
        01329801\n\
        10456732\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            macro_rules! cells {
                [ $( $cell:expr, )* ] => { vec![ $( Cell($cell as u8), )* ] }
            }

            vec![
                Solution(
                    Grid2D::try_from_cells_and_dimensions(
                        cells![
                            '0', '1', '2', '3', '1', '2', '3', '4', '8', '7', '6', '5', '9', '8',
                            '7', '6',
                        ],
                        4_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                ),
                Solution(
                    Grid2D::try_from_cells_and_dimensions(
                        cells![
                            '.', '.', '.', '0', '.', '.', '.', '.', '.', '.', '1', '.', '.', '.',
                            '.', '.', '.', '2', '.', '.', '.', '6', '5', '4', '3', '4', '5', '6',
                            '7', '.', '.', '.', '.', '.', '7', '8', '.', '.', '.', '.', '.', '8',
                            '9', '.', '.', '.', '.', '.', '9',
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                ),
                Solution(
                    Grid2D::try_from_cells_and_dimensions(
                        cells![
                            '.', '.', '9', '0', '.', '.', '9', '.', '.', '.', '1', '.', '9', '8',
                            '.', '.', '.', '2', '.', '.', '7', '6', '5', '4', '3', '4', '5', '6',
                            '7', '6', '5', '.', '9', '8', '7', '8', '7', '6', '.', '.', '.', '.',
                            '9', '8', '7', '.', '.', '.', '.',
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                ),
                Solution(
                    Grid2D::try_from_cells_and_dimensions(
                        cells![
                            '1', '0', '.', '.', '9', '.', '.', '2', '.', '.', '.', '8', '.', '.',
                            '3', '.', '.', '.', '7', '.', '.', '4', '5', '6', '7', '6', '5', '4',
                            '.', '.', '.', '8', '.', '.', '3', '.', '.', '.', '9', '.', '.', '2',
                            '.', '.', '.', '.', '.', '0', '1',
                        ],
                        7_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                ),
                Solution(
                    Grid2D::try_from_cells_and_dimensions(
                        cells![
                            '8', '9', '0', '1', '0', '1', '2', '3', '7', '8', '1', '2', '1', '8',
                            '7', '4', '8', '7', '4', '3', '0', '9', '6', '5', '9', '6', '5', '4',
                            '9', '8', '7', '4', '4', '5', '6', '7', '8', '9', '0', '3', '3', '2',
                            '0', '1', '9', '0', '1', '2', '0', '1', '3', '2', '9', '8', '0', '1',
                            '1', '0', '4', '5', '6', '7', '3', '2',
                        ],
                        8_i32 * IVec2::ONE,
                    )
                    .unwrap(),
                ),
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
    fn test_iter_trailheads_scores() {
        for (index, trailheads) in [
            vec![Trailhead {
                pos: (0_i32, 0_i32).into(),
                score: 1_u32,
                rating: 0_u32,
            }],
            vec![Trailhead {
                pos: (3_i32, 0_i32).into(),
                score: 2_u32,
                rating: 0_u32,
            }],
            vec![Trailhead {
                pos: (3_i32, 0_i32).into(),
                score: 4_u32,
                rating: 0_u32,
            }],
            vec![
                Trailhead {
                    pos: (1_i32, 0_i32).into(),
                    score: 1_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (5_i32, 6_i32).into(),
                    score: 2_u32,
                    rating: 0_u32,
                },
            ],
            vec![
                Trailhead {
                    pos: (2_i32, 0_i32).into(),
                    score: 5_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (4_i32, 0_i32).into(),
                    score: 6_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (4_i32, 2_i32).into(),
                    score: 5_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (6_i32, 4_i32).into(),
                    score: 3_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (2_i32, 5_i32).into(),
                    score: 1_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (5_i32, 5_i32).into(),
                    score: 3_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (0_i32, 6_i32).into(),
                    score: 5_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (6_i32, 6_i32).into(),
                    score: 3_u32,
                    rating: 0_u32,
                },
                Trailhead {
                    pos: (1_i32, 7_i32).into(),
                    score: 5_u32,
                    rating: 0_u32,
                },
            ],
        ]
        .into_iter()
        .enumerate()
        {
            for (actual_trailhead, expected_trailhead) in
                solution(index).iter_trailheads().zip(trailheads)
            {
                assert_eq!(actual_trailhead.pos, expected_trailhead.pos);
                assert_eq!(actual_trailhead.score, expected_trailhead.score);
            }
        }
    }

    #[test]
    fn test_trailhead_score_sum() {
        for (index, trailhead_score_sum) in
            [1_u32, 2_u32, 4_u32, 3_u32, 36_u32].into_iter().enumerate()
        {
            assert_eq!(solution(index).trailhead_score_sum(), trailhead_score_sum);
        }
    }

    #[test]
    fn test_trailhead_rating_sum() {
        for (index, trailhead_rating_sum) in [None, None, None, None, Some(81_u32)]
            .into_iter()
            .enumerate()
        {
            if let Some(trailhead_rating_sum) = trailhead_rating_sum {
                assert_eq!(solution(index).trailhead_rating_sum(), trailhead_rating_sum);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
