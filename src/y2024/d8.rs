use {
    crate::*,
    glam::IVec2,
    nom::{
        character::complete::one_of,
        combinator::{map, success, verify},
        error::Error,
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 8: Resonant Collinearity ---

You find yourselves on the roof of a top-secret Easter Bunny installation.

While The Historians do their thing, you take a look at the familiar huge antenna. Much to your surprise, it seems to have been reconfigured to emit a signal that makes people 0.1% more likely to buy Easter Bunny brand Imitation Mediocre Chocolate as a Christmas gift! Unthinkable!

Scanning across the city, you find that there are actually many such antennas. Each antenna is tuned to a specific frequency indicated by a single lowercase letter, uppercase letter, or digit. You create a map (your puzzle input) of these antennas. For example:

............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............

The signal only applies its nefarious effect at specific antinodes based on the resonant frequencies of the antennas. In particular, an antinode occurs at any point that is perfectly in line with two antennas of the same frequency - but only when one of the antennas is twice as far away as the other. This means that for any pair of antennas with the same frequency, there are two antinodes, one on either side of them.

So, for these two antennas with frequency a, they create the two antinodes marked with #:

..........
...#......
..........
....a.....
..........
.....a....
..........
......#...
..........
..........

Adding a third antenna with the same frequency creates several more antinodes. It would ideally add four antinodes, but two are off the right side of the map, so instead it adds only two:

..........
...#......
#.........
....a.....
........a.
.....a....
..#.......
......#...
..........
..........

Antennas with different frequencies don't create antinodes; A and a count as different frequencies. However, antinodes can occur at locations that contain antennas. In this diagram, the lone antenna with frequency capital A creates no antinodes but has a lowercase-a-frequency antinode at its location:

..........
...#......
#.........
....a.....
........a.
.....a....
..#.......
......A...
..........
..........

The first example has antennas with two different frequencies, so the antinodes they create look like this, plus an antinode overlapping the topmost A-frequency antenna:

......#....#
...#....0...
....#0....#.
..#....0....
....0....#..
.#....A.....
...#........
#......#....
........A...
.........A..
..........#.
..........#.

Because the topmost A-frequency antenna overlaps with a 0-frequency antinode, there are 14 total unique locations that contain an antinode within the bounds of the map.

Calculate the impact of the signal. How many unique locations within the bounds of the map contain an antinode?

--- Part Two ---

Watching over your shoulder as you work, one of The Historians asks if you took the effects of resonant harmonics into your calculations.

Whoops!

After updating your model, it turns out that an antinode occurs at any grid position exactly in line with at least two antennas of the same frequency, regardless of distance. This means that some of the new antinodes will occur at the position of each antenna (unless that antenna is the only one of its frequency).

So, these three T-frequency antennas now create many antinodes:

T....#....
...T......
.T....#...
.........#
..#.......
..........
...#......
..........
....#.....
..........

In fact, the three T-frequency antennas are all exactly in line with two antennas, so they are all also antinodes! This brings the total number of antinodes in the above example to 9.

The original example now has 34 antinodes, including the antinodes that appear on every antenna:

##....#....#
.#.#....0...
..#.#0....#.
..##...0....
....0....#..
.#...#A....#
...#..#.....
#....#.#....
..#.....A...
....#....A..
.#........#.
...#......##

Calculate the impact of the signal using this updated model. How many unique locations within the bounds of the map contain an antinode? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Cell(u8);

impl Cell {
    const VALID_CHARS: &'static str = "\
        .\
        0123456789\
        ABCDEFGHIJKLMNOPQRSTUVWXYZ\
        abcdefghijklmnopqrstuvwxyz";
    const DEFAULT: Self = Self(Self::VALID_CHARS.as_bytes()[0_usize]);

    fn is_empty(self) -> bool {
        self.0 == b'.'
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// SAFETY: See `Parse` implementation.
unsafe impl IsValidAscii for Cell {}

impl Parse for Cell {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            map(map(one_of(Self::VALID_CHARS), u8::try_from), Result::unwrap),
            Self,
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct FrequencyGroup {
    frequency: Cell,
    antenna_range: Range<u8>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    dimensions: IVec2,
    antennae: Vec<IVec2>,
    frequency_groups: Vec<FrequencyGroup>,
}

impl Solution {
    fn contains(&self, pos: IVec2) -> bool {
        grid_2d_contains(pos, self.dimensions)
    }

    fn try_insert_antinode(&self, antinode: IVec2, antinodes: &mut Vec<IVec2>) -> bool {
        self.contains(antinode)
            && if let Err(antinode_index) = antinodes
                .binary_search_by_key(&sortable_index_from_pos(antinode), |antinode| {
                    sortable_index_from_pos(*antinode)
                })
            {
                antinodes.insert(antinode_index, antinode);

                true
            } else {
                false
            }
    }

    fn compute_antinodes(&self, complex: bool) -> Vec<IVec2> {
        let mut antinodes: Vec<IVec2> = Vec::new();

        for frequency_group in &self.frequency_groups {
            for antenna_index_a in (frequency_group.antenna_range.start
                ..frequency_group.antenna_range.end - 1_u8)
                .as_range_usize()
            {
                let antenna_a: IVec2 = self.antennae[antenna_index_a];

                for antenna_index_b in
                    antenna_index_a + 1_usize..frequency_group.antenna_range.end as usize
                {
                    let antenna_b: IVec2 = self.antennae[antenna_index_b];
                    let delta: IVec2 = antenna_b - antenna_a;

                    if complex {
                        let delta: IVec2 = delta
                            / extended_euclidean_algorithm(delta.x as i64, delta.y as i64).gcd
                                as i32;
                        let mut antinode: IVec2 = antenna_a;

                        while self.contains(antinode) {
                            self.try_insert_antinode(antinode, &mut antinodes);
                            antinode -= delta;
                        }

                        antinode = antenna_a + delta;

                        while self.contains(antinode) {
                            self.try_insert_antinode(antinode, &mut antinodes);
                            antinode += delta;
                        }
                    } else {
                        self.try_insert_antinode(antenna_a - delta, &mut antinodes);
                        self.try_insert_antinode(antenna_b + delta, &mut antinodes);

                        if delta % 3_i32 == IVec2::ZERO {
                            let one_third_delta: IVec2 = delta / 3_i32;

                            self.try_insert_antinode(antenna_a + one_third_delta, &mut antinodes);
                            self.try_insert_antinode(antenna_b - one_third_delta, &mut antinodes);
                        }
                    }
                }
            }
        }

        antinodes
    }

    fn count_antinodes(&self, complex: bool) -> usize {
        self.compute_antinodes(complex).len()
    }

    fn insert_antenna(&mut self, antenna: IVec2, frequency: Cell) {
        let frequency_group_index: usize = match self
            .frequency_groups
            .binary_search_by_key(&frequency.0, |frequency_group| frequency_group.frequency.0)
        {
            Ok(frequency_group_index) => {
                let antenna_range: &mut Range<u8> =
                    &mut self.frequency_groups[frequency_group_index].antenna_range;

                self.antennae.insert(antenna_range.end as usize, antenna);

                antenna_range.end += 1_u8;

                frequency_group_index
            }
            Err(frequency_group_index) => {
                let antenna_range_start: u8 = self
                    .frequency_groups
                    .get(frequency_group_index)
                    .map_or_else(
                        || self.antennae.len() as u8,
                        |frequency_group| frequency_group.antenna_range.start,
                    );

                self.antennae.insert(antenna_range_start as usize, antenna);
                self.frequency_groups.insert(
                    frequency_group_index,
                    FrequencyGroup {
                        frequency,
                        antenna_range: antenna_range_start..antenna_range_start + 1_u8,
                    },
                );

                frequency_group_index
            }
        };

        for next_frequency_group in
            self.frequency_groups[frequency_group_index + 1_usize..].iter_mut()
        {
            next_frequency_group.antenna_range.start += 1_u8;
            next_frequency_group.antenna_range.end += 1_u8;
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, grid): (&str, Grid2D<Cell>) = Grid2D::parse(input)?;

        let dimensions: IVec2 = grid.dimensions();

        let mut solution: Self = Self {
            dimensions,
            antennae: Vec::new(),
            frequency_groups: Vec::new(),
        };

        for (index, cell) in grid.cells().iter().enumerate() {
            if !cell.is_empty() {
                verify(success(()), |_| {
                    u8::try_from(solution.antennae.len()).is_ok()
                })(input)?;

                solution.insert_antenna(grid.pos_from_index(index), *cell);
            }
        }

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// Apparently in my user case checking for antinodes between antennae didn't make a difference:
    /// no pairs of antennae were located at multiples of 3 deltas from one another.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_antinodes(false));
    }

    /// Maybe `compute_antinodes` should be returning a hash set instead?
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_antinodes(true));
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
        ............\n\
        ........0...\n\
        .....0......\n\
        .......0....\n\
        ....0.......\n\
        ......A.....\n\
        ............\n\
        ............\n\
        ........A...\n\
        .........A..\n\
        ............\n\
        ............\n",
        "\
        ..........\n\
        ..........\n\
        ..........\n\
        ....a.....\n\
        ..........\n\
        .....a....\n\
        ..........\n\
        ..........\n\
        ..........\n\
        ..........\n",
        "\
        ..........\n\
        ..........\n\
        ..........\n\
        ....a.....\n\
        ........a.\n\
        .....a....\n\
        ..........\n\
        ..........\n\
        ..........\n\
        ..........\n",
        "\
        ..........\n\
        ..........\n\
        ..........\n\
        ....a.....\n\
        ........a.\n\
        .....a....\n\
        ..........\n\
        ......A...\n\
        ..........\n\
        ..........\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    dimensions: 12_i32 * IVec2::ONE,
                    antennae: vec![
                        (8_i32, 1_i32).into(),
                        (5_i32, 2_i32).into(),
                        (7_i32, 3_i32).into(),
                        (4_i32, 4_i32).into(),
                        (6_i32, 5_i32).into(),
                        (8_i32, 8_i32).into(),
                        (9_i32, 9_i32).into(),
                    ],
                    frequency_groups: vec![
                        FrequencyGroup {
                            frequency: Cell(b'0'),
                            antenna_range: 0_u8..4_u8,
                        },
                        FrequencyGroup {
                            frequency: Cell(b'A'),
                            antenna_range: 4_u8..7_u8,
                        },
                    ],
                },
                Solution {
                    dimensions: 10_i32 * IVec2::ONE,
                    antennae: vec![(4_i32, 3_i32).into(), (5_i32, 5_i32).into()],
                    frequency_groups: vec![FrequencyGroup {
                        frequency: Cell(b'a'),
                        antenna_range: 0_u8..2_u8,
                    }],
                },
                Solution {
                    dimensions: 10_i32 * IVec2::ONE,
                    antennae: vec![
                        (4_i32, 3_i32).into(),
                        (8_i32, 4_i32).into(),
                        (5_i32, 5_i32).into(),
                    ],
                    frequency_groups: vec![FrequencyGroup {
                        frequency: Cell(b'a'),
                        antenna_range: 0_u8..3_u8,
                    }],
                },
                Solution {
                    dimensions: 10_i32 * IVec2::ONE,
                    antennae: vec![
                        (6_i32, 7_i32).into(),
                        (4_i32, 3_i32).into(),
                        (8_i32, 4_i32).into(),
                        (5_i32, 5_i32).into(),
                    ],
                    frequency_groups: vec![
                        FrequencyGroup {
                            frequency: Cell(b'A'),
                            antenna_range: 0_u8..1_u8,
                        },
                        FrequencyGroup {
                            frequency: Cell(b'a'),
                            antenna_range: 1_u8..4_u8,
                        },
                    ],
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
    fn test_compute_simple_antinodes() {
        for (index, antinodes) in [
            vec![
                IVec2::new(6_i32, 0_i32),
                (11_i32, 0_i32).into(),
                (3_i32, 1_i32).into(),
                (4_i32, 2_i32).into(),
                (10_i32, 2_i32).into(),
                (2_i32, 3_i32).into(),
                (9_i32, 4_i32).into(),
                (1_i32, 5_i32).into(),
                (6_i32, 5_i32).into(),
                (3_i32, 6_i32).into(),
                (0_i32, 7_i32).into(),
                (7_i32, 7_i32).into(),
                (10_i32, 10_i32).into(),
                (10_i32, 11_i32).into(),
            ],
            vec![(3_i32, 1_i32).into(), (6_i32, 7_i32).into()],
            vec![
                (3_i32, 1_i32).into(),
                (0_i32, 2_i32).into(),
                (2_i32, 6_i32).into(),
                (6_i32, 7_i32).into(),
            ],
            vec![
                (3_i32, 1_i32).into(),
                (0_i32, 2_i32).into(),
                (2_i32, 6_i32).into(),
                (6_i32, 7_i32).into(),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).compute_antinodes(false), antinodes);
        }
    }

    #[test]
    fn test_count_complex_antinodes() {
        for (solution_str, antinode_str) in [
            (
                "\
                T.........\n\
                ...T......\n\
                .T........\n\
                ..........\n\
                ..........\n\
                ..........\n\
                ..........\n\
                ..........\n\
                ..........\n\
                ..........\n",
                "\
                T....#....\n\
                ...T......\n\
                .T....#...\n\
                .........#\n\
                ..#.......\n\
                ..........\n\
                ...#......\n\
                ..........\n\
                ....#.....\n\
                ..........\n",
            ),
            (
                "\
                ............\n\
                ........0...\n\
                .....0......\n\
                .......0....\n\
                ....0.......\n\
                ......A.....\n\
                ............\n\
                ............\n\
                ........A...\n\
                .........A..\n\
                ............\n\
                ............\n",
                "\
                ##....#....#\n\
                .#.#....0...\n\
                ..#.#0....#.\n\
                ..##...0....\n\
                ....0....#..\n\
                .#...#A....#\n\
                ...#..#.....\n\
                #....#.#....\n\
                ..#.....A...\n\
                ....#....A..\n\
                .#........#.\n\
                ...#......##\n",
            ),
        ] {
            let solution: Solution = solution_str.try_into().unwrap();

            assert_eq!(
                solution.count_antinodes(true),
                antinode_str
                    .as_bytes()
                    .iter()
                    .filter(|b| !matches!(*b, b'\n' | b'.'))
                    .count()
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
