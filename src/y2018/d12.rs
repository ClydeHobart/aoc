use {
    crate::*,
    bitvec::{prelude::*, view::BitView},
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::{many0_count, many_m_n},
        sequence::{terminated, tuple},
        Err, IResult,
    },
    num::{One, Zero},
    std::{collections::HashMap, mem::swap},
};

/* --- Day 12: Subterranean Sustainability ---

The year 518 is significantly more underground than your history books implied. Either that, or you've arrived in a vast cavern network under the North Pole.

After exploring a little, you discover a long tunnel that contains a row of small pots as far as you can see to your left and right. A few of them contain plants - someone is trying to grow things in these geothermally-heated caves.

The pots are numbered, with 0 in front of you. To the left, the pots are numbered -1, -2, -3, and so on; to the right, 1, 2, 3.... Your puzzle input contains a list of pots from 0 to the right and whether they do (#) or do not (.) currently contain a plant, the initial state. (No other pots currently contain plants.) For example, an initial state of #..##.... indicates that pots 0, 3, and 4 currently contain plants.

Your puzzle input also contains some notes you find on a nearby table: someone has been trying to figure out how these plants spread to nearby pots. Based on the notes, for each generation of plants, a given pot has or does not have a plant based on whether that pot (and the two pots on either side of it) had a plant in the last generation. These are written as LLCRR => N, where L are pots to the left, C is the current pot being considered, R are the pots to the right, and N is whether the current pot will have a plant in the next generation. For example:

    A note like ..#.. => . means that a pot that contains a plant but with no plants within two pots of it will not have a plant in it during the next generation.
    A note like ##.## => . means that an empty pot with two plants on each side of it will remain empty in the next generation.
    A note like .##.# => # means that a pot has a plant in a given generation if, in the previous generation, there were plants in that pot, the one immediately to the left, and the one two pots to the right, but not in the ones immediately to the right and two to the left.

It's not clear what these plants are for, but you're sure it's important, so you'd like to make sure the current configuration of plants is sustainable by determining what will happen after 20 generations.

For example, given the following input:

initial state: #..#.#..##......###...###

...## => #
..#.. => #
.#... => #
.#.#. => #
.#.## => #
.##.. => #
.#### => #
#.#.# => #
#.### => #
##.#. => #
##.## => #
###.. => #
###.# => #
####. => #

For brevity, in this example, only the combinations which do produce a plant are listed. (Your input includes all possible combinations.) Then, the next 20 generations will look like this:

                 1         2         3
       0         0         0         0
 0: ...#..#.#..##......###...###...........
 1: ...#...#....#.....#..#..#..#...........
 2: ...##..##...##....#..#..#..##..........
 3: ..#.#...#..#.#....#..#..#...#..........
 4: ...#.#..#...#.#...#..#..##..##.........
 5: ....#...##...#.#..#..#...#...#.........
 6: ....##.#.#....#...#..##..##..##........
 7: ...#..###.#...##..#...#...#...#........
 8: ...#....##.#.#.#..##..##..##..##.......
 9: ...##..#..#####....#...#...#...#.......
10: ..#.#..#...#.##....##..##..##..##......
11: ...#...##...#.#...#.#...#...#...#......
12: ...##.#.#....#.#...#.#..##..##..##.....
13: ..#..###.#....#.#...#....#...#...#.....
14: ..#....##.#....#.#..##...##..##..##....
15: ..##..#..#.#....#....#..#.#...#...#....
16: .#.#..#...#.#...##...#...#.#..##..##...
17: ..#...##...#.#.#.#...##...#....#...#...
18: ..##.#.#....#####.#.#.#...##...##..##..
19: .#..###.#..#.#.#######.#.#.#..#.#...#..
20: .#....##....#####...#######....#.#..##.

The generation is shown along the left, where 0 is the initial state. The pot numbers are shown along the top, where 0 labels the center pot, negative-numbered pots extend to the left, and positive pots extend toward the right. Remember, the initial state begins at pot 0, which is not the leftmost pot used in this example.

After one generation, only seven plants remain. The one in pot 0 matched the rule looking for ..#.., the one in pot 4 matched the rule looking for .#.#., pot 9 matched .##.., and so on.

In this example, after 20 generations, the pots shown as # contain plants, the furthest left of which is pot -2, and the furthest right of which is pot 34. Adding up all the numbers of plant-containing pots after the 20th generation produces 325.

After 20 generations, what is the sum of the numbers of all pots which contain a plant?

--- Part Two ---

You realize that 20 generations aren't enough. After all, these plants will need to last another 1500 years to even reach your timeline, not to mention your future.

After fifty billion (50000000000) generations, what is the sum of the numbers of all pots which contain a plant? */

const PLANT_SPREAD_RULES_INPUT_LEN: usize = 5_usize;
const PLANT_SPREAD_WINDOW_TO_CENTER_OFFSET: usize = PLANT_SPREAD_RULES_INPUT_LEN / 2_usize;

type PlantSpreadRulesBitArray = BitArr!(for 1_usize << PLANT_SPREAD_RULES_INPUT_LEN, in u32);

type PotsWithPlantsStorage = u32;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    generation: usize,
    curr_pots_with_plants: Vec<PotsWithPlantsStorage>,
    next_pots_with_plants: Vec<PotsWithPlantsStorage>,
    pot_index_offset: i64,
    plant_spread_rules: PlantSpreadRulesBitArray,
}

impl Solution {
    const GENERATIONS: usize = 20_usize;
    const MANY_GENERATIONS: usize = 50_000_000_000_usize;

    fn first_plant_spread_bits_have_plant(pots_with_plants_storage: PotsWithPlantsStorage) -> bool {
        pots_with_plants_storage
            & ((PotsWithPlantsStorage::one() << PLANT_SPREAD_RULES_INPUT_LEN)
                - PotsWithPlantsStorage::one())
            != PotsWithPlantsStorage::zero()
    }

    fn last_plant_spread_bits_have_plant(pots_with_plants_storage: PotsWithPlantsStorage) -> bool {
        pots_with_plants_storage
            >> (PotsWithPlantsStorage::BITS as usize - PLANT_SPREAD_RULES_INPUT_LEN)
            != PotsWithPlantsStorage::zero()
    }

    fn iter_pots_with_plants(&self) -> impl Iterator<Item = i64> + '_ {
        self.curr_pots_with_plants
            .view_bits::<Lsb0>()
            .iter_ones()
            .map(|pot_with_plant| pot_with_plant as i64 - self.pot_index_offset)
    }

    fn pots_with_plants_sum_after_generations(&self, generations: usize) -> i64 {
        let mut solution: Self = self.clone();

        for _ in 0_usize..generations {
            solution.next_generation();
        }

        solution.iter_pots_with_plants().sum()
    }

    fn pots_with_plants_hash(&self) -> u64 {
        let pots_with_plants: &BitSlice<PotsWithPlantsStorage> =
            self.curr_pots_with_plants.view_bits();
        let pots_with_plants: &BitSlice<PotsWithPlantsStorage> = if pots_with_plants.any() {
            &pots_with_plants[pots_with_plants.leading_zeros()
                ..pots_with_plants.len() - pots_with_plants.trailing_zeros()]
        } else {
            &pots_with_plants[..0_usize]
        };

        pots_with_plants.compute_hash()
    }

    fn pots_with_plants_sum_after_many_generations(&self, generations: usize) -> i64 {
        let mut solution: Solution = self.clone();
        let mut pots_with_plants_hash: u64 = solution.pots_with_plants_hash();
        let mut pots_with_plants_hash_to_generation: HashMap<u64, usize> = HashMap::new();

        while !pots_with_plants_hash_to_generation.contains_key(&pots_with_plants_hash)
            && solution.generation < generations
        {
            pots_with_plants_hash_to_generation.insert(pots_with_plants_hash, solution.generation);
            solution.next_generation();
            pots_with_plants_hash = solution.pots_with_plants_hash();
        }

        if solution.generation == generations {
            self.iter_pots_with_plants().sum()
        } else {
            let curr_first_pot_with_plant: Option<i64> = solution.iter_pots_with_plants().next();

            if let Some(curr_first_pot_with_plant) = curr_first_pot_with_plant {
                let prev_generation: usize =
                    pots_with_plants_hash_to_generation[&pots_with_plants_hash];
                let curr_generation: usize = solution.generation;
                let generation_period: usize = curr_generation - prev_generation;
                let all_cyclical_generations: usize = generations - prev_generation;
                let cycles: usize = all_cyclical_generations / generation_period;
                let partial_cycle_generations: usize = all_cyclical_generations % generation_period;

                // Re-run up until `prev_generation`
                solution = self.clone();

                while solution.generation < prev_generation {
                    solution.next_generation();
                }

                // Safe to unwrap because it has the same hash as before, which had a first pot with
                // plant
                let prev_first_pot_with_plant: i64 =
                    solution.iter_pots_with_plants().next().unwrap();
                let pot_with_plant_offset_per_cycle: i64 =
                    curr_first_pot_with_plant - prev_first_pot_with_plant;
                let pot_with_plant_offset_after_all_cycles: i64 =
                    pot_with_plant_offset_per_cycle * cycles as i64;

                for _ in 0_usize..partial_cycle_generations {
                    solution.next_generation();
                }

                solution
                    .iter_pots_with_plants()
                    .map(|pot_with_plant| pot_with_plant + pot_with_plant_offset_after_all_cycles)
                    .sum()
            } else {
                // There are no pots with plants. It's safe to assume that this isn't a recoverable
                // state, as that would bust things.
                // In that case, the sum after a super huge number of generations is 0
                0_i64
            }
        }
    }

    fn next_generation(&mut self) {
        self.next_pots_with_plants.clear();
        self.next_pots_with_plants.resize(
            self.curr_pots_with_plants.len(),
            PotsWithPlantsStorage::zero(),
        );

        let next_pots_with_plants_bits: &mut BitSlice<PotsWithPlantsStorage> =
            self.next_pots_with_plants.view_bits_mut();

        for (adjacent_plants_index, adjacent_plants) in self
            .curr_pots_with_plants
            .view_bits::<Lsb0>()
            .windows(PLANT_SPREAD_RULES_INPUT_LEN)
            .enumerate()
        {
            next_pots_with_plants_bits.set(
                adjacent_plants_index + PLANT_SPREAD_WINDOW_TO_CENTER_OFFSET,
                self.plant_spread_rules[adjacent_plants.load::<usize>()],
            );
        }

        if Self::first_plant_spread_bits_have_plant(*self.next_pots_with_plants.first().unwrap()) {
            self.next_pots_with_plants
                .insert(0_usize, PotsWithPlantsStorage::zero());
            self.pot_index_offset += PotsWithPlantsStorage::BITS as i64;
        }

        if Self::last_plant_spread_bits_have_plant(*self.next_pots_with_plants.last().unwrap()) {
            self.next_pots_with_plants
                .push(PotsWithPlantsStorage::zero());
        }

        swap(
            &mut self.curr_pots_with_plants,
            &mut self.next_pots_with_plants,
        );

        self.generation += 1_usize;
    }

    #[cfg(test)]
    fn as_string(&self) -> String {
        self.curr_pots_with_plants
            .view_bits::<Lsb0>()
            .iter()
            .by_vals()
            .map(Pixel::from)
            .map(char::from)
            .collect()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("initial state: "),
                |input: &'i str| {
                    let mut curr_pots_with_plants: Vec<u32> = vec![0_u32, 0_u32];
                    let mut pot_index: usize = u32::BITS as usize;

                    let input: &str = many0_count(map(Pixel::parse, |pixel| {
                        curr_pots_with_plants
                            .view_bits_mut::<Lsb0>()
                            .set(pot_index, pixel.is_light());

                        pot_index += 1_usize;

                        if pot_index % u32::BITS as usize == 0_usize {
                            curr_pots_with_plants.push(0_u32);
                        }
                    }))(input)?
                    .0;

                    if Self::last_plant_spread_bits_have_plant(
                        *curr_pots_with_plants.last().unwrap(),
                    ) {
                        curr_pots_with_plants.push(0_u32);
                    }

                    Ok((
                        input,
                        (0_usize, curr_pots_with_plants, Vec::new(), u32::BITS as i64),
                    ))
                },
                line_ending,
                line_ending,
                |input: &'i str| {
                    let mut plant_spread_rules: PlantSpreadRulesBitArray =
                        PlantSpreadRulesBitArray::ZERO;

                    let input: &str = many0_count(terminated(
                        |input: &'i str| {
                            let mut rule_index: usize = 0_usize;
                            let mut rule_bit_mask: usize = 1_usize;

                            let input: &str = tuple((
                                many_m_n(
                                    PLANT_SPREAD_RULES_INPUT_LEN,
                                    PLANT_SPREAD_RULES_INPUT_LEN,
                                    map(Pixel::parse, |pixel| {
                                        rule_index |= rule_bit_mask * pixel.is_light() as usize;
                                        rule_bit_mask <<= 1_u32;
                                    }),
                                ),
                                tag(" => "),
                            ))(input)?
                            .0;

                            let input: &str = map(Pixel::parse, |pixel| {
                                plant_spread_rules.set(rule_index, pixel.is_light());
                            })(input)?
                            .0;

                            Ok((input, ()))
                        },
                        opt(line_ending),
                    ))(input)?
                    .0;

                    Ok((input, plant_spread_rules))
                },
            )),
            |(
                _,
                (generation, curr_pots_with_plants, next_pots_with_plants, pot_index_offset),
                _,
                _,
                plant_spread_rules,
            )| Self {
                generation,
                curr_pots_with_plants,
                next_pots_with_plants,
                pot_index_offset,
                plant_spread_rules,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Guessing q2 is going to be "okay, now 2000 generations".
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.pots_with_plants_sum_after_generations(Self::GENERATIONS));
    }

    /// a bit more than 2000, took a while to sort out what I would and what I wouldn't need to
    /// track to detect a cycle.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.pots_with_plants_sum_after_many_generations(Self::MANY_GENERATIONS));
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        initial state: #..#.#..##......###...###\n\
        \n\
        ...## => #\n\
        ..#.. => #\n\
        .#... => #\n\
        .#.#. => #\n\
        .#.## => #\n\
        .##.. => #\n\
        .#### => #\n\
        #.#.# => #\n\
        #.### => #\n\
        ##.#. => #\n\
        ##.## => #\n\
        ###.. => #\n\
        ###.# => #\n\
        ####. => #\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                generation: 0_usize,
                curr_pots_with_plants: vec![0_u32, 0b_1110001110000001100101001_u32],
                next_pots_with_plants: Vec::new(),
                pot_index_offset: 32_i64,
                plant_spread_rules: bitarr![u32, Lsb0;
                    0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                    1, 1, 0, 1, 1, 0
                ],
            }]
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
    fn test_iter_pots_with_plants() {
        for (index, pots_with_plants) in [vec![
            0_i64, 3_i64, 5_i64, 8_i64, 9_i64, 16_i64, 17_i64, 18_i64, 22_i64, 23_i64, 24_i64,
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_pots_with_plants()
                    .collect::<Vec<i64>>(),
                pots_with_plants
            );
        }
    }

    #[test]
    fn test_next_generation() {
        for (index, next_generations) in [vec![
            bitvec![
                0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
                0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
            ],
            bitvec![
                0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            ],
            bitvec![
                0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
            ],
            bitvec![
                0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
            ],
            bitvec![
                0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
            ],
        ]]
        .into_iter()
        .enumerate()
        {
            let mut solution: Solution = solution(index).clone();

            for next_generation in next_generations {
                assert_eq!(
                    solution.iter_pots_with_plants().collect::<Vec<i64>>(),
                    next_generation
                        .iter_ones()
                        .map(|index| index as i64 - 3_i64)
                        .collect::<Vec<i64>>()
                );

                solution.next_generation();
            }
        }

        let mut solution: Solution = solution(0_usize).clone();

        for _ in 0_usize..200_usize {
            println!("{}", solution.as_string());

            solution.next_generation();
        }
    }

    #[test]
    fn test_pots_with_plants_sum_after_generations() {
        for (index, pots_with_plants_sum_after_generations) in [325_i64].into_iter().enumerate() {
            assert_eq!(
                solution(index).pots_with_plants_sum_after_generations(Solution::GENERATIONS),
                pots_with_plants_sum_after_generations
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
