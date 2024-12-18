use {
    crate::*,
    nom::{
        bytes::complete::tag,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::terminated,
        Err, IResult,
    },
    num::Integer,
    std::{collections::HashMap, mem::swap, ops::Range},
};

/* --- Day 11: Plutonian Pebbles ---

The ancient civilization on Pluto was known for its ability to manipulate spacetime, and while The Historians explore their infinite corridors, you've noticed a strange set of physics-defying stones.

At first glance, they seem like normal stones: they're arranged in a perfectly straight line, and each stone has a number engraved on it.

The strange part is that every time you blink, the stones change.

Sometimes, the number engraved on a stone changes. Other times, a stone might split in two, causing all the other stones to shift over a bit to make room in their perfectly straight line.

As you observe them for a while, you find that the stones have a consistent behavior. Every time you blink, the stones each simultaneously change according to the first applicable rule in this list:

    If the stone is engraved with the number 0, it is replaced by a stone engraved with the number 1.
    If the stone is engraved with a number that has an even number of digits, it is replaced by two stones. The left half of the digits are engraved on the new left stone, and the right half of the digits are engraved on the new right stone. (The new numbers don't keep extra leading zeroes: 1000 would become stones 10 and 0.)
    If none of the other rules apply, the stone is replaced by a new stone; the old stone's number multiplied by 2024 is engraved on the new stone.

No matter how the stones change, their order is preserved, and they stay on their perfectly straight line.

How will the stones evolve if you keep blinking at them? You take a note of the number engraved on each stone in the line (your puzzle input).

If you have an arrangement of five stones engraved with the numbers 0 1 10 99 999 and you blink once, the stones transform as follows:

    The first stone, 0, becomes a stone marked 1.
    The second stone, 1, is multiplied by 2024 to become 2024.
    The third stone, 10, is split into a stone marked 1 followed by a stone marked 0.
    The fourth stone, 99, is split into two stones marked 9.
    The fifth stone, 999, is replaced by a stone marked 2021976.

So, after blinking once, your five stones would become an arrangement of seven stones engraved with the numbers 1 2024 1 0 9 9 2021976.

Here is a longer example:

Initial arrangement:
125 17

After 1 blink:
253000 1 7

After 2 blinks:
253 0 2024 14168

After 3 blinks:
512072 1 20 24 28676032

After 4 blinks:
512 72 2024 2 0 2 4 2867 6032

After 5 blinks:
1036288 7 2 20 24 4048 1 4048 8096 28 67 60 32

After 6 blinks:
2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2

In this example, after blinking six times, you would have 22 stones. After blinking 25 times, you would have 55312 stones!

Consider the arrangement of stones in front of you. How many stones will you have after blinking 25 times?

--- Part Two ---

The Historians sure are taking a long time. To be fair, the infinite corridors are very large.

How many stones would you have after blinking a total of 75 times? */

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Stone(u64);

impl Stone {
    fn blink(&mut self) -> Option<Self> {
        if self.0 == 0_u64 {
            self.0 = 1_u64;

            None
        } else {
            let digits: u32 = if self.0 == 0_u64 {
                1_u32
            } else {
                self.0.ilog10() + 1_u32
            };

            if digits.is_even() {
                let power_of_ten: u64 = 10_u64.pow(digits / 2_u32);
                let split_stone: Self = Self(self.0 % power_of_ten);

                self.0 /= power_of_ten;

                Some(split_stone)
            } else {
                self.0 *= 2024_u64;

                None
            }
        }
    }
}

impl Parse for Stone {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self)(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct StoneCount {
    stone: Stone,
    count: u64,
}

impl StoneCount {
    fn new(stone: Stone) -> Self {
        Self {
            stone,
            count: 1_u64,
        }
    }

    fn add(stone_counts: &mut Vec<Self>, start: usize, stone_count: StoneCount) {
        match stone_counts[start..]
            .binary_search_by_key(&stone_count.stone, |stone_count| stone_count.stone)
        {
            Ok(range_index) => {
                stone_counts[start + range_index].count += stone_count.count;
            }
            Err(range_index) => stone_counts.insert(start + range_index, stone_count),
        }
    }
}

#[derive(Default)]
struct StoneSplitMap {
    stones: LinkedList<Stone, u32>,
    stone_counts: Vec<StoneCount>,
    stone_to_stone_count_range: HashMap<Stone, Range<u32>>,
    blinks: usize,
}

impl StoneSplitMap {
    const MAX_BLINKS: usize = 10_usize;

    fn blink(stones: &mut LinkedList<Stone, u32>) {
        let mut curr_stone_index_option: Option<usize> = stones.get_head();

        while let Some(curr_stone_index) = curr_stone_index_option {
            if let Some(split_stone) = stones.get_mut(curr_stone_index).unwrap().blink() {
                let split_stone_index: usize = stones.insert(
                    split_stone,
                    stones.get_next(curr_stone_index).unwrap_or(stones.len()),
                );

                curr_stone_index_option = stones.get_next(split_stone_index);
            } else {
                curr_stone_index_option = stones.get_next(curr_stone_index);
            }
        }
    }

    fn get_or_add_stone_counts(&mut self, stone: Stone) -> &[StoneCount] {
        let stone_count_range: Range<u32> = self
            .stone_to_stone_count_range
            .get(&stone)
            .cloned()
            .unwrap_or_else(|| {
                self.stones.clear();
                self.stones.push_back(stone);

                for _ in 0_usize..self.blinks {
                    Self::blink(&mut self.stones);
                }

                let stone_count_range_start: usize = self.stone_counts.len();

                for stone in self.stones.iter_data().copied() {
                    StoneCount::add(
                        &mut self.stone_counts,
                        stone_count_range_start,
                        StoneCount::new(stone),
                    );
                }

                let stone_count_range_end: usize = self.stone_counts.len();
                let stone_count_range: Range<u32> =
                    stone_count_range_start as u32..stone_count_range_end as u32;

                self.stone_to_stone_count_range
                    .insert(stone, stone_count_range.clone());

                stone_count_range
            });

        &self.stone_counts[stone_count_range.as_range_usize()]
    }

    fn run_blinks(
        &mut self,
        curr_stone_counts: &mut Vec<StoneCount>,
        next_stone_counts: &mut Vec<StoneCount>,
    ) {
        next_stone_counts.clear();

        for curr_stone_count in curr_stone_counts.iter() {
            for split_stone_count in self.get_or_add_stone_counts(curr_stone_count.stone) {
                StoneCount::add(
                    next_stone_counts,
                    0_usize,
                    StoneCount {
                        stone: split_stone_count.stone,
                        count: curr_stone_count.count * split_stone_count.count,
                    },
                );
            }
        }

        swap(curr_stone_counts, next_stone_counts);
    }

    fn clear(&mut self) {
        self.stones.clear();
        self.stone_counts.clear();
        self.stone_to_stone_count_range.clear();
    }

    fn set_blinks(&mut self, blinks: usize) {
        self.clear();
        self.blinks = blinks;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(Vec<Stone>);

impl Solution {
    const Q1_BLINKS: usize = 25_usize;
    const Q2_BLINKS: usize = 75_usize;

    fn stone_count_after_blinks(&self, mut blinks: usize) -> usize {
        let mut curr_stone_counts: Vec<StoneCount> = Vec::new();
        let mut next_stone_counts: Vec<StoneCount> = Vec::new();
        let mut stone_split_map: StoneSplitMap = StoneSplitMap::default();

        stone_split_map.set_blinks(StoneSplitMap::MAX_BLINKS);

        for stone in self.0.iter().copied() {
            StoneCount::add(&mut curr_stone_counts, 0_usize, StoneCount::new(stone));
        }

        while blinks >= StoneSplitMap::MAX_BLINKS {
            stone_split_map.run_blinks(&mut curr_stone_counts, &mut next_stone_counts);
            blinks -= StoneSplitMap::MAX_BLINKS;
        }

        stone_split_map.set_blinks(blinks);
        stone_split_map.run_blinks(&mut curr_stone_counts, &mut next_stone_counts);

        curr_stone_counts
            .into_iter()
            .map(|stone_count| stone_count.count)
            .sum::<u64>() as usize
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Stone::parse, opt(tag(" ")))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Should've made the linked list class sooner, I feel like I've stood this up 5 times the past
    /// 3 weeks.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.stone_count_after_blinks(Self::Q1_BLINKS));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.stone_count_after_blinks(Self::Q2_BLINKS));
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

    const SOLUTION_STRS: &'static [&'static str] = &["0 1 10 99 999", "125 17"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    Stone(0_u64),
                    Stone(1_u64),
                    Stone(10_u64),
                    Stone(99_u64),
                    Stone(999_u64),
                ]),
                Solution(vec![Stone(125_u64), Stone(17_u64)]),
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
    fn test_blink() {
        for (index, blink_stones) in [
            vec![vec![
                Stone(1_u64),
                Stone(2024_u64),
                Stone(1_u64),
                Stone(0_u64),
                Stone(9_u64),
                Stone(9_u64),
                Stone(2021976_u64),
            ]],
            vec![
                vec![Stone(253000_u64), Stone(1_u64), Stone(7_u64)],
                vec![
                    Stone(253_u64),
                    Stone(0_u64),
                    Stone(2024_u64),
                    Stone(14168_u64),
                ],
                vec![
                    Stone(512072_u64),
                    Stone(1_u64),
                    Stone(20_u64),
                    Stone(24_u64),
                    Stone(28676032_u64),
                ],
                vec![
                    Stone(512_u64),
                    Stone(72_u64),
                    Stone(2024_u64),
                    Stone(2_u64),
                    Stone(0_u64),
                    Stone(2_u64),
                    Stone(4_u64),
                    Stone(2867_u64),
                    Stone(6032_u64),
                ],
                vec![
                    Stone(1036288_u64),
                    Stone(7_u64),
                    Stone(2_u64),
                    Stone(20_u64),
                    Stone(24_u64),
                    Stone(4048_u64),
                    Stone(1_u64),
                    Stone(4048_u64),
                    Stone(8096_u64),
                    Stone(28_u64),
                    Stone(67_u64),
                    Stone(60_u64),
                    Stone(32_u64),
                ],
                vec![
                    Stone(2097446912_u64),
                    Stone(14168_u64),
                    Stone(4048_u64),
                    Stone(2_u64),
                    Stone(0_u64),
                    Stone(2_u64),
                    Stone(4_u64),
                    Stone(40_u64),
                    Stone(48_u64),
                    Stone(2024_u64),
                    Stone(40_u64),
                    Stone(48_u64),
                    Stone(80_u64),
                    Stone(96_u64),
                    Stone(2_u64),
                    Stone(8_u64),
                    Stone(6_u64),
                    Stone(7_u64),
                    Stone(6_u64),
                    Stone(0_u64),
                    Stone(3_u64),
                    Stone(2_u64),
                ],
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let mut stones: LinkedList<Stone, u32> = solution(index).0.clone().into();

            for blink_stones in blink_stones {
                StoneSplitMap::blink(&mut stones);

                assert_eq!(
                    stones.iter_data().copied().collect::<Vec<Stone>>(),
                    blink_stones
                );
            }
        }
    }

    #[test]
    fn test_stone_count_after_blinks() {
        for (index, blinks_and_stone_counts) in [
            vec![(1_usize, 7_usize)],
            vec![(6_usize, 22_usize), (Solution::Q1_BLINKS, 55312_usize)],
        ]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (blinks, stone_count) in blinks_and_stone_counts {
                assert_eq!(solution.stone_count_after_blinks(blinks), stone_count);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
