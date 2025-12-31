use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::map,
        error::Error,
        multi::{many0, separated_list0},
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 5: Cafeteria ---

As the forklifts break through the wall, the Elves are delighted to discover that there was a cafeteria on the other side after all.

You can hear a commotion coming from the kitchen. "At this rate, we won't have any time left to put the wreaths up in the dining hall!" Resolute in your quest, you investigate.

"If only we hadn't switched to the new inventory management system right before Christmas!" another Elf exclaims. You ask what's going on.

The Elves in the kitchen explain the situation: because of their complicated new inventory management system, they can't figure out which of their ingredients are fresh and which are spoiled. When you ask how it works, they give you a copy of their database (your puzzle input).

The database operates on ingredient IDs. It consists of a list of fresh ingredient ID ranges, a blank line, and a list of available ingredient IDs. For example:

3-5
10-14
16-20
12-18

1
5
8
11
17
32

The fresh ID ranges are inclusive: the range 3-5 means that ingredient IDs 3, 4, and 5 are all fresh. The ranges can also overlap; an ingredient ID is fresh if it is in any range.

The Elves are trying to determine which of the available ingredient IDs are fresh. In this example, this is done as follows:

    Ingredient ID 1 is spoiled because it does not fall into any range.
    Ingredient ID 5 is fresh because it falls into range 3-5.
    Ingredient ID 8 is spoiled.
    Ingredient ID 11 is fresh because it falls into range 10-14.
    Ingredient ID 17 is fresh because it falls into range 16-20 as well as range 12-18.
    Ingredient ID 32 is spoiled.

So, in this example, 3 of the available ingredient IDs are fresh.

Process the database file from the new inventory management system. How many of the available ingredient IDs are fresh?

--- Part Two ---

The Elves start bringing their spoiled inventory to the trash chute at the back of the kitchen.

So that they can stop bugging you when they get new inventory, the Elves would like to know all of the IDs that the fresh ingredient ID ranges consider to be fresh. An ingredient ID is still considered fresh if it is in any range.

Now, the second section of the database (the available ingredient IDs) is irrelevant. Here are the fresh ingredient ID ranges from the above example:

3-5
10-14
16-20
12-18

The ingredient IDs that these ranges consider to be fresh are 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and 20. So, in this example, the fresh ingredient ID ranges consider a total of 14 ingredient IDs to be fresh.

Process the database file again. How many ingredient IDs are considered to be fresh according to the fresh ingredient ID ranges? */

type IngredientId = u64;

#[derive(Clone, Copy, PartialEq)]
struct IngredientIdValue {
    is_fresh: bool,
}

impl IngredientIdValue {
    const SPOILED: Self = Self { is_fresh: false };

    const FRESH: Self = Self { is_fresh: true };
}

impl RegionTreeValue for IngredientIdValue {
    fn insert_value_into_leaf_with_matching_range(&mut self, other: &Self) {
        *self = *other;
    }

    fn should_convert_leaf_to_parent(&self, other: &Self) -> bool {
        self.ne(other)
    }

    fn get_leaf<const D: usize, I: RangeIntTrait>(
        &self,
        _range: &RangeD<I, D>,
        _child_range: &RangeD<I, D>,
    ) -> Self {
        *self
    }

    fn try_convert_parent_to_leaf<'a, I>(mut iter: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a,
    {
        iter.try_fold(None, |previous_value, &value| {
            (previous_value.is_none() || previous_value == Some(value)).then_some(Some(value))
        })
        .flatten()
    }
}

type IngredientIdBinaryTree = BinaryTree<IngredientIdValue, IngredientId>;
type IngredientIdRange1 = Range1<IngredientId>;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    fresh_ingredient_id_ranges: Vec<Range<IngredientId>>,
    available_ingredient_ids: Vec<IngredientId>,
}

impl Solution {
    fn parse_ingredient_id<'i>(input: &'i str) -> IResult<&'i str, IngredientId> {
        parse_integer(input)
    }

    fn parse_ingredient_id_range<'i>(input: &'i str) -> IResult<&'i str, Range<IngredientId>> {
        map(
            separated_pair(
                Self::parse_ingredient_id,
                tag("-"),
                Self::parse_ingredient_id,
            ),
            |(start, end_inclusive)| start..end_inclusive + 1 as IngredientId,
        )(input)
    }

    fn build_ingredient_id_binary_tree(&self) -> IngredientIdBinaryTree {
        let mut ingredient_id_binary_tree: IngredientIdBinaryTree = IngredientIdBinaryTree::new(
            IngredientIdRange1::all_regions(),
            IngredientIdValue::SPOILED,
        );

        for fresh_ingredient_id_range in self
            .fresh_ingredient_id_ranges
            .iter()
            .cloned()
            .flat_map(IngredientIdRange1::iter_from_start_and_end_1d)
        {
            ingredient_id_binary_tree.insert(&fresh_ingredient_id_range, &IngredientIdValue::FRESH);
        }

        ingredient_id_binary_tree
    }

    fn is_ingredient_id_fresh(
        ingredient_id_binary_tree: &IngredientIdBinaryTree,
        ingredient_id: IngredientId,
    ) -> bool {
        let queried_ingredient_id_range_1: IngredientIdRange1 =
            ([ingredient_id]..[ingredient_id + 1]).try_into().unwrap();

        let mut intersecting_leaf_value: Option<IngredientIdValue> = None;

        ingredient_id_binary_tree.visit_all_leaves(
            |ingredient_id_range_1| {
                ingredient_id_range_1
                    .try_intersect(&queried_ingredient_id_range_1)
                    .is_some()
            },
            |ingredient_id_range_1, &ingredient_id_value| {
                if ingredient_id_range_1
                    .try_intersect(&queried_ingredient_id_range_1)
                    .is_none()
                {
                    // Keep searching.
                    true
                } else {
                    intersecting_leaf_value = Some(ingredient_id_value);

                    // We found a match.
                    false
                }
            },
        );

        intersecting_leaf_value.map_or(false, |ingredient_id_value| ingredient_id_value.is_fresh)
    }

    fn iter_available_fresh_ingredient_ids(&self) -> impl Iterator<Item = IngredientId> + '_ {
        let ingredient_id_binary_tree: IngredientIdBinaryTree =
            self.build_ingredient_id_binary_tree();

        self.available_ingredient_ids
            .iter()
            .filter(move |&&available_ingredient_id| {
                Self::is_ingredient_id_fresh(&ingredient_id_binary_tree, available_ingredient_id)
            })
            .copied()
    }

    fn count_available_fresh_ingredient_ids(&self) -> usize {
        self.iter_available_fresh_ingredient_ids().count()
    }

    fn count_fresh_ingredient_ids(&self) -> usize {
        const SHOULD_KEEP_GOING: bool = true;

        let ingredient_id_binary_tree: IngredientIdBinaryTree =
            self.build_ingredient_id_binary_tree();

        let mut fresh_ingredient_id_count: usize = 0_usize;

        ingredient_id_binary_tree.visit_all_leaves(
            |_| SHOULD_KEEP_GOING,
            |leaf_ingredient_id_range_1, ingredient_id_range_value| {
                if ingredient_id_range_value.is_fresh {
                    fresh_ingredient_id_count += leaf_ingredient_id_range_1.len_product();
                }

                SHOULD_KEEP_GOING
            },
        );

        fresh_ingredient_id_count
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                many0(terminated(Self::parse_ingredient_id_range, line_ending)),
                line_ending,
                separated_list0(line_ending, parse_integer),
            ),
            |(fresh_ingredient_id_ranges, available_ingredient_ids)| Self {
                fresh_ingredient_id_ranges,
                available_ingredient_ids,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_available_fresh_ingredient_ids());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_fresh_ingredient_ids());
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
        3-5\n\
        10-14\n\
        16-20\n\
        12-18\n\
        \n\
        1\n\
        5\n\
        8\n\
        11\n\
        17\n\
        32\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                fresh_ingredient_id_ranges: vec![
                    3..(5 + 1),
                    10..(14 + 1),
                    16..(20 + 1),
                    12..(18 + 1),
                ],
                available_ingredient_ids: vec![1, 5, 8, 11, 17, 32],
            }]
        })[index]
    }

    fn ingredient_id_binary_tree(index: usize) -> &'static IngredientIdBinaryTree {
        static ONCE_LOCK: OnceLock<Vec<IngredientIdBinaryTree>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            (0_usize..SOLUTION_STRS.len())
                .map(|index| solution(index).build_ingredient_id_binary_tree())
                .collect()
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
    fn test_is_ingredient_id_fresh() {
        for (index, is_ingredient_id_fresh_list) in [vec![false, true, false, true, true, false]]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);
            let ingredient_id_binary_tree: &IngredientIdBinaryTree =
                ingredient_id_binary_tree(index);

            for (available_ingredient_id, is_ingredient_id_fresh) in solution
                .available_ingredient_ids
                .iter()
                .copied()
                .zip(is_ingredient_id_fresh_list)
            {
                assert_eq!(
                    Solution::is_ingredient_id_fresh(
                        ingredient_id_binary_tree,
                        available_ingredient_id
                    ),
                    is_ingredient_id_fresh
                );
            }
        }
    }

    #[test]
    fn test_count_fresh_ingredient_ids() {
        for (index, fresh_ingredient_id_count) in [14_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_fresh_ingredient_ids(),
                fresh_ingredient_id_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
