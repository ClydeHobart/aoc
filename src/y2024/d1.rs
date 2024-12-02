use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
};

/* --- Day 1: Historian Hysteria ---

The Chief Historian is always present for the big Christmas sleigh launch, but nobody has seen him in months! Last anyone heard, he was visiting locations that are historically significant to the North Pole; a group of Senior Historians has asked you to accompany them as they check the places they think he was most likely to visit.

As each location is checked, they will mark it on their list with a star. They figure the Chief Historian must be in one of the first fifty places they'll look, so in order to save Christmas, you need to help them get fifty stars on their list before Santa takes off on December 25th.

Collect stars by solving puzzles. Two puzzles will be made available on each day in the Advent calendar; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

You haven't even left yet and the group of Elvish Senior Historians has already hit a problem: their list of locations to check is currently empty. Eventually, someone decides that the best place to check first would be the Chief Historian's office.

Upon pouring into the office, everyone confirms that the Chief Historian is indeed nowhere to be found. Instead, the Elves discover an assortment of notes and lists of historically significant locations! This seems to be the planning the Chief Historian was doing before he left. Perhaps these notes can be used to determine which locations to search?

Throughout the Chief's office, the historically significant locations are listed not by name but by a unique number called the location ID. To make sure they don't miss anything, The Historians split into two groups, each searching the office and trying to create their own complete list of location IDs.

There's just one problem: by holding the two lists up side by side (your puzzle input), it quickly becomes clear that the lists aren't very similar. Maybe you can help The Historians reconcile their lists?

For example:

3   4
4   3
2   5
1   3
3   9
3   3

Maybe the lists are only off by a small amount! To find out, pair up the numbers and measure how far apart they are. Pair up the smallest number in the left list with the smallest number in the right list, then the second-smallest left number with the second-smallest right number, and so on.

Within each pair, figure out how far apart the two numbers are; you'll need to add up all of those distances. For example, if you pair up a 3 from the left list with a 7 from the right list, the distance apart is 4; if you pair up a 9 with a 3, the distance apart is 6.

In the example list above, the pairs and distances would be as follows:

    The smallest number in the left list is 1, and the smallest number in the right list is 3. The distance between them is 2.
    The second-smallest number in the left list is 2, and the second-smallest number in the right list is another 3. The distance between them is 1.
    The third-smallest number in both lists is 3, so the distance between them is 0.
    The next numbers to pair up are 3 and 4, a distance of 1.
    The fifth-smallest numbers in each list are 3 and 5, a distance of 2.
    Finally, the largest number in the left list is 4, while the largest number in the right list is 9; these are a distance 5 apart.

To find the total distance between the left list and the right list, add up the distances between all of the pairs you found. In the example above, this is 2 + 1 + 0 + 1 + 2 + 5, a total distance of 11!

Your actual left and right lists contain many location IDs. What is the total distance between your lists?

--- Part Two ---

Your analysis only confirmed what everyone feared: the two lists of location IDs are indeed very different.

Or are they?

The Historians can't agree on which group made the mistakes or how to read most of the Chief's handwriting, but in the commotion you notice an interesting detail: a lot of location IDs appear in both lists! Maybe the other numbers aren't location IDs at all but rather misinterpreted handwriting.

This time, you'll need to figure out exactly how often each number from the left list appears in the right list. Calculate a total similarity score by adding up each number in the left list after multiplying it by the number of times that number appears in the right list.

Here are the same example lists again:

3   4
4   3
2   5
1   3
3   9
3   3

For these example lists, here is the process of finding the similarity score:

    The first number in the left list is 3. It appears in the right list three times, so the similarity score increases by 3 * 3 = 9.
    The second number in the left list is 4. It appears in the right list once, so the similarity score increases by 4 * 1 = 4.
    The third number in the left list is 2. It does not appear in the right list, so the similarity score does not increase (2 * 0 = 0).
    The fourth number, 1, also does not appear in the right list.
    The fifth number, 3, appears in the right list three times; the similarity score increases by 9.
    The last number, 3, appears in the right list three times; the similarity score again increases by 9.

So, for these example lists, the similarity score at the end of this process is 31 (9 + 4 + 0 + 0 + 9 + 9).

Once again consider your left and right lists. What is their similarity score? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ListElement {
    left: i32,
    right: i32,
}

impl Parse for ListElement {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(parse_integer, tag("   "), parse_integer),
            |(left, right)| Self { left, right },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<ListElement>);

impl Solution {
    fn lists(&self) -> (Vec<i32>, Vec<i32>) {
        let left: Vec<i32> = self
            .0
            .iter()
            .map(|list_element| list_element.left)
            .collect();
        let right: Vec<i32> = self
            .0
            .iter()
            .map(|list_element| list_element.right)
            .collect();

        (left, right)
    }

    fn sorted_lists(&self) -> (Vec<i32>, Vec<i32>) {
        let (mut left, mut right): (Vec<i32>, Vec<i32>) = self.lists();

        left.sort();
        right.sort();

        (left, right)
    }

    fn iter_sorted_pairwise_dists(&self) -> impl Iterator<Item = i32> {
        let (left, right): (Vec<i32>, Vec<i32>) = self.sorted_lists();

        left.into_iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
    }

    fn sorted_pairwise_dist_sum(&self) -> i32 {
        self.iter_sorted_pairwise_dists().sum()
    }

    fn count_occurrences_in_sorted_list(sorted_list: &[i32], target: i32) -> i32 {
        let index_ge_target: usize =
            sorted_list.partition_point(|&sorted_element| sorted_element < target);

        sorted_list
            .get(index_ge_target)
            .and_then(|&sorted_element_ge_target| {
                (sorted_element_ge_target == target).then(|| {
                    sorted_list[index_ge_target..]
                        .partition_point(|&sorted_element| sorted_element == target)
                        as i32
                })
            })
            .unwrap_or_default()
    }

    fn iter_sorted_similarity_scores(&self) -> impl Iterator<Item = i32> {
        let (left, right): (Vec<i32>, Vec<i32>) = self.sorted_lists();

        let mut prev_left_and_similarity_score: Option<(i32, i32)> = None;

        left.into_iter().map(move |curr_left| {
            let curr_similarity_score: i32 = prev_left_and_similarity_score
                .and_then(|(prev_left, prev_similarity_score)| {
                    (curr_left == prev_left).then_some(prev_similarity_score)
                })
                .unwrap_or_else(|| {
                    if curr_left == 0_i32 {
                        0_i32
                    } else {
                        curr_left * Self::count_occurrences_in_sorted_list(&right, curr_left)
                    }
                });

            prev_left_and_similarity_score = Some((curr_left, curr_similarity_score));

            curr_similarity_score
        })
    }

    fn sorted_similarity_score_sum(&self) -> i32 {
        self.iter_sorted_similarity_scores().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(ListElement::parse, opt(line_ending))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sorted_pairwise_dist_sum());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sorted_similarity_score_sum());
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
        3   4\n\
        4   3\n\
        2   5\n\
        1   3\n\
        3   9\n\
        3   3\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                ListElement {
                    left: 3_i32,
                    right: 4_i32,
                },
                ListElement {
                    left: 4_i32,
                    right: 3_i32,
                },
                ListElement {
                    left: 2_i32,
                    right: 5_i32,
                },
                ListElement {
                    left: 1_i32,
                    right: 3_i32,
                },
                ListElement {
                    left: 3_i32,
                    right: 9_i32,
                },
                ListElement {
                    left: 3_i32,
                    right: 3_i32,
                },
            ])]
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
    fn test_iter_sorted_pairwise_dists() {
        for (index, sorted_pairwise_dists) in [vec![2_i32, 1_i32, 0_i32, 1_i32, 2_i32, 5_i32]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_sorted_pairwise_dists()
                    .collect::<Vec<i32>>(),
                sorted_pairwise_dists
            );
        }
    }

    #[test]
    fn test_sorted_pairwise_dist_sum() {
        for (index, sorted_pairwise_dist_sum) in [11_i32].into_iter().enumerate() {
            assert_eq!(
                solution(index).sorted_pairwise_dist_sum(),
                sorted_pairwise_dist_sum
            );
        }
    }

    #[test]
    fn test_iter_sorted_similarity_scores() {
        for (index, sorted_similarity_scores) in [vec![0_i32, 0_i32, 9_i32, 9_i32, 9_i32, 4_i32]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_sorted_similarity_scores()
                    .collect::<Vec<i32>>(),
                sorted_similarity_scores
            );
        }
    }

    #[test]
    fn test_sorted_similarity_score_sum() {
        for (index, sorted_similarity_score_sum) in [31_i32].into_iter().enumerate() {
            assert_eq!(
                solution(index).sorted_similarity_score_sum(),
                sorted_similarity_score_sum
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
