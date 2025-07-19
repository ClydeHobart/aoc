use {
    crate::*,
    nom::{character::complete::line_ending, combinator::map, error::Error, Err, IResult},
};

/* --- Day 17: No Such Thing as Too Much ---

The elves bought too much eggnog again - 150 liters this time. To fit it all into your refrigerator, you'll need to move it into smaller containers. You take an inventory of the capacities of the available containers.

For example, suppose you have containers of size 20, 15, 10, 5, and 5 liters. If you need to store 25 liters, there are four ways to do it:

    15 and 10
    20 and 5 (the first 5)
    20 and 5 (the second 5)
    15, 5, and 5

Filling all containers entirely, how many different combinations of containers can exactly fit all 150 liters of eggnog?

--- Part Two ---

While playing with all the containers in the kitchen, another load of eggnog arrives! The shipping and receiving department is requesting as many containers as you can spare.

Find the minimum number of containers that can exactly fit all 150 liters of eggnog. How many different ways can you fill that number of containers and still hold exactly 150 litres?

In the example above, the minimum number of containers was two. There were three ways to use that many containers, and so the answer there would be 3. */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(SspInputArrayVec);

impl Solution {
    const LITERS_OF_EGGNOG: i32 = 150_i32;

    fn ssp_input(&self, liters_of_eggnog: i32) -> SspInput {
        SspInput {
            values: self.0.clone(),
            target_sum: liters_of_eggnog,
        }
    }

    fn container_combination_count(&self, liters_of_eggnog: i32) -> usize {
        self.ssp_input(liters_of_eggnog).into_iter().count()
    }

    fn min_containers_and_container_combination_count(
        &self,
        liters_of_eggnog: i32,
    ) -> (usize, usize) {
        let ssp_input: SspInput = self.ssp_input(liters_of_eggnog);
        let mut ssp_subset_with_sum_iterator: SspSubsetWithSumIterator = ssp_input.into_iter();

        let min_containers: usize = (&mut ssp_subset_with_sum_iterator)
            .map(|containers| containers.count_ones())
            .min()
            .unwrap_or_default();

        ssp_subset_with_sum_iterator.clique_iterator.reset();

        (
            min_containers,
            ssp_subset_with_sum_iterator
                .filter(|containers| containers.count_ones() == min_containers)
                .count(),
        )
    }

    fn container_combination_count_with_min_containers(&self, liters_of_eggnog: i32) -> usize {
        self.min_containers_and_container_combination_count(liters_of_eggnog)
            .1
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            parse_separated_array_vec(parse_integer, line_ending),
            |mut array_vec| {
                array_vec.sort();

                Self(array_vec)
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Not bad, had to restructure maximal clique into a clique iterator.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.container_combination_count(Self::LITERS_OF_EGGNOG));
    }

    /// Thankful I added a rest to the clique iterator
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.min_containers_and_container_combination_count(Self::LITERS_OF_EGGNOG));
        } else {
            dbg!(self.container_combination_count_with_min_containers(Self::LITERS_OF_EGGNOG));
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

    const SOLUTION_STRS: &'static [&'static str] = &["20\n15\n10\n5\n5\n"];
    const LITERS_OF_EGGNOG: i32 = 25_i32;

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                [5_i32, 5_i32, 10_i32, 15_i32, 20_i32].into_iter().collect(),
            )]
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
    fn test_container_combination_count() {
        for (index, container_combination_count) in [4_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).container_combination_count(LITERS_OF_EGGNOG),
                container_combination_count
            );
        }
    }

    #[test]
    fn test_min_containers_and_container_combination_count() {
        for (index, min_containers_and_container_combination_count) in
            [(2_usize, 3_usize)].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).min_containers_and_container_combination_count(LITERS_OF_EGGNOG),
                min_containers_and_container_combination_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
