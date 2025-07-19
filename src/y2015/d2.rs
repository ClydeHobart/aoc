use {
    crate::*,
    glam::{UVec3, Vec3Swizzles},
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        multi::separated_list0, Err, IResult,
    },
};

/* --- Day 2: I Was Told There Would Be No Math ---

The elves are running low on wrapping paper, and so they need to submit an order for more. They have a list of the dimensions (length l, width w, and height h) of each present, and only want to order exactly as much as they need.

Fortunately, every present is a box (a perfect right rectangular prism), which makes calculating the required wrapping paper for each gift a little easier: find the surface area of the box, which is 2*l*w + 2*w*h + 2*h*l. The elves also need a little extra paper for each present: the area of the smallest side.

For example:

    A present with dimensions 2x3x4 requires 2*6 + 2*12 + 2*8 = 52 square feet of wrapping paper plus 6 square feet of slack, for a total of 58 square feet.
    A present with dimensions 1x1x10 requires 2*1 + 2*10 + 2*10 = 42 square feet of wrapping paper plus 1 square foot of slack, for a total of 43 square feet.

All numbers in the elves' list are in feet. How many total square feet of wrapping paper should they order?

--- Part Two ---

The elves are also running low on ribbon. Ribbon is all the same width, so they only have to worry about the length they need to order, which they would again like to be exact.

The ribbon required to wrap a present is the shortest distance around its sides, or the smallest perimeter of any one face. Each present also requires a bow made out of ribbon as well; the feet of ribbon required for the perfect bow is equal to the cubic feet of volume of the present. Don't ask how they tie the bow, though; they'll never tell.

For example:

    A present with dimensions 2x3x4 requires 2+2+3+3 = 10 feet of ribbon to wrap the present plus 2*3*4 = 24 feet of ribbon for the bow, for a total of 34 feet.
    A present with dimensions 1x1x10 requires 1+1+1+1 = 4 feet of ribbon to wrap the present plus 1*1*10 = 10 feet of ribbon for the bow, for a total of 14 feet.

How many total feet of ribbon should they order? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<UVec3>);

impl Solution {
    fn surface_area_and_slack(dimensions: UVec3) -> (u32, u32) {
        let side_areas: UVec3 = dimensions * dimensions.yzx();

        (2_u32 * side_areas.element_sum(), side_areas.min_element())
    }

    fn smallest_perimeter_and_volume(dimensions: UVec3) -> (u32, u32) {
        (
            (2_u32 * (dimensions + dimensions.yzx())).min_element(),
            dimensions.element_product(),
        )
    }

    fn total_wrapping_paper_area(&self) -> u32 {
        self.0
            .iter()
            .copied()
            .map(Self::surface_area_and_slack)
            .map(|(surface_area, slack)| surface_area + slack)
            .sum()
    }

    fn total_ribbon_length(&self) -> u32 {
        self.0
            .iter()
            .copied()
            .map(Self::smallest_perimeter_and_volume)
            .map(|(smallest_perimeter, volume)| smallest_perimeter + volume)
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_list0(
                line_ending,
                map(parse_separated_array(parse_integer, tag("x")), UVec3::from),
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// cheese
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_wrapping_paper_area());
    }

    /// cheese
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_ribbon_length());
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

    const SOLUTION_STRS: &'static [&'static str] = &["2x3x4", "1x1x10"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![(2_u32, 3_u32, 4_u32).into()]),
                Solution(vec![(1_u32, 1_u32, 10_u32).into()]),
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
    fn test_surface_area_and_slack() {
        for (index, surface_areas_and_slacks) in [vec![(52_u32, 6_u32)], vec![(42_u32, 1_u32)]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index)
                    .0
                    .iter()
                    .copied()
                    .map(Solution::surface_area_and_slack)
                    .collect::<Vec<(u32, u32)>>(),
                surface_areas_and_slacks
            );
        }
    }

    #[test]
    fn test_smallest_perimeter_and_volume() {
        for (index, smallest_perimeters_and_volumes) in
            [vec![(10_u32, 24_u32)], vec![(4_u32, 10_u32)]]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index)
                    .0
                    .iter()
                    .copied()
                    .map(Solution::smallest_perimeter_and_volume)
                    .collect::<Vec<(u32, u32)>>(),
                smallest_perimeters_and_volumes
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
