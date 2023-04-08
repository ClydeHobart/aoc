use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::digit1,
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::terminated,
        Err,
    },
    std::str::FromStr,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<u16>);

impl Solution {
    fn linear_cost(pos_a: u16, pos_b: u16) -> u32 {
        pos_a.abs_diff(pos_b) as u32
    }

    fn triangular_cost(pos_a: u16, pos_b: u16) -> u32 {
        let linear_cost: u32 = Self::linear_cost(pos_a, pos_b);

        linear_cost * (linear_cost + 1_u32) / 2_u32
    }

    fn total_cost<F: Fn(u16, u16) -> u32>(&self, align_pos: u16, cost: F) -> u32 {
        self.0.iter().map(|pos| cost(*pos, align_pos)).sum()
    }

    fn total_linear_cost(&self, align_pos: u16) -> u32 {
        self.total_cost(align_pos, Self::linear_cost)
    }

    fn total_triangular_cost(&self, align_pos: u16) -> u32 {
        self.total_cost(align_pos, Self::triangular_cost)
    }

    fn total_costs<F: Fn(&Solution, u16) -> u32 + 'static>(
        &self,
        cost: F,
    ) -> impl Iterator<Item = u32> + '_ {
        (0_u16..self.0.last().map(|pos| *pos + 1_u16).unwrap_or_default())
            .map(move |align_pos| cost(self, align_pos))
    }

    fn total_linear_costs(&self) -> impl Iterator<Item = u32> + '_ {
        self.total_costs(Self::total_linear_cost)
    }

    fn total_triangular_costs(&self) -> impl Iterator<Item = u32> + '_ {
        self.total_costs(Self::total_triangular_cost)
    }

    fn minimum_total_linear_cost(&self) -> u32 {
        self.total_linear_costs().min().unwrap_or_default()
    }

    fn minimum_total_triangular_cost(&self) -> u32 {
        self.total_triangular_costs().min().unwrap_or_default()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.minimum_total_linear_cost());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.minimum_total_triangular_cost());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        let mut iter = iterator(
            input,
            terminated(map_res(digit1, u16::from_str), opt(tag(","))),
        );

        let mut positions: Vec<u16> = iter.collect();

        iter.finish()?;
        positions.sort();

        Ok(Self(positions))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const POSITIONS_STR: &str = "16,1,2,0,4,2,7,1,2,14";

    lazy_static! {
        static ref SOLUTION: Solution = new_solution();
    }

    fn new_solution() -> Solution {
        Solution(vec![0, 1, 1, 2, 2, 2, 4, 7, 14, 16])
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(POSITIONS_STR), Ok(new_solution()));
    }

    mod linear {
        use super::*;

        #[test]
        fn test_total_linear_cost() {
            for (pos, cost) in [
                (2_u16, 37_u32),
                (1_u16, 41_u32),
                (3_u16, 39_u32),
                (10_u16, 71_u32),
            ] {
                assert_eq!(SOLUTION.total_linear_cost(pos), cost);
            }
        }

        #[test]
        fn test_minimum_total_linear_cost() {
            assert_eq!(SOLUTION.minimum_total_linear_cost(), 37_u32);
        }

        #[test]
        fn test_total_linear_costs() {
            assert_eq!(
                SOLUTION.total_linear_costs().collect::<Vec<u32>>(),
                vec![49, 41, 37, 39, 41, 45, 49, 53, 59, 65, 71, 77, 83, 89, 95, 103, 111]
            );
        }
    }

    mod triangular {
        use super::*;

        #[test]
        fn test_total_triangular_cost() {
            for (pos, cost) in [(5_u16, 168_u32), (2_u16, 206_u32)] {
                assert_eq!(SOLUTION.total_triangular_cost(pos), cost);
            }
        }

        #[test]
        fn test_minimum_total_triangular_cost() {
            assert_eq!(SOLUTION.minimum_total_triangular_cost(), 168_u32);
        }

        #[test]
        fn test_total_triangular_costs() {
            assert_eq!(
                SOLUTION.total_triangular_costs().collect::<Vec<u32>>(),
                vec![
                    290, 242, 206, 183, 170, 168, 176, 194, 223, 262, 311, 370, 439, 518, 607, 707,
                    817,
                ]
            );
        }
    }
}
