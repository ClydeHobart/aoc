use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{map_res, opt},
        error::Error,
        Err,
    },
    std::str::FromStr,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution;

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        todo!();
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        todo!();
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const SOLUTION_STR: &str = "";

    lazy_static! {
        static ref SOLUTION: Solution = solution();
    }

    fn solution() -> Solution {
        todo!();
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR), Ok(solution()))
    }
}
