use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{iterator, map, map_res, opt},
        error::Error,
        sequence::{preceded, terminated},
        Err,
    },
    std::str::FromStr,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Command {
    Forward(i32),
    Down(i32),
    Up(i32),
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Command>);

impl Solution {
    fn compute_h_pos_and_depth(&self) -> (i32, i32) {
        self.0
            .iter()
            .copied()
            .fold((0_i32, 0_i32), |(h_pos, depth), command| match command {
                Command::Forward(x) => (h_pos + x, depth),
                Command::Down(x) => (h_pos, depth + x),
                Command::Up(x) => (h_pos, depth - x),
            })
    }

    fn compute_h_pos_and_depth_with_aim(&self) -> (i32, i32) {
        let mut aim: i32 = 0_i32;

        self.0
            .iter()
            .copied()
            .fold((0_i32, 0_i32), |(h_pos, depth), command| match command {
                Command::Forward(x) => (h_pos + x, depth + aim * x),
                Command::Down(x) => {
                    aim += x;

                    (h_pos, depth)
                }
                Command::Up(x) => {
                    aim -= x;

                    (h_pos, depth)
                }
            })
    }

    fn compute_product(&self) -> i32 {
        let (h_pos, depth): (i32, i32) = self.compute_h_pos_and_depth();

        h_pos * depth
    }

    fn compute_product_with_aim(&self) -> i32 {
        let (h_pos, depth): (i32, i32) = self.compute_h_pos_and_depth_with_aim();

        h_pos * depth
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_product());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_product_with_aim());
    }
}

impl<'a> TryFrom<&'a str> for Solution {
    type Error = Err<Error<&'a str>>;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut iter = iterator(
            value,
            terminated(
                alt((
                    preceded(
                        tag("forward "),
                        map(map_res(digit1, i32::from_str), Command::Forward),
                    ),
                    preceded(
                        tag("down "),
                        map(map_res(digit1, i32::from_str), Command::Down),
                    ),
                    preceded(tag("up "), map(map_res(digit1, i32::from_str), Command::Up)),
                )),
                opt(line_ending),
            ),
        );

        let result: Result<Self, Self::Error> = Ok(Self(iter.collect()));

        iter.finish()?;

        result
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const COMMANDS_STR: &str = "forward 5\ndown 5\nforward 8\nup 3\ndown 8\nforward 2\n";

    lazy_static! {
        static ref SOLUTION: Solution = new_solutions();
    }

    fn new_solutions() -> Solution {
        use Command::*;

        Solution(vec![
            Forward(5),
            Down(5),
            Forward(8),
            Up(3),
            Down(8),
            Forward(2),
        ])
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(COMMANDS_STR), Ok(new_solutions()));
    }

    #[test]
    fn test_compute_h_pos_and_depth() {
        assert_eq!(SOLUTION.compute_h_pos_and_depth(), (15, 10));
    }

    #[test]
    fn test_compute_product() {
        assert_eq!(SOLUTION.compute_product(), 150);
    }

    #[test]
    fn test_compute_h_pos_and_depth_with_aim() {
        assert_eq!(SOLUTION.compute_h_pos_and_depth_with_aim(), (15, 60));
    }

    #[test]
    fn test_compute_product_with_aim() {
        assert_eq!(SOLUTION.compute_product_with_aim(), 900);
    }
}
