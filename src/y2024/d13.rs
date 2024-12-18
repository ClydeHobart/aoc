use {
    crate::*,
    glam::I64Vec2,
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        multi::separated_list0, sequence::tuple, Err, IResult,
    },
};

/* --- Day 13: Claw Contraption ---

Next up: the lobby of a resort on a tropical island. The Historians take a moment to admire the hexagonal floor tiles before spreading out.

Fortunately, it looks like the resort has a new arcade! Maybe you can win some prizes from the claw machines?

The claw machines here are a little unusual. Instead of a joystick or directional buttons to control the claw, these machines have two buttons labeled A and B. Worse, you can't just put in a token and play; it costs 3 tokens to push the A button and 1 token to push the B button.

With a little experimentation, you figure out that each machine's buttons are configured to move the claw a specific amount to the right (along the X axis) and a specific amount forward (along the Y axis) each time that button is pressed.

Each machine contains one prize; to win the prize, the claw must be positioned exactly above the prize on both the X and Y axes.

You wonder: what is the smallest number of tokens you would have to spend to win as many prizes as possible? You assemble a list of every machine's button behavior and prize location (your puzzle input). For example:

Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279

This list describes the button configuration and prize location of four different claw machines.

For now, consider just the first claw machine in the list:

    Pushing the machine's A button would move the claw 94 units along the X axis and 34 units along the Y axis.
    Pushing the B button would move the claw 22 units along the X axis and 67 units along the Y axis.
    The prize is located at X=8400, Y=5400; this means that from the claw's initial position, it would need to move exactly 8400 units along the X axis and exactly 5400 units along the Y axis to be perfectly aligned with the prize in this machine.

The cheapest way to win the prize is by pushing the A button 80 times and the B button 40 times. This would line up the claw along the X axis (because 80*94 + 40*22 = 8400) and along the Y axis (because 80*34 + 40*67 = 5400). Doing this would cost 80*3 tokens for the A presses and 40*1 for the B presses, a total of 280 tokens.

For the second and fourth claw machines, there is no combination of A and B presses that will ever win a prize.

For the third claw machine, the cheapest way to win the prize is by pushing the A button 38 times and the B button 86 times. Doing this would cost a total of 200 tokens.

So, the most prizes you could possibly win is two; the minimum tokens you would have to spend to win all (two) prizes is 480.

You estimate that each button would need to be pressed no more than 100 times to win a prize. How else would someone be expected to play?

Figure out how to win as many prizes as possible. What is the fewest tokens you would have to spend to win all possible prizes?

--- Part Two ---

As you go to win the first prize, you discover that the claw is nowhere near where you expected it would be. Due to a unit conversion error in your measurements, the position of every prize is actually 10000000000000 higher on both the X and Y axis!

Add 10000000000000 to the X and Y position of every prize. After making this change, the example above would now look like this:

Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=10000000008400, Y=10000000005400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=10000000012748, Y=10000000012176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=10000000007870, Y=10000000006450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=10000000018641, Y=10000000010279

Now, it is only possible to win a prize on the second and fourth claw machines. Unfortunately, it will take many more than 100 presses to do so.

Using the corrected prize coordinates, figure out how to win as many prizes as possible. What is the fewest tokens you would have to spend to win all possible prizes? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ClawMachine {
    button_a: I64Vec2,
    button_b: I64Vec2,
    prize: I64Vec2,
}

impl ClawMachine {
    const BUTTON_A_TOKEN_COST: i64 = 3_i64;
    const BUTTON_B_TOKEN_COST: i64 = 1_i64;

    fn parse_button<'i>(name: &'static str) -> impl FnMut(&'i str) -> IResult<&'i str, I64Vec2> {
        map(
            tuple((
                tag("Button "),
                tag(name),
                tag(": X"),
                parse_integer,
                tag(", Y"),
                parse_integer,
            )),
            |(_, _, _, x, _, y)| I64Vec2 { x, y },
        )
    }

    fn try_compute_a_and_b(&self, prize_offset: I64Vec2) -> Option<(i64, i64)> {
        let I64Vec2 { x: x_a, y: y_a } = self.button_a;
        let I64Vec2 { x: x_b, y: y_b } = self.button_b;
        let I64Vec2 { x: x_p, y: y_p } = self.prize + prize_offset;

        // a = x_p * (x_a * y_b - x_b * y_a) - x_b * (x_a * y_p - x_p * y_a)
        //     -------------------------------------------------------------
        //     x_a * (x_a * y_b - x_b * y_a)
        //
        // b = x_a * y_p - x_p * y_a
        //     -----------------------
        //     x_a * y_b - x_b * y_a
        let numerator_b: i64 = x_a * y_p - x_p * y_a;
        let denominator_b: i64 = x_a * y_b - x_b * y_a;
        let numerator_a: i64 = x_p * denominator_b - x_b * (x_a * y_p - x_p * y_a);
        let denominator_a: i64 = x_a * denominator_b;

        (denominator_a != 0_i64
            && (numerator_a % denominator_a == 0_i64)
            && (numerator_b % denominator_b == 0_i64))
            .then(|| (numerator_a / denominator_a, numerator_b / denominator_b))
            .filter(|&(a, b)| a >= 0_i64 && b >= 0_i64)
    }

    fn try_compute_token_cost(&self, prize_offset: I64Vec2) -> Option<i64> {
        self.try_compute_a_and_b(prize_offset)
            .map(|(a, b)| a * Self::BUTTON_A_TOKEN_COST + b * Self::BUTTON_B_TOKEN_COST)
    }
}

impl Parse for ClawMachine {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                Self::parse_button("A"),
                line_ending,
                Self::parse_button("B"),
                line_ending,
                tag("Prize: X="),
                parse_integer,
                tag(", Y="),
                parse_integer,
            )),
            |(button_a, _, button_b, _, _, x, _, y)| Self {
                button_a,
                button_b,
                prize: I64Vec2 { x, y },
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<ClawMachine>);

impl Solution {
    /// Ten trilly!
    const PRIZE_OFFSET: I64Vec2 = I64Vec2 {
        x: 10_000_000_000_000_i64,
        y: 10_000_000_000_000_i64,
    };

    fn total_token_cost(&self, prize_offset: I64Vec2) -> i64 {
        self.0
            .iter()
            .filter_map(|claw_machine| claw_machine.try_compute_token_cost(prize_offset))
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_list0(tuple((line_ending, line_ending)), ClawMachine::parse),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Luckily I had to solve some systems of equations at work this week lol.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_token_cost(I64Vec2::ZERO));
    }

    /// cheeeeese
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_token_cost(Self::PRIZE_OFFSET));
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
        Button A: X+94, Y+34\n\
        Button B: X+22, Y+67\n\
        Prize: X=8400, Y=5400\n\
        \n\
        Button A: X+26, Y+66\n\
        Button B: X+67, Y+21\n\
        Prize: X=12748, Y=12176\n\
        \n\
        Button A: X+17, Y+86\n\
        Button B: X+84, Y+37\n\
        Prize: X=7870, Y=6450\n\
        \n\
        Button A: X+69, Y+23\n\
        Button B: X+27, Y+71\n\
        Prize: X=18641, Y=10279\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                ClawMachine {
                    button_a: (94_i64, 34_i64).into(),
                    button_b: (22_i64, 67_i64).into(),
                    prize: (8400, 5400).into(),
                },
                ClawMachine {
                    button_a: (26_i64, 66_i64).into(),
                    button_b: (67_i64, 21_i64).into(),
                    prize: (12748, 12176).into(),
                },
                ClawMachine {
                    button_a: (17_i64, 86_i64).into(),
                    button_b: (84_i64, 37_i64).into(),
                    prize: (7870, 6450).into(),
                },
                ClawMachine {
                    button_a: (69_i64, 23_i64).into(),
                    button_b: (27_i64, 71_i64).into(),
                    prize: (18641, 10279).into(),
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
    fn test_try_compute_a_and_b() {
        for (index, as_and_bs) in [vec![
            Some((80_i64, 40_i64)),
            None,
            Some((38_i64, 86_i64)),
            None,
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .0
                    .iter()
                    .map(|claw_machine| claw_machine.try_compute_a_and_b(I64Vec2::ZERO))
                    .collect::<Vec<Option<(i64, i64)>>>(),
                as_and_bs
            );
        }
    }

    #[test]
    fn test_try_compute_token_cost() {
        for (index, token_costs) in [vec![Some(280_i64), None, Some(200_i64), None]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index)
                    .0
                    .iter()
                    .map(|claw_machine| claw_machine.try_compute_token_cost(I64Vec2::ZERO))
                    .collect::<Vec<Option<i64>>>(),
                token_costs
            );
        }
    }

    #[test]
    fn test_total_token_cost() {
        for (index, total_token_cost) in [480_i64].into_iter().enumerate() {
            assert_eq!(
                solution(index).total_token_cost(I64Vec2::ZERO),
                total_token_cost
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
