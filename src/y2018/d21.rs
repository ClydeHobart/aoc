use {
    crate::{
        y2018::{
            d16::{OpCode, RegisterRaw},
            d19::Program,
        },
        *,
    },
    nom::{combinator::map, error::Error, Err, IResult},
    std::collections::HashSet,
};

/* --- Day 21: Chronal Conversion ---

You should have been watching where you were going, because as you wander the new North Pole base, you trip and fall into a very deep hole!

Just kidding. You're falling through time again.

If you keep up your current pace, you should have resolved all of the temporal anomalies by the next time the device activates. Since you have very little interest in browsing history in 500-year increments for the rest of your life, you need to find a way to get back to your present time.

After a little research, you discover two important facts about the behavior of the device:

First, you discover that the device is hard-wired to always send you back in time in 500-year increments. Changing this is probably not feasible.

Second, you discover the activation system (your puzzle input) for the time travel module. Currently, it appears to run forever without halting.

If you can cause the activation system to halt at a specific moment, maybe you can make the device send you so far back in time that you cause an integer underflow in time itself and wrap around back to your current time!

The device executes the program as specified in manual section one and manual section two.

Your goal is to figure out how the program works and cause it to halt. You can only control register 0; every other register begins at 0 as usual.

Because time travel is a dangerous activity, the activation system begins with a few instructions which verify that bitwise AND (via bani) does a numeric operation and not an operation as if the inputs were interpreted as strings. If the test fails, it enters an infinite loop re-running the test instead of allowing the program to execute normally. If the test passes, the program continues, and assumes that all other bitwise operations (banr, bori, and borr) also interpret their inputs as numbers. (Clearly, the Elves who wrote this system were worried that someone might introduce a bug while trying to emulate this system with a scripting language.)

What is the lowest non-negative integer value for register 0 that causes the program to halt after executing the fewest instructions? (Executing the same instruction multiple times counts as multiple instructions executed.) */

struct Constants {
    in_0_h_eq_a: RegisterRaw,
    in_1_h_eq_h_and_b: RegisterRaw,
    in_2_h_eq_h_eq_b: RegisterRaw,
    in_7_h_eq_a: RegisterRaw,
    in_11_h_eq_h_mul_b: RegisterRaw,
}

impl Constants {
    const IN_6_G_EQ_H_OR_B: RegisterRaw = 0x10000 as RegisterRaw;
    const IN_8_F_EQ_G_AND_B: RegisterRaw = 0xFF as RegisterRaw;
    const IN_10_H_EQ_H_AND_B: RegisterRaw = 0xFFFFFF as RegisterRaw;
    const IN_12_H_EQ_H_AND_B: RegisterRaw = 0xFFFFFF as RegisterRaw;
    const IN_13_F_EQ_A_GT_G: RegisterRaw = 0x100 as RegisterRaw;
    const IN_19_E_EQ_E_MUL_B: RegisterRaw = 0x100 as RegisterRaw;
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Program);

impl Solution {
    fn try_constants(&self) -> Option<Constants> {
        const H: RegisterRaw = 5 as RegisterRaw;

        Some(Constants {
            in_0_h_eq_a: self
                .0
                .try_get_constant(0_usize, OpCode::SetI, None, None, H)?,
            in_1_h_eq_h_and_b: self
                .0
                .try_get_constant(1_usize, OpCode::BAnI, Some(H), None, H)?,
            in_2_h_eq_h_eq_b: self
                .0
                .try_get_constant(2_usize, OpCode::EqRI, Some(H), None, H)?,
            in_7_h_eq_a: self
                .0
                .try_get_constant(7_usize, OpCode::SetI, None, None, H)?,
            in_11_h_eq_h_mul_b: self.0.try_get_constant(
                11_usize,
                OpCode::MulI,
                Some(H),
                None,
                H,
            )?,
        })
    }

    fn map_h(constants: &Constants, h: RegisterRaw) -> RegisterRaw {
        let mut g: RegisterRaw = h | Constants::IN_6_G_EQ_H_OR_B;
        let mut h: RegisterRaw = constants.in_7_h_eq_a;

        while {
            h = (((h + (g & Constants::IN_8_F_EQ_G_AND_B)) & Constants::IN_10_H_EQ_H_AND_B)
                * constants.in_11_h_eq_h_mul_b)
                & Constants::IN_12_H_EQ_H_AND_B;

            Constants::IN_13_F_EQ_A_GT_G <= g
        } {
            g /= Constants::IN_19_E_EQ_E_MUL_B;
        }

        h
    }

    fn filter_constants(constants: &Constants) -> bool {
        (constants.in_0_h_eq_a & constants.in_1_h_eq_h_and_b) == constants.in_2_h_eq_h_eq_b
    }

    fn try_shortest_halt(&self) -> Option<RegisterRaw> {
        self.try_constants()
            .filter(Self::filter_constants)
            .map(|constants| Self::map_h(&constants, 0))
    }

    fn try_longest_halt(&self) -> Option<RegisterRaw> {
        self.try_constants()
            .filter(Self::filter_constants)
            .map(|constants| {
                let mut prev_hs: HashSet<RegisterRaw> = HashSet::new();
                let mut prev_h: Option<RegisterRaw> = None;
                let mut h: RegisterRaw = 0;

                while prev_hs.insert(h) {
                    prev_h = Some(h);
                    h = Self::map_h(&constants, h);
                }

                prev_h.unwrap()
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Program::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Took forever to solve this because I mistranslated 65536 as 0x100 instead of 0x10000. The
    /// logic this runs is effectively
    /// ```notrust
    /// while {
    ///     g = h | Constants::IN_6_G_EQ_H_OR_B;
    ///     h = constants.in_7_h_eq_a;
    ///     while {
    ///         h = (((h + (g & Constants::IN_8_F_EQ_G_AND_B))
    ///             & Constants::IN_10_H_EQ_H_AND_B)
    ///             * constants.in_11_h_eq_h_mul_b)
    ///             & Constants::IN_12_H_EQ_H_AND_B;
    ///         Constants::IN_13_F_EQ_A_GT_G <= g
    ///     } {
    ///         g /= Constants::IN_19_E_EQ_E_MUL_B;
    ///     }
    ///     h != d
    /// } {}
    /// ```
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Err(e) = self.0.try_print_simplified_to_file(module_path!()) {
                eprintln!("{e}");
            }
        }

        dbg!(self.try_shortest_halt());
    }

    /// I think this one's incorrect: in theory, having a register `d` of 0 would take even longer,
    /// since `h` is first compared with `d` after operations are performed on it.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_longest_halt());
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

    const SOLUTION_STRS: &'static [&'static str] = &[""];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![])[index]
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
    fn test_input() {
        let args: Args = Args::parse(module_path!()).unwrap().1;

        Solution::both(&args);
    }
}
