use {
    crate::{
        y2016::d12::{Instruction, Solution as D12Solution, Value},
        *,
    },
    nom::{combinator::map, error::Error, Err, IResult},
};

/* --- Day 25: Clock Signal ---

You open the door and find yourself on the roof. The city sprawls away from you for miles and miles.

There's not much time now - it's already Christmas, but you're nowhere near the North Pole, much too far to deliver these stars to the sleigh in time.

However, maybe the huge antenna up here can offer a solution. After all, the sleigh doesn't need the stars, exactly; it needs the timing data they provide, and you happen to have a massive signal generator right here.

You connect the stars you have to your prototype computer, connect that to the antenna, and begin the transmission.

Nothing happens.

You call the service number printed on the side of the antenna and quickly explain the situation. "I'm not sure what kind of equipment you have connected over there," he says, "but you need a clock signal." You try to explain that this is a signal for a clock.

"No, no, a clock signal - timing information so the antenna computer knows how to read the data you're sending it. An endless, alternating pattern of 0, 1, 0, 1, 0, 1, 0, 1, 0, 1...." He trails off.

You ask if the antenna can handle a clock signal at the frequency you would need to use for the data from the stars. "There's no way it can! The only antenna we've installed capable of that is on top of a top-secret Easter Bunny installation, and you're definitely not-" You hang up the phone.

You've extracted the antenna's clock signal generation assembunny code (your puzzle input); it looks mostly compatible with code you worked on just recently.

This antenna code, being a signal generator, uses one extra instruction:

    out x transmits x (either an integer or the value of a register) as the next value for the clock signal.

The code takes a value (via register a) that describes the signal to generate, but you're not sure how it's used. You'll have to find the input to produce the right signal through experimentation.

What is the lowest positive integer that can be used to initialize register a and cause the code to output a clock signal of 0, 1, 0, 1... repeating forever? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    const M_INSTRUCTION_INDEX: usize = 1_usize;
    const N_INSTRUCTION_INDEX: usize = 2_usize;

    /// The smallest A value that yields the desired clock signal can be computed by the
    /// following procedure:
    /// * Let B be the number of bits in M * N.
    /// * Round B up to the nearest even number.
    /// * Let S be the number represented by the bit sequence "10" repeated B/2 times.
    /// * If S < M * N, append "10" to S
    /// * Let A be S - M * N
    fn compute_a_for_m_and_n(m: i32, n: i32) -> i32 {
        let product: i32 = m * n;

        // The number of bits in the product of m and n, rounded up to the nearest multiple of 2.
        let bits: u32 = product.ilog2() + 2_u32 / 2_u32 * 2_u32;
        let s: i32 = 0x2AAAAAAA_i32 & ((1_i32 << bits) - 1_i32);

        (if s < product { (s << 2_i32) | 2_i32 } else { s }) - product
    }

    /// Similar to Day 23, without including my user input, it's hard to explain this one
    /// thoroughly. Essentially, the assembunny code takes A + M * N, where A is the value initially
    /// in the A register, and M and N are two constants (presumably specific to the user), and
    /// emits the bits of this number in LSB order, until the MSB 1 has been emitted, then it
    /// repeats. All other constants present are relevant to the logic necessary to perform this
    /// bit streaming procedure. In order to preserve the privacy of the user-specific input, this
    /// function doesn't verify the structure actually performs this procedure. Instead, it just
    /// attempts to find the M and N constants, and returns what A would be if the rest of the
    /// structure were to match.
    ///
    /// The smallest A value that yields the desired clock signal can be computed by the
    /// following procedure:
    /// * Let B be the number of bits in M * N.
    /// * Round B up to the nearest even number.
    /// * Let S be the number represented by the bit sequence "10" repeated B/2 times.
    /// * Let A be S - M * N
    fn a_for_clock_signal(&self) -> Option<i32> {
        self.0
            .get(Self::M_INSTRUCTION_INDEX)
            .zip(self.0.get(Self::N_INSTRUCTION_INDEX))
            .map(
                |(m_instruction, n_instruciton)| match (m_instruction, n_instruciton) {
                    (
                        Instruction::Cpy {
                            x: Value::Constant(m),
                            y: _,
                        },
                        Instruction::Cpy {
                            x: Value::Constant(n),
                            y: _,
                        },
                    ) => Some(Self::compute_a_for_m_and_n(*m, *n)),
                    _ => None,
                },
            )
            .flatten()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(D12Solution::parse_instructions, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.a_for_clock_signal());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {}
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}
