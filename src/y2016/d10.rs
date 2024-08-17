use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{iterator, map, opt},
        error::Error,
        sequence::{preceded, separated_pair, terminated, tuple},
        Err, IResult,
    },
    num::Integer,
    std::str::FromStr,
};

#[cfg(test)]
use std::fmt::{Debug, Formatter, Result as FmtResult};

/* --- Day 10: Balance Bots ---

You come upon a factory in which many robots are zooming around handing small microchips to each other.

Upon closer examination, you notice that each bot only proceeds when it has two microchips, and once it does, it gives each one to a different bot or puts it in a marked "output" bin. Sometimes, bots take microchips from "input" bins, too.

Inspecting one of the microchips, it seems like they each contain a single number; the bots must use some logic to decide what to do with each chip. You access the local control computer and download the bots' instructions (your puzzle input).

Some of the instructions specify that a specific-valued microchip should be given to a specific bot; the rest of the instructions indicate what a given bot should do with its lower-value or higher-value chip.

For example, consider the following instructions:

value 5 goes to bot 2
bot 2 gives low to bot 1 and high to bot 0
value 3 goes to bot 1
bot 1 gives low to output 1 and high to bot 0
bot 0 gives low to output 2 and high to output 0
value 2 goes to bot 2

    Initially, bot 1 starts with a value-3 chip, and bot 2 starts with a value-2 chip and a value-5 chip.
    Because bot 2 has two microchips, it gives its lower one (2) to bot 1 and its higher one (5) to bot 0.
    Then, bot 1 has two microchips; it puts the value-2 chip in output 1 and gives the value-3 chip to bot 0.
    Finally, bot 0 has two microchips; it puts the 3 in output 2 and the 5 in output 0.

In the end, output bin 0 contains a value-5 microchip, output bin 1 contains a value-2 microchip, and output bin 2 contains a value-3 microchip. In this configuration, bot number 2 is responsible for comparing value-5 microchips with value-2 microchips.

Based on your instructions, what is the number of the bot that is responsible for comparing value-61 microchips with value-17 microchips?

--- Part Two ---

What do you get if you multiply together the values of one chip in each of outputs 0, 1, and 2? */

fn parse_tagged_wrapped_integer<'i, I: FromStr + Integer, T, F: Fn(I) -> T>(
    tag_str: &'static str,
    f: F,
) -> impl FnMut(&'i str) -> IResult<&'i str, T> {
    map(preceded(tag(tag_str), parse_integer), f)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Bot(u8);

impl Parse for Bot {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        parse_tagged_wrapped_integer("bot ", Self)(input)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Value(u8);

impl Parse for Value {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        parse_tagged_wrapped_integer("value ", Self)(input)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct InputInstruction {
    bot: Bot,
    value: Value,
}

impl Parse for InputInstruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(Value::parse, tag(" goes to "), Bot::parse),
            |(value, bot)| Self { bot, value },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Output(u8);

impl Parse for Output {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        parse_tagged_wrapped_integer("output ", Self)(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Recipient {
    Bot(Bot),
    Output(Output),
}

impl Parse for Recipient {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((map(Bot::parse, Self::Bot), map(Output::parse, Self::Output)))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Recipients {
    low: Recipient,
    high: Recipient,
}

impl Parse for Recipients {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag(" gives low to "),
                Recipient::parse,
                tag(" and high to "),
                Recipient::parse,
            )),
            |(_, low, _, high)| Self { low, high },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct BotInstruction {
    bot: Bot,
    recipients: Recipients,
}

impl Parse for BotInstruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((Bot::parse, Recipients::parse)),
            |(bot, recipients)| Self { bot, recipients },
        )(input)
    }
}

enum Instruction {
    Input(InputInstruction),
    Bot(BotInstruction),
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(InputInstruction::parse, Self::Input),
            map(BotInstruction::parse, Self::Bot),
        ))(input)
    }
}

#[cfg_attr(test, derive(PartialEq))]
struct BotInstructions([Option<Recipients>; Self::LEN]);

impl BotInstructions {
    const LEN: usize = u8::MAX as usize + 1_usize;
}

impl Default for BotInstructions {
    fn default() -> Self {
        Self(LargeArrayDefault::large_array_default())
    }
}

#[cfg(test)]
impl Debug for BotInstructions {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("BotInstructions")?;

        let mut debug_list = f.debug_list();

        for bot_instruction in self.0.iter().map(Option::as_ref).flatten() {
            debug_list.entry(bot_instruction);
        }

        debug_list.finish()
    }
}

#[derive(Clone, Copy, Default)]
struct BotState {
    a: Option<Value>,
    b: Option<Value>,
}

impl BotState {
    fn value_count(self) -> usize {
        self.a.into_iter().chain(self.b).count()
    }

    fn try_push(&mut self, value: Value) -> bool {
        if let Some(a) = self.a.replace(value) {
            if self.b.replace(a).is_some() {
                false
            } else {
                true
            }
        } else {
            true
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct OutputState {
    output: Output,
    value: Value,
}

struct State {
    bots: [BotState; Self::BOTS_LEN],
    output: Vec<OutputState>,
}

impl State {
    const BOTS_LEN: usize = u8::MAX as usize + 1_usize;

    fn can_process_bot_recipient(&self, recipient: Recipient) -> bool {
        match recipient {
            Recipient::Bot(bot) => self.bots[bot.0 as usize].value_count() < 2_usize,
            Recipient::Output(_) => true,
        }
    }

    fn can_process_bot(&self, bot: Bot, recipients: Option<Recipients>) -> bool {
        self.bots[bot.0 as usize].value_count() == 2_usize
            && recipients
                .map(|recipients| {
                    self.can_process_bot_recipient(recipients.low)
                        && self.can_process_bot_recipient(recipients.high)
                })
                .unwrap_or_default()
    }

    fn process_bot_recipient(&mut self, recipient: Recipient, value: Value) {
        match recipient {
            Recipient::Bot(bot) => assert!(self.bots[bot.0 as usize].try_push(value)),
            Recipient::Output(output) => self.output.push(OutputState { output, value }),
        }
    }

    fn process_bot(&mut self, bot: Bot, recipients: Recipients) {
        let bot_state: BotState = self.bots[bot.0 as usize];

        let mut values: [Value; 2_usize] = [bot_state.a.unwrap(), bot_state.b.unwrap()];

        values.sort();

        self.process_bot_recipient(recipients.low, values[0_usize]);
        self.process_bot_recipient(recipients.high, values[1_usize]);
    }

    fn try_process_bot(&mut self, bot: Bot, recipients: Option<Recipients>) -> bool {
        if self.can_process_bot(bot, recipients) {
            self.process_bot(bot, recipients.unwrap());

            true
        } else {
            false
        }
    }

    fn process<F: FnMut(Bot, BotState, Recipients)>(
        &mut self,
        bot_instructions: &BotInstructions,
        f: Option<F>,
    ) {
        type BotsBitArray = BitArr!(for State::BOTS_LEN);

        let mut f: Option<F> = f;
        let mut one_value_bots: BotsBitArray = BotsBitArray::ZERO;
        let mut two_value_bots: BotsBitArray = BotsBitArray::ZERO;

        for (bot_index, bot_state) in self.bots.iter().enumerate() {
            match bot_state.value_count() {
                1_usize => one_value_bots.set(bot_index, true),
                2_usize => two_value_bots.set(bot_index, true),
                _ => (),
            }
        }

        while two_value_bots.as_bitslice().any() {
            let prev_two_value_bots: BotsBitArray = two_value_bots;

            for bot_index in prev_two_value_bots.iter_ones() {
                let recipients: Option<Recipients> = bot_instructions.0[bot_index];
                let bot: Bot = Bot(bot_index as u8);

                if self.try_process_bot(bot, recipients) {
                    let recipients: Recipients = recipients.unwrap();

                    if let Some(f) = f.as_mut() {
                        f(bot, self.bots[bot_index], recipients);
                    }

                    self.bots[bot_index] = BotState::default();

                    two_value_bots.set(bot_index, false);

                    for recipient in [recipients.low, recipients.high] {
                        match recipient {
                            Recipient::Bot(recipient_bot) => {
                                let recipient_bot_index: usize = recipient_bot.0 as usize;

                                match self.bots[recipient_bot_index].value_count() {
                                    1_usize => {
                                        one_value_bots.set(recipient_bot_index, true);
                                    }
                                    2_usize => {
                                        one_value_bots.set(recipient_bot_index, false);
                                        two_value_bots.set(recipient_bot_index, true);
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            Recipient::Output(_) => {}
                        }
                    }
                }
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    input_instructions: Vec<InputInstruction>,
    bot_instructions: BotInstructions,
}

impl Solution {
    fn try_initial_state(&self) -> Result<State, InputInstruction> {
        let mut bots: [BotState; State::BOTS_LEN] = LargeArrayDefault::large_array_default();

        for input_instruction in &self.input_instructions {
            if !bots[input_instruction.bot.0 as usize].try_push(input_instruction.value) {
                Err(*input_instruction)?;
            }
        }

        Ok(State {
            bots,
            output: Vec::new(),
        })
    }

    fn try_process_internal<F: FnMut(Bot, BotState, Recipients)>(
        &self,
        f: Option<F>,
    ) -> Option<Vec<OutputState>> {
        let mut state: State = self
            .try_initial_state()
            .map_err(|input_instruction| {
                eprintln!(
                    "Failed to initialize state due to invalid input instruction \
                    {input_instruction:?} for a bot that already has two values."
                );
            })
            .ok()?;

        state.process(&self.bot_instructions, f);

        Some(state.output)
    }

    fn try_bot_that_compares_17_to_61(&self) -> Option<Bot> {
        let mut bot_that_compares_17_to_61: Option<Bot> = None;

        self.try_process_internal(Some(|bot: Bot, bot_state: BotState, _: Recipients| {
            let mut values: [Value; 2_usize] = [bot_state.a.unwrap(), bot_state.b.unwrap()];

            values.sort();

            if values == [Value(17_u8), Value(61_u8)] {
                bot_that_compares_17_to_61 = Some(bot);
            }
        }));

        bot_that_compares_17_to_61
    }

    fn try_process(&self) -> Option<Vec<OutputState>> {
        fn processor(_: Bot, _: BotState, _: Recipients) {}

        self.try_process_internal(Some(processor))
    }

    fn try_product_of_outputs_0_to_2(&self) -> Option<u32> {
        self.try_process().map(|output| {
            output
                .into_iter()
                .filter_map(|output_state| {
                    (output_state.output.0 <= 2_u8).then_some(output_state.value.0 as u32)
                })
                .product()
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut input_instructions: Vec<InputInstruction> = Vec::new();
        let mut bot_instructions: BotInstructions = BotInstructions::default();

        let mut instruction_iter =
            iterator(input, terminated(Instruction::parse, opt(line_ending)));

        for instruction in &mut instruction_iter {
            match instruction {
                Instruction::Input(input_instruction) => input_instructions.push(input_instruction),
                Instruction::Bot(bot_instruction) => {
                    bot_instructions.0[bot_instruction.bot.0 as usize] =
                        Some(bot_instruction.recipients);
                }
            }
        }

        let (input, ()): (&str, ()) = instruction_iter.finish()?;

        input_instructions.sort();

        Ok((
            input,
            Self {
                input_instructions,
                bot_instructions,
            },
        ))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_bot_that_compares_17_to_61());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_product_of_outputs_0_to_2());
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

    const SOLUTION_STR: &'static str = "\
        value 5 goes to bot 2\n\
        bot 2 gives low to bot 1 and high to bot 0\n\
        value 3 goes to bot 1\n\
        bot 1 gives low to output 1 and high to bot 0\n\
        bot 0 gives low to output 2 and high to output 0\n\
        value 2 goes to bot 2\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        fn rb(bot: u8) -> Recipient {
            Recipient::Bot(Bot(bot))
        }

        fn ro(output: u8) -> Recipient {
            Recipient::Output(Output(output))
        }

        macro_rules! input_instructions {
            [ $( ($value:expr, $bot:expr), )* ] => {
                vec![ $( InputInstruction { value: Value($value), bot: Bot($bot) }, )* ]
            }
        }

        macro_rules! bot_instructions {
            [ $( ($bot:expr, $low:expr, $high:expr), )* ] => { {
                let mut bot_instructions: BotInstructions = BotInstructions::default();

                $(
                    bot_instructions.0[$bot as usize] = Some(Recipients {
                        low: $low,
                        high: $high,
                    });
                )*

                bot_instructions
            } }
        }

        ONCE_LOCK.get_or_init(|| Solution {
            input_instructions: input_instructions![(3, 1), (2, 2), (5, 2),],
            bot_instructions: bot_instructions![
                (2, rb(1), rb(0)),
                (1, ro(1), rb(0)),
                (0, ro(2), ro(0)),
            ],
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_process() {
        assert_eq!(
            solution().try_process(),
            Some(vec![
                OutputState {
                    output: Output(1_u8),
                    value: Value(2_u8)
                },
                OutputState {
                    output: Output(2_u8),
                    value: Value(3_u8)
                },
                OutputState {
                    output: Output(0_u8),
                    value: Value(5_u8)
                },
            ])
        )
    }
}
