use {
    crate::*,
    bitvec::{domain::Domain, prelude::*},
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, verify},
        error::Error,
        multi::separated_list0,
        sequence::tuple,
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 6: Probably a Fire Hazard ---

Because your neighbors keep defeating you in the holiday house decorating contest year after year, you've decided to deploy one million lights in a 1000x1000 grid.

Furthermore, because you've been especially nice this year, Santa has mailed you instructions on how to display the ideal lighting configuration.

Lights in your grid are numbered from 0 to 999 in each direction; the lights at each corner are at 0,0, 0,999, 999,999, and 999,0. The instructions include whether to turn on, turn off, or toggle various inclusive ranges given as coordinate pairs. Each coordinate pair represents opposite corners of a rectangle, inclusive; a coordinate pair like 0,0 through 2,2 therefore refers to 9 lights in a 3x3 square. The lights all start turned off.

To defeat your neighbors this year, all you have to do is set up your lights by doing the instructions Santa sent you in order.

For example:

    turn on 0,0 through 999,999 would turn on (or leave on) every light.
    toggle 0,0 through 999,0 would toggle the first line of 1000 lights, turning off the ones that were on, and turning on the ones that were off.
    turn off 499,499 through 500,500 would turn off (or leave off) the middle four lights.

After following the instructions, how many lights are lit?

--- Part Two ---

You just finish implementing your winning light pattern when you realize you mistranslated Santa's message from Ancient Nordic Elvish.

The light grid you bought actually has individual brightness controls; each light can have a brightness of zero or more. The lights all start at zero.

The phrase turn on actually means that you should increase the brightness of those lights by 1.

The phrase turn off actually means that you should decrease the brightness of those lights by 1, to a minimum of zero.

The phrase toggle actually means that you should increase the brightness of those lights by 2.

What is the total brightness of all lights combined after following Santa's instructions?

For example:

    turn on 0,0 through 0,0 would increase the total brightness by 1.
    toggle 0,0 through 999,999 would increase the total brightness by 2000000. */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum LightAction {
    TurnOn,
    TurnOff,
    Toggle,
}

impl Parse for LightAction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(tag("turn on"), |_| Self::TurnOn),
            map(tag("turn off"), |_| Self::TurnOff),
            map(tag("toggle"), |_| Self::Toggle),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Instruction {
    light_action: LightAction,
    region: SmallRangeInclusive<IVec2>,
}

impl Instruction {
    fn parse_coordinate<'i>(input: &'i str) -> IResult<&'i str, IVec2> {
        map(
            parse_separated_array(parse_integer, tag(",")),
            IVec2::from_array,
        )(input)
    }

    fn iter_ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let x: usize = self.region.start.x as usize;
        let row_len: usize = self.region.end.x as usize - x + 1_usize;

        (self.region.start.y as usize..=self.region.end.y as usize).map(move |y| {
            let start: usize = y * Solution::SIDE_LEN + x;

            start..start + row_len
        })
    }

    fn execute(&self, grid: &mut BitSlice) {
        for range in self.iter_ranges() {
            let row: &mut BitSlice = &mut grid[range];

            match self.light_action {
                LightAction::TurnOn => row.fill(true),
                LightAction::TurnOff => row.fill(false),
                LightAction::Toggle => match row.domain_mut() {
                    Domain::Enclave(mut enclave) => {
                        enclave.invert();
                    }
                    Domain::Region { head, body, tail } => {
                        if let Some(mut head) = head {
                            head.invert();
                        }

                        for element in body {
                            *element ^= usize::MAX;
                        }

                        if let Some(mut tail) = tail {
                            tail.invert();
                        }
                    }
                },
            }
        }
    }

    fn execute_ancient_nordic_elvish(&self, grid: &mut [u8]) {
        for range in self.iter_ranges() {
            let row: &mut [u8] = &mut grid[range];

            match self.light_action {
                LightAction::TurnOn => {
                    for light in row {
                        *light += 1_u8;
                    }
                }
                LightAction::TurnOff => {
                    for light in row {
                        *light = light.saturating_sub(1_u8);
                    }
                }
                LightAction::Toggle => {
                    for light in row {
                        *light += 2_u8;
                    }
                }
            }
        }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        verify(
            map(
                tuple((
                    LightAction::parse,
                    tag(" "),
                    Self::parse_coordinate,
                    tag(" through "),
                    Self::parse_coordinate,
                )),
                |(light_action, _, start, _, end)| Self {
                    light_action,
                    region: SmallRangeInclusive { start, end },
                },
            ),
            |instruction| {
                instruction.region.start.cmple(instruction.region.end).all()
                    && instruction.region.start.cmpge(IVec2::ZERO).all()
                    && instruction.region.end.cmplt(Solution::DIMENSIONS).all()
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    const SIDE_LEN: usize = 1000_usize;
    const DIMENSIONS: IVec2 = IVec2::new(Self::SIDE_LEN as i32, Self::SIDE_LEN as i32);
    const GRID_AREA: usize = Self::DIMENSIONS.x as usize * Self::DIMENSIONS.y as usize;

    fn grid(&self) -> BitVec {
        let mut grid: BitVec = bitvec![0; Self::GRID_AREA];

        for instruction in &self.0 {
            instruction.execute(&mut grid);
        }

        grid
    }

    fn count_lit_lights(&self) -> usize {
        self.grid().count_ones()
    }

    fn total_brightness(&self) -> usize {
        let mut grid: Vec<u8> = vec![0_u8; Self::GRID_AREA];

        for instruction in &self.0 {
            instruction.execute_ancient_nordic_elvish(&mut grid);
        }

        grid.into_iter().map(usize::from).sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(separated_list0(line_ending, Instruction::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// `bitvec` only has an `invert` for a partial element???
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_lit_lights());
    }

    /// That was kind of silly. Not a fan.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_brightness());
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
        turn on 0,0 through 999,999\n\
        toggle 0,0 through 999,0\n\
        turn off 499,499 through 500,500\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                Instruction {
                    light_action: LightAction::TurnOn,
                    region: ((0_i32, 0_i32).into()..=(999_i32, 999_i32).into()).into(),
                },
                Instruction {
                    light_action: LightAction::Toggle,
                    region: ((0_i32, 0_i32).into()..=(999_i32, 0_i32).into()).into(),
                },
                Instruction {
                    light_action: LightAction::TurnOff,
                    region: ((499_i32, 499_i32).into()..=(500_i32, 500_i32).into()).into(),
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
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
