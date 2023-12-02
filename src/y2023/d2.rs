use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{map, map_res, opt},
        error::Error,
        multi::{fold_many0, fold_many_m_n, many0},
        sequence::{delimited, separated_pair, terminated},
        Err, IResult,
    },
    std::str::FromStr,
};

#[derive(Clone, Copy)]
enum CubeColor {
    Red,
    Green,
    Blue,
}

impl CubeColor {
    const RED: &'static str = "red";
    const GREEN: &'static str = "green";
    const BLUE: &'static str = "blue";

    fn str(self) -> &'static str {
        match self {
            Self::Red => Self::RED,
            Self::Green => Self::GREEN,
            Self::Blue => Self::BLUE,
        }
    }

    fn match_arm<'i>(self) -> impl Fn(&'i str) -> IResult<&'i str, Self> {
        move |input| map(tag(self.str()), |_| self)(input)
    }
}

impl Parse for CubeColor {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::Red.match_arm(),
            Self::Green.match_arm(),
            Self::Blue.match_arm(),
        ))(input)
    }
}

struct CubeCount {
    count: u8,
    color: CubeColor,
}

impl Parse for CubeCount {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(map_res(digit1, u8::from_str), tag(" "), CubeColor::parse),
            |(count, color)| Self { count, color },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Default)]
struct Set {
    red: u8,
    green: u8,
    blue: u8,
}

impl Set {
    fn all_le(self, other: Self) -> bool {
        self.red <= other.red && self.green <= other.green && self.blue <= other.blue
    }
}

impl Parse for Set {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        fold_many_m_n(
            1_usize,
            3_usize,
            terminated(CubeCount::parse, opt(tag(", "))),
            Set::default,
            |mut set, cube_count| {
                match cube_count.color {
                    CubeColor::Red => set.red += cube_count.count,
                    CubeColor::Green => set.green += cube_count.count,
                    CubeColor::Blue => set.blue += cube_count.count,
                }

                set
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Game {
    id: u8,
    max: Set,
}

impl Parse for Game {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, id): (&str, u8) =
            map_res(delimited(tag("Game "), digit1, tag(": ")), u8::from_str)(input)?;

        map(
            fold_many0(
                terminated(Set::parse, opt(tag("; "))),
                Set::default,
                |max, set| Set {
                    red: max.red.max(set.red),
                    green: max.green.max(set.green),
                    blue: max.blue.max(set.blue),
                },
            ),
            move |max| Self { id, max },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution(Vec<Game>);

impl Solution {
    const BAG_SET: Set = Set {
        red: 12_u8,
        green: 13_u8,
        blue: 14_u8,
    };

    fn sum_possible_game_ids(&self) -> u32 {
        self.0
            .iter()
            .filter(|game| game.max.all_le(Self::BAG_SET))
            .map(|game| game.id as u32)
            .sum()
    }

    fn sum_game_max_powers(&self) -> u32 {
        self.0
            .iter()
            .map(|game| game.max.red as u32 * game.max.green as u32 * game.max.blue as u32)
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Game::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_possible_game_ids());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_game_max_powers());
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
        Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green\n\
        Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue\n\
        Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red\n\
        Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red\n\
        Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Game {
                    id: 1_u8,
                    max: Set {
                        red: 4_u8,
                        green: 2_u8,
                        blue: 6_u8,
                    },
                },
                Game {
                    id: 2_u8,
                    max: Set {
                        red: 1_u8,
                        green: 3_u8,
                        blue: 4_u8,
                    },
                },
                Game {
                    id: 3_u8,
                    max: Set {
                        red: 20_u8,
                        green: 13_u8,
                        blue: 6_u8,
                    },
                },
                Game {
                    id: 4_u8,
                    max: Set {
                        red: 14_u8,
                        green: 3_u8,
                        blue: 15_u8,
                    },
                },
                Game {
                    id: 5_u8,
                    max: Set {
                        red: 6_u8,
                        green: 3_u8,
                        blue: 2_u8,
                    },
                },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_sum_possible_game_ids() {
        assert_eq!(solution().sum_possible_game_ids(), 8_u32);
    }

    #[test]
    fn test_sum_game_max_powers() {
        assert_eq!(solution().sum_game_max_powers(), 2286_u32);
    }
}
