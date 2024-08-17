use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::{many0, many1},
        sequence::terminated,
        Err, IResult,
    },
};

/* --- Day 2: Bathroom Security ---

You arrive at Easter Bunny Headquarters under cover of darkness. However, you left in such a rush that you forgot to use the bathroom! Fancy office buildings like this one usually have keypad locks on their bathrooms, so you search the front desk for the code.

"In order to improve security," the document you find says, "bathroom codes will no longer be written down. Instead, please memorize and follow the procedure below to access the bathrooms."

The document goes on to explain that each button to be pressed can be found by starting on the previous button and moving to adjacent buttons on the keypad: U moves up, D moves down, L moves left, and R moves right. Each line of instructions corresponds to one button, starting at the previous button (or, for the first line, the "5" button); press whatever button you're on at the end of each line. If a move doesn't lead to a button, ignore it.

You can't hold it much longer, so you decide to figure out the code as you walk to the bathroom. You picture a keypad like this:

1 2 3
4 5 6
7 8 9

Suppose your instructions are:

ULL
RRDDD
LURDL
UUUUD

    You start at "5" and move up (to "2"), left (to "1"), and left (you can't, and stay on "1"), so the first button is 1.
    Starting from the previous button ("1"), you move right twice (to "3") and then down three times (stopping at "9" after two moves and ignoring the third), ending up with 9.
    Continuing from "9", you move left, up, right, down, and left, ending with 8.
    Finally, you move up four times (stopping at "2"), then down once, ending with 5.

So, in this example, the bathroom code is 1985.

Your puzzle input is the instructions from the document you found at the front desk. What is the bathroom code?

--- Part Two ---

You finally arrive at the bathroom (it's a several minute walk from the lobby so visitors can behold the many fancy conference rooms and water coolers on this floor) and go to punch in the code. Much to your bladder's dismay, the keypad is not at all like you imagined it. Instead, you are confronted with the result of hundreds of man-hours of bathroom-keypad-design meetings:

    1
  2 3 4
5 6 7 8 9
  A B C
    D

You still start at "5" and stop when you're at an edge, but given the same instructions as above, the outcome is very different:

    You start at "5" and don't move at all (up and left are both edges), ending at 5.
    Continuing from "5", you move right twice and down three times (through "6", "7", "B", "D", "D"), ending at D.
    Then, from "D", you move five more times (through "D", "B", "C", "C", "B"), ending at B.
    Finally, after five more moves, you end at 3.

So, given the actual keypad layout, the code would be 5DB3.

Using the same instructions in your puzzle input, what is the correct bathroom code? */

trait Keypad {
    fn start_pos() -> IVec2;
    fn is_in_bounds(pos: IVec2) -> bool;
    fn digit(pos: IVec2) -> char;
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ButtonInstructions(Vec<Direction>);

impl ButtonInstructions {
    fn parse_instruction_branch<'i>(
        tag_str: &'i str,
        dir: Direction,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Direction> {
        map(tag(tag_str), move |_| dir)
    }

    fn parse_instruction<'i>(input: &'i str) -> IResult<&'i str, Direction> {
        alt((
            Self::parse_instruction_branch("U", Direction::North),
            Self::parse_instruction_branch("R", Direction::East),
            Self::parse_instruction_branch("D", Direction::South),
            Self::parse_instruction_branch("L", Direction::West),
        ))(input)
    }

    fn process<K: Keypad>(&self, pos: IVec2) -> IVec2 {
        self.0
            .iter()
            .copied()
            .map(Direction::vec)
            .fold(pos, |pos, delta| {
                let pos_plus_delta: IVec2 = pos + delta;

                if K::is_in_bounds(pos_plus_delta) {
                    pos_plus_delta
                } else {
                    pos
                }
            })
    }
}

impl Parse for ButtonInstructions {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many1(Self::parse_instruction), Self)(input)
    }
}

struct OrthogonalKeypad;

impl Keypad for OrthogonalKeypad {
    fn start_pos() -> IVec2 {
        IVec2::ZERO
    }

    fn is_in_bounds(pos: IVec2) -> bool {
        pos.cmpge(IVec2::NEG_ONE).all() && pos.cmple(IVec2::ONE).all()
    }

    fn digit(pos: IVec2) -> char {
        let rem_euclid: IVec2 = (pos + IVec2::ONE).rem_euclid(3_i32 * IVec2::ONE);

        (b'1' + rem_euclid.x as u8 + (3_i32 * rem_euclid.y) as u8) as char
    }
}

struct DiagonalKeypad;

impl Keypad for DiagonalKeypad {
    fn start_pos() -> IVec2 {
        IVec2::new(-2_i32, 0_i32)
    }

    fn is_in_bounds(pos: IVec2) -> bool {
        manhattan_magnitude_2d(pos) <= 2_i32
    }

    fn digit(pos: IVec2) -> char {
        assert!(Self::is_in_bounds(pos));

        match pos.y {
            -2_i32 => '1',
            -1_i32 => ((pos.x + 1_i32) as u8 + b'2') as char,
            0_i32 => ((pos.x + 2_i32) as u8 + b'5') as char,
            1_i32 => ((pos.x + 1_i32) as u8 + b'A') as char,
            2_i32 => 'D',
            _ => unreachable!(),
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<ButtonInstructions>);

impl Solution {
    fn iter_poses<K: Keypad>(&self) -> impl Iterator<Item = IVec2> + '_ {
        let mut pos: IVec2 = K::start_pos();

        self.0.iter().map(move |button_instructions| {
            pos = button_instructions.process::<K>(pos);

            pos
        })
    }

    fn bathroom_code<K: Keypad>(&self) -> String {
        self.iter_poses::<K>().map(K::digit).collect()
    }

    fn orthogonal_bathroom_code(&self) -> String {
        self.bathroom_code::<OrthogonalKeypad>()
    }

    fn diagonal_bathroom_code(&self) -> String {
        self.bathroom_code::<DiagonalKeypad>()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(ButtonInstructions::parse, opt(line_ending))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.orthogonal_bathroom_code());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.diagonal_bathroom_code());
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

    const SOLUTION_STR: &'static str = "ULL\n\
        RRDDD\n\
        LURDL\n\
        UUUUD\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            use Direction::{East as R, North as U, South as D, West as L};

            macro_rules! solution {
                [ $( [ $( $dir:expr, )* ], )* ] => {
                    Solution(vec![ $(
                        ButtonInstructions(vec![ $(
                            $dir,
                        )* ]),
                    )* ])
                };
            }

            solution![
                [U, L, L,],
                [R, R, D, D, D,],
                [L, U, R, D, L,],
                [U, U, U, U, D,],
            ]
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_poses() {
        macro_rules! ivec2s {
            [ $( ($x:expr, $y:expr), )* ] => {
                vec![ $( IVec2::new($x, $y), )* ]
            };
        }

        assert_eq!(
            solution()
                .iter_poses::<OrthogonalKeypad>()
                .collect::<Vec<IVec2>>(),
            ivec2s![(-1, -1), (1, 1), (0, 1), (0, 0),]
        );
        assert_eq!(
            solution()
                .iter_poses::<DiagonalKeypad>()
                .collect::<Vec<IVec2>>(),
            ivec2s![(-2, 0), (0, 2), (0, 1), (0, -1),]
        );
    }

    #[test]
    fn test_digit() {
        for index in 0_i32..9_i32 {
            assert_eq!(
                OrthogonalKeypad::digit(IVec2::new(index % 3_i32, index / 3_i32) - IVec2::ONE),
                (b'1' + index as u8) as char
            );
        }

        for (x, y, c) in [
            (0, -2, '1'),
            (-1, -1, '2'),
            (0, -1, '3'),
            (1, -1, '4'),
            (-2, 0, '5'),
            (-1, 0, '6'),
            (0, 0, '7'),
            (1, 0, '8'),
            (2, 0, '9'),
            (-1, 1, 'A'),
            (0, 1, 'B'),
            (1, 1, 'C'),
            (0, 2, 'D'),
        ] {
            assert_eq!(DiagonalKeypad::digit(IVec2::new(x, y)), c);
        }
    }

    #[test]
    fn test_bathroom_code() {
        assert_eq!(solution().orthogonal_bathroom_code(), "1985".to_owned());
        assert_eq!(solution().diagonal_bathroom_code(), "5DB3".to_owned());
    }
}
