use {
    crate::*,
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
    std::iter::repeat_n,
};

/* --- Day 1: Secret Entrance ---

The Elves have good news and bad news.

The good news is that they've discovered project management! This has given them the tools they need to prevent their usual Christmas emergency. For example, they now know that the North Pole decorations need to be finished soon so that other critical tasks can start on time.

The bad news is that they've realized they have a different emergency: according to their resource planning, none of them have any time left to decorate the North Pole!

To save Christmas, the Elves need you to finish decorating the North Pole by December 12th.

Collect stars by solving puzzles. Two puzzles will be made available on each day; the second puzzle is unlocked when you complete the first. Each puzzle grants one star. Good luck!

You arrive at the secret entrance to the North Pole base ready to start decorating. Unfortunately, the password seems to have been changed, so you can't get in. A document taped to the wall helpfully explains:

"Due to new security protocols, the password is locked in the safe below. Please see the attached document for the new combination."

The safe has a dial with only an arrow on it; around the dial are the numbers 0 through 99 in order. As you turn the dial, it makes a small click noise as it reaches each number.

The attached document (your puzzle input) contains a sequence of rotations, one per line, which tell you how to open the safe. A rotation starts with an L or R which indicates whether the rotation should be to the left (toward lower numbers) or to the right (toward higher numbers). Then, the rotation has a distance value which indicates how many clicks the dial should be rotated in that direction.

So, if the dial were pointing at 11, a rotation of R8 would cause the dial to point at 19. After that, a rotation of L19 would cause it to point at 0.

Because the dial is a circle, turning the dial left from 0 one click makes it point at 99. Similarly, turning the dial right from 99 one click makes it point at 0.

So, if the dial were pointing at 5, a rotation of L10 would cause it to point at 95. After that, a rotation of R5 could cause it to point at 0.

The dial starts by pointing at 50.

You could follow the instructions, but your recent required official North Pole secret entrance security training seminar taught you that the safe is actually a decoy. The actual password is the number of times the dial is left pointing at 0 after any rotation in the sequence.

For example, suppose the attached document contained the following rotations:

L68
L30
R48
L5
R60
L55
L1
L99
R14
L82

Following these rotations would cause the dial to move as follows:

    The dial starts by pointing at 50.
    The dial is rotated L68 to point at 82.
    The dial is rotated L30 to point at 52.
    The dial is rotated R48 to point at 0.
    The dial is rotated L5 to point at 95.
    The dial is rotated R60 to point at 55.
    The dial is rotated L55 to point at 0.
    The dial is rotated L1 to point at 99.
    The dial is rotated L99 to point at 0.
    The dial is rotated R14 to point at 14.
    The dial is rotated L82 to point at 32.

Because the dial points at 0 a total of three times during this process, the password in this example is 3.

Analyze the rotations in your attached document. What's the actual password to open the door?

--- Part Two ---

You're sure that's the right password, but the door won't open. You knock, but nobody answers. You build a snowman while you think.

As you're rolling the snowballs for your snowman, you find another security document that must have fallen into the snow:

"Due to newer security protocols, please use password method 0x434C49434B until further notice."

You remember from the training seminar that "method 0x434C49434B" means you're actually supposed to count the number of times any click causes the dial to point at 0, regardless of whether it happens during a rotation or at the end of one.

Following the same rotations as in the above example, the dial points at zero a few extra times during its rotations:

    The dial starts by pointing at 50.
    The dial is rotated L68 to point at 82; during this rotation, it points at 0 once.
    The dial is rotated L30 to point at 52.
    The dial is rotated R48 to point at 0.
    The dial is rotated L5 to point at 95.
    The dial is rotated R60 to point at 55; during this rotation, it points at 0 once.
    The dial is rotated L55 to point at 0.
    The dial is rotated L1 to point at 99.
    The dial is rotated L99 to point at 0.
    The dial is rotated R14 to point at 14.
    The dial is rotated L82 to point at 32; during this rotation, it points at 0 once.

In this example, the dial points at 0 three times at the end of a rotation, plus three more times during a rotation. So, in this example, the new password would be 6.

Be careful: if the dial were pointing at 50, a single rotation like R1000 would cause the dial to point at 0 ten times before returning back to 50!

Using password method 0x434C49434B, what is the password to open the door? */

/// # Invariants
/// * Is in the range `[0, Self::COUNT)`.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
struct DialValue(i32);

#[cfg_attr(test, derive(Debug, PartialEq))]
struct DialValueRotateOutput {
    dial_value: DialValue,
    skipped_zeroes: usize,
}

impl DialValue {
    const ZERO: Self = Self(0_i32);
    const INITIAL: Self = Self(50_i32);
    const COUNT: Self = Self(100_i32);

    fn invert(self) -> Self {
        Self((Self::COUNT.0 - self.0) % Self::COUNT.0)
    }

    fn rotate(self, rotation: Rotation) -> DialValueRotateOutput {
        let is_rotation_sign_negative: bool = rotation.sign() < 0_i32;

        let mut dial_value: DialValue = self;

        if is_rotation_sign_negative {
            dial_value = dial_value.invert();
        }

        dial_value.0 += rotation.magnitude();

        let mut skipped_zeroes: usize = (dial_value.0 / Self::COUNT.0) as usize;

        dial_value.0 %= Self::COUNT.0;

        if dial_value.0 == 0_i32 {
            skipped_zeroes -= 1_usize;
        }

        if is_rotation_sign_negative {
            dial_value = dial_value.invert();
        }

        DialValueRotateOutput {
            dial_value,
            skipped_zeroes,
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

/// # Invariants
/// * Is non-zero.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Rotation(i32);

impl Rotation {
    fn sign(self) -> i32 {
        self.0.signum()
    }

    fn magnitude(self) -> i32 {
        self.0.abs()
    }

    fn parse_sign<'i>(input: &'i str) -> IResult<&'i str, i32> {
        alt((map(tag("L"), |_| -1_i32), map(tag("R"), |_| 1_i32)))(input)
    }

    fn parse_magnitude<'i>(input: &'i str) -> IResult<&'i str, i32> {
        verify(parse_integer, |&rotation_magnitude| {
            rotation_magnitude > 0_i32
        })(input)
    }
}

impl Parse for Rotation {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((Self::parse_sign, Self::parse_magnitude)),
            |(rotation_direction, rotation_magnitude)| {
                Self(rotation_direction * rotation_magnitude)
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Rotation>);

impl Solution {
    fn iter_dial_values(
        &self,
        use_alternate_password_method: bool,
    ) -> impl Iterator<Item = DialValue> + '_ {
        let mut dial_value: DialValue = DialValue::INITIAL;

        [dial_value]
            .into_iter()
            .chain(self.0.iter().flat_map(move |&rotation| {
                let output: DialValueRotateOutput = dial_value.rotate(rotation);

                dial_value = output.dial_value;

                repeat_n(
                    DialValue::ZERO,
                    if use_alternate_password_method {
                        output.skipped_zeroes
                    } else {
                        0_usize
                    },
                )
                .chain([dial_value])
            }))
    }

    fn actual_door_password(&self) -> usize {
        self.iter_dial_values(false)
            .filter(DialValue::is_zero)
            .count()
    }

    fn actual_door_password_with_alternate_password_method(&self) -> usize {
        self.iter_dial_values(true)
            .filter(DialValue::is_zero)
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(separated_list0(line_ending, Rotation::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.actual_door_password());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.actual_door_password_with_alternate_password_method());
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
    use {super::*, static_assertions::const_assert_eq, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        L68\n\
        L30\n\
        R48\n\
        L5\n\
        R60\n\
        L55\n\
        L1\n\
        L99\n\
        R14\n\
        L82\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            const L: i32 = -1_i32;
            const R: i32 = 1_i32;

            macro_rules! rotations {
                ( $( ( $direction:expr, $magnitude:expr ) ),* $(,)? ) => { vec![ $(
                    Rotation($direction * $magnitude),
                )* ] };
            }

            vec![Solution(rotations![
                (L, 68),
                (L, 30),
                (R, 48),
                (L, 5),
                (R, 60),
                (L, 55),
                (L, 1),
                (L, 99),
                (R, 14),
                (L, 82),
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
    fn test_iter_dial_values_false() {
        for (index, dial_values) in [vec![
            50_i32, 82_i32, 52_i32, 0_i32, 95_i32, 55_i32, 0_i32, 99_i32, 0_i32, 14_i32, 32_i32,
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_dial_values(false)
                    .map(|dial_value| dial_value.0)
                    .collect::<Vec<i32>>(),
                dial_values
            );
        }
    }

    #[test]
    fn test_iter_dial_values_true() {
        for (index, dial_values) in [vec![
            50_i32, 0_i32, 82_i32, 52_i32, 0_i32, 95_i32, 0_i32, 55_i32, 0_i32, 99_i32, 0_i32,
            14_i32, 0_i32, 32_i32,
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_dial_values(true)
                    .map(|dial_value| dial_value.0)
                    .collect::<Vec<i32>>(),
                dial_values
            );
        }
    }

    #[test]
    fn test_dial_value_invert() {
        for dial_value_inner in 0_i32..DialValue::COUNT.0 {
            let dial_value: DialValue = DialValue(dial_value_inner);
            let inverted_dial_value: DialValue = DialValue(if dial_value_inner == 0_i32 {
                0_i32
            } else {
                DialValue::COUNT.0 - dial_value_inner
            });

            assert_eq!(dial_value.invert(), inverted_dial_value);
        }
    }

    #[test]
    fn test_dial_value_rotate() {
        const_assert_eq!(DialValue::COUNT.0 % 4_i32, 0_i32);

        const NO_ROTATION: i32 = 0_i32;
        const QUARTER_ROTATION: i32 = DialValue::COUNT.0 / 4_i32;
        const HALF_ROTATION: i32 = DialValue::COUNT.0 / 2_i32;
        const THREE_QUARTER_ROTATION: i32 = 3_i32 * DialValue::COUNT.0 / 4_i32;
        const FULL_ROTATION: i32 = DialValue::COUNT.0;

        for (dial_value, rotation, dial_value_rotate_output) in []
            .into_iter()
            // Simple tests
            .chain([
                (
                    DialValue(0_i32),
                    Rotation(1_i32),
                    DialValueRotateOutput {
                        dial_value: DialValue(1_i32),
                        skipped_zeroes: 0_usize,
                    },
                ),
                (
                    DialValue(0_i32),
                    Rotation(2_i32),
                    DialValueRotateOutput {
                        dial_value: DialValue(2_i32),
                        skipped_zeroes: 0_usize,
                    },
                ),
                (
                    DialValue(0_i32),
                    Rotation(3_i32),
                    DialValueRotateOutput {
                        dial_value: DialValue(3_i32),
                        skipped_zeroes: 0_usize,
                    },
                ),
            ])
            // Simple quarter rotation tests, positive
            .chain(
                [].into_iter()
                    // No initial rotation
                    .chain([
                        (
                            DialValue(NO_ROTATION),
                            Rotation(QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(QUARTER_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(HALF_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(HALF_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(THREE_QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(THREE_QUARTER_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(FULL_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(NO_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                    ])
                    // Quarter initial rotation
                    .chain([
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(HALF_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(HALF_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(THREE_QUARTER_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(THREE_QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(NO_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(FULL_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(QUARTER_ROTATION),
                                skipped_zeroes: 1_usize,
                            },
                        ),
                    ]),
            )
            // Simple quarter rotation tests, negative
            .chain(
                [].into_iter()
                    // No initial rotation
                    .chain([
                        (
                            DialValue(NO_ROTATION),
                            Rotation(-QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(THREE_QUARTER_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(-HALF_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(HALF_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(-THREE_QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(QUARTER_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(NO_ROTATION),
                            Rotation(-FULL_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(NO_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                    ])
                    // Quarter initial rotation
                    .chain([
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(-QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(NO_ROTATION),
                                skipped_zeroes: 0_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(-HALF_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(THREE_QUARTER_ROTATION),
                                skipped_zeroes: 1_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(-THREE_QUARTER_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(HALF_ROTATION),
                                skipped_zeroes: 1_usize,
                            },
                        ),
                        (
                            DialValue(QUARTER_ROTATION),
                            Rotation(-FULL_ROTATION),
                            DialValueRotateOutput {
                                dial_value: DialValue(QUARTER_ROTATION),
                                skipped_zeroes: 1_usize,
                            },
                        ),
                    ]),
            )
            // Complex quarter rotation tests, positive
            .chain([
                (
                    DialValue(HALF_ROTATION),
                    Rotation(HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 0_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(FULL_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(HALF_ROTATION),
                        skipped_zeroes: 1_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(3 * HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 1_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(2 * FULL_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(HALF_ROTATION),
                        skipped_zeroes: 2_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(5 * HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 2_usize,
                    },
                ),
            ])
            // Complex quarter rotation tests, negative
            .chain([
                (
                    DialValue(HALF_ROTATION),
                    Rotation(-HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 0_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(-FULL_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(HALF_ROTATION),
                        skipped_zeroes: 1_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(-3 * HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 1_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(-2 * FULL_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(HALF_ROTATION),
                        skipped_zeroes: 2_usize,
                    },
                ),
                (
                    DialValue(HALF_ROTATION),
                    Rotation(-5 * HALF_ROTATION),
                    DialValueRotateOutput {
                        dial_value: DialValue(NO_ROTATION),
                        skipped_zeroes: 2_usize,
                    },
                ),
            ])
        {
            assert_eq!(dial_value.rotate(rotation), dial_value_rotate_output);
        }
    }

    #[test]
    fn test_compute_skipped_zeroes() {}

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
