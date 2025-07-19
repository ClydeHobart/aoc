use {
    crate::*,
    arrayvec::ArrayVec,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map_opt, success, verify},
        error::Error,
        multi::separated_list1,
        sequence::tuple,
        Err, IResult,
    },
    std::cmp::Ordering,
    strum::{EnumCount, EnumIter, EnumVariantNames, IntoEnumIterator},
};

/* --- Day 16: Aunt Sue ---

Your Aunt Sue has given you a wonderful gift, and you'd like to send her a thank you card. However, there's a small problem: she signed it "From, Aunt Sue".

You have 500 Aunts named "Sue".

So, to avoid sending the card to the wrong person, you need to figure out which Aunt Sue (which you conveniently number 1 to 500, for sanity) gave you the gift. You open the present and, as luck would have it, good ol' Aunt Sue got you a My First Crime Scene Analysis Machine! Just what you wanted. Or needed, as the case may be.

The My First Crime Scene Analysis Machine (MFCSAM for short) can detect a few specific compounds in a given sample, as well as how many distinct kinds of those compounds there are. According to the instructions, these are what the MFCSAM can detect:

    children, by human DNA age analysis.
    cats. It doesn't differentiate individual breeds.
    Several seemingly random breeds of dog: samoyeds, pomeranians, akitas, and vizslas.
    goldfish. No other kinds of fish.
    trees, all in one group.
    cars, presumably by exhaust or gasoline or something.
    perfumes, which is handy, since many of your Aunts Sue wear a few kinds.

In fact, many of your Aunts Sue have many of these. You put the wrapping from the gift into the MFCSAM. It beeps inquisitively at you a few times and then prints out a message on ticker tape:

children: 3
cats: 7
samoyeds: 2
pomeranians: 3
akitas: 0
vizslas: 0
goldfish: 5
trees: 3
cars: 2
perfumes: 1

You make a list of the things you can remember about each Aunt Sue. Things missing from your list aren't zero - you simply don't remember the value.

What is the number of the Sue that got you the gift?

--- Part Two ---

As you're about to send the thank you note, something in the MFCSAM's instructions catches your eye. Apparently, it has an outdated retroencabulator, and so the output from the machine isn't exact values - some of them indicate ranges.

In particular, the cats and trees readings indicates that there are greater than that many (due to the unpredictable nuclear decay of cat dander and tree pollen), while the pomeranians and goldfish readings indicate that there are fewer than that many (due to the modial interaction of magnetoreluctance).

What is the number of the real Aunt Sue? */

#[derive(Clone, Copy, EnumCount, EnumIter, EnumVariantNames, PartialEq)]
#[strum(serialize_all = "snake_case")]
enum CompoundType {
    Children,
    Cats,
    Samoyeds,
    Pomeranians,
    Akitas,
    Vizslas,
    Goldfish,
    Trees,
    Cars,
    Perfumes,
}

impl CompoundType {
    fn cmp_target_ordering(self) -> Ordering {
        Ordering::Equal
    }

    fn cmp_target_ordering_given_outdated_retroencabulator(self) -> Ordering {
        match self {
            Self::Cats | Self::Trees => Ordering::Greater,
            Self::Pomeranians | Self::Goldfish => Ordering::Less,
            _ => Ordering::Equal,
        }
    }
}

impl_Parse_for_Enum!(CompoundType);

type CompoundValueRaw = u8;
type CompoundValue = Index<CompoundValueRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct AuntSue([CompoundValue; CompoundType::COUNT]);

impl AuntSue {
    const TARGET: Self = Self([
        CompoundValue::new_raw(3_u8),
        CompoundValue::new_raw(7_u8),
        CompoundValue::new_raw(2_u8),
        CompoundValue::new_raw(3_u8),
        CompoundValue::new_raw(0_u8),
        CompoundValue::new_raw(0_u8),
        CompoundValue::new_raw(5_u8),
        CompoundValue::new_raw(3_u8),
        CompoundValue::new_raw(2_u8),
        CompoundValue::new_raw(1_u8),
    ]);

    fn could_be_target_internal<F: Fn(CompoundType) -> Ordering>(
        &self,
        cmp_target_ordering: F,
    ) -> bool {
        CompoundType::iter()
            .zip(self.0.into_iter().zip(Self::TARGET.0))
            .all(
                |(compound_type, (self_compound_value, other_compound_value))| {
                    !self_compound_value.is_valid()
                        || self_compound_value.cmp(&other_compound_value)
                            == cmp_target_ordering(compound_type)
                },
            )
    }

    fn could_be_target(&self) -> bool {
        self.could_be_target_internal(CompoundType::cmp_target_ordering)
    }

    fn could_be_target_given_outdated_retroencabulator(&self) -> bool {
        self.could_be_target_internal(
            CompoundType::cmp_target_ordering_given_outdated_retroencabulator,
        )
    }
}

impl Parse for AuntSue {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut aunt_sue: Self = Self::default();

        let input: &str = verify(
            separated_list1(tag(", "), |input| {
                map_opt(
                    tuple((CompoundType::parse, tag(": "), parse_integer::<usize>)),
                    |(compound_type, _, compount_value)| {
                        let compound_value_mut: &mut CompoundValue =
                            &mut aunt_sue.0[compound_type as usize];

                        (!compound_value_mut.is_valid()).then(|| {
                            *compound_value_mut = compount_value.into();
                        })
                    },
                )(input)
            }),
            |separated_list: &Vec<()>| separated_list.len() <= CompoundType::COUNT,
        )(input)?
        .0;

        Ok((input, aunt_sue))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(ArrayVec<AuntSue, { Self::AUNT_SUE_CAPACITY }>);

impl Solution {
    const AUNT_SUE_COUNT: usize = 500_usize;

    // The first Aunt Sue is the target, so that the indices line up with the numbers.
    const AUNT_SUE_CAPACITY: usize = Self::AUNT_SUE_COUNT + 1_usize;

    fn find_target_aunt_sue_index_internal<F: Fn(&AuntSue) -> bool>(
        &self,
        could_be_target: F,
    ) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .skip(1_usize)
            .find_map(|(aunt_sue_index, aunt_sue)| {
                could_be_target(aunt_sue).then_some(aunt_sue_index)
            })
    }

    fn find_target_aunt_sue_index(&self) -> Option<usize> {
        self.find_target_aunt_sue_index_internal(AuntSue::could_be_target)
    }

    fn find_target_aunt_sue_index_given_outdated_retroencabulator(&self) -> Option<usize> {
        self.find_target_aunt_sue_index_internal(
            AuntSue::could_be_target_given_outdated_retroencabulator,
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self(ArrayVec::new());

        solution.0.push(AuntSue::TARGET);

        let input: &str = verify(
            separated_list1(line_ending, |input| {
                let expected_sue_index: usize = solution.0.len();

                verify(success(()), |_| !solution.0.is_full())(input)?;

                let (input, (_, _, _, aunt_sue)): (&str, (_, _, _, AuntSue)) =
                    tuple((
                        tag("Sue "),
                        verify(parse_integer::<usize>, |&real_sue_index| {
                            real_sue_index == expected_sue_index
                        }),
                        tag(": "),
                        AuntSue::parse,
                    ))(input)?;

                solution.0.push(aunt_sue);

                Ok((input, ()))
            }),
            |separated_list: &Vec<()>| separated_list.len() == Self::AUNT_SUE_COUNT,
        )(input)?
        .0;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// Difficult to write tests for this one.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_target_aunt_sue_index());
    }

    /// I appreciate the retroencabulator reference.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_target_aunt_sue_index_given_outdated_retroencabulator());
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
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
