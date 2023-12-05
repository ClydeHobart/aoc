use {
    crate::*,
    bitvec::BitArr,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::{fold_many0, many0},
        sequence::{terminated, tuple},
        Err, IResult,
    },
};

const MAX_NUMBERS: usize = 99_usize;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct NumberSet(BitArr!(for MAX_NUMBERS));

impl Parse for NumberSet {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        fold_many0(
            terminated(parse_integer::<usize>, space0),
            NumberSet::default,
            |mut number_set, number| {
                number_set.0.set(number, true);

                number_set
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Card {
    winning_numbers: NumberSet,
    held_numbers: NumberSet,
}

impl Card {
    fn winning_numbers(&self) -> NumberSet {
        NumberSet(self.winning_numbers.0 & self.held_numbers.0)
    }

    fn count_winning_numbers(&self) -> usize {
        self.winning_numbers().0.count_ones()
    }

    fn points(&self) -> usize {
        let winning_numbers: usize = self.count_winning_numbers();

        if winning_numbers == 0_usize {
            0_usize
        } else {
            1_usize << (winning_numbers - 1_usize)
        }
    }
}

impl Parse for Card {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, _): (&str, _) = tuple((
            tag("Card"),
            space0,
            parse_integer::<usize>,
            tag(":"),
            space0,
        ))(input)?;
        let (input, winning_numbers): (&str, NumberSet) = NumberSet::parse(input)?;
        let (input, _): (&str, _) = tuple((tag("|"), space0))(input)?;
        let (input, held_numbers): (&str, NumberSet) = NumberSet::parse(input)?;

        Ok((
            input,
            Self {
                winning_numbers,
                held_numbers,
            },
        ))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Card>);

impl Solution {
    fn sum_points(&self) -> usize {
        self.0.iter().map(|card| card.points()).sum()
    }

    fn card_counts(&self) -> Vec<usize> {
        let cards_len: usize = self.0.len();
        let mut card_counts: Vec<usize> = vec![1_usize; cards_len];

        for (card_index, winning_numbers) in self
            .0
            .iter()
            .map(|card| card.count_winning_numbers())
            .enumerate()
        {
            let additional_cards: usize = card_counts[card_index];
            let start: usize = card_index + 1_usize;
            let end: usize = cards_len.min(start + winning_numbers);

            for card_count in &mut card_counts[start..end] {
                *card_count += additional_cards;
            }
        }

        card_counts
    }

    fn sum_card_counts(&self) -> usize {
        self.card_counts().into_iter().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Card::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_points());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_card_counts());
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
        Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53\n\
        Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19\n\
        Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1\n\
        Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83\n\
        Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36\n\
        Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11\n";

    macro_rules! number_set {
        [ $( $number:expr, )* ] => { {
            let mut number_set: NumberSet = NumberSet::default();

            $(
                number_set.0.set($number, true);
            )*

            number_set
        } }
    }

    macro_rules! card {
        ( $( $winning_number:expr ),* ; $( $held_number:expr ),* ) => { {
            Card {
                winning_numbers: number_set![ $( $winning_number, )* ],
                held_numbers: number_set![ $( $held_number, )* ],
            }
        } }
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                card!(41, 48, 83, 86, 17; 83, 86,  6, 31, 17,  9, 48, 53),
                card!(13, 32, 20, 16, 61; 61, 30, 68, 82, 17, 32, 24, 19),
                card!( 1, 21, 53, 59, 44; 69, 82, 63, 72, 16, 21, 14,  1),
                card!(41, 92, 73, 84, 69; 59, 84, 76, 51, 58,  5, 54, 83),
                card!(87, 83, 26, 28, 32; 88, 30, 70, 12, 93, 22, 82, 36),
                card!(31, 18, 13, 56, 72; 74, 77, 10, 23, 35, 67, 36, 11),
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_winning_numbers() {
        for (real_winning_numbers, expected_winning_numbers) in solution()
            .0
            .iter()
            .map(|card| card.winning_numbers().0.iter_ones().collect::<Vec<usize>>())
            .zip([
                &[17_usize, 48_usize, 83_usize, 86_usize][..],
                &[32_usize, 61_usize][..],
                &[1_usize, 21_usize],
                &[84_usize][..],
                &[][..],
                &[][..],
            ])
        {
            assert_eq!(real_winning_numbers, expected_winning_numbers);
        }
    }

    #[test]
    fn test_points() {
        assert_eq!(
            solution()
                .0
                .iter()
                .map(|card| card.points())
                .collect::<Vec<usize>>(),
            vec![8_usize, 2_usize, 2_usize, 1_usize, 0_usize, 0_usize]
        );
    }

    #[test]
    fn test_sum_points() {
        assert_eq!(solution().sum_points(), 13_usize);
    }

    #[test]
    fn test_card_counts() {
        assert_eq!(
            solution().card_counts(),
            vec![1_usize, 2_usize, 4_usize, 8_usize, 14_usize, 1_usize]
        );
    }

    #[test]
    fn test_sum_card_counts() {
        assert_eq!(solution().sum_card_counts(), 30_usize);
    }
}
