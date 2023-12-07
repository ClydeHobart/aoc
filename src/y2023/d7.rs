use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::{fold_many_m_n, many0},
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::{
        cmp::{Ordering, Reverse},
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
    },
};

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Default, Eq, PartialEq)]
    enum CardType {
        #[default]
        Two = TWO = b'2',
        Three = THREE = b'3',
        Four = FOUR = b'4',
        Five = FIVE = b'5',
        Six = SIX = b'6',
        Seven = SEVEN = b'7',
        Eight = EIGHT = b'8',
        Nine = NINE = b'9',
        Ten = TEN = b'T',
        Jack = JACK = b'J',
        Queen = QUEEN = b'Q',
        King = KING = b'K',
        Ace = ACE = b'A',
    }
}

impl CardType {
    fn value(self) -> u32 {
        match self {
            CardType::Two => 2_u32,
            CardType::Three => 3_u32,
            CardType::Four => 4_u32,
            CardType::Five => 5_u32,
            CardType::Six => 6_u32,
            CardType::Seven => 7_u32,
            CardType::Eight => 8_u32,
            CardType::Nine => 9_u32,
            CardType::Ten => 10_u32,
            CardType::Jack => 11_u32,
            CardType::Queen => 12_u32,
            CardType::King => 13_u32,
            CardType::Ace => 14_u32,
        }
    }
}

impl CardType
where
    Self: IsValidAscii,
{
    fn fmt_internal(&self, f: &mut Formatter<'_>) -> FmtResult {
        // SAFETY: Guaranteed by `Self: IsValidAscii`
        f.write_str(unsafe { from_utf8_unchecked(&[*self as u8]) })
    }

    fn joker_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Jack, Self::Jack) => Ordering::Equal,
            (Self::Jack, _) => Ordering::Less,
            (_, Self::Jack) => Ordering::Greater,
            (_, _) => self.cmp(other),
        }
    }
}

impl Debug for CardType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.fmt_internal(f)
    }
}

impl Ord for CardType {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().cmp(&other.value())
    }
}

impl PartialOrd for CardType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct Hand([CardType; Hand::CARDS]);

impl Hand {
    const CARDS: usize = 5_usize;

    fn hand_type(self) -> HandType {
        const CARD_TYPE_COUNTS_LEN: usize = CardType::STR.len();

        let default_card_value: usize = CardType::default().value() as usize;

        let mut card_type_counts: [u8; CARD_TYPE_COUNTS_LEN] = [0_u8; CARD_TYPE_COUNTS_LEN];

        for card_type in self.0 {
            card_type_counts[card_type.value() as usize - default_card_value] += 1_u8;
        }

        card_type_counts.sort_by_key(|card_type_count| Reverse(*card_type_count));

        match (card_type_counts[0_usize], card_type_counts[1_usize]) {
            (5_u8, 0_u8) => HandType::FiveOfAKind,
            (4_u8, 1_u8) => HandType::FourOfAKind,
            (3_u8, 2_u8) => HandType::FullHouse,
            (3_u8, 1_u8) => HandType::ThreeOfAKind,
            (2_u8, 2_u8) => HandType::TwoPair,
            (2_u8, 1_u8) => HandType::OnePair,
            (1_u8, 1_u8) => HandType::HighCard,
            _ => unreachable!(),
        }
    }

    fn joker_hand_type(self, hand_type: HandType) -> HandType {
        let joker_count: usize = self
            .0
            .iter()
            .filter(|card_type| **card_type == CardType::Jack)
            .count();

        match (joker_count, hand_type) {
            (0_usize, _) => hand_type,
            (1_usize, HandType::HighCard) => HandType::OnePair, // J2345 => 22345
            (1_usize, HandType::OnePair) => HandType::ThreeOfAKind, // J2234 => 22234
            (1_usize, HandType::TwoPair) => HandType::FullHouse, // J2233 => 22233
            (1_usize, HandType::ThreeOfAKind) => HandType::FourOfAKind, // J2223 => 22223
            (1_usize, HandType::FourOfAKind) => HandType::FiveOfAKind, // J2222 => 22222
            (2_usize, HandType::OnePair) => HandType::ThreeOfAKind, // JJ234 => 22234
            (2_usize, HandType::TwoPair) => HandType::FourOfAKind, // JJ223 => 22223
            (2_usize, HandType::FullHouse) => HandType::FiveOfAKind, // JJ222 => 22222
            (3_usize, HandType::ThreeOfAKind) => HandType::FourOfAKind, // JJJ23 => 22223
            (3_usize, HandType::FullHouse) => HandType::FiveOfAKind, // JJJ22 => 22222
            (4_usize, HandType::FourOfAKind) => HandType::FiveOfAKind, // JJJJ2 => 22222
            (5_usize, HandType::FiveOfAKind) => HandType::FiveOfAKind, // JJJJJ => 22222
            _ => unreachable!(),
        }
    }

    fn cmp_card_types(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }

    fn joker_cmp_card_types(&self, other: &Self) -> Ordering {
        self.0
            .iter()
            .zip(other.0.iter())
            .try_fold(Ordering::Equal, |_, (self_card_type, other_card_type)| {
                let ordering: Ordering = self_card_type.joker_cmp(other_card_type);
                match ordering {
                    Ordering::Equal => Ok(Ordering::Equal),
                    _ => Err(ordering),
                }
            })
            .map_or_else(|ordering| ordering, |ordering| ordering)
    }
}

impl Parse for Hand {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            fold_many_m_n(
                Self::CARDS,
                Self::CARDS,
                CardType::parse,
                || (Self::default(), 0_usize),
                |(mut hand, index), card_type| {
                    hand.0[index] = card_type;

                    (hand, index + 1_usize)
                },
            ),
            |(hand, _)| hand,
        )(input)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(u16)]
enum HandType {
    HighCard,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    FullHouse,
    FourOfAKind,
    FiveOfAKind,
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Eq, PartialEq)]
struct ListEntry {
    hand: Hand,
    hand_type: HandType,
    joker_hand_type: HandType,
    bid: u32,
}

impl ListEntry {
    fn joker_cmp(&self, other: &Self) -> Ordering {
        self.joker_hand_type
            .cmp(&other.joker_hand_type)
            .then_with(|| {
                self.hand
                    .joker_cmp_card_types(&other.hand)
                    .then_with(|| self.bid.cmp(&other.bid))
            })
    }
}

impl Ord for ListEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hand_type.cmp(&other.hand_type).then_with(|| {
            self.hand
                .cmp_card_types(&other.hand)
                .then_with(|| self.bid.cmp(&other.bid))
        })
    }
}

impl Parse for ListEntry {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(Hand::parse, tag(" "), parse_integer::<u32>),
            |(hand, bid)| {
                let hand_type: HandType = hand.hand_type();
                let joker_hand_type: HandType = hand.joker_hand_type(hand_type);

                Self {
                    hand,
                    hand_type,
                    joker_hand_type,
                    bid,
                }
            },
        )(input)
    }
}

impl PartialOrd for ListEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub struct Solution(Vec<ListEntry>);

impl Solution {
    fn sort(&mut self) {
        self.0.sort();
    }

    fn joker_sort(&mut self) {
        self.0
            .sort_by(|list_entry_a, list_entry_b| list_entry_a.joker_cmp(list_entry_b));
    }

    fn iter_ranked_list_entries(&self) -> impl Iterator<Item = (usize, &ListEntry)> + '_ {
        self.0
            .iter()
            .enumerate()
            .map(|(index, list_entry)| (index + 1_usize, list_entry))
    }

    fn iter_winnings(&self) -> impl Iterator<Item = u32> + '_ {
        self.iter_ranked_list_entries()
            .map(|(rank, list_entry)| rank as u32 * list_entry.bid)
    }

    fn total_winnings(&self) -> u32 {
        self.iter_winnings().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(ListEntry::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        self.sort();

        dbg!(self.total_winnings());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        self.joker_sort();

        dbg!(self.total_winnings());
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
        32T3K 765\n\
        T55J5 684\n\
        KK677 28\n\
        KTJJT 220\n\
        QQQJA 483\n";

    fn solution() -> &'static Solution {
        use CardType::{
            Ace as A, Five as N5, Jack as J, King as K, Queen as Q, Seven as N7, Six as N6,
            Ten as T, Three as N3, Two as N2,
        };

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                ListEntry {
                    hand: Hand([N3, N2, T, N3, K]),
                    hand_type: HandType::OnePair,
                    joker_hand_type: HandType::OnePair,
                    bid: 765_u32,
                },
                ListEntry {
                    hand: Hand([T, N5, N5, J, N5]),
                    hand_type: HandType::ThreeOfAKind,
                    joker_hand_type: HandType::FourOfAKind,
                    bid: 684_u32,
                },
                ListEntry {
                    hand: Hand([K, K, N6, N7, N7]),
                    hand_type: HandType::TwoPair,
                    joker_hand_type: HandType::TwoPair,
                    bid: 28_u32,
                },
                ListEntry {
                    hand: Hand([K, T, J, J, T]),
                    hand_type: HandType::TwoPair,
                    joker_hand_type: HandType::FourOfAKind,
                    bid: 220_u32,
                },
                ListEntry {
                    hand: Hand([Q, Q, Q, J, A]),
                    hand_type: HandType::ThreeOfAKind,
                    joker_hand_type: HandType::FourOfAKind,
                    bid: 483_u32,
                },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_sort() {
        let mut local_solution: Solution = solution().clone();

        local_solution.sort();

        assert_eq!(
            local_solution,
            Solution(vec![
                solution().0[0_usize].clone(),
                solution().0[3_usize].clone(),
                solution().0[2_usize].clone(),
                solution().0[1_usize].clone(),
                solution().0[4_usize].clone(),
            ])
        )
    }

    #[test]
    fn test_joker_sort() {
        let mut local_solution: Solution = solution().clone();

        local_solution.joker_sort();

        assert_eq!(
            local_solution,
            Solution(vec![
                solution().0[0_usize].clone(),
                solution().0[2_usize].clone(),
                solution().0[1_usize].clone(),
                solution().0[4_usize].clone(),
                solution().0[3_usize].clone(),
            ])
        )
    }

    #[test]
    fn test_iter_ranked_list_entries() {
        let mut local_solution: Solution = solution().clone();

        local_solution.sort();

        assert_eq!(
            local_solution
                .iter_ranked_list_entries()
                .collect::<Vec<(usize, &ListEntry)>>(),
            vec![
                (1_usize, &solution().0[0_usize]),
                (2_usize, &solution().0[3_usize]),
                (3_usize, &solution().0[2_usize]),
                (4_usize, &solution().0[1_usize]),
                (5_usize, &solution().0[4_usize]),
            ]
        )
    }

    #[test]
    fn test_total_winnings() {
        let mut solution: Solution = solution().clone();

        solution.sort();

        assert_eq!(solution.total_winnings(), 6440_u32);
    }
}
