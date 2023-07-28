use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{digit1, line_ending, multispace0},
        combinator::{map, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{delimited, separated_pair, terminated, tuple},
        Err, IResult,
    },
    std::{ops::Add, str::FromStr},
};

#[derive(Clone, Copy, Default, PartialEq)]
struct Path {
    bits: u32,
    len: u32,
}

impl Path {
    fn new(path: Self, right: bool) -> Self {
        let mut path: Self = path;

        path.push(right);

        path
    }

    fn peek(&self) -> bool {
        (self.bits & (1_u32 << self.len - 1_u32)) != 0_u32
    }

    fn push(&mut self, right: bool) {
        self.bits |= (right as u32) << self.len;
        self.len += 1_u32;
    }

    fn pop(&mut self) -> bool {
        self.len -= 1_u32;

        let mask: u32 = 1_u32 << self.len;
        let right: bool = (self.bits & mask) != 0_u32;

        self.bits &= !mask;

        right
    }

    fn pop_front(&mut self) -> bool {
        let right: bool = (self.bits & 1_u32) != 0_u32;

        self.bits >>= 1_u32;
        self.len -= 1_u32;

        right
    }
}

type Pair = [Element; 2_usize];

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug)]
enum Element {
    RegularNumber(u32),
    Pair(Box<Pair>),
}

impl Element {
    const EXPLODE_PATH_LEN_THRESHOLD: u32 = 4_u32;
    const SPLIT_REGULAR_NUMBER_THRESHOLD: u32 = 10_u32;

    fn try_find_pair_to_explode(&self, path: Path) -> Option<Path> {
        match self {
            Self::RegularNumber(_) => None,
            Self::Pair(pair) => match pair.as_ref() {
                [Self::RegularNumber(_), Self::RegularNumber(_)]
                    if path.len >= Self::EXPLODE_PATH_LEN_THRESHOLD =>
                {
                    Some(path)
                }
                [left, right] => left
                    .try_find_pair_to_explode(Path::new(path, false))
                    .or_else(|| right.try_find_pair_to_explode(Path::new(path, true))),
            },
        }
    }

    fn try_find_regular_number_to_split(&self, path: Path) -> Option<Path> {
        match self {
            Self::RegularNumber(regular_number)
                if *regular_number >= Self::SPLIT_REGULAR_NUMBER_THRESHOLD =>
            {
                Some(path)
            }
            Self::RegularNumber(_) => None,
            Self::Pair(pair) => [false, true].into_iter().find_map(|right| {
                let path: Path = Path::new(path, right);

                pair[path.peek() as usize].try_find_regular_number_to_split(path)
            }),
        }
    }

    fn get_regular_number(&self) -> Option<&u32> {
        match self {
            Self::RegularNumber(regular_number) => Some(regular_number),
            _ => None,
        }
    }

    fn get_regular_number_mut(&mut self) -> Option<&mut u32> {
        match self {
            Self::RegularNumber(regular_number) => Some(regular_number),
            _ => None,
        }
    }

    fn get_pair(&self) -> Option<&Pair> {
        match self {
            Self::Pair(pair) => Some(pair),
            _ => None,
        }
    }

    fn get_pair_mut(&mut self) -> Option<&mut Pair> {
        match self {
            Self::Pair(pair) => Some(pair),
            _ => None,
        }
    }

    fn get(&self, path: Path) -> Option<&Self> {
        if path.len == 0_u32 {
            Some(self)
        } else if let Some(pair) = self.get_pair() {
            let mut path: Path = path;
            let right: bool = path.pop_front();

            pair[right as usize].get(path)
        } else {
            None
        }
    }

    fn get_mut(&mut self, path: Path) -> Option<&mut Self> {
        if path.len == 0_u32 {
            Some(self)
        } else if let Some(pair) = self.get_pair_mut() {
            let mut path: Path = path;
            let right: bool = path.pop_front();

            pair[right as usize].get_mut(path)
        } else {
            None
        }
    }

    fn magnitude(&self) -> u32 {
        match self {
            Self::RegularNumber(regular_number) => *regular_number as u32,
            Self::Pair(pair) => {
                3_u32 * pair[0_usize].magnitude() + 2_u32 * pair[1_usize].magnitude()
            }
        }
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((Self::parse_regular_number, Self::parse_pair))(input)
    }

    fn parse_regular_number<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map_res(digit1, u32::from_str), Self::RegularNumber)(input)
    }

    fn parse_pair<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            delimited(
                tag("["),
                separated_pair(Self::parse, tuple((tag(","), multispace0)), Self::parse),
                tag("]"),
            ),
            |(left, right)| Self::Pair(Box::new([left, right])),
        )(input)
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug)]
struct SnailfishNumber(Element);

impl SnailfishNumber {
    fn reduce(&mut self) {
        while self.try_explode() || self.try_split() {}
    }

    fn try_explode(&mut self) -> bool {
        self.0
            .try_find_pair_to_explode(Path::default())
            .map(|path: Path| {
                for right in [false, true] {
                    let regular_number_path: Path = Path::new(path, right);
                    let regular_number: u32 = self
                        .0
                        .get(regular_number_path)
                        .and_then(Element::get_regular_number)
                        .copied()
                        .unwrap();

                    if let Some(next_regular_number_path) =
                        self.get_next_regular_number(regular_number_path, right)
                    {
                        *self
                            .0
                            .get_mut(next_regular_number_path)
                            .and_then(Element::get_regular_number_mut)
                            .unwrap() += regular_number;
                    }
                }

                *self.0.get_mut(path).unwrap() = Element::RegularNumber(0_u32);

                true
            })
            .unwrap_or_default()
    }

    fn try_split(&mut self) -> bool {
        self.0
            .try_find_regular_number_to_split(Path::default())
            .map(|path: Path| {
                let element: &mut Element = self.0.get_mut(path).unwrap();
                let regular_number: u32 = *element.get_regular_number().unwrap();

                *element = Element::Pair(Box::new([
                    Element::RegularNumber(regular_number / 2_u32),
                    Element::RegularNumber((regular_number + 1_u32) / 2_u32),
                ]));

                true
            })
            .unwrap_or_default()
    }

    fn get_next_regular_number(&self, path: Path, right: bool) -> Option<Path> {
        let mut path: Path = path;

        // First, walk up the tree until we can walk over
        while path.len > 0_u32 && (path.peek() == right) {
            path.pop();
        }

        if path.len == 0_u32 {
            None
        } else {
            // Walk over once
            path.pop();
            path.push(right);

            // Walk down the tree until we've reached a leaf
            while self
                .0
                .get(path)
                .and_then(Element::get_regular_number)
                .is_none()
            {
                path.push(!right);
            }

            Some(path)
        }
    }

    fn magnitude(&self) -> u32 {
        self.0.magnitude()
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Element::parse_pair, Self)(input)
    }
}

impl Add for SnailfishNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut sum: Self = Self(Element::Pair(Box::new([self.0, rhs.0])));

        sum.reduce();

        sum
    }
}

impl<'i> TryFrom<&'i str> for SnailfishNumber {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<SnailfishNumber>);

impl Solution {
    fn sum(&self) -> Option<SnailfishNumber> {
        if self.0.is_empty() {
            None
        } else {
            let mut iter = self.0.iter();
            let mut sum: SnailfishNumber = iter.next().unwrap().clone();

            for snailfish_number in iter {
                sum = sum + snailfish_number.clone();
            }

            Some(sum)
        }
    }

    fn magnitude_of_sum(&self) -> Option<u32> {
        self.sum().as_ref().map(SnailfishNumber::magnitude)
    }

    fn maximum_magnitude(&self) -> Option<u32> {
        (0_usize..self.0.len())
            .flat_map(|left_index| {
                (0_usize..self.0.len()).map(move |right_index| (left_index, right_index))
            })
            .filter_map(|(left_index, right_index)| {
                if left_index != right_index {
                    Some((self.0[left_index].clone() + self.0[right_index].clone()).magnitude())
                } else {
                    None
                }
            })
            .max()
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(SnailfishNumber::parse, opt(line_ending))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let sum: Option<SnailfishNumber> = self.sum();

            dbg!(&sum);
            dbg!(sum.as_ref().map(SnailfishNumber::magnitude));
        } else {
            dbg!(self.magnitude_of_sum());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.maximum_magnitude());
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
    use {
        super::*,
        std::{stringify, sync::OnceLock},
    };

    macro_rules! pair {
        [$left:literal,$right:literal] => { Element::Pair(Box::new([
            Element::RegularNumber($left),
            Element::RegularNumber($right)
        ])) };
        [$left:literal,[$right_left:tt,$right_right:tt]] => { Element::Pair(Box::new([
            Element::RegularNumber($left),
            pair![$right_left,$right_right]
        ])) };
        [[$left_left:tt,$left_right:tt],$right:literal] => { Element::Pair(Box::new([
            pair![$left_left,$left_right],
            Element::RegularNumber($right),
        ])) };
        [[$left_left:tt,$left_right:tt],[$right_left:tt,$right_right:tt]] => { Element::Pair(Box::new([
            pair![$left_left,$left_right],
            pair![$right_left,$right_right]
        ])) };
    }

    macro_rules! snailfish_number {
        [$left:tt,$right:tt] => { SnailfishNumber(pair![$left,$right]) }
    }

    macro_rules! snailfish_number_try_from_str {
        [ $( [$left:tt,$right:tt], )* ] => {
            const PAIR_TRY_FROM_STR_STRS: &[&str] = &[ $(
                stringify!([$left, $right]),
            )* ];

            fn snailfish_number_try_from_str_snailfish_numbers() -> &'static [SnailfishNumber] {
                static ONCE_LOCK: OnceLock<Vec<SnailfishNumber>> = OnceLock::new();

                &ONCE_LOCK.get_or_init(|| vec![ $(
                    snailfish_number![$left, $right],
                )* ])
            }
        };
    }

    snailfish_number_try_from_str![
        [1, 2],
        [[1, 2], 3],
        [9, [8, 7]],
        [[1, 9], [8, 5]],
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 9],
        [[[9, [3, 8]], [[0, 9], 6]], [[[3, 7], [4, 9]], 3]],
        [
            [[[1, 3], [5, 3]], [[1, 3], [8, 7]]],
            [[[4, 9], [6, 9]], [[8, 2], [7, 3]]]
        ],
    ];

    macro_rules! snailfish_number_tuples {
        [ $( ( $( [$left:tt,$right:tt] ),+ ), )* ] => { [ $(
            ( $(
                snailfish_number![$left, $right],
            )+ ),
        )* ] }
    }

    macro_rules! solution {
        [ $( [$left:tt,$right:tt] ),* $(,)? ] => { Solution(vec![ $(
            snailfish_number![$left, $right],
        )* ]) }
    }

    #[test]
    fn test_snailfish_number_try_from_str() {
        for (pair_str, pair) in PAIR_TRY_FROM_STR_STRS
            .iter()
            .copied()
            .zip(snailfish_number_try_from_str_snailfish_numbers().iter())
        {
            assert_eq!(pair_str.try_into().as_ref(), Ok(pair));
        }
    }

    #[test]
    fn test_snailfish_number_try_explode() {
        for (mut initial, exploded) in snailfish_number_tuples![
            ([[[[[9, 8], 1], 2], 3], 4], [[[[0, 9], 2], 3], 4]),
            ([7, [6, [5, [4, [3, 2]]]]], [7, [6, [5, [7, 0]]]]),
            ([[6, [5, [4, [3, 2]]]], 1], [[6, [5, [7, 0]]], 3]),
            (
                [[3, [2, [1, [7, 3]]]], [6, [5, [4, [3, 2]]]]],
                [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]]
            ),
            (
                [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]],
                [[3, [2, [8, 0]]], [9, [5, [7, 0]]]]
            ),
            (
                [[[[[4, 3], 4], 4], [7, [[8, 4], 9]]], [1, 1]],
                [[[[0, 7], 4], [7, [[8, 4], 9]]], [1, 1]]
            ),
            (
                [[[[0, 7], 4], [7, [[8, 4], 9]]], [1, 1]],
                [[[[0, 7], 4], [15, [0, 13]]], [1, 1]]
            ),
            (
                [[[[0, 7], 4], [[7, 8], [0, [6, 7]]]], [1, 1]],
                [[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]]
            ),
        ] {
            assert!(initial.try_explode());
            assert_eq!(initial, exploded);
        }
    }

    #[test]
    fn test_snailfish_number_try_split() {
        for (mut initial, split) in snailfish_number_tuples![
            (
                [[[[0, 7], 4], [15, [0, 13]]], [1, 1]],
                [[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]]
            ),
            (
                [[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]],
                [[[[0, 7], 4], [[7, 8], [0, [6, 7]]]], [1, 1]]
            ),
        ] {
            assert!(initial.try_split());
            assert_eq!(initial, split);
        }
    }

    #[test]
    fn test_snailfish_number_add() {
        for (left, right, sum) in snailfish_number_tuples![
            (
                [[[[4, 3], 4], 4], [7, [[8, 4], 9]]],
                [1, 1],
                [[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]]
            ),
            (
                [[[0, [4, 5]], [0, 0]], [[[4, 5], [2, 6]], [9, 5]]],
                [7, [[[3, 7], [4, 3]], [[6, 3], [8, 8]]]],
                [
                    [[[4, 0], [5, 4]], [[7, 7], [6, 0]]],
                    [[8, [7, 7]], [[7, 9], [5, 0]]]
                ]
            ),
            (
                [
                    [[[4, 0], [5, 4]], [[7, 7], [6, 0]]],
                    [[8, [7, 7]], [[7, 9], [5, 0]]]
                ],
                [[2, [[0, 8], [3, 4]]], [[[6, 7], 1], [7, [1, 6]]]],
                [
                    [[[6, 7], [6, 7]], [[7, 7], [0, 7]]],
                    [[[8, 7], [7, 7]], [[8, 8], [8, 0]]]
                ]
            ),
            (
                [
                    [[[6, 7], [6, 7]], [[7, 7], [0, 7]]],
                    [[[8, 7], [7, 7]], [[8, 8], [8, 0]]]
                ],
                [
                    [[[2, 4], 7], [6, [0, 5]]],
                    [[[6, 8], [2, 8]], [[2, 1], [4, 5]]]
                ],
                [
                    [[[7, 0], [7, 7]], [[7, 7], [7, 8]]],
                    [[[7, 7], [8, 8]], [[7, 7], [8, 7]]]
                ]
            ),
            (
                [
                    [[[7, 0], [7, 7]], [[7, 7], [7, 8]]],
                    [[[7, 7], [8, 8]], [[7, 7], [8, 7]]]
                ],
                [7, [5, [[3, 8], [1, 4]]]],
                [
                    [[[7, 7], [7, 8]], [[9, 5], [8, 7]]],
                    [[[6, 8], [0, 8]], [[9, 9], [9, 0]]]
                ]
            ),
            (
                [
                    [[[7, 7], [7, 8]], [[9, 5], [8, 7]]],
                    [[[6, 8], [0, 8]], [[9, 9], [9, 0]]]
                ],
                [[2, [2, 2]], [8, [8, 1]]],
                [
                    [[[6, 6], [6, 6]], [[6, 0], [6, 7]]],
                    [[[7, 7], [8, 9]], [8, [8, 1]]]
                ]
            ),
            (
                [
                    [[[6, 6], [6, 6]], [[6, 0], [6, 7]]],
                    [[[7, 7], [8, 9]], [8, [8, 1]]]
                ],
                [2, 9],
                [[[[6, 6], [7, 7]], [[0, 7], [7, 7]]], [[[5, 5], [5, 6]], 9]]
            ),
            (
                [[[[6, 6], [7, 7]], [[0, 7], [7, 7]]], [[[5, 5], [5, 6]], 9]],
                [1, [[[9, 3], 9], [[9, 0], [0, 7]]]],
                [
                    [[[7, 8], [6, 7]], [[6, 8], [0, 8]]],
                    [[[7, 7], [5, 0]], [[5, 5], [5, 6]]]
                ]
            ),
            (
                [
                    [[[7, 8], [6, 7]], [[6, 8], [0, 8]]],
                    [[[7, 7], [5, 0]], [[5, 5], [5, 6]]]
                ],
                [[[5, [7, 4]], 7], 1],
                [[[[7, 7], [7, 7]], [[8, 7], [8, 7]]], [[[7, 0], [7, 7]], 9]]
            ),
            (
                [[[[7, 7], [7, 7]], [[8, 7], [8, 7]]], [[[7, 0], [7, 7]], 9]],
                [[[[4, 2], 2], 6], [8, 7]],
                [
                    [[[8, 7], [7, 7]], [[8, 6], [7, 7]]],
                    [[[0, 7], [6, 6]], [8, 7]]
                ]
            ),
        ] {
            assert_eq!(left + right, sum);
        }
    }

    #[test]
    fn test_solution_sum() {
        for (solution, sum) in [
            (
                solution![[1, 1], [2, 2], [3, 3], [4, 4]],
                snailfish_number![[[[1, 1], [2, 2]], [3, 3]], [4, 4]],
            ),
            (
                solution![[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                snailfish_number![[[[3, 0], [5, 3]], [4, 4]], [5, 5]],
            ),
            (
                solution![[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                snailfish_number![[[[5, 0], [7, 4]], [5, 5]], [6, 6]],
            ),
            (
                solution![
                    [[[0, [4, 5]], [0, 0]], [[[4, 5], [2, 6]], [9, 5]]],
                    [7, [[[3, 7], [4, 3]], [[6, 3], [8, 8]]]],
                    [[2, [[0, 8], [3, 4]]], [[[6, 7], 1], [7, [1, 6]]]],
                    [
                        [[[2, 4], 7], [6, [0, 5]]],
                        [[[6, 8], [2, 8]], [[2, 1], [4, 5]]]
                    ],
                    [7, [5, [[3, 8], [1, 4]]]],
                    [[2, [2, 2]], [8, [8, 1]]],
                    [2, 9],
                    [1, [[[9, 3], 9], [[9, 0], [0, 7]]]],
                    [[[5, [7, 4]], 7], 1],
                    [[[[4, 2], 2], 6], [8, 7]],
                ],
                snailfish_number![
                    [[[8, 7], [7, 7]], [[8, 6], [7, 7]]],
                    [[[0, 7], [6, 6]], [8, 7]]
                ],
            ),
            (
                solution![
                    [[[0, [5, 8]], [[1, 7], [9, 6]]], [[4, [1, 2]], [[1, 4], 2]]],
                    [[[5, [2, 8]], 4], [5, [[9, 9], 0]]],
                    [6, [[[6, 2], [5, 6]], [[7, 6], [4, 7]]]],
                    [[[6, [0, 7]], [0, 9]], [4, [9, [9, 0]]]],
                    [[[7, [6, 4]], [3, [1, 3]]], [[[5, 5], 1], 9]],
                    [[6, [[7, 3], [3, 2]]], [[[3, 8], [5, 7]], 4]],
                    [[[[5, 4], [7, 7]], 8], [[8, 3], 8]],
                    [[9, 3], [[9, 9], [6, [4, 9]]]],
                    [[2, [[7, 7], 7]], [[5, 8], [[9, 3], [0, 2]]]],
                    [[[[5, 2], 5], [8, [3, 7]]], [[5, [7, 5]], [4, 4]]],
                ],
                snailfish_number![
                    [[[6, 6], [7, 6]], [[7, 7], [7, 0]]],
                    [[[7, 7], [7, 7]], [[7, 8], [9, 9]]]
                ],
            ),
        ] {
            assert_eq!(solution.sum(), Some(sum));
        }
    }

    #[test]
    fn test_element_magnitude() {
        for (pair, magnitude) in [
            (pair![9, 1], 29_u32),
            (pair![1, 9], 21_u32),
            (pair![[9, 1], [1, 9]], 129_u32),
            (pair![[1, 2], [[3, 4], 5]], 143_u32),
            (pair![[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]], 1384_u32),
            (pair![[[[1, 1], [2, 2]], [3, 3]], [4, 4]], 445_u32),
            (pair![[[[3, 0], [5, 3]], [4, 4]], [5, 5]], 791_u32),
            (pair![[[[5, 0], [7, 4]], [5, 5]], [6, 6]], 1137_u32),
            (
                pair![
                    [[[8, 7], [7, 7]], [[8, 6], [7, 7]]],
                    [[[0, 7], [6, 6]], [8, 7]]
                ],
                3488_u32,
            ),
            (
                pair![
                    [[[6, 6], [7, 6]], [[7, 7], [7, 0]]],
                    [[[7, 7], [7, 7]], [[7, 8], [9, 9]]]
                ],
                4140_u32,
            ),
        ] {
            assert_eq!(pair.magnitude(), magnitude);
        }
    }

    #[test]
    fn test_solution_maximum_magnitude() {
        assert_eq!(
            solution![
                [[[0, [5, 8]], [[1, 7], [9, 6]]], [[4, [1, 2]], [[1, 4], 2]]],
                [[[5, [2, 8]], 4], [5, [[9, 9], 0]]],
                [6, [[[6, 2], [5, 6]], [[7, 6], [4, 7]]]],
                [[[6, [0, 7]], [0, 9]], [4, [9, [9, 0]]]],
                [[[7, [6, 4]], [3, [1, 3]]], [[[5, 5], 1], 9]],
                [[6, [[7, 3], [3, 2]]], [[[3, 8], [5, 7]], 4]],
                [[[[5, 4], [7, 7]], 8], [[8, 3], 8]],
                [[9, 3], [[9, 9], [6, [4, 9]]]],
                [[2, [[7, 7], 7]], [[5, 8], [[9, 3], [0, 2]]]],
                [[[[5, 2], 5], [8, [3, 7]]], [[5, [7, 5]], [4, 4]]],
            ]
            .maximum_magnitude(),
            Some(3993_u32)
        );
    }
}
