use {
    crate::*,
    glam::IVec2,
    nom::{character::complete::satisfy, combinator::map, error::Error, Err, IResult},
};

/* --- Day 6: Signals and Noise ---

Something is jamming your communications with Santa. Fortunately, your signal is only partially jammed, and protocol in situations like this is to switch to a simple repetition code to get the message through.

In this model, the same message is sent repeatedly. You've recorded the repeating message signal (your puzzle input), but the data seems quite corrupted - almost too badly to recover. Almost.

All you need to do is figure out which character is most frequent for each position. For example, suppose you had recorded the following messages:

eedadn
drvtee
eandsr
raavrd
atevrs
tsrnev
sdttsa
rasrtv
nssdts
ntnada
svetve
tesnvt
vntsnd
vrdear
dvrsen
enarar

The most common character in the first column is e; in the second, a; in the third, s, and so on. Combining these characters returns the error-corrected message, easter.

Given the recording in your puzzle input, what is the error-corrected version of the message being sent?

--- Part Two ---

Of course, that would be the message - if you hadn't agreed to use a modified repetition code instead.

In this modified code, the sender instead transmits what looks like random data, but for each character, the character they actually want to send is slightly less likely than the others. Even after signal-jamming noise, you can look at the letter distributions in each column and choose the least common letter to reconstruct the original message.

In the above example, the least common character in the first column is a; in the second, d, and so on. Repeating this process for the remaining characters produces the original message, advent.

Given the recording in your puzzle input and this new decoding methodology, what is the original message that Santa is trying to send? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct MessageByte(u8);

impl Parse for MessageByte {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(|c| c.is_ascii_lowercase()), |c| Self(c as u8))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<MessageByte>);

impl Solution {
    fn iter_column(&self, x: i32) -> impl Iterator<Item = u8> + '_ {
        CellIter2D::until_boundary(&self.0, IVec2::new(x, 0_i32), Direction::South)
            .map(|pos| self.0.get(pos).unwrap().0)
    }

    fn letter_counts_for_column(&self, x: i32) -> LetterCounts {
        self.iter_column(x).into()
    }

    fn iter_most_common_letter(&self) -> impl Iterator<Item = u8> + '_ {
        (0_i32..self.0.dimensions().x).map(|x| self.letter_counts_for_column(x).0[0_usize].letter)
    }

    fn iter_least_common_present_letter(&self) -> impl Iterator<Item = u8> + '_ {
        (0_i32..self.0.dimensions().x).map(|x| {
            self.letter_counts_for_column(x)
                .0
                .into_iter()
                .rev()
                .filter(|letter_count| letter_count.count > 0_u8)
                .next()
                .unwrap()
                .letter
        })
    }

    fn most_common_letter_string(&self) -> String {
        self.iter_most_common_letter().map(char::from).collect()
    }

    fn least_common_present_letter_string(&self) -> String {
        self.iter_least_common_present_letter()
            .map(char::from)
            .collect()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.most_common_letter_string());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.least_common_present_letter_string());
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
        eedadn\n\
        drvtee\n\
        eandsr\n\
        raavrd\n\
        atevrs\n\
        tsrnev\n\
        sdttsa\n\
        rasrtv\n\
        nssdts\n\
        ntnada\n\
        svetve\n\
        tesnvt\n\
        vntsnd\n\
        vrdear\n\
        dvrsen\n\
        enarar\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        macro_rules! solution {
            ( $width:expr, $( [ $( $c:expr ),* ], )* ) => {
                Solution(Grid2D::try_from_cells_and_width(
                    vec![ $( $( MessageByte($c as u8), )* )* ],
                    $width
                ).unwrap())
            };
        }

        ONCE_LOCK.get_or_init(|| {
            solution!(
                6_usize,
                ['e', 'e', 'd', 'a', 'd', 'n'],
                ['d', 'r', 'v', 't', 'e', 'e'],
                ['e', 'a', 'n', 'd', 's', 'r'],
                ['r', 'a', 'a', 'v', 'r', 'd'],
                ['a', 't', 'e', 'v', 'r', 's'],
                ['t', 's', 'r', 'n', 'e', 'v'],
                ['s', 'd', 't', 't', 's', 'a'],
                ['r', 'a', 's', 'r', 't', 'v'],
                ['n', 's', 's', 'd', 't', 's'],
                ['n', 't', 'n', 'a', 'd', 'a'],
                ['s', 'v', 'e', 't', 'v', 'e'],
                ['t', 'e', 's', 'n', 'v', 't'],
                ['v', 'n', 't', 's', 'n', 'd'],
                ['v', 'r', 'd', 'e', 'a', 'r'],
                ['d', 'v', 'r', 's', 'e', 'n'],
                ['e', 'n', 'a', 'r', 'a', 'r'],
            )
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_most_common_letter_string() {
        assert_eq!(solution().most_common_letter_string(), "easter".to_owned());
    }

    #[test]
    fn test_least_common_present_letter_string() {
        assert_eq!(
            solution().least_common_present_letter_string(),
            "advent".to_owned()
        );
    }
}
