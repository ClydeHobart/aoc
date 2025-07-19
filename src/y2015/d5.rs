use {
    crate::*,
    nom::{
        bytes::complete::{tag, take_while1, take_while_m_n},
        character::complete::{anychar, line_ending, satisfy},
        combinator::map,
        error::Error,
        multi::{many_till, separated_list0},
        sequence::tuple,
        Err, IResult,
    },
};

/* --- Day 5: Doesn't He Have Intern-Elves For This? ---

Santa needs help figuring out which strings in his text file are naughty or nice.

A nice string is one with all of the following properties:

    It contains at least three vowels (aeiou only), like aei, xazegov, or aeiouaeiouaeiou.
    It contains at least one letter that appears twice in a row, like xx, abcdde (dd), or aabbccdd (aa, bb, cc, or dd).
    It does not contain the strings ab, cd, pq, or xy, even if they are part of one of the other requirements.

For example:

    ugknbfddgicrmopn is nice because it has at least three vowels (u...i...o...), a double letter (...dd...), and none of the disallowed substrings.
    aaa is nice because it has at least three vowels and a double letter, even though the letters used by different rules overlap.
    jchzalrnumimnmhp is naughty because it has no double letter.
    haegwjzuvuyypxyu is naughty because it contains the string xy.
    dvszwmarrgswjxmb is naughty because it contains only one vowel.

How many strings are nice?

--- Part Two ---

Realizing the error of his ways, Santa has switched to a better model of determining whether a string is naughty or nice. None of the old rules apply, as they are all clearly ridiculous.

Now, a nice string is one with all of the following properties:

    It contains a pair of any two letters that appears at least twice in the string without overlapping, like xyxy (xy) or aabcdefgaa (aa), but not like aaa (aa, but it overlaps).
    It contains at least one letter which repeats with exactly one letter between them, like xyx, abcdefeghi (efe), or even aaa.

For example:

    qjhvhtzxzqqjkmpb is nice because is has a pair that appears twice (qj) and a letter that repeats with exactly one letter between them (zxz).
    xxyxx is nice because it has a pair that appears twice and a letter that repeats with one between, even though the letters used by each rule overlap.
    uurcxstgmygtbstg is naughty because it has a pair (tg) but no repeat with a single letter between them.
    ieodomkazucvgmuy is naughty because it has a repeating letter with one between (odo), but no pair that appears twice.

How many strings are nice under these new rules? */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<String>);

impl Solution {
    const VOWELS: &'static str = "aeiou";
    const DENIED_PAIRS: &'static [&'static str] = &["ab", "cd", "pq", "xy"];

    fn contains_at_least_three_vowels(string: &str) -> bool {
        string.chars().filter(|&c| Self::VOWELS.contains(c)).count() >= 3_usize
    }

    fn contains_at_least_one_double_letter(string: &str) -> bool {
        string
            .as_bytes()
            .windows(2_usize)
            .any(|pair| pair.first().unwrap() == pair.last().unwrap())
    }

    fn does_not_contain_denied_pair(string: &str) -> bool {
        Self::DENIED_PAIRS
            .iter()
            .copied()
            .all(|denied_pair| !string.contains(denied_pair))
    }

    fn is_q1_nice(string: &str) -> bool {
        Self::contains_at_least_three_vowels(string)
            && Self::contains_at_least_one_double_letter(string)
            && Self::does_not_contain_denied_pair(string)
    }

    fn parse_with_optional_ignored_prefix<'i, F: FnMut(&'i str) -> IResult<&'i str, &'i str>>(
        f: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, &'i str> {
        map(many_till(map(anychar, |_| ()), f), |(_, value)| value)
    }

    fn parse_non_overlapping_repeating_pair<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        let (input, pair): (&str, &str) = take_while_m_n(2_usize, 2_usize, |_| true)(input)?;

        Self::parse_with_optional_ignored_prefix(tag(pair))(input)
    }

    fn contains_non_overlapping_repeating_pair(string: &str) -> bool {
        Self::parse_with_optional_ignored_prefix(Self::parse_non_overlapping_repeating_pair)(string)
            .is_ok()
    }

    fn parse_repeating_letter_with_one_letter_between<'i>(
        input: &'i str,
    ) -> IResult<&'i str, &'i str> {
        let original_input: &str = input;
        let (input, expected_char): (&str, char) = anychar(input)?;
        let input: &str =
            tuple((anychar, satisfy(|actual_char| actual_char == expected_char)))(input)?.0;

        Ok((input, &original_input[..1_usize]))
    }

    fn contains_repeating_letter_with_one_letter_between(string: &str) -> bool {
        Self::parse_with_optional_ignored_prefix(
            Self::parse_repeating_letter_with_one_letter_between,
        )(string)
        .is_ok()
    }

    fn is_q2_nice(string: &str) -> bool {
        Self::contains_non_overlapping_repeating_pair(string)
            && Self::contains_repeating_letter_with_one_letter_between(string)
    }

    fn count_q1_nice_strings(&self) -> usize {
        self.0
            .iter()
            .filter(|&string| Self::is_q1_nice(string))
            .count()
    }

    fn count_q2_nice_strings(&self) -> usize {
        self.0
            .iter()
            .filter(|&string| Self::is_q2_nice(string))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_list0(
                line_ending,
                map(take_while1(|c: char| c.is_ascii_lowercase()), String::from),
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I didn't realize there were multiple strings until surprisingly late into this question.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_q1_nice_strings());
    }

    /// I spent a fair amount of time trying to cook up a lazy evaluator for `.*<pattern>`, but then
    /// I found `nom`'s `many_till`.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_q2_nice_strings());
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        "aei\n\
        xazegov\n\
        aeiouaeiouaeiou\n",
        "xx\n\
        abcdde\n\
        aabbccdd\n",
        "ugknbfddgicrmopn\n\
        aaa\n\
        jchzalrnumimnmhp\n\
        haegwjzuvuyypxyu\n\
        dvszwmarrgswjxmb\n",
        "xyxy\n\
        aabcdefgaa\n\
        aaa\n",
        "xyx\n\
        abcdefeghi\n\
        aaa\n",
        "qjhvhtzxzqqjkmpb\n\
        xxyxx\n\
        uurcxstgmygtbstg\n\
        ieodomkazucvgmuy\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    "aei".into(),
                    "xazegov".into(),
                    "aeiouaeiouaeiou".into(),
                ]),
                Solution(vec!["xx".into(), "abcdde".into(), "aabbccdd".into()]),
                Solution(vec![
                    "ugknbfddgicrmopn".into(),
                    "aaa".into(),
                    "jchzalrnumimnmhp".into(),
                    "haegwjzuvuyypxyu".into(),
                    "dvszwmarrgswjxmb".into(),
                ]),
                Solution(vec!["xyxy".into(), "aabcdefgaa".into(), "aaa".into()]),
                Solution(vec!["xyx".into(), "abcdefeghi".into(), "aaa".into()]),
                Solution(vec![
                    "qjhvhtzxzqqjkmpb".into(),
                    "xxyxx".into(),
                    "uurcxstgmygtbstg".into(),
                    "ieodomkazucvgmuy".into(),
                ]),
            ]
        })[index]
    }

    fn test_solution_strings<F: Fn(&str) -> bool>(index: usize, f: F) -> Vec<bool> {
        solution(index).0.iter().map(|string| f(string)).collect()
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
    fn test_contains_at_least_three_vowels() {
        for (index, contains_at_least_three_vowels) in [
            vec![true, true, true],
            vec![false, false, false],
            vec![true, true, true, true, false],
            vec![false, true, true],
            vec![false, true, true],
            vec![false, false, false, true],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::contains_at_least_three_vowels),
                contains_at_least_three_vowels,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_contains_at_least_one_double_letter() {
        for (index, contains_at_least_one_double_letter) in [
            vec![false, false, false],
            vec![true, true, true],
            vec![true, true, false, true, true],
            vec![false, true, true],
            vec![false, false, true],
            vec![true, true, true, false],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::contains_at_least_one_double_letter),
                contains_at_least_one_double_letter,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_does_not_contain_denied_pair() {
        for (index, does_not_contain_denied_pair) in [
            vec![true, true, true],
            vec![true, false, false],
            vec![true, true, true, false, true],
            vec![false, false, true],
            vec![false, false, true],
            vec![true, false, true, true],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::does_not_contain_denied_pair),
                does_not_contain_denied_pair,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_is_q1_nice() {
        for (index, is_q1_nice) in [
            vec![false, false, false],
            vec![false, false, false],
            vec![true, true, false, false, false],
            vec![false, false, true],
            vec![false, false, true],
            vec![false, false, false, false],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::is_q1_nice),
                is_q1_nice,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_contains_non_overlapping_repeating_pair() {
        for (index, contains_non_overlapping_repeating_pair) in [
            vec![false, false, true],
            vec![false, false, false],
            vec![false, false, false, false, false],
            vec![true, true, false],
            vec![false, false, false],
            vec![true, true, true, false],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::contains_non_overlapping_repeating_pair),
                contains_non_overlapping_repeating_pair,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_contains_repeating_letter_with_one_letter_between() {
        for (index, contains_repeating_letter_with_one_letter_between) in [
            vec![false, false, false],
            vec![false, false, false],
            vec![false, true, true, true, false],
            vec![true, false, true],
            vec![true, true, true],
            vec![true, true, false, true],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(
                    index,
                    Solution::contains_repeating_letter_with_one_letter_between
                ),
                contains_repeating_letter_with_one_letter_between,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_is_q2_nice() {
        for (index, is_q2_nice) in [
            vec![false, false, false],
            vec![false, false, false],
            vec![false, false, false, false, false],
            vec![true, false, false],
            vec![false, false, false],
            vec![true, true, false, false],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                test_solution_strings(index, Solution::is_q2_nice),
                is_q2_nice,
                "index: {index}"
            );
        }
    }

    #[test]
    fn test_count_q1_nice_strings() {
        for (index, q1_nice_strings_count) in [0_usize, 0_usize, 2_usize, 1_usize, 1_usize, 0_usize]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).count_q1_nice_strings(),
                q1_nice_strings_count
            );
        }
    }

    #[test]
    fn test_count_q2_nice_strings() {
        for (index, q2_nice_strings_count) in [0_usize, 0_usize, 0_usize, 1_usize, 0_usize, 2_usize]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index).count_q2_nice_strings(),
                q2_nice_strings_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
