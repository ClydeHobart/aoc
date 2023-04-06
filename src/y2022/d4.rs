use {
    crate::*,
    nom::{
        character::complete::{line_ending, not_line_ending},
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::terminated,
        Err,
    },
    std::{
        num::ParseIntError,
        ops::RangeInclusive,
        str::{FromStr, Split},
    },
};

/// A section assignment, describing an inclusive range of section ID numbers
///
/// # Run-time Invariants
///
/// The `RangeInclusive<u8>` must not be empty
#[cfg_attr(test, derive(Debug, PartialEq))]
struct SectionAssignment(RangeInclusive<u8>);

impl SectionAssignment {
    /// Returns whether `self` fully contains a given other section assignment
    ///
    /// # Arguments
    ///
    /// * `other` - The other section assignment to check if whether it's fully contained by `self`
    fn fully_contains(&self, other: &Self) -> bool {
        self.0.contains(other.0.start()) && self.0.contains(other.0.end())
    }

    /// Returns whether `self` overlaps with a given other section assignment
    ///
    /// # Arguments
    ///
    /// * `other` - The other section assignment to check for overlapping against
    ///
    /// # Undefined Behavior
    ///
    /// If either `self` or `other` represent empty ranges, the run-time invariant for
    /// `SectionAssignment` is broken, and calling this function produces UB.
    fn overlaps(&self, other: &Self) -> bool {
        // Consider the following situations:
        //  * `!lhs && !rhs`: The invariant is broken, so this is UB that need not be handled
        //      * `self`:      S-----E
        //        `other`:   E---------S
        //
        //  * `lhs && !rhs`: `self` and `other` do not overlap
        //      * `self`:      S-----E
        //        `other`: S-E
        //
        //  * `!lhs && rhs`: `self` and `other` do not overlap
        //      * `self`:      S-----E
        //        `other`:             S-E
        //
        //  * `lhs && rhs`: `self` and `other` do not overlap
        //      * `self`:      S-----E
        //        `other`:           S-E
        //
        //      * `self`:      S-----E
        //        `other`:       S-----E
        //
        //      * `self`:      S-----E
        //        `other`:       S---E
        //
        //      * `self`:      S-----E
        //        `other`:       S-E
        //
        //      * `self`:      S-----E
        //        `other`:     S---E
        //
        //      * `self`:      S-----E
        //        `other`:   S-----E
        //
        //      * `self`:      S-----E
        //        `other`:   S-E
        self.0.end() >= other.0.start() && self.0.start() <= other.0.end()
    }
}

/// A possible error encountered while parsing a section assignment from a string slice
#[derive(Debug)]
enum SectionAssignmentParseError {
    /// There was no token for the start of the range
    NoStartToken,

    /// Parsing the token for the start of the range produced an error
    StartFailedToParse(ParseIntError),

    /// There was no token for the end of the range
    NoEndToken,

    /// Parsing the token for the end of the range produced an error
    EndFailedToParse(ParseIntError),

    /// There was an extra token after parsing the start and end of the range
    FoundExtraToken,

    /// The end of the range is less than the start of the range, which would produce an invalid
    /// `SectionAssignment`
    RangeWasEmpty,
}

impl TryFrom<&str> for SectionAssignment {
    type Error = SectionAssignmentParseError;

    /// Tries to parse a string slice into a `SectionAssignment`
    ///
    /// # Arguments
    ///
    /// * `section_assignment_str` - The string slice to attempt to parse into a `SectionAssignment`
    ///
    /// # Errors
    ///
    /// If an error is encountered, an `Err`-wrapped `SectionAssignmentParseError` is returned.
    fn try_from(section_assignment_str: &str) -> Result<Self, Self::Error> {
        use SectionAssignmentParseError as Error;

        let mut section_bound_iter: Split<char> = section_assignment_str.split('-');

        let start: u8 = u8::from_str(section_bound_iter.next().ok_or(Error::NoStartToken)?)
            .map_err(Error::StartFailedToParse)?;
        let end: u8 = u8::from_str(section_bound_iter.next().ok_or(Error::NoEndToken)?)
            .map_err(Error::EndFailedToParse)?;

        if section_bound_iter.next().is_some() {
            Err(Error::FoundExtraToken)
        } else if end < start {
            Err(Error::RangeWasEmpty)
        } else {
            Ok(SectionAssignment(start..=end))
        }
    }
}

/// A pair of section assignments
#[cfg_attr(test, derive(Debug, PartialEq))]
struct SectionAssignmentPair(SectionAssignment, SectionAssignment);

impl SectionAssignmentPair {
    /// Returns whether one of the section assignments is fully contained within the other section
    /// assignment
    fn one_fully_contains_other(&self) -> bool {
        self.0.fully_contains(&self.1) || self.1.fully_contains(&self.0)
    }

    /// Returns whether the two section assignments overlap
    fn is_overlapping(&self) -> bool {
        self.0.overlaps(&self.1)
    }
}

/// A possible error encountered while parsing a section assignment pair from a string slice
#[derive(Debug)]
enum SectionAssignmentPairParseError {
    /// There was no token for the first section assignment (field 0)
    NoZeroToken,

    /// Parsing the token for the first section assignment (field 0) produced an error
    ZeroFailedToParse(SectionAssignmentParseError),

    /// There was no token for the second section assignment (field 1)
    NoOneToken,

    /// Parsing the token for the second section assignment (field 1) produced an error
    OneFailedToParse(SectionAssignmentParseError),

    /// There was an extra token after parsing both section assignments
    ExtraTokenFound,
}

impl TryFrom<&str> for SectionAssignmentPair {
    type Error = SectionAssignmentPairParseError;

    /// Tries to parse a string slice into a `SectionAssignmentPair`
    ///
    /// # Arguments
    ///
    /// * `section_assignment_pair_str` - The string slice to attempt to parse into a
    ///   `SectionAssignmentPair`
    ///
    /// # Errors
    ///
    /// If an error is encountered, an `Err`-wrapped `SectionAssignmentPairParseError` is returned.
    fn try_from(section_assignment_pair_str: &str) -> Result<Self, Self::Error> {
        use SectionAssignmentPairParseError as Error;

        let mut section_assignment_iter: Split<char> = section_assignment_pair_str.split(',');

        let zero: SectionAssignment = section_assignment_iter
            .next()
            .ok_or(Error::NoZeroToken)?
            .try_into()
            .map_err(Error::ZeroFailedToParse)?;
        let one: SectionAssignment = section_assignment_iter
            .next()
            .ok_or(Error::NoOneToken)?
            .try_into()
            .map_err(Error::OneFailedToParse)?;

        if section_assignment_iter.next().is_none() {
            Ok(SectionAssignmentPair(zero, one))
        } else {
            Err(Error::ExtraTokenFound)
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<SectionAssignmentPair>);

impl Solution {
    fn count_sa_pairs_with_fully_contained_sas(&self) -> usize {
        self.0
            .iter()
            .filter(|sa_pair| sa_pair.one_fully_contains_other())
            .count()
    }

    fn count_overlapping_sa_pairs(&self) -> usize {
        self.0
            .iter()
            .filter(|sa_pair| sa_pair.is_overlapping())
            .count()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_sa_pairs_with_fully_contained_sas());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_overlapping_sa_pairs());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        let mut iter = iterator(
            input,
            terminated(
                map_res(not_line_ending, SectionAssignmentPair::try_from),
                opt(line_ending),
            ),
        );

        let result: Result<Self, Self::Error> = Ok(Self(iter.collect()));

        iter.finish()?;

        result
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const SA_PAIRS_STR: &str = concat!(
        "2-4,6-8\n",
        "2-3,4-5\n",
        "5-7,7-9\n",
        "2-8,3-7\n",
        "6-6,4-6\n",
        "2-6,4-8\n"
    );

    lazy_static! {
        static ref SOLUTION: Solution = new_solution();
    }

    fn new_solution() -> Solution {
        macro_rules! solution {
            [ $( ( $sa_0:expr, $sa_1:expr ), )* ] => {
                Solution(vec![ $(
                    SectionAssignmentPair(SectionAssignment($sa_0), SectionAssignment($sa_1)),
                )* ])
            };
        }

        solution![
            (2..=4, 6..=8),
            (2..=3, 4..=5),
            (5..=7, 7..=9),
            (2..=8, 3..=7),
            (6..=6, 4..=6),
            (2..=6, 4..=8),
        ]
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SA_PAIRS_STR), Ok(new_solution()));
    }

    #[test]
    fn test_count_sa_pairs_with_fully_contained_sas() {
        assert_eq!(SOLUTION.count_sa_pairs_with_fully_contained_sas(), 2_usize);
    }

    #[test]
    fn test_count_overlapping_sa_pairs() {
        assert_eq!(SOLUTION.count_overlapping_sa_pairs(), 4_usize);
    }
}
