use {
    crate::*,
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

/// Iterate over the section assignment pairs produced by a string slice
///
/// # Arguments
///
/// * `section_assignment_pairs_str` - The string slice to split parse section assignment pairs
///   from, with individual section assignment pairs delineated by `'\n'`
///
/// # Errors
///
/// If a section assignment pair fails to parse, an error message is printed describing the error.
fn iter_section_assignment_pairs(
    section_assignment_pairs_str: &str,
) -> impl Iterator<Item = SectionAssignmentPair> + '_ {
    section_assignment_pairs_str.split('\n').filter_map(
        |section_assignment_pair_str: &str| -> Option<SectionAssignmentPair> {
            match SectionAssignmentPair::try_from(section_assignment_pair_str) {
                Ok(section_assignment_pair) => Some(section_assignment_pair),
                Err(sa_pair_parse_error) => {
                    eprintln!(
                        "Failed to parse section assignment pair \"{}\": {:?}",
                        section_assignment_pair_str, sa_pair_parse_error
                    );

                    None
                }
            }
        },
    )
}

/// Counts the number of section assignment pairs where one section assignment fully contains the
/// other section assignment
///
/// # Arguments
///
/// * `section_assignment_pairs_str` - The string slice to split parse section assignment pairs
///   from, with individual section assignment pairs delineated by `'\n'`
fn count_sa_pairs_with_fully_contained_sas(section_assignment_pairs_str: &str) -> usize {
    iter_section_assignment_pairs(section_assignment_pairs_str)
        .filter(SectionAssignmentPair::one_fully_contains_other)
        .count()
}

/// Counts the number of section assignment pairs where the section assignments overlap
///
/// # Arguments
///
/// * `section_assignment_pairs_str` - The string slice to split parse section assignment pairs
///   from, with individual section assignment pairs delineated by `'\n'`
fn count_overlapping_sa_pairs(section_assignment_pairs_str: &str) -> usize {
    iter_section_assignment_pairs(section_assignment_pairs_str)
        .filter(SectionAssignmentPair::is_overlapping)
        .count()
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day4.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                println!(
                    "count_sa_pairs_with_fully_contained_sas == {}\n\
                    count_overlapping_sa_pairs == {}",
                    count_sa_pairs_with_fully_contained_sas(input),
                    count_overlapping_sa_pairs(input)
                );
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

pub struct Solution {}
