use {
    crate::*,
    nom::{
        bytes::complete::take_while,
        character::complete::{alpha0, anychar},
        combinator::{iterator, map, map_opt, map_res},
        error::Error,
        multi::many1_count,
        sequence::preceded,
        Err, IResult,
    },
    rayon::iter::{IntoParallelIterator, ParallelIterator},
    std::{
        collections::HashSet,
        fmt::{Display, Formatter, Result as FmtResult},
        ops::Range,
    },
};

/* --- Day 5: Alchemical Reduction ---

You've managed to sneak in to the prototype suit manufacturing lab. The Elves are making decent progress, but are still struggling with the suit's size reduction capabilities.

While the very latest in 1518 alchemical technology might have solved their problem eventually, you can do better. You scan the chemical composition of the suit's material and discover that it is formed by extremely long polymers (one of which is available as your puzzle input).

The polymer is formed by smaller units which, when triggered, react with each other such that two adjacent units of the same type and opposite polarity are destroyed. Units' types are represented by letters; units' polarity is represented by capitalization. For instance, r and R are units with the same type but opposite polarity, whereas r and s are entirely different types and do not react.

For example:

    In aA, a and A react, leaving nothing behind.
    In abBA, bB destroys itself, leaving aA. As above, this then destroys itself, leaving nothing.
    In abAB, no two adjacent units are of the same type, and so nothing happens.
    In aabAAB, even though aa and AA are of the same type, their polarities match, and so nothing happens.

Now, consider a larger example, dabAcCaCBAcCcaDA:

dabAcCaCBAcCcaDA  The first 'cC' is removed.
dabAaCBAcCcaDA    This creates 'Aa', which is removed.
dabCBAcCcaDA      Either 'cC' or 'Cc' are removed (the result is the same).
dabCBAcaDA        No further actions can be taken.

After all possible reactions, the resulting polymer contains 10 units.

How many units remain after fully reacting the polymer you scanned? (Note: in this puzzle and others, the input is large; if you copy/paste your input, make sure you get the whole thing.)

--- Part Two ---

Time to improve the polymer.

One of the unit types is causing problems; it's preventing the polymer from collapsing as much as it should. Your goal is to figure out which unit type is causing the most problems, remove all instances of it (regardless of polarity), fully react the remaining polymer, and measure its length.

For example, again using the polymer dabAcCaCBAcCcaDA from above:

    Removing all A/a units produces dbcCCBcCcD. Fully reacting this polymer produces dbCBcD, which has length 6.
    Removing all B/b units produces daAcCaCAcCcaDA. Fully reacting this polymer produces daCAcaDA, which has length 8.
    Removing all C/c units produces dabAaBAaDA. Fully reacting this polymer produces daDA, which has length 4.
    Removing all D/d units produces abAcCaCBAcCcaA. Fully reacting this polymer produces abCBAc, which has length 6.

In this example, removing all C/c units was best, producing the answer 4.

What is the length of the shortest polymer you can produce by removing all units of exactly one type and fully reacting the result? */

// There are 50000 characters in the user input. The index to the last element can successfully be
// stored in a `u16`.
type UnitRangeRangeIndex = u16;

// The number of active `UnitRange`s is monotonically decreasing.
type UnitRangeIndexRaw = u16;
type UnitRangeIndex = Index<UnitRangeIndexRaw>;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct UnitRangeLink {
    prev: UnitRangeIndex,
    next: UnitRangeIndex,
}

impl UnitRangeLink {
    fn new(prev: UnitRangeIndex, next: UnitRangeIndex) -> Self {
        assert!(prev.is_valid());
        assert!(next.is_valid());

        Self { prev, next }
    }
}

#[derive(Default)]
struct UnitRangeLinkSet(HashSet<UnitRangeLink>);

impl UnitRangeLinkSet {
    fn insert(&mut self, prev: UnitRangeIndex, next: UnitRangeIndex) {
        self.0.insert(UnitRangeLink { prev, next });
    }

    fn pop(&mut self) -> Option<UnitRangeLink> {
        self.0.iter().copied().next().map(|next| {
            self.0.remove(&next);

            next
        })
    }

    fn remove(&mut self, unit_range_link: UnitRangeLink) {
        self.0.remove(&unit_range_link);
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn extend<I: IntoIterator<Item = UnitRangeLink>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

/// A range referring to a slice of units.
///
/// Invariants:
/// 1. A `UnitRange` is either invalid (the `Default` implementation), and is considered dead/
///    unimportant, or it is valid, and all the other invariants are maintained.
/// 2. `range` is a valid range relative to the slice of units that generated this `UnitRange`.
/// 3. The slice of units referred to by a `UnitRange` is unreactive in isolation.
/// 4. If the index for a `UnitRange` is 0, it is the head and tail tracker, and it has length 0,
///    otherwise the slice of units referred to by a `UnitRange` is not empty (though it may have
///    length 1).
/// 5. If `prev` is valid, it can safely be used as an index to refer to the previous `UnitRange` in
///    the unit slice, potentially following some reactions.
/// 6. If `next` is valid, it can safely be used as an index to refer to the next `UnitRange` in the
///    unit slice, potentially following some reactions.
/// 7. Given two `UnitRange`s, `a` and `b`, where `a.next` is the index of `b`, `b.prev` is the
///    index of `a`, and vice versa.
#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct UnitRange {
    range: Range<UnitRangeRangeIndex>,
    prev: UnitRangeIndex,
    next: UnitRangeIndex,
}

impl UnitRange {
    fn units_react(unit_a: u8, unit_b: u8) -> bool {
        match (unit_a, unit_b) {
            (b'A'..=b'Z', b'a'..=b'z') => (unit_a | ASCII_CASE_MASK) == unit_b,
            (b'a'..=b'z', b'A'..=b'Z') => unit_a == (unit_b | ASCII_CASE_MASK),
            _ => false,
        }
    }

    fn should_skip_unit(unit_to_skip: Option<u8>) -> impl Fn(u8) -> bool {
        move |c| unit_to_skip.map_or(false, |unit_to_skip| (c | ASCII_CASE_MASK) == unit_to_skip)
    }

    fn should_skip_unit_char(unit_to_skip: Option<u8>) -> impl Fn(char) -> bool {
        move |c| Self::should_skip_unit(unit_to_skip)(c as u8)
    }

    fn parse<'i>(unit_to_skip: Option<u8>) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        move |input: &str| {
            let mut prev_unit: u8 = b' ';

            map(
                map_res(
                    many1_count(map_opt(anychar, move |curr_unit: char| {
                        let curr_unit: u8 = curr_unit.try_into().ok()?;

                        (!Self::should_skip_unit(unit_to_skip)(curr_unit)
                            && !Self::units_react(prev_unit, curr_unit))
                        .then(|| {
                            prev_unit = curr_unit;
                        })
                    })),
                    u16::try_from,
                ),
                |len| Self {
                    range: UnitRangeRangeIndex::default()..len,
                    prev: UnitRangeIndex::invalid(),
                    next: UnitRangeIndex::invalid(),
                },
            )(input)
        }
    }

    fn invalid() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    fn is_valid(&self) -> bool {
        self.prev.is_valid() && self.next.is_valid()
    }
}

struct UnitRangeIndexIter<'u> {
    unit_ranges: &'u [UnitRange],
    next: UnitRangeIndex,
}

impl<'u> Iterator for UnitRangeIndexIter<'u> {
    type Item = UnitRangeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        (!Polymer::is_head_and_tail_unit_range(self.next)).then(|| {
            let next: UnitRangeIndex = self.next;

            self.next = self.unit_ranges[next.get()].next;

            next
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Polymer<'u> {
    unit_ranges: Vec<UnitRange>,
    units: &'u str,
}

impl<'u> Polymer<'u> {
    fn is_head_and_tail_unit_range(unit_range_index: UnitRangeIndex) -> bool {
        unit_range_index == 0_usize.into()
    }

    fn parse(unit_to_skip: Option<u8>) -> impl FnMut(&'u str) -> IResult<&'u str, Self> {
        move |input: &'u str| {
            let original_input: &str = input;
            let mut unit_ranges: Vec<UnitRange> = vec![UnitRange::default()];
            let mut offset: UnitRangeRangeIndex = UnitRangeRangeIndex::default();
            let mut unit_range_iter = iterator(input, |input: &'u str| {
                let mut skipped_len: usize = 0_usize;
                let (input, mut unit_range): (&str, UnitRange) = preceded(
                    map(
                        take_while(UnitRange::should_skip_unit_char(unit_to_skip)),
                        |skipped: &str| {
                            skipped_len = skipped.len();
                        },
                    ),
                    UnitRange::parse(unit_to_skip),
                )(input)?;

                offset += skipped_len as UnitRangeRangeIndex;
                unit_range.range.start = offset;
                unit_range.range.end += offset;
                offset = unit_range.range.end;

                Ok((input, unit_range))
            });

            unit_ranges.extend(&mut unit_range_iter);

            let input: &str = unit_range_iter.finish()?.0;
            let unit_ranges_len: usize = unit_ranges.len();

            for (index, unit_range) in unit_ranges.iter_mut().enumerate() {
                unit_range.prev = ((index + unit_ranges_len - 1_usize) % unit_ranges_len).into();
                unit_range.next = ((index + 1_usize) % unit_ranges_len).into();
            }

            let units: &str = &original_input[..offset as usize];

            Ok((input, Self { unit_ranges, units }))
        }
    }

    fn is_unit_range_link_reactive(&self, unit_range_link: UnitRangeLink) -> bool {
        UnitRange::units_react(
            *self.units[self.unit_ranges[unit_range_link.prev.get()]
                .range
                .as_range_usize()]
            .as_bytes()
            .last()
            .unwrap(),
            *self.units[self.unit_ranges[unit_range_link.next.get()]
                .range
                .as_range_usize()]
            .as_bytes()
            .first()
            .unwrap(),
        )
    }

    fn gather_initial_reaction_candidates(&self, reaction_candidates: &mut UnitRangeLinkSet) {
        reaction_candidates.clear();
        reaction_candidates.extend(
            self.unit_ranges
                .iter()
                .enumerate()
                .skip(1_usize)
                .filter_map(|(prev_raw, prev_unit_range)| {
                    (prev_unit_range.is_valid() && prev_unit_range.next != 0_usize.into()).then(
                        || {
                            let prev: UnitRangeIndex = prev_raw.into();
                            let next: UnitRangeIndex = prev_unit_range.next;
                            let next_unit_range: &UnitRange = &self.unit_ranges[next.get()];

                            assert_eq!(prev, next_unit_range.prev);

                            UnitRangeLink { prev, next }
                        },
                    )
                }),
        );
    }

    fn iter_unit_range_indices(&self) -> impl Iterator<Item = UnitRangeIndex> + '_ {
        UnitRangeIndexIter {
            unit_ranges: &self.unit_ranges,
            next: self.unit_ranges.first().unwrap().next,
        }
    }

    fn iter_unit_ranges(&self) -> impl Iterator<Item = &UnitRange> {
        self.iter_unit_range_indices()
            .map(|unit_range_index| &self.unit_ranges[unit_range_index.get()])
    }

    fn react(&mut self) {
        let mut reaction_candidates: UnitRangeLinkSet = UnitRangeLinkSet::default();

        self.gather_initial_reaction_candidates(&mut reaction_candidates);

        while let Some(reaction_candidate) = reaction_candidates.pop() {
            let old_prev: UnitRangeIndex = reaction_candidate.prev;
            let old_next: UnitRangeIndex = reaction_candidate.next;

            assert!(old_prev.is_valid());
            assert!(!Self::is_head_and_tail_unit_range(old_prev));
            assert!(old_next.is_valid());
            assert!(!Self::is_head_and_tail_unit_range(old_next));
            assert!(old_prev < old_next);

            if self.is_unit_range_link_reactive(reaction_candidate) {
                let (old_prev_unit_ranges, old_next_unit_ranges): (
                    &mut [UnitRange],
                    &mut [UnitRange],
                ) = self.unit_ranges.split_at_mut(old_next.get());
                let old_prev_unit_range: &mut UnitRange = &mut old_prev_unit_ranges[old_prev.get()];
                let old_next_unit_range: &mut UnitRange = &mut old_next_unit_ranges[0_usize];

                assert!(!old_prev_unit_range.is_empty());
                assert!(!old_next_unit_range.is_empty());

                old_prev_unit_range.range.end -= 1 as UnitRangeRangeIndex;
                old_next_unit_range.range.start += 1 as UnitRangeRangeIndex;

                let new_prev: UnitRangeIndex = if old_prev_unit_range.is_empty() {
                    let new_prev: UnitRangeIndex = old_prev_unit_range.prev;

                    *old_prev_unit_range = UnitRange::invalid();
                    reaction_candidates.remove(UnitRangeLink::new(new_prev, old_prev));

                    new_prev
                } else {
                    old_prev
                };

                let new_next: UnitRangeIndex = if old_next_unit_range.is_empty() {
                    let new_next: UnitRangeIndex = old_next_unit_range.next;

                    *old_next_unit_range = UnitRange::invalid();
                    reaction_candidates.remove(UnitRangeLink::new(old_next, new_next));

                    new_next
                } else {
                    old_next
                };

                self.unit_ranges[new_prev.get()].next = new_next;
                self.unit_ranges[new_next.get()].prev = new_prev;

                if !Self::is_head_and_tail_unit_range(new_prev)
                    && !Self::is_head_and_tail_unit_range(new_next)
                {
                    reaction_candidates.insert(new_prev, new_next);
                }
            }
        }
    }

    fn units_len(&self) -> usize {
        self.iter_unit_ranges()
            .map(|unit_range| unit_range.range.len())
            .sum()
    }
}

impl<'u> Display for Polymer<'u> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.iter_unit_ranges().try_fold((), |_, unit_range| {
            f.write_str(&self.units[unit_range.range.as_range_usize()])
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(String);

impl Solution {
    fn polymer<'u>(&'u self) -> Polymer<'u> {
        Polymer::parse(None)(&self.0).unwrap().1
    }

    fn unreactive_polymer<'u>(&'u self) -> Polymer<'u> {
        let mut polymer: Polymer = self.polymer();

        polymer.react();

        polymer
    }

    fn unreactive_polymer_units_len(&self) -> usize {
        self.unreactive_polymer().units_len()
    }

    fn removed_unit_polymer<'u>(&'u self, unit_to_remove: u8) -> Polymer<'u> {
        Polymer::parse(Some(unit_to_remove))(&self.0).unwrap().1
    }

    fn removed_unit_type_and_shortest_polymer<'u>(&'u self) -> (char, Polymer<'u>) {
        (b'a'..=b'z')
            .into_par_iter()
            .map(|unit| {
                let mut polymer: Polymer = self.removed_unit_polymer(unit);

                polymer.react();

                (unit as char, polymer)
            })
            .min_by_key(|(_, polymer)| polymer.units_len())
            .unwrap()
    }

    fn shortest_polymer_units_len_after_removing_unit_type(&self) -> usize {
        self.removed_unit_type_and_shortest_polymer().1.units_len()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map(alpha0, String::from), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Linked lists aren't super easy in Rust with minimal allocation.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let unreactive_polymer: Polymer = self.unreactive_polymer();

            dbg!(unreactive_polymer.units_len());

            println!("unreactive_polymer:\n\"{unreactive_polymer}\"");
        } else {
            dbg!(self.unreactive_polymer_units_len());
        }
    }

    /// Rayon really comes in handy on this one. Originally I did an implementation where the string
    /// with removed units was actually allocated and constructed in memory, as opposed to just
    /// omitting the removed units using `UnitRange`s. This seemed too slow. I'm not sure if I just
    /// didn't have enough patience, or if it was actually that slow due to reallocating so often.
    /// Either way, the current implementation is a lot cleaner and doesn't reallocate for the
    /// string itself (though a different allocation is used for each `Vec<UnitRange>`).
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let (removed_unit, polymer): (char, Polymer) =
                self.removed_unit_type_and_shortest_polymer();

            dbg!(removed_unit);

            println!("units_len: {}\npolymer:\n{}", polymer.units_len(), &polymer);
        } else {
            dbg!(self.shortest_polymer_units_len_after_removing_unit_type());
        }
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

    const SOLUTION_STRS: &'static [&'static str] =
        &["aA", "abBA", "abAB", "aabAAB", "dabAcCaCBAcCcaDA"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution("aA".into()),
                Solution("abBA".into()),
                Solution("abAB".into()),
                Solution("aabAAB".into()),
                Solution("dabAcCaCBAcCcaDA".into()),
            ]
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
    fn test_polymer() {
        for (index, polymer) in [
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 2_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..1_u16,
                        prev: 0_usize.into(),
                        next: 2_usize.into(),
                    },
                    UnitRange {
                        range: 1_u16..2_u16,
                        prev: 1_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "aA",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 2_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..2_u16,
                        prev: 0_usize.into(),
                        next: 2_usize.into(),
                    },
                    UnitRange {
                        range: 2_u16..4_u16,
                        prev: 1_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "abBA",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 1_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..4_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "abAB",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 1_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..6_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "aabAAB",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 4_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..5_u16,
                        prev: 0_usize.into(),
                        next: 2_usize.into(),
                    },
                    UnitRange {
                        range: 5_u16..11_u16,
                        prev: 1_usize.into(),
                        next: 3_usize.into(),
                    },
                    UnitRange {
                        range: 11_u16..12_u16,
                        prev: 2_usize.into(),
                        next: 4_usize.into(),
                    },
                    UnitRange {
                        range: 12_u16..16_u16,
                        prev: 3_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "dabAcCaCBAcCcaDA",
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).polymer(), polymer);
        }
    }

    #[test]
    fn test_unreactive_polymer() {
        for (index, unreactive_polymer) in [
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                    UnitRange::invalid(),
                    UnitRange::invalid(),
                ],
                units: "aA",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                    UnitRange::invalid(),
                    UnitRange::invalid(),
                ],
                units: "abBA",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 1_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..4_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "abAB",
            },
            Polymer {
                unit_ranges: vec![
                    UnitRange {
                        range: 0_u16..0_u16,
                        prev: 1_usize.into(),
                        next: 1_usize.into(),
                    },
                    UnitRange {
                        range: 0_u16..6_u16,
                        prev: 0_usize.into(),
                        next: 0_usize.into(),
                    },
                ],
                units: "aabAAB",
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).unreactive_polymer(), unreactive_polymer);
        }
    }

    #[test]
    fn test_display_fmt() {
        for (index, unreactive_polymer_string) in ["", "", "abAB", "aabAAB", "dabCBAcaDA"]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);

            assert_eq!(format!("{}", solution.polymer()), SOLUTION_STRS[index]);
            assert_eq!(
                format!("{}", solution.unreactive_polymer()),
                unreactive_polymer_string
            );
        }
    }

    #[test]
    fn test_removed_unit_polymer() {
        for (unit_to_remove, polymer) in [
            (b'a', "dbcCCBcCcD"),
            (b'b', "daAcCaCAcCcaDA"),
            (b'c', "dabAaBAaDA"),
            (b'd', "abAcCaCBAcCcaA"),
        ]
        .into_iter()
        {
            assert_eq!(
                format!("{}", solution(4_usize).removed_unit_polymer(unit_to_remove)),
                polymer
            );
        }
    }

    #[test]
    fn test_removed_unit_type_and_shortest_polymer() {
        let (removed_unit, polymer): (char, Polymer) =
            solution(4_usize).removed_unit_type_and_shortest_polymer();

        assert_eq!(removed_unit, 'c');
        assert_eq!(polymer.units_len(), 4_usize);
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
