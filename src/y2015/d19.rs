#![allow(unused)]

use {
    crate::*,
    arrayvec::ArrayVec,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::{line_ending, satisfy},
        combinator::{map, map_opt, success, verify},
        error::Error,
        multi::{many1, separated_list1},
        sequence::tuple,
        Err, IResult,
    },
    std::{
        collections::HashSet,
        hash::{DefaultHasher, Hash, Hasher},
        ops::Range,
        str::from_utf8_unchecked,
    },
};

/* --- Day 19: Medicine for Rudolph ---

Rudolph the Red-Nosed Reindeer is sick! His nose isn't shining very brightly, and he needs medicine.

Red-Nosed Reindeer biology isn't similar to regular reindeer biology; Rudolph is going to need custom-made medicine. Unfortunately, Red-Nosed Reindeer chemistry isn't similar to regular reindeer chemistry, either.

The North Pole is equipped with a Red-Nosed Reindeer nuclear fusion/fission plant, capable of constructing any Red-Nosed Reindeer molecule you need. It works by starting with some input molecule and then doing a series of replacements, one per step, until it has the right molecule.

However, the machine has to be calibrated before it can be used. Calibration involves determining the number of molecules that can be generated in one step from a given starting point.

For example, imagine a simpler machine that supports only the following replacements:

H => HO
H => OH
O => HH

Given the replacements above and starting with HOH, the following molecules could be generated:

    HOOH (via H => HO on the first H).
    HOHO (via H => HO on the second H).
    OHOH (via H => OH on the first H).
    HOOH (via H => OH on the second H).
    HHHH (via O => HH).

So, in the example above, there are 4 distinct molecules (not five, because HOOH appears twice) after one replacement from HOH. Santa's favorite molecule, HOHOHO, can become 7 distinct molecules (over nine replacements: six from H, and three from O).

The machine replaces without regard for the surrounding characters. For example, given the string H2O, the transition H => OO would result in OO2O.

Your puzzle input describes all of the possible replacements and, at the bottom, the medicine molecule for which you need to calibrate the machine. How many distinct molecules can be created after all the different ways you can do one replacement on the medicine molecule?

--- Part Two ---

Now that the machine is calibrated, you're ready to begin molecule fabrication.

Molecule fabrication always begins with just a single electron, e, and applying replacements one at a time, just like the ones during calibration.

For example, suppose you have the following replacements:

e => H
e => O
H => HO
H => OH
O => HH

If you'd like to make HOH, you start with e, and then make the following replacements:

    e => O to get O
    O => HH to get HH
    H => OH (on the second H) to get HOH

So, you could make HOH after 3 steps. Santa's favorite molecule, HOHOHO, can be made in 6 steps.

How long will it take to make the medicine? Given the available replacements and the medicine molecule in your puzzle input, what is the fewest number of steps to go from e to the medicine molecule? */

type ElementIndexRaw = u8;
type ElementIndex = Index<ElementIndexRaw>;

const MAX_ELEMENT_ID_LEN: usize = 2_usize;

type ElementId = StaticString<MAX_ELEMENT_ID_LEN>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct ElementData {
    replacements: ReplacementRange,
}

type Element = TableElement<ElementId, ElementData>;
type ElementTable = Table<ElementId, ElementData, ElementIndexRaw>;

type ElementListIndexRaw = u16;
type ElementListIndex = Index<ElementListIndexRaw>;

type ElementListRange = Range<ElementListIndex>;

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ReplacementData {
    replaced_element: ElementIndex,
    replacing_elements: ElementListRange,
}

type ReplacementIndexRaw = u8;
type ReplacementIndex = Index<ReplacementIndexRaw>;
type ReplacementRange = Range<ReplacementIndex>;

const MIN_REPLACEMENT_LEN: usize = 2_usize;
const MAX_MOLECULE_LEN: usize = 300_usize;

type InverseReplacementTrieNodeIndexRaw = u8;
type InverseReplacementTrieNodeIndex = Index<InverseReplacementTrieNodeIndexRaw>;

#[derive(Default)]
struct InverseReplacementTrieNodeData {
    replacing_element_index: ElementIndex,
    replacement_index: ReplacementIndex,
    child_node_index: InverseReplacementTrieNodeIndex,
    sibling_node_index: InverseReplacementTrieNodeIndex,
}

struct InverseReplacementTrie(Vec<InverseReplacementTrieNodeData>);

impl InverseReplacementTrie {
    fn insert_replacing_element(
        &mut self,
        replacing_element_index: ElementIndex,
        head_node_index: InverseReplacementTrieNodeIndex,
    ) -> InverseReplacementTrieNodeIndex {
        let is_empty: bool = self.0.is_empty();
        let head_node_index_is_valid: bool = head_node_index.is_valid();

        assert!(!is_empty || !head_node_index_is_valid);

        if is_empty || !head_node_index_is_valid {
            let node_index: InverseReplacementTrieNodeIndex = self.0.len().into();

            self.0.push(InverseReplacementTrieNodeData {
                replacing_element_index,
                ..Default::default()
            });

            node_index
        } else {
            let mut found_node: bool = false;
            let mut prev_node_index: InverseReplacementTrieNodeIndex =
                InverseReplacementTrieNodeIndex::invalid();
            let mut curr_node_index: InverseReplacementTrieNodeIndex = head_node_index;

            while !found_node && curr_node_index.is_valid() {
                let node: &InverseReplacementTrieNodeData = &self.0[curr_node_index.get()];

                found_node = node.replacing_element_index == replacing_element_index;
                prev_node_index = curr_node_index;
                curr_node_index = node.sibling_node_index;
            }

            if !found_node {
                let sibling_node_index: InverseReplacementTrieNodeIndex = self.0.len().into();

                self.0.push(InverseReplacementTrieNodeData {
                    replacing_element_index,
                    ..Default::default()
                });

                assert!(prev_node_index.is_valid());

                self.0[prev_node_index.get()].sibling_node_index = sibling_node_index;

                sibling_node_index
            } else {
                prev_node_index
            }
        }
    }

    fn insert_replacement(
        &mut self,
        replacement_index: ReplacementIndex,
        replacing_elements: &[ElementIndex],
    ) {
        assert!(replacing_elements.len() >= MIN_REPLACEMENT_LEN);

        let node_index: InverseReplacementTrieNodeIndex = replacing_elements
            .iter()
            .copied()
            .fold(
                (
                    InverseReplacementTrieNodeIndex::invalid(),
                    InverseReplacementTrieNodeIndex::invalid(),
                ),
                |(parent_node_index, child_node_index), replacing_element| {
                    let node_index: InverseReplacementTrieNodeIndex =
                        self.insert_replacing_element(replacing_element, child_node_index);

                    if !child_node_index.is_valid() {
                        self.0[parent_node_index.get()].child_node_index = child_node_index;
                    }

                    (node_index, self.0[node_index.get()].child_node_index)
                },
            )
            .0;

        let node_data: &mut InverseReplacementTrieNodeData = &mut self.0[node_index.get()];

        assert!(!node_data.replacement_index.is_valid());

        node_data.replacement_index = replacement_index;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    element_table: ElementTable,
    element_list: Vec<ElementIndex>,
    replacements: Vec<ReplacementData>,
    molecule: ElementListRange,
}

impl Solution {
    fn range_usize_from_range_index<I: IndexRawTrait>(range: &Range<Index<I>>) -> Range<usize> {
        range.start.get()..range.end.get()
    }

    fn parse_element_id<'i>(input: &'i str) -> IResult<&'i str, ElementId> {
        alt((
            map(tag("e"), |element_id: &str| element_id.try_into().unwrap()),
            map(
                tuple((
                    satisfy(|c| c.is_ascii_uppercase()),
                    take_while_m_n(0_usize, MAX_ELEMENT_ID_LEN - 1_usize, |c: char| {
                        c.is_ascii_lowercase()
                    }),
                )),
                |(a, b): (char, &str)| {
                    let bytes: ArrayVec<u8, MAX_ELEMENT_ID_LEN> = [a as u8]
                        .into_iter()
                        .chain(b.as_bytes().iter().copied())
                        .collect();

                    unsafe { from_utf8_unchecked(&bytes) }.try_into().unwrap()
                },
            ),
        ))(input)
    }

    fn try_extract_len<T, I: IndexRawTrait>(values: &mut Option<&mut Vec<T>>) -> Option<usize> {
        if let Some(values) = copy_opt_mut(values) {
            I::from_usize(values.len()).map(|len| len.to_usize().unwrap())
        } else {
            Some(0_usize)
        }
    }

    fn parse_element_index<'i, F: FnMut(ElementId) -> ElementIndex>(
        mut element_index_from_element_id: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, ElementIndex> {
        move |input| {
            map(Self::parse_element_id, |element_id| {
                element_index_from_element_id(element_id)
            })(input)
        }
    }

    fn parse_replacement<'a, 'i, F: FnMut(&'i str) -> IResult<&'i str, ElementIndex> + 'a>(
        mut parse_element_index: F,
        mut element_list: Option<&'a mut Vec<ElementIndex>>,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, ReplacementData> + 'a {
        move |input| {
            let (input, replaced_element): (&str, ElementIndex) = parse_element_index(input)?;
            let input: &str = tag(" => ")(input)?.0;
            let replacing_elements_start: ElementListIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ElementListIndexRaw>(&mut element_list)
            })(input)?
            .1
            .into();
            let input: &str = many1(map(
                |input| parse_element_index(input),
                |element_index| {
                    if let Some(element_list) = copy_opt_mut(&mut element_list) {
                        element_list.push(element_index);
                    }
                },
            ))(input)?
            .0;
            let replacing_elements_end: ElementListIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ElementListIndexRaw>(&mut element_list)
            })(input)?
            .1
            .into();

            Ok((
                input,
                ReplacementData {
                    replaced_element,
                    replacing_elements: replacing_elements_start..replacing_elements_end,
                },
            ))
        }
    }

    fn parse_element<'a, 'i, F: FnMut(&'i str) -> IResult<&'i str, ElementIndex> + 'a>(
        mut parse_element_index: F,
        mut element_list: Option<&'a mut Vec<ElementIndex>>,
        mut replacements: Option<&'a mut Vec<ReplacementData>>,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, (ElementIndex, ElementData)> + 'a {
        move |input| {
            let element_index: ElementIndex = parse_element_index(input)?.1;
            let replacements_start: ReplacementIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ReplacementIndexRaw>(&mut replacements)
            })(input)?
            .1
            .into();
            let input: &str = separated_list1(line_ending, |input| {
                verify(
                    |input| parse_element_index(input),
                    |&next_element_index| next_element_index == element_index,
                )(input)?;

                let (input, replacement): (&str, ReplacementData) = Self::parse_replacement(
                    |input| parse_element_index(input),
                    copy_opt_mut(&mut element_list),
                )(input)?;

                if let Some(replacements) = copy_opt_mut(&mut replacements) {
                    replacements.push(replacement);
                }

                Ok((input, ()))
            })(input)?
            .0;
            let replacements_end: ReplacementIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ReplacementIndexRaw>(&mut replacements)
            })(input)?
            .1
            .into();

            Ok((
                input,
                (
                    element_index,
                    ElementData {
                        replacements: replacements_start..replacements_end,
                    },
                ),
            ))
        }
    }

    fn parse_internal<'a, 'i>(
        solution: &'a mut Self,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, ()> + 'a {
        |input| {
            let is_first_pass: bool = solution.element_table.as_slice().is_empty();
            let input: &str = separated_list1(line_ending, |input| {
                let (element_list, replacements): (
                    Option<&mut Vec<ElementIndex>>,
                    Option<&mut Vec<ReplacementData>>,
                ) = if is_first_pass {
                    (None, None)
                } else {
                    (
                        Some(&mut solution.element_list),
                        Some(&mut solution.replacements),
                    )
                };

                let (input, (element_index, element_data)): (&str, (ElementIndex, ElementData)) =
                    Self::parse_element(
                        Self::parse_element_index(|element_id| {
                            if is_first_pass {
                                solution
                                    .element_table
                                    .find_or_add_index_binary_search(&element_id);

                                ElementIndex::invalid()
                            } else {
                                solution.element_table.find_index_binary_search(&element_id)
                            }
                        }),
                        element_list,
                        replacements,
                    )(input)?;

                if !is_first_pass {
                    solution.element_table.as_slice_mut()[element_index.get()].data = element_data;
                }

                Ok((input, ()))
            })(input)?
            .0;
            let input: &str = tuple((line_ending, line_ending))(input)?.0;

            let mut element_list: Option<&mut Vec<ElementIndex>> =
                (!is_first_pass).then_some(&mut solution.element_list);

            let molecule_start: ElementListIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ElementListIndexRaw>(&mut element_list)
            })(input)?
            .1
            .into();
            let input: &str = many1(map(Self::parse_element_id, |element_id| {
                if let Some(element_list) = copy_opt_mut(&mut element_list) {
                    element_list.push(solution.element_table.find_index_binary_search(&element_id));
                } else {
                    solution
                        .element_table
                        .find_or_add_index_binary_search(&element_id);
                }
            }))(input)?
            .0;
            let molecule_end: ElementListIndex = map_opt(success(()), |_| {
                Self::try_extract_len::<_, ElementListIndexRaw>(&mut element_list)
            })(input)?
            .1
            .into();

            if !is_first_pass {
                solution.molecule = molecule_start..molecule_end;
            }

            Ok((input, ()))
        }
    }

    fn iter_elements_in_molecule_ranges<'s>(
        &'s self,
        molecule_ranges: &'s [Range<ElementListIndex>],
    ) -> impl Iterator<Item = ElementIndex> + 's {
        molecule_ranges.iter().flat_map(|molecule_range| {
            self.element_list[Self::range_usize_from_range_index(molecule_range)]
                .iter()
                .copied()
        })
    }

    fn molecule_hash(&self, molecule_ranges: &[Range<ElementListIndex>]) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        for element in self.iter_elements_in_molecule_ranges(molecule_ranges) {
            element.hash(&mut hasher);
        }

        hasher.finish()
    }

    fn replacement_molecule_hashes(&self) -> HashSet<u64> {
        let self_molecule_range: Range<ElementListIndex> = self.molecule.clone();

        let self_molecule_elements: &[ElementIndex] =
            &self.element_list[Self::range_usize_from_range_index(&self_molecule_range)];
        let mut molecule_ranges: [Range<ElementListIndex>; 3_usize] = [
            self_molecule_range.clone(),
            ElementListIndex::invalid()..ElementListIndex::invalid(),
            self_molecule_range.clone(),
        ];
        let mut molecule_hashes: HashSet<u64> = HashSet::new();

        for (molecule_offset, original_molecule_element) in
            self_molecule_elements.iter().enumerate()
        {
            let midpoint: ElementListIndex =
                (self_molecule_range.start.get() + molecule_offset).into();

            molecule_ranges[0_usize].end = midpoint;
            molecule_ranges[2_usize].start = (midpoint.get() + 1_usize).into();

            let replacement_ranges: &Range<ReplacementIndex> = &self.element_table.as_slice()
                [original_molecule_element.get()]
            .data
            .replacements;

            if replacement_ranges.start.is_valid() && replacement_ranges.end.is_valid() {
                for replacement in
                    &self.replacements[Self::range_usize_from_range_index(replacement_ranges)]
                {
                    molecule_ranges[1_usize] = replacement.replacing_elements.clone();
                    molecule_hashes.insert(self.molecule_hash(&molecule_ranges));
                }
            }
        }

        molecule_hashes
    }

    fn replacement_molecule_count(&self) -> usize {
        self.replacement_molecule_hashes().len()
    }

    // fn inverse_replacement_trie(&self) -> InverseReplacmentTrie {
    //     let mut inverse_replacement_trie: InverseReplacmentTrie = InverseReplacmentTrie::new();

    //     inverse_replacement_trie
    // }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Default::default();

        Self::parse_internal(&mut solution)(input)?;

        let output: &str = Self::parse_internal(&mut solution)(input)?.0;

        verify(success(()), |_| {
            Self::range_usize_from_range_index(&solution.molecule).len() <= MAX_MOLECULE_LEN
                && solution
                    .replacements
                    .iter()
                    .try_fold(
                        HashSet::<&[ElementIndex]>::new(),
                        |mut present_replacements, replacement_data| {
                            let replacing_elements: Range<usize> =
                                Self::range_usize_from_range_index(
                                    &replacement_data.replacing_elements,
                                );

                            (replacing_elements.len() >= MIN_REPLACEMENT_LEN
                                && present_replacements
                                    .insert(&solution.element_list[replacing_elements]))
                            .then_some(present_replacements)
                        },
                    )
                    .is_some()
        })(input)?;

        Ok((output, solution))
    }
}

impl RunQuestions for Solution {
    /// Q2 is going to have something to do with the e replacments.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.replacement_molecule_count());
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        let molecule: Range<usize> = self.molecule.start.get()..self.molecule.end.get();

        dbg!(self.molecule.start.get());
        dbg!(molecule.len());

        todo!();
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        H => HO\n\
        H => OH\n\
        O => HH\n\
        \n\
        HOH\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                element_table: vec![
                    Element {
                        id: "H".try_into().unwrap(),
                        data: ElementData {
                            replacements: ReplacementIndex::from(0_usize)..2_usize.into(),
                        },
                    },
                    Element {
                        id: "O".try_into().unwrap(),
                        data: ElementData {
                            replacements: ReplacementIndex::from(2_usize)..3_usize.into(),
                        },
                    },
                ]
                .try_into()
                .unwrap(),
                element_list: vec![
                    ElementIndex::from(0_usize),
                    1_usize.into(),
                    1_usize.into(),
                    0_usize.into(),
                    0_usize.into(),
                    0_usize.into(),
                    0_usize.into(),
                    1_usize.into(),
                    0_usize.into(),
                ],
                replacements: vec![
                    ReplacementData {
                        replaced_element: 0_usize.into(),
                        replacing_elements: 0_usize.into()..2_usize.into(),
                    },
                    ReplacementData {
                        replaced_element: 0_usize.into(),
                        replacing_elements: 2_usize.into()..4_usize.into(),
                    },
                    ReplacementData {
                        replaced_element: 1_usize.into(),
                        replacing_elements: 4_usize.into()..6_usize.into(),
                    },
                ],
                molecule: ElementListIndex::from(6_usize)..9_usize.into(),
            }]
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
    fn test_replacement_molecule_hashes() {
        for (index, replacement_molecule_hashes_count) in [4_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).replacement_molecule_hashes().len(),
                replacement_molecule_hashes_count
            );
        }
    }

    #[test]
    fn test_input() {
        let args: Args = Args::parse(module_path!()).unwrap().1;

        Solution::both(&args);
    }
}
