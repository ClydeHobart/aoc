#![allow(unused_imports, dead_code)]

use {
    crate::*,
    arrayvec::ArrayVec,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::{line_ending, satisfy},
        combinator::{map, map_opt, success, verify},
        error::Error as NomError,
        multi::{many1, separated_list1},
        sequence::tuple,
        Err, IResult,
    },
    std::{
        borrow::BorrowMut,
        cell::{Ref, RefCell, RefMut},
        collections::{HashMap, HashSet},
        fmt::{
            Debug, DebugList, Display, Error as FmtError, Formatter, Result as FmtResult, Write,
        },
        hash::{DefaultHasher, Hash, Hasher},
        iter::from_fn,
        ops::Range,
        str::from_utf8_unchecked,
        sync::LazyLock,
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

struct ElementIndexIter<'e, I: Clone + Iterator<Item = ElementIndex>> {
    element_table: &'e ElementTable,
    element_index_iter: I,
}
impl<'e, I: Clone + Iterator<Item = ElementIndex>> ElementIndexIter<'e, I> {
    fn new(element_table: &'e ElementTable, element_index_iter: I) -> Self {
        Self {
            element_table,
            element_index_iter,
        }
    }
}

impl<'e, I: Clone + Iterator<Item = ElementIndex>> Debug for ElementIndexIter<'e, I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut debug_list: DebugList = f.debug_list();

        for element_index in self.element_index_iter.clone() {
            debug_list.entry(&element_index);
        }

        debug_list.finish()
    }
}

impl<'e, I: Clone + Iterator<Item = ElementIndex>> Display for ElementIndexIter<'e, I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.element_index_iter
            .clone()
            .try_for_each(|element_index| {
                if element_index.is_valid() {
                    element_index.fmt(f)
                } else {
                    f.write_str(
                        self.element_table.as_slice()[element_index.get()]
                            .id
                            .as_str(),
                    )
                }
            })
    }
}

type ElementListIndexRaw = u32;
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

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct FastestEReductionSearchVertex(u64);

impl FastestEReductionSearchVertex {
    fn invalid() -> Self {
        static INVALID: LazyLock<FastestEReductionSearchVertex> = LazyLock::new(|| {
            let element_list: &[ElementIndex] = &[];

            element_list.into()
        });

        *INVALID
    }

    fn from_iter<I: Iterator<Item = ElementIndex>>(element_index_iter: I) -> Self {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        for element_index in element_index_iter {
            element_index.hash(&mut hasher);
        }

        Self(hasher.finish())
    }

    fn is_valid(self) -> bool {
        self != Self::invalid()
    }
}

impl Default for FastestEReductionSearchVertex {
    fn default() -> Self {
        Self::invalid()
    }
}

impl From<&[ElementIndex]> for FastestEReductionSearchVertex {
    fn from(value: &[ElementIndex]) -> Self {
        Self::from_iter(value.iter().copied())
    }
}

type ReplacementCount = u16;

#[derive(Clone)]
struct FastestEReductionSearchVertexData {
    element_list_range: ElementListRange,
    replacement_location: ElementListIndex,
    replacement_count: ReplacementCount,
    replacement_index: ReplacementIndex,
    prev_vertex: FastestEReductionSearchVertex,
}

#[derive(Default)]
struct FastestEReductionSearchMutableState {
    element_list: Vec<ElementIndex>,
    pending_element_list: Vec<ElementIndex>,
    vertex_data_map: HashMap<FastestEReductionSearchVertex, FastestEReductionSearchVertexData>,
}

impl FastestEReductionSearchMutableState {
    fn inverse_replacement_element_list<'e>(
        element_list: &'e [ElementIndex],
        solution: &Solution,
        replacement_index: ReplacementIndex,
        replacement_start_element_list_index: ElementListIndex,
    ) -> (&'e [ElementIndex], ElementIndex, &'e [ElementIndex]) {
        let replacement_data: &ReplacementData = &solution.replacements[replacement_index.get()];

        let solution_replacement_element_list: &[ElementIndex] = &solution.element_list
            [Solution::range_usize_from_range_index(&replacement_data.replacing_elements)];
        let replacement_end_element_list_index: ElementListIndex =
            (replacement_start_element_list_index.get() + solution_replacement_element_list.len())
                .into();
        let target_replacement_element_list: &[ElementIndex] = element_list
            .get(
                replacement_start_element_list_index.get()
                    ..replacement_end_element_list_index.get(),
            )
            .unwrap();

        assert_eq_break(
            solution_replacement_element_list,
            target_replacement_element_list,
        );

        (
            &element_list[..replacement_start_element_list_index.get()],
            replacement_data.replaced_element,
            &element_list[replacement_end_element_list_index.get()..],
        )
    }
}

type InverseReplacementMapNodeIndexRaw = u16;
type InverseReplacementMapNodeIndex = Index<InverseReplacementMapNodeIndexRaw>;

struct FastestEReductionSearcher<'s> {
    solution: &'s Solution,
    e: ElementIndex,
    start_vertex: FastestEReductionSearchVertex,
    end_vertex: FastestEReductionSearchVertex,
    inverse_replacement_map:
        LinkedTrie<ElementIndex, ReplacementIndex, InverseReplacementMapNodeIndex>,
    max_replacement_element_list_len_delta: ElementListIndex,
    mutable_state: RefCell<FastestEReductionSearchMutableState>,
}

impl<'s> WeightedGraphSearch for FastestEReductionSearcher<'s> {
    type Vertex = FastestEReductionSearchVertex;
    type Cost = ReplacementCount;

    fn start(&self) -> &Self::Vertex {
        &self.start_vertex
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        let mutable_state: Ref<FastestEReductionSearchMutableState> = self.mutable_state.borrow();

        mutable_state.element_list[Solution::range_usize_from_range_index(
            &mutable_state.vertex_data_map[vertex].element_list_range,
        )] == [self.e]
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        Vec::new()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.mutable_state.borrow().vertex_data_map[vertex].replacement_count
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        let element_list_len: usize = Solution::range_usize_from_range_index(
            &self.mutable_state.borrow().vertex_data_map[vertex].element_list_range,
        )
        .len();
        let remaining_replacement_len: usize = element_list_len - 1_usize;

        if remaining_replacement_len == 0_usize {
            0 as ReplacementCount
        } else {
            ((remaining_replacement_len - 1_usize)
                / self.max_replacement_element_list_len_delta.get()
                + 1_usize) as ReplacementCount
        }
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        let mut mutable_state: RefMut<FastestEReductionSearchMutableState> =
            self.mutable_state.borrow_mut();
        let FastestEReductionSearchMutableState {
            ref mut element_list,
            ref mut pending_element_list,
            ref mut vertex_data_map,
        } = *mutable_state;

        let vertex_data: FastestEReductionSearchVertexData = vertex_data_map[vertex].clone();
        let vertex_element_list: &[ElementIndex] =
            &element_list[Solution::range_usize_from_range_index(&vertex_data.element_list_range)];

        dbg!(vertex_element_list.len());

        // For future `FastestEReductionSearchVertexData` construction.
        let replacement_count: ReplacementCount = vertex_data.replacement_count + 1;
        let prev_vertex: FastestEReductionSearchVertex = *vertex;

        for start_element_list_index in 0_usize..vertex_element_list.len() {
            // For future `FastestEReductionSearchVertexData` construction.
            let replacement_location: ElementListIndex = start_element_list_index.into();

            // For `from_fn`
            let mut element_index_iter = vertex_element_list[start_element_list_index..].iter();
            let mut inverse_replacement_map_node_index: InverseReplacementMapNodeIndex =
                InverseReplacementMapNodeIndex::invalid();
            let mut failed_to_find_child_node: bool = false;

            for replacement_index in from_fn(|| {
                (!failed_to_find_child_node)
                    .then_some(element_index_iter.next())
                    .flatten()
                    .and_then(|element_index| {
                        if let Some(next_inverse_replacement_map_node_index) =
                            self.inverse_replacement_map.find_child_node_index_sorted(
                                element_index,
                                inverse_replacement_map_node_index,
                            )
                        {
                            inverse_replacement_map_node_index =
                                next_inverse_replacement_map_node_index;

                            Some(next_inverse_replacement_map_node_index)
                        } else {
                            failed_to_find_child_node = true;

                            None
                        }
                    })
            })
            .filter_map(|inverse_replacement_map_node_index| {
                self.inverse_replacement_map
                    .try_get_node(inverse_replacement_map_node_index)
                    .unwrap()
                    .kvp
                    .value
                    .copied()
            }) {
                let (
                    pre_replacement_element_list,
                    replaced_element_index,
                    post_replacement_element_list,
                ): (&[ElementIndex], ElementIndex, &[ElementIndex]) =
                    FastestEReductionSearchMutableState::inverse_replacement_element_list(
                        &vertex_element_list,
                        &self.solution,
                        replacement_index,
                        start_element_list_index.into(),
                    );
                let neighbor: FastestEReductionSearchVertex =
                    FastestEReductionSearchVertex::from_iter(
                        Solution::element_index_iter_from_pre_replaced_and_post(
                            pre_replacement_element_list,
                            replaced_element_index,
                            post_replacement_element_list,
                        ),
                    );

                if vertex_data_map
                    .get(&neighbor)
                    .map_or(true, |neighbor_data| {
                        neighbor_data.replacement_count > replacement_count
                    })
                {
                    let start_pending_element_list_len: usize = pending_element_list.len();
                    let neighbor_start_element_list_index: ElementListIndex =
                        (start_pending_element_list_len + element_list.len()).into();

                    pending_element_list.extend_from_slice(pre_replacement_element_list);
                    pending_element_list.push(replaced_element_index);
                    pending_element_list.extend_from_slice(post_replacement_element_list);

                    let end_pending_element_list_len: usize = pending_element_list.len();
                    let neighbor_end_element_list_index: ElementListIndex =
                        (neighbor_start_element_list_index.get()
                            + (end_pending_element_list_len - start_pending_element_list_len))
                            .into();

                    vertex_data_map.insert(
                        neighbor,
                        FastestEReductionSearchVertexData {
                            element_list_range: neighbor_start_element_list_index
                                ..neighbor_end_element_list_index,
                            replacement_location,
                            replacement_count,
                            replacement_index,
                            prev_vertex,
                        },
                    );

                    neighbors.push(OpenSetElement(neighbor, 1 as ReplacementCount));
                }
            }
        }

        element_list.extend(pending_element_list.drain(..));
    }

    fn update_vertex(
        &mut self,
        _from: &Self::Vertex,
        to: &Self::Vertex,
        _cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        if self.is_end(to) {
            self.end_vertex = *to;
        }
    }

    fn reset(&mut self) {
        let mut mutable_state: RefMut<FastestEReductionSearchMutableState> =
            self.mutable_state.borrow_mut();
        let FastestEReductionSearchMutableState {
            ref mut element_list,
            ref mut pending_element_list,
            ref mut vertex_data_map,
        } = *mutable_state;

        element_list.clear();
        pending_element_list.clear();
        vertex_data_map.clear();
        self.e = self
            .solution
            .element_table
            .find_index_binary_search(&"e".try_into().unwrap());
        element_list.extend_from_slice(
            &self.solution.element_list
                [Solution::range_usize_from_range_index(&self.solution.molecule)],
        );
        self.start_vertex = element_list[..].into();
        self.end_vertex = FastestEReductionSearchVertex::invalid();
        vertex_data_map.insert(
            self.start_vertex,
            FastestEReductionSearchVertexData {
                element_list_range: 0_usize.into()..element_list.len().into(),
                replacement_location: ElementListIndex::invalid(),
                replacement_count: 0,
                replacement_index: ReplacementIndex::invalid(),
                prev_vertex: FastestEReductionSearchVertex::invalid(),
            },
        );

        for (replacement_index, replacement_data) in self.solution.replacements.iter().enumerate() {
            let replacing_elements: Range<usize> =
                Solution::range_usize_from_range_index(&replacement_data.replacing_elements);
            let replacement_element_list_len_delta: ElementListIndex =
                (replacing_elements.len() - 1_usize).into();

            if self
                .max_replacement_element_list_len_delta
                .opt()
                .filter(|max_replacement_element_list_len_delta| {
                    max_replacement_element_list_len_delta.get()
                        > replacement_element_list_len_delta.get()
                })
                .is_none()
            {
                self.max_replacement_element_list_len_delta = replacement_element_list_len_delta;
            }

            self.inverse_replacement_map.insert_sorted(
                self.solution.element_list[replacing_elements]
                    .iter()
                    .copied(),
                replacement_index.into(),
            );
        }
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

    fn element_index_iter_from_pre_replaced_and_post<'e>(
        pre_replacement_element_list: &'e [ElementIndex],
        replaced_element_index: ElementIndex,
        post_replacement_element_list: &'e [ElementIndex],
    ) -> impl Clone + Iterator<Item = ElementIndex> + 'e {
        pre_replacement_element_list
            .iter()
            .copied()
            .chain([replaced_element_index])
            .chain(post_replacement_element_list.iter().copied())
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

    fn fastest_e_reduction_searcher(&self) -> FastestEReductionSearcher {
        FastestEReductionSearcher {
            solution: self,
            e: Default::default(),
            start_vertex: Default::default(),
            end_vertex: Default::default(),
            inverse_replacement_map: Default::default(),
            max_replacement_element_list_len_delta: Default::default(),
            mutable_state: Default::default(),
        }
    }

    fn try_fastest_e_reduction(&self) -> Option<FastestEReductionSearcher> {
        let mut fastest_e_reduction_searcher: FastestEReductionSearcher =
            self.fastest_e_reduction_searcher();

        fastest_e_reduction_searcher.reset();

        (fastest_e_reduction_searcher.e.is_valid()
            && fastest_e_reduction_searcher
                .max_replacement_element_list_len_delta
                .opt()
                .filter(|max_replacement_element_list_len_delta| {
                    max_replacement_element_list_len_delta.get() > 0_usize
                })
                .is_some())
        .then(|| fastest_e_reduction_searcher.run_a_star())
        .flatten()
        .map(|_| fastest_e_reduction_searcher)
    }

    fn try_fastest_e_reduction_replacement_count(&self) -> Option<ReplacementCount> {
        self.try_fastest_e_reduction().map(|fastest_e_reduction| {
            fastest_e_reduction.mutable_state.borrow().vertex_data_map
                [&fastest_e_reduction.end_vertex]
                .replacement_count
        })
    }

    fn print_elements(
        &self,
        element_list: &[ElementIndex],
        string: &mut String,
    ) -> Result<(), FmtError> {
        write!(
            string,
            "{}",
            ElementIndexIter::new(&self.element_table, element_list.iter().copied())
        )
    }

    fn replacements_from_fastest_e_reduction(
        &self,
        fastest_e_reduction: &FastestEReductionSearcher,
    ) -> Result<Vec<String>, FmtError> {
        let mut vertex: FastestEReductionSearchVertex = fastest_e_reduction.end_vertex;

        from_fn(|| {
            (vertex != fastest_e_reduction.start_vertex).then(|| {
                let next_vertex: FastestEReductionSearchVertex = vertex;

                vertex =
                    fastest_e_reduction.mutable_state.borrow().vertex_data_map[&vertex].prev_vertex;

                next_vertex
            })
        })
        .try_fold(Vec::new(), |mut replacements, vertex| {
            let mutable_state: Ref<FastestEReductionSearchMutableState> =
                fastest_e_reduction.mutable_state.borrow();
            let vertex_data: &FastestEReductionSearchVertexData =
                &mutable_state.vertex_data_map[&vertex];
            let replacement_location: usize = vertex_data.replacement_location.get();
            let element_list: &[ElementIndex] = &mutable_state.element_list
                [Self::range_usize_from_range_index(&vertex_data.element_list_range)];
            let pre_replacement_element_list: &[ElementIndex] =
                &element_list[..replacement_location];
            let replaced_element_index: ElementIndex = element_list[replacement_location];
            let replacing_element_list: &[ElementIndex] = &self.element_list
                [Self::range_usize_from_range_index(
                    &self.replacements[vertex_data.replacement_index.get()].replacing_elements,
                )];
            let post_replacement_element_list: &[ElementIndex] =
                &element_list[replacement_location + 1_usize..];

            let mut replacement_string: String = String::new();

            if !pre_replacement_element_list.is_empty() {
                self.print_elements(pre_replacement_element_list, &mut replacement_string)?;
                write!(&mut replacement_string, ", ")?;
            }

            write!(&mut replacement_string, "(")?;
            self.print_elements(&[replaced_element_index], &mut replacement_string)?;
            write!(&mut replacement_string, " => ")?;
            self.print_elements(replacing_element_list, &mut replacement_string)?;
            write!(&mut replacement_string, ")")?;

            if !post_replacement_element_list.is_empty() {
                write!(&mut replacement_string, ", ")?;
                self.print_elements(post_replacement_element_list, &mut replacement_string)?;
            }

            replacements.push(replacement_string);

            Ok(replacements)
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Default::default();

        Self::parse_internal(&mut solution)(input)?;

        let output: &str = Self::parse_internal(&mut solution)(input)?.0;

        verify(success(()), |_| {
            solution
                .replacements
                .iter()
                .try_fold(
                    HashSet::<&[ElementIndex]>::new(),
                    |mut present_replacements, replacement_data| {
                        present_replacements
                            .insert(
                                &solution.element_list[Self::range_usize_from_range_index(
                                    &replacement_data.replacing_elements,
                                )],
                            )
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
        if !args.verbose {
            dbg!(self.try_fastest_e_reduction_replacement_count());
        } else if let Some(fastest_e_reduction) = self.try_fastest_e_reduction() {
            match self.replacements_from_fastest_e_reduction(&fastest_e_reduction) {
                Ok(replacements) => {
                    dbg!(replacements.len());
                    dbg!(replacements);
                }
                Result::Err(err) => eprintln!("{err}"),
            }
        } else {
            eprintln!("Failed to find fastest e reduction");
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<NomError<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &[
        "\
        H => HO\n\
        H => OH\n\
        O => HH\n\
        \n\
        HOH\n",
        "\
        e => H\n\
        e => O\n\
        H => HO\n\
        H => OH\n\
        O => HH\n\
        \n\
        HOH\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
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
                },
                Solution {
                    element_table: vec![
                        Element {
                            id: "H".try_into().unwrap(),
                            data: ElementData {
                                replacements: 2_usize.into()..4_usize.into(),
                            },
                        },
                        Element {
                            id: "O".try_into().unwrap(),
                            data: ElementData {
                                replacements: 4_usize.into()..5_usize.into(),
                            },
                        },
                        Element {
                            id: "e".try_into().unwrap(),
                            data: ElementData {
                                replacements: 0_usize.into()..2_usize.into(),
                            },
                        },
                    ]
                    .try_into()
                    .unwrap(),
                    element_list: vec![
                        0_usize.into(),
                        1_usize.into(),
                        0_usize.into(),
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
                            replaced_element: 2_usize.into(),
                            replacing_elements: 0_usize.into()..1_usize.into(),
                        },
                        ReplacementData {
                            replaced_element: 2_usize.into(),
                            replacing_elements: 1_usize.into()..2_usize.into(),
                        },
                        ReplacementData {
                            replaced_element: 0_usize.into(),
                            replacing_elements: 2_usize.into()..4_usize.into(),
                        },
                        ReplacementData {
                            replaced_element: 0_usize.into(),
                            replacing_elements: 4_usize.into()..6_usize.into(),
                        },
                        ReplacementData {
                            replaced_element: 1_usize.into(),
                            replacing_elements: 6_usize.into()..8_usize.into(),
                        },
                    ],
                    molecule: 8_usize.into()..11_usize.into(),
                },
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
    fn test_replacement_molecule_hashes() {
        for (index, replacement_molecule_hashes_count) in [4_usize, 4_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).replacement_molecule_hashes().len(),
                replacement_molecule_hashes_count
            );
        }
    }

    #[test]
    fn test_try_fastest_e_reduction_replacement_count() {
        for (index, fastest_e_reduction_replacement_count) in
            [None, Some(3 as ReplacementCount)].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_fastest_e_reduction_replacement_count(),
                fastest_e_reduction_replacement_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
