use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0_count,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        mem::size_of,
        str::from_utf8,
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Ord, PartialOrd))]
struct Tag([u8; Tag::SIZE]);

impl Tag {
    const MIN_SIZE: usize = 1_usize;
    const SIZE: usize = 2_usize;
    const START: Self = Self([u8::MIN; Tag::SIZE]);
    const END: Self = Self([u8::MAX; Tag::SIZE]);

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(tag("start"), |_| Self::START),
            map(tag("end"), |_| Self::END),
            map(
                alt((
                    take_while_m_n(Self::MIN_SIZE, Self::SIZE, |c: char| c.is_ascii_lowercase()),
                    take_while_m_n(Self::MIN_SIZE, Self::SIZE, |c: char| c.is_ascii_uppercase()),
                )),
                Self::from_valid_non_terminal_str,
            ),
        ))(input)
    }

    fn from_valid_non_terminal_str(input: &str) -> Self {
        let mut tag: Self = Self::START;

        tag.0[..input.len()].copy_from_slice(input.as_bytes());

        tag
    }

    fn is_small_or_terminal(self) -> bool {
        self == Self::START || self == Self::END || (self.0[0_usize] as char).is_ascii_lowercase()
    }

    fn as_str(&self) -> &str {
        const INVALID_TAG_STATE: &str = "`Tag` was in an invalid state!";

        match *self {
            Self::START => "start",
            Self::END => "end",
            _ if self.0[1_usize] != 0_u8 => from_utf8(&self.0).expect(INVALID_TAG_STATE),
            _ => from_utf8(&self.0[..1_usize]).expect(INVALID_TAG_STATE),
        }
    }

    #[cfg(test)]
    fn try_from_str(input: &str) -> Option<Self> {
        Some(match input {
            "start" => Self::START,
            "end" => Self::END,
            _ if Self::is_valid_non_terminal_str(input) => Self::from_valid_non_terminal_str(input),
            _ => None?,
        })
    }

    #[cfg(test)]
    fn is_valid_non_terminal_str(input: &str) -> bool {
        (Self::MIN_SIZE..=Self::SIZE).contains(&input.len())
            && (input.chars().all(|c| c.is_ascii_lowercase())
                || input.chars().all(|c| c.is_ascii_uppercase()))
    }
}

impl Debug for Tag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
#[cfg_attr(test, derive(Debug))]
struct Set([u8; Set::SIZE]);

impl Set {
    const SIZE: usize = Vertex::SIZE - Tag::SIZE;
    const MAX_INDEX: usize = Set::SIZE * u8::BITS as usize;

    fn from_slice(slice: &[usize]) -> Self {
        let mut neighbors: Self = Self::default();
        let bits: &mut BitSlice<u8> = neighbors.0.as_mut_bits::<Lsb0>();

        for index in slice.iter().copied() {
            bits.set(index, true);
        }

        neighbors
    }

    fn add(&mut self, index: usize) {
        self.0.as_mut_bits::<Lsb0>().set(index, true);
    }

    fn remove(&mut self, index: usize) {
        self.0.as_mut_bits::<Lsb0>().set(index, false);
    }

    fn has(self, index: usize) -> bool {
        self.0.as_bits::<Lsb0>()[index]
    }

    fn iter_non_start(&self) -> impl Iterator<Item = usize> + '_ {
        self.0
            .as_bits::<Lsb0>()
            .iter_ones()
            .filter(|index| *index != Solution::START)
    }

    fn next_non_start(self) -> Option<usize> {
        self.iter_non_start().next()
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Vertex {
    tag: Tag,
    neighbors: Set,
}

impl Vertex {
    const SIZE: usize = size_of::<u32>();

    fn new(tag: Tag) -> Self {
        Self {
            tag,
            neighbors: Set::default(),
        }
    }
}

#[derive(Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Paths {
    count: usize,

    #[cfg(test)]
    paths: Vec<Vec<Tag>>,
}

struct StackElement {
    vertex: Vertex,
    index: usize,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Vertex>);

impl Solution {
    const START: usize = 0_usize;
    const END: usize = 1_usize;

    fn stack_element(&self, index: usize) -> StackElement {
        StackElement {
            vertex: self.0[index],
            index,
        }
    }

    fn count_paths_internal(&self, can_revisit: bool) -> Paths {
        let mut paths: Paths = Paths::default();
        let mut stack: Vec<StackElement> = vec![self.stack_element(Self::START)];
        let mut present: Set = Set::from_slice(&[Self::START]);
        let mut revisit: Option<usize> = None;

        while let Some(stack_element) = stack.last_mut() {
            if stack_element.index == Self::END {
                let index: usize = stack_element.index;

                paths.count += 1_usize;

                #[cfg(test)]
                paths.paths.push(
                    stack
                        .iter()
                        .map(|stack_element| stack_element.vertex.tag)
                        .collect(),
                );

                stack.pop();
                present.remove(index);
            } else if let Some(next) = stack_element.vertex.neighbors.next_non_start() {
                stack_element.vertex.neighbors.remove(next);

                if !self.0[next].tag.is_small_or_terminal() || !present.has(next) {
                    stack.push(self.stack_element(next));
                    present.add(next);
                } else if next != Self::START && can_revisit && revisit.is_none() {
                    stack.push(self.stack_element(next));
                    present.add(next);
                    revisit = Some(next);
                }
            } else {
                let index: usize = stack_element.index;

                stack.pop();

                if revisit.map_or(false, |revisit_index| index == revisit_index) {
                    revisit = None;
                } else {
                    present.remove(index);
                }
            }
        }

        #[cfg(test)]
        paths.paths.sort();

        paths
    }

    fn count_paths(&self) -> usize {
        self.count_paths_internal(false).count
    }

    fn count_paths_with_revisit(&self) -> usize {
        self.count_paths_internal(true).count
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_paths());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_paths_with_revisit());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        struct SolutionWithMap {
            solution: Solution,
            tag_to_index: HashMap<Tag, usize>,
        }

        impl SolutionWithMap {
            fn new() -> Self {
                let mut solution_with_map: Self = Self {
                    solution: Solution(Vec::with_capacity(2_usize)),
                    tag_to_index: HashMap::new(),
                };

                solution_with_map.add_tag(Tag::START);
                solution_with_map.add_tag(Tag::END);

                solution_with_map
            }

            fn add_tag(&mut self, tag: Tag) -> Option<usize> {
                if let Some(index) = self.tag_to_index.get(&tag) {
                    Some(*index)
                } else {
                    let index: usize = self.solution.0.len();

                    if index < Set::MAX_INDEX {
                        self.tag_to_index.insert(tag, index);
                        self.solution.0.push(Vertex::new(tag));

                        Some(index)
                    } else {
                        None
                    }
                }
            }

            fn add_neighbors(&mut self, a: Tag, b: Tag) -> Option<()> {
                let a_index: usize = self.add_tag(a)?;
                let b_index: usize = self.add_tag(b)?;

                self.solution.0[a_index as usize].neighbors.add(b_index);
                self.solution.0[b_index as usize].neighbors.add(a_index);

                Some(())
            }
        }

        let mut solution_with_map: SolutionWithMap = SolutionWithMap::new();

        many0_count(map_opt(
            terminated(
                separated_pair(Tag::parse, tag("-"), Tag::parse),
                opt(line_ending),
            ),
            |(a, b)| solution_with_map.add_neighbors(a, b),
        ))(input)?;

        Ok(solution_with_map.solution)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const SOLUTION_1_STR: &str = concat!(
        "start-A\n",
        "start-b\n",
        "A-c\n",
        "A-b\n",
        "b-d\n",
        "A-end\n",
        "b-end\n",
    );
    const SOLUTION_2_STR: &str = concat!(
        "dc-end\n",
        "HN-start\n",
        "start-kj\n",
        "dc-start\n",
        "dc-HN\n",
        "LN-dc\n",
        "HN-end\n",
        "kj-sa\n",
        "kj-HN\n",
        "kj-dc\n",
    );
    const SOLUTION_3_STR: &str = concat!(
        "fs-end\n",
        "he-DX\n",
        "fs-he\n",
        "start-DX\n",
        "pj-DX\n",
        "end-zg\n",
        "zg-sl\n",
        "zg-pj\n",
        "pj-he\n",
        "RW-he\n",
        "fs-DX\n",
        "pj-RW\n",
        "zg-RW\n",
        "start-pj\n",
        "he-WI\n",
        "zg-he\n",
        "pj-fs\n",
        "start-RW\n",
    );

    lazy_static! {
        static ref SOLUTION_1: Solution = solution_1();
        static ref SOLUTION_2: Solution = solution_2();
        static ref SOLUTION_3: Solution = solution_3();
    }

    macro_rules! tag {
        ($tag:expr) => {
            Tag::try_from_str($tag).unwrap()
        };
    }

    macro_rules! solution {
        [ $( { $tag:expr, [ $( $neighbor:expr ),* ] }, )* ] => { Solution(vec![ $( Vertex {
            tag: tag!($tag),
            neighbors: Set::from_slice(&[ $( $neighbor, )* ]),
        }, )* ]) };
    }

    fn solution_1() -> Solution {
        solution![
            { "start",  [ 2, 3 ] }, // 0
            { "end",    [ 2, 3 ] }, // 1
            { "A",      [ 0, 1, 3, 4 ] }, // 2
            { "b",      [ 0, 1, 2, 5 ] }, // 3
            { "c",      [ 2 ] }, // 4
            { "d",      [ 3 ] }, // 5
        ]
    }

    fn solution_2() -> Solution {
        solution![
            { "start",  [ 2, 3, 4 ] }, // 0
            { "end",    [ 2, 3 ] }, // 1
            { "dc",     [ 0, 1, 3, 4, 5 ] }, // 2
            { "HN",     [ 0, 1, 2, 4 ] }, // 3
            { "kj",     [ 0, 2, 3, 6 ] }, // 4
            { "LN",     [ 2 ] }, // 5
            { "sa",     [ 4 ] }, // 6
        ]
    }

    fn solution_3() -> Solution {
        solution![
            { "start",  [ 4, 5, 8 ] }, // 0
            { "end",    [ 2, 6 ] }, // 1
            { "fs",     [ 1, 3, 4, 5 ] }, // 2
            { "he",     [ 2, 4, 5, 6, 8, 9 ] }, // 3
            { "DX",     [ 0, 2, 3, 5 ] }, // 4
            { "pj",     [ 0, 2, 3, 4, 6, 8 ] }, // 5
            { "zg",     [ 1, 3, 5, 7, 8 ] }, // 6
            { "sl",     [ 6 ] }, // 7
            { "RW",     [ 0, 3, 5, 6 ] }, // 8
            { "WI",     [ 3 ] }, // 9
        ]
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_1_STR), Ok(solution_1()));
        assert_eq!(Solution::try_from(SOLUTION_2_STR), Ok(solution_2()));
        assert_eq!(Solution::try_from(SOLUTION_3_STR), Ok(solution_3()));
    }

    #[test]
    fn test_count_paths() {
        macro_rules! paths {
            { $count:expr, [ $( [ $( $tag:expr ),* ], )* ], } => {
                Paths {
                    count: $count,
                    paths: vec![ $( vec![ $( tag!($tag), )* ], )* ]
                }
            };
        }

        assert_eq!(
            SOLUTION_1.count_paths_internal(false),
            paths! {
                10,
                [
                    [ "start", "A", "b", "A", "c", "A", "end" ],
                    [ "start", "A", "b", "A", "end" ],
                    [ "start", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "b", "A", "end" ],
                    [ "start", "A", "c", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "end" ],
                    [ "start", "A", "end" ],
                    [ "start", "b", "A", "c", "A", "end" ],
                    [ "start", "b", "A", "end" ],
                    [ "start", "b", "end" ],
                ],
            }
        );
        assert_eq!(
            SOLUTION_2.count_paths_internal(false),
            paths! {
                19,
                [
                    [ "start", "HN", "dc", "HN", "kj", "HN", "end" ],
                    [ "start", "HN", "dc", "HN", "end" ],
                    [ "start", "HN", "dc", "kj", "HN", "end" ],
                    [ "start", "HN", "dc", "end" ],
                    [ "start", "HN", "kj", "HN", "dc", "HN", "end" ],
                    [ "start", "HN", "kj", "HN", "dc", "end" ],
                    [ "start", "HN", "kj", "HN", "end" ],
                    [ "start", "HN", "kj", "dc", "HN", "end" ],
                    [ "start", "HN", "kj", "dc", "end" ],
                    [ "start", "HN", "end" ],
                    [ "start", "dc", "HN", "kj", "HN", "end" ],
                    [ "start", "dc", "HN", "end" ],
                    [ "start", "dc", "kj", "HN", "end" ],
                    [ "start", "dc", "end" ],
                    [ "start", "kj", "HN", "dc", "HN", "end" ],
                    [ "start", "kj", "HN", "dc", "end" ],
                    [ "start", "kj", "HN", "end" ],
                    [ "start", "kj", "dc", "HN", "end" ],
                    [ "start", "kj", "dc", "end" ],
                ],
            }
        );
        assert_eq!(SOLUTION_3.count_paths(), 226_usize);
    }

    #[test]
    fn test_count_paths_with_revisit() {
        macro_rules! paths {
            { $count:expr, [ $( [ $( $tag:expr ),* ], )* ], } => {
                Paths {
                    count: $count,
                    paths: vec![ $( vec![ $( tag!($tag), )* ], )* ]
                }
            };
        }

        assert_eq!(
            SOLUTION_1.count_paths_internal(true),
            paths! {
                36,
                [
                    [ "start", "A", "b", "A", "b", "A", "c", "A", "end" ],
                    [ "start", "A", "b", "A", "b", "A", "end" ],
                    [ "start", "A", "b", "A", "b", "end" ],
                    [ "start", "A", "b", "A", "c", "A", "b", "A", "end" ],
                    [ "start", "A", "b", "A", "c", "A", "b", "end" ],
                    [ "start", "A", "b", "A", "c", "A", "c", "A", "end" ],
                    [ "start", "A", "b", "A", "c", "A", "end" ],
                    [ "start", "A", "b", "A", "end" ],
                    [ "start", "A", "b", "d", "b", "A", "c", "A", "end" ],
                    [ "start", "A", "b", "d", "b", "A", "end" ],
                    [ "start", "A", "b", "d", "b", "end" ],
                    [ "start", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "b", "A", "b", "A", "end" ],
                    [ "start", "A", "c", "A", "b", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "b", "A", "c", "A", "end" ],
                    [ "start", "A", "c", "A", "b", "A", "end" ],
                    [ "start", "A", "c", "A", "b", "d", "b", "A", "end" ],
                    [ "start", "A", "c", "A", "b", "d", "b", "end" ],
                    [ "start", "A", "c", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "c", "A", "b", "A", "end" ],
                    [ "start", "A", "c", "A", "c", "A", "b", "end" ],
                    [ "start", "A", "c", "A", "c", "A", "end" ],
                    [ "start", "A", "c", "A", "end" ],
                    [ "start", "A", "end" ],
                    [ "start", "b", "A", "b", "A", "c", "A", "end" ],
                    [ "start", "b", "A", "b", "A", "end" ],
                    [ "start", "b", "A", "b", "end" ],
                    [ "start", "b", "A", "c", "A", "b", "A", "end" ],
                    [ "start", "b", "A", "c", "A", "b", "end" ],
                    [ "start", "b", "A", "c", "A", "c", "A", "end" ],
                    [ "start", "b", "A", "c", "A", "end" ],
                    [ "start", "b", "A", "end" ],
                    [ "start", "b", "d", "b", "A", "c", "A", "end" ],
                    [ "start", "b", "d", "b", "A", "end" ],
                    [ "start", "b", "d", "b", "end" ],
                    [ "start", "b", "end" ],
                ],
            }
        );
        assert_eq!(SOLUTION_2.count_paths_with_revisit(), 103_usize);
        assert_eq!(SOLUTION_3.count_paths_with_revisit(), 3509_usize);
    }
}
