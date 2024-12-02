use {
    crate::*,
    nom::{
        bytes::complete::tag,
        combinator::{map, opt},
        error::Error,
        multi::many_m_n,
        sequence::{terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 8: Memory Maneuver ---

The sleigh is much easier to pull than you'd expect for something its weight. Unfortunately, neither you nor the Elves know which way the North Pole is from here.

You check your wrist device for anything that might help. It seems to have some kind of navigation system! Activating the navigation system produces more bad news: "Failed to start navigation system. Could not read software license file."

The navigation system's license file consists of a list of numbers (your puzzle input). The numbers define a data structure which, when processed, produces some kind of tree that can be used to calculate the license number.

The tree is made up of nodes; a single, outermost node forms the tree's root, and it contains all other nodes in the tree (or contains nodes that contain nodes, and so on).

Specifically, a node consists of:

    A header, which is always exactly two numbers:
        The quantity of child nodes.
        The quantity of metadata entries.
    Zero or more child nodes (as specified in the header).
    One or more metadata entries (as specified in the header).

Each child node is itself a node that has its own header, child nodes, and metadata. For example:

2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2
A----------------------------------
    B----------- C-----------
                     D-----

In this example, each node of the tree is also marked with an underline starting with a letter for easier identification. In it, there are four nodes:

    A, which has 2 child nodes (B, C) and 3 metadata entries (1, 1, 2).
    B, which has 0 child nodes and 3 metadata entries (10, 11, 12).
    C, which has 1 child node (D) and 1 metadata entry (2).
    D, which has 0 child nodes and 1 metadata entry (99).

The first check done on the license file is to simply add up all of the metadata entries. In this example, that sum is 1+1+2+10+11+12+2+99=138.

What is the sum of all metadata entries?

--- Part Two ---

The second check is slightly more complicated: you need to find the value of the root node (A in the example above).

The value of a node depends on whether it has child nodes.

If a node has no child nodes, its value is the sum of its metadata entries. So, the value of node B is 10+11+12=33, and the value of node D is 99.

However, if a node does have child nodes, the metadata entries become indexes which refer to those child nodes. A metadata entry of 1 refers to the first child node, 2 to the second, 3 to the third, and so on. The value of this node is the sum of the values of the child nodes referenced by the metadata entries. If a referenced child node does not exist, that reference is skipped. A child node can be referenced multiple time and counts each time it is referenced. A metadata entry of 0 does not refer to any child node.

For example, again using the above nodes:

    Node C has one metadata entry, 2. Because node C has only one child node, 2 references a child node which does not exist, and so the value of node C is 0.
    Node A has three metadata entries: 1, 1, and 2. The 1 references node A's first child node, B, and the 2 references node A's second child node, C. Because node B has a value of 33 and node C has a value of 0, the value of node A is 33+33+0=66.

So, in this example, the value of the root node is 66.

What is the value of the root node? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Node {
    children: Vec<Node>,
    metadata: Vec<u8>,
}

impl Node {
    fn visit_all_metadata_internal<F: FnMut(u8)>(&self, f: &mut F) {
        for metadata in self.metadata.iter() {
            f(*metadata);
        }

        for child in &self.children {
            child.visit_all_metadata_internal(f);
        }
    }

    fn visit_all_metadata<F: FnMut(u8)>(&self, f: F) {
        let mut f: F = f;

        self.visit_all_metadata_internal(&mut f);
    }

    fn value(&self) -> u32 {
        if self.children.is_empty() {
            self.metadata.iter().copied().map(u32::from).sum()
        } else {
            self.metadata
                .iter()
                .filter_map(|metadata_index| {
                    metadata_index
                        .checked_sub(1_u8)
                        .and_then(|metadata_index| self.children.get(metadata_index as usize))
                        .map(|child| child.value())
                })
                .sum()
        }
    }
}

impl Parse for Node {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, (children_len, metadata_len)): (&str, (usize, usize)) = map(
            tuple((parse_integer, tag(" "), parse_integer, tag(" "))),
            |(children_len, _, metadata_len, _)| (children_len, metadata_len),
        )(input)?;

        map(
            tuple((
                many_m_n(
                    children_len,
                    children_len,
                    terminated(Self::parse, opt(tag(" "))),
                ),
                many_m_n(
                    metadata_len,
                    metadata_len,
                    terminated(parse_integer, opt(tag(" "))),
                ),
            )),
            |(children, metadata)| Self { children, metadata },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Node);

impl Solution {
    fn sum_all_metadata(&self) -> u32 {
        let mut all_metadata_sum: u32 = 0_u32;

        self.0
            .visit_all_metadata(|metadata| all_metadata_sum += metadata as u32);

        all_metadata_sum
    }

    fn value(&self) -> u32 {
        self.0.value()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Node::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// I wish it was easier to iterate trees than with a helper function.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_all_metadata());
    }

    /// Not super challenging
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.value());
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

    const SOLUTION_STRS: &'static [&'static str] = &["2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(Node {
                children: vec![
                    Node {
                        children: Vec::new(),
                        metadata: vec![10_u8, 11_u8, 12_u8],
                    },
                    Node {
                        children: vec![Node {
                            children: Vec::new(),
                            metadata: vec![99_u8],
                        }],
                        metadata: vec![2_u8],
                    },
                ],
                metadata: vec![1_u8, 1_u8, 2_u8],
            })]
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
    fn test_visit_all_metadata() {
        for (index, expected_metadata) in [vec![1_u8, 1_u8, 2_u8, 10_u8, 11_u8, 12_u8, 2_u8, 99_u8]]
            .into_iter()
            .enumerate()
        {
            let mut actual_metadata: Vec<u8> = Vec::new();

            solution(index)
                .0
                .visit_all_metadata(|metadata| actual_metadata.push(metadata));

            assert_eq!(actual_metadata, expected_metadata);
        }
    }

    #[test]
    fn test_sum_all_metadata() {
        for (index, all_metadata_sum) in [138_u32].into_iter().enumerate() {
            assert_eq!(solution(index).sum_all_metadata(), all_metadata_sum);
        }
    }

    #[test]
    fn test_value() {
        for (index, value) in [66_u32].into_iter().enumerate() {
            assert_eq!(solution(index).value(), value);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
