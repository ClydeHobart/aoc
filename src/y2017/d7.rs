use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::{alpha1, line_ending},
        combinator::{map, map_opt, map_res, opt, success},
        error::Error,
        multi::{many0, many_m_n},
        sequence::{delimited, preceded},
        Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        fmt::{Debug, Formatter, Result as FmtResult},
    },
};

/* --- Day 7: Recursive Circus ---

Wandering further through the circuits of the computer, you come upon a tower of programs that have gotten themselves into a bit of trouble. A recursive algorithm has gotten out of hand, and now they're balanced precariously in a large tower.

One program at the bottom supports the entire tower. It's holding a large disc, and on the disc are balanced several more sub-towers. At the bottom of these sub-towers, standing on the bottom disc, are other programs, each holding their own disc, and so on. At the very tops of these sub-sub-sub-...-towers, many programs stand simply keeping the disc below them balanced but with no disc of their own.

You offer to help, but first you need to understand the structure of these towers. You ask each program to yell out their name, their weight, and (if they're holding a disc) the names of the programs immediately above them balancing on that disc. You write this information down (your puzzle input). Unfortunately, in their panic, they don't do this in an orderly fashion; by the time you're done, you're not sure which program gave which information.

For example, if your list is the following:

pbga (66)
xhth (57)
ebii (61)
havc (66)
ktlj (57)
fwft (72) -> ktlj, cntj, xhth
qoyq (66)
padx (45) -> pbga, havc, qoyq
tknk (41) -> ugml, padx, fwft
jptl (61)
ugml (68) -> gyxo, ebii, jptl
gyxo (61)
cntj (57)

...then you would be able to recreate the structure of the towers that looks like this:

                gyxo
              /
         ugml - ebii
       /      \
      |         jptl
      |
      |         pbga
     /        /
tknk --- padx - havc
     \        \
      |         qoyq
      |
      |         ktlj
       \      /
         fwft - cntj
              \
                xhth

In this example, tknk is at the bottom of the tower (the bottom program), and is holding up ugml, padx, and fwft. Those programs are, in turn, holding up other programs; in this example, none of those programs are holding up any other programs, and are all the tops of their own towers. (The actual tower balancing in front of you is much larger.)

Before you're ready to help them, you need to make sure your information is correct. What is the name of the bottom program?

--- Part Two ---

The programs explain the situation: they can't get down. Rather, they could get down, if they weren't expending all of their energy trying to keep the tower balanced. Apparently, one program has the wrong weight, and until it's fixed, they're stuck here.

For any program holding a disc, each program standing on that disc forms a sub-tower. Each of those sub-towers are supposed to be the same weight, or the disc itself isn't balanced. The weight of a tower is the sum of the weights of the programs in that tower.

In the example above, this means that for ugml's disc to be balanced, gyxo, ebii, and jptl must all have the same weight, and they do: 61.

However, for tknk to be balanced, each of the programs standing on its disc and all programs above it must each match. This means that the following sums must all be the same:

    ugml + (gyxo + ebii + jptl) = 68 + (61 + 61 + 61) = 251
    padx + (pbga + havc + qoyq) = 45 + (66 + 66 + 66) = 243
    fwft + (ktlj + cntj + xhth) = 72 + (57 + 57 + 57) = 243

As you can see, tknk's disc is unbalanced: ugml's stack is heavier than the other two. Even though the nodes above ugml are balanced, ugml itself is too heavy: it needs to be 8 units lighter for its stack to weigh 243 and keep the towers balanced. If this change were made, its weight would be 60.

Given that exactly one program is the wrong weight, what would its weight need to be to balance the entire tower? */

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct NodeIndex(usize);

impl NodeIndex {
    fn invalid() -> Self {
        Self(usize::MAX)
    }

    fn is_valid(self) -> bool {
        self != Self::invalid()
    }

    fn get(self) -> usize {
        self.0
    }
}

impl Debug for NodeIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.is_valid() {
            f.write_fmt(format_args!("{}", self.0))
        } else {
            f.write_str("<invalid>")
        }
    }
}

impl Default for NodeIndex {
    fn default() -> Self {
        Self::invalid()
    }
}

type Name = StaticString<{ Node::NAME_LEN }>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct Node {
    name: Name,
    weight: u16,
    children: [NodeIndex; Self::CHILDREN_LEN],
    parent: NodeIndex,
}

impl Node {
    const NAME_LEN: usize = 7_usize;
    const CHILDREN_LEN: usize = 7_usize;

    fn set_data(&mut self, weight: u16, children: &[NodeIndex; Node::CHILDREN_LEN]) {
        self.weight = weight;
        self.children = children.clone();
    }

    fn parse_name<'i>(input: &'i str) -> IResult<&'i str, Name> {
        map_res(alpha1, Name::try_from)(input)
    }

    fn valid_children_len(&self) -> usize {
        self.children
            .partition_point(|child_index| child_index.is_valid())
    }

    fn valid_children(&self) -> &[NodeIndex] {
        &self.children[..self.valid_children_len()]
    }

    fn valid_children_mut(&mut self) -> &mut [NodeIndex] {
        let valid_children_len: usize = self.valid_children_len();

        &mut self.children[..valid_children_len]
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    nodes: Vec<Node>,
}

#[derive(Default, Clone, Copy)]
struct IncorrectWeightCandidate {
    candidate: NodeIndex,
    expected_weight: i32,
}

#[derive(Default)]
struct IncorrectWeightCandidates {
    candidates: [IncorrectWeightCandidate; 2_usize],
    len: usize,
}

impl IncorrectWeightCandidates {
    fn pop(&mut self, index: usize) {
        assert!(index < self.len);

        self.candidates[index..self.len].rotate_left(1_usize);
        self.len -= 1_usize;
    }
}

impl Solution {
    fn find_or_add_node(&mut self, name: Name) -> NodeIndex {
        self.nodes
            .iter()
            .position(|node| node.name == name)
            .map_or_else(
                || {
                    let node_index: NodeIndex = NodeIndex(self.nodes.len());

                    self.nodes.push(Node {
                        name,
                        ..Node::default()
                    });

                    node_index
                },
                |index| NodeIndex(index),
            )
    }

    fn try_find_root_node(&mut self) -> Option<NodeIndex> {
        self.nodes
            .iter()
            .position(|node| !node.parent.is_valid())
            .map(NodeIndex)
    }

    fn topologically_sort(&mut self, root: NodeIndex) {
        assert!(root.is_valid());

        let name_to_curr_index: HashMap<Name, NodeIndex> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.name.clone(), NodeIndex(index)))
            .collect();
        let mut curr_index_to_next_index: Vec<NodeIndex> =
            vec![NodeIndex::invalid(); self.nodes.len()];
        let mut next_index: NodeIndex = NodeIndex(0_usize);
        let mut to_visit_queue: VecDeque<NodeIndex> = vec![root].into();

        while let Some(curr_index) = to_visit_queue.pop_front() {
            curr_index_to_next_index[curr_index.get()] = next_index;
            next_index.0 += 1_usize;
            to_visit_queue.extend(
                self.nodes[curr_index.get()]
                    .valid_children()
                    .iter()
                    .copied(),
            );
        }

        self.nodes
            .sort_by_key(|node| curr_index_to_next_index[name_to_curr_index[&node.name].get()].0);

        for node in self.nodes.iter_mut() {
            for child in node.valid_children_mut() {
                *child = curr_index_to_next_index[child.get()];
            }

            if node.parent.is_valid() {
                node.parent = curr_index_to_next_index[node.parent.get()];
            }
        }
    }

    fn bottom_program_name(&self) -> Option<Name> {
        self.nodes.first().map(|node| node.name.clone())
    }

    fn try_find_incorrect_weight_candidates(
        &self,
        parent: NodeIndex,
        total_weights: &[i32],
    ) -> Result<IncorrectWeightCandidates, Box<String>> {
        let children: &[NodeIndex] = self.nodes[parent.get()].valid_children();

        if children.len() < 2_usize {
            Ok(Default::default())
        } else {
            let mut children_weights: [(i32, u8); Node::CHILDREN_LEN] =
                [(-1_i32, 0_u8); Node::CHILDREN_LEN];
            let mut children_weights_len: usize = 0_usize;

            for child in children {
                let child_weight: i32 = total_weights[child.get()];

                if let Some(children_weights_index) = children_weights[..children_weights_len]
                    .iter()
                    .map(|(weight, _)| *weight)
                    .position(|weight| weight == child_weight)
                {
                    children_weights[children_weights_index].1 += 1_u8;
                } else {
                    children_weights[children_weights_len] = (child_weight, 1_u8);
                    children_weights_len += 1_usize;
                }
            }

            match children_weights_len {
                0_usize => unreachable!(),
                1_usize => Ok(Default::default()),
                2_usize => {
                    if children.len() == 2_usize {
                        let child_a: NodeIndex = children[0_usize];
                        let child_b: NodeIndex = children[1_usize];
                        let total_weight_a: i32 = total_weights[child_a.get()];
                        let total_weight_b: i32 = total_weights[child_b.get()];

                        Ok(IncorrectWeightCandidates {
                            candidates: [
                                IncorrectWeightCandidate {
                                    candidate: child_a,
                                    expected_weight: total_weight_b,
                                },
                                IncorrectWeightCandidate {
                                    candidate: child_b,
                                    expected_weight: total_weight_a,
                                },
                            ],
                            len: 2_usize,
                        })
                    } else {
                        let incorrect_total_weight: i32 = children_weights[..children_weights_len]
                            .iter()
                            .find_map(|(weight, count)| (*count == 1_u8).then_some(*weight))
                            .ok_or_else(|| {
                                format!(
                                    "`incorrect_weight_candidates` failed to find a weight that \
                                    only one child has."
                                )
                            })?;
                        let child: NodeIndex = *children
                            .iter()
                            .find(|child| total_weights[child.get()] == incorrect_total_weight)
                            .unwrap();
                        let correct_total_weight: i32 = children_weights
                            .iter()
                            .find_map(|(weight, _)| {
                                (*weight != incorrect_total_weight).then_some(*weight)
                            })
                            .unwrap();

                        Ok(IncorrectWeightCandidates {
                            candidates: [
                                IncorrectWeightCandidate {
                                    candidate: child,
                                    expected_weight: correct_total_weight,
                                },
                                Default::default(),
                            ],
                            len: 1_usize,
                        })
                    }
                }
                _ => {
                    std::hint::black_box(());

                    Err(format!(
                        "`incorrect_weight_candidates` found {children_weights_len} distinct weights, \
                        when either 1 or 2 were expected."
                    )
                    .into())
                }
            }
        }
    }

    fn try_find_incorrect_weight_candidate(
        &self,
        parent: NodeIndex,
        expected_delta: Option<i32>,
        total_weights: &[i32],
    ) -> Result<Option<IncorrectWeightCandidate>, Box<String>> {
        let children: &[NodeIndex] = self.nodes[parent.get()].valid_children();

        match children.len() {
            0_usize => Ok(None),
            1_usize => self.try_find_incorrect_weight_candidate(
                children[0_usize],
                expected_delta,
                total_weights,
            ),
            _ => {
                let mut incorrect_weight_candidates: IncorrectWeightCandidates =
                    self.try_find_incorrect_weight_candidates(parent, total_weights)?;

                for index in (0_usize..incorrect_weight_candidates.len).into_iter().rev() {
                    let incorrect_weight_candidate: &mut IncorrectWeightCandidate =
                        &mut incorrect_weight_candidates.candidates[index];
                    let actual_weight: i32 =
                        total_weights[incorrect_weight_candidate.candidate.get()];

                    if expected_delta.map_or(false, |expected_delta| {
                        expected_delta != incorrect_weight_candidate.expected_weight - actual_weight
                    }) {
                        incorrect_weight_candidates.pop(index);
                    } else if let Some(child_incorrect_weight_candidate) = self
                        .try_find_incorrect_weight_candidate(
                            incorrect_weight_candidate.candidate,
                            Some(incorrect_weight_candidate.expected_weight - actual_weight),
                            total_weights,
                        )?
                    {
                        // A lower node takes precedence.
                        *incorrect_weight_candidate = child_incorrect_weight_candidate;
                    }
                }

                match incorrect_weight_candidates.len {
                    0_usize => Ok(None),
                    1_usize => Ok(Some(incorrect_weight_candidates.candidates[0_usize])),
                    2_usize => {
                        // Greater node indices reflect nodes that are lower in the tree.
                        Ok(Some(
                            incorrect_weight_candidates
                                .candidates
                                .into_iter()
                                .max_by_key(|incorrect_weight_candidate| {
                                    incorrect_weight_candidate.candidate.0
                                })
                                .unwrap(),
                        ))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn sum_child_weights(&self, parent: NodeIndex, total_weights: &[i32]) -> i32 {
        self.nodes[parent.get()]
            .valid_children()
            .iter()
            .map(|child| total_weights[child.get()])
            .sum()
    }

    fn try_find_correct_weight(&self) -> Result<i32, Box<String>> {
        let mut total_weights: Vec<i32> = vec![-1_i32; self.nodes.len()];

        for (index, node) in self.nodes.iter().enumerate().rev() {
            total_weights[index] =
                node.weight as i32 + self.sum_child_weights(NodeIndex(index), &total_weights);
        }

        self.try_find_incorrect_weight_candidate(NodeIndex(0_usize), None, &total_weights)?
            .ok_or_else(|| {
                format!("`try_find_correct_weight` failed to find a valid candidate").into()
            })
            .map(|incorrect_weight_candidate| {
                incorrect_weight_candidate.expected_weight
                    - self.sum_child_weights(incorrect_weight_candidate.candidate, &total_weights)
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        let input: &str = many0(|input: &'i str| -> IResult<&'i str, ()> {
            let (input, name): (&str, Name) = Node::parse_name(input)?;
            let node_index: NodeIndex = solution.find_or_add_node(name);
            let (input, weight): (&str, u16) =
                delimited(tag(" ("), parse_integer, tag(")"))(input)?;
            let mut children: [NodeIndex; Node::CHILDREN_LEN] = Default::default();
            let mut children_len: usize = 0_usize;
            let input: &str = opt(preceded(
                tag(" -> "),
                many_m_n(
                    1_usize,
                    Node::CHILDREN_LEN,
                    map(preceded(opt(tag(", ")), Node::parse_name), |child_name| {
                        let child_node_index: NodeIndex = solution.find_or_add_node(child_name);
                        children[children_len] = child_node_index;
                        children_len += 1_usize;
                    }),
                ),
            ))(input)?
            .0;
            let input: &str = opt(line_ending)(input)?.0;

            for child_node_index in children
                .iter()
                .filter(|child_node_index| child_node_index.is_valid())
            {
                solution.nodes[child_node_index.get()].parent = node_index;
            }

            solution.nodes[node_index.get()].set_data(weight, &children);

            Ok((input, ()))
        })(input)?
        .0;

        let input: &str = map_opt(success(()), |_| {
            solution
                .try_find_root_node()
                .map(|root| solution.topologically_sort(root))
        })(input)?
        .0;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// Took a surprisingly long amount of time to figure out how to parse this correctly. Again,
    /// could've accelerated that by using more dynamically sized lists.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.bottom_program_name());
    }

    /// This took a while to work out, in particular due to the edge case of how to handle when
    /// there are only two children and their weights differ. Determining the correct node to change
    /// after encountering that fork was a bit of a headache, followed by confusion before I
    /// realized I wasn't summing up the total weights correctly.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_find_correct_weight()).ok();
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
        pbga (66)\n\
        xhth (57)\n\
        ebii (61)\n\
        havc (66)\n\
        ktlj (57)\n\
        fwft (72) -> ktlj, cntj, xhth\n\
        qoyq (66)\n\
        padx (45) -> pbga, havc, qoyq\n\
        tknk (41) -> ugml, padx, fwft\n\
        jptl (61)\n\
        ugml (68) -> gyxo, ebii, jptl\n\
        gyxo (61)\n\
        cntj (57)\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! children {
            [ $( $child:expr, )* ] => { {
                let mut children: [NodeIndex; Node::CHILDREN_LEN] = Default::default();
                let children_slice: &[NodeIndex] = &[ $( NodeIndex($child), )* ];

                children[..children_slice.len()].copy_from_slice(children_slice);

                children
            } }
        }

        macro_rules! solution {
            {
                [ $( {
                    $name:expr,
                    $weight:expr,
                    [ $( $child:expr, )* ],
                    $parent:expr,
                }, )* ],
            } => {
                Solution {
                    nodes: vec![ $( Node {
                        name: Name::try_from($name).unwrap(),
                        weight: $weight,
                        children: children![ $( $child, )* ],
                        parent: NodeIndex($parent),
                    }, )* ],
                }
            }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![solution! { [
                { "tknk", 41_u16, [1_usize, 2_usize, 3_usize, ], usize::MAX, }, // 0
                { "ugml", 68_u16, [4_usize, 5_usize, 6_usize, ], 0_usize, }, // 1
                { "padx", 45_u16, [7_usize, 8_usize, 9_usize, ], 0_usize, }, // 2
                { "fwft", 72_u16, [10_usize, 11_usize, 12_usize, ], 0_usize, }, // 3
                { "gyxo", 61_u16, [], 1_usize, }, // 4
                { "ebii", 61_u16, [], 1_usize, }, // 5
                { "jptl", 61_u16, [], 1_usize, }, // 6
                { "pbga", 66_u16, [], 2_usize, }, // 7
                { "havc", 66_u16, [], 2_usize, }, // 8
                { "qoyq", 66_u16, [], 2_usize, }, // 9
                { "ktlj", 57_u16, [], 3_usize, }, // 10
                { "cntj", 57_u16, [], 3_usize, }, // 11
                { "xhth", 57_u16, [], 3_usize, }, // 12
            ], }]
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
    fn test_bottom_program_name() {
        for (index, bottom_program_name) in
            [Name::try_from("tknk").unwrap()].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).bottom_program_name(),
                Some(bottom_program_name)
            );
        }
    }

    #[test]
    fn test_try_find_correct_weight() {
        for (index, try_find_correct_weight) in [Ok(60_i32)].into_iter().enumerate() {
            assert_eq!(
                solution(index).try_find_correct_weight(),
                try_find_correct_weight
            );
        }
    }
}
