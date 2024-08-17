use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::ops::{Range, RangeInclusive},
};

/* --- Day 20: Firewall Rules ---

You'd like to set up a small hidden computer here so you can use it to get back into the network later. However, the corporate firewall only allows communication with certain external IP addresses.

You've retrieved the list of blocked IPs from the firewall, but the list seems to be messy and poorly maintained, and it's not clear which IPs are allowed. Also, rather than being written in dot-decimal notation, they are written as plain 32-bit integers, which can have any value from 0 through 4294967295, inclusive.

For example, suppose only the values 0 through 9 were valid, and that you retrieved the following blacklist:

5-8
0-2
4-7

The blacklist specifies ranges of IPs (inclusive of both the start and end value) that are not allowed. Then, the only IPs that this firewall allows are 3 and 9, since those are the only numbers not in any range.

Given the list of blocked IPs you retrieved from the firewall (your puzzle input), what is the lowest-valued IP that is not blocked?

--- Part Two ---

How many IPs are allowed by the blacklist? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct RangeInc {
    start: u32,
    end: u32,
}

impl From<RangeInc> for RangeInclusive<u32> {
    fn from(value: RangeInc) -> Self {
        value.start..=value.end
    }
}

impl From<RangeInclusive<u32>> for RangeInc {
    fn from(value: RangeInclusive<u32>) -> Self {
        let (start, end): (u32, u32) = value.into_inner();

        Self { start, end }
    }
}

impl From<RangeInc> for Range<usize> {
    fn from(value: RangeInc) -> Self {
        let start: usize = value.start as usize;
        let end: usize = value.end as usize + 1_usize;

        start..end
    }
}

impl Parse for RangeInc {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(parse_integer::<u32>, tag("-"), parse_integer::<u32>),
            |(start, end)| (start..=end).into(),
        )(input)
    }
}

impl TryFrom<Range<usize>> for RangeInc {
    type Error = ();

    fn try_from(value: Range<usize>) -> Result<Self, Self::Error> {
        u32::try_from(value.start)
            .ok()
            .zip(
                value
                    .end
                    .checked_sub(1_usize)
                    .and_then(|end| u32::try_from(end).ok()),
            )
            .map(|(start, end)| Self { start, end })
            .ok_or(())
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Node {
    range: RangeInc,
    node_type: NodeType,
}

impl Node {
    fn new(range: RangeInc) -> Self {
        Self {
            range,
            node_type: NodeType::Leaf { allow: true },
        }
    }

    fn denies_all(&self) -> bool {
        match &self.node_type {
            NodeType::Leaf { allow } => !*allow,
            NodeType::Parent(children) => children.iter().all(Self::denies_all),
        }
    }

    fn try_min_allowed(&self) -> Option<u32> {
        match &self.node_type {
            NodeType::Leaf { allow } => (*allow).then_some(self.range.start),
            NodeType::Parent(children) => children.iter().filter_map(Self::try_min_allowed).next(),
        }
    }

    fn allowed_count(&self) -> usize {
        match &self.node_type {
            NodeType::Leaf { allow } => *allow as usize * Range::<usize>::from(self.range).len(),
            NodeType::Parent(children) => children.iter().map(Self::allowed_count).sum(),
        }
    }

    // Returns whether the node is denies all.
    fn deny(&mut self, range: Range<usize>) -> bool {
        let self_range: Range<usize> = self.range.into();

        if let Some(intersection) = try_non_empty_intersection(range, self_range.clone()) {
            if intersection == self_range {
                self.node_type = NodeType::Leaf { allow: false };

                true
            } else {
                match &mut self.node_type {
                    NodeType::Leaf { allow } => {
                        if *allow {
                            let mid: usize = (self_range.start + self_range.end) / 2_usize;
                            let mut children: Box<[Self; 2_usize]> = Box::new([
                                Self::new((self_range.start..mid).try_into().unwrap()),
                                Self::new((mid..self_range.end).try_into().unwrap()),
                            ]);

                            // Don't bother paying attention to the return values of the `deny`
                            // here, since at least one of them won't deny all.
                            for child in children.iter_mut() {
                                child.deny(intersection.clone());
                            }

                            self.node_type = NodeType::Parent(children);

                            false
                        } else {
                            true
                        }
                    }
                    NodeType::Parent(children) => {
                        if children.iter_mut().fold(true, |denies_all, child| {
                            child.deny(intersection.clone()) && denies_all
                        }) {
                            self.node_type = NodeType::Leaf { allow: false };

                            true
                        } else {
                            false
                        }
                    }
                }
            }
        } else {
            self.denies_all()
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum NodeType {
    Leaf { allow: bool },
    Parent(Box<[Node; 2_usize]>),
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<RangeInc>);

impl Solution {
    const RANGE: RangeInc = RangeInc {
        start: 0_u32,
        end: u32::MAX,
    };

    fn node(&self, range: RangeInc) -> Node {
        let mut node: Node = Node::new(range);

        for range in self.0.iter() {
            node.deny((*range).into());
        }

        node
    }

    fn try_min_allowed(&self, range: RangeInc) -> Option<u32> {
        self.node(range).try_min_allowed()
    }

    fn allowed_count(&self, range: RangeInc) -> usize {
        self.node(range).allowed_count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(RangeInc::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_min_allowed(Self::RANGE));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.allowed_count(Self::RANGE));
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
        5-8\n\
        0-2\n\
        4-7\n";
    const RANGE: RangeInc = RangeInc {
        start: 0_u32,
        end: 9_u32,
    };

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                (5_u32..=8_u32).into(),
                (0_u32..=2_u32).into(),
                (4_u32..=7_u32).into(),
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_deny() {
        macro_rules! node {
            ($range:expr, $allow:literal) => {
                Node {
                    range: ($range).into(),
                    node_type: NodeType::Leaf { allow: $allow },
                }
            };
            ($range:expr, $left:expr, $right:expr) => {
                Node {
                    range: ($range).into(),
                    node_type: NodeType::Parent(Box::new([$left, $right])),
                }
            };
        }

        let mut node: Node = Node::new(RANGE);
        let mut range_iter = solution()
            .0
            .iter()
            .map(|range_inc| Range::<usize>::from(*range_inc));

        node.deny(range_iter.next().unwrap());

        assert_eq!(
            &node,
            &node!(
                0_u32..=9_u32,
                node!(0_u32..=4_u32, true),
                node!(
                    5_u32..=9_u32,
                    node!(5..=6_u32, false),
                    node!(
                        7_u32..=9_u32,
                        node!(7_u32..=7_u32, false),
                        node!(
                            8_u32..=9_u32,
                            node!(8_u32..=8_u32, false),
                            node!(9_u32..=9_u32, true)
                        )
                    )
                )
            )
        );

        node.deny(range_iter.next().unwrap());

        assert_eq!(
            &node,
            &node!(
                0_u32..=9_u32,
                node!(
                    0_u32..=4_u32,
                    node!(0_u32..=1_u32, false),
                    node!(
                        2_u32..=4_u32,
                        node!(2_u32..=2_u32, false),
                        node!(3_u32..=4_u32, true)
                    )
                ),
                node!(
                    5_u32..=9_u32,
                    node!(5..=6_u32, false),
                    node!(
                        7_u32..=9_u32,
                        node!(7_u32..=7_u32, false),
                        node!(
                            8_u32..=9_u32,
                            node!(8_u32..=8_u32, false),
                            node!(9_u32..=9_u32, true)
                        )
                    )
                )
            )
        );

        node.deny(range_iter.next().unwrap());

        assert_eq!(
            &node,
            &node!(
                0_u32..=9_u32,
                node!(
                    0_u32..=4_u32,
                    node!(0_u32..=1_u32, false),
                    node!(
                        2_u32..=4_u32,
                        node!(2_u32..=2_u32, false),
                        node!(
                            3_u32..=4_u32,
                            node!(3_u32..=3_u32, true),
                            node!(4_u32..=4_u32, false)
                        )
                    )
                ),
                node!(
                    5_u32..=9_u32,
                    node!(5..=6_u32, false),
                    node!(
                        7_u32..=9_u32,
                        node!(7_u32..=7_u32, false),
                        node!(
                            8_u32..=9_u32,
                            node!(8_u32..=8_u32, false),
                            node!(9_u32..=9_u32, true)
                        )
                    )
                )
            )
        );

        node.deny(3_usize..4_usize);

        assert_eq!(
            &node,
            &node!(
                0_u32..=9_u32,
                node!(0_u32..=4_u32, false),
                node!(
                    5_u32..=9_u32,
                    node!(5..=6_u32, false),
                    node!(
                        7_u32..=9_u32,
                        node!(7_u32..=7_u32, false),
                        node!(
                            8_u32..=9_u32,
                            node!(8_u32..=8_u32, false),
                            node!(9_u32..=9_u32, true)
                        )
                    )
                )
            )
        );

        node.deny(9_usize..10_usize);

        assert_eq!(&node, &node!(0_u32..=9_u32, false));
    }

    #[test]
    fn test_try_min_allowed() {
        assert_eq!(solution().try_min_allowed(RANGE), Some(3_u32));
    }
}
