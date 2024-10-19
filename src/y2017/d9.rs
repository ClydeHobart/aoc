use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{anychar, none_of},
        combinator::{cond, map, opt, success, verify},
        error::Error,
        multi::{many0, many0_count},
        sequence::{delimited, preceded, tuple},
        Err, IResult, Parser,
    },
    std::ops::Range,
};

/* --- Day 9: Stream Processing ---

A large stream blocks your path. According to the locals, it's not safe to cross the stream at the moment because it's full of garbage. You look down at the stream; rather than water, you discover that it's a stream of characters.

You sit for a while and record part of the stream (your puzzle input). The characters represent groups - sequences that begin with { and end with }. Within a group, there are zero or more other things, separated by commas: either another group or garbage. Since groups can contain other groups, a } only closes the most-recently-opened unclosed group - that is, they are nestable. Your puzzle input represents a single, large group which itself contains many smaller ones.

Sometimes, instead of a group, you will find garbage. Garbage begins with < and ends with >. Between those angle brackets, almost any character can appear, including { and }. Within garbage, < has no special meaning.

In a futile attempt to clean up the garbage, some program has canceled some of the characters within it using !: inside garbage, any character that comes after ! should be ignored, including <, >, and even another !.

You don't see any characters that deviate from these rules. Outside garbage, you only find well-formed groups, and garbage always terminates according to the rules above.

Here are some self-contained pieces of garbage:

    <>, empty garbage.
    <random characters>, garbage containing random characters.
    <<<<>, because the extra < are ignored.
    <{!>}>, because the first > is canceled.
    <!!>, because the second ! is canceled, allowing the > to terminate the garbage.
    <!!!>>, because the second ! and the first > are canceled.
    <{o"i!a,<{i<a>, which ends at the first >.

Here are some examples of whole streams and the number of groups they contain:

    {}, 1 group.
    {{{}}}, 3 groups.
    {{},{}}, also 3 groups.
    {{{},{},{{}}}}, 6 groups.
    {<{},{},{{}}>}, 1 group (which itself contains garbage).
    {<a>,<a>,<a>,<a>}, 1 group.
    {{<a>},{<a>},{<a>},{<a>}}, 5 groups.
    {{<!>},{<!>},{<!>},{<a>}}, 2 groups (since all but the last > are canceled).

Your goal is to find the total score for all groups in your input. Each group is assigned a score which is one more than the score of the group that immediately contains it. (The outermost group gets a score of 1.)

    {}, score of 1.
    {{{}}}, score of 1 + 2 + 3 = 6.
    {{},{}}, score of 1 + 2 + 2 = 5.
    {{{},{},{{}}}}, score of 1 + 2 + 3 + 3 + 3 + 4 = 16.
    {<a>,<a>,<a>,<a>}, score of 1.
    {{<ab>},{<ab>},{<ab>},{<ab>}}, score of 1 + 2 + 2 + 2 + 2 = 9.
    {{<!!>},{<!!>},{<!!>},{<!!>}}, score of 1 + 2 + 2 + 2 + 2 = 9.
    {{<a!>},{<a!>},{<a!>},{<ab>}}, score of 1 + 2 = 3.

What is the total score for all groups in your input?

--- Part Two ---

Now, you're ready to remove the garbage.

To prove you've removed it, you need to count all of the characters within the garbage. The leading and trailing < and > don't count, nor do any canceled characters or the ! doing the canceling.

    <>, 0 characters.
    <random characters>, 17 characters.
    <<<<>, 3 characters.
    <{!>}>, 2 characters.
    <!!>, 0 characters.
    <!!!>>, 0 characters.
    <{o"i!a,<{i<a>, 10 characters.

How many non-canceled characters are within the garbage in your puzzle input? */

fn parse_with_wrapped_range<'i, O, F: Parser<&'i str, O, NomError<&'i str>>>(
    mut f: F,
) -> impl Parser<&'i str, (Range<u16>, O), NomError<&'i str>> {
    move |input: &'i str| {
        let start_input_len: usize = input.len();
        let (input, output): (&str, O) = f.parse(input)?;
        let end_input_len: usize = input.len();
        let range_len: usize = start_input_len - end_input_len;
        let input: &str = verify(success(()), |_| range_len <= u16::MAX as usize)(input)?.0;

        Ok((input, (0_u16..range_len as u16, output)))
    }
}

struct GroupNodeData<'g> {
    group: Option<&'g Group>,
    garbage: Option<&'g Garbage>,
    depth: usize,

    #[allow(dead_code)]
    stream: &'g str,
}

trait GroupNodeTrait {
    fn fix_ranges(&mut self, offset: u16) -> u16;

    fn aggregate_nodes<'g>(
        &'g self,
        depth: usize,
        stream: &'g str,
        nodes: &mut Vec<GroupNodeData<'g>>,
    );
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Group {
    range: Range<u16>,
    children: Vec<GroupNode>,
}

impl GroupNodeTrait for Group {
    fn fix_ranges(&mut self, offset: u16) -> u16 {
        self.range.start += offset;
        self.range.end += offset;

        // Add one for the '{'.
        let mut child_offset: u16 = offset + 1_u16;

        for child in &mut self.children {
            // Add one for the ','.
            child_offset = child.fix_ranges(child_offset) + 1_u16;
        }

        self.range.end
    }

    fn aggregate_nodes<'g>(
        &'g self,
        depth: usize,
        stream: &'g str,
        nodes: &mut Vec<GroupNodeData<'g>>,
    ) {
        nodes.push(GroupNodeData {
            group: Some(self),
            garbage: None,
            depth,
            stream: &stream[self.range.as_range_usize()],
        });

        for child in &self.children {
            child.aggregate_nodes(depth + 1_usize, stream, nodes);
        }
    }
}

impl Parse for Group {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut first_child: bool = true;

        map(
            parse_with_wrapped_range(delimited(
                tag("{"),
                many0(preceded(
                    move |input: &'i str| -> IResult<&'i str, ()> {
                        let prev_first_child: bool = first_child;

                        first_child = false;

                        map(cond(!prev_first_child, tag(",")), |_| ())(input)
                    },
                    GroupNode::parse,
                )),
                tag("}"),
            )),
            |(range, children)| Self { range, children },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Garbage {
    range: Range<u16>,
    canceled_pairs: usize,
}

impl Garbage {
    fn parse_canceled_pair<'i>(input: &'i str) -> IResult<&'i str, ()> {
        map(tuple((tag("!"), anychar)), |_| ())(input)
    }

    fn parse_non_canceled_chars<'i>(input: &'i str) -> IResult<&'i str, ()> {
        map(many0_count(none_of("!>")), |_| ())(input)
    }

    fn non_canceled_chars_count(&self) -> usize {
        self.range.len() - 2_usize * (self.canceled_pairs + 1_usize)
    }
}

impl GroupNodeTrait for Garbage {
    fn fix_ranges(&mut self, offset: u16) -> u16 {
        self.range.start += offset;
        self.range.end += offset;

        self.range.end
    }

    fn aggregate_nodes<'g>(
        &'g self,
        depth: usize,
        stream: &'g str,
        nodes: &mut Vec<GroupNodeData<'g>>,
    ) {
        nodes.push(GroupNodeData {
            group: None,
            garbage: Some(self),
            depth,
            stream: &stream[self.range.as_range_usize()],
        });
    }
}

impl Parse for Garbage {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            parse_with_wrapped_range(delimited(
                tag("<"),
                preceded(
                    Self::parse_non_canceled_chars,
                    many0_count(tuple((
                        Self::parse_canceled_pair,
                        opt(Self::parse_non_canceled_chars),
                    ))),
                ),
                tag(">"),
            )),
            |(range, canceled_pairs)| Self {
                range,
                canceled_pairs,
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum GroupNode {
    Group(Group),
    Garbage(Garbage),
}

impl From<Garbage> for GroupNode {
    fn from(value: Garbage) -> Self {
        Self::Garbage(value)
    }
}

impl From<Group> for GroupNode {
    fn from(value: Group) -> Self {
        Self::Group(value)
    }
}

impl GroupNodeTrait for GroupNode {
    fn fix_ranges(&mut self, offset: u16) -> u16 {
        match self {
            Self::Group(group) => group.fix_ranges(offset),
            Self::Garbage(garbage) => garbage.fix_ranges(offset),
        }
    }

    fn aggregate_nodes<'g>(
        &'g self,
        depth: usize,
        stream: &'g str,
        nodes: &mut Vec<GroupNodeData<'g>>,
    ) {
        match self {
            Self::Group(group) => group.aggregate_nodes(depth, stream, nodes),
            Self::Garbage(garbage) => garbage.aggregate_nodes(depth, stream, nodes),
        }
    }
}

impl Parse for GroupNode {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Group::parse, Self::from),
            map(Garbage::parse, Self::from),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    stream: String,
    group: Group,
}

impl Solution {
    fn get_nodes(&self) -> Vec<GroupNodeData> {
        let mut nodes: Vec<GroupNodeData> = Vec::new();

        self.group
            .aggregate_nodes(0_usize, &self.stream, &mut nodes);

        nodes
    }

    fn iter_groups(&self) -> impl Iterator<Item = GroupNodeData> {
        self.get_nodes()
            .into_iter()
            .filter(|node| node.group.is_some())
    }

    #[cfg(test)]
    fn count_groups(&self) -> usize {
        self.iter_groups().count()
    }

    fn total_score(&self) -> usize {
        self.iter_groups().map(|node| node.depth + 1_usize).sum()
    }

    fn iter_garbage(&self) -> impl Iterator<Item = GroupNodeData> {
        self.get_nodes()
            .into_iter()
            .filter(|node| node.garbage.is_some())
    }

    fn non_canceled_garbage_chars_count(&self) -> usize {
        self.iter_garbage()
            .map(|node| node.garbage.unwrap().non_canceled_chars_count())
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (remaining_input, mut group): (&str, Group) = Group::parse(input)?;

        group.fix_ranges(0_u16);

        let stream: String = input[..input.len() - remaining_input.len()].into();

        Ok((remaining_input, Self { stream, group }))
    }
}

impl RunQuestions for Solution {
    /// I could put together a custom type to iterate over the `GroupNodeData` without having to
    /// allocate a vector for them, but I'd still need to allocate a DFS stack, so it wasn't worth
    /// it.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.total_score());
    }

    /// I'm glad this didn't require refactoring `Garbage` to have a `Vec` field...
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.non_canceled_garbage_chars_count());
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

    const GARBAGE_STRS: &'static [&'static str] = &[
        "<>",
        "<random characters>",
        "<<<<>",
        "<{!>}>",
        "<!!>",
        "<!!!>>",
        "<{o\"i!a,<{i<a>",
    ];
    const SOLUTION_STRS: &'static [&'static str] = &[
        "{}",
        "{{{}}}",
        "{{},{}}",
        "{{{},{},{{}}}}",
        "{<{},{},{{}}>}",
        "{<a>,<a>,<a>,<a>}",
        "{{<a>},{<a>},{<a>},{<a>}}",
        "{{<!>},{<!>},{<!>},{<a>}}",
        "{{<ab>},{<ab>},{<ab>},{<ab>}}",
        "{{<!!>},{<!!>},{<!!>},{<!!>}}",
        "{{<a!>},{<a!>},{<a!>},{<ab>}}",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    stream: "{}".into(),
                    group: Group {
                        range: 0_u16..2_u16,
                        children: vec![],
                    },
                },
                Solution {
                    stream: "{{{}}}".into(),
                    group: Group {
                        range: 0_u16..6_u16,
                        children: vec![Group {
                            range: 1_u16..5_u16,
                            children: vec![Group {
                                range: 2_u16..4_u16,
                                children: vec![],
                            }
                            .into()],
                        }
                        .into()],
                    },
                },
                Solution {
                    stream: "{{},{}}".into(),
                    group: Group {
                        range: 0_u16..7_u16,
                        children: vec![
                            Group {
                                range: 1_u16..3_u16,
                                children: vec![],
                            }
                            .into(),
                            Group {
                                range: 4_u16..6_u16,
                                children: vec![],
                            }
                            .into(),
                        ],
                    },
                },
                Solution {
                    stream: "{{{},{},{{}}}}".into(),
                    group: Group {
                        range: 0_u16..14_u16,
                        children: vec![Group {
                            range: 1_u16..13_u16,
                            children: vec![
                                Group {
                                    range: 2_u16..4_u16,
                                    children: vec![],
                                }
                                .into(),
                                Group {
                                    range: 5_u16..7_u16,
                                    children: vec![],
                                }
                                .into(),
                                Group {
                                    range: 8_u16..12_u16,
                                    children: vec![Group {
                                        range: 9_u16..11_u16,
                                        children: vec![],
                                    }
                                    .into()],
                                }
                                .into(),
                            ],
                        }
                        .into()],
                    },
                },
                Solution {
                    stream: "{<{},{},{{}}>}".into(),
                    group: Group {
                        range: 0_u16..14_u16,
                        children: vec![Garbage {
                            range: 1_u16..13_u16,
                            canceled_pairs: 0_usize,
                        }
                        .into()],
                    },
                },
                Solution {
                    stream: "{<a>,<a>,<a>,<a>}".into(),
                    group: Group {
                        range: 0_u16..17_u16,
                        children: vec![
                            Garbage {
                                range: 1_u16..4_u16,
                                canceled_pairs: 0_usize,
                            }
                            .into(),
                            Garbage {
                                range: 5_u16..8_u16,
                                canceled_pairs: 0_usize,
                            }
                            .into(),
                            Garbage {
                                range: 9_u16..12_u16,
                                canceled_pairs: 0_usize,
                            }
                            .into(),
                            Garbage {
                                range: 13_u16..16_u16,
                                canceled_pairs: 0_usize,
                            }
                            .into(),
                        ],
                    },
                },
                Solution {
                    stream: "{{<a>},{<a>},{<a>},{<a>}}".into(),
                    group: Group {
                        range: 0_u16..25_u16,
                        children: vec![
                            Group {
                                range: 1_u16..6_u16,
                                children: vec![Garbage {
                                    range: 2_u16..5_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 7_u16..12_u16,
                                children: vec![Garbage {
                                    range: 8_u16..11_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 13_u16..18_u16,
                                children: vec![Garbage {
                                    range: 14_u16..17_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 19_u16..24_u16,
                                children: vec![Garbage {
                                    range: 20_u16..23_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                        ],
                    },
                },
                Solution {
                    stream: "{{<!>},{<!>},{<!>},{<a>}}".into(),
                    group: Group {
                        range: 0_u16..25_u16,
                        children: vec![Group {
                            range: 1_u16..24_u16,
                            children: vec![Garbage {
                                range: 2_u16..23_u16,
                                canceled_pairs: 3_usize,
                            }
                            .into()],
                        }
                        .into()],
                    },
                },
                Solution {
                    stream: "{{<ab>},{<ab>},{<ab>},{<ab>}}".into(),
                    group: Group {
                        range: 0_u16..29_u16,
                        children: vec![
                            Group {
                                range: 1_u16..7_u16,
                                children: vec![Garbage {
                                    range: 2_u16..6_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 8_u16..14_u16,
                                children: vec![Garbage {
                                    range: 9_u16..13_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 15_u16..21_u16,
                                children: vec![Garbage {
                                    range: 16_u16..20_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 22_u16..28_u16,
                                children: vec![Garbage {
                                    range: 23_u16..27_u16,
                                    canceled_pairs: 0_usize,
                                }
                                .into()],
                            }
                            .into(),
                        ],
                    },
                },
                Solution {
                    stream: "{{<!!>},{<!!>},{<!!>},{<!!>}}".into(),
                    group: Group {
                        range: 0_u16..29_u16,
                        children: vec![
                            Group {
                                range: 1_u16..7_u16,
                                children: vec![Garbage {
                                    range: 2_u16..6_u16,
                                    canceled_pairs: 1_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 8_u16..14_u16,
                                children: vec![Garbage {
                                    range: 9_u16..13_u16,
                                    canceled_pairs: 1_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 15_u16..21_u16,
                                children: vec![Garbage {
                                    range: 16_u16..20_u16,
                                    canceled_pairs: 1_usize,
                                }
                                .into()],
                            }
                            .into(),
                            Group {
                                range: 22_u16..28_u16,
                                children: vec![Garbage {
                                    range: 23_u16..27_u16,
                                    canceled_pairs: 1_usize,
                                }
                                .into()],
                            }
                            .into(),
                        ],
                    },
                },
                Solution {
                    stream: "{{<a!>},{<a!>},{<a!>},{<ab>}}".into(),
                    group: Group {
                        range: 0_u16..29_u16,
                        children: vec![Group {
                            range: 1_u16..28_u16,
                            children: vec![Garbage {
                                range: 2_u16..27_u16,
                                canceled_pairs: 3_usize,
                            }
                            .into()],
                        }
                        .into()],
                    },
                },
            ]
        })[index]
    }

    #[test]
    fn test_garbage_parse() {
        for garbage_str in GARBAGE_STRS.into_iter().copied() {
            let (output, garbage): (&str, Garbage) = Garbage::parse(&garbage_str).unwrap();

            assert_eq!(output, "");
            assert_eq!(garbage.range, 0_u16..garbage_str.len() as u16);
        }
    }

    #[test]
    fn test_garbage_non_canceled_chars_count() {
        for (index, non_canceled_chars_count) in [
            0_usize, 17_usize, 3_usize, 2_usize, 0_usize, 0_usize, 10_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                Garbage::parse(GARBAGE_STRS[index])
                    .unwrap()
                    .1
                    .non_canceled_chars_count(),
                non_canceled_chars_count
            );
        }
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
    fn test_count_groups() {
        for (index, count_groups) in [
            1_usize, 3_usize, 3_usize, 6_usize, 1_usize, 1_usize, 5_usize, 2_usize, 5_usize,
            5_usize, 2_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).count_groups(), count_groups);
        }
    }

    #[test]
    fn test_total_score() {
        for (index, total_score) in [
            1_usize, 6_usize, 5_usize, 16_usize, 1_usize, 1_usize, 9_usize, 3_usize, 9_usize,
            9_usize, 3_usize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).total_score(), total_score);
        }
    }
}
