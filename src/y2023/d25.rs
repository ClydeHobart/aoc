use {
    crate::*,
    nom::{
        bytes::complete::{tag, take_while_m_n},
        character::complete::{line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    rand::{prelude::*, rngs::ThreadRng, thread_rng},
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
    },
};

#[derive(Clone, Copy, Eq, Hash, Ord, PartialOrd, PartialEq)]
struct Tag([u8; Self::LEN]);

impl Tag {
    const LEN: usize = 3_usize;
}

impl Debug for Tag {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // This isn't safe, but I'm too tired to care
        f.write_str(unsafe { from_utf8_unchecked(&self.0) })
    }
}

impl Parse for Tag {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            take_while_m_n(Self::LEN, Self::LEN, |c: char| c.is_ascii_lowercase()),
            Tag::from,
        )(input)
    }
}

impl From<&str> for Tag {
    fn from(value: &str) -> Self {
        let mut tag: Tag = Tag(Default::default());

        tag.0.copy_from_slice(value.as_bytes());

        tag
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Graph(HashMap<Tag, Vec<Tag>>);

impl Graph {
    fn iter_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.0
            .iter()
            .flat_map(|(key, value)| value.iter().map(|value_tag| Edge::new(*key, *value_tag)))
    }
}

impl Parse for Graph {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(separated_pair(
                Tag::parse,
                tag(": "),
                terminated(many0(terminated(Tag::parse, space0)), opt(line_ending)),
            )),
            |key_value_pairs| {
                let half_graph: Self = Self(key_value_pairs.into_iter().collect());
                let mut graph: Self = Self(HashMap::new());
                let mut push_mapping = |from: Tag, to: Tag| {
                    if !graph.0.contains_key(&from) {
                        graph.0.insert(from, Vec::new());
                    }

                    graph.0.get_mut(&from).unwrap().push(to);
                };

                for edge in half_graph.iter_edges() {
                    let tag_a: Tag = edge.0[0_usize];
                    let tag_b: Tag = edge.0[1_usize];

                    push_mapping(tag_a, tag_b);
                    push_mapping(tag_b, tag_a);
                }

                for (_, values) in graph.0.iter_mut() {
                    values.sort();
                }

                graph
            },
        )(input)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialOrd, PartialEq)]
struct Edge([Tag; 2_usize]);

impl Edge {
    fn new(tag_a: Tag, tag_b: Tag) -> Self {
        let mut edge: Self = Self([tag_a, tag_b]);

        edge.0.sort();

        edge
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Partition {
    edges: [Edge; 3_usize],
    sub_graph_a: usize,
    sub_graph_b: usize,
}

struct NewPartitionFinder<'g> {
    graph: &'g Graph,
    start: Option<Tag>,
    end: Option<Tag>,
    invalid_edges: Vec<Edge>,
    parent_map: HashMap<Tag, Tag>,
}

impl<'g> NewPartitionFinder<'g> {
    const VERTEX_PAIR_SAMPLE_SIZE: usize = 300_usize;

    fn shuffle_vertex_pair_indices(vertex_pair_indices: &mut Vec<usize>, vertex_pairs_len: usize) {
        vertex_pair_indices.clear();
        vertex_pair_indices.extend(0_usize..vertex_pairs_len);

        let end: usize = vertex_pairs_len.min(Self::VERTEX_PAIR_SAMPLE_SIZE);

        let mut rng: ThreadRng = thread_rng();

        for i in 0_usize..end - 1_usize {
            let j: usize = i + rng.gen_range(0_usize..vertex_pairs_len - i);
            let temp: usize = vertex_pair_indices[i];

            vertex_pair_indices[i] = vertex_pair_indices[j];
            vertex_pair_indices[j] = temp;
        }
    }

    fn find_partition(&mut self, verbose: bool) -> Partition {
        let vertices: Vec<Tag> = self.graph.0.iter().map(|(key, _)| *key).collect();
        let mut vertex_pair_indices: Vec<usize> = Vec::new();
        let mut vertex_frequencies: HashMap<Tag, usize> = HashMap::new();
        let mut most_popular_nodes: Vec<(Tag, usize)> = Vec::new();
        let mut pass_count: usize = 0_usize;

        loop {
            if verbose {
                pass_count = dbg!(pass_count) + 1_usize;
            }

            let vertex_pairs_len: usize = vertices.len() * vertices.len();

            vertex_frequencies.clear();
            Self::shuffle_vertex_pair_indices(&mut vertex_pair_indices, vertex_pairs_len);

            let mut increment_frequency = |tag: Tag| {
                if !vertex_frequencies.contains_key(&tag) {
                    vertex_frequencies.insert(tag, 0_usize);
                }

                *vertex_frequencies.get_mut(&tag).unwrap() += 1_usize;
            };

            self.invalid_edges.clear();

            for (index, vertex_pair_index) in vertex_pair_indices
                .drain(..vertex_pairs_len.min(Self::VERTEX_PAIR_SAMPLE_SIZE))
                .enumerate()
            {
                if verbose {
                    dbg!(index);
                }

                let start_index: usize = vertex_pair_index / vertices.len();
                let mut end_index: usize = vertex_pair_index % vertices.len();

                if end_index == start_index {
                    end_index = (end_index + 1_usize) % vertices.len();
                }

                let start: Tag = vertices[start_index];
                let end: Tag = vertices[end_index];

                self.start = Some(start);
                self.end = Some(end);
                self.run();

                let mut vertex: Tag = end;

                while vertex != start {
                    increment_frequency(vertex);
                    vertex = *self.parent_map.get(&vertex).unwrap();
                }

                increment_frequency(start);
            }

            most_popular_nodes.clear();

            for _ in 0_usize..6_usize {
                let next_most_popular_node: (Tag, usize) = vertex_frequencies
                    .iter()
                    .filter_map(|(tag, count)| {
                        (!most_popular_nodes
                            .iter()
                            .any(|(popular_tag, _)| *popular_tag == *tag))
                        .then_some((*tag, *count))
                    })
                    .max_by_key(|(_, count)| *count)
                    .unwrap();

                most_popular_nodes.push(next_most_popular_node);
            }

            most_popular_nodes.sort_by_key(|(_, count)| *count);

            if most_popular_nodes
                .chunks_exact(2_usize)
                .all(|most_popular_nodes_pair| {
                    let popular_node_a: (Tag, usize) = most_popular_nodes_pair[0_usize];
                    let popular_node_b: (Tag, usize) = most_popular_nodes_pair[1_usize];

                    (popular_node_a.1 == popular_node_b.1)
                        && self
                            .graph
                            .0
                            .get(&popular_node_a.0)
                            .unwrap()
                            .iter()
                            .any(|a_neighbor| *a_neighbor == popular_node_b.0)
                })
            {
                self.invalid_edges.clear();
                self.invalid_edges
                    .extend(most_popular_nodes.chunks_exact(2_usize).map(
                        |most_popular_node_pair| {
                            Edge::new(
                                most_popular_node_pair[0_usize].0,
                                most_popular_node_pair[1_usize].0,
                            )
                        },
                    ));

                self.end = None;

                self.run();

                let mut edges: [Edge; 3_usize] = [
                    self.invalid_edges[0_usize],
                    self.invalid_edges[1_usize],
                    self.invalid_edges[2_usize],
                ];

                edges.sort();

                // Add 1, since there's no entry for the start
                let sub_graph_a: usize = self.parent_map.len() + 1_usize;
                let sub_graph_b: usize = self.graph.0.len() - sub_graph_a;

                return Partition {
                    edges,
                    sub_graph_a: sub_graph_a.min(sub_graph_b),
                    sub_graph_b: sub_graph_a.max(sub_graph_b),
                };
            }
        }
    }
}

impl<'g> BreadthFirstSearch for NewPartitionFinder<'g> {
    type Vertex = Tag;

    fn start(&self) -> &Self::Vertex {
        self.start.as_ref().unwrap()
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        Some(*vertex) == self.end
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        // We don't need to be allocating this each time, we can use iters
        Vec::new()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(
            self.graph
                .0
                .get(vertex)
                .unwrap()
                .iter()
                .copied()
                .filter(|value_tag| {
                    let edge: Edge = Edge::new(*vertex, *value_tag);

                    !self
                        .invalid_edges
                        .iter()
                        .any(|invalid_edge| *invalid_edge == edge)
                }),
        );
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.parent_map.insert(*to, *from);
    }

    fn reset(&mut self) {
        self.parent_map.clear();
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Graph);

impl Solution {
    fn new_find_partition(&self, verbose: bool) -> Partition {
        let mut new_partition_finder: NewPartitionFinder = NewPartitionFinder {
            graph: &self.0,
            start: None,
            end: None,
            invalid_edges: Vec::new(),
            parent_map: HashMap::new(),
        };

        new_partition_finder.find_partition(verbose)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Graph::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let partition: Partition = self.new_find_partition(true);

            dbg!(partition.sub_graph_a * partition.sub_graph_b, partition);
        } else {
            let partition: Partition = self.new_find_partition(false);

            dbg!(partition.sub_graph_a * partition.sub_graph_b);
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
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

    const SOLUTION_STR: &'static str = "\
        jqt: rhn xhk nvd\n\
        rsh: frs pzl lsr\n\
        xhk: hfx\n\
        cmg: qnr nvd lhk bvb\n\
        rhn: xhk bvb hfx\n\
        bvb: xhk hfx\n\
        pzl: lsr hfx nvd\n\
        qnr: nvd\n\
        ntq: jqt hfx bvb xhk\n\
        nvd: lhk\n\
        lsr: lhk\n\
        rzs: qnr cmg lsr rsh\n\
        frs: qnr lhk lsr\n";

    fn solution() -> &'static Solution {
        macro_rules! tags {
            { $( ( $key:expr, [ $( $value_tag:expr, )* ], ), )* } => { [ $(
                ( Tag::from($key), vec![ $( Tag::from($value_tag), )* ] ),
            )* ].into_iter().collect() }
        }

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(Graph(tags! {
                (
                    "bvb",
                    [
                        "cmg",
                        "hfx",
                        "ntq",
                        "rhn",
                        "xhk",
                    ],
                ),
                (
                    "cmg",
                    [
                        "bvb",
                        "lhk",
                        "nvd",
                        "qnr",
                        "rzs",
                    ],
                ),
                (
                    "frs",
                    [
                        "lhk",
                        "lsr",
                        "qnr",
                        "rsh",
                    ],
                ),
                (
                    "hfx",
                    [
                        "bvb",
                        "ntq",
                        "pzl",
                        "rhn",
                        "xhk",
                    ],
                ),
                (
                    "jqt",
                    [
                        "ntq",
                        "nvd",
                        "rhn",
                        "xhk",
                    ],
                ),
                (
                    "lhk",
                    [
                        "cmg",
                        "frs",
                        "lsr",
                        "nvd",
                    ],
                ),
                (
                    "lsr",
                    [
                        "frs",
                        "lhk",
                        "pzl",
                        "rsh",
                        "rzs",
                    ],
                ),
                (
                    "ntq",
                    [
                        "bvb",
                        "hfx",
                        "jqt",
                        "xhk",
                    ],
                ),
                (
                    "nvd",
                    [
                        "cmg",
                        "jqt",
                        "lhk",
                        "pzl",
                        "qnr",
                    ],
                ),
                (
                    "pzl",
                    [
                        "hfx",
                        "lsr",
                        "nvd",
                        "rsh",
                    ],
                ),
                (
                    "qnr",
                    [
                        "cmg",
                        "frs",
                        "nvd",
                        "rzs",
                    ],
                ),
                (
                    "rhn",
                    [
                        "bvb",
                        "hfx",
                        "jqt",
                        "xhk",
                    ],
                ),
                (
                    "rsh",
                    [
                        "frs",
                        "lsr",
                        "pzl",
                        "rzs",
                    ],
                ),
                (
                    "rzs",
                    [
                        "cmg",
                        "lsr",
                        "qnr",
                        "rsh",
                    ],
                ),
                (
                    "xhk",
                    [
                        "bvb",
                        "hfx",
                        "jqt",
                        "ntq",
                        "rhn",
                    ],
                ),
            }))
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }
}
