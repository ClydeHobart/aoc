use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{cond, map, map_opt, map_res, opt, success, verify},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 12: Digital Plumber ---

Walking along the memory banks of the stream, you find a small village that is experiencing a little confusion: some programs can't communicate with each other.

Programs in this village communicate using a fixed system of pipes. Messages are passed between programs using these pipes, but most programs aren't connected to each other directly. Instead, programs pass messages between each other until the message reaches the intended recipient.

For some reason, though, some of these messages aren't ever reaching their intended recipient, and the programs suspect that some pipes are missing. They would like you to investigate.

You walk through the village and record the ID of each program and the IDs with which it can communicate directly (your puzzle input). Each program has one or more programs with which it can communicate, and these pipes are bidirectional; if 8 says it can communicate with 11, then 11 will say it can communicate with 8.

You need to figure out how many programs are in the group that contains program ID 0.

For example, suppose you go door-to-door like a travelling salesman and record the following list:

0 <-> 2
1 <-> 1
2 <-> 0, 3, 4
3 <-> 2, 4
4 <-> 2, 3, 6
5 <-> 6
6 <-> 4, 5

In this example, the following programs are in the group that contains program ID 0:

    Program 0 by definition.
    Program 2, directly connected to program 0.
    Program 3 via program 2.
    Program 4 via program 2.
    Program 5 via programs 6, then 4, then 2.
    Program 6 via programs 4, then 2.

Therefore, a total of 6 programs are in this group; all but program 1, which has a pipe that connects it to itself.

How many programs are in the group that contains program ID 0?

--- Part Two ---

There are more programs than just the ones in the group containing program ID 0. The rest of them have no way of reaching that group, and still might have no way of reaching each other.

A group is a collection of programs that can all communicate via pipes either directly or indirectly. The programs you identified just a moment ago are all part of the same group. Now, they would like you to determine the total number of groups.

In the example above, there were 2 groups: one consisting of programs 0,2,3,4,5,6, and the other consisting solely of program 1.

How many groups are there in total? */

type ProgramIndex = TableIndex<u16>;

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Program {
    neighbors_range: Range<u16>,

    #[allow(dead_code)]
    has_pipe_to_self: bool,
}

impl Program {
    fn parse<'i>(
        neighbors: &mut Vec<ProgramIndex>,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, (ProgramIndex, Self)> + '_ {
        move |input| {
            let (input, (neighbors_start, program_index, _)): (&str, (u16, ProgramIndex, _)) =
                tuple((
                    map_res(success(()), |_| neighbors.len().try_into()),
                    map(parse_integer::<usize>, ProgramIndex::from),
                    tag(" <-> "),
                ))(input)?;
            let mut has_neighbors: bool = false;
            let mut has_pipe_to_self: bool = false;
            let input: &str = many0(map(
                preceded(
                    move |input| {
                        let has_neighbors_value: bool = has_neighbors;

                        has_neighbors = true;

                        cond(has_neighbors_value, tag(", "))(input)
                    },
                    parse_integer::<usize>,
                ),
                |neighbor_program_index_raw| {
                    let neighbor_program_index: ProgramIndex = neighbor_program_index_raw.into();

                    if neighbor_program_index == program_index {
                        has_pipe_to_self = true;
                    } else {
                        neighbors.push(neighbor_program_index);
                    }
                },
            ))(input)?
            .0;
            let (input, neighbors_end): (&str, u16) =
                map_res(success(()), |_| neighbors.len().try_into())(input)?;

            let neighbors_range: Range<u16> = neighbors_start..neighbors_end;

            neighbors[neighbors_range.clone().as_range_usize()].sort();

            Ok((
                input,
                (
                    program_index,
                    Self {
                        neighbors_range,
                        has_pipe_to_self,
                    },
                ),
            ))
        }
    }

    fn neighbors<'n>(&self, full_neighbors: &'n [ProgramIndex]) -> &'n [ProgramIndex] {
        &full_neighbors[self.neighbors_range.as_range_usize()]
    }
}

struct ProgramSubGraphFinder<'s> {
    solution: &'s Solution,
    start: ProgramIndex,
    parents: Vec<ProgramIndex>,
}

impl<'s> BreadthFirstSearch for ProgramSubGraphFinder<'s> {
    type Vertex = ProgramIndex;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend_from_slice(
            &self.solution.programs[vertex.get()].neighbors(&self.solution.neighbors),
        );
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.parents[to.get()] = *from;
    }

    fn reset(&mut self) {
        self.parents.clear();
        self.parents
            .resize(self.solution.programs.len(), ProgramIndex::invalid());
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    programs: Vec<Program>,
    neighbors: Vec<ProgramIndex>,
}

impl Solution {
    fn validate_neighbors(&self) -> bool {
        self.programs
            .iter()
            .enumerate()
            .all(|(program_index_a, program_a)| {
                let program_index_a: ProgramIndex = program_index_a.into();

                program_a
                    .neighbors(&self.neighbors)
                    .iter()
                    .copied()
                    .all(|program_index_b| {
                        self.programs[program_index_b.get()]
                            .neighbors(&self.neighbors)
                            .binary_search(&program_index_a)
                            .is_ok()
                    })
            })
    }

    fn sub_graph_incl_0_programs_count(&self) -> usize {
        let mut program_sub_graph_finder: ProgramSubGraphFinder = ProgramSubGraphFinder {
            solution: self,
            start: 0_usize.into(),
            parents: Vec::new(),
        };

        program_sub_graph_finder.run();

        program_sub_graph_finder
            .parents
            .iter()
            .filter(|parent| parent.is_valid())
            .count()
            + 1_usize
    }

    fn disjoint_sub_graph_count(&self) -> usize {
        let mut program_sub_graph_finder: ProgramSubGraphFinder = ProgramSubGraphFinder {
            solution: self,
            start: ProgramIndex::invalid(),
            parents: Vec::new(),
        };
        let mut discovered: BitVec = bitvec![0; self.programs.len()];
        let mut disjoint_sub_graph_count: usize = 0_usize;

        while let Some(start) = discovered.iter_zeros().next().map(ProgramIndex::from) {
            disjoint_sub_graph_count += 1_usize;
            discovered.set(start.get(), true);
            program_sub_graph_finder.start = start;
            program_sub_graph_finder.run();

            for descendant in program_sub_graph_finder
                .parents
                .iter()
                .enumerate()
                .filter_map(|(descendant, descendent_parent)| {
                    descendent_parent.is_valid().then_some(descendant)
                })
            {
                discovered.set(descendant, true);
            }
        }

        disjoint_sub_graph_count
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        verify(
            |input| {
                let mut neighbors: Vec<ProgramIndex> = Vec::new();
                let mut program_index: ProgramIndex = 0_usize.into();

                let (input, programs): (&str, Vec<Program>) = many0(map_opt(
                    terminated(Program::parse(&mut neighbors), opt(line_ending)),
                    |(parsed_program_index, program)| {
                        (parsed_program_index == program_index).then(|| {
                            program_index = (program_index.get() + 1_usize).into();

                            program
                        })
                    },
                ))(input)?;

                Ok((
                    input,
                    Self {
                        programs,
                        neighbors,
                    },
                ))
            },
            Solution::validate_neighbors,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Hardest part was parsing. Parser looks clean, and the BFS implementation was easy (with the
    /// trait already established).
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sub_graph_incl_0_programs_count());
    }

    /// After some googling, I might not be using the correct term here by "subgraph" or "disjoint
    /// subgraph", but the context of the problem should make it clear what I mean by these terms.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.disjoint_sub_graph_count());
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
        0 <-> 2\n\
        1 <-> 1\n\
        2 <-> 0, 3, 4\n\
        3 <-> 2, 4\n\
        4 <-> 2, 3, 6\n\
        5 <-> 6\n\
        6 <-> 4, 5\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                programs: vec![
                    Program {
                        neighbors_range: 0_u16..1_u16,
                        has_pipe_to_self: false,
                    },
                    Program {
                        neighbors_range: 1_u16..1_u16,
                        has_pipe_to_self: true,
                    },
                    Program {
                        neighbors_range: 1_u16..4_u16,
                        has_pipe_to_self: false,
                    },
                    Program {
                        neighbors_range: 4_u16..6_u16,
                        has_pipe_to_self: false,
                    },
                    Program {
                        neighbors_range: 6_u16..9_u16,
                        has_pipe_to_self: false,
                    },
                    Program {
                        neighbors_range: 9_u16..10_u16,
                        has_pipe_to_self: false,
                    },
                    Program {
                        neighbors_range: 10_u16..12_u16,
                        has_pipe_to_self: false,
                    },
                ],
                neighbors: vec![
                    2_usize.into(),
                    0_usize.into(),
                    3_usize.into(),
                    4_usize.into(),
                    2_usize.into(),
                    4_usize.into(),
                    2_usize.into(),
                    3_usize.into(),
                    6_usize.into(),
                    6_usize.into(),
                    4_usize.into(),
                    5_usize.into(),
                ],
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
    fn test_sub_graph_incl_0_programs_count() {
        for (index, sub_graph_incl_0_programs_count) in [6_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).sub_graph_incl_0_programs_count(),
                sub_graph_incl_0_programs_count
            );
        }
    }

    #[test]
    fn test_disjoint_sub_graph_count() {
        for (index, disjoint_sub_graph_count) in [2_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).disjoint_sub_graph_count(),
                disjoint_sub_graph_count
            );
        }
    }
}
