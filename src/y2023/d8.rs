use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, satisfy},
        combinator::{map, map_opt, opt},
        error::Error,
        multi::{fold_many_m_n, many0},
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::{
        collections::HashMap,
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8,
    },
};

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug, PartialEq))]
    #[derive(Copy, Clone)]
    enum Instruction {
        Left = LEFT = b'L',
        Right = RIGHT = b'R',
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Name([u8; Name::LEN]);

impl Name {
    const LEN: usize = 3_usize;
    const LAST_INDEX: usize = Name::LEN - 1_usize;

    const fn new(byte: u8) -> Self {
        Self([byte; Name::LEN])
    }
}

impl Debug for Name {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(from_utf8(&self.0).unwrap())
    }
}

impl Default for Name {
    fn default() -> Self {
        Self::new(b' ')
    }
}

impl Parse for Name {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            fold_many_m_n(
                Self::LEN,
                Self::LEN,
                satisfy(|c| c.is_ascii_alphanumeric()),
                || (Self::default(), 0_u8),
                |(mut name, index), c| {
                    name.0[index as usize] = c as u8;

                    (name, index + 1_u8)
                },
            ),
            |(name, _)| name,
        )(input)
    }
}

#[cfg(test)]
impl TryFrom<&str> for Name {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.is_ascii() && value.len() == Self::LEN {
            let mut name: Self = Self::default();

            name.0.copy_from_slice(value.as_bytes());

            Ok(name)
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct Node<I = Name, N = u16> {
    id: I,
    left: N,
    right: N,
}

impl<I, N: Copy> Node<I, N> {
    fn neighbor(&self, instruction: Instruction) -> N {
        match instruction {
            Instruction::Left => self.left,
            Instruction::Right => self.right,
        }
    }
}

impl<N> Node<Name, N> {
    const START_NAME: Name = Name::new(b'A');
    const END_NAME: Name = Name::new(b'Z');

    fn is_end(&self) -> bool {
        self.id == Self::END_NAME
    }

    fn last_byte(&self) -> u8 {
        self.id.0[Name::LAST_INDEX]
    }

    fn is_ghost_start(&self) -> bool {
        self.last_byte() == b'A'
    }

    fn is_ghost_end(&self) -> bool {
        self.last_byte() == b'Z'
    }
}

impl<I: Parse, N: Parse> Parse for Node<I, N> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                I::parse,
                tag(" = ("),
                N::parse,
                tag(", "),
                N::parse,
                tag(")"),
            )),
            |(id, _, left, _, right, _)| Self { id, left, right },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    instructions: Vec<Instruction>,
    start: Option<u16>,
    end: Option<u16>,
    nodes: Vec<Node<Name, u16>>,
}

impl Solution {
    fn steps_internal<F: Fn(&Node) -> bool>(&self, start_index: u16, is_end: F) -> usize {
        self.instructions
            .iter()
            .cycle()
            .enumerate()
            .try_fold(start_index, |index, (steps, instruction)| {
                let node: &Node = &self.nodes[index as usize];

                if is_end(node) {
                    Err(steps)
                } else {
                    Ok(node.neighbor(*instruction))
                }
            })
            .unwrap_err()
    }

    fn steps(&self) -> Option<usize> {
        self.end
            .and_then(|_| Some(self.steps_internal(self.start?, Node::is_end)))
    }

    fn iter_ghost_start_indices(&self) -> impl Iterator<Item = u16> + '_ {
        self.nodes.iter().enumerate().filter_map(|(index, node)| {
            if node.is_ghost_start() {
                Some(index as u16)
            } else {
                None
            }
        })
    }

    fn ghost_steps(&self) -> usize {
        // I originally had some code in here that tried to verify this would work ahead of time by
        // checking for a cycle, but it wasn't correct. Luckily, the examples and my input are a
        // cycles of this fashion.
        let mut prime_factorization: HashMap<u32, u32> = HashMap::new();

        for prime_factor in self.iter_ghost_start_indices().flat_map(|start_index| {
            iter_prime_factors(self.steps_internal(start_index, Node::is_ghost_end))
        }) {
            if let Some(exponent) = prime_factorization.get_mut(&(prime_factor.prime as u32)) {
                *exponent = (*exponent).max(prime_factor.exponent as u32);
            } else {
                prime_factorization.insert(prime_factor.prime as u32, prime_factor.exponent as u32);
            }
        }

        prime_factorization
            .into_iter()
            .map(|(prime, exponent)| (prime as usize).pow(exponent))
            .product()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                many0(Instruction::parse),
                line_ending,
                line_ending,
                many0(terminated(Node::<Name, Name>::parse, opt(line_ending))),
            )),
            |(instructions, _, _, name_nodes)| {
                let name_to_index: HashMap<Name, u16> = name_nodes
                    .iter()
                    .enumerate()
                    .map(|(index, node)| (node.id, index as u16))
                    .collect();
                let start: Option<u16> = name_to_index.get(&Node::<Name, u16>::START_NAME).copied();
                let end: Option<u16> = name_to_index.get(&Node::<Name, u16>::END_NAME).copied();

                let mut nodes: Vec<Node<Name, u16>> = Vec::with_capacity(name_nodes.len());

                name_nodes.iter().try_for_each(|name_node| {
                    let left: u16 = *name_to_index.get(&name_node.left)?;
                    let right: u16 = *name_to_index.get(&name_node.right)?;

                    nodes.push(Node::<Name, u16> {
                        id: name_node.id,
                        left,
                        right,
                    });

                    Some(())
                })?;

                Some(Solution {
                    instructions,
                    start,
                    end,
                    nodes,
                })
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.steps());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.ghost_steps());
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        "RL\n\
        \n\
        AAA = (BBB, CCC)\n\
        BBB = (DDD, EEE)\n\
        CCC = (ZZZ, GGG)\n\
        DDD = (DDD, DDD)\n\
        EEE = (EEE, EEE)\n\
        GGG = (GGG, GGG)\n\
        ZZZ = (ZZZ, ZZZ)\n",
        "LLR\n\
        \n\
        AAA = (BBB, BBB)\n\
        BBB = (AAA, ZZZ)\n\
        ZZZ = (ZZZ, ZZZ)\n",
        "LR\n\
        \n\
        11A = (11B, XXX)\n\
        11B = (XXX, 11Z)\n\
        11Z = (11B, XXX)\n\
        22A = (22B, XXX)\n\
        22B = (22C, 22C)\n\
        22C = (22Z, 22Z)\n\
        22Z = (22B, 22B)\n\
        XXX = (XXX, XXX)\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        use Instruction::{Left as L, Right as R};

        let name = |byte: u8| Name::new(byte);
        let name_str = |name: &str| Name::try_from(name).unwrap();

        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    instructions: vec![R, L],
                    start: Some(0_u16),
                    end: Some(6_u16),
                    nodes: vec![
                        Node {
                            id: name(b'A'),
                            left: 1_u16,
                            right: 2_u16,
                        },
                        Node {
                            id: name(b'B'),
                            left: 3_u16,
                            right: 4_u16,
                        },
                        Node {
                            id: name(b'C'),
                            left: 6_u16,
                            right: 5_u16,
                        },
                        Node {
                            id: name(b'D'),
                            left: 3_u16,
                            right: 3_u16,
                        },
                        Node {
                            id: name(b'E'),
                            left: 4_u16,
                            right: 4_u16,
                        },
                        Node {
                            id: name(b'G'),
                            left: 5_u16,
                            right: 5_u16,
                        },
                        Node {
                            id: name(b'Z'),
                            left: 6_u16,
                            right: 6_u16,
                        },
                    ],
                },
                Solution {
                    instructions: vec![L, L, R],
                    start: Some(0_u16),
                    end: Some(2_u16),
                    nodes: vec![
                        Node {
                            id: name(b'A'),
                            left: 1_u16,
                            right: 1_u16,
                        },
                        Node {
                            id: name(b'B'),
                            left: 0_u16,
                            right: 2_u16,
                        },
                        Node {
                            id: name(b'Z'),
                            left: 2_u16,
                            right: 2_u16,
                        },
                    ],
                },
                Solution {
                    instructions: vec![L, R],
                    start: None,
                    end: None,
                    nodes: vec![
                        Node {
                            id: name_str("11A"),
                            left: 1_u16,
                            right: 7_u16,
                        },
                        Node {
                            id: name_str("11B"),
                            left: 7_u16,
                            right: 2_u16,
                        },
                        Node {
                            id: name_str("11Z"),
                            left: 1_u16,
                            right: 7_u16,
                        },
                        Node {
                            id: name_str("22A"),
                            left: 4_u16,
                            right: 7_u16,
                        },
                        Node {
                            id: name_str("22B"),
                            left: 5_u16,
                            right: 5_u16,
                        },
                        Node {
                            id: name_str("22C"),
                            left: 6_u16,
                            right: 6_u16,
                        },
                        Node {
                            id: name_str("22Z"),
                            left: 4_u16,
                            right: 4_u16,
                        },
                        Node {
                            id: name_str("XXX"),
                            left: 7_u16,
                            right: 7_u16,
                        },
                    ],
                },
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().cloned().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_steps() {
        for (index, steps) in [2_usize, 6_usize].into_iter().enumerate() {
            assert_eq!(solution(index).steps(), Some(steps));
        }
    }

    #[test]
    fn test_ghost_steps() {
        assert_eq!(solution(2_usize).ghost_steps(), 6_usize);
    }
}
