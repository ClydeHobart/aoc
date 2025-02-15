use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt},
        error::Error,
        multi::separated_list0,
        sequence::separated_pair,
        Err, IResult,
    },
    std::{cmp::Ordering, ops::BitAnd},
};

/* --- Day 23: LAN Party ---

As The Historians wander around a secure area at Easter Bunny HQ, you come across posters for a LAN party scheduled for today! Maybe you can find it; you connect to a nearby datalink port and download a map of the local network (your puzzle input).

The network map provides a list of every connection between two computers. For example:

kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn

Each line of text in the network map represents a single connection; the line kh-tc represents a connection between the computer named kh and the computer named tc. Connections aren't directional; tc-kh would mean exactly the same thing.

LAN parties typically involve multiplayer games, so maybe you can locate it by finding groups of connected computers. Start by looking for sets of three computers where each computer in the set is connected to the other two computers.

In this example, there are 12 such sets of three inter-connected computers:

aq,cg,yn
aq,vc,wq
co,de,ka
co,de,ta
co,ka,ta
de,ka,ta
kh,qp,ub
qp,td,wh
tb,vc,wq
tc,td,wh
td,wh,yn
ub,vc,wq

If the Chief Historian is here, and he's at the LAN party, it would be best to know that right away. You're pretty sure his computer's name starts with t, so consider only sets of three computers where at least one computer's name starts with t. That narrows the list down to 7 sets of three inter-connected computers:

co,de,ta
co,ka,ta
de,ka,ta
qp,td,wh
tb,vc,wq
tc,td,wh
td,wh,yn

Find all the sets of three inter-connected computers. How many contain at least one computer with a name that starts with t?

--- Part Two ---

There are still way too many results to go through them all. You'll have to find the LAN party another way and go there yourself.

Since it doesn't seem like any employees are around, you figure they must all be at the LAN party. If that's true, the LAN party will be the largest set of computers that are all connected to each other. That is, for each computer at the LAN party, that computer will have a connection to every other computer at the LAN party.

In the above example, the largest set of computers that are all connected to each other is made up of co, de, ka, and ta. Each computer in this set has a connection to every other computer in the set:

ka-co
ta-co
de-co
ta-ka
de-ta
ka-de

The LAN party posters say that the password to get into the LAN party is the name of every computer at the LAN party, sorted alphabetically, then joined together with commas. (The people running the LAN party are clearly a bunch of nerds.) In this example, the password would be co,de,ka,ta.

What is the password to get into the LAN party? */

type ComputerIndexRaw = u16;
type ComputerIndex = Index<ComputerIndexRaw>;
type ComputerId = StaticString<{ Solution::COMPUTER_ID_LEN }>;
type ComputersBitArray = BitArr!(for Solution::COMPUTERS_LEN);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct ComputerData {
    connections: ComputersBitArray,
}

type ComputerTable = Table<ComputerId, ComputerData, ComputerIndexRaw>;
type ComputerConnection<T = ComputerIndex> = (T, T);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct Set<I>([I; 3_usize]);

type ComputerIdSet = Set<ComputerId>;
type ComputerIndexSet = Set<ComputerIndex>;

#[derive(Clone, Default)]
pub struct LanPartyAdditionalCliqueState;

impl AdditionalCliqueStateTrait for LanPartyAdditionalCliqueState {
    fn is_valid(&self) -> bool {
        true
    }

    fn maximal_cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }

    fn maximal_cmp_always_returns_equal(&self) -> bool {
        true
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    computers: ComputerTable,
}

impl Solution {
    const COMPUTER_ID_LEN: usize = 2_usize;
    const COMPUTERS_LEN: usize = 10_usize * u64::BITS as usize;

    fn parse_computer_id<'i>(input: &'i str) -> IResult<&'i str, ComputerId> {
        StaticString::parse_char1(Self::COMPUTER_ID_LEN, |c| c.is_ascii_lowercase())(input)
    }

    fn parse_computer_connection<'i>(
        input: &'i str,
    ) -> IResult<&'i str, ComputerConnection<ComputerId>> {
        separated_pair(Self::parse_computer_id, tag("-"), Self::parse_computer_id)(input)
    }

    fn computer_id_could_be_chief_historian(computer_id: ComputerId) -> bool {
        computer_id.as_str().as_bytes()[0_usize] == b't'
    }

    fn computer_id_set_could_contain_chief_historian(computer_id_set: &ComputerIdSet) -> bool {
        computer_id_set
            .0
            .iter()
            .cloned()
            .any(Self::computer_id_could_be_chief_historian)
    }

    fn iter_computer_index_sets(&self) -> impl Iterator<Item = ComputerIndexSet> + '_ {
        (0_usize..self.computers.as_slice().len()).flat_map(move |computer_index_a| {
            let computer_index_a: ComputerIndex = computer_index_a.into();
            let connections_a: &ComputersBitArray = &self.computers.as_slice()
                [computer_index_a.get()]
            .data
            .connections;
            let computer_index_b_offset: usize = computer_index_a.get() + 1_usize;

            connections_a[computer_index_b_offset..]
                .iter_ones()
                .flat_map(move |computer_index_b| {
                    let computer_index_b: ComputerIndex =
                        (computer_index_b + computer_index_b_offset).into();
                    let connections_b: &ComputersBitArray = &self.computers.as_slice()
                        [computer_index_b.get()]
                    .data
                    .connections;
                    let connections_a_and_b: ComputersBitArray =
                        connections_a.bitand(connections_b);
                    let computer_index_c_offset: usize = computer_index_b.get() + 1_usize;

                    connections_a_and_b
                        .into_iter()
                        .skip(computer_index_c_offset)
                        .enumerate()
                        .filter_map(move |(computer_index_c, is_connection)| {
                            is_connection.then(|| {
                                let computer_index_c: ComputerIndex =
                                    (computer_index_c + computer_index_c_offset).into();

                                Set([computer_index_a, computer_index_b, computer_index_c])
                            })
                        })
                })
        })
    }

    fn computer_id_set_from_computer_index_set(
        &self,
        computer_index_set: ComputerIndexSet,
    ) -> ComputerIdSet {
        let mut computer_id_set: ComputerIdSet = ComputerIdSet::default();

        for (computer_index, computer_id) in computer_index_set
            .0
            .into_iter()
            .zip(computer_id_set.0.iter_mut())
        {
            *computer_id = self.computers.as_slice()[computer_index.get()].id.clone();
        }

        computer_id_set
    }

    fn iter_possible_chief_historian_id_sets(&self) -> impl Iterator<Item = ComputerIdSet> + '_ {
        self.iter_computer_index_sets()
            .map(|computer_index_set| {
                self.computer_id_set_from_computer_index_set(computer_index_set)
            })
            .filter(Self::computer_id_set_could_contain_chief_historian)
    }

    fn count_possible_chief_historian_id_sets(&self) -> usize {
        self.iter_possible_chief_historian_id_sets().count()
    }

    fn lan_party_password(&self) -> String {
        self.run()
            .clique
            .iter_ones()
            .enumerate()
            .flat_map(|(clique_index, computer_index)| {
                [
                    if clique_index > 0_usize { "," } else { "" },
                    self.computers.as_slice()[computer_index].id.as_str(),
                ]
            })
            .collect()
    }
}

impl MaximumClique for Solution {
    type BitArray = ComputersBitArray;

    type AdditionalCliqueState = LanPartyAdditionalCliqueState;

    fn vertex_count(&self) -> usize {
        self.computers.as_slice().len()
    }

    fn integrate_vertex(
        &self,
        _additional_clique_state: &mut Self::AdditionalCliqueState,
        _vertex_index: usize,
    ) {
    }

    fn get_neighbors(&self, vertex_index: usize) -> &Self::BitArray {
        &self.computers.as_slice()[vertex_index].data.connections
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut computers: ComputerTable = ComputerTable::new();

        separated_list0(
            line_ending,
            map_opt(
                Self::parse_computer_connection,
                |(computer_id_a, computer_id_b)| {
                    (computer_id_a != computer_id_b).then(|| {
                        computers.find_or_add_index(&computer_id_a);
                        computers.find_or_add_index(&computer_id_b);
                    })
                },
            ),
        )(input)?;

        computers.sort_by_id();

        let input: &str = separated_list0(
            line_ending,
            map(
                Self::parse_computer_connection,
                |(computer_id_a, computer_id_b)| {
                    let computer_index_a: ComputerIndex =
                        computers.find_index_binary_search(&computer_id_a);
                    let computer_index_b: ComputerIndex =
                        computers.find_index_binary_search(&computer_id_b);

                    computers.as_slice_mut()[computer_index_a.get()]
                        .data
                        .connections
                        .set(computer_index_b.get(), true);
                    computers.as_slice_mut()[computer_index_b.get()]
                        .data
                        .connections
                        .set(computer_index_a.get(), true);
                },
            ),
        )(input)?
        .0;

        Ok((input, Self { computers }))
    }
}

impl RunQuestions for Solution {
    /// Most time consuming process was writing the test cases.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_possible_chief_historian_id_sets());
    }

    /// Honestly really surprised I got this one on the first try.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.lan_party_password());
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
        kh-tc\n\
        qp-kh\n\
        de-cg\n\
        ka-co\n\
        yn-aq\n\
        qp-ub\n\
        cg-tb\n\
        vc-aq\n\
        tb-ka\n\
        wh-tc\n\
        yn-cg\n\
        kh-ub\n\
        ta-co\n\
        de-co\n\
        tc-td\n\
        tb-wq\n\
        wh-td\n\
        ta-ka\n\
        td-qp\n\
        aq-cg\n\
        wq-ub\n\
        ub-vc\n\
        de-ta\n\
        wq-aq\n\
        wq-vc\n\
        wh-yn\n\
        ka-de\n\
        kh-ta\n\
        co-tc\n\
        wh-qp\n\
        tb-vc\n\
        td-yn\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            macro_rules! computersbitarr {
                [ $bits:expr ] => { {
                    let mut computers: ComputersBitArray = ComputersBitArray::ZERO;

                    for index in $bits.view_bits::<Lsb0>().iter_ones() {
                        computers.set(index, true);
                    }

                    computers
                } }
            }

            vec![Solution {
                computers: vec![
                    TableElement {
                        id: "aq".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_1101_0000_0000_0010_usize],
                        },
                    },
                    TableElement {
                        id: "cg".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_1000_0001_0000_1001_usize],
                        },
                    },
                    TableElement {
                        id: "co".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0000_0010_1001_1000_usize],
                        },
                    },
                    TableElement {
                        id: "de".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0000_0000_1001_0110_usize],
                        },
                    },
                    TableElement {
                        id: "ka".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0000_0001_1000_1100_usize],
                        },
                    },
                    TableElement {
                        id: "kh".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0000_1010_1100_0000_usize],
                        },
                    },
                    TableElement {
                        id: "qp".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0010_1100_0010_0000_usize],
                        },
                    },
                    TableElement {
                        id: "ta".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0000_0000_0011_1100_usize],
                        },
                    },
                    TableElement {
                        id: "tb".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0101_0000_0001_0010_usize],
                        },
                    },
                    TableElement {
                        id: "tc".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0010_0100_0010_0100_usize],
                        },
                    },
                    TableElement {
                        id: "td".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_1010_0010_0100_0000_usize],
                        },
                    },
                    TableElement {
                        id: "ub".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0101_0000_0110_0000_usize],
                        },
                    },
                    TableElement {
                        id: "vc".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0100_1001_0000_0001_usize],
                        },
                    },
                    TableElement {
                        id: "wh".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_1000_0110_0100_0000_usize],
                        },
                    },
                    TableElement {
                        id: "wq".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0001_1001_0000_0001_usize],
                        },
                    },
                    TableElement {
                        id: "yn".try_into().unwrap(),
                        data: ComputerData {
                            connections: computersbitarr![0b_0010_0100_0000_0011_usize],
                        },
                    },
                ]
                .try_into()
                .unwrap(),
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
    fn test_iter_computer_index_set() {
        for (index, computer_index_sets) in [vec![
            Set([0_usize.into(), 1_usize.into(), 15_usize.into()]),
            Set([0_usize.into(), 12_usize.into(), 14_usize.into()]),
            Set([2_usize.into(), 3_usize.into(), 4_usize.into()]),
            Set([2_usize.into(), 3_usize.into(), 7_usize.into()]),
            Set([2_usize.into(), 4_usize.into(), 7_usize.into()]),
            Set([3_usize.into(), 4_usize.into(), 7_usize.into()]),
            Set([5_usize.into(), 6_usize.into(), 11_usize.into()]),
            Set([6_usize.into(), 10_usize.into(), 13_usize.into()]),
            Set([8_usize.into(), 12_usize.into(), 14_usize.into()]),
            Set([9_usize.into(), 10_usize.into(), 13_usize.into()]),
            Set([10_usize.into(), 13_usize.into(), 15_usize.into()]),
            Set([11_usize.into(), 12_usize.into(), 14_usize.into()]),
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_computer_index_sets()
                    .collect::<Vec<Set<ComputerIndex>>>(),
                computer_index_sets
            );
        }
    }

    #[test]
    fn test_iter_possible_chief_historian_id_sets() {
        for (index, possible_chief_historian_id_sets) in [vec![
            Set::<ComputerId>([
                "co".try_into().unwrap(),
                "de".try_into().unwrap(),
                "ta".try_into().unwrap(),
            ]),
            Set([
                "co".try_into().unwrap(),
                "ka".try_into().unwrap(),
                "ta".try_into().unwrap(),
            ]),
            Set([
                "de".try_into().unwrap(),
                "ka".try_into().unwrap(),
                "ta".try_into().unwrap(),
            ]),
            Set([
                "qp".try_into().unwrap(),
                "td".try_into().unwrap(),
                "wh".try_into().unwrap(),
            ]),
            Set([
                "tb".try_into().unwrap(),
                "vc".try_into().unwrap(),
                "wq".try_into().unwrap(),
            ]),
            Set([
                "tc".try_into().unwrap(),
                "td".try_into().unwrap(),
                "wh".try_into().unwrap(),
            ]),
            Set([
                "td".try_into().unwrap(),
                "wh".try_into().unwrap(),
                "yn".try_into().unwrap(),
            ]),
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_possible_chief_historian_id_sets()
                    .collect::<Vec<ComputerIdSet>>(),
                possible_chief_historian_id_sets
            );
        }
    }

    #[test]
    fn test_lan_party_password() {
        for (index, lan_party_password) in ["co,de,ka,ta"].into_iter().enumerate() {
            assert_eq!(solution(index).lan_party_password(), lan_party_password);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
