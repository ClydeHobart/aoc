use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, success, verify},
        error::Error,
        multi::separated_list1,
        sequence::{preceded, separated_pair},
        Err, IResult,
    },
    std::{
        collections::VecDeque,
        mem::{transmute, MaybeUninit},
    },
};

/* --- Day 7: Some Assembly Required ---

This year, Santa brought little Bobby Tables a set of wires and bitwise logic gates! Unfortunately, little Bobby is a little under the recommended age range, and he needs help assembling the circuit.

Each wire has an identifier (some lowercase letters) and can carry a 16-bit signal (a number from 0 to 65535). A signal is provided to each wire by a gate, another wire, or some specific value. Each wire can only get a signal from one source, but can provide its signal to multiple destinations. A gate provides no signal until all of its inputs have a signal.

The included instructions booklet describes how to connect the parts together: x AND y -> z means to connect wires x and y to an AND gate, and then connect its output to wire z.

For example:

    123 -> x means that the signal 123 is provided to wire x.
    x AND y -> z means that the bitwise AND of wire x and wire y is provided to wire z.
    p LSHIFT 2 -> q means that the value from wire p is left-shifted by 2 and then provided to wire q.
    NOT e -> f means that the bitwise complement of the value from wire e is provided to wire f.

Other possible gates include OR (bitwise OR) and RSHIFT (right-shift). If, for some reason, you'd like to emulate the circuit instead, almost all programming languages (for example, C, JavaScript, or Python) provide operators for these gates.

For example, here is a simple circuit:

123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i

After it is run, these are the signals on the wires:

d: 72
e: 507
f: 492
g: 114
h: 65412
i: 65079
x: 123
y: 456

In little Bobby's kit's instructions booklet (provided as your puzzle input), what signal is ultimately provided to wire a?

--- Part Two ---

Now, take the signal you got on wire a, override wire b to that signal, and reset the other wires (including wire a). What new signal is ultimately provided to wire a? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum WireValue<W = WireIndex> {
    Wire(W),
    Constant(u16),
}

impl WireValue {
    fn get_value(self, values: &[u16]) -> u16 {
        match self {
            WireValue::Wire(wire_index) => values[wire_index.get()],
            WireValue::Constant(constant) => constant,
        }
    }
}

impl<W> WireValue<W> {
    fn get_wire(&self) -> Option<W>
    where
        W: Clone,
    {
        match self {
            WireValue::Wire(wire_id) => Some(wire_id.clone()),
            WireValue::Constant(_) => None,
        }
    }
}

impl WireValue<WireId> {
    fn as_index_wire_value(&self, wires: &MaybeUninitWireTable) -> WireValue {
        match self {
            WireValue::Wire(wire_id) => WireValue::Wire(wires.find_index_binary_search(wire_id)),
            WireValue::Constant(constant) => WireValue::Constant(*constant),
        }
    }
}

impl Parse for WireValue<WireId> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(parse_wire_id, Self::Wire),
            map(parse_integer, Self::Constant),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
enum WireInputType {
    Value,
    Not,
    And,
    Or,
    LShift,
    RShift,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct WireInput<W = WireIndex> {
    wire_input_type: WireInputType,
    a: WireValue<W>,
    b: Option<WireValue<W>>,
}

impl WireInput {
    fn get_value(&self, values: &[u16]) -> u16 {
        match self.wire_input_type {
            WireInputType::Value => self.a.get_value(values),
            WireInputType::Not => !self.a.get_value(values),
            WireInputType::And => self.a.get_value(values) & self.b.unwrap().get_value(values),
            WireInputType::Or => self.a.get_value(values) | self.b.unwrap().get_value(values),
            WireInputType::LShift => self.a.get_value(values) << self.b.unwrap().get_value(values),
            WireInputType::RShift => self.a.get_value(values) >> self.b.unwrap().get_value(values),
        }
    }
}

impl<W: Clone> WireInput<W> {
    fn get_wire_a(&self) -> Option<W> {
        self.a.get_wire()
    }

    fn get_wire_b(&self) -> Option<W> {
        self.b.as_ref().and_then(WireValue::get_wire)
    }

    fn iter_wires(&self) -> impl Iterator<Item = W> {
        self.get_wire_a().into_iter().chain(self.get_wire_b())
    }
}

impl WireInput<WireId> {
    fn as_index_wire_input(&self, wires: &MaybeUninitWireTable) -> WireInput {
        WireInput {
            wire_input_type: self.wire_input_type,
            a: self.a.as_index_wire_value(wires),
            b: self.b.as_ref().map(|b| b.as_index_wire_value(wires)),
        }
    }
}

impl Parse for WireInput<WireId> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            preceded(
                tag("NOT "),
                map(WireValue::<WireId>::parse, |a| Self {
                    wire_input_type: WireInputType::Not,
                    a,
                    b: None,
                }),
            ),
            |input| {
                let (input, a): (&str, WireValue<WireId>) = WireValue::<WireId>::parse(input)?;
                let (input, wire_input_type): (&str, WireInputType) = alt((
                    map(tag(" AND "), |_| WireInputType::And),
                    map(tag(" OR "), |_| WireInputType::Or),
                    map(tag(" LSHIFT "), |_| WireInputType::LShift),
                    map(tag(" RSHIFT "), |_| WireInputType::RShift),
                    map(success(()), |_| WireInputType::Value),
                ))(input)?;

                if wire_input_type == WireInputType::Value {
                    Ok((
                        input,
                        Self {
                            wire_input_type,
                            a,
                            b: None,
                        },
                    ))
                } else {
                    let (input, b): (&str, WireValue<WireId>) =
                        verify(WireValue::<WireId>::parse, |b| {
                            !matches!(
                                wire_input_type,
                                WireInputType::LShift | WireInputType::RShift
                            ) || matches!(
                                b,
                                WireValue::Constant(constant) if *constant < u16::BITS as u16
                            )
                        })(input)?;

                    Ok((
                        input,
                        Self {
                            wire_input_type,
                            a,
                            b: Some(b),
                        },
                    ))
                }
            },
        ))(input)
    }
}

const MAX_WIRE_ID_LEN: usize = 2_usize;
const MAX_WIRES_LEN: usize = 384_usize;

type WireBitArray = BitArr!(for MAX_WIRES_LEN);
type WireId = StaticString<MAX_WIRE_ID_LEN>;

fn parse_wire_id<'i>(input: &'i str) -> IResult<&'i str, WireId> {
    WireId::parse_char1(1_usize, |c| c.is_ascii_lowercase())(input)
}

type WireIndexRaw = u16;
type WireIndex = Index<WireIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
struct WireData<W = WireIndex> {
    wire_input: WireInput<W>,
    out_neighbors: WireBitArray,
}

impl WireData<WireId> {
    fn as_index_wire_data(&self, wires: &MaybeUninitWireTable) -> WireData {
        WireData {
            wire_input: self.wire_input.as_index_wire_input(wires),
            out_neighbors: self.out_neighbors,
        }
    }
}

struct MaybeUninitWireData(MaybeUninit<WireData>);

impl Default for MaybeUninitWireData {
    fn default() -> Self {
        Self(MaybeUninit::uninit())
    }
}

type Wire<W = WireIndex> = TableElement<WireId, WireData<W>>;
type WireTable = Table<WireId, WireData, WireIndexRaw>;
type MaybeUninitWireTable = Table<WireId, MaybeUninitWireData, WireIndexRaw>;

#[derive(Default)]
struct BlockedNeighbors {
    blocked_a_in_neighbors: WireBitArray,
    blocked_b_in_neighbors: WireBitArray,
}

impl BlockedNeighbors {
    const BLOCKED_NEIGHBOR_FUNCS: &'static [(
        fn(&WireInput) -> Option<WireIndex>,
        fn(&BlockedNeighbors, WireIndex) -> bool,
    )] = &[
        (WireInput::get_wire_a, Self::is_a_in_neighbor_blocked),
        (WireInput::get_wire_b, Self::is_b_in_neighbor_blocked),
    ];

    fn is_a_in_neighbor_blocked(&self, out_neighbor_wire_index: WireIndex) -> bool {
        self.blocked_a_in_neighbors[out_neighbor_wire_index.get()]
    }

    fn is_b_in_neighbor_blocked(&self, out_neighbor_wire_index: WireIndex) -> bool {
        self.blocked_b_in_neighbors[out_neighbor_wire_index.get()]
    }
}

struct WireTopologicalOrderFinder<'w> {
    wires: &'w [Wire],
    blocked_neighbors: BlockedNeighbors,
    edge_count: usize,
}

impl<'w> WireTopologicalOrderFinder<'w> {
    fn iter_in_neighbors<'s>(
        &'s self,
        out_neighbor_wire_index: WireIndex,
    ) -> impl Iterator<Item = WireIndex> + 's
    where
        'w: 's,
    {
        let out_neighbor_wire_input: &WireInput =
            &self.wires[out_neighbor_wire_index.get()].data.wire_input;

        BlockedNeighbors::BLOCKED_NEIGHBOR_FUNCS
            .iter()
            .copied()
            .filter_map(move |(get_wire, is_in_neighbor_blocked)| {
                get_wire(out_neighbor_wire_input).filter(|_| {
                    !is_in_neighbor_blocked(&self.blocked_neighbors, out_neighbor_wire_index)
                })
            })
    }
}

impl<'w> Kahn for WireTopologicalOrderFinder<'w> {
    type Vertex = WireIndex;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>) {
        initial_set.clear();
        initial_set.extend(
            self.wires
                .iter()
                .enumerate()
                .filter_map(|(wire_index, wire)| {
                    (wire.data.wire_input.iter_wires().count() == 0_usize)
                        .then(|| WireIndex::from(wire_index))
                }),
        );
    }

    fn out_neighbors(
        &self,
        &in_neighbor_wire_index: &Self::Vertex,
        neighbors: &mut Vec<Self::Vertex>,
    ) {
        neighbors.clear();
        neighbors.extend(
            self.wires[in_neighbor_wire_index.get()]
                .data
                .out_neighbors
                .iter_ones()
                .map(WireIndex::from)
                .filter_map(|out_neighbor_wire_index| {
                    self.iter_in_neighbors(out_neighbor_wire_index)
                        .any(|in_neighbor_of_out_neighbor_wire_index| {
                            in_neighbor_of_out_neighbor_wire_index == in_neighbor_wire_index
                        })
                        .then(|| out_neighbor_wire_index)
                }),
        );
    }

    fn remove_edge(
        &mut self,
        &in_neighbor_wire_index: &Self::Vertex,
        &out_neighbor_wire_index: &Self::Vertex,
    ) {
        if self.wires[out_neighbor_wire_index.get()]
            .data
            .wire_input
            .get_wire_a()
            == Some(in_neighbor_wire_index)
        {
            &mut self.blocked_neighbors.blocked_a_in_neighbors
        } else {
            &mut self.blocked_neighbors.blocked_b_in_neighbors
        }
        .set(out_neighbor_wire_index.get(), true);

        self.edge_count -= 1_usize;
    }

    fn has_in_neighbors(&self, &out_neighbor_wire_index: &Self::Vertex) -> bool {
        self.iter_in_neighbors(out_neighbor_wire_index).count() > 0_usize
    }

    fn any_edges_exist(&self) -> bool {
        self.edge_count > 0_usize
    }

    fn reset(&mut self) {
        self.blocked_neighbors = BlockedNeighbors::default();
        self.edge_count = self
            .wires
            .iter()
            .map(|wire| wire.data.wire_input.iter_wires().count())
            .sum();
    }

    fn order_set(&self, set: &mut VecDeque<Self::Vertex>) {
        set.make_contiguous().sort();
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    wires: WireTable,
    a: WireIndex,
    b: WireIndex,
}

impl Solution {
    fn iter_wire_ids(wire: &Wire<WireId>) -> impl Iterator<Item = WireId> {
        [wire.id.clone()]
            .into_iter()
            .chain(wire.data.wire_input.iter_wires())
    }

    fn parse_wire<'i>(input: &'i str) -> IResult<&'i str, Wire<WireId>> {
        map(
            separated_pair(WireInput::<WireId>::parse, tag(" -> "), parse_wire_id),
            |(wire_input, id)| Wire {
                id,
                data: WireData {
                    wire_input,
                    out_neighbors: WireBitArray::ZERO,
                },
            },
        )(input)
    }

    fn try_compute_topological_order(&self) -> Option<Vec<WireIndex>> {
        WireTopologicalOrderFinder {
            wires: self.wires.as_slice(),
            blocked_neighbors: BlockedNeighbors::default(),
            edge_count: 0_usize,
        }
        .run()
    }

    fn compute_final_values(
        &self,
        topological_order: &[WireIndex],
        override_value: Option<(WireIndex, u16)>,
        out_values: &mut [u16],
    ) {
        out_values.fill(0_u16);

        for wire_index in topological_order.iter().copied() {
            let value: u16 = override_value
                .as_ref()
                .and_then(|&(override_wire_index, override_value)| {
                    (override_wire_index == wire_index).then_some(override_value)
                })
                .unwrap_or_else(|| {
                    self.wires.as_slice()[wire_index.get()]
                        .data
                        .wire_input
                        .get_value(out_values)
                });

            out_values[wire_index.get()] = value;
        }
    }

    fn try_compute_final_a_value(&self) -> Option<u16> {
        self.a
            .is_valid()
            .then(|| {
                self.try_compute_topological_order()
                    .map(|topological_order| {
                        let mut values: [u16; MAX_WIRES_LEN] = [0_u16; MAX_WIRES_LEN];

                        self.compute_final_values(&topological_order, None, &mut values);

                        values[self.a.get()]
                    })
            })
            .flatten()
    }

    fn try_compute_final_a_value_with_overridden_b(&self) -> Option<u16> {
        (self.a.is_valid() && self.b.is_valid())
            .then(|| {
                self.try_compute_topological_order()
                    .map(|topological_order| {
                        let mut values: [u16; MAX_WIRES_LEN] = [0_u16; MAX_WIRES_LEN];

                        self.compute_final_values(&topological_order, None, &mut values);

                        let override_value: Option<(WireIndex, u16)> =
                            Some((self.b, values[self.a.get()]));

                        self.compute_final_values(&topological_order, override_value, &mut values);

                        values[self.a.get()]
                    })
            })
            .flatten()
    }

    fn init_out_neighbors(&mut self) {
        for out_neighbor_index in 0_usize..self.wires.as_slice().len() {
            let wire_input: WireInput = self.wires.as_slice()[out_neighbor_index].data.wire_input;

            if let WireValue::Wire(in_neighbor_index) = wire_input.a {
                self.wires.as_slice_mut()[in_neighbor_index.get()]
                    .data
                    .out_neighbors
                    .set(out_neighbor_index, true);
            }

            if let Some(WireValue::Wire(in_neighbor_index)) = wire_input.b {
                self.wires.as_slice_mut()[in_neighbor_index.get()]
                    .data
                    .out_neighbors
                    .set(out_neighbor_index, true);
            }
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut wires: MaybeUninitWireTable = MaybeUninitWireTable::new();

        separated_list1(
            line_ending,
            map(Self::parse_wire, |wire| {
                for wire_id in Self::iter_wire_ids(&wire) {
                    wires.find_or_add_index(&wire_id);
                }
            }),
        )(input)?;

        wires.sort_by_id();

        verify(success(()), |_| wires.as_slice().len() <= MAX_WIRES_LEN)("too many wires")?;

        let mut present_wires: WireBitArray = WireBitArray::ZERO;
        let mut found_invalid_wire: bool = false;
        let mut found_duplicate_wire: bool = false;

        let input: &str = separated_list1(
            line_ending,
            map_opt(Self::parse_wire, |wire| {
                let wire_index: WireIndex = wires.find_index_binary_search(&wire.id);

                if !wire_index.is_valid() {
                    found_invalid_wire = true;

                    None
                } else if present_wires[wire_index.get()] {
                    found_duplicate_wire = true;

                    None
                } else {
                    present_wires.set(wire_index.get(), true);

                    let wire_data: WireData = wire.data.as_index_wire_data(&wires);

                    wires.as_slice_mut()[wire_index.get()]
                        .data
                        .0
                        .write(wire_data);

                    Some(())
                }
            }),
        )(input)?
        .0;

        verify(success(()), |_| !found_invalid_wire)("found invalid wire")?;
        verify(success(()), |_| !found_duplicate_wire)("found duplicate wire")?;
        verify(success(()), |_| {
            present_wires.count_ones() == wires.as_slice().len()
        })("not all wires initialized")?;

        let a: WireIndex = wires.find_index_binary_search(&"a".try_into().unwrap());
        let b: WireIndex = wires.find_index_binary_search(&"b".try_into().unwrap());

        let mut solution: Self = Self {
            // SAFETY: All wires are present and accounted for.
            wires: unsafe { transmute(wires) },
            a,
            b,
        };

        solution.init_out_neighbors();

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// (written after part 2) I was really hoping it didn't turn out like the adder one from 2024.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_final_a_value());
    }

    /// Not super interesting to watch, but I'm at least glad that it didn't require recomputinug
    /// the topological order.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_final_a_value_with_overridden_b());
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
        123 -> x\n\
        456 -> y\n\
        x AND y -> d\n\
        x OR y -> e\n\
        x LSHIFT 2 -> f\n\
        y RSHIFT 2 -> g\n\
        NOT x -> h\n\
        NOT y -> i\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                wires: vec![
                    Wire {
                        id: "d".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::And,
                                a: WireValue::Wire(6_usize.into()),
                                b: Some(WireValue::Wire(7_usize.into())),
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "e".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::Or,
                                a: WireValue::Wire(6_usize.into()),
                                b: Some(WireValue::Wire(7_usize.into())),
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "f".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::LShift,
                                a: WireValue::Wire(6_usize.into()),
                                b: Some(WireValue::Constant(2_u16)),
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "g".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::RShift,
                                a: WireValue::Wire(7_usize.into()),
                                b: Some(WireValue::Constant(2_u16)),
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "h".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::Not,
                                a: WireValue::Wire(6_usize.into()),
                                b: None,
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "i".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::Not,
                                a: WireValue::Wire(7_usize.into()),
                                b: None,
                            },
                            out_neighbors: WireBitArray::ZERO,
                        },
                    },
                    Wire {
                        id: "x".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::Value,
                                a: WireValue::Constant(123_u16),
                                b: None,
                            },
                            out_neighbors: bitarr_typed!(WireBitArray; 1, 1, 1, 0, 1, 0, 0, 0),
                        },
                    },
                    Wire {
                        id: "y".try_into().unwrap(),
                        data: WireData {
                            wire_input: WireInput {
                                wire_input_type: WireInputType::Value,
                                a: WireValue::Constant(456_u16),
                                b: None,
                            },
                            out_neighbors: bitarr_typed!(WireBitArray; 1, 1, 0, 1, 0, 1, 0, 0),
                        },
                    },
                ]
                .try_into()
                .unwrap(),
                a: WireIndex::invalid(),
                b: WireIndex::invalid(),
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
    fn test_try_compute_topological_order() {
        for (index, topological_order) in [Some(vec![
            WireIndex::from(6_usize),
            2_usize.into(),
            4_usize.into(),
            7_usize.into(),
            0_usize.into(),
            1_usize.into(),
            3_usize.into(),
            5_usize.into(),
        ])]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).try_compute_topological_order(),
                topological_order
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
