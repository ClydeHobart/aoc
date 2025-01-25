use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{line_ending, one_of},
        combinator::{map, map_opt, opt, success, verify},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::collections::VecDeque,
};

/* --- Day 24: Crossed Wires ---

You and The Historians arrive at the edge of a large grove somewhere in the jungle. After the last incident, the Elves installed a small device that monitors the fruit. While The Historians search the grove, one of them asks if you can take a look at the monitoring device; apparently, it's been malfunctioning recently.

The device seems to be trying to produce a number through some boolean logic gates. Each gate has two inputs and one output. The gates all operate on values that are either true (1) or false (0).

    AND gates output 1 if both inputs are 1; if either input is 0, these gates output 0.
    OR gates output 1 if one or both inputs is 1; if both inputs are 0, these gates output 0.
    XOR gates output 1 if the inputs are different; if the inputs are the same, these gates output 0.

Gates wait until both inputs are received before producing output; wires can carry 0, 1 or no value at all. There are no loops; once a gate has determined its output, the output will not change until the whole system is reset. Each wire is connected to at most one gate output, but can be connected to many gate inputs.

Rather than risk getting shocked while tinkering with the live system, you write down all of the gate connections and initial wire values (your puzzle input) so you can consider them in relative safety. For example:

x00: 1
x01: 1
x02: 1
y00: 0
y01: 1
y02: 0

x00 AND y00 -> z00
x01 XOR y01 -> z01
x02 OR y02 -> z02

Because gates wait for input, some wires need to start with a value (as inputs to the entire system). The first section specifies these values. For example, x00: 1 means that the wire named x00 starts with the value 1 (as if a gate is already outputting that value onto that wire).

The second section lists all of the gates and the wires connected to them. For example, x00 AND y00 -> z00 describes an instance of an AND gate which has wires x00 and y00 connected to its inputs and which will write its output to wire z00.

In this example, simulating these gates eventually causes 0 to appear on wire z00, 0 to appear on wire z01, and 1 to appear on wire z02.

Ultimately, the system is trying to produce a number by combining the bits on all wires starting with z. z00 is the least significant bit, then z01, then z02, and so on.

In this example, the three output bits form the binary number 100 which is equal to the decimal number 4.

Here's a larger example:

x00: 1
x01: 0
x02: 1
x03: 1
x04: 0
y00: 1
y01: 1
y02: 1
y03: 1
y04: 1

ntg XOR fgs -> mjb
y02 OR x01 -> tnw
kwq OR kpj -> z05
x00 OR x03 -> fst
tgd XOR rvg -> z01
vdt OR tnw -> bfw
bfw AND frj -> z10
ffh OR nrd -> bqk
y00 AND y03 -> djm
y03 OR y00 -> psh
bqk OR frj -> z08
tnw OR fst -> frj
gnj AND tgd -> z11
bfw XOR mjb -> z00
x03 OR x00 -> vdt
gnj AND wpb -> z02
x04 AND y00 -> kjc
djm OR pbm -> qhw
nrd AND vdt -> hwm
kjc AND fst -> rvg
y04 OR y02 -> fgs
y01 AND x02 -> pbm
ntg OR kjc -> kwq
psh XOR fgs -> tgd
qhw XOR tgd -> z09
pbm OR djm -> kpj
x03 XOR y03 -> ffh
x00 XOR y04 -> ntg
bfw OR bqk -> z06
nrd XOR fgs -> wpb
frj XOR qhw -> z04
bqk OR frj -> z07
y03 OR x01 -> nrd
hwm AND bqk -> z03
tgd XOR rvg -> z12
tnw OR pbm -> gnj

After waiting for values on all wires starting with z, the wires in this system have the following values:

bfw: 1
bqk: 1
djm: 1
ffh: 0
fgs: 1
frj: 1
fst: 1
gnj: 1
hwm: 1
kjc: 0
kpj: 1
kwq: 0
mjb: 1
nrd: 1
ntg: 0
pbm: 1
psh: 1
qhw: 1
rvg: 0
tgd: 0
tnw: 1
vdt: 1
wpb: 0
z00: 0
z01: 0
z02: 0
z03: 1
z04: 0
z05: 1
z06: 1
z07: 1
z08: 1
z09: 1
z10: 1
z11: 0
z12: 0

Combining the bits from all wires starting with z produces the binary number 0011111101000. Converting this number to decimal produces 2024.

Simulate the system of gates and wires. What decimal number does it output on the wires starting with z?

--- Part Two ---

After inspecting the monitoring device more closely, you determine that the system you're simulating is trying to add two binary numbers.

Specifically, it is treating the bits on wires starting with x as one binary number, treating the bits on wires starting with y as a second binary number, and then attempting to add those two numbers together. The output of this operation is produced as a binary number on the wires starting with z. (In all three cases, wire 00 is the least significant bit, then 01, then 02, and so on.)

The initial values for the wires in your puzzle input represent just one instance of a pair of numbers that sum to the wrong value. Ultimately, any two binary numbers provided as input should be handled correctly. That is, for any combination of bits on wires starting with x and wires starting with y, the sum of the two numbers those bits represent should be produced as a binary number on the wires starting with z.

For example, if you have an addition system with four x wires, four y wires, and five z wires, you should be able to supply any four-bit number on the x wires, any four-bit number on the y numbers, and eventually find the sum of those two numbers as a five-bit number on the z wires. One of the many ways you could provide numbers to such a system would be to pass 11 on the x wires (1011 in binary) and 13 on the y wires (1101 in binary):

x00: 1
x01: 1
x02: 0
x03: 1
y00: 1
y01: 0
y02: 1
y03: 1

If the system were working correctly, then after all gates are finished processing, you should find 24 (11+13) on the z wires as the five-bit binary number 11000:

z00: 0
z01: 0
z02: 0
z03: 1
z04: 1

Unfortunately, your actual system needs to add numbers with many more bits and therefore has many more wires.

Based on forensic analysis of scuff marks and scratches on the device, you can tell that there are exactly four pairs of gates whose output wires have been swapped. (A gate can only be in at most one such pair; no gate's output was swapped multiple times.)

For example, the system below is supposed to find the bitwise AND of the six-bit number on x00 through x05 and the six-bit number on y00 through y05 and then write the result as a six-bit number on z00 through z05:

x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00

However, in this example, two pairs of gates have had their output wires swapped, causing the system to produce wrong answers. The first pair of gates with swapped outputs is x00 AND y00 -> z05 and x05 AND y05 -> z00; the second pair of gates is x01 AND y01 -> z02 and x02 AND y02 -> z01. Correcting these two swaps results in this system that works as intended for any set of initial values on wires that start with x or y:

x00 AND y00 -> z00
x01 AND y01 -> z01
x02 AND y02 -> z02
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z05

In this example, two pairs of gates have outputs that are involved in a swap. By sorting their output wires' names and joining them with commas, the list of wires involved in swaps is z00,z01,z02,z05.

Of course, your actual system is much more complex than this, and the gates that need their outputs swapped could be anywhere, not just attached to a wire starting with z. If you were to determine that you need to swap output wires aaa with eee, ooo with z99, bbb with ccc, and aoc with z24, your answer would be aaa,aoc,bbb,ccc,eee,ooo,z24,z99.

Your system of gates and wires has four pairs of gates which need their output wires swapped - eight wires in total. Determine which four pairs of gates need their outputs swapped so that your system correctly performs addition; what do you get if you sort the names of the eight wires involved in a swap and then join those names with commas? */

type WireIndexRaw = u16;
type WireIndex = Index<WireIndexRaw>;
type WiresBitArray = BitArr!(for Solution::WIRES_LEN);
type WireGatesArray = [GateIndex; Solution::WIRES_LEN];
type WireId = StaticString<{ Solution::WIRE_ID_LEN }>;

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
enum WireInput {
    Constant(bool),
    Gate(GateIndex),
}

impl Default for WireInput {
    fn default() -> Self {
        Self::Gate(GateIndex::invalid())
    }
}

impl From<WireInput> for GateIndex {
    fn from(value: WireInput) -> Self {
        match value {
            WireInput::Gate(gate_index) => gate_index,
            _ => GateIndex::invalid(),
        }
    }
}

impl TryFrom<WireInput> for bool {
    type Error = ();

    fn try_from(value: WireInput) -> Result<Self, Self::Error> {
        match value {
            WireInput::Constant(constant) => Ok(constant),
            _ => Err(()),
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct WireData {
    input: WireInput,
    output_gates: GatesBitArray,
}

type Wire = TableElement<WireId, WireData>;
type WireTable = Table<WireId, WireData, WireIndexRaw>;
type GateIndexRaw = u8;
type GateIndex = Index<GateIndexRaw>;
type GatesBitArray = BitArr!(for Solution::GATES_LEN);
type GateWiresArray = [WireIndex; Solution::GATES_LEN];

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum GateOperation {
    #[default]
    And,
    Or,
    Xor,
}

impl GateOperation {
    fn operate(self, a: bool, b: bool) -> bool {
        match self {
            Self::And => a && b,
            Self::Or => a || b,
            Self::Xor => a ^ b,
        }
    }
}

impl Parse for GateOperation {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(tag("AND"), |_| Self::And),
            map(tag("OR"), |_| Self::Or),
            map(tag("XOR"), |_| Self::Xor),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Default, Ord, PartialEq, PartialOrd)]
struct Gate<W = WireIndex> {
    input_wire_a: W,
    operation: GateOperation,
    input_wire_b: W,
    output_wire: W,
}

impl Parse for Gate<WireId> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((
                Solution::parse_wire_id,
                tag(" "),
                GateOperation::parse,
                tag(" "),
                Solution::parse_wire_id,
                tag(" -> "),
                Solution::parse_wire_id,
                opt(line_ending),
            )),
            |(input_wire_a, _, operation, _, input_wire_b, _, output_wire, _)| {
                (Solution::try_bit_index_for_wire_id(output_wire.clone(), 'x').is_none()
                    && Solution::try_bit_index_for_wire_id(output_wire.clone(), 'y').is_none())
                .then(|| Self {
                    input_wire_a,
                    operation,
                    input_wire_b,
                    output_wire,
                })
            },
        )(input)
    }
}

type GateList = IdList<Gate, GateIndexRaw>;

struct GateMapping {
    gate_output_wires: GateWiresArray,
    wire_input_gates: WireGatesArray,
}

impl Default for GateMapping {
    fn default() -> Self {
        Self {
            gate_output_wires: LargeArrayDefault::large_array_default(),
            wire_input_gates: LargeArrayDefault::large_array_default(),
        }
    }
}

#[derive(Default)]
struct GateStates {
    input_wires_a: GatesBitArray,
    input_wires_b: GatesBitArray,
}

struct TopologicalWireOrderFinder<'a> {
    solution: &'a Solution,
    gate_mapping: &'a GateMapping,
    gate_states: &'a mut GateStates,
}

impl<'a> TopologicalWireOrderFinder<'a> {
    fn try_output_wire(&self, wire_index: WireIndex, gate_index: GateIndex) -> Option<WireIndex> {
        let gate_index: usize = gate_index.get();
        let gate: Gate = self.solution.gates.as_id_slice()[gate_index];

        if wire_index == gate.input_wire_a {
            self.gate_states.input_wires_a[gate_index]
                .then(|| self.gate_mapping.gate_output_wires[gate_index])
        } else if wire_index == gate.input_wire_b {
            self.gate_states.input_wires_b[gate_index]
                .then(|| self.gate_mapping.gate_output_wires[gate_index])
        } else {
            None
        }
    }
}

impl<'a> Kahn for TopologicalWireOrderFinder<'a> {
    type Vertex = WireIndex;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>) {
        initial_set.clear();
        initial_set.extend((0_usize..self.solution.wires.as_slice().len()).filter_map(
            |wire_index| {
                let wire_index: WireIndex = wire_index.into();

                (!self.has_in_neighbors(&wire_index)).then_some(wire_index)
            },
        ));
    }

    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        neighbors.extend(
            self.solution.wires.as_slice()[vertex.get()]
                .data
                .output_gates
                .iter_ones()
                .filter_map(|gate_index| self.try_output_wire(*vertex, gate_index.into())),
        );
    }

    fn remove_edge(&mut self, &from: &Self::Vertex, to: &Self::Vertex) {
        let gate_index: usize = self.gate_mapping.wire_input_gates[to.get()].get();
        let gate: Gate = self.solution.gates.as_id_slice()[gate_index];

        if from == gate.input_wire_a {
            self.gate_states.input_wires_a.set(gate_index, false);
        } else {
            assert_eq!(from, gate.input_wire_b);
            self.gate_states.input_wires_b.set(gate_index, false);
        }
    }

    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool {
        let gate_index: GateIndex = self.gate_mapping.wire_input_gates[vertex.get()];

        gate_index.is_valid() && {
            let gate_index: usize = gate_index.get();

            self.gate_states.input_wires_a[gate_index] || self.gate_states.input_wires_b[gate_index]
        }
    }

    fn any_edges_exist(&self) -> bool {
        self.gate_states.input_wires_a.any() || self.gate_states.input_wires_b.any()
    }

    fn reset(&mut self) {
        let gates_len: usize = self.solution.gates.as_id_slice().len();

        self.gate_states.input_wires_a[..gates_len].fill(true);
        self.gate_states.input_wires_b[..gates_len].fill(true);
    }

    fn order_set(&self, _set: &mut VecDeque<Self::Vertex>) {}
}

#[derive(Default)]
struct CircuitEvaluatorState {
    gate_mapping: GateMapping,
    gate_states: GateStates,
    kahn_state: KahnState<WireIndex>,
    wire_states: WiresBitArray,
}

impl CircuitEvaluatorState {
    fn compute_bus(&self, solution: &Solution, bus_id: char) -> u64 {
        let mut bus: u64 = 0_u64;

        for wire_index in self.wire_states.iter_ones() {
            if let Some(bit_index) = Solution::try_bit_index_for_wire_id(
                solution.wires.as_slice()[wire_index].id.clone(),
                bus_id,
            ) {
                bus.view_bits_mut::<Lsb0>().set(bit_index, true);
            }
        }

        bus
    }

    fn try_evaluate(&mut self, solution: &Solution) -> Option<()> {
        TopologicalWireOrderFinder {
            solution,
            gate_mapping: &self.gate_mapping,
            gate_states: &mut self.gate_states,
        }
        .run_internal(&mut self.kahn_state)
        .then(|| {
            for wire_index in self.kahn_state.list.drain(..) {
                let wire_index: usize = wire_index.get();
                let wire_state: bool = if let Some(gate_index) =
                    self.gate_mapping.wire_input_gates[wire_index].opt()
                {
                    let gate: Gate = solution.gates.as_id_slice()[gate_index.get()];

                    gate.operation.operate(
                        self.wire_states[gate.input_wire_a.get()],
                        self.wire_states[gate.input_wire_b.get()],
                    )
                } else {
                    self.wire_states[wire_index]
                };

                self.wire_states.set(wire_index, wire_state);
            }
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    wires: WireTable,
    gates: GateList,
    x: u64,
    y: u64,
    x_bits: usize,
    y_bits: usize,
    z_bits: usize,
}

#[derive(Clone, Copy, Default)]
struct FullAdderWires {
    x: WireIndex,
    y: WireIndex,

    /// Invalid for bit 0, otherwise should be valid
    carry_in: WireIndex,

    /// For bit 0, should be z
    x_xor_y: WireIndex,

    /// For bit 0, should be carry out
    x_and_y: WireIndex,

    /// For bit 0, should be invalid, otherwise should be valid
    carry_in_and_x_xor_y: WireIndex,
    z: WireIndex,

    /// For last bit, should be last z
    carry_out: WireIndex,
}

impl Solution {
    const WIRE_ID_LEN: usize = 3_usize;
    const WIRES_LEN: usize = 5_usize * usize::BITS as usize;
    const GATES_LEN: usize = 1_usize << u8::BITS;

    fn parse_wire_id<'i>(input: &'i str) -> IResult<&'i str, WireId> {
        WireId::parse_char1(Self::WIRE_ID_LEN, |c| {
            c.is_ascii_lowercase() || c.is_ascii_digit()
        })(input)
    }

    fn try_bit_index_for_wire_id(wire_id: WireId, bus_id: char) -> Option<usize> {
        preceded(one_of(&[bus_id][..]), parse_integer)(wire_id.as_str())
            .ok()
            .and_then(|(_, bit_index)| (bit_index < u64::BITS as usize).then_some(bit_index))
    }

    fn parse_wire_id_and_wire_input<'i>(input: &'i str) -> IResult<&'i str, (WireId, WireInput)> {
        map(
            tuple((
                verify(Self::parse_wire_id, |wire_id| {
                    Self::try_bit_index_for_wire_id(wire_id.clone(), 'x').is_some()
                        || Self::try_bit_index_for_wire_id(wire_id.clone(), 'y').is_some()
                }),
                tag(": "),
                map(one_of("01"), |c| WireInput::Constant(c == '1')),
                line_ending,
            )),
            |(wire_id, _, wire_input, _)| (wire_id, wire_input),
        )(input)
    }

    fn compute_bus_parameters(wires: &[Wire], bus_id: char) -> (u64, usize) {
        let mut bus: u64 = 0_u64;
        let mut bus_bits: usize = 0_usize;

        for wire in wires {
            if let Some(bit_index) = Self::try_bit_index_for_wire_id(wire.id.clone(), bus_id) {
                bus.view_bits_mut::<Lsb0>()
                    .set(bit_index, wire.data.input.try_into().unwrap_or_default());
                bus_bits = bus_bits.max(bit_index + 1_usize);
            }
        }

        (bus, bus_bits)
    }

    fn process_wire_result(
        bit_index: usize,
        verbose: bool,
        source_wire_a_label: &str,
        source_wire_b_label: &str,
        derived_wire: &mut WireIndex,
        computed_derived_wire: WireIndex,
        derived_wire_label: &str,
        should_overwrite_derived_wire: bool,
        erroneous_wires: &mut WiresBitArray,
    ) {
        if !computed_derived_wire.is_valid() {
            if verbose {
                println!(
                    "at bit {bit_index}, {derived_wire_label} couldn't be found from \
                    {source_wire_a_label} and {source_wire_b_label}({}candidate currently \
                    exists)",
                    if derived_wire.is_valid() { "" } else { "no " }
                );
            }
        } else {
            let derived_wire_is_valid: bool = derived_wire.is_valid();
            let wires_match: bool = computed_derived_wire == *derived_wire;

            if !derived_wire_is_valid || !wires_match {
                if verbose {
                    println!(
                        "at bit {bit_index}, correct {derived_wire_label} was found from \
                        {source_wire_a_label} and {source_wire_b_label}, {}correcting",
                        if should_overwrite_derived_wire {
                            ""
                        } else {
                            "not "
                        }
                    );
                }

                if derived_wire_is_valid && !wires_match {
                    erroneous_wires.set(computed_derived_wire.get(), true);
                    erroneous_wires.set(derived_wire.get(), true);
                }

                if should_overwrite_derived_wire {
                    *derived_wire = computed_derived_wire;
                }
            }
        }
    }

    fn init_gate_mapping(&self, gate_mapping: &mut GateMapping) {
        gate_mapping.wire_input_gates.fill(GateIndex::invalid());

        for (gate_index, (gate, gate_output_wire)) in self
            .gates
            .as_id_slice()
            .iter()
            .zip(gate_mapping.gate_output_wires.iter_mut())
            .enumerate()
        {
            *gate_output_wire = gate.output_wire;
            gate_mapping.wire_input_gates[gate.output_wire.get()] = gate_index.into();
        }
    }

    fn init_wire_states_for_x_and_y(&self, x: u64, y: u64, wire_states: &mut WiresBitArray) {
        wire_states.fill(false);

        for (wire, mut wire_state) in self.wires.as_slice().iter().zip(wire_states.iter_mut()) {
            *wire_state = if let Some(x_bit_index) =
                Self::try_bit_index_for_wire_id(wire.id.clone(), 'x')
            {
                x.view_bits::<Lsb0>()[x_bit_index]
            } else if let Some(y_bit_index) = Self::try_bit_index_for_wire_id(wire.id.clone(), 'y')
            {
                y.view_bits::<Lsb0>()[y_bit_index]
            } else {
                false
            };
        }
    }

    fn init_wire_states(&self, wire_states: &mut WiresBitArray) {
        self.init_wire_states_for_x_and_y(self.x, self.y, wire_states)
    }

    fn try_compute_final_z_value(&self) -> Option<u64> {
        let mut circuit_evaluator_state: CircuitEvaluatorState = CircuitEvaluatorState::default();

        self.init_gate_mapping(&mut circuit_evaluator_state.gate_mapping);
        self.init_wire_states(&mut circuit_evaluator_state.wire_states);

        circuit_evaluator_state
            .try_evaluate(self)
            .map(|_| circuit_evaluator_state.compute_bus(self, 'z'))
    }

    /// Returns `None` if an input is invalid, `Some(WireIndex::invalid())` if one couldn't be
    /// found, or `Some(wire_index)`.
    fn try_find_gate_output_from_inputs(
        &self,
        input_wire_a: WireIndex,
        input_wire_b: WireIndex,
        operation: GateOperation,
    ) -> Option<WireIndex> {
        (input_wire_a.is_valid() && input_wire_b.is_valid()).then(|| {
            let output_gates_a: &GatesBitArray =
                &self.wires.as_slice()[input_wire_a.get()].data.output_gates;
            let output_gates_b: &GatesBitArray =
                &self.wires.as_slice()[input_wire_b.get()].data.output_gates;

            output_gates_a
                .iter_ones()
                .filter_map(|gate_index| {
                    let gate: Gate = self.gates.as_id_slice()[gate_index];

                    (output_gates_b[gate_index] && gate.operation == operation)
                        .then_some(gate.output_wire)
                })
                .next()
                .unwrap_or_default()
        })
    }

    /// Returns `None` if the input or output is invalid, `Some(WireIndex::invalid())` if one
    /// couldn't be found, or `Some(wire_index)`.
    fn try_find_gate_input_from_input_and_output(
        &self,
        input_wire: WireIndex,
        output_wire: WireIndex,
        operation: GateOperation,
    ) -> Option<WireIndex> {
        (input_wire.is_valid() && output_wire.is_valid()).then(|| {
            self.wires.as_slice()[input_wire.get()]
                .data
                .output_gates
                .iter_ones()
                .filter_map(|gate_index| {
                    let gate: Gate = self.gates.as_id_slice()[gate_index];

                    (gate.operation == operation).then(|| {
                        if input_wire == gate.input_wire_a {
                            Some(gate.input_wire_b)
                        } else if input_wire == gate.input_wire_b {
                            Some(gate.input_wire_a)
                        } else {
                            None
                        }
                    })
                })
                .flatten()
                .next()
                .unwrap_or_default()
        })
    }

    fn find_erroneous_wire_connected_to_gate(
        &self,
        bit_index: usize,
        operation: GateOperation,
        verbose: bool,
        input_wire_a: &mut WireIndex,
        input_wire_a_label: &str,
        input_wire_b: &mut WireIndex,
        input_wire_b_label: &str,
        output_wire: &mut WireIndex,
        output_wire_label: &str,
        erroneous_wires: &mut WiresBitArray,
    ) {
        let input_wire_a_is_valid: bool = input_wire_a.is_valid();
        let input_wire_b_is_valid: bool = input_wire_a.is_valid();
        let output_wire_is_valid: bool = output_wire.is_valid();
        let output_from_input_wire_a_and_input_wire_b: Option<WireIndex> =
            self.try_find_gate_output_from_inputs(*input_wire_a, *input_wire_b, operation);
        let input_wire_a_from_input_wire_b_and_output_wire: Option<WireIndex> =
            self.try_find_gate_input_from_input_and_output(*input_wire_b, *output_wire, operation);
        let input_wire_b_from_input_wire_a_and_output_wire: Option<WireIndex> =
            self.try_find_gate_input_from_input_and_output(*input_wire_a, *output_wire, operation);

        if let Some(output_from_input_wire_a_and_input_wire_b) =
            output_from_input_wire_a_and_input_wire_b
        {
            Self::process_wire_result(
                bit_index,
                verbose,
                input_wire_a_label,
                input_wire_b_label,
                output_wire,
                output_from_input_wire_a_and_input_wire_b,
                output_wire_label,
                !output_wire_is_valid,
                erroneous_wires,
            );
        }

        if let Some(input_wire_a_from_input_wire_b_and_output_wire) =
            input_wire_a_from_input_wire_b_and_output_wire
        {
            Self::process_wire_result(
                bit_index,
                verbose,
                input_wire_b_label,
                output_wire_label,
                input_wire_a,
                input_wire_a_from_input_wire_b_and_output_wire,
                input_wire_a_label,
                !input_wire_a_is_valid,
                erroneous_wires,
            );
        }

        if let Some(input_wire_b_from_input_wire_a_and_output_wire) =
            input_wire_b_from_input_wire_a_and_output_wire
        {
            Self::process_wire_result(
                bit_index,
                verbose,
                input_wire_a_label,
                output_wire_label,
                input_wire_b,
                input_wire_b_from_input_wire_a_and_output_wire,
                input_wire_b_label,
                !input_wire_b_is_valid,
                erroneous_wires,
            );
        }
    }

    fn try_find_erroneous_wires(&self, verbose: bool) -> Option<WiresBitArray> {
        (self.x_bits == self.y_bits && self.x_bits + 1_usize == self.z_bits)
            .then_some(())
            .and_then(|_| {
                let full_bits: usize = self.x_bits;
                let mut full_adders: Vec<FullAdderWires> =
                    vec![FullAdderWires::default(); full_bits];
                let mut erroneous_wires: WiresBitArray = WiresBitArray::ZERO;

                for (wire_index, wire) in self.wires.as_slice().iter().enumerate() {
                    let wire_index: WireIndex = wire_index.into();

                    if let Some(bit_index) = Self::try_bit_index_for_wire_id(wire.id.clone(), 'x') {
                        full_adders[bit_index].x = wire_index;
                    } else if let Some(bit_index) =
                        Self::try_bit_index_for_wire_id(wire.id.clone(), 'y')
                    {
                        full_adders[bit_index].y = wire_index;
                    } else if let Some(bit_index) =
                        Self::try_bit_index_for_wire_id(wire.id.clone(), 'z')
                    {
                        if bit_index < full_bits {
                            full_adders[bit_index].z = wire_index;
                        }
                    }
                }

                let mut carry_in: WireIndex = WireIndex::invalid();

                for (bit_index, full_adder_wires) in full_adders.iter_mut().enumerate() {
                    full_adder_wires.carry_in = carry_in;
                    full_adder_wires.x.opt()?;
                    full_adder_wires.y.opt()?;
                    full_adder_wires.z.opt()?;

                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::Xor,
                        verbose,
                        &mut full_adder_wires.x,
                        "x",
                        &mut full_adder_wires.y,
                        "y",
                        &mut full_adder_wires.x_xor_y,
                        "x_xor_y",
                        &mut erroneous_wires,
                    );
                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::And,
                        verbose,
                        &mut full_adder_wires.x,
                        "x",
                        &mut full_adder_wires.y,
                        "y",
                        &mut full_adder_wires.x_and_y,
                        "x_and_y",
                        &mut erroneous_wires,
                    );

                    if bit_index == 0_usize {
                        if full_adder_wires.x_xor_y != full_adder_wires.z {
                            match (
                                full_adder_wires.x_xor_y.is_valid(),
                                full_adder_wires.z.is_valid(),
                            ) {
                                (true, true) => {
                                    if verbose {
                                        println!("at bit 0, x_xor_y was not z, correcting.");
                                    }

                                    erroneous_wires.set(full_adder_wires.x_xor_y.get(), true);
                                    erroneous_wires.set(full_adder_wires.z.get(), true);
                                }
                                (true, false) => full_adder_wires.z = full_adder_wires.x_xor_y,
                                (false, true) => full_adder_wires.x_xor_y = full_adder_wires.z,
                                (false, false) => (),
                            }
                        }

                        full_adder_wires.carry_out = full_adder_wires.x_and_y;
                        carry_in = full_adder_wires.carry_out;
                    } else {
                        self.find_erroneous_wire_connected_to_gate(
                            bit_index,
                            GateOperation::Xor,
                            verbose,
                            &mut full_adder_wires.carry_in,
                            "carry_in",
                            &mut full_adder_wires.x_xor_y,
                            "x_xor_y",
                            &mut full_adder_wires.z,
                            "z",
                            &mut erroneous_wires,
                        );
                        self.find_erroneous_wire_connected_to_gate(
                            bit_index,
                            GateOperation::And,
                            verbose,
                            &mut full_adder_wires.carry_in,
                            "carry_in",
                            &mut full_adder_wires.x_xor_y,
                            "x_xor_y",
                            &mut full_adder_wires.carry_in_and_x_xor_y,
                            "carry_in_and_x_xor_y",
                            &mut erroneous_wires,
                        );
                        self.find_erroneous_wire_connected_to_gate(
                            bit_index,
                            GateOperation::Or,
                            verbose,
                            &mut full_adder_wires.carry_in_and_x_xor_y,
                            "carry_in_and_x_xor_y",
                            &mut full_adder_wires.x_and_y,
                            "x_and_y",
                            &mut full_adder_wires.carry_out,
                            "carry_out",
                            &mut erroneous_wires,
                        );

                        carry_in = full_adder_wires.carry_out;
                    }
                }

                let mut carry_out: WireIndex = WireIndex::invalid();

                for (bit_index, full_adder_wires) in full_adders
                    .iter_mut()
                    .enumerate()
                    .rev()
                    .take(full_bits.saturating_sub(1_usize))
                {
                    full_adder_wires.carry_out = carry_out;

                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::Or,
                        verbose,
                        &mut full_adder_wires.carry_in_and_x_xor_y,
                        "carry_in_and_x_xor_y",
                        &mut full_adder_wires.x_and_y,
                        "x_and_y",
                        &mut full_adder_wires.carry_out,
                        "carry_out",
                        &mut erroneous_wires,
                    );
                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::And,
                        verbose,
                        &mut full_adder_wires.carry_in,
                        "carry_in",
                        &mut full_adder_wires.x_xor_y,
                        "x_xor_y",
                        &mut full_adder_wires.carry_in_and_x_xor_y,
                        "carry_in_and_x_xor_y",
                        &mut erroneous_wires,
                    );
                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::Xor,
                        verbose,
                        &mut full_adder_wires.carry_in,
                        "carry_in",
                        &mut full_adder_wires.x_xor_y,
                        "x_xor_y",
                        &mut full_adder_wires.z,
                        "z",
                        &mut erroneous_wires,
                    );
                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::And,
                        verbose,
                        &mut full_adder_wires.x,
                        "x",
                        &mut full_adder_wires.y,
                        "y",
                        &mut full_adder_wires.x_and_y,
                        "x_and_y",
                        &mut erroneous_wires,
                    );
                    self.find_erroneous_wire_connected_to_gate(
                        bit_index,
                        GateOperation::Xor,
                        verbose,
                        &mut full_adder_wires.x,
                        "x",
                        &mut full_adder_wires.y,
                        "y",
                        &mut full_adder_wires.x_xor_y,
                        "x_xor_y",
                        &mut erroneous_wires,
                    );

                    carry_out = full_adder_wires.carry_in;
                }

                Some(erroneous_wires)
            })
    }

    fn try_find_erroneous_wire_names(&self, verbose: bool) -> Option<String> {
        self.try_find_erroneous_wires(verbose)
            .map(|erroneous_wires| {
                erroneous_wires
                    .iter_ones()
                    .enumerate()
                    .flat_map(|(list_index, wire_index)| {
                        (list_index > 0_usize)
                            .then_some(",")
                            .into_iter()
                            .chain([self.wires.as_slice()[wire_index].id.as_str()])
                    })
                    .collect()
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut wires: WireTable = WireTable::default();
        let mut gates: GateList = GateList::default();

        let input: &str = terminated(
            many0(map(
                Self::parse_wire_id_and_wire_input,
                |(wire_id, wire_input)| {
                    let wire_index: WireIndex = wires.find_or_add_index(&wire_id);

                    // Even if this wire gets moved somewhere else after the sort, this data will be
                    // moved with it and will still be valid.
                    wires.as_slice_mut()[wire_index.get()].data.input = wire_input;
                },
            )),
            line_ending,
        )(input)?
        .0;

        many0(map(Gate::<WireId>::parse, |gate| {
            wires.find_or_add_index(&gate.input_wire_a);
            wires.find_or_add_index(&gate.input_wire_b);
            wires.find_or_add_index(&gate.output_wire);
        }))(input)?;

        // All wire IDs are known, so sort the table now.
        wires.sort_by_id();

        // Wire indices are now stable.
        let input: &str = many0(map(Gate::<WireId>::parse, |gate| {
            let input_wire_a: WireIndex = wires.find_index_binary_search(&gate.input_wire_a);
            let operation: GateOperation = gate.operation;
            let input_wire_b: WireIndex = wires.find_index_binary_search(&gate.input_wire_b);
            let output_wire: WireIndex = wires.find_index_binary_search(&gate.output_wire);

            gates.find_or_add_index(&Gate {
                input_wire_a,
                operation,
                input_wire_b,
                output_wire,
            });
        }))(input)?
        .0;

        // All gate IDs are now known, so sort the table now.
        gates.sort_by_id();

        // Gate indices are now stable.
        for (gate_index, gate) in gates.as_id_slice().iter().enumerate() {
            wires.as_slice_mut()[gate.input_wire_a.get()]
                .data
                .output_gates
                .set(gate_index, true);
            wires.as_slice_mut()[gate.input_wire_b.get()]
                .data
                .output_gates
                .set(gate_index, true);

            let output_wire_input: &mut WireInput =
                &mut wires.as_slice_mut()[gate.output_wire.get()].data.input;
            let output_wire_input_value: WireInput = map_opt(success(()), |_| {
                (*output_wire_input == WireInput::default())
                    .then(|| WireInput::Gate(gate_index.into()))
            })(input)?
            .1;

            *output_wire_input = output_wire_input_value;
        }

        verify(success(()), |_| {
            wires
                .as_slice()
                .iter()
                .all(|wire| wire.data.input != WireInput::default())
                && wires.as_slice().len() <= Self::WIRES_LEN
                && gates.as_slice().len() <= Self::GATES_LEN
        })(input)?;

        let (x, x_bits): (u64, usize) = Self::compute_bus_parameters(wires.as_slice(), 'x');
        let (y, y_bits): (u64, usize) = Self::compute_bus_parameters(wires.as_slice(), 'y');
        let z_bits: usize = Self::compute_bus_parameters(wires.as_slice(), 'z').1;

        Ok((
            input,
            Self {
                wires,
                gates,
                x,
                y,
                x_bits,
                y_bits,
                z_bits,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// Prediction for q2: feeding Z back in as the constants somehow?
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.gates.as_id_slice().len());
        dbg!(self.try_compute_final_z_value());
    }

    /// This took a while, and I was definitely going in the wrong direction for a time.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.try_find_erroneous_wire_names(args.verbose));
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
        "\
        x00: 1\n\
        x01: 1\n\
        x02: 1\n\
        y00: 0\n\
        y01: 1\n\
        y02: 0\n\
        \n\
        x00 AND y00 -> z00\n\
        x01 XOR y01 -> z01\n\
        x02 OR y02 -> z02\n",
        "\
        x00: 1\n\
        x01: 0\n\
        x02: 1\n\
        x03: 1\n\
        x04: 0\n\
        y00: 1\n\
        y01: 1\n\
        y02: 1\n\
        y03: 1\n\
        y04: 1\n\
        \n\
        ntg XOR fgs -> mjb\n\
        y02 OR x01 -> tnw\n\
        kwq OR kpj -> z05\n\
        x00 OR x03 -> fst\n\
        tgd XOR rvg -> z01\n\
        vdt OR tnw -> bfw\n\
        bfw AND frj -> z10\n\
        ffh OR nrd -> bqk\n\
        y00 AND y03 -> djm\n\
        y03 OR y00 -> psh\n\
        bqk OR frj -> z08\n\
        tnw OR fst -> frj\n\
        gnj AND tgd -> z11\n\
        bfw XOR mjb -> z00\n\
        x03 OR x00 -> vdt\n\
        gnj AND wpb -> z02\n\
        x04 AND y00 -> kjc\n\
        djm OR pbm -> qhw\n\
        nrd AND vdt -> hwm\n\
        kjc AND fst -> rvg\n\
        y04 OR y02 -> fgs\n\
        y01 AND x02 -> pbm\n\
        ntg OR kjc -> kwq\n\
        psh XOR fgs -> tgd\n\
        qhw XOR tgd -> z09\n\
        pbm OR djm -> kpj\n\
        x03 XOR y03 -> ffh\n\
        x00 XOR y04 -> ntg\n\
        bfw OR bqk -> z06\n\
        nrd XOR fgs -> wpb\n\
        frj XOR qhw -> z04\n\
        bqk OR frj -> z07\n\
        y03 OR x01 -> nrd\n\
        hwm AND bqk -> z03\n\
        tgd XOR rvg -> z12\n\
        tnw OR pbm -> gnj\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    wires: vec![
                        Wire {
                            id: "x00".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(true),
                                output_gates: bitarr_typed![GatesBitArray;1,0,0],
                            },
                        },
                        Wire {
                            id: "x01".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(true),
                                output_gates: bitarr_typed![GatesBitArray;0,1,0],
                            },
                        },
                        Wire {
                            id: "x02".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(true),
                                output_gates: bitarr_typed![GatesBitArray;0,0,1],
                            },
                        },
                        Wire {
                            id: "y00".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(false),
                                output_gates: bitarr_typed![GatesBitArray;1,0,0],
                            },
                        },
                        Wire {
                            id: "y01".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(true),
                                output_gates: bitarr_typed![GatesBitArray;0,1,0],
                            },
                        },
                        Wire {
                            id: "y02".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Constant(false),
                                output_gates: bitarr_typed![GatesBitArray;0,0,1],
                            },
                        },
                        Wire {
                            id: "z00".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Gate(0_usize.into()),
                                output_gates: GatesBitArray::ZERO,
                            },
                        },
                        Wire {
                            id: "z01".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Gate(1_usize.into()),
                                output_gates: GatesBitArray::ZERO,
                            },
                        },
                        Wire {
                            id: "z02".try_into().unwrap(),
                            data: WireData {
                                input: WireInput::Gate(2_usize.into()),
                                output_gates: GatesBitArray::ZERO,
                            },
                        },
                    ]
                    .try_into()
                    .unwrap(),
                    gates: vec![
                        Gate {
                            input_wire_a: 0_usize.into(),
                            operation: GateOperation::And,
                            input_wire_b: 3_usize.into(),
                            output_wire: 6_usize.into(),
                        },
                        Gate {
                            input_wire_a: 1_usize.into(),
                            operation: GateOperation::Xor,
                            input_wire_b: 4_usize.into(),
                            output_wire: 7_usize.into(),
                        },
                        Gate {
                            input_wire_a: 2_usize.into(),
                            operation: GateOperation::Or,
                            input_wire_b: 5_usize.into(),
                            output_wire: 8_usize.into(),
                        },
                    ]
                    .try_into()
                    .unwrap(),
                    x: 7_u64,
                    y: 2_u64,
                    x_bits: 3_usize,
                    y_bits: 3_usize,
                    z_bits: 3_usize,
                },
                SOLUTION_STRS[1_usize].try_into().unwrap(),
            ]
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
    fn test_try_compute_final_z_value() {
        for (index, final_z_value) in [Some(4_u64), Some(2024_u64)].into_iter().enumerate() {
            assert_eq!(solution(index).try_compute_final_z_value(), final_z_value);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
