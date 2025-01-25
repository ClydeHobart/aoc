use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while1},
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        mem::take,
        ops::{Add, Mul, Range, Sub},
        str::from_utf8,
    },
};

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, PartialEq)]
enum ModuleType {
    Broadcaster,
    FlipFlop,
    Conjunction,
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
struct Label([u8; Self::LEN]);

impl Label {
    const LEN: usize = 2_usize;
    const BROADCASTER: Self = Self([b'B', b'R']);
    const RECEIVER: Self = Self([b'r', b'x']);

    #[allow(dead_code)]
    fn as_str(&self) -> &str {
        from_utf8(&self.0).unwrap()
    }
}

impl Parse for Label {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(take_while1(|c: char| c.is_ascii_lowercase()), Self::from)(input)
    }
}

impl From<&str> for Label {
    fn from(value: &str) -> Self {
        let mut label: Self = Self::default();
        let len: usize = value.len().min(Self::LEN);

        label.0[..len].copy_from_slice(&value.as_bytes()[..len]);

        label
    }
}

struct ModuleTemp {
    label: Label,
    module_type: Option<ModuleType>,
    inputs: Vec<Label>,
    outputs: Vec<Label>,
}

impl Parse for ModuleTemp {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                alt((
                    map(tag("broadcaster"), |_| {
                        (Label::BROADCASTER, Some(ModuleType::Broadcaster))
                    }),
                    map(preceded(tag("%"), Label::parse), |label| {
                        (label, Some(ModuleType::FlipFlop))
                    }),
                    map(preceded(tag("&"), Label::parse), |label| {
                        (label, Some(ModuleType::Conjunction))
                    }),
                    map(Label::parse, |label| (label, None)),
                )),
                tag(" -> "),
                many0(terminated(Label::parse, opt(tag(", ")))),
            )),
            |((label, module_type), _, outputs)| Self {
                label,
                module_type,
                inputs: Vec::new(),
                outputs,
            },
        )(input)
    }
}

struct ModuleConfigurationTemp(Vec<ModuleTemp>);

impl ModuleConfigurationTemp {
    fn label_to_index_map(&self) -> HashMap<Label, u8> {
        self.0
            .iter()
            .enumerate()
            .map(|(index, module)| (module.label, index as u8))
            .collect()
    }
}

impl Parse for ModuleConfigurationTemp {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            many0(terminated(ModuleTemp::parse, opt(line_ending))),
            |modules| {
                let mut module_configuration_temp: Self = Self(modules);
                let mut label_to_index: HashMap<Label, u8> =
                    module_configuration_temp.label_to_index_map();

                for module_index in 0_usize..module_configuration_temp.0.len() {
                    let label: Label = module_configuration_temp.0[module_index].label;
                    let outputs: Vec<Label> =
                        take(&mut module_configuration_temp.0[module_index].outputs);

                    for output in outputs.iter().copied() {
                        if let Some(output_index) = label_to_index.get(&output) {
                            module_configuration_temp.0[*output_index as usize]
                                .inputs
                                .push(label);
                        } else {
                            let output_index: u8 = module_configuration_temp.0.len() as u8;

                            module_configuration_temp.0.push(ModuleTemp {
                                label: output,
                                module_type: None,
                                inputs: vec![label],
                                outputs: Vec::new(),
                            });
                            label_to_index.insert(output, output_index);
                        }
                    }

                    module_configuration_temp.0[module_index].outputs = outputs;
                }

                if label_to_index.contains_key(&Label::BROADCASTER) {
                    Some(module_configuration_temp)
                } else {
                    None
                }
            },
        )(input)
    }
}

#[allow(dead_code)]
#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct Module {
    label: Label,
    module_type: Option<ModuleType>,
    inputs: Range<u8>,
    outputs: Range<u8>,
}

impl Module {
    fn states_len(&self) -> u8 {
        match self.module_type {
            Some(ModuleType::Broadcaster) => 0_u8,
            Some(ModuleType::FlipFlop) => 1_u8,
            Some(ModuleType::Conjunction) => self.inputs.end - self.inputs.start,
            None => 0_u8,
        }
    }
}

type ModuleStateBitArr = BitArr!(for u8::MAX as usize + 1_usize);

struct ModuleConfigurationState {
    state_ranges: Vec<Range<u8>>,
    states: ModuleStateBitArr,
    received_high: ModuleStateBitArr,
}

impl ModuleConfigurationState {
    fn reset(&mut self) {
        self.states.fill(false);
    }
}

struct Pulse {
    sender: u8,
    receiver: u8,
    is_high: bool,
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Default)]
struct PulseCountsState {
    high: usize,
    low: usize,
}

impl PulseCountsState {
    fn product(&self) -> usize {
        self.high * self.low
    }
}

impl Add for PulseCountsState {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            high: self.high + rhs.high,
            low: self.low + rhs.low,
        }
    }
}

impl Mul<PulseCountsState> for usize {
    type Output = PulseCountsState;

    fn mul(self, rhs: PulseCountsState) -> Self::Output {
        PulseCountsState {
            high: self * rhs.high,
            low: self * rhs.low,
        }
    }
}

impl Sub for PulseCountsState {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            high: self.high - rhs.high,
            low: self.low - rhs.low,
        }
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Debug, Default)]
struct CountsState {
    button_presses: usize,
    pulse_counts: PulseCountsState,
}

impl Add for CountsState {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            button_presses: self.button_presses + rhs.button_presses,
            pulse_counts: self.pulse_counts + rhs.pulse_counts,
        }
    }
}

impl Mul<CountsState> for usize {
    type Output = CountsState;

    fn mul(self, rhs: CountsState) -> Self::Output {
        CountsState {
            button_presses: self * rhs.button_presses,
            pulse_counts: self * rhs.pulse_counts,
        }
    }
}

impl Sub for CountsState {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            button_presses: self.button_presses - rhs.button_presses,
            pulse_counts: self.pulse_counts - rhs.pulse_counts,
        }
    }
}

#[derive(Default)]
struct PulseState {
    queue: VecDeque<Pulse>,
    counts_state: CountsState,
}

impl PulseState {
    fn clear_queue(&mut self) {
        self.queue.clear();
    }

    fn reset(&mut self) {
        self.clear_queue();
        self.counts_state = CountsState::default();
    }

    fn push(&mut self, pulse: Pulse) {
        if pulse.is_high {
            self.counts_state.pulse_counts.high += 1_usize;
        } else {
            self.counts_state.pulse_counts.low += 1_usize;
        }

        self.queue.push_back(pulse);
    }

    fn pop(&mut self) -> Option<Pulse> {
        self.queue.pop_front()
    }
}

struct ModuleStateMap(HashMap<ModuleStateBitArr, CountsState>);

impl ModuleStateMap {
    fn reset(&mut self) {
        self.0.clear();
    }
}

impl Default for ModuleStateMap {
    fn default() -> Self {
        let mut module_state_map: Self = Self(HashMap::new());

        module_state_map.reset();

        module_state_map
    }
}

struct SimulationState {
    module_configuration: ModuleConfigurationState,
    pulse: PulseState,
    module_state_map: ModuleStateMap,
}

impl SimulationState {
    fn reset(&mut self) {
        self.module_configuration.reset();
        self.pulse.reset();
        self.module_state_map.reset();
    }
}

#[allow(dead_code)]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct CycleReport {
    button_presses_at_cycle_found: usize,
    cycle_start: CountsState,
}

#[allow(dead_code)]
#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct Report {
    final_pulse_counts: PulseCountsState,
    cycle_report: Option<CycleReport>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct ModuleConfiguration {
    modules: Vec<Module>,
    neighbors: Vec<u8>,
    broadcaster: u8,
    receiver: Option<u8>,
}

impl ModuleConfiguration {
    const WARMUP_BUTTON_PRESSES: usize = 1000_usize;

    fn module_configuration_state(&self) -> ModuleConfigurationState {
        let mut start: u8 = 0_u8;
        let state_ranges: Vec<Range<u8>> = self
            .modules
            .iter()
            .map(|module| {
                let end: u8 = start + module.states_len();
                let range: Range<u8> = start..end;

                start = end;

                range
            })
            .collect();

        ModuleConfigurationState {
            state_ranges,
            states: Default::default(),
            received_high: Default::default(),
        }
    }

    fn simulation_state(&self) -> SimulationState {
        SimulationState {
            module_configuration: self.module_configuration_state(),
            pulse: PulseState::default(),
            module_state_map: ModuleStateMap::default(),
        }
    }

    fn button_pulse(&self) -> Pulse {
        Pulse {
            sender: u8::MAX,
            receiver: self.broadcaster,
            is_high: false,
        }
    }

    fn send_pulse(&self, state: &mut SimulationState, sender: u8, is_high: bool) {
        for receiver in self.neighbors[self.modules[sender as usize].outputs.as_range_usize()]
            .iter()
            .copied()
        {
            state.pulse.push(Pulse {
                sender,
                receiver,
                is_high,
            });
        }
    }

    fn press_button(&self, state: &mut SimulationState) {
        state.pulse.counts_state.button_presses += 1_usize;
        state.pulse.clear_queue();
        state.pulse.push(self.button_pulse());

        while let Some(pulse) = state.pulse.pop() {
            let module: &Module = &self.modules[pulse.receiver as usize];

            if let Some(module_type) = module.module_type {
                match module_type {
                    ModuleType::Broadcaster => {
                        let sender: u8 = pulse.receiver;
                        let is_high: bool = pulse.is_high;

                        self.send_pulse(state, sender, is_high)
                    }
                    ModuleType::FlipFlop => {
                        if !pulse.is_high {
                            let sender: u8 = pulse.receiver;
                            let state_index: usize =
                                state.module_configuration.state_ranges[sender as usize].start
                                    as usize;
                            let is_high: bool = !state.module_configuration.states[state_index];
                            state.module_configuration.states.set(state_index, is_high);

                            if is_high {
                                state
                                    .module_configuration
                                    .received_high
                                    .set(state_index, true);
                            }

                            self.send_pulse(state, sender, is_high);
                        }
                    }
                    ModuleType::Conjunction => {
                        let receiver: usize = pulse.receiver as usize;
                        let states_range: Range<usize> =
                            state.module_configuration.state_ranges[receiver].as_range_usize();
                        let states = &mut state.module_configuration.states[states_range.clone()];
                        let sender_state_index: usize = self.neighbors
                            [self.modules[receiver].inputs.as_range_usize()]
                        .iter()
                        .copied()
                        .position(|input| input == pulse.sender)
                        .unwrap();

                        states.set(sender_state_index, pulse.is_high);

                        if pulse.is_high {
                            state
                                .module_configuration
                                .received_high
                                .set(states_range.start + sender_state_index, true);
                        }

                        let sender: u8 = pulse.receiver as u8;
                        let is_high: bool = !states.all();

                        self.send_pulse(state, sender, is_high);
                    }
                }
            }
        }
    }

    fn find_cycle(&self, state: &mut SimulationState, button_presses: usize) -> bool {
        state.reset();

        let mut completed_all_button_presses: bool = button_presses == 0_usize;

        while !state
            .module_state_map
            .0
            .contains_key(&state.module_configuration.states)
            && !completed_all_button_presses
        {
            state.module_state_map.0.insert(
                state.module_configuration.states.clone(),
                state.pulse.counts_state.clone(),
            );
            self.press_button(state);
            completed_all_button_presses =
                state.pulse.counts_state.button_presses == button_presses;
        }

        !completed_all_button_presses
    }

    fn report_for_button_presses(&self, button_presses: usize) -> Report {
        let mut state: SimulationState = self.simulation_state();

        if self.find_cycle(&mut state, button_presses) {
            let counts_state: CountsState = state.pulse.counts_state;
            let button_presses_at_cycle_found: usize = counts_state.button_presses;
            let cycle_start: CountsState = state
                .module_state_map
                .0
                .get(&state.module_configuration.states)
                .unwrap()
                .clone();
            let counts_state_diff: CountsState = counts_state.clone() - cycle_start.clone();
            let remaining_button_presses: usize = button_presses - button_presses_at_cycle_found;
            let remaining_cycles: usize =
                remaining_button_presses / counts_state_diff.button_presses;

            state.pulse.counts_state = counts_state + remaining_cycles * counts_state_diff;

            while state.pulse.counts_state.button_presses < button_presses {
                self.press_button(&mut state);
            }

            Report {
                final_pulse_counts: state.pulse.counts_state.pulse_counts,
                cycle_report: Some(CycleReport {
                    button_presses_at_cycle_found,
                    cycle_start,
                }),
            }
        } else {
            Report {
                final_pulse_counts: state.pulse.counts_state.pulse_counts,
                cycle_report: None,
            }
        }
    }

    fn warmup_report(&self) -> Report {
        self.report_for_button_presses(Self::WARMUP_BUTTON_PRESSES)
    }

    fn warmup_pulse_count_product(&self) -> usize {
        self.warmup_report().final_pulse_counts.product()
    }

    fn parents(&self, module: u8) -> &[u8] {
        &self.neighbors[self.modules[module as usize].inputs.as_range_usize()]
    }

    fn module_is_conjunction(&self, module: u8) -> bool {
        self.modules[module as usize].module_type == Some(ModuleType::Conjunction)
    }

    fn parents_are_conjunctions(&self, module: u8) -> bool {
        self.parents(module)
            .iter()
            .all(|parent| self.module_is_conjunction(*parent))
    }

    fn grandparents_are_conjunctions(&self, module: u8) -> bool {
        self.parents(module)
            .iter()
            .all(|parent| self.parents_are_conjunctions(*parent))
    }

    fn great_grandparents_are_conjunctions(&self, module: u8) -> bool {
        self.parents(module)
            .iter()
            .all(|parent| self.grandparents_are_conjunctions(*parent))
    }

    fn state_indices_to_watch(&self, state: &SimulationState) -> Option<ModuleStateBitArr> {
        self.receiver
            .filter(|receiver| {
                self.parents_are_conjunctions(*receiver)
                    && self.grandparents_are_conjunctions(*receiver)
                    && self.great_grandparents_are_conjunctions(*receiver)
            })
            .map(|receiver| {
                let mut state_indices_to_watch: ModuleStateBitArr = ModuleStateBitArr::default();

                for state_index in self.parents(receiver).iter().flat_map(|parent| {
                    state.module_configuration.state_ranges[*parent as usize].as_range_usize()
                }) {
                    state_indices_to_watch.set(state_index, true);
                }

                state_indices_to_watch
            })
    }

    fn button_presses_until_machine_turned_on(&self, verbose: bool) -> Option<usize> {
        let mut state: SimulationState = self.simulation_state();

        if let Some(state_indices_to_watch) = self.state_indices_to_watch(&state) {
            let mut state_indices_to_watch: Vec<(usize, bool)> = state_indices_to_watch
                .iter_ones()
                .map(|state_index| (state_index, true))
                .collect();
            let mut state_indices_to_watch_count: usize = state_indices_to_watch.len();
            let mut prime_factorization: HashMap<u32, u32> = HashMap::new();

            while state_indices_to_watch_count > 0_usize {
                self.press_button(&mut state);

                let button_presses: usize = state.pulse.counts_state.button_presses;

                if verbose && button_presses % 10_usize == 0_usize {
                    dbg!(button_presses);
                }

                for (state_index, should_watch) in state_indices_to_watch.iter_mut() {
                    if *should_watch && state.module_configuration.received_high[*state_index] {
                        *should_watch = false;
                        state_indices_to_watch_count -= 1_usize;

                        for prime_factor in iter_prime_factors(button_presses) {
                            if let Some(exponent) =
                                prime_factorization.get_mut(&(prime_factor.prime as u32))
                            {
                                *exponent = (*exponent).max(prime_factor.exponent as u32);
                            } else {
                                prime_factorization.insert(
                                    prime_factor.prime as u32,
                                    prime_factor.exponent as u32,
                                );
                            }
                        }
                    }
                }
            }

            Some(
                prime_factorization
                    .into_iter()
                    .map(|(prime, exponent)| (prime as usize).pow(exponent))
                    .product(),
            )
        } else {
            None
        }
    }
}

impl From<ModuleConfigurationTemp> for ModuleConfiguration {
    fn from(value: ModuleConfigurationTemp) -> Self {
        let mut modules: Vec<Module> = value
            .0
            .iter()
            .map(|module_temp| Module {
                label: module_temp.label,
                module_type: module_temp.module_type,
                inputs: 0..0,
                outputs: 0..0,
            })
            .collect();
        let mut neighbors: Vec<u8> = Vec::new();
        let label_to_index: HashMap<Label, u8> = value.label_to_index_map();
        let broadcaster: u8 = *label_to_index.get(&Label::BROADCASTER).unwrap();
        let receiver: Option<u8> = label_to_index.get(&Label::RECEIVER).copied();
        let mut append_neighbors = |neighbor_labels: &[Label]| -> Range<u8> {
            let start: u8 = neighbors.len() as u8;

            for label in neighbor_labels.iter() {
                neighbors.push(*label_to_index.get(label).unwrap());
            }

            let end: u8 = neighbors.len() as u8;

            start..end
        };

        for (module, module_temp) in modules.iter_mut().zip(value.0.into_iter()) {
            module.inputs = append_neighbors(&module_temp.inputs);
            module.outputs = append_neighbors(&module_temp.outputs);
        }

        Self {
            modules,
            neighbors,
            broadcaster,
            receiver,
        }
    }
}

impl Parse for ModuleConfiguration {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(ModuleConfigurationTemp::parse, Self::from)(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(ModuleConfiguration);

impl Solution {
    fn warmup_report(&self) -> Report {
        self.0.warmup_report()
    }

    fn warmup_pulse_count_product(&self) -> usize {
        self.0.warmup_pulse_count_product()
    }

    fn button_presses_until_machine_turned_on(&self, verbose: bool) -> Option<usize> {
        self.0.button_presses_until_machine_turned_on(verbose)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(ModuleConfiguration::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let report: Report = self.warmup_report();

            dbg!(&report);
            dbg!(report.final_pulse_counts.product());
        } else {
            dbg!(self.warmup_pulse_count_product());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.button_presses_until_machine_turned_on(args.verbose));
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
        broadcaster -> a, b, c\n\
        %a -> b\n\
        %b -> c\n\
        %c -> inv\n\
        &inv -> a\n",
        "\
        broadcaster -> a\n\
        %a -> inv, con\n\
        &inv -> b\n\
        %b -> con\n\
        &con -> output\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        const B: Option<ModuleType> = Some(ModuleType::Broadcaster);
        const F: Option<ModuleType> = Some(ModuleType::FlipFlop);
        const C: Option<ModuleType> = Some(ModuleType::Conjunction);

        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! modules {
            [ $( $label:expr, $module_type:expr, $inputs:expr, $outputs:expr; )* ] => { vec![ $(
                Module {
                    label: $label.into(),
                    module_type: $module_type,
                    inputs: $inputs,
                    outputs: $outputs,
                },
            )* ] }
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(ModuleConfiguration {
                    modules: modules![
                        "BR", B, 0..0, 0..3; // 0
                        "a", F, 3..5, 5..6; // 1
                        "b", F, 6..8, 8..9; // 2
                        "c", F, 9..11, 11..12; // 3
                        "inv", C, 12..13, 13..14; // 4
                    ],
                    neighbors: vec![
                        1, 2, 3, // broadcaster out
                        0, 4, // a in
                        2, // a out
                        0, 1, // b in
                        3, // b out
                        0, 2, // c in
                        4, // c out
                        3, // inv in
                        1, // inv out
                    ],
                    broadcaster: 0,
                    receiver: None,
                }),
                Solution(ModuleConfiguration {
                    // broadcaster -> a
                    // %a -> inv, con
                    // &inv -> b
                    // %b -> con
                    // &con -> output
                    modules: modules![
                        "BR", B, 0..0, 0..1; // 0
                        "a", F, 1..2, 2..4; // 1
                        "inv", C, 4..5, 5..6; // 2
                        "b", F, 6..7, 7..8; // 3
                        "con", C, 8..10, 10..11; // 4
                        "output", None, 11..12, 12..12; // 5
                    ],
                    neighbors: vec![
                        1, // broadcaster in
                        0, // a in
                        2, 4, // a out
                        1, // inv in
                        3, // inv out
                        2, // b in
                        4, // b out,
                        1, 3, // con in
                        5, // con out
                        4, // output in
                    ],
                    broadcaster: 0,
                    receiver: None,
                }),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_warmup_report() {
        for (index, report) in [
            Report {
                final_pulse_counts: PulseCountsState {
                    high: 4000_usize,
                    low: 8000_usize,
                },
                cycle_report: Some(CycleReport {
                    button_presses_at_cycle_found: 1_usize,
                    cycle_start: CountsState::default(),
                }),
            },
            Report {
                final_pulse_counts: PulseCountsState {
                    high: 2750_usize,
                    low: 4250_usize,
                },
                cycle_report: Some(CycleReport {
                    button_presses_at_cycle_found: 4_usize,
                    cycle_start: CountsState::default(),
                }),
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).warmup_report(), report);
        }
    }
}
