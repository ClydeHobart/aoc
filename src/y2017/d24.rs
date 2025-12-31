use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::{
        cmp::Ordering,
        collections::{HashMap, VecDeque},
        fmt::{Debug, Formatter, Result as FmtResult, Write},
        hash::{Hash, Hasher},
    },
};

/* --- Day 24: Electromagnetic Moat ---

The CPU itself is a large, black building surrounded by a bottomless pit. Enormous metal tubes extend outward from the side of the building at regular intervals and descend down into the void. There's no way to cross, but you need to get inside.

No way, of course, other than building a bridge out of the magnetic components strewn about nearby.

Each component has two ports, one on each end. The ports come in all different types, and only matching types can be connected. You take an inventory of the components by their port types (your puzzle input). Each port is identified by the number of pins it uses; more pins mean a stronger connection for your bridge. A 3/7 component, for example, has a type-3 port on one side, and a type-7 port on the other.

Your side of the pit is metallic; a perfect surface to connect a magnetic, zero-pin port. Because of this, the first port you use must be of type 0. It doesn't matter what type of port you end with; your goal is just to make the bridge as strong as possible.

The strength of a bridge is the sum of the port types in each component. For example, if your bridge is made of components 0/3, 3/7, and 7/4, your bridge has a strength of 0+3 + 3+7 + 7+4 = 24.

For example, suppose you had the following components:

0/2
2/2
2/3
3/4
3/5
0/1
10/1
9/10

With them, you could make the following valid bridges:

    0/1
    0/1--10/1
    0/1--10/1--9/10
    0/2
    0/2--2/3
    0/2--2/3--3/4
    0/2--2/3--3/5
    0/2--2/2
    0/2--2/2--2/3
    0/2--2/2--2/3--3/4
    0/2--2/2--2/3--3/5

(Note how, as shown by 10/1, order of ports within a component doesn't matter. However, you may only use each port on a component once.)

Of these bridges, the strongest one is 0/1--10/1--9/10; it has a strength of 0+1 + 1+10 + 10+9 = 31.

What is the strength of the strongest bridge you can make with the components you have available?

--- Part Two ---

The bridge you've built isn't long enough; you can't jump the rest of the way.

In the example above, there are two longest bridges:

    0/2--2/2--2/3--3/4
    0/2--2/2--2/3--3/5

Of them, the one which uses the 3/5 component is stronger; its strength is 0+2 + 2+2 + 2+3 + 3+5 = 19.

What is the strength of the longest bridge you can make? If you can make multiple bridges of the longest length, pick the strongest one. */

#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Port {
    pins: u8,
}

impl Port {
    const MAX_PINS: u8 = Solution::PORT_TO_COMPONENTS_LEN as u8 - 1_u8;
    const ZERO: Self = Self { pins: 0_u8 };

    fn try_new(pins: u8) -> Option<Self> {
        (pins <= Self::MAX_PINS).then_some(Self { pins })
    }

    fn as_usize(self) -> usize {
        self.pins as usize
    }
}

impl Debug for Port {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.pins.fmt(f)
    }
}

impl Parse for Port {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(parse_integer, Self::try_new)(input)
    }
}

#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Component {
    ports: [Port; Self::PORTS_LEN],
}

impl Component {
    const PORTS_LEN: usize = 2_usize;

    fn new((port_0, port_1): (Port, Port)) -> Self {
        let mut component: Self = Self {
            ports: [port_0, port_1],
        };

        component.ports.sort();

        component
    }

    fn other_port(self, port: Port) -> Port {
        self.ports
            .into_iter()
            .find(|self_port| *self_port != port)
            .unwrap_or(port)
    }

    fn strength(self) -> u32 {
        self.ports.into_iter().map(|port| port.pins as u32).sum()
    }
}

impl Debug for Component {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.ports
            .into_iter()
            .enumerate()
            .try_for_each(|(port_index, port)| {
                if port_index != 0_usize {
                    f.write_char('/')?;
                }

                port.fmt(f)
            })
    }
}

impl Parse for Component {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(Port::parse, tag("/"), Port::parse),
            Self::new,
        )(input)
    }
}

type ComponentIndexRaw = u8;
type ComponentIndex = Index<ComponentIndexRaw>;
type ComponentList = IdList<Component, ComponentIndexRaw>;
type ComponentBitArr = BitArr!(for Solution::MAX_COMPONENTS_LEN, in u32);

#[derive(Clone, Debug, Eq)]
struct BridgeStateEndComponent(ComponentIndex);

impl PartialEq for BridgeStateEndComponent {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Hash for BridgeStateEndComponent {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

#[derive(Clone, Eq, Debug, Hash, PartialEq)]
struct BridgeState {
    components: ComponentBitArr,
    strength: u32,
    end_port: Port,
    end_component: BridgeStateEndComponent,
}

impl BridgeState {
    const START: Self = Self {
        components: BitArray::ZERO,
        strength: 0_u32,
        end_port: Port::ZERO,
        end_component: BridgeStateEndComponent(ComponentIndex::INVALID),
    };

    fn strength(&self) -> u32 {
        self.strength
    }

    fn length(&self) -> u32 {
        self.components.count_ones() as u32
    }

    fn cmp_stronger(&self, other: &Self) -> Ordering {
        self.strength().cmp(&other.strength())
    }

    fn cmp_longer(&self, other: &Self) -> Ordering {
        self.length()
            .cmp(&other.length())
            .then_with(|| self.cmp_stronger(other))
    }
}

type PortToComponents = [ComponentBitArr; Solution::PORT_TO_COMPONENTS_LEN];

struct OptimalBridgeFinder<'s> {
    solution: &'s Solution,
    child_to_parent_end_component: HashMap<BridgeState, ComponentIndex>,
    strongest_bridge: BridgeState,
    longest_bridge: BridgeState,
}

impl<'s> BreadthFirstSearch for OptimalBridgeFinder<'s> {
    type Vertex = BridgeState;

    fn start(&self) -> &Self::Vertex {
        &Self::Vertex::START
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<Self::Vertex> = VecDeque::new();
        let mut curr_vertex: Self::Vertex = vertex.clone();

        while {
            path.push_front(curr_vertex.clone());

            curr_vertex != *self.start()
        } {
            let mut prev_vertex: Self::Vertex = curr_vertex.clone();

            prev_vertex
                .components
                .set(curr_vertex.end_component.0.get(), false);
            prev_vertex.end_component = BridgeStateEndComponent(
                self.child_to_parent_end_component
                    .get(&curr_vertex)
                    .unwrap()
                    .clone(),
            );

            let end_component: Component =
                self.solution.components.as_id_slice()[curr_vertex.end_component.0.get()];

            prev_vertex.end_port = end_component.other_port(curr_vertex.end_port);
            prev_vertex.strength -= end_component.strength();
            curr_vertex = prev_vertex;
        }

        path.into()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        neighbors.extend(
            (self.solution.port_to_components[vertex.end_port.as_usize()] & !vertex.components)
                .iter_ones()
                .map(|component_index| {
                    let mut next_vertex: Self::Vertex = vertex.clone();

                    next_vertex.components.set(component_index, true);
                    next_vertex.end_component = BridgeStateEndComponent(component_index.into());

                    let component: Component =
                        self.solution.components.as_id_slice()[component_index];

                    next_vertex.strength += component.strength();
                    next_vertex.end_port = component.other_port(vertex.end_port);

                    next_vertex
                }),
        );
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        if to.cmp_stronger(&self.strongest_bridge).is_gt() {
            self.strongest_bridge = to.clone();
        }

        if to.cmp_longer(&self.longest_bridge).is_gt() {
            self.longest_bridge = to.clone();
        }

        self.child_to_parent_end_component
            .insert(to.clone(), from.end_component.0);
    }

    fn reset(&mut self) {
        self.child_to_parent_end_component.clear();
        self.strongest_bridge = self.start().clone();
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    components: ComponentList,
    port_to_components: PortToComponents,
}

impl Solution {
    const MAX_COMPONENTS_LEN: usize = u64::BITS as usize;
    const PORT_TO_COMPONENTS_LEN: usize = 64_usize;

    fn optimal_bridge_property<F: Fn(&BridgeState) -> u32>(
        bridge_state_path: &[BridgeState],
        f: F,
    ) -> u32 {
        f(bridge_state_path.last().unwrap())
    }

    fn run_optimal_bridge_finder<'s>(&'s self) -> OptimalBridgeFinder<'s> {
        let mut optimal_bridge_finder: OptimalBridgeFinder = OptimalBridgeFinder {
            solution: self,
            child_to_parent_end_component: HashMap::new(),
            strongest_bridge: BridgeState::START,
            longest_bridge: BridgeState::START,
        };

        optimal_bridge_finder.run();

        optimal_bridge_finder
    }

    fn strongest_bridge_state_path(&self) -> Vec<BridgeState> {
        let optimal_bridge_finder: OptimalBridgeFinder = self.run_optimal_bridge_finder();

        optimal_bridge_finder.path_to(&optimal_bridge_finder.strongest_bridge)
    }

    fn longest_bridge_state_path(&self) -> Vec<BridgeState> {
        let optimal_bridge_finder: OptimalBridgeFinder = self.run_optimal_bridge_finder();

        optimal_bridge_finder.path_to(&optimal_bridge_finder.longest_bridge)
    }

    fn strongest_bridge_strength(&self) -> u32 {
        Self::optimal_bridge_property(&self.strongest_bridge_state_path(), BridgeState::strength)
    }

    fn longest_bridge_strength(&self) -> u32 {
        Self::optimal_bridge_property(&self.longest_bridge_state_path(), BridgeState::strength)
    }

    fn component_path_from_bridge_state_path(
        &self,
        bridge_state_path: &[BridgeState],
    ) -> Vec<Component> {
        bridge_state_path
            .iter()
            .filter_map(|bridge_state| bridge_state.end_component.0.opt())
            .map(|component_index| self.components.as_id_slice()[component_index.get()])
            .collect()
    }

    fn optimal_bridge_component_path_and_property<F: Fn(&BridgeState) -> u32>(
        &self,
        bridge_state_path: &[BridgeState],
        f: F,
    ) -> (Vec<Component>, u32) {
        (
            self.component_path_from_bridge_state_path(bridge_state_path),
            Self::optimal_bridge_property(bridge_state_path, f),
        )
    }

    fn strongest_bridge_component_path_and_strength(&self) -> (Vec<Component>, u32) {
        self.optimal_bridge_component_path_and_property(
            &self.strongest_bridge_state_path(),
            BridgeState::strength,
        )
    }

    fn longest_bridge_component_path_and_strength(&self) -> (Vec<Component>, u32) {
        self.optimal_bridge_component_path_and_property(
            &self.longest_bridge_state_path(),
            BridgeState::strength,
        )
    }
}

impl From<ComponentList> for Solution {
    fn from(components: ComponentList) -> Self {
        let mut port_to_components: PortToComponents = PortToComponents::large_array_default();

        for (component_index, component) in components.as_id_slice().iter().enumerate() {
            for port in component.ports {
                port_to_components[port.as_usize()].set(component_index, true);
            }
        }

        Self {
            components,
            port_to_components,
        }
    }
}

impl TryFrom<Vec<Component>> for Solution {
    type Error = Box<String>;

    fn try_from(value: Vec<Component>) -> Result<Self, Self::Error> {
        ComponentList::try_from(value).map(Self::from)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_res(
            many0(terminated(Component::parse, opt(line_ending))),
            Self::try_from,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// This was cool. It was interesting creating a type with a non-trivial `PartialEq` and `Hash`
    /// implementation. I think part 2 will point out that after the first component, components
    /// need to end with at least 1 pin facing outward?
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.strongest_bridge_component_path_and_strength());
        } else {
            dbg!(self.strongest_bridge_strength());
        }
    }

    /// I misread this initially, I thought it was asking for the length of the longest bridge,
    /// which is why everything is templated to fetch the length property.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.longest_bridge_component_path_and_strength());
        } else {
            dbg!(self.longest_bridge_strength());
        }
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
        0/2\n\
        2/2\n\
        2/3\n\
        3/4\n\
        3/5\n\
        0/1\n\
        10/1\n\
        9/10\n"];

    macro_rules! components {
        [ $( ( $port_0:expr, $port_1:expr ), )* ] => {
            vec![ $( Component::new((
                Port::try_new($port_0).unwrap(),
                Port::try_new($port_1).unwrap()
            )), )* ]
        }
    }

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![{
                let components: ComponentList = components![
                    (0_u8, 1_u8),
                    (0_u8, 2_u8),
                    (1_u8, 10_u8),
                    (2_u8, 2_u8),
                    (2_u8, 3_u8),
                    (3_u8, 4_u8),
                    (3_u8, 5_u8),
                    (9_u8, 10_u8),
                ]
                .try_into()
                .unwrap();

                let mut port_to_components: PortToComponents =
                    PortToComponents::large_array_default();

                for (port, components) in [
                    (0_usize, &[0_usize, 1_usize][..]),
                    (1_usize, &[0_usize, 2_usize]),
                    (2_usize, &[1_usize, 3_usize, 4_usize]),
                    (3_usize, &[4_usize, 5_usize, 6_usize]),
                    (4_usize, &[5_usize]),
                    (5_usize, &[6_usize]),
                    (9_usize, &[7_usize]),
                    (10_usize, &[2_usize, 7_usize]),
                ] {
                    for component in components {
                        port_to_components[port].set(*component, true);
                    }
                }

                Solution {
                    components,
                    port_to_components,
                }
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
    fn test_strongest_bridge_component_path_and_strength() {
        for (index, strongest_bridge_component_path_and_strength) in [(
            components![(0_u8, 1_u8), (1_u8, 10_u8), (9_u8, 10_u8),],
            31_u32,
        )]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).strongest_bridge_component_path_and_strength(),
                strongest_bridge_component_path_and_strength
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
