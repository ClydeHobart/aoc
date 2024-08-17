use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{alpha1, line_ending},
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many1_count,
        sequence::{delimited, tuple},
        Err, IResult,
    },
    std::{
        collections::{HashMap, VecDeque},
        fmt::{Display, Formatter, Result as FmtResult},
        ops::Range,
    },
};

/* --- Day 11: Radioisotope Thermoelectric Generators ---

You come upon a column of four floors that have been entirely sealed off from the rest of the building except for a small dedicated lobby. There are some radiation warnings and a big sign which reads "Radioisotope Testing Facility".

According to the project status board, this facility is currently being used to experiment with Radioisotope Thermoelectric Generators (RTGs, or simply "generators") that are designed to be paired with specially-constructed microchips. Basically, an RTG is a highly radioactive rock that generates electricity through heat.

The experimental RTGs have poor radiation containment, so they're dangerously radioactive. The chips are prototypes and don't have normal radiation shielding, but they do have the ability to generate an electromagnetic radiation shield when powered. Unfortunately, they can only be powered by their corresponding RTG. An RTG powering a microchip is still dangerous to other microchips.

In other words, if a chip is ever left in the same area as another RTG, and it's not connected to its own RTG, the chip will be fried. Therefore, it is assumed that you will follow procedure and keep chips connected to their corresponding RTG when they're in the same room, and away from other RTGs otherwise.

These microchips sound very interesting and useful to your current activities, and you'd like to try to retrieve them. The fourth floor of the facility has an assembling machine which can make a self-contained, shielded computer for you to take with you - that is, if you can bring it all of the RTGs and microchips.

Within the radiation-shielded part of the facility (in which it's safe to have these pre-assembly RTGs), there is an elevator that can move between the four floors. Its capacity rating means it can carry at most yourself and two RTGs or microchips in any combination. (They're rigged to some heavy diagnostic equipment - the assembling machine will detach it for you.) As a security measure, the elevator will only function if it contains at least one RTG or microchip. The elevator always stops on each floor to recharge, and this takes long enough that the items within it and the items on that floor can irradiate each other. (You can prevent this if a Microchip and its Generator end up on the same floor in this way, as they can be connected while the elevator is recharging.)

You make some notes of the locations of each component of interest (your puzzle input). Before you don a hazmat suit and start moving things around, you'd like to have an idea of what you need to do.

When you enter the containment area, you and the elevator will start on the first floor.

For example, suppose the isolated area has the following arrangement:

The first floor contains a hydrogen-compatible microchip and a lithium-compatible microchip.
The second floor contains a hydrogen generator.
The third floor contains a lithium generator.
The fourth floor contains nothing relevant.

As a diagram (F# for a Floor number, E for Elevator, H for Hydrogen, L for Lithium, M for Microchip, and G for Generator), the initial state looks like this:

F4 .  .  .  .  .
F3 .  .  .  LG .
F2 .  HG .  .  .
F1 E  .  HM .  LM

Then, to get everything up to the assembling machine on the fourth floor, the following steps could be taken:

    Bring the Hydrogen-compatible Microchip to the second floor, which is safe because it can get power from the Hydrogen Generator:

    F4 .  .  .  .  .
    F3 .  .  .  LG .
    F2 E  HG HM .  .
    F1 .  .  .  .  LM

    Bring both Hydrogen-related items to the third floor, which is safe because the Hydrogen-compatible microchip is getting power from its generator:

    F4 .  .  .  .  .
    F3 E  HG HM LG .
    F2 .  .  .  .  .
    F1 .  .  .  .  LM

    Leave the Hydrogen Generator on floor three, but bring the Hydrogen-compatible Microchip back down with you so you can still use the elevator:

    F4 .  .  .  .  .
    F3 .  HG .  LG .
    F2 E  .  HM .  .
    F1 .  .  .  .  LM

    At the first floor, grab the Lithium-compatible Microchip, which is safe because Microchips don't affect each other:

    F4 .  .  .  .  .
    F3 .  HG .  LG .
    F2 .  .  .  .  .
    F1 E  .  HM .  LM

    Bring both Microchips up one floor, where there is nothing to fry them:

    F4 .  .  .  .  .
    F3 .  HG .  LG .
    F2 E  .  HM .  LM
    F1 .  .  .  .  .

    Bring both Microchips up again to floor three, where they can be temporarily connected to their corresponding generators while the elevator recharges, preventing either of them from being fried:

    F4 .  .  .  .  .
    F3 E  HG HM LG LM
    F2 .  .  .  .  .
    F1 .  .  .  .  .

    Bring both Microchips to the fourth floor:

    F4 E  .  HM .  LM
    F3 .  HG .  LG .
    F2 .  .  .  .  .
    F1 .  .  .  .  .

    Leave the Lithium-compatible microchip on the fourth floor, but bring the Hydrogen-compatible one so you can still use the elevator; this is safe because although the Lithium Generator is on the destination floor, you can connect Hydrogen-compatible microchip to the Hydrogen Generator there:

    F4 .  .  .  .  LM
    F3 E  HG HM LG .
    F2 .  .  .  .  .
    F1 .  .  .  .  .

    Bring both Generators up to the fourth floor, which is safe because you can connect the Lithium-compatible Microchip to the Lithium Generator upon arrival:

    F4 E  HG .  LG LM
    F3 .  .  HM .  .
    F2 .  .  .  .  .
    F1 .  .  .  .  .

    Bring the Lithium Microchip with you to the third floor so you can use the elevator:

    F4 .  HG .  LG .
    F3 E  .  HM .  LM
    F2 .  .  .  .  .
    F1 .  .  .  .  .

    Bring both Microchips to the fourth floor:

    F4 E  HG HM LG LM
    F3 .  .  .  .  .
    F2 .  .  .  .  .
    F1 .  .  .  .  .

In this arrangement, it takes 11 steps to collect all of the objects at the fourth floor for assembly. (Each elevator stop counts as one step, even if nothing is added to or removed from it.)

In your situation, what is the minimum number of steps required to bring all of the objects to the fourth floor?

--- Part Two ---

You step into the cleanroom separating the lobby from the isolated area and put on the hazmat suit.

Upon entering the isolated containment area, however, you notice some extra parts on the first floor that weren't listed on the record outside:

    An elerium generator.
    An elerium-compatible microchip.
    A dilithium generator.
    A dilithium-compatible microchip.

These work just like the other generators and microchips. You'll have to get them up to assembly as well.

What is the minimum number of steps required to bring all of the objects, including these four new ones, to the fourth floor? */

fn parse_tagged_str_wrapper<'i, T, F: Fn(&'i str) -> T>(
    tag_str: &'static str,
    f: F,
) -> impl FnMut(&'i str) -> IResult<&'i str, T> {
    map(delimited(tag("a "), alpha1, tag(tag_str)), f)
}

struct Generator<'s>(&'s str);

impl<'s> Generator<'s> {
    fn parse(input: &'s str) -> IResult<&'s str, Self> {
        parse_tagged_str_wrapper(" generator", Self)(input)
    }
}

struct Microchip<'s>(&'s str);

impl<'s> Microchip<'s> {
    fn parse(input: &'s str) -> IResult<&'s str, Self> {
        parse_tagged_str_wrapper("-compatible microchip", Self)(input)
    }
}

enum FloorElement<'s> {
    Generator(Generator<'s>),
    Microchip(Microchip<'s>),
}

impl<'s> FloorElement<'s> {
    fn parse(input: &'s str) -> IResult<&'s str, Self> {
        alt((
            map(Generator::parse, Self::Generator),
            map(Microchip::parse, Self::Microchip),
        ))(input)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct RadioisotopeState(u8);

impl RadioisotopeState {
    const INVALID: Self = Self(u8::MAX);
    const INVALID_FLOOR: u32 = 0xF_u32;
    const MIN_FLOOR: u32 = 0_u32;
    const MAX_FLOOR: u32 = 3_u32;
    const GENERATOR: Range<usize> = Self::MICROCHIP.end..u8::BITS as usize;
    const MICROCHIP: Range<usize> = 0_usize..((u8::BITS >> 1_u32) as usize);

    #[cfg(test)]
    fn new(generator: u32, microchip: u32) -> Self {
        let mut radioisotope_state: Self = Self::INVALID;

        radioisotope_state.set_generator_floor(generator);
        radioisotope_state.set_microchip_floor(microchip);

        radioisotope_state
    }

    fn new_elevator(elevator_floor: u32) -> Self {
        let mut radioisotope_state: Self = Self::INVALID;

        radioisotope_state.set_elevator_floor(elevator_floor);

        radioisotope_state
    }

    fn iter_neighboring_floors(floor: u32) -> impl Iterator<Item = u32> {
        (floor > Self::MIN_FLOOR)
            .then(|| floor - 1_u32)
            .into_iter()
            .chain((floor < Self::MAX_FLOOR).then(|| floor + 1_u32))
    }

    fn bits(&self) -> &BitSlice<u8, Lsb0> {
        self.0.view_bits()
    }

    fn get_generator_floor(self) -> u32 {
        self.bits()[Self::GENERATOR].load()
    }

    fn is_generator_valid(self) -> bool {
        self.get_generator_floor() != Self::INVALID_FLOOR
    }

    fn get_microchip_floor(self) -> u32 {
        if self.is_generator_valid() {
            self.bits()[Self::MICROCHIP].load()
        } else {
            Self::INVALID_FLOOR
        }
    }

    fn get_elevator_floor(self) -> u32 {
        if !self.is_generator_valid() {
            self.bits()[Self::MICROCHIP].load()
        } else {
            Self::INVALID_FLOOR
        }
    }

    fn try_get_floor<F: Fn(Self) -> u32>(self, f: F) -> Option<u32> {
        let floor: u32 = f(self);

        (floor != Self::INVALID_FLOOR).then_some(floor)
    }

    fn try_get_microchip_floor(self) -> Option<u32> {
        self.try_get_floor(Self::get_microchip_floor)
    }

    fn try_get_generator_floor(self) -> Option<u32> {
        self.try_get_floor(Self::get_generator_floor)
    }

    fn try_get_elevator_floor(self) -> Option<u32> {
        self.try_get_floor(Self::get_elevator_floor)
    }

    fn bits_mut(&mut self) -> &mut BitSlice<u8, Lsb0> {
        self.0.view_bits_mut()
    }

    fn set_generator_floor(&mut self, generator_floor: u32) {
        self.bits_mut()[Self::GENERATOR].store(generator_floor);
    }

    fn set_microchip_floor(&mut self, microchip_floor: u32) {
        self.bits_mut()[Self::MICROCHIP].store(microchip_floor);
    }

    fn set_elevator_floor(&mut self, elevator_floor: u32) {
        *self = Self::INVALID;
        self.set_microchip_floor(elevator_floor);
    }
}

impl Default for RadioisotopeState {
    fn default() -> Self {
        Self::INVALID
    }
}

#[derive(Clone, Copy, Debug)]
struct FloorElementState {
    index: u32,
    floor: u32,
}

#[derive(Clone, Copy, Debug)]
struct ElevatorPassenger {
    index: u32,
    is_generator: bool,
}

impl ElevatorPassenger {
    const INVALID: Self = Self {
        index: u32::MAX,
        is_generator: false,
    };

    fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct State([RadioisotopeState; Self::LEN]);

impl State {
    const LEN: usize = 8_usize;

    fn normalize(&mut self) {
        self.0.sort();
    }

    fn iter_floor_elements<'a, F: Fn(RadioisotopeState) -> Option<u32> + 'a>(
        &'a self,
        f: F,
    ) -> impl Iterator<Item = FloorElementState> + 'a {
        self.0.iter().copied().enumerate().filter_map(
            move |(floor_element_index, radioisotope_state)| {
                f(radioisotope_state).map(|floor| FloorElementState {
                    index: floor_element_index as u32,
                    floor,
                })
            },
        )
    }

    fn iter_microchips(&self) -> impl Iterator<Item = FloorElementState> + '_ {
        self.iter_floor_elements(RadioisotopeState::try_get_microchip_floor)
    }

    fn iter_microchips_for_floor(&self, floor: u32) -> impl Iterator<Item = u32> + '_ {
        self.iter_microchips()
            .filter_map(move |floor_element_state| {
                (floor_element_state.floor == floor).then_some(floor_element_state.index)
            })
    }

    fn iter_generators(&self) -> impl Iterator<Item = FloorElementState> + '_ {
        self.iter_floor_elements(RadioisotopeState::try_get_generator_floor)
    }

    fn iter_generators_for_floor(&self, floor: u32) -> impl Iterator<Item = u32> + '_ {
        self.iter_generators()
            .filter_map(move |floor_element_state| {
                (floor_element_state.floor == floor).then_some(floor_element_state.index)
            })
    }

    fn fries_microchip(self) -> bool {
        let at_risk_microchip_floors: [bool; 4_usize] = self.iter_microchips().fold(
            Default::default(),
            |mut at_risk_microchip_floors, floor_element_state| {
                let is_microchip_at_risk: &mut bool =
                    &mut at_risk_microchip_floors[floor_element_state.floor as usize];

                *is_microchip_at_risk = *is_microchip_at_risk
                    || self.0[floor_element_state.index as usize].get_generator_floor()
                        != floor_element_state.floor;

                at_risk_microchip_floors
            },
        );
        let generator_floors: [bool; 4_usize] = self.iter_generators().fold(
            Default::default(),
            |mut generator_floors, floor_element_state| {
                generator_floors[floor_element_state.floor as usize] = true;

                generator_floors
            },
        );

        at_risk_microchip_floors
            .into_iter()
            .zip(generator_floors)
            .any(|(is_microchip_at_risk, floor_has_generator)| {
                is_microchip_at_risk && floor_has_generator
            })
    }

    fn elevator_index(self) -> u32 {
        self.0
            .into_iter()
            .enumerate()
            .filter_map(|(index, radioisotope_state)| {
                radioisotope_state
                    .try_get_elevator_floor()
                    .map(|_| index as u32)
            })
            .next()
            .unwrap()
    }

    fn iter_elevator_passengers(
        &self,
        elevator_index: u32,
    ) -> impl Iterator<Item = [ElevatorPassenger; 2_usize]> + '_ {
        let elevator_floor: u32 = self.0[elevator_index as usize].get_elevator_floor();

        // The possibilities for what goes in the elevator are:
        // * 1 microchip
        // * 2 microchips
        // * 1 microchip and 1 generator (both of the same index)
        // * 1 generator
        // * 2 generators
        self.iter_microchips_for_floor(elevator_floor)
            .flat_map(move |microchip_index_a| {
                let elevator_passenger_a: ElevatorPassenger = ElevatorPassenger {
                    index: microchip_index_a,
                    is_generator: false,
                };

                // * 1 microchip
                [[elevator_passenger_a, ElevatorPassenger::INVALID]]
                    .into_iter()
                    // * 2 microchips
                    .chain(self.iter_microchips_for_floor(elevator_floor).filter_map(
                        move |microchip_index_b| {
                            (microchip_index_b > microchip_index_a).then(|| {
                                [
                                    elevator_passenger_a,
                                    ElevatorPassenger {
                                        index: microchip_index_b,
                                        is_generator: false,
                                    },
                                ]
                            })
                        },
                    ))
                    // * 1 microchip and 1 generator (both of the same index)
                    .chain(
                        (self.0[microchip_index_a as usize].get_generator_floor()
                            == elevator_floor)
                            .then(|| {
                                [
                                    elevator_passenger_a,
                                    ElevatorPassenger {
                                        index: microchip_index_a,
                                        is_generator: true,
                                    },
                                ]
                            }),
                    )
            })
            .chain(self.iter_generators_for_floor(elevator_floor).flat_map(
                move |generator_index_a| {
                    let elevator_passenger_a: ElevatorPassenger = ElevatorPassenger {
                        index: generator_index_a,
                        is_generator: true,
                    };

                    // * 1 generator
                    [[elevator_passenger_a, ElevatorPassenger::INVALID]]
                        .into_iter()
                        // * 2 generators
                        .chain(self.iter_generators_for_floor(elevator_floor).filter_map(
                            move |generator_index_b| {
                                (generator_index_b > generator_index_a).then(|| {
                                    [
                                        elevator_passenger_a,
                                        ElevatorPassenger {
                                            index: generator_index_b,
                                            is_generator: true,
                                        },
                                    ]
                                })
                            },
                        ))
                },
            ))
    }

    fn iter_neighbors(&self) -> impl Iterator<Item = Self> + '_ {
        let elevator_index: u32 = self.elevator_index();
        let elevator_floor: u32 = self.0[elevator_index as usize].get_elevator_floor();

        self.iter_elevator_passengers(elevator_index)
            .flat_map(move |elevator_passengers| {
                RadioisotopeState::iter_neighboring_floors(elevator_floor).filter_map(
                    move |elevator_floor| {
                        let mut state: Self = *self;

                        state.0[elevator_index as usize].set_elevator_floor(elevator_floor);

                        for elevator_passenger in elevator_passengers
                            .into_iter()
                            .filter(ElevatorPassenger::is_valid)
                        {
                            let radioisotope_state: &mut RadioisotopeState =
                                &mut state.0[elevator_passenger.index as usize];

                            if elevator_passenger.is_generator {
                                radioisotope_state.set_generator_floor(elevator_floor);
                            } else {
                                radioisotope_state.set_microchip_floor(elevator_floor);
                            }
                        }

                        (!state.fries_microchip()).then(|| {
                            state.normalize();

                            state
                        })
                    },
                )
            })
    }
}

impl Display for State {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let elevator_index: u32 = self.elevator_index();

        for floor_index in (0_u32..4_u32).rev() {
            write!(
                f,
                "F{} {}  ",
                floor_index + 1_u32,
                if self.0[elevator_index as usize].get_microchip_floor() == floor_index {
                    'E'
                } else {
                    '.'
                }
            )?;

            for radioisotope_index in 0_usize..elevator_index as usize {
                let radioisotope_char: char = (radioisotope_index as u8 + b'A') as char;
                let floor_has_generator: bool =
                    self.0[radioisotope_index].get_generator_floor() == floor_index;
                let floor_has_microchip: bool =
                    self.0[radioisotope_index].get_microchip_floor() == floor_index;

                write!(
                    f,
                    "{}{} {}{} ",
                    if floor_has_generator {
                        radioisotope_char
                    } else {
                        '.'
                    },
                    if floor_has_generator { 'G' } else { ' ' },
                    if floor_has_microchip {
                        radioisotope_char
                    } else {
                        '.'
                    },
                    if floor_has_microchip { 'M' } else { ' ' },
                )?;
            }

            writeln!(f, "")?;
        }

        Ok(())
    }
}

#[cfg(test)]
struct RadioisotopeStateIter<I: Iterator<Item = RadioisotopeState>>(I);

#[cfg(test)]
impl<I: Iterator<Item = RadioisotopeState>> From<I> for RadioisotopeStateIter<I> {
    fn from(value: I) -> Self {
        Self(value)
    }
}

#[cfg(test)]
impl<I: Iterator<Item = RadioisotopeState>> TryFrom<RadioisotopeStateIter<I>> for State {
    type Error = ();

    fn try_from(value: RadioisotopeStateIter<I>) -> Result<Self, Self::Error> {
        let mut state: Self = Self::default();

        for (index, radioisotope_state) in value.0.into_iter().enumerate() {
            match state.0.get_mut(index) {
                Some(radioisotope_state_dest) => *radioisotope_state_dest = radioisotope_state,
                None => Err(())?,
            }
        }

        state.normalize();

        Ok(state)
    }
}

struct PathToAssemblyFinder {
    start_state: State,
    end_state: State,
    previous_map: HashMap<State, State>,
}

impl BreadthFirstSearch for PathToAssemblyFinder {
    type Vertex = State;

    fn start(&self) -> &Self::Vertex {
        &self.start_state
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.end_state
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<State> = VecDeque::new();
        let mut state: State = *vertex;

        while state != self.start_state {
            path.push_front(state);
            state = *self.previous_map.get(&state).unwrap();
        }

        path.push_front(self.start_state);

        path.into()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();

        neighbors.extend(vertex.iter_neighbors());
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.previous_map.insert(*to, *from);
    }

    fn reset(&mut self) {
        self.previous_map.clear();
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    start_state: State,
}

impl Solution {
    fn end_state(initial_state: State) -> State {
        let mut end_state: State = State::default();
        let elevator_index: u32 = initial_state.elevator_index();

        for radioisotope_index in 0_usize..elevator_index as usize {
            let radioisotope_state: &mut RadioisotopeState = &mut end_state.0[radioisotope_index];

            radioisotope_state.set_generator_floor(RadioisotopeState::MAX_FLOOR);
            radioisotope_state.set_microchip_floor(RadioisotopeState::MAX_FLOOR);
        }

        end_state.0[elevator_index as usize].set_elevator_floor(RadioisotopeState::MAX_FLOOR);

        end_state
    }

    fn try_start_state_with_extra_radioisotopes(&self) -> Option<State> {
        let elevator_index: usize = self.start_state.elevator_index() as usize;
        let next_elevator_index: usize = elevator_index + 2_usize;

        (next_elevator_index < State::LEN).then(|| {
            let mut start_state: State = self.start_state;

            for radioisotope_index in elevator_index..next_elevator_index {
                start_state.0[radioisotope_index].set_generator_floor(0_u32);
                start_state.0[radioisotope_index].set_microchip_floor(0_u32);
            }

            start_state.0[next_elevator_index].set_elevator_floor(0_u32);

            start_state.normalize();

            start_state
        })
    }

    fn try_path_to_assembly_for_start_state(start_state: State) -> Option<Vec<State>> {
        let mut path_to_assembly_finder: PathToAssemblyFinder = PathToAssemblyFinder {
            start_state,
            end_state: Self::end_state(start_state),
            previous_map: HashMap::new(),
        };

        path_to_assembly_finder.run()
    }

    fn try_path_to_assembly(&self) -> Option<Vec<State>> {
        Self::try_path_to_assembly_for_start_state(self.start_state)
    }

    fn try_steps_to_assembly(&self) -> Option<usize> {
        self.try_path_to_assembly()
            .map(|path| path.len().checked_sub(1_usize))
            .flatten()
    }

    fn try_path_to_assembly_with_extra_radioisotopes(&self) -> Option<Vec<State>> {
        self.try_start_state_with_extra_radioisotopes()
            .and_then(Self::try_path_to_assembly_for_start_state)
    }

    fn try_steps_to_assembly_with_extra_radioisotopes(&self) -> Option<usize> {
        self.try_path_to_assembly_with_extra_radioisotopes()
            .map(|path| path.len().checked_sub(1_usize))
            .flatten()
    }

    fn print_path(path: &[State]) {
        for (index, state) in path.iter().copied().enumerate() {
            println!("state {index}:\n{state}");
        }
    }

    fn verbose_path_and_steps_to_assembly(start_state: State) {
        println!(
            "start:\n{}\nend:\n{}",
            start_state,
            Self::end_state(start_state)
        );

        match Self::try_path_to_assembly_for_start_state(start_state) {
            Some(path) => {
                println!(
                    "Success! path.len() == {}, steps == {:?}",
                    path.len(),
                    path.len().checked_sub(1_usize)
                );

                Self::print_path(&path);
            }
            None => {
                eprintln!("Failure!");
            }
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        const FLOOR_TAGS: &'static [&'static str] = &["first", "second", "third", "fourth"];

        let mut input: &str = input;
        let mut start_state: State = State::default();
        let mut radioisotopes: Vec<&'i str> = Vec::new();

        for (floor_index, floor_tag) in FLOOR_TAGS.iter().copied().enumerate() {
            let floor: u32 = floor_index as u32;

            input = tuple((
                tag("The "),
                tag(floor_tag),
                tag(" floor contains "),
                alt((
                    map(tag("nothing relevant"), |_| {}),
                    map(
                        many1_count(map_opt(
                            tuple((
                                opt(tag(",")),
                                opt(tag(" ")),
                                opt(tag("and ")),
                                FloorElement::parse,
                            )),
                            |(_, _, _, floor_element)| {
                                let mut get_radioisotope_index = |radioisotope_str: &'i str| {
                                    radioisotopes
                                        .iter()
                                        .copied()
                                        .position(|radioisotope| radioisotope == radioisotope_str)
                                        .or_else(|| {
                                            let radioisotope_index: usize = radioisotopes.len();

                                            (radioisotope_index < State::LEN).then(|| {
                                                radioisotopes.push(radioisotope_str);

                                                radioisotope_index
                                            })
                                        })
                                };

                                match floor_element {
                                    FloorElement::Generator(Generator(radioisotope)) => {
                                        start_state.0[get_radioisotope_index(radioisotope)?]
                                            .set_generator_floor(floor);
                                    }
                                    FloorElement::Microchip(Microchip(radioisotope)) => {
                                        start_state.0[get_radioisotope_index(radioisotope)?]
                                            .set_microchip_floor(floor);
                                    }
                                }

                                Some(())
                            },
                        )),
                        |_| {},
                    ),
                )),
                tag("."),
                opt(line_ending),
            ))(input)?
            .0;
        }

        *start_state.0.last_mut().unwrap() = RadioisotopeState::new_elevator(0_u32);

        start_state.normalize();

        Ok((input, Self { start_state }))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            Self::verbose_path_and_steps_to_assembly(self.start_state);
        } else {
            dbg!(self.try_steps_to_assembly());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(start_state) = self.try_start_state_with_extra_radioisotopes() {
                Self::verbose_path_and_steps_to_assembly(start_state);
            } else {
                eprintln!("Failed to add extra isotopes!");
            }
        } else {
            dbg!(self.try_steps_to_assembly_with_extra_radioisotopes());
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

    const SOLUTION_STR: &'static str = "\
        The first floor contains a hydrogen-compatible microchip and a lithium-compatible \
            microchip.\n\
        The second floor contains a hydrogen generator.\n\
        The third floor contains a lithium generator.\n\
        The fourth floor contains nothing relevant.\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            start_state: State([
                RadioisotopeState::new(1_u32, 0_u32),
                RadioisotopeState::new(2_u32, 0_u32),
                RadioisotopeState::new_elevator(0_u32),
                RadioisotopeState::INVALID,
                RadioisotopeState::INVALID,
                RadioisotopeState::INVALID,
                RadioisotopeState::INVALID,
                RadioisotopeState::INVALID,
            ]),
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_steps_to_assembly() {
        assert_eq!(solution().try_steps_to_assembly(), Some(11_usize));
    }

    #[test]
    fn test_iter_neighbors() {
        use RadioisotopeState as RS;
        const IF: u32 = RS::INVALID_FLOOR;

        macro_rules! bs {
            [ $( ( $generator:expr, $microchip:expr ), )* ] => {
                RadioisotopeStateIter([ $( RS::new($generator, $microchip), )* ].into_iter())
                    .try_into()
                    .unwrap()
            }
        }

        let states: &[State] = &[
            bs![(IF, 0), (1, 0), (2, 0),],
            bs![(IF, 1), (1, 1), (2, 0),],
            bs![(IF, 2), (2, 0), (2, 2),],
            bs![(IF, 1), (2, 0), (2, 1),],
            bs![(IF, 0), (2, 0), (2, 0),],
            bs![(IF, 1), (2, 1), (2, 1),],
            bs![(IF, 2), (2, 2), (2, 2),],
            bs![(IF, 3), (2, 3), (2, 3),],
            bs![(IF, 2), (2, 2), (2, 3),],
            bs![(IF, 3), (3, 2), (3, 3),],
            bs![(IF, 2), (3, 2), (3, 2),],
            bs![(IF, 3), (3, 3), (3, 3),],
        ];

        for states in states.windows(2_usize) {
            let from: State = states[0_usize];
            let to: State = states[1_usize];
            let from_neighbors: Vec<State> = from.iter_neighbors().collect();

            if !from_neighbors.contains(&to) {
                eprintln!("from:\n{from}\nto:\n{to}\nneighbors:\n");

                for (index, from_neighbor) in from_neighbors.into_iter().enumerate() {
                    eprintln!("neighbor {index}:\n{from_neighbor}\n");
                }

                assert!(false);
            }
        }
    }
}
