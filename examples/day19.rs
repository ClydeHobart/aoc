use {
    self::MaterialType::*,
    aoc_2022::*,
    rayon::prelude::*,
    std::{
        collections::HashMap,
        hash::Hash,
        mem::transmute,
        num::ParseIntError,
        ops::Range,
        str::{FromStr, Split},
        time::Instant,
    },
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

/// # Notes
///
/// I watched some of Chris Biscardi's video (https://www.youtube.com/watch?v=zrxBpFfjDBE) on this
/// day prior to attempting it myself, so I have them to thank for:
/// * suggesting `rayon` to parallelize the computation for each blueprint
/// * the filtering criteria for `OreRobotCosts`, `ClayRobotCosts`, and `ObsidianRobotCosts`
/// * trimming sibling states of successful `GeodeRobot` states

trait BuildRobot {
    fn build_robot(self, state: &State) -> State;
    fn try_build_robot(self, state: &State, blueprint: Blueprint) -> Option<State>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OreRobotCosts {
    ore: u16,
}

impl BuildRobot for OreRobotCosts {
    fn build_robot(self, state: &State) -> State {
        let mut state: State = state.step();

        state.robots.ore += 1_u16;
        state.materials.ore -= self.ore;

        state
    }

    fn try_build_robot(self, state: &State, blueprint: Blueprint) -> Option<State> {
        if state.materials.ore >= self.ore && state.robots.ore < blueprint.max_ore {
            Some(self.build_robot(state))
        } else {
            None
        }
    }
}

impl<'s> TryFrom<&'s str> for OreRobotCosts {
    type Error = ParseCostsError<'s>;

    fn try_from(ore_robot_costs_str: &'s str) -> Result<Self, Self::Error> {
        Ok(Self {
            ore: Blueprint::parse_ore_and_other(ore_robot_costs_str, "ore", None)?.0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ClayRobotCosts {
    ore: u16,
}

impl BuildRobot for ClayRobotCosts {
    fn build_robot(self, state: &State) -> State {
        let mut state: State = state.step();

        state.robots.clay += 1_u16;
        state.materials.ore -= self.ore;

        state
    }

    fn try_build_robot(self, state: &State, blueprint: Blueprint) -> Option<State> {
        if state.materials.ore >= self.ore && state.robots.clay < blueprint.obsidian.clay {
            Some(self.build_robot(state))
        } else {
            None
        }
    }
}

impl<'s> TryFrom<&'s str> for ClayRobotCosts {
    type Error = ParseCostsError<'s>;

    fn try_from(clay_robot_costs_str: &'s str) -> Result<Self, Self::Error> {
        Ok(Self {
            ore: Blueprint::parse_ore_and_other(clay_robot_costs_str, "clay", None)?.0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ObsidianRobotCosts {
    ore: u16,
    clay: u16,
}

impl BuildRobot for ObsidianRobotCosts {
    fn build_robot(self, state: &State) -> State {
        let mut state: State = state.step();

        state.robots.obsidian += 1_u16;
        state.materials.ore -= self.ore;
        state.materials.clay -= self.clay;

        state
    }

    fn try_build_robot(self, state: &State, blueprint: Blueprint) -> Option<State> {
        if state.materials.ore >= self.ore
            && state.materials.clay >= self.clay
            && state.robots.obsidian < blueprint.geode.obsidian
        {
            Some(self.build_robot(state))
        } else {
            None
        }
    }
}

impl<'s> TryFrom<&'s str> for ObsidianRobotCosts {
    type Error = ParseCostsError<'s>;

    fn try_from(obsidian_robot_costs_str: &'s str) -> Result<Self, Self::Error> {
        let (ore, clay) =
            Blueprint::parse_ore_and_other(obsidian_robot_costs_str, "obsidian", Some("clay"))?;

        Ok(Self { ore, clay })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GeodeRobotCosts {
    ore: u16,
    obsidian: u16,
}

impl BuildRobot for GeodeRobotCosts {
    fn build_robot(self, state: &State) -> State {
        let mut state: State = state.step();

        state.robots.geode += 1_u16;
        state.materials.ore -= self.ore;
        state.materials.obsidian -= self.obsidian;

        state
    }

    fn try_build_robot(self, state: &State, _blueprint: Blueprint) -> Option<State> {
        if state.materials.ore >= self.ore && state.materials.obsidian >= self.obsidian {
            Some(self.build_robot(state))
        } else {
            None
        }
    }
}

impl<'s> TryFrom<&'s str> for GeodeRobotCosts {
    type Error = ParseCostsError<'s>;

    fn try_from(geode_robot_costs_str: &'s str) -> Result<Self, Self::Error> {
        let (ore, obsidian) =
            Blueprint::parse_ore_and_other(geode_robot_costs_str, "geode", Some("obsidian."))?;

        Ok(Self { ore, obsidian })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Blueprint {
    id: u16,
    ore: OreRobotCosts,
    clay: ClayRobotCosts,
    obsidian: ObsidianRobotCosts,
    geode: GeodeRobotCosts,
    max_ore: u16,
}

#[derive(Debug, PartialEq)]
enum ParseCostsError<'s> {
    NoEachToken,
    InvalidEachToken(&'s str),
    NoMaterialToken,
    InvalidMaterialToken(&'s str),
    NoRobotToken,
    InvalidRobotToken(&'s str),
    NoCostsToken,
    InvalidCostsToken(&'s str),
    NoOreToken,
    FailedToParseOre(ParseIntError),
    NoOreTextToken,
    InvalidOreTextToken(&'s str),
    NoAndToken,
    InvalidAndToken(&'s str),
    NoOtherToken,
    FailedToParseOther(ParseIntError),
    NoOtherTextToken,
    InvalidOtherTextToken(&'s str),
    ExtraTokenFound(&'s str),
}

impl Blueprint {
    fn parse_token<'s, I: Iterator<Item = &'s str>, E: Sized, E2: Fn(&'s str) -> E>(
        token_iter: &mut I,
        expected: &str,
        e1: E,
        e2: E2,
    ) -> Result<(), E> {
        match token_iter.next() {
            None => Err(e1),
            Some(actual) if actual == expected => Ok(()),
            Some(invalid_token) => Err(e2(invalid_token)),
        }
    }

    fn parse_ore_and_other<'s>(
        costs_str: &'s str,
        material_str: &str,
        other_str: Option<&str>,
    ) -> Result<(u16, u16), ParseCostsError<'s>> {
        use ParseCostsError::*;

        let mut costs_token_iter: Split<char> = costs_str.split(' ');

        Self::parse_token(&mut costs_token_iter, "Each", NoEachToken, InvalidEachToken)?;
        Self::parse_token(
            &mut costs_token_iter,
            material_str,
            NoMaterialToken,
            InvalidMaterialToken,
        )?;
        Self::parse_token(
            &mut costs_token_iter,
            "robot",
            NoRobotToken,
            InvalidRobotToken,
        )?;
        Self::parse_token(
            &mut costs_token_iter,
            "costs",
            NoCostsToken,
            InvalidCostsToken,
        )?;

        let ore: u16 =
            u16::from_str(costs_token_iter.next().ok_or(NoOreToken)?).map_err(FailedToParseOre)?;

        Self::parse_token(
            &mut costs_token_iter,
            "ore",
            NoOreTextToken,
            InvalidOreTextToken,
        )?;

        let other: u16 = other_str.map_or(Ok(0_u16), |other_str| {
            Self::parse_token(&mut costs_token_iter, "and", NoAndToken, InvalidAndToken)?;

            let other: u16 = u16::from_str(costs_token_iter.next().ok_or(NoOtherToken)?)
                .map_err(FailedToParseOther)?;

            Self::parse_token(
                &mut costs_token_iter,
                other_str,
                NoOtherTextToken,
                InvalidOtherTextToken,
            )?;

            Ok(other)
        })?;

        match costs_token_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok((ore, other)),
        }
    }
}

#[derive(Debug, PartialEq)]
enum ParseBlueprintError<'s> {
    NoBlueprintToken,
    InvalidBlueprintToken(&'s str),
    NoIdToken,
    FailedToParseId(ParseIntError),
    NoOreRobotCostsToken,
    FailedToParseOreRobotCosts(ParseCostsError<'s>),
    NoClayRobotCostsToken,
    FailedToParseClayRobotCosts(ParseCostsError<'s>),
    NoObsidianRobotCostsToken,
    FailedToParseObsidianRobotCosts(ParseCostsError<'s>),
    NoGeodeRobotCostsToken,
    FailedToParseGeodeRobotCosts(ParseCostsError<'s>),
    ExtraTokenFound(&'s str),
}

impl<'s> TryFrom<&'s str> for Blueprint {
    type Error = ParseBlueprintError<'s>;

    fn try_from(blue_print_str: &'s str) -> Result<Self, Self::Error> {
        use ParseBlueprintError::*;

        let mut blueprint_token_iter: Split<&str> = blue_print_str.split(": ");
        let mut header_token_iter: Split<char> = blueprint_token_iter
            .next()
            .ok_or(NoBlueprintToken)?
            .split(' ');

        Self::parse_token(
            &mut header_token_iter,
            "Blueprint",
            NoBlueprintToken,
            InvalidBlueprintToken,
        )?;

        let id: u16 =
            u16::from_str(header_token_iter.next().ok_or(NoIdToken)?).map_err(FailedToParseId)?;

        if let Some(extra_token) = header_token_iter.next() {
            Err(ExtraTokenFound(extra_token))
        } else {
            let mut body_token_iter: Split<&str> = blueprint_token_iter
                .next()
                .ok_or(NoOreRobotCostsToken)?
                .split(". ");

            let ore: OreRobotCosts = body_token_iter
                .next()
                .ok_or(NoOreRobotCostsToken)?
                .try_into()
                .map_err(FailedToParseOreRobotCosts)?;
            let clay: ClayRobotCosts = body_token_iter
                .next()
                .ok_or(NoClayRobotCostsToken)?
                .try_into()
                .map_err(FailedToParseClayRobotCosts)?;
            let obsidian: ObsidianRobotCosts = body_token_iter
                .next()
                .ok_or(NoObsidianRobotCostsToken)?
                .try_into()
                .map_err(FailedToParseObsidianRobotCosts)?;
            let geode: GeodeRobotCosts = body_token_iter
                .next()
                .ok_or(NoGeodeRobotCostsToken)?
                .try_into()
                .map_err(FailedToParseGeodeRobotCosts)?;
            let max_ore: u16 = clay.ore.max(obsidian.ore).max(geode.ore);

            match body_token_iter.next() {
                Some(extra_token) => Err(ExtraTokenFound(extra_token)),
                None => Ok(Self {
                    id,
                    ore,
                    clay,
                    obsidian,
                    geode,
                    max_ore,
                }),
            }
        }
    }
}

#[derive(Debug, PartialEq)]
struct Blueprints(Vec<Blueprint>);

impl Blueprints {
    fn end_results(
        &self,
        max_time: usize,
        max_blueprints: usize,
        root: State,
        status_updates: bool,
    ) -> Vec<EndResult> {
        self.0
            .par_iter()
            .take(max_blueprints)
            .map(|blueprint| {
                let start: Instant = Instant::now();
                let end_result: EndResult = Explorer::new(*blueprint, max_time, root.clone()).run();
                let end: Instant = Instant::now();

                if status_updates {
                    println!(
                        "Blueprint {} finished in {}ms",
                        blueprint.id,
                        (end - start).as_millis()
                    );
                }

                end_result
            })
            .collect()
    }

    fn quality_level_sum_for_end_results(&self, end_results: &[EndResult]) -> usize {
        self.0
            .iter()
            .zip(end_results.iter())
            .map(|(blueprint, end_result)| blueprint.id as usize * end_result.geodes as usize)
            .sum()
    }

    fn quality_level_sum(
        &self,
        max_time: usize,
        max_blueprints: usize,
        root: State,
        status_updates: bool,
    ) -> usize {
        self.quality_level_sum_for_end_results(&self.end_results(
            max_time,
            max_blueprints,
            root,
            status_updates,
        ))
    }

    fn max_geodes_product_for_end_results(end_results: &[EndResult]) -> usize {
        end_results
            .iter()
            .map(|end_result| end_result.geodes as usize)
            .product()
    }

    fn max_geodes_product(
        &self,
        max_time: usize,
        max_blueprints: usize,
        root: State,
        status_updates: bool,
    ) -> usize {
        Self::max_geodes_product_for_end_results(&self.end_results(
            max_time,
            max_blueprints,
            root,
            status_updates,
        ))
    }
}

#[derive(Debug, PartialEq)]
enum ParseBlueprintsError<'s> {
    FailedToParseBlueprint(ParseBlueprintError<'s>),
    InvalidBlueprintId { expected: u16, actual: u16 },
}

impl<'s> TryFrom<&'s str> for Blueprints {
    type Error = ParseBlueprintsError<'s>;

    fn try_from(blueprints_str: &'s str) -> Result<Self, Self::Error> {
        use ParseBlueprintsError::*;

        let mut blueprints: Self = Self(Vec::new());
        let mut id: u16 = 1_u16;

        for blueprint_str in blueprints_str.split('\n') {
            let blueprint: Blueprint = blueprint_str.try_into().map_err(FailedToParseBlueprint)?;

            if blueprint.id != id {
                return Err(InvalidBlueprintId {
                    expected: id,
                    actual: blueprint.id,
                });
            }

            blueprints.0.push(blueprint);
            id += 1_u16;
        }

        Ok(blueprints)
    }
}

/// A material type that could also be a robot, for all but the `Invalid` variant. The attribute
/// `#[allow(dead_code)]` is present since the compiler can't tell that `MaterialTypeIter` will
/// construct all variants.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum MaterialType {
    Invalid = 1_u8,
    Ore = 2_u8,
    Clay = 3_u8,
    Obsidian = 4_u8,
    Geode = 5_u8,
}

impl MaterialType {
    #[inline(always)]
    fn iter() -> MaterialTypeIter {
        MaterialTypeIter(Self::Geode as u8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaterialTypeIter(u8);

impl MaterialTypeIter {
    const VALID_RANGE: Range<u8> = Invalid as u8..(Geode as u8 + 1_u8);
    const DONE: Self = Self(0_u8);

    fn prev(self) -> Option<MaterialType> {
        Self(self.0 + 1_u8).next()
    }
}

impl Default for MaterialTypeIter {
    fn default() -> Self {
        MaterialType::iter()
    }
}

impl Iterator for MaterialTypeIter {
    type Item = MaterialType;

    fn next(&mut self) -> Option<Self::Item> {
        if Self::VALID_RANGE.contains(&self.0) {
            let option: Option<MaterialType> = Some(
                // SAFETY: We've just verified that `self.0` is in the valid range
                unsafe { transmute::<u8, MaterialType>(self.0) },
            );

            self.0 -= 1_u8;

            option
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Default, Debug, Eq, Hash, PartialEq)]
#[repr(align(8))]
struct Counts {
    ore: u16,
    clay: u16,
    obsidian: u16,
    geode: u16,
}

impl Counts {
    #[inline(always)]
    const fn as_u64(self) -> u64 {
        // SAFETY: `Counts` has `align(8)`, and it's 8 bytes
        unsafe { transmute(self) }
    }

    #[inline(always)]
    const fn from_u64(value: u64) -> Self {
        // SAFETY: `Counts` has `align(8)`, and it's 8 bytes
        unsafe { transmute(value) }
    }
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
struct State {
    robots: Counts,
    materials: Counts,
}

impl State {
    #[inline(always)]
    const fn step(&self) -> Self {
        Self {
            robots: self.robots,
            // There's not enough time for these counts to get high enough where we'd have to worry
            // about carryover, so we can save time by adding them all together as a `u32`
            materials: Counts::from_u64(self.materials.as_u64() + self.robots.as_u64()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EndResultLink {
    geodes: u16,
    next: Option<MaterialType>,
}

#[derive(Clone, Debug, Default)]
struct StackState {
    state: State,
    best: Option<(u16, MaterialType)>,
    iter: MaterialTypeIter,
}

#[derive(Debug, Default)]
struct EndResult {
    geodes: u16,
    path: Vec<(State, Option<MaterialType>)>,
}

struct Explorer {
    blueprint: Blueprint,
    states_by_time: Vec<HashMap<State, Option<EndResultLink>>>,
    stack: Vec<StackState>,
    max_time: usize,
    root: State,
}

impl Explorer {
    fn new(blueprint: Blueprint, max_time: usize, root: State) -> Self {
        let mut states_by_time: Vec<HashMap<State, Option<EndResultLink>>> =
            Vec::with_capacity(max_time + 1_usize);
        let mut stack: Vec<StackState> = Vec::with_capacity(max_time + 1_usize);

        states_by_time.resize_with(max_time + 1_usize, HashMap::new);
        states_by_time[0_usize].insert(root.clone(), None);
        stack.push(StackState {
            state: root.clone(),
            ..Default::default()
        });

        Self {
            blueprint,
            states_by_time,
            stack,
            max_time,
            root,
        }
    }

    fn stack_time(&self) -> isize {
        self.stack.len() as isize - 1_isize
    }

    fn build_robot(&self, state: &State, material_type: MaterialType) -> State {
        match material_type {
            Invalid => state.step(),
            Ore => self.blueprint.ore.build_robot(state),
            Clay => self.blueprint.clay.build_robot(state),
            Obsidian => self.blueprint.obsidian.build_robot(state),
            Geode => self.blueprint.geode.build_robot(state),
        }
    }

    fn try_build_robot(&self, state: &State, material_type: MaterialType) -> Option<State> {
        if self.stack_time() < self.max_time as isize {
            match material_type {
                Invalid => Some(state.step()),
                Ore => self.blueprint.ore.try_build_robot(state, self.blueprint),
                Clay => self.blueprint.clay.try_build_robot(state, self.blueprint),
                Obsidian => self
                    .blueprint
                    .obsidian
                    .try_build_robot(state, self.blueprint),
                Geode => self.blueprint.geode.try_build_robot(state, self.blueprint),
            }
        } else {
            None
        }
    }

    fn run(&mut self) -> EndResult {
        while let Some(stack_state) = self.stack.last().cloned() {
            // If the state is already known
            if self.states_by_time[self.stack_time() as usize].get(&stack_state.state)
                .unwrap()
                .is_some()
                // Or there are no more child states to explore
                || {
                    let stack_state: &mut StackState = self.stack.last_mut().unwrap();

                    if let Some(material_type) = stack_state.iter.next() {
                        let curr_state: State = stack_state.state.clone();

                        if let Some(next_state) = self.try_build_robot(&curr_state, material_type)
                        {
                            // We can add a child state from this material
                            self.stack.push(StackState { state: next_state.clone(), ..Default::default() });

                            let stack_time: usize = self.stack_time() as usize;
                            let next_map: &mut HashMap<State, Option<EndResultLink>> =
                                &mut self.states_by_time[stack_time];

                            if !next_map.contains_key(&next_state) {
                                next_map.insert(next_state, None);
                            }
                        }

                        false
                    } else {
                        // There are no more child states to explore
                        true
                    }
                }
            {
                // Pop the stack state, and update as necessary
                let stack_time: usize = self.stack_time() as usize;
                let curr_stack_state: StackState = self.stack.pop().unwrap();

                let end_result_link: &mut Option<EndResultLink> = self.states_by_time[stack_time]
                    .get_mut(&curr_stack_state.state)
                    .unwrap();

                if end_result_link.is_none() {
                    let (geodes, next): (u16, Option<MaterialType>) = curr_stack_state.best.map_or(
                        (curr_stack_state.state.materials.geode, None),
                        |(geodes, material_type)| (geodes, Some(material_type)),
                    );

                    *end_result_link = Some(EndResultLink { geodes, next });
                }

                let geodes: u16 = end_result_link.as_ref().unwrap().geodes;

                if let Some(prev_stack_state) = self.stack.last_mut() {
                    if prev_stack_state
                        .best
                        .clone()
                        .map(|(geodes, _)| geodes)
                        .unwrap_or_default()
                        <= geodes
                    {
                        let prev_material_type: MaterialType =
                            prev_stack_state.iter.prev().unwrap();

                        if prev_material_type == Geode {
                            // We just successfully searched a geode robot child state. No sibling
                            // will be better, since it will directly result in fewer geodes being
                            // produced ultimately
                            prev_stack_state.iter = MaterialTypeIter::DONE;
                        }

                        prev_stack_state.best = Some((geodes, prev_material_type));
                    }
                }
            }
        }

        let mut end_result: EndResult = Default::default();
        let mut state: State = self.root.clone();
        let mut time: usize = 0_usize;

        while time <= self.max_time {
            let end_result_link: EndResultLink =
                self.states_by_time[time].get(&state).unwrap().unwrap();

            if time == 0_usize {
                end_result.geodes = end_result_link.geodes;
            }

            end_result.path.push((state.clone(), end_result_link.next));

            if let Some(material_type) = end_result_link.next {
                state = self.build_robot(&state, material_type);
                time += 1_usize;
            } else {
                break;
            }
        }

        assert_eq!(
            end_result.path.len(),
            self.max_time + 1_usize,
            "Failed to build path for blueprint {}",
            self.blueprint.id,
        );

        end_result
    }
}

const QUESTION_1_MAX_TIME: usize = 24_usize;
const QUESTION_2_MAX_TIME: usize = 32_usize;
const QUESTION_1_MAX_BLUEPRINTS: usize = usize::MAX;
const QUESTION_2_MAX_BLUEPRINTS: usize = 3_usize;
const ROOT: State = State {
    robots: Counts {
        ore: 1_u16,
        clay: 0_u16,
        obsidian: 0_u16,
        geode: 0_u16,
    },
    materials: Counts {
        ore: 0_u16,
        clay: 0_u16,
        obsidian: 0_u16,
        geode: 0_u16,
    },
};

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day19.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                match Blueprints::try_from(input) {
                    Ok(blueprints) => {
                        dbg!(blueprints.quality_level_sum(
                            QUESTION_1_MAX_TIME,
                            QUESTION_1_MAX_BLUEPRINTS,
                            ROOT,
                            args.verbose
                        ));
                        dbg!(blueprints.max_geodes_product(
                            QUESTION_2_MAX_TIME,
                            QUESTION_2_MAX_BLUEPRINTS,
                            ROOT,
                            args.verbose
                        ));
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                }
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLUEPRINTS_STR: &str = concat!(
        "Blueprint 1: \
            Each ore robot costs 4 ore. \
            Each clay robot costs 2 ore. \
            Each obsidian robot costs 3 ore and 14 clay. \
            Each geode robot costs 2 ore and 7 obsidian.\n",
        "Blueprint 2: \
            Each ore robot costs 2 ore. \
            Each clay robot costs 3 ore. \
            Each obsidian robot costs 3 ore and 8 clay. \
            Each geode robot costs 3 ore and 12 obsidian."
    );

    lazy_static! {
        static ref BLUEPRINTS: Blueprints = blueprints();
    }

    #[test]
    fn test_blueprints_try_from_str() {
        assert_eq!(BLUEPRINTS_STR.try_into().as_ref(), Ok(&*BLUEPRINTS));
    }

    #[test]
    fn test_question_1() {
        let end_results: Vec<EndResult> =
            BLUEPRINTS.end_results(QUESTION_1_MAX_TIME, QUESTION_1_MAX_BLUEPRINTS, ROOT, true);

        assert_eq!(
            end_results
                .iter()
                .map(|end_result| end_result.geodes)
                .collect::<Vec<u16>>(),
            vec![9_u16, 12_u16]
        );
        assert_eq!(
            BLUEPRINTS.quality_level_sum_for_end_results(&end_results),
            33_usize
        );
    }

    #[test]
    fn test_question_2() {
        let end_results: Vec<EndResult> =
            BLUEPRINTS.end_results(QUESTION_2_MAX_TIME, QUESTION_2_MAX_BLUEPRINTS, ROOT, true);

        assert_eq!(
            end_results
                .iter()
                .map(|end_result| end_result.geodes)
                .collect::<Vec<u16>>(),
            vec![56_u16, 62_u16]
        );
        assert_eq!(
            Blueprints::max_geodes_product_for_end_results(&end_results),
            3472_usize,
        )
    }

    fn blueprints() -> Blueprints {
        Blueprints(vec![
            Blueprint {
                id: 1_u16,
                ore: OreRobotCosts { ore: 4_u16 },
                clay: ClayRobotCosts { ore: 2_u16 },
                obsidian: ObsidianRobotCosts {
                    ore: 3_u16,
                    clay: 14_u16,
                },
                geode: GeodeRobotCosts {
                    ore: 2_u16,
                    obsidian: 7_u16,
                },
                max_ore: 3_u16,
            },
            Blueprint {
                id: 2_u16,
                ore: OreRobotCosts { ore: 2_u16 },
                clay: ClayRobotCosts { ore: 3_u16 },
                obsidian: ObsidianRobotCosts {
                    ore: 3_u16,
                    clay: 8_u16,
                },
                geode: GeodeRobotCosts {
                    ore: 3_u16,
                    obsidian: 12_u16,
                },
                max_ore: 3_u16,
            },
        ])
    }
}
