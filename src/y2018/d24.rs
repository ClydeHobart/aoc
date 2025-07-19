use {
    crate::*,
    arrayvec::ArrayVec,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_opt, opt, success, verify},
        error::Error,
        multi::{many0_count, separated_list0},
        sequence::{delimited, preceded, tuple},
        Err, IResult,
    },
    std::{
        cmp::{Ordering, Reverse},
        iter::repeat,
        u16,
    },
};

/* --- Day 24: Immune System Simulator 20XX ---

After a weird buzzing noise, you appear back at the man's cottage. He seems relieved to see his friend, but quickly notices that the little reindeer caught some kind of cold while out exploring.

The portly man explains that this reindeer's immune system isn't similar to regular reindeer immune systems:

The immune system and the infection each have an army made up of several groups; each group consists of one or more identical units. The armies repeatedly fight until only one army has units remaining.

Units within a group all have the same hit points (amount of damage a unit can take before it is destroyed), attack damage (the amount of damage each unit deals), an attack type, an initiative (higher initiative units attack first and win ties), and sometimes weaknesses or immunities. Here is an example group:

18 units each with 729 hit points (weak to fire; immune to cold, slashing)
 with an attack that does 8 radiation damage at initiative 10

Each group also has an effective power: the number of units in that group multiplied by their attack damage. The above group has an effective power of 18 * 8 = 144. Groups never have zero or negative units; instead, the group is removed from combat.

Each fight consists of two phases: target selection and attacking.

During the target selection phase, each group attempts to choose one target. In decreasing order of effective power, groups choose their targets; in a tie, the group with the higher initiative chooses first. The attacking group chooses to target the group in the enemy army to which it would deal the most damage (after accounting for weaknesses and immunities, but not accounting for whether the defending group has enough units to actually receive all of that damage).

If an attacking group is considering two defending groups to which it would deal equal damage, it chooses to target the defending group with the largest effective power; if there is still a tie, it chooses the defending group with the highest initiative. If it cannot deal any defending groups damage, it does not choose a target. Defending groups can only be chosen as a target by one attacking group.

At the end of the target selection phase, each group has selected zero or one groups to attack, and each group is being attacked by zero or one groups.

During the attacking phase, each group deals damage to the target it selected, if any. Groups attack in decreasing order of initiative, regardless of whether they are part of the infection or the immune system. (If a group contains no units, it cannot attack.)

The damage an attacking group deals to a defending group depends on the attacking group's attack type and the defending group's immunities and weaknesses. By default, an attacking group would deal damage equal to its effective power to the defending group. However, if the defending group is immune to the attacking group's attack type, the defending group instead takes no damage; if the defending group is weak to the attacking group's attack type, the defending group instead takes double damage.

The defending group only loses whole units from damage; damage is always dealt in such a way that it kills the most units possible, and any remaining damage to a unit that does not immediately kill it is ignored. For example, if a defending group contains 10 units with 10 hit points each and receives 75 damage, it loses exactly 7 units and is left with 3 units at full health.

After the fight is over, if both armies still contain units, a new fight begins; combat only ends once one army has lost all of its units.

For example, consider the following armies:

Immune System:
17 units each with 5390 hit points (weak to radiation, bludgeoning) with
 an attack that does 4507 fire damage at initiative 2
989 units each with 1274 hit points (immune to fire; weak to bludgeoning,
 slashing) with an attack that does 25 slashing damage at initiative 3

Infection:
801 units each with 4706 hit points (weak to radiation) with an attack
 that does 116 bludgeoning damage at initiative 1
4485 units each with 2961 hit points (immune to radiation; weak to fire,
 cold) with an attack that does 12 slashing damage at initiative 4

If these armies were to enter combat, the following fights, including details during the target selection and attacking phases, would take place:

Immune System:
Group 1 contains 17 units
Group 2 contains 989 units
Infection:
Group 1 contains 801 units
Group 2 contains 4485 units

Infection group 1 would deal defending group 1 185832 damage
Infection group 1 would deal defending group 2 185832 damage
Infection group 2 would deal defending group 2 107640 damage
Immune System group 1 would deal defending group 1 76619 damage
Immune System group 1 would deal defending group 2 153238 damage
Immune System group 2 would deal defending group 1 24725 damage

Infection group 2 attacks defending group 2, killing 84 units
Immune System group 2 attacks defending group 1, killing 4 units
Immune System group 1 attacks defending group 2, killing 51 units
Infection group 1 attacks defending group 1, killing 17 units

Immune System:
Group 2 contains 905 units
Infection:
Group 1 contains 797 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 184904 damage
Immune System group 2 would deal defending group 1 22625 damage
Immune System group 2 would deal defending group 2 22625 damage

Immune System group 2 attacks defending group 1, killing 4 units
Infection group 1 attacks defending group 2, killing 144 units

Immune System:
Group 2 contains 761 units
Infection:
Group 1 contains 793 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 183976 damage
Immune System group 2 would deal defending group 1 19025 damage
Immune System group 2 would deal defending group 2 19025 damage

Immune System group 2 attacks defending group 1, killing 4 units
Infection group 1 attacks defending group 2, killing 143 units

Immune System:
Group 2 contains 618 units
Infection:
Group 1 contains 789 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 183048 damage
Immune System group 2 would deal defending group 1 15450 damage
Immune System group 2 would deal defending group 2 15450 damage

Immune System group 2 attacks defending group 1, killing 3 units
Infection group 1 attacks defending group 2, killing 143 units

Immune System:
Group 2 contains 475 units
Infection:
Group 1 contains 786 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 182352 damage
Immune System group 2 would deal defending group 1 11875 damage
Immune System group 2 would deal defending group 2 11875 damage

Immune System group 2 attacks defending group 1, killing 2 units
Infection group 1 attacks defending group 2, killing 142 units

Immune System:
Group 2 contains 333 units
Infection:
Group 1 contains 784 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 181888 damage
Immune System group 2 would deal defending group 1 8325 damage
Immune System group 2 would deal defending group 2 8325 damage

Immune System group 2 attacks defending group 1, killing 1 unit
Infection group 1 attacks defending group 2, killing 142 units

Immune System:
Group 2 contains 191 units
Infection:
Group 1 contains 783 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 181656 damage
Immune System group 2 would deal defending group 1 4775 damage
Immune System group 2 would deal defending group 2 4775 damage

Immune System group 2 attacks defending group 1, killing 1 unit
Infection group 1 attacks defending group 2, killing 142 units

Immune System:
Group 2 contains 49 units
Infection:
Group 1 contains 782 units
Group 2 contains 4434 units

Infection group 1 would deal defending group 2 181424 damage
Immune System group 2 would deal defending group 1 1225 damage
Immune System group 2 would deal defending group 2 1225 damage

Immune System group 2 attacks defending group 1, killing 0 units
Infection group 1 attacks defending group 2, killing 49 units

Immune System:
No groups remain.
Infection:
Group 1 contains 782 units
Group 2 contains 4434 units

In the example above, the winning army ends up with 782 + 4434 = 5216 units.

You scan the reindeer's condition (your puzzle input); the white-bearded man looks nervous. As it stands now, how many units would the winning army have?

--- Part Two ---

Things aren't looking good for the reindeer. The man asks whether more milk and cookies would help you think.

If only you could give the reindeer's immune system a boost, you might be able to change the outcome of the combat.

A boost is an integer increase in immune system units' attack damage. For example, if you were to boost the above example's immune system's units by 1570, the armies would instead look like this:

Immune System:
17 units each with 5390 hit points (weak to radiation, bludgeoning) with
 an attack that does 6077 fire damage at initiative 2
989 units each with 1274 hit points (immune to fire; weak to bludgeoning,
 slashing) with an attack that does 1595 slashing damage at initiative 3

Infection:
801 units each with 4706 hit points (weak to radiation) with an attack
 that does 116 bludgeoning damage at initiative 1
4485 units each with 2961 hit points (immune to radiation; weak to fire,
 cold) with an attack that does 12 slashing damage at initiative 4

With this boost, the combat proceeds differently:

Immune System:
Group 2 contains 989 units
Group 1 contains 17 units
Infection:
Group 1 contains 801 units
Group 2 contains 4485 units

Infection group 1 would deal defending group 2 185832 damage
Infection group 1 would deal defending group 1 185832 damage
Infection group 2 would deal defending group 1 53820 damage
Immune System group 2 would deal defending group 1 1577455 damage
Immune System group 2 would deal defending group 2 1577455 damage
Immune System group 1 would deal defending group 2 206618 damage

Infection group 2 attacks defending group 1, killing 9 units
Immune System group 2 attacks defending group 1, killing 335 units
Immune System group 1 attacks defending group 2, killing 32 units
Infection group 1 attacks defending group 2, killing 84 units

Immune System:
Group 2 contains 905 units
Group 1 contains 8 units
Infection:
Group 1 contains 466 units
Group 2 contains 4453 units

Infection group 1 would deal defending group 2 108112 damage
Infection group 1 would deal defending group 1 108112 damage
Infection group 2 would deal defending group 1 53436 damage
Immune System group 2 would deal defending group 1 1443475 damage
Immune System group 2 would deal defending group 2 1443475 damage
Immune System group 1 would deal defending group 2 97232 damage

Infection group 2 attacks defending group 1, killing 8 units
Immune System group 2 attacks defending group 1, killing 306 units
Infection group 1 attacks defending group 2, killing 29 units

Immune System:
Group 2 contains 876 units
Infection:
Group 2 contains 4453 units
Group 1 contains 160 units

Infection group 2 would deal defending group 2 106872 damage
Immune System group 2 would deal defending group 2 1397220 damage
Immune System group 2 would deal defending group 1 1397220 damage

Infection group 2 attacks defending group 2, killing 83 units
Immune System group 2 attacks defending group 2, killing 427 units

After a few fights...

Immune System:
Group 2 contains 64 units
Infection:
Group 2 contains 214 units
Group 1 contains 19 units

Infection group 2 would deal defending group 2 5136 damage
Immune System group 2 would deal defending group 2 102080 damage
Immune System group 2 would deal defending group 1 102080 damage

Infection group 2 attacks defending group 2, killing 4 units
Immune System group 2 attacks defending group 2, killing 32 units

Immune System:
Group 2 contains 60 units
Infection:
Group 1 contains 19 units
Group 2 contains 182 units

Infection group 1 would deal defending group 2 4408 damage
Immune System group 2 would deal defending group 1 95700 damage
Immune System group 2 would deal defending group 2 95700 damage

Immune System group 2 attacks defending group 1, killing 19 units

Immune System:
Group 2 contains 60 units
Infection:
Group 2 contains 182 units

Infection group 2 would deal defending group 2 4368 damage
Immune System group 2 would deal defending group 2 95700 damage

Infection group 2 attacks defending group 2, killing 3 units
Immune System group 2 attacks defending group 2, killing 30 units

After a few more fights...

Immune System:
Group 2 contains 51 units
Infection:
Group 2 contains 40 units

Infection group 2 would deal defending group 2 960 damage
Immune System group 2 would deal defending group 2 81345 damage

Infection group 2 attacks defending group 2, killing 0 units
Immune System group 2 attacks defending group 2, killing 27 units

Immune System:
Group 2 contains 51 units
Infection:
Group 2 contains 13 units

Infection group 2 would deal defending group 2 312 damage
Immune System group 2 would deal defending group 2 81345 damage

Infection group 2 attacks defending group 2, killing 0 units
Immune System group 2 attacks defending group 2, killing 13 units

Immune System:
Group 2 contains 51 units
Infection:
No groups remain.

This boost would allow the immune system's armies to win! It would be left with 51 units.

You don't even know how you could boost the reindeer's immune system or what effect it might have, so you need to be cautious and find the smallest boost that would allow the immune system to win.

How many units does the immune system have left after getting the smallest boost it needs to win? */

const ARMY_ID_LEN: usize = 15_usize;
const MAX_ARMY_COUNT: usize = 4_usize;

type ArmyId = StaticString<ARMY_ID_LEN>;
type ArmyIndexRaw = u8;
type ArmyIndex = Index<ArmyIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct ArmyData {
    groups: GroupBitArray,
}

#[allow(dead_code)]
type Army = TableElement<ArmyId, ArmyData>;
type ArmyTable = Table<ArmyId, ArmyData, ArmyIndexRaw>;
type ArmyArrayVec<T> = ArrayVec<T, MAX_ARMY_COUNT>;

const ATTACK_TYPE_ID_LEN: usize = 15_usize;
const MAX_ATTACK_TYPE_COUNT: usize = u8::BITS as usize;

type AttackTypeId = StaticString<ATTACK_TYPE_ID_LEN>;
type AttackTypeIndexRaw = u8;
type AttackTypeIndex = Index<AttackTypeIndexRaw>;
type AttackTypeList = IdList<AttackTypeId, AttackTypeIndexRaw>;
type AttackTypeBitArray = BitArr!(for MAX_ATTACK_TYPE_COUNT, in u8);

const MAX_GROUP_COUNT: usize = u32::BITS as usize;

type GroupIndexRaw = u8;
type GroupIndex = Index<GroupIndexRaw>;

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Default, Eq, PartialEq)]
struct GroupId {
    initiative: u8,
}

impl Ord for GroupId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.initiative.cmp(&other.initiative).reverse()
    }
}

impl PartialOrd for GroupId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default, PartialEq, Clone, Copy)]
struct GroupState {
    units: u16,
    attack_damage: u16,
    effective_power: u32,
}

impl GroupState {
    fn new(units: u16, attack_damage: u16) -> Self {
        GroupState {
            units,
            attack_damage,
            effective_power: units as u32 * attack_damage as u32,
        }
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct TargetSelectionKey {
    damage_dealt: u32,
    effective_power: u32,
    initiative: u8,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct GroupData {
    army_index: ArmyIndex,

    #[allow(dead_code)]
    army_group_index: u8,
    units: u16,
    hit_points: u16,
    immunities: AttackTypeBitArray,
    weaknesses: AttackTypeBitArray,
    attack_damage: u16,
    attack_type_index: AttackTypeIndex,
}

impl GroupData {
    fn parse_attack_types<'i, F: FnMut(&'i str) -> IResult<&'i str, AttackTypeIndex>>(
        tag_str: &'i str,
        mut parse_attack_type_index: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, AttackTypeBitArray> {
        move |input| {
            let mut attack_types: AttackTypeBitArray = AttackTypeBitArray::ZERO;

            let input: &str = tuple((
                tag(tag_str),
                tag(" to "),
                separated_list0(
                    tag(", "),
                    map(
                        |input| parse_attack_type_index(input),
                        |attack_type_index| {
                            if attack_type_index.is_valid() {
                                attack_types.set(attack_type_index.get(), true);
                            }
                        },
                    ),
                ),
            ))(input)?
            .0;

            Ok((input, attack_types))
        }
    }

    fn parse_immunities_and_weaknesses<
        'i,
        F: FnMut(&'i str) -> IResult<&'i str, AttackTypeIndex>,
    >(
        mut parse_attack_type_index: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, (AttackTypeBitArray, AttackTypeBitArray)> {
        move |input| {
            let mut input: &str = input;
            let mut immunities: Option<AttackTypeBitArray>;
            let weaknesses: Option<AttackTypeBitArray>;

            let (input_value, immunities_value): (&str, Option<AttackTypeBitArray>) =
                opt(Self::parse_attack_types("immune", |input| {
                    parse_attack_type_index(input)
                }))(input)?;

            input = input_value;
            immunities = immunities_value;

            if immunities.is_some() {
                let (input_value, weaknesses_value): (&str, Option<AttackTypeBitArray>) =
                    opt(preceded(
                        tag("; "),
                        Self::parse_attack_types("weak", |input| parse_attack_type_index(input)),
                    ))(input)?;

                input = input_value;
                weaknesses = weaknesses_value;
            } else {
                let (input_value, weaknesses_value): (&str, Option<AttackTypeBitArray>) =
                    opt(Self::parse_attack_types("weak", |input| {
                        parse_attack_type_index(input)
                    }))(input)?;

                input = input_value;
                weaknesses = weaknesses_value;

                if weaknesses.is_some() {
                    let (input_value, immunities_value): (&str, Option<AttackTypeBitArray>) =
                        opt(preceded(
                            tag("; "),
                            Self::parse_attack_types("immune", |input| {
                                parse_attack_type_index(input)
                            }),
                        ))(input)?;

                    input = input_value;
                    immunities = immunities_value;
                }
            }

            Ok((
                input,
                (
                    immunities.unwrap_or_default(),
                    weaknesses.unwrap_or_default(),
                ),
            ))
        }
    }

    fn parse<'i, F: FnMut(&'i str) -> IResult<&'i str, AttackTypeIndex>>(
        army_index: ArmyIndex,
        army_group_index: u8,
        mut parse_attack_type_index: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Group> {
        move |input| {
            let (input, (units, _, hit_points, _)): (&str, (u16, _, u16, _)) =
                tuple((
                    verify(parse_integer, |&units| units > 0_u16),
                    tag(" units each with "),
                    verify(parse_integer, |&hit_points| hit_points > 0_u16),
                    tag(" hit points"),
                ))(input)?;
            let (input, (immunities, weaknesses)): (
                &str,
                (AttackTypeBitArray, AttackTypeBitArray),
            ) = map(
                opt(delimited(
                    tag(" ("),
                    Self::parse_immunities_and_weaknesses(|input| parse_attack_type_index(input)),
                    tag(")"),
                )),
                Option::unwrap_or_default,
            )(input)?;
            let (input, (_, attack_damage, _, attack_type_index, _, initiative)): (
                &str,
                (_, u16, _, AttackTypeIndex, _, u8),
            ) = tuple((
                tag(" with an attack that does "),
                verify(parse_integer, |&attack_damage| attack_damage > 0_u16),
                tag(" "),
                |input| parse_attack_type_index(input),
                tag(" damage at initiative "),
                parse_integer,
            ))(input)?;

            Ok((
                input,
                Group {
                    id: GroupId { initiative },
                    data: GroupData {
                        army_index,
                        army_group_index,
                        units,
                        hit_points,
                        immunities,
                        weaknesses,
                        attack_damage,
                        attack_type_index,
                    },
                },
            ))
        }
    }

    fn initial_group_state(&self, army_boosts: &[u16]) -> GroupState {
        GroupState::new(
            self.units,
            self.attack_damage + army_boosts[self.army_index.get()],
        )
    }

    fn damage_dealt_multiplier(&self, other: &Self) -> u32 {
        if other.weaknesses[self.attack_type_index.get()] {
            2_u32
        } else if !other.immunities[self.attack_type_index.get()] {
            1_u32
        } else {
            0_u32
        }
    }
}

type Group = TableElement<GroupId, GroupData>;
type GroupTable = Table<GroupId, GroupData, GroupIndexRaw>;
type GroupBitArray = BitArr!(for MAX_GROUP_COUNT, in u32);
type GroupArrayVec<T> = ArrayVec<T, MAX_GROUP_COUNT>;

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct TargetSelectionOrderKey {
    effective_power: u32,
    initiative: u8,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct SolutionState {
    group_states: GroupArrayVec<GroupState>,
    army_group_counts: ArmyArrayVec<u8>,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    armies: ArmyTable,

    #[allow(dead_code)]
    attack_types: AttackTypeList,
    groups: GroupTable,
}

impl Solution {
    const IMMUNE_SYSTEM_ARMY_ID_STR: &'static str = "Immune System";

    fn immune_system_army_id() -> ArmyId {
        Self::IMMUNE_SYSTEM_ARMY_ID_STR.try_into().unwrap()
    }

    fn parse_internal<
        'i,
        F: FnMut(&'i str) -> IResult<&'i str, ArmyIndex>,
        G: FnMut(&'i str) -> IResult<&'i str, AttackTypeIndex>,
        H: FnMut(Group),
    >(
        mut parse_army_index: F,
        mut parse_attack_type_index: G,
        mut process_group: H,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, ()> {
        move |input| {
            map(
                separated_list0(parse_line_endings(2_usize), |input| {
                    let (input, army_index): (&str, ArmyIndex) = parse_army_index(input)?;
                    let input: &str = tag(":")(input)?.0;

                    let mut army_group_index: u8 = 0_u8;

                    let input: &str = many0_count(tuple((
                        line_ending,
                        map(
                            |input| {
                                let (input, next_army_group_index): (&str, u8) =
                                    map_opt(success(()), |_| army_group_index.checked_add(1_u8))(
                                        input,
                                    )?;

                                army_group_index = next_army_group_index;

                                GroupData::parse(army_index, army_group_index, |input| {
                                    parse_attack_type_index(input)
                                })(input)
                            },
                            |group| process_group(group),
                        ),
                    )))(input)?
                    .0;

                    Ok((input, ()))
                }),
                |_| (),
            )(input)
        }
    }

    fn parse_army_id<'i>(input: &'i str) -> IResult<&'i str, ArmyId> {
        ArmyId::parse_char1(1_usize, |c| c != ':')(input)
    }

    fn parse_attack_type_id<'i>(input: &'i str) -> IResult<&'i str, AttackTypeId> {
        AttackTypeId::parse_char1(1_usize, |c| c.is_ascii_lowercase())(input)
    }

    fn initial_group_states(&self, army_boosts: &[u16]) -> GroupArrayVec<GroupState> {
        self.groups
            .as_slice()
            .iter()
            .map(|group| group.data.initial_group_state(army_boosts))
            .collect()
    }

    fn initial_army_group_counts(&self) -> ArmyArrayVec<u8> {
        self.armies
            .as_slice()
            .iter()
            .map(|army| army.data.groups.count_ones() as u8)
            .collect()
    }

    fn no_army_boosts(&self) -> ArmyArrayVec<u16> {
        repeat(0_u16).take(self.armies.as_slice().len()).collect()
    }

    fn immune_system_army_index(&self) -> ArmyIndex {
        self.armies.find_index(&Self::immune_system_army_id())
    }

    fn initial_solution_state(&self, army_boosts: &[u16]) -> SolutionState {
        SolutionState {
            group_states: self.initial_group_states(army_boosts),
            army_group_counts: self.initial_army_group_counts(),
        }
    }

    fn get_target_selection_order_and_valid_targets(
        &self,
        group_states: &[GroupState],
        target_selection_order: &mut GroupArrayVec<GroupIndex>,
        valid_targets: &mut GroupBitArray,
    ) {
        let target_selection_order_keys: GroupArrayVec<TargetSelectionOrderKey> = self
            .groups
            .as_slice()
            .iter()
            .zip(group_states.iter())
            .map(|(group, group_state)| TargetSelectionOrderKey {
                effective_power: group_state.effective_power,
                initiative: group.id.initiative,
            })
            .collect();

        target_selection_order.clear();
        valid_targets.fill(false);
        target_selection_order.extend(group_states.iter().enumerate().filter_map(
            |(group_index, group_state)| {
                (group_state.units > 0_u16).then(|| {
                    valid_targets.set(group_index, true);

                    group_index.into()
                })
            },
        ));

        target_selection_order
            .sort_by_key(|group_index| Reverse(&target_selection_order_keys[group_index.get()]));
    }

    fn try_select_target(
        &self,
        group_states: &[GroupState],
        valid_targets: GroupBitArray,
        group_index: GroupIndex,
    ) -> GroupIndex {
        let groups: &[Group] = self.groups.as_slice();
        let group_data: &GroupData = &groups[group_index.get()].data;
        let army_index: ArmyIndex = group_data.army_index;
        let effective_power: u32 = group_states[group_index.get()].effective_power;

        valid_targets
            .iter_ones()
            .filter_map(|target_group_index| {
                let target_group: &Group = &groups[target_group_index];
                let damage_dealt: u32 =
                    effective_power * group_data.damage_dealt_multiplier(&target_group.data);

                (target_group.data.army_index != army_index && damage_dealt > 0_u32).then(|| {
                    let target_selection_key: TargetSelectionKey = TargetSelectionKey {
                        damage_dealt,
                        effective_power: group_states[target_group_index].effective_power,
                        initiative: target_group.id.initiative,
                    };

                    (target_group_index.into(), target_selection_key)
                })
            })
            .max_by_key(|&(_, target_selection_key)| target_selection_key)
            .map(|(target_group_index, _)| target_group_index)
            .unwrap_or_default()
    }

    fn target_selection_phase(
        &self,
        group_states: &[GroupState],
        targets: &mut GroupArrayVec<GroupIndex>,
    ) {
        let mut target_selection_order: GroupArrayVec<GroupIndex> = GroupArrayVec::new();
        let mut valid_targets: GroupBitArray = GroupBitArray::ZERO;

        self.get_target_selection_order_and_valid_targets(
            group_states,
            &mut target_selection_order,
            &mut valid_targets,
        );
        targets.clear();
        targets.extend(repeat(GroupIndex::invalid()).take(self.groups.as_slice().len()));

        for group_index in target_selection_order {
            let target_group_index: GroupIndex =
                self.try_select_target(group_states, valid_targets, group_index);

            if target_group_index.is_valid() {
                valid_targets.set(target_group_index.get(), false);
            }

            targets[group_index.get()] = target_group_index;
        }
    }

    fn attacking_phase(
        &self,
        targets: &[GroupIndex],
        group_states: &mut [GroupState],
        army_group_counts: &mut [u8],
    ) -> u16 {
        let groups: &[Group] = self.groups.as_slice();

        targets
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(group_index, target_group_index)| {
                let effective_power: u32 = group_states[group_index].effective_power;

                (target_group_index.is_valid() && effective_power > 0_u32).then(|| {
                    let group_data: &GroupData = &groups[group_index].data;
                    let target_group_data: &GroupData = &groups[target_group_index.get()].data;
                    let damage_dealt: u32 =
                        effective_power * group_data.damage_dealt_multiplier(target_group_data);
                    let target_group_state: &mut GroupState =
                        &mut group_states[target_group_index.get()];
                    let target_units: u16 = target_group_state.units;
                    let killed_units: u16 = ((damage_dealt / target_group_data.hit_points as u32)
                        as u16)
                        .min(target_units);

                    *target_group_state = GroupState::new(
                        target_units - killed_units,
                        target_group_state.attack_damage,
                    );

                    if killed_units == target_units {
                        army_group_counts[target_group_data.army_index.get()] -= 1_u8;
                    }

                    killed_units
                })
            })
            .sum()
    }

    fn fight(&self, solution_state: &mut SolutionState) -> u16 {
        let mut targets: GroupArrayVec<GroupIndex> = GroupArrayVec::new();

        self.target_selection_phase(&solution_state.group_states, &mut targets);

        self.attacking_phase(
            &targets,
            &mut solution_state.group_states,
            &mut solution_state.army_group_counts,
        )
    }

    fn war(&self, solution_state: &mut SolutionState) -> ArmyIndex {
        let mut winning_army_index: ArmyIndex = ArmyIndex::invalid();

        while {
            if let Some(winning_army_index_value) = solution_state
                .army_group_counts
                .iter()
                .enumerate()
                .try_fold(
                    ArmyIndex::invalid(),
                    |remaining_army, (army_index, &army_group_count)| {
                        if army_group_count == 0_u8 {
                            Some(remaining_army)
                        } else if remaining_army.is_valid() {
                            None
                        } else {
                            Some(army_index.into())
                        }
                    },
                )
            {
                winning_army_index = winning_army_index_value;

                false
            } else {
                self.fight(solution_state) > 0_u16
            }
        } {}

        winning_army_index
    }

    fn winning_army_index_and_unit_count(&self, army_boosts: &[u16]) -> (ArmyIndex, u32) {
        let mut solution_state: SolutionState = self.initial_solution_state(army_boosts);

        let winning_army_index: ArmyIndex = self.war(&mut solution_state);

        (
            winning_army_index,
            solution_state
                .group_states
                .into_iter()
                .map(|group_state| group_state.units as u32)
                .sum(),
        )
    }

    fn winning_army_unit_count(&self) -> u32 {
        self.winning_army_index_and_unit_count(&self.no_army_boosts())
            .1
    }

    fn immune_system_winning_army_unit_count_with_min_boost(&self) -> u32 {
        let immune_system_army_index: ArmyIndex = self.immune_system_army_index();
        let immune_system_army_index_usize: usize = immune_system_army_index.get();

        let mut army_boosts: ArmyArrayVec<u16> = self.no_army_boosts();
        let mut winning_army_index_and_unit_count: (ArmyIndex, u32) =
            self.winning_army_index_and_unit_count(&army_boosts);

        if winning_army_index_and_unit_count.0 == immune_system_army_index {
            winning_army_index_and_unit_count.1
        } else {
            let mut boost: u16 = 1_u16;

            // This is the largest boost that doesn't result in the Immune System winning.
            let mut min_boost: u16 = 0_u16;

            // Keep doubling min_minimum_boost until it results in a win.
            while {
                army_boosts[immune_system_army_index_usize] = boost;

                winning_army_index_and_unit_count =
                    self.winning_army_index_and_unit_count(&army_boosts);

                winning_army_index_and_unit_count.0 != immune_system_army_index
            } {
                min_boost = boost;
                boost *= 2_u16;
            }

            // This should be tied to the winning army unit count produced by the smallest boost
            // that does result in the Immune System winning.
            let mut winning_army_unit_count: u32 = winning_army_index_and_unit_count.1;
            let mut delta: u16 = boost - min_boost;

            while delta > 1_u16 {
                delta /= 2_u16;
                boost = min_boost + delta;
                army_boosts[immune_system_army_index_usize] = boost;

                winning_army_index_and_unit_count =
                    self.winning_army_index_and_unit_count(&army_boosts);

                if winning_army_index_and_unit_count.0 == immune_system_army_index {
                    winning_army_unit_count = winning_army_index_and_unit_count.1;
                } else {
                    min_boost = boost;
                }
            }

            winning_army_unit_count
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut armies: ArmyTable = ArmyTable::new();
        let mut attack_types: AttackTypeList = AttackTypeList::new();
        let mut groups: GroupTable = GroupTable::new();

        Self::parse_internal(
            map(Self::parse_army_id, |army_id| {
                armies.find_or_add_index(&army_id)
            }),
            map(Self::parse_attack_type_id, |attack_type_id| {
                attack_types.find_or_add_index(&attack_type_id);

                AttackTypeIndex::invalid()
            }),
            |group| {
                groups.find_or_add_index(&group.id);
            },
        )(input)?;

        verify(success(()), |_| {
            armies.as_slice().len() <= MAX_ARMY_COUNT
                && attack_types.as_slice().len() <= MAX_ATTACK_TYPE_COUNT
                && groups.as_slice().len() <= MAX_GROUP_COUNT
        })(input)?;

        attack_types.sort_by_id();
        groups.sort_by_id();

        let input: &str = Self::parse_internal(
            map(Self::parse_army_id, |army_id| armies.find_index(&army_id)),
            map(Self::parse_attack_type_id, |attack_type_id| {
                attack_types.find_index_binary_search(&attack_type_id)
            }),
            |group| {
                let group_index: GroupIndex = groups.find_index_binary_search(&group.id);

                groups.as_slice_mut()[group_index.get()].data = group.data;
            },
        )(input)?
        .0;

        for (group_index, group) in groups.as_slice().iter().enumerate() {
            armies.as_slice_mut()[group.data.army_index.get()]
                .data
                .groups
                .set(group_index, true);
        }

        Ok((
            input,
            Self {
                armies,
                attack_types,
                groups,
            },
        ))
    }
}

impl RunQuestions for Solution {
    /// Took a while to get this one. I initially had tests that expected specific output from each
    /// phase, but due to the way their output orders groups, things wouldn't match (while still)
    /// being functionally correct.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.winning_army_unit_count());
    }

    /// My user input ran into a case where a stalemate was reached, as the immune system's selected
    /// targets had too many hitpoints per unit to be able to kill off any units. Wasn't expecting
    /// that, but I guess it makes sense.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.immune_system_winning_army_unit_count_with_min_boost());
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
        Immune System:\n\
        17 units each with 5390 hit points (weak to radiation, bludgeoning) with an attack that \
        does 4507 fire damage at initiative 2\n\
        989 units each with 1274 hit points (immune to fire; weak to bludgeoning, slashing) with \
        an attack that does 25 slashing damage at initiative 3\n\
        \n\
        Infection:\n\
        801 units each with 4706 hit points (weak to radiation) with an attack that does 116 \
        bludgeoning damage at initiative 1\n\
        4485 units each with 2961 hit points (immune to radiation; weak to fire, cold) with an \
        attack that does 12 slashing damage at initiative 4\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                armies: vec![
                    Army {
                        id: "Immune System".try_into().unwrap(),
                        data: ArmyData {
                            groups: bitarr_typed!(GroupBitArray; 0, 1, 1, 0),
                        },
                    },
                    Army {
                        id: "Infection".try_into().unwrap(),
                        data: ArmyData {
                            groups: bitarr_typed!(GroupBitArray; 1, 0, 0, 1),
                        },
                    },
                ]
                .try_into()
                .unwrap(),
                attack_types: ["bludgeoning", "cold", "fire", "radiation", "slashing"]
                    .into_iter()
                    .map(AttackTypeId::try_from)
                    .map(Result::unwrap)
                    .collect::<Vec<AttackTypeId>>()
                    .try_into()
                    .unwrap(),
                groups: vec![
                    Group {
                        id: GroupId { initiative: 4_u8 },
                        data: GroupData {
                            army_index: 1_usize.into(),
                            army_group_index: 2_u8,
                            units: 4485_u16,
                            hit_points: 2961_u16,
                            immunities: bitarr_typed!(AttackTypeBitArray; 0, 0, 0, 1, 0),
                            weaknesses: bitarr_typed!(AttackTypeBitArray; 0, 1, 1, 0, 0),
                            attack_damage: 12_u16,
                            attack_type_index: 4_usize.into(),
                        },
                    },
                    Group {
                        id: GroupId { initiative: 3_u8 },
                        data: GroupData {
                            army_index: 0_usize.into(),
                            army_group_index: 2_u8,
                            units: 989_u16,
                            hit_points: 1274_u16,
                            immunities: bitarr_typed!(AttackTypeBitArray; 0, 0, 1, 0, 0),
                            weaknesses: bitarr_typed!(AttackTypeBitArray; 1, 0, 0, 0, 1),
                            attack_damage: 25_u16,
                            attack_type_index: 4_usize.into(),
                        },
                    },
                    Group {
                        id: GroupId { initiative: 2_u8 },
                        data: GroupData {
                            army_index: 0_usize.into(),
                            army_group_index: 1_u8,
                            units: 17_u16,
                            hit_points: 5390_u16,
                            immunities: bitarr_typed!(AttackTypeBitArray; 0, 0, 0, 0, 0),
                            weaknesses: bitarr_typed!(AttackTypeBitArray; 1, 0, 0, 1, 0),
                            attack_damage: 4507_u16,
                            attack_type_index: 2_usize.into(),
                        },
                    },
                    Group {
                        id: GroupId { initiative: 1_u8 },
                        data: GroupData {
                            army_index: 1_usize.into(),
                            army_group_index: 1_u8,
                            units: 801_u16,
                            hit_points: 4706_u16,
                            immunities: bitarr_typed!(AttackTypeBitArray; 0, 0, 0, 0, 0),
                            weaknesses: bitarr_typed!(AttackTypeBitArray; 0, 0, 0, 1, 0),
                            attack_damage: 116_u16,
                            attack_type_index: 0_usize.into(),
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
    fn test_get_target_selection_order_and_valid_targets() {
        #[derive(Debug, Default, PartialEq)]
        struct Output {
            target_selection_order: GroupArrayVec<GroupIndex>,
            valid_targets: GroupBitArray,
        }

        for (index, expected_output) in [Output {
            target_selection_order: [
                GroupIndex::from(3_usize),
                2_usize.into(),
                0_usize.into(),
                1_usize.into(),
            ]
            .into_iter()
            .collect(),
            valid_targets: bitarr_typed!(GroupBitArray; 1, 1, 1, 1),
        }]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);
            let mut real_output: Output = Output::default();

            solution.get_target_selection_order_and_valid_targets(
                &solution.initial_group_states(&solution.no_army_boosts()),
                &mut real_output.target_selection_order,
                &mut real_output.valid_targets,
            );

            assert_eq!(real_output, expected_output);
        }
    }

    #[test]
    fn test_target_selection_phase() {
        for (index, expected_targets) in [[
            GroupIndex::from(1_usize),
            3_usize.into(),
            0_usize.into(),
            2_usize.into(),
        ]
        .into_iter()
        .collect::<GroupArrayVec<GroupIndex>>()]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);
            let mut real_targets: GroupArrayVec<GroupIndex> = GroupArrayVec::new();

            solution.target_selection_phase(
                &solution.initial_group_states(&solution.no_army_boosts()),
                &mut real_targets,
            );

            assert_eq!(real_targets, expected_targets);
        }
    }

    #[test]
    fn test_fight() {
        for (index, expected_solution_state) in [[
            SolutionState {
                group_states: [
                    GroupState::new(4485_u16, 12_u16),
                    GroupState::new(989_u16, 25_u16),
                    GroupState::new(17_u16, 4507_u16),
                    GroupState::new(801_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [2_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(905_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(797_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(761_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(793_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(618_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(789_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(475_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(786_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(333_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(784_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(191_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(783_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(49_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(782_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [1_u8, 2_u8].into_iter().collect(),
            },
            SolutionState {
                group_states: [
                    GroupState::new(4434_u16, 12_u16),
                    GroupState::new(0_u16, 25_u16),
                    GroupState::new(0_u16, 4507_u16),
                    GroupState::new(782_u16, 116_u16),
                ]
                .into_iter()
                .collect(),
                army_group_counts: [0_u8, 2_u8].into_iter().collect(),
            },
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            let mut real_solution_state: SolutionState =
                solution.initial_solution_state(&solution.no_army_boosts());

            for expected_output in expected_solution_state {
                assert_eq!(real_solution_state, expected_output);

                solution.fight(&mut real_solution_state);
            }
        }
    }

    #[test]
    fn test_war() {
        for (index, expected_solution_state) in [SolutionState {
            group_states: [
                GroupState::new(4434_u16, 12_u16),
                GroupState::new(0_u16, 25_u16),
                GroupState::new(0_u16, 4507_u16),
                GroupState::new(782_u16, 116_u16),
            ]
            .into_iter()
            .collect(),
            army_group_counts: [0_u8, 2_u8].into_iter().collect(),
        }]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            let mut real_solution_state: SolutionState =
                solution.initial_solution_state(&solution.no_army_boosts());

            solution.war(&mut real_solution_state);

            assert_eq!(real_solution_state, expected_solution_state);
        }
    }

    #[test]
    fn test_winning_army_unit_count() {
        for (index, winning_army_unit_count) in [5216_u32].into_iter().enumerate() {
            assert_eq!(
                solution(index).winning_army_unit_count(),
                winning_army_unit_count
            );
        }
    }

    #[test]
    fn test_immune_system_winning_army_unit_count_with_min_boost() {
        for (index, immune_system_winning_army_unit_count_with_min_boost) in
            [51_u32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).immune_system_winning_army_unit_count_with_min_boost(),
                immune_system_winning_army_unit_count_with_min_boost
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
