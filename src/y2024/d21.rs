use {
    crate::*,
    bitvec::prelude::*,
    glam::{BVec2, IVec2},
    nom::{
        character::complete::line_ending,
        combinator::{map, opt, verify},
        error::Error,
        multi::many_m_n,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::{cell::RefCell, collections::HashMap, mem::swap, ops::Range, rc::Rc},
    strum::{EnumIter, IntoEnumIterator},
};

/* --- Day 21: Keypad Conundrum ---

As you teleport onto Santa's Reindeer-class starship, The Historians begin to panic: someone from their search party is missing. A quick life-form scan by the ship's computer reveals that when the missing Historian teleported, he arrived in another part of the ship.

The door to that area is locked, but the computer can't open it; it can only be opened by physically typing the door codes (your puzzle input) on the numeric keypad on the door.

The numeric keypad has four rows of buttons: 789, 456, 123, and finally an empty gap followed by 0A. Visually, they are arranged like this:

+---+---+---+
| 7 | 8 | 9 |
+---+---+---+
| 4 | 5 | 6 |
+---+---+---+
| 1 | 2 | 3 |
+---+---+---+
    | 0 | A |
    +---+---+

Unfortunately, the area outside the door is currently depressurized and nobody can go near the door. A robot needs to be sent instead.

The robot has no problem navigating the ship and finding the numeric keypad, but it's not designed for button pushing: it can't be told to push a specific button directly. Instead, it has a robotic arm that can be controlled remotely via a directional keypad.

The directional keypad has two rows of buttons: a gap / ^ (up) / A (activate) on the first row and < (left) / v (down) / > (right) on the second row. Visually, they are arranged like this:

    +---+---+
    | ^ | A |
+---+---+---+
| < | v | > |
+---+---+---+

When the robot arrives at the numeric keypad, its robotic arm is pointed at the A button in the bottom right corner. After that, this directional keypad remote control must be used to maneuver the robotic arm: the up / down / left / right buttons cause it to move its arm one button in that direction, and the A button causes the robot to briefly move forward, pressing the button being aimed at by the robotic arm.

For example, to make the robot type 029A on the numeric keypad, one sequence of inputs on the directional keypad you could use is:

    < to move the arm from A (its initial position) to 0.
    A to push the 0 button.
    ^A to move the arm to the 2 button and push it.
    >^^A to move the arm to the 9 button and push it.
    vvvA to move the arm to the A button and push it.

In total, there are three shortest possible sequences of button presses on this directional keypad that would cause the robot to type 029A: <A^A>^^AvvvA, <A^A^>^AvvvA, and <A^A^^>AvvvA.

Unfortunately, the area containing this directional keypad remote control is currently experiencing high levels of radiation and nobody can go near it. A robot needs to be sent instead.

When the robot arrives at the directional keypad, its robot arm is pointed at the A button in the upper right corner. After that, a second, different directional keypad remote control is used to control this robot (in the same way as the first robot, except that this one is typing on a directional keypad instead of a numeric keypad).

There are multiple shortest possible sequences of directional keypad button presses that would cause this robot to tell the first robot to type 029A on the door. One such sequence is v<<A>>^A<A>AvA<^AA>A<vAAA>^A.

Unfortunately, the area containing this second directional keypad remote control is currently -40 degrees! Another robot will need to be sent to type on that directional keypad, too.

There are many shortest possible sequences of directional keypad button presses that would cause this robot to tell the second robot to tell the first robot to eventually type 029A on the door. One such sequence is <vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A.

Unfortunately, the area containing this third directional keypad remote control is currently full of Historians, so no robots can find a clear path there. Instead, you will have to type this sequence yourself.

Were you to choose this sequence of button presses, here are all of the buttons that would be pressed on your directional keypad, the two robots' directional keypads, and the numeric keypad:

<vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A
v<<A>>^A<A>AvA<^AA>A<vAAA>^A
<A^A>^^AvvvA
029A

In summary, there are the following keypads:

    One directional keypad that you are using.
    Two directional keypads that robots are using.
    One numeric keypad (on a door) that a robot is using.

It is important to remember that these robots are not designed for button pushing. In particular, if a robot arm is ever aimed at a gap where no button is present on the keypad, even for an instant, the robot will panic unrecoverably. So, don't do that. All robots will initially aim at the keypad's A key, wherever it is.

To unlock the door, five codes will need to be typed on its numeric keypad. For example:

029A
980A
179A
456A
379A

For each of these, here is a shortest sequence of button presses you could type to cause the desired code to be typed on the numeric keypad:

029A: <vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A
980A: <v<A>>^AAAvA^A<vA<AA>>^AvAA<^A>A<v<A>A>^AAAvA<^A>A<vA>^A<A>A
179A: <v<A>>^A<vA<A>>^AAvAA<^A>A<v<A>>^AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A
456A: <v<A>>^AA<vA<A>>^AAvAA<^A>A<vA>^A<A>A<vA>^A<A>A<v<A>A>^AAvA<^A>A
379A: <v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A

The Historians are getting nervous; the ship computer doesn't remember whether the missing Historian is trapped in the area containing a giant electromagnet or molten lava. You'll need to make sure that for each of the five codes, you find the shortest sequence of button presses necessary.

The complexity of a single code (like 029A) is equal to the result of multiplying these two values:

    The length of the shortest sequence of button presses you need to type on your directional keypad in order to cause the code to be typed on the numeric keypad; for 029A, this would be 68.
    The numeric part of the code (ignoring leading zeroes); for 029A, this would be 29.

In the above example, complexity of the five codes can be found by calculating 68 * 29, 60 * 980, 68 * 179, 64 * 456, and 64 * 379. Adding these together produces 126384.

Find the fewest number of button presses you'll need to perform in order to cause the robot in front of the door to type each code. What is the sum of the complexities of the five codes on your list?

--- Part Two ---

Just as the missing Historian is released, The Historians realize that a second member of their search party has also been missing this entire time!

A quick life-form scan reveals the Historian is also trapped in a locked area of the ship. Due to a variety of hazards, robots are once again dispatched, forming another chain of remote control keypads managing robotic-arm-wielding robots.

This time, many more robots are involved. In summary, there are the following keypads:

    One directional keypad that you are using.
    25 directional keypads that robots are using.
    One numeric keypad (on a door) that a robot is using.

The keypads form a chain, just like before: your directional keypad controls a robot which is typing on a directional keypad which controls a robot which is typing on a directional keypad... and so on, ending with the robot which is typing on the numeric keypad.

The door codes are the same this time around; only the number of robots and directional keypads has changed.

Find the fewest number of button presses you'll need to perform in order to cause the robot in front of the door to type each code. What is the sum of the complexities of the five codes on your list? */

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, Default, EnumIter, Eq, Hash, Ord, PartialEq, PartialOrd)]
    enum Key {
        #[default]
        Activate = ACTIVATE = b'A',
        Zero = ZERO = b'0',
        One = ONE = b'1',
        Two = TWO = b'2',
        Three = THREE = b'3',
        Four = FOUR = b'4',
        Five = FIVE = b'5',
        Six = SIX = b'6',
        Seven = SEVEN = b'7',
        Eight = EIGHT = b'8',
        Nine = NINE = b'9',
        Up = UP = b'^',
        Down = DOWN = b'v',
        Left = LEFT = b'<',
        Right = RIGHT = b'>',
    }
}

impl Key {
    const ACTIVATE_VEC: IVec2 = IVec2::ZERO;
    const ZERO_OR_UP_VEC: IVec2 = IVec2::NEG_X;
    const DOWN_VEC: IVec2 = IVec2::new(-1_i32, 1_i32);
    const LEFT_VEC: IVec2 = IVec2::new(-2_i32, 1_i32);
    const RIGHT_VEC: IVec2 = IVec2::Y;
    const GAP_VEC: IVec2 = IVec2::new(-2_i32, 0_i32);

    #[allow(unused)]
    const NUMPAD_KEYS: &'static [Key] = &[
        Key::Activate,
        Key::Zero,
        Key::One,
        Key::Two,
        Key::Three,
        Key::Four,
        Key::Five,
        Key::Six,
        Key::Seven,
        Key::Eight,
        Key::Nine,
    ];

    fn parse_digit<'i>(input: &'i str) -> IResult<&'i str, Self> {
        verify(Self::parse, |key| key.is_digit())(input)
    }

    fn parse_activate<'i>(input: &'i str) -> IResult<&'i str, Self> {
        verify(Self::parse, |&key| key == Self::Activate)(input)
    }

    fn string_from_iter<I: Iterator<Item = Key>>(iter: I) -> String {
        iter.map(|key| key as u8 as char).collect()
    }

    fn try_digit(self) -> Option<u8> {
        match self as u8 {
            Self::ZERO..=Self::NINE => Some(self as u8 - Self::ZERO),
            _ => None,
        }
    }

    fn is_digit(self) -> bool {
        self.try_digit().is_some()
    }

    fn vec(self) -> IVec2 {
        match self as u8 {
            Self::ONE..=Self::NINE => {
                let index: i32 = (self as u8 - Self::ONE) as i32;

                IVec2::new(-2_i32, -1_i32) + (index % 3_i32) * IVec2::X + (index / 3) * IVec2::NEG_Y
            }
            Self::ZERO | Self::UP => Self::ZERO_OR_UP_VEC,
            Self::ACTIVATE => Self::ACTIVATE_VEC,
            Self::LEFT => Self::LEFT_VEC,
            Self::DOWN => Self::DOWN_VEC,
            Self::RIGHT => Self::RIGHT_VEC,
            _ => unreachable!(),
        }
    }
}

type KeyPairIndexRaw = u8;
type KeyPairIndex = Index<KeyPairIndexRaw>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct KeyPairConfig(u32);

impl KeyPairConfig {
    /// This was found by iterating over all possible configs, filtering for those that satisfied
    /// all the example constraints, then minimizing over the key count with extra layers for my
    /// personal input. I spent over a day on part 2. Not fun.
    const OPTIMAL: KeyPairConfig = Self(279616_u32);

    #[allow(unused)]
    const FLAGS: u32 = 24_u32;

    #[allow(unused)]
    const COUNT: u32 = 1_u32 << Self::FLAGS;

    fn index(key_pair: KeyPair) -> usize {
        let delta: IVec2 = key_pair.delta();
        let delta_abs: IVec2 = delta.abs();

        if delta_abs.min_element() == 0_i32 {
            // Won't matter, as it'll be a slice of just a single `Key` regardless
            0_usize
        } else {
            let transform: IVec2 = delta
                + IVec2::new(2_i32, 3_i32)
                + (delta.x > 0_i32) as i32 * IVec2::NEG_X
                + (delta.y > 0_i32) as i32 * IVec2::NEG_Y;

            (transform.x + 4_i32 * transform.y) as usize
        }
    }

    fn vertical_first(self, key_pair: KeyPair) -> bool {
        self.0.view_bits::<Lsb0>()[Self::index(key_pair)]
    }
}

type KeyPairCache = Rc<RefCell<HashMap<KeyPair, &'static [Key]>>>;

#[derive(Clone, Copy, Default, Hash, Eq, Ord, PartialEq, PartialOrd)]
struct KeyPair {
    curr: Key,
    next: Key,
}

impl KeyPair {
    const COUNT: usize = 145_usize;

    fn new_cache() -> KeyPairCache {
        Rc::new(RefCell::new(HashMap::new()))
    }

    fn iter() -> impl Iterator<Item = Self> {
        Key::iter()
            .flat_map(|curr| Key::iter().map(move |next| Self { curr, next }))
            .filter(|key_pair| key_pair.is_valid())
    }

    #[allow(unused)]
    fn iter_numpad() -> impl Iterator<Item = Self> {
        Key::NUMPAD_KEYS
            .iter()
            .copied()
            .flat_map(|curr| Key::iter().map(move |next| Self { curr, next }))
    }

    fn iter_from_keys<I: Iterator<Item = Key>>(iter: I) -> impl Iterator<Item = Self> {
        let mut curr: Key = Key::Activate;

        iter.map(move |next| {
            let key_pair: KeyPair = KeyPair { curr, next };

            curr = next;

            key_pair
        })
    }

    fn shortest_sequence_for_iter_and_config<I: Iterator<Item = Key>>(
        iter: I,
        config: KeyPairConfig,
        cache: KeyPairCache,
    ) -> impl Iterator<Item = Key> {
        Self::iter_from_keys(iter).flat_map(move |key_pair| {
            key_pair
                .shortest_sequence_with_config(config, &cache.clone())
                .iter()
                .copied()
                .chain([Key::Activate])
        })
    }

    fn shortest_sequence_for_iter<I: Iterator<Item = Key>>(
        iter: I,
        cache: KeyPairCache,
    ) -> impl Iterator<Item = Key> {
        Self::shortest_sequence_for_iter_and_config(iter, KeyPairConfig::OPTIMAL, cache)
    }

    fn is_valid(self) -> bool {
        self.curr == Key::Activate
            || self.next == Key::Activate
            || (self.curr.is_digit() == self.next.is_digit())
    }

    fn delta(self) -> IVec2 {
        self.next.vec() - self.curr.vec()
    }

    fn shortest_sequence_with_config(
        self,
        config: KeyPairConfig,
        cache: &KeyPairCache,
    ) -> &'static [Key] {
        const KEYS: &'static [Key] = &[
            Key::Up,
            Key::Up,
            Key::Up,
            // 3
            Key::Right,
            Key::Right,
            // 5
            Key::Down,
            Key::Down,
            Key::Down,
            // 8
            Key::Left,
            Key::Left,
            // 10
            Key::Up,
            Key::Up,
            Key::Up,
            // 13
            Key::Left,
            Key::Left,
            // 15
            Key::Down,
            Key::Down,
            Key::Down,
            // 18
            Key::Right,
            Key::Right,
            // 20
            Key::Up,
            Key::Up,
            Key::Up,
        ];

        let shortest_sequence: &'static [Key] = {
            let mut cache = cache.borrow_mut();

            if !cache.contains_key(&self) {
                let curr_vec: IVec2 = self.curr.vec();
                let next_vec: IVec2 = self.next.vec();
                let delta: IVec2 = next_vec - curr_vec;
                let delta_abs: IVec2 = delta.abs();
                let delta_cmpge_zero: BVec2 = delta.cmpge(IVec2::ZERO);
                let min: IVec2 = curr_vec.min(next_vec);
                let max: IVec2 = curr_vec.max(next_vec);
                let vertical_first: bool =
                    if (Key::GAP_VEC.cmpge(min) & Key::GAP_VEC.cmple(max)).all() {
                        (Key::GAP_VEC - curr_vec).y == 0_i32
                    } else {
                        config.vertical_first(self)
                    };
                let key_range: Range<usize> =
                    match (vertical_first, delta_cmpge_zero.x, delta_cmpge_zero.y) {
                        // Left, Up
                        (false, false, false) => {
                            10 - delta_abs.x as usize..10 + delta_abs.y as usize
                        }
                        // Left, Down
                        (false, false, true) => {
                            15 - delta_abs.x as usize..15 + delta_abs.y as usize
                        }
                        // Right, Up
                        (false, true, false) => {
                            20 - delta_abs.x as usize..20 + delta_abs.y as usize
                        }
                        // Right, Down
                        (false, true, true) => 5 - delta_abs.x as usize..5 + delta_abs.y as usize,
                        // Up, Left
                        (true, false, false) => {
                            13 - delta_abs.y as usize..13 + delta_abs.x as usize
                        }
                        // Down, Left
                        (true, false, true) => 8 - delta_abs.y as usize..8 + delta_abs.x as usize,
                        // Up, Right
                        (true, true, false) => 3 - delta_abs.y as usize..3 + delta_abs.x as usize,
                        // Down, Right
                        (true, true, true) => 18 - delta_abs.y as usize..18 + delta_abs.x as usize,
                    };

                cache.insert(self, &KEYS[key_range]);
            }

            cache[&self]
        };

        shortest_sequence
    }
}

#[derive(Default)]
struct KeyPairData {
    key_range: Range<u16>,
}

struct KeyPairTable {
    table: Table<KeyPair, KeyPairData, KeyPairIndexRaw>,
    keys: Vec<Key>,
}

impl KeyPairTable {
    fn new_with_config(config: KeyPairConfig, cache: &KeyPairCache) -> Self {
        let mut key_pair_table: Self = Self {
            table: Table::default(),
            keys: Vec::new(),
        };

        for key_pair in KeyPair::iter() {
            let key_range_start: u16 = key_pair_table.keys.len() as u16;

            key_pair_table.keys.extend(
                key_pair
                    .shortest_sequence_with_config(config, cache)
                    .iter()
                    .copied()
                    .chain([Key::Activate]),
            );

            let key_range_end: u16 = key_pair_table.keys.len() as u16;
            let key_pair_index: KeyPairIndex = key_pair_table.table.find_or_add_index(&key_pair);
            key_pair_table.table.as_slice_mut()[key_pair_index.get()]
                .data
                .key_range = key_range_start..key_range_end;
        }

        key_pair_table.table.sort_by_id();
        key_pair_table
    }

    fn new(cache: &KeyPairCache) -> Self {
        Self::new_with_config(KeyPairConfig::OPTIMAL, cache)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Default)]
struct Code([Key; Self::LEN]);

impl Code {
    const DIGIT_COUNT: usize = 3_usize;
    const LEN: usize = Self::DIGIT_COUNT + 1_usize;
    const EXTRA_NUMPAD_LAYERS: usize = 26_usize;

    fn numeric_part(self) -> usize {
        self.0[..Self::DIGIT_COUNT]
            .iter()
            .fold(0_usize, |numeric_part, key| {
                numeric_part * 10_usize + key.try_digit().unwrap() as usize
            })
    }

    fn iter_depressurized_robot_keys(self) -> impl Iterator<Item = Key> {
        self.0.into_iter()
    }

    fn iter_irradiated_robot_keys(self, cache: KeyPairCache) -> impl Iterator<Item = Key> {
        KeyPair::shortest_sequence_for_iter(self.iter_depressurized_robot_keys(), cache)
    }

    fn iter_cold_robot_keys(self, cache: KeyPairCache) -> impl Iterator<Item = Key> {
        KeyPair::shortest_sequence_for_iter(self.iter_irradiated_robot_keys(cache.clone()), cache)
    }

    fn iter_human_keys(self, cache: KeyPairCache) -> impl Iterator<Item = Key> {
        KeyPair::shortest_sequence_for_iter(self.iter_cold_robot_keys(cache.clone()), cache)
    }

    fn complexity(self, cache: KeyPairCache) -> usize {
        self.iter_human_keys(cache).count() * self.numeric_part()
    }

    fn key_count_with_numpad_layers(
        self,
        key_pair_table: &KeyPairTable,
        numpad_layers: usize,
    ) -> usize {
        let mut key_pair_counts_a: [usize; KeyPair::COUNT] = [0_usize; KeyPair::COUNT];
        let mut key_pair_counts_b: [usize; KeyPair::COUNT] = key_pair_counts_a.clone();
        let mut curr_key_pair_counts: &mut [usize; KeyPair::COUNT] = &mut key_pair_counts_a;
        let mut next_key_pair_counts: &mut [usize; KeyPair::COUNT] = &mut key_pair_counts_b;

        for key_pair in KeyPair::iter_from_keys(self.iter_depressurized_robot_keys()) {
            let key_pair_index: KeyPairIndex =
                key_pair_table.table.find_index_binary_search(&key_pair);

            assert!(key_pair_index.is_valid());

            curr_key_pair_counts[key_pair_index.get()] += 1_usize;
        }

        for _ in 0_usize..numpad_layers {
            next_key_pair_counts.fill(0_usize);

            for (curr_key_pair_index, curr_key_pair_count) in
                curr_key_pair_counts.iter().copied().enumerate()
            {
                if curr_key_pair_count != 0_usize {
                    for next_key_pair in KeyPair::iter_from_keys(
                        key_pair_table.keys[key_pair_table.table.as_slice()[curr_key_pair_index]
                            .data
                            .key_range
                            .as_range_usize()]
                        .iter()
                        .copied(),
                    ) {
                        let next_key_pair_index: KeyPairIndex = key_pair_table
                            .table
                            .find_index_binary_search(&next_key_pair);

                        assert!(next_key_pair_index.is_valid());

                        next_key_pair_counts[next_key_pair_index.get()] += curr_key_pair_count;
                    }
                }
            }

            swap(&mut curr_key_pair_counts, &mut next_key_pair_counts);
        }

        curr_key_pair_counts.iter().copied().sum()
    }

    fn complexity_with_extra_layers(
        self,
        key_pair_table: &KeyPairTable,
        numpad_layers: usize,
    ) -> usize {
        self.key_count_with_numpad_layers(key_pair_table, numpad_layers)
            * self.numeric_part() as usize
    }
}

impl Parse for Code {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut code: Self = Self::default();
        let mut key_index: usize = 0_usize;

        let input: &str = tuple((
            many_m_n(
                Self::DIGIT_COUNT,
                Self::DIGIT_COUNT,
                map(Key::parse_digit, |key| {
                    code.0[key_index] = key;
                    key_index += 1_usize;
                }),
            ),
            Key::parse_activate,
        ))(input)?
        .0;

        Ok((input, code))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution([Code; Self::LEN]);

impl Solution {
    const LEN: usize = 5_usize;

    fn complexity_sum(&self) -> usize {
        let cache: KeyPairCache = KeyPair::new_cache();

        self.0
            .iter()
            .map(|code| code.complexity(cache.clone()))
            .sum()
    }

    fn complexity_sum_with_extra_layers(&self) -> usize {
        let cache: KeyPairCache = KeyPair::new_cache();
        let key_pair_table: KeyPairTable = KeyPairTable::new(&cache);

        self.0
            .iter()
            .map(|code| {
                code.complexity_with_extra_layers(&key_pair_table, Code::EXTRA_NUMPAD_LAYERS)
            })
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();
        let mut code_index: usize = 0_usize;

        let input: &str = many_m_n(
            Self::LEN,
            Self::LEN,
            terminated(
                map(Code::parse, |code| {
                    solution.0[code_index] = code;
                    code_index += 1_usize;
                }),
                opt(line_ending),
            ),
        )(input)?
        .0;

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// JFC this was tough
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.complexity_sum());

        if args.verbose {
            let cache: KeyPairCache = KeyPair::new_cache();

            for code in self.0 {
                dbg!(Key::string_from_iter(code.iter_human_keys(cache.clone())));
            }
        }
    }

    /// Really had to think outside of the box to figure out how to find the correct solution to
    /// this one. `KeyPairConfiguration` took forever to optimize, and many many attempts. It
    /// started out as a `OnceLock` but then enough stuff got packed into it that it was taking too
    /// long to determine. I'm not certain that it's the correct constant for other user inputs,
    /// either, but the method of computing `OPTIMAL` (which is still present in `tests`) should be
    /// portable to other user inputs.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.complexity_sum_with_extra_layers());
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
    #[allow(unused_imports)]
    use {
        super::*,
        rayon::iter::{IntoParallelIterator, ParallelIterator},
        std::{collections::HashSet, sync::OnceLock, time::Instant},
    };

    const SOLUTION_STRS: &'static [&'static str] = &["\
        029A\n\
        980A\n\
        179A\n\
        456A\n\
        379A\n"];
    const COMPLEXITIES: [usize; Solution::LEN] = [
        1972_usize,
        58800_usize,
        12172_usize,
        29184_usize,
        24256_usize,
    ];
    const KEY_COUNTS_WITH_EXTRA_LAYERS: [usize; Solution::LEN] = [
        82050061710_usize,
        72242026390_usize,
        81251039228_usize,
        80786362258_usize,
        77985628636_usize,
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        use Key::{
            Activate as Ac, Eight as K8, Five as K5, Four as K4, Nine as K9, One as K1,
            Seven as K7, Six as K6, Three as K3, Two as K2, Zero as K0,
        };

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution([
                Code([K0, K2, K9, Ac]),
                Code([K9, K8, K0, Ac]),
                Code([K1, K7, K9, Ac]),
                Code([K4, K5, K6, Ac]),
                Code([K3, K7, K9, Ac]),
            ])]
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
    fn test_key_pair_shortest_sequence_for_iter() {
        let code: Code = solution(0_usize).0[0_usize];
        let cache: KeyPairCache = KeyPair::new_cache();
        let key_pair_table: KeyPairTable = KeyPairTable::new(&cache);

        for (actual_keys, expected_keys, numpad_layers) in [
            (
                Key::string_from_iter(code.iter_depressurized_robot_keys()),
                "029A",
                0_usize,
            ),
            (
                Key::string_from_iter(code.iter_irradiated_robot_keys(cache.clone())),
                "<A^A>^^AvvvA",
                1_usize,
            ),
            (
                Key::string_from_iter(code.iter_cold_robot_keys(cache.clone())),
                "v<<A>>^A<A>AvA<^AA>A<vAAA>^A",
                2_usize,
            ),
        ] {
            assert_eq!(
                actual_keys.len(),
                expected_keys.len(),
                "actual:\n{actual_keys}\nexpected:\n{expected_keys}"
            );
            assert_eq!(
                code.key_count_with_numpad_layers(&key_pair_table, numpad_layers),
                expected_keys.len()
            );
        }

        for (index, human_keys_list) in [[
            "<vA<AA>>^AvAA<^A>A<v<A>>^AvA^A<vA>^A<v<A>^A>AAvA^A<v<A>A>^AAAvA<^A>A",
            "<v<A>>^AAAvA^A<vA<AA>>^AvAA<^A>A<v<A>A>^AAAvA<^A>A<vA>^A<A>A",
            "<v<A>>^A<vA<A>>^AAvAA<^A>A<v<A>>^AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A",
            "<v<A>>^AA<vA<A>>^AAvAA<^A>A<vA>^A<A>A<vA>^A<A>A<v<A>A>^AAvA<^A>A",
            "<v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A",
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (code_index, expected_human_keys) in human_keys_list.into_iter().enumerate() {
                let actual_human_keys: String =
                    Key::string_from_iter(solution.0[code_index].iter_human_keys(cache.clone()));

                assert_eq!(
                    actual_human_keys.len(),
                    expected_human_keys.len(),
                    "code_index: {code_index}\n\
                    actual:\n{actual_human_keys}\n\
                    expected:\n{expected_human_keys}"
                );
                assert_eq!(
                    solution.0[code_index].key_count_with_numpad_layers(&key_pair_table, 3_usize),
                    expected_human_keys.len()
                );
            }
        }
    }

    #[test]
    fn test_complexity() {
        let cache: KeyPairCache = KeyPair::new_cache();

        for (index, complexities) in [COMPLEXITIES].into_iter().enumerate() {
            let solution: &Solution = solution(index);

            for (index, complexity) in complexities.into_iter().enumerate() {
                assert_eq!(solution.0[index].complexity(cache.clone()), complexity);
            }
        }
    }

    #[test]
    fn test_complexity_sum() {
        for (index, complexity_sum) in [126384_usize].into_iter().enumerate() {
            assert_eq!(solution(index).complexity_sum(), complexity_sum);
        }
    }

    #[test]
    fn test_key_count_with_numpad_layers() {
        for (index, key_counts_with_numpad_layers) in
            [KEY_COUNTS_WITH_EXTRA_LAYERS].into_iter().enumerate()
        {
            let solution: &Solution = solution(index);
            let cache: KeyPairCache = KeyPair::new_cache();
            let key_pair_table: KeyPairTable = KeyPairTable::new(&cache);

            for (code_index, key_count_with_numpad_layers) in
                key_counts_with_numpad_layers.into_iter().enumerate()
            {
                assert_eq!(
                    solution.0[code_index]
                        .key_count_with_numpad_layers(&key_pair_table, Code::EXTRA_NUMPAD_LAYERS),
                    key_count_with_numpad_layers,
                    "code_index: {code_index}"
                );
            }
        }
    }

    #[test]
    fn test_key_pair_iter_count() {
        assert_eq!(KeyPair::iter().count(), KeyPair::COUNT);
    }

    #[test]
    fn test_find_valid_configs() {
        // const NUMPAD_LAYERS: usize = 3_usize;
        // const EXTRA_NUMPAD_LAYERS: usize = 26_usize;
        // // Change this to your part 1 answer
        // const INPUT_COMPLEXITY_SUM: usize = 179444_usize;

        // let example_solution: &Solution = solution(0_usize);
        // let args: Args = Args::parse(module_path!()).unwrap().1;
        // let input_solution: Solution = args.try_to_intermediate::<Solution>().unwrap();
        // let start: Instant = Instant::now();
        // let config_and_min_complexity_sum_with_extra_layers: Option<(KeyPairConfig, usize)> =
        //     (0_u32..KeyPairConfig::COUNT)
        //         .into_par_iter()
        //         .map(KeyPairConfig)
        //         .filter_map(|config| {
        //             let cache: KeyPairCache = KeyPair::new_cache();
        //             let key_pair_table: KeyPairTable =
        //                 KeyPairTable::new_with_config(config, &cache);

        //             (example_solution
        //                 .0
        //                 .iter()
        //                 .zip(COMPLEXITIES.into_iter().zip(KEY_COUNTS_WITH_EXTRA_LAYERS))
        //                 .all(|(code, (complexity, key_count_with_extra_layers))| {
        //                     code.complexity_with_extra_layers(&key_pair_table, NUMPAD_LAYERS)
        //                         == complexity
        //                         && code.key_count_with_numpad_layers(
        //                             &key_pair_table,
        //                             EXTRA_NUMPAD_LAYERS,
        //                         ) == key_count_with_extra_layers
        //                 })
        //                 && input_solution
        //                     .0
        //                     .iter()
        //                     .map(|code| {
        //                         code.complexity_with_extra_layers(&key_pair_table, NUMPAD_LAYERS)
        //                     })
        //                     .sum::<usize>()
        //                     == INPUT_COMPLEXITY_SUM)
        //                 .then_some((config, key_pair_table))
        //         })
        //         .map(|(config, key_pair_table)| {
        //             let complexity_sum_with_extra_layers: usize = input_solution
        //                 .0
        //                 .iter()
        //                 .map(|code| {
        //                     code.complexity_with_extra_layers(&key_pair_table, EXTRA_NUMPAD_LAYERS)
        //                 })
        //                 .sum();

        //             (config, complexity_sum_with_extra_layers)
        //         })
        //         .min_by_key(|&(_, complexity_sum_with_extra_layers)| {
        //             complexity_sum_with_extra_layers
        //         });
        // let end: Instant = Instant::now();

        // dbg!(config_and_min_complexity_sum_with_extra_layers, end - start);
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
