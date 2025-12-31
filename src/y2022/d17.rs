use {
    crate::*,
    num::integer::lcm,
    std::{collections::VecDeque, mem::transmute},
    strum::EnumCount,
};

#[cfg(test)]
use {std::{iter::Rev, ops::Range, slice::IterMut}, static_assertions::const_assert};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, EnumCount)]
#[repr(usize)]
enum RockType {
    #[default]
    HorizontalLine,
    Plus,
    RightAngle,
    VerticalLine,
    Square,
}

impl RockType {
    const MASKS: [&'static [u16]; RockType::COUNT] = [
        &[0b1111000_u16],
        &[0b10000_u16, 0b111000_u16, 0b10000_u16],
        &[0b111000_u16, 0b100000_u16, 0b100000_u16],
        &[0b1000_u16, 0b1000_u16, 0b1000_u16, 0b1000_u16],
        &[0b11000_u16, 0b11000_u16],
    ];
    const MAX_HEIGHT: usize = Self::max_height();

    const fn mask(self) -> &'static [u16] {
        Self::MASKS[self as usize]
    }

    const fn next_const(self) -> Self {
        // SAFETY: `RockType` has `repr(usize)`, so it shares byte size with `usize`, it starts at
        // a value of 0, and increments until (and excluding) `Rocktype::COUNT`. Taking the modulus
        // of a `usize` using `RockType::COUNT` will map any `usize` to a valid underlying value
        unsafe { transmute((self as usize + 1_usize) % RockType::COUNT) }
    }

    const fn max_height() -> usize {
        let mut max_height: usize = 0_usize;
        let mut index: usize = 0_usize;

        while index < Self::COUNT {
            let height: usize = Self::MASKS[index].len();

            if height > max_height {
                max_height = height;
            }

            index += 1_usize;
        }

        max_height
    }
}

impl Iterator for RockType {
    type Item = Self;

    fn next(&mut self) -> Option<Self::Item> {
        let option: Option<Self> = Some(*self);

        *self = RockType::next_const(*self);

        option
    }
}

#[derive(Clone, Copy, Default)]
struct FallingRock([u16; RockType::MAX_HEIGHT]);

impl FallingRock {
    const WALL_MASK: u16 = 0b100000001_u16;

    fn try_jet_push(mut self, jet_push_direction: JetPushDirection) -> Option<Self> {
        for layer in self.0.iter_mut() {
            *layer = layer.rotate_left(jet_push_direction as u32);

            if *layer & Self::WALL_MASK != 0_u16 {
                return None;
            }
        }

        Some(self)
    }
}

impl From<RockType> for FallingRock {
    fn from(rock_type: RockType) -> Self {
        let mut falling_rock: Self = Self::default();

        let layers: &[u16] = rock_type.mask();

        falling_rock.0[..layers.len()].copy_from_slice(layers);

        falling_rock
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
enum JetPushDirection {
    Left = 15_u32,
    Right = 1_u32,
}

#[derive(Debug, PartialEq)]
pub struct InvalidJetPushDirectionChar(char);

impl TryFrom<char> for JetPushDirection {
    type Error = InvalidJetPushDirectionChar;

    fn try_from(jet_push_direction_char: char) -> Result<Self, Self::Error> {
        use JetPushDirection::*;

        match jet_push_direction_char {
            '<' => Ok(Left),
            '>' => Ok(Right),
            _ => Err(InvalidJetPushDirectionChar(jet_push_direction_char)),
        }
    }
}

#[derive(Debug, PartialEq)]
struct JetPattern(Vec<JetPushDirection>);

impl JetPattern {
    fn iter<'j>(&'j self) -> JetPatternIterator<'j> {
        JetPatternIterator {
            jet_pattern: self,
            index: 0_usize,
        }
    }
}

impl TryFrom<&str> for JetPattern {
    type Error = InvalidJetPushDirectionChar;

    fn try_from(jet_pattern_str: &str) -> Result<Self, Self::Error> {
        let mut jet_pattern: Self = Self(Vec::with_capacity(jet_pattern_str.len()));

        for char in jet_pattern_str.chars() {
            jet_pattern.0.push(char.try_into()?);
        }

        Ok(jet_pattern)
    }
}

struct JetPatternIterator<'j> {
    jet_pattern: &'j JetPattern,
    index: usize,
}

impl<'j> Iterator for JetPatternIterator<'j> {
    type Item = JetPushDirection;

    fn next(&mut self) -> Option<Self::Item> {
        if self.jet_pattern.0.is_empty() {
            None
        } else {
            let option: Option<JetPushDirection> = Some(self.jet_pattern.0[self.index]);

            self.index = (self.index + 1_usize) % self.jet_pattern.0.len();

            option
        }
    }
}

struct FallingRockSimulation<'j> {
    jet_pattern_iterator: JetPatternIterator<'j>,
    resting_layers: VecDeque<u16>,
    popped_layers: usize,
    falling_rock_bottom: usize,
    rock_type: RockType,
    falling_rock: FallingRock,
}

impl<'j> FallingRockSimulation<'j> {
    const VERTICAL_CLEARANCE: usize = 3_usize;
    const MAX_LEN: usize = 1024_usize;

    fn new(jet_pattern: &'j JetPattern) -> Self {
        let mut falling_rock_simulation: FallingRockSimulation = Self {
            jet_pattern_iterator: jet_pattern.iter(),
            resting_layers: VecDeque::new(),
            popped_layers: 0_usize,
            falling_rock_bottom: 0_usize,
            rock_type: Default::default(),
            falling_rock: Default::default(),
        };

        falling_rock_simulation.drop_new_rock();

        falling_rock_simulation
    }

    fn drop_new_rock(&mut self) {
        self.falling_rock_bottom = self.resting_layers.len() + Self::VERTICAL_CLEARANCE;
        self.falling_rock = self.rock_type.next().unwrap().into();
    }

    fn do_layers_overlap(&self, falling_rock_bottom: usize, falling_rock: FallingRock) -> bool {
        self.resting_layers
            .iter()
            .skip(falling_rock_bottom)
            .copied()
            .zip(falling_rock.0.iter().copied())
            .any(|(resting_layer, falling_layer)| (resting_layer & falling_layer) != 0_u16)
    }

    fn try_jet_push(&mut self) -> bool {
        if let Some(falling_rock) = self
            .jet_pattern_iterator
            .next()
            .and_then(|jet_push_direction| self.falling_rock.try_jet_push(jet_push_direction))
            .filter(|falling_rock| !self.do_layers_overlap(self.falling_rock_bottom, *falling_rock))
        {
            self.falling_rock = falling_rock;

            true
        } else {
            false
        }
    }

    fn falling_layer_count(&self) -> usize {
        self.falling_rock
            .0
            .iter()
            .position(|falling_layer| *falling_layer == 0_u16)
            .unwrap_or(self.falling_rock.0.len())
    }

    fn integrate_falling_rock(&mut self) {
        let falling_layer_count: usize = self.falling_layer_count();
        let mut falling_rock_top: usize = self.falling_rock_bottom + falling_layer_count;

        if self.resting_layers.len() < falling_rock_top {
            if falling_rock_top > Self::MAX_LEN {
                let layers_to_pop: usize = falling_rock_top - Self::MAX_LEN;

                for _ in 0_usize..layers_to_pop {
                    self.resting_layers.pop_front();
                }

                self.popped_layers += layers_to_pop;
                self.falling_rock_bottom -= layers_to_pop;
                falling_rock_top = Self::MAX_LEN;
            }

            self.resting_layers.resize(falling_rock_top, 0_u16);
        }

        for (resting_layer, falling_layer) in self
            .resting_layers
            .iter_mut()
            .take(falling_rock_top)
            .skip(self.falling_rock_bottom)
            .zip(self.falling_rock.0[..falling_layer_count].iter())
        {
            *resting_layer |= *falling_layer;
        }

        self.drop_new_rock();
    }

    fn try_fall_down(&mut self) -> bool {
        if let Some(falling_rock_bottom) =
            self.falling_rock_bottom
                .checked_sub(1_usize)
                .filter(|falling_rock_bottom| {
                    !self.do_layers_overlap(*falling_rock_bottom, self.falling_rock)
                })
        {
            self.falling_rock_bottom = falling_rock_bottom;

            true
        } else {
            self.integrate_falling_rock();

            false
        }
    }

    fn run_until_rock_rests(&mut self) {
        while {
            self.try_jet_push();

            self.try_fall_down()
        } {}
    }

    fn run_until_n_rocks_rest(&mut self, n: usize) {
        for _ in 0_usize..n {
            self.run_until_rock_rests();
        }
    }

    fn resting_layer_count(&self) -> usize {
        self.resting_layers.len() + self.popped_layers
    }

    fn resting_layer_count_at_large_n(jet_pattern: &JetPattern, n: usize, status_updates: bool) -> Option<usize> {
        // These needed *significant* adjustment from their example input values to produce a non-
        // `None` result for my input
        const WARMUP_CYCLES: usize = 1024;
        const VERIFICATION_CYCLES: usize = 1024;

        let cycle_rocks: usize = lcm(jet_pattern.0.len(), RockType::COUNT);

        let mut falling_rock_simulation: FallingRockSimulation =
            FallingRockSimulation::new(jet_pattern);

        if n < cycle_rocks * (WARMUP_CYCLES + VERIFICATION_CYCLES) {
            falling_rock_simulation.run_until_n_rocks_rest(n);

            Some(falling_rock_simulation.resting_layer_count())
        } else {
            let mut layers_deltas: Vec<usize> =
                Vec::with_capacity(VERIFICATION_CYCLES);
            let mut rocks: usize = 0_usize;
            let mut layers: usize = 0_usize;
            let mut old_layers: usize = 0_usize;

            // Warm it up without tracking the deltas
            for _ in 0_usize..WARMUP_CYCLES {
                falling_rock_simulation.run_until_n_rocks_rest(cycle_rocks);
                rocks += cycle_rocks;

                let new_layers: usize = falling_rock_simulation.resting_layer_count();

                layers += new_layers - old_layers;
                old_layers = new_layers;

                if status_updates {
                    println!("{rocks} rocks now resting, forming {layers} layers");
                }
            }

            let mut old_delta: usize = 0_usize;

            // Run multiple cycles, tracking the deltas
            for _ in 0_usize..VERIFICATION_CYCLES {
                falling_rock_simulation.run_until_n_rocks_rest(cycle_rocks);
                rocks += cycle_rocks;

                let new_layers: usize = falling_rock_simulation.resting_layer_count();
                let layers_delta: usize =
                    new_layers - old_layers;

                layers_deltas.push(layers_delta);
                layers += layers_delta;
                old_layers = new_layers;

                if status_updates {
                    println!("{rocks} rocks now resting, forming {layers} layers with a delta of {layers_delta} (delta delta {})", (layers_delta as isize) - (old_delta as isize));
                }

                old_delta = layers_delta;
            }

            let period: Option<usize> = 'option: loop {
                for period in 1_usize..VERIFICATION_CYCLES / 2_usize {
                    let reference_cycle: &[usize] = &layers_deltas[..period];

                    if layers_deltas[period..]
                        .chunks(period)
                        .all(|cycle| {
                            cycle
                                .iter()
                                .copied()
                                .zip(reference_cycle.iter().copied())
                                .all(|(delta, reference_delta)| delta == reference_delta)
                        })
                    {
                        break 'option Some(period);
                    }
                }

                break 'option None;
            };
            period.map(|period| {
                let partial_period_cycles: usize = VERIFICATION_CYCLES % period;

                if partial_period_cycles != 0_usize {
                    for _ in 0_usize..period - partial_period_cycles {
                        falling_rock_simulation.run_until_n_rocks_rest(cycle_rocks);
                        rocks += cycle_rocks;
        
                        let new_layers: usize = falling_rock_simulation.resting_layer_count();
        
                        layers += new_layers - old_layers;
                        old_layers = new_layers;

                        if status_updates {
                            println!("{rocks} rocks now resting, forming {layers} layers");
                        }
                    }
                }

                let period_rocks: usize = period * cycle_rocks;
                let period_layers: usize = layers_deltas[..period].iter().sum();

                let mut rocks_remaining: usize = n - rocks;
    
                let periods_remaining: usize = rocks_remaining / period_rocks;
                let extrapolated_rocks: usize = periods_remaining * period_rocks;
                let extrapolated_layers: usize = periods_remaining * period_layers;
    
                rocks += extrapolated_rocks;
                layers += extrapolated_layers;

                if status_updates {
                    println!(
                        "Extrapolated {extrapolated_rocks} rocks, forming {extrapolated_layers} layers,\n\
                        {rocks} rocks now resting, forming {layers} layers"
                    );
                }

                rocks_remaining -= extrapolated_rocks;
                falling_rock_simulation.run_until_n_rocks_rest(rocks_remaining);
                rocks += rocks_remaining;
                layers +=
                    falling_rock_simulation.resting_layer_count() - old_layers;
        
                assert_eq!(
                    rocks, n,
                    "We should have {n} rocks remaining at this point"
                );

                if status_updates {
                    println!("{rocks} rocks now resting, forming {layers} layers");
                }

                layers
            })
        }
    }

    #[cfg(test)]
    fn string(&self) -> String {
        const LAYER_BYTES: usize = 10_usize;

        let falling_layer_count: usize = self.falling_layer_count();
        let layer_count: usize = self
            .resting_layers
            .len()
            .max(self.falling_rock_bottom + falling_layer_count);
        let byte_count: usize = (layer_count + 1_usize) * LAYER_BYTES - 1_usize;

        let mut bytes: Vec<u8> = Vec::with_capacity(byte_count);

        bytes.resize(byte_count, b'.');

        const_assert!(LAYER_BYTES >= 2_usize);

        let falling_layer_range: Range<usize> =
            self.falling_rock_bottom..self.falling_rock_bottom + falling_layer_count;

        for (layer_index, layer_bytes) in bytes.chunks_exact_mut(LAYER_BYTES).rev().enumerate() {
            let print_layer = |layer_bytes: &mut [u8], layer: u16, byte: u8| {
                let mut mask: u16 = 1_u16;

                for layer_byte in layer_bytes.iter_mut() {
                    if layer & mask != 0_u16 {
                        *layer_byte = byte;
                    }

                    mask <<= 1_u16;
                }
            };

            layer_bytes[0_usize] = b'|';

            let mut rev_iter: Rev<IterMut<u8>> = layer_bytes.iter_mut().rev();

            *rev_iter.next().unwrap() = b'\n';
            *rev_iter.next().unwrap() = b'|';

            if let Some(resting_layer) = self.resting_layers.get(layer_index).copied() {
                print_layer(layer_bytes, resting_layer, b'#');
            }

            if falling_layer_range.contains(&layer_index) {
                print_layer(
                    layer_bytes,
                    self.falling_rock.0[layer_index - self.falling_rock_bottom],
                    b'@',
                );
            }
        }

        let last_bytes_start: usize = bytes.len() - LAYER_BYTES + 1_usize;
        let last_bytes: &mut [u8] = &mut bytes[last_bytes_start..];

        last_bytes.fill(b'-');
        *last_bytes.first_mut().unwrap() = b'+';
        *last_bytes.last_mut().unwrap() = b'+';

        String::from_utf8(bytes).unwrap()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(JetPattern);

impl Solution {
    fn resting_layer_count_when_n_rocks_rest(&self, n: usize) -> usize {
        let mut falling_rock_simulation: FallingRockSimulation = FallingRockSimulation::new(&self.0);

        falling_rock_simulation
        .run_until_n_rocks_rest(n);

        falling_rock_simulation.resting_layer_count()
    }

    fn resting_layer_count_when_large_n_rocks_rest(&self, n: usize, status_updates: bool) -> Option<usize> {
        FallingRockSimulation::resting_layer_count_at_large_n(
            &self.0,
            n,
            status_updates
        )
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.resting_layer_count_when_n_rocks_rest(2022_usize));
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.resting_layer_count_when_large_n_rocks_rest(
            1_000_000_000_000_usize,
            args.verbose));
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = InvalidJetPushDirectionChar;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const JET_PATTERN_STR: &str = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>";
    const FALLING_ROCK_SIMULATION_STRING_0_ROCKS: &str = concat!(
        "|..@@@@.|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "+-------+",
    );
    const FALLING_ROCK_SIMULATION_STRING_1_ROCK: &str = concat!(
        "|...@...|\n",
        "|..@@@..|\n",
        "|...@...|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|..####.|\n",
        "+-------+",
    );
    const FALLING_ROCK_SIMULATION_STRING_2_ROCKS: &str = concat!(
        "|....@..|\n",
        "|....@..|\n",
        "|..@@@..|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|...#...|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+",
    );
    const FALLING_ROCK_SIMULATION_STRING_3_ROCKS_THROUGH_10_ROCKS: &str = concat!(
        "|..@....|\n",
        "|..@....|\n",
        "|..@....|\n",
        "|..@....|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|..#....|\n",
        "|..#....|\n",
        "|####...|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|..@@...|\n",
        "|..@@...|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|..@@@@.|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|...@...|\n",
        "|..@@@..|\n",
        "|...@...|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|.####..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|....@..|\n",
        "|....@..|\n",
        "|..@@@..|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|..#....|\n",
        "|.###...|\n",
        "|..#....|\n",
        "|.####..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|..@....|\n",
        "|..@....|\n",
        "|..@....|\n",
        "|..@....|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|.....#.|\n",
        "|.....#.|\n",
        "|..####.|\n",
        "|.###...|\n",
        "|..#....|\n",
        "|.####..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|..@@...|\n",
        "|..@@...|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|....#..|\n",
        "|....#..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|..####.|\n",
        "|.###...|\n",
        "|..#....|\n",
        "|.####..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+\n",
        "\n",
        "|..@@@@.|\n",
        "|.......|\n",
        "|.......|\n",
        "|.......|\n",
        "|....#..|\n",
        "|....#..|\n",
        "|....##.|\n",
        "|##..##.|\n",
        "|######.|\n",
        "|.###...|\n",
        "|..#....|\n",
        "|.####..|\n",
        "|....##.|\n",
        "|....##.|\n",
        "|....#..|\n",
        "|..#.#..|\n",
        "|..#.#..|\n",
        "|#####..|\n",
        "|..###..|\n",
        "|...#...|\n",
        "|..####.|\n",
        "+-------+",
    );

    fn jet_pattern() -> &'static JetPattern {
        use {JetPushDirection::Left as L, JetPushDirection::Right as R};

        static ONCE_LOCK: OnceLock<JetPattern> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| JetPattern(vec![
            R, R, R, L, L, R, L, R, R, L, L, L, R, R, L, R, R, R, L, L, L, R, R, R, L, L, L, R, L,
            L, L, R, R, L, R, R, L, L, R, R,
        ]))
    }

    #[test]
    fn test_jet_pattern_try_from_str() {
        assert_eq!(JET_PATTERN_STR.try_into().as_ref(), Ok(jet_pattern()));
    }

    #[test]
    fn test_falling_rock_simulation() {
        let mut falling_rock_simulation: FallingRockSimulation =
            FallingRockSimulation::new(jet_pattern());

        assert_eq!(
            falling_rock_simulation.string(),
            FALLING_ROCK_SIMULATION_STRING_0_ROCKS
        );
        assert!(falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(!falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(!falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(falling_rock_simulation.try_jet_push());
        assert!(!falling_rock_simulation.try_fall_down());
        assert_eq!(
            falling_rock_simulation.string(),
            FALLING_ROCK_SIMULATION_STRING_1_ROCK
        );
        assert!(falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(falling_rock_simulation.try_jet_push());
        assert!(falling_rock_simulation.try_fall_down());
        assert!(falling_rock_simulation.try_jet_push());
        assert!(!falling_rock_simulation.try_fall_down());
        assert_eq!(
            falling_rock_simulation.string(),
            FALLING_ROCK_SIMULATION_STRING_2_ROCKS
        );

        for (index, expected) in FALLING_ROCK_SIMULATION_STRING_3_ROCKS_THROUGH_10_ROCKS
            .split("\n\n")
            .enumerate()
        {
            falling_rock_simulation.run_until_rock_rests();

            let actual: String = falling_rock_simulation.string();

            if actual != expected {
                panic!(
                    "After {} rocks have fallen, the actual doesn't match the expectetd!\n\
                    \n\
                    actual:\n\
                    \n\
                    {}\n\
                    \n\
                    expected:\n\
                    \n\
                    {}",
                    index + 3_usize,
                    actual,
                    expected
                );
            }
        }

        falling_rock_simulation.run_until_n_rocks_rest(2022_usize - 10_usize);

        assert_eq!(falling_rock_simulation.resting_layer_count(), 3068_usize);
    }

    #[test]
    fn test_falling_rock_simulation_one_trillion_rocks() {
        assert_eq!(
            FallingRockSimulation::resting_layer_count_at_large_n(
                jet_pattern(),
                1_000_000_000_000_usize,
                true
            ),
            Some(1_514_285_714_288_usize)
        );
    }
}
