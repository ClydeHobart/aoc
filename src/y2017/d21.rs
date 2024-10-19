use {
    crate::*,
    bitvec::{prelude::*, view::BitView},
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{cond, map, map_opt, opt},
        error::Error,
        multi::{many0, many_m_n},
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
    num::Integer,
    static_assertions::const_assert,
    std::{cmp::Ordering, mem::swap},
    strum::EnumCount,
};

/* --- Day 21: Fractal Art ---

You find a program trying to generate some art. It uses a strange process that involves repeatedly enhancing the detail of an image through a set of rules.

The image consists of a two-dimensional square grid of pixels that are either on (#) or off (.). The program always begins with this pattern:

.#.
..#
###

Because the pattern is both 3 pixels wide and 3 pixels tall, it is said to have a size of 3.

Then, the program repeats the following process:

    If the size is evenly divisible by 2, break the pixels up into 2x2 squares, and convert each 2x2 square into a 3x3 square by following the corresponding enhancement rule.
    Otherwise, the size is evenly divisible by 3; break the pixels up into 3x3 squares, and convert each 3x3 square into a 4x4 square by following the corresponding enhancement rule.

Because each square of pixels is replaced by a larger one, the image gains pixels and so its size increases.

The artist's book of enhancement rules is nearby (your puzzle input); however, it seems to be missing rules. The artist explains that sometimes, one must rotate or flip the input pattern to find a match. (Never rotate or flip the output pattern, though.) Each pattern is written concisely: rows are listed as single units, ordered top-down, and separated by slashes. For example, the following rules correspond to the adjacent patterns:

../.#  =  ..
          .#

                .#.
.#./..#/###  =  ..#
                ###

                        #..#
#..#/..../#..#/.##.  =  ....
                        #..#
                        .##.

When searching for a rule to use, rotate and flip the pattern as necessary. For example, all of the following patterns match the same rule:

.#.   .#.   #..   ###
..#   #..   #.#   ..#
###   ###   ##.   .#.

Suppose the book contained the following two rules:

../.# => ##./#../...
.#./..#/### => #..#/..../..../#..#

As before, the program begins with this pattern:

.#.
..#
###

The size of the grid (3) is not divisible by 2, but it is divisible by 3. It divides evenly into a single square; the square matches the second rule, which produces:

#..#
....
....
#..#

The size of this enhanced grid (4) is evenly divisible by 2, so that rule is used. It divides evenly into four squares:

#.|.#
..|..
--+--
..|..
#.|.#

Each of these squares matches the same rule (../.# => ##./#../...), three of which require some flipping and rotation to line up with the rule. The output for the rule is the same in all four cases:

##.|##.
#..|#..
...|...
---+---
##.|##.
#..|#..
...|...

Finally, the squares are joined into a new grid:

##.##.
#..#..
......
##.##.
#..#..
......

Thus, after 2 iterations, the grid contains 12 pixels that are on.

How many pixels stay on after 5 iterations?

--- Part Two ---

How many pixels stay on after 18 iterations? */

define_super_trait! {
    trait PatternElementTrait where Self: Copy + Default + PartialEq {}
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
struct Pattern<T: PatternElementTrait, const SIDE_LEN: usize>([[T; SIDE_LEN]; SIDE_LEN]);

impl<T: PatternElementTrait, const SIDE_LEN: usize> Pattern<T, SIDE_LEN> {
    const fn side_len() -> i32 {
        SIDE_LEN as i32
    }

    const fn side_len_non_static(&self) -> i32 {
        Self::side_len()
    }

    fn can_get_grid_at_offset(grid: &Grid2D<T>, offset: IVec2) -> bool {
        grid.contains(offset) && grid.contains(offset + (Self::side_len() - 1_i32) * IVec2::ONE)
    }

    fn iter_pos_at_offset(offset: IVec2) -> impl Iterator<Item = IVec2> {
        let side_len: i32 = Self::side_len();

        (side_len > 0_i32)
            .then(|| {
                let y_delta: IVec2 = side_len * IVec2::Y;
                let x_delta: IVec2 = side_len * IVec2::X;

                CellIter2D::try_from(offset..offset + y_delta)
                    .unwrap()
                    .flat_map(move |row_pos| {
                        CellIter2D::try_from(row_pos..row_pos + x_delta).unwrap()
                    })
            })
            .into_iter()
            .flatten()
    }

    fn try_from_grid_at_offset(grid: &Grid2D<T>, offset: IVec2) -> Option<Self> {
        Self::can_get_grid_at_offset(grid, offset).then_some(())?;

        let mut pattern: Self = Self::default();

        for (pos, element) in Self::iter_pos_at_offset(offset).zip(pattern.iter_mut()) {
            *element = *grid.get(pos)?;
        }

        Some(pattern)
    }

    fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.0.iter().flatten().copied()
    }

    fn apply_map(&self, map: &Pattern<IVec2, SIDE_LEN>) -> Self {
        let mut result: Self = Self::default();

        for (value, pos) in self.iter().zip(map.iter()) {
            *result.get_mut(pos) = value;
        }

        result
    }

    fn flip(&self) -> Self {
        self.apply_map(&Pattern::<IVec2, SIDE_LEN>::FLIP_MAP)
    }

    fn rotate_1(&self) -> Self {
        self.apply_map(&Pattern::<IVec2, SIDE_LEN>::ROTATE_1_MAP)
    }

    fn rotate_2(&self) -> Self {
        self.apply_map(&Pattern::<IVec2, SIDE_LEN>::ROTATE_2_MAP)
    }

    fn rotate_3(&self) -> Self {
        self.apply_map(&Pattern::<IVec2, SIDE_LEN>::ROTATE_3_MAP)
    }

    fn should_flip(&self) -> bool {
        let self_flipped: Self = self.flip();

        self_flipped != *self
            && self_flipped.rotate_1() != *self
            && self_flipped.rotate_2() != *self
            && self_flipped.rotate_3() != *self
    }

    fn should_rotate_1(&self) -> bool {
        self.rotate_1() != *self
    }

    fn should_rotate_2(&self) -> bool {
        self.rotate_2() != *self
    }

    fn should_rotate_3(&self) -> bool {
        self.rotate_3() != *self
    }

    fn try_to_grid_at_offset(&self, offset: IVec2, grid: &mut Grid2D<T>) -> Option<()> {
        Self::can_get_grid_at_offset(grid, offset).then_some(())?;

        for (pos, value) in Self::iter_pos_at_offset(offset).zip(self.iter()) {
            *grid.get_mut(pos).unwrap() = value;
        }

        Some(())
    }

    fn get_mut(&mut self, pos: IVec2) -> &mut T {
        &mut self.0[pos.y as usize][pos.x as usize]
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut().flatten()
    }
}

impl<const SIDE_LEN: usize> Pattern<IVec2, SIDE_LEN> {
    const FLIP_MAP: Self = Self::flip_map();
    const ROTATE_1_MAP: Self = Self::rotate_1_map();
    const ROTATE_2_MAP: Self = Self::rotate_2_map();
    const ROTATE_3_MAP: Self = Self::rotate_3_map();

    const fn flip_map() -> Self {
        let mut map: Self = Self([[IVec2::ZERO; SIDE_LEN]; SIDE_LEN]);
        let mut pos: IVec2 = IVec2::ZERO;

        while (pos.y as usize) < SIDE_LEN {
            pos.x = 0_i32;

            while (pos.x as usize) < SIDE_LEN {
                map.0[pos.y as usize][pos.x as usize] = IVec2::new(pos.y, pos.x);

                pos.x += 1_i32;
            }

            pos.y += 1_i32;
        }

        map
    }

    const fn rotate_1_map() -> Self {
        let side_len_minus_one: i32 = SIDE_LEN as i32 - 1_i32;

        let mut map: Self = Self([[IVec2::ZERO; SIDE_LEN]; SIDE_LEN]);
        let mut pos: IVec2 = IVec2::ZERO;

        while (pos.y as usize) < SIDE_LEN {
            pos.x = 0_i32;

            while (pos.x as usize) < SIDE_LEN {
                map.0[pos.y as usize][pos.x as usize] =
                    IVec2::new(side_len_minus_one - pos.y, pos.x);

                pos.x += 1_i32;
            }

            pos.y += 1_i32;
        }

        map
    }

    const fn rotate_2_map() -> Self {
        let side_len_minus_one: i32 = SIDE_LEN as i32 - 1_i32;

        let mut map: Self = Self([[IVec2::ZERO; SIDE_LEN]; SIDE_LEN]);
        let mut pos: IVec2 = IVec2::ZERO;

        while (pos.y as usize) < SIDE_LEN {
            pos.x = 0_i32;

            while (pos.x as usize) < SIDE_LEN {
                map.0[pos.y as usize][pos.x as usize] =
                    IVec2::new(side_len_minus_one - pos.x, side_len_minus_one - pos.y);

                pos.x += 1_i32;
            }

            pos.y += 1_i32;
        }

        map
    }

    const fn rotate_3_map() -> Self {
        let side_len_minus_one: i32 = SIDE_LEN as i32 - 1_i32;

        let mut map: Self = Self([[IVec2::ZERO; SIDE_LEN]; SIDE_LEN]);
        let mut pos: IVec2 = IVec2::ZERO;

        while (pos.y as usize) < SIDE_LEN {
            pos.x = 0_i32;

            while (pos.x as usize) < SIDE_LEN {
                map.0[pos.y as usize][pos.x as usize] =
                    IVec2::new(pos.y, side_len_minus_one - pos.x);

                pos.x += 1_i32;
            }

            pos.y += 1_i32;
        }

        map
    }
}

impl<const SIDE_LEN: usize> Pattern<Pixel, SIDE_LEN> {
    fn from_bits<B: BitView>(bits: B) -> Self {
        bits.view_bits::<Lsb0>()
            .iter()
            .map(|bit| Pixel::from(*bit))
            .collect()
    }

    fn light_count(&self) -> usize {
        self.0
            .iter()
            .flatten()
            .filter(|pixel| pixel.is_light())
            .count()
    }
}

impl<T: PatternElementTrait, const SIDE_LEN: usize> FromIterator<T> for Pattern<T, SIDE_LEN>
where
    Self: Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut pattern: Self = Self::default();

        for (value, element) in iter.into_iter().zip(pattern.iter_mut()) {
            *element = value;
        }

        pattern
    }
}

impl<T: PatternElementTrait, const SIDE_LEN: usize> Default for Pattern<T, SIDE_LEN> {
    fn default() -> Self {
        Self([[T::default(); SIDE_LEN]; SIDE_LEN])
    }
}

impl<T: PatternElementTrait + Parse, const SIDE_LEN: usize> Parse for Pattern<T, SIDE_LEN>
where
    Self: Default,
{
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut pattern: Self = Self::default();
        let mut row: usize = 0_usize;

        let side_len: usize = pattern.0.len();
        let input: &str = many_m_n(side_len, side_len, |input| {
            let row_slice: &mut [T] = &mut pattern.0[row];
            let mut col = 0_usize;

            let input: &str = preceded(
                cond(row != 0_usize, tag("/")),
                many_m_n(
                    side_len,
                    side_len,
                    map(T::parse, |value| {
                        row_slice[col] = value;

                        col += 1_usize;
                    }),
                ),
            )(input)?
            .0;

            row += 1_usize;

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, pattern))
    }
}

#[derive(EnumCount)]
enum EnhancementRuleFlag {
    ShouldFlip,
    ShouldRotate1,
    ShouldRotate2,
    ShouldRotate3,
}

type EnhancementRuleFlagsBitArr = BitArr!(for EnhancementRuleFlag::COUNT, in u8);

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, PartialEq)]
struct EnhancementRuleFlags(EnhancementRuleFlagsBitArr);

impl EnhancementRuleFlags {
    fn new<const SIDE_LEN: usize>(pattern: &Pattern<Pixel, SIDE_LEN>) -> Self {
        use EnhancementRuleFlag::*;

        let mut enhancement_rule_flags: Self = Self(BitArray::ZERO);

        enhancement_rule_flags
            .0
            .set(ShouldFlip as usize, pattern.should_flip());
        enhancement_rule_flags
            .0
            .set(ShouldRotate1 as usize, pattern.should_rotate_1());
        enhancement_rule_flags
            .0
            .set(ShouldRotate2 as usize, pattern.should_rotate_2());
        enhancement_rule_flags
            .0
            .set(ShouldRotate3 as usize, pattern.should_rotate_3());

        enhancement_rule_flags
    }

    fn should_flip(self) -> bool {
        self.0[EnhancementRuleFlag::ShouldFlip as usize]
    }

    fn should_rotate_1(self) -> bool {
        self.0[EnhancementRuleFlag::ShouldRotate1 as usize]
    }

    fn should_rotate_2(self) -> bool {
        self.0[EnhancementRuleFlag::ShouldRotate2 as usize]
    }

    fn should_rotate_3(self) -> bool {
        self.0[EnhancementRuleFlag::ShouldRotate3 as usize]
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Eq, PartialEq)]
struct EnhancementRule<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize> {
    input: Pattern<Pixel, INPUT_SIDE_LEN>,
    output: Pattern<Pixel, OUTPUT_SIDE_LEN>,
    light_count: u8,
    flags: EnhancementRuleFlags,
}

impl<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize>
    EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>
{
    fn matches_no_flip(&self, pattern: &Pattern<Pixel, INPUT_SIDE_LEN>) -> bool {
        *pattern == self.input
            || self.flags.should_rotate_1() && pattern.rotate_1() == self.input
            || self.flags.should_rotate_2() && pattern.rotate_2() == self.input
            || self.flags.should_rotate_3() && pattern.rotate_3() == self.input
    }

    fn matches(&self, pattern: &Pattern<Pixel, INPUT_SIDE_LEN>) -> bool {
        self.matches_no_flip(pattern)
            || self.flags.should_flip() && self.matches_no_flip(&pattern.flip())
    }
}

impl<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize> Ord
    for EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.light_count.cmp(&other.light_count).then_with(|| {
            self.input
                .cmp(&other.input)
                .then_with(|| self.output.cmp(&other.output))
        })
    }
}

impl<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize> PartialOrd
    for EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize> Parse
    for EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>
where
    Pattern<Pixel, INPUT_SIDE_LEN>: Parse,
    Pattern<Pixel, OUTPUT_SIDE_LEN>: Parse,
{
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(
            tuple((Pattern::parse, tag(" => "), Pattern::parse)),
            |(input, _, output)| {
                let light_count: u8 = input.light_count().try_into().ok()?;
                let flags: EnhancementRuleFlags = EnhancementRuleFlags::new(&input);

                Some(Self {
                    input,
                    output,
                    light_count,
                    flags,
                })
            },
        )(input)
    }
}

type SmallEnhancementRule =
    EnhancementRule<{ Solution::SMALL_INPUT_SIDE_LEN }, { Solution::SMALL_OUTPUT_SIDE_LEN }>;
type LargeEnhancementRule =
    EnhancementRule<{ Solution::LARGE_INPUT_SIDE_LEN }, { Solution::LARGE_OUTPUT_SIDE_LEN }>;

struct Grid {
    curr_grid: Grid2D<Pixel>,
    next_grid: Grid2D<Pixel>,
}

impl Grid {
    const INITIAL_PATTERN: Pattern<Pixel, { Solution::LARGE_INPUT_SIDE_LEN }> = Pattern({
        use pixel::*;

        [[D, L, D], [D, D, L], [L, L, L]]
    });

    fn new() -> Self {
        let mut curr_grid: Grid2D<Pixel> =
            Grid2D::default(Self::INITIAL_PATTERN.side_len_non_static() * IVec2::ONE);

        Self::INITIAL_PATTERN.try_to_grid_at_offset(IVec2::ZERO, &mut curr_grid);

        let next_grid: Grid2D<Pixel> = Grid2D::default(IVec2::ZERO);

        Self {
            curr_grid,
            next_grid,
        }
    }

    fn light_count(&self) -> usize {
        self.curr_grid
            .cells()
            .iter()
            .filter(|pixel| pixel.is_light())
            .count()
    }
}

struct PatternCounts {
    curr_counts: Vec<usize>,
    next_counts: Vec<usize>,
}

// type SmallInputPattern = Pattern<Pixel, { Solution::SMALL_INPUT_SIDE_LEN }>;
type LargeInputPattern = Pattern<Pixel, { Solution::LARGE_INPUT_SIDE_LEN }>;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    small_enhancement_rules: Vec<SmallEnhancementRule>,
    large_enhancement_rules: Vec<LargeEnhancementRule>,
}

impl Solution {
    const SMALL_INPUT_SIDE_LEN: usize = 2_usize;
    const SMALL_OUTPUT_SIDE_LEN: usize = Self::SMALL_INPUT_SIDE_LEN + 1_usize;
    const LARGE_INPUT_SIDE_LEN: usize = 3_usize;
    const LARGE_OUTPUT_SIDE_LEN: usize = Self::LARGE_INPUT_SIDE_LEN + 1_usize;
    const SMALL_ENHANCEMENT_COUNT: usize = 5_usize;
    const LARGE_ENHANCEMENT_COUNT: usize = 18_usize;
    const CYCLE_LEN: usize = 3_usize;
    const CYCLE_GROWTH_FACTOR: usize = Self::cycle_growth_factor();

    const fn grow_small(len: usize) -> usize {
        len / Self::SMALL_INPUT_SIDE_LEN * Self::SMALL_OUTPUT_SIDE_LEN
    }

    const fn grow_large(len: usize) -> usize {
        len / Self::LARGE_INPUT_SIDE_LEN * Self::LARGE_OUTPUT_SIDE_LEN
    }

    const fn cycle_growth_factor() -> usize {
        const LEN_0: usize = Grid::INITIAL_PATTERN.side_len_non_static() as usize;

        const_assert!(LEN_0 % 2_usize != 0_usize);

        const LEN_1: usize = Solution::grow_large(LEN_0);

        const_assert!(LEN_1 % 2_usize == 0_usize);

        const LEN_2: usize = Solution::grow_small(LEN_1);

        const_assert!(LEN_2 % 2_usize == 0_usize);

        const LEN_3: usize = Solution::grow_small(LEN_2);

        const_assert!(LEN_3 % 2_usize != 0_usize);

        LEN_3 / LEN_0
    }

    fn try_find_enhancement_rule<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize>(
        enhancement_rules: &[EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>],
        pattern: &Pattern<Pixel, INPUT_SIDE_LEN>,
    ) -> Option<usize> {
        let light_count: u8 = pattern.light_count() as u8;
        let start_enhancement_rule: usize = enhancement_rules
            .partition_point(|enhancement_rule| light_count > enhancement_rule.light_count);
        let end_enhancement_rule: usize = start_enhancement_rule
            + enhancement_rules[start_enhancement_rule..]
                .partition_point(|enhancement_rule| light_count == enhancement_rule.light_count);

        enhancement_rules[start_enhancement_rule..end_enhancement_rule]
            .iter()
            .position(|enhancement_rule| enhancement_rule.matches(pattern))
            .map(|position| position + start_enhancement_rule)
    }

    fn try_enhance_internal<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize>(
        enhancement_rules: &[EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>],
        grid: &mut Grid,
    ) -> Option<()> {
        let input_side_len: i32 = INPUT_SIDE_LEN as i32;
        let output_side_len: i32 = OUTPUT_SIDE_LEN as i32;
        let input_pattern_dimensions: IVec2 = grid.curr_grid.dimensions() / input_side_len;

        grid.next_grid
            .clear_and_resize(output_side_len * input_pattern_dimensions, Pixel::Dark);

        for pattern_pos in (0_i32..input_pattern_dimensions.y)
            .flat_map(|y| (0_i32..input_pattern_dimensions.x).map(move |x| IVec2::new(x, y)))
        {
            let input_pattern: Pattern<Pixel, INPUT_SIDE_LEN> =
                Pattern::try_from_grid_at_offset(&grid.curr_grid, input_side_len * pattern_pos)?;

            let enhancement_rule_index: usize =
                Self::try_find_enhancement_rule(enhancement_rules, &input_pattern)?;

            enhancement_rules[enhancement_rule_index]
                .output
                .try_to_grid_at_offset(output_side_len * pattern_pos, &mut grid.next_grid)?;
        }

        swap(&mut grid.curr_grid, &mut grid.next_grid);

        Some(())
    }

    fn pattern_pos_to_pixel_pos() -> IVec2 {
        Grid::INITIAL_PATTERN.side_len_non_static() * IVec2::ONE
    }

    fn try_enhance(&self, grid: &mut Grid) -> Option<()> {
        if grid.curr_grid.dimensions().x % 2 == 0_i32 {
            Self::try_enhance_internal(&self.small_enhancement_rules, grid)
        } else {
            Self::try_enhance_internal(&self.large_enhancement_rules, grid)
        }
    }

    fn try_grid_after_enhancements(&self, enhancement_count: usize) -> Option<Grid> {
        (0_usize..enhancement_count).try_fold(Grid::new(), |mut grid, _| {
            self.try_enhance(&mut grid).map(|_| grid)
        })
    }

    fn try_light_count_after_enhancements(&self, enhancement_count: usize) -> Option<usize> {
        self.try_grid_after_enhancements(enhancement_count)
            .map(|grid| grid.light_count())
    }

    fn question_internal(&self, verbose: bool, enhancement_count: usize) {
        if verbose {
            if let Some(grid) = self.try_grid_after_enhancements(enhancement_count) {
                let light_count: usize = grid.light_count();

                println!("{}", String::from(grid.curr_grid));
                dbg!(light_count);
            } else {
                eprintln!("Failed to enhance grid {} times", enhancement_count);
            }
        } else {
            dbg!(self.try_light_count_after_enhancements(enhancement_count));
        }
    }

    fn is_complete_internal<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize>(
        enhancement_rules: &[EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>],
    ) -> bool {
        (0_usize..(2_usize << (INPUT_SIDE_LEN * INPUT_SIDE_LEN))).all(|bits| {
            let pattern: Pattern<Pixel, INPUT_SIDE_LEN> = Pattern::from_bits(bits);

            Self::try_find_enhancement_rule(enhancement_rules, &pattern).is_some()
        })
    }

    fn light_count_for_pattern_counts<const INPUT_SIDE_LEN: usize, const OUTPUT_SIDE_LEN: usize>(
        enhancement_rules: &[EnhancementRule<INPUT_SIDE_LEN, OUTPUT_SIDE_LEN>],
        pattern_counts: &[usize],
    ) -> usize {
        assert_eq!(enhancement_rules.len(), pattern_counts.len());

        pattern_counts
            .iter()
            .zip(enhancement_rules.iter())
            .map(|(pattern_count, enhancement_rule)| {
                *pattern_count * enhancement_rule.input.light_count()
            })
            .sum()
    }

    fn is_complete(&self) -> bool {
        Self::is_complete_internal(&self.small_enhancement_rules)
            && Self::is_complete_internal(&self.large_enhancement_rules)
    }

    fn try_pattern_counts(&self) -> Option<PatternCounts> {
        self.is_complete().then(|| {
            let small_enhancement_rules_len: usize = self.small_enhancement_rules.len();
            let counts_len: usize =
                small_enhancement_rules_len + self.large_enhancement_rules.len();
            let mut curr_counts: Vec<usize> = vec![0_usize; counts_len];
            let next_counts: Vec<usize> = vec![0_usize; counts_len];

            curr_counts[small_enhancement_rules_len
                + Self::try_find_enhancement_rule(
                    &self.large_enhancement_rules,
                    &Grid::INITIAL_PATTERN,
                )
                .unwrap()] = 1_usize;

            PatternCounts {
                curr_counts,
                next_counts,
            }
        })
    }

    fn patterns_per_grid_side(&self) -> i32 {
        (self.large_enhancement_rules.len() as f32).sqrt().ceil() as i32
    }

    fn grid_for_full_cycles(&self) -> Grid {
        let patterns_per_grid_side: i32 = self.patterns_per_grid_side();
        let pattern_pos_to_pixel_pos: IVec2 = Self::pattern_pos_to_pixel_pos();

        let mut curr_grid: Grid2D<Pixel> =
            Grid2D::default(patterns_per_grid_side * pattern_pos_to_pixel_pos);

        for (pattern_index, enhancement_rule) in self.large_enhancement_rules.iter().enumerate() {
            let pattern_pos: IVec2 = IVec2::new(
                pattern_index as i32 % patterns_per_grid_side,
                pattern_index as i32 / patterns_per_grid_side,
            );

            enhancement_rule
                .input
                .try_to_grid_at_offset(pattern_pos * pattern_pos_to_pixel_pos, &mut curr_grid)
                .unwrap();
        }

        let next_grid: Grid2D<Pixel> = Grid2D::default(IVec2::ZERO);

        Grid {
            curr_grid,
            next_grid,
        }
    }

    fn pattern_counts_after_full_cycles(
        &self,
        enhancement_count: usize,
        pattern_counts: &mut PatternCounts,
    ) {
        let cycles: usize = enhancement_count / Self::CYCLE_LEN;

        if cycles > 0_usize {
            let mut grid: Grid = self.grid_for_full_cycles();
            let big_patterns_per_grid_side: i32 = self.patterns_per_grid_side();
            let pattern_pos_to_pixel_pos: IVec2 = Self::pattern_pos_to_pixel_pos();
            let big_pattern_pos_to_pixel_pos: IVec2 =
                Self::CYCLE_GROWTH_FACTOR as i32 * pattern_pos_to_pixel_pos;

            (0_usize..Self::CYCLE_LEN)
                .try_for_each(|_| self.try_enhance(&mut grid))
                .unwrap();

            assert!(!grid.curr_grid.dimensions().x.is_even());

            let pattern_count: usize = self.large_enhancement_rules.len();

            let mut cycle_pattern_count_map: Grid2D<u8> =
                Grid2D::default(pattern_count as i32 * IVec2::ONE);

            for cycle_input_pattern_index in 0_i32..pattern_count as i32 {
                let big_pattern_pos: IVec2 = IVec2::new(
                    cycle_input_pattern_index % big_patterns_per_grid_side,
                    cycle_input_pattern_index / big_patterns_per_grid_side,
                );
                let big_pattern_pixel_pos: IVec2 = big_pattern_pos * big_pattern_pos_to_pixel_pos;

                for pattern_pos in (0_i32..Self::CYCLE_GROWTH_FACTOR as i32).flat_map(|y| {
                    (0_i32..Self::CYCLE_GROWTH_FACTOR as i32).map(move |x| IVec2 { x, y })
                }) {
                    let pattern: LargeInputPattern = LargeInputPattern::try_from_grid_at_offset(
                        &grid.curr_grid,
                        big_pattern_pixel_pos + pattern_pos * pattern_pos_to_pixel_pos,
                    )
                    .unwrap();
                    let cycle_output_pattern_index: i32 =
                        Self::try_find_enhancement_rule(&self.large_enhancement_rules, &pattern)
                            .unwrap() as i32;

                    *cycle_pattern_count_map
                        .get_mut(IVec2::new(
                            cycle_output_pattern_index,
                            cycle_input_pattern_index,
                        ))
                        .unwrap() += 1_u8;
                }
            }

            let pattern_index_offset: usize = self.small_enhancement_rules.len();

            for _ in 0_usize..cycles {
                let curr_counts: &[usize] = &pattern_counts.curr_counts[pattern_index_offset..];
                let next_counts: &mut [usize] =
                    &mut pattern_counts.next_counts[pattern_index_offset..];

                next_counts.fill(0_usize);

                for (input_pattern_index, count) in curr_counts.iter().enumerate() {
                    let input_pattern_index: i32 = input_pattern_index as i32;
                    let count: usize = *count;

                    for output_pattern_index in 0_i32..pattern_count as i32 {
                        next_counts[output_pattern_index as usize] += *cycle_pattern_count_map
                            .get(IVec2::new(output_pattern_index as i32, input_pattern_index))
                            .unwrap()
                            as usize
                            * count;
                    }
                }

                swap(
                    &mut pattern_counts.curr_counts,
                    &mut pattern_counts.next_counts,
                );
            }
        }
    }

    fn pattern_counts_after_partial_cycles(
        &self,
        enhancement_count: usize,
        _pattern_counts: &mut PatternCounts,
    ) {
        let enhancement_count: usize = enhancement_count % Self::CYCLE_LEN;

        if enhancement_count != 0_usize {
            // I've lost interest in this problem at this point. The initial implementation is fast
            // enough for problem 2 input, and this case is for a number of enhancements that we
            // haven't been asked for.
            unimplemented!();
        }
    }

    fn try_light_count_after_many_enhancements(&self, enhancement_count: usize) -> Option<usize> {
        let mut pattern_counts: PatternCounts = self.try_pattern_counts()?;

        self.pattern_counts_after_full_cycles(enhancement_count, &mut pattern_counts);
        self.pattern_counts_after_partial_cycles(enhancement_count, &mut pattern_counts);

        Some(
            Self::light_count_for_pattern_counts(
                &self.small_enhancement_rules,
                &pattern_counts.curr_counts[..self.small_enhancement_rules.len()],
            ) + Self::light_count_for_pattern_counts(
                &self.large_enhancement_rules,
                &pattern_counts.curr_counts[self.small_enhancement_rules.len()..],
            ),
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                many0(terminated(EnhancementRule::parse, opt(line_ending))),
                many0(terminated(EnhancementRule::parse, opt(line_ending))),
            )),
            |(mut small_enhancement_rules, mut large_enhancement_rules)| {
                small_enhancement_rules.sort();
                large_enhancement_rules.sort();

                Self {
                    small_enhancement_rules,
                    large_enhancement_rules,
                }
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Rust const generics have a long way to go until they're anywhere near as capable/feasible
    /// for production-level code as C++ and Zig's alternatives. Yikes.
    ///
    /// Expecting part 2 to be something like "run it a bajillion times".
    fn q1_internal(&mut self, args: &QuestionArgs) {
        self.question_internal(args.verbose, Self::SMALL_ENHANCEMENT_COUNT);
    }

    /// Not quite as "bajillion" as I had expected! I think that with the end size of the grid,
    /// there's an arguement to be made that the expectation is a smarter answer, but my
    /// implementation just took, I don't know, 15s tops to compute? I'll take that.
    ///
    /// If I were to smarten it up, it'd be something along this line of reasoning:
    ///
    /// Consider the size of the grid after consecutive iterations:
    ///
    /// | Iteration | Size | Prime Factorization |
    /// | --------- | ---- | ------------------- |
    /// | 0         |    3 | 2 ^ 0 * 3 ^ 1       |
    /// | 1         |    4 | 2 ^ 2 * 3 ^ 0       |
    /// | 2         |    6 | 2 ^ 1 * 3 ^ 1       |
    /// | 3         |    9 | 2 ^ 0 * 3 ^ 2       |
    /// | 4         |   12 | 2 ^ 2 * 3 ^ 1       |
    /// | 5         |   18 | 2 ^ 1 * 3 ^ 2       |
    /// | 6         |   27 | 2 ^ 0 * 3 ^ 3       |
    ///
    /// Each 3 iterations increases the size by a factor of 3. Without doing more math than I care
    /// to do right now, there's some finite number less than 2 ^ 9 = 512 discrete 3x3 pattern
    /// isomorphic groups. Let's call this number N. Over the courses of 3 iterations, each 3x3
    /// pattern will map to a vector of length N filled with at least N - 9 copies of 0, where each
    /// cell corresponds to the number of copies of that 3x3 pattern in the resulting output...
    ///
    /// Update: I didn't really finish the train of thought of that comment, instead opting to just
    /// implement it. I didn't fully flesh out the case where the number of enhancements isn't a
    /// multiple of 3, but it works for the case of 3, and faster than the naive implementation. The
    /// payoff increases for larger numbers, as it reduces it to a linear computation in time
    /// instead of exponential.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            self.question_internal(args.verbose, Self::LARGE_ENHANCEMENT_COUNT);
        } else {
            dbg!(self.try_light_count_after_many_enhancements(Self::LARGE_ENHANCEMENT_COUNT));
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
    use {super::*, pixel::*, std::sync::OnceLock};

    const SOLUTION_STRS: &'static [&'static str] = &["\
        ../.# => ##./#../...\n\
        .#./..#/### => #..#/..../..../#..#\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                small_enhancement_rules: vec![EnhancementRule {
                    input: Pattern([[D, D], [D, L]]),
                    output: Pattern([[L, L, D], [L, D, D], [D, D, D]]),
                    light_count: 1_u8,
                    flags: EnhancementRuleFlags(BitArray::new([0b1110_u8])),
                }],
                large_enhancement_rules: vec![EnhancementRule {
                    input: Pattern([[D, L, D], [D, D, L], [L, L, L]]),
                    output: Pattern([[L, D, D, L], [D, D, D, D], [D, D, D, D], [L, D, D, L]]),
                    light_count: 5_u8,
                    flags: EnhancementRuleFlags(BitArray::new([0b1111_u8])),
                }],
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
    fn test_try_enhance() {
        for (index, grid_strs) in [&[
            "\
            #..#\n\
            ....\n\
            ....\n\
            #..#\n",
            "\
            ##.##.\n\
            #..#..\n\
            ......\n\
            ##.##.\n\
            #..#..\n\
            ......\n",
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            let mut grid: Grid = Grid::new();

            for grid_str in grid_strs.iter().copied() {
                solution.try_enhance(&mut grid).unwrap();

                assert_eq!(grid.curr_grid, Grid2D::parse(grid_str).unwrap().1);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
