use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::{is_a, tag},
        character::complete::line_ending,
        combinator::map_res,
        error::Error,
        multi::separated_list1,
        sequence::tuple,
        Err, IResult,
    },
    std::{collections::HashMap, ops::Range},
};

/* --- Day 19: Linen Layout ---

Today, The Historians take you up to the hot springs on Gear Island! Very suspiciously, absolutely nothing goes wrong as they begin their careful search of the vast field of helixes.

Could this finally be your chance to visit the onsen next door? Only one way to find out.

After a brief conversation with the reception staff at the onsen front desk, you discover that you don't have the right kind of money to pay the admission fee. However, before you can leave, the staff get your attention. Apparently, they've heard about how you helped at the hot springs, and they're willing to make a deal: if you can simply help them arrange their towels, they'll let you in for free!

Every towel at this onsen is marked with a pattern of colored stripes. There are only a few patterns, but for any particular pattern, the staff can get you as many towels with that pattern as you need. Each stripe can be white (w), blue (u), black (b), red (r), or green (g). So, a towel with the pattern ggr would have a green stripe, a green stripe, and then a red stripe, in that order. (You can't reverse a pattern by flipping a towel upside-down, as that would cause the onsen logo to face the wrong way.)

The Official Onsen Branding Expert has produced a list of designs - each a long sequence of stripe colors - that they would like to be able to display. You can use any towels you want, but all of the towels' stripes must exactly match the desired design. So, to display the design rgrgr, you could use two rg towels and then an r towel, an rgr towel and then a gr towel, or even a single massive rgrgr towel (assuming such towel patterns were actually available).

To start, collect together all of the available towel patterns and the list of desired designs (your puzzle input). For example:

r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb

The first line indicates the available towel patterns; in this example, the onsen has unlimited towels with a single red stripe (r), unlimited towels with a white stripe and then a red stripe (wr), and so on.

After the blank line, the remaining lines each describe a design the onsen would like to be able to display. In this example, the first design (brwrr) indicates that the onsen would like to be able to display a black stripe, a red stripe, a white stripe, and then two red stripes, in that order.

Not all designs will be possible with the available towels. In the above example, the designs are possible or impossible as follows:

    brwrr can be made with a br towel, then a wr towel, and then finally an r towel.
    bggr can be made with a b towel, two g towels, and then an r towel.
    gbbr can be made with a gb towel and then a br towel.
    rrbgbr can be made with r, rb, g, and br.
    ubwu is impossible.
    bwurrg can be made with bwu, r, r, and g.
    brgr can be made with br, g, and r.
    bbrgwb is impossible.

In this example, 6 of the eight designs are possible with the available towel patterns.

To get into the onsen as soon as possible, consult your list of towel patterns and desired designs carefully. How many designs are possible?

--- Part Two ---

The staff don't really like some of the towel arrangements you came up with. To avoid an endless cycle of towel rearrangement, maybe you should just give them every possible option.

Here are all of the different ways the above example's designs can be made:

brwrr can be made in two different ways: b, r, wr, r or br, wr, r.

bggr can only be made with b, g, g, and r.

gbbr can be made 4 different ways:

    g, b, b, r
    g, b, br
    gb, b, r
    gb, br

rrbgbr can be made 6 different ways:

    r, r, b, g, b, r
    r, r, b, g, br
    r, r, b, gb, r
    r, rb, g, b, r
    r, rb, g, br
    r, rb, gb, r

bwurrg can only be made with bwu, r, r, and g.

brgr can be made in two different ways: b, r, g, r or br, g, r.

ubwu and bbrgwb are still impossible.

Adding up all of the ways the towels in this example could be arranged into the desired designs yields 16 (2 + 1 + 4 + 6 + 1 + 2).

They'll let you into the onsen as soon as you have the list. What do you get if you add up the number of different ways you could make each design? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Pattern {
    color_range: Range<u16>,
}

impl Pattern {
    const VALID_COLORS: &'static str = "wubrg";

    fn parse<'c, 'i: 'c>(
        colors: &'c mut String,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Self> + 'c {
        |input: &str| {
            map_res(is_a(Self::VALID_COLORS), |colors_str: &str| {
                colors.len().try_into().and_then(|color_range_start| {
                    (color_range_start as usize + colors_str.len())
                        .try_into()
                        .map(|color_range_end| {
                            colors.push_str(colors_str);

                            Self {
                                color_range: color_range_start..color_range_end,
                            }
                        })
                })
            })(input)
        }
    }

    fn get<'c>(&self, colors: &'c str) -> &'c str {
        &colors[self.color_range.as_range_usize()]
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    colors: String,
    available_patterns: Vec<Pattern>,
    design_patterns: Vec<Pattern>,
}

impl Solution {
    fn count_design_pattern_possibilities<'s, 'd>(
        &'s self,
        design_pattern_cache: &'d mut HashMap<&'s str, usize>,
        design_pattern: &'s str,
    ) -> usize {
        if !design_pattern_cache.contains_key(design_pattern) {
            let first_design_pattern_byte: u8 = design_pattern.as_bytes()[0_usize];
            let available_patterns_start: usize =
                self.available_patterns
                    .partition_point(|available_pattern| {
                        available_pattern.get(&self.colors).as_bytes()[0_usize]
                            < first_design_pattern_byte
                    });
            let design_pattern_possibility_count: usize = self.available_patterns
                [available_patterns_start..self.available_patterns.len()]
                .iter()
                .map(|available_pattern| available_pattern.get(&self.colors))
                .take_while(|available_pattern_str| {
                    available_pattern_str.as_bytes()[0_usize] == first_design_pattern_byte
                })
                .map(|available_pattern_str| {
                    if design_pattern == available_pattern_str {
                        1_usize
                    } else if !design_pattern.starts_with(available_pattern_str) {
                        0_usize
                    } else {
                        self.count_design_pattern_possibilities(
                            design_pattern_cache,
                            &design_pattern[available_pattern_str.len()..],
                        )
                    }
                })
                .sum();

            design_pattern_cache.insert(design_pattern, design_pattern_possibility_count);
        }

        design_pattern_cache[design_pattern]
    }

    fn possible_design_patterns(&self) -> (HashMap<&str, usize>, BitVec) {
        let mut design_pattern_cache: HashMap<&str, usize> = HashMap::new();
        let mut possible_design_patterns: BitVec = bitvec![0; self.design_patterns.len()];

        for (index, design_pattern) in self.design_patterns.iter().enumerate() {
            possible_design_patterns.set(
                index,
                self.count_design_pattern_possibilities(
                    &mut design_pattern_cache,
                    design_pattern.get(&self.colors),
                ) > 0_usize,
            );
        }

        (design_pattern_cache, possible_design_patterns)
    }

    fn count_possible_design_patterns(&self) -> usize {
        self.possible_design_patterns().1.count_ones()
    }

    fn count_total_design_pattern_possibilities(&self) -> usize {
        let (design_pattern_cache, possible_design_patterns): (HashMap<&str, usize>, BitVec) =
            self.possible_design_patterns();

        possible_design_patterns
            .iter_ones()
            .map(|index| design_pattern_cache[self.design_patterns[index].get(&self.colors)])
            .sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut colors: String = String::with_capacity(input.len());

        let (input, mut available_patterns): (&str, Vec<Pattern>) =
            separated_list1(tag(", "), Pattern::parse(&mut colors))(input)?;
        let input: &str = tuple((line_ending, line_ending))(input)?.0;
        let (input, design_patterns): (&str, Vec<Pattern>) =
            separated_list1(line_ending, Pattern::parse(&mut colors))(input)?;

        available_patterns.sort_by_key(|pattern| pattern.get(&colors));

        Ok((
            input,
            Self {
                colors,
                available_patterns,
                design_patterns,
            },
        ))
    }
}

impl RunQuestions for Solution {
    // dYnAmIc PrOgRaMmInG/mEmOiZaTiOn
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_possible_design_patterns());
    }

    // *gulp* that's a big number
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_total_design_pattern_possibilities());
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
        r, wr, b, g, bwu, rb, gb, br\n\
        \n\
        brwrr\n\
        bggr\n\
        gbbr\n\
        rrbgbr\n\
        ubwu\n\
        bwurrg\n\
        brgr\n\
        bbrgwb\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            macro_rules! patterns {
                [ $( $color_range:expr, )* ] => {
                    vec![ $( Pattern { color_range: $color_range }, )* ]
                }
            }

            vec![Solution {
                colors: "rwrbgbwurbgbbrbrwrrbggrgbbrrrbgbrubwubwurrgbrgrbbrgwb".into(),
                available_patterns: patterns![
                    3..4,   // b
                    12..14, // br
                    5..8,   // bwu
                    4..5,   // g
                    10..12, // gb
                    0..1,   // r
                    8..10,  // rb
                    1..3,   // wr
                ],
                design_patterns: patterns![
                    14..19,
                    19..23,
                    23..27,
                    27..33,
                    33..37,
                    37..43,
                    43..47,
                    47..53,
                ],
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
    fn test_possible_design_patterns() {
        for (index, possible_design_patterns) in
            [bitvec![1, 1, 1, 1, 0, 1, 1, 0]].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).possible_design_patterns().1,
                possible_design_patterns
            );
        }
    }

    #[test]
    fn test_count_possible_design_patterns() {
        for (index, possible_design_patterns_count) in [6_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_possible_design_patterns(),
                possible_design_patterns_count
            );
        }
    }

    #[test]
    fn test_count_total_design_pattern_possibilities() {
        for (index, total_design_pattern_possibilities_count) in [16_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).count_total_design_pattern_possibilities(),
                total_design_pattern_possibilities_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
