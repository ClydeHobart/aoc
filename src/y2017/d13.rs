use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
};

/* --- Day 13: Packet Scanners ---

You need to cross a vast firewall. The firewall consists of several layers, each with a security scanner that moves back and forth across the layer. To succeed, you must not be detected by a scanner.

By studying the firewall briefly, you are able to record (in your puzzle input) the depth of each layer and the range of the scanning area for the scanner within it, written as depth: range. Each layer has a thickness of exactly 1. A layer at depth 0 begins immediately inside the firewall; a layer at depth 1 would start immediately after that.

For example, suppose you've recorded the following:

0: 3
1: 2
4: 4
6: 4

This means that there is a layer immediately inside the firewall (with range 3), a second layer immediately after that (with range 2), a third layer which begins at depth 4 (with range 4), and a fourth layer which begins at depth 6 (also with range 4). Visually, it might look like this:

 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]

Within each layer, a security scanner moves back and forth within its range. Each security scanner starts at the top and moves down until it reaches the bottom, then moves up until it reaches the top, and repeats. A security scanner takes one picosecond to move one step. Drawing scanners as S, the first few picoseconds look like this:


Picosecond 0:
 0   1   2   3   4   5   6
[S] [S] ... ... [S] ... [S]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]

Picosecond 1:
 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

Picosecond 2:
 0   1   2   3   4   5   6
[ ] [S] ... ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]

Picosecond 3:
 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] ... [ ]
[S] [S]         [ ]     [ ]
[ ]             [ ]     [ ]
                [S]     [S]

Your plan is to hitch a ride on a packet about to move through the firewall. The packet will travel along the top of each layer, and it moves at one layer per picosecond. Each picosecond, the packet moves one layer forward (its first move takes it into layer 0), and then the scanners move one step. If there is a scanner at the top of the layer as your packet enters it, you are caught. (If a scanner moves into the top of its layer while you are there, you are not caught: it doesn't have time to notice you before you leave.) If you were to do this in the configuration above, marking your current position with parentheses, your passage through the firewall would look like this:

Initial state:
 0   1   2   3   4   5   6
[S] [S] ... ... [S] ... [S]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]

Picosecond 0:
 0   1   2   3   4   5   6
(S) [S] ... ... [S] ... [S]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
( ) [ ] ... ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]


Picosecond 1:
 0   1   2   3   4   5   6
[ ] ( ) ... ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] (S) ... ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]


Picosecond 2:
 0   1   2   3   4   5   6
[ ] [S] (.) ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] (.) ... [ ] ... [ ]
[S] [S]         [ ]     [ ]
[ ]             [ ]     [ ]
                [S]     [S]


Picosecond 3:
 0   1   2   3   4   5   6
[ ] [ ] ... (.) [ ] ... [ ]
[S] [S]         [ ]     [ ]
[ ]             [ ]     [ ]
                [S]     [S]

 0   1   2   3   4   5   6
[S] [S] ... (.) [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[ ]             [S]     [S]
                [ ]     [ ]


Picosecond 4:
 0   1   2   3   4   5   6
[S] [S] ... ... ( ) ... [ ]
[ ] [ ]         [ ]     [ ]
[ ]             [S]     [S]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] ... ... ( ) ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]


Picosecond 5:
 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] (.) [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [S] ... ... [S] (.) [S]
[ ] [ ]         [ ]     [ ]
[S]             [ ]     [ ]
                [ ]     [ ]


Picosecond 6:
 0   1   2   3   4   5   6
[ ] [S] ... ... [S] ... (S)
[ ] [ ]         [ ]     [ ]
[S]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] ... ( )
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

In this situation, you are caught in layers 0 and 6, because your packet entered the layer when its scanner was at the top when you entered it. You are not caught in layer 1, since the scanner moved into the top of the layer once you were already there.

The severity of getting caught on a layer is equal to its depth multiplied by its range. (Ignore layers in which you do not get caught.) The severity of the whole trip is the sum of these values. In the example above, the trip severity is 0*3 + 6*4 = 24.

Given the details of the firewall you've recorded, if you leave immediately, what is the severity of your whole trip?

--- Part Two ---

Now, you need to pass through the firewall without being caught - easier said than done.

You can't control the speed of the packet, but you can delay it any number of picoseconds. For each picosecond you delay the packet before beginning your trip, all security scanners move one step. You're not in the firewall during this time; you don't enter layer 0 until you stop delaying the packet.

In the example above, if you delay 10 picoseconds (picoseconds 0 - 9), you won't get caught:

State after delaying:
 0   1   2   3   4   5   6
[ ] [S] ... ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]

Picosecond 10:
 0   1   2   3   4   5   6
( ) [S] ... ... [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]

 0   1   2   3   4   5   6
( ) [ ] ... ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]


Picosecond 11:
 0   1   2   3   4   5   6
[ ] ( ) ... ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[S] (S) ... ... [S] ... [S]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]


Picosecond 12:
 0   1   2   3   4   5   6
[S] [S] (.) ... [S] ... [S]
[ ] [ ]         [ ]     [ ]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] (.) ... [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]


Picosecond 13:
 0   1   2   3   4   5   6
[ ] [ ] ... (.) [ ] ... [ ]
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [S] ... (.) [ ] ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]


Picosecond 14:
 0   1   2   3   4   5   6
[ ] [S] ... ... ( ) ... [ ]
[ ] [ ]         [ ]     [ ]
[S]             [S]     [S]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] ... ... ( ) ... [ ]
[S] [S]         [ ]     [ ]
[ ]             [ ]     [ ]
                [S]     [S]


Picosecond 15:
 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] (.) [ ]
[S] [S]         [ ]     [ ]
[ ]             [ ]     [ ]
                [S]     [S]

 0   1   2   3   4   5   6
[S] [S] ... ... [ ] (.) [ ]
[ ] [ ]         [ ]     [ ]
[ ]             [S]     [S]
                [ ]     [ ]


Picosecond 16:
 0   1   2   3   4   5   6
[S] [S] ... ... [ ] ... ( )
[ ] [ ]         [ ]     [ ]
[ ]             [S]     [S]
                [ ]     [ ]

 0   1   2   3   4   5   6
[ ] [ ] ... ... [ ] ... ( )
[S] [S]         [S]     [S]
[ ]             [ ]     [ ]
                [ ]     [ ]

Because all smaller delays would get you caught, the fewest number of picoseconds you would need to delay to get through safely is 10.

What is the fewest number of picoseconds that you need to delay the packet to pass through the firewall without being caught? */

#[cfg_attr(test, derive(Debug))]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct FirewallLayer {
    depth: u8,
    range: u8,
}

impl FirewallLayer {
    fn period(self) -> u8 {
        (self.range - 1_u8) * 2_u8
    }

    fn pos_at_time(self, time: usize) -> u8 {
        let half_period: u8 = self.range - 1_u8;
        let period: u8 = half_period * 2_u8;
        let mod_pos: u8 = (time % period as usize) as u8;

        if mod_pos < half_period {
            mod_pos
        } else {
            period - mod_pos
        }
    }

    fn severity(self) -> u32 {
        self.depth as u32 * self.range as u32
    }

    fn catches_player_with_delay(self, delay: usize) -> bool {
        self.pos_at_time(self.depth as usize + delay) == 0_u8
    }

    fn catches_player(self) -> bool {
        self.catches_player_with_delay(0_usize)
    }
}

impl Parse for FirewallLayer {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(parse_integer, tag(": "), parse_integer),
            |(depth, range)| Self { depth, range },
        )(input)
    }
}

const MAX_OFFSET: usize = u16::BITS as usize * 3_usize;

type OffsetBitArray = BitArr!(for MAX_OFFSET, in u16);

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<FirewallLayer>);

impl Solution {
    fn iter_caught_firewall_layers(&self) -> impl Iterator<Item = FirewallLayer> + '_ {
        self.0
            .iter()
            .copied()
            .filter(|firewall_layer| firewall_layer.catches_player())
    }

    fn sum_severity_of_caught_firewall_layers(&self) -> u32 {
        self.iter_caught_firewall_layers()
            .map(FirewallLayer::severity)
            .sum()
    }

    fn min_delay_to_not_get_caught(&self) -> Option<usize> {
        (self
            .0
            .iter()
            .map(|firewall_layer| firewall_layer.range)
            .max()
            .unwrap_or_default() as usize
            <= MAX_OFFSET)
            .then(|| {
                let mut period_and_blocked_offsets: Vec<(i32, OffsetBitArray)> = Vec::new();

                for firewall_layer in &self.0 {
                    let period: i32 = firewall_layer.period() as i32;
                    let index: usize = period_and_blocked_offsets
                        .binary_search_by_key(&period, |(period, _)| *period)
                        .unwrap_or_else(|index| {
                            period_and_blocked_offsets
                                .insert(index, (period, OffsetBitArray::ZERO));

                            index
                        });
                    let blocked_offsets: &mut OffsetBitArray =
                        &mut period_and_blocked_offsets[index].1;

                    blocked_offsets.set(
                        (period - firewall_layer.depth as i32).rem_euclid(period) as usize,
                        true,
                    );
                }

                let mut delay: usize = 0_usize;

                while period_and_blocked_offsets
                    .iter()
                    .any(|(period, blocked_offset)| blocked_offset[delay % *period as usize])
                {
                    delay += 1_usize;
                }

                delay
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(FirewallLayer::parse, opt(line_ending))),
            |mut firewall_layers| {
                firewall_layers.sort();

                Self(firewall_layers)
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Seems like one of the problems from 2016
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_severity_of_caught_firewall_layers());
    }

    /// Probably not an optimal solution here. It seems like there's a way to leverage LCM to find
    /// one that'll work quicker?
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.min_delay_to_not_get_caught());
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
        0: 3\n\
        1: 2\n\
        4: 4\n\
        6: 4\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                FirewallLayer {
                    depth: 0_u8,
                    range: 3_u8,
                },
                FirewallLayer {
                    depth: 1_u8,
                    range: 2_u8,
                },
                FirewallLayer {
                    depth: 4_u8,
                    range: 4_u8,
                },
                FirewallLayer {
                    depth: 6_u8,
                    range: 4_u8,
                },
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
    fn test_iter_caught_firewall_layers() {
        for (index, caught_firewall_layers) in [vec![
            FirewallLayer {
                depth: 0_u8,
                range: 3_u8,
            },
            FirewallLayer {
                depth: 6_u8,
                range: 4_u8,
            },
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_caught_firewall_layers()
                    .collect::<Vec<_>>(),
                caught_firewall_layers
            );
        }
    }

    #[test]
    fn test_sum_severity_of_caught_firewall_layers() {
        for (index, sum_severity_of_cuaght_firewall_layers) in [24_u32].into_iter().enumerate() {
            assert_eq!(
                solution(index).sum_severity_of_caught_firewall_layers(),
                sum_severity_of_cuaght_firewall_layers
            );
        }
    }

    #[test]
    fn test_min_delay_to_not_get_caught() {
        for (index, min_delay_to_not_get_caught) in [Some(10_usize)].into_iter().enumerate() {
            assert_eq!(
                solution(index).min_delay_to_not_get_caught(),
                min_delay_to_not_get_caught
            );
        }
    }
}
