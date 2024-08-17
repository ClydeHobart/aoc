use {
    crate::*,
    nom::{
        bytes::complete::{tag, take_while1},
        character::complete::line_ending,
        combinator::{iterator, opt},
        error::Error,
        sequence::{delimited, tuple},
        Err, IResult,
    },
    num::Integer,
    std::ops::Range,
};

/* --- Day 7: Internet Protocol Version 7 ---

While snooping around the local network of EBHQ, you compile a list of IP addresses (they're IPv7, of course; IPv6 is much too limited). You'd like to figure out which IPs support TLS (transport-layer snooping).

An IP supports TLS if it has an Autonomous Bridge Bypass Annotation, or ABBA. An ABBA is any four-character sequence which consists of a pair of two different characters followed by the reverse of that pair, such as xyyx or abba. However, the IP also must not have an ABBA within any hypernet sequences, which are contained by square brackets.

For example:

    abba[mnop]qrst supports TLS (abba outside square brackets).
    abcd[bddb]xyyx does not support TLS (bddb is within square brackets, even though xyyx is outside square brackets).
    aaaa[qwer]tyui does not support TLS (aaaa is invalid; the interior characters must be different).
    ioxxoj[asdfgh]zxcvbn supports TLS (oxxo is outside square brackets, even though it's within a larger string).

How many IPs in your puzzle input support TLS?

--- Part Two ---

You would also like to know which IPs support SSL (super-secret listening).

An IP supports SSL if it has an Area-Broadcast Accessor, or ABA, anywhere in the supernet sequences (outside any square bracketed sections), and a corresponding Byte Allocation Block, or BAB, anywhere in the hypernet sequences. An ABA is any three-character sequence which consists of the same character twice with a different character between them, such as xyx or aba. A corresponding BAB is the same characters but in reversed positions: yxy and bab, respectively.

For example:

    aba[bab]xyz supports SSL (aba outside square brackets with corresponding bab within square brackets).
    xyx[xyx]xyx does not support SSL (xyx, but no corresponding yxy).
    aaa[kek]eke supports SSL (eke in supernet with corresponding kek in hypernet; the aaa sequence is not related, because the interior character must be different).
    zazbz[bzb]cdb supports SSL (zaz has no corresponding aza, but zbz has a corresponding bzb, even though zaz and zbz overlap).

How many IPs in your puzzle input support SSL? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct IPAddress {
    sequences: Range<u16>,
}

impl IPAddress {
    fn sequences<'s>(&self, solution: &'s Solution) -> &'s [Sequence] {
        &solution.sequences[self.sequences.as_range_usize()]
    }

    fn iter_filtered_sequences<'s, F: Fn(&usize) -> bool + 's>(
        &self,
        solution: &'s Solution,
        f: F,
    ) -> impl Iterator<Item = &'s Sequence> + 's {
        self.sequences(solution)
            .iter()
            .enumerate()
            .filter_map(move |(index, sequence)| f(&index).then_some(sequence))
    }

    fn iter_supernet_sequences<'s>(
        &self,
        solution: &'s Solution,
    ) -> impl Iterator<Item = &'s Sequence> + 's {
        self.iter_filtered_sequences(solution, usize::is_even)
    }

    fn iter_hypernet_sequences<'s>(
        &self,
        solution: &'s Solution,
    ) -> impl Iterator<Item = &'s Sequence> + 's {
        self.iter_filtered_sequences(solution, usize::is_odd)
    }

    fn supports_tls(&self, solution: &Solution) -> bool {
        self.iter_supernet_sequences(solution)
            .any(|sequence| Sequence::contains_abba(sequence.bytes(solution)))
            && self
                .iter_hypernet_sequences(solution)
                .all(|sequence| !Sequence::contains_abba(sequence.bytes(solution)))
    }

    fn supports_ssl(&self, solution: &Solution) -> bool {
        self.iter_supernet_sequences(solution)
            .flat_map(|sequence| Sequence::iter_abas(sequence.bytes(solution)))
            .any(|aba| {
                self.iter_hypernet_sequences(solution)
                    .any(|sequence| Sequence::contains_bab(sequence.bytes(solution), aba))
            })
    }
}

#[derive(Clone, Copy, PartialEq)]
struct ABSequence<const LEN: usize> {
    a: u8,
    b: u8,
}

impl<const LEN: usize> ABSequence<LEN> {
    fn len() -> usize {
        LEN
    }

    fn is_valid(bytes: &[u8]) -> bool {
        Self::try_from(bytes).is_ok()
    }

    fn invert(self) -> Self {
        Self {
            a: self.b,
            b: self.a,
        }
    }
}

type ABBA = ABSequence<4_usize>;
type ABA = ABSequence<3_usize>;

impl<const LEN: usize> TryFrom<&[u8]> for ABSequence<LEN> {
    type Error = ();

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        let len: usize = ABSequence::<LEN>::len();

        if value.len() != len {
            Err(())
        } else {
            let mut abc: [u8; 3_usize] = Default::default();

            abc.copy_from_slice(&value[..3_usize]);

            let [a, b, c]: [u8; 3_usize] = abc;

            if !match len {
                3_usize => a == c && a != b,
                4_usize => {
                    let d: u8 = value[3_usize];

                    a == d && b == c && a != b
                }
                _ => false,
            } {
                Err(())
            } else {
                Ok(Self { a, b })
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Sequence {
    bytes: Range<u32>,
}

impl Sequence {
    fn contains_abba(bytes: &[u8]) -> bool {
        bytes.windows(ABBA::len()).any(ABBA::is_valid)
    }

    fn iter_abas(bytes: &[u8]) -> impl Iterator<Item = ABA> + '_ {
        bytes
            .windows(ABA::len())
            .filter_map(|bytes| bytes.try_into().ok())
    }

    fn contains_bab(bytes: &[u8], aba: ABA) -> bool {
        let bab: ABA = aba.invert();

        bytes
            .windows(ABA::len())
            .filter_map(|bytes| ABA::try_from(bytes).ok())
            .any(|candidate_bab| candidate_bab == bab)
    }

    fn bytes<'s>(&self, solution: &'s Solution) -> &'s [u8] {
        &solution.bytes[self.bytes.as_range_usize()]
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    ip_addresses: Vec<IPAddress>,
    sequences: Vec<Sequence>,
    bytes: Vec<u8>,
}

impl Solution {
    fn parse_sequence<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        take_while1(|c: char| c.is_ascii_lowercase())(input)
    }

    fn parse_sequence_pair<'i>(input: &'i str) -> IResult<&'i str, (&'i str, Option<&'i str>)> {
        tuple((
            Self::parse_sequence,
            opt(delimited(tag("["), Self::parse_sequence, tag("]"))),
        ))(input)
    }

    fn tls_supporting_ip_address_count(&self) -> usize {
        self.ip_addresses
            .iter()
            .filter(|ip_address| ip_address.supports_tls(self))
            .count()
    }

    fn ssl_supporting_ip_address_count(&self) -> usize {
        self.ip_addresses
            .iter()
            .filter(|ip_address| ip_address.supports_ssl(self))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut remaining_input: &str = input;
        let mut ip_addresses: Vec<IPAddress> = Vec::new();
        let mut sequences: Vec<Sequence> = Vec::new();
        let mut byte_offset: u32 = 0_u32;

        loop {
            let ip_address_sequence_start: u16 = sequences.len() as u16;

            let mut sequence_pair_iter = iterator(remaining_input, Self::parse_sequence_pair);

            for (supernet_sequence, hypernet_sequence) in &mut sequence_pair_iter {
                let push_sequence =
                    |sequence: &str, byte_offset: &mut u32, sequences: &mut Vec<Sequence>| {
                        let sequence_byte_start: u32 = *byte_offset;
                        let sequence_byte_end: u32 = sequence_byte_start + sequence.len() as u32;

                        *byte_offset = sequence_byte_end;
                        sequences.push(Sequence {
                            bytes: sequence_byte_start..sequence_byte_end,
                        });
                    };

                push_sequence(supernet_sequence, &mut byte_offset, &mut sequences);

                if let Some(hypernet_sequence) = hypernet_sequence {
                    byte_offset += 1_u32;
                    push_sequence(hypernet_sequence, &mut byte_offset, &mut sequences);
                    byte_offset += 1_u32;
                }
            }

            let (pre_line_ending_input, ()) = sequence_pair_iter.finish()?;
            let (post_line_ending_input, _) = opt(line_ending)(pre_line_ending_input)?;

            byte_offset += (pre_line_ending_input.len() - post_line_ending_input.len()) as u32;
            remaining_input = post_line_ending_input;

            let ip_address_sequence_end: u16 = sequences.len() as u16;

            if ip_address_sequence_start == ip_address_sequence_end {
                break;
            } else {
                ip_addresses.push(IPAddress {
                    sequences: ip_address_sequence_start..ip_address_sequence_end,
                })
            }
        }

        let bytes: Vec<u8> = input[..input.len() - remaining_input.len()]
            .as_bytes()
            .to_vec();

        Ok((
            remaining_input,
            Self {
                ip_addresses,
                sequences,
                bytes,
            },
        ))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.tls_supporting_ip_address_count());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.ssl_supporting_ip_address_count());
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
        abba[mnop]qrst\n\
        abcd[bddb]xyyx\n\
        aaaa[qwer]tyui\n\
        ioxxoj[asdfgh]zxcvbn\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            ip_addresses: vec![
                IPAddress { sequences: 0..3 },
                IPAddress { sequences: 3..6 },
                IPAddress { sequences: 6..9 },
                IPAddress { sequences: 9..12 },
            ],
            sequences: vec![
                Sequence { bytes: 0..4 },
                Sequence { bytes: 5..9 },
                Sequence { bytes: 10..14 },
                Sequence { bytes: 15..19 },
                Sequence { bytes: 20..24 },
                Sequence { bytes: 25..29 },
                Sequence { bytes: 30..34 },
                Sequence { bytes: 35..39 },
                Sequence { bytes: 40..44 },
                Sequence { bytes: 45..51 },
                Sequence { bytes: 52..58 },
                Sequence { bytes: 59..65 },
            ],
            bytes: SOLUTION_STR.as_bytes().to_vec(),
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_supports_tls() {
        assert_eq!(
            solution()
                .ip_addresses
                .iter()
                .map(|ip_address| ip_address.supports_tls(solution()))
                .collect::<Vec<bool>>(),
            vec![true, false, false, true]
        );
    }

    #[test]
    fn test_supports_ssl() {
        let solution: Solution = Solution::try_from(
            "\
        aba[bab]xyz\n\
        xyx[xyx]xyx\n\
        aaa[kek]eke\n\
        zazbz[bzb]cdb\n",
        )
        .unwrap();

        assert_eq!(
            solution
                .ip_addresses
                .iter()
                .map(|ip_address| ip_address.supports_ssl(&solution))
                .collect::<Vec<bool>>(),
            vec![true, false, true, true]
        );
    }
}
