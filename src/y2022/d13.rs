use {
    crate::*,
    std::sync::OnceLock,
    std::{
        cmp::Ordering,
        num::ParseIntError,
        str::{from_utf8_unchecked, FromStr, Split},
    },
};

#[derive(Debug, PartialEq)]
struct BracketPair {
    left: usize,
    right: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Packet {
    List(Vec<Packet>),
    Int(u32),
}

impl Packet {
    fn divider_2() -> &'static Self {
        static ONCE_LOCK: OnceLock<Packet> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Self::List(vec![Self::List(vec![Self::Int(2_u32)])]))
    }

    fn divider_6() -> &'static Self {
        static ONCE_LOCK: OnceLock<Packet> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Self::List(vec![Self::List(vec![Self::Int(6_u32)])]))
    }

    fn slice_cmp(
        left_slice: &[Packet],
        right_slice: &[Packet],
        allow_equal: bool,
    ) -> Option<Ordering> {
        use Ordering::*;

        let mut ordering: Option<Ordering> = Some(Equal);

        for index in 0_usize..left_slice.len().max(right_slice.len()) {
            match (left_slice.get(index), right_slice.get(index)) {
                (Some(left), Some(right)) => match left.partial_cmp(right) {
                    Some(not_equal) if not_equal.is_ne() => return Some(not_equal),
                    other => {
                        ordering = ordering.and(other);
                    }
                },
                (None, Some(_)) => return Some(Less),
                (Some(_), None) => return Some(Greater),
                (None, None) => unreachable!(),
            }
        }

        if allow_equal {
            ordering
        } else {
            None
        }
    }
}

impl PartialOrd for Packet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Packet::*;

        match (self, other) {
            (Int(left_int), Int(right_int)) => left_int.partial_cmp(right_int),
            (Int(left_int), List(right_list)) => {
                Self::slice_cmp(&[Int(*left_int)], right_list, false)
            }
            (List(left_list), Int(right_int)) => {
                Self::slice_cmp(left_list, &[Int(*right_int)], false)
            }
            (List(left_list), List(right_list)) => Self::slice_cmp(left_list, right_list, true),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum PacketParseError {
    InvalidChar { index: usize, c: char },
    BracketMismatch,
    FailedToParseInt(ParseIntError),
    UnexpectedEndOfByteSlice,
    UnexpectedComma,
    ExpectedComma,
    NotAllBytesUsed,
}

impl TryFrom<&str> for Packet {
    type Error = PacketParseError;

    fn try_from(packet_str: &str) -> Result<Self, Self::Error> {
        use PacketParseError::*;

        let mut brackets: usize = 0_usize;

        for (index, c) in packet_str.chars().enumerate() {
            if c == '[' {
                brackets += 1_usize;
            } else if c == ']' {
                brackets = brackets.checked_sub(1_usize).ok_or(BracketMismatch)?;
            } else if c != ',' && !c.is_ascii_digit() {
                return Err(InvalidChar { index, c });
            }
        }

        fn try_from_bytes(packet_bytes: &[u8]) -> Result<(Packet, usize), PacketParseError> {
            if packet_bytes[0_usize] as char == '[' {
                // Parse a list
                let mut packets: Vec<Packet> = Vec::new();
                let mut bytes_parsed: usize = 1_usize;

                loop {
                    let next_byte: u8 = *packet_bytes
                        .get(bytes_parsed)
                        .ok_or(UnexpectedEndOfByteSlice)?;

                    if next_byte == b',' {
                        if packets.len() == 0_usize {
                            return Err(UnexpectedComma);
                        }

                        bytes_parsed += 1_usize;
                    } else if next_byte == b']' {
                        bytes_parsed += 1_usize;

                        break;
                    } else if packets.len() != 0_usize {
                        return Err(ExpectedComma);
                    }

                    let (child_packet, child_bytes_parsed) =
                        try_from_bytes(&packet_bytes[bytes_parsed..])?;

                    packets.push(child_packet);
                    bytes_parsed += child_bytes_parsed;
                }

                Ok((Packet::List(packets), bytes_parsed))
            } else {
                // Parse an integer
                let digits_end: usize = packet_bytes
                    .iter()
                    .position(|byte| !(*byte as char).is_ascii_digit())
                    .unwrap_or(packet_bytes.len());
                u32::from_str(
                    // SAFETY: `try_from_bytes` is only defined in a scope where all input to it has
                    // already been vetted as being either an ASCII digit, `,`, `[`, or `]`
                    unsafe { from_utf8_unchecked(&packet_bytes[..digits_end]) },
                )
                .map(|integer| (Packet::Int(integer), digits_end))
                .map_err(FailedToParseInt)
            }
        }

        if brackets != 0_usize {
            Err(BracketMismatch)
        } else {
            let (packet, parsed_bytes) = try_from_bytes(packet_str.as_bytes())?;

            if parsed_bytes == packet_str.len() {
                Ok(packet)
            } else {
                Err(NotAllBytesUsed)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PacketPair {
    left: Packet,
    right: Packet,
}

impl PacketPair {
    fn partial_cmp(&self) -> Option<Ordering> {
        self.left.partial_cmp(&self.right)
    }
}

#[derive(Debug, PartialEq)]
pub enum PacketPairParseError<'s> {
    NoLeftToken,
    FailedToParseLeft(PacketParseError),
    NoRightToken,
    FailedtoParseRight(PacketParseError),
    ExtraTokenFound(&'s str),
}

impl<'s> TryFrom<&'s str> for PacketPair {
    type Error = PacketPairParseError<'s>;

    fn try_from(packet_pair_str: &'s str) -> Result<Self, Self::Error> {
        use PacketPairParseError::*;

        let mut packet_iter: Split<char> = packet_pair_str.split('\n');

        let left: Packet = match packet_iter.next() {
            None => Err(NoLeftToken),
            Some(left_str) => left_str.try_into().map_err(FailedToParseLeft),
        }?;
        let right: Packet = match packet_iter.next() {
            None => Err(NoRightToken),
            Some(right_str) => right_str.try_into().map_err(FailedtoParseRight),
        }?;

        match packet_iter.next() {
            Some(extra_token) => Err(ExtraTokenFound(extra_token)),
            None => Ok(Self { left, right }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PacketPairs(Vec<PacketPair>);

impl<'s> TryFrom<&'s str> for PacketPairs {
    type Error = PacketPairParseError<'s>;

    fn try_from(packet_pairs_str: &'s str) -> Result<Self, Self::Error> {
        let mut packet_pairs: Self = Self(Vec::new());

        for packet_pair_str in packet_pairs_str.split("\n\n") {
            packet_pairs.0.push(packet_pair_str.try_into()?);
        }

        Ok(packet_pairs)
    }
}

#[cfg_attr(test, derive(Clone))]
#[derive(Debug, PartialEq, PartialOrd)]
struct Packets(Vec<Packet>);

impl Packets {
    fn decoder_key(&mut self) -> usize {
        (self
            .0
            .iter()
            .position(|packet| *packet == *Packet::divider_2())
            .unwrap()
            + 1_usize)
            * (self
                .0
                .iter()
                .position(|packet| *packet == *Packet::divider_6())
                .unwrap()
                + 1_usize)
    }
}

impl From<PacketPairs> for Packets {
    fn from(packet_pairs: PacketPairs) -> Self {
        let mut packets: Self = Self(
            packet_pairs
                .0
                .into_iter()
                .map(|packet_pair| [packet_pair.left, packet_pair.right].into_iter())
                .flatten()
                .collect(),
        );

        packets.0.push(Packet::divider_2().clone());
        packets.0.push(Packet::divider_6().clone());
        packets
            .0
            .sort_unstable_by(|left, right| left.partial_cmp(&right).unwrap_or(Ordering::Equal));

        packets
    }
}

impl TryFrom<&str> for Packets {
    type Error = PacketParseError;

    fn try_from(packets_str: &str) -> Result<Self, Self::Error> {
        let mut packets: Self = Self(Vec::new());

        for packet_str in packets_str.split('\n') {
            packets.0.push(packet_str.try_into()?);
        }

        Ok(packets)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(PacketPairs);

impl Solution {
    fn right_order_pair_index_sum(&self) -> usize {
        self.0
             .0
            .iter()
            .enumerate()
            .filter(|(_, packet_pair)| packet_pair.partial_cmp() == Some(Ordering::Less))
            .map(|(index, _)| index + 1_usize)
            .sum()
    }

    fn decoder_key(&self) -> usize {
        Packets::from(self.0.clone()).decoder_key()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.right_order_pair_index_sum());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.decoder_key());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = PacketPairParseError<'i>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, Packet::Int as I};

    macro_rules! l { [$($i: expr),*] => { Packet::List(vec![ $( $i, )* ]) }; }

    const PACKET_PAIRS_STR: &str = concat!(
        "[1,1,3,1,1]\n",
        "[1,1,5,1,1]\n",
        "\n",
        "[[1],[2,3,4]]\n",
        "[[1],4]\n",
        "\n",
        "[9]\n",
        "[[8,7,6]]\n",
        "\n",
        "[[4,4],4,4]\n",
        "[[4,4],4,4,4]\n",
        "\n",
        "[7,7,7,7]\n",
        "[7,7,7]\n",
        "\n",
        "[]\n",
        "[3]\n",
        "\n",
        "[[[]]]\n",
        "[[]]\n",
        "\n",
        "[1,[2,[3,[4,[5,6,7]]]],8,9]\n",
        "[1,[2,[3,[4,[5,6,0]]]],8,9]",
    );
    const PACKETS_STR: &str = concat!(
        "[]\n",
        "[[]]\n",
        "[[[]]]\n",
        "[1,1,3,1,1]\n",
        "[1,1,5,1,1]\n",
        "[[1],[2,3,4]]\n",
        "[1,[2,[3,[4,[5,6,0]]]],8,9]\n",
        "[1,[2,[3,[4,[5,6,7]]]],8,9]\n",
        "[[1],4]\n",
        "[[2]]\n",
        "[3]\n",
        "[[4,4],4,4]\n",
        "[[4,4],4,4,4]\n",
        "[[6]]\n",
        "[7,7,7]\n",
        "[7,7,7,7]\n",
        "[[8,7,6]]\n",
        "[9]",
    );

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(PacketPairs(vec![
                PacketPair {
                    left: l![I(1), I(1), I(3), I(1), I(1)],
                    right: l![I(1), I(1), I(5), I(1), I(1)],
                },
                PacketPair {
                    left: l![l![I(1)], l![I(2), I(3), I(4)]],
                    right: l![l![I(1)], I(4)],
                },
                PacketPair {
                    left: l![I(9)],
                    right: l![l![I(8), I(7), I(6)]],
                },
                PacketPair {
                    left: l![l![I(4), I(4)], I(4), I(4)],
                    right: l![l![I(4), I(4)], I(4), I(4), I(4)],
                },
                PacketPair {
                    left: l![I(7), I(7), I(7), I(7)],
                    right: l![I(7), I(7), I(7)],
                },
                PacketPair {
                    left: l![],
                    right: l![I(3)],
                },
                PacketPair {
                    left: l![l![l![]]],
                    right: l![l![]],
                },
                PacketPair {
                    left: l![
                        I(1),
                        l![I(2), l![I(3), l![I(4), l![I(5), I(6), I(7)]]]],
                        I(8),
                        I(9)
                    ],
                    right: l![
                        I(1),
                        l![I(2), l![I(3), l![I(4), l![I(5), I(6), I(0)]]]],
                        I(8),
                        I(9)
                    ],
                },
            ]))
        })
    }

    fn packets() -> &'static Packets {
        static ONCE_LOCK: OnceLock<Packets> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Packets(vec![
                l![],
                l![l![]],
                l![l![l![]]],
                l![I(1), I(1), I(3), I(1), I(1)],
                l![I(1), I(1), I(5), I(1), I(1)],
                l![l![I(1)], l![I(2), I(3), I(4)]],
                l![
                    I(1),
                    l![I(2), l![I(3), l![I(4), l![I(5), I(6), I(0)]]]],
                    I(8),
                    I(9)
                ],
                l![
                    I(1),
                    l![I(2), l![I(3), l![I(4), l![I(5), I(6), I(7)]]]],
                    I(8),
                    I(9)
                ],
                l![l![I(1)], I(4)],
                l![l![I(2)]],
                l![I(3)],
                l![l![I(4), I(4)], I(4), I(4)],
                l![l![I(4), I(4)], I(4), I(4), I(4)],
                l![l![I(6)]],
                l![I(7), I(7), I(7)],
                l![I(7), I(7), I(7), I(7)],
                l![l![I(8), I(7), I(6)]],
                l![I(9)],
            ])
        })
    }

    #[test]
    fn test_solution_try_from_str() {
        let real_solution: Result<Solution, PacketPairParseError> = PACKET_PAIRS_STR.try_into();

        pretty_assert_eq!(real_solution.as_ref(), Ok(solution()));
    }

    #[test]
    fn test_packet_partial_cmp() {
        use Ordering::*;

        let orderings: Vec<Ordering> = solution()
            .0
             .0 // How do you live with yourself, rust_fmt?
            .iter()
            .map(PacketPair::partial_cmp)
            .map(Option::unwrap)
            .collect::<Vec<Ordering>>();

        assert_eq!(
            orderings,
            vec![Less, Less, Greater, Less, Greater, Less, Greater, Greater]
        )
    }

    #[test]
    fn test_packets_try_from_str() {
        let real_packets: Result<Packets, PacketParseError> = PACKETS_STR.try_into();

        pretty_assert_eq!(real_packets.as_ref(), Ok(packets()));
    }

    #[test]
    fn test_packet_partial_cmp_vs_partial_eq() {
        let a: Packet = l![I(1)];
        let b: Packet = l![l![I(1)]];

        assert_eq!(matches!(a.partial_cmp(&b), Some(Ordering::Equal)), a.eq(&b));
    }

    #[test]
    fn test_packets_from_packet_pairs() {
        let real_packets: Packets = Packets::from(solution().0.clone());

        pretty_assert_eq!(&real_packets, packets());
    }

    #[test]
    fn test_right_order_pair_index_sum() {
        assert_eq!(solution().right_order_pair_index_sum(), 13_usize);
    }

    #[test]
    fn test_decoder_key() {
        assert_eq!(packets().clone().decoder_key(), 140_usize);
    }
}
