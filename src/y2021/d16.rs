use {
    crate::*,
    nom::{
        combinator::{iterator, map},
        error::{Error, ErrorKind},
        multi::{many0_count, many1_count},
        sequence::{terminated, tuple},
        Err, IResult, InputIter, InputLength, Needed,
    },
    num::PrimInt,
    std::{
        cmp::Ordering,
        iter::Enumerate,
        mem::{size_of, transmute},
        ops::{Bound, RangeBounds},
        slice::SliceIndex,
    },
};

struct HexStrIter<'i> {
    hex_str: &'i str,
    hexadigit: u8,
    start_bit: u8,
    end_bit: u8,
}

impl<'i> HexStrIter<'i> {
    fn current_end_bit(&self) -> u8 {
        if self.hex_str.is_empty() {
            self.end_bit
        } else {
            0_u8
        }
    }
}

impl<'i> From<HexStr<'i>> for HexStrIter<'i> {
    fn from(hex_str: HexStr<'i>) -> Self {
        if let Some(hexadigit) = hex_str.hex_str.chars().next() {
            Self {
                hex_str: &hex_str.hex_str[1_usize..],
                hexadigit: HexStr::char_to_byte(hexadigit),
                start_bit: hex_str.start_bit,
                end_bit: hex_str.end_bit,
            }
        } else {
            Self {
                hex_str: hex_str.hex_str,
                hexadigit: 0_u8,
                start_bit: hex_str.start_bit,
                end_bit: hex_str.end_bit,
            }
        }
    }
}

impl<'i> Iterator for HexStrIter<'i> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_bit > self.current_end_bit() {
            self.start_bit -= 1_u8;

            Some((self.hexadigit & 1_u8 << self.start_bit) != 0_u8)
        } else if let Some(hexadigit) = self.hex_str.chars().next() {
            self.hex_str = &self.hex_str[1_usize..];
            self.hexadigit = HexStr::char_to_byte(hexadigit);
            self.start_bit = HexStr::BITS_PER_HEX_BYTE;

            self.next()
        } else {
            None
        }
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Copy, Clone)]
struct HexStr<'i> {
    hex_str: &'i str,
    start_bit: u8,
    end_bit: u8,
}

impl<'i> HexStr<'i> {
    const BITS_PER_HEX_BYTE: u8 = 4_u8;

    const fn char_to_byte(c: char) -> u8 {
        if c.is_ascii_digit() {
            c as u8 - b'0'
        } else {
            const ALPHA_OFFSET: u8 = b'A' - 10_u8;

            c as u8 - ALPHA_OFFSET
        }
    }

    fn len(&self) -> usize {
        if self.hex_str.is_empty() {
            0_usize
        } else {
            ((self.hex_str.len() - 1_usize) * Self::BITS_PER_HEX_BYTE as usize
                + self.start_bit as usize)
                .saturating_sub(self.end_bit as usize)
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }

    fn normalize(&self) -> Self {
        let (start_byte, start_bit): (usize, u8) = if self.start_bit == 0_u8 {
            (1_usize, Self::BITS_PER_HEX_BYTE)
        } else {
            (0_usize, self.start_bit)
        };
        let (end_byte, end_bit): (usize, u8) = if self.end_bit == Self::BITS_PER_HEX_BYTE {
            (self.hex_str.len() - 1_usize, 0_u8)
        } else {
            (self.hex_str.len(), self.end_bit)
        };

        Self {
            hex_str: &self.hex_str[start_byte..end_byte],
            start_bit,
            end_bit,
        }
    }

    fn byte_and_bit_from_bit(&self, bit: usize) -> (usize, u8) {
        if self.start_bit as usize > bit {
            (0_usize, self.start_bit - bit as u8)
        } else {
            let bit: usize = bit - self.start_bit as usize;

            (
                bit / Self::BITS_PER_HEX_BYTE as usize + 1_usize,
                Self::BITS_PER_HEX_BYTE - (bit % Self::BITS_PER_HEX_BYTE as usize) as u8,
            )
        }
    }

    fn index<I: SliceIndex<str> + RangeBounds<usize>>(&self, index: I) -> Self {
        self.get(index).unwrap()
    }

    fn get<I: SliceIndex<str> + RangeBounds<usize>>(&self, index: I) -> Option<Self> {
        let hex_str: Self = self.normalize();
        let start_bit: usize = match index.start_bound() {
            Bound::Included(inclusive_start) => *inclusive_start,
            Bound::Excluded(exclusive_start) => *exclusive_start + 1_usize,
            Bound::Unbounded => 0_usize,
        };
        let (end_bit, verified): (usize, bool) = match index.end_bound() {
            Bound::Included(inclusive_end) => (*inclusive_end + 1_usize, false),
            Bound::Excluded(exclusive_end) => (*exclusive_end, false),
            Bound::Unbounded => (hex_str.len(), true),
        };

        if !verified && end_bit > hex_str.len() || start_bit > end_bit {
            None
        } else {
            let (start_byte, start_bit): (usize, u8) = self.byte_and_bit_from_bit(start_bit);
            let (mut end_byte, mut end_bit): (usize, u8) = self.byte_and_bit_from_bit(end_bit);

            if end_bit == Self::BITS_PER_HEX_BYTE {
                end_bit = 0_u8;
            } else {
                end_byte += 1_usize;
            }

            Some(Self {
                hex_str: &hex_str.hex_str[start_byte..end_byte],
                start_bit,
                end_bit,
            })
        }
    }

    fn try_into_prim_int<T: PrimInt>(&self) -> Option<T> {
        if self.len() <= T::zero().count_zeros() as usize {
            let mut prim_int: T = T::zero();

            for bit in self.iter_elements() {
                prim_int = prim_int << 1_usize;

                if bit {
                    prim_int = prim_int | T::one();
                }
            }

            Some(prim_int)
        } else {
            None
        }
    }
}

impl<'i> InputIter for HexStr<'i> {
    type Item = bool;
    type Iter = Enumerate<HexStrIter<'i>>;
    type IterElem = HexStrIter<'i>;

    fn iter_indices(&self) -> Self::Iter {
        self.iter_elements().enumerate()
    }

    fn iter_elements(&self) -> Self::IterElem {
        (*self).into()
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.iter_elements().position(predicate)
    }

    fn slice_index(&self, _count: usize) -> Result<usize, Needed> {
        // I can't tell what this is supposed to do. Let's hope it's not needed
        Err(Needed::Unknown)
    }
}

impl<'i> InputLength for HexStr<'i> {
    fn input_len(&self) -> usize {
        self.len()
    }
}

impl<'i> PartialEq for HexStr<'i> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter_elements()
                .zip(other.iter_elements())
                .all(|(self_bit, other_bit)| self_bit == other_bit)
    }
}

impl<'i> TryFrom<&'i str> for HexStr<'i> {
    type Error = ();

    fn try_from(value: &'i str) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Ok(Self {
                hex_str: value,
                start_bit: 0_u8,
                end_bit: 0_u8,
            })
        } else if value
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, 'A'..='F'))
        {
            Ok(Self {
                hex_str: value,
                start_bit: Self::BITS_PER_HEX_BYTE,
                end_bit: 0_u8,
            })
        } else {
            Err(())
        }
    }
}

#[allow(dead_code)]
#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
#[repr(u8)]
enum OperatorType {
    Sum,
    Product,
    Minimum,
    Maximum,
    // Literal
    GreaterThan = 5,
    LessThan,
    EqualTo,
}

impl TryFrom<u8> for OperatorType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value < (1_u8 << Packet::PACKET_TYPE_ID_LEN) && value != Packet::PACKET_TYPE_ID_LITERAL {
            // SAFETY: OperatorType has repr(u8), and we just verified value is one of the seven
            // valid values
            Ok(unsafe { transmute(value) })
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
enum Packet {
    Literal {
        packet_version: u8,
        value: u64,
    },
    Operator {
        packet_version: u8,
        operator_type: OperatorType,
        sub_packets: Vec<Packet>,
    },
}

impl Packet {
    const PACKET_VERSION_LEN: usize = 3_usize;
    const PACKET_TYPE_ID_LEN: usize = 3_usize;
    const PACKET_TYPE_ID_LITERAL: u8 = 4_u8;
    const LITERAL_GROUP_PREFIX_LEN: usize = 1_usize;
    const LITERAL_GROUP_PAYLOAD_LEN: usize = 4_usize;
    const LEN_TYPE_ID_LEN: usize = 1_usize;
    const LEN_TYPE_ID_0_TOTAL_LEN_LEN: usize = 15_usize;
    const LEN_TYPE_ID_1_SUB_PACKETS_LEN_LEN: usize = 11_usize;

    fn sum_packet_versions(&self) -> u32 {
        match self {
            Self::Literal { packet_version, .. } => *packet_version as u32,
            Self::Operator {
                packet_version,
                sub_packets,
                ..
            } => {
                *packet_version as u32
                    + sub_packets
                        .iter()
                        .map(Self::sum_packet_versions)
                        .sum::<u32>()
            }
        }
    }

    fn value(&self) -> u64 {
        match self {
            Self::Literal { value, .. } => *value,
            Self::Operator {
                operator_type,
                sub_packets,
                ..
            } => match operator_type {
                OperatorType::Sum => sub_packets.iter().map(Self::value).sum(),
                OperatorType::Product => sub_packets.iter().map(Self::value).product(),
                OperatorType::Minimum => sub_packets.iter().map(Self::value).min().unwrap(),
                OperatorType::Maximum => sub_packets.iter().map(Self::value).max().unwrap(),
                OperatorType::GreaterThan => {
                    assert_eq!(sub_packets.len(), 2_usize);

                    (sub_packets[0_usize].value() > sub_packets[1_usize].value()) as u64
                }
                OperatorType::LessThan => {
                    assert_eq!(sub_packets.len(), 2_usize);

                    (sub_packets[0_usize].value() < sub_packets[1_usize].value()) as u64
                }
                OperatorType::EqualTo => {
                    assert_eq!(sub_packets.len(), 2_usize);

                    (sub_packets[0_usize].value() == sub_packets[1_usize].value()) as u64
                }
            },
        }
    }

    fn parse<'i>(value: HexStr<'i>) -> IResult<HexStr<'i>, Self> {
        let (value, (packet_version, packet_type_id)): (HexStr, (u8, u8)) = tuple((
            Self::parse_value::<{ Self::PACKET_VERSION_LEN }, u8>,
            Self::parse_value::<{ Self::PACKET_TYPE_ID_LEN }, u8>,
        ))(value)?;

        match packet_type_id {
            Self::PACKET_TYPE_ID_LITERAL => Self::parse_literal(packet_version, value),
            // UNWRAP SAFETY: packet_type_id can only have 1s in its least-significant 3 bits,
            // guaranteeing packet_type_id < 8, and it's not the type ID of a literal.
            _ => Self::parse_operator(packet_version, packet_type_id.try_into().unwrap(), value),
        }
    }

    fn parse_value<'i, const BITS: usize, T: PrimInt>(value: HexStr<'i>) -> IResult<HexStr<'i>, T> {
        if value.len() >= { BITS } {
            // Const generics need work
            // const_assert!(BITS <= size_of::<T>() * u8::BITS as usize);
            assert!(BITS <= size_of::<T>() * u8::BITS as usize);

            Ok((
                value.index({ BITS }..),
                value.index(..{ BITS }).try_into_prim_int::<T>().unwrap(),
            ))
        } else {
            Err(Err::Failure(Error::new(value, ErrorKind::Eof)))
        }
    }

    fn parse_literal<'i>(packet_version: u8, value: HexStr<'i>) -> IResult<HexStr<'i>, Self> {
        let mut should_continue: bool = true;
        let mut literal_value: u64 = 0_u64;

        let (value, _) = many1_count(terminated(
            |value| {
                if should_continue {
                    let (value, prefix) =
                        Self::parse_value::<{ Self::LITERAL_GROUP_PREFIX_LEN }, u8>(value)?;
                    should_continue = prefix != 0_u8;

                    Ok((value, ()))
                } else {
                    Err(Err::Error(Error::new(value, ErrorKind::Many0Count)))
                }
            },
            map(
                Self::parse_value::<{ Self::LITERAL_GROUP_PAYLOAD_LEN }, u64>,
                |payload| {
                    literal_value <<= Self::LITERAL_GROUP_PAYLOAD_LEN;
                    literal_value |= payload;
                },
            ),
        ))(value)?;

        Ok((
            value,
            Self::Literal {
                packet_version,
                value: literal_value,
            },
        ))
    }

    fn parse_operator<'i>(
        packet_version: u8,
        operator_type: OperatorType,
        value: HexStr<'i>,
    ) -> IResult<HexStr<'i>, Self> {
        let (value, len_type_id) = Self::parse_value::<{ Self::LEN_TYPE_ID_LEN }, u8>(value)?;
        let (value, sub_packets) = match len_type_id {
            0_u8 => Self::parse_sub_packets_len_type_0(value),
            1_u8 => Self::parse_sub_packets_len_type_1(value),
            _ => unreachable!(),
        }?;

        Ok((
            value,
            Self::Operator {
                packet_version,
                operator_type,
                sub_packets,
            },
        ))
    }

    fn parse_sub_packets_len_type_0<'i>(value: HexStr<'i>) -> IResult<HexStr<'i>, Vec<Self>> {
        let (value, total_len) =
            Self::parse_value::<{ Self::LEN_TYPE_ID_0_TOTAL_LEN_LEN }, usize>(value)?;

        if let Some(sub_packets_value) = value.get(..total_len) {
            let mut iter = iterator(sub_packets_value, |value: HexStr<'i>| {
                if value.is_empty() {
                    Err(Err::Error(Error::new(value, ErrorKind::Many0Count)))
                } else {
                    Self::parse(value)
                }
            });
            let sub_packets: Vec<Self> = iter.collect();
            let (remaining_value, _) = iter.finish()?;

            if remaining_value.is_empty() {
                Ok((value.index(total_len..), sub_packets))
            } else {
                Err(Err::Failure(Error::new(value, ErrorKind::TooLarge)))
            }
        } else {
            Err(Err::Failure(Error::new(value, ErrorKind::Eof)))
        }
    }

    fn parse_sub_packets_len_type_1<'i>(value: HexStr<'i>) -> IResult<HexStr<'i>, Vec<Self>> {
        let (value, sub_packets_len) =
            Self::parse_value::<{ Self::LEN_TYPE_ID_1_SUB_PACKETS_LEN_LEN }, usize>(value)?;
        let mut sub_packets: Vec<Self> = Vec::with_capacity(sub_packets_len);
        let (remaining_value, _) = many0_count(|value| {
            if sub_packets.len() >= sub_packets_len {
                Err(Err::Error(Error::new(value, ErrorKind::Many0Count)))
            } else {
                let (value, sub_packet) = Self::parse(value)?;

                sub_packets.push(sub_packet);

                Ok((value, ()))
            }
        })(value)?;

        match sub_packets.len().cmp(&sub_packets_len) {
            Ordering::Less => Err(Err::Failure(Error::new(value, ErrorKind::Eof))),
            Ordering::Equal => Ok((remaining_value, sub_packets)),
            Ordering::Greater => Err(Err::Failure(Error::new(value, ErrorKind::TooLarge))),
        }
    }

    #[cfg(test)]
    fn sub_packets(&self) -> Option<&[Packet]> {
        match self {
            Self::Literal { .. } => None,
            Self::Operator { sub_packets, .. } => Some(sub_packets),
        }
    }
}

impl<'i> TryFrom<&'i str> for Packet {
    type Error = Err<Error<&'i str>>;

    fn try_from(value: &'i str) -> Result<Self, Self::Error> {
        match HexStr::try_from(value) {
            Ok(value) => match Self::parse(value) {
                Ok((_, packet)) => Ok(packet),
                Err(err) => Err(err.map_input(|value| value.hex_str)),
            },
            Err(()) => Err(Err::Failure(Error::new(value, ErrorKind::Fail))),
        }
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub struct Solution(Packet);

impl Solution {
    fn sum_packet_versions(&self) -> u32 {
        self.0.sum_packet_versions()
    }

    fn value(&self) -> u64 {
        self.0.value()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_packet_versions());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.value());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    macro_rules! hex_str {
        ($hex_str:expr, $start_bit:expr, $end_bit:expr) => {
            HexStr {
                hex_str: $hex_str,
                start_bit: $start_bit,
                end_bit: $end_bit,
            }
        };
        ($hex_str:expr) => {
            HexStr {
                hex_str: $hex_str,
                start_bit: HexStr::BITS_PER_HEX_BYTE,
                end_bit: 0_u8,
            }
        };
    }

    const EXAMPLES: usize = 15_usize;
    const PACKET_STRS: [&str; EXAMPLES] = [
        "D2FE28",
        "38006F45291200",
        "EE00D40C823060",
        "8A004A801A8002F478",
        "620080001611562C8802118E34",
        "C0015000016115A2E0802F182340",
        "A0016C880162017C3686B18A3D4780",
        "C200B40A82",
        "04005AC33890",
        "880086C3E88112",
        "CE00C43D881120",
        "D8005AC2A8F0",
        "F600BC2D8F",
        "9C005AC2F8F0",
        "9C0141080250320F1802104A08",
    ];
    const INITIALIZERS: &[fn() -> Solution] = &[
        || {
            Solution(Packet::Literal {
                packet_version: 6_u8,
                value: 2021_u64,
            })
        },
        || {
            Solution(Packet::Operator {
                packet_version: 1_u8,
                operator_type: OperatorType::LessThan,
                sub_packets: vec![
                    Packet::Literal {
                        packet_version: 6_u8,
                        value: 10_u64,
                    },
                    Packet::Literal {
                        packet_version: 2_u8,
                        value: 20_u64,
                    },
                ],
            })
        },
        || {
            Solution(Packet::Operator {
                packet_version: 7_u8,
                operator_type: OperatorType::Maximum,
                sub_packets: vec![
                    Packet::Literal {
                        packet_version: 2_u8,
                        value: 1_u64,
                    },
                    Packet::Literal {
                        packet_version: 4_u8,
                        value: 2_u64,
                    },
                    Packet::Literal {
                        packet_version: 1_u8,
                        value: 3_u64,
                    },
                ],
            })
        },
    ];
    const ABCDEF: HexStr = hex_str!("ABCDEF");
    const HEX_STR_0: HexStr = hex_str!(PACKET_STRS[0]);
    const HEX_STR_1: HexStr = hex_str!(PACKET_STRS[1]);
    const HEX_STR_2: HexStr = hex_str!(PACKET_STRS[2]);

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCKS: OnceLock<Vec<OnceLock<Solution>>> = OnceLock::new();

        ONCE_LOCKS.get_or_init(|| {
            let mut once_locks: Vec<OnceLock<Solution>> = Vec::with_capacity(EXAMPLES);

            for _ in 0_usize..EXAMPLES {
                once_locks.push(OnceLock::new());
            }

            once_locks
        })[index]
            .get_or_init(|| {
                INITIALIZERS.get(index).map_or_else(
                    || PACKET_STRS[index].try_into().unwrap(),
                    |initializer| initializer(),
                )
            })
    }

    #[test]
    fn test_hex_str_iter_elements() {
        assert_eq!(
            ABCDEF.iter_elements().collect::<Vec<bool>>(),
            vec![
                true, false, true, false, true, false, true, true, true, true, false, false, true,
                true, false, true, true, true, true, false, true, true, true, true,
            ]
        );
    }

    #[test]
    fn test_hex_str_index() {
        assert_eq!(ABCDEF.index(..), ABCDEF);
        assert_eq!(ABCDEF.index(..12), hex_str!("ABC"));
        assert_eq!(ABCDEF.index(12..), hex_str!("DEF"));
        assert_eq!(ABCDEF.index(12..), hex_str!("DEF"));
        assert_eq!(ABCDEF.index(1..), hex_str!("ABCDEF", 3, 0));
        assert_eq!(ABCDEF.index(1..), hex_str!("579BDE", 4, 1));
        assert_eq!(ABCDEF.index(..23), hex_str!("ABCDEF", 4, 1));
        assert_eq!(ABCDEF.index(..23), hex_str!("55E6F7", 3, 0));
    }

    #[test]
    fn test_packet_parse_value() {
        assert_eq!(
            Packet::parse_value::<3, u8>(HEX_STR_0.index(0..)),
            Ok((HEX_STR_0.index(3..), 6))
        );
        assert_eq!(
            Packet::parse_value::<3, u8>(HEX_STR_0.index(3..)),
            Ok((HEX_STR_0.index(6..), 4))
        );
        assert_eq!(
            Packet::parse_value::<1, u8>(HEX_STR_0.index(6..)),
            Ok((HEX_STR_0.index(7..), 1))
        );
        assert_eq!(
            Packet::parse_value::<4, u8>(HEX_STR_0.index(7..)),
            Ok((HEX_STR_0.index(11..), 7))
        );
        assert_eq!(
            Packet::parse_value::<1, u8>(HEX_STR_0.index(11..)),
            Ok((HEX_STR_0.index(12..), 1))
        );
        assert_eq!(
            Packet::parse_value::<4, u8>(HEX_STR_0.index(12..)),
            Ok((HEX_STR_0.index(16..), 14))
        );
        assert_eq!(
            Packet::parse_value::<1, u8>(HEX_STR_0.index(16..)),
            Ok((HEX_STR_0.index(17..), 0))
        );
        assert_eq!(
            Packet::parse_value::<4, u8>(HEX_STR_0.index(17..)),
            Ok((HEX_STR_0.index(21..), 5))
        );
    }

    #[test]
    fn test_packet_parse_literal() {
        assert_eq!(
            Packet::parse_literal(6_u8, HEX_STR_0.index(6..)),
            Ok((HEX_STR_0.index(21..), solution(0_usize).0.clone()))
        );
    }

    #[test]
    fn test_packet_parse_sub_packets_len_type_0() {
        assert_eq!(
            Packet::parse_sub_packets_len_type_0(HEX_STR_1.index(7..)),
            Ok((
                HEX_STR_1.index(49..),
                solution(1_usize).0.sub_packets().unwrap().into()
            ))
        );
    }

    #[test]
    fn test_packet_parse_sub_packets_len_type_1() {
        assert_eq!(
            Packet::parse_sub_packets_len_type_1(HEX_STR_2.index(7..)),
            Ok((
                HEX_STR_2.index(51..),
                solution(2_usize).0.sub_packets().unwrap().into()
            ))
        );
    }

    #[test]
    fn test_solution_try_from_str() {
        assert_eq!(
            Solution::try_from(PACKET_STRS[0]).as_ref(),
            Ok(solution(0_usize))
        );
        assert_eq!(
            Solution::try_from(PACKET_STRS[1]).as_ref(),
            Ok(solution(1_usize))
        );
        assert_eq!(
            Solution::try_from(PACKET_STRS[2]).as_ref(),
            Ok(solution(2_usize))
        );

        match &solution(3_usize).0 {
            Packet::Operator {
                packet_version,
                sub_packets,
                ..
            } => {
                assert_eq!(*packet_version, 4_u8);
                assert_eq!(sub_packets.len(), 1_usize);

                match sub_packets.first().unwrap() {
                    Packet::Operator {
                        packet_version,
                        sub_packets,
                        ..
                    } => {
                        assert_eq!(*packet_version, 1_u8);
                        assert_eq!(sub_packets.len(), 1_usize);

                        match sub_packets.first().unwrap() {
                            Packet::Operator {
                                packet_version,
                                sub_packets,
                                ..
                            } => {
                                assert_eq!(*packet_version, 5_u8);
                                assert_eq!(sub_packets.len(), 1_usize);

                                match sub_packets.first().unwrap() {
                                    Packet::Literal { packet_version, .. } => {
                                        assert_eq!(*packet_version, 6_u8)
                                    }
                                    _ => panic!("Unexpected Packet variant for Solution 3"),
                                }
                            }
                            _ => panic!("Unexpected Packet variant for Solution 3"),
                        }
                    }
                    _ => panic!("Unexpected Packet variant for Solution 3"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 3"),
        }

        match &solution(4_usize).0 {
            Packet::Operator {
                packet_version,
                sub_packets,
                ..
            } => {
                assert_eq!(*packet_version, 3_u8);
                assert_eq!(sub_packets.len(), 2_usize);

                for sub_packet in sub_packets.iter() {
                    match sub_packet {
                        Packet::Operator { sub_packets, .. } => {
                            assert_eq!(sub_packets.len(), 2_usize);

                            for sub_packet in sub_packets.iter() {
                                assert!(matches!(sub_packet, Packet::Literal { .. }));
                            }
                        }
                        _ => panic!("Unexpected Packet variant for Solution 4"),
                    }
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 4"),
        }

        match &solution(5_usize).0 {
            Packet::Operator { sub_packets, .. } => {
                assert_eq!(sub_packets.len(), 2_usize);

                for sub_packet in sub_packets.iter() {
                    match sub_packet {
                        Packet::Operator { sub_packets, .. } => {
                            assert_eq!(sub_packets.len(), 2_usize);

                            for sub_packet in sub_packets.iter() {
                                assert!(matches!(sub_packet, Packet::Literal { .. }));
                            }
                        }
                        _ => panic!("Unexpected Packet variant for Solution 5"),
                    }
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 5"),
        }

        match &solution(6_usize).0 {
            Packet::Operator { sub_packets, .. } => {
                assert_eq!(sub_packets.len(), 1_usize);

                match sub_packets.first().unwrap() {
                    Packet::Operator { sub_packets, .. } => {
                        assert_eq!(sub_packets.len(), 1_usize);

                        match sub_packets.first().unwrap() {
                            Packet::Operator { sub_packets, .. } => {
                                assert_eq!(sub_packets.len(), 5_usize);

                                for sub_packet in sub_packets.iter() {
                                    assert!(matches!(sub_packet, Packet::Literal { .. }));
                                }
                            }
                            _ => panic!("Unexpected Packet variant for Solution 6"),
                        }
                    }
                    _ => panic!("Unexpected Packet variant for Solution 6"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 6"),
        }

        match &solution(7_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::Sum);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 1_u64),
                    _ => panic!("Unexpected Packet variant for Solution 7"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 2_u64),
                    _ => panic!("Unexpected Packet variant for Solution 7"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 7"),
        }

        match &solution(8_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::Product);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 6_u64),
                    _ => panic!("Unexpected Packet variant for Solution 8"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 9_u64),
                    _ => panic!("Unexpected Packet variant for Solution 8"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 8"),
        }

        match &solution(9_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::Minimum);
                assert_eq!(sub_packets.len(), 3_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 7_u64),
                    _ => panic!("Unexpected Packet variant for Solution 9"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 8_u64),
                    _ => panic!("Unexpected Packet variant for Solution 9"),
                }

                match &sub_packets[2_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 9_u64),
                    _ => panic!("Unexpected Packet variant for Solution 9"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 9"),
        }

        match &solution(10_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::Maximum);
                assert_eq!(sub_packets.len(), 3_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 7_u64),
                    _ => panic!("Unexpected Packet variant for Solution 10"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 8_u64),
                    _ => panic!("Unexpected Packet variant for Solution 10"),
                }

                match &sub_packets[2_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 9_u64),
                    _ => panic!("Unexpected Packet variant for Solution 10"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 10"),
        }

        match &solution(11_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::LessThan);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 5_u64),
                    _ => panic!("Unexpected Packet variant for Solution 11"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 15_u64),
                    _ => panic!("Unexpected Packet variant for Solution 11"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 11"),
        }

        match &solution(12_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::GreaterThan);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 5_u64),
                    _ => panic!("Unexpected Packet variant for Solution 12"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 15_u64),
                    _ => panic!("Unexpected Packet variant for Solution 12"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 12"),
        }

        match &solution(13_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::EqualTo);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 5_u64),
                    _ => panic!("Unexpected Packet variant for Solution 13"),
                }

                match &sub_packets[1_usize] {
                    Packet::Literal { value, .. } => assert_eq!(*value, 15_u64),
                    _ => panic!("Unexpected Packet variant for Solution 13"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 13"),
        }

        match &solution(14_usize).0 {
            Packet::Operator {
                operator_type,
                sub_packets,
                ..
            } => {
                assert_eq!(*operator_type, OperatorType::EqualTo);
                assert_eq!(sub_packets.len(), 2_usize);

                match &sub_packets[0_usize] {
                    Packet::Operator {
                        operator_type,
                        sub_packets,
                        ..
                    } => {
                        assert_eq!(*operator_type, OperatorType::Sum);
                        assert_eq!(sub_packets.len(), 2_usize);

                        match &sub_packets[0_usize] {
                            Packet::Literal { value, .. } => assert_eq!(*value, 1_u64),
                            _ => panic!("Unexpected Packet variant for Solution 14"),
                        }

                        match &sub_packets[1_usize] {
                            Packet::Literal { value, .. } => assert_eq!(*value, 3_u64),
                            _ => panic!("Unexpected Packet variant for Solution 14"),
                        }
                    }
                    _ => panic!("Unexpected Packet variant for Solution 14"),
                }

                match &sub_packets[1_usize] {
                    Packet::Operator {
                        operator_type,
                        sub_packets,
                        ..
                    } => {
                        assert_eq!(*operator_type, OperatorType::Product);
                        assert_eq!(sub_packets.len(), 2_usize);

                        match &sub_packets[0_usize] {
                            Packet::Literal { value, .. } => assert_eq!(*value, 2_u64),
                            _ => panic!("Unexpected Packet variant for Solution 14"),
                        }

                        match &sub_packets[1_usize] {
                            Packet::Literal { value, .. } => assert_eq!(*value, 2_u64),
                            _ => panic!("Unexpected Packet variant for Solution 14"),
                        }
                    }
                    _ => panic!("Unexpected Packet variant for Solution 14"),
                }
            }
            _ => panic!("Unexpected Packet variant for Solution 14"),
        }
    }

    #[test]
    fn test_sum_packet_versions() {
        assert_eq!(solution(0_usize).sum_packet_versions(), 6_u32);
        assert_eq!(solution(1_usize).sum_packet_versions(), 9_u32);
        assert_eq!(solution(2_usize).sum_packet_versions(), 14_u32);
        assert_eq!(solution(3_usize).sum_packet_versions(), 16_u32);
        assert_eq!(solution(4_usize).sum_packet_versions(), 12_u32);
        assert_eq!(solution(5_usize).sum_packet_versions(), 23_u32);
        assert_eq!(solution(6_usize).sum_packet_versions(), 31_u32);
    }

    #[test]
    fn test_value() {
        assert_eq!(solution(7_usize).value(), 3_u64);
        assert_eq!(solution(8_usize).value(), 54_u64);
        assert_eq!(solution(9_usize).value(), 7_u64);
        assert_eq!(solution(10_usize).value(), 9_u64);
        assert_eq!(solution(11_usize).value(), 1_u64);
        assert_eq!(solution(12_usize).value(), 0_u64);
        assert_eq!(solution(13_usize).value(), 0_u64);
        assert_eq!(solution(14_usize).value(), 1_u64);
    }
}
