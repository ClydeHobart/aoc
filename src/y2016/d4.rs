use {
    crate::*,
    nom::{
        bytes::complete::{tag, take_while, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::many0,
        sequence::{delimited, terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 4: Security Through Obscurity ---

Finally, you come across an information kiosk with a list of rooms. Of course, the list is encrypted and full of decoy data, but the instructions to decode the list are barely hidden nearby. Better remove the decoy data first.

Each room consists of an encrypted name (lowercase letters separated by dashes) followed by a dash, a sector ID, and a checksum in square brackets.

A room is real (not a decoy) if the checksum is the five most common letters in the encrypted name, in order, with ties broken by alphabetization. For example:

    aaaaa-bbb-z-y-x-123[abxyz] is a real room because the most common letters are a (5), b (3), and then a tie between x, y, and z, which are listed alphabetically.
    a-b-c-d-e-f-g-h-987[abcde] is a real room because although the letters are all tied (1 of each), the first five are listed alphabetically.
    not-a-real-room-404[oarel] is a real room.
    totally-real-room-200[decoy] is not.

Of the real rooms from the list above, the sum of their sector IDs is 1514.

What is the sum of the sector IDs of the real rooms?

--- Part Two ---

With all the decoy data out of the way, it's time to decrypt this list and get moving.

The room names are encrypted by a state-of-the-art shift cipher, which is nearly unbreakable without the right software. However, the information kiosk designers at Easter Bunny HQ were not expecting to deal with a master cryptographer like yourself.

To decrypt a room name, rotate each letter forward through the alphabet a number of times equal to the room's sector ID. A becomes B, B becomes C, Z becomes A, and so on. Dashes become spaces.

For example, the real name for qzmt-zixmtkozy-ivhz-343 is very encrypted name.

What is the sector ID of the room where North Pole objects are stored? */

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Room {
    encrypted_name: String,
    sector_id: u32,
    checksum: [u8; Self::CHECKSUM_LEN],
}

impl Room {
    const CHECKSUM_LEN: usize = 5_usize;

    fn is_real(&self) -> bool {
        LetterCounts::from_str(self.encrypted_name.as_str())
            .0
            .into_iter()
            .zip(self.checksum)
            .all(|(letter_count, checksum_letter)| letter_count.letter == checksum_letter)
    }

    fn real_name(&self) -> String {
        let letter_count: u8 = LetterCounts::LEN as u8;
        let shift: u8 = (self.sector_id % letter_count as u32) as u8;

        self.encrypted_name
            .as_bytes()
            .iter()
            .copied()
            .map(|b| {
                if b == b'-' {
                    ' '
                } else {
                    ((b - b'a' + shift) % letter_count + b'a') as char
                }
            })
            .collect()
    }
}

impl Parse for Room {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                map_opt(
                    take_while(|c: char| c.is_ascii_lowercase() || c == '-'),
                    |encrypted_name: &str| {
                        let encrypted_name_len: usize = encrypted_name.len();

                        if encrypted_name_len >= 2_usize
                            && encrypted_name.as_bytes()[encrypted_name_len - 1_usize] == b'-'
                            && encrypted_name.as_bytes()[encrypted_name_len - 2_usize]
                                .is_ascii_lowercase()
                        {
                            Some(encrypted_name[..encrypted_name_len - 1_usize].into())
                        } else {
                            None
                        }
                    },
                ),
                parse_integer,
                delimited(
                    tag("["),
                    take_while_m_n(Self::CHECKSUM_LEN, Self::CHECKSUM_LEN, |c: char| {
                        c.is_ascii_lowercase()
                    }),
                    tag("]"),
                ),
            )),
            |(encrypted_name, sector_id, checksum_str)| {
                let mut checksum: [u8; Self::CHECKSUM_LEN] = [b'a'; Self::CHECKSUM_LEN];

                checksum.copy_from_slice(checksum_str.as_bytes());

                Self {
                    encrypted_name,
                    sector_id,
                    checksum,
                }
            },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Room>);

impl Solution {
    fn iter_real_rooms(&self) -> impl Iterator<Item = &Room> {
        self.0.iter().filter(|room| room.is_real())
    }

    fn sum_real_sector_ids(&self) -> u32 {
        self.iter_real_rooms().map(|room| room.sector_id).sum()
    }

    fn print_real_room_names(&self) {
        for room in self.iter_real_rooms() {
            println!("{}: {}", room.sector_id, room.real_name());
        }
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Room::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let room_count: usize = self.0.len();
            let mut real_room_count: usize = 0_usize;
            let mut real_sector_id_sum: u32 = 0_u32;

            for room in self.0.iter().filter(|room| room.is_real()) {
                real_room_count += 1_usize;
                real_sector_id_sum += room.sector_id;
            }

            dbg!(room_count, real_room_count, real_sector_id_sum);
        } else {
            dbg!(self.sum_real_sector_ids());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        self.print_real_room_names();
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

    const SOLUTION_STR: &'static str = concat!(
        "aaaaa-bbb-z-y-x-123[abxyz]\n", // is a real room because the most common letters are a (5), b (3), and then a tie between x, y, and z, which are listed alphabetically.
        "a-b-c-d-e-f-g-h-987[abcde]\n", // is a real room because although the letters are all tied (1 of each), the first five are listed alphabetically.
        "not-a-real-room-404[oarel]\n", // is a real room.
        "totally-real-room-200[decoy]\n", // is not.
        "qzmt-zixmtkozy-ivhz-343[aaaaa]\n"
    );

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        macro_rules! solution {
            [ $( { $encrypted_name:expr, $sector_id:expr, [ $checksum: expr ] }, )* ] => {
                Solution(vec![ $(
                    Room {
                        encrypted_name: $encrypted_name.into(),
                        sector_id: $sector_id,
                        checksum: {
                            let mut checksum: [u8; Room::CHECKSUM_LEN] = Default::default();

                            checksum.copy_from_slice($checksum.as_bytes());

                            checksum
                        },
                    },
                )* ])
            };
        }

        ONCE_LOCK.get_or_init(|| {
            solution![
                { "aaaaa-bbb-z-y-x", 123, ["abxyz"] }, // is a real room because the most common letters are a (5), b (3), and then a tie between x, y, and z, which are listed alphabetically.
                { "a-b-c-d-e-f-g-h", 987, ["abcde"] }, // is a real room because although the letters are all tied (1 of each), the first five are listed alphabetically.
                { "not-a-real-room", 404, ["oarel"] }, // is a real room.
                { "totally-real-room", 200, ["decoy"] }, // is not.
                { "qzmt-zixmtkozy-ivhz", 343, ["aaaaa"] }, // real name test of not real room
            ]
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_is_real() {
        assert_eq!(
            solution()
                .0
                .iter()
                .map(Room::is_real)
                .collect::<Vec<bool>>(),
            vec![true, true, true, false, false]
        );
    }

    #[test]
    fn test_sum_real_sector_ids() {
        assert_eq!(solution().sum_real_sector_ids(), 1514_u32);
    }

    #[test]
    fn test_real_name() {
        assert_eq!(
            solution().0[4_usize].real_name(),
            "very encrypted name".to_owned()
        );
    }
}
