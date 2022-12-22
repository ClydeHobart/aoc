use {
    aoc_2022::*,
    std::{cmp::Ordering, num::ParseIntError, str::FromStr},
};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ListElement {
    number: i64,
    next_index: u32,
    prev_index: u32,
}

impl ListElement {
    fn prev_index(self) -> u32 {
        self.prev_index
    }

    fn next_index(self) -> u32 {
        self.next_index
    }

    fn prev_index_mut(&mut self) -> &mut u32 {
        &mut self.prev_index
    }

    fn next_index_mut(&mut self) -> &mut u32 {
        &mut self.next_index
    }
}

#[derive(Debug, PartialEq)]
struct EncryptedFile {
    list: Vec<ListElement>,
    zero: usize,
    modulo_divisor: i64,
}

impl EncryptedFile {
    fn mix(&mut self) {
        for index in 0_usize..self.list.len() {
            // Moving one number is equivalent to keeping it stationary while rotation the
            // remaining `self.list.len() - 1_usize` elements, so the period is `self.list.len() -
            // 1_usize`, not `self.list.len()`
            let number: i64 = self.list[index].number % self.modulo_divisor;

            match number.cmp(&0_i64) {
                Ordering::Less => {
                    self.mix_index_and_number(
                        index,
                        number,
                        ListElement::prev_index,
                        ListElement::next_index,
                        ListElement::prev_index_mut,
                        ListElement::next_index_mut,
                    );
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    self.mix_index_and_number(
                        index,
                        number,
                        ListElement::next_index,
                        ListElement::prev_index,
                        ListElement::next_index_mut,
                        ListElement::prev_index_mut,
                    );
                }
            }
        }
    }

    fn mix_index_and_number<
        N: Fn(ListElement) -> u32,
        P: Fn(ListElement) -> u32,
        NM: Fn(&mut ListElement) -> &mut u32,
        PM: Fn(&mut ListElement) -> &mut u32,
    >(
        &mut self,
        index_1: usize,
        number: i64,
        next_index: N,
        prev_index: P,
        next_index_mut: NM,
        prev_index_mut: PM,
    ) {
        let mut index_2: usize = index_1;

        for _ in 0_i64..number.abs() {
            index_2 = next_index(self.list[index_2]) as usize;
        }

        let index_1_next: usize = next_index(self.list[index_1]) as usize;
        let index_1_prev: usize = prev_index(self.list[index_1]) as usize;
        let index_2_next: usize = next_index(self.list[index_2]) as usize;

        let mut set_neighbors = |prev_index: usize, next_index: usize| {
            *next_index_mut(&mut self.list[prev_index]) = next_index as u32;
            *prev_index_mut(&mut self.list[next_index]) = prev_index as u32;
        };

        set_neighbors(index_1_prev, index_1_next);
        set_neighbors(index_2, index_1);
        set_neighbors(index_1, index_2_next);
    }

    fn grove_coordinates(&self, delta: usize) -> [i64; 3_usize] {
        let mut index: usize = self.zero;
        let mut grove_coordinates: [i64; 3_usize] = [0_i64; 3_usize];

        for grove_coordinate in grove_coordinates.iter_mut() {
            for _ in 0_usize..delta {
                index = self.list[index].next_index as usize;
            }

            *grove_coordinate = self.list[index].number;
        }

        grove_coordinates
    }

    fn grove_coordinates_sum(&self, delta: usize) -> i64 {
        self.grove_coordinates(delta).into_iter().sum()
    }

    fn grove_coordinates_sum_after_mixing(&mut self, delta: usize) -> i64 {
        self.mix();

        self.grove_coordinates_sum(delta)
    }
}

#[derive(Debug, PartialEq)]
enum ParseEncryptedFileError {
    FailedToParseNumber(ParseIntError),
    TooFewNumbers,
    TooManyNumbers,
    NoZeroPresent,
}

impl TryFrom<&str> for EncryptedFile {
    type Error = ParseEncryptedFileError;

    fn try_from(encrypted_file_str: &str) -> Result<Self, Self::Error> {
        use ParseEncryptedFileError::*;

        let mut encrypted_file: Self = Self {
            list: Vec::new(),
            zero: usize::MAX,
            modulo_divisor: 0_i64,
        };

        for list_element_str in encrypted_file_str.split('\n') {
            let number: i64 = i64::from_str(list_element_str).map_err(FailedToParseNumber)?;

            if number == 0_i64 {
                encrypted_file.zero = encrypted_file.list.len();
            }

            encrypted_file.list.push(ListElement {
                number,
                ..Default::default()
            });
        }

        if encrypted_file.list.len() <= 2_usize {
            Err(TooFewNumbers)
        } else if encrypted_file.list.len() > i64::MAX as usize + 1_usize {
            Err(TooManyNumbers)
        } else if encrypted_file.zero == usize::MAX {
            Err(NoZeroPresent)
        } else {
            encrypted_file.modulo_divisor = (encrypted_file.list.len() - 1_usize) as i64;

            // Slice has no `windows_mut` :(
            for prev_index in 0_usize..encrypted_file.list.len() - 1_usize {
                let next_index: usize = prev_index + 1_usize;
                let adjacent_pair: &mut [ListElement] =
                    &mut encrypted_file.list[prev_index..=next_index];

                adjacent_pair[0_usize].next_index = next_index as u32;
                adjacent_pair[1_usize].prev_index = prev_index as u32;
            }

            encrypted_file.list.last_mut().unwrap().next_index = 0_u32;

            let prev_index: u32 = encrypted_file.list.len() as u32 - 1_u32;

            encrypted_file.list.first_mut().unwrap().prev_index = prev_index;

            Ok(encrypted_file)
        }
    }
}

const DELTA: usize = 1_000_usize;

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day20.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match EncryptedFile::try_from(input) {
                    Ok(mut encrypted_file) => {
                        dbg!(encrypted_file.grove_coordinates_sum_after_mixing(DELTA));
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                },
            )
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCRYPTED_FILE_STR: &str = "1\n2\n-3\n3\n-2\n0\n4";

    lazy_static! {
        static ref ENCRYPTED_FILE: EncryptedFile = encrypted_file();
        static ref ENCRYPTED_FILE_MIXED: EncryptedFile = encrypted_file_mixed();
    }

    macro_rules! list_elements {
        [ $( ($prev:literal <- $num:literal -> $next:literal), )* ] => { vec![ $(
            ListElement {
                number: $num,
                next_index: $next,
                prev_index: $prev
            },
        )* ] };
    }

    #[test]
    fn test_encrypted_file_try_from_str() {
        assert_eq!(ENCRYPTED_FILE_STR.try_into().as_ref(), Ok(&*ENCRYPTED_FILE));
    }

    #[test]
    fn test_encrypted_file_mix() {
        let mut encrypted_file: EncryptedFile = encrypted_file();

        encrypted_file.mix();

        assert_eq!(encrypted_file, *ENCRYPTED_FILE_MIXED);
    }

    #[test]
    fn test_encrypted_file_grove_coordinates() {
        assert_eq!(
            ENCRYPTED_FILE_MIXED.grove_coordinates(DELTA),
            [4_i64, -3_i64, 2_i64]
        );
    }

    #[test]
    fn test_encrypted_file_grove_coordinates_sum() {
        assert_eq!(ENCRYPTED_FILE_MIXED.grove_coordinates_sum(DELTA), 3_i64);
        assert_eq!(
            encrypted_file().grove_coordinates_sum_after_mixing(DELTA),
            3_i64
        );
    }

    fn encrypted_file() -> EncryptedFile {
        EncryptedFile {
            list: list_elements![
                (6 <-  1 -> 1),
                (0 <-  2 -> 2),
                (1 <- -3 -> 3),
                (2 <-  3 -> 4),
                (3 <- -2 -> 5),
                (4 <-  0 -> 6),
                (5 <-  4 -> 0),
            ],
            zero: 5_usize,
            modulo_divisor: 6_i64,
        }
    }

    fn encrypted_file_mixed() -> EncryptedFile {
        EncryptedFile {
            list: list_elements![
                (4 <-  1 -> 1),
                (0 <-  2 -> 2),
                (1 <- -3 -> 6),
                (5 <-  3 -> 4),
                (3 <- -2 -> 0),
                (6 <-  0 -> 3),
                (2 <-  4 -> 5),
            ],
            zero: 5_usize,
            modulo_divisor: 6_i64,
        }
    }
}
