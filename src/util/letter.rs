pub use letter_counts::*;

use std::ops::Range;

mod letter_counts;

pub const MIN_ASCII_LOWERCASE_LETTER: u8 = b'a';
pub const MAX_ASCII_LOWERCASE_LETTER: u8 = b'z';
pub const ASCII_LOWERCASE_LETTER_RANGE: Range<u8> =
    MIN_ASCII_LOWERCASE_LETTER..MAX_ASCII_LOWERCASE_LETTER + 1_u8;
pub const LETTER_COUNT: usize =
    (MAX_ASCII_LOWERCASE_LETTER - MIN_ASCII_LOWERCASE_LETTER + 1_u8) as usize;

pub const fn index_from_ascii_lowercase_letter(ascii_lowercase_letter: u8) -> usize {
    (ascii_lowercase_letter - MIN_ASCII_LOWERCASE_LETTER) as usize
}

pub const fn ascii_lowercase_letter_from_index(index: usize) -> u8 {
    index as u8 + MIN_ASCII_LOWERCASE_LETTER
}
