use std::cmp::Ordering;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct LetterCount {
    pub letter: u8,
    pub count: u8,
}

impl Ord for LetterCount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for LetterCount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.count
                .cmp(&other.count)
                .reverse()
                .then_with(|| self.letter.cmp(&other.letter)),
        )
    }
}

pub struct LetterCounts(pub [LetterCount; Self::LEN]);

impl LetterCounts {
    pub const LEN: usize = 26_usize;

    pub fn from_str(value: &str) -> Self {
        value.as_bytes().iter().copied().into()
    }
}

impl<T: Iterator<Item = u8>> From<T> for LetterCounts {
    fn from(value: T) -> Self {
        let mut letter_counts: Self = Self::default();

        for letter in value.filter(u8::is_ascii_lowercase) {
            letter_counts.0[(letter - b'a') as usize].count += 1_u8;
        }

        letter_counts.0.sort();

        letter_counts
    }
}

impl Default for LetterCounts {
    fn default() -> Self {
        let mut letter_counts: Self = Self(
            [LetterCount {
                letter: 0_u8,
                count: 0_u8,
            }; Self::LEN],
        );

        for (index, letter_count) in letter_counts.0.iter_mut().enumerate() {
            letter_count.letter = b'a' + index as u8;
        }

        letter_counts
    }
}
