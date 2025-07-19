use {super::*, std::cmp::Ordering};

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct LetterCount {
    pub letter: u8,
    pub count: u8,
}

impl Ord for LetterCount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .reverse()
            .then_with(|| self.letter.cmp(&other.letter))
    }
}

impl PartialOrd for LetterCount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Eq, Hash, PartialEq)]
pub struct LetterCounts(pub [LetterCount; LETTER_COUNT]);

impl LetterCounts {
    pub fn from_str(value: &str) -> Self {
        value.as_bytes().iter().copied().into()
    }

    pub fn iter_letter_counts_with_count(&self, count: u8) -> impl Iterator<Item = &LetterCount> {
        self.0
            .iter()
            .skip_while(move |letter_count| letter_count.count > count)
            .take_while(move |letter_count| letter_count.count == count)
    }

    pub fn try_find_letter_count_with_count(&self, count: u8) -> Option<&LetterCount> {
        self.iter_letter_counts_with_count(count).next()
    }
}

impl<T: Iterator<Item = u8>> From<T> for LetterCounts {
    fn from(value: T) -> Self {
        let mut letter_counts: Self = Self::default();

        for letter in value.filter(u8::is_ascii_lowercase) {
            letter_counts.0[index_from_ascii_lowercase_letter(letter)].count += 1_u8;
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
            }; LETTER_COUNT],
        );

        for (index, letter_count) in letter_counts.0.iter_mut().enumerate() {
            letter_count.letter = ascii_lowercase_letter_from_index(index);
        }

        letter_counts
    }
}
