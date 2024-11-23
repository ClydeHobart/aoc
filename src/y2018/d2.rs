use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::take_while1, character::complete::line_ending, combinator::opt,
        error::Error, multi::fold_many0, sequence::terminated, Err, IResult,
    },
    std::ops::Range,
};

/* --- Day 2: Inventory Management System ---

You stop falling through time, catch your breath, and check the screen on the device. "Destination reached. Current Year: 1518. Current Location: North Pole Utility Closet 83N10." You made it! Now, to find those anomalies.

Outside the utility closet, you hear footsteps and a voice. "...I'm not sure either. But now that so many people have chimneys, maybe he could sneak in that way?" Another voice responds, "Actually, we've been working on a new kind of suit that would let him fit through tight spaces like that. But, I heard that a few days ago, they lost the prototype fabric, the design plans, everything! Nobody on the team can even seem to remember important details of the project!"

"Wouldn't they have had enough fabric to fill several boxes in the warehouse? They'd be stored together, so the box IDs should be similar. Too bad it would take forever to search the warehouse for two similar box IDs..." They walk too far away to hear any more.

Late at night, you sneak to the warehouse - who knows what kinds of paradoxes you could cause if you were discovered - and use your fancy wrist device to quickly scan every box and produce a list of the likely candidates (your puzzle input).

To make sure you didn't miss any, you scan the likely candidate boxes again, counting the number that have an ID containing exactly two of any letter and then separately counting those with exactly three of any letter. You can multiply those two counts together to get a rudimentary checksum and compare it to what your device predicts.

For example, if you see the following box IDs:

    abcdef contains no letters that appear exactly two or three times.
    bababc contains two a and three b, so it counts for both.
    abbcde contains two b, but no letter appears exactly three times.
    abcccd contains three c, but no letter appears exactly two times.
    aabcdd contains two a and two d, but it only counts once.
    abcdee contains two e.
    ababab contains three a and three b, but it only counts once.

Of these box IDs, four of them contain a letter which appears exactly twice, and three of them contain a letter which appears exactly three times. Multiplying these together produces a checksum of 4 * 3 = 12.

What is the checksum for your list of box IDs?

--- Part Two ---

Confident that your list of box IDs is complete, you're ready to find the boxes full of prototype fabric.

The boxes will have IDs which differ by exactly one character at the same position in both strings. For example, given the following box IDs:

abcde
fghij
klmno
pqrst
fguij
axcye
wvxyz

The IDs abcde and axcye are close, but they differ by two characters (the second and fourth). However, the IDs fghij and fguij differ by exactly one character, the third (h and u). Those must be the correct boxes.

What letters are common between the two correct box IDs? (In the example above, this is found by removing the differing character from either ID, producing fgij.) */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    id_ranges: Vec<Range<u16>>,
    id_string: String,
}

impl Solution {
    const LETTER_COUNT_A: u8 = 2_u8;
    const LETTER_COUNT_B: u8 = 3_u8;

    fn iter_differing_char_indices<'i>(
        id_a: &'i str,
        id_b: &'i str,
    ) -> impl Iterator<Item = usize> + 'i {
        id_a.chars()
            .zip(id_b.chars())
            .enumerate()
            .filter_map(|(index, (char_a, char_b))| (char_a != char_b).then_some(index))
            .chain({
                let id_a_len: usize = id_a.len();
                let id_b_len: usize = id_b.len();
                let id_len_range: Range<usize> = id_a_len.min(id_b_len)..id_a_len.max(id_b_len);

                (!id_len_range.is_empty())
                    .then_some(id_len_range)
                    .into_iter()
                    .flatten()
            })
    }

    fn iter_ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.id_ranges
            .iter()
            .map(|id_range| &self.id_string[id_range.as_range_usize()])
    }

    fn iter_id_letter_counts(&self) -> impl Iterator<Item = LetterCounts> + '_ {
        self.iter_ids().map(LetterCounts::from_str)
    }

    fn checksum(&self) -> i32 {
        let id_counts: IVec2 = self
            .iter_id_letter_counts()
            .map(|letter_counts| {
                IVec2::new(
                    letter_counts
                        .try_find_letter_count_with_count(Self::LETTER_COUNT_A)
                        .is_some() as i32,
                    letter_counts
                        .try_find_letter_count_with_count(Self::LETTER_COUNT_B)
                        .is_some() as i32,
                )
            })
            .sum();

        id_counts.x * id_counts.y
    }

    fn id_count(&self) -> usize {
        self.id_ranges.len()
    }

    fn get_id(&self, index: usize) -> &str {
        &self.id_string[self.id_ranges[index].as_range_usize()]
    }

    fn iter_id_pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        (0_usize..self.id_count()).flat_map(move |index_a| {
            ((index_a + 1_usize)..self.id_count())
                .map(move |index_b| (self.get_id(index_a), self.get_id(index_b)))
        })
    }

    fn find_prototype_fabric_box_ids_and_common_letters(&self) -> Option<(&str, &str, String)> {
        self.iter_id_pairs().find_map(|(id_a, id_b)| {
            let mut differing_char_indices_iter =
                Self::iter_differing_char_indices(id_a, id_b).peekable();

            let differing_char_index: Option<usize> = differing_char_indices_iter.peek().copied();

            (differing_char_indices_iter.count() == 1_usize).then(|| {
                let differing_char_index: usize = differing_char_index.unwrap();

                (
                    id_a,
                    id_b,
                    format!(
                        "{}{}",
                        &id_a[..differing_char_index],
                        &id_a[differing_char_index + 1_usize..]
                    ),
                )
            })
        })
    }

    fn find_prototype_fabric_box_id_common_letters(&self) -> Option<String> {
        self.find_prototype_fabric_box_ids_and_common_letters()
            .map(|(_, _, common_letters)| common_letters)
    }

    fn push_id(&mut self, id: &str) {
        let id_range_start: u16 = self.id_string.len() as u16;

        self.id_string.push_str(id);

        let id_range_end: u16 = self.id_string.len() as u16;

        self.id_ranges.push(id_range_start..id_range_end);
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        fold_many0(
            terminated(
                take_while1(|c: char| c.is_ascii_lowercase()),
                opt(line_ending),
            ),
            Self::default,
            |mut solution, id| {
                solution.push_id(id);

                solution
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.checksum());
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.find_prototype_fabric_box_ids_and_common_letters());
        } else {
            dbg!(self.find_prototype_fabric_box_id_common_letters());
        }
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        "\
        abcdef\n\
        bababc\n\
        abbcde\n\
        abcccd\n\
        aabcdd\n\
        abcdee\n\
        ababab\n",
        "\
        abcde\n\
        fghij\n\
        klmno\n\
        pqrst\n\
        fguij\n\
        axcye\n\
        wvxyz\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    id_ranges: vec![
                        0_u16..6_u16,
                        6_u16..12_u16,
                        12_u16..18_u16,
                        18_u16..24_u16,
                        24_u16..30_u16,
                        30_u16..36_u16,
                        36_u16..42_u16,
                    ],
                    id_string: "abcdefbababcabbcdeabcccdaabcddabcdeeababab".into(),
                },
                Solution {
                    id_ranges: vec![
                        0_u16..5_u16,
                        5_u16..10_u16,
                        10_u16..15_u16,
                        15_u16..20_u16,
                        20_u16..25_u16,
                        25_u16..30_u16,
                        30_u16..35_u16,
                    ],
                    id_string: "abcdefghijklmnopqrstfguijaxcyewvxyz".into(),
                },
            ]
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
    fn test_checksum() {
        for (index, checksum) in [12_i32, 0_i32].into_iter().enumerate() {
            assert_eq!(solution(index).checksum(), checksum);
        }
    }

    #[test]
    fn test_find_prototype_fabric_box_ids_and_common_letters() {
        for (index, found_prototype_fabric_box_ids_and_common_letters) in [
            ("abcdef", "abcdee", "abcde".to_owned()),
            ("fghij", "fguij", "fgij".to_owned()),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).find_prototype_fabric_box_ids_and_common_letters(),
                Some(found_prototype_fabric_box_ids_and_common_letters)
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
