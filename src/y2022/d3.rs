use crate::*;

pub struct Solution(String);

impl Solution {
    /// Tries to convert an item character into a priority, as described by
    /// https://adventofcode.com/2022/day/3
    ///
    /// # Arguments
    ///
    /// * `item` - The item character to attempt to convert into a priority
    ///
    /// # Errors
    ///
    /// This function will print an error message and return `None` if `item` is not ASCII
    /// alphabetic.
    fn try_item_to_priority(item: char) -> Option<u32> {
        const LOWERCASE_A_OFFSET: u32 = 'a' as u32 - 1_u32;
        const UPPERCASE_A_OFFSET: u32 = 'A' as u32 - 27_u32;

        if item.is_ascii_alphabetic() {
            Some(if item.is_ascii_lowercase() {
                item as u32 - LOWERCASE_A_OFFSET
            } else {
                item as u32 - UPPERCASE_A_OFFSET
            })
        } else {
            eprintln!("Character '{}' is not ASCII alphabetic", item);

            None
        }
    }

    /// Takes inventory of the items present in a sequence, storing whether or not an item is
    /// present in the bit corresponding to the item's priority of a `u64`
    ///
    /// # Arguments
    ///
    /// * `item_sequence` - The sequence of items to take inventory of
    fn items_present_in_sequence(item_sequence: &str) -> u64 {
        let mut items_present_in_sequence: u64 = 0_u64;

        for item in item_sequence.chars() {
            if let Some(priority) = Self::try_item_to_priority(item) {
                items_present_in_sequence |= 1_u64 << priority;
            }
        }

        items_present_in_sequence
    }

    /// Computes the priority of a rucksack string slice, as described by
    /// https://adventofcode.com/2022/day/3
    ///
    /// This uses `items_present_in_sequence` to take inventory of the first compartment, then
    /// returns early when a matching item is found in the second compartment.
    ///
    /// # Arguments
    ///
    /// * `rucksack` - A string slice representing the items present in a rucksack, with the first
    ///   half of the characters representing the first compartment, and vice versa with the second
    ///   half and second compartment
    ///
    /// # Errors
    ///
    /// If the string slice does not have an even length, an error message is printed, and the last
    /// byte is ignored
    ///
    /// If the byte at the midpoint of the string slice is not a `char` boundary, `rucksack` is
    /// ill-formatted for this problem. In this case, an error message is printed, and 0 is
    /// returned.
    ///
    /// If an item couldn't be found that's present in both compartments, an error message is
    /// printed, and 0 is returned.
    fn compute_priority_of_rucksack(rucksack: &str) -> u32 {
        let mut rucksack_len: usize = rucksack.len();

        if rucksack_len % 2_usize == 1_usize {
            eprintln!(
                "Rucksack \"{}\" has odd length {}; ignoring last byte",
                rucksack, rucksack_len
            );

            rucksack_len -= 1_usize;
        }

        let midpoint: usize = rucksack_len / 2_usize;

        if !rucksack.is_char_boundary(midpoint) {
            eprintln!(
                "Rucksack \"{}\" has doesn't have a char boundary at its midpoint byte; \
                terminating early",
                rucksack
            );

            return 0_u32;
        }

        let first_compartment_items: u64 = Self::items_present_in_sequence(&rucksack[..midpoint]);

        for second_compartment_item in rucksack[midpoint..].chars() {
            if let Some(priority) = Self::try_item_to_priority(second_compartment_item) {
                if (first_compartment_items & 1_u64 << priority) != 0_u64 {
                    return priority;
                }
            }
        }

        eprintln!(
            "Couldn't find matching items in both compartments of rucksack \"{}\"",
            rucksack
        );

        0_u32
    }

    /// Sums the priorities of line-break-delineated rucksacks
    ///
    /// # Arguments
    ///
    /// * `rucksacks` - A string slice representing rucksacks, with individual rucksacks delineated by
    ///   `'\n'`
    fn sum_rucksack_priorities(&self) -> u32 {
        self.0
            .split('\n')
            .map(Self::compute_priority_of_rucksack)
            .sum()
    }

    /// Sums the priorities of rucksack groups of a given size
    ///
    /// The priority of a group is the priority of the single item within a rucksack group of a given
    /// size that all rucksacks contain. There should only be one such item.
    ///
    /// # Arguments
    ///
    /// * `rucksacks` - A string slice representing rucksacks, with individual rucksacks delineated by
    ///   `'\n'`
    /// * `group_size` - The size of the group of rucksacks to find the unique shared item of, which
    ///   should be at least 2
    ///
    /// # Errors
    ///
    /// If `group_size` is less than two, an error message is printed, and 3 is used instead.
    ///
    /// If ill-formated input is supplied such that there is more than one item shared by all
    /// `group_size` rucksacks of a given group, the lowest-priority shared item will have its priority
    /// used.
    fn sum_rucksack_priorities_for_groups(&self, mut group_size: usize) -> u32 {
        if group_size < 2_usize {
            eprintln!("A group of size {} is too small; using 3", group_size);

            group_size = 3_usize;
        }

        let group_size_minus_1: usize = group_size - 1_usize;
        let mut priority_sum: u32 = 0_u32;
        let mut group_items: u64 = u64::MAX;

        for (rucksack_index, rucksack_items) in self
            .0
            .split('\n')
            .map(Self::items_present_in_sequence)
            .enumerate()
        {
            group_items &= rucksack_items;

            if rucksack_index % group_size == group_size_minus_1 {
                priority_sum += group_items.trailing_zeros();
                group_items = u64::MAX;
            }
        }

        priority_sum
    }

    /// Sums the priorities of consecutive rucksack trios, using `sum_rucksack_priorities_for_groups`
    fn sum_rucksack_priorities_for_groups_of_3(&self) -> u32 {
        self.sum_rucksack_priorities_for_groups(3_usize)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_rucksack_priorities());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_rucksack_priorities_for_groups_of_3());
    }
}

impl TryFrom<&str> for Solution {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(Solution(value.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUCKSACKS_STR: &str = concat!(
        "vJrwpWtwJgWrhcsFMMfFFhFp\n",
        "jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL\n",
        "PmmdzqPrVvPwwTWBwg\n",
        "wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn\n",
        "ttgJtRGJQctTZtZT\n",
        "CrZsJsPPZsGzwwsLwLmpwMDw",
    );

    #[test]
    fn test_sum_rucksack_priorities() {
        assert_eq!(
            Solution::try_from(RUCKSACKS_STR)
                .unwrap()
                .sum_rucksack_priorities(),
            157
        );
    }

    #[test]
    fn test_sum_rucksack_priorities_for_groups_of_3() {
        assert_eq!(
            Solution::try_from(RUCKSACKS_STR)
                .unwrap()
                .sum_rucksack_priorities_for_groups_of_3(),
            70
        );
    }
}
