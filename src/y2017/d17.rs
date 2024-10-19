use {
    crate::*,
    nom::{combinator::map, error::Error, Err, IResult},
};

#[cfg(test)]
use std::fmt::Write;

/* --- Day 17: Spinlock ---

Suddenly, whirling in the distance, you notice what looks like a massive, pixelated hurricane: a deadly spinlock. This spinlock isn't just consuming computing power, but memory, too; vast, digital mountains are being ripped from the ground and consumed by the vortex.

If you don't move quickly, fixing that printer will be the least of your problems.

This spinlock's algorithm is simple but efficient, quickly consuming everything in its path. It starts with a circular buffer containing only the value 0, which it marks as the current position. It then steps forward through the circular buffer some number of steps (your puzzle input) before inserting the first new value, 1, after the value it stopped on. The inserted value becomes the current position. Then, it steps forward from there the same number of steps, and wherever it stops, inserts after it the second new value, 2, and uses that as the new current position again.

It repeats this process of stepping forward, inserting a new value, and using the location of the inserted value as the new current position a total of 2017 times, inserting 2017 as its final operation, and ending with a total of 2018 values (including 0) in the circular buffer.

For example, if the spinlock were to step 3 times per insert, the circular buffer would begin to evolve like this (using parentheses to mark the current position after each iteration of the algorithm):

    (0), the initial state before any insertions.
    0 (1): the spinlock steps forward three times (0, 0, 0), and then inserts the first value, 1, after it. 1 becomes the current position.
    0 (2) 1: the spinlock steps forward three times (0, 1, 0), and then inserts the second value, 2, after it. 2 becomes the current position.
    0  2 (3) 1: the spinlock steps forward three times (1, 0, 2), and then inserts the third value, 3, after it. 3 becomes the current position.

And so on:

    0  2 (4) 3  1
    0 (5) 2  4  3  1
    0  5  2  4  3 (6) 1
    0  5 (7) 2  4  3  6  1
    0  5  7  2  4  3 (8) 6  1
    0 (9) 5  7  2  4  3  8  6  1

Eventually, after 2017 insertions, the section of the circular buffer near the last insertion looks like this:

1512  1134  151 (2017) 638  1513  851

Perhaps, if you can identify the value that will ultimately be after the last value written (2017), you can short-circuit the spinlock. In this example, that would be 638.

What is the value after 2017 in your completed circular buffer?

--- Part Two ---

The spinlock does not short-circuit. Instead, it gets more angry. At least, you assume that's what happened; it's spinning significantly faster than it was a moment ago.

You have good news and bad news.

The good news is that you have improved calculations for how to stop the spinlock. They indicate that you actually need to identify the value after 0 in the current state of the circular buffer.

The bad news is that while you were determining this, the spinlock has just finished inserting its fifty millionth value (50000000).

What is the value after 0 the moment 50000000 is inserted? */

#[derive(Clone, Copy, Default)]
struct Node {
    next_pos: u32,
}

struct Buffer {
    nodes: Vec<Node>,
    offset: u32,
}

impl Buffer {
    fn new(offset: u32) -> Self {
        Self {
            nodes: vec![Node::default()],
            offset,
        }
    }

    fn insert(&mut self) {
        let nodes_len: usize = self.nodes.len();

        assert!(nodes_len < u32::MAX as usize);

        let curr_pos: u32 = nodes_len as u32;
        let prev_pos: u32 = {
            let mut pos: u32 = curr_pos - 1_u32;

            for _ in 0_u32..self.offset % curr_pos {
                pos = self.nodes[pos as usize].next_pos;
            }

            pos
        };
        let prev: &mut Node = &mut self.nodes[prev_pos as usize];
        let next_pos: u32 = prev.next_pos;

        prev.next_pos = curr_pos;
        self.nodes.push(Node { next_pos });
    }

    fn insert_many(&mut self, insertions: usize) {
        assert!(self.nodes.len() + insertions < u32::MAX as usize);

        self.nodes.reserve_exact(insertions);

        for _ in 0_usize..insertions {
            self.insert();
        }
    }

    #[cfg(test)]
    fn as_string_in_list_order(&self) -> String {
        let nodes_len: usize = self.nodes.len();
        let curr_pos: u32 = nodes_len as u32 - 1_u32;

        let mut string: String = String::new();
        let mut printed_nodes: usize = 0_usize;
        let mut pos: u32 = 0_u32;

        while printed_nodes < nodes_len {
            write!(
                &mut string,
                "{}{}{pos}{}",
                if printed_nodes != 0_usize { " " } else { "" },
                if pos == curr_pos { "(" } else { "" },
                if pos == curr_pos { ")" } else { "" },
            )
            .ok();
            printed_nodes += 1_usize;
            pos = self.nodes[pos as usize].next_pos;
        }

        string
    }

    #[cfg(test)]
    fn as_string_in_memory_order(&self) -> String {
        let mut string: String = String::new();

        for node in &self.nodes {
            write!(&mut string, "{:2} ", node.next_pos).ok();
        }

        string
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
pub struct Solution(u32);

impl Solution {
    const Q1_INSERTIONS: usize = 2017_usize;
    const Q1_VALUE: u32 = 2017_u32;
    const Q2_INSERTIONS: usize = 50_000_000_usize;
    const Q2_VALUE: u32 = 0_u32;

    fn linked_list_buffer(self) -> Buffer {
        Buffer::new(self.0)
    }

    fn value_after_value_after_insertions(self, insertions: usize, value: u32) -> u32 {
        assert!(value as usize <= insertions);

        let mut buffer: Buffer = self.linked_list_buffer();

        buffer.insert_many(insertions);

        buffer.nodes[value as usize].next_pos
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Pretty trivial. Expecting part 2 to be doozy.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.value_after_value_after_insertions(Self::Q1_INSERTIONS, Self::Q1_VALUE));
    }

    /// I had to restructure to a linked list, since rotating the `VecDeque` (my initial
    /// implementation) I think would've gotten to unwieldly? Lots of copying memory around. This
    /// does far fewer writes, but it does a lot of reading, likley with poor cach locality, since
    /// the full list is on the order of 200MB.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.value_after_value_after_insertions(Self::Q2_INSERTIONS, Self::Q2_VALUE));
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

    const SOLUTION_STRS: &'static [&'static str] = &["3"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![Solution(3_u32)])[index]
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
    fn test_buffer_insert() {
        let mut buffer: Buffer = solution(0_usize).linked_list_buffer();

        assert_eq!(buffer.as_string_in_list_order(), "(0)");

        for string in [
            "0 (1)",
            "0 (2) 1",
            "0 2 (3) 1",
            "0 2 (4) 3 1",
            "0 (5) 2 4 3 1",
            "0 5 2 4 3 (6) 1",
            "0 5 (7) 2 4 3 6 1",
            "0 5 7 2 4 3 (8) 6 1",
            "0 (9) 5 7 2 4 3 8 6 1",
        ] {
            buffer.insert();

            assert_eq!(buffer.as_string_in_list_order(), string);
        }
    }

    #[test]
    fn test_value_after_value_after_insertions() {
        for (index, value_after_value_after_insertions) in [638_u32].into_iter().enumerate() {
            assert_eq!(
                solution(index).value_after_value_after_insertions(
                    Solution::Q1_INSERTIONS,
                    Solution::Q1_VALUE
                ),
                value_after_value_after_insertions
            );
        }
    }

    #[test]
    fn test_print_triangle() {
        let mut buffer: Buffer = Buffer::new(5_u32);

        println!("{}", buffer.as_string_in_memory_order());

        for _ in 0_usize..200_usize {
            buffer.insert();

            let string: String = buffer.as_string_in_memory_order();

            println!("{}", &string[..150_usize.min(string.len())]);
        }
    }
}
