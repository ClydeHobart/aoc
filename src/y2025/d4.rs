use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
};

/* --- Day 4: Printing Department ---

You ride the escalator down to the printing department. They're clearly getting ready for Christmas; they have lots of large rolls of paper everywhere, and there's even a massive printer in the corner (to handle the really big print jobs).

Decorating here will be easy: they can make their own decorations. What you really need is a way to get further into the North Pole base while the elevators are offline.

"Actually, maybe we can help with that," one of the Elves replies when you ask for help. "We're pretty sure there's a cafeteria on the other side of the back wall. If we could break through the wall, you'd be able to keep moving. It's too bad all of our forklifts are so busy moving those big rolls of paper around."

If you can optimize the work the forklifts are doing, maybe they would have time to spare to break through the wall.

The rolls of paper (@) are arranged on a large grid; the Elves even have a helpful diagram (your puzzle input) indicating where everything is located.

For example:

..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.

The forklifts can only access a roll of paper if there are fewer than four rolls of paper in the eight adjacent positions. If you can figure out which rolls of paper the forklifts can access, they'll spend less time looking and more time breaking down the wall to the cafeteria.

In this example, there are 13 rolls of paper that can be accessed by a forklift (marked with x):

..xx.xx@x.
x@@.@.@.@@
@@@@@.x.@@
@.@@@@..@.
x@.@@@@.@x
.@@@@@@@.@
.@.@.@.@@@
x.@@@.@@@@
.@@@@@@@@.
x.x.@@@.x.

Consider your complete diagram of the paper roll locations. How many rolls of paper can be accessed by a forklift?

--- Part Two ---

Now, the Elves just need help accessing as much of the paper as they can.

Once a roll of paper can be accessed by a forklift, it can be removed. Once a roll of paper is removed, the forklifts might be able to access more rolls of paper, which they might also be able to remove. How many total rolls of paper could the Elves remove if they keep repeating this process?

Starting with the same example as above, here is one way you could remove as many rolls of paper as possible, using highlighted @ to indicate that a roll of paper is about to be removed, and using x to indicate that a roll of paper was just removed:

Initial state:
..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.

Remove 13 rolls of paper:
..xx.xx@x.
x@@.@.@.@@
@@@@@.x.@@
@.@@@@..@.
x@.@@@@.@x
.@@@@@@@.@
.@.@.@.@@@
x.@@@.@@@@
.@@@@@@@@.
x.x.@@@.x.

Remove 12 rolls of paper:
.......x..
.@@.x.x.@x
x@@@@...@@
x.@@@@..x.
.@.@@@@.x.
.x@@@@@@.x
.x.@.@.@@@
..@@@.@@@@
.x@@@@@@@.
....@@@...

Remove 7 rolls of paper:
..........
.x@.....x.
.@@@@...xx
..@@@@....
.x.@@@@...
..@@@@@@..
...@.@.@@x
..@@@.@@@@
..x@@@@@@.
....@@@...

Remove 5 rolls of paper:
..........
..x.......
.x@@@.....
..@@@@....
...@@@@...
..x@@@@@..
...@.@.@@.
..x@@.@@@x
...@@@@@@.
....@@@...

Remove 2 rolls of paper:
..........
..........
..x@@.....
..@@@@....
...@@@@...
...@@@@@..
...@.@.@@.
...@@.@@@.
...@@@@@x.
....@@@...

Remove 1 roll of paper:
..........
..........
...@@.....
..x@@@....
...@@@@...
...@@@@@..
...@.@.@@.
...@@.@@@.
...@@@@@..
....@@@...

Remove 1 roll of paper:
..........
..........
...x@.....
...@@@....
...@@@@...
...@@@@@..
...@.@.@@.
...@@.@@@.
...@@@@@..
....@@@...

Remove 1 roll of paper:
..........
..........
....x.....
...@@@....
...@@@@...
...@@@@@..
...@.@.@@.
...@@.@@@.
...@@@@@..
....@@@...

Remove 1 roll of paper:
..........
..........
..........
...x@@....
...@@@@...
...@@@@@..
...@.@.@@.
...@@.@@@.
...@@@@@..
....@@@...

Stop once no more rolls of paper are accessible by a forklift. In this example, a total of 43 rolls of paper can be removed.

Start with your original diagram. How many rolls of paper in total can be removed by the Elves and their forklifts? */

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, PartialEq)]
    #[cfg_attr(test, derive(Debug))]
    enum Cell {
        Empty = EMPTY = b'.',
        PaperRoll = PAPER_ROLL = b'@',
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Cell>);

impl Solution {
    const MAX_PAPER_ROLL_NEIGHBORS_FOR_FORKLIFT_ACCESSIBLE_POS: usize = 3_usize;

    fn is_pos_forklift_accessible(&self, pos: IVec2) -> bool {
        assert!(self.0.contains(pos));

        iter_neighbors(pos)
            .filter_map(|neighbor_pos| self.0.get(neighbor_pos))
            .cloned()
            .filter(|&cell| cell == Cell::PaperRoll)
            .count()
            <= Self::MAX_PAPER_ROLL_NEIGHBORS_FOR_FORKLIFT_ACCESSIBLE_POS
    }

    fn iter_forklift_accessible_paper_rolls(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.0
            .iter_positions_with_cell(&Cell::PaperRoll)
            .filter(|&pos| self.is_pos_forklift_accessible(pos))
    }

    fn count_forklift_accessible_paper_rolls(&self) -> usize {
        self.iter_forklift_accessible_paper_rolls().count()
    }

    fn remove_all_forklift_accessible_paper_rolls(&self) -> (Self, usize) {
        let mut solution: Self = self.clone();
        let mut total_forklift_accessible_paper_rolls_count: usize = 0_usize;
        let mut forklift_accessible_paper_rolls: BitVec = bitvec![0; self.0.cells().len()];

        while {
            for pos in solution.iter_forklift_accessible_paper_rolls() {
                forklift_accessible_paper_rolls.set(solution.0.index_from_pos(pos), true);
            }

            forklift_accessible_paper_rolls
                .iter_ones()
                .fold(false, |_, index| {
                    *solution
                        .0
                        .get_mut(solution.0.pos_from_index(index))
                        .unwrap() = Cell::Empty;
                    total_forklift_accessible_paper_rolls_count += 1_usize;

                    true
                })
        } {
            forklift_accessible_paper_rolls.fill(false);
        }

        (solution, total_forklift_accessible_paper_rolls_count)
    }

    fn count_total_forklift_accessible_paper_rolls(&self) -> usize {
        self.remove_all_forklift_accessible_paper_rolls().1
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_forklift_accessible_paper_rolls());
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let (solution, total_forklift_accessible_paper_rolls_count) =
                self.remove_all_forklift_accessible_paper_rolls();

            println!("{}", String::from(solution.0));

            dbg!(total_forklift_accessible_paper_rolls_count);
        } else {
            dbg!(self.count_total_forklift_accessible_paper_rolls());
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        ..@@.@@@@.\n\
        @@@.@.@.@@\n\
        @@@@@.@.@@\n\
        @.@@@@..@.\n\
        @@.@@@@.@@\n\
        .@@@@@@@.@\n\
        .@.@.@.@@@\n\
        @.@@@.@@@@\n\
        .@@@@@@@@.\n\
        @.@.@@@.@.\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{Empty as E, PaperRoll as P};

            vec![Solution(
                Grid2D::try_from_cells_and_dimensions(
                    vec![
                        E, E, P, P, E, P, P, P, P, E, P, P, P, E, P, E, P, E, P, P, P, P, P, P, P,
                        E, P, E, P, P, P, E, P, P, P, P, E, E, P, E, P, P, E, P, P, P, P, E, P, P,
                        E, P, P, P, P, P, P, P, E, P, E, P, E, P, E, P, E, P, P, P, P, E, P, P, P,
                        E, P, P, P, P, E, P, P, P, P, P, P, P, P, E, P, E, P, E, P, P, P, E, P, E,
                    ],
                    (10, 10).into(),
                )
                .unwrap(),
            )]
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
    fn test_count_forklift_accessible_paper_rolls() {
        for (index, forklift_accessible_paper_roll_count) in [13_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_forklift_accessible_paper_rolls(),
                forklift_accessible_paper_roll_count
            );
        }
    }

    #[test]
    fn test_count_total_forklift_accessible_paper_rolls() {
        for (index, total_forklift_accessible_paper_rolls_count) in
            [43_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).count_total_forklift_accessible_paper_rolls(),
                total_forklift_accessible_paper_rolls_count
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
