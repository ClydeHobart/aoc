use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

/* --- Day 4: Ceres Search ---

"Looks like the Chief's not here. Next!" One of The Historians pulls out a device and pushes the only button on it. After a brief flash, you recognize the interior of the Ceres monitoring station!

As the search for the Chief continues, a small Elf who lives on the station tugs on your shirt; she'd like to know if you could help her with her word search (your puzzle input). She only has to find one word: XMAS.

This word search allows words to be horizontal, vertical, diagonal, written backwards, or even overlapping other words. It's a little unusual, though, as you don't merely need to find one instance of XMAS - you need to find all of them. Here are a few ways XMAS might appear, where irrelevant characters have been replaced with .:

..X...
.SAMX.
.A..A.
XMAS.S
.X....

The actual word search will be full of letters instead. For example:

MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX

In this word search, XMAS occurs a total of 18 times; here's the same word search again, but where letters not involved in any XMAS have been replaced with .:

....XXMAS.
.SAMXMS...
...S..A...
..A.A.MS.X
XMASAMX.MM
X.....XA.A
S.S.S.S.SS
.A.A.A.A.A
..M.M.M.MM
.X.X.XMASX

Take a look at the little Elf's word search. How many times does XMAS appear?

--- Part Two ---

The Elf looks quizzically at you. Did you misunderstand the assignment?

Looking for the instructions, you flip over the word search to find that this isn't actually an XMAS puzzle; it's an X-MAS puzzle in which you're supposed to find two MAS in the shape of an X. One way to achieve that is like this:

M.S
.A.
M.S

Irrelevant characters have again been replaced with . in the above diagram. Within the X, each MAS can be written forwards or backwards.

Here's the same example from before, but this time all of the X-MASes have been kept instead:

.M.S......
..A..MSMS.
.M.S.MAA..
..A.ASMSM.
.M.S.M....
..........
S.S.S.S.S.
.A.A.A.A..
M.M.M.M.M.
..........

In this example, an X-MAS appears 9 times.

Flip the word search from the instructions back over to the word search side and try again. How many times does an X-MAS appear? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, EnumCount, EnumIter, PartialEq)]
    enum Letter {
        X = X_VALUE = b'X',
        M = M_VALUE = b'M',
        A = A_VALUE = b'A',
        S = S_VALUE = b'S',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Letter>);

impl Solution {
    const WORD_LEN: usize = Letter::COUNT;
    const WORD_LEN_MINUS_ONE: i32 = Self::WORD_LEN as i32 - 1_i32;
    const X_SIDE_LEN: i32 = Self::WORD_LEN_MINUS_ONE;
    const X_SIDE_LEN_MINUS_ONE: i32 = Self::X_SIDE_LEN - 1_i32;

    fn iter_dirs() -> impl Iterator<Item = IVec2> {
        Direction::iter().flat_map(|dir| {
            [false, true]
                .into_iter()
                .map(move |is_diagonal| dir.vec() + (is_diagonal as i32 * dir.next().vec()))
        })
    }

    fn matches_word(&self, pos: IVec2, dir: IVec2) -> bool {
        self.0.contains(pos)
            && self.0.contains(pos + Self::WORD_LEN_MINUS_ONE * dir)
            && Letter::iter()
                .enumerate()
                .all(|(index, letter)| *self.0.get(pos + index as i32 * dir).unwrap() == letter)
    }

    fn iter_poses_and_word_dirs(&self) -> impl Iterator<Item = (IVec2, IVec2)> + '_ {
        (0_usize..self.0.cells().len()).flat_map(move |index| {
            Self::iter_dirs().map(move |dir| (self.0.pos_from_index(index), dir))
        })
    }

    fn count_word_apperances(&self) -> usize {
        self.iter_poses_and_word_dirs()
            .filter(|&(pos, dir)| self.matches_word(pos, dir))
            .count()
    }

    fn matches_x(&self, pos: IVec2, dir: Direction) -> bool {
        let dir_vec: IVec2 = dir.vec();
        let dir_next_vec: IVec2 = dir.next().vec();
        let diag_1_dir: IVec2 = dir_vec + dir_next_vec;

        self.0.contains(pos)
            && self
                .0
                .contains(pos + Self::X_SIDE_LEN_MINUS_ONE * diag_1_dir)
            && {
                let diag_1_pos: IVec2 = pos;
                let diag_2_dir: IVec2 = dir_vec - dir_next_vec;
                let diag_2_pos: IVec2 = pos + Self::X_SIDE_LEN_MINUS_ONE * dir_next_vec;

                Letter::iter()
                    .skip(1_usize)
                    .enumerate()
                    .all(|(index, letter)| {
                        *self.0.get(diag_1_pos + index as i32 * diag_1_dir).unwrap() == letter
                            && *self.0.get(diag_2_pos + index as i32 * diag_2_dir).unwrap()
                                == letter
                    })
            }
    }

    fn iter_poses_and_x_dirs(&self) -> impl Iterator<Item = (IVec2, Direction)> + '_ {
        (0_usize..self.0.cells().len()).flat_map(move |index| {
            Direction::iter().map(move |dir| (self.0.pos_from_index(index), dir))
        })
    }

    fn count_x_appearances(&self) -> usize {
        self.iter_poses_and_x_dirs()
            .filter(|&(pos, dir)| self.matches_x(pos, dir))
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// `Grid2D`, `Direction`, and `define_cell` came in handy.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_word_apperances());
    }

    /// Easily extendable.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_x_appearances());
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
        MMMSXXMASM\n\
        MSAMXMSMSA\n\
        AMXSXMAAMM\n\
        MSAMASMSMX\n\
        XMASAMXAMM\n\
        XXAMMXXAMA\n\
        SMSMSASXSS\n\
        SAXAMASAAA\n\
        MAMMMXMMMM\n\
        MXMXAXMASX\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Letter::*;

            vec![Solution(
                Grid2D::try_from_cells_and_dimensions(
                    vec![
                        M, M, M, S, X, X, M, A, S, M, M, S, A, M, X, M, S, M, S, A, A, M, X, S, X,
                        M, A, A, M, M, M, S, A, M, A, S, M, S, M, X, X, M, A, S, A, M, X, A, M, M,
                        X, X, A, M, M, X, X, A, M, A, S, M, S, M, S, A, S, X, S, S, S, A, X, A, M,
                        A, S, A, A, A, M, A, M, M, M, X, M, M, M, M, M, X, M, X, A, X, M, A, S, X,
                    ],
                    10_i32 * IVec2::ONE,
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
    fn test_count_word_apperances() {
        for (index, word_appearance_count) in [18_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).count_word_apperances(),
                word_appearance_count
            );
        }
    }

    #[test]
    fn test_count_x_appearances() {
        for (index, x_appearance_count) in [9_usize].into_iter().enumerate() {
            assert_eq!(solution(index).count_x_appearances(), x_appearance_count);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
