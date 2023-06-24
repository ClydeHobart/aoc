use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{iterator, map_res, opt},
        error::Error,
        sequence::{preceded, terminated, tuple},
        Err,
    },
    std::{iter::once, ops::Range, str::FromStr},
};

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Cell {
    board: u8,
    pos: u8,
}

#[derive(Clone, Copy)]
struct CellWithValue {
    value: u8,
    cell: Cell,
}

impl Eq for CellWithValue {}

impl Ord for CellWithValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialEq for CellWithValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for CellWithValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
struct WinningBoard {
    board_state: u32,
    board: u8,
    value: u8,
}

#[allow(dead_code)]
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
struct ScoredWinningBoard {
    board_score: u32,
    final_score: u32,
    winning_board: WinningBoard,
}

trait SolutionConsts {
    const COLS_PER_BOARD: Self;
    const ROWS_PER_BOARD: Self;
    const VALUES_PER_BOARD: Self;
}

macro_rules! impl_solution_consts_for {
    ($type:ty) => {
        impl SolutionConsts for $type {
            const COLS_PER_BOARD: $type = 5;
            const ROWS_PER_BOARD: $type = 5;
            const VALUES_PER_BOARD: $type = <$type>::COLS_PER_BOARD * <$type>::ROWS_PER_BOARD;
        }
    };
}

impl_solution_consts_for!(u8);
impl_solution_consts_for!(usize);

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    drawn_nums: Vec<u8>,
    values: Vec<u8>,
    cells: Vec<Cell>,
    ranges: Vec<Range<u16>>,
}

impl Solution {
    const ROW_0_MASK: u32 = 0b11111_u32;
    const COL_0_MASK: u32 = 0b100001000010000100001_u32;

    fn get_cells(&self, value: u8) -> &[Cell] {
        let Range::<u16> { start, end } = self.ranges[value as usize];

        &self.cells[start as usize..end as usize]
    }

    fn score_board(&self, winning_board: WinningBoard) -> u32 {
        let start: usize = winning_board.board as usize * usize::VALUES_PER_BOARD;

        winning_board.board_state.view_bits::<Lsb0>()[..usize::VALUES_PER_BOARD]
            .iter()
            .zip(self.values[start..start + usize::VALUES_PER_BOARD].iter())
            .filter_map(|(bit, value)| if !*bit { Some(*value as u32) } else { None })
            .sum()
    }

    fn score_winning_board(&self, winning_board: WinningBoard) -> ScoredWinningBoard {
        let board_score: u32 = self.score_board(winning_board);
        let final_score: u32 = board_score * winning_board.value as u32;

        ScoredWinningBoard {
            board_score,
            final_score,
            winning_board,
        }
    }

    fn iter_winning_boards(&self) -> impl Iterator<Item = WinningBoard> + '_ {
        let mut board_states: Vec<u32> = vec![0_u32; self.values.len() / usize::VALUES_PER_BOARD];

        self.drawn_nums
            .iter()
            .copied()
            .flat_map(|value| {
                self.get_cells(value)
                    .iter()
                    .copied()
                    .zip(once(value).cycle())
            })
            .filter_map(move |(Cell { board, pos }, value)| {
                const BOARD_HAS_WON_MASK: u32 = 1_u32 << u8::VALUES_PER_BOARD;

                let board_state: u32 = board_states[board as usize] | (1_u32 << pos);

                if board_state & BOARD_HAS_WON_MASK != 0_u32 {
                    None
                } else {
                    board_states[board as usize] = board_state;

                    let (row, col): (u8, u8) = (pos / u8::COLS_PER_BOARD, pos % u8::COLS_PER_BOARD);
                    let row_mask: u32 = Self::ROW_0_MASK << (row * u8::COLS_PER_BOARD);
                    let col_mask: u32 = Self::COL_0_MASK << col;

                    if board_state & row_mask != row_mask && board_state & col_mask != col_mask {
                        None
                    } else {
                        board_states[board as usize] |= BOARD_HAS_WON_MASK;

                        Some(WinningBoard {
                            board_state,
                            board,
                            value,
                        })
                    }
                }
            })
    }

    fn compute_first_scored_winning_board(&self) -> Option<ScoredWinningBoard> {
        self.iter_winning_boards()
            .next()
            .map(|winning_board| self.score_winning_board(winning_board))
    }

    fn compute_last_scored_winning_board(&self) -> Option<ScoredWinningBoard> {
        self.iter_winning_boards()
            .last()
            .map(|winning_board| self.score_winning_board(winning_board))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_first_scored_winning_board());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_last_scored_winning_board());
    }
}

impl<'a> TryFrom<&'a str> for Solution {
    type Error = Err<Error<&'a str>>;

    fn try_from(input: &'a str) -> Result<Self, Self::Error> {
        let mut iter = iterator(
            input,
            terminated(map_res(digit1, u8::from_str), opt(tag(","))),
        );

        let drawn_nums: Vec<u8> = iter.collect();
        let (input, _): (&str, _) = iter.finish()?;
        let (input, _): (&str, _) = tuple((line_ending, line_ending))(input)?;
        let num_boards: usize = input.lines().count() / (usize::ROWS_PER_BOARD + 1_usize);
        let num_cells: usize = num_boards * usize::VALUES_PER_BOARD;

        let mut board: u8 = 0_u8;
        let mut pos: u8 = 0_u8;
        let mut iter = iterator(input, |input: &'a str| {
            let (new_input, value): (&str, u8) =
                preceded(opt(tag(" ")), map_res(digit1, u8::from_str))(input)?;

            let cell_with_value: CellWithValue = CellWithValue {
                value,
                cell: Cell { board, pos },
            };

            pos += 1_u8;

            Ok((
                if pos % u8::COLS_PER_BOARD != 0_u8 {
                    tag(" ")(new_input)?.0
                } else if pos % u8::VALUES_PER_BOARD != 0_u8 {
                    line_ending(new_input)?.0
                } else {
                    pos = 0_u8;
                    board += 1_u8;

                    opt(tuple((line_ending, line_ending)))(new_input)?.0
                },
                cell_with_value,
            ))
        });
        let mut values: Vec<u8> = Vec::with_capacity(num_cells);
        let mut cells_with_values: Vec<CellWithValue> = Vec::with_capacity(num_cells);

        for cell_with_value in &mut iter {
            values.push(cell_with_value.value);
            cells_with_values.push(cell_with_value);
        }

        iter.finish()?;
        cells_with_values.sort();

        let mut cells: Vec<Cell> = Vec::with_capacity(cells_with_values.len());

        for cell_with_value in cells_with_values.iter() {
            cells.push(cell_with_value.cell);
        }

        let ranges_len: usize = cells_with_values.last().map_or(0_usize, |cell_with_value| {
            (cell_with_value.value + 1_u8) as usize
        });

        let mut start: usize = 0_usize;
        let mut end: usize = 1_usize;
        let mut ranges: Vec<Range<u16>> = Vec::with_capacity(ranges_len);

        ranges.resize(ranges_len, 0_u16..0_u16);

        while start < cells_with_values.len() {
            while end < cells_with_values.len()
                && cells_with_values[start].value == cells_with_values[end].value
            {
                end += 1_usize;
            }

            ranges[cells_with_values[start].value as usize] = start as u16..end as u16;
            start = end;
            end = start + 1_usize;
        }

        Ok(Self {
            drawn_nums,
            values,
            cells,
            ranges,
        })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const BINGO_STR: &str = concat!(
        "7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1\n",
        "\n",
        "22 13 17 11  0\n",
        " 8  2 23  4 24\n",
        "21  9 14 16  7\n",
        " 6 10  3 18  5\n",
        " 1 12 20 15 19\n",
        "\n",
        " 3 15  0  2 22\n",
        " 9 18 13 17  5\n",
        "19  8  7 25 23\n",
        "20 11 10 24  4\n",
        "14 21 16 12  6\n",
        "\n",
        "14 21 17 24  4\n",
        "10 16 15  9 19\n",
        "18  8 23 26 20\n",
        "22 11 13  6  5\n",
        " 2  0 12  3  7\n",
        "\n"
    );

    fn solution() -> &'static Solution {
        macro_rules! cells {
            [$( ($board:expr, $pos:expr), )*] => {
                vec![ $( Cell { board: $board, pos: $pos }, )* ]
            };
        }

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            drawn_nums: vec![
                7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21, 24, 10, 16, 13, 6, 15, 25, 12, 22, 18, 20, 8,
                19, 3, 26, 1,
            ],
            values: vec![
                22, 13, 17, 11, 0, 8, 2, 23, 4, 24, 21, 9, 14, 16, 7, 6, 10, 3, 18, 5, 1, 12, 20,
                15, 19, 3, 15, 0, 2, 22, 9, 18, 13, 17, 5, 19, 8, 7, 25, 23, 20, 11, 10, 24, 4, 14,
                21, 16, 12, 6, 14, 21, 17, 24, 4, 10, 16, 15, 9, 19, 18, 8, 23, 26, 20, 22, 11, 13,
                6, 5, 2, 0, 12, 3, 7,
            ],
            cells: cells![
                // 0
                (0, 4),
                (1, 2),
                (2, 21),
                // 1
                (0, 20),
                // 2
                (0, 6),
                (1, 3),
                (2, 20),
                // 3
                (0, 17),
                (1, 0),
                (2, 23),
                // 4
                (0, 8),
                (1, 19),
                (2, 4),
                // 5
                (0, 19),
                (1, 9),
                (2, 19),
                // 6
                (0, 15),
                (1, 24),
                (2, 18),
                // 7
                (0, 14),
                (1, 12),
                (2, 24),
                // 8
                (0, 5),
                (1, 11),
                (2, 11),
                // 9
                (0, 11),
                (1, 5),
                (2, 8),
                // 10
                (0, 16),
                (1, 17),
                (2, 5),
                // 11
                (0, 3),
                (1, 16),
                (2, 16),
                // 12
                (0, 21),
                (1, 23),
                (2, 22),
                // 13
                (0, 1),
                (1, 7),
                (2, 17),
                // 14
                (0, 12),
                (1, 20),
                (2, 0),
                // 15
                (0, 23),
                (1, 1),
                (2, 7),
                // 16
                (0, 13),
                (1, 22),
                (2, 6),
                // 17
                (0, 2),
                (1, 8),
                (2, 2),
                // 18
                (0, 18),
                (1, 6),
                (2, 10),
                // 19
                (0, 24),
                (1, 10),
                (2, 9),
                // 20
                (0, 22),
                (1, 15),
                (2, 14),
                // 21
                (0, 10),
                (1, 21),
                (2, 1),
                // 22
                (0, 0),
                (1, 4),
                (2, 15),
                // 23
                (0, 7),
                (1, 14),
                (2, 12),
                // 24
                (0, 9),
                (1, 18),
                (2, 3),
                // 25
                (1, 13),
                // 26
                (2, 13),
            ],
            ranges: vec![
                0..3,   // 0
                3..4,   // 1
                4..7,   // 2
                7..10,  // 3
                10..13, // 4
                13..16, // 5
                16..19, // 6
                19..22, // 7
                22..25, // 8
                25..28, // 9
                28..31, // 10
                31..34, // 11
                34..37, // 12
                37..40, // 13
                40..43, // 14
                43..46, // 15
                46..49, // 16
                49..52, // 17
                52..55, // 18
                55..58, // 19
                58..61, // 20
                61..64, // 21
                64..67, // 22
                67..70, // 23
                70..73, // 24
                73..74, // 25
                74..75, // 26
            ],
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(BINGO_STR).as_ref(), Ok(solution()))
    }

    #[test]
    fn test_compute_first_scored_winning_board() {
        assert_eq!(
            solution().compute_first_scored_winning_board(),
            Some(ScoredWinningBoard {
                board_score: 188_u32,
                final_score: 4512_u32,
                winning_board: WinningBoard {
                    board_state: 0b10011_10010_00100_01000_11111_u32,
                    board: 2_u8,
                    value: 24_u8
                }
            })
        )
    }

    #[test]
    fn test_compute_last_scored_winning_board() {
        const MIDDLE_COLUMN: u32 = Solution::COL_0_MASK << 2_u32;

        assert_eq!(
            solution()
                .compute_last_scored_winning_board()
                .map(|mut scored_winning_board| {
                    scored_winning_board.winning_board.board_state &= MIDDLE_COLUMN;

                    scored_winning_board
                }),
            Some(ScoredWinningBoard {
                board_score: 148_u32,
                final_score: 1924_u32,
                winning_board: WinningBoard {
                    board_state: MIDDLE_COLUMN,
                    board: 1_u8,
                    value: 13_u8
                }
            })
        )
    }
}
