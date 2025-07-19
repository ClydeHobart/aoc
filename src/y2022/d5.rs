use {
    crate::*,
    std::{
        cmp::Ordering,
        iter::Peekable,
        num::ParseIntError,
        ops::Range,
        str::{from_utf8, FromStr, Split, Utf8Error},
        string::FromUtf8Error,
    },
};

/// A cell of the cargo bay drawing, as described by https://adventofcode.com/2022/day/5
#[derive(Clone, Copy, Debug, PartialEq)]
enum DrawingCell {
    /// An empty space in the drawing present above any columns that are not the tallest,
    /// represented by the regex pattern `   `
    Empty,

    /// A cargo crate marked with an uppercase capital letter, represented by the regex pattern
    /// `\[[A-Z]\]`
    Crate(u8),

    /// A column index at the foot of the drawing, represented by the regex pattern ` [1-9] `
    ///
    /// The index is expected to be a single digit representing the 1-based array index of the
    /// column. It is stored with the offset removed, such that `b'1'` is stored as `1_u8`
    ColumnIndex(u8),
}

/// An error encountered while parsing a `DrawingCell` from a byte slice
#[derive(Debug, PartialEq)]
pub enum DrawingCellParseError {
    /// The byte slice is not a valid UTF-8-encoded string slice
    NotUtf8(Utf8Error),

    /// The string slice is not valid ASCII
    NotAscii,

    /// The string slice does not have length 3
    ///
    /// The `usize` payload is the actual length of the string slice
    InvalidLength(usize),

    /// The string slice was not a valid representation of a `DrawingCell`
    ///
    /// See the comments on its variants for the accepted values. The payload is the 3 bytes of the
    /// string slice, which can be assumed to be valid ASCII text.
    InvalidPattern([u8; 3_usize]),
}

impl TryFrom<&[u8]> for DrawingCell {
    type Error = DrawingCellParseError;

    /// Tries to parse a byte slice into a `DrawingCell`
    ///
    /// # Arguments
    ///
    /// * `drawing_cell_bytes` - The byte slice to attempt to parse into a `DrawingCell`
    ///
    /// # Errors
    ///
    /// If an error is encountered, an `Err`-wrapped `DrawingCellParseError` is returned.
    fn try_from(drawing_cell_bytes: &[u8]) -> Result<Self, Self::Error> {
        use DrawingCellParseError as Error;

        let drawing_cell_str: &str = from_utf8(drawing_cell_bytes).map_err(Error::NotUtf8)?;

        if !drawing_cell_str.is_ascii() {
            Err(Error::NotAscii)
        } else if drawing_cell_str.len() != 3_usize {
            Err(Error::InvalidLength(drawing_cell_str.len()))
        } else {
            let mut drawing_cell_bytes_copy: [u8; 3_usize] = [0_u8; 3_usize];

            drawing_cell_bytes_copy.copy_from_slice(&drawing_cell_bytes);

            let middle_byte: u8 = drawing_cell_bytes_copy[1_usize];
            let middle_char: char = middle_byte as char;

            match drawing_cell_bytes_copy {
                [b' ', b' ', b' '] => Ok(DrawingCell::Empty),
                [b'[', _, b']'] if middle_char.is_ascii_uppercase() => {
                    Ok(DrawingCell::Crate(middle_byte))
                }
                [b' ', _, b' '] if middle_char.is_ascii_digit() => {
                    Ok(DrawingCell::ColumnIndex(middle_byte - ZERO_OFFSET))
                }
                _ => Err(Error::InvalidPattern(drawing_cell_bytes_copy)),
            }
        }
    }
}

#[derive(Debug, PartialEq)]
struct DrawingGrid {
    cells: Vec<DrawingCell>,

    /// The number of columns present in the grid
    columns: usize,
}

#[derive(Debug, PartialEq)]
pub enum DrawingGridParseError {
    NoInitialToken,
    InvalidLineLength(usize),
    InvalidGapByte(u8),
    InvalidCharBoundary([u8; 3_usize]),
    DrawingCellFailedToParse(DrawingCellParseError),
}

impl DrawingGrid {
    fn reserve_row(&mut self) {
        self.cells.reserve(self.columns);
    }

    fn parse_row(
        &mut self,
        drawing_row_str: &str,
        expected_len: usize,
    ) -> Result<(), DrawingGridParseError> {
        use DrawingGridParseError as Error;

        if drawing_row_str.len() != expected_len {
            return Err(Error::InvalidLineLength(drawing_row_str.len()));
        }

        self.reserve_row();

        let columns_minus_one: usize = self.columns - 1_usize;
        let drawing_row_bytes: &[u8] = drawing_row_str.as_bytes();

        for column_index in 0_usize..self.columns {
            let drawing_cell_start: usize = 4_usize * column_index;

            if column_index != columns_minus_one {
                let gap_byte: u8 = drawing_row_bytes[drawing_cell_start + 3_usize];

                if gap_byte != b' ' {
                    return Err(Error::InvalidGapByte(gap_byte));
                }
            }

            let drawing_cell_range: Range<usize> = drawing_cell_start..drawing_cell_start + 3_usize;

            if !drawing_row_str.is_char_boundary(drawing_cell_start) {
                let mut drawing_cell_bytes: [u8; 3_usize] = [0_u8; 3_usize];

                drawing_cell_bytes.copy_from_slice(&drawing_row_bytes[drawing_cell_range]);

                return Err(Error::InvalidCharBoundary(drawing_cell_bytes));
            }

            self.cells.push(
                drawing_row_bytes[drawing_cell_range]
                    .try_into()
                    .map_err(Error::DrawingCellFailedToParse)?,
            );
        }

        Ok(())
    }
}

impl TryFrom<&str> for DrawingGrid {
    type Error = DrawingGridParseError;

    fn try_from(drawing_grid_str: &str) -> Result<Self, Self::Error> {
        use DrawingGridParseError as Error;

        let mut drawing_row_str_iter: Peekable<Split<char>> =
            drawing_grid_str.split('\n').peekable();

        let expected_len: usize = drawing_row_str_iter
            .peek()
            .ok_or(Error::NoInitialToken)?
            .len();
        let columns: usize = {
            let len_plus_one: usize = expected_len + 1_usize;

            if len_plus_one % 4_usize != 0_usize {
                Err(Error::InvalidLineLength(expected_len))
            } else {
                Ok(len_plus_one / 4_usize)
            }
        }?;

        let mut drawing_grid: Self = Self {
            cells: Vec::new(),
            columns,
        };

        for drawing_row_str in drawing_row_str_iter {
            drawing_grid.parse_row(drawing_row_str, expected_len)?;
        }

        Ok(drawing_grid)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
struct ProcedureStep {
    count: usize,
    from: u8,
    to: u8,
}

impl ProcedureStep {
    fn from_index(&self) -> usize {
        self.from as usize - 1_usize
    }

    fn to_index(&self) -> usize {
        self.to as usize - 1_usize
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ProcedureStepParseError {
    NoMoveToken,
    InvalidMoveToken,
    NoCountToken,
    FailedToParseCount(ParseIntError),
    NoFromTextToken,
    InvalidFromTextToken,
    NoFromToken,
    FailedToParseFrom(ParseIntError),
    NoToTextToken,
    InvalidToTextToken,
    NoToToken,
    FailedToParseTo(ParseIntError),
    ExtraTokenFound,
}

impl TryFrom<&str> for ProcedureStep {
    type Error = ProcedureStepParseError;

    fn try_from(procedure_step_str: &str) -> Result<Self, Self::Error> {
        use ProcedureStepParseError as Error;

        let mut iter: Split<char> = procedure_step_str.split(' ');

        match iter.next() {
            None => Err(Error::NoMoveToken),
            Some("move") => Ok(()),
            _ => Err(Error::InvalidMoveToken),
        }?;

        let count: usize = usize::from_str(iter.next().ok_or(Error::NoCountToken)?)
            .map_err(Error::FailedToParseCount)?;

        match iter.next() {
            None => Err(Error::NoFromTextToken),
            Some("from") => Ok(()),
            _ => Err(Error::InvalidFromTextToken),
        }?;

        let from: u8 = u8::from_str(iter.next().ok_or(Error::NoFromToken)?)
            .map_err(Error::FailedToParseFrom)?;

        match iter.next() {
            None => Err(Error::NoToTextToken),
            Some("to") => Ok(()),
            _ => Err(Error::InvalidToTextToken),
        }?;

        let to: u8 =
            u8::from_str(iter.next().ok_or(Error::NoToToken)?).map_err(Error::FailedToParseTo)?;

        match iter.next() {
            Some(_) => Err(Error::ExtraTokenFound),
            None => Ok(Self { count, from, to }),
        }
    }
}

trait CrateMover {
    fn move_crates(count: usize, from: &mut Vec<u8>, to: &mut Vec<u8>);
}

/// Do not operate without a license
struct CrateMover9000;

impl CrateMover for CrateMover9000 {
    fn move_crates(count: usize, from: &mut Vec<u8>, to: &mut Vec<u8>) {
        for crate_byte in from.drain(from.len() - count..).rev() {
            to.push(crate_byte);
        }
    }
}

/// Do not operate without a license
struct CrateMover9001;

impl CrateMover for CrateMover9001 {
    fn move_crates(count: usize, from: &mut Vec<u8>, to: &mut Vec<u8>) {
        for crate_byte in from.drain(from.len() - count..) {
            to.push(crate_byte);
        }
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct CargoStacks(Vec<Vec<u8>>);

#[derive(Debug)]
enum ProcedureStepFailedReason {
    #[allow(dead_code)]
    FromIndexTooLarge(usize),
    #[allow(dead_code)]
    ToIndexTooLarge(usize),
    IdenticalIndices,
    #[allow(dead_code)]
    InadequateCount(usize),
}

#[allow(dead_code)]
#[derive(Debug)]
struct ProcedureStepFailed {
    procedure_step: ProcedureStep,
    reason: ProcedureStepFailedReason,
}

impl CargoStacks {
    fn columns(&self) -> usize {
        self.0.len()
    }

    fn try_procedure_step<CM: CrateMover>(
        &mut self,
        procedure_step: ProcedureStep,
    ) -> Result<(), ProcedureStepFailed> {
        use ProcedureStepFailedReason as Reason;

        let columns: usize = self.columns();
        let from_index: usize = procedure_step.from_index();
        let to_index: usize = procedure_step.to_index();

        macro_rules! validate_index {
            ($index:ident, $reason_variant:ident) => {
                if $index >= columns {
                    return Err(ProcedureStepFailed {
                        procedure_step,
                        reason: Reason::$reason_variant(columns),
                    });
                }
            };
        }

        validate_index!(from_index, FromIndexTooLarge);
        validate_index!(to_index, ToIndexTooLarge);

        let (from, to): (&mut Vec<u8>, &mut Vec<u8>) = match from_index.cmp(&to_index) {
            Ordering::Less => {
                let (min, max): (&mut [Vec<u8>], &mut [Vec<u8>]) = self.0.split_at_mut(to_index);

                (&mut min[from_index], &mut max[0_usize])
            }
            Ordering::Equal => {
                return Err(ProcedureStepFailed {
                    procedure_step,
                    reason: Reason::IdenticalIndices,
                });
            }
            Ordering::Greater => {
                let (min, max): (&mut [Vec<u8>], &mut [Vec<u8>]) = self.0.split_at_mut(from_index);

                (&mut max[0_usize], &mut min[to_index])
            }
        };

        let from_len: usize = from.len();
        let final_len: isize = from_len as isize - procedure_step.count as isize;

        if final_len < 0_isize {
            return Err(ProcedureStepFailed {
                procedure_step,
                reason: Reason::InadequateCount(from_len),
            });
        }

        to.reserve(procedure_step.count);
        CM::move_crates(procedure_step.count, from, to);

        Ok(())
    }

    fn stack_tops(&self) -> Result<String, FromUtf8Error> {
        String::from_utf8(
            self.0
                .iter()
                .map(|cargo_stack: &Vec<u8>| -> u8 { cargo_stack.last().copied().unwrap_or(b' ') })
                .collect(),
        )
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum CargoStacksParseError {
    FailedToParseDrawingGrid(DrawingGridParseError),
    InvalidColumnCount(usize),
    InvalidCellCount(usize),
    GridContainsNoLastRow,
    InvalidLastRow,
    InvalidCrateFound {
        row: usize,
        column: u8,
        crate_byte: u8,
    },
    InvalidColumnIndexFound {
        row: usize,
        column: u8,
        column_index_byte: u8,
    },
}

impl TryFrom<DrawingGrid> for CargoStacks {
    type Error = CargoStacksParseError;

    fn try_from(DrawingGrid { mut cells, columns }: DrawingGrid) -> Result<Self, Self::Error> {
        use CargoStacksParseError as Error;

        // The current understanding of the string format is that each column index is a single
        // digit, starting at 1. If that is not the case, code needs altering. We also rely on this
        // to an extent by using a `u32` to keep track of whether a column has received an empty
        // crate
        if columns > 9_usize {
            return Err(Error::InvalidColumnCount(columns));
        }

        let cells_len: usize = cells.len();

        // We're expecting `calls_len` to be a multiple of `columns`
        if cells_len % columns != 0_usize {
            return Err(Error::InvalidCellCount(cells_len));
        }

        let row_count: isize = (cells_len / columns) as isize - 1_isize;

        // There at least needs to be 1 row for the `ColumnIndex` variants
        if row_count < 0_isize {
            return Err(Error::GridContainsNoLastRow);
        }

        // Validate that the last row is properly indexed `ColumnIndex` instances
        if !cells.drain(cells_len - columns..).enumerate().all(
            |(column_index, drawing_cell): (usize, DrawingCell)| -> bool {
                matches!(
                    drawing_cell,
                    DrawingCell::ColumnIndex(column_index_byte)
                        if column_index as u8 + 1_u8 == column_index_byte
                )
            },
        ) {
            return Err(Error::InvalidLastRow);
        }

        let mut cargo_stacks: Self = Self(vec![Vec::with_capacity(row_count as usize); columns]);
        let mut column_has_received_empty_crate: u32 = 0_u32;

        while !cells.is_empty() {
            let mut column_bit: u32 = 1_u32;

            for ((column_index, drawing_cell), cargo_stack) in cells
                .drain(cells.len() - columns..)
                .enumerate()
                .zip(cargo_stacks.0.iter_mut())
            {
                let column: u8 = column_index as u8 + 1_u8;

                match drawing_cell {
                    DrawingCell::Empty => {
                        column_has_received_empty_crate |= column_bit;
                    }
                    DrawingCell::Crate(crate_byte) => {
                        if column_has_received_empty_crate & column_bit != 0_u32 {
                            return Err(Error::InvalidCrateFound {
                                row: cargo_stack.len(),
                                column,
                                crate_byte,
                            });
                        }

                        cargo_stack.push(crate_byte);
                    }
                    DrawingCell::ColumnIndex(column_index_byte) => {
                        return Err(Error::InvalidColumnIndexFound {
                            row: cargo_stack.len(),
                            column,
                            column_index_byte,
                        });
                    }
                }

                column_bit <<= 1_u32;
            }
        }

        Ok(cargo_stacks)
    }
}

impl TryFrom<&str> for CargoStacks {
    type Error = CargoStacksParseError;

    fn try_from(drawing_grid_str: &str) -> Result<Self, Self::Error> {
        DrawingGrid::try_from(drawing_grid_str)
            .map_err(CargoStacksParseError::FailedToParseDrawingGrid)?
            .try_into()
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct RearrangementProcedure(Vec<ProcedureStep>);

impl TryFrom<&str> for RearrangementProcedure {
    type Error = ProcedureStepParseError;

    fn try_from(rearrangement_procedure_str: &str) -> Result<Self, Self::Error> {
        let mut rearrangement_procedure: Self = Self(Vec::new());

        for procedure_step_str in rearrangement_procedure_str.split('\n') {
            rearrangement_procedure
                .0
                .push(procedure_step_str.try_into()?);
        }

        Ok(rearrangement_procedure)
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct CargoStacksSimulation {
    cargo_stacks: CargoStacks,
    rearrangement_procedure: RearrangementProcedure,
}

impl CargoStacksSimulation {
    fn run<CM: CrateMover>(self) -> Result<CargoStacks, ProcedureStepFailed> {
        let CargoStacksSimulation {
            mut cargo_stacks,
            rearrangement_procedure,
        } = self;

        for procedure_step in rearrangement_procedure.0 {
            cargo_stacks.try_procedure_step::<CM>(procedure_step)?;
        }

        Ok(cargo_stacks)
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum CargoStacksSimulationParseError {
    NoCargoStacksToken,
    FailedToParseCargoStacks(CargoStacksParseError),
    NoRearrangementProcedureToken,
    FailedToParseRearrangementProcedure(ProcedureStepParseError),
    ExtraTokenFound,
}

impl TryFrom<&str> for CargoStacksSimulation {
    type Error = CargoStacksSimulationParseError;

    fn try_from(cargo_stacks_simulation_str: &str) -> Result<Self, Self::Error> {
        use CargoStacksSimulationParseError as Error;

        let mut iter: Split<&str> = cargo_stacks_simulation_str.split("\n\n");

        let cargo_stacks: CargoStacks = match iter.next() {
            None => Err(Error::NoCargoStacksToken),
            Some(cargo_stacks_str) => cargo_stacks_str
                .try_into()
                .map_err(Error::FailedToParseCargoStacks),
        }?;

        let rearrangement_procedure: RearrangementProcedure = match iter.next() {
            None => Err(Error::NoRearrangementProcedureToken),
            Some(rearrangement_procedure_str) => rearrangement_procedure_str
                .try_into()
                .map_err(Error::FailedToParseRearrangementProcedure),
        }?;

        match iter.next() {
            Some(_) => Err(Error::ExtraTokenFound),
            None => Ok(Self {
                cargo_stacks,
                rearrangement_procedure,
            }),
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]

pub struct Solution(CargoStacksSimulation);

impl Solution {
    fn try_compute_stack_tops<CM: CrateMover>(&self) -> Option<String> {
        let cargo_stacks: CargoStacks = match self.0.clone().run::<CM>() {
            Ok(cargo_stacks) => cargo_stacks,
            Err(procedure_step_failed) => {
                eprintln!("Running simulation failed: {:#?}", procedure_step_failed);

                return None;
            }
        };

        let stack_tops: String = match cargo_stacks.stack_tops() {
            Ok(stack_tops) => stack_tops,
            Err(from_utf8_error) => {
                eprintln!(
                    "Failed to parse stack tops as a valid UTF-8 string: {:#?}",
                    from_utf8_error
                );

                return None;
            }
        };

        Some(stack_tops)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_stack_tops::<CrateMover9000>());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_stack_tops::<CrateMover9001>());
    }
}

impl TryFrom<&str> for Solution {
    type Error = CargoStacksSimulationParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const DRAWING_GRID_STR: &str = concat!(
        "    [D]    \n",
        "[N] [C]    \n",
        "[Z] [M] [P]\n",
        " 1   2   3 "
    );
    const REARRANGEMENT_PROCEDURE_STR: &str = concat!(
        "move 1 from 2 to 1\n",
        "move 3 from 1 to 3\n",
        "move 2 from 2 to 1\n",
        "move 1 from 1 to 2"
    );
    const CARGO_STACKS_SIMULATION_STR: &str = concat!(
        "    [D]    \n",
        "[N] [C]    \n",
        "[Z] [M] [P]\n",
        " 1   2   3 \n",
        "\n",
        "move 1 from 2 to 1\n",
        "move 3 from 1 to 3\n",
        "move 2 from 2 to 1\n",
        "move 1 from 1 to 2"
    );

    fn drawing_grid() -> &'static DrawingGrid {
        static ONCE_LOCK: OnceLock<DrawingGrid> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            use DrawingCell::*;

            let emty = || Empty;
            let c = |c: char| Crate(c as u8);
            let idx = |i: u8| ColumnIndex(i);

            DrawingGrid {
                cells: [
                    [emty(), c('D'), emty()],
                    [c('N'), c('C'), emty()],
                    [c('Z'), c('M'), c('P')],
                    [idx(1), idx(2), idx(3)],
                ]
                .iter()
                .map(|drawing_cells| drawing_cells.iter())
                .flatten()
                .copied()
                .collect(),
                columns: 3_usize,
            }
        })
    }

    fn cargo_stacks() -> &'static CargoStacks {
        static ONCE_LOCK: OnceLock<CargoStacks> = OnceLock::new();

        ONCE_LOCK
            .get_or_init(|| CargoStacks(vec![vec![b'Z', b'N'], vec![b'M', b'C', b'D'], vec![b'P']]))
    }

    fn rearrangement_procedure() -> &'static RearrangementProcedure {
        static ONCE_LOCK: OnceLock<RearrangementProcedure> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            RearrangementProcedure(vec![
                ProcedureStep {
                    count: 1,
                    from: 2,
                    to: 1,
                },
                ProcedureStep {
                    count: 3,
                    from: 1,
                    to: 3,
                },
                ProcedureStep {
                    count: 2,
                    from: 2,
                    to: 1,
                },
                ProcedureStep {
                    count: 1,
                    from: 1,
                    to: 2,
                },
            ])
        })
    }

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(CargoStacksSimulation {
                cargo_stacks: cargo_stacks().clone(),
                rearrangement_procedure: rearrangement_procedure().clone(),
            })
        })
    }

    #[test]
    fn test_drawing_grid_try_from_str() {
        assert_eq!(DRAWING_GRID_STR.try_into().as_ref(), Ok(drawing_grid()));
    }

    #[test]
    fn test_cargo_stacks_try_from_str() {
        assert_eq!(DRAWING_GRID_STR.try_into().as_ref(), Ok(cargo_stacks()));
    }

    #[test]
    fn test_rearrangement_procedure_try_from_str() {
        assert_eq!(
            REARRANGEMENT_PROCEDURE_STR.try_into().as_ref(),
            Ok(rearrangement_procedure())
        )
    }

    #[test]
    fn test_solution_try_from_str() {
        assert_eq!(
            CARGO_STACKS_SIMULATION_STR.try_into().as_ref(),
            Ok(solution())
        )
    }

    #[test]
    fn test_crate_mover_9000() {
        assert_eq!(
            solution().try_compute_stack_tops::<CrateMover9000>(),
            Some("CMZ".into())
        );
    }

    #[test]
    fn test_crate_mover_9001() {
        assert_eq!(
            solution().try_compute_stack_tops::<CrateMover9001>(),
            Some("MCD".into())
        );
    }
}
