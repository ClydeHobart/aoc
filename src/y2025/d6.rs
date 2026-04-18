use {
    crate::*,
    arrayvec::ArrayVec,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while},
        character::complete::line_ending,
        combinator::{cut, map, map_opt, opt, success},
        error::Error,
        multi::{many0_count, many_m_n},
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::{
        iter::{repeat, repeat_with},
        ops::{AddAssign, MulAssign},
    },
};

/* --- Day 6: Trash Compactor ---

After helping the Elves in the kitchen, you were taking a break and helping them re-enact a movie scene when you over-enthusiastically jumped into the garbage chute!

A brief fall later, you find yourself in a garbage smasher. Unfortunately, the door's been magnetically sealed.

As you try to find a way out, you are approached by a family of cephalopods! They're pretty sure they can get the door open, but it will take some time. While you wait, they're curious if you can help the youngest cephalopod with her math homework.

Cephalopod math doesn't look that different from normal math. The math worksheet (your puzzle input) consists of a list of problems; each problem has a group of numbers that need to be either added (+) or multiplied (*) together.

However, the problems are arranged a little strangely; they seem to be presented next to each other in a very long horizontal list. For example:

123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +

Each problem's numbers are arranged vertically; at the bottom of the problem is the symbol for the operation that needs to be performed. Problems are separated by a full column of only spaces. The left/right alignment of numbers within each problem can be ignored.

So, this worksheet contains four problems:

    123 * 45 * 6 = 33210
    328 + 64 + 98 = 490
    51 * 387 * 215 = 4243455
    64 + 23 + 314 = 401

To check their work, cephalopod students are given the grand total of adding together all of the answers to the individual problems. In this worksheet, the grand total is 33210 + 490 + 4243455 + 401 = 4277556.

Of course, the actual worksheet is much wider. You'll need to make sure to unroll it completely so that you can read the problems clearly.

Solve the problems on the math worksheet. What is the grand total found by adding together all of the answers to the individual problems?

--- Part Two ---

The big cephalopods come back to check on how things are going. When they see that your grand total doesn't match the one expected by the worksheet, they realize they forgot to explain how to read cephalopod math.

Cephalopod math is written right-to-left in columns. Each number is given in its own column, with the most significant digit at the top and the least significant digit at the bottom. (Problems are still separated with a column consisting only of spaces, and the symbol at the bottom of the problem is still the operator to use.)

Here's the example worksheet again:

123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +

Reading the problems right-to-left one column at a time, the problems are now quite different:

    The rightmost problem is 4 + 431 + 623 = 1058
    The second problem from the right is 175 * 581 * 32 = 3253600
    The third problem from the right is 8 + 248 + 369 = 625
    Finally, the leftmost problem is 356 * 24 * 1 = 8544

Now, the grand total is 1058 + 3253600 + 625 + 8544 = 3263827.

Solve the problems on the math worksheet again. What is the grand total found by adding together all of the answers to the individual problems? */

type Constant = u16;

const MAX_COLUMN_CONSTANTS: usize = digits(Constant::MAX as usize);
const DECIMAL_BASE: Constant = 10 as Constant;

type ColumnConstants = ArrayVec<Constant, MAX_COLUMN_CONSTANTS>;

const MAX_GRID_HEIGHT: usize = MAX_COLUMN_CONSTANTS + 1_usize;

type ColumnValue = u64;

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy)]
    #[cfg_attr(test, derive(Debug, PartialEq))]
    enum Op {
        Add = ADD = b'+',
        Mul = MUL = b'*',
    }
}

impl Op {
    fn get_initial_column_value(self) -> ColumnValue {
        match self {
            Self::Add => 0 as ColumnValue,
            Self::Mul => 1 as ColumnValue,
        }
    }

    fn get_column_value_accumulator(self) -> fn(&mut ColumnValue, ColumnValue) -> () {
        match self {
            Self::Add => ColumnValue::add_assign,
            Self::Mul => ColumnValue::mul_assign,
        }
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct ColumnFooter {
    op: Op,
    column_width: u8,
    is_left_aligned: Option<bool>,
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
enum Cell {
    Constant(Constant),
    ColumnFooter(ColumnFooter),
}

impl Cell {
    fn parse_space0<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        take_while(|c| c == ' ')(input)
    }

    fn parse_constant<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self::Constant)(input)
    }

    fn parse_space_delimited_constant<'i>(
        input: &'i str,
    ) -> IResult<&'i str, (&'i str, Self, &'i str)> {
        tuple((Self::parse_space0, Self::parse_constant, Self::parse_space0))(input)
    }

    fn parse_column_footer<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Op::parse, |op| {
            Self::ColumnFooter(ColumnFooter {
                op,
                column_width: 0_u8,
                is_left_aligned: None,
            })
        })(input)
    }

    fn parse_column_footer_and_space<'i>(input: &'i str) -> IResult<&'i str, (Self, &'i str)> {
        tuple((Self::parse_column_footer, Self::parse_space0))(input)
    }

    fn try_get_constant(self) -> Option<Constant> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }

    fn try_get_column_footer(self) -> Option<ColumnFooter> {
        match self {
            Self::ColumnFooter(column_footer) => Some(column_footer),
            _ => None,
        }
    }

    fn try_get_op(self) -> Option<Op> {
        self.try_get_column_footer()
            .map(|column_footer| column_footer.op)
    }

    fn try_get_column_footer_mut(&mut self) -> Option<&mut ColumnFooter> {
        match self {
            Self::ColumnFooter(column_footer) => Some(column_footer),
            _ => None,
        }
    }
}

impl Parse for Cell {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Self::parse_space_delimited_constant, |(_, constant, _)| {
                constant
            }),
            map(Self::parse_column_footer_and_space, |(column_footer, _)| {
                column_footer
            }),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Cell>);

impl Solution {
    fn get_column_footer_row_index_for_grid(grid: &Grid2D<Cell>) -> i32 {
        grid.max_dimensions().y
    }

    fn does_grid_have_footer_row(grid: &Grid2D<Cell>) -> bool {
        let op_row_index: i32 = Self::get_column_footer_row_index_for_grid(grid);

        op_row_index >= 1_i32
            && grid.iter_positions_and_cells().all(|(pos, cell)| {
                if pos.y != op_row_index {
                    cell.try_get_constant().is_some()
                } else {
                    cell.try_get_op().is_some()
                }
            })
    }

    fn iter_row_strs<'i>(
        input: &'i str,
        grid: &Grid2D<Cell>,
    ) -> impl Iterator<Item = &'i str> + 'i {
        input.lines().take(grid.height())
    }

    fn get_footer_row_str<'i>(input: &'i str, grid: &Grid2D<Cell>) -> &'i str {
        Self::iter_row_strs(input, grid)
            .skip(grid.height() - 1_usize)
            .next()
            .unwrap()
    }

    fn initialize_footer_row_column_widths(input: &str, grid: &mut Grid2D<Cell>) {
        let footer_row_str: &str = Self::get_footer_row_str(input, grid);
        let final_column_index: i32 = grid.width() as i32 - 1_i32;
        let op_row_index: i32 = Self::get_column_footer_row_index_for_grid(grid);
        let mut column_index: i32 = 0_i32;

        many0_count(map(Cell::parse_column_footer_and_space, |(_, space)| {
            grid.get_mut((column_index, op_row_index).into())
                .unwrap()
                .try_get_column_footer_mut()
                .unwrap()
                .column_width = space.len() as u8 + (column_index == final_column_index) as u8;
            column_index += 1_i32;
        }))(footer_row_str)
        .unwrap();
    }

    fn try_get_constant(cell_pos: IVec2, grid: &Grid2D<Cell>) -> Option<Constant> {
        grid.get(cell_pos)?.try_get_constant()
    }

    fn try_get_column_footer_mut(
        column_footer_pos: IVec2,
        grid: &mut Grid2D<Cell>,
    ) -> Option<&mut ColumnFooter> {
        grid.get_mut(column_footer_pos)?.try_get_column_footer_mut()
    }

    fn try_update_alignment(
        cell_pos: IVec2,
        column_footer_pos: IVec2,
        grid: &mut Grid2D<Cell>,
        input: &mut &str,
    ) -> Option<()> {
        let constant: Constant = Self::try_get_constant(cell_pos, grid).break_on_none()?;
        let column_footer: &mut ColumnFooter =
            Self::try_get_column_footer_mut(column_footer_pos, grid).break_on_none()?;
        let column_width: usize = column_footer.column_width as usize;
        let cell_str: &str = input.get(..column_width).break_on_none()?;

        *input = &input[column_width..];

        let (left_space, cell, right_space): (&str, Cell, &str) =
            Cell::parse_space_delimited_constant(cell_str)
                .ok()
                .break_on_none()?
                .1;

        (cell.try_get_constant()? == constant)
            .then_some(())
            .break_on_none()?;

        let could_be_left_aligned: bool = left_space.is_empty();
        let could_be_right_aligned: bool = right_space.is_empty();

        (could_be_left_aligned || could_be_right_aligned)
            .then_some(())
            .break_on_none()?;

        match column_footer.is_left_aligned {
            Some(is_left_aligned) => {
                (if is_left_aligned {
                    could_be_left_aligned
                } else {
                    could_be_right_aligned
                })
                .then_some(())
                .break_on_none()?;
            }
            None => {
                column_footer.is_left_aligned =
                    match (could_be_left_aligned, could_be_right_aligned) {
                        (true, true) => {
                            (cell_pos.y == column_footer_pos.y - 1_i32).then_some(false)
                        }
                        (true, false) => Some(true),
                        (false, true) => Some(false),
                        (false, false) => unreachable!(),
                    };
            }
        }

        Some(())
    }

    fn parse_alignment<'i, 'g>(
        grid: &'g mut Grid2D<Cell>,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, ()> + 'g {
        let dimensions: IVec2 = grid.dimensions();
        let column_footer_start: IVec2 = (0_i32, dimensions.y - 1_i32).into();
        let column_footer_end: IVec2 = (dimensions.x, column_footer_start.y).into();
        let mut iter = grid.iter_positions().take(grid.area() - grid.width()).zip(
            repeat_with(move || {
                CellIter2D::try_from(column_footer_start..column_footer_end).unwrap()
            })
            .flatten(),
        );

        move |input| {
            let (cell_pos, column_footer_pos): (IVec2, IVec2) =
                map_opt(success(()), |_| iter.next())(input)?.1;
            let mut input: &str = input;
            let updated_is_left_aligned: Option<()> =
                Self::try_update_alignment(cell_pos, column_footer_pos, grid, &mut input)
                    .break_on_none();

            cut(map_opt(success(()), move |_| updated_is_left_aligned))(input)
        }
    }

    fn try_initialize_footer_row_alignment(input: &str, grid: &mut Grid2D<Cell>) -> bool {
        let grid_width: usize = grid.width();
        let grid_height: usize = grid.height() - 1_usize;

        many_m_n(
            grid_height,
            grid_height,
            terminated(
                many_m_n(
                    grid_width,
                    grid_width,
                    terminated(Self::parse_alignment(grid), opt(tag(" "))),
                ),
                opt(line_ending),
            ),
        )(input)
        .is_ok()
    }

    fn try_initialize_footer_row(input: &str, grid: &mut Grid2D<Cell>) -> bool {
        grid.width() > 0_usize
            && grid.height() <= MAX_GRID_HEIGHT
            && Self::does_grid_have_footer_row(grid)
            && {
                Self::initialize_footer_row_column_widths(input, grid);

                Self::try_initialize_footer_row_alignment(input, grid)
            }
    }

    fn get_column_footer_row_index(&self) -> i32 {
        Self::get_column_footer_row_index_for_grid(&self.0)
    }

    fn get_constant(&self, pos: IVec2) -> Constant {
        self.0
            .get(pos)
            .copied()
            .unwrap()
            .try_get_constant()
            .unwrap()
    }

    fn get_column_footer(&self, column_index: i32) -> ColumnFooter {
        self.0
            .get((column_index, self.get_column_footer_row_index()).into())
            .unwrap()
            .try_get_column_footer()
            .unwrap()
    }

    fn get_op(&self, pos: IVec2) -> Op {
        self.0.get(pos).copied().unwrap().try_get_op().unwrap()
    }

    fn iter_op_poses(&self) -> impl Iterator<Item = IVec2> {
        CellIter2D::until_boundary(
            &self.0,
            (0_i32, self.get_column_footer_row_index()).into(),
            Direction::East,
        )
    }

    fn iter_ops(&self) -> impl Iterator<Item = (IVec2, Op)> + '_ {
        self.iter_op_poses().map(|pos| (pos, self.get_op(pos)))
    }

    fn iter_column_constant_poses(&self, column_index: i32) -> impl Iterator<Item = IVec2> {
        CellIter2D::try_from(
            (column_index, 0_i32).into()..(column_index, self.get_column_footer_row_index()).into(),
        )
        .unwrap()
    }

    fn iter_column_constants(&self, column_index: i32) -> impl Iterator<Item = Constant> + '_ {
        self.iter_column_constant_poses(column_index)
            .map(|pos| self.get_constant(pos))
    }

    fn compute_aligned_column_constants(&self, column_index: i32) -> ColumnConstants {
        let column_footer: ColumnFooter = self.get_column_footer(column_index);

        self.iter_column_constants(column_index)
            .map(|constant| {
                if !column_footer.is_left_aligned.unwrap() {
                    constant
                } else {
                    constant
                        * DECIMAL_BASE.pow(
                            column_footer.column_width as u32 - digits(constant as usize) as u32,
                        )
                }
            })
            .collect()
    }

    fn compute_cephalopod_column_constants(&self, column_index: i32) -> ColumnConstants {
        let column_footer: ColumnFooter = self.get_column_footer(column_index);
        let mut aligned_column_constants: ColumnConstants =
            self.compute_aligned_column_constants(column_index);

        repeat(())
            .take(column_footer.column_width as usize)
            .map(|_| {
                aligned_column_constants.iter_mut().fold(
                    0 as Constant,
                    |cephalopod_column_constant, aligned_column_constant| {
                        let next_digit: Constant = *aligned_column_constant % DECIMAL_BASE;

                        *aligned_column_constant /= DECIMAL_BASE;

                        cephalopod_column_constant * DECIMAL_BASE + next_digit
                    },
                )
            })
            .collect()
    }

    // fn iter_cephalopod_column_constants(
    //     &self,
    //     column_index: i32,
    // ) -> impl Iterator<Item = Constant> {
    //     let column_footer: ColumnFooter = self
    //         .0
    //         .get((column_index, self.get_column_footer_row_index()).into())
    //         .unwrap()
    //         .try_get_column_footer()
    //         .unwrap();
    // }

    fn compute_column_value(&self, column_index: i32, op: Op) -> ColumnValue {
        let column_value_accumulator = op.get_column_value_accumulator();

        self.iter_column_constants(column_index).fold(
            op.get_initial_column_value(),
            |mut column_value, constant| {
                column_value_accumulator(&mut column_value, constant as ColumnValue);

                column_value
            },
        )
    }

    fn iter_column_values(&self) -> impl Iterator<Item = ColumnValue> + '_ {
        self.iter_ops()
            .map(|(op_pos, op)| self.compute_column_value(op_pos.x, op))
    }

    fn compute_grand_total(&self) -> ColumnValue {
        self.iter_column_values().sum()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::<Cell>::parse, |mut grid| {
            Self::try_initialize_footer_row(input, &mut grid).then(|| Self(grid))
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_grand_total());
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        todo!();
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

    const SOLUTION_STRS: &'static [&'static str] = &[concat!(
        "123 328  51 64 \n",
        " 45 64  387 23 \n",
        "  6 98  215 314\n",
        "*   +   *   +  \n",
    )];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                Grid2D::try_from_cells_and_dimensions(
                    vec![
                        Cell::Constant(123),
                        Cell::Constant(328),
                        Cell::Constant(51),
                        Cell::Constant(64),
                        Cell::Constant(45),
                        Cell::Constant(64),
                        Cell::Constant(387),
                        Cell::Constant(23),
                        Cell::Constant(6),
                        Cell::Constant(98),
                        Cell::Constant(215),
                        Cell::Constant(314),
                        Cell::ColumnFooter(ColumnFooter {
                            op: Op::Mul,
                            column_width: 3_u8,
                            is_left_aligned: Some(false),
                        }),
                        Cell::ColumnFooter(ColumnFooter {
                            op: Op::Add,
                            column_width: 3_u8,
                            is_left_aligned: Some(true),
                        }),
                        Cell::ColumnFooter(ColumnFooter {
                            op: Op::Mul,
                            column_width: 3_u8,
                            is_left_aligned: Some(false),
                        }),
                        Cell::ColumnFooter(ColumnFooter {
                            op: Op::Add,
                            column_width: 3_u8,
                            is_left_aligned: Some(true),
                        }),
                    ],
                    (4_i32, 4_i32).into(),
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
    fn test_iter_column_values() {
        for (index, column_values) in [vec![33210 as ColumnValue, 490, 4243455, 401]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                solution(index)
                    .iter_column_values()
                    .collect::<Vec<ColumnValue>>(),
                column_values
            );
        }
    }

    #[test]
    fn test_compute_grand_total() {
        for (index, grand_total) in [4277556 as ColumnValue].into_iter().enumerate() {
            assert_eq!(solution(index).compute_grand_total(), grand_total);
        }
    }

    #[test]
    fn test_compute_aligned_column_constants() {
        for (index, aligned_column_constants_list) in [vec![
            vec![123, 045, 006],
            vec![328, 640, 980],
            vec![051, 387, 215],
            vec![640, 230, 314 as Constant],
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (column_index, aligned_column_constants) in
                aligned_column_constants_list.into_iter().enumerate()
            {
                let computed_aligned_column_constants: Vec<Constant> = solution
                    .compute_aligned_column_constants(column_index as i32)
                    .into_iter()
                    .collect();

                debug_assert_eq!(computed_aligned_column_constants, aligned_column_constants);
            }
        }
    }

    #[test]
    fn test_compute_cephalopod_column_constants() {
        for (index, cephalopod_column_constants_list) in [vec![
            vec![346, 24, 1],
            vec![8, 248, 369],
            vec![175, 581, 32],
            vec![4, 431, 623 as Constant],
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (column_index, cephalopod_column_constants) in
                cephalopod_column_constants_list.into_iter().enumerate()
            {
                let computed_cephalopod_column_constants: Vec<Constant> = solution
                    .compute_cephalopod_column_constants(column_index as i32)
                    .into_iter()
                    .collect();

                debug_assert_eq!(computed_cephalopod_column_constants, cephalopod_column_constants);
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
