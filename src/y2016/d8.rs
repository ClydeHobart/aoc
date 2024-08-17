use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::mem::swap,
};

/* --- Day 8: Two-Factor Authentication ---

You come across a door implementing what you can only assume is an implementation of two-factor authentication after a long game of requirements telephone.

To get past the door, you first swipe a keycard (no problem; there was one on a nearby desk). Then, it displays a code on a little screen, and you type that code on a keypad. Then, presumably, the door unlocks.

Unfortunately, the screen has been smashed. After a few minutes, you've taken everything apart and figured out how it works. Now you just have to work out what the screen would have displayed.

The magnetic strip on the card you swiped encodes a series of instructions for the screen; these instructions are your puzzle input. The screen is 50 pixels wide and 6 pixels tall, all of which start off, and is capable of three somewhat peculiar operations:

    rect AxB turns on all of the pixels in a rectangle at the top-left of the screen which is A wide and B tall.
    rotate row y=A by B shifts all of the pixels in row A (0 is the top row) right by B pixels. Pixels that would fall off the right end appear at the left end of the row.
    rotate column x=A by B shifts all of the pixels in column A (0 is the left column) down by B pixels. Pixels that would fall off the bottom appear at the top of the column.

For example, here is a simple sequence on a smaller screen:

    rect 3x2 creates a small rectangle in the top-left corner:

    ###....
    ###....
    .......

    rotate column x=1 by 1 rotates the second column down by one pixel:

    #.#....
    ###....
    .#.....

    rotate row y=0 by 4 rotates the top row right by four pixels:

    ....#.#
    ###....
    .#.....

    rotate column x=1 by 1 again rotates the second column down by one pixel, causing the bottom pixel to wrap back to the top:

    .#..#.#
    #.#....
    .#.....

As you can see, this display technology is extremely powerful, and will soon dominate the tiny-code-displaying-screen market. That's what the advertisement on the back of the display tries to convince you, anyway.

There seems to be an intermediate check of the voltage used by the display: after you swipe your card, if the screen did work, how many pixels should be lit?

--- Part Two ---

You notice that the screen is only capable of displaying capital letters; in the font it uses, each letter is 5 pixels wide and 6 tall.

After you swipe your card, what code is the screen trying to display? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
enum Instruction {
    Rect(IVec2),
    RotateRow { y: i32, delta: i32 },
    RotateCol { x: i32, delta: i32 },
}

impl Instruction {
    fn process(self, grid: &mut Grid2D<Pixel>) {
        match self {
            Instruction::Rect(size) => {
                for pos in CellIter2D::try_from(IVec2::ZERO..(size.y * IVec2::Y))
                    .unwrap()
                    .flat_map(|row_pos| {
                        CellIter2D::try_from(row_pos..(row_pos + size.x * IVec2::X)).unwrap()
                    })
                {
                    *grid.get_mut(pos).unwrap() = Pixel::Light;
                }
            }
            Instruction::RotateRow { y, delta } => {
                let cols: usize = grid.dimensions().x as usize;
                let cells_start: usize = y as usize * cols;
                let cells_end: usize = cells_start + cols;

                grid.cells_mut()[cells_start..cells_end].rotate_right(delta as usize);
            }
            Instruction::RotateCol { x, delta } => {
                let rows: i32 = grid.dimensions().y;
                let mut temp_pixel: Pixel;
                let mut y_start: i32 = 0_i32;
                let mut rotated_cells: i32 = 0_i32;

                while rotated_cells < rows {
                    let mut y: i32 = y_start;

                    temp_pixel = *grid.get(IVec2::new(x, y)).unwrap();

                    loop {
                        y = (y + delta) % rows;
                        swap(grid.get_mut(IVec2::new(x, y)).unwrap(), &mut temp_pixel);
                        rotated_cells += 1_i32;

                        if y == y_start {
                            break;
                        }
                    }

                    y_start += 1_i32;
                }
            }
        }
    }
}

impl Parse for Instruction {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        terminated(
            alt((
                map(
                    tuple((tag("rect "), parse_integer, tag("x"), parse_integer)),
                    |(_, x, _, y)| Self::Rect(IVec2::new(x, y)),
                ),
                map(
                    tuple((
                        tag("rotate row y="),
                        parse_integer,
                        tag(" by "),
                        parse_integer,
                    )),
                    |(_, y, _, delta)| Self::RotateRow { y, delta },
                ),
                map(
                    tuple((
                        tag("rotate column x="),
                        parse_integer,
                        tag(" by "),
                        parse_integer,
                    )),
                    |(_, x, _, delta)| Self::RotateCol { x, delta },
                ),
            )),
            opt(line_ending),
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Instruction>);

impl Solution {
    const GRID_DIMENSIONS: IVec2 = IVec2::new(50_i32, 6_i32);

    fn new_grid() -> Grid2D<Pixel> {
        Grid2D::default(Self::GRID_DIMENSIONS)
    }

    fn process(&self, grid: &mut Grid2D<Pixel>) {
        for instruction in &self.0 {
            instruction.process(grid);
        }
    }

    fn lit_pixel_count_for_grid(grid: &Grid2D<Pixel>) -> usize {
        grid.cells().iter().filter(|pixel| pixel.is_light()).count()
    }

    fn lit_pixel_count(&self) -> usize {
        let mut grid: Grid2D<Pixel> = Self::new_grid();

        self.process(&mut grid);

        Self::lit_pixel_count_for_grid(&grid)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(Instruction::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.lit_pixel_count());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        let mut grid: Grid2D<Pixel> = Self::new_grid();

        self.process(&mut grid);

        println!("{}", String::from(grid));
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

    const SOLUTION_STR: &'static str = "\
        rect 3x2\n\
        rotate column x=1 by 1\n\
        rotate row y=0 by 4\n\
        rotate column x=1 by 1\n";
    const GRID_DIMENSIONS: IVec2 = IVec2::new(7_i32, 3_i32);

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Instruction::Rect(IVec2::new(3_i32, 2_i32)),
                Instruction::RotateCol {
                    x: 1_i32,
                    delta: 1_i32,
                },
                Instruction::RotateRow {
                    y: 0_i32,
                    delta: 4_i32,
                },
                Instruction::RotateCol {
                    x: 1_i32,
                    delta: 1_i32,
                },
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_instruction_process() {
        let solution: &Solution = solution();

        let mut grid: Grid2D<Pixel> = Grid2D::default(GRID_DIMENSIONS);

        solution.0[0_usize].process(&mut grid);

        assert_eq!(
            String::from(grid.clone()),
            "\
            ###....\n\
            ###....\n\
            .......\n"
        );

        solution.0[1_usize].process(&mut grid);

        assert_eq!(
            String::from(grid.clone()),
            "\
            #.#....\n\
            ###....\n\
            .#.....\n"
        );

        solution.0[2_usize].process(&mut grid);

        assert_eq!(
            String::from(grid.clone()),
            "\
            ....#.#\n\
            ###....\n\
            .#.....\n"
        );

        solution.0[3_usize].process(&mut grid);

        assert_eq!(
            String::from(grid),
            "\
            .#..#.#\n\
            #.#....\n\
            .#.....\n"
        );

        fn new_grid() -> Grid2D<Pixel> {
            use Pixel::{Dark as D, Light as L};

            Grid2D::try_from_cells_and_width(
                vec![
                    L, D, D, D, D, D, D, D, L, D, L, D, D, D, D, D, L, D, D, D, L, D, D, D, L, D,
                    D, D, D, D, L, D, L, D, D, D, D, D, D, D, L, D, D, D, D, D, D, D, L, D, L, D,
                    D, D, D, D, L, D, D, D, L, D, D, D, L, D, D, D, D, D, L, D, L, D, D, D, D, D,
                    D, D, L,
                ],
                9_usize,
            )
            .unwrap()
        }

        grid = new_grid();

        Instruction::Rect(3 * IVec2::ONE).process(&mut grid);

        assert_eq!(
            String::from(grid),
            "\
            ###.....#\n\
            ###....#.\n\
            ###...#..\n\
            ...#.#...\n\
            ....#....\n\
            ...#.#...\n\
            ..#...#..\n\
            .#.....#.\n\
            #.......#\n"
        );

        grid = new_grid();

        for y in 0_i32..grid.dimensions().y {
            Instruction::RotateRow { y, delta: 3_i32 }.process(&mut grid);
        }

        assert_eq!(
            String::from(grid),
            "\
            ..##.....\n\
            .#..#....\n\
            #....#...\n\
            ......#.#\n\
            .......#.\n\
            ......#.#\n\
            #....#...\n\
            .#..#....\n\
            ..##.....\n"
        );

        grid = new_grid();

        for x in 0_i32..grid.dimensions().x {
            Instruction::RotateCol { x, delta: 3_i32 }.process(&mut grid);
        }

        assert_eq!(
            String::from(grid),
            "\
            ..#...#..\n\
            .#.....#.\n\
            #.......#\n\
            #.......#\n\
            .#.....#.\n\
            ..#...#..\n\
            ...#.#...\n\
            ....#....\n\
            ...#.#...\n"
        );
    }

    #[test]
    fn test_lit_pixel_count_for_grid() {
        let mut grid: Grid2D<Pixel> = Grid2D::default(GRID_DIMENSIONS);

        solution().process(&mut grid);

        assert_eq!(Solution::lit_pixel_count_for_grid(&grid), 6_usize);
    }
}
