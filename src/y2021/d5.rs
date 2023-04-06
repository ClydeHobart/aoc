use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{iterator, map, map_res, opt},
        error::Error,
        sequence::{separated_pair, terminated},
        Err, Parser,
    },
    std::{
        iter::{IntoIterator, Iterator},
        ops::Range,
        str::FromStr,
    },
};

struct LineIter {
    range: Range<IVec2>,
    delta: IVec2,
}

impl Iterator for LineIter {
    type Item = IVec2;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.start != self.range.end {
            let next: Option<IVec2> = Some(self.range.start);

            self.range.start += self.delta;

            next
        } else {
            None
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Line {
    a: IVec2,
    b: IVec2,
}

impl Line {
    fn parse_ivec2<'i>() -> impl Parser<&'i str, IVec2, Error<&'i str>> {
        map(
            separated_pair(
                map_res(digit1, i32::from_str),
                tag(","),
                map_res(digit1, i32::from_str),
            ),
            |(x, y)| IVec2 { x, y },
        )
    }

    fn parse<'i>() -> impl Parser<&'i str, Self, Error<&'i str>> {
        map(
            separated_pair(Self::parse_ivec2(), tag(" -> "), Self::parse_ivec2()),
            |(a, b)| Self { a, b },
        )
    }

    #[inline]
    fn is_horizontal(self) -> bool {
        self.a.y == self.b.y
    }

    #[inline]
    fn is_vertical(self) -> bool {
        self.a.x == self.b.x
    }

    #[inline]
    fn is_horizontal_or_vertical(self) -> bool {
        self.is_horizontal() || self.is_vertical()
    }
}

impl IntoIterator for Line {
    type Item = IVec2;
    type IntoIter = LineIter;

    fn into_iter(self) -> Self::IntoIter {
        let mut delta: IVec2 = self.b - self.a;

        delta /= delta.abs().max(IVec2::ONE);

        LineIter {
            range: self.a..self.b + delta,
            delta,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    lines: Vec<Line>,
    range: Range<IVec2>,
}

impl Solution {
    fn compute_overlaps_and_grid<P: FnMut(&Line) -> bool>(
        &self,
        predicate: P,
    ) -> (usize, Grid2D<u8>) {
        let dimensions: IVec2 = self.range.end - self.range.start;

        let mut grid: Grid2D<u8> = Grid2D::try_from_cell_and_dimensions(
            vec![0_u8; dimensions.x as usize * dimensions.y as usize],
            dimensions,
        )
        .unwrap();

        (
            self.lines
                .iter()
                .copied()
                .filter(predicate)
                .flat_map(Line::into_iter)
                .map(|pos| {
                    grid.get_mut(pos - self.range.start)
                        .map_or(0_usize, |cell| {
                            let cell_val: u8 = *cell;

                            *cell = cell_val.saturating_add(1_u8);

                            (cell_val == 1_u8) as usize
                        })
                })
                .sum::<usize>(),
            grid,
        )
    }

    fn filter_horizontal_and_vertical_overlaps(line: &Line) -> bool {
        line.is_horizontal_or_vertical()
    }

    fn filter_all(_: &Line) -> bool {
        true
    }

    fn try_grid_to_string(mut grid: Grid2D<u8>) -> Grid2DStringResult {
        for cell in grid.cells_mut() {
            let cell_val: u8 = *cell;

            const LETTER_OFFSET: u8 = b'A' - 0xA_u8;

            *cell = match cell_val {
                0_u8 => b'.',
                1_u8..=9_u8 => cell_val + b'0',
                0xA_u8..=0xE_u8 => cell_val + LETTER_OFFSET,
                _ => b'F',
            };
        }

        Grid2DString(grid).try_as_string()
    }

    fn compute_horizontal_and_vertical_overlaps(&self) -> usize {
        self.compute_overlaps_and_grid(Self::filter_horizontal_and_vertical_overlaps)
            .0
    }

    fn try_horizontal_and_vertical_grid(&self) -> Grid2DStringResult {
        Self::try_grid_to_string(
            self.compute_overlaps_and_grid(Self::filter_horizontal_and_vertical_overlaps)
                .1,
        )
    }

    fn compute_all_overlaps(&self) -> usize {
        self.compute_overlaps_and_grid(Self::filter_all).0
    }

    fn try_all_grid(&self) -> Grid2DStringResult {
        Self::try_grid_to_string(self.compute_overlaps_and_grid(Self::filter_all).1)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.compute_horizontal_and_vertical_overlaps());

        if args.verbose {
            match self.try_horizontal_and_vertical_grid() {
                Ok(string) => println!("{string}"),
                Err(err) => eprintln!(
                    "Failed to convert grid with horizontal and vertical lines.\nError:\n{err:#?}"
                ),
            }
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.compute_all_overlaps());

        if args.verbose {
            match self.try_all_grid() {
                Ok(string) => println!("{string}"),
                Err(err) => eprintln!("Failed to convert grid with all lines.\nError:\n{err:#?}"),
            }
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        let mut iter = iterator(input, terminated(Line::parse(), opt(line_ending)));

        let lines: Vec<Line> = iter.collect();

        iter.finish()?;

        let (min, max): (IVec2, IVec2) = lines.iter().fold(
            (IVec2::ONE * i32::MAX, IVec2::ONE * i32::MIN),
            |(min, max), line| (min.min(line.a.min(line.b)), max.max(line.a.max(line.b))),
        );

        Ok(Self {
            lines,
            range: min..max + IVec2::ONE,
        })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static};

    const LINES_STR: &str = concat!(
        "0,9 -> 5,9\n",
        "8,0 -> 0,8\n",
        "9,4 -> 3,4\n",
        "2,2 -> 2,1\n",
        "7,0 -> 7,4\n",
        "6,4 -> 2,0\n",
        "0,9 -> 2,9\n",
        "3,4 -> 1,4\n",
        "0,0 -> 8,8\n",
        "5,5 -> 8,2\n",
    );

    lazy_static! {
        static ref SOLUTION: Solution = new_solution();
    }

    fn new_solution() -> Solution {
        macro_rules! solution {
            [
                $( ($ax:expr, $ay:expr) -> ($bx:expr, $by:expr) ),* ;
                ( $min_x:expr, $min_y:expr )..( $max_x:expr, $max_y:expr )
            ] => { Solution {
                lines: vec![
                    $( Line { a: IVec2 { x: $ax, y: $ay }, b: IVec2 { x: $bx, y: $by } }, )*
                ],
                range: IVec2 { x: $min_x, y: $min_y }..IVec2 { x: $max_x, y: $max_y }
            } };
        }

        solution![
            (0, 9) -> (5, 9),
            (8, 0) -> (0, 8),
            (9, 4) -> (3, 4),
            (2, 2) -> (2, 1),
            (7, 0) -> (7, 4),
            (6, 4) -> (2, 0),
            (0, 9) -> (2, 9),
            (3, 4) -> (1, 4),
            (0, 0) -> (8, 8),
            (5, 5) -> (8, 2);
            (0, 0)..(10, 10)
        ]
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(LINES_STR), Ok(new_solution()));
    }

    #[test]
    fn test_compute_horizontal_and_vertical_overlaps() {
        assert_eq!(SOLUTION.compute_horizontal_and_vertical_overlaps(), 5_usize);
    }

    #[test]
    fn test_try_horizontal_and_vertical_grid() {
        const GRID: &str = concat!(
            ".......1..\n",
            "..1....1..\n",
            "..1....1..\n",
            ".......1..\n",
            ".112111211\n",
            "..........\n",
            "..........\n",
            "..........\n",
            "..........\n",
            "222111....\n",
        );

        assert_eq!(SOLUTION.try_horizontal_and_vertical_grid(), Ok(GRID.into()));
    }

    #[test]
    fn test_compute_all_overlaps() {
        assert_eq!(SOLUTION.compute_all_overlaps(), 12_usize);
    }

    #[test]
    fn test_try_all_grid() {
        const GRID: &str = concat!(
            "1.1....11.\n",
            ".111...2..\n",
            "..2.1.111.\n",
            "...1.2.2..\n",
            ".112313211\n",
            "...1.2....\n",
            "..1...1...\n",
            ".1.....1..\n",
            "1.......1.\n",
            "222111....\n",
        );

        assert_eq!(SOLUTION.try_all_grid(), Ok(GRID.into()));
    }
}
