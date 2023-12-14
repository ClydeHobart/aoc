use {
    crate::*,
    glam::IVec2,
    nom::{
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::terminated,
        Err, IResult,
    },
};

#[cfg_attr(test, derive(Debug, PartialEq))]
enum ReflectionLine {
    Horizontal(i32),
    Vertical(i32),
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Pattern(Grid2D<Pixel>);

impl Pattern {
    fn pixels_are_the_same(&self) -> impl Fn((IVec2, IVec2)) -> bool + '_ {
        |(pixel_a, pixel_b)| self.0.get(pixel_a) == self.0.get(pixel_b)
    }

    fn pixels_are_different(&self) -> impl Fn(&(IVec2, IVec2)) -> bool + '_ {
        |(pixel_a, pixel_b)| self.0.get(*pixel_a) != self.0.get(*pixel_b)
    }

    fn iter_row_pos_pairs(&self, y: i32) -> impl Iterator<Item = (IVec2, IVec2)> + '_ {
        CellIter2D::until_boundary(&self.0, IVec2::new(0_i32, y), Direction::South).zip(
            CellIter2D::until_boundary(&self.0, IVec2::new(0_i32, y - 1_i32), Direction::North),
        )
    }

    fn iter_pos_pairs_for_row_pos_pair(
        &self,
        bottom_row_pos: IVec2,
        top_row_pos: IVec2,
    ) -> impl Iterator<Item = (IVec2, IVec2)> + '_ {
        CellIter2D::until_boundary(&self.0, bottom_row_pos, Direction::East).zip(
            CellIter2D::until_boundary(&self.0, top_row_pos, Direction::East),
        )
    }

    fn iter_col_pos_pairs(&self, x: i32) -> impl Iterator<Item = (IVec2, IVec2)> + '_ {
        CellIter2D::until_boundary(&self.0, IVec2::new(x, 0_i32), Direction::East).zip(
            CellIter2D::until_boundary(&self.0, IVec2::new(x - 1_i32, 0_i32), Direction::West),
        )
    }

    fn iter_pos_pairs_for_col_pos_pair(
        &self,
        right_col_pos: IVec2,
        left_col_pos: IVec2,
    ) -> impl Iterator<Item = (IVec2, IVec2)> + '_ {
        CellIter2D::until_boundary(&self.0, right_col_pos, Direction::South).zip(
            CellIter2D::until_boundary(&self.0, left_col_pos, Direction::South),
        )
    }

    fn find_reflection_line(&self) -> Option<ReflectionLine> {
        (1_i32..self.0.dimensions().y)
            .into_iter()
            .find(|y| {
                self.iter_row_pos_pairs(*y)
                    .all(|(bottom_row_pos, top_row_pos)| {
                        self.iter_pos_pairs_for_row_pos_pair(bottom_row_pos, top_row_pos)
                            .all(self.pixels_are_the_same())
                    })
            })
            .map(ReflectionLine::Horizontal)
            .into_iter()
            .chain(
                (1_i32..self.0.dimensions().x)
                    .into_iter()
                    .find(|x| {
                        self.iter_col_pos_pairs(*x)
                            .all(|(right_col_pos, left_col_pos)| {
                                self.iter_pos_pairs_for_col_pos_pair(right_col_pos, left_col_pos)
                                    .all(self.pixels_are_the_same())
                            })
                    })
                    .map(ReflectionLine::Vertical),
            )
            .next()
    }

    fn find_smudged_reflection_line(&self) -> Option<ReflectionLine> {
        (1_i32..self.0.dimensions().y)
            .into_iter()
            .find(|y| {
                self.iter_row_pos_pairs(*y).try_fold(
                    false,
                    |found_smudge: bool, (bottom_row_pos, top_row_pos)| {
                        let different_pixel_count: usize = self
                            .iter_pos_pairs_for_row_pos_pair(bottom_row_pos, top_row_pos)
                            .filter(self.pixels_are_different())
                            .count();

                        match different_pixel_count {
                            0_usize => Ok(found_smudge),
                            1_usize => {
                                if found_smudge {
                                    Err(())
                                } else {
                                    Ok(true)
                                }
                            }
                            _ => Err(()),
                        }
                    },
                ) == Ok(true)
            })
            .map(ReflectionLine::Horizontal)
            .into_iter()
            .chain(
                (1_i32..self.0.dimensions().x)
                    .into_iter()
                    .find(|x| {
                        self.iter_col_pos_pairs(*x).try_fold(
                            false,
                            |found_smudge: bool, (right_col_pos, left_col_pos)| {
                                let different_pixel_count: usize = self
                                    .iter_pos_pairs_for_col_pos_pair(right_col_pos, left_col_pos)
                                    .filter(self.pixels_are_different())
                                    .count();

                                match different_pixel_count {
                                    0_usize => Ok(found_smudge),
                                    1_usize => {
                                        if found_smudge {
                                            Err(())
                                        } else {
                                            Ok(true)
                                        }
                                    }
                                    _ => Err(()),
                                }
                            },
                        ) == Ok(true)
                    })
                    .map(ReflectionLine::Vertical),
            )
            .next()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Pattern>);

impl Solution {
    fn iter_reflection_lines(&self) -> impl Iterator<Item = Option<ReflectionLine>> + '_ {
        self.0.iter().map(Pattern::find_reflection_line)
    }

    fn summarize_reflection_lines<I: Iterator<Item = Option<ReflectionLine>>>(iter: I) -> i32 {
        iter.map(|opt_reflection_line| {
            opt_reflection_line.map_or(0_i32, |reflection_line| match reflection_line {
                ReflectionLine::Horizontal(y) => 100_i32 * y,
                ReflectionLine::Vertical(x) => x,
            })
        })
        .sum()
    }

    fn summarize_notes(&self) -> i32 {
        Self::summarize_reflection_lines(self.iter_reflection_lines())
    }

    fn iter_smudged_reflection_lines(&self) -> impl Iterator<Item = Option<ReflectionLine>> + '_ {
        self.0.iter().map(Pattern::find_smudged_reflection_line)
    }

    fn summarize_smudged_notes(&self) -> i32 {
        Self::summarize_reflection_lines(self.iter_smudged_reflection_lines())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(
                map(Grid2D::<Pixel>::parse, Pattern),
                opt(line_ending),
            )),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.summarize_notes());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.summarize_smudged_notes());
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
        #.##..##.\n\
        ..#.##.#.\n\
        ##......#\n\
        ##......#\n\
        ..#.##.#.\n\
        ..##..##.\n\
        #.#.##.#.\n\
        \n\
        #...##..#\n\
        #....#..#\n\
        ..##..###\n\
        #####.##.\n\
        #####.##.\n\
        ..##..###\n\
        #....#..#\n";

    fn solution() -> &'static Solution {
        use Pixel::{Dark as D, Light as L};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(vec![
                Pattern(
                    Grid2D::try_from_cells_and_width(
                        vec![
                            L, D, L, L, D, D, L, L, D, D, D, L, D, L, L, D, L, D, L, L, D, D, D, D,
                            D, D, L, L, L, D, D, D, D, D, D, L, D, D, L, D, L, L, D, L, D, D, D, L,
                            L, D, D, L, L, D, L, D, L, D, L, L, D, L, D,
                        ],
                        9_usize,
                    )
                    .unwrap(),
                ),
                Pattern(
                    Grid2D::try_from_cells_and_width(
                        vec![
                            L, D, D, D, L, L, D, D, L, L, D, D, D, D, L, D, D, L, D, D, L, L, D, D,
                            L, L, L, L, L, L, L, L, D, L, L, D, L, L, L, L, L, D, L, L, D, D, D, L,
                            L, D, D, L, L, L, L, D, D, D, D, L, D, D, L,
                        ],
                        9_usize,
                    )
                    .unwrap(),
                ),
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_reflection_lines() {
        assert_eq!(
            solution()
                .iter_reflection_lines()
                .collect::<Vec<Option<ReflectionLine>>>(),
            vec![
                Some(ReflectionLine::Vertical(5_i32)),
                Some(ReflectionLine::Horizontal(4_i32))
            ]
        );
    }

    #[test]
    fn test_summarize_notes() {
        assert_eq!(solution().summarize_notes(), 405_i32);
    }

    #[test]
    fn test_iter_smudged_reflection_lines() {
        assert_eq!(
            solution()
                .iter_smudged_reflection_lines()
                .collect::<Vec<Option<ReflectionLine>>>(),
            vec![
                Some(ReflectionLine::Horizontal(3_i32)),
                Some(ReflectionLine::Horizontal(1_i32))
            ]
        );
    }

    #[test]
    fn test_summarize_smudged_notes() {
        assert_eq!(solution().summarize_smudged_notes(), 400_i32);
    }
}
