use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{map, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{delimited, preceded, separated_pair, terminated},
        Err, IResult,
    },
    std::{cmp::Ordering, collections::VecDeque, mem::transmute, str::FromStr},
};

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
enum Fold {
    X(i32),
    Y(i32),
}

impl Fold {
    // fn

    fn extreme(self, value: i32) -> IVec2 {
        match self {
            Self::X(x) => IVec2::new(x, value),
            Self::Y(y) => IVec2::new(value, y),
        }
    }

    fn min(self) -> IVec2 {
        self.extreme(i32::MIN)
    }

    fn max(self) -> IVec2 {
        self.extreme(i32::MAX)
    }

    fn compare(self) -> fn(&IVec2, &IVec2) -> Ordering {
        match self {
            Self::X(_) => |a, b| a.x.cmp(&b.x).then_with(|| a.y.cmp(&b.y)),
            Self::Y(_) => |a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)),
        }
    }

    fn fold(self, dot: IVec2) -> IVec2 {
        match self {
            Self::X(x) => IVec2::new(2_i32 * x - dot.x, dot.y),
            Self::Y(y) => IVec2::new(dot.x, 2_i32 * y - dot.y),
        }
    }
}

#[repr(u8)]
enum DotCell {
    Occupied = b'#',
    Vacant = b'.',
}

impl Default for DotCell {
    fn default() -> Self {
        Self::Vacant
    }
}

struct DotGrid {
    grid: Grid2D<DotCell>,

    #[cfg(test)]
    offset: IVec2,
}

impl DotGrid {
    fn string(&self) -> String {
        // SAFETY: `Grid2D<DotCell>` is a `Grid2D` of a `u8`-sized type, and `Grid2DString` is a
        // new-type of a `Grid2D<u8>`.
        unsafe { transmute::<&Grid2D<DotCell>, &Grid2DString>(&self.grid) }
            .try_as_string()
            .unwrap_or_default()
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    dots: Vec<IVec2>,
    folds: VecDeque<Fold>,
}

impl Solution {
    fn parse_i32<'i>(input: &'i str) -> IResult<&'i str, i32> {
        map_res(digit1, i32::from_str)(input)
    }

    fn parse_dot<'i>(input: &'i str) -> IResult<&'i str, IVec2> {
        terminated(
            map(
                separated_pair(Self::parse_i32, tag(","), Self::parse_i32),
                IVec2::from,
            ),
            opt(line_ending),
        )(input)
    }

    fn parse_fold<'i>(input: &'i str) -> IResult<&'i str, Fold> {
        delimited(
            tag("fold along "),
            alt((
                preceded(tag("x="), map(Self::parse_i32, Fold::X)),
                preceded(tag("y="), map(Self::parse_i32, Fold::Y)),
            )),
            opt(line_ending),
        )(input)
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(
                many0(Self::parse_dot),
                line_ending,
                map(many0(Self::parse_fold), VecDeque::from),
            ),
            Self::from,
        )(input)
    }

    fn run_folds(&self, num_folds: Option<usize>) -> Self {
        let num_folds: usize = num_folds.unwrap_or(usize::MAX).min(self.folds.len());

        let mut solution: Self = self.clone();

        let Solution { dots, folds } = &mut solution;

        for fold in folds.drain(..num_folds) {
            let compare: fn(&IVec2, &IVec2) -> Ordering = fold.compare();

            dots.sort_by(compare);

            let max: IVec2 = fold.max();
            let partition: usize = dots.partition_point(|dot| compare(dot, &max).is_le());

            for index in partition..dots.len() {
                let mut folded_dot: IVec2 = fold.fold(dots[index]);

                if dots[..partition]
                    .binary_search_by(|dot| compare(dot, &folded_dot))
                    .is_ok()
                {
                    folded_dot = IVec2::new(i32::MAX, i32::MAX);
                }

                dots[index] = folded_dot;
            }

            dots.sort_by(compare);

            let min: IVec2 = fold.min();

            dots.truncate(dots.partition_point(|dot| compare(dot, &min).is_lt()));
        }

        solution
    }

    fn dot_grid(&self) -> DotGrid {
        if self.dots.is_empty() {
            DotGrid {
                grid: Grid2D::empty(IVec2::ZERO),

                #[cfg(test)]
                offset: IVec2::ZERO,
            }
        } else {
            let (min, max): (IVec2, IVec2) = self.dots.iter().fold(
                (i32::MAX * IVec2::ONE, i32::MIN * IVec2::ONE),
                |(min, max), dot| (min.min(*dot), max.max(*dot)),
            );
            let dimensions: IVec2 = max - min + IVec2::ONE;

            let mut grid: Grid2D<DotCell> = Grid2D::default(dimensions);

            for dot in self.dots.iter() {
                *grid.get_mut(*dot - min).unwrap() = DotCell::Occupied;
            }

            DotGrid {
                grid,
                #[cfg(test)]
                offset: min,
            }
        }
    }

    fn count_dots_after_1_fold(&self) -> usize {
        self.run_folds(Some(1_usize)).dots.len()
    }

    fn code(&self) -> String {
        self.run_folds(None).dot_grid().string()
    }
}

impl From<(Vec<IVec2>, VecDeque<Fold>)> for Solution {
    fn from((dots, folds): (Vec<IVec2>, VecDeque<Fold>)) -> Self {
        Self { dots, folds }
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let solution_after_1_fold: Solution = self.run_folds(Some(1_usize));

            dbg!(solution_after_1_fold.dots.len());
            eprintln!("{}", solution_after_1_fold.dot_grid().string());
        } else {
            dbg!(self.count_dots_after_1_fold());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        eprintln!("{}", self.code());
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
    use {super::*, lazy_static::lazy_static};

    const SOLUTION_STR: &str = concat!(
        "6,10\n",
        "0,14\n",
        "9,10\n",
        "0,3\n",
        "10,4\n",
        "4,11\n",
        "6,0\n",
        "6,12\n",
        "4,1\n",
        "0,13\n",
        "10,12\n",
        "3,4\n",
        "3,0\n",
        "8,4\n",
        "1,10\n",
        "2,14\n",
        "8,10\n",
        "9,0\n",
        "\n",
        "fold along y=7\n",
        "fold along x=5\n",
    );

    lazy_static! {
        static ref SOLUTION_AFTER_0_FOLDS: Solution = solution_after_0_folds();
        static ref SOLUTION_AFTER_1_FOLD: Solution = solution_after_1_fold();
        static ref SOLUTION_AFTER_2_FOLDS: Solution = solution_after_2_folds();
    }

    macro_rules! solution {
        { [ $( ( $x:expr, $y:expr ), )* ], [ $( $fold:ident($i:expr), )* ], } => {
            Solution {
                dots: vec![ $( IVec2::new($x, $y), )* ],
                folds: vec![ $( Fold::$fold($i), )* ].into(),
            }
        };
    }

    fn solution_after_0_folds() -> Solution {
        solution! {
            [
                (6, 10),
                (0, 14),
                (9, 10),
                (0, 3),
                (10, 4),
                (4, 11),
                (6, 0),
                (6, 12),
                (4, 1),
                (0, 13),
                (10, 12),
                (3, 4),
                (3, 0),
                (8, 4),
                (1, 10),
                (2, 14),
                (8, 10),
                (9, 0),
            ],
            [ Y(7), X(5), ],
        }
    }

    fn solution_after_1_fold() -> Solution {
        solution! {
            [
                (0, 0),
                (2, 0),
                (3, 0),
                (6, 0),
                (9, 0),
                (0, 1),
                (4, 1),
                (6, 2),
                (10, 2),
                (0, 3),
                (4, 3),
                (1, 4),
                (3, 4),
                (6, 4),
                (8, 4),
                (9, 4),
                (10, 4),
            ],
            [ X(5), ],
        }
    }

    fn solution_after_2_folds() -> Solution {
        solution! {
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 4),
                (2, 0),
                (2, 4),
                (3, 0),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
                (4, 4),
            ],
            [],
        }
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(
            Solution::try_from(SOLUTION_STR),
            Ok(solution_after_0_folds())
        )
    }

    #[test]
    fn test_run_folds() {
        assert_eq!(
            SOLUTION_AFTER_0_FOLDS.run_folds(Some(1_usize)),
            *SOLUTION_AFTER_1_FOLD
        );
        assert_eq!(
            SOLUTION_AFTER_1_FOLD.run_folds(Some(1_usize)),
            *SOLUTION_AFTER_2_FOLDS
        );
        assert_eq!(
            SOLUTION_AFTER_0_FOLDS.run_folds(Some(2_usize)),
            *SOLUTION_AFTER_2_FOLDS
        );
        assert_eq!(
            SOLUTION_AFTER_0_FOLDS.run_folds(None),
            *SOLUTION_AFTER_2_FOLDS
        );
    }

    #[test]
    fn test_dot_grid() {
        let dot_grid_after_0_folds: DotGrid = SOLUTION_AFTER_0_FOLDS.dot_grid();

        assert_eq!(dot_grid_after_0_folds.offset, IVec2::ZERO);
        assert_eq!(
            dot_grid_after_0_folds.string(),
            concat!(
                "...#..#..#.\n",
                "....#......\n",
                "...........\n",
                "#..........\n",
                "...#....#.#\n",
                "...........\n",
                "...........\n",
                "...........\n",
                "...........\n",
                "...........\n",
                ".#....#.##.\n",
                "....#......\n",
                "......#...#\n",
                "#..........\n",
                "#.#........\n",
            )
        );

        let dot_grid_after_1_fold: DotGrid = SOLUTION_AFTER_1_FOLD.dot_grid();

        assert_eq!(dot_grid_after_1_fold.offset, IVec2::ZERO);
        assert_eq!(
            dot_grid_after_1_fold.string(),
            concat!(
                "#.##..#..#.\n",
                "#...#......\n",
                "......#...#\n",
                "#...#......\n",
                ".#.#..#.###\n",
            )
        );

        let dot_grid_after_2_folds: DotGrid = SOLUTION_AFTER_2_FOLDS.dot_grid();

        assert_eq!(dot_grid_after_2_folds.offset, IVec2::ZERO);
        assert_eq!(
            dot_grid_after_2_folds.string(),
            "#####\n#...#\n#...#\n#...#\n#####\n",
        );
    }

    #[test]
    fn test_count_dots_after_1_fold() {
        assert_eq!(SOLUTION_AFTER_0_FOLDS.count_dots_after_1_fold(), 17_usize);
    }
}
