use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, tuple},
        AsChar, Err, IResult,
    },
    strum::IntoEnumIterator,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct DigDirection {
    dir: Direction,
    dist: i32,
}

impl DigDirection {
    fn parse_direction_branch<'i>(
        tag_str: &'static str,
        dir: Direction,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Direction> {
        map(tag(tag_str), move |_| dir)
    }

    fn parse_direction<'i>(input: &'i str) -> IResult<&'i str, Direction> {
        alt((
            Self::parse_direction_branch("U", Direction::North),
            Self::parse_direction_branch("R", Direction::East),
            Self::parse_direction_branch("D", Direction::South),
            Self::parse_direction_branch("L", Direction::West),
        ))(input)
    }

    fn vec(&self) -> IVec2 {
        self.dist * self.dir.vec()
    }
}

impl Parse for DigDirection {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((Self::parse_direction, tag(" "), parse_integer::<i32>)),
            |(dir, _, dist)| Self { dir, dist },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct DigDirectionWithColor {
    dig_dir: DigDirection,
    color: u32,
}

impl DigDirectionWithColor {
    fn parse_color(input: &str) -> IResult<&str, u32> {
        // Modified from the homepage for the `nom` crate: https://docs.rs/nom/latest/nom/
        preceded(
            tag("#"),
            map_res(
                take_while_m_n(6_usize, 6_usize, char::is_hex_digit),
                |input| u32::from_str_radix(input, 16_u32),
            ),
        )(input)
    }

    fn dig_dir_from_color(&self) -> DigDirection {
        let dist: i32 = (self.color >> 4_u32) as i32;
        let dir: Direction = match self.color & 0x3_u32 {
            0_u32 => Direction::East,
            1_u32 => Direction::South,
            2_u32 => Direction::West,
            3_u32 => Direction::North,
            _ => unreachable!(),
        };

        DigDirection { dir, dist }
    }
}

impl Parse for DigDirectionWithColor {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                DigDirection::parse,
                tag(" ("),
                Self::parse_color,
                tag(")"),
                opt(line_ending),
            )),
            |(dig_dir, _, color, _, _)| Self { dig_dir, color },
        )(input)
    }
}

struct InteriorDigger<'g> {
    grid: &'g mut Grid2D<Pixel>,
    start: IVec2,
}

impl<'g> BreadthFirstSearch for InteriorDigger<'g> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(
            Direction::iter()
                .map(|dir| *vertex + dir.vec())
                .filter(|pos| self.grid.get(*pos).copied() == Some(Pixel::Dark)),
        );
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        *self.grid.get_mut(*to).unwrap() = Pixel::Light;
    }

    fn reset(&mut self) {
        self.update_parent(&IVec2::ZERO, &self.start.clone());
    }
}

struct Digger<'s> {
    solution: &'s Solution,
    grid: Grid2D<Pixel>,
    start: IVec2,
}

impl<'s> Digger<'s> {
    fn dig_trenches(&mut self) {
        for (curr_pos, dig_direction) in self.solution.iter_pos_and_dig_dir(self.start) {
            let next_pos: IVec2 = curr_pos + dig_direction.vec();

            for pos in CellIter2D::try_from(curr_pos..=next_pos).unwrap() {
                *self.grid.get_mut(pos).unwrap() = Pixel::Light;
            }
        }
    }

    fn dig_interior(&mut self) {
        if let Some((curr_pos, dig_direction)) = self
            .solution
            .iter_pos_and_dig_dir(self.start)
            .find(|(_, dig_direction)| dig_direction.dist > 1_i32)
        {
            let interior_dir: Direction = if Solution::is_loop_clockwise(Solution::signed_area(
                self.solution.iter_pos_and_dig_dir(IVec2::ZERO),
            )) {
                dig_direction.dir.next()
            } else {
                dig_direction.dir.prev()
            };

            let mut interior_digger: InteriorDigger = InteriorDigger {
                grid: &mut self.grid,
                start: curr_pos + dig_direction.dir.vec() + interior_dir.vec(),
            };

            interior_digger.run();
        }
    }

    fn string(&self) -> String {
        String::from(self.grid.clone())
    }

    fn cubic_meters_of_lava(&self) -> usize {
        self.grid
            .cells()
            .iter()
            .filter(|pixel| pixel.is_light())
            .count()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<DigDirectionWithColor>);

impl Solution {
    fn iter_pos_and_dig_dir(
        &self,
        offset: IVec2,
    ) -> impl Iterator<Item = (IVec2, DigDirection)> + '_ {
        let mut curr_pos: IVec2 = offset;

        self.0.iter().map(move |dig_dir_w_col| {
            let dig_dir: DigDirection = dig_dir_w_col.dig_dir.clone();
            let delta: IVec2 = dig_dir.vec();
            let pos: IVec2 = curr_pos;

            curr_pos += delta;

            (pos, dig_dir)
        })
    }

    fn iter_pos_and_dig_dir_from_colors(&self) -> impl Iterator<Item = (IVec2, DigDirection)> + '_ {
        let mut curr_pos: IVec2 = IVec2::ZERO;

        self.0.iter().map(move |dig_dir_w_col| {
            let dig_dir: DigDirection = dig_dir_w_col.dig_dir_from_color();
            let delta: IVec2 = dig_dir.vec();
            let pos: IVec2 = curr_pos;

            curr_pos += delta;

            (pos, dig_dir)
        })
    }

    fn digger(&self) -> Digger {
        let mut min: IVec2 = IVec2::ZERO;
        let mut max: IVec2 = IVec2::ZERO;

        for (pos, _) in self.iter_pos_and_dig_dir(IVec2::ZERO) {
            min = min.min(pos);
            max = max.max(pos);
        }

        let start: IVec2 = -min;
        let dimensions: IVec2 = max - min + IVec2::ONE;
        let grid: Grid2D<Pixel> = Grid2D::default(dimensions);
        let solution: &Self = self;

        Digger {
            solution,
            grid,
            start,
        }
    }

    fn perimeter<I: Iterator<Item = (IVec2, DigDirection)>>(iter: I) -> usize {
        iter.map(|(_, dig_direction)| dig_direction.dist as usize)
            .sum()
    }

    fn signed_area<I: Iterator<Item = (IVec2, DigDirection)>>(iter: I) -> isize {
        // An implementation of the shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
        iter.map(|(curr_pos, dig_direction)| {
            let next_pos: IVec2 = curr_pos + dig_direction.vec();

            (next_pos.x - curr_pos.x) as isize * (next_pos.y + curr_pos.y) as isize
        })
        .sum::<isize>()
            / 2_isize
    }

    fn is_loop_clockwise(signed_area: isize) -> bool {
        signed_area < 0_isize
    }

    fn cubic_meters_of_lava_from_area_and_perimeter(area: usize, perimeter: usize) -> usize {
        area + (perimeter / 2_usize) + 1_usize
    }

    fn cubic_meters_of_lava(&self) -> usize {
        Self::cubic_meters_of_lava_from_area_and_perimeter(
            Self::signed_area(self.iter_pos_and_dig_dir(IVec2::ZERO)).abs() as usize,
            Self::perimeter(self.iter_pos_and_dig_dir(IVec2::ZERO)),
        )
    }

    fn cubic_meters_of_lava_from_colors(&self) -> usize {
        Self::cubic_meters_of_lava_from_area_and_perimeter(
            Self::signed_area(self.iter_pos_and_dig_dir_from_colors()).abs() as usize,
            Self::perimeter(self.iter_pos_and_dig_dir_from_colors()),
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(DigDirectionWithColor::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let mut digger: Digger = self.digger();

            digger.dig_trenches();
            digger.dig_interior();

            dbg!(digger.cubic_meters_of_lava());

            println!("\n\n{}\n\n", digger.string());
        } else {
            dbg!(self.cubic_meters_of_lava());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.cubic_meters_of_lava_from_colors());
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
        R 6 (#70c710)\n\
        D 5 (#0dc571)\n\
        L 2 (#5713f0)\n\
        D 2 (#d2c081)\n\
        R 2 (#59c680)\n\
        D 2 (#411b91)\n\
        L 5 (#8ceee2)\n\
        U 2 (#caa173)\n\
        L 1 (#1b58a2)\n\
        U 2 (#caa171)\n\
        R 2 (#7807d2)\n\
        U 3 (#a77fa3)\n\
        L 2 (#015232)\n\
        U 2 (#7a21e3)\n";
    const TRENCHES_STR: &'static str = "\
        #######\n\
        #.....#\n\
        ###...#\n\
        ..#...#\n\
        ..#...#\n\
        ###.###\n\
        #...#..\n\
        ##..###\n\
        .#....#\n\
        .######\n";
    const INTERIOR_STR: &'static str = "\
        #######\n\
        #######\n\
        #######\n\
        ..#####\n\
        ..#####\n\
        #######\n\
        #####..\n\
        #######\n\
        .######\n\
        .######\n";

    fn solution() -> &'static Solution {
        use Direction::{East as R, North as U, South as D, West as L};

        macro_rules! dig_directions {
            [ $( $dir:ident, $dist:expr, $color:expr; )*] => {
                vec![ $(
                    DigDirectionWithColor {
                        dig_dir: DigDirection {
                            dir: $dir,
                            dist: $dist,
                        },
                        color: $color,
                    },
                )*]
            };
        }

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(dig_directions![
                R, 6, 0x70c710;
                D, 5, 0x0dc571;
                L, 2, 0x5713f0;
                D, 2, 0xd2c081;
                R, 2, 0x59c680;
                D, 2, 0x411b91;
                L, 5, 0x8ceee2;
                U, 2, 0xcaa173;
                L, 1, 0x1b58a2;
                U, 2, 0xcaa171;
                R, 2, 0x7807d2;
                U, 3, 0xa77fa3;
                L, 2, 0x015232;
                U, 2, 0x7a21e3;
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_digger_dig_trenches() {
        let mut digger: Digger = solution().digger();

        digger.dig_trenches();

        assert_eq!(digger.cubic_meters_of_lava(), 38_usize);
        assert_eq!(digger.string(), TRENCHES_STR.to_owned());
    }

    #[test]
    fn test_digger_dig_interior() {
        let mut digger: Digger = solution().digger();

        digger.dig_trenches();
        digger.dig_interior();

        assert_eq!(digger.cubic_meters_of_lava(), 62_usize);
        assert_eq!(digger.string(), INTERIOR_STR.to_owned());
    }

    #[test]
    fn test_cubic_meters_of_lave() {
        assert_eq!(solution().cubic_meters_of_lava(), 62_usize);
    }

    #[test]
    fn test_cubic_meters_of_lava_from_colors() {
        assert_eq!(
            solution().cubic_meters_of_lava_from_colors(),
            952408144115_usize
        );
    }
}
