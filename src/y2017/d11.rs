use {
    crate::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::tag,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::terminated,
        Err, IResult,
    },
};

/* --- Day 11: Hex Ed ---

Crossing the bridge, you've barely reached the other side of the stream when a program comes up to you, clearly in distress. "It's my child process," she says, "he's gotten lost in an infinite grid!"

Fortunately for her, you have plenty of experience with infinite grids.

Unfortunately for you, it's a hex grid.

The hexagons ("hexes") in this grid are aligned such that adjacent hexes can be found to the north, northeast, southeast, south, southwest, and northwest:

  \ n  /
nw +--+ ne
  /    \
-+      +-
  \    /
sw +--+ se
  / s  \

You have the path the child process took. Starting where he started, you need to determine the fewest number of steps required to reach him. (A "step" means to move from the hex you are in to any adjacent hex.)

For example:

    ne,ne,ne is 3 steps away.
    ne,ne,sw,sw is 0 steps away (back where you started).
    ne,ne,s,s is 2 steps away (se,se).
    se,sw,se,sw,sw is 3 steps away (s,s,sw). */

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<HexDirection>);

impl Solution {
    fn parse_hex_dir<'i>(input: &'i str) -> IResult<&'i str, HexDirection> {
        alt((
            map(tag("ne"), |_| HexDirection::NorthEast),
            map(tag("nw"), |_| HexDirection::NorthWest),
            map(tag("n"), |_| HexDirection::North),
            map(tag("se"), |_| HexDirection::SouthEast),
            map(tag("sw"), |_| HexDirection::SouthWest),
            map(tag("s"), |_| HexDirection::South),
        ))(input)
    }

    fn pos(&self) -> IVec2 {
        self.0.iter().copied().map(HexDirection::vec).sum()
    }

    fn hex_manhattan_distance(&self) -> i32 {
        hex_manhattan_magnitude(self.pos())
    }

    fn iter_poses(&self) -> impl Iterator<Item = IVec2> + '_ {
        let mut pos: IVec2 = IVec2::ZERO;

        [IVec2::ZERO]
            .into_iter()
            .chain(self.0.iter().map(move |hex_dir| {
                pos += hex_dir.vec();
                pos
            }))
    }

    fn max_hex_manhattan_distance(&self) -> i32 {
        self.iter_poses()
            .map(hex_manhattan_magnitude)
            .max()
            .unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Self::parse_hex_dir, opt(tag(",")))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Not bad. I think already having a relatively established "grid utilities" helped here.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            dbg!(self.pos());
        }

        dbg!(self.hex_manhattan_distance());
    }

    /// Cheese
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.max_hex_manhattan_distance());
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

    const SOLUTION_STRS: &'static [&'static str] =
        &["ne,ne,ne", "ne,ne,sw,sw", "ne,ne,s,s", "se,sw,se,sw,sw"];

    fn solution(index: usize) -> &'static Solution {
        use HexDirection::{NorthEast as NE, South as S, SouthEast as SE, SouthWest as SW};

        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![NE, NE, NE]),
                Solution(vec![NE, NE, SW, SW]),
                Solution(vec![NE, NE, S, S]),
                Solution(vec![SE, SW, SE, SW, SW]),
            ]
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
    fn test_hex_manhattan_distance() {
        for (index, hex_manhattan_distance) in [3_i32, 0_i32, 2_i32, 3_i32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).hex_manhattan_distance(),
                hex_manhattan_distance
            );
        }
    }
}
