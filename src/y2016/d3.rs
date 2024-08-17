use {
    crate::*,
    nom::{
        character::complete::{line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 3: Squares With Three Sides ---

Now that you can think clearly, you move deeper into the labyrinth of hallways and office furniture that makes up this part of Easter Bunny HQ. This must be a graphic design department; the walls are covered in specifications for triangles.

Or are they?

The design document gives the side lengths of each triangle it describes, but... 5 10 25? Some of these aren't triangles. You can't help but mark the impossible ones.

In a valid triangle, the sum of any two sides must be larger than the remaining side. For example, the "triangle" given above is impossible, because 5 + 10 is not larger than 25.

In your puzzle input, how many of the listed triangles are possible?

--- Part Two ---

Now that you've helpfully marked up their design documents, it occurs to you that triangles are specified in groups of three vertically. Each set of three numbers in a column specifies a triangle. Rows are unrelated.

For example, given the following specification, numbers with the same hundreds digit would be part of the same triangle:

101 301 501
102 302 502
103 303 503
201 401 601
202 402 602
203 403 603

In your puzzle input, and instead reading by columns, how many of the listed triangles are possible? */

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
struct TriangleSideLengths([u32; Self::LEN]);

impl TriangleSideLengths {
    const LEN: usize = 3_usize;

    fn parse_side_length<'i>(input: &'i str) -> IResult<&'i str, u32> {
        preceded(space0, parse_integer)(input)
    }

    fn get_side_length(&self, index: usize) -> u32 {
        self.0[index % self.0.len()]
    }

    fn is_possible(&self) -> bool {
        (0_usize..self.0.len()).all(|offset| {
            self.get_side_length(0_usize + offset) + self.get_side_length(1_usize + offset)
                > self.get_side_length(2_usize + offset)
        })
    }
}

impl From<(u32, u32, u32)> for TriangleSideLengths {
    fn from((a, b, c): (u32, u32, u32)) -> Self {
        Self([a, b, c])
    }
}

impl Parse for TriangleSideLengths {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                Self::parse_side_length,
                Self::parse_side_length,
                Self::parse_side_length,
            )),
            Self::from,
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<TriangleSideLengths>);

impl Solution {
    fn iter_transposed_triangle_side_lengths(
        &self,
    ) -> impl Iterator<Item = TriangleSideLengths> + '_ {
        self.0
            .chunks_exact(TriangleSideLengths::LEN)
            .flat_map(|row| {
                row[0_usize]
                    .0
                    .iter()
                    .zip(row[1_usize].0.iter())
                    .zip(row[2_usize].0.iter())
                    .map(|((a, b), c)| TriangleSideLengths([*a, *b, *c]))
            })
    }

    fn count_possible_transposed_triangles(&self) -> usize {
        self.iter_transposed_triangle_side_lengths()
            .filter(TriangleSideLengths::is_possible)
            .count()
    }

    fn count_possible_triangles(&self) -> usize {
        self.0
            .iter()
            .filter(|triangle_side_lengths| triangle_side_lengths.is_possible())
            .count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            many0(terminated(TriangleSideLengths::parse, opt(line_ending))),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_possible_triangles());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_possible_transposed_triangles());
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

    const SOLUTION_STR: &'static str = "5 10 25\n\
        5 25 10\n\
        10 5 25\n\
        10 25 5\n\
        25 5 10\n\
        25 10 5\n";

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        macro_rules! solution {
            [ $( ($a:expr, $b:expr, $c:expr), )* ] => {
                Solution(vec![ $( TriangleSideLengths([$a, $b, $c]), )* ])
            };
        }

        &ONCE_LOCK.get_or_init(|| {
            vec![
                solution![
                    (5, 10, 25),
                    (5, 25, 10),
                    (10, 5, 25),
                    (10, 25, 5),
                    (25, 5, 10),
                    (25, 10, 5),
                ],
                solution![
                    (101, 301, 501),
                    (102, 302, 502),
                    (103, 303, 503),
                    (201, 401, 601),
                    (202, 402, 602),
                    (203, 403, 603),
                ],
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(
            Solution::try_from(SOLUTION_STR).as_ref(),
            Ok(solution(0_usize))
        );
    }

    #[test]
    fn test_count_possible_triangles() {
        assert_eq!(solution(0_usize).count_possible_triangles(), 0_usize);
    }

    #[test]
    fn test_count_possible_transposed_triangles() {
        assert_eq!(
            solution(1_usize).count_possible_transposed_triangles(),
            6_usize
        );
    }
}
