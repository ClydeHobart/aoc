use {
    crate::*,
    glam::IVec2,
    nom::{
        combinator::{map_opt, success},
        error::Error,
        Err, IResult,
    },
};

/* --- Day 19: A Series of Tubes ---

Somehow, a network packet got lost and ended up here. It's trying to follow a routing diagram (your puzzle input), but it's confused about where to go.

Its starting point is just off the top of the diagram. Lines (drawn with |, -, and +) show the path it needs to take, starting by going down onto the only line connected to the top of the diagram. It needs to follow this path until it reaches the end (located somewhere within the diagram) and stop there.

Sometimes, the lines cross over each other; in these cases, it needs to continue going the same direction, and only turn left or right when there's no other option. In addition, someone has left letters on the line; these also don't change its direction, but it can use them to keep track of where it's been. For example:

     |
     |  +--+
     A  |  C
 F---|----E|--+
     |  |  |  D
     +B-+  +--+

Given this diagram, the packet needs to take the following path:

    Starting at the only line touching the top of the diagram, it must go down, pass through A, and continue onward to the first +.
    Travel right, up, and right, passing through B in the process.
    Continue down (collecting C), right, and up (collecting D).
    Finally, go all the way left through E and stopping at F.

Following the path to the end, the letters it sees on its path are ABCDEF.

The little packet looks up at you, hoping you can help it find the way. What letters will it see (in the order it would see them) if it follows the path? (The routing diagram is very wide; make sure you view it without line wrapping.)

--- Part Two ---

The packet is curious how many steps it needs to go.

For example, using the same routing diagram from the example above...

     |
     |  +--+
     A  |  C
 F---|--|-E---+
     |  |  |  D
     +B-+  +--+

...the packet would go:

    6 steps down (including the first line at the top of the diagram).
    3 steps right.
    4 steps up.
    3 steps right.
    4 steps down.
    3 steps right.
    2 steps up.
    13 steps left (including the F it stops on).

This would result in a total of 38 steps.

How many steps does the packet need to go? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        Empty = EMPTY = b' ',
        HorizontalPipe = HORIZONTAL_PIPE = b'-',
        VerticalPipe = VERTICAL_PIPE = b'|',
        CornerPipe = CORNER_PIPE = b'+',
        A = A_VALUE = b'A',
        B = B_VALUE = b'B',
        C = C_VALUE = b'C',
        D = D_VALUE = b'D',
        E = E_VALUE = b'E',
        F = F_VALUE = b'F',
        G = G_VALUE = b'G',
        H = H_VALUE = b'H',
        I = I_VALUE = b'I',
        J = J_VALUE = b'J',
        K = K_VALUE = b'K',
        L = L_VALUE = b'L',
        M = M_VALUE = b'M',
        N = N_VALUE = b'N',
        O = O_VALUE = b'O',
        P = P_VALUE = b'P',
        Q = Q_VALUE = b'Q',
        R = R_VALUE = b'R',
        S = S_VALUE = b'S',
        T = T_VALUE = b'T',
        U = U_VALUE = b'U',
        V = V_VALUE = b'V',
        W = W_VALUE = b'W',
        X = X_VALUE = b'X',
        Y = Y_VALUE = b'Y',
        Z = Z_VALUE = b'Z',
    }
}

impl Cell {
    fn is_empty(self) -> bool {
        self == Self::Empty
    }

    fn is_pipe(self) -> bool {
        matches!(
            self,
            Self::HorizontalPipe | Self::VerticalPipe | Self::CornerPipe
        )
    }

    fn is_letter(self) -> bool {
        !self.is_empty() && !self.is_pipe()
    }
}

impl From<Direction> for Cell {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North | Direction::South => Self::VerticalPipe,
            Direction::East | Direction::West => Self::HorizontalPipe,
        }
    }
}

struct PacketPosIter<'g> {
    grid: &'g Grid2D<Cell>,
    pos: IVec2,
    dir: Direction,
}

impl<'g> Iterator for PacketPosIter<'g> {
    type Item = IVec2;

    fn next(&mut self) -> Option<Self::Item> {
        self.grid
            .get(self.pos)
            .copied()
            .filter(|cell| !cell.is_empty())
            .map(|cell| {
                let next: IVec2 = self.pos;

                if cell != Cell::CornerPipe {
                    self.pos += self.dir.vec();
                } else if let Some((pos, dir)) = [self.dir.prev(), self.dir.next()]
                    .into_iter()
                    .map(|dir| (self.pos + dir.vec(), dir))
                    .find(|(pos, _)| !self.grid.get(*pos).unwrap().is_empty())
                {
                    self.pos = pos;
                    self.dir = dir;
                } else {
                    self.pos = IVec2::NEG_ONE;
                }

                next
            })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Cell>,
    start: IVec2,
}

impl Solution {
    fn packet_pos_iter<'g>(&'g self) -> PacketPosIter<'g> {
        PacketPosIter {
            grid: &self.grid,
            pos: self.start,
            dir: Direction::South,
        }
    }

    fn map_pos_iter_to_letter_iter<'i, I: Iterator<Item = IVec2> + 'i>(
        &'i self,
        pos_iter: I,
    ) -> impl Iterator<Item = char> + 'i {
        pos_iter.filter_map(|pos| {
            self.grid
                .get(pos)
                .copied()
                .filter(|cell| cell.is_letter())
                .map(|cell| cell as u8 as char)
        })
    }

    fn packet_letters(&self) -> String {
        self.map_pos_iter_to_letter_iter(self.packet_pos_iter())
            .collect()
    }

    fn packet_pos_count(&self) -> usize {
        self.packet_pos_iter().count()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, grid): (&str, Grid2D<Cell>) = Grid2D::parse(input)?;
        let start: IVec2 = map_opt(success(()), |_| {
            CellIter2D::try_from(IVec2::ZERO..grid.dimensions() * IVec2::X)
                .ok()?
                .find(|pos| *grid.get(*pos).unwrap() == Cell::VerticalPipe)
        })(input)?
        .1;

        Ok((input, Self { grid, start }))
    }
}

impl RunQuestions for Solution {
    /// I'm anticipating that this will handle pipe intersections (like at (5, 3) in the example)
    /// differently, instead either 1) forcing the packet into one of the directions of the top-most
    /// pipe, preferring either clockwise or counter-clockwise or 2) forcing the packet to always
    /// turn clockwise or counter-clockwise at these. That's why `map_pos_iter_to_letter_iter` looks
    /// the way it does.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.packet_letters());
    }

    /// Dang, this was extremely easy for part 2 on a day 19.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.packet_pos_count());
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
        "     |          \n",
        "     |  +--+    \n",
        "     A  |  C    \n",
        " F---|----E|--+ \n",
        "     |  |  |  D \n",
        "     +B-+  +--+ \n",
    )];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use Cell::{
                CornerPipe as K, Empty as M, HorizontalPipe as H, VerticalPipe as V, A, B, C, D, E,
                F,
            };

            vec![Solution {
                grid: Grid2D::try_from_cells_and_width(
                    vec![
                        M, M, M, M, M, V, M, M, M, M, M, M, M, M, M, M, M, M, M, M, M, V, M, M, K,
                        H, H, K, M, M, M, M, M, M, M, M, M, A, M, M, V, M, M, C, M, M, M, M, M, F,
                        H, H, H, V, H, H, H, H, E, V, H, H, K, M, M, M, M, M, M, V, M, M, V, M, M,
                        V, M, M, D, M, M, M, M, M, M, K, B, H, K, M, M, K, H, H, K, M,
                    ],
                    16_usize,
                )
                .unwrap(),
                start: IVec2::new(5_i32, 0_i32),
            }]
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
    fn test_packet_pos_iter() {
        for (index, packet_poses) in [vec![
            IVec2::new(5_i32, 0_i32),
            IVec2::new(5_i32, 1_i32),
            IVec2::new(5_i32, 2_i32),
            IVec2::new(5_i32, 3_i32),
            IVec2::new(5_i32, 4_i32),
            IVec2::new(5_i32, 5_i32),
            IVec2::new(6_i32, 5_i32),
            IVec2::new(7_i32, 5_i32),
            IVec2::new(8_i32, 5_i32),
            IVec2::new(8_i32, 4_i32),
            IVec2::new(8_i32, 3_i32),
            IVec2::new(8_i32, 2_i32),
            IVec2::new(8_i32, 1_i32),
            IVec2::new(9_i32, 1_i32),
            IVec2::new(10_i32, 1_i32),
            IVec2::new(11_i32, 1_i32),
            IVec2::new(11_i32, 2_i32),
            IVec2::new(11_i32, 3_i32),
            IVec2::new(11_i32, 4_i32),
            IVec2::new(11_i32, 5_i32),
            IVec2::new(12_i32, 5_i32),
            IVec2::new(13_i32, 5_i32),
            IVec2::new(14_i32, 5_i32),
            IVec2::new(14_i32, 4_i32),
            IVec2::new(14_i32, 3_i32),
            IVec2::new(13_i32, 3_i32),
            IVec2::new(12_i32, 3_i32),
            IVec2::new(11_i32, 3_i32),
            IVec2::new(10_i32, 3_i32),
            IVec2::new(9_i32, 3_i32),
            IVec2::new(8_i32, 3_i32),
            IVec2::new(7_i32, 3_i32),
            IVec2::new(6_i32, 3_i32),
            IVec2::new(5_i32, 3_i32),
            IVec2::new(4_i32, 3_i32),
            IVec2::new(3_i32, 3_i32),
            IVec2::new(2_i32, 3_i32),
            IVec2::new(1_i32, 3_i32),
        ]]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).packet_pos_iter().collect::<Vec<IVec2>>(),
                packet_poses
            );
        }
    }

    #[test]
    fn test_packet_letters() {
        for (index, packet_letters) in ["ABCDEF"].into_iter().enumerate() {
            assert_eq!(solution(index).packet_letters(), packet_letters);
        }
    }

    #[test]
    fn test_packet_pos_count() {
        for (index, packet_pos_count) in [38_usize].into_iter().enumerate() {
            assert_eq!(solution(index).packet_pos_count(), packet_pos_count);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
