use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    std::{
        cell::RefCell,
        fmt::Debug,
        marker::PhantomData,
        mem::{size_of_val, transmute, MaybeUninit},
        num::{NonZeroU32, ParseIntError},
        ops::AddAssign,
        slice::Iter,
        str::{FromStr, Split},
    },
};

/// Analogous enum to `aoc::Direction`, but specifically for parsing character codes.
#[repr(u8)]
enum XZDirection {
    /// Analog to `Direction::North`
    Up,

    /// Analog to `Direction::East`
    Right,

    /// Analog to `Direction::South`
    Down,

    /// Analog to `Direction::West`
    Left,
}

impl From<XZDirection> for Direction {
    fn from(xz_direction: XZDirection) -> Self {
        (xz_direction as u8).into()
    }
}

#[derive(Debug, PartialEq)]
pub struct InvalidXZDirectionChar(char);

impl TryFrom<char> for XZDirection {
    type Error = InvalidXZDirectionChar;

    fn try_from(xz_direction_char: char) -> Result<Self, Self::Error> {
        Ok(match xz_direction_char {
            'U' => XZDirection::Up,
            'R' => XZDirection::Right,
            'D' => XZDirection::Down,
            'L' => XZDirection::Left,
            _ => Err(InvalidXZDirectionChar(xz_direction_char))?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Motion {
    dir: Direction,
    dist: u32,
}

impl Motion {
    fn vec(&self) -> IVec2 {
        self.dist as i32 * self.dir.vec()
    }
}

impl From<Motion> for IVec2 {
    fn from(motion: Motion) -> Self {
        motion.vec()
    }
}

#[derive(Debug, PartialEq)]
pub enum MotionParseError<'s> {
    NoDirToken,
    InvalidDirTokenLength(&'s str),
    FailedToParseDir(InvalidXZDirectionChar),
    NoDistToken,
    FailedToParseDist(ParseIntError),
    DistTooLarge(u32),
    ExtraTokenFound,
}

impl<'s> TryFrom<&'s str> for Motion {
    type Error = MotionParseError<'s>;

    fn try_from(motion_str: &'s str) -> Result<Self, Self::Error> {
        use MotionParseError as Error;

        let mut token_iter: Split<char> = motion_str.split(' ');

        let dir: Direction = match token_iter.next() {
            None => Err(Error::NoDirToken),
            Some(dir_str) if dir_str.len() == 1_usize => {
                Ok(XZDirection::try_from(dir_str.chars().next().unwrap())
                    .map_err(Error::FailedToParseDir)?
                    .into())
            }
            Some(dir_str) => Err(Error::InvalidDirTokenLength(dir_str)),
        }?;

        let dist: u32 = match token_iter.next() {
            None => Err(Error::NoDistToken),
            Some(dist_str) => NonZeroU32::from_str(dist_str).map_err(Error::FailedToParseDist),
        }?
        .get();

        if dist > i32::MAX as u32 {
            Err(Error::DistTooLarge(dist))
        } else if token_iter.next().is_some() {
            Err(Error::ExtraTokenFound)
        } else {
            Ok(Self { dir, dist })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RopeState<const N: usize>([IVec2; N]);

impl<const N: usize> RopeState<N> {
    const N: usize = N;
    const TAIL: usize = Self::N - 1_usize;

    fn new(initial: IVec2) -> Self {
        let mut mu_rope_state: MaybeUninit<RopeState<N>> = MaybeUninit::uninit();
        let mu_rope_state_size: usize = size_of_val(&mu_rope_state);

        // SAFETY: We're transmuting from a `MaybeUninit` struct to an array of `MaybeUninit`
        // structs of the same size and alignment
        let mu_vec_array: &mut [MaybeUninit<IVec2>; N] = unsafe { transmute(&mut mu_rope_state) };

        // sanity check to make sure our assumptions aren't wrong. The compiler complains easily
        // when using const generics, so let's just make sure this is correct
        assert_eq!(size_of_val(mu_vec_array), mu_rope_state_size);

        for mu_vec in mu_vec_array {
            mu_vec.write(initial);
        }

        // SAFETY: We just initialized each member of `mu_rope_state`, doubly guaranteed by the
        // assert
        unsafe { mu_rope_state.assume_init() }
    }

    pub fn head(&self) -> &IVec2 {
        &self.0[0_usize]
    }

    pub fn tail(&self) -> &IVec2 {
        &self.0[Self::TAIL]
    }

    pub fn head_mut(&mut self) -> &mut IVec2 {
        &mut self.0[0_usize]
    }

    pub fn tail_mut(&mut self) -> &mut IVec2 {
        &mut self.0[Self::TAIL]
    }
}

impl<const N: usize> RopeState<N>
where
    Self: Default,
{
    pub fn from_head_and_tail(head: IVec2, tail: IVec2) -> Self {
        let mut rope_state: Self = Self::default();

        *rope_state.head_mut() = head;
        *rope_state.tail_mut() = tail;

        rope_state
    }
}

impl<const N: usize> AddAssign<IVec2> for RopeState<N> {
    fn add_assign(&mut self, mut impulse: IVec2) {
        if Self::N == 0_usize {
            return;
        }

        let mut previous: IVec2 = self.0[0_usize] + 2_i32 * impulse;

        for knot in &mut self.0 {
            let delta: IVec2 = previous - *knot;
            let abs: IVec2 = delta.abs();

            // This knot doesn't move, and the impulse isn't great enough to move any subsequent
            // knots
            if abs.x.max(abs.y) <= 1_i32 {
                break;
            }

            impulse = delta.clamp(IVec2::NEG_ONE, IVec2::ONE);
            *knot += impulse;
            previous = *knot;
        }
    }
}

impl<const N: usize> Default for RopeState<N>
where
    [IVec2; N]: Default,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Debug, PartialEq)]
struct MotionSequence(Vec<Motion>);

impl MotionSequence {
    fn compute_initial_and_dimensions(&self) -> (IVec2, IVec2) {
        let mut curr: IVec2 = IVec2::ZERO;
        let mut min: IVec2 = IVec2::ZERO;
        let mut max: IVec2 = IVec2::ZERO;

        for motion in self.0.iter() {
            curr += IVec2::from(motion.clone());
            min = min.min(curr);
            max = max.max(curr);
        }

        let initial: IVec2 = -min;
        let dimensions: IVec2 = initial + max + IVec2::ONE;

        (initial, dimensions)
    }

    fn iter<'a, const N: usize>(&'a self, initial: IVec2) -> RopeStateIter<'a, N> {
        RopeStateIter {
            motion_iter: self.0.iter(),
            rope_state: RefCell::new(RopeState::new(initial)),
            in_progress_motion: Motion {
                dir: Direction::North,
                dist: 0_u32,
            },
            started: false,
        }
    }
}

impl<'s> TryFrom<&'s str> for MotionSequence {
    type Error = MotionParseError<'s>;

    fn try_from(motion_sequence_str: &'s str) -> Result<Self, Self::Error> {
        let mut motion_sequence: MotionSequence = MotionSequence(Vec::new());

        for motion_str in motion_sequence_str.split('\n') {
            motion_sequence.0.push(motion_str.try_into()?);
        }

        Ok(motion_sequence)
    }
}

struct RopeStateIter<'a, const N: usize> {
    motion_iter: Iter<'a, Motion>,
    rope_state: RefCell<RopeState<N>>,
    in_progress_motion: Motion,
    started: bool,
}

impl<'a, const N: usize> Iterator for RopeStateIter<'a, N> {
    type Item = RefCell<RopeState<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;

            Some(self.rope_state.clone())
        } else {
            if self.in_progress_motion.dist == 0_u32 {
                if let Some(next_motion) = self.motion_iter.next() {
                    self.in_progress_motion = next_motion.clone();
                }
            }

            if self.in_progress_motion.dist != 0_u32 {
                *self.rope_state.borrow_mut() += self.in_progress_motion.dir.vec();
                self.in_progress_motion.dist -= 1_u32;

                Some(unsafe { transmute(self.rope_state.clone()) })
            } else {
                None
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HasVisited<const N: usize>(BitArray<u16>, PhantomData<[(); N]>);

impl<const N: usize> HasVisited<N> {
    const N: usize = N;
    const TAIL: usize = Self::N - 1_usize;

    fn new(head: bool, tail: bool) -> Self {
        let mut has_visited: HasVisited<N> = HasVisited::default();

        has_visited.set_head(head);
        has_visited.set_tail(tail);

        has_visited
    }

    fn set_head(&mut self, head: bool) {
        self.0.set(0_usize, head);
    }

    fn set_tail(&mut self, tail: bool) {
        self.0.set(Self::TAIL, tail);
    }

    fn get(self, index: usize) -> bool {
        self.0[index]
    }

    fn set(&mut self, index: usize, value: bool) {
        self.0.set(index, value);
    }
}

impl<const N: usize> TryFrom<char> for HasVisited<N> {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        Ok(match value {
            ' ' => Self::new(false, false),
            '\\' => Self::new(false, true),
            '/' => Self::new(true, false),
            'X' => Self::new(true, true),
            _ => Err(())?,
        })
    }
}

trait HasVisitedGrid {
    type State;

    fn visit<I: Iterator<Item = Self::State>>(&mut self, state_iter: I);
    fn count_visited(&self, index: usize) -> usize;
    fn count_tails(&self) -> usize;
}

impl<const N: usize> HasVisitedGrid for Grid2D<HasVisited<N>> {
    type State = RefCell<RopeState<N>>;

    fn visit<I: Iterator<Item = Self::State>>(&mut self, state_iter: I) {
        for state in state_iter {
            for (index, pos) in state.borrow().0.iter().enumerate() {
                self.get_mut(*pos).unwrap().set(index, true);
            }
        }
    }

    fn count_visited(&self, index: usize) -> usize {
        self.cells()
            .iter()
            .filter(|has_visited: &&HasVisited<N>| has_visited.get(index))
            .count()
    }

    fn count_tails(&self) -> usize {
        self.count_visited(HasVisited::<N>::TAIL)
    }
}

type SolutionHasVisitedGrid = Grid2D<HasVisited<{ Solution::N }>>;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(SolutionHasVisitedGrid);

impl Solution {
    const TAIL_1: usize = 1_usize;
    const TAIL_2: usize = 9_usize;
    const N: usize = if Solution::TAIL_1 > Solution::TAIL_2 {
        Solution::TAIL_1
    } else {
        Solution::TAIL_2
    } + 1_usize;

    fn count_visited_tail_1(&self) -> usize {
        self.0.count_visited(Self::TAIL_1)
    }

    fn count_visited_tail_2(&self) -> usize {
        self.0.count_visited(Self::TAIL_2)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_visited_tail_1());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_visited_tail_2());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = MotionParseError<'i>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        let motion_sequence: MotionSequence = input.try_into()?;
        let (initial, dimensions): (IVec2, IVec2) =
            motion_sequence.compute_initial_and_dimensions();

        let mut has_visited_grid: SolutionHasVisitedGrid =
            SolutionHasVisitedGrid::default(dimensions);

        has_visited_grid.visit(motion_sequence.iter(initial));

        Ok(Self(has_visited_grid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 2_usize;
    const MOTION_SEQUENCE_STR: &str = "\
        R 4\n\
        U 4\n\
        L 3\n\
        D 1\n\
        R 4\n\
        D 1\n\
        L 5\n\
        R 2";

    #[test]
    fn test_motion_sequence_try_from_str() {
        assert_eq!(
            MotionSequence::try_from(MOTION_SEQUENCE_STR),
            Ok(example_motion_sequence())
        );
    }

    #[test]
    fn test_compute_initial_and_dimensions() {
        assert_eq!(
            example_motion_sequence().compute_initial_and_dimensions(),
            example_initial_and_dimensions()
        );
    }

    #[test]
    fn test_state_iter() {
        assert_eq!(
            example_motion_sequence()
                .iter(example_initial_and_dimensions().0)
                .map(
                    |ref_cell_rope_state: RefCell<RopeState<N>>| ref_cell_rope_state
                        .borrow()
                        .clone()
                )
                .collect::<Vec<RopeState<N>>>(),
            example_states()
        );
    }

    #[test]
    fn test_visit() {
        let mut has_visited_grid: Grid2D<HasVisited<N>> =
            Grid2D::default(example_initial_and_dimensions().1);

        has_visited_grid.visit(example_states().into_iter().map(RefCell::new));

        assert_eq!(has_visited_grid, example_has_visited_grid());
    }

    #[test]
    fn test_count_tails() {
        assert_eq!(example_has_visited_grid().count_tails(), 13_usize);
    }

    #[test]
    fn test_full() {
        let motion_sequence: MotionSequence =
            MotionSequence::try_from(MOTION_SEQUENCE_STR).unwrap();
        let (initial, dimensions): (IVec2, IVec2) =
            motion_sequence.compute_initial_and_dimensions();

        let mut has_visited_grid: Grid2D<HasVisited<N>> = Grid2D::default(dimensions);

        has_visited_grid.visit(motion_sequence.iter(initial));

        assert_eq!(has_visited_grid.count_tails(), 13_usize);
    }

    #[test]
    fn test_ten_knots() {
        const N: usize = 10_usize;
        const MOTION_SEQUENCE_STR: &str = "\
            R 5\n\
            U 8\n\
            L 8\n\
            D 3\n\
            R 17\n\
            D 10\n\
            L 25\n\
            U 20";

        let motion_sequence: MotionSequence =
            MotionSequence::try_from(MOTION_SEQUENCE_STR).unwrap();
        let (initial, dimensions): (IVec2, IVec2) =
            motion_sequence.compute_initial_and_dimensions();

        let mut has_visited_grid: Grid2D<HasVisited<N>> = Grid2D::default(dimensions);

        has_visited_grid.visit(motion_sequence.iter(initial));

        assert_eq!(has_visited_grid.count_tails(), 36_usize);
    }

    fn example_initial_and_dimensions() -> (IVec2, IVec2) {
        (IVec2::new(0_i32, 4_i32), IVec2::new(6_i32, 5_i32))
    }

    fn example_motion_sequence() -> MotionSequence {
        use Direction::*;

        macro_rules! motion_sequence {
            [$(($dir:ident, $dist:expr),)*] => {
                MotionSequence(vec![
                    $( Motion { dir: $dir, dist: $dist }, )*
                ])
            };
        }

        motion_sequence![
            (East, 4),
            (North, 4),
            (West, 3),
            (South, 1),
            (East, 4),
            (South, 1),
            (West, 5),
            (East, 2),
        ]
    }

    fn example_states() -> Vec<RopeState<N>> {
        macro_rules! states {
            [$((h: ($hx:expr, $hy:expr), t: ($tx:expr, $ty:expr)),)*] => {
                vec![
                    $( RopeState::from_head_and_tail(IVec2::new($hx, $hy), IVec2::new($tx, $ty)), )*
                ]
            };
        }

        states![
            (h: (0, 4), t: (0, 4)),
            (h: (1, 4), t: (0, 4)),
            (h: (2, 4), t: (1, 4)),
            (h: (3, 4), t: (2, 4)),
            (h: (4, 4), t: (3, 4)),
            (h: (4, 3), t: (3, 4)),
            (h: (4, 2), t: (4, 3)),
            (h: (4, 1), t: (4, 2)),
            (h: (4, 0), t: (4, 1)),
            (h: (3, 0), t: (4, 1)),
            (h: (2, 0), t: (3, 0)),
            (h: (1, 0), t: (2, 0)),
            (h: (1, 1), t: (2, 0)),
            (h: (2, 1), t: (2, 0)),
            (h: (3, 1), t: (2, 0)),
            (h: (4, 1), t: (3, 1)),
            (h: (5, 1), t: (4, 1)),
            (h: (5, 2), t: (4, 1)),
            (h: (4, 2), t: (4, 1)),
            (h: (3, 2), t: (4, 1)),
            (h: (2, 2), t: (3, 2)),
            (h: (1, 2), t: (2, 2)),
            (h: (0, 2), t: (1, 2)),
            (h: (1, 2), t: (1, 2)),
            (h: (2, 2), t: (1, 2)),
        ]
    }

    fn example_has_visited_grid() -> Grid2D<HasVisited<N>> {
        // `rust_fmt` insists on restructuring this array, so separate constants it is
        const ROW_0: &str = " /XX/ ";
        const ROW_1: &str = " //XX/";
        const ROW_2: &str = "/XXXX/";
        const ROW_3: &str = "    X ";
        const ROW_4: &str = "XXXX/ ";
        const HAS_VISITED_GRID_STRS: [&str; 5_usize] = [ROW_0, ROW_1, ROW_2, ROW_3, ROW_4];

        Grid2D::try_from_cells_and_width(
            HAS_VISITED_GRID_STRS
                .iter()
                .map(|s: &&str| s.chars())
                .flatten()
                .map(HasVisited::<N>::try_from)
                .map(Result::unwrap)
                .collect(),
            6_usize,
        )
        .unwrap()
    }
}
