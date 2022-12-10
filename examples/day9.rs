use {
    aoc_2022::*,
    clap::Parser,
    glam::IVec2,
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        num::{NonZeroU32, ParseIntError},
        ops::AddAssign,
        slice::Iter,
        str::{FromStr, Split},
    },
};

/// Analogous enum to `aoc_2022::Direction`, but specifically for parsing character codes.
#[repr(usize)]
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
        (xz_direction as usize).into()
    }
}

#[derive(Debug, PartialEq)]
struct InvalidXZDirectionChar(char);

impl TryFrom<char> for XZDirection {
    type Error = InvalidXZDirectionChar;

    fn try_from(xz_direction_char: char) -> Result<Self, Self::Error> {
        Ok(match xz_direction_char {
            'u' | 'U' => XZDirection::Up,
            'r' | 'R' => XZDirection::Right,
            'd' | 'D' => XZDirection::Down,
            'l' | 'L' => XZDirection::Left,
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
enum MotionParseError<'s> {
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
struct State {
    head: IVec2,
    tail: IVec2,
}

impl State {
    fn new(initial: IVec2) -> Self {
        Self {
            head: initial,
            tail: initial,
        }
    }
}

impl AddAssign<Direction> for State {
    fn add_assign(&mut self, dir: Direction) {
        self.head += dir.vec();

        let delta: IVec2 = self.head - self.tail;
        let abs: IVec2 = delta.abs();

        if abs.x.max(abs.y) > 1_i32 {
            self.tail += delta.clamp(IVec2::NEG_ONE, IVec2::ONE);
        }
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

    fn iter(&self, initial: IVec2) -> StateIter {
        StateIter {
            motion_iter: self.0.iter(),
            state: State::new(initial),
            in_progress_motion: Motion {
                dir: Direction::North,
                dist: 0_u32,
            },
            finished: false,
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

struct StateIter<'m> {
    motion_iter: Iter<'m, Motion>,
    state: State,
    in_progress_motion: Motion,
    finished: bool,
}

impl<'m> StateIter<'m> {
    fn step(&mut self) {
        if self.in_progress_motion.dist != 0_u32 {
            self.state += self.in_progress_motion.dir;
            self.in_progress_motion.dist -= 1_u32;
        } else {
            eprintln!("Calling `StateIter::step` on iter with no more distance");
        }
    }
}

impl<'m> Iterator for StateIter<'m> {
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.finished {
            let option: Option<State> = Some(self.state.clone());

            if self.in_progress_motion.dist != 0_u32 {
                self.step();
            } else if let Some(next_motion) = self.motion_iter.next() {
                self.in_progress_motion = next_motion.clone();
                self.step();
            } else {
                self.finished = true;
            }

            option
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct HasVisited(u8);

impl HasVisited {
    const TAIL: u8 = 1_u8 << 0_u32;
    const HEAD: u8 = 1_u8 << 1_u32;
    const TAIL_OFFSET: u32 = Self::TAIL.trailing_zeros();
    const HEAD_OFFSET: u32 = Self::HEAD.trailing_zeros();

    const fn new(head: bool, tail: bool) -> Self {
        Self(((head as u8) << Self::HEAD_OFFSET) | ((tail as u8) << Self::TAIL_OFFSET))
    }

    #[inline]
    fn get_head(self) -> bool {
        (self.0 & Self::HEAD) != 0_u8
    }

    #[inline]
    fn get_tail(self) -> bool {
        (self.0 & Self::TAIL) != 0_u8
    }

    fn set_head(&mut self, head: bool) {
        self.0 = (self.0 & !Self::HEAD) | ((head as u8) << Self::HEAD_OFFSET);
    }

    fn set_tail(&mut self, tail: bool) {
        self.0 = (self.0 & !Self::TAIL) | ((tail as u8) << Self::TAIL_OFFSET);
    }
}

impl Debug for HasVisited {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("HasVisited")
            .field("head", &self.get_head())
            .field("tail", &self.get_tail())
            .finish()
    }
}

impl TryFrom<char> for HasVisited {
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
    fn visit<I: Iterator<Item = State>>(&mut self, state_iter: I);

    fn count_tails(&self) -> usize;
}

impl HasVisitedGrid for Grid<HasVisited> {
    fn visit<I: Iterator<Item = State>>(&mut self, state_iter: I) {
        for state in state_iter {
            self.get_mut(state.head).unwrap().set_head(true);
            self.get_mut(state.tail).unwrap().set_tail(true);
        }
    }

    fn count_tails(&self) -> usize {
        self.cells()
            .iter()
            .filter(|has_visited: &&HasVisited| has_visited.get_tail())
            .count()
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day9.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match MotionSequence::try_from(input) {
                    Ok(motion_sequence) => {
                        let (initial, dimensions): (IVec2, IVec2) =
                            motion_sequence.compute_initial_and_dimensions();

                        let mut has_visited_grid: Grid<HasVisited> = Grid::default(dimensions);

                        has_visited_grid.visit(motion_sequence.iter(initial));

                        println!(
                            "has_visited_grid.count_tails() == {}",
                            has_visited_grid.count_tails()
                        );
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                },
            )
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                .collect::<Vec<State>>(),
            example_states()
        );
    }

    #[test]
    fn test_visit() {
        let mut has_visited_grid: Grid<HasVisited> =
            Grid::default(example_initial_and_dimensions().1);

        has_visited_grid.visit(example_states().into_iter());

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

        let mut has_visited_grid: Grid<HasVisited> = Grid::default(dimensions);

        has_visited_grid.visit(motion_sequence.iter(initial));

        assert_eq!(has_visited_grid.count_tails(), 13_usize);
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

    fn example_states() -> Vec<State> {
        macro_rules! states {
            [$((h: ($hx:expr, $hy:expr), t: ($tx:expr, $ty:expr)),)*] => {
                vec![
                    $( State { head: IVec2::new($hx, $hy), tail: IVec2::new($tx, $ty) }, )*
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

    fn example_has_visited_grid() -> Grid<HasVisited> {
        // `rust_fmt` insists on restructuring this array, so separate constants it is
        const ROW_0: &str = " /XX/ ";
        const ROW_1: &str = " //XX/";
        const ROW_2: &str = "/XXXX/";
        const ROW_3: &str = "    X ";
        const ROW_4: &str = "XXXX/ ";
        const HAS_VISITED_GRID_STRS: [&str; 5_usize] = [ROW_0, ROW_1, ROW_2, ROW_3, ROW_4];

        Grid::try_from_cells_and_width(
            HAS_VISITED_GRID_STRS
                .iter()
                .map(|s: &&str| s.chars())
                .flatten()
                .map(HasVisited::try_from)
                .map(Result::unwrap)
                .collect(),
            6_usize,
        )
        .unwrap()
    }
}
