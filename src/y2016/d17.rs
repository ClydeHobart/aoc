use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    md5::compute,
    nom::{character::complete::not_line_ending, combinator::map, error::Error, Err, IResult},
    std::{
        cell::{RefCell, RefMut},
        collections::{HashSet, VecDeque},
        fmt::{Debug, Formatter, Result as FmtResult},
    },
    strum::{EnumCount, IntoEnumIterator},
};

/* --- Day 17: Two Steps Forward ---

You're trying to access a secure vault protected by a 4x4 grid of small rooms connected by doors. You start in the top-left room (marked S), and you can access the vault (marked V) once you reach the bottom-right room:

#########
#S| | | #
#-#-#-#-#
# | | | #
#-#-#-#-#
# | | | #
#-#-#-#-#
# | | |
####### V

Fixed walls are marked with #, and doors are marked with - or |.

The doors in your current room are either open or closed (and locked) based on the hexadecimal MD5 hash of a passcode (your puzzle input) followed by a sequence of uppercase characters representing the path you have taken so far (U for up, D for down, L for left, and R for right).

Only the first four characters of the hash are used; they represent, respectively, the doors up, down, left, and right from your current position. Any b, c, d, e, or f means that the corresponding door is open; any other character (any number or a) means that the corresponding door is closed and locked.

To access the vault, all you need to do is reach the bottom-right room; reaching this room opens the vault and all doors in the maze.

For example, suppose the passcode is hijkl. Initially, you have taken no steps, and so your path is empty: you simply find the MD5 hash of hijkl alone. The first four characters of this hash are ced9, which indicate that up is open (c), down is open (e), left is open (d), and right is closed and locked (9). Because you start in the top-left corner, there are no "up" or "left" doors to be open, so your only choice is down.

Next, having gone only one step (down, or D), you find the hash of hijklD. This produces f2bc, which indicates that you can go back up, left (but that's a wall), or right. Going right means hashing hijklDR to get 5745 - all doors closed and locked. However, going up instead is worthwhile: even though it returns you to the room you started in, your path would then be DU, opening a different set of doors.

After going DU (and then hashing hijklDU to get 528e), only the right door is open; after going DUR, all doors lock. (Fortunately, your actual passcode is not hijkl).

Passcodes actually used by Easter Bunny Vault Security do allow access to the vault if you know the right path. For example:

    If your passcode were ihgpwlah, the shortest path would be DDRRRD.
    With kglvqrro, the shortest path would be DDUDRLRRUDRD.
    With ulqzkmiv, the shortest would be DRURDRUDDLLDLUURRDULRLDUUDDDRR.

Given your vault's passcode, what is the shortest path (the actual path, not just the length) to reach the vault?

--- Part Two ---

You're curious how robust this security solution really is, and so you decide to find longer and longer paths which still provide access to the vault. You remember that paths always end the first time they reach the bottom-right room (that is, they can never pass through it, only end in it).

For example:

    If your passcode were ihgpwlah, the longest path would take 370 steps.
    With kglvqrro, the longest path would be 492 steps long.
    With ulqzkmiv, the longest path would be 830 steps long.

What is the length of the longest path that reaches the vault? */

#[derive(Clone, Default, Eq, Hash, PartialEq)]
struct DirPath<const PATH_BITARR_LEN: usize = 2_usize> {
    path_bitarr: BitArray<[usize; PATH_BITARR_LEN]>,
    len: usize,
}

const BITS_PER_DIR: usize = 2_usize;
const DIRS_PER_USIZE: usize = usize::BITS as usize / BITS_PER_DIR;
const BITS_PER_NIBBLE: usize = u8::BITS as usize / 2_usize;

const DIR_BYTES: [u8; Direction::COUNT] = [b'U', b'R', b'D', b'L'];

fn compute_is_door<I1: Iterator<Item = Direction>>(
    dir_iter: I1,
    hash_buffer: &mut String,
    truncate_len: usize,
) -> [bool; Direction::COUNT] {
    hash_buffer.truncate(truncate_len);

    for dir in dir_iter {
        hash_buffer.push(DIR_BYTES[dir as usize] as char);
    }

    let mut is_door: [bool; Direction::COUNT] = [false; Direction::COUNT];

    for (dir_nibble, dir) in compute(hash_buffer.as_bytes())
        .0
        .as_bits::<Msb0>()
        .chunks_exact(BITS_PER_NIBBLE)
        .map(|dir_nibble_bits| dir_nibble_bits.load::<u8>())
        .zip([
            Direction::North,
            Direction::South,
            Direction::West,
            Direction::East,
        ])
    {
        is_door[dir as usize] = dir_nibble >= 0xb_u8;
    }

    is_door
}

impl<const PATH_BITARR_LEN: usize> DirPath<PATH_BITARR_LEN> {
    const fn path_bitarr_len() -> usize {
        PATH_BITARR_LEN
    }

    const fn capacity(&self) -> usize {
        Self::path_bitarr_len() * DIRS_PER_USIZE
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }

    fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    fn iter_dirs(&self) -> impl Iterator<Item = Direction> + '_ {
        self.path_bitarr[..self.len * BITS_PER_DIR]
            .chunks_exact(BITS_PER_DIR)
            .map(|dir_bits| dir_bits.load::<u8>().into())
    }

    fn iter_chars(&self) -> impl Iterator<Item = char> + '_ {
        self.iter_dirs().map(|dir| DIR_BYTES[dir as usize] as char)
    }

    fn pos(&self) -> IVec2 {
        self.iter_dirs().map(Direction::vec).sum()
    }

    fn compute_is_door(
        &self,
        hash_buffer: &mut String,
        passcode_len: usize,
    ) -> [bool; Direction::COUNT] {
        compute_is_door(self.iter_dirs(), hash_buffer, passcode_len)
    }

    fn push(&mut self, dir: Direction) {
        assert!(!self.is_full());

        let start: usize = self.len * BITS_PER_DIR;

        self.path_bitarr[start..start + BITS_PER_DIR].store(dir as u8);
        self.len += 1_usize;
    }

    fn pop(&mut self) -> Direction {
        assert!(!self.is_empty());

        self.len -= 1_usize;

        let start: usize = self.len * BITS_PER_DIR;

        self.path_bitarr[start..start + BITS_PER_DIR]
            .load::<u8>()
            .into()
    }
}

impl<const PATH_BITARR_LEN: usize> Debug for DirPath<PATH_BITARR_LEN> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_tuple("DirPath")
            .field(&self.iter_chars().collect::<String>())
            .finish()
    }
}

impl<const PATH_BITARR_LEN: usize> FromIterator<Direction> for DirPath<PATH_BITARR_LEN> {
    fn from_iter<T: IntoIterator<Item = Direction>>(iter: T) -> Self {
        let mut dir_path: Self = Default::default();

        for dir in iter {
            dir_path.push(dir);
        }

        dir_path
    }
}

struct ShortestPathFinder<const PATH_BITARR_LEN: usize = 2_usize> {
    start_dir_path: DirPath<PATH_BITARR_LEN>,
    start: IVec2,
    end: IVec2,
    passcode_len: usize,
    hash_buffer: RefCell<String>,
    visited: HashSet<DirPath<PATH_BITARR_LEN>>,
}

impl<const PATH_BITARR_LEN: usize> WeightedGraphSearch for ShortestPathFinder<PATH_BITARR_LEN> {
    type Vertex = DirPath<PATH_BITARR_LEN>;
    type Cost = usize;

    fn start(&self) -> &Self::Vertex {
        &self.start_dir_path
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.pos() == self.end - self.start
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<Self::Vertex> = VecDeque::with_capacity(vertex.len() + 1_usize);
        let mut dir_path: Self::Vertex = vertex.clone();

        for _ in 0_usize..=vertex.len() {
            path.push_front(dir_path.clone());

            if !dir_path.is_empty() {
                dir_path.pop();
            }
        }

        path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        if self.visited.contains(vertex) {
            vertex.len()
        } else {
            usize::MAX
        }
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        manhattan_distance_2d(vertex.pos(), self.end) as usize
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        if !vertex.is_full() {
            let mut hash_buffer: RefMut<String> = self.hash_buffer.borrow_mut();
            let is_door: [bool; Direction::COUNT] =
                vertex.compute_is_door(&mut hash_buffer, self.passcode_len);

            if is_door.into_iter().any(|is_door| is_door) {
                let pos: IVec2 = vertex.pos() + self.start;

                neighbors.extend(Direction::iter().filter_map(|dir| {
                    let neighbor_pos: IVec2 = pos + dir.vec();

                    (is_door[dir as usize]
                        && neighbor_pos.cmpge(self.start).all()
                        && neighbor_pos.cmple(self.end).all())
                    .then(|| {
                        let mut neighbor: Self::Vertex = vertex.clone();

                        neighbor.push(dir);

                        OpenSetElement(neighbor, 1_usize)
                    })
                }));
            }
        }
    }

    fn update_vertex(
        &mut self,
        _from: &Self::Vertex,
        to: &Self::Vertex,
        _cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.visited.insert(to.clone());
    }

    fn reset(&mut self) {
        self.visited.clear();
        self.visited.insert(self.start_dir_path.clone());
    }
}

#[derive(Clone, Copy)]
struct PathElement {
    pos: IVec2,
    is_door: [bool; Direction::COUNT],
    is_new: bool,
}

impl PathElement {
    fn new(is_door: [bool; Direction::COUNT], pos: IVec2, start: IVec2, end: IVec2) -> Self {
        let mut is_door: [bool; Direction::COUNT] = is_door;

        for dir in Direction::iter() {
            let neighbor_pos: IVec2 = pos + dir.vec();

            is_door[dir as usize] = is_door[dir as usize]
                && neighbor_pos.cmpge(start).all()
                && neighbor_pos.cmple(end).all();
        }

        Self {
            is_new: true,
            pos,
            is_door,
        }
    }

    fn get_next_door(&self) -> Option<Direction> {
        self.is_door
            .iter()
            .position(|is_door| *is_door)
            .map(|index| (index as u8).into())
    }
}

struct LongestPathFinder {
    start: IVec2,
    end: IVec2,
    passcode_len: usize,
    hash_buffer: String,
    longest_path_len: usize,
    path: Vec<PathElement>,
}

impl LongestPathFinder {
    fn new(start: IVec2, end: IVec2, passcode: &str) -> Self {
        Self {
            start,
            end,
            passcode_len: passcode.len(),
            hash_buffer: passcode.into(),
            longest_path_len: 0_usize,
            path: Vec::new(),
        }
    }

    fn new_path_element(&mut self, dir: Option<Direction>, prev_pos: IVec2) -> PathElement {
        let truncate_len: usize = self.hash_buffer.len();
        let is_door: [bool; Direction::COUNT] =
            compute_is_door(dir.into_iter(), &mut self.hash_buffer, truncate_len);

        PathElement::new(
            is_door,
            prev_pos + dir.map(Direction::vec).unwrap_or_default(),
            self.start,
            self.end,
        )
    }

    fn pop(&mut self) {
        self.hash_buffer.truncate(
            self.passcode_len
                .max(self.hash_buffer.len().saturating_sub(1_usize)),
        );
        self.path.pop();
    }

    fn run(&mut self) -> Option<usize> {
        self.hash_buffer.truncate(self.passcode_len);
        self.path.clear();

        if self.start == self.end {
            Some(0_usize)
        } else {
            let path_element: PathElement = self.new_path_element(None, self.start);

            self.path.push(path_element);

            while !self.path.is_empty() {
                let last_path_element: PathElement = *self.path.last().unwrap();

                if last_path_element.is_new {
                    if last_path_element.pos == self.end {
                        self.longest_path_len = self
                            .longest_path_len
                            .max(self.hash_buffer.len() - self.passcode_len);
                        self.pop();
                    } else {
                        self.path.last_mut().unwrap().is_new = false;
                    }
                } else if let Some(dir) = last_path_element.get_next_door() {
                    let prev_pos: IVec2 = last_path_element.pos;

                    self.path.last_mut().unwrap().is_door[dir as usize] = false;

                    let path_element: PathElement = self.new_path_element(Some(dir), prev_pos);

                    self.path.push(path_element);
                } else {
                    self.pop();
                }
            }

            (self.longest_path_len != 0_usize).then_some(self.longest_path_len)
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(String);

impl Solution {
    const START: IVec2 = IVec2::ZERO;
    const END: IVec2 = IVec2::new(3_i32, 3_i32);

    fn try_shortest_path_to_end(&self, start: IVec2, end: IVec2) -> Option<DirPath> {
        let mut path_finder: ShortestPathFinder = ShortestPathFinder {
            start_dir_path: Default::default(),
            start,
            end,
            passcode_len: self.0.len(),
            hash_buffer: RefCell::new(self.0.clone()),
            visited: HashSet::new(),
        };

        path_finder
            .run_a_star()
            .map(|path| path.last().unwrap().clone())
    }

    fn try_shortest_path_to_end_string(&self, start: IVec2, end: IVec2) -> Option<String> {
        self.try_shortest_path_to_end(start, end)
            .map(|dir_path| dir_path.iter_chars().collect())
    }

    fn try_longest_path_to_end_len(&self, start: IVec2, end: IVec2) -> Option<usize> {
        LongestPathFinder::new(start, end, &self.0).run()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(map(not_line_ending, String::from), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_shortest_path_to_end_string(Self::START, Self::END));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_longest_path_to_end_len(Self::START, Self::END));
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

    const SOLUTION_STRS: &'static [&'static str] = &["ihgpwlah", "kglvqrro", "ulqzkmiv"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution("ihgpwlah".into()),
                Solution("kglvqrro".into()),
                Solution("ulqzkmiv".into()),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.into_iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_compute_is_door() {
        use Direction::*;

        let mut hash_buffer: String = "hijkl".into();
        let passcode_len: usize = hash_buffer.len();

        assert_eq!(
            DirPath::<1_usize>::from_iter([]).compute_is_door(&mut hash_buffer, passcode_len),
            [true, false, true, true]
        );
        assert_eq!(
            DirPath::<1_usize>::from_iter([South]).compute_is_door(&mut hash_buffer, passcode_len),
            [true, true, false, true]
        );
        assert_eq!(
            DirPath::<1_usize>::from_iter([South, East])
                .compute_is_door(&mut hash_buffer, passcode_len),
            [false, false, false, false]
        );
        assert_eq!(
            DirPath::<1_usize>::from_iter([South, North])
                .compute_is_door(&mut hash_buffer, passcode_len),
            [false, true, false, false]
        );
        assert_eq!(
            DirPath::<1_usize>::from_iter([South, North, East])
                .compute_is_door(&mut hash_buffer, passcode_len),
            [false, false, false, false]
        );
    }

    #[test]
    fn test_try_shortest_path_to_end_string() {
        for (passcode, path) in [
            ("ihgpwlah", "DDRRRD"),
            ("kglvqrro", "DDUDRLRRUDRD"),
            ("ulqzkmiv", "DRURDRUDDLLDLUURRDULRLDUUDDDRR"),
        ] {
            let solution: Solution = passcode.try_into().unwrap();

            assert_eq!(
                solution.try_shortest_path_to_end_string(Solution::START, Solution::END),
                Some(path.to_owned())
            );
        }
    }

    #[test]
    fn test_try_longest_path_to_end_len() {
        for (passcode, len) in [
            ("hijkl", None),
            ("ihgpwlah", Some(370_usize)),
            ("kglvqrro", Some(492_usize)),
            ("ulqzkmiv", Some(830_usize)),
        ] {
            let solution: Solution = passcode.try_into().unwrap();

            assert_eq!(
                solution.try_longest_path_to_end_len(Solution::START, Solution::END),
                len
            );
        }
    }
}
