use {
    crate::{
        y2017::d10::{DenseHash, Solution as D10Solution},
        *,
    },
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        character::complete::not_line_ending,
        combinator::{map_res, success},
        error::Error,
        Err, IResult,
    },
    static_assertions::const_assert_eq,
    std::{
        alloc::{alloc_zeroed, Layout},
        fmt::Write,
        mem::{size_of, transmute},
    },
    strum::IntoEnumIterator,
};

/* --- Day 14: Disk Defragmentation ---

Suddenly, a scheduled job activates the system's disk defragmenter. Were the situation different, you might sit and watch it for a while, but today, you just don't have that kind of time. It's soaking up valuable system resources that are needed elsewhere, and so the only option is to help it finish its task as soon as possible.

The disk in question consists of a 128x128 grid; each square of the grid is either free or used. On this disk, the state of the grid is tracked by the bits in a sequence of knot hashes.

A total of 128 knot hashes are calculated, each corresponding to a single row in the grid; each hash contains 128 bits which correspond to individual grid squares. Each bit of a hash indicates whether that square is free (0) or used (1).

The hash inputs are a key string (your puzzle input), a dash, and a number from 0 to 127 corresponding to the row. For example, if your key string were flqrgnkx, then the first row would be given by the bits of the knot hash of flqrgnkx-0, the second row from the bits of the knot hash of flqrgnkx-1, and so on until the last row, flqrgnkx-127.

The output of a knot hash is traditionally represented by 32 hexadecimal digits; each of these digits correspond to 4 bits, for a total of 4 * 32 = 128 bits. To convert to bits, turn each hexadecimal digit to its equivalent binary value, high-bit first: 0 becomes 0000, 1 becomes 0001, e becomes 1110, f becomes 1111, and so on; a hash that begins with a0c2017... in hexadecimal would begin with 10100000110000100000000101110000... in binary.

Continuing this process, the first 8 rows and columns for key flqrgnkx appear as follows, using # to denote used squares, and . to denote free ones:

##.#.#..-->
.#.#.#.#
....#.#.
#.#.##.#
.##.#...
##..#..#
.#...#..
##.#.##.-->
|      |
V      V

In this example, 8108 squares are used across the entire 128x128 grid.

Given your actual key string, how many squares are used?

--- Part Two ---

Now, all the defragmenter needs to know is the number of regions. A region is a group of used squares that are all adjacent, not including diagonals. Every used square is in exactly one region: lone used squares form their own isolated regions, while several adjacent squares all count as a single region.

In the example above, the following nine regions are visible, each marked with a distinct digit:

11.2.3..-->
.1.2.3.4
....5.6.
7.8.55.9
.88.5...
88..5..8
.8...8..
88.8.88.-->
|      |
V      V

Of particular interest is the region marked 8; while it does not appear contiguous in this small view, all of the squares marked 8 are connected when considering the whole 128x128 grid. In total, in this example, 1242 regions are present.

How many regions are present given your key string? */

type DenseHashArray = [DenseHash; Solution::ROWS];

trait FindRegions
where
    Self: Sized,
{
    fn next_start(&self) -> Option<IVec2>;
    fn is_valid(&self, pos: IVec2) -> bool;
    fn is_used(&self, pos: IVec2) -> bool;
    fn set_curr_region(&mut self, region: usize);
    fn add_pos_to_curr_region(&mut self, pos: IVec2);

    fn find_regions(&mut self) -> usize {
        let mut find_regions_state: FindRegionsState<Self> = FindRegionsState {
            find_regions: self,
            start: IVec2::ZERO,
        };
        let mut region: usize = 0_usize;

        while let Some(start) = find_regions_state.find_regions.next_start() {
            find_regions_state.start = start;
            find_regions_state.find_regions.set_curr_region(region);
            find_regions_state
                .find_regions
                .add_pos_to_curr_region(start);
            find_regions_state.run();
            region += 1_usize;
        }

        region
    }
}

struct FindRegionsState<'f, F: FindRegions + ?Sized>
where
    Self: Sized,
{
    find_regions: &'f mut F,
    start: IVec2,
}

impl<'f, F: FindRegions> BreadthFirstSearch for FindRegionsState<'f, F> {
    type Vertex = IVec2;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unimplemented!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(Direction::iter().filter_map(|dir| {
            let pos: IVec2 = *vertex + dir.vec();

            (self.find_regions.is_valid(pos) && self.find_regions.is_used(pos)).then_some(pos)
        }));
    }

    fn update_parent(&mut self, _from: &Self::Vertex, to: &Self::Vertex) {
        self.find_regions.add_pos_to_curr_region(*to);
    }

    fn reset(&mut self) {}
}

struct RegionCounter<'s> {
    solution: &'s Solution,
    used_but_not_visited: BitVec,
}

impl<'s> RegionCounter<'s> {
    fn set(&mut self, pos: IVec2, value: bool) {
        self.used_but_not_visited
            .set((pos.x + pos.y * Solution::DIMENSIONS.x) as usize, value);
    }
}

impl<'s> FindRegions for RegionCounter<'s> {
    fn next_start(&self) -> Option<IVec2> {
        self.used_but_not_visited.iter_ones().next().map(|index| {
            IVec2::new(
                index as i32 % Solution::DIMENSIONS.x,
                index as i32 / Solution::DIMENSIONS.x,
            )
        })
    }

    fn is_valid(&self, pos: IVec2) -> bool {
        Solution::is_valid(pos)
    }

    fn is_used(&self, pos: IVec2) -> bool {
        self.solution.is_used(pos)
    }

    fn set_curr_region(&mut self, _region: usize) {}

    fn add_pos_to_curr_region(&mut self, pos: IVec2) {
        self.set(pos, false);
    }
}

impl<'s> From<&'s Solution> for RegionCounter<'s> {
    fn from(solution: &'s Solution) -> Self {
        let mut region_counter: Self = Self {
            solution,
            used_but_not_visited: bitvec![0; (Solution::DIMENSIONS.x * Solution::DIMENSIONS.y) as usize],
        };

        for (y, dense_hash) in solution.0.iter().enumerate() {
            for x in dense_hash.as_bits::<Lsb0>().iter_ones() {
                region_counter.set(IVec2::new(x as i32, y as i32), true);
            }
        }

        region_counter
    }
}

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Default, PartialEq)]
    enum Square {
        #[default]
        Free = FREE = b'.',
        Used = USED = b'#',
        Region0 = REGION_0 = b'0',
        Region1 = REGION_1 = b'1',
        Region2 = REGION_2 = b'2',
        Region3 = REGION_3 = b'3',
        Region4 = REGION_4 = b'4',
        Region5 = REGION_5 = b'5',
        Region6 = REGION_6 = b'6',
        Region7 = REGION_7 = b'7',
        Region8 = REGION_8 = b'8',
        Region9 = REGION_9 = b'9',
    }
}

impl From<usize> for Square {
    fn from(value: usize) -> Self {
        // SAFETY: We just made sure it's in the range 0..=9, which will map `value` into one of the
        // region variants.
        unsafe { transmute((value % 10_usize) as u8 + Self::REGION_0) }
    }
}

struct RegionFinder {
    grid: Grid2D<Square>,
    region: Square,
}

impl FindRegions for RegionFinder {
    fn next_start(&self) -> Option<IVec2> {
        self.grid
            .cells()
            .iter()
            .position(|square| *square == Square::Used)
            .map(|index| self.grid.pos_from_index(index))
    }

    fn is_valid(&self, pos: IVec2) -> bool {
        self.grid.contains(pos)
    }

    fn is_used(&self, pos: IVec2) -> bool {
        *self.grid.get(pos).unwrap() == Square::Used
    }

    fn set_curr_region(&mut self, region: usize) {
        self.region = region.into();
    }

    fn add_pos_to_curr_region(&mut self, pos: IVec2) {
        *self.grid.get_mut(pos).unwrap() = self.region;
    }
}

impl<'s> From<&'s Solution> for RegionFinder {
    fn from(value: &'s Solution) -> Self {
        let mut region_finder: Self = Self {
            grid: Grid2D::default(Solution::DIMENSIONS),
            region: Square::default(),
        };

        for (y, dense_hash) in value.0.iter().enumerate() {
            for x in dense_hash.as_bits::<Lsb0>().iter_ones() {
                *region_finder
                    .grid
                    .get_mut(IVec2::new(x as i32, y as i32))
                    .unwrap() = Square::Used;
            }
        }

        region_finder
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Box<DenseHashArray>);

impl Solution {
    const ROWS: usize = 128_usize;
    const DIMENSIONS: IVec2 = {
        const_assert_eq!(Solution::ROWS, size_of::<DenseHash>() * u8::BITS as usize);

        IVec2::new(Self::ROWS as i32, Self::ROWS as i32)
    };

    fn new() -> Self {
        Self(unsafe {
            // SAFETY: `DenseHashArray` is all `u8`s.
            Box::from_raw(alloc_zeroed(Layout::new::<DenseHashArray>()) as *mut DenseHashArray)
        })
    }

    fn is_valid(pos: IVec2) -> bool {
        pos.cmpge(IVec2::ZERO).all() && pos.cmplt(Self::DIMENSIONS).all()
    }

    fn is_used(&self, pos: IVec2) -> bool {
        assert!(Self::is_valid(pos));

        self.0[pos.y as usize].as_bits::<Lsb0>()[pos.x as usize]
    }

    fn count_used_squares(&self) -> usize {
        self.0
            .iter()
            .map(|dense_hash| dense_hash.as_bits::<Lsb0>().count_ones())
            .sum()
    }

    fn count_regions(&self) -> usize {
        let mut region_counter: RegionCounter = self.into();

        region_counter.find_regions()
    }

    fn find_regions(&self) -> (Grid2D<Square>, usize) {
        let mut region_finder: RegionFinder = self.into();

        let region_count: usize = region_finder.find_regions();

        (region_finder.grid, region_count)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, key): (&str, &str) = not_line_ending(input)?;

        let key_len_plus_one: usize = key.len() + 1_usize;
        let mut knot_hash_input: String =
            String::with_capacity(key_len_plus_one + digits(Self::ROWS as u32 - 1_u32));

        map_res(success(()), |_| write!(&mut knot_hash_input, "{}-", key))(input)?;

        let mut solution: Self = Self::new();

        for (index, dense_hash) in solution.0.iter_mut().enumerate() {
            knot_hash_input.truncate(key_len_plus_one);

            map_res(success(()), |_| write!(&mut knot_hash_input, "{}", index))(input)?;

            *dense_hash = D10Solution::knot_hash_for_str(&knot_hash_input);

            for byte in dense_hash {
                *byte = byte.reverse_bits();
            }
        }

        Ok((input, solution))
    }
}

impl RunQuestions for Solution {
    /// I feel like part 2 is probably going to be some sort of graph algorithm, as that's common
    /// for grid problems, but because we already have a bitvector representation of the data, it
    /// feels wasteful to expand that out unnecessarily.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_used_squares());
    }

    /// Two different implementations for verbose vs non-verbose, but they share a decent amount of
    /// code still through traits.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let (grid, region_count): (Grid2D<Square>, usize) = self.find_regions();

            println!("{}", String::from(grid));
            dbg!(region_count);
        } else {
            dbg!(self.count_regions());
        }
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

    const SOLUTION_STRS: &'static [&'static str] = &["flqrgnkx"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| vec![Solution::try_from(SOLUTION_STRS[0_usize]).unwrap()])[index]
    }

    #[test]
    fn test_try_from_str_and_get() {
        for (index, expected_grid_corner_string) in ["\
            ##.#.#..\
            .#.#.#.#\
            ....#.#.\
            #.#.##.#\
            .##.#...\
            ##..#..#\
            .#...#..\
            ##.#.##."]
        .into_iter()
        .enumerate()
        {
            let grid_corner_string: String = CellIter2D::try_from(IVec2::ZERO..8_i32 * IVec2::Y)
                .unwrap()
                .flat_map(|row_pos| {
                    CellIter2D::try_from(IVec2::ZERO..8_i32 * IVec2::X)
                        .unwrap()
                        .map(move |col_pos| IVec2::new(col_pos.x, row_pos.y))
                })
                .map(|pos| {
                    if solution(index).is_used(pos) {
                        '#'
                    } else {
                        '.'
                    }
                })
                .collect();

            assert_eq!(grid_corner_string, expected_grid_corner_string);
        }
    }

    #[test]
    fn test_count_used_squares() {
        for (index, count_used_squares) in [8108_usize].into_iter().enumerate() {
            assert_eq!(solution(index).count_used_squares(), count_used_squares);
        }
    }

    #[test]
    fn test_count_regions() {
        for (index, count_regions) in [1242_usize].into_iter().enumerate() {
            assert_eq!(solution(index).count_regions(), count_regions);
        }
    }

    #[test]
    fn test_find_regions() {
        for (index, find_regions) in [1242_usize].into_iter().enumerate() {
            assert_eq!(solution(index).find_regions().1, find_regions);
        }
    }
}
