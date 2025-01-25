use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    num::PrimInt,
    std::{collections::HashMap, mem::swap},
};

/* --- Day 18: Settlers of The North Pole ---

On the outskirts of the North Pole base construction project, many Elves are collecting lumber.

The lumber collection area is 50 acres by 50 acres; each acre can be either open ground (.), trees (|), or a lumberyard (#). You take a scan of the area (your puzzle input).

Strange magic is at work here: each minute, the landscape looks entirely different. In exactly one minute, an open acre can fill with trees, a wooded acre can be converted to a lumberyard, or a lumberyard can be cleared to open ground (the lumber having been sent to other projects).

The change to each acre is based entirely on the contents of that acre as well as the number of open, wooded, or lumberyard acres adjacent to it at the start of each minute. Here, "adjacent" means any of the eight acres surrounding that acre. (Acres on the edges of the lumber collection area might have fewer than eight adjacent acres; the missing acres aren't counted.)

In particular:

    An open acre will become filled with trees if three or more adjacent acres contained trees. Otherwise, nothing happens.
    An acre filled with trees will become a lumberyard if three or more adjacent acres were lumberyards. Otherwise, nothing happens.
    An acre containing a lumberyard will remain a lumberyard if it was adjacent to at least one other lumberyard and at least one acre containing trees. Otherwise, it becomes open.

These changes happen across all acres simultaneously, each of them using the state of all acres at the beginning of the minute and changing to their new form by the end of that same minute. Changes that happen during the minute don't affect each other.

For example, suppose the lumber collection area is instead only 10 by 10 acres with this initial configuration:

Initial state:
.#.#...|#.
.....#|##|
.|..|...#.
..|#.....#
#.#|||#|#|
...#.||...
.|....|...
||...#|.#|
|.||||..|.
...#.|..|.

After 1 minute:
.......##.
......|###
.|..|...#.
..|#||...#
..##||.|#|
...#||||..
||...|||..
|||||.||.|
||||||||||
....||..|.

After 2 minutes:
.......#..
......|#..
.|.|||....
..##|||..#
..###|||#|
...#|||||.
|||||||||.
||||||||||
||||||||||
.|||||||||

After 3 minutes:
.......#..
....|||#..
.|.||||...
..###|||.#
...##|||#|
.||##|||||
||||||||||
||||||||||
||||||||||
||||||||||

After 4 minutes:
.....|.#..
...||||#..
.|.#||||..
..###||||#
...###||#|
|||##|||||
||||||||||
||||||||||
||||||||||
||||||||||

After 5 minutes:
....|||#..
...||||#..
.|.##||||.
..####|||#
.|.###||#|
|||###||||
||||||||||
||||||||||
||||||||||
||||||||||

After 6 minutes:
...||||#..
...||||#..
.|.###|||.
..#.##|||#
|||#.##|#|
|||###||||
||||#|||||
||||||||||
||||||||||
||||||||||

After 7 minutes:
...||||#..
..||#|##..
.|.####||.
||#..##||#
||##.##|#|
|||####|||
|||###||||
||||||||||
||||||||||
||||||||||

After 8 minutes:
..||||##..
..|#####..
|||#####|.
||#...##|#
||##..###|
||##.###||
|||####|||
||||#|||||
||||||||||
||||||||||

After 9 minutes:
..||###...
.||#####..
||##...##.
||#....###
|##....##|
||##..###|
||######||
|||###||||
||||||||||
||||||||||

After 10 minutes:
.||##.....
||###.....
||##......
|##.....##
|##.....##
|##....##|
||##.####|
||#####|||
||||#|||||
||||||||||

After 10 minutes, there are 37 wooded acres and 31 lumberyards. Multiplying the number of wooded acres by the number of lumberyards gives the total resource value after ten minutes: 37 * 31 = 1147.

What will the total resource value of the lumber collection area be after 10 minutes?

--- Part Two ---

This important natural resource will need to last for at least thousands of years. Are the Elves collecting this lumber sustainably?

What will the total resource value of the lumber collection area be after 1000000000 minutes? */

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, Eq, Hash, PartialEq)]
    enum Acre {
        Open = OPEN = b'.',
        Trees = TREES = b'|',
        Lumberyard = LUMBERYARD = b'#',
    }
}

#[derive(Clone, Copy, Default)]
struct AcreCounts<I: PrimInt + Default> {
    open: I,
    trees: I,
    lumberyard: I,
}

impl<I: PrimInt + Default> FromIterator<Acre> for AcreCounts<I> {
    fn from_iter<T: IntoIterator<Item = Acre>>(iter: T) -> Self {
        iter.into_iter()
            .fold(Self::default(), |mut neighbor_counts, acre| {
                match acre {
                    Acre::Open => neighbor_counts.open = neighbor_counts.open + I::one(),
                    Acre::Trees => neighbor_counts.trees = neighbor_counts.trees + I::one(),
                    Acre::Lumberyard => {
                        neighbor_counts.lumberyard = neighbor_counts.lumberyard + I::one()
                    }
                }

                neighbor_counts
            })
    }
}

#[derive(Clone)]
struct LumberCollectionArea {
    curr_acres: Grid2D<Acre>,
    next_acres: Grid2D<Acre>,
    minutes: usize,
    hash_to_minutes: HashMap<u64, usize>,
}

impl LumberCollectionArea {
    const SMALL_MINUTES: usize = 10_usize;
    const LARGE_MINUTES: usize = 1_000_000_000_usize;

    fn iter_neighbors(&self, pos: IVec2) -> impl Iterator<Item = Acre> + '_ {
        let pos_minus_one: IVec2 = pos - IVec2::ONE;
        let pos_plus_one: IVec2 = pos + IVec2::ONE;
        let min: IVec2 = pos_minus_one.max(IVec2::ZERO);
        let max: IVec2 = pos_plus_one.min(self.curr_acres.max_dimensions());
        let x_len: usize = (max.x - min.x + 1_i32) as usize;

        (pos_minus_one.y == min.y)
            .then(|| {
                let start: usize = self.curr_acres.index_from_pos(min);

                self.curr_acres.cells()[start..start + x_len]
                    .iter()
                    .copied()
            })
            .into_iter()
            .flatten()
            .chain(
                (min.x < pos.x)
                    .then(|| *self.curr_acres.get(IVec2::new(min.x, pos.y)).unwrap())
                    .into_iter(),
            )
            .chain(
                (max.x > pos.x)
                    .then(|| *self.curr_acres.get(IVec2::new(max.x, pos.y)).unwrap())
                    .into_iter(),
            )
            .chain(
                (pos_plus_one.y == max.y)
                    .then(|| {
                        let start: usize = self.curr_acres.index_from_pos(IVec2::new(min.x, max.y));

                        self.curr_acres.cells()[start..start + x_len]
                            .iter()
                            .copied()
                    })
                    .into_iter()
                    .flatten(),
            )
    }

    fn try_update_pos(&self, acre: Acre, pos: IVec2) -> Option<Acre> {
        let acre_counts: AcreCounts<u8> = self.iter_neighbors(pos).collect();

        match acre {
            Acre::Open => (acre_counts.trees >= 3_u8).then_some(Acre::Trees),
            Acre::Trees => (acre_counts.lumberyard >= 3_u8).then_some(Acre::Lumberyard),
            Acre::Lumberyard => {
                (acre_counts.trees == 0_u8 || acre_counts.lumberyard == 0_u8).then_some(Acre::Open)
            }
        }
    }

    fn grid_string(&self) -> String {
        self.curr_acres.clone().into()
    }

    fn self_after_minutes(&self, minutes: usize) -> Self {
        let mut solution: Self = self.clone();

        solution.update(minutes);

        solution
    }

    fn total_resource_value(&self) -> usize {
        let acre_counts: AcreCounts<usize> = self.curr_acres.cells().iter().copied().collect();

        acre_counts.trees * acre_counts.lumberyard
    }

    fn total_resource_value_after_minutes(&self, minutes: usize) -> usize {
        self.self_after_minutes(minutes).total_resource_value()
    }

    fn update_internal(&mut self) {
        self.next_acres
            .cells_mut()
            .copy_from_slice(self.curr_acres.cells());

        for (pos, &acre) in self.curr_acres.iter_positions_and_cells() {
            if let Some(acre) = self.try_update_pos(acre, pos) {
                *self.next_acres.get_mut(pos).unwrap() = acre;
            }
        }

        swap(&mut self.curr_acres, &mut self.next_acres);

        self.minutes += 1_usize;
    }

    fn update(&mut self, minutes: usize) {
        let mut minutes: usize = minutes;

        while minutes > 0_usize {
            self.update_internal();

            minutes -= 1_usize;

            let hash: u64 = self.curr_acres.cells().compute_hash();

            if let Some(&cycle_start_minutes) = self.hash_to_minutes.get(&hash) {
                let cycle_end_minutes: usize = self.minutes;
                let period: usize = cycle_end_minutes - cycle_start_minutes;
                let (remaining_periods, remaining): (usize, usize) =
                    (minutes / period, minutes % period);

                minutes = remaining;
                self.minutes += remaining_periods * period;
            } else {
                self.hash_to_minutes.insert(hash, self.minutes);
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution<const SIZE: i32 = 50_i32>(Grid2D<Acre>);

impl<const SIZE: i32> Solution<SIZE> {
    fn lumber_collection_area(&self) -> LumberCollectionArea {
        LumberCollectionArea {
            curr_acres: self.0.clone(),
            next_acres: self.0.clone(),
            minutes: 0_usize,
            hash_to_minutes: [(self.0.cells().compute_hash(), 0_usize)]
                .into_iter()
                .collect(),
        }
    }

    fn lumber_collection_area_after_minutes(&self, minutes: usize) -> LumberCollectionArea {
        let mut lumber_collection_area: LumberCollectionArea = self.lumber_collection_area();

        lumber_collection_area.update(minutes);

        lumber_collection_area
    }

    fn total_resource_value_after_minutes(&self, minutes: usize) -> usize {
        self.lumber_collection_area()
            .total_resource_value_after_minutes(minutes)
    }
}

impl<const SIZE: i32> Parse for Solution<SIZE> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |grid| {
            let dimensions: IVec2 = grid.dimensions();

            (dimensions == SIZE * IVec2::ONE).then(|| Self(grid))
        })(input)
    }
}

impl RunQuestions for Solution {
    /// I want to hook this up to a timer to let it run instead of just all running behind the
    /// scenes, but I'm not sure if the paths of doing so (just printing each time, using special
    /// characters to move the cursor back to the start each time, using something ncurses-like to
    /// handle it for me, or using a straight up TUI crate) will achieve the desired effect well
    /// enough w/o requiring too much effort.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let lumber_collection_area: LumberCollectionArea =
                self.lumber_collection_area_after_minutes(LumberCollectionArea::SMALL_MINUTES);

            dbg!(lumber_collection_area.total_resource_value());
            println!("{}", lumber_collection_area.grid_string());
        } else {
            dbg!(self.total_resource_value_after_minutes(LumberCollectionArea::SMALL_MINUTES));
        }
    }

    /// when i see big number, i cache. it's what i do.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let lumber_collection_area: LumberCollectionArea =
                self.lumber_collection_area_after_minutes(LumberCollectionArea::LARGE_MINUTES);

            dbg!(lumber_collection_area.total_resource_value());
            println!("{}", lumber_collection_area.grid_string());
        } else {
            dbg!(self.total_resource_value_after_minutes(LumberCollectionArea::LARGE_MINUTES));
        }
    }
}

impl<'i, const SIZE: i32> TryFrom<&'i str> for Solution<SIZE> {
    type Error = Err<Error<&'i str>>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self::parse(input)?.1)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SIZE: i32 = 10_i32;
    const SOLUTION_STRS: &'static [&'static str] = &["\
        .#.#...|#.\n\
        .....#|##|\n\
        .|..|...#.\n\
        ..|#.....#\n\
        #.#|||#|#|\n\
        ...#.||...\n\
        .|....|...\n\
        ||...#|.#|\n\
        |.||||..|.\n\
        ...#.|..|.\n"];

    fn solution(index: usize) -> &'static Solution<SIZE> {
        use Acre::{Lumberyard as L, Open as O, Trees as T};

        static ONCE_LOCK: OnceLock<Vec<Solution<SIZE>>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                Grid2D::try_from_cells_and_dimensions(
                    vec![
                        O, L, O, L, O, O, O, T, L, O, O, O, O, O, O, L, T, L, L, T, O, T, O, O, T,
                        O, O, O, L, O, O, O, T, L, O, O, O, O, O, L, L, O, L, T, T, T, L, T, L, T,
                        O, O, O, L, O, T, T, O, O, O, O, T, O, O, O, O, T, O, O, O, T, T, O, O, O,
                        L, T, O, L, T, T, O, T, T, T, T, O, O, T, O, O, O, O, L, O, T, O, O, T, O,
                    ],
                    SIZE * IVec2::ONE,
                )
                .unwrap(),
            )]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(
                Solution::<SIZE>::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_update() {
        for (index, grid_strings) in [vec![
            "\
            .......##.\n\
            ......|###\n\
            .|..|...#.\n\
            ..|#||...#\n\
            ..##||.|#|\n\
            ...#||||..\n\
            ||...|||..\n\
            |||||.||.|\n\
            ||||||||||\n\
            ....||..|.\n",
            "\
            .......#..\n\
            ......|#..\n\
            .|.|||....\n\
            ..##|||..#\n\
            ..###|||#|\n\
            ...#|||||.\n\
            |||||||||.\n\
            ||||||||||\n\
            ||||||||||\n\
            .|||||||||\n",
            "\
            .......#..\n\
            ....|||#..\n\
            .|.||||...\n\
            ..###|||.#\n\
            ...##|||#|\n\
            .||##|||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            .....|.#..\n\
            ...||||#..\n\
            .|.#||||..\n\
            ..###||||#\n\
            ...###||#|\n\
            |||##|||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            ....|||#..\n\
            ...||||#..\n\
            .|.##||||.\n\
            ..####|||#\n\
            .|.###||#|\n\
            |||###||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            ...||||#..\n\
            ...||||#..\n\
            .|.###|||.\n\
            ..#.##|||#\n\
            |||#.##|#|\n\
            |||###||||\n\
            ||||#|||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            ...||||#..\n\
            ..||#|##..\n\
            .|.####||.\n\
            ||#..##||#\n\
            ||##.##|#|\n\
            |||####|||\n\
            |||###||||\n\
            ||||||||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            ..||||##..\n\
            ..|#####..\n\
            |||#####|.\n\
            ||#...##|#\n\
            ||##..###|\n\
            ||##.###||\n\
            |||####|||\n\
            ||||#|||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            ..||###...\n\
            .||#####..\n\
            ||##...##.\n\
            ||#....###\n\
            |##....##|\n\
            ||##..###|\n\
            ||######||\n\
            |||###||||\n\
            ||||||||||\n\
            ||||||||||\n",
            "\
            .||##.....\n\
            ||###.....\n\
            ||##......\n\
            |##.....##\n\
            |##.....##\n\
            |##....##|\n\
            ||##.####|\n\
            ||#####|||\n\
            ||||#|||||\n\
            ||||||||||\n",
        ]]
        .into_iter()
        .enumerate()
        {
            let mut lumber_collection_area: LumberCollectionArea =
                solution(index).lumber_collection_area();

            for grid_string in grid_strings {
                lumber_collection_area.update(1_usize);

                assert_eq!(lumber_collection_area.grid_string(), grid_string);
            }
        }
    }

    #[test]
    fn test_total_resource_value_after_minutes() {
        for (index, total_resource_value_after_minutes) in [1147_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index)
                    .total_resource_value_after_minutes(LumberCollectionArea::SMALL_MINUTES),
                total_resource_value_after_minutes
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
