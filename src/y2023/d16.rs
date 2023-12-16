use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
    std::{
        cell::{Ref, RefCell, RefMut},
        collections::{HashSet, VecDeque},
        ops::Range,
    },
};

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Tile {
        EmptySpace = EMPTY_SPACE = b'.',
        UpMirror = UP_MIRROR = b'/',
        DownMirror = DOWN_MIRROR = b'\\',
        VerticalMirror = VERTICAL_MIRROR = b'|',
        HorizontalMirror = HORIZONTAL_MIRROR = b'-',
    }
}

impl Tile {
    fn interact_with_beam(self, dir: Direction) -> (Direction, Option<Direction>) {
        match self {
            Tile::EmptySpace => (dir, None),
            Tile::UpMirror => {
                // Direction::North => Direction::East
                // Direction::East => Direction::North
                // Direction::South => Direction::West
                // Direction::West => Direction::South
                (Direction::from_u8(dir as u8 ^ 1_u8), None)
            }
            Tile::DownMirror => {
                // Direction::North => Direction::West
                // Direction::East => Direction::South
                // Direction::South => Direction::East
                // Direction::West => Direction::North
                (Direction::from_u8(!(dir as u8)), None)
            }
            Tile::VerticalMirror => match dir {
                Direction::North | Direction::South => (dir, None),
                Direction::East | Direction::West => (dir.next(), Some(dir.prev())),
            },
            Tile::HorizontalMirror => match dir {
                Direction::East | Direction::West => (dir, None),
                Direction::North | Direction::South => (dir.next(), Some(dir.prev())),
            },
        }
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Eq, Hash, PartialEq)]
struct Beam {
    pos: IVec2,
    dir: Direction,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct EnergizedGrid(Grid2D<Pixel>);

impl EnergizedGrid {
    fn energized_tile_count(&self) -> usize {
        self.0
            .cells()
            .iter()
            .copied()
            .filter(|pixel| pixel.is_light())
            .count()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct SolutionData {
    beams: HashSet<Beam>,
    new_beams: VecDeque<Beam>,
    energized_grid: EnergizedGrid,
}

impl SolutionData {
    fn new(dimensions: IVec2) -> Self {
        Self {
            beams: HashSet::new(),
            new_beams: VecDeque::new(),
            energized_grid: EnergizedGrid(Grid2D::default(dimensions)),
        }
    }

    fn initialize(&mut self, initial_beam: Beam) {
        self.beams.clear();
        self.beams.insert(initial_beam.clone());
        self.new_beams.clear();
        self.new_beams.push_back(initial_beam);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Tile>,
    data: RefCell<SolutionData>,
}

impl Solution {
    const INITIAL_BEAM: Beam = Beam {
        pos: IVec2::ZERO,
        dir: Direction::East,
    };

    fn add_beam(&self, old_pos: IVec2, dir: Direction) {
        let beam: Beam = Beam {
            pos: old_pos + dir.vec(),
            dir,
        };

        let mut solution_data: RefMut<SolutionData> = self.data.borrow_mut();

        if self.grid.contains(beam.pos) && solution_data.beams.insert(beam.clone()) {
            solution_data.new_beams.push_back(beam);
        }
    }

    fn initialize(&self, initial_beam: Beam) {
        self.data.borrow_mut().initialize(initial_beam);
    }

    fn pop_beam(&self) -> Option<Beam> {
        self.data.borrow_mut().new_beams.pop_front()
    }

    fn disperse_beam(&self, initial_beam: Beam) -> Ref<HashSet<Beam>> {
        self.initialize(initial_beam);

        while let Some(beam) = self.pop_beam() {
            let (new_dir_a, new_dir_b): (Direction, Option<Direction>) = self
                .grid
                .get(beam.pos)
                .unwrap()
                .interact_with_beam(beam.dir);

            self.add_beam(beam.pos, new_dir_a);

            if let Some(new_dir_b) = new_dir_b {
                self.add_beam(beam.pos, new_dir_b);
            }
        }

        Ref::map(self.data.borrow(), |data| &data.beams)
    }

    fn energized_grid(&self, initial_beam: Beam) -> Ref<EnergizedGrid> {
        self.disperse_beam(initial_beam);

        {
            let mut solution_data: RefMut<SolutionData> = self.data.borrow_mut();
            let solution_data: &mut SolutionData = &mut *solution_data;

            solution_data.energized_grid.0.cells_mut().fill(Pixel::Dark);

            for beam in solution_data.beams.iter() {
                *solution_data.energized_grid.0.get_mut(beam.pos).unwrap() = Pixel::Light;
            }
        }

        Ref::map(self.data.borrow(), |data| &data.energized_grid)
    }

    fn energized_tile_count(&self) -> usize {
        self.energized_grid(Self::INITIAL_BEAM)
            .energized_tile_count()
    }

    fn corner(&self, index: usize) -> IVec2 {
        match index & 3_usize {
            0_usize => IVec2::new(self.grid.max_dimensions().x, 0_i32),
            1_usize => IVec2::ZERO,
            2_usize => IVec2::new(0_i32, self.grid.max_dimensions().y),
            3_usize => self.grid.max_dimensions(),
            _ => unimplemented!(),
        }
    }

    fn iter_initial_beams(&self) -> impl Iterator<Item = Beam> + '_ {
        (0_usize..4_usize).flat_map(|index| {
            let start: IVec2 = self.corner(index);
            let end: IVec2 = self.corner(index + 1_usize);
            let range: Range<IVec2> = start..end;
            let dir: Direction = Direction::try_from(range.clone()).unwrap().prev();

            CellIter2D::try_from(range)
                .unwrap()
                .map(move |pos| Beam { pos, dir })
        })
    }

    fn maximally_energized_grid(&self) -> EnergizedGrid {
        self.iter_initial_beams()
            .max_by_key(|initial_beam| {
                self.energized_grid(initial_beam.clone())
                    .energized_tile_count()
            })
            .map(|initial_beam| self.energized_grid(initial_beam).clone())
            .unwrap()
    }

    fn maximally_energized_tile_count(&self) -> usize {
        self.iter_initial_beams()
            .map(|initial_beam| self.energized_grid(initial_beam).energized_tile_count())
            .max()
            .unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::<Tile>::parse, |grid| {
            let dimensions: IVec2 = grid.dimensions();

            Self {
                grid,
                data: RefCell::new(SolutionData::new(dimensions)),
            }
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let energized_grid: Ref<EnergizedGrid> = self.energized_grid(Solution::INITIAL_BEAM);

            dbg!(energized_grid.energized_tile_count());

            println!("\n{}", String::from(energized_grid.0.clone()));
        } else {
            dbg!(self.energized_tile_count());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let energized_grid: EnergizedGrid = self.maximally_energized_grid();

            dbg!(energized_grid.energized_tile_count());

            println!("\n{}", String::from(energized_grid.0.clone()));
        } else {
            dbg!(self.maximally_energized_tile_count());
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
    use {
        super::*,
        std::sync::{Mutex, MutexGuard, OnceLock},
    };

    const SOLUTION_STR: &'static str = "\
        .|...\\....\n\
        |.-.\\.....\n\
        .....|-...\n\
        ........|.\n\
        ..........\n\
        .........\\\n\
        ..../.\\\\..\n\
        .-.-/..|..\n\
        .|....-|.\\\n\
        ..//.|....\n";
    const ENERGIZED_GRID_STR: &'static str = "\
        ######....\n\
        .#...#....\n\
        .#...#####\n\
        .#...##...\n\
        .#...##...\n\
        .#...##...\n\
        .#..####..\n\
        ########..\n\
        .#######..\n\
        .#...#.#..\n";

    fn solution() -> MutexGuard<'static, Solution> {
        use Tile::{
            DownMirror as D, EmptySpace as E, HorizontalMirror as H, UpMirror as U,
            VerticalMirror as V,
        };

        static ONCE_LOCK: OnceLock<Mutex<Solution>> = OnceLock::new();

        ONCE_LOCK
            .get_or_init(|| {
                Mutex::new(Solution {
                    grid: Grid2D::try_from_cells_and_width(
                        vec![
                            E, V, E, E, E, D, E, E, E, E, V, E, H, E, D, E, E, E, E, E, E, E, E, E,
                            E, V, H, E, E, E, E, E, E, E, E, E, E, E, V, E, E, E, E, E, E, E, E, E,
                            E, E, E, E, E, E, E, E, E, E, E, D, E, E, E, E, U, E, D, D, E, E, E, H,
                            E, H, U, E, E, V, E, E, E, V, E, E, E, E, H, V, E, D, E, E, U, U, E, V,
                            E, E, E, E,
                        ],
                        10_usize,
                    )
                    .unwrap(),
                    data: RefCell::new(SolutionData::new(10_i32 * IVec2::ONE)),
                })
            })
            .lock()
            .unwrap()
    }

    #[test]
    fn test_try_from_str() {
        let solution: MutexGuard<Solution> = solution();

        assert_eq!(
            Solution::try_from(SOLUTION_STR)
                .as_ref()
                .map(|solution| &solution.grid),
            Ok(&*solution).map(|solution| &solution.grid)
        );
    }

    #[test]
    fn test_energized_grid() {
        let solution: MutexGuard<Solution> = solution();

        assert_eq!(
            &*solution.energized_grid(Solution::INITIAL_BEAM),
            &EnergizedGrid(Grid2D::try_from(ENERGIZED_GRID_STR).unwrap())
        );
    }

    #[test]
    fn test_energized_tile_count() {
        let solution: MutexGuard<Solution> = solution();

        assert_eq!(solution.energized_tile_count(), 46_usize);
    }

    #[test]
    fn test_maximally_energized_tile_count() {
        let solution: MutexGuard<Solution> = solution();

        assert_eq!(solution.maximally_energized_tile_count(), 51_usize);
    }
}
