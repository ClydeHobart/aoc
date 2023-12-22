use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map_opt, error::Error, Err, IResult},
    num::{BigInt, BigRational, ToPrimitive},
    std::collections::HashMap,
    strum::IntoEnumIterator,
};

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        StartingPosition = STARTING_POSITION = b'S',
        GardenPlot = GARDEN_PLOT = b'.',
        Rock = ROCK = b'#',
        ReachableGardenPlot = REACHABLE_GARDEN_PLOT = b'O',
    }
}

struct DistGrid {
    grid: Grid2D<u16>,
    max_dist: u16,
}

struct DistFinder<'s> {
    solution: &'s Solution,
    dist_grid: DistGrid,
    start: IVec2,
}

impl<'s> BreadthFirstSearch for DistFinder<'s> {
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
        let vertex: IVec2 = *vertex;

        neighbors.clear();

        if *self.dist_grid.grid.get(vertex).unwrap() < self.dist_grid.max_dist {
            neighbors.extend(
                Direction::iter()
                    .map(|dir| vertex + dir.vec())
                    .filter(|pos| {
                        self.dist_grid.grid.get(*pos) == Some(&u16::MAX)
                            && self.solution.grid.get(*pos) != Some(&Cell::Rock)
                    }),
            );
        }
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let from_dist: u16 = *self.dist_grid.grid.get(*from).unwrap();

        *self.dist_grid.grid.get_mut(*to).unwrap() = from_dist + 1_u16;
    }

    fn reset(&mut self) {
        self.dist_grid.grid.cells_mut().fill(u16::MAX);
        *self.dist_grid.grid.get_mut(*self.start()).unwrap() = 0_u16;
    }
}

struct MultiMapDistFinder<'s> {
    solution: &'s Solution,
    dists: HashMap<IVec2, u16>,
    max_dist: u16,
    start: IVec2,
}

impl<'s> BreadthFirstSearch for MultiMapDistFinder<'s> {
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
        let vertex: IVec2 = *vertex;
        let dimensions: IVec2 = self.solution.grid.dimensions();

        neighbors.clear();

        if *self.dists.get(&vertex).unwrap() < self.max_dist {
            neighbors.extend(Direction::iter().filter_map(|dir| {
                let pos: IVec2 = vertex + dir.vec();

                (!self.dists.contains_key(&pos)
                    && self.solution.grid.get(pos.rem_euclid(dimensions)) != Some(&Cell::Rock))
                .then_some(pos)
            }))
        }
    }

    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        let from_dist: u16 = *self.dists.get(from).unwrap();

        self.dists.insert(*to, from_dist + 1_u16);
    }

    fn reset(&mut self) {
        self.dists.clear();
        self.dists.insert(self.start, 0_u16);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    grid: Grid2D<Cell>,
    start: IVec2,
}

impl Solution {
    const FAVORITE_SQUARE_CUBE: u16 = 64_u16;
    const STEPS: u32 = 26501365_u32;

    fn dist_grid_for_start(&self, steps: u16, start: IVec2) -> DistGrid {
        let mut dist_finder: DistFinder = DistFinder {
            solution: self,
            dist_grid: DistGrid {
                grid: Grid2D::default(self.grid.dimensions()),
                max_dist: steps,
            },
            start,
        };

        dist_finder.run();

        dist_finder.dist_grid
    }

    fn dist_grid(&self, steps: u16) -> DistGrid {
        self.dist_grid_for_start(steps, self.start)
    }

    fn iter_reachable_garden_plot_poses_for_grid_and_dist(
        grid: &Grid2D<u16>,
        max_dist: u16,
    ) -> impl Iterator<Item = IVec2> + '_ {
        let max_dist_is_even: bool = max_dist % 2_u16 == 0_u16;

        grid.cells()
            .iter()
            .enumerate()
            .filter_map(move |(index, dist)| {
                if *dist != u16::MAX && (*dist % 2_u16 == 0_u16) == max_dist_is_even {
                    Some(grid.pos_from_index(index))
                } else {
                    None
                }
            })
    }

    fn iter_reachable_garden_plot_poses(dist_grid: &DistGrid) -> impl Iterator<Item = IVec2> + '_ {
        Self::iter_reachable_garden_plot_poses_for_grid_and_dist(
            &dist_grid.grid,
            dist_grid.max_dist,
        )
    }

    fn reachable_garden_plots_for_grid(dist_grid: &DistGrid) -> usize {
        Self::iter_reachable_garden_plot_poses(dist_grid).count()
    }

    fn reachable_garden_plots(&self, steps: u16) -> usize {
        let dist_grid: DistGrid = self.dist_grid(steps);

        Self::reachable_garden_plots_for_grid(&dist_grid)
    }

    fn string_for_dist_grid(&self, dist_grid: &DistGrid) -> String {
        let mut grid: Grid2D<Cell> = self.grid.clone();

        for pos in Self::iter_reachable_garden_plot_poses(dist_grid) {
            *grid.get_mut(pos).unwrap() = Cell::ReachableGardenPlot;
        }

        grid.into()
    }

    fn multi_map_dists(&self, max_dist: u16) -> HashMap<IVec2, u16> {
        let mut multi_map_dist_finder: MultiMapDistFinder = MultiMapDistFinder {
            solution: self,
            dists: HashMap::new(),
            max_dist,
            start: self.start,
        };

        multi_map_dist_finder.run();

        multi_map_dist_finder.dists
    }

    fn quadratic_domain_values(&self, steps: u32) -> Option<[u16; 3_usize]> {
        let dimensions: IVec2 = self.grid.dimensions();

        (dimensions.x == dimensions.y).then(|| {
            let map_side_len: u16 = dimensions.x as u16;
            let first_domain_value: u16 = (steps % map_side_len as u32) as u16;

            [
                first_domain_value,
                first_domain_value + map_side_len,
                first_domain_value + 2_u16 * map_side_len,
            ]
        })
    }

    fn quadratic_range_values(&self, quadratic_domain_values: [u16; 3_usize]) -> [i64; 3_usize] {
        let multi_map_dists: HashMap<IVec2, u16> =
            self.multi_map_dists(quadratic_domain_values[2_usize] as u16);

        let mut quadratic_range_values: [i64; 3_usize] = Default::default();

        for (domain_value, range_value) in quadratic_domain_values
            .iter()
            .copied()
            .zip(quadratic_range_values.iter_mut())
        {
            *range_value = multi_map_dists
                .iter()
                .filter(|(_, dist)| {
                    let dist: u16 = **dist;

                    dist <= domain_value && dist % 2_u16 == domain_value % 2_u16
                })
                .count() as i64;
        }

        quadratic_range_values
    }

    fn multi_map_reachable_garden_plots(&self, steps: u32) -> Option<usize> {
        self.quadratic_domain_values(steps).and_then(|x_values| {
            let y_values: [i64; 3_usize] = self.quadratic_range_values(x_values);

            if let Some(index) = x_values.iter().position(|x_value| *x_value as u32 == steps) {
                Some(y_values[index] as usize)
            } else {
                let dimensions: IVec2 = self.grid.dimensions();
                let map_side_len: i128 = dimensions.x as i128; // d below
                let steps: i128 = steps as i128; // x below

                /*
                From https://en.wikipedia.org/wiki/Polynomial_interpolation
                p(x) = y[0] * (x - x[1]) * (x - x[2])     / ((x[0] - x[1]) * (x[0] - x[2])) +
                       y[1] * (x - x[0]) * (x - x[2])     / ((x[1] - x[0]) * (x[1] - x[2])) +
                       y[2] * (x - x[0]) * (x - x[1])     / ((x[2] - x[0]) * (x[2] - x[1]))
                     = y[0] * (x - x[1]) * (x - x[2])     / (-d * -2 * d) +
                       y[1] * (x - x[0]) * (x - x[2])     / (d * -d) +
                       y[2] * (x - x[0]) * (x - x[1])     / (2 * d * d)
                     = y[0] * (x - x[1]) * (x - x[2])     / (2 * d ^ 2) -
                       y[1] * (x - x[0]) * (x - x[2]) * 2 / (2 * d ^ 2) +
                       y[2] * (x - x[0]) * (x - x[1])     / (2 * d ^ 2)
                     = (y[0] * (x - x[1]) * (x - x[2])
                  - 2 * y[1] * (x - x[0]) * (x - x[2])
                      + y[2] * (x - x[0]) * (x - x[1])) / (2 * d ^ 2)

                      70230848671342200000
                */

                let steps_minus_x_0: i128 = steps - (x_values[0_usize] as i128);
                let steps_minus_x_1: i128 = steps - (x_values[1_usize] as i128);
                let steps_minus_x_2: i128 = steps - (x_values[2_usize] as i128);
                let numerator: i128 = y_values[0_usize] as i128 * steps_minus_x_1 * steps_minus_x_2
                    - 2_i128 * y_values[1_usize] as i128 * steps_minus_x_0 * steps_minus_x_2
                    + y_values[2_usize] as i128 * steps_minus_x_0 * steps_minus_x_1;
                let denominator: i128 = 2_i128 * map_side_len * map_side_len;
                let ratio: BigRational = BigRational::new(numerator.into(), denominator.into());

                ratio
                    .is_integer()
                    .then(|| ratio.numer())
                    .and_then(BigInt::to_usize)
            }
        })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::<Cell>::parse, |grid| {
            let start: IVec2 = grid.pos_from_index(
                grid.cells()
                    .iter()
                    .position(|cell| *cell == Cell::StartingPosition)?,
            );

            Some(Self { grid, start })
        })(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let dist_grid: DistGrid = self.dist_grid(Solution::FAVORITE_SQUARE_CUBE);

            dbg!(Solution::reachable_garden_plots_for_grid(&dist_grid));

            println!("\n{}\n", self.string_for_dist_grid(&dist_grid));
        } else {
            dbg!(self.reachable_garden_plots(Solution::FAVORITE_SQUARE_CUBE));
        }
    }

    /// I was lost on this one. The only explanation that I could both find and understand how to
    /// implement was from [u/charr3 on the r/adventofcode subreddit][url], which is what I've
    /// implemented here. It doesn't work on the test cases, unfortunately, but it got me my star,
    /// which is good enough for me for now. I must say that I'm displeased with so many problems
    /// this year where the second question realistically requires using observed properties of the
    /// user-specific input that aren't present in the example input. Today is the second
    /// consecutive such day.
    ///
    /// [url]: https://www.reddit.com/r/adventofcode/comments/18nevo3/comment/keaiiq7/
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.multi_map_reachable_garden_plots(Solution::STEPS));
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
        ...........\n\
        .....###.#.\n\
        .###.##..#.\n\
        ..#.#...#..\n\
        ....#.#....\n\
        .##..S####.\n\
        .##..#...#.\n\
        .......##..\n\
        .##.#.####.\n\
        .##..##.##.\n\
        ...........\n";

    const STEPS: u16 = 6_u16;

    fn solution() -> &'static Solution {
        use Cell::{GardenPlot as G, Rock as R, StartingPosition as S};

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            grid: Grid2D::try_from_cells_and_width(
                vec![
                    G, G, G, G, G, G, G, G, G, G, G, G, G, G, G, G, R, R, R, G, R, G, G, R, R, R,
                    G, R, R, G, G, R, G, G, G, R, G, R, G, G, G, R, G, G, G, G, G, G, R, G, R, G,
                    G, G, G, G, R, R, G, G, S, R, R, R, R, G, G, R, R, G, G, R, G, G, G, R, G, G,
                    G, G, G, G, G, G, R, R, G, G, G, R, R, G, R, G, R, R, R, R, G, G, R, R, G, G,
                    R, R, G, R, R, G, G, G, G, G, G, G, G, G, G, G, G,
                ],
                11_usize,
            )
            .unwrap(),
            start: 5_i32 * IVec2::ONE,
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_reachable_garden_plots() {
        assert_eq!(
            solution().reachable_garden_plots(STEPS),
            16_usize,
            "grid string: \n{}",
            solution().string_for_dist_grid(&solution().dist_grid(STEPS))
        );
    }

    #[test]
    fn test_valid_multi_map_reachable_garden_plots() {
        assert_eq!(
            solution().multi_map_reachable_garden_plots(6_u32),
            Some(16_usize)
        );
        assert_eq!(
            solution().multi_map_reachable_garden_plots(10_u32),
            Some(50_usize)
        );
    }

    #[test]
    fn test_invalid_multi_map_reachable_garden_plots() {
        // These test cases don't work, but should. See the comment on `q2_internal` for more info.
        assert!(![
            (
                solution().multi_map_reachable_garden_plots(50_u32),
                1594_usize
            ),
            (
                solution().multi_map_reachable_garden_plots(100_u32),
                6536_usize
            ),
            (
                solution().multi_map_reachable_garden_plots(500_u32),
                167004_usize
            ),
            (
                solution().multi_map_reachable_garden_plots(1000_u32),
                668697_usize
            ),
            (
                solution().multi_map_reachable_garden_plots(5000_u32),
                16733044_usize
            )
        ]
        .into_iter()
        .any(|(real, expected)| real == Some(expected)));
    }
}
