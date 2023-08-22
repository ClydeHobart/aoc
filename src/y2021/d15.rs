use {
    crate::*,
    derive_deref::{Deref, DerefMut},
    glam::IVec2,
    std::{
        collections::VecDeque,
        ops::{Add, Range},
    },
    strum::IntoEnumIterator,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy, Deref)]
#[repr(transparent)]
pub struct RiskLevel(u8);

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
pub struct InvalidRiskLevel;

impl RiskLevel {
    const MIN: u8 = 1;
    const MAX: u8 = 9;
    const ASCII_OFFSET: u8 = b'0';
    const ASCII_RANGE: Range<char> = ((RiskLevel::MIN + RiskLevel::ASCII_OFFSET) as char)
        ..((RiskLevel::MAX + RiskLevel::ASCII_OFFSET + 1_u8) as char);
}

impl Add for RiskLevel {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sum: u8 = self.0 + rhs.0;

        Self(if sum > Self::MAX {
            (sum - Self::MIN) % Self::MAX + Self::MIN
        } else {
            sum
        })
    }
}

impl Default for RiskLevel {
    fn default() -> Self {
        Self(Self::MIN)
    }
}

impl TryFrom<char> for RiskLevel {
    type Error = InvalidRiskLevel;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        if Self::ASCII_RANGE.contains(&value) {
            Ok(Self(value as u8 - Self::ASCII_OFFSET))
        } else {
            Err(InvalidRiskLevel)
        }
    }
}

#[derive(Clone, Copy)]
struct FinderCell {
    total_risk: u32,
    previous: Option<Direction>,
}

impl Default for FinderCell {
    fn default() -> Self {
        Self {
            total_risk: u32::MAX,
            previous: None,
        }
    }
}

struct MinimalTotalRiskPathFinder<'s> {
    solution: &'s Solution,
    finder_grid: Grid2D<FinderCell>,
}

impl<'s> AStar for MinimalTotalRiskPathFinder<'s> {
    type Vertex = IVec2;
    type Cost = u32;

    fn start(&self) -> &Self::Vertex {
        &IVec2::ZERO
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.solution.max_dimensions()
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut reverse_path: VecDeque<IVec2> =
            VecDeque::with_capacity((abs_sum_2d(self.solution.max_dimensions())) as usize);
        let mut vertex: IVec2 = *vertex;

        reverse_path.push_front(vertex);

        let mut finder_cell: FinderCell = *self.finder_grid.get(vertex).unwrap();

        while let Some(previous) = finder_cell.previous {
            vertex += previous.vec();
            reverse_path.push_front(vertex);
            finder_cell = *self.finder_grid.get(vertex).unwrap();
        }

        reverse_path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.finder_grid.get(*vertex).unwrap().total_risk
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        abs_sum_2d(self.solution.max_dimensions() - *vertex) as u32
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();
        neighbors.extend(
            Direction::iter()
                .map(|dir: Direction| *vertex + dir.vec())
                .filter_map(|vertex| {
                    self.solution
                        .get(vertex)
                        .map(|cost| OpenSetElement(vertex, (**cost) as u32))
                }),
        )
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        let finder_cell: &mut FinderCell = self.finder_grid.get_mut(*to).unwrap();

        finder_cell.total_risk = cost;
        finder_cell.previous = Some((*from - *to).try_into().unwrap());
    }

    fn reset(&mut self) {
        self.finder_grid.cells_mut().fill_with(Default::default);
        self.finder_grid.get_mut(IVec2::ZERO).unwrap().total_risk = 0_u32;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Deref, DerefMut)]
pub struct Solution(Grid2D<RiskLevel>);

impl Solution {
    const ENTIRE_CAVE_SCALE: i32 = 5_i32;

    fn find_lowest_total_risk_path(&self) -> Option<(Vec<IVec2>, u32)> {
        MinimalTotalRiskPathFinder {
            solution: self,
            finder_grid: Grid2D::default(self.dimensions()),
        }
        .run()
        .map(|path: Vec<IVec2>| {
            let total_risk: u32 = path
                .iter()
                .skip(1_usize)
                .map(|pos: &IVec2| **self.get(*pos).unwrap() as u32)
                .sum();
            (path, total_risk)
        })
    }

    fn entire_cave(&self) -> Self {
        let dimensions: IVec2 = self.dimensions();
        let mut entire_cave: Self = Self(Grid2D::default(Self::ENTIRE_CAVE_SCALE * dimensions));

        for tile_y_iter in
            CellIter2D::try_from(IVec2::ZERO..Self::ENTIRE_CAVE_SCALE * IVec2::Y).unwrap()
        {
            let risk_level_offset_y: RiskLevel = RiskLevel(tile_y_iter.y as u8);

            for tile_iter in
                CellIter2D::try_from(tile_y_iter..tile_y_iter + Self::ENTIRE_CAVE_SCALE * IVec2::X)
                    .unwrap()
            {
                let tile_offset: IVec2 = dimensions * tile_iter;
                let risk_level_offset: RiskLevel =
                    RiskLevel(tile_iter.x as u8) + risk_level_offset_y;

                for cell_y_iter in
                    CellIter2D::try_from(IVec2::ZERO..dimensions.y * IVec2::Y).unwrap()
                {
                    for cell_iter in
                        CellIter2D::try_from(cell_y_iter..cell_y_iter + dimensions.x * IVec2::X)
                            .unwrap()
                    {
                        *entire_cave.0.get_mut(cell_iter + tile_offset).unwrap() =
                            *self.get(cell_iter).unwrap() + risk_level_offset;
                    }
                }
            }
        }

        entire_cave
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        let lowest_total_risk_path: Option<(Vec<IVec2>, u32)> = self.find_lowest_total_risk_path();
        let total_risk: Option<u32> = lowest_total_risk_path
            .as_ref()
            .map(|(_, total_risk)| *total_risk);

        dbg!(total_risk);

        if args.verbose {
            let path: Option<&Vec<IVec2>> = lowest_total_risk_path.as_ref().map(|(path, _)| path);

            dbg!(path);
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        let lowest_total_risk_path_of_entire_cave: Option<(Vec<IVec2>, u32)> =
            self.entire_cave().find_lowest_total_risk_path();
        let total_risk_of_entire_cave: Option<u32> = lowest_total_risk_path_of_entire_cave
            .as_ref()
            .map(|(_, total_risk)| *total_risk);

        dbg!(total_risk_of_entire_cave);

        if args.verbose {
            let path_of_entire_cave: Option<&Vec<IVec2>> = lowest_total_risk_path_of_entire_cave
                .as_ref()
                .map(|(path, _)| path);

            dbg!(path_of_entire_cave);
        }
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = GridParseError<'i, InvalidRiskLevel>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(input.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const SOLUTION_STR: &str = concat!(
        "1163751742\n",
        "1381373672\n",
        "2136511328\n",
        "3694931569\n",
        "7463417111\n",
        "1319128137\n",
        "1359912421\n",
        "3125421639\n",
        "1293138521\n",
        "2311944581\n",
    );
    const ENTIRE_CAVE_STR: &str = concat!(
        "11637517422274862853338597396444961841755517295286\n",
        "13813736722492484783351359589446246169155735727126\n",
        "21365113283247622439435873354154698446526571955763\n",
        "36949315694715142671582625378269373648937148475914\n",
        "74634171118574528222968563933317967414442817852555\n",
        "13191281372421239248353234135946434524615754563572\n",
        "13599124212461123532357223464346833457545794456865\n",
        "31254216394236532741534764385264587549637569865174\n",
        "12931385212314249632342535174345364628545647573965\n",
        "23119445813422155692453326671356443778246755488935\n",
        "22748628533385973964449618417555172952866628316397\n",
        "24924847833513595894462461691557357271266846838237\n",
        "32476224394358733541546984465265719557637682166874\n",
        "47151426715826253782693736489371484759148259586125\n",
        "85745282229685639333179674144428178525553928963666\n",
        "24212392483532341359464345246157545635726865674683\n",
        "24611235323572234643468334575457944568656815567976\n",
        "42365327415347643852645875496375698651748671976285\n",
        "23142496323425351743453646285456475739656758684176\n",
        "34221556924533266713564437782467554889357866599146\n",
        "33859739644496184175551729528666283163977739427418\n",
        "35135958944624616915573572712668468382377957949348\n",
        "43587335415469844652657195576376821668748793277985\n",
        "58262537826937364893714847591482595861259361697236\n",
        "96856393331796741444281785255539289636664139174777\n",
        "35323413594643452461575456357268656746837976785794\n",
        "35722346434683345754579445686568155679767926678187\n",
        "53476438526458754963756986517486719762859782187396\n",
        "34253517434536462854564757396567586841767869795287\n",
        "45332667135644377824675548893578665991468977611257\n",
        "44961841755517295286662831639777394274188841538529\n",
        "46246169155735727126684683823779579493488168151459\n",
        "54698446526571955763768216687487932779859814388196\n",
        "69373648937148475914825958612593616972361472718347\n",
        "17967414442817852555392896366641391747775241285888\n",
        "46434524615754563572686567468379767857948187896815\n",
        "46833457545794456865681556797679266781878137789298\n",
        "64587549637569865174867197628597821873961893298417\n",
        "45364628545647573965675868417678697952878971816398\n",
        "56443778246755488935786659914689776112579188722368\n",
        "55172952866628316397773942741888415385299952649631\n",
        "57357271266846838237795794934881681514599279262561\n",
        "65719557637682166874879327798598143881961925499217\n",
        "71484759148259586125936169723614727183472583829458\n",
        "28178525553928963666413917477752412858886352396999\n",
        "57545635726865674683797678579481878968159298917926\n",
        "57944568656815567976792667818781377892989248891319\n",
        "75698651748671976285978218739618932984172914319528\n",
        "56475739656758684176786979528789718163989182927419\n",
        "67554889357866599146897761125791887223681299833479\n",
    );

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            macro_rules! risk_levels {
                [ $( [ $( $risk_level:expr ),* ], )* ] => {
                    vec![ $( $( RiskLevel($risk_level), )* )* ]
                };
            }

            Solution(
                Grid2D::try_from_cells_and_dimensions(
                    risk_levels![
                        [1, 1, 6, 3, 7, 5, 1, 7, 4, 2],
                        [1, 3, 8, 1, 3, 7, 3, 6, 7, 2],
                        [2, 1, 3, 6, 5, 1, 1, 3, 2, 8],
                        [3, 6, 9, 4, 9, 3, 1, 5, 6, 9],
                        [7, 4, 6, 3, 4, 1, 7, 1, 1, 1],
                        [1, 3, 1, 9, 1, 2, 8, 1, 3, 7],
                        [1, 3, 5, 9, 9, 1, 2, 4, 2, 1],
                        [3, 1, 2, 5, 4, 2, 1, 6, 3, 9],
                        [1, 2, 9, 3, 1, 3, 8, 5, 2, 1],
                        [2, 3, 1, 1, 9, 4, 4, 5, 8, 1],
                    ],
                    IVec2::new(10_i32, 10_i32),
                )
                .unwrap(),
            )
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_find_lowest_total_risk_path() {
        macro_rules! path {
            [ $( ($x:expr, $y:expr), )* ] => {
                vec![ $( IVec2::new($x, $y), )* ]
            };
        }

        assert_eq!(
            solution().find_lowest_total_risk_path(),
            Some((
                path![
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 2),
                    (2, 2),
                    (3, 2),
                    (4, 2),
                    (5, 2),
                    (6, 2),
                    (6, 3),
                    (7, 3),
                    (7, 4),
                    (8, 4), // The example uses (7, 5), but Direction explores East first
                    (8, 5),
                    (8, 6),
                    (8, 7),
                    (8, 8),
                    (9, 8),
                    (9, 9),
                ],
                40_u32
            ))
        );
        assert_eq!(
            solution()
                .entire_cave()
                .find_lowest_total_risk_path()
                .map(|(_, total_risk)| total_risk),
            Some(315_u32)
        );
    }

    #[test]
    fn test_entire_cave() {
        assert_eq!(
            solution().entire_cave(),
            Solution::try_from(ENTIRE_CAVE_STR).unwrap()
        );
    }
}
