use {
    crate::*,
    glam::IVec2,
    nom::{combinator::map, error::Error, Err, IResult},
};

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Cell {
        East = EAST = b'>',
        South = SOUTH = b'v',
        Empty = EMPTY = b'.',
    }
}

impl Cell {
    fn try_get_dir(self) -> Option<Direction> {
        match self {
            Cell::East => Some(Direction::East),
            Cell::South => Some(Direction::South),
            Cell::Empty => None,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution(Grid2D<Cell>);

impl Solution {
    fn steps_until_stationary(&mut self) -> usize {
        let mut steps_until_stationary: usize = 1_usize;
        let mut to_be_moved: Vec<IVec2> = Vec::new();

        while self.step(Some(&mut to_be_moved)) > 0_usize {
            steps_until_stationary += 1_usize;
        }

        steps_until_stationary
    }

    fn step(&mut self, to_be_moved: Option<&mut Vec<IVec2>>) -> usize {
        let mut local_to_be_moved: Vec<IVec2> = Vec::new();
        let to_be_moved: &mut Vec<IVec2> = to_be_moved.unwrap_or(&mut local_to_be_moved);

        self.move_herd(Cell::East, to_be_moved) + self.move_herd(Cell::South, to_be_moved)
    }

    fn move_herd(&mut self, herd: Cell, to_be_moved: &mut Vec<IVec2>) -> usize {
        let dimensions: IVec2 = self.0.dimensions();
        let herd_vec_add_dimensions: IVec2 = herd.try_get_dir().unwrap().vec() + dimensions;

        to_be_moved.clear();
        to_be_moved.extend(
            self.0
                .cells()
                .iter()
                .enumerate()
                .filter_map(|(index, cell)| {
                    if *cell == herd {
                        let pos: IVec2 = self.0.pos_from_index(index);

                        if *self
                            .0
                            .get((pos + herd_vec_add_dimensions) % dimensions)
                            .unwrap()
                            == Cell::Empty
                        {
                            Some(pos)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }),
        );

        let moved_sea_cucumber_count: usize = to_be_moved.len();

        for pos in to_be_moved.drain(..) {
            *self.0.get_mut(pos).unwrap() = Cell::Empty;
            *self
                .0
                .get_mut((pos + herd_vec_add_dimensions) % dimensions)
                .unwrap() = herd;
        }

        moved_sea_cucumber_count
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.steps_until_stationary());

        if args.verbose {
            println!("{}", String::from(self.0.clone()))
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {}
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
        std::{collections::HashMap, sync::OnceLock},
    };

    struct SolutionData {
        str: &'static str,
        steps: usize,
    }

    impl SolutionData {
        const fn new(str: &'static str, steps: usize) -> Self {
            Self { str, steps }
        }
    }

    static SOLUTIONS: OnceLock<HashMap<(usize, usize), OnceLock<Solution>>> = OnceLock::new();

    const SOLUTION_DATAS: &[&[SolutionData]] = &[
        &[
            SolutionData::new(
                "\
                v...>>.vv>\n\
                .vv>>.vv..\n\
                >>.>v>...v\n\
                >>v>>.>.v.\n\
                v>v.vv.v..\n\
                >.>>..v...\n\
                .vv..>.>v.\n\
                v.v..>>v.v\n\
                ....v..v.>\n",
                0_usize,
            ),
            SolutionData::new(
                "\
                ....>.>v.>\n\
                v.v>.>v.v.\n\
                >v>>..>v..\n\
                >>v>v>.>.v\n\
                .>v.v...v.\n\
                v>>.>vvv..\n\
                ..v...>>..\n\
                vv...>>vv.\n\
                >.v.v..v.v\n",
                1_usize,
            ),
            SolutionData::new(
                "\
                >.v.v>>..v\n\
                v.v.>>vv..\n\
                >v>.>.>.v.\n\
                >>v>v.>v>.\n\
                .>..v....v\n\
                .>v>>.v.v.\n\
                v....v>v>.\n\
                .vv..>>v..\n\
                v>.....vv.\n",
                2_usize,
            ),
            SolutionData::new(
                "\
                v>v.v>.>v.\n\
                v...>>.v.v\n\
                >vv>.>v>..\n\
                >>v>v.>.v>\n\
                ..>....v..\n\
                .>.>v>v..v\n\
                ..v..v>vv>\n\
                v.v..>>v..\n\
                .v>....v..\n",
                3_usize,
            ),
            SolutionData::new(
                "\
                v>..v.>>..\n\
                v.v.>.>.v.\n\
                >vv.>>.v>v\n\
                >>.>..v>.>\n\
                ..v>v...v.\n\
                ..>>.>vv..\n\
                >.v.vv>v.v\n\
                .....>>vv.\n\
                vvv>...v..\n",
                4_usize,
            ),
            SolutionData::new(
                "\
                vv>...>v>.\n\
                v.v.v>.>v.\n\
                >.v.>.>.>v\n\
                >v>.>..v>>\n\
                ..v>v.v...\n\
                ..>.>>vvv.\n\
                .>...v>v..\n\
                ..v.v>>v.v\n\
                v.v.>...v.\n",
                5_usize,
            ),
            SolutionData::new(
                "\
                ..>..>>vv.\n\
                v.....>>.v\n\
                ..v.v>>>v>\n\
                v>.>v.>>>.\n\
                ..v>v.vv.v\n\
                .v.>>>.v..\n\
                v.v..>v>..\n\
                ..v...>v.>\n\
                .vv..v>vv.\n",
                10_usize,
            ),
            SolutionData::new(
                "\
                v>.....>>.\n\
                >vv>.....v\n\
                .>v>v.vv>>\n\
                v>>>v.>v.>\n\
                ....vv>v..\n\
                .v.>>>vvv.\n\
                ..v..>>vv.\n\
                v.v...>>.v\n\
                ..v.....v>\n",
                20_usize,
            ),
            SolutionData::new(
                "\
                .vv.v..>>>\n\
                v>...v...>\n\
                >.v>.>vv.>\n\
                >v>.>.>v.>\n\
                .>..v.vv..\n\
                ..v>..>>v.\n\
                ....v>..>v\n\
                v.v...>vv>\n\
                v.v...>vvv\n",
                30_usize,
            ),
            SolutionData::new(
                "\
                >>v>v..v..\n\
                ..>>v..vv.\n\
                ..>>>v.>.v\n\
                ..>>>>vvv>\n\
                v.....>...\n\
                v.v...>v>>\n\
                >vv.....v>\n\
                .>v...v.>v\n\
                vvv.v..v.>\n",
                40_usize,
            ),
            SolutionData::new(
                "\
                ..>>v>vv.v\n\
                ..v.>>vv..\n\
                v.>>v>>v..\n\
                ..>>>>>vv.\n\
                vvv....>vv\n\
                ..v....>>>\n\
                v>.......>\n\
                .vv>....v>\n\
                .>v.vv.v..\n",
                50_usize,
            ),
            SolutionData::new(
                "\
                ..>>v>vv..\n\
                ..v.>>vv..\n\
                ..>>v>>vv.\n\
                ..>>>>>vv.\n\
                v......>vv\n\
                v>v....>>v\n\
                vvv...>..>\n\
                >vv.....>.\n\
                .>v.vv.v..\n",
                55_usize,
            ),
            SolutionData::new(
                "\
                ..>>v>vv..\n\
                ..v.>>vv..\n\
                ..>>v>>vv.\n\
                ..>>>>>vv.\n\
                v......>vv\n\
                v>v....>>v\n\
                vvv....>.>\n\
                >vv......>\n\
                .>v.vv.v..\n",
                56_usize,
            ),
            SolutionData::new(
                "\
                ..>>v>vv..\n\
                ..v.>>vv..\n\
                ..>>v>>vv.\n\
                ..>>>>>vv.\n\
                v......>vv\n\
                v>v....>>v\n\
                vvv.....>>\n\
                >vv......>\n\
                .>v.vv.v..\n",
                57_usize,
            ),
            SolutionData::new(
                "\
                ..>>v>vv..\n\
                ..v.>>vv..\n\
                ..>>v>>vv.\n\
                ..>>>>>vv.\n\
                v......>vv\n\
                v>v....>>v\n\
                vvv.....>>\n\
                >vv......>\n\
                .>v.vv.v..\n",
                58_usize,
            ),
        ],
        &[
            SolutionData::new("...>>>>>...", 0_usize),
            SolutionData::new("...>>>>.>..", 1_usize),
            SolutionData::new("...>>>.>.>.", 2_usize),
        ],
        &[
            SolutionData::new(
                "\
                ..........\n\
                .>v....v..\n\
                .......>..\n\
                ..........\n",
                0_usize,
            ),
            SolutionData::new(
                "\
                ..........\n\
                .>........\n\
                ..v....v>.\n\
                ..........\n",
                1_usize,
            ),
        ],
        &[
            SolutionData::new(
                "\
                ...>...\n\
                .......\n\
                ......>\n\
                v.....>\n\
                ......>\n\
                .......\n\
                ..vvv..\n",
                0_usize,
            ),
            SolutionData::new(
                "\
                ..vv>..\n\
                .......\n\
                >......\n\
                v.....>\n\
                >......\n\
                .......\n\
                ....v..\n",
                1_usize,
            ),
            SolutionData::new(
                "\
                ....v>.\n\
                ..vv...\n\
                .>.....\n\
                ......>\n\
                v>.....\n\
                .......\n\
                .......\n",
                2_usize,
            ),
            SolutionData::new(
                "\
                ......>\n\
                ..v.v..\n\
                ..>v...\n\
                >......\n\
                ..>....\n\
                v......\n\
                .......\n",
                3_usize,
            ),
            SolutionData::new(
                "\
                >......\n\
                ..v....\n\
                ..>.v..\n\
                .>.v...\n\
                ...>...\n\
                .......\n\
                v......\n",
                4_usize,
            ),
        ],
    ];

    fn solution(group: usize, solution_data: &SolutionData) -> &'static Solution {
        SOLUTIONS
            .get_or_init(|| {
                SOLUTION_DATAS
                    .iter()
                    .copied()
                    .enumerate()
                    .flat_map(|(group, solution_datas)| {
                        solution_datas.iter().map(move |solution_data| {
                            ((group, solution_data.steps), OnceLock::new())
                        })
                    })
                    .collect()
            })
            .get(&(group, solution_data.steps))
            .unwrap()
            .get_or_init(|| solution_data.str.try_into().unwrap())
    }

    #[test]
    fn test_solution_try_from_str() {
        use Cell::{East as E, Empty as M, South as S};

        assert_eq!(
            Solution::try_from(SOLUTION_DATAS[0][0].str),
            Ok(Solution(
                Grid2D::try_from_cells_and_dimensions(
                    vec![
                        S, M, M, M, E, E, M, S, S, E, M, S, S, E, E, M, S, S, M, M, E, E, M, E, S,
                        E, M, M, M, S, E, E, S, E, E, M, E, M, S, M, S, E, S, M, S, S, M, S, M, M,
                        E, M, E, E, M, M, S, M, M, M, M, S, S, M, M, E, M, E, S, M, S, M, S, M, M,
                        E, E, S, M, S, M, M, M, M, S, M, M, S, M, E,
                    ],
                    IVec2::new(10_i32, 9_i32),
                )
                .unwrap(),
            ))
        );
    }

    #[test]
    fn test_solution_step() {
        let mut to_be_moved: Vec<IVec2> = Vec::new();

        for (group, solution_datas) in SOLUTION_DATAS.iter().copied().enumerate() {
            for solution_data_pair in solution_datas.windows(2_usize) {
                let curr_solution_data: &SolutionData = &solution_data_pair[0_usize];
                let next_solution_data: &SolutionData = &solution_data_pair[1_usize];
                let mut curr_solution: Solution = solution(group, curr_solution_data).clone();
                let next_solution: &Solution = solution(group, next_solution_data);

                for _ in curr_solution_data.steps..next_solution_data.steps {
                    curr_solution.step(Some(&mut to_be_moved));
                }

                if curr_solution != *next_solution {
                    assert!(
                        false,
                        "\
                            group == {group}\n\
                            curr_solution_data.steps == {}\n\
                            curr_solution ==\n\
                            {}\n\
                            next_solution_data.steps == {}\n\
                            next_solution ==\n\
                            {}\n",
                        curr_solution_data.steps,
                        String::from(curr_solution.0),
                        next_solution_data.steps,
                        next_solution_data.str
                    )
                }
            }
        }
    }

    #[test]
    fn test_solution_steps_until_stationary() {
        assert_eq!(
            solution(0_usize, &SOLUTION_DATAS[0_usize][0_usize])
                .clone()
                .steps_until_stationary(),
            58_usize
        );
    }
}
