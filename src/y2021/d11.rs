use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    std::mem::{transmute, MaybeUninit},
};

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct CharOutOfBounds(char);

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Light(u8);

impl TryFrom<char> for Light {
    type Error = CharOutOfBounds;

    fn try_from(height: char) -> Result<Self, Self::Error> {
        match height {
            '0'..='9' => Ok(Light(height as u8)),
            _ => Err(CharOutOfBounds(height)),
        }
    }
}

struct FlashQueue {
    data: [usize; Solution::CELLS],
    head: usize,
    len: usize,
}

impl FlashQueue {
    #[inline(always)]
    fn new() -> Self {
        // SAFETY: `0_u8` for all bytes of a `usize` is `0_usize`
        unsafe { MaybeUninit::zeroed().assume_init() }
    }

    fn push(&mut self, index: usize) {
        assert!(self.len < Solution::CELLS, "Cannot push any more elements");

        self.data[(self.head + self.len) % Solution::CELLS] = index;
        self.len += 1_usize;
    }

    fn pop(&mut self) -> Option<usize> {
        if self.len != 0_usize {
            let option: Option<usize> = Some(self.data[self.head]);

            self.head = (self.head + 1_usize) % Solution::CELLS;
            self.len -= 1_usize;

            option
        } else {
            None
        }
    }
}

type CellsBitArray = BitArr!(for Solution::CELLS);

pub struct RunSolution<'s> {
    solution: &'s mut Solution,
    flash_queue: FlashQueue,
    flashed_this_step: CellsBitArray,
    steps: usize,
}

impl<'s> RunSolution<'s> {
    #[inline(always)]
    fn new(solution: &'s mut Solution, steps: usize) -> Self {
        Self {
            solution,
            flash_queue: FlashQueue::new(),
            flashed_this_step: CellsBitArray::ZERO,
            steps,
        }
    }

    fn run(mut self) -> usize {
        let mut flashed: usize = 0_usize;

        while self.steps > 0_usize {
            self.steps -= 1_usize;

            for index in 0_usize..Solution::CELLS {
                self.increment_light_by_index(index);
            }

            while let Some(index) = self.flash_queue.pop() {
                const NEIGHBOR_DELTAS: [IVec2; 8_usize] = [
                    IVec2::new(1_i32, 0_i32),
                    IVec2::new(1_i32, 1_i32),
                    IVec2::new(0_i32, 1_i32),
                    IVec2::new(-1_i32, 1_i32),
                    IVec2::new(-1_i32, 0_i32),
                    IVec2::new(-1_i32, -1_i32),
                    IVec2::new(0_i32, -1_i32),
                    IVec2::new(1_i32, -1_i32),
                ];

                let pos: IVec2 = self.solution.0.pos_from_index(index);

                for delta in NEIGHBOR_DELTAS {
                    self.increment_light_by_pos(pos + delta);
                }
            }

            for index in self.flashed_this_step.iter_ones() {
                self.solution.0.cells_mut()[index].0 = b'0';
            }

            flashed += self.flashed_this_step.count_ones();
            self.flashed_this_step.fill(false);
        }

        flashed
    }

    fn increment_light_by_index(&mut self, index: usize) {
        if !self.flashed_this_step[index] {
            let light: &mut Light = &mut self.solution.0.cells_mut()[index];
            let new_light: u8 = light.0 + 1_u8;

            if new_light > Solution::MAX {
                self.flash_queue.push(index);
                self.flashed_this_step.set(index, true);
            }

            light.0 = new_light;
        }
    }

    fn increment_light_by_pos(&mut self, pos: IVec2) {
        if let Some(index) = self.solution.0.try_index_from_pos(pos) {
            self.increment_light_by_index(index);
        }
    }
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Light>);

impl Solution {
    const MAX: u8 = b'9';
    const DIMENSIONS: IVec2 = IVec2::new(10_i32, 10_i32);
    const CELLS: usize = Solution::DIMENSIONS.x as usize * Solution::DIMENSIONS.y as usize;

    fn run(&mut self, steps: usize) -> usize {
        RunSolution::new(self, steps).run()
    }

    fn string(&self) -> String {
        // SAFETY: `Grid2DString` is just a new-type for `Grid2D<u8>`, and `Self` is just a new-type
        // for `Grid2D<Height>`, where `Height` is just a new-type for `u8`.
        unsafe { transmute::<&Self, &Grid2DString>(self) }
            .try_as_string()
            .unwrap_or_default()
    }

    fn run_100_steps(&self) -> (usize, String) {
        let mut solution: Solution = self.clone();

        let flashes: usize = solution.run(100_usize);
        let string: String = solution.string();

        (flashes, string)
    }

    fn steps_until_synchronization(&self) -> usize {
        let mut solution: Solution = self.clone();
        let mut steps: usize = 1_usize;

        while solution.run(1_usize) != Self::CELLS {
            steps += 1_usize;
        }

        steps
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        let (flashes_after_100_steps, string): (usize, String) = self.run_100_steps();

        dbg!(flashes_after_100_steps);

        if args.verbose {
            eprintln!("string:\n{string}");
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.steps_until_synchronization());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = GridParseError<'i, CharOutOfBounds>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(Grid2D::try_from(input)?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, lazy_static::lazy_static, std::ops::Range};

    const SOLUTION_STR: &str = concat!(
        "5483143223\n",
        "2745854711\n",
        "5264556173\n",
        "6141336146\n",
        "6357385478\n",
        "4167524645\n",
        "2176841721\n",
        "6882881134\n",
        "4846848554\n",
        "5283751526\n",
    );

    lazy_static! {
        static ref SOLUTION: Solution = solution();
    }

    fn solution() -> Solution {
        macro_rules! lights {
            [ $( [ $( $light:expr ),* ], )* ] => { vec![ $( $( Light($light), )* )* ] };
        }

        Solution(
            Grid2D::try_from_cells_and_dimensions(
                lights![
                    [b'5', b'4', b'8', b'3', b'1', b'4', b'3', b'2', b'2', b'3'],
                    [b'2', b'7', b'4', b'5', b'8', b'5', b'4', b'7', b'1', b'1'],
                    [b'5', b'2', b'6', b'4', b'5', b'5', b'6', b'1', b'7', b'3'],
                    [b'6', b'1', b'4', b'1', b'3', b'3', b'6', b'1', b'4', b'6'],
                    [b'6', b'3', b'5', b'7', b'3', b'8', b'5', b'4', b'7', b'8'],
                    [b'4', b'1', b'6', b'7', b'5', b'2', b'4', b'6', b'4', b'5'],
                    [b'2', b'1', b'7', b'6', b'8', b'4', b'1', b'7', b'2', b'1'],
                    [b'6', b'8', b'8', b'2', b'8', b'8', b'1', b'1', b'3', b'4'],
                    [b'4', b'8', b'4', b'6', b'8', b'4', b'8', b'5', b'5', b'4'],
                    [b'5', b'2', b'8', b'3', b'7', b'5', b'1', b'5', b'2', b'6'],
                ],
                Solution::DIMENSIONS,
            )
            .unwrap_or_else(|| Grid2D::empty(Solution::DIMENSIONS)),
        )
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR), Ok(solution()))
    }

    #[test]
    fn test_run() {
        let mut solution: Solution = SOLUTION.clone();
        let mut total_steps: usize = 0_usize;
        let mut flashes: usize = 0_usize;

        const STRS_AND_STEPS: [(&str, usize); 19_usize] = [
            // After step 1:
            (
                concat!(
                    "6594254334\n",
                    "3856965822\n",
                    "6375667284\n",
                    "7252447257\n",
                    "7468496589\n",
                    "5278635756\n",
                    "3287952832\n",
                    "7993992245\n",
                    "5957959665\n",
                    "6394862637\n",
                ),
                1_usize,
            ),
            // After step 2:
            (
                concat!(
                    "8807476555\n",
                    "5089087054\n",
                    "8597889608\n",
                    "8485769600\n",
                    "8700908800\n",
                    "6600088989\n",
                    "6800005943\n",
                    "0000007456\n",
                    "9000000876\n",
                    "8700006848\n",
                ),
                1_usize,
            ),
            // After step 3:
            (
                concat!(
                    "0050900866\n",
                    "8500800575\n",
                    "9900000039\n",
                    "9700000041\n",
                    "9935080063\n",
                    "7712300000\n",
                    "7911250009\n",
                    "2211130000\n",
                    "0421125000\n",
                    "0021119000\n",
                ),
                1_usize,
            ),
            // After step 4:
            (
                concat!(
                    "2263031977\n",
                    "0923031697\n",
                    "0032221150\n",
                    "0041111163\n",
                    "0076191174\n",
                    "0053411122\n",
                    "0042361120\n",
                    "5532241122\n",
                    "1532247211\n",
                    "1132230211\n",
                ),
                1_usize,
            ),
            // After step 5:
            (
                concat!(
                    "4484144000\n",
                    "2044144000\n",
                    "2253333493\n",
                    "1152333274\n",
                    "1187303285\n",
                    "1164633233\n",
                    "1153472231\n",
                    "6643352233\n",
                    "2643358322\n",
                    "2243341322\n",
                ),
                1_usize,
            ),
            // After step 6:
            (
                concat!(
                    "5595255111\n",
                    "3155255222\n",
                    "3364444605\n",
                    "2263444496\n",
                    "2298414396\n",
                    "2275744344\n",
                    "2264583342\n",
                    "7754463344\n",
                    "3754469433\n",
                    "3354452433\n",
                ),
                1_usize,
            ),
            // After step 7:
            (
                concat!(
                    "6707366222\n",
                    "4377366333\n",
                    "4475555827\n",
                    "3496655709\n",
                    "3500625609\n",
                    "3509955566\n",
                    "3486694453\n",
                    "8865585555\n",
                    "4865580644\n",
                    "4465574644\n",
                ),
                1_usize,
            ),
            // After step 8:
            (
                concat!(
                    "7818477333\n",
                    "5488477444\n",
                    "5697666949\n",
                    "4608766830\n",
                    "4734946730\n",
                    "4740097688\n",
                    "6900007564\n",
                    "0000009666\n",
                    "8000004755\n",
                    "6800007755\n",
                ),
                1_usize,
            ),
            // After step 9:
            (
                concat!(
                    "9060000644\n",
                    "7800000976\n",
                    "6900000080\n",
                    "5840000082\n",
                    "5858000093\n",
                    "6962400000\n",
                    "8021250009\n",
                    "2221130009\n",
                    "9111128097\n",
                    "7911119976\n",
                ),
                1_usize,
            ),
            // After step 10:
            (
                concat!(
                    "0481112976\n",
                    "0031112009\n",
                    "0041112504\n",
                    "0081111406\n",
                    "0099111306\n",
                    "0093511233\n",
                    "0442361130\n",
                    "5532252350\n",
                    "0532250600\n",
                    "0032240000\n",
                ),
                1_usize,
            ),
            // After step 20:
            (
                concat!(
                    "3936556452\n",
                    "5686556806\n",
                    "4496555690\n",
                    "4448655580\n",
                    "4456865570\n",
                    "5680086577\n",
                    "7000009896\n",
                    "0000000344\n",
                    "6000000364\n",
                    "4600009543\n",
                ),
                10_usize,
            ),
            // After step 30:
            (
                concat!(
                    "0643334118\n",
                    "4253334611\n",
                    "3374333458\n",
                    "2225333337\n",
                    "2229333338\n",
                    "2276733333\n",
                    "2754574565\n",
                    "5544458511\n",
                    "9444447111\n",
                    "7944446119\n",
                ),
                10_usize,
            ),
            // After step 40:
            (
                concat!(
                    "6211111981\n",
                    "0421111119\n",
                    "0042111115\n",
                    "0003111115\n",
                    "0003111116\n",
                    "0065611111\n",
                    "0532351111\n",
                    "3322234597\n",
                    "2222222976\n",
                    "2222222762\n",
                ),
                10_usize,
            ),
            // After step 50:
            (
                concat!(
                    "9655556447\n",
                    "4865556805\n",
                    "4486555690\n",
                    "4458655580\n",
                    "4574865570\n",
                    "5700086566\n",
                    "6000009887\n",
                    "8000000533\n",
                    "6800000633\n",
                    "5680000538\n",
                ),
                10_usize,
            ),
            // After step 60:
            (
                concat!(
                    "2533334200\n",
                    "2743334640\n",
                    "2264333458\n",
                    "2225333337\n",
                    "2225333338\n",
                    "2287833333\n",
                    "3854573455\n",
                    "1854458611\n",
                    "1175447111\n",
                    "1115446111\n",
                ),
                10_usize,
            ),
            // After step 70:
            (
                concat!(
                    "8211111164\n",
                    "0421111166\n",
                    "0042111114\n",
                    "0004211115\n",
                    "0000211116\n",
                    "0065611111\n",
                    "0532351111\n",
                    "7322235117\n",
                    "5722223475\n",
                    "4572222754\n",
                ),
                10_usize,
            ),
            // After step 80:
            (
                concat!(
                    "1755555697\n",
                    "5965555609\n",
                    "4486555680\n",
                    "4458655580\n",
                    "4570865570\n",
                    "5700086566\n",
                    "7000008666\n",
                    "0000000990\n",
                    "0000000800\n",
                    "0000000000\n",
                ),
                10_usize,
            ),
            // After step 90:
            (
                concat!(
                    "7433333522\n",
                    "2643333522\n",
                    "2264333458\n",
                    "2226433337\n",
                    "2222433338\n",
                    "2287833333\n",
                    "2854573333\n",
                    "4854458333\n",
                    "3387779333\n",
                    "3333333333\n",
                ),
                10_usize,
            ),
            // After step 100:
            (
                concat!(
                    "0397666866\n",
                    "0749766918\n",
                    "0053976933\n",
                    "0004297822\n",
                    "0004229892\n",
                    "0053222877\n",
                    "0532222966\n",
                    "9322228966\n",
                    "7922286866\n",
                    "6789998766\n",
                ),
                10_usize,
            ),
        ];

        let mut run_range = |flashes: &mut usize, range: Range<usize>| {
            for (string, steps) in STRS_AND_STEPS[range].iter().copied() {
                total_steps += steps;
                *flashes += solution.run(steps);

                assert_eq!(solution.string(), string);
            }
        };

        run_range(&mut flashes, 0_usize..10_usize);

        assert_eq!(flashes, 204_usize);

        run_range(&mut flashes, 10_usize..19_usize);

        assert_eq!(flashes, 1_656_usize);
    }

    #[test]
    fn test_steps_until_syncrhonization() {
        assert_eq!(SOLUTION.steps_until_synchronization(), 195_usize);
    }
}
