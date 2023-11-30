use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::fold_many_m_n,
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    std::{mem::swap, ops::Range},
};

#[derive(Clone, Debug, Default, PartialEq)]
struct ImageEnhancementAlgorithm(BitArr!(for ImageEnhancementAlgorithm::LEN));

impl ImageEnhancementAlgorithm {
    const INDEX_SIDE_LEN: usize = 3_usize;
    const INDEX_BITS: usize =
        ImageEnhancementAlgorithm::INDEX_SIDE_LEN * ImageEnhancementAlgorithm::INDEX_SIDE_LEN;
    const LEN: usize = 1_usize << ImageEnhancementAlgorithm::INDEX_BITS;
    const DELTAS: [IVec2; ImageEnhancementAlgorithm::INDEX_BITS] =
        ImageEnhancementAlgorithm::deltas();

    const fn deltas() -> [IVec2; ImageEnhancementAlgorithm::INDEX_BITS] {
        const RANGE: Range<i32> =
            -1_i32..(ImageEnhancementAlgorithm::INDEX_SIDE_LEN as i32 - 1_i32);

        let mut deltas: [IVec2; ImageEnhancementAlgorithm::INDEX_BITS] =
            [IVec2::ZERO; ImageEnhancementAlgorithm::INDEX_BITS];
        let mut index: usize = 0_usize;
        let mut y: i32 = RANGE.end - 1_i32;

        while y >= RANGE.start {
            let mut x: i32 = RANGE.end - 1_i32;

            while x >= RANGE.start {
                deltas[index] = IVec2::new(x, y);
                index += 1_usize;
                x -= 1_i32;
            }

            y -= 1_i32
        }

        deltas
    }
}

impl Parse for ImageEnhancementAlgorithm {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut image_enhancement_algorithm: Self = Default::default();
        let mut index: usize = 0_usize;

        let (input, _) = fold_many_m_n(
            Self::LEN,
            Self::LEN,
            terminated(Pixel::parse, opt(line_ending)),
            || (),
            |_, pixel| {
                image_enhancement_algorithm.0.set(index, pixel.is_light());
                index += 1_usize;
            },
        )(input)?;

        Ok((input, image_enhancement_algorithm))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
pub struct Solution {
    image_enhancement_algorithm: ImageEnhancementAlgorithm,
    curr_grid: Grid2D<Pixel>,
    next_grid: Grid2D<Pixel>,
    enhancements: usize,
    expanse: Pixel,
}

impl Solution {
    fn count_light_pixels_after_enhancements(&mut self, enhancements: usize) -> usize {
        self.enhance_until(enhancements);

        self.count_light_pixels()
    }

    fn count_light_pixels(&self) -> usize {
        self.curr_grid
            .cells()
            .iter()
            .filter(|pixel| pixel.is_light())
            .count()
    }

    fn as_string(&self) -> String {
        self.curr_grid.clone().into()
    }

    fn enhance_until(&mut self, enhancements: usize) {
        for _ in self.enhancements..enhancements {
            if !self.enhance_internal() {
                break;
            }
        }
    }

    fn enhance_internal(&mut self) -> bool {
        // next_grid has an extra row and column on each side
        let next_dimensions: IVec2 = self.curr_grid.dimensions() + 2_i32 * IVec2::ONE;

        self.next_grid.resize(next_dimensions, Pixel::Dark);

        let mut enhanced: bool = false;

        for (next_index, next_pixel) in self.next_grid.cells_mut().iter_mut().enumerate() {
            let next_pos: IVec2 = IVec2::new(
                next_index as i32 % next_dimensions.x,
                next_index as i32 / next_dimensions.x,
            );
            let curr_pos: IVec2 = next_pos - IVec2::ONE;
            let algorithm_index: usize = Self::index(&self.curr_grid, curr_pos, self.expanse);
            let next_pixel_val: Pixel = self.image_enhancement_algorithm.0[algorithm_index].into();

            if !enhanced {
                enhanced =
                    next_pixel_val != self.curr_grid.get(curr_pos).copied().unwrap_or_default();
            }

            *next_pixel = next_pixel_val;
        }

        swap(&mut self.curr_grid, &mut self.next_grid);
        self.enhancements += 1_usize;
        self.expanse = self.image_enhancement_algorithm.0[if self.expanse.is_light() {
            ImageEnhancementAlgorithm::LEN - 1_usize
        } else {
            0_usize
        }]
        .into();

        enhanced
    }

    fn index(grid: &Grid2D<Pixel>, pos: IVec2, expanse: Pixel) -> usize {
        let expanse_is_light: bool = expanse.is_light();

        ImageEnhancementAlgorithm::DELTAS
            .iter()
            .copied()
            .enumerate()
            .fold(0_usize, |index, (bit_index, delta)| {
                if grid
                    .get(pos + delta)
                    .copied()
                    .map_or(expanse_is_light, Pixel::is_light)
                {
                    index | (1_usize << bit_index)
                } else {
                    index
                }
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            separated_pair(ImageEnhancementAlgorithm::parse, line_ending, Grid2D::parse),
            |(image_enhancement_algorithm, curr_grid)| Self {
                image_enhancement_algorithm,
                curr_grid,
                next_grid: Grid2D::empty(IVec2::ZERO),
                enhancements: 0_usize,
                expanse: Pixel::Dark,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.count_light_pixels_after_enhancements(2_usize));

        if args.verbose {
            println!("self.as_string():\n{}", self.as_string());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.count_light_pixels_after_enhancements(50_usize));

        if args.verbose {
            println!("self.as_string():\n{}", self.as_string());
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

    const SOLUTION_STRS: &[&str] = &[
        "..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..##\n\
        #..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###\n\
        .######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#.\n\
        .#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#.....\n\
        .#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#..\n\
        ...####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.....\n\
        ..##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#\n\
        \n\
        #..#.\n\
        #....\n\
        ##..#\n\
        ..#..\n\
        ..###\n",
        include_str!("../../input/y2021/d20.txt"),
    ];

    const GRID_STRS: &[&str] = &[
        "\
        #..#.\n\
        #....\n\
        ##..#\n\
        ..#..\n\
        ..###\n",
        "\
        .##.##.\n\
        #..#.#.\n\
        ##.#..#\n\
        ####..#\n\
        .#..##.\n\
        ..##..#\n\
        ...#.#.\n",
        "\
        .......#.\n\
        .#..#.#..\n\
        #.#...###\n\
        #...##.#.\n\
        #.....#.#\n\
        .#.#####.\n\
        ..#.#####\n\
        ...##.##.\n\
        ....###..\n",
    ];

    fn solution() -> &'static Solution {
        use Pixel::{Dark as D, Light as L};

        macro_rules! image_enhancement_algorithm {
            [ $( $reverse_block:literal, )* ] => {
                ImageEnhancementAlgorithm(BitArray::new([ $(
                    reverse($reverse_block),
                )* ]))
            };
        }

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            image_enhancement_algorithm: image_enhancement_algorithm![
                0b_0010100111110101010111011000001110110100111011110011111001000010_usize,
                0b_0100110011100111111011100011110010011111001100101111100011010100_usize,
                0b_1011001010000001011101111110111011110001011011001001001111100000_usize,
                0b_1010000111001011000000100000100100100110010001101111110111101111_usize,
                0b_0101000100000001001010100011110110100000010010001101011001000110_usize,
                0b_1011001110100000010100000001010101111011101100010000011110100100_usize,
                0b_1011010000110010111100001100011001000100000010100000001000000011_usize,
                0b_0011110010001010100011001010011100111110000000010011110000001001_usize,
            ],
            curr_grid: Grid2D::try_from_cells_and_dimensions(
                vec![
                    L, D, D, L, D, L, D, D, D, D, L, L, D, D, L, D, D, L, D, D, D, D, L, L, L,
                ],
                IVec2::new(5_i32, 5_i32),
            )
            .unwrap(),
            next_grid: Grid2D::empty(IVec2::ZERO),
            enhancements: 0_usize,
            expanse: D,
        })
    }

    fn reverse(mut block: usize) -> usize {
        let mut rev_block: usize = 0_usize;

        for _ in 0_u32..usize::BITS {
            rev_block = (rev_block << 1_u32) | (block & 1_usize);
            block >>= 1_u32;
        }

        rev_block
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(
            Solution::try_from(SOLUTION_STRS[0_usize]).as_ref(),
            Ok(solution())
        );
    }

    #[test]
    fn test_try_as_string() {
        assert_eq!(solution().as_string(), GRID_STRS[0_usize].to_owned())
    }

    #[test]
    fn test_enhance() {
        let mut solution: Solution = solution().clone();

        solution.enhance_until(1_usize);
        assert_eq!(solution.as_string(), GRID_STRS[1_usize].to_owned());
        solution.enhance_until(2_usize);
        assert_eq!(solution.as_string(), GRID_STRS[2_usize].to_owned());
    }

    #[test]
    fn test_count_light_pixels() {
        let mut solution: Solution = solution().clone();

        solution.enhance_until(2_usize);
        assert_eq!(solution.count_light_pixels(), 35_usize);
        solution.enhance_until(50_usize);
        assert_eq!(solution.count_light_pixels(), 3351_usize);

        let mut solution: Solution = Solution::try_from(SOLUTION_STRS[1_usize]).unwrap();

        solution.enhance_until(2_usize);
        assert!(solution.count_light_pixels() < 5860_usize);
    }

    #[test]
    fn test_count_light_pixels_after_enhancements() {
        let mut solution: Solution = solution().clone();

        assert_eq!(
            solution.count_light_pixels_after_enhancements(2_usize),
            35_usize
        );
        assert_eq!(
            solution.count_light_pixels_after_enhancements(50_usize),
            3351_usize
        );
    }
}
