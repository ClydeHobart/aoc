use {crate::*, glam::IVec2, std::fmt::Debug};

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
struct Height(u8);

#[allow(dead_code)]
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct CharIsNotAsciiDigit(char);

impl TryFrom<char> for Height {
    type Error = CharIsNotAsciiDigit;

    fn try_from(height_char: char) -> Result<Self, Self::Error> {
        if height_char.is_ascii_digit() {
            Ok(Height(height_char as u8 - ZERO_OFFSET))
        } else {
            Err(CharIsNotAsciiDigit(height_char))
        }
    }
}

#[derive(Debug, Default)]
struct IsVisible(bool);

#[derive(Default)]
struct ComputeIsVisible {
    max_row_height: u8,
}

impl GridVisitor for ComputeIsVisible {
    type Old = Height;
    type New = IsVisible;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        old_grid: &Grid2D<Self::Old>,
        _rev_dir: Direction,
        pos: IVec2,
    ) {
        if !new.0 {
            new.0 = old_grid.is_border(pos) || old.0 > self.max_row_height
        }

        self.max_row_height = self.max_row_height.max(old.0);
    }
}

#[derive(Debug)]
struct ScenicScore(u32);

impl Default for ScenicScore {
    fn default() -> Self {
        Self(1_u32)
    }
}

#[derive(Default)]
struct ComputeScenicScore {
    height_to_viewing_distance: [u32; 10_usize],
}

impl GridVisitor for ComputeScenicScore {
    type Old = Height;
    type New = ScenicScore;

    fn visit_cell(
        &mut self,
        new: &mut Self::New,
        old: &Self::Old,
        _old_grid: &Grid2D<Self::Old>,
        _rev_dir: Direction,
        _pos: IVec2,
    ) {
        let height_index: usize = old.0 as usize;

        if new.0 != 0_u32 {
            new.0 *= self.height_to_viewing_distance[height_index];
        }

        // All cells not taller than this cell can now only see this one
        self.height_to_viewing_distance[..=height_index].fill(1_u32);

        if let Some(taller_slice) = self
            .height_to_viewing_distance
            .get_mut(height_index + 1_usize..)
        {
            for taller_viewing_distance in taller_slice {
                *taller_viewing_distance += 1_u32;
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<Height>);

impl Solution {
    fn compute_is_visible_count(&self) -> usize {
        ComputeIsVisible::visit_grid(&self.0)
            .cells()
            .iter()
            .filter(|is_visible: &&IsVisible| is_visible.0)
            .count()
    }

    fn compute_scenic_score_max(&self) -> u32 {
        ComputeScenicScore::visit_grid(&self.0)
            .cells()
            .iter()
            .map(|scenic_score: &ScenicScore| scenic_score.0)
            .max()
            .unwrap_or_default()
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_is_visible_count());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_scenic_score_max());
    }
}

impl<'i> TryFrom<&'i str> for Solution {
    type Error = GridParseError<'i, CharIsNotAsciiDigit>;

    fn try_from(input: &'i str) -> Result<Self, Self::Error> {
        Ok(Self(Grid2D::<Height>::try_from(input)?))
    }
}

#[cfg(test)]
mod tests {
    use {super::*, std::sync::OnceLock};

    const HEIGHT_GRID_STR: &str = "\
        30373\n\
        25512\n\
        65332\n\
        33549\n\
        35390";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            macro_rules! heights { [ $( $height:expr ),* ] => { vec![ $( Height($height), )* ] }; }

            const DIMENSIONS: IVec2 = IVec2::new(5_i32, 5_i32);

            Solution(
                Grid2D::try_from_cells_and_dimensions(
                    heights![
                        3, 0, 3, 7, 3, 2, 5, 5, 1, 2, 6, 5, 3, 3, 2, 3, 3, 5, 4, 9, 3, 5, 3, 9, 0
                    ],
                    DIMENSIONS,
                )
                .unwrap_or_else(|| Grid2D::empty(DIMENSIONS)),
            )
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(HEIGHT_GRID_STR).as_ref(), Ok(solution()))
    }

    #[test]
    fn test_compute_is_visible_count() {
        assert_eq!(solution().compute_is_visible_count(), 21_usize);
    }

    #[test]
    fn test_compute_scenic_score_max() {
        assert_eq!(solution().compute_scenic_score_max(), 8_u32);
    }
}
