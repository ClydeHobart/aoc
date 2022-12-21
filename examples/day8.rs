use {aoc_2022::*, glam::IVec2, std::fmt::Debug};

#[derive(Debug)]
struct Height(u8);

#[derive(Debug)]
struct CharIsNotAsciiDigit(char);

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

trait IsVisibleTrait {
    fn count(&self) -> usize;
}

impl IsVisibleTrait for Grid2D<IsVisible> {
    fn count(&self) -> usize {
        self.cells()
            .iter()
            .filter(|is_visible: &&IsVisible| is_visible.0)
            .count()
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

trait ScenicScoreTrait {
    fn max(&self) -> u32;
}

impl ScenicScoreTrait for Grid2D<ScenicScore> {
    fn max(&self) -> u32 {
        self.cells()
            .iter()
            .map(|scenic_score: &ScenicScore| scenic_score.0)
            .max()
            .unwrap_or_default()
    }
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day8.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(
                input_file_path,
                |input: &str| match Grid2D::<Height>::try_from(input) {
                    Ok(height_grid) => {
                        println!(
                            "ComputeIsVisible::visit_grid(&height_grid).count() == {}\n\
                            ComputeScenicScore::visit_grid(&height_grid).max() == {}",
                            ComputeIsVisible::visit_grid(&height_grid).count(),
                            ComputeScenicScore::visit_grid(&height_grid).max()
                        );
                    }
                    Err(error) => {
                        panic!("{error:#?}")
                    }
                },
            )
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}

#[test]
fn test() {
    const HEIGHT_GRID_STR: &str = "\
        30373\n\
        25512\n\
        65332\n\
        33549\n\
        35390";

    match Grid2D::<Height>::try_from(HEIGHT_GRID_STR) {
        Ok(height_grid) => {
            let is_visible_grid: Grid2D<IsVisible> = ComputeIsVisible::visit_grid(&height_grid);

            assert_eq!(
                is_visible_grid.count(),
                21_usize,
                "height_grid: {height_grid:#?}\n\nis_visible_grid: {is_visible_grid:#?}"
            );

            let scenic_score_grid: Grid2D<ScenicScore> =
                ComputeScenicScore::visit_grid(&height_grid);

            assert_eq!(
                scenic_score_grid.max(),
                8_u32,
                "height_grid: {height_grid:#?}\n\ncenic_score_grid: {scenic_score_grid:#?}"
            );
        }
        Err(error) => panic!("{error:#?}"),
    }
}
