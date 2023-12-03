use {
    crate::*,
    glam::IVec2,
    nom::{character::complete::satisfy, combinator::map, error::Error, Err, IResult},
    std::{
        collections::{HashMap, HashSet},
        fmt::{Debug, Formatter, Result as FmtResult, Write},
    },
};

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Copy)]
struct Cell(u8);

impl Debug for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char(self.0 as char)
    }
}

impl Default for Cell {
    fn default() -> Self {
        Cell(b'.')
    }
}

// SAFETY: `Cell` is only constructed from valid ASCII bytes.
unsafe impl IsValidAscii for Cell {}

impl Parse for Cell {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(satisfy(|c| c.is_ascii()), |c| Self(c as u8))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct CellGrid(Grid2D<Cell>);

impl CellGrid {
    fn get_char(&self, pos: IVec2) -> char {
        self.0.get(pos).unwrap().0 as char
    }

    fn is_digit(&self, pos: IVec2) -> bool {
        self.get_char(pos).is_ascii_digit()
    }

    fn is_symbol(&self, pos: IVec2) -> bool {
        let c: char = self.get_char(pos);

        !c.is_ascii_alphanumeric() && c != '.'
    }

    fn write_number(&mut self, mut number: u32, pos: IVec2) {
        let digits_len: usize = digits(number);

        let mut digits: [Cell; U32_DIGITS as usize] = [Cell(b'0'); U32_DIGITS as usize];
        let mut digit_index: usize = 0_usize;

        while number > 0_u32 {
            digits[digit_index] = Cell((number % 10_u32) as u8 + b'0');
            number /= 10_u32;
            digit_index += 1_usize;
        }

        for (digit, pos) in digits[..digits_len]
            .iter()
            .rev()
            .copied()
            .zip(CellIter2D::until_boundary(&self.0, pos, Direction::East))
        {
            self.write_cell(digit, pos);
        }
    }

    fn write_cell(&mut self, cell: Cell, pos: IVec2) {
        *self.0.get_mut(pos).unwrap() = cell;
    }
}

impl From<Solution> for CellGrid {
    fn from(value: Solution) -> Self {
        let mut max_pos: IVec2 = IVec2::ZERO;

        for (number, pos) in value.iter_numbers() {
            max_pos = max_pos.max(IVec2::new(pos.x + digits(number) as i32 - 1_i32, pos.y));
        }

        for (_, pos) in value.iter_symbols() {
            max_pos = max_pos.max(pos);
        }

        let mut cell_grid: CellGrid = CellGrid(Grid2D::default(max_pos + IVec2::ONE));

        for (number, pos) in value.iter_numbers() {
            cell_grid.write_number(number, pos);
        }

        for (symbol, pos) in value.iter_symbols() {
            cell_grid.write_cell(Cell(symbol), pos);
        }

        cell_grid
    }
}

impl Parse for CellGrid {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(Grid2D::parse, Self)(input)
    }
}

enum GearNumbers {
    None,
    One(u32),
    Two(u32, u32),
    TooMany,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Gear {
    number_a: u32,
    number_b: u32,

    #[cfg(test)]
    pos: IVec2,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    numbers: Vec<u32>,
    symbols: Vec<u8>,
    positions: Vec<IVec2>,
}

impl Solution {
    fn iter_numbers(&self) -> impl Iterator<Item = (u32, IVec2)> + '_ {
        self.numbers
            .iter()
            .copied()
            .zip(self.positions.iter().copied())
    }

    fn iter_symbols(&self) -> impl Iterator<Item = (u8, IVec2)> + '_ {
        self.symbols
            .iter()
            .copied()
            .zip(self.positions[self.numbers.len()..].iter().copied())
    }

    fn iter_positions_adjacent_to_number(number: u32, pos: IVec2) -> impl Iterator<Item = IVec2> {
        let digits: i32 = digits(number) as i32;
        let nw_corner: IVec2 = pos - IVec2::ONE;
        let se_corner: IVec2 = IVec2::new(pos.x + digits, pos.y + 1_i32);
        let ne_corner: IVec2 = IVec2::new(se_corner.x, nw_corner.y);
        let sw_corner: IVec2 = IVec2::new(nw_corner.x, se_corner.y);

        CellIter2D::try_from(nw_corner..ne_corner)
            .unwrap()
            .chain(CellIter2D::try_from(ne_corner..se_corner).unwrap())
            .chain(
                CellIter2D::try_from(se_corner..sw_corner)
                    .unwrap()
                    .chain(CellIter2D::try_from(sw_corner..nw_corner).unwrap()),
            )
    }

    fn iter_part_numbers(&self) -> impl Iterator<Item = u32> + '_ {
        let mut symbols_positions: HashSet<IVec2> = HashSet::with_capacity(self.symbols.len());

        for (_, pos) in self.iter_symbols() {
            symbols_positions.insert(pos);
        }

        self.iter_numbers().filter_map(move |(number, pos)| {
            if Self::iter_positions_adjacent_to_number(number, pos)
                .any(|pos| symbols_positions.contains(&pos))
            {
                Some(number)
            } else {
                None
            }
        })
    }

    fn iter_gears(&self) -> impl Iterator<Item = Gear> {
        let mut gear_numbers: HashMap<IVec2, GearNumbers> = self
            .iter_symbols()
            .filter_map(|(symbol, pos)| {
                if symbol == b'*' {
                    Some((pos, GearNumbers::None))
                } else {
                    None
                }
            })
            .collect();

        for (number, pos) in self.iter_numbers().flat_map(|(number, pos)| {
            Self::iter_positions_adjacent_to_number(number, pos).map(move |pos| (number, pos))
        }) {
            if let Some(gear_numbers) = gear_numbers.get_mut(&pos) {
                *gear_numbers = match gear_numbers {
                    GearNumbers::None => GearNumbers::One(number),
                    GearNumbers::One(number_a) => GearNumbers::Two(*number_a, number),
                    _ => GearNumbers::TooMany,
                };
            }
        }

        gear_numbers
            .into_iter()
            .filter_map(|(_pos, gear_numbers)| match gear_numbers {
                GearNumbers::Two(number_a, number_b) => Some(Gear {
                    number_a,
                    number_b,

                    #[cfg(test)]
                    pos: _pos,
                }),
                _ => None,
            })
    }

    fn sum_part_numbers(&self) -> u32 {
        self.iter_part_numbers().sum()
    }

    fn sum_gear_ratios(&self) -> u32 {
        self.iter_gears()
            .map(|gear| gear.number_a * gear.number_b)
            .sum()
    }
}

impl From<CellGrid> for Solution {
    fn from(value: CellGrid) -> Self {
        let mut solution: Self = Self::default();

        for pos in value.0.iter_positions().filter(|pos| {
            let prev_pos: IVec2 = *pos - IVec2::X;

            value.is_digit(*pos) && (value.0.get(prev_pos).is_none() || !value.is_digit(prev_pos))
        }) {
            let mut number: u32 = 0_u32;

            for digit_pos in CellIter2D::until_boundary(&value.0, pos, Direction::East) {
                let c: char = value.get_char(digit_pos);

                if !c.is_ascii_digit() {
                    break;
                }

                number = number * 10_u32 + (c as u8 - b'0') as u32;
            }

            solution.numbers.push(number);
            solution.positions.push(pos);
        }

        for pos in value.0.iter_positions().filter(|pos| value.is_symbol(*pos)) {
            solution.symbols.push(value.get_char(pos) as u8);
            solution.positions.push(pos);
        }

        solution
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(CellGrid::parse, Self::from)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_part_numbers());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_gear_ratios());
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
        std::{mem::swap, sync::OnceLock},
    };

    const SOLUTION_STR: &'static str = "\
        467..114..\n\
        ...*......\n\
        ..35..633.\n\
        ......#...\n\
        617*......\n\
        .....+.58.\n\
        ..592.....\n\
        ......755.\n\
        ...$.*....\n\
        .664.598..\n";

    fn solution() -> &'static Solution {
        let v = IVec2::new;

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            let mut solution: Solution = Solution::default();

            for (number, pos) in [
                (467, v(0, 0)),
                (114, v(5, 0)),
                (35, v(2, 2)),
                (633, v(6, 2)),
                (617, v(0, 4)),
                (58, v(7, 5)),
                (592, v(2, 6)),
                (755, v(6, 7)),
                (664, v(1, 9)),
                (598, v(5, 9)),
            ] {
                solution.numbers.push(number);
                solution.positions.push(pos);
            }

            for (symbol, pos) in [
                (b'*', v(3, 1)),
                (b'#', v(6, 3)),
                (b'*', v(3, 4)),
                (b'+', v(5, 5)),
                (b'$', v(3, 8)),
                (b'*', v(5, 8)),
            ] {
                solution.symbols.push(symbol);
                solution.positions.push(pos);
            }

            solution
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_part_numbers() {
        assert_eq!(
            solution().iter_part_numbers().collect::<Vec<u32>>(),
            vec![467_u32, 35_u32, 633_u32, 617_u32, 592_u32, 755_u32, 664_u32, 598_u32]
        );
    }

    #[test]
    fn test_sum_part_numbers() {
        assert_eq!(solution().sum_part_numbers(), 4361_u32);
    }

    #[test]
    fn test_iter_gears() {
        let mut gears: Vec<Gear> = solution().iter_gears().collect();

        for gear in gears.iter_mut() {
            if gear.number_a > gear.number_b {
                swap(&mut gear.number_a, &mut gear.number_b);
            }
        }

        gears.sort_by(|gear_a, gear_b| {
            gear_a.pos.x.cmp(&gear_b.pos.x).then_with(|| {
                gear_a.pos.y.cmp(&gear_b.pos.y).then_with(|| {
                    gear_a
                        .number_a
                        .cmp(&gear_b.number_a)
                        .then_with(|| gear_a.number_b.cmp(&gear_b.number_b))
                })
            })
        });

        assert_eq!(
            gears,
            vec![
                Gear {
                    number_a: 35_u32,
                    number_b: 467_u32,
                    pos: IVec2::new(3_i32, 1_i32),
                },
                Gear {
                    number_a: 598_u32,
                    number_b: 755_u32,
                    pos: IVec2::new(5_i32, 8_i32),
                },
            ]
        );
    }

    #[test]
    fn test_sum_gear_ratios() {
        assert_eq!(solution().sum_gear_ratios(), 467835_u32);
    }
}
