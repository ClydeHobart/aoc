use {
    self::trinary::*,
    crate::*,
    nom::{combinator::map, error::Error, Err, IResult},
};

/* --- Day 19: An Elephant Named Joseph ---

The Elves contact you over a highly secure emergency channel. Back at the North Pole, the Elves are busy misunderstanding White Elephant parties.

Each Elf brings a present. They all sit in a circle, numbered starting with position 1. Then, starting with the first Elf, they take turns stealing all the presents from the Elf to their left. An Elf with no presents is removed from the circle and does not take turns.

For example, with five Elves (numbered 1 to 5):

  1
5   2
 4 3

    Elf 1 takes Elf 2's present.
    Elf 2 has no presents and is skipped.
    Elf 3 takes Elf 4's present.
    Elf 4 has no presents and is also skipped.
    Elf 5 takes Elf 1's two presents.
    Neither Elf 1 nor Elf 2 have any presents, so both are skipped.
    Elf 3 takes Elf 5's three presents.

So, with five Elves, the Elf that sits starting in position 3 gets all the presents.

With the number of Elves given in your puzzle input, which Elf gets all the presents?

--- Part Two ---

Realizing the folly of their present-exchange rules, the Elves agree to instead steal presents from the Elf directly across the circle. If two Elves are across the circle, the one on the left (from the perspective of the stealer) is stolen from. The other rules remain unchanged: Elves with no presents are removed from the circle entirely, and the other elves move in slightly to keep the circle evenly spaced.

For example, with five Elves (again numbered 1 to 5):

    The Elves sit in a circle; Elf 1 goes first:

      1
    5   2
     4 3

    Elves 3 and 4 are across the circle; Elf 3's present is stolen, being the one to the left. Elf 3 leaves the circle, and the rest of the Elves move in:

      1           1
    5   2  -->  5   2
     4 -          4

    Elf 2 steals from the Elf directly across the circle, Elf 5:

      1         1
    -   2  -->     2
      4         4

    Next is Elf 4 who, choosing between Elves 1 and 2, steals from Elf 1:

     -          2
        2  -->
     4          4

    Finally, Elf 2 steals from Elf 4:

     2
        -->  2
     -

So, with five Elves, the Elf that sits starting in position 2 gets all the presents.

With the number of Elves given in your puzzle input, which Elf now gets all the presents? */

mod trinary {
    fn ilog3(value: i32) -> i32 {
        value.checked_ilog(3_i32).unwrap_or_default() as i32
    }

    /// most sig trit mask
    fn mstm(value: i32) -> i32 {
        3_i32.pow(ilog3(value) as u32)
    }

    pub fn last_elf_with_presents_index(value: i32) -> i32 {
        let value: i32 = value - 1_i32;
        let mstm: i32 = mstm(value);
        let lstm: i32 = mstm - 1_i32;
        let vmmstm: i32 = value - mstm;
        let sd: i32 = 2_i32 * (vmmstm - lstm) + lstm;
        let ltdmstm: bool = value < 2_i32 * mstm;

        (if ltdmstm { vmmstm } else { sd }).max(0_i32)
    }

    #[cfg(test)]
    mod tests {
        use {
            super::*,
            std::{cmp::Ordering, ops::Range},
        };

        /// least sig trits mask
        fn lstm(value: i32) -> i32 {
            mstm(value) - 1_i32
        }

        /// value minus most sig trit mask
        fn vmmstm(value: i32) -> i32 {
            value - mstm(value)
        }

        /// scaled diff
        fn sd(value: i32) -> i32 {
            let lstm: i32 = lstm(value);

            2_i32 * (vmmstm(value) - lstm) + lstm
        }

        /// double most sig trit mask
        fn dmstm(value: i32) -> i32 {
            2_i32 * mstm(value)
        }

        /// less than double most sig trit mask
        fn ltdmstm(value: i32) -> i32 {
            (value < 2_i32 * mstm(value)) as i32
        }

        /// value minus most sig trit mask or scaled diff
        fn vmmstmosd(value: i32) -> i32 {
            if ltdmstm(value) == 1_i32 {
                vmmstm(value)
            } else {
                sd(value)
            }
        }

        /// shifted value minus most sig trit mask or scaled diff
        fn s_vmmstmosd(value: i32) -> i32 {
            vmmstmosd(value - 1_i32)
        }

        fn target(value: i32) -> i32 {
            const TARGET_VALUES: &'static [i32] = &[
                0, 0, 2, 0, 1, 2, 4, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22,
                24, 26, 0, 1, 2, 3, 4,
            ];
            const OFFSET: i32 = 1_i32;

            let index: i32 = value - OFFSET;

            usize::try_from(index)
                .ok()
                .map(|index| TARGET_VALUES.get(index))
                .flatten()
                .copied()
                .unwrap_or(-1_i32)
        }

        fn printed_width(value: i32) -> usize {
            match value.cmp(&0_i32) {
                Ordering::Less => (-value).ilog10() as usize + 2_usize,
                Ordering::Equal => 1_usize,
                Ordering::Greater => value.ilog10() as usize + 1_usize,
            }
        }

        macro_rules! funcs {
            [ $( $func:ident, )* ] => {
                [ $( ($func, stringify!($func)), )* ]
            }
        }

        fn print_table(funcs: &[(fn(i32) -> i32, &'static str)], domain: Range<i32>) {
            assert!(!funcs.is_empty());
            assert!(!domain.is_empty());

            const FUNCS_COL_HEADER: &'static str = "Funcs";

            let funcs_len: usize = funcs.len();
            let funcs_col_width: usize = funcs
                .iter()
                .map(|(_, name)| name.len())
                .chain([FUNCS_COL_HEADER.len()])
                .max()
                .unwrap_or_default();
            let domain_len: usize = domain.len();
            let values: Vec<i32> = funcs
                .iter()
                .flat_map(|(func, _)| domain.clone().map(func))
                .collect();
            let col_widths: Vec<usize> = domain
                .clone()
                .enumerate()
                .map(|(col, value)| {
                    (0_usize..funcs_len)
                        .map(|row| printed_width(values[row * domain_len + col]))
                        .chain([printed_width(value)])
                        .max()
                        .unwrap_or_default()
                })
                .collect();

            print!("{0:>1$}", FUNCS_COL_HEADER, funcs_col_width);

            for (col, value) in domain.enumerate() {
                print!("|{0:>1$}", value, col_widths[col]);
            }

            println!(
                "\n{0:-<1$}",
                "",
                funcs_col_width + domain_len + col_widths.iter().copied().sum::<usize>()
            );

            for ((_, name), row_values) in funcs.iter().zip(values.chunks(domain_len)) {
                print!("{0:>1$}", name, funcs_col_width);

                for (col, value) in row_values.iter().enumerate() {
                    print!(
                        "{0}{1:>2$}",
                        if col == 0_usize { '|' } else { ' ' },
                        value,
                        col_widths[col]
                    );
                }

                println!();
            }
        }

        #[test]
        fn test_print_table() {
            print_table(
                &funcs![
                    mstm,
                    dmstm,
                    ltdmstm,
                    vmmstm,
                    lstm,
                    sd,
                    vmmstmosd,
                    s_vmmstmosd,
                    last_elf_with_presents_index,
                    target,
                ],
                1_i32..29_i32,
            );
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
pub struct Solution(u32);

impl Solution {
    fn last_elf_with_presents(self, verbose: bool) -> u32 {
        self.0.checked_ilog2().map_or(0_u32, |shl| {
            let msb: u32 = 1_u32 << shl;
            let remaining: u32 = self.0 - msb;
            let index: u32 = remaining << 1_u32;
            let last_elf: u32 = index + 1_u32;

            if verbose {
                println!(
                    " 0b{:032b}\n\
                     -0b{:032b}\n\
                     ---{:-<32}\n \
                      0b{:032b}\n\
                     *  {:32}\n\
                     ---{:-<32}\n \
                      0b{:032b}\n\
                     +  {:32}\n\
                     ---{:-<32}\n \
                      0b{:032b}",
                    self.0, msb, "", remaining, 2, "", index, 1, "", last_elf
                );
            }

            last_elf
        })
    }

    fn updated_last_elf_with_presents(self) -> u32 {
        last_elf_with_presents_index(self.0 as i32) as u32 + 1_u32
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.last_elf_with_presents(args.verbose));
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.updated_last_elf_with_presents());
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

    const SOLUTION_STR: &'static str = "5\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution(5_u32))
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_last_elf_with_presents() {
        assert_eq!(solution().last_elf_with_presents(false), 3_u32);
    }
}
