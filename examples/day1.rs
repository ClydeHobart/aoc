use {
    aoc_2022::*,
    std::{mem::MaybeUninit, str::FromStr},
};

/// Iterate over the sume of calories held by each elf
///
/// # Arguments
///
/// * `input` - The full input string with individual elf inventories delineated by `"\n\n"`, and
///   individual calorie quantities delineated by `'\n'`
fn iter_elf_calories(input: &str) -> impl Iterator<Item = u32> + '_ {
    input.split("\n\n").map(|elf_calories_str: &str| -> u32 {
        elf_calories_str
            .split('\n')
            .map(|calories_str: &str| -> u32 {
                match u32::from_str(calories_str) {
                    Ok(calories) => calories,
                    Err(err) => {
                        eprintln!(
                            "Encountered ParseIntError {} while parsing \"{}\"",
                            err, calories_str
                        );

                        0_u32
                    }
                }
            })
            .sum()
    })
}

/// Returns the maximum number of calories held by an elf as described by
/// https://adventofcode.com/2022/day/1
///
/// # Arguments
///
/// * `input` - The full input string with individual elf inventories delineated by `"\n\n"`, and
///   individual calorie quantities delineated by `'\n'`
fn calories_sum_of_max_calories_elf(input: &str) -> u32 {
    match iter_elf_calories(input).max() {
        Some(calories_sum_of_max_calories_elf) => calories_sum_of_max_calories_elf,
        None => {
            eprintln!("Iterator yielded no maximum. Were there any lines?");

            0
        }
    }
}

/// Returns the sum of the calories held by the N elves with the highest number of calories as
/// described by https://adventofcode.com/2022/day/1#part2
///
/// # Arguments
///
/// * `input` - The full input string with individual elf inventories delineated by `"\n\n"`, and
///   individual calorie quantities delineated by `'\n'`
fn calories_sum_of_top_n_calories_elves<const N: usize>(input: &str) -> u32 {
    // SAFETY: `0_u32` is 4 consecutive `0_u8` bytes in memory, and `MaybeUninit` guarantees
    // alignment
    let mut top_n_calories: [u32; N] = unsafe { MaybeUninit::zeroed().assume_init() };
    let n: usize = top_n_calories.len();

    if n == 0 {
        eprintln!("calories_sum_of_top_n_calories_elves() called with N == 0, terminating early");

        return 0;
    }

    let n_minus_1 = n - 1_usize;

    for elf_calories in iter_elf_calories(input) {
        for index in 0_usize..n {
            if elf_calories > top_n_calories[index] {
                top_n_calories[n_minus_1] = elf_calories;
                top_n_calories[index..].rotate_right(1_usize);

                break;
            }
        }
    }

    top_n_calories.into_iter().sum()
}

/// Returns the sum of the calories held by the three elves with the highest number of calories as
/// described by https://adventofcode.com/2022/day/1#part2
///
/// # Arguments
///
/// * `input` - The full input string with individual elf inventories delineated by `"\n\n"`, and
///   individual calorie quantities delineated by `'\n'`
fn calories_sum_of_top_3_calories_elves(input: &str) -> u32 {
    calories_sum_of_top_n_calories_elves::<3_usize>(input)
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day1.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                println!(
                    "calories_sum_of_max_calories_elf == {}\n\
                    calories_sum_of_top_3_calories_elves == {}",
                    calories_sum_of_max_calories_elf(input),
                    calories_sum_of_top_3_calories_elves(input)
                );
            })
        }
    {
        eprintln!(
            "Encountered error {} when opening file \"{}\"",
            err, input_file_path
        );
    }
}
