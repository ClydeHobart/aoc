pub use {graph::*, grid::*};

use {
    clap::Parser,
    memmap::Mmap,
    nom::{
        bytes::complete::tag,
        combinator::{map, map_res, rest},
        sequence::tuple,
        IResult,
    },
    std::{
        any::type_name,
        cmp::{max, min},
        fmt::Debug,
        fs::File,
        io::{Error as IoError, ErrorKind, Result as IoResult},
        ops::{Deref, DerefMut},
        str::{from_utf8, FromStr, Utf8Error},
    },
};

mod graph;
mod grid;

#[derive(Debug, Parser)]
pub struct QuestionArgs {
    /// Print extra information, if there is any
    #[arg(short, long, default_value_t)]
    pub verbose: bool,
}

/// Arguments for program execution
#[derive(Debug, Parser)]
pub struct Args {
    /// Input file path
    #[arg(short, long, default_value_t)]
    input_file_path: String,

    /// The year to run
    #[arg(short, long)]
    pub year: u16,

    /// The day to run
    #[arg(short, long, value_parser = clap::value_parser!(u8).range(0..=25))]
    pub day: u8,

    /// The question to run, both if omitted
    #[arg(short, long, default_value_t, value_parser = clap::value_parser!(u8).range(0..=2))]
    pub question: u8,

    #[command(flatten)]
    pub question_args: QuestionArgs,
}

impl Args {
    fn try_to_intermediate<I>(&self) -> Option<I>
    where
        I: for<'a> TryFrom<&'a str>,
        for<'a> <I as TryFrom<&'a str>>::Error: Debug,
    {
        let default_file_path: String;
        let file_path: &str = if self.input_file_path.is_empty() {
            default_file_path = format!("input/y{}/d{}.txt", self.year, self.day);

            &default_file_path
        } else {
            &self.input_file_path
        };

        // SAFETY: This isn't truly safe, we're just hoping nobody touches our file before we're
        // done parsing it
        unsafe {
            open_utf8_file(file_path, |s| {
                s.try_into().map_or_else(
                    |error| {
                        eprintln!(
                            "Failed to convert file \"{file_path}\" to type {}:\n{error:#?}",
                            type_name::<I>()
                        );

                        None
                    },
                    Some,
                )
            })
        }
        .unwrap_or_else(|error| {
            eprintln!("Failed to open UTF-8 file \"{file_path}\":\n{error}");

            None
        })
    }
}

pub trait RunQuestions
where
    Self: Sized + for<'a> TryFrom<&'a str>,
    for<'a> <Self as TryFrom<&'a str>>::Error: Debug,
{
    fn q2_internal(&mut self, args: &QuestionArgs);
    fn q1_internal(&mut self, args: &QuestionArgs);

    fn q1(args: &Args) {
        if let Some(mut intermediate) = args.try_to_intermediate::<Self>() {
            intermediate.q1_internal(&args.question_args);
        }
    }

    fn q2(args: &Args) {
        if let Some(mut intermediate) = args.try_to_intermediate::<Self>() {
            intermediate.q2_internal(&args.question_args);
        }
    }

    fn both(args: &Args) {
        if let Some(mut intermediate) = args.try_to_intermediate::<Self>() {
            intermediate.q1_internal(&args.question_args);
            intermediate.q2_internal(&args.question_args);
        }
    }
}

#[derive(Clone)]
pub struct Day {
    pub q1: fn(&Args),
    pub q2: fn(&Args),
    pub both: fn(&Args),
}

impl Day {
    fn run(&self, args: &Args) {
        match args.question {
            0 => (self.both)(args),
            1 => (self.q1)(args),
            2 => (self.q2)(args),
            question => unreachable!(
                "A valid Args will have a question value in the range 0..=2, but {question} was \
                encountered.\n\
                Args:\n\
                {args:#?}"
            ),
        }
    }
}

pub struct DayParams<'a> {
    pub string: &'a str,
    pub option: Option<u8>,
    pub day: Day,
}

pub struct Year {
    days: Vec<Option<Day>>,
    min: u8,
}

fn parse_tagged_int<'i, I: FromStr>(t: &str, input: &'i str) -> IResult<&'i str, I> {
    map(tuple((tag(t), map_res(rest, I::from_str))), |(_, i)| i)(input)
}

impl Year {
    fn run(&self, args: &Args) {
        match args
            .day
            .checked_sub(self.min)
            .and_then(|day| self.days.get(day as usize))
        {
            None => panic!(
                "Queried day {} is out of the range of valid days, {}..{}.\n\
                Args:\n\
                {args:#?}",
                args.day,
                self.min,
                self.min as usize + self.days.len()
            ),
            Some(None) => panic!(
                "Queried day {} has no registered questions.\n\
                Args:\n\
                {args:#?}",
                args.day
            ),
            Some(Some(day)) => day.run(args),
        }
    }

    fn try_from_day_params(mut day_params: Vec<DayParams>) -> Option<Self> {
        let (min, max): (u8, u8) = day_params
            .iter_mut()
            .filter_map(|DayParams { string, option, .. }| {
                parse_tagged_int("d", string).map_or_else(
                    |error| {
                        eprintln!(
                            "Invalid day string \"{}\"\n\
                            Error:\n\
                            {error}",
                            string
                        );

                        None
                    },
                    |(_, day)| {
                        *option = Some(day);

                        Some(day)
                    },
                )
            })
            .fold((u8::MAX, u8::MIN), |(min, max), day| {
                (min.min(day), max.max(day))
            });

        if min == u8::MAX {
            None
        } else {
            let size: usize = (max + 1 - min) as usize;
            let mut days: Vec<Option<Day>> = Vec::with_capacity(size);

            days.resize_with(size, || None);

            for DayParams { option, day, .. } in day_params.into_iter() {
                days[(option.unwrap() - min) as usize] = Some(day);
            }

            Some(Year { days, min })
        }
    }
}

pub struct YearParams<'a> {
    pub string: &'a str,
    pub option: Option<u16>,
    pub day_params: Vec<DayParams<'a>>,
}

#[derive(Default)]
pub struct Solutions {
    years: Vec<Option<Year>>,
    min: u16,
}

impl Solutions {
    pub fn run(&self, args: &Args) {
        match args
            .year
            .checked_sub(self.min)
            .and_then(|year| self.years.get(year as usize))
        {
            None => panic!(
                "Queried year {} is out of the range of valid years, {}..{}.\n\
                Args:\n\
                {args:#?}",
                args.year,
                self.min,
                self.min as usize + self.years.len()
            ),
            Some(None) => panic!(
                "Queried year {} has no registered days.\n\
                Args:\n\
                {args:#?}",
                args.year
            ),
            Some(Some(days)) => days.run(args),
        }
    }

    pub fn try_from_year_params(mut year_params: Vec<YearParams>) -> Option<Self> {
        let (min, max): (u16, u16) = year_params
            .iter_mut()
            .filter_map(|YearParams { string, option, .. }| {
                parse_tagged_int("y", string).map_or_else(
                    |error| {
                        eprintln!(
                            "Invalid year string \"{}\"\n\
                            Error:\n\
                            {error}",
                            string
                        );

                        None
                    },
                    |(_, year)| {
                        *option = Some(year);

                        Some(year)
                    },
                )
            })
            .fold((u16::MAX, u16::MIN), |(min, max), year| {
                (min.min(year), max.max(year))
            });

        if min == u16::MAX {
            None
        } else {
            let size: usize = (max + 1 - min) as usize;
            let mut years: Vec<Option<Year>> = Vec::with_capacity(size);

            years.resize_with(size, || None);

            for YearParams {
                option, day_params, ..
            } in year_params.into_iter()
            {
                years[(option.unwrap() - min) as usize] = Year::try_from_day_params(day_params);
            }

            Some(Solutions { years, min })
        }
    }
}

#[macro_export]
macro_rules! solutions {
    [ $( ( $year:ident, [ $( $day:ident, )* ] ), )* ] => {
        $(
            pub mod $year {
                $(
                    pub mod $day;
                )*
            }
        )*

        lazy_static::lazy_static! {
            pub static ref SOLUTIONS: Solutions = Solutions::try_from_year_params(vec![ $(
                YearParams {
                    string: stringify!($year),
                    option: None,
                    day_params: vec![ $(
                        DayParams {
                            string: stringify!($day),
                            option: None,
                            day: Day {
                                q1: $year::$day::Solution::q1,
                                q2: $year::$day::Solution::q2,
                                both: $year::$day::Solution::both,
                            }
                        },
                    )* ]
                },
            )* ]).unwrap_or_else(Solutions::default);
        }
    };
}

/// Opens a memory-mapped UTF-8 file at a specified path, and passes in a `&str` over the file to a
/// provided callback function
///
/// # Arguments
///
/// * `file_path` - A string slice file path to open as a read-only file
/// * `f` - A callback function to invoke on the contents of the file as a string slice
///
/// # Errors
///
/// This function returns a `Result::Err`-wrapped `std::io::Error` if an error has occurred.
/// Possible causes are:
///
/// * `std::fs::File::open` was unable to open a read-only file at `file_path`
/// * `memmap::Mmap::map` fails to create an `Mmap` instance for the opened file
/// * `std::str::from_utf8` determines the file is not in valid UTF-8 format
///
/// `f` is only executed *iff* an error is not encountered.
///
/// # Safety
///
/// This function uses `Mmap::map`, which is an unsafe function. There is no guarantee that an
/// external process won't modify the file after it is opened as read-only.
///
/// # Undefined Behavior
///
/// Related to the **Safety** section above, it is UB if the opened file is modified by an external
/// process while this function is referring to it as an immutable string slice. For more info on
/// this, see:
///
/// * https://www.reddit.com/r/rust/comments/wyq3ih/why_are_memorymapped_files_unsafe/
/// * https://users.rust-lang.org/t/how-unsafe-is-mmap/19635
/// * https://users.rust-lang.org/t/is-there-no-safe-way-to-use-mmap-in-rust/70338
pub unsafe fn open_utf8_file<T, F: FnOnce(&str) -> T>(file_path: &str, f: F) -> IoResult<T> {
    let file: File = File::open(file_path)?;

    // SAFETY: This operation is unsafe
    let mmap: Mmap = Mmap::map(&file)?;
    let bytes: &[u8] = &mmap;
    let utf8_str: &str = from_utf8(bytes).map_err(|utf8_error: Utf8Error| -> IoError {
        IoError::new(ErrorKind::InvalidData, utf8_error)
    })?;

    Ok(f(utf8_str))
}

pub fn min_and_max<T: Ord + Copy>(v1: T, v2: T) -> (T, T) {
    (min(v1, v2), max(v1, v2))
}

pub struct TokenStream<'i, 't, I: Iterator<Item = &'t str>>(&'i mut I);

impl<'i, 't, I: Iterator<Item = &'t str>> TokenStream<'i, 't, I> {
    pub fn new(iter: &'i mut I) -> Self {
        Self(iter)
    }
}

impl<'i, 't, I: Iterator<Item = &'t str>> Deref for TokenStream<'i, 't, I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'i, 't, I: Iterator<Item = &'t str>> DerefMut for TokenStream<'i, 't, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

pub fn unreachable_any<T, U>(_: T) -> U {
    unreachable!();
}

pub const LOWERCASE_A_OFFSET: u8 = b'a';
pub const UPPERCASE_A_OFFSET: u8 = b'A';
pub const ZERO_OFFSET: u8 = b'0';

pub fn validate_prefix_and_suffix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    suffix: &str,
    f: F,
) -> Result<&'s str, E> {
    if value.len() >= prefix.len() + suffix.len()
        && value.get(..prefix.len()).map_or(false, |p| p == prefix)
        && value
            .get(value.len() - suffix.len()..)
            .map_or(false, |s| s == suffix)
    {
        Ok(&value[prefix.len()..value.len() - suffix.len()])
    } else {
        Err(f(value))
    }
}

pub fn validate_prefix<'s, E, F: FnOnce(&'s str) -> E>(
    value: &'s str,
    prefix: &str,
    f: F,
) -> Result<&'s str, E> {
    validate_prefix_and_suffix(value, prefix, "", f)
}
