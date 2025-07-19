pub use {
    bit::*,
    graph::*,
    grid::*,
    imat3::*,
    index::*,
    knapsack::*,
    letter::*,
    linked_list::*,
    linked_trie::*,
    necklace::*,
    no_drop_vec::*,
    nom::{error::Error as NomError, Err as NomErr},
    pixel::Pixel,
    region_tree::*,
    set_and_vec::*,
    static_string::*,
    table::*,
};

use {
    arrayvec::ArrayVec,
    clap::Parser,
    glam::IVec3,
    memmap::Mmap,
    nom::{
        branch::alt,
        bytes::complete::{tag, take},
        character::complete::{digit1, line_ending},
        combinator::{all_consuming, cond, map, map_res, opt, rest, success},
        multi::many_m_n,
        sequence::{preceded, tuple},
        IResult, Parser as NomParser,
    },
    num::{Integer, NumCast, PrimInt, ToPrimitive},
    std::{
        alloc::{alloc, Layout},
        any::type_name,
        cmp::{max, min, Ordering},
        fmt::{Debug, Formatter, Result as FmtResult},
        fs::File,
        hash::{DefaultHasher, Hash, Hasher},
        hint::black_box,
        io::{Error as IoError, ErrorKind, Result as IoResult},
        iter::from_fn,
        mem::{size_of, transmute, ManuallyDrop, MaybeUninit},
        ops::{Deref, DerefMut, Div, Range, RangeInclusive, Rem},
        str::{from_utf8, FromStr, Utf8Error},
        task::Poll,
    },
};

#[cfg(test)]
use std::rc::Rc;

mod bit;
mod graph;
mod grid;
mod imat3;
mod index;
mod knapsack;
mod letter;
mod linked_list;
mod linked_trie;
pub mod minimal_value_with_all_digit_pairs;
mod necklace;
mod no_drop_vec;
pub mod pixel;
mod region_tree;
mod set_and_vec;
mod static_string;
mod table;

#[allow(dead_code, unused_imports, unused_variables)]
mod template;

// Taken from `core::num` for faster conversion.
pub const ASCII_CASE_MASK: u8 = 0b0010_0000;

pub type NomErrStr<'s> = NomErr<NomError<&'s str>>;

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
    pub fn try_to_intermediate<I>(&self) -> Option<I>
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

impl Parse for Args {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, (_, year, _, day)): (&str, (_, u16, _, u8)) =
            tuple((tag("aoc::y"), parse_integer, tag("::d"), parse_integer))(input)?;

        let args: Self = Self {
            input_file_path: String::new(),
            year,
            day,
            question: 0_u8,
            question_args: QuestionArgs { verbose: true },
        };

        Ok((input, args))
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
    [ $( ( $year:ident, [ $( $day:ident ),* $(,)?] ) ),* $(,)? ] => {
        $(
            pub mod $year {
                $(
                    pub mod $day;
                )*
            }
        )*

        pub fn solutions() -> &'static Solutions {
            static ONCE_LOCK: std::sync::OnceLock<Solutions> = std::sync::OnceLock::new();

            ONCE_LOCK.get_or_init(|| Solutions::try_from_year_params(vec![ $(
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
            )* ]).unwrap_or_else(Solutions::default))
        }
    };
}

#[macro_export]
macro_rules! pretty_assert_eq {
    ($left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;

        if left != right {
            panic!(
                "pretty assertion failed: `(left == right)`\nleft: {left:#?}\nright: {right:#?}"
            );
        }
    }};
}

#[macro_export]
macro_rules! define_super_trait {
    {
        $pub:vis trait $super_trait:ident
            where Self : $first_trait:ident $( + $other_trait:ident )* $( , )?
        {}
    } => {
        $pub trait $super_trait
            where Self : $first_trait $( + $other_trait )*
        {}

        impl<T: $first_trait $( + $other_trait )*> $super_trait for T {}
    }
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

pub fn min_and_max<T: Copy + Ord>(v1: T, v2: T) -> (T, T) {
    (min(v1, v2), max(v1, v2))
}

pub fn div_and_rem<T: Copy + Div<Output = T> + Rem<Output = T>>(lhs: T, rhs: T) -> (T, T) {
    (lhs / rhs, lhs % rhs)
}

pub fn try_intersection<T: Ord + Copy>(range1: Range<T>, range2: Range<T>) -> Option<Range<T>> {
    match (range1.start.cmp(&range1.end), range2.start.cmp(&range2.end)) {
        (Ordering::Less, Ordering::Less) => (range1.end >= range2.start
            && range1.start <= range2.end)
            .then_some(range1.start.max(range2.start)..range1.end.min(range2.end)),
        (Ordering::Less, Ordering::Equal) => range1.contains(&range2.start).then_some(range2),
        (Ordering::Equal, Ordering::Less) => range2.contains(&range1.start).then_some(range1),
        (Ordering::Equal, Ordering::Equal) => (range1.start == range2.start).then_some(range1),
        _ => None,
    }
}

pub fn try_non_empty_intersection<T: Ord + Copy>(
    range1: Range<T>,
    range2: Range<T>,
) -> Option<Range<T>> {
    try_intersection(range1, range2).filter(|range| !range.is_empty())
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

pub fn copy_opt_mut<'a, 'b: 'a, T>(value: &'a mut Option<&'b mut T>) -> Option<&'a mut T> {
    value.as_mut().map(|value| &mut **value)
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

pub fn parse_integer<'i, I: FromStr + Integer>(input: &'i str) -> IResult<&'i str, I> {
    map(
        tuple((
            alt((
                map(tag("-"), |_| I::zero() - I::one()),
                map(opt(tag("+")), |_| I::one()),
            )),
            map_res(digit1, I::from_str),
        )),
        |(sign, bound)| sign * bound,
    )(input)
}

pub fn parse_integer_n_bytes<'i, I: FromStr + Integer>(
    n: usize,
) -> impl FnMut(&'i str) -> IResult<&'i str, I> {
    map_res(take(n), |input| {
        all_consuming(parse_integer)(input).map(|(_, integer)| integer)
    })
}

pub fn parse_separated_array_vec<
    'i,
    const LEN: usize,
    O: Default,
    O2,
    F: NomParser<&'i str, O, NomError<&'i str>>,
    G: NomParser<&'i str, O2, NomError<&'i str>>,
>(
    mut f: F,
    mut g: G,
) -> impl FnMut(&'i str) -> IResult<&'i str, ArrayVec<O, LEN>> {
    move |input: &'i str| {
        let mut array_vec: ArrayVec<O, LEN> = ArrayVec::new();

        let input: &str = many_m_n(0_usize, LEN, |input: &'i str| {
            let (input, value): (&str, O) = preceded(
                cond(!array_vec.is_empty(), |input| g.parse(input)),
                |input| f.parse(input),
            )(input)?;

            array_vec.push(value);

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, array_vec))
    }
}

pub fn parse_array_vec<
    'i,
    const LEN: usize,
    O: Default,
    F: NomParser<&'i str, O, NomError<&'i str>>,
>(
    f: F,
) -> impl FnMut(&'i str) -> IResult<&'i str, ArrayVec<O, LEN>> {
    parse_separated_array_vec(f, success(()))
}

pub fn parse_terminated_array_vec<
    'i,
    const LEN: usize,
    O: Default,
    O2,
    F: NomParser<&'i str, O, NomError<&'i str>>,
    G: NomParser<&'i str, O2, NomError<&'i str>>,
>(
    mut f: F,
    mut g: G,
) -> impl FnMut(&'i str) -> IResult<&'i str, ArrayVec<O, LEN>> {
    move |input: &'i str| {
        let (input, array_vec): (&str, ArrayVec<O, LEN>) =
            parse_separated_array_vec(|input| f.parse(input), |input| g.parse(input))(input)?;

        let input: &str = g.parse(input)?.0;

        Ok((input, array_vec))
    }
}

pub fn parse_array<
    'i,
    const LEN: usize,
    O: Default,
    F: NomParser<&'i str, O, NomError<&'i str>>,
>(
    f: F,
) -> impl FnMut(&'i str) -> IResult<&'i str, [O; LEN]> {
    map_res(parse_array_vec(f), ArrayVec::into_inner)
}

pub fn parse_separated_array<
    'i,
    const LEN: usize,
    O: Default,
    O2,
    F: NomParser<&'i str, O, NomError<&'i str>>,
    G: NomParser<&'i str, O2, NomError<&'i str>>,
>(
    f: F,
    g: G,
) -> impl FnMut(&'i str) -> IResult<&'i str, [O; LEN]> {
    map_res(parse_separated_array_vec(f, g), ArrayVec::into_inner)
}

pub fn parse_terminated_array<
    'i,
    const LEN: usize,
    O: Default,
    O2,
    F: NomParser<&'i str, O, NomError<&'i str>>,
    G: NomParser<&'i str, O2, NomError<&'i str>>,
>(
    f: F,
    g: G,
) -> impl FnMut(&'i str) -> IResult<&'i str, [O; LEN]> {
    map_res(parse_terminated_array_vec(f, g), ArrayVec::into_inner)
}

pub fn parse_line_endings<'i>(n: usize) -> impl FnMut(&'i str) -> IResult<&'i str, ()> {
    map(many_m_n(n, n, map(line_ending, |_| ())), |_| ())
}

pub trait Parse: Sized {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self>;
}

#[macro_export]
macro_rules! impl_Parse_for_Enum {
    ($enum:ty) => {
        impl Parse for $enum {
            fn parse<'i>(input: &'i str) -> nom::IResult<&'i str, Self> {
                use {
                    nom::{bytes::complete::tag, combinator::map},
                    strum::{IntoEnumIterator, VariantNames},
                };

                Self::iter()
                    .try_for_each(|enum_value| {
                        match map(tag(Self::VARIANTS[enum_value as usize]), |_| enum_value)(input) {
                            Ok(value) => Result::Err(Ok(value)),
                            Result::Err(nom::Err::Error(_)) => Ok(()),
                            Result::Err(error) => Result::Err(Result::Err(error)),
                        }
                    })
                    .map_or_else(
                        |result| result,
                        |_| {
                            Result::Err(nom::Err::Error(Error {
                                input,
                                code: nom::error::ErrorKind::Alt,
                            }))
                        },
                    )
            }
        }
    };
}

pub const fn imat3_const_inverse(imat3: &IMat3) -> IMat3 {
    let tmp0: IVec3 = ivec3_const_cross(imat3.y_axis, imat3.z_axis);
    let tmp1: IVec3 = ivec3_const_cross(imat3.z_axis, imat3.x_axis);
    let tmp2: IVec3 = ivec3_const_cross(imat3.x_axis, imat3.y_axis);
    let det: i32 = ivec3_const_dot(imat3.z_axis, tmp2);

    IMat3::from_rows(
        ivec3_const_div_i32(tmp0, det),
        ivec3_const_div_i32(tmp1, det),
        ivec3_const_div_i32(tmp2, det),
    )
}

pub const fn imat3_const_mul_ivec3(lhs: &IMat3, rhs: IVec3) -> IVec3 {
    ivec3_const_add(
        ivec3_const_add(
            ivec3_const_mul_i32(lhs.x_axis, rhs.x),
            ivec3_const_mul_i32(lhs.y_axis, rhs.y),
        ),
        ivec3_const_mul_i32(lhs.z_axis, rhs.z),
    )
}

pub const fn imat3_const_transpose(imat3: &IMat3) -> IMat3 {
    IMat3::from_rows(imat3.x_axis, imat3.y_axis, imat3.z_axis)
}

pub const fn imat3_const_mul(lhs: &IMat3, rhs: &IMat3) -> IMat3 {
    IMat3::from_cols(
        imat3_const_mul_ivec3(lhs, rhs.x_axis),
        imat3_const_mul_ivec3(lhs, rhs.y_axis),
        imat3_const_mul_ivec3(lhs, rhs.z_axis),
    )
}

pub const fn ivec3_const_add(lhs: IVec3, rhs: IVec3) -> IVec3 {
    IVec3::new(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z)
}

pub const fn ivec3_const_cross(lhs: IVec3, rhs: IVec3) -> IVec3 {
    IVec3 {
        x: lhs.y * rhs.z - rhs.y * lhs.z,
        y: lhs.z * rhs.x - rhs.z * lhs.x,
        z: lhs.x * rhs.y - rhs.x * lhs.y,
    }
}

pub const fn ivec3_const_dot(lhs: IVec3, rhs: IVec3) -> i32 {
    (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z)
}

pub const fn ivec3_const_div_i32(lhs: IVec3, rhs: i32) -> IVec3 {
    IVec3::new(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs)
}

pub const fn ivec3_const_mul_i32(lhs: IVec3, rhs: i32) -> IVec3 {
    IVec3::new(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs)
}

pub const fn triangle_number(n: usize) -> usize {
    n * (n + 1_usize) / 2_usize
}

pub trait TryAsRange<T: NumCast> {
    fn try_as_range(&self) -> Option<Range<T>>;
}

impl<T: Clone + Copy + ToPrimitive + Sized, U: NumCast> TryAsRange<U> for Range<T> {
    fn try_as_range(&self) -> Option<Range<U>> {
        Some(<U as NumCast>::from(self.start)?..<U as NumCast>::from(self.end)?)
    }
}

#[cfg(target_pointer_width = "16")]
pub type UsizeEquivalentInteger = u16;

#[cfg(target_pointer_width = "32")]
pub type UsizeEquivalentInteger = u32;

#[cfg(target_pointer_width = "64")]
pub type UsizeEquivalentInteger = u64;

pub trait AsRangeUsize {
    fn as_range_usize(&self) -> Range<usize>;
}

impl<T: Clone + Copy + ToPrimitive + Sized> AsRangeUsize for Range<T>
where
    UsizeEquivalentInteger: From<T>,
{
    fn as_range_usize(&self) -> Range<usize> {
        self.try_as_range().unwrap()
    }
}

define_super_trait! {
    pub trait SmallRangeInclusiveTraits where Self: Clone + Copy + Default + Debug + PartialEq {}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmallRangeInclusive<T: SmallRangeInclusiveTraits> {
    pub start: T,
    pub end: T,
}

impl<T: SmallRangeInclusiveTraits> SmallRangeInclusive<T> {
    pub const fn new(start: T, end: T) -> Self {
        Self { start, end }
    }

    pub const fn as_range_inclusive(&self) -> RangeInclusive<T> {
        self.start..=self.end
    }

    /// For compatibility with `range_inclusive_len`
    pub const fn start(&self) -> &T {
        &self.start
    }

    /// For compatibility with `range_inclusive_len`
    pub const fn end(&self) -> &T {
        &self.end
    }
}

impl<T: SmallRangeInclusiveTraits> From<RangeInclusive<T>> for SmallRangeInclusive<T> {
    fn from(value: RangeInclusive<T>) -> Self {
        Self {
            start: value.start().clone(),
            end: value.end().clone(),
        }
    }
}

impl<T: SmallRangeInclusiveTraits> From<SmallRangeInclusive<T>> for RangeInclusive<T> {
    fn from(value: SmallRangeInclusive<T>) -> Self {
        value.start..=value.end
    }
}

#[macro_export]
macro_rules! const_range_inclusive_len {
    ($range_inclusive:expr) => {{
        let diff = *$range_inclusive.end() - *$range_inclusive.start() + 1;

        #[allow(unused_comparisons)]
        if diff < 0 {
            0_usize
        } else {
            diff as usize
        }
    }};
}

pub trait ComputeHash {
    fn compute_hash(&self) -> u64;
}

impl<T: Hash> ComputeHash for T {
    fn compute_hash(&self) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();

        self.hash(&mut hasher);

        hasher.finish()
    }
}

pub const fn factorial(value: usize) -> usize {
    match value {
        0_usize..=1_usize => 1_usize,
        _ => value * factorial(value - 1_usize),
    }
}

pub const fn digits(value: u32) -> usize {
    if value == 0_u32 {
        1_usize
    } else {
        value.ilog10() as usize + 1_usize
    }
}

pub const U32_DIGITS: usize = digits(u32::MAX);

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct PrimeFactor {
    pub prime: usize,
    pub exponent: u8,
}

struct ConstTryGetPrimeFactorResult {
    remaining: usize,
    prime_factor: PrimeFactor,
}

const fn const_try_get_prime_factor(
    remaining: usize,
    divisor: usize,
) -> Option<ConstTryGetPrimeFactorResult> {
    if remaining <= 1_usize {
        None
    } else {
        let mut exponent: u8 = 0_u8;
        let mut remaining: usize = remaining;

        while remaining % divisor == 0_usize {
            remaining /= divisor;
            exponent += 1_u8;
        }

        if exponent == 0_u8 {
            None
        } else {
            Some(ConstTryGetPrimeFactorResult {
                remaining,
                prime_factor: PrimeFactor {
                    prime: divisor,
                    exponent,
                },
            })
        }
    }
}

struct ConstTryGetNextPrimeFactorResult {
    remaining: usize,
    divisor: usize,
    prime_factor: PrimeFactor,
}

const fn const_try_get_next_prime_factor(
    value: usize,
    remaining: usize,
    divisor: usize,
) -> Option<ConstTryGetNextPrimeFactorResult> {
    let mut result: Option<ConstTryGetNextPrimeFactorResult> = None;
    let mut divisor: usize = divisor;

    if divisor == 2_usize {
        let next_divisor: usize = 3_usize;

        if let Some(const_try_get_prime_factor_result) =
            const_try_get_prime_factor(remaining, divisor)
        {
            result = Some(ConstTryGetNextPrimeFactorResult {
                remaining: const_try_get_prime_factor_result.remaining,
                divisor: next_divisor,
                prime_factor: const_try_get_prime_factor_result.prime_factor,
            });
        } else {
            divisor = next_divisor;
        }
    }

    // Could actually be even
    let max_odd: usize = value / 2_usize;

    while result.is_none() && divisor <= max_odd {
        let mut next_divisor: usize = divisor + 2_usize;

        if next_divisor > max_odd {
            next_divisor = value;
        }

        if let Some(const_try_get_prime_factor_result) =
            const_try_get_prime_factor(remaining, divisor)
        {
            result = Some(ConstTryGetNextPrimeFactorResult {
                remaining: const_try_get_prime_factor_result.remaining,
                divisor: next_divisor,
                prime_factor: const_try_get_prime_factor_result.prime_factor,
            });
        } else {
            divisor = next_divisor;
        }
    }

    if result.is_none() && divisor == value {
        if let Some(const_try_get_prime_factor_result) =
            const_try_get_prime_factor(remaining, divisor)
        {
            result = Some(ConstTryGetNextPrimeFactorResult {
                remaining: const_try_get_prime_factor_result.remaining,
                divisor: usize::MAX,
                prime_factor: const_try_get_prime_factor_result.prime_factor,
            });
        }
    }

    result
}

pub const fn const_is_prime(value: usize) -> bool {
    let remaining: usize = value;
    let divisor: usize = 2_usize;

    if let Some(result) = const_try_get_next_prime_factor(value, remaining, divisor) {
        const_try_get_next_prime_factor(value, result.remaining, result.divisor).is_none()
    } else {
        false
    }
}

pub const fn const_is_composite(value: usize) -> bool {
    let remaining: usize = value;
    let divisor: usize = 2_usize;

    if let Some(result) = const_try_get_next_prime_factor(value, remaining, divisor) {
        const_try_get_next_prime_factor(value, result.remaining, result.divisor).is_some()
    } else {
        false
    }
}

pub const fn max_distinct_primes<I: PrimInt>() -> usize {
    let max: usize = usize::MAX >> ((size_of::<usize>() - size_of::<I>()) * u8::BITS as usize);
    let mut product: usize = 2_usize;
    let mut distinct_primes: usize = 1_usize;
    let mut candidate: usize = 3_usize;

    while product < max {
        if const_is_prime(candidate) {
            product = match product.checked_mul(candidate) {
                Some(product) => product,
                None => max,
            };

            // This will add one more than needed, but it's better to save the branching here and
            // just subtract one off at the end.
            distinct_primes += 1_usize;
        }

        candidate += 2_usize;
    }

    distinct_primes - 1_usize
}

pub trait MaxDistinctPrimes {
    const MAX_DISTINCT_PRIMES: usize;
}

impl<I: PrimInt> MaxDistinctPrimes for I {
    const MAX_DISTINCT_PRIMES: usize = max_distinct_primes::<I>();
}

pub struct PrimeFactors {
    primes: [usize; usize::MAX_DISTINCT_PRIMES],
    exponents: [u8; usize::MAX_DISTINCT_PRIMES],
    len: u8,
}

impl PrimeFactors {
    pub const fn new() -> Self {
        PrimeFactors {
            primes: [0_usize; usize::MAX_DISTINCT_PRIMES],
            exponents: [0_u8; usize::MAX_DISTINCT_PRIMES],
            len: 0_u8,
        }
    }

    pub fn iter_prime_factors(&self) -> impl Iterator<Item = PrimeFactor> + '_ {
        self.primes[..self.len as usize]
            .iter()
            .zip(self.exponents[..self.len as usize].iter())
            .map(|(&prime, &exponent)| PrimeFactor { prime, exponent })
    }

    pub fn push(&mut self, prime_factor: PrimeFactor) {
        let index: usize = self.len as usize;

        self.primes[index] = prime_factor.prime;
        self.exponents[index] = prime_factor.exponent;
        self.len += 1_u8;
    }

    pub fn get_prime_factor(&self, index: usize) -> PrimeFactor {
        assert!(index < self.len as usize);

        PrimeFactor {
            prime: self.primes[index],
            exponent: self.exponents[index],
        }
    }
}

pub const fn compute_prime_factors(value: usize) -> PrimeFactors {
    let mut prime_factors: PrimeFactors = PrimeFactors::new();
    let mut remaining: usize = value;
    let mut divisor: usize = 2_usize;

    while {
        if let Some(result) = const_try_get_next_prime_factor(value, remaining, divisor) {
            remaining = result.remaining;
            divisor = result.divisor;

            let index: usize = prime_factors.len as usize;

            prime_factors.primes[index] = result.prime_factor.prime;
            prime_factors.exponents[index] = result.prime_factor.exponent;
            prime_factors.len += 1_u8;

            true
        } else {
            false
        }
    } {}

    prime_factors
}

fn try_get_prime_factor(remaining: &mut usize, divisor: usize) -> Option<PrimeFactor> {
    const_try_get_prime_factor(*remaining, divisor).map(|result| {
        *remaining = result.remaining;

        result.prime_factor
    })
}

/// Iterate over the prime factors of a given number.
///
/// This is an implementation of https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
pub fn iter_prime_factors(value: usize) -> impl Iterator<Item = PrimeFactor> {
    let mut remaining: usize = value;

    [2_usize]
        .into_iter()
        .chain((3_usize..=value / 2_usize).step_by(2_usize))
        .chain([value])
        .filter_map(move |divisor| try_get_prime_factor(&mut remaining, divisor))
}

pub fn is_prime(value: usize) -> bool {
    let mut prime_factor_iter = iter_prime_factors(value);

    prime_factor_iter.next().is_some() && prime_factor_iter.next().is_none()
}

pub fn is_composite(value: usize) -> bool {
    let mut prime_factor_iter = iter_prime_factors(value);

    prime_factor_iter.next().is_some() && prime_factor_iter.next().is_some()
}

#[test]
fn test_iter_prime_factors() {
    for (value, prime_factors) in [
        (
            12_usize,
            vec![
                PrimeFactor {
                    prime: 2_usize,
                    exponent: 2_u8,
                },
                PrimeFactor {
                    prime: 3_usize,
                    exponent: 1_u8,
                },
            ],
        ),
        (
            120_usize,
            vec![
                PrimeFactor {
                    prime: 2_usize,
                    exponent: 3_u8,
                },
                PrimeFactor {
                    prime: 3_usize,
                    exponent: 1_u8,
                },
                PrimeFactor {
                    prime: 5_usize,
                    exponent: 1_u8,
                },
            ],
        ),
        (
            315_usize,
            vec![
                PrimeFactor {
                    prime: 3_usize,
                    exponent: 2_u8,
                },
                PrimeFactor {
                    prime: 5_usize,
                    exponent: 1_u8,
                },
                PrimeFactor {
                    prime: 7_usize,
                    exponent: 1_u8,
                },
            ],
        ),
        (
            41_usize,
            vec![PrimeFactor {
                prime: 41_usize,
                exponent: 1_u8,
            }],
        ),
        (
            22411_usize,
            vec![
                PrimeFactor {
                    prime: 73_usize,
                    exponent: 1_u8,
                },
                PrimeFactor {
                    prime: 307_usize,
                    exponent: 1_u8,
                },
            ],
        ),
        (0_usize, Vec::new()),
        (1_usize, Vec::new()),
    ] {
        assert_eq!(
            iter_prime_factors(value).collect::<Vec<PrimeFactor>>(),
            prime_factors
        );
        assert_eq!(
            compute_prime_factors(value)
                .iter_prime_factors()
                .collect::<Vec<PrimeFactor>>(),
            prime_factors
        );
    }
}

pub fn iter_factors(value: usize) -> impl Iterator<Item = usize> {
    (value <= 1_usize).then_some(value).into_iter().chain({
        (value > 1_usize)
            .then(|| {
                let mut prime_factors: PrimeFactors = PrimeFactors::new();

                for prime_factor in iter_prime_factors(value) {
                    prime_factors.push(prime_factor);
                }

                let mut exponents: [u8; usize::MAX_DISTINCT_PRIMES] =
                    [0_u8; usize::MAX_DISTINCT_PRIMES];
                let mut is_done: bool = false;

                from_fn(move || {
                    (!is_done).then(|| {
                        let factor: usize = prime_factors
                            .iter_prime_factors()
                            .zip(exponents.iter().copied())
                            .map(|(prime_factor, exponent)| prime_factor.prime.pow(exponent as u32))
                            .product();

                        is_done = prime_factors
                            .iter_prime_factors()
                            .zip(exponents[..prime_factors.len as usize].iter_mut())
                            .try_for_each(|(prime_factor, exponent)| {
                                *exponent += 1_u8;

                                (*exponent > prime_factor.exponent).then(|| {
                                    *exponent = 0_u8;
                                })
                            })
                            .is_some();

                        factor
                    })
                })
            })
            .into_iter()
            .flatten()
    })
}

#[test]
fn test_iter_factors() {
    for (value, factors) in [
        (
            12_usize,
            vec![1_usize, 2_usize, 4_usize, 3_usize, 6_usize, 12_usize],
        ),
        (
            120_usize,
            vec![
                1_usize, 2_usize, 4_usize, 8_usize, 3_usize, 6_usize, 12_usize, 24_usize, 5_usize,
                10_usize, 20_usize, 40_usize, 15_usize, 30_usize, 60_usize, 120_usize,
            ],
        ),
        (
            315_usize,
            vec![
                1_usize, 3_usize, 9_usize, 5_usize, 15_usize, 45_usize, 7_usize, 21_usize,
                63_usize, 35_usize, 105_usize, 315_usize,
            ],
        ),
        (41_usize, vec![1_usize, 41_usize]),
        (22411_usize, vec![1_usize, 73_usize, 307_usize, 22411_usize]),
        (0_usize, vec![0_usize]),
        (1_usize, vec![1_usize]),
    ] {
        assert_eq!(iter_factors(value).collect::<Vec<usize>>(), factors);
    }
}

// This is an implementation of https://en.wikipedia.org/wiki/Greatest_common_divisor#Binary_GCD_algorithm
pub fn compute_gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    let mut d: u32 = 0_u32;

    match (a.is_even(), b.is_even()) {
        (true, true) => {
            d = a.trailing_zeros().min(b.trailing_zeros());
            a >>= d;
            b >>= d;
        }
        (true, false) => {
            a >>= a.trailing_zeros();
        }
        (false, true) => {
            b >>= b.trailing_zeros();
        }
        (false, false) => {}
    }

    while a != b {
        if a > b {
            let diff: u32 = a - b;

            a = diff >> diff.trailing_zeros();
        } else {
            let diff: u32 = b - a;

            b = diff >> diff.trailing_zeros();
        }
    }

    a << d
}

#[test]
fn test_compute_gcd() {
    assert_eq!(compute_gcd(2_u32, 3_u32), 1_u32);
    assert_eq!(compute_gcd(12_u32, 3_u32), 3_u32);
    assert_eq!(compute_gcd(12_u32, 3_u32), 3_u32);
    assert_eq!(compute_gcd(25_u32, 3_u32), 1_u32);
    assert_eq!(compute_gcd(25_u32, 10_u32), 5_u32);
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct ExtendedEuclideanAlgorithmOutput {
    pub gcd: i64,
    pub x: i64,
    pub y: i64,
}

// This is an implementation of https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
pub fn extended_euclidean_algorithm(a: i64, b: i64) -> ExtendedEuclideanAlgorithmOutput {
    struct Terms {
        r: i64,
        s: i64,
        t: i64,
    }

    impl From<Terms> for ExtendedEuclideanAlgorithmOutput {
        fn from(value: Terms) -> Self {
            Self {
                gcd: value.r,
                x: value.s,
                y: value.t,
            }
        }
    }

    let [mut prev_terms, mut curr_terms]: [Terms; 2_usize] = [
        Terms {
            r: a,
            s: 1_i64,
            t: 0_i64,
        },
        Terms {
            r: b,
            s: 0_i64,
            t: 1_i64,
        },
    ];

    loop {
        let q: i64 = prev_terms.r.div_euclid(curr_terms.r);
        let next_terms: Terms = Terms {
            r: prev_terms.r.rem_euclid(curr_terms.r),
            s: prev_terms.s - q * curr_terms.s,
            t: prev_terms.t - q * curr_terms.t,
        };

        if next_terms.r == 0_i64 {
            return curr_terms.into();
        } else {
            prev_terms = curr_terms;
            curr_terms = next_terms;
        }
    }
}

#[test]
fn test_extended_euclidean_algorithm() {
    assert_eq!(
        extended_euclidean_algorithm(240_i64, 46_i64),
        ExtendedEuclideanAlgorithmOutput {
            gcd: 2_i64,
            x: -9_i64,
            y: 47_i64,
        }
    );
}

#[macro_export]
macro_rules! define_cell {
    {
        #[repr(u8)]
        $(#[$attr:meta])*
        $pub:vis enum $cell:ident { $(
            $(#[$variant_attr:meta])*
            $variant:ident = $variant_const:ident = $variant_u8:expr
        ),* $(,)? }
    } => {
        #[repr(u8)]
        $(#[$attr])*
        $pub enum $cell { $(
            $(#[$variant_attr])*
            $variant = Self::$variant_const,
        )* }

        impl $cell {
            $(
                const $variant_const: u8 = $variant_u8;
            )*
            const STR: &'static str =
                // SAFETY: Trivial
                unsafe { ::std::str::from_utf8_unchecked(&[$(
                    $cell::$variant_const,
                )*]) };
        }

        unsafe impl IsValidAscii for $cell {}

        impl Parse for $cell {
            fn parse<'i>(input: &'i str) -> ::nom::IResult<&'i str, Self> {
                ::nom::combinator::map(
                    ::nom::character::complete::one_of($cell::STR),
                    |value: char| { $cell::try_from(value).unwrap() }
                )(input)
            }
        }

        impl TryFrom<u8> for $cell {
            type Error = ();

            fn try_from(value: u8) -> Result<Self, Self::Error> {
                match value {
                    $(
                        Self::$variant_const => Ok(Self::$variant),
                    )*
                    _ => Err(()),
                }
            }
        }

        impl TryFrom<char> for $cell {
            type Error = ();

            fn try_from(value: char) -> Result<Self, Self::Error> {
                (value as u8).try_into()
            }
        }
    }
}

pub trait LargeArrayDefault {
    fn large_array_default() -> Self;
}

impl<T: Default, const N: usize> LargeArrayDefault for [T; N] {
    fn large_array_default() -> Self {
        let mut maybe_uninit_array_of_t: MaybeUninit<Self> = MaybeUninit::uninit();

        {
            // SAFETY: We're transmuting to an array of `MaybeUninit` elements of the same size.
            let array_of_maybe_uninit_t: &mut [MaybeUninit<T>; N] =
                unsafe { transmute(&mut maybe_uninit_array_of_t) };

            for maybe_uninit_t in array_of_maybe_uninit_t {
                maybe_uninit_t.write(T::default());
            }
        }

        // SAFETY: All elements were just initialized above
        unsafe { maybe_uninit_array_of_t.assume_init() }
    }
}

pub fn boxed_slice_from_array<T: Unpin, const N: usize>(array: [T; N]) -> Box<[T]> {
    let array: ManuallyDrop<[T; N]> = ManuallyDrop::new(array);

    // SAFETY: The compiler gives the following error when trying to transmute by value:
    // ```
    // error[E0512]: cannot transmute between types of different sizes, or dependently-sized types
    // ```
    // We know, of course, that these types are actually the same size and alignment. Note that this
    // has to be an array of `MaybeUninit` instead of `ManuallyDrop` because we can only take it by
    // reference, yet we want to assume ownership (without cloning or copying) of each element.
    // `MaybeUninit::assume_init_read` can achieve that, but `ManuallyDrop` has no such method.
    let array: &[MaybeUninit<T>; N] = unsafe { transmute(&array) };

    let array_ptr: *mut [T; N] = unsafe { alloc(Layout::new::<[T; N]>()) } as *mut [T; N];

    {
        let array_mut: &mut [MaybeUninit<T>; N] =
            // SAFETY: In reality, elements of this array are currently uninitialized. This pointer
            // cast just explicitly acknowledges that.
            unsafe {(array_ptr as *mut [MaybeUninit<T>; N]).as_mut()}.unwrap();

        for (src, dst) in array.iter().zip(array_mut.iter_mut()) {
            // SAFETY: Each element of src is really an element of the original `array` function
            // parameter, which is known to be initialized.
            dst.write(unsafe { src.assume_init_read() });
        }
    }

    // SAFETY: At this point, all elements of array_ptr have been initialized, and the shadowed
    // `array` will drop without calling drop on the elements that have now been moved into the
    // slice.
    unsafe { Box::from_raw(array_ptr) }
}

#[macro_export]
macro_rules! bitarr_typed {
    [ $bitarr:ty; $($bit:expr),*  $(,)? ] => { {
        let mut bitarr: $bitarr = <$bitarr>::ZERO;

        for (index, bit) in [ $( ($bit as i32) == 1_i32, )* ].into_iter().enumerate() {
            bitarr.set(index, bit);
        }

        bitarr
    } }
}

pub struct DebugString(pub String);

impl Debug for DebugString {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(&self.0)
    }
}

#[test]
fn test_boxed_slice_from_array() {
    let rc: Rc<()> = Rc::new(());

    let array: [Rc<()>; 4_usize] = [rc.clone(), rc.clone(), rc.clone(), rc.clone()];

    assert_eq!(Rc::strong_count(&rc), 5_usize);

    {
        let _boxed_slice: Box<[Rc<()>]> = boxed_slice_from_array(array);

        assert_eq!(Rc::strong_count(&rc), 5_usize);
    }

    assert_eq!(Rc::strong_count(&rc), 1_usize);
}

pub fn debug_break() {
    black_box(());
}

pub fn cond_debug_break(cond: bool) {
    if cond {
        debug_break();
    }
}

pub fn option_from_poll<T>(poll: Poll<T>) -> Option<T> {
    match poll {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

pub fn poll_from_option<T>(option: Option<T>) -> Poll<T> {
    match option {
        Some(value) => Poll::Ready(value),
        None => Poll::Pending,
    }
}

pub fn add_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_add(&b) {
        Some(sum) => sum,
        None => {
            panic!();
        }
    }
}

pub fn sub_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_sub(&b) {
        Some(difference) => difference,
        None => {
            panic!();
        }
    }
}

pub fn mul_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_mul(&b) {
        Some(product) => product,
        None => {
            panic!();
        }
    }
}

pub fn div_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_div(&b) {
        Some(quotient) => quotient,
        None => {
            panic!();
        }
    }
}

pub fn assert_eq_break<T: Debug + PartialEq>(left: T, right: T) {
    if left != right {
        assert_eq!(left, right);
    }
}

pub fn map_break<T>(value: T) -> T {
    black_box(());

    value
}

pub trait UnwrapBreak<T> {
    fn unwrap_break(self) -> T;
}

impl<T> UnwrapBreak<T> for Option<T> {
    fn unwrap_break(self) -> T {
        if self.is_none() {
            panic!()
        } else {
            self.unwrap()
        }
    }
}
