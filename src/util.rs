pub use {graph::*, grid::*, imat3::*, letter_counts::*};

use {
    clap::Parser,
    glam::IVec3,
    memmap::Mmap,
    nom::{
        bytes::complete::tag,
        character::complete::digit1,
        combinator::{map, map_res, opt, rest},
        sequence::tuple,
        IResult,
    },
    num::{Integer, NumCast, ToPrimitive},
    std::{
        any::type_name,
        cmp::{max, min, Ordering},
        fmt::Debug,
        fs::File,
        io::{Error as IoError, ErrorKind, Result as IoResult},
        mem::{transmute, MaybeUninit},
        ops::{Deref, DerefMut, Range, RangeInclusive},
        str::{from_utf8, FromStr, Utf8Error},
    },
};

mod graph;
mod grid;
mod imat3;
mod letter_counts;
pub mod minimal_value_with_all_digit_pairs;

#[allow(dead_code, unused_imports, unused_variables)]
mod template;

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
            map(opt(tag("-")), |minus| {
                if minus.is_some() {
                    I::zero() - I::one()
                } else {
                    I::one()
                }
            }),
            map_res(digit1, I::from_str),
        )),
        |(sign, bound)| sign * bound,
    )(input)
}

pub trait Parse: Sized {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self>;
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

pub trait SmallRangeInclusiveTraits: Clone + Default + Debug + PartialEq {}

impl<T: Clone + Debug + Default + PartialEq> SmallRangeInclusiveTraits for T {}

#[derive(Clone, Debug, PartialEq)]
pub struct SmallRangeInclusive<T: SmallRangeInclusiveTraits> {
    pub start: T,
    pub end: T,
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

impl<T: Copy + SmallRangeInclusiveTraits> Copy for SmallRangeInclusive<T> {}

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
    pub prime: u32,
    pub exponent: u32,
}

fn try_get_prime_factor(value: &mut u32, divisor: u32) -> Option<PrimeFactor> {
    let mut local_value: u32 = *value;
    let mut exponent: u32 = 0_u32;

    if local_value != 1_u32 {
        while local_value % divisor == 0_u32 {
            local_value /= divisor;
            exponent += 1_u32;
        }

        *value = local_value;
    }

    if exponent != 0_u32 {
        Some(PrimeFactor {
            prime: divisor,
            exponent,
        })
    } else {
        None
    }
}

/// Iterate over the prime factors of a given number.
///
/// This is an implementation of https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
pub fn iter_prime_factors(mut value: u32) -> impl Iterator<Item = PrimeFactor> {
    [2_u32]
        .into_iter()
        .chain((3_u32..=value / 2_u32).step_by(2_usize))
        .chain([value])
        .filter_map(move |divisor| try_get_prime_factor(&mut value, divisor))
}

#[test]
fn test_iter_prime_factors() {
    assert_eq!(
        iter_prime_factors(12_u32).collect::<Vec<PrimeFactor>>(),
        vec![
            PrimeFactor {
                prime: 2_u32,
                exponent: 2_u32
            },
            PrimeFactor {
                prime: 3_u32,
                exponent: 1_u32
            },
        ]
    );
    assert_eq!(
        iter_prime_factors(315_u32).collect::<Vec<PrimeFactor>>(),
        vec![
            PrimeFactor {
                prime: 3_u32,
                exponent: 2_u32
            },
            PrimeFactor {
                prime: 5_u32,
                exponent: 1_u32
            },
            PrimeFactor {
                prime: 7_u32,
                exponent: 1_u32
            },
        ]
    );
    assert_eq!(
        iter_prime_factors(41_u32).collect::<Vec<PrimeFactor>>(),
        vec![PrimeFactor {
            prime: 41_u32,
            exponent: 1_u32
        }]
    );
    assert_eq!(
        iter_prime_factors(22411_u32).collect::<Vec<PrimeFactor>>(),
        vec![
            PrimeFactor {
                prime: 73_u32,
                exponent: 1_u32
            },
            PrimeFactor {
                prime: 307_u32,
                exponent: 1_u32
            },
        ]
    );
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

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub enum Pixel {
        #[default]
        Dark = DARK = b'.',
        Light = LIGHT = b'#',
    }
}

impl Pixel {
    pub fn is_light(self) -> bool {
        matches!(self, Self::Light)
    }
}

impl From<bool> for Pixel {
    fn from(value: bool) -> Self {
        if value {
            Self::Light
        } else {
            Self::Dark
        }
    }
}

impl From<Pixel> for bool {
    fn from(value: Pixel) -> Self {
        value.is_light()
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
