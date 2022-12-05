use {
    aoc_2022::*,
    clap::Parser,
    std::{cmp::Ordering, convert::TryFrom, mem::transmute, str::Chars},
};

/// An enum to represent either a play in the game "Rock, Paper, Scissors"
#[derive(Clone, Copy)]
#[repr(u8)]
enum Rps {
    Rock,
    Paper,
    Scissors,
}

impl Rps {
    /// Compares `self` with another `Rps`, returning `Ordering::Less` if `self` loses,
    /// `Ordering::Equal` if both tie, and `Ordering::Greater` if `self` wins
    ///
    /// Note that this is intentionally not an implementation of `Ord`/`PartialOrd`, since they
    /// require transitivity between results, which is incompatible with the game "Rock, Paper,
    /// Scissors".
    ///
    /// # Arguments
    ///
    /// * `other` - Another play of "Rock, Paper, Scissors" to compare `self` against
    fn cmp(self, other: Self) -> Ordering {
        ((self as i8 - other as i8 + 1_i8).rem_euclid(3_i8) - 1_i8).cmp(&0_i8)
    }

    /// Computes the score for this play, as described by https://adventofcode.com/2022/day/2
    fn score(self) -> u32 {
        self as u32 + 1_u32
    }
}

impl TryFrom<u8> for Rps {
    type Error = ();

    /// Tries to convert a `u8` into an `Rps`
    ///
    /// # Arguments
    ///
    /// * `value` - The `u8` to attempt to convert into an `Rps`
    ///
    /// # Errors
    ///
    /// This will return `Err(())` if `value` is out of the range [0, 2]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value < 3_u8 {
            Ok(
                // SAFETY: We just validated value is in the correct range
                unsafe { transmute(value) },
            )
        } else {
            Err(())
        }
    }
}

/// A new-type of `Rps` for the opponent, for the sake of being able to have multiple types
/// implement `TryFrom<char>` differently, depending on who's play is being parsed
#[derive(Clone, Copy)]
struct Opponent(Rps);

impl TryFrom<char> for Opponent {
    type Error = ();

    /// Tries to convert a `char` into an `Opponent`, as described by
    /// https://adventofcode.com/2022/day/2
    ///
    /// # Arguments
    ///
    /// * `value` - The `char` to attempt to convert into an `Opponent`
    ///
    /// # Errors
    ///
    /// This will return `Err(())` if `value` isn't 'A', 'B', nor 'C'.
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'A' => Ok(Self(Rps::Rock)),
            'B' => Ok(Self(Rps::Paper)),
            'C' => Ok(Self(Rps::Scissors)),
            _ => Err(()),
        }
    }
}

/// A new-type of `Rps` for "you", for the sake of being able to have multiple types implement
/// `TryFrom<char>` differently, depending on who's play is being parsed
#[derive(Clone, Copy)]
struct Response(Rps);

impl From<(Opponent, Outcome)> for Response {
    /// Converts an opponent's play and a round outcome into the player's response to the opponent
    /// that will produce the outcome.
    fn from((Opponent(opponent), Outcome(outcome)): (Opponent, Outcome)) -> Self {
        // `i8::rem_euclid` returns a value in the range `[0_i8, rhs)`, each value of which is fully
        // representable as a `u8`. This is also the same range `Rps::try_from` expects, so this
        // `unwrap` call won't panic
        Self(
            ((opponent as i8 + outcome as i8).rem_euclid(3_i8) as u8)
                .try_into()
                .unwrap(),
        )
    }
}

impl TryFrom<char> for Response {
    type Error = ();

    /// Tries to convert a `char` into a `Response`, as described by
    /// https://adventofcode.com/2022/day/2
    ///
    /// # Arguments
    ///
    /// * `value` - The `char` to attempt to convert into a `Response`
    ///
    /// # Errors
    ///
    /// This will return `Err(())` if `value` isn't 'X', 'Y', nor 'Z'.
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'X' => Ok(Self(Rps::Rock)),
            'Y' => Ok(Self(Rps::Paper)),
            'Z' => Ok(Self(Rps::Scissors)),
            _ => Err(()),
        }
    }
}

/// A new-type of `Ordering` for the outcome of the round, for the sake of being able to derive
/// `TryFrom<char>` for it
#[derive(Clone, Copy)]
struct Outcome(Ordering);

impl Outcome {
    /// Computes the score of the round outcome, as described by https://adventofcode.com/2022/day/2
    fn score(self) -> u32 {
        (self.0 as i8 + 1_i8) as u32 * 3_u32
    }
}

impl From<(Response, Opponent)> for Outcome {
    /// Converts a response and an opponent's play into a round outcome
    fn from((Response(response), Opponent(opponent)): (Response, Opponent)) -> Self {
        Self(response.cmp(opponent))
    }
}

impl TryFrom<char> for Outcome {
    type Error = ();

    /// Tries to convert a `char` into an `Outcome`, as described by
    /// https://adventofcode.com/2022/day/2
    ///
    /// # Arguments
    ///
    /// * `value` - The `char` to attempt to convert into an `Outcome`
    ///
    /// # Errors
    ///
    /// This will return `Err(())` if `value` isn't 'X', 'Y', nor 'Z'.
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'X' => Ok(Self(Ordering::Less)),
            'Y' => Ok(Self(Ordering::Equal)),
            'Z' => Ok(Self(Ordering::Greater)),
            _ => Err(()),
        }
    }
}

/// The correct round understanding as described by https://adventofcode.com/2022/day/2: each line
/// is an opponent's play and the round outcome
#[derive(Clone, Copy)]
struct Round {
    response: Response,
    outcome: Outcome,
}

impl Round {
    /// Computes the score of the round, as described by https://adventofcode.com/2022/day/2
    fn score(self) -> u32 {
        self.response.0.score() + self.outcome.score()
    }

    /// Tries to convert a `&str` line into a `Round`, expecting a 3-char string slice that matches
    /// the regex expression `[ABC] [XYZ]`, representing an opponent's play followed by the player's
    /// response
    ///
    /// # Arguments
    ///
    /// * `value` - The string slice to attempt to convert into a `Round`
    ///
    /// # Errors
    ///
    /// This will return `None` if `value` cannot be properly parsed. See `impl TryFrom<char> for
    /// Opponent` and `impl TryFrom<char> for Response` for more info.
    fn try_from_opponent_and_response(value: &str) -> Option<Self> {
        let mut chars: Chars = value.chars();

        let opponent: Opponent = chars.next()?.try_into().ok()?;

        if chars.next()? != ' ' {
            return None;
        }

        let response: Response = chars.next()?.try_into().ok()?;

        if chars.next().is_some() {
            return None;
        }

        let outcome: Outcome = (response, opponent).into();

        Some(Self { response, outcome })
    }

    /// Tries to convert a `&str` line into a `Round`, expecting a 3-char string slice that matches
    /// the regex expression `[ABC] [XYZ]`, representing an opponent's play followed by the round
    /// outcome
    ///
    /// # Arguments
    ///
    /// * `value` - The string slice to attempt to convert into a `Round`
    ///
    /// # Errors
    ///
    /// This will return `None` if `value` cannot be properly parsed. See `impl TryFrom<char> for
    /// Opponent` and `impl TryFrom<char> for Outcome` for more info.
    fn try_from_opponent_and_outcome(value: &str) -> Option<Self> {
        let mut chars: Chars = value.chars();

        let opponent: Opponent = chars.next()?.try_into().ok()?;

        if chars.next()? != ' ' {
            return None;
        }

        let outcome: Outcome = chars.next()?.try_into().ok()?;

        if chars.next().is_some() {
            return None;
        }

        let response: Response = (opponent, outcome).into();

        Some(Self { response, outcome })
    }
}

/// Iterate over the rounds of an input string slice, using a specified function to parse string
/// slice lines
///
/// # Arguments
///
/// * `input` - The input string slice to parse rounds from and iterate over, with individual rounds
///   delineated by `'\n'`
/// * `f` - The function used to parse individual rounds from `input`
fn iter_rounds<'a, F: for<'b> Fn(&'b str) -> Option<Round> + 'a>(
    input: &'a str,
    f: F,
) -> impl Iterator<Item = Round> + '_ {
    input.split('\n').filter_map(f)
}

/// Sum the total score of all rounds of an input string slice, using a specified function to parse
/// string slice lines
///
/// # Arguments
///
/// * `input` - The input string slice to sum the round scores of, with individual rounds
///   delineated by `'\n'`
/// * `f` - The function used to parse individual rounds from `input`
fn total_score<F: for<'a> Fn(&'a str) -> Option<Round>>(input: &str, f: F) -> u32 {
    iter_rounds(input, f).map(Round::score).sum()
}

fn main() {
    let args: Args = Args::parse();
    let input_file_path: &str = args.input_file_path("input/day2.txt");

    if let Err(err) =
        // SAFETY: This operation is unsafe, we're just hoping nobody else touches the file while
        // this program is executing
        unsafe {
            open_utf8_file(input_file_path, |input: &str| {
                println!(
                    "total_score(input, Round::try_from_opponent_and_response) == {}\n\
                    total_score(input, Round::try_from_opponent_and_outcome) == {}",
                    total_score(input, Round::try_from_opponent_and_response),
                    total_score(input, Round::try_from_opponent_and_outcome)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Verify that `Rps::cmp` does the math correctly
    fn test_rps_cmp() {
        macro_rules! compare_pairs {
            ($($lhs:ident $op:ident $rhs:ident,)+) => {
                $(
                    assert_eq!(Rps::$lhs.cmp(Rps::$rhs), Ordering::$op);
                )+
            };
        }

        compare_pairs!(
            Rock     Equal   Rock,
            Rock     Less    Paper,
            Rock     Greater Scissors,
            Paper    Greater Rock,
            Paper    Equal   Paper,
            Paper    Less    Scissors,
            Scissors Less    Rock,
            Scissors Greater Paper,
            Scissors Equal   Scissors,
        );
    }
}
