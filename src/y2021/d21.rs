use {
    crate::*,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        sequence::{delimited, preceded, tuple},
        Err, IResult,
    },
    num::{Integer, NumCast},
    std::{collections::HashMap, fmt::Debug},
};

trait Die {
    fn rolls(&self) -> u32;

    fn roll(&mut self) -> u32;
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct DeterministicDie {
    sides: u32,
    next: u32,
    rolls: u32,
}

impl DeterministicDie {
    const D100: Self = DeterministicDie::new(100_u32);

    const fn new(sides: u32) -> Self {
        Self {
            sides,
            next: 1_u32,
            rolls: 0_u32,
        }
    }
}

impl Die for DeterministicDie {
    fn rolls(&self) -> u32 {
        self.rolls
    }

    fn roll(&mut self) -> u32 {
        let roll: u32 = self.next;

        self.next = self.next % self.sides + 1_u32;
        self.rolls += 1_u32;

        roll
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlayerState<I: Copy + Debug + Integer + NumCast> {
    space: I,
    score: I,
}

impl<I: Copy + Debug + Integer + NumCast> PlayerState<I> {
    fn new(space: I) -> Self {
        Self {
            space,
            score: I::zero(),
        }
    }

    fn take_turn<D: Die>(&mut self, die: &mut D) {
        let mut space_delta: I = I::zero();

        for _ in 0_usize..Solution::ROLLS_PER_TURN {
            space_delta = space_delta + <I as NumCast>::from(die.roll()).unwrap();
        }

        self.space = (self.space + space_delta - I::one())
            % <I as NumCast>::from(Solution::SPACES_U32).unwrap()
            + I::one();
        self.score = self.score + self.space;
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct GameState<D: Die> {
    player_1_state: PlayerState<u32>,
    player_2_state: PlayerState<u32>,
    die: D,
}

impl<D: Die> GameState<D> {
    fn play(&mut self) {
        loop {
            self.player_1_state.take_turn(&mut self.die);

            if self.player_1_state.score >= Solution::DETERMINISTIC_WINNING_SCORE {
                break;
            }

            self.player_2_state.take_turn(&mut self.die);

            if self.player_2_state.score >= Solution::DETERMINISTIC_WINNING_SCORE {
                break;
            }
        }
    }

    fn losing_score(&self) -> u32 {
        self.player_1_state.score.min(self.player_2_state.score)
    }

    fn die_rolls(&self) -> u32 {
        self.die.rolls()
    }
}

type SmallPlayerState = PlayerState<u8>;
type SmallPlayerStatePair = (SmallPlayerState, SmallPlayerState);
type WinningUniverseCounts = (usize, usize);
type SpaceDeltaDistribution = [usize; QuantumCache::SPACE_DELTA_DISTRIBUTION_LEN];

#[derive(Default)]
struct QuantumCache(HashMap<SmallPlayerStatePair, WinningUniverseCounts>);

impl QuantumCache {
    const SIDES: usize = 3_usize;
    const MAX_SPACE_DELTA: usize = QuantumCache::SIDES * Solution::ROLLS_PER_TURN;
    const MIN_SPACE_DELTA: usize = Solution::ROLLS_PER_TURN;
    const SPACE_DELTA_DISTRIBUTION_LEN: usize =
        QuantumCache::MAX_SPACE_DELTA + 1_usize - QuantumCache::MIN_SPACE_DELTA;
    const SPACE_DELTA_DISTRIBUTION: SpaceDeltaDistribution =
        QuantumCache::space_delta_distribution();

    fn get_or_compute(
        &mut self,
        curr_player_state: SmallPlayerState,
        next_player_state: SmallPlayerState,
    ) -> WinningUniverseCounts {
        let small_player_state_pair: SmallPlayerStatePair = (curr_player_state, next_player_state);

        if let Some(winning_universe_counts) = self.0.get(&small_player_state_pair) {
            *winning_universe_counts
        } else if next_player_state.score >= Solution::QUANTUM_WINNING_SCORE {
            let winning_universe_counts: WinningUniverseCounts = (0_usize, 1_usize);

            self.0
                .insert(small_player_state_pair, winning_universe_counts);

            winning_universe_counts
        } else {
            let mut winning_universe_counts: WinningUniverseCounts = (0_usize, 0_usize);

            for (offset_space_delta, frequency) in
                Self::SPACE_DELTA_DISTRIBUTION.into_iter().enumerate()
            {
                const MIN_SPACE_DELTA_MINUS_1: u8 = QuantumCache::MIN_SPACE_DELTA as u8 - 1_u8;

                let mut curr_player_state: SmallPlayerState = curr_player_state;

                curr_player_state.space =
                    (curr_player_state.space + offset_space_delta as u8 + MIN_SPACE_DELTA_MINUS_1)
                        % Solution::SPACES_U8
                        + 1_u8;
                curr_player_state.score += curr_player_state.space;

                let (next_player_winning_universes, curr_player_winning_universes): WinningUniverseCounts =
                    self.get_or_compute(next_player_state, curr_player_state);
                winning_universe_counts.0 += frequency * curr_player_winning_universes;
                winning_universe_counts.1 += frequency * next_player_winning_universes;
            }

            self.0
                .insert(small_player_state_pair, winning_universe_counts);

            winning_universe_counts
        }
    }

    const fn space_delta_distribution() -> SpaceDeltaDistribution {
        const PERMUTATIONS: usize = QuantumCache::SIDES.pow(Solution::ROLLS_PER_TURN as u32);

        let mut space_delta_distribution: SpaceDeltaDistribution =
            [0_usize; QuantumCache::SPACE_DELTA_DISTRIBUTION_LEN];
        let mut permutation: usize = 0_usize;

        while permutation < PERMUTATIONS {
            let mut die_index: usize = 0_usize;
            let mut space_delta: usize = 0_usize;
            let mut remaining_permutation: usize = permutation;

            while die_index < Solution::ROLLS_PER_TURN {
                die_index += 1_usize;
                space_delta += remaining_permutation % QuantumCache::SIDES + 1_usize;
                remaining_permutation /= QuantumCache::SIDES;
            }

            space_delta_distribution[space_delta - QuantumCache::MIN_SPACE_DELTA] += 1_usize;
            permutation += 1_usize;
        }

        space_delta_distribution
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
pub struct Solution {
    player_1_space: u32,
    player_2_space: u32,
}

impl Solution {
    const SPACES_U32: u32 = 10_u32;
    const SPACES_U8: u8 = Solution::SPACES_U32 as u8;
    const ROLLS_PER_TURN: usize = 3_usize;
    const DETERMINISTIC_WINNING_SCORE: u32 = 1000_u32;
    const QUANTUM_WINNING_SCORE: u8 = 21_u8;

    fn new_game<D: Die>(&self, die: D) -> GameState<D> {
        GameState {
            player_1_state: PlayerState::new(self.player_1_space),
            player_2_state: PlayerState::new(self.player_2_space),
            die,
        }
    }

    fn game_state_with_100_sided_deterministic_die(&self) -> GameState<impl Die + Debug> {
        self.new_game(DeterministicDie::D100)
    }

    fn play_with_100_sided_deterministic_die(&self) -> u32 {
        let mut game_state: GameState<_> = self.new_game(DeterministicDie::D100);

        game_state.play();

        game_state.losing_score() * game_state.die_rolls()
    }

    fn play_with_dirac_die(&self) -> WinningUniverseCounts {
        let mut quantum_cache: QuantumCache = Default::default();

        quantum_cache.get_or_compute(
            PlayerState::new(self.player_1_space as u8),
            PlayerState::new(self.player_2_space as u8),
        )
    }

    fn max_of_winning_universe_counts(winning_universe_counts: WinningUniverseCounts) -> usize {
        winning_universe_counts.0.max(winning_universe_counts.1)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                preceded(tag("Player 1 starting position: "), parse_integer::<u32>),
                delimited(
                    tuple((line_ending, tag("Player 2 starting position: "))),
                    parse_integer::<u32>,
                    opt(line_ending),
                ),
            )),
            |(player_1_pos, player_2_pos)| Self {
                player_1_space: player_1_pos,
                player_2_space: player_2_pos,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let mut game_state: GameState<_> = self.game_state_with_100_sided_deterministic_die();

            game_state.play();

            dbg!(
                game_state.losing_score() * game_state.die_rolls(),
                game_state
            );
        } else {
            dbg!(self.play_with_100_sided_deterministic_die());
        }
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let winning_universe_counts: WinningUniverseCounts = self.play_with_dirac_die();

            dbg!(
                winning_universe_counts,
                Self::max_of_winning_universe_counts(winning_universe_counts)
            );
        } else {
            dbg!(Self::max_of_winning_universe_counts(
                self.play_with_dirac_die()
            ));
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

    const SOLUTION_STR: &str = "\
    Player 1 starting position: 4\n\
    Player 2 starting position: 8\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            player_1_space: 4,
            player_2_space: 8,
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_deterministic_die_rolls() {
        let mut d100: DeterministicDie = DeterministicDie::D100;

        for roll in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ] {
            assert_eq!(d100.roll(), roll);
        }
    }

    #[test]
    fn test_player_state_take_turn() {
        let mut game_state: GameState<_> = solution().new_game(DeterministicDie::D100);

        game_state.player_1_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_1_state,
            PlayerState {
                space: 10,
                score: 10,
            }
        );
        game_state.player_2_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_2_state,
            PlayerState { space: 3, score: 3 }
        );
        game_state.player_1_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_1_state,
            PlayerState {
                space: 4,
                score: 14,
            }
        );
        game_state.player_2_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_2_state,
            PlayerState { space: 6, score: 9 }
        );
        game_state.player_1_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_1_state,
            PlayerState {
                space: 6,
                score: 20,
            }
        );
        game_state.player_2_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_2_state,
            PlayerState {
                space: 7,
                score: 16
            }
        );
        game_state.player_1_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_1_state,
            PlayerState {
                space: 6,
                score: 26,
            }
        );
        game_state.player_2_state.take_turn(&mut game_state.die);
        assert_eq!(
            game_state.player_2_state,
            PlayerState {
                space: 6,
                score: 22
            }
        );
    }

    #[test]
    fn test_game_state_play() {
        let mut game_state: GameState<_> = solution().new_game(DeterministicDie::D100);

        game_state.play();

        assert_eq!(
            game_state,
            GameState {
                player_1_state: PlayerState {
                    space: 10,
                    score: 1000
                },
                player_2_state: PlayerState {
                    space: 3,
                    score: 745
                },
                die: DeterministicDie {
                    sides: 100,
                    next: 94,
                    rolls: 993
                }
            }
        );
    }

    #[test]
    fn test_solution_play_with_100_sided_deterministic_die() {
        assert_eq!(solution().play_with_100_sided_deterministic_die(), 739785);
    }

    #[test]
    fn test_solution_play_with_dirac_die() {
        assert_eq!(
            solution().play_with_dirac_die(),
            (444_356_092_776_315_usize, 341_960_390_180_808_usize)
        );
    }
}
