use {
    crate::*,
    nom::{combinator::map, error::Error, Err, IResult},
};

/* --- Day 14: Chocolate Charts ---

You finally have a chance to look at all of the produce moving around. Chocolate, cinnamon, mint, chili peppers, nutmeg, vanilla... the Elves must be growing these plants to make hot chocolate! As you realize this, you hear a conversation in the distance. When you go to investigate, you discover two Elves in what appears to be a makeshift underground kitchen/laboratory.

The Elves are trying to come up with the ultimate hot chocolate recipe; they're even maintaining a scoreboard which tracks the quality score (0-9) of each recipe.

Only two recipes are on the board: the first recipe got a score of 3, the second, 7. Each of the two Elves has a current recipe: the first Elf starts with the first recipe, and the second Elf starts with the second recipe.

To create new recipes, the two Elves combine their current recipes. This creates new recipes from the digits of the sum of the current recipes' scores. With the current recipes' scores of 3 and 7, their sum is 10, and so two new recipes would be created: the first with score 1 and the second with score 0. If the current recipes' scores were 2 and 3, the sum, 5, would only create one recipe (with a score of 5) with its single digit.

The new recipes are added to the end of the scoreboard in the order they are created. So, after the first round, the scoreboard is 3, 7, 1, 0.

After all new recipes are added to the scoreboard, each Elf picks a new current recipe. To do this, the Elf steps forward through the scoreboard a number of recipes equal to 1 plus the score of their current recipe. So, after the first round, the first Elf moves forward 1 + 3 = 4 times, while the second Elf moves forward 1 + 7 = 8 times. If they run out of recipes, they loop back around to the beginning. After the first round, both Elves happen to loop around until they land on the same recipe that they had in the beginning; in general, they will move to different recipes.

Drawing the first Elf as parentheses and the second Elf as square brackets, they continue this process:

(3)[7]
(3)[7] 1  0
 3  7  1 [0](1) 0
 3  7  1  0 [1] 0 (1)
(3) 7  1  0  1  0 [1] 2
 3  7  1  0 (1) 0  1  2 [4]
 3  7  1 [0] 1  0 (1) 2  4  5
 3  7  1  0 [1] 0  1  2 (4) 5  1
 3 (7) 1  0  1  0 [1] 2  4  5  1  5
 3  7  1  0  1  0  1  2 [4](5) 1  5  8
 3 (7) 1  0  1  0  1  2  4  5  1  5  8 [9]
 3  7  1  0  1  0  1 [2] 4 (5) 1  5  8  9  1  6
 3  7  1  0  1  0  1  2  4  5 [1] 5  8  9  1 (6) 7
 3  7  1  0 (1) 0  1  2  4  5  1  5 [8] 9  1  6  7  7
 3  7 [1] 0  1  0 (1) 2  4  5  1  5  8  9  1  6  7  7  9
 3  7  1  0 [1] 0  1  2 (4) 5  1  5  8  9  1  6  7  7  9  2

The Elves think their skill will improve after making a few recipes (your puzzle input). However, that could take ages; you can speed this up considerably by identifying the scores of the ten recipes after that. For example:

    If the Elves think their skill will improve after making 9 recipes, the scores of the ten recipes after the first nine on the scoreboard would be 5158916779 (highlighted in the last line of the diagram).
    After 5 recipes, the scores of the next ten would be 0124515891.
    After 18 recipes, the scores of the next ten would be 9251071085.
    After 2018 recipes, the scores of the next ten would be 5941429882.

What are the scores of the ten recipes immediately after the number of recipes in your puzzle input?

--- Part Two ---

As it turns out, you got the Elves' plan backwards. They actually want to know how many recipes appear on the scoreboard to the left of the first recipes whose scores are the digits from your puzzle input.

    51589 first appears after 9 recipes.
    01245 first appears after 5 recipes.
    92510 first appears after 18 recipes.
    59414 first appears after 2018 recipes.

How many recipes appear on the scoreboard to the left of the score sequence in your puzzle input? */

struct RecipeTracker {
    quality_scores: String,
    elf_a: usize,
    elf_b: usize,
}

impl RecipeTracker {
    const QUALITY_SCORE_0: u8 = 3_u8;
    const QUALITY_SCORE_1: u8 = 7_u8;

    fn new() -> Self {
        Self {
            quality_scores: [
                Self::byte_to_char(Self::QUALITY_SCORE_0),
                Self::byte_to_char(Self::QUALITY_SCORE_1),
            ]
            .into_iter()
            .collect(),
            elf_a: 0_usize,
            elf_b: 1_usize,
        }
    }

    const fn byte_to_char(b: u8) -> char {
        (b + b'0') as char
    }

    fn add_recipes(&mut self) {
        let quality_score_a: u8 = self.quality_scores.as_bytes()[self.elf_a] - b'0';
        let quality_score_b: u8 = self.quality_scores.as_bytes()[self.elf_b] - b'0';
        let quality_score_sum: u8 = quality_score_a + quality_score_b;

        if quality_score_sum >= 10_u8 {
            self.quality_scores.push(Self::byte_to_char(1_u8));
        }

        self.quality_scores
            .push(Self::byte_to_char(quality_score_sum % 10_u8));
        self.elf_a = (self.elf_a + quality_score_a as usize + 1_usize) % self.quality_scores.len();
        self.elf_b = (self.elf_b + quality_score_b as usize + 1_usize) % self.quality_scores.len();
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(usize);

impl Solution {
    fn ten_scores_after_improvement(&self) -> String {
        let mut recipe_tracker: RecipeTracker = RecipeTracker::new();
        let needed_quality_scores: usize = self.0 + 10_usize;

        while recipe_tracker.quality_scores.len() < needed_quality_scores {
            recipe_tracker.add_recipes();
        }

        recipe_tracker.quality_scores[self.0..self.0 + 10_usize].into()
    }

    fn recipes_to_the_left_of_input(&self) -> usize {
        let input: String = format!("{}", self.0);
        let input_len_plus_one: usize = input.len() + 1_usize;

        let mut recipe_tracker: RecipeTracker = RecipeTracker::new();

        while !recipe_tracker.quality_scores[recipe_tracker
            .quality_scores
            .len()
            .saturating_sub(input_len_plus_one)..]
            .contains(&input)
        {
            recipe_tracker.add_recipes();
        }

        recipe_tracker.quality_scores.find(&input).unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(parse_integer, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Easy.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.ten_scores_after_improvement());
    }

    /// This question was phrased horribly.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.recipes_to_the_left_of_input());
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

    const SOLUTION_STRS: &'static [&'static str] = &["9", "5", "18", "2018"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(9_usize),
                Solution(5_usize),
                Solution(18_usize),
                Solution(2018_usize),
            ]
        })[index]
    }

    #[test]
    fn test_try_from_str() {
        for (index, solution_str) in SOLUTION_STRS.iter().copied().enumerate() {
            assert_eq!(
                Solution::try_from(solution_str).as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_ten_scores_after_improvement() {
        for (index, ten_scores_after_improvement) in
            ["5158916779", "0124515891", "9251071085", "5941429882"]
                .into_iter()
                .enumerate()
        {
            assert_eq!(
                solution(index).ten_scores_after_improvement(),
                ten_scores_after_improvement
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
