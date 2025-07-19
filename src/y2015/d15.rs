use {
    crate::*,
    arrayvec::ArrayVec,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_res, verify},
        error::Error,
        multi::separated_list1,
        sequence::{preceded, tuple},
        Err, IResult,
    },
    num::{Integer, PrimInt, Signed},
    std::{cmp::Ordering, str::FromStr},
    strum::{EnumCount, EnumIter, EnumVariantNames, IntoEnumIterator, VariantNames},
};

/* --- Day 15: Science for Hungry People ---

Today, you set out on the task of perfecting your milk-dunking cookie recipe. All you have to do is find the right balance of ingredients.

Your recipe leaves room for exactly 100 teaspoons of ingredients. You make a list of the remaining ingredients you could use to finish the recipe (your puzzle input) and their properties per teaspoon:

    capacity (how well it helps the cookie absorb milk)
    durability (how well it keeps the cookie intact when full of milk)
    flavor (how tasty it makes the cookie)
    texture (how it improves the feel of the cookie)
    calories (how many calories it adds to the cookie)

You can only measure ingredients in whole-teaspoon amounts accurately, and you have to be accurate so you can reproduce your results in the future. The total score of a cookie can be found by adding up each of the properties (negative totals become 0) and then multiplying together everything except calories.

For instance, suppose you have these two ingredients:

Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8
Cinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3

Then, choosing to use 44 teaspoons of butterscotch and 56 teaspoons of cinnamon (because the amounts of each ingredient must add up to 100) would result in a cookie with the following properties:

    A capacity of 44*-1 + 56*2 = 68
    A durability of 44*-2 + 56*3 = 80
    A flavor of 44*6 + 56*-2 = 152
    A texture of 44*3 + 56*-1 = 76

Multiplying these together (68 * 80 * 152 * 76, ignoring calories for now) results in a total score of 62842880, which happens to be the best score possible given these ingredients. If any properties had produced a negative total, it would have instead become zero, causing the whole score to multiply to zero.

Given the ingredients in your kitchen and their properties, what is the total score of the highest-scoring cookie you can make?

--- Part Two ---

Your cookie recipe becomes wildly popular! Someone asks if you can make another recipe that has exactly 500 calories per cookie (so they can use it as a meal replacement). Keep the rest of your award-winning process the same (100 teaspoons, same ingredients, same scoring system).

For example, given the ingredients above, if you had instead selected 40 teaspoons of butterscotch and 60 teaspoons of cinnamon (which still adds to 100), the total calorie count would be 40*8 + 60*3 = 500. The total score would go down, though: only 57600000, the best you can do in such trying circumstances.

Given the ingredients in your kitchen and their properties, what is the total score of the highest-scoring cookie you can make with a calorie total of 500? */

#[derive(Clone, Copy, EnumCount, EnumIter, EnumVariantNames, PartialEq)]
#[strum(serialize_all = "snake_case")]
enum PropertyType {
    Capacity,
    Durability,
    Flavor,
    Texture,
    Calories,
}

impl PropertyType {
    fn tag_str(self) -> &'static str {
        &Self::VARIANTS[self as usize]
    }
}

define_super_trait! {
    trait PropertyValue where Self: Default + FromStr + Integer + PrimInt + Signed {}
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Default)]
struct PropertyValues<P: PropertyValue>([P; PropertyType::COUNT]);

impl<P: PropertyValue> Parse for PropertyValues<P> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut property_type_iter: PropertyTypeIter = PropertyType::iter();

        map(
            parse_separated_array(
                move |input| {
                    let property_type: PropertyType = property_type_iter.next().unwrap();

                    map(
                        tuple((tag(property_type.tag_str()), tag(" "), parse_integer)),
                        |(_, _, property_value)| property_value,
                    )(input)
                },
                tag(", "),
            ),
            Self,
        )(input)
    }
}

type IngredientIndexRaw = u8;

const MIN_INGREDIENT_ID_LEN: usize = 1_usize;
const MAX_INGREDIENT_ID_LEN: usize = 12_usize;

type IngredientId = StaticString<MAX_INGREDIENT_ID_LEN>;

type IngredientDataPropertyValueRaw = i8;

#[cfg_attr(test, derive(Debug, PartialEq))]
struct IngredientData(PropertyValues<IngredientDataPropertyValueRaw>);

impl Parse for IngredientData {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(preceded(tag(": "), PropertyValues::parse), Self)(input)
    }
}

type Ingredient = TableElement<IngredientId, IngredientData>;

impl Parse for Ingredient {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                IngredientId::parse_char1(MIN_INGREDIENT_ID_LEN, |c| c.is_ascii_alphabetic()),
                IngredientData::parse,
            )),
            |(id, data)| Self { id, data },
        )(input)
    }
}

const MAX_INGREDIENT_TABLE_LEN: usize = 4_usize;

type IngredientTable = Table<IngredientId, IngredientData, IngredientIndexRaw>;
type IngredientCountArrayVec = ArrayVec<u8, MAX_INGREDIENT_TABLE_LEN>;

struct IngredientCountArrayVecIter {
    ingredients: u8,
    ingredient_count_sum: u8,
    ingredient_counts: IngredientCountArrayVec,
}

impl IngredientCountArrayVecIter {
    fn from_ingredients_and_ingredient_count_sum(
        ingredients: u8,
        ingredient_count_sum: u8,
    ) -> Self {
        let mut iter: Self = Self {
            ingredients,
            ingredient_count_sum,
            ingredient_counts: Default::default(),
        };

        iter.push_new_ingredient_counts();

        iter
    }

    fn remaining_ingredient_count(&self) -> u8 {
        self.ingredient_count_sum - self.ingredient_counts.iter().copied().sum::<u8>()
    }

    fn push_new_ingredient_counts(&mut self) {
        let ingredients: usize = self.ingredients as usize;
        let ingredients_minus_one: usize = ingredients - 1_usize;

        while self.ingredient_counts.len() < ingredients {
            let new_ingredient_count: u8 =
                match self.ingredient_counts.len().cmp(&ingredients_minus_one) {
                    Ordering::Less => 0_u8,
                    Ordering::Equal => self.remaining_ingredient_count(),
                    Ordering::Greater => panic!(),
                };

            self.ingredient_counts.push(new_ingredient_count);
        }
    }
}

impl Iterator for IngredientCountArrayVecIter {
    type Item = IngredientCountArrayVec;

    fn next(&mut self) -> Option<Self::Item> {
        let ingredients: usize = self.ingredients as usize;

        while (1_usize..ingredients).contains(&self.ingredient_counts.len()) {
            let current_count: u8 = self.ingredient_counts.pop().unwrap();

            if current_count < self.remaining_ingredient_count() {
                self.ingredient_counts.push(current_count + 1_u8);
                self.push_new_ingredient_counts();
            }
        }

        (self.ingredient_counts.len() == ingredients).then(|| {
            let ingredient_count_array_vec: IngredientCountArrayVec =
                self.ingredient_counts.clone();

            self.ingredient_counts.pop();

            ingredient_count_array_vec
        })
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug)]
struct HighestScoringCookie {
    #[allow(dead_code)]
    ingredient_counts: IngredientCountArrayVec,
    property_values: PropertyValues<i32>,
    total_score: i32,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(IngredientTable);

impl Solution {
    const INGREDIENT_COUNT_SUM: u8 = 100_u8;
    const CALORIES: i32 = 500_i32;

    fn ingredient_table_len(&self) -> usize {
        self.0.as_slice().len()
    }

    fn iter_all_ingredient_counts(&self) -> impl Iterator<Item = IngredientCountArrayVec> {
        IngredientCountArrayVecIter::from_ingredients_and_ingredient_count_sum(
            self.ingredient_table_len() as u8,
            Self::INGREDIENT_COUNT_SUM,
        )
    }

    fn property_values_for_ingredient_counts(
        &self,
        ingredient_counts: IngredientCountArrayVec,
    ) -> PropertyValues<i32> {
        let mut property_values: PropertyValues<i32> = Default::default();

        for (ingredient, ingredient_count) in self.0.as_slice().iter().zip(ingredient_counts) {
            let ingredient_count: i32 = ingredient_count as i32;

            for (ingredient_property_value, sum_property_value) in ingredient
                .data
                .0
                 .0
                .into_iter()
                .zip(property_values.0.iter_mut())
            {
                *sum_property_value += ingredient_property_value as i32 * ingredient_count;
            }
        }

        property_values
    }

    fn total_score(&self, property_values: &PropertyValues<i32>) -> i32 {
        PropertyType::iter()
            .filter(|&property_type| property_type != PropertyType::Calories)
            .map(|property_type| property_values.0[property_type as usize].max(0_i32))
            .product()
    }

    fn highest_scoring_cookie(&self, calories: Option<i32>) -> HighestScoringCookie {
        self.iter_all_ingredient_counts()
            .map(|ingredient_counts| {
                let property_values: PropertyValues<i32> =
                    self.property_values_for_ingredient_counts(ingredient_counts.clone());
                let total_score: i32 = self.total_score(&property_values);

                HighestScoringCookie {
                    ingredient_counts,
                    property_values,
                    total_score,
                }
            })
            .filter(|highest_scoring_cookie| {
                calories
                    .map(|calories| {
                        highest_scoring_cookie.property_values.0[PropertyType::Calories as usize]
                            == calories
                    })
                    .unwrap_or(true)
            })
            .max_by_key(|highest_scoring_cookie| highest_scoring_cookie.total_score)
            .unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            map_res(
                verify(
                    separated_list1(line_ending, Ingredient::parse),
                    |ingredients: &Vec<Ingredient>| ingredients.len() <= MAX_INGREDIENT_TABLE_LEN,
                ),
                IngredientTable::try_from,
            ),
            |mut ingredient_table| {
                ingredient_table.sort_by_id();

                Self(ingredient_table)
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I feel like there's probably a matrix operation that matches some of the math here, but I'm
    /// too tired to figure that out, plus it wouldn't be generalized (in terms of type information)
    /// over a variable number of ingredients.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        let highest_scoring_cookie: HighestScoringCookie = self.highest_scoring_cookie(None);

        if args.verbose {
            dbg!(highest_scoring_cookie);
        } else {
            dbg!(highest_scoring_cookie.total_score);
        }
    }

    /// Rusts iterators are truly something special.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        let highest_scoring_cookie: HighestScoringCookie =
            self.highest_scoring_cookie(Some(Self::CALORIES));

        if args.verbose {
            dbg!(highest_scoring_cookie);
        } else {
            dbg!(highest_scoring_cookie.total_score);
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8\n\
        Cinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(
                vec![
                    Ingredient {
                        id: "Butterscotch".try_into().unwrap(),
                        data: IngredientData(PropertyValues([-1_i8, -2_i8, 6_i8, 3_i8, 8_i8])),
                    },
                    Ingredient {
                        id: "Cinnamon".try_into().unwrap(),
                        data: IngredientData(PropertyValues([2_i8, 3_i8, -2_i8, -1_i8, 3_i8])),
                    },
                ]
                .try_into()
                .unwrap(),
            )]
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
    fn test_property_values_for_ingredient_counts() {
        for (index, (ingredient_counts, property_values)) in [(
            [44_u8, 56_u8]
                .into_iter()
                .collect::<IngredientCountArrayVec>(),
            PropertyValues([68_i32, 80_i32, 152_i32, 76_i32, 520_i32]),
        )]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).property_values_for_ingredient_counts(ingredient_counts),
                property_values
            );
        }
    }

    #[test]
    fn test_total_score() {
        for (index, (property_values, total_score)) in [(
            PropertyValues([68_i32, 80_i32, 152_i32, 76_i32, 520_i32]),
            62842880_i32,
        )]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).total_score(&property_values), total_score);
        }
    }

    #[test]
    fn test_highest_scoring_cookie() {
        for (index, highest_scoring_cookie) in [HighestScoringCookie {
            ingredient_counts: [44_u8, 56_u8].into_iter().collect(),
            property_values: PropertyValues([68_i32, 80_i32, 152_i32, 76_i32, 520_i32]),
            total_score: 62842880_i32,
        }]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                solution(index).highest_scoring_cookie(None),
                highest_scoring_cookie
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
