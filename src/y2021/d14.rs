use {
    crate::{
        util::minimal_value_with_all_digit_pairs::{try_sequence, MAX_BASE},
        *,
    },
    derive_deref::{Deref, DerefMut},
    nom::{
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_opt, opt},
        error::Error,
        multi::{many0, many0_count, many_m_n},
        sequence::{separated_pair, terminated},
        Err, IResult,
    },
    num::Num,
    static_assertions::const_assert,
    std::{
        mem::{swap, take, transmute},
        ops::AddAssign,
    },
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct ElementIndex(u8);

impl From<usize> for ElementIndex {
    fn from(element_index: usize) -> Self {
        ElementIndex(element_index as u8)
    }
}

impl From<ElementIndex> for usize {
    fn from(element_index: ElementIndex) -> Self {
        element_index.0 as usize
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default, Deref, DerefMut)]
#[repr(transparent)]
struct Polymer(Vec<ElementIndex>);

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Rules {
    Complete(Vec<ElementIndex>),
    Incomplete(Vec<Option<ElementIndex>>),
}

impl Rules {
    fn try_make_complete(&mut self) -> bool {
        match self {
            Self::Complete(_) => true,
            Self::Incomplete(rules) => {
                if rules.iter().all(Option::is_some) {
                    *self = Rules::Complete(take(rules).into_iter().map(Option::unwrap).collect());

                    true
                } else {
                    false
                }
            }
        }
    }

    fn frequencies_steps(&self, num_elements: usize) -> u8 {
        let iter = (0_usize..num_elements).map(|index| index * num_elements + index);

        (if match self {
            Self::Complete(rules) => iter
                .map(|index| rules[index])
                .enumerate()
                .any(|(in_index, out_index)| in_index == out_index.into()),
            Self::Incomplete(rules) => {
                iter.map(|index| rules[index])
                    .enumerate()
                    .any(|(in_index, out_index)| {
                        out_index.map_or(false, |out_index| in_index == out_index.into())
                    })
            }
        } {
            u8::BITS - 1_u32
        } else {
            u8::BITS
        }) as u8
    }
}

impl Default for Rules {
    fn default() -> Self {
        Self::Incomplete(Vec::new())
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct RuleFrequencies {
    rule_counts: Vec<u8>,
    steps: u8,
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Default, Deref, DerefMut)]
#[repr(transparent)]
struct ElementFrequencies(Vec<(char, usize)>);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct Solution {
    elements: Vec<char>,
    polymer_template: Polymer,
    rules: Rules,
    rule_frequencies: RuleFrequencies,
}

type Pair = [u8; Solution::PAIR_SIZE];

impl Solution {
    const PAIR_SIZE: usize = 2_usize;
    const MAX_ELEMENTS_LEN: usize = 1_usize << u8::BITS;

    fn is_ascii_uppercase(c: char) -> bool {
        c.is_ascii_uppercase()
    }

    fn pair_from_valid_slice(input: &[u8]) -> Pair {
        let mut pair: Pair = Pair::default();

        pair.clone_from_slice(input);

        pair
    }

    fn pair_from_valid_str(input: &str) -> Pair {
        Self::pair_from_valid_slice(input.as_bytes())
    }

    fn first_byte(input: &str) -> u8 {
        input.as_bytes()[0_usize]
    }

    fn parse_byte<'i>(input: &'i str) -> IResult<&'i str, u8> {
        map(
            take_while_m_n(1_usize, 1_usize, Self::is_ascii_uppercase),
            Self::first_byte,
        )(input)
    }

    fn parse_pair_insertion_rule<'i>(input: &'i str) -> IResult<&'i str, (Pair, u8)> {
        terminated(
            separated_pair(
                map(
                    take_while_m_n(Self::PAIR_SIZE, Self::PAIR_SIZE, Self::is_ascii_uppercase),
                    Self::pair_from_valid_str,
                ),
                tag(" -> "),
                Self::parse_byte,
            ),
            opt(line_ending),
        )(input)
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut solution: Self = Self::default();

        let (input, polymer_template) = map(
            many0(map_opt(Self::parse_byte, |element| {
                solution.try_get_or_add_element_index(element)
            })),
            Polymer,
        )(input)?;

        solution.polymer_template = polymer_template;

        let (input, _) = many_m_n(2_usize, 2_usize, line_ending)(input)?;

        // The index of `Rules` (list) elements is dependent on the total number of elements in the
        // `Solution`, so iterate once through to finalize that
        many0_count(map_opt(
            Self::parse_pair_insertion_rule,
            |([left_element, right_element], mid_element)| {
                solution.try_get_or_add_element_index(left_element)?;
                solution.try_get_or_add_element_index(right_element)?;
                solution.try_get_or_add_element_index(mid_element)?;

                Some(())
            },
        ))(input)?;

        let num_elements: usize = solution.num_elements();
        let get_index =
            |element: u8| Self::try_get_element_index(&solution.elements, element).unwrap();

        let mut rules: Vec<Option<ElementIndex>> = vec![None; num_elements * num_elements];

        let (input, _) = many0_count(map(
            Self::parse_pair_insertion_rule,
            |([left_element, right_element], mid_element)| {
                rules[Self::rule_index_from_elements(
                    num_elements,
                    get_index(left_element),
                    get_index(right_element),
                )] = Some(get_index(mid_element));
            },
        ))(input)?;

        solution.rules = Rules::Incomplete(rules);
        solution.rules.try_make_complete();
        solution.rule_frequencies = solution.rule_frequencies();

        Ok((input, solution))
    }

    #[inline(always)]
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn try_get_element_index(elements: &Vec<char>, element: u8) -> Option<ElementIndex> {
        let element: char = element as char;

        elements
            .iter()
            .position(|existing_element| *existing_element == element)
            .map(From::from)
    }

    fn try_get_or_add_element_index(&mut self, element: u8) -> Option<ElementIndex> {
        if let Some(element_index) = Self::try_get_element_index(&self.elements, element) {
            Some(element_index)
        } else if self.num_elements() < Self::MAX_ELEMENTS_LEN {
            let element_index: ElementIndex = self.num_elements().into();

            self.elements.push(element as char);

            Some(element_index)
        } else {
            None
        }
    }

    #[inline(always)]
    fn rule_index_from_elements(
        num_elements: usize,
        left_element_index: ElementIndex,
        right_element_index: ElementIndex,
    ) -> usize {
        usize::from(left_element_index) * num_elements + usize::from(right_element_index)
    }

    #[inline(always)]
    fn left_element_index_from_rule_index(num_elements: usize, rule_index: usize) -> ElementIndex {
        (rule_index / num_elements).into()
    }

    #[inline(always)]
    fn rule_index(
        &self,
        left_element_index: ElementIndex,
        right_element_index: ElementIndex,
    ) -> usize {
        Self::rule_index_from_elements(self.num_elements(), left_element_index, right_element_index)
    }

    fn new_element_frequencies(&self) -> ElementFrequencies {
        ElementFrequencies(
            self.elements
                .iter()
                .copied()
                .map(|element| (element, 0_usize))
                .collect(),
        )
    }

    fn element_frequencies_after_pair_insertion_process(
        &self,
        polymer_template: &Polymer,
        mut steps: usize,
    ) -> ElementFrequencies {
        let num_elements: usize = self.num_elements();
        let num_elements_squared: usize = num_elements * num_elements;
        let rule_frequencies_steps: usize = self.rule_frequencies.steps as usize;

        let mut curr_rule_counts: Vec<usize> = vec![0_usize; num_elements * num_elements];
        let mut next_rule_counts: Vec<usize> = curr_rule_counts.clone();

        self.fill_rule_counts(&*polymer_template, &mut curr_rule_counts);

        let increment_counts = |rule_counts: &Vec<u8>,
                                curr_rule_counts: &mut Vec<usize>,
                                next_rule_counts: &mut Vec<usize>| {
            next_rule_counts.fill(0_usize);

            for (rule_counts_for_rule, curr_rule_count) in rule_counts
                .chunks_exact(num_elements_squared)
                .zip(curr_rule_counts.iter().copied())
            {
                for (rule_count_for_rule, next_rule_count) in rule_counts_for_rule
                    .iter()
                    .copied()
                    .zip(next_rule_counts.iter_mut())
                {
                    *next_rule_count += curr_rule_count * rule_count_for_rule as usize;
                }
            }

            swap(curr_rule_counts, next_rule_counts);
        };

        while steps != 0_usize {
            if steps >= rule_frequencies_steps {
                increment_counts(
                    &self.rule_frequencies.rule_counts,
                    &mut curr_rule_counts,
                    &mut next_rule_counts,
                );
                steps -= rule_frequencies_steps;
            } else {
                increment_counts(
                    &self.rule_counts_for_steps(steps),
                    &mut curr_rule_counts,
                    &mut next_rule_counts,
                );
                steps = 0_usize;
            }
        }

        let num_elements: usize = self.num_elements();
        let mut element_frequencies: ElementFrequencies = self.new_element_frequencies();

        for (rule_index, rule_count) in curr_rule_counts.into_iter().enumerate() {
            element_frequencies[usize::from(Self::left_element_index_from_rule_index(
                num_elements,
                rule_index,
            ))]
            .1 += rule_count;
        }

        if let Some(element_index) = polymer_template.last() {
            element_frequencies[usize::from(*element_index)].1 += 1_usize;
        }

        element_frequencies
    }

    fn element_frequency_range(element_frequencies: &ElementFrequencies) -> usize {
        if element_frequencies.len() < 2_usize {
            0_usize
        } else {
            let (min, max): (usize, usize) = element_frequencies
                .iter()
                .copied()
                .fold((usize::MAX, usize::MIN), |(min, max), (_, frequency)| {
                    (min.min(frequency), max.max(frequency))
                });

            max - min
        }
    }

    #[cfg(test)]
    fn polymer_as_string(&self, polymer: Polymer) -> String {
        // SAFETY: `Polymer` is a new-type of `Vec<ElementIndex>`, and `ElementIndex` is a new-type
        // of `u8`
        let mut polymer_bytes: Vec<u8> = unsafe { transmute(polymer) };

        for byte in polymer_bytes.iter_mut() {
            *byte = self.elements[*byte as usize] as u8;
        }

        String::from_utf8(polymer_bytes).unwrap_or_default()
    }

    fn run_pair_insertion_process_for_template(
        &self,
        polymer_template: &Polymer,
        steps: usize,
    ) -> (Polymer, Vec<usize>) {
        if polymer_template.len() < 2_usize {
            (
                polymer_template.clone(),
                if polymer_template.is_empty() {
                    vec![0_usize]
                } else {
                    vec![0_usize, 0_usize]
                },
            )
        } else {
            match &self.rules {
                // The final length will be (`polymer_template.len() - 1) * (1 << steps) + 1`, and
                // we can easily index into the gaps without requiring additional allocation
                Rules::Complete(rules) => {
                    let mut end: usize = polymer_template.len() - 1_usize;
                    let mut step_size: usize = 1_usize << steps;
                    let mut polymer: Polymer =
                        Polymer(vec![ElementIndex(u8::MAX); end * step_size + 1_usize]);
                    let mut offsets: Vec<usize> = vec![0_usize; polymer_template.len()];

                    for (index, element_index) in polymer_template.iter().copied().enumerate() {
                        let step_size_index: usize = index * step_size;

                        offsets[index] = step_size_index;
                        polymer[step_size_index] = element_index;
                    }

                    while step_size > 1_usize {
                        let half_step_size: usize = step_size >> 1_u32;

                        for index in 0_usize..end {
                            let left: usize = index * step_size;

                            polymer[left + half_step_size] =
                                rules[self.rule_index(polymer[left], polymer[left + step_size])];
                        }

                        step_size = half_step_size;
                        end <<= 1_u32;
                    }

                    (polymer, offsets)
                }
                // The final length isn't easily determined, so allocate two and alternate between
                // them
                Rules::Incomplete(rules) => {
                    let capacity: usize = polymer_template.len() * (1_usize << steps);
                    let mut curr_polymer: Polymer = Polymer(Vec::with_capacity(capacity));
                    let mut next_polymer: Polymer = Polymer(Vec::with_capacity(capacity));
                    let mut offsets: Vec<usize> =
                        (0_usize..polymer_template.len()).into_iter().collect();

                    curr_polymer.extend(polymer_template.iter().copied());

                    for _ in 0_usize..steps {
                        next_polymer.clear();

                        let mut offset_index: usize = 0_usize;
                        let mut curr_offset: usize = offsets[offset_index];

                        for (pair_index, pair) in curr_polymer.windows(Self::PAIR_SIZE).enumerate()
                        {
                            if pair_index == curr_offset {
                                offsets[offset_index] = next_polymer.len();
                                offset_index += 1_usize;
                                curr_offset = offsets[offset_index];
                            }

                            next_polymer.push(pair[0_usize]);

                            if let Some(index) =
                                rules[self.rule_index(pair[0_usize], pair[1_usize])]
                            {
                                next_polymer.push(index);
                            }
                        }

                        if let Some(element_index) = curr_polymer.last() {
                            next_polymer.push(*element_index);
                        }

                        swap(&mut curr_polymer, &mut next_polymer);
                    }

                    *offsets.last_mut().unwrap() = curr_polymer.len() - 1_usize;

                    (curr_polymer, offsets)
                }
            }
        }
    }

    #[cfg(test)]
    fn run_pair_insertion_process(&self, steps: usize) -> Polymer {
        self.run_pair_insertion_process_for_template(&self.polymer_template, steps)
            .0
    }

    fn rule_frequencies_polymer_template(&self) -> Polymer {
        match self.num_elements() {
            0_usize => Polymer::default(),
            1_usize => Polymer(vec![ElementIndex(0_u8); 2_usize]),
            num_elements => {
                const_assert!(Solution::MAX_ELEMENTS_LEN <= MAX_BASE);

                // SAFETY: `Polymer` is a new-type of `Vec<ElementIndex>`, and `ElementIndex` is a
                // new-type of `u8`
                unsafe { transmute(try_sequence(num_elements).unwrap()) }
            }
        }
    }

    fn fill_rule_counts<T: AddAssign + Clone + Num>(
        &self,
        polymer: &[ElementIndex],
        rule_counts: &mut [T],
    ) {
        let num_elements: usize = self.num_elements();

        assert_eq!(rule_counts.len(), num_elements * num_elements);

        for pair in polymer.windows(Self::PAIR_SIZE) {
            rule_counts
                [Self::rule_index_from_elements(num_elements, pair[0_usize], pair[1_usize])] +=
                T::one();
        }
    }

    fn rule_counts_for_steps(&self, steps: usize) -> Vec<u8> {
        let num_elements: usize = self.num_elements();
        let rule_frequencies_polymer_template: Polymer = self.rule_frequencies_polymer_template();
        let (rule_frequencies_polymer, offsets): (Polymer, Vec<usize>) =
            self.run_pair_insertion_process_for_template(&rule_frequencies_polymer_template, steps);
        let num_elements_squared: usize = num_elements * num_elements;

        let mut rule_counts: Vec<u8> = vec![0_u8; num_elements_squared * num_elements_squared];

        for (rule_frequency_pair, offset_pair) in rule_frequencies_polymer_template
            .windows(Self::PAIR_SIZE)
            .zip(offsets.windows(Self::PAIR_SIZE))
        {
            let rule_counts_start: usize = Self::rule_index_from_elements(
                num_elements,
                rule_frequency_pair[0_usize],
                rule_frequency_pair[1_usize],
            ) * num_elements_squared;

            self.fill_rule_counts(
                &rule_frequencies_polymer[offset_pair[0_usize]..=offset_pair[1_usize]],
                &mut rule_counts[rule_counts_start..rule_counts_start + num_elements_squared],
            );
        }

        rule_counts
    }

    fn rule_frequencies_steps(&self) -> u8 {
        self.rules.frequencies_steps(self.num_elements())
    }

    fn rule_frequencies(&self) -> RuleFrequencies {
        let steps: u8 = self.rule_frequencies_steps();
        let rule_counts: Vec<u8> = self.rule_counts_for_steps(steps as usize);

        RuleFrequencies { rule_counts, steps }
    }

    fn element_frequencies_after_steps(&self, steps: usize) -> ElementFrequencies {
        self.element_frequencies_after_pair_insertion_process(&self.polymer_template, steps)
    }

    fn frequency_range_after_steps(&self, steps: usize) -> usize {
        Self::element_frequency_range(&self.element_frequencies_after_steps(steps))
    }

    fn print_frequency_range_after_steps(&self, steps: usize, verbose: bool) {
        if verbose {
            let element_frequencies_after_steps: ElementFrequencies =
                self.element_frequencies_after_steps(steps);
            let frequency_range_after_steps: usize =
                Self::element_frequency_range(&element_frequencies_after_steps);

            dbg!(frequency_range_after_steps, element_frequencies_after_steps);
        } else {
            dbg!(self.frequency_range_after_steps(steps));
        }
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        self.print_frequency_range_after_steps(10_usize, args.verbose);
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        self.print_frequency_range_after_steps(40_usize, args.verbose);
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
    use {super::*, lazy_static::lazy_static, std::collections::HashMap};

    const SOLUTION_1_STR: &str = concat!(
        "NNCB\n",
        "\n",
        "CH -> B\n",
        "HH -> N\n",
        "CB -> H\n",
        "NH -> C\n",
        "HB -> C\n",
        "HC -> B\n",
        "HN -> C\n",
        "NN -> C\n",
        "BH -> H\n",
        "NC -> B\n",
        "NB -> B\n",
        "BN -> B\n",
        "BB -> N\n",
        "BC -> B\n",
        "CC -> N\n",
        "CN -> C\n",
    );

    const SOLUTION_2_STR: &str = concat!(
        "AABBA\n",
        "\n",
        "AA -> B\n",
        "AB -> A\n",
        "BA -> A\n",
        "BB -> A\n",
    );

    const SOLUTION_3_STR: &str = "\
        AABBA\n\
        \n\
        AA -> A\n\
        AB -> A\n\
        BA -> B\n";

    lazy_static! {
        static ref SOLUTION_1: Solution = solution_1();
        static ref SOLUTION_2: Solution = solution_2();
        static ref SOLUTION_3: Solution = solution_3();
    }

    macro_rules! polymer { [ $( $element_index:expr ),* $(,)? ] => {
        Polymer(vec![ $( ElementIndex($element_index), )* ])
    } }

    macro_rules! rule_frequencies { { [ $( $rule_count:expr ),* $(,)? ], $steps:expr } => {
        RuleFrequencies {
            rule_counts: vec![ $( $rule_count, )* ],
            steps: $steps
        }
    } }

    macro_rules! solution {
        {
            [ $( $element:expr ),* $(,)? ],
            [ $( $element_index:expr ),* $(,)? ],
            $rules:expr
        } => { {
            let mut solution: Solution = Solution {
                elements: vec![ $( $element, )* ],
                polymer_template: polymer![ $( $element_index, )* ],
                rules: $rules,
                rule_frequencies: RuleFrequencies::default()
            };

            solution.rule_frequencies = solution.rule_frequencies();

            solution
        } };

        {
            [ $( $element:expr ),* $(,)? ],
            [ $( $element_index:expr ),* $(,)? ],
            $rules:expr,
            {
                [ $( $rule_count:expr ),* $(,)? ],
                $steps:expr
            }
        } => {
            Solution {
                elements: vec![ $( $element, )* ],
                polymer_template: polymer![ $( $element_index, )* ],
                rules: $rules,
                rule_frequencies: rule_frequencies! {
                    [ $( $rule_count, )* ],
                    $steps
                }
            }
        };
    }

    fn solution_1() -> Solution {
        solution! {
            ['N', 'C', 'B', 'H'],
            [0, 0, 1, 2],
            Rules::Complete(vec![
                1, 2, 2, 1,
                1, 0, 3, 2,
                2, 2, 0, 3,
                1, 2, 1, 0,
            ].into_iter().map(ElementIndex).collect())
        }
    }

    fn solution_2() -> Solution {
        solution! {
            ['A', 'B'],
            [0, 0, 1, 1, 0],
            Rules::Complete(vec![1, 0, 0, 0].into_iter().map(ElementIndex).collect()),
            {
                [
                    86, 85, 85, 0, // AA
                    85, 86, 85, 0, // AB
                    85, 85, 86, 0, // BA
                    86, 85, 85, 0, // BB
                ],//AA  AB  BA  BB
                8
            }
        }
    }

    fn solution_3() -> Solution {
        solution! {
            ['A', 'B'],
            [0, 0, 1, 1, 0],
            Rules::Incomplete(
                vec![Some(0), Some(0), Some(1), None]
                    .into_iter()
                    .map(|option| option.map(ElementIndex))
                    .collect()
            )
        }
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_1_STR), Ok(solution_1()));
        assert_eq!(Solution::try_from(SOLUTION_2_STR), Ok(solution_2()));
        assert_eq!(Solution::try_from(SOLUTION_3_STR), Ok(solution_3()));
    }

    #[test]
    fn test_run() {
        for (index, polymer) in [
            "NNCB",
            "NCNBCHB",
            "NBCCNBBBCBHCB",
            "NBBBCNCCNBBNBNBBCHBHHBCHB",
            "NBBNBNBBCCNBCNCCNBBNBBNBBBNBBNBBCBHCBHHNHCBBCBHCB",
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                SOLUTION_1.polymer_as_string(SOLUTION_1.run_pair_insertion_process(index)),
                polymer
            );
        }
    }

    #[test]
    fn test_frequencies() {
        for (steps, r_element_frequencies, e_element_frequencies) in [
            (
                10_usize,
                &[
                    ('B', 1749_usize),
                    ('C', 298_usize),
                    ('H', 161_usize),
                    ('N', 865_usize),
                ][..],
            ),
            (
                40_usize,
                &[('B', 2_192_039_569_602_usize), ('H', 3_849_876_073_usize)][..],
            ),
        ]
        .into_iter()
        .map(|(steps, element_frequencies)| {
            (
                steps,
                SOLUTION_1.element_frequencies_after_steps(steps),
                element_frequencies
                    .iter()
                    .copied()
                    .collect::<HashMap<char, usize>>(),
            )
        }) {
            for (element, r_frequency, e_frequency) in r_element_frequencies
                .iter()
                .copied()
                .filter_map(|(element, r_frequency)| {
                    e_element_frequencies
                        .get(&element)
                        .map(|e_frequency| (element, r_frequency, *e_frequency))
                })
            {
                assert_eq!(
                    r_frequency, e_frequency,
                    "steps: {steps}, \
                    element: {element}, \
                    r_frequency: {r_frequency}, \
                    e_frequency: {e_frequency}"
                );
            }
        }
    }

    #[test]
    fn test_rule_frequencies_polymer_template() {
        assert_eq!(
            SOLUTION_1.rule_frequencies_polymer_template(),
            polymer![0, 0, 1, 0, 2, 0, 3, 1, 1, 2, 1, 3, 2, 2, 3, 3, 0]
        );

        let solution_2_rule_frequencies_polymer_template: Polymer =
            SOLUTION_2.rule_frequencies_polymer_template();

        assert_eq!(
            solution_2_rule_frequencies_polymer_template,
            polymer![0, 0, 1, 1, 0]
        );
        assert_eq!(
            solution_2_rule_frequencies_polymer_template,
            SOLUTION_2.polymer_template
        );

        let solution_3_rule_frequencies_polymer_template: Polymer =
            SOLUTION_3.rule_frequencies_polymer_template();

        assert_eq!(
            solution_3_rule_frequencies_polymer_template,
            polymer![0, 0, 1, 1, 0]
        );
        assert_eq!(
            solution_3_rule_frequencies_polymer_template,
            SOLUTION_3.polymer_template
        );
    }

    #[test]
    fn test_rule_frequencies_steps() {
        assert_eq!(SOLUTION_1.rule_frequencies_steps(), u8::BITS as u8);
        assert_eq!(SOLUTION_2.rule_frequencies_steps(), u8::BITS as u8);
        assert_eq!(SOLUTION_3.rule_frequencies_steps(), u8::BITS as u8 - 1_u8);
    }
}
