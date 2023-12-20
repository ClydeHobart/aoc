use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::{tag, take_while_m_n},
        character::complete::line_ending,
        combinator::{map, map_opt, map_res, opt},
        error::Error,
        multi::{many0, many0_count},
        sequence::{terminated, tuple},
        Err, IResult,
    },
    std::{
        cmp::Ordering,
        collections::HashMap,
        ops::{Mul, Range},
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, EnumCount, EnumIter, PartialEq)]
enum RatingType {
    ExtremelyCoolLooking,
    Musical,
    Aerodynamic,
    Shiny,
}

impl RatingType {
    const STRS: [&'static str; RatingType::COUNT] = ["x", "m", "a", "s"];

    const fn str(self) -> &'static str {
        Self::STRS[self as usize]
    }

    fn alt_branch<'i>(self) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        map(tag(self.str()), move |_| self)
    }
}

impl Parse for RatingType {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            Self::ExtremelyCoolLooking.alt_branch(),
            Self::Musical.alt_branch(),
            Self::Aerodynamic.alt_branch(),
            Self::Shiny.alt_branch(),
        ))(input)
    }
}

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Comparison {
        LessThan = LESS_THAN = b'<',
        GreaterThan = GREATER_THAN = b'>',
    }
}

impl From<Comparison> for Ordering {
    fn from(value: Comparison) -> Self {
        match value {
            Comparison::LessThan => Self::Less,
            Comparison::GreaterThan => Self::Greater,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Copy)]
struct Condition {
    rating_type: RatingType,
    comparison: Comparison,
    value: u16,
}

impl Condition {
    fn evaluate(&self, part_ratings: &PartRatings) -> bool {
        part_ratings.0[self.rating_type as usize].cmp(&self.value)
            == Ordering::from(self.comparison)
    }

    fn opposite(&self) -> Self {
        match self.comparison {
            Comparison::LessThan => Self {
                rating_type: self.rating_type,
                comparison: Comparison::GreaterThan,
                value: self.value - 1_u16,
            },
            Comparison::GreaterThan => Self {
                rating_type: self.rating_type,
                comparison: Comparison::LessThan,
                value: self.value + 1_u16,
            },
        }
    }
}

impl Parse for Condition {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((RatingType::parse, Comparison::parse, parse_integer::<u16>)),
            |(rating_type, comparison, value)| Self {
                rating_type,
                comparison,
                value,
            },
        )(input)
    }
}

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum Decision {
        Acceptance = ACCEPTANCE = b'A',
        Rejection = REJECTION = b'R',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum Destination<W = u16> {
    Workflow(W),
    Decision(Decision),
}

impl<W: Parse> Parse for Destination<W> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        alt((
            map(Decision::parse, Self::Decision),
            map(W::parse, Self::Workflow),
        ))(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Rule<W = u16> {
    cond: Option<Condition>,
    dest: Destination<W>,
}

impl<W: Parse> Parse for Rule<W> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                opt(terminated(Condition::parse, tag(":"))),
                Destination::<W>::parse,
            )),
            |(cond, dest)| Self { cond, dest },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
struct Label([u8; Self::LEN]);

impl Label {
    const LEN: usize = 3_usize;
}

impl Parse for Label {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            take_while_m_n(1_usize, Self::LEN, |c: char| c.is_ascii_lowercase()),
            |label_str: &str| label_str.try_into().unwrap(),
        )(input)
    }
}

impl TryFrom<&str> for Label {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.len() <= Self::LEN {
            let mut label: Label = Self::default();

            label.0[..value.len()].copy_from_slice(value.as_bytes());

            Ok(label)
        } else {
            Err(())
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Workflow {
    id: Label,
    rule_range: Range<u16>,
}

#[derive(Clone)]
struct PartRatingRange([Range<u16>; RatingType::COUNT]);

impl PartRatingRange {
    const MIN: u16 = 1_u16;
    const MAX: u16 = 4000_u16;
    const START: u16 = Self::MIN;
    const END: u16 = Self::MAX + 1_u16;
    const FULL_RANGE: Range<u16> = Self::START..Self::END;

    fn is_empty(&self) -> bool {
        self.0.iter().any(Range::is_empty)
    }

    fn distinct_combinations(&self) -> usize {
        self.0.iter().map(Range::len).product()
    }
}

impl Default for PartRatingRange {
    fn default() -> Self {
        Self([Self::FULL_RANGE; RatingType::COUNT])
    }
}

impl From<Condition> for PartRatingRange {
    fn from(
        Condition {
            rating_type,
            comparison,
            value,
        }: Condition,
    ) -> Self {
        let mut part_rating_range: Self = Self::default();

        part_rating_range.0[rating_type as usize] = match comparison {
            Comparison::LessThan => Self::START..value,
            Comparison::GreaterThan => value + 1_u16..Self::END,
        };

        part_rating_range
    }
}

impl Mul for PartRatingRange {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut output: Self = Self::default();

        for (index, (lhs_range, rhs_range)) in self.0.into_iter().zip(rhs.0).enumerate() {
            output.0[index] =
                lhs_range.start.max(rhs_range.start)..lhs_range.end.min(rhs_range.end);
        }

        output
    }
}

type PartRatingRangeIndex = u16;

#[derive(Default)]
struct PartRatingRangeMap {
    workflow_index_to_range: HashMap<u16, Range<PartRatingRangeIndex>>,
    part_rating_ranges: Vec<PartRatingRange>,
}

impl PartRatingRangeMap {
    fn populate(&mut self, workflow_list: &WorkflowList, workflow_index: u16) {
        if !self.workflow_index_to_range.contains_key(&workflow_index) {
            for rule in workflow_list.rules(workflow_index) {
                if let Destination::Workflow(child_workflow_index) = &rule.dest {
                    self.populate(workflow_list, *child_workflow_index);
                }
            }

            let range_start: PartRatingRangeIndex =
                self.part_rating_ranges.len() as PartRatingRangeIndex;
            let mut part_rating_range: PartRatingRange = PartRatingRange::default();

            for rule in workflow_list.rules(workflow_index) {
                let (new_part_rating_range, rule_part_rating_range): (
                    PartRatingRange,
                    PartRatingRange,
                ) = rule
                    .cond
                    .as_ref()
                    .map(|cond| {
                        let opposite_cond: Condition = cond.opposite();

                        (
                            part_rating_range.clone() * PartRatingRange::from(opposite_cond),
                            part_rating_range.clone() * PartRatingRange::from(cond.clone()),
                        )
                    })
                    .unwrap_or_else(|| (part_rating_range.clone(), part_rating_range.clone()));

                part_rating_range = new_part_rating_range;

                if !rule_part_rating_range.is_empty() {
                    match &rule.dest {
                        Destination::Workflow(child_workflow_index) => {
                            for part_rating_range_index in self
                                .workflow_index_to_range
                                .get(child_workflow_index)
                                .unwrap()
                                .clone()
                                .as_range_usize()
                            {
                                let child_part_rating_range: PartRatingRange =
                                    rule_part_rating_range.clone()
                                        * self.part_rating_ranges[part_rating_range_index].clone();

                                if !child_part_rating_range.is_empty() {
                                    self.part_rating_ranges.push(child_part_rating_range);
                                }
                            }
                        }
                        Destination::Decision(Decision::Acceptance) => {
                            self.part_rating_ranges.push(rule_part_rating_range);
                        }
                        Destination::Decision(Decision::Rejection) => (),
                    }
                }

                if part_rating_range.is_empty() {
                    break;
                }
            }

            let range_end: PartRatingRangeIndex =
                self.part_rating_ranges.len() as PartRatingRangeIndex;

            self.workflow_index_to_range
                .insert(workflow_index, range_start..range_end);
        }
    }

    fn distinct_combinations(&self, workflow_index: u16) -> usize {
        self.part_rating_ranges[self
            .workflow_index_to_range
            .get(&workflow_index)
            .unwrap()
            .as_range_usize()]
        .iter()
        .map(PartRatingRange::distinct_combinations)
        .sum()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct WorkflowList {
    workflows: Vec<Workflow>,
    rules: Vec<Rule>,
    in_index: u16,
}

impl WorkflowList {
    const IN_LABEL: Label = Label([b'i', b'n', 0_u8]);

    fn parse_workflows_and_rules<'i>(
        input: &'i str,
    ) -> IResult<&'i str, (Vec<Workflow>, Vec<Rule<Label>>)> {
        let mut workflows: Vec<Workflow> = Vec::new();
        let mut rules: Vec<Rule<Label>> = Vec::new();

        let input: &str = many0_count(|input: &'i str| {
            let rule_range_start: u16 = rules.len() as u16;
            let (input, id): (&str, Label) = terminated(Label::parse, tag("{"))(input)?;
            let input: &str = tuple((
                many0_count(|input: &'i str| {
                    let (input, rule): (&str, Rule<Label>) =
                        terminated(Rule::<Label>::parse, opt(tag(",")))(input)?;

                    rules.push(rule);

                    Ok((input, ()))
                }),
                tag("}"),
                opt(line_ending),
            ))(input)?
            .0;

            let rule_range_end: u16 = rules.len() as u16;

            workflows.push(Workflow {
                id,
                rule_range: rule_range_start..rule_range_end,
            });

            Ok((input, ()))
        })(input)?
        .0;

        Ok((input, (workflows, rules)))
    }

    fn rules(&self, workflow_index: u16) -> &[Rule] {
        &self.rules[self.workflows[workflow_index as usize]
            .rule_range
            .as_range_usize()]
    }

    fn decision(&self, part_ratings: &PartRatings) -> Decision {
        let mut workflow_index: u16 = self.in_index;

        loop {
            for rule in self.rules(workflow_index) {
                if rule
                    .cond
                    .as_ref()
                    .map(|cond| cond.evaluate(part_ratings))
                    .unwrap_or(true)
                {
                    match rule.dest {
                        Destination::Workflow(new_workflow_index) => {
                            workflow_index = new_workflow_index;

                            break;
                        }
                        Destination::Decision(decision) => return decision,
                    }
                }
            }
        }
    }

    fn part_rating_range_map(&self) -> PartRatingRangeMap {
        let mut part_rating_range_map: PartRatingRangeMap = PartRatingRangeMap::default();

        part_rating_range_map.populate(self, self.in_index);

        part_rating_range_map
    }
}

impl Parse for WorkflowList {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_res(Self::parse_workflows_and_rules, Self::try_from)(input)
    }
}

impl TryFrom<(Vec<Workflow>, Vec<Rule<Label>>)> for WorkflowList {
    type Error = ();

    fn try_from(
        (workflows, label_rules): (Vec<Workflow>, Vec<Rule<Label>>),
    ) -> Result<Self, Self::Error> {
        let id_to_index: HashMap<Label, u16> = workflows
            .iter()
            .enumerate()
            .map(|(index, workflow)| (workflow.id, index as u16))
            .collect();
        let in_index: u16 = id_to_index.get(&Self::IN_LABEL).copied().ok_or(())?;
        let mut rules: Vec<Rule> = Vec::with_capacity(label_rules.len());

        label_rules.into_iter().try_fold((), |_, rule| {
            let cond: Option<Condition> = rule.cond;
            let dest: Destination = match rule.dest {
                Destination::Workflow(id) => id_to_index
                    .get(&id)
                    .copied()
                    .map_or(Err(()), |index| Ok(Destination::Workflow(index)))?,
                Destination::Decision(decision) => Destination::Decision(decision),
            };

            rules.push(Rule { cond, dest });

            Ok(())
        })?;

        Ok(Self {
            workflows,
            rules,
            in_index,
        })
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Copy, Clone, Default)]
struct PartRatings([u16; RatingType::COUNT]);

impl Parse for PartRatings {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut part_ratings: Self = Self::default();

        let mut input: &str = tag("{")(input)?.0;

        for rating_type in RatingType::iter() {
            let (remaining_input, (_, _, part_rating, _)): (&str, (_, _, u16, _)) =
                tuple((
                    map_opt(RatingType::parse, |input_rating_type| {
                        if input_rating_type == rating_type {
                            Some(())
                        } else {
                            None
                        }
                    }),
                    tag("="),
                    parse_integer::<u16>,
                    opt(tag(",")),
                ))(input)?;

            input = remaining_input;
            part_ratings.0[rating_type as usize] = part_rating;
        }

        let input: &str = tag("}")(input)?.0;

        Ok((input, part_ratings))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    workflow_list: WorkflowList,
    parts: Vec<PartRatings>,
}

impl Solution {
    fn iter_parts(&self) -> impl Iterator<Item = &PartRatings> {
        self.parts.iter()
    }

    fn iter_accepted_parts(&self) -> impl Iterator<Item = &PartRatings> {
        self.iter_parts().filter(|part_ratings| {
            self.workflow_list.decision(part_ratings) == Decision::Acceptance
        })
    }

    fn sum_accepted_part_ratings(&self) -> u32 {
        self.iter_accepted_parts()
            .flat_map(|part_ratings| part_ratings.0.iter().copied().map(u32::from))
            .sum()
    }

    fn distinct_combinations(&self) -> usize {
        self.workflow_list
            .part_rating_range_map()
            .distinct_combinations(self.workflow_list.in_index)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                WorkflowList::parse,
                line_ending,
                many0(terminated(PartRatings::parse, opt(line_ending))),
            )),
            |(workflow_list, _, parts)| Self {
                workflow_list,
                parts,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.sum_accepted_part_ratings());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.distinct_combinations());
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

    const SOLUTION_STR: &'static str = "\
        px{a<2006:qkq,m>2090:A,rfg}\n\
        pv{a>1716:R,A}\n\
        lnx{m>1548:A,A}\n\
        rfg{s<537:gd,x>2440:R,A}\n\
        qs{s>3448:A,lnx}\n\
        qkq{x<1416:A,crn}\n\
        crn{x>2662:A,R}\n\
        in{s<1351:px,qqz}\n\
        qqz{s>2770:qs,m<1801:hdj,R}\n\
        gd{a>3333:R,R}\n\
        hdj{m>838:A,pv}\n\
        \n\
        {x=787,m=2655,a=1222,s=2876}\n\
        {x=1679,m=44,a=2067,s=496}\n\
        {x=2036,m=264,a=79,s=2244}\n\
        {x=2461,m=1339,a=466,s=291}\n\
        {x=2127,m=1623,a=2188,s=1013}\n";

    fn solution() -> &'static Solution {
        use {
            Comparison::{GreaterThan as G, LessThan as L},
            Decision::{Acceptance as A, Rejection as R},
            Destination::{Decision as D, Workflow as W},
            RatingType::{Aerodynamic as a, ExtremelyCoolLooking as x, Musical as m, Shiny as s},
        };

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        macro_rules! workflows {
            [ $( $id:expr => $rule_range:expr, )* ] => { vec![ $( Workflow {
                id: $id.try_into().unwrap(),
                rule_range: $rule_range,
            }, )* ] }
        }

        struct CondTemp(Option<Condition>);

        impl From<()> for CondTemp {
            fn from(_: ()) -> Self {
                Self(None)
            }
        }

        impl From<(RatingType, Comparison, u16)> for CondTemp {
            fn from((rating_type, comparison, value): (RatingType, Comparison, u16)) -> Self {
                Self(Some(Condition {
                    rating_type,
                    comparison,
                    value,
                }))
            }
        }

        macro_rules! rules {
            [ $( $dest:expr $( => $rating_type:ident $comparison:ident $value:expr  )? , )* ] => {
                vec![ $( Rule {
                    cond: CondTemp::from(( $( $rating_type, $comparison, $value )? )).0,
                    dest: $dest,
                }, )* ]
            }
        }

        ONCE_LOCK.get_or_init(|| Solution {
            workflow_list: WorkflowList {
                workflows: workflows![
                    "px" => 0..3, // 0
                    "pv" => 3..5, // 1
                    "lnx" => 5..7, // 2
                    "rfg" => 7..10, // 3
                    "qs" => 10..12, // 4
                    "qkq" => 12..14, // 5
                    "crn" => 14..16, // 6
                    "in" => 16..18, // 7
                    "qqz" => 18..21, // 8
                    "gd" => 21..23, // 9
                    "hdj" => 23..25, // 10
                ],
                rules: rules![
                    W(5) => a L 2006,
                    D(A) => m G 2090,
                    W(3),
                    D(R) => a G 1716,
                    D(A),
                    D(A) => m G 1548,
                    D(A),
                    W(9) => s L 537,
                    D(R) => x G 2440,
                    D(A),
                    D(A) => s G 3448,
                    W(2),
                    D(A) => x L 1416,
                    W(6),
                    D(A) => x G 2662,
                    D(R),
                    W(0) => s L 1351,
                    W(8),
                    W(4) => s G 2770,
                    W(10) => m L 1801,
                    D(R),
                    D(R) => a G 3333,
                    D(R),
                    D(A) => m G 838,
                    W(1),
                ],
                in_index: 7_u16,
            },
            parts: vec![
                PartRatings([787, 2655, 1222, 2876]),
                PartRatings([1679, 44, 2067, 496]),
                PartRatings([2036, 264, 79, 2244]),
                PartRatings([2461, 1339, 466, 291]),
                PartRatings([2127, 1623, 2188, 1013]),
            ],
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_accepted_parts() {
        assert_eq!(
            solution()
                .iter_accepted_parts()
                .copied()
                .collect::<Vec<PartRatings>>(),
            vec![
                PartRatings([787, 2655, 1222, 2876]),
                PartRatings([2036, 264, 79, 2244]),
                PartRatings([2127, 1623, 2188, 1013]),
            ]
        )
    }

    #[test]
    fn test_sum_accepted_part_ratings() {
        assert_eq!(solution().sum_accepted_part_ratings(), 19114);
    }

    #[test]
    fn test_condition_opposite() {
        use {
            Comparison::{GreaterThan as G, LessThan as L},
            RatingType::ExtremelyCoolLooking as X,
        };

        assert_eq!(
            Condition {
                rating_type: X,
                comparison: L,
                value: 2000
            }
            .opposite(),
            Condition {
                rating_type: X,
                comparison: G,
                value: 1999
            }
        );
        assert_eq!(
            Condition {
                rating_type: X,
                comparison: G,
                value: 1999
            }
            .opposite(),
            Condition {
                rating_type: X,
                comparison: L,
                value: 2000
            }
        );
    }

    #[test]
    fn test_distinct_combinations() {
        assert_eq!(solution().distinct_combinations(), 167409079868000_usize);
    }
}
