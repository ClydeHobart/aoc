use {
    crate::*,
    bitvec::prelude::*,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, satisfy},
        combinator::{map, map_opt, map_res, success},
        error::Error,
        multi::{many_m_n, separated_list0, separated_list1},
        sequence::{separated_pair, terminated, tuple},
        Err, IResult,
    },
    static_assertions::const_assert,
    std::{
        collections::VecDeque,
        ops::{Range, RangeInclusive},
        str::{from_utf8_unchecked, FromStr},
    },
};

/* --- Day 5: Print Queue ---

Satisfied with their search on Ceres, the squadron of scholars suggests subsequently scanning the stationery stacks of sub-basement 17.

The North Pole printing department is busier than ever this close to Christmas, and while The Historians continue their search of this historically significant facility, an Elf operating a very familiar printer beckons you over.

The Elf must recognize you, because they waste no time explaining that the new sleigh launch safety manual updates won't print correctly. Failure to update the safety manuals would be dire indeed, so you offer your services.

Safety protocols clearly indicate that new pages for the safety manuals must be printed in a very specific order. The notation X|Y means that if both page number X and page number Y are to be produced as part of an update, page number X must be printed at some point before page number Y.

The Elf has for you both the page ordering rules and the pages to produce in each update (your puzzle input), but can't figure out whether each update has the pages in the right order.

For example:

47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47

The first section specifies the page ordering rules, one per line. The first rule, 47|53, means that if an update includes both page number 47 and page number 53, then page number 47 must be printed at some point before page number 53. (47 doesn't necessarily need to be immediately before 53; other pages are allowed to be between them.)

The second section specifies the page numbers of each update. Because most safety manuals are different, the pages needed in the updates are different too. The first update, 75,47,61,53,29, means that the update consists of page numbers 75, 47, 61, 53, and 29.

To get the printers going as soon as possible, start by identifying which updates are already in the right order.

In the above example, the first update (75,47,61,53,29) is in the right order:

    75 is correctly first because there are rules that put each other page after it: 75|47, 75|61, 75|53, and 75|29.
    47 is correctly second because 75 must be before it (75|47) and every other page must be after it according to 47|61, 47|53, and 47|29.
    61 is correctly in the middle because 75 and 47 are before it (75|61 and 47|61) and 53 and 29 are after it (61|53 and 61|29).
    53 is correctly fourth because it is before page number 29 (53|29).
    29 is the only page left and so is correctly last.

Because the first update does not include some page numbers, the ordering rules involving those missing page numbers are ignored.

The second and third updates are also in the correct order according to the rules. Like the first update, they also do not include every page number, and so only some of the ordering rules apply - within each update, the ordering rules that involve missing page numbers are not used.

The fourth update, 75,97,47,61,53, is not in the correct order: it would print 75 before 97, which violates the rule 97|75.

The fifth update, 61,13,29, is also not in the correct order, since it breaks the rule 29|13.

The last update, 97,13,75,29,47, is not in the correct order due to breaking several rules.

For some reason, the Elves also need to know the middle page number of each update being printed. Because you are currently only printing the correctly-ordered updates, you will need to find the middle page number of each correctly-ordered update. In the above example, the correctly-ordered updates are:

75,47,61,53,29
97,61,53,29,13
75,29,13

These have middle page numbers of 61, 53, and 29 respectively. Adding these page numbers together gives 143.

Of course, you'll need to be careful: the actual list of page ordering rules is bigger and more complicated than the above example.

Determine which updates are already in the correct order. What do you get if you add up the middle page number from those correctly-ordered updates?

--- Part Two ---

While the Elves get to work printing the correctly-ordered updates, you have a little time to fix the rest of them.

For each of the incorrectly-ordered updates, use the page ordering rules to put the page numbers in the right order. For the above example, here are the three incorrectly-ordered updates and their correct orderings:

    75,97,47,61,53 becomes 97,75,47,61,53.
    61,13,29 becomes 61,29,13.
    97,13,75,29,47 becomes 97,75,47,29,13.

After taking only the incorrectly-ordered updates and ordering them correctly, their middle page numbers are 47, 29, and 47. Adding these together produces 123.

Find the updates which are not in the correct order. What do you get if you add up the middle page numbers after correctly ordering just those updates? */

type PageId = [u8; PageData::ID_LEN];
type PageIndexRaw = u8;

const_assert!(PageIndexRaw::MAX as usize >= PageData::DISTINCT_ID_COUNT);

type PageIndex = Index<PageIndexRaw>;
type PageBitArray = BitArr!(for PageData::DISTINCT_ID_COUNT, in u32);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct PageData {
    pages_before: PageBitArray,
    pages_after: PageBitArray,
}

impl PageData {
    const ID_DIGIT_START: u8 = b'1';
    const ID_DIGIT_END: u8 = b'9';
    const ID_DIGIT_RANGE: SmallRangeInclusive<u8> =
        SmallRangeInclusive::new(Self::ID_DIGIT_START, Self::ID_DIGIT_END);
    const ID_DIGIT_RANGE_INCLUSIVE: RangeInclusive<u8> = Self::ID_DIGIT_RANGE.as_range_inclusive();
    const ID_DIGIT_RANGE_LEN: usize = const_range_inclusive_len!(Self::ID_DIGIT_RANGE);
    const ID_LEN: usize = 2_usize;
    const DISTINCT_ID_COUNT: usize = Self::ID_DIGIT_RANGE_LEN.pow(Self::ID_LEN as u32);

    fn parse_page_id<'i>(input: &'i str) -> IResult<&'i str, PageId> {
        let mut page_id: PageId = PageId::default();
        let mut digit_index: usize = 0_usize;

        let input: &str = many_m_n(
            Self::ID_LEN,
            Self::ID_LEN,
            map(
                satisfy(|c| {
                    u8::try_from(c)
                        .ok()
                        .map_or(false, |b| Self::ID_DIGIT_RANGE_INCLUSIVE.contains(&b))
                }),
                |c| {
                    page_id[digit_index] = c as u8;
                    digit_index += 1_usize;
                },
            ),
        )(input)?
        .0;

        Ok((input, page_id))
    }

    fn parse_page_ordering_rule<'i>(input: &'i str) -> IResult<&'i str, (PageId, PageId)> {
        separated_pair(Self::parse_page_id, tag("|"), Self::parse_page_id)(input)
    }

    fn page_number_from_page_id(page_id: PageId) -> u8 {
        // SAFETY: see `parse_page_id`
        u8::from_str(unsafe { from_utf8_unchecked(&page_id) }).unwrap()
    }
}

type Page = TableElement<PageId, PageData>;
type PageTable = Table<PageId, PageData, PageIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Update {
    page_range: Range<u16>,
}

impl Update {
    fn is_in_correct_order(&self, solution: &Solution) -> bool {
        solution.pages[self.page_range.as_range_usize()]
            .windows(2_usize)
            .all(|pages| {
                let page_before: PageIndex = *pages.first().unwrap();
                let page_after: PageIndex = *pages.last().unwrap();

                // solution.page_table.as_slice()[page_before.get()]
                //     .data
                //     .pages_after[page_after.get()]
                //     && solution.page_table.as_slice()[page_after.get()]
                //         .data
                //         .pages_before[page_before.get()]

                // They need not have a relationship in the affirmative, but no relationship in the negative
                !solution.page_table.as_slice()[page_before.get()]
                    .data
                    .pages_before[page_after.get()]
                    && !solution.page_table.as_slice()[page_after.get()]
                        .data
                        .pages_after[page_before.get()]
            })
    }

    fn try_compute_correct_order(
        &self,
        solution: &Solution,
        update_correct_order_finder: &mut UpdateCorrectOrderFinder,
    ) -> Option<Vec<PageIndex>> {
        update_correct_order_finder.enabled_pages = PageBitArray::ZERO;

        for page_index in solution.pages[self.page_range.as_range_usize()].iter() {
            update_correct_order_finder
                .enabled_pages
                .set(page_index.get(), true);
        }

        update_correct_order_finder.run()
    }

    fn middle_page_number(&self, solution: &Solution) -> u8 {
        let middle_page: u16 = (self.page_range.end + self.page_range.start) / 2_u16;

        PageData::page_number_from_page_id(
            solution.page_table.as_slice()[solution.pages[middle_page as usize].get()].id,
        )
    }
}

struct UpdateCorrectOrderFinder<'p> {
    page_table: &'p PageTable,
    disabled_page_data: Vec<PageData>,
    enabled_pages: PageBitArray,
}

impl<'p> UpdateCorrectOrderFinder<'p> {
    fn enabled_page_bit_array(
        base_page_bit_array: &PageBitArray,
        disabled_page_bit_array: &PageBitArray,
    ) -> PageBitArray {
        *base_page_bit_array & !*disabled_page_bit_array
    }

    fn new(page_table: &'p PageTable) -> Self {
        let disabled_page_data: Vec<PageData> =
            vec![PageData::default(); page_table.as_slice().len()];
        let enabled_pages: PageBitArray = PageBitArray::ZERO;

        Self {
            page_table,
            disabled_page_data,
            enabled_pages,
        }
    }

    fn iter_page_indices(&self) -> impl Iterator<Item = PageIndex> + '_ {
        (0_usize..self.disabled_page_data.len())
            .map(PageIndex::from)
            .filter(|page_index| self.enabled_pages[page_index.get()])
    }

    fn get_enabled_page_data(&self, page_index: PageIndex) -> PageData {
        let base_page_data: &PageData = &self.page_table.as_slice()[page_index.get()].data;
        let disabled_page_data: &PageData = &self.disabled_page_data[page_index.get()];

        PageData {
            pages_before: Self::enabled_page_bit_array(
                &base_page_data.pages_before,
                &disabled_page_data.pages_before,
            ),
            pages_after: Self::enabled_page_bit_array(
                &base_page_data.pages_after,
                &disabled_page_data.pages_after,
            ),
        }
    }
}

impl<'p> Kahn for UpdateCorrectOrderFinder<'p> {
    type Vertex = PageIndex;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>) {
        initial_set.clear();
        initial_set.extend(
            self.iter_page_indices()
                .filter(|page_index| !self.has_in_neighbors(&page_index)),
        );
    }

    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(
            self.get_enabled_page_data(*vertex)
                .pages_after
                .iter_ones()
                .map(PageIndex::from),
        )
    }

    fn remove_edge(&mut self, from: &Self::Vertex, to: &Self::Vertex) {
        self.disabled_page_data[from.get()]
            .pages_after
            .set(to.get(), true);
        self.disabled_page_data[to.get()]
            .pages_before
            .set(from.get(), true);
    }

    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool {
        self.get_enabled_page_data(*vertex).pages_before.any()
    }

    fn any_edges_exist(&self) -> bool {
        self.iter_page_indices()
            .any(|page_index| self.has_in_neighbors(&page_index))
    }

    fn reset(&mut self) {
        let disabled_pages: PageBitArray = !self.enabled_pages;

        self.disabled_page_data.fill(PageData {
            pages_before: disabled_pages,
            pages_after: disabled_pages,
        });
    }

    fn order_set(&self, _set: &mut VecDeque<Self::Vertex>) {}
}

/// # Invariants
///
/// 1. A page ID (AKA "page number", but not "page index") must consist of 2 digits in the range
///     [1,9].
/// 2. An update must consist of a positive odd number of pages.
/// 3. The sum of page counts for each update must be no more than `u16::MAX`.
///
/// # Corollaries
///
/// 1. There can be at most 9 * 9 = 81 distinct pages.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    page_table: PageTable,
    pages: Vec<PageIndex>,
    updates: Vec<Update>,
}

impl Solution {
    fn correct_order_update_middle_page_number_sum(&self) -> u32 {
        self.updates
            .iter()
            .filter(|update| update.is_in_correct_order(self))
            .map(|update| update.middle_page_number(self) as u32)
            .sum()
    }

    fn try_incorrect_order_update_corrected_order_middle_page_number_sum(&self) -> Option<u32> {
        let mut update_correct_order_finder: UpdateCorrectOrderFinder =
            UpdateCorrectOrderFinder::new(&self.page_table);

        self.updates
            .iter()
            .filter(|update| !update.is_in_correct_order(self))
            .try_fold(0_u32, |sum, update| {
                update
                    .try_compute_correct_order(self, &mut update_correct_order_finder)
                    .map(|correct_order| {
                        sum + PageData::page_number_from_page_id(
                            self.page_table.as_slice()
                                [correct_order[correct_order.len() / 2_usize].get()]
                            .id,
                        ) as u32
                    })
            })
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let mut page_table: PageTable = PageTable::new();

        let updates_input: &str = terminated(
            separated_list0(
                line_ending,
                map(
                    PageData::parse_page_ordering_rule,
                    |(page_id_before, page_id_after)| {
                        page_table.find_or_add_index(&page_id_before);
                        page_table.find_or_add_index(&page_id_after);
                    },
                ),
            ),
            tuple((line_ending, line_ending)),
        )(input)?
        .0;

        separated_list0(
            line_ending,
            map_opt(
                separated_list1(
                    tag(","),
                    map(PageData::parse_page_id, |page_id| {
                        page_table.find_or_add_index(&page_id);
                    }),
                ),
                |page_ids| (page_ids.len() % 2_usize == 1_usize).then_some(()),
            ),
        )(updates_input)?;

        page_table.as_slice_mut().sort_by_key(|page| page.id);

        let input: &str = terminated(
            separated_list0(
                line_ending,
                map(
                    PageData::parse_page_ordering_rule,
                    |(page_id_before, page_id_after)| {
                        let page_index_before: PageIndex = page_table.find_index(&page_id_before);
                        let page_index_after: PageIndex = page_table.find_index(&page_id_after);

                        assert!(page_index_before.is_valid());
                        assert!(page_index_after.is_valid());

                        let pages: &mut [Page] = page_table.as_slice_mut();

                        pages[page_index_before.get()]
                            .data
                            .pages_after
                            .set(page_index_after.get(), true);
                        pages[page_index_after.get()]
                            .data
                            .pages_before
                            .set(page_index_before.get(), true);
                    },
                ),
            ),
            tuple((line_ending, line_ending)),
        )(input)?
        .0;

        let mut pages: Vec<PageIndex> = Vec::new();
        let mut updates: Vec<Update> = Vec::new();

        let input: &str = separated_list0(line_ending, |input: &'i str| {
            let update_page_start: u16 = map_res(success(()), |_| pages.len().try_into())(input)?.1;
            let input: &'i str = separated_list1(
                tag(","),
                map(PageData::parse_page_id, |page_id| {
                    let page_index: PageIndex = page_table.find_index(&page_id);

                    assert!(page_index.is_valid());

                    pages.push(page_index);
                }),
            )(input)?
            .0;
            let update_page_end: u16 = map_res(success(()), |_| pages.len().try_into())(input)?.1;

            updates.push(Update {
                page_range: update_page_start..update_page_end,
            });

            Ok((input, ()))
        })(input)?
        .0;

        Ok((
            input,
            Self {
                page_table,
                pages,
                updates,
            },
        ))
    }
}

impl RunQuestions for Solution {
    // Parsing this took forever because I accidentally used a streaming variant instead of the
    // complete variant. Also, the user data has cycles in it?
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.correct_order_update_middle_page_number_sum());
    }

    // Lots of `Vec` allocation, not great, but I'm not going to alter the graph algorithm interface
    // because this one instance wants to run it numerous times.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_incorrect_order_update_corrected_order_middle_page_number_sum());
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
        47|53\n\
        97|13\n\
        97|61\n\
        97|47\n\
        75|29\n\
        61|13\n\
        75|53\n\
        29|13\n\
        97|29\n\
        53|29\n\
        61|53\n\
        97|53\n\
        61|29\n\
        47|13\n\
        75|47\n\
        97|75\n\
        47|61\n\
        75|61\n\
        47|29\n\
        75|13\n\
        53|13\n\
        \n\
        75,47,61,53,29\n\
        97,61,53,29,13\n\
        75,29,13\n\
        75,97,47,61,53\n\
        61,13,29\n\
        97,13,75,29,47\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            macro_rules! id {
                ($id:expr) => {{
                    let mut id: [u8; PageData::ID_LEN] = Default::default();

                    id.copy_from_slice($id.as_bytes());

                    id
                }};
            }

            macro_rules! pages {
                [ $($page:expr),* ] => { {
                    let mut pages: PageBitArray = PageBitArray::ZERO;

                    for (index, page) in [ $( $page, )* ].into_iter().enumerate() {
                        pages.set(index, page == 1);
                    }

                    pages
                } }
            }

            vec![Solution {
                page_table: vec![
                    Page {
                        id: id!("13"),
                        data: PageData {
                            pages_before: pages![0, 1, 1, 1, 1, 1, 1],
                            pages_after: PageBitArray::ZERO,
                        },
                    },
                    Page {
                        id: id!("29"),
                        data: PageData {
                            pages_before: pages![0, 0, 1, 1, 1, 1, 1],
                            pages_after: pages![1, 0, 0, 0, 0, 0, 0],
                        },
                    },
                    Page {
                        id: id!("47"),
                        data: PageData {
                            pages_before: pages![0, 0, 0, 0, 0, 1, 1],
                            pages_after: pages![1, 1, 0, 1, 1, 0, 0],
                        },
                    },
                    Page {
                        id: id!("53"),
                        data: PageData {
                            pages_before: pages![0, 0, 1, 0, 1, 1, 1],
                            pages_after: pages![1, 1, 0, 0, 0, 0, 0],
                        },
                    },
                    Page {
                        id: id!("61"),
                        data: PageData {
                            pages_before: pages![0, 0, 1, 0, 0, 1, 1],
                            pages_after: pages![1, 1, 0, 1, 0, 0, 0],
                        },
                    },
                    Page {
                        id: id!("75"),
                        data: PageData {
                            pages_before: pages![0, 0, 0, 0, 0, 0, 1],
                            pages_after: pages![1, 1, 1, 1, 1, 0, 0],
                        },
                    },
                    Page {
                        id: id!("97"),
                        data: PageData {
                            pages_before: PageBitArray::ZERO,
                            pages_after: pages![1, 1, 1, 1, 1, 1, 0],
                        },
                    },
                ]
                .try_into()
                .unwrap(),
                pages: vec![
                    5_usize.into(),
                    2_usize.into(),
                    4_usize.into(),
                    3_usize.into(),
                    1_usize.into(),
                    6_usize.into(),
                    4_usize.into(),
                    3_usize.into(),
                    1_usize.into(),
                    0_usize.into(),
                    5_usize.into(),
                    1_usize.into(),
                    0_usize.into(),
                    5_usize.into(),
                    6_usize.into(),
                    2_usize.into(),
                    4_usize.into(),
                    3_usize.into(),
                    4_usize.into(),
                    0_usize.into(),
                    1_usize.into(),
                    6_usize.into(),
                    0_usize.into(),
                    5_usize.into(),
                    1_usize.into(),
                    2_usize.into(),
                ],
                updates: vec![
                    Update {
                        page_range: 0_u16..5_u16,
                    },
                    Update {
                        page_range: 5_u16..10_u16,
                    },
                    Update {
                        page_range: 10_u16..13_u16,
                    },
                    Update {
                        page_range: 13_u16..18_u16,
                    },
                    Update {
                        page_range: 18_u16..21_u16,
                    },
                    Update {
                        page_range: 21_u16..26_u16,
                    },
                ],
            }]
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
    fn test_is_in_correct_order() {
        for (index, is_in_correct_order) in [vec![true, true, true, false, false, false]]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);

            assert_eq!(
                solution
                    .updates
                    .iter()
                    .map(|update| update.is_in_correct_order(solution))
                    .collect::<Vec<bool>>(),
                is_in_correct_order
            );
        }
    }

    #[test]
    fn test_correct_order_update_middle_page_number_sum() {
        for (index, correct_order_update_middle_page_number_sum) in
            [143_u32].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).correct_order_update_middle_page_number_sum(),
                correct_order_update_middle_page_number_sum
            );
        }
    }

    #[test]
    fn test_try_incorrect_order_update_corrected_order_middle_page_number_sum() {
        for (index, incorrect_order_update_corrected_order_middle_page_number_sum) in
            [Some(123_u32)].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).try_incorrect_order_update_corrected_order_middle_page_number_sum(),
                incorrect_order_update_corrected_order_middle_page_number_sum
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
