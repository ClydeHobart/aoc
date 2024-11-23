use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, map_res, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
};

/* --- Day 3: No Matter How You Slice It ---

The Elves managed to locate the chimney-squeeze prototype fabric for Santa's suit (thanks to someone who helpfully wrote its box IDs on the wall of the warehouse in the middle of the night). Unfortunately, anomalies are still affecting them - nobody can even agree on how to cut the fabric.

The whole piece of fabric they're working on is a very large square - at least 1000 inches on each side.

Each Elf has made a claim about which area of fabric would be ideal for Santa's suit. All claims have an ID and consist of a single rectangle with edges parallel to the edges of the fabric. Each claim's rectangle is defined as follows:

    The number of inches between the left edge of the fabric and the left edge of the rectangle.
    The number of inches between the top edge of the fabric and the top edge of the rectangle.
    The width of the rectangle in inches.
    The height of the rectangle in inches.

A claim like #123 @ 3,2: 5x4 means that claim ID 123 specifies a rectangle 3 inches from the left edge, 2 inches from the top edge, 5 inches wide, and 4 inches tall. Visually, it claims the square inches of fabric represented by # (and ignores the square inches of fabric represented by .) in the diagram below:

...........
...........
...#####...
...#####...
...#####...
...#####...
...........
...........
...........

The problem is that many of the claims overlap, causing two or more claims to cover part of the same areas. For example, consider the following claims:

#1 @ 1,3: 4x4
#2 @ 3,1: 4x4
#3 @ 5,5: 2x2

Visually, these claim the following areas:

........
...2222.
...2222.
.11XX22.
.11XX22.
.111133.
.111133.
........

The four square inches marked with X are claimed by both 1 and 2. (Claim 3, while adjacent to the others, does not overlap either of them.)

If the Elves all proceed with their own plans, none of them will have enough fabric. How many square inches of fabric are within two or more claims? */

type ClaimId = u16;
type ClaimIndexRaw = u16;
type ClaimIndex = TableIndex<ClaimIndexRaw>;
type ClaimTableElement = TableElement<ClaimId, Claim>;
type ClaimTable = Table<ClaimId, Claim, ClaimIndexRaw>;

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct Claim {
    pos: IVec2,
    dimensions: IVec2,
}

impl Claim {
    fn iter_range_2s(&self) -> impl Iterator<Item = Range2> {
        Range2::iter_from_start_and_end(self.pos..self.pos + self.dimensions)
    }

    fn parse_claim_table_element<'i>(input: &'i str) -> IResult<&'i str, ClaimTableElement> {
        map(
            tuple((
                tag("#"),
                parse_integer,
                tag(" @ "),
                parse_integer,
                tag(","),
                parse_integer,
                tag(": "),
                parse_integer,
                tag("x"),
                parse_integer,
            )),
            |(_, id, _, pos_x, _, pos_y, _, dimensions_x, _, dimensions_y)| ClaimTableElement {
                id,
                data: Self {
                    pos: IVec2::new(pos_x, pos_y),
                    dimensions: IVec2::new(dimensions_x, dimensions_y),
                },
            },
        )(input)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Fabric {
    #[default]
    ZeroClaims,
    OneClaim(ClaimIndex),
    MultipleClaims,
}

impl RegionTreeValue for Fabric {
    fn insert_value_into_leaf_with_matching_range(&mut self, other: &Self) {
        match (&self, other) {
            (Self::ZeroClaims, Self::OneClaim(claim_index)) => *self = Self::OneClaim(*claim_index),
            (Self::OneClaim(_), Self::OneClaim(_)) => *self = Self::MultipleClaims,
            (Self::MultipleClaims, Self::OneClaim(_)) => (),
            (_, _) => unimplemented!(),
        }
    }

    fn should_convert_leaf_to_parent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::MultipleClaims, _) => false,
            (_, Self::ZeroClaims) => false,
            _ => true,
        }
    }

    fn get_leaf<const D: usize, I: RangeIntTrait>(
        &self,
        _range: &RangeD<I, D>,
        _child_range: &RangeD<I, D>,
    ) -> Self {
        *self
    }

    fn try_convert_parent_to_leaf<'a, I>(mut iter: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a,
    {
        iter.try_fold(None, |expectation, fabric| {
            expectation
                .map(|expectation| expectation == *fabric)
                .unwrap_or(true)
                .then_some(Some(*fabric))
        })
        .flatten()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(ClaimTable);

impl Solution {
    fn fabric_quad_tree(&self) -> QuadTree<Fabric> {
        let mut fabric_quad_tree: QuadTree<Fabric> = QuadTree::new(
            (IVec2::ZERO..IVec2::ONE * (1_i32 << 10_u32))
                .try_into()
                .unwrap(),
            Fabric::ZeroClaims,
        );

        for (index, claim) in self.0.as_slice().iter().enumerate() {
            for range_2 in claim.data.iter_range_2s() {
                fabric_quad_tree.insert(&range_2, &Fabric::OneClaim(index.into()));
            }
        }

        fabric_quad_tree
    }

    fn overlap_fabric_area(&self) -> usize {
        let mut area: usize = 0_usize;

        self.fabric_quad_tree().visit_all_leaves(
            |_| true,
            |range, fabric| {
                if *fabric == Fabric::MultipleClaims {
                    area += range.len_product();
                }

                true
            },
        );

        area
    }

    fn does_claim_have_overlap(
        &self,
        fabric_quad_tree: &QuadTree<Fabric>,
        claim_index: ClaimIndex,
    ) -> bool {
        self.0.as_slice()[claim_index.get()]
            .data
            .iter_range_2s()
            .any(|range_2| {
                let mut does_claim_have_overlap: bool = false;

                fabric_quad_tree.visit_all_leaves(
                    |parent_range_2| range_2.try_intersect(parent_range_2).is_some(),
                    |child_range_2, fabric| {
                        does_claim_have_overlap = range_2.try_intersect(child_range_2).is_some()
                            && *fabric == Fabric::MultipleClaims;

                        !does_claim_have_overlap
                    },
                );

                does_claim_have_overlap
            })
    }

    fn find_no_overlap_claim_id(&self) -> Option<u16> {
        let fabric_quad_tree: QuadTree<Fabric> = self.fabric_quad_tree();
        let mut claims_with_overlaps: BitVec = bitvec![0; self.0.as_slice().len()];
        let mut no_overlap_claim_id: Option<u16> = None;

        fabric_quad_tree.visit_all_leaves(
            |_| true,
            |_range, fabric| {
                if let Fabric::OneClaim(claim_index) = fabric {
                    if !claims_with_overlaps[claim_index.get()] {
                        if self.does_claim_have_overlap(&fabric_quad_tree, *claim_index) {
                            claims_with_overlaps.set(claim_index.get(), true);
                        } else {
                            no_overlap_claim_id = Some(self.0.as_slice()[claim_index.get()].id);
                        }
                    }
                }

                no_overlap_claim_id.is_none()
            },
        );

        no_overlap_claim_id
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            map_res(
                many0(terminated(
                    Claim::parse_claim_table_element,
                    opt(line_ending),
                )),
                ClaimTable::try_from,
            ),
            Self,
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I got side tracked a while on this one, putting together the `RegionTree` utility
    /// (generalized to support higher dimensions!). The proximity of December snapped me into high
    /// gear on this one to wrap up my side quest and finish this.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.overlap_fabric_area());
    }

    /// Slight tweaks to the API for iterating over leafs to be able to prune out areas of the tree.
    /// Once I added a means to abort an iteration, it conveniently also removed the need to use
    /// `RefCell` (I added the `parent_predicate` first, which I was using to check if
    /// `no_overlap_claim_id` was `Some` to avoid further searching, but that required throwing it
    /// in separate `RefCell`s, one for each closure).
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.find_no_overlap_claim_id());
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

    const SOLUTION_STRS: &'static [&'static str] = &[
        "#123 @ 3,2: 5x4",
        "\
        #1 @ 1,3: 4x4\n\
        #2 @ 3,1: 4x4\n\
        #3 @ 5,5: 2x2\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            let vec = vec![
                Solution(
                    vec![ClaimTableElement {
                        id: 123_u16,
                        data: Claim {
                            pos: IVec2::new(3_i32, 2_i32),
                            dimensions: IVec2::new(5_i32, 4_i32),
                        },
                    }]
                    .try_into()
                    .unwrap(),
                ),
                Solution(
                    vec![
                        ClaimTableElement {
                            id: 1_u16,
                            data: Claim {
                                pos: IVec2::new(1_i32, 3_i32),
                                dimensions: IVec2::new(4_i32, 4_i32),
                            },
                        },
                        ClaimTableElement {
                            id: 2_u16,
                            data: Claim {
                                pos: IVec2::new(3_i32, 1_i32),
                                dimensions: IVec2::new(4_i32, 4_i32),
                            },
                        },
                        ClaimTableElement {
                            id: 3_u16,
                            data: Claim {
                                pos: IVec2::new(5_i32, 5_i32),
                                dimensions: IVec2::new(2_i32, 2_i32),
                            },
                        },
                    ]
                    .try_into()
                    .unwrap(),
                ),
            ];
            let vec = vec;
            vec
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
    fn test_overlap_fabric_area() {
        for (index, overlap_fabric_area) in [0_usize, 4_usize].into_iter().enumerate() {
            assert_eq!(solution(index).overlap_fabric_area(), overlap_fabric_area);
        }
    }

    #[test]
    fn test_find_no_overlap_claim_id() {
        for (index, no_overlap_claim_id) in [Some(123_u16), Some(3_u16)].into_iter().enumerate() {
            assert_eq!(
                solution(index).find_no_overlap_claim_id(),
                no_overlap_claim_id
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
