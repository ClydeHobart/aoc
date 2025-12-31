use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec3,
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        multi::separated_list1, sequence::tuple, Err, IResult,
    },
    std::{cmp::Reverse, ops::BitAndAssign},
};

/* --- Day 23: Experimental Emergency Teleportation ---

Using your torch to search the darkness of the rocky cavern, you finally locate the man's friend: a small reindeer.

You're not sure how it got so far in this cave. It looks sick - too sick to walk - and too heavy for you to carry all the way back. Sleighs won't be invented for another 1500 years, of course.

The only option is experimental emergency teleportation.

You hit the "experimental emergency teleportation" button on the device and push I accept the risk on no fewer than 18 different warning messages. Immediately, the device deploys hundreds of tiny nanobots which fly around the cavern, apparently assembling themselves into a very specific formation. The device lists the X,Y,Z position (pos) for each nanobot as well as its signal radius (r) on its tiny screen (your puzzle input).

Each nanobot can transmit signals to any integer coordinate which is a distance away from it less than or equal to its signal radius (as measured by Manhattan distance). Coordinates a distance away of less than or equal to a nanobot's signal radius are said to be in range of that nanobot.

Before you start the teleportation process, you should determine which nanobot is the strongest (that is, which has the largest signal radius) and then, for that nanobot, the total number of nanobots that are in range of it, including itself.

For example, given the following nanobots:

pos=<0,0,0>, r=4
pos=<1,0,0>, r=1
pos=<4,0,0>, r=3
pos=<0,2,0>, r=1
pos=<0,5,0>, r=3
pos=<0,0,3>, r=1
pos=<1,1,1>, r=1
pos=<1,1,2>, r=1
pos=<1,3,1>, r=1

The strongest nanobot is the first one (position 0,0,0) because its signal radius, 4 is the largest. Using that nanobot's location and signal radius, the following nanobots are in or out of range:

    The nanobot at 0,0,0 is distance 0 away, and so it is in range.
    The nanobot at 1,0,0 is distance 1 away, and so it is in range.
    The nanobot at 4,0,0 is distance 4 away, and so it is in range.
    The nanobot at 0,2,0 is distance 2 away, and so it is in range.
    The nanobot at 0,5,0 is distance 5 away, and so it is not in range.
    The nanobot at 0,0,3 is distance 3 away, and so it is in range.
    The nanobot at 1,1,1 is distance 3 away, and so it is in range.
    The nanobot at 1,1,2 is distance 4 away, and so it is in range.
    The nanobot at 1,3,1 is distance 5 away, and so it is not in range.

In this example, in total, 7 nanobots are in range of the nanobot with the largest signal radius.

Find the nanobot with the largest signal radius. How many nanobots are in range of its signals?

--- Part Two ---

Now, you just need to figure out where to position yourself so that you're actually teleported when the nanobots activate.

To increase the probability of success, you need to find the coordinate which puts you in range of the largest number of nanobots. If there are multiple, choose one closest to your position (0,0,0, measured by manhattan distance).

For example, given the following nanobot formation:

pos=<10,12,12>, r=2
pos=<12,14,12>, r=2
pos=<16,12,12>, r=4
pos=<14,14,14>, r=6
pos=<50,50,50>, r=200
pos=<10,10,10>, r=5

Many coordinates are in range of some of the nanobots in this formation. However, only the coordinate 12,12,12 is in range of the most nanobots: it is in range of the first five, but is not in range of the nanobot at 10,10,10. (All other coordinates are in range of fewer than five nanobots.) This coordinate's distance from 0,0,0 is 36.

Find the coordinates that are in range of the largest number of nanobots. What is the shortest manhattan distance between any of those points and 0,0,0? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct NanoBot {
    pos: IVec3,
    r: i32,
}

impl NanoBot {
    fn is_in_range(&self, pos: &IVec3) -> bool {
        manhattan_distance_3d(&self.pos, pos) <= self.r
    }

    fn iter_in_range_nano_bots<'a>(
        &'a self,
        nano_bots: &'a [Self],
    ) -> impl Iterator<Item = &'a Self> {
        nano_bots
            .iter()
            .filter(|nano_bot| self.is_in_range(&nano_bot.pos))
    }
}

impl Parse for NanoBot {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("pos=<"),
                map(parse_separated_array(parse_integer, tag(",")), IVec3::from),
                tag(">, r="),
                parse_integer,
            )),
            |(_, pos, _, r)| NanoBot { pos, r },
        )(input)
    }
}

const MAX_NANO_BOT_COUNT: usize = 1_usize << 10_usize;

type NanoBotBitArray = BitArr!(for MAX_NANO_BOT_COUNT);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct OctahedronIntersection([i32; Self::LEN]);

impl OctahedronIntersection {
    const DIMENSION: usize = 3_usize;
    const LEN: usize = 1_usize << Self::DIMENSION;
    const HALF_LEN: usize = Self::LEN / 2_usize;

    /// ```notrust
    /// [
    ///     ( 1, 1, 1),
    ///     (-1, 1, 1),
    ///     ( 1,-1, 1),
    ///     (-1,-1, 1),
    ///     ( 1, 1,-1),
    ///     (-1, 1,-1),
    ///     ( 1,-1,-1),
    ///     (-1,-1,-1),
    /// ]
    /// ```
    const DOT_PRODUCT_FACTORS: [IVec3; Self::LEN] = Self::dot_product_factors();

    const fn all() -> Self {
        Self([i32::MAX; Self::LEN])
    }

    const fn dot_product_factors() -> [IVec3; Self::LEN] {
        let mut dot_product_factors: [IVec3; Self::LEN] = [IVec3::ONE; Self::LEN];
        let mut dot_product_factor_index: usize = 0_usize;

        while dot_product_factor_index < dot_product_factors.len() {
            let mut dot_product_factor_array: [i32; Self::DIMENSION] = [1_i32; Self::DIMENSION];
            let mut dot_product_factor_array_index: usize = 0_usize;

            while dot_product_factor_array_index < dot_product_factor_array.len() {
                if (dot_product_factor_index & (1_usize << dot_product_factor_array_index))
                    != 0_usize
                {
                    dot_product_factor_array[dot_product_factor_array_index] = -1_i32;
                }

                dot_product_factor_array_index += 1_usize;
            }

            dot_product_factors[dot_product_factor_index] =
                IVec3::from_array(dot_product_factor_array);
            dot_product_factor_index += 1_usize;
        }

        dot_product_factors
    }

    fn new(pos: IVec3, r: i32) -> Self {
        let mut octahedron_intersection: Self = Self([r; Self::LEN]);

        for (constraint, dot_product_factor) in octahedron_intersection
            .0
            .iter_mut()
            .zip(Self::DOT_PRODUCT_FACTORS)
        {
            *constraint += pos.dot(dot_product_factor);
        }

        octahedron_intersection
    }

    fn iter_constraint_diffs(&self, pos: IVec3) -> impl Iterator<Item = i32> + '_ {
        self.0
            .iter()
            .zip(Self::DOT_PRODUCT_FACTORS)
            .map(move |(&constraint, dot_product_factor)| pos.dot(dot_product_factor) - constraint)
    }

    /// Point `(x, y, z)` is in the intersection iff for all `i` in `[0, 8)`:
    /// ```notrust
    /// k_i_x * x + k_i_y * y + k_i_z * z <= constraint_i
    /// ```
    #[cfg(test)]
    fn contains(&self, pos: IVec3) -> bool {
        self.iter_constraint_diffs(pos)
            .all(|constraint_diff| constraint_diff <= 0)
    }

    /// Continuing the terminology from `contains`, `k_i` is `(k_i_x, k_i_y, k_i_z)`, and together
    /// `k_0` through `k_7` are the elements of `DOT_PRODUCT_FACTORS`. By the construction of
    /// `DOT_PRODUCT_FACTORS`, if `j = 7 - i`, `k_i = -k_j`. Revisiting the inequality, this gives:
    /// ```notrust
    ///                 k_j_x * x + k_j_y * y + k_j_z * z <= constraint_j
    ///                -k_j_x * x - k_j_y * y - k_j_z * z >= -constraint_j
    /// constraint_i >= k_i_x * x + k_i_y * y + k_i_z * z >= -constraint_j
    /// -constraint_j <= constraint_i
    /// ```
    fn is_empty(&self) -> bool {
        self.0[..Self::HALF_LEN]
            .iter()
            .zip(self.0[Self::HALF_LEN..].iter().rev())
            .any(|(&constraint_j, &constraint_i)| -constraint_j > constraint_i)
    }

    /// Returns the minimum Manhattan distance from the position to the intersection. If the
    /// intersection is empty, this return value has no meaning.
    fn manhattan_distance(&self, pos: IVec3) -> i32 {
        self.iter_constraint_diffs(pos).max().unwrap().max(0_i32)
    }

    /// Tries to return the minimum Manhattan distance from the position to the intersection, if the
    /// intersection is non-empty, or `None` otherwise.
    #[cfg(test)]
    fn try_manhattan_distance(&self, pos: IVec3) -> Option<i32> {
        (!self.is_empty()).then(|| self.manhattan_distance(pos))
    }
}

impl BitAndAssign<&NanoBot> for OctahedronIntersection {
    fn bitand_assign(&mut self, rhs: &NanoBot) {
        for (constraint, dot_product_factor) in self.0.iter_mut().zip(Self::DOT_PRODUCT_FACTORS) {
            *constraint = (*constraint).min(rhs.pos.dot(dot_product_factor) + rhs.r);
        }
    }
}

impl Default for OctahedronIntersection {
    fn default() -> Self {
        Self::all()
    }
}

impl From<&NanoBot> for OctahedronIntersection {
    fn from(value: &NanoBot) -> Self {
        Self::new(value.pos, value.r)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
struct NanoBotUserCliqueState {
    octahedron_intersection: OctahedronIntersection,
    clique_cardinality: u32,
    manhattan_distance: i32,
}

impl NanoBotUserCliqueState {
    fn cmp_key(&self) -> (u32, Reverse<i32>) {
        (self.clique_cardinality, Reverse(self.manhattan_distance))
    }
}

struct NanoBotData<'n> {
    nano_bots: &'n [NanoBot],
    neighbors: Vec<NanoBotBitArray>,
    maximum_clique_state: Option<CliqueState<NanoBotBitArray, NanoBotUserCliqueState>>,
}

impl<'n> NanoBotData<'n> {
    fn new(nano_bots: &'n [NanoBot]) -> Self {
        let mut neighbors: Vec<NanoBotBitArray> = vec![Default::default(); nano_bots.len()];

        for index_a in 0_usize..nano_bots.len().saturating_sub(1_usize) {
            let octahedron_intersection_a: OctahedronIntersection = (&nano_bots[index_a]).into();
            let index_b_start: usize = index_a + 1_usize;
            let (neighbors_a, neighbors_b): (&mut [NanoBotBitArray], &mut [NanoBotBitArray]) =
                neighbors.split_at_mut(index_b_start);

            for index_b in index_b_start..nano_bots.len() {
                let mut octahedron_intersection: OctahedronIntersection =
                    octahedron_intersection_a.clone();

                octahedron_intersection &= &nano_bots[index_b];

                if !octahedron_intersection.is_empty() {
                    neighbors_a[index_a].set(index_b, true);
                    neighbors_b[index_b - index_b_start].set(index_a, true);
                }
            }
        }

        Self {
            nano_bots,
            neighbors,
            maximum_clique_state: None,
        }
    }
}

impl<'n> UserCliqueIteratorTrait for NanoBotData<'n> {
    type BitArray = NanoBotBitArray;

    type UserCliqueState = NanoBotUserCliqueState;

    fn vertex_count(&self) -> usize {
        self.nano_bots.len()
    }

    fn integrate_vertex(
        &self,
        vertex_index: usize,
        _clique: &Self::BitArray,
        inout_user_clique_state: &mut Self::UserCliqueState,
    ) {
        inout_user_clique_state.octahedron_intersection &= &self.nano_bots[vertex_index];
        inout_user_clique_state.clique_cardinality += 1_u32;
        inout_user_clique_state.manhattan_distance = inout_user_clique_state
            .octahedron_intersection
            .manhattan_distance(IVec3::ZERO);
    }

    fn is_clique_state_valid(
        &self,
        _clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
    ) -> bool {
        !user_clique_state.octahedron_intersection.is_empty()
    }

    fn visit_clique(&mut self, clique_state: &CliqueState<Self::BitArray, Self::UserCliqueState>) {
        if self
            .maximum_clique_state
            .as_ref()
            .map(|maximum_clique_state| {
                maximum_clique_state.user_clique_state.cmp_key()
                    < clique_state.user_clique_state.cmp_key()
            })
            .unwrap_or(true)
        {
            self.maximum_clique_state = Some(clique_state.clone());
        }
    }

    fn get_neighbors(
        &self,
        vertex_index: usize,
        _clique: &Self::BitArray,
        _user_clique_state: &Self::UserCliqueState,
        out_neighbors: &mut Self::BitArray,
    ) {
        *out_neighbors = self.neighbors[vertex_index].clone();
    }

    fn should_visit_neighbors(
        &self,
        _clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
        neighbors: &Self::BitArray,
    ) -> bool {
        self.maximum_clique_state
            .as_ref()
            .map(|maximum_clique_state| {
                user_clique_state.clique_cardinality as usize + neighbors.count_ones()
                    >= maximum_clique_state.user_clique_state.clique_cardinality as usize
            })
            .unwrap_or(true)
    }
}

type NanoBotCliqueState = CliqueState<NanoBotBitArray, NanoBotUserCliqueState>;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<NanoBot>);

impl Solution {
    fn strongest_nano_bot(&self) -> &NanoBot {
        self.0.iter().max_by_key(|nano_bot| nano_bot.r).unwrap()
    }

    fn count_nano_bots_in_range_of_strongest_nano_bot(&self) -> usize {
        self.strongest_nano_bot()
            .iter_in_range_nano_bots(&self.0)
            .count()
    }

    fn nano_bot_data<'n>(&'n self) -> NanoBotData<'n> {
        NanoBotData::new(&self.0)
    }

    fn maximum_clique_state(&self) -> NanoBotCliqueState {
        let mut iterator = self.nano_bot_data().iter();

        (&mut iterator).for_each(|_| ());

        iterator.user_clique_iterator.maximum_clique_state.unwrap()
    }

    fn minimum_manhattan_distance_of_maximum_clique(&self) -> i32 {
        self.maximum_clique_state()
            .user_clique_state
            .manhattan_distance
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(separated_list1(line_ending, NanoBot::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Curious what part 2 will be, particularly as it pertains to data structures. I hope this
    /// doesn't require an oct-tree.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_nano_bots_in_range_of_strongest_nano_bot());
    }

    /// I like maximum clique!
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let maximum_clique_state = self.maximum_clique_state();

            dbg!(maximum_clique_state.clique.count_ones());
            dbg!(maximum_clique_state.user_clique_state.manhattan_distance);
        } else {
            dbg!(self.minimum_manhattan_distance_of_maximum_clique());
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
    use {
        super::*,
        std::{iter::repeat_with, sync::OnceLock},
    };

    const SOLUTION_STRS: &'static [&'static str] = &[
        "\
        pos=<0,0,0>, r=4\n\
        pos=<1,0,0>, r=1\n\
        pos=<4,0,0>, r=3\n\
        pos=<0,2,0>, r=1\n\
        pos=<0,5,0>, r=3\n\
        pos=<0,0,3>, r=1\n\
        pos=<1,1,1>, r=1\n\
        pos=<1,1,2>, r=1\n\
        pos=<1,3,1>, r=1\n",
        "\
        pos=<10,12,12>, r=2\n\
        pos=<12,14,12>, r=2\n\
        pos=<16,12,12>, r=4\n\
        pos=<14,14,14>, r=6\n\
        pos=<50,50,50>, r=200\n\
        pos=<10,10,10>, r=5\n",
    ];
    const POSES: &'static [IVec3] = &[
        IVec3::ZERO,
        IVec3::X,
        IVec3::Y,
        IVec3::Z,
        IVec3::NEG_X,
        IVec3::NEG_Y,
        IVec3::NEG_Z,
        IVec3::ONE,
        IVec3::NEG_ONE,
        IVec3::new(1_i32, 2_i32, 3_i32),
        IVec3::new(4_i32, 5_i32, 6_i32),
        IVec3::new(7_i32, 8_i32, 9_i32),
    ];
    const AXES: &'static [IVec3] = &[
        IVec3::X,
        IVec3::Y,
        IVec3::Z,
        IVec3::NEG_X,
        IVec3::NEG_Y,
        IVec3::NEG_Z,
    ];
    const RS: &'static [i32] = &[0, 1, 2, 3, 4];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    NanoBot {
                        pos: (0_i32, 0_i32, 0_i32).into(),
                        r: 4_i32,
                    },
                    NanoBot {
                        pos: (1_i32, 0_i32, 0_i32).into(),
                        r: 1_i32,
                    },
                    NanoBot {
                        pos: (4_i32, 0_i32, 0_i32).into(),
                        r: 3_i32,
                    },
                    NanoBot {
                        pos: (0_i32, 2_i32, 0_i32).into(),
                        r: 1_i32,
                    },
                    NanoBot {
                        pos: (0_i32, 5_i32, 0_i32).into(),
                        r: 3_i32,
                    },
                    NanoBot {
                        pos: (0_i32, 0_i32, 3_i32).into(),
                        r: 1_i32,
                    },
                    NanoBot {
                        pos: (1_i32, 1_i32, 1_i32).into(),
                        r: 1_i32,
                    },
                    NanoBot {
                        pos: (1_i32, 1_i32, 2_i32).into(),
                        r: 1_i32,
                    },
                    NanoBot {
                        pos: (1_i32, 3_i32, 1_i32).into(),
                        r: 1_i32,
                    },
                ]),
                Solution(vec![
                    NanoBot {
                        pos: (10_i32, 12_i32, 12_i32).into(),
                        r: 2_i32,
                    },
                    NanoBot {
                        pos: (12_i32, 14_i32, 12_i32).into(),
                        r: 2_i32,
                    },
                    NanoBot {
                        pos: (16_i32, 12_i32, 12_i32).into(),
                        r: 4_i32,
                    },
                    NanoBot {
                        pos: (14_i32, 14_i32, 14_i32).into(),
                        r: 6_i32,
                    },
                    NanoBot {
                        pos: (50_i32, 50_i32, 50_i32).into(),
                        r: 200_i32,
                    },
                    NanoBot {
                        pos: (10_i32, 10_i32, 10_i32).into(),
                        r: 5_i32,
                    },
                ]),
            ]
        })[index]
    }

    fn iter_poses() -> impl Iterator<Item = IVec3> {
        POSES.iter().copied()
    }

    fn iter_axes() -> impl Iterator<Item = IVec3> {
        AXES.iter().copied()
    }

    struct AxisPair {
        axis_a: IVec3,
        axis_b: IVec3,
    }

    fn iter_axis_pairs() -> impl Iterator<Item = AxisPair> {
        iter_axes()
            .zip(repeat_with(iter_axes).flatten().skip(1_usize))
            .map(|(axis_a, axis_b)| AxisPair { axis_a, axis_b })
    }

    struct AxisTrio {
        axis_a: IVec3,
        axis_b: IVec3,
        axis_c: IVec3,
    }

    fn iter_axis_trios() -> impl Iterator<Item = AxisTrio> {
        iter_axis_pairs().map(|AxisPair { axis_a, axis_b }| AxisTrio {
            axis_a,
            axis_b,
            axis_c: axis_a.cross(axis_b),
        })
    }

    fn iter_rs() -> impl Iterator<Item = i32> {
        RS.iter().copied()
    }

    struct OctahedronParams {
        pos: IVec3,
        r: i32,
    }

    fn iter_octahedron_params() -> impl Iterator<Item = OctahedronParams> {
        iter_poses().flat_map(|pos| iter_rs().map(move |r| OctahedronParams { pos, r }))
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
    fn test_strongest_nano_bot() {
        for (index, strongest_nano_bot) in [NanoBot {
            pos: IVec3::ZERO,
            r: 4_i32,
        }]
        .into_iter()
        .enumerate()
        {
            assert_eq!(*solution(index).strongest_nano_bot(), strongest_nano_bot);
        }
    }

    #[test]
    fn test_is_in_range() {
        for (index, is_in_range) in [vec![true, true, true, true, false, true, true, true, false]]
            .into_iter()
            .enumerate()
        {
            let solution: &Solution = solution(index);
            let strongest_nano_bot: &NanoBot = solution.strongest_nano_bot();

            assert_eq!(
                solution
                    .0
                    .iter()
                    .map(|nano_bot| strongest_nano_bot.is_in_range(&nano_bot.pos))
                    .collect::<Vec<bool>>(),
                is_in_range
            );
        }
    }

    #[test]
    fn test_count_nano_bots_in_range_of_strongest_nano_bot() {
        for (index, nano_bots_in_range_of_strongest_nano_bot_count) in
            [7_usize].into_iter().enumerate()
        {
            assert_eq!(
                solution(index).count_nano_bots_in_range_of_strongest_nano_bot(),
                nano_bots_in_range_of_strongest_nano_bot_count
            );
        }
    }

    #[test]
    fn test_octahedron_intersection_new() {
        for OctahedronParams { pos, r } in iter_octahedron_params() {
            assert_eq!(
                OctahedronIntersection::new(pos, r),
                OctahedronIntersection([
                    r + pos.x + pos.y + pos.z, // +X +Y +Z
                    r - pos.x + pos.y + pos.z, // -X +Y +Z
                    r + pos.x - pos.y + pos.z, // +X -Y +Z
                    r - pos.x - pos.y + pos.z, // -X -Y +Z
                    r + pos.x + pos.y - pos.z, // +X +Y -Z
                    r - pos.x + pos.y - pos.z, // -X +Y -Z
                    r + pos.x - pos.y - pos.z, // +X -Y -Z
                    r - pos.x - pos.y - pos.z, // -X -Y -Z
                ]),
                "pos == {pos:?}, r == {r}"
            );
        }
    }

    fn test_octahedron_intersection_contains_for_iter<
        I: IntoIterator<Item = (IVec3, Option<IVec3>)>,
    >(
        octahedron_intersection: &OctahedronIntersection,
        _axes: Option<&OblongParams>,
        iter: I,
    ) {
        for (pos, delta) in iter {
            assert!(octahedron_intersection.contains(pos));

            if let Some(delta) = delta {
                assert!(!octahedron_intersection.contains(pos + delta));
            }
        }
    }

    #[test]
    fn test_octahedron_intersection_contains() {
        for OctahedronParams { pos, r } in iter_octahedron_params() {
            let octahedron_intersection: OctahedronIntersection =
                OctahedronIntersection::new(pos, r);

            test_octahedron_intersection_contains_for_iter(
                &octahedron_intersection,
                None,
                [(pos, None)]
                    .into_iter()
                    .chain(iter_axes().map(|axis| (pos + r * axis, Some(axis)))),
            );
        }
    }

    struct OblongParams {
        pos: IVec3,
        axis_trio: AxisTrio,
    }

    fn iter_oblong_params() -> impl Iterator<Item = OblongParams> {
        iter_poses()
            .flat_map(|pos| iter_axis_trios().map(move |axis_trio| OblongParams { pos, axis_trio }))
    }

    fn test_oblong_octahedron_intersection<
        F1: Fn(&OctahedronIntersection, Option<&OblongParams>, Vec<(IVec3, Option<IVec3>)>),
        F2: Fn(&OctahedronIntersection, &OblongParams),
    >(
        f1: F1,
        f2: F2,
    ) {
        for oblong_params in iter_oblong_params() {
            let mut octahedron_intersection: OctahedronIntersection = OctahedronIntersection::all();

            octahedron_intersection &= &NanoBot {
                pos: oblong_params.pos
                    + oblong_params.axis_trio.axis_a
                    + oblong_params.axis_trio.axis_b,
                r: 4_i32,
            };
            octahedron_intersection &= &NanoBot {
                pos: oblong_params.pos
                    - oblong_params.axis_trio.axis_a
                    - oblong_params.axis_trio.axis_b,
                r: 4_i32,
            };

            f1(
                &octahedron_intersection,
                Some(&oblong_params),
                vec![
                    (
                        oblong_params.pos + 1_i32 * oblong_params.axis_trio.axis_a
                            - 1_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        None,
                    ),
                    (
                        oblong_params.pos - 1_i32 * oblong_params.axis_trio.axis_a
                            + 1_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        None,
                    ),
                    (
                        oblong_params.pos + 3_i32 * oblong_params.axis_trio.axis_a
                            - 1_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        Some(oblong_params.axis_trio.axis_a),
                    ),
                    (
                        oblong_params.pos + 1_i32 * oblong_params.axis_trio.axis_a
                            - 3_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        Some(-oblong_params.axis_trio.axis_b),
                    ),
                    (
                        oblong_params.pos - 1_i32 * oblong_params.axis_trio.axis_a
                            + 3_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        Some(oblong_params.axis_trio.axis_b),
                    ),
                    (
                        oblong_params.pos - 3_i32 * oblong_params.axis_trio.axis_a
                            + 1_i32 * oblong_params.axis_trio.axis_b
                            + 0_i32 * oblong_params.axis_trio.axis_c,
                        Some(-oblong_params.axis_trio.axis_a),
                    ),
                    (
                        oblong_params.pos + 1_i32 * oblong_params.axis_trio.axis_a
                            - 1_i32 * oblong_params.axis_trio.axis_b
                            + 2_i32 * oblong_params.axis_trio.axis_c,
                        Some(oblong_params.axis_trio.axis_c),
                    ),
                    (
                        oblong_params.pos + 1_i32 * oblong_params.axis_trio.axis_a
                            - 1_i32 * oblong_params.axis_trio.axis_b
                            - 2_i32 * oblong_params.axis_trio.axis_c,
                        Some(-oblong_params.axis_trio.axis_c),
                    ),
                    (
                        oblong_params.pos - 1_i32 * oblong_params.axis_trio.axis_a
                            + 1_i32 * oblong_params.axis_trio.axis_b
                            + 2_i32 * oblong_params.axis_trio.axis_c,
                        Some(oblong_params.axis_trio.axis_c),
                    ),
                    (
                        oblong_params.pos - 1_i32 * oblong_params.axis_trio.axis_a
                            + 1_i32 * oblong_params.axis_trio.axis_b
                            - 2_i32 * oblong_params.axis_trio.axis_c,
                        Some(-oblong_params.axis_trio.axis_c),
                    ),
                ],
            );

            octahedron_intersection &= &NanoBot {
                pos: oblong_params.pos + oblong_params.axis_trio.axis_a
                    - oblong_params.axis_trio.axis_b,
                r: 4_i32,
            };
            octahedron_intersection &= &NanoBot {
                pos: oblong_params.pos - oblong_params.axis_trio.axis_a
                    + oblong_params.axis_trio.axis_b,
                r: 4_i32,
            };

            f2(&octahedron_intersection, &oblong_params);
        }
    }

    #[test]
    fn test_octahedron_intersection_bitand_assign() {
        for OctahedronParams { pos, r } in iter_octahedron_params() {
            let mut octahedron_intersection: OctahedronIntersection = OctahedronIntersection::all();

            octahedron_intersection &= &NanoBot { pos, r };

            assert_eq!(octahedron_intersection, OctahedronIntersection::new(pos, r));
        }

        for OctahedronParams { pos, r } in iter_octahedron_params() {
            for axis in iter_axes() {
                let mut octahedron_intersection: OctahedronIntersection =
                    OctahedronIntersection::all();

                octahedron_intersection &= &NanoBot { pos: pos + axis, r };
                octahedron_intersection &= &NanoBot { pos: pos - axis, r };

                if r == 0_i32 {
                    assert!(octahedron_intersection.is_empty());
                } else {
                    assert_eq!(
                        octahedron_intersection,
                        OctahedronIntersection::new(pos, r - 1_i32)
                    );
                }
            }
        }

        test_oblong_octahedron_intersection(
            test_octahedron_intersection_contains_for_iter,
            |octahedron_intersection, oblong_params| {
                assert_eq!(
                    *octahedron_intersection,
                    OctahedronIntersection::new(oblong_params.pos, 2_i32)
                );
            },
        );
    }

    fn test_octahedron_intersection_manhattan_distance_for_iter<
        I: IntoIterator<Item = (IVec3, Option<i32>)>,
    >(
        octahedron_intersection: &OctahedronIntersection,
        iter: I,
    ) {
        for (pos, manhattan_distance) in iter {
            assert_eq_break(
                octahedron_intersection.try_manhattan_distance(pos),
                manhattan_distance,
            );
        }
    }

    #[test]
    fn test_octahedron_intersection_manhattan_distance() {
        for OctahedronParams { pos, r } in iter_octahedron_params() {
            let octahedron_intersection: OctahedronIntersection =
                OctahedronIntersection::new(pos, r);

            test_octahedron_intersection_manhattan_distance_for_iter(
                &octahedron_intersection,
                [(pos, Some(0_i32))]
                    .into_iter()
                    .chain(iter_axes().map(|axis| (pos + r * axis, Some(0_i32))))
                    .chain(
                        (r > 0_i32)
                            .then(|| {
                                iter_axis_pairs()
                                    .map(|AxisPair { axis_a, axis_b }| {
                                        (pos + r * (axis_a + axis_b), Some(r))
                                    })
                                    .chain(iter_axis_trios().map(
                                        |AxisTrio {
                                             axis_a,
                                             axis_b,
                                             axis_c,
                                         }| {
                                            (pos + r * (axis_a + axis_b + axis_c), Some(2_i32 * r))
                                        },
                                    ))
                            })
                            .into_iter()
                            .flatten(),
                    ),
            );
        }

        test_oblong_octahedron_intersection(
            |octahedron_intersection, _, poses_and_deltas| {
                test_octahedron_intersection_manhattan_distance_for_iter(
                    octahedron_intersection,
                    poses_and_deltas
                        .iter()
                        .copied()
                        .flat_map(|(pos, delta)| {
                            [(pos, Some(0_i32))]
                                .into_iter()
                                .chain(delta.into_iter().flat_map(move |delta| {
                                    [
                                        (pos + 1_i32 * delta, Some(1_i32)),
                                        (pos + 2_i32 * delta, Some(2_i32)),
                                    ]
                                }))
                        })
                        .chain(
                            [
                                (2_usize, 3_usize),
                                (2_usize, 4_usize),
                                (2_usize, 6_usize),
                                (2_usize, 7_usize),
                                (3_usize, 5_usize),
                                (3_usize, 6_usize),
                                (3_usize, 7_usize),
                                (4_usize, 5_usize),
                                (4_usize, 8_usize),
                                (4_usize, 9_usize),
                                (5_usize, 8_usize),
                                (5_usize, 9_usize),
                                (6_usize, 8_usize),
                                (7_usize, 9_usize),
                            ]
                            .into_iter()
                            .flat_map(|(start_index, end_index)| {
                                let start: IVec3 = poses_and_deltas[start_index].0;
                                let end: IVec3 = poses_and_deltas[end_index].0;
                                let delta: IVec3 = start - end;
                                let delta_signum: IVec3 = delta.signum();
                                let axis_a: IVec3 = delta_signum
                                    * if delta_signum.x != 0_i32 {
                                        IVec3::X
                                    } else {
                                        IVec3::Y
                                    };
                                let axis_b: IVec3 = delta_signum - axis_a;
                                let side_len: i32 = delta.abs().max_element();

                                (0_i32..=side_len).into_iter().flat_map(move |a| {
                                    (0_i32..=side_len).map(move |b| {
                                        let pos: IVec3 = end + a * axis_a + b * axis_b;

                                        (
                                            pos,
                                            Some(if octahedron_intersection.contains(pos) {
                                                0_i32
                                            } else {
                                                (a - b).abs()
                                            }),
                                        )
                                    })
                                })
                            }),
                        ),
                )
            },
            |_, _| {},
        );
    }

    #[test]
    fn test_maximum_clique_run() {
        assert_eq!(
            solution(1_usize).maximum_clique_state(),
            CliqueState {
                clique: bitarr_typed!(NanoBotBitArray; 1, 1, 1, 1, 1, 0),
                user_clique_state: NanoBotUserCliqueState {
                    octahedron_intersection: OctahedronIntersection::new(
                        (12_i32, 12_i32, 12_i32).into(),
                        0_i32
                    ),
                    clique_cardinality: 5_u32,
                    manhattan_distance: 36_i32
                }
            }
        );
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
