use {
    crate::*,
    nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::line_ending,
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{preceded, separated_pair, terminated, tuple},
        Err, IResult, Parser,
    },
    std::{
        alloc::{alloc, Layout},
        mem::MaybeUninit,
        ops::{DerefMut, Range},
    },
};

type Ranges = [Range<i32>; 3_usize];

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct RebootStep {
    ranges: Ranges,
    on: bool,
}

impl RebootStep {
    fn parse_component_range<'i>(
        component_tag: &'static str,
    ) -> impl Parser<&'i str, Range<i32>, Error<&'i str>> {
        preceded(
            tuple((tag(component_tag), tag("="))),
            map(
                separated_pair(parse_integer::<i32>, tag(".."), parse_integer::<i32>),
                |(start, end_inclusive)| start..(end_inclusive + 1_i32),
            ),
        )
    }
}

impl Parse for RebootStep {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                terminated(
                    alt((map(tag("on"), |_| true), map(tag("off"), |_| false))),
                    tag(" "),
                ),
                terminated(Self::parse_component_range("x"), tag(",")),
                terminated(Self::parse_component_range("y"), tag(",")),
                terminated(Self::parse_component_range("z"), opt(line_ending)),
            )),
            |(on, x_range, y_range, z_range)| Self {
                ranges: [x_range, y_range, z_range],
                on,
            },
        )(input)
    }
}

// #[cfg(not(test))]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Children(Box<[Node]>);

impl Children {
    fn as_slice(&self) -> &[Node] {
        self.0.deref()
    }

    fn as_mut_slice(&mut self) -> &mut [Node] {
        self.0.deref_mut()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
enum NodeData {
    Leaf { on: bool },
    Parent { children: Children },
}

impl NodeData {
    fn default_parent<const CHILDREN: usize>() -> Self {
        let children: Children = Children(Self::box_children::<CHILDREN>());

        Self::Parent { children }
    }

    fn box_children<const CHILDREN: usize>() -> Box<[Node]> {
        // SAFETY: Follow the expression within `from_raw` below.
        unsafe {
            Box::from_raw({
                let children_ptr: *mut [Node; CHILDREN] =
                    alloc(Layout::new::<[Node; CHILDREN]>()) as *mut [Node; CHILDREN];

                {
                    let maybe_uninit_children: &mut [MaybeUninit<Node>; CHILDREN] =
                    // SAFETY: In reality, elements of this array are currently
                    // uninitialized. This pointer cast just explicitly acknowledges that.
                    (children_ptr as *mut [MaybeUninit<Node>; CHILDREN])
                        .as_mut()
                        .unwrap();

                    for maybe_uninit_child in maybe_uninit_children.iter_mut() {
                        maybe_uninit_child.write(Default::default());
                    }
                }

                // SAFETY: At this point, all elements of the array have been initialzied.
                children_ptr
            })
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum RangeContainsRangeResult {
    Disjoint,
    RightContainsLeft,
    RightStartBisectsLeft,
    RightEndBisectsLeft,
    LeftContainsRight,
}

use std::ops::Deref;

use RangeContainsRangeResult as RCRR;

impl RangeContainsRangeResult {
    fn new(left: Range<i32>, right: Range<i32>) -> Self {
        if left.start >= right.end || right.start >= left.end {
            Self::Disjoint
        } else {
            match (
                left.contains(&right.start.saturating_sub(1_i32)),
                left.contains(&right.end),
            ) {
                (false, false) => Self::RightContainsLeft,
                (false, true) => Self::RightEndBisectsLeft,
                (true, false) => Self::RightStartBisectsLeft,
                (true, true) => Self::LeftContainsRight,
            }
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct Node {
    ranges: Ranges,
    node_data: NodeData,
}

impl Node {
    const GLOBAL_RANGE: Range<i32> = i32::MIN..i32::MAX;
    const GLOBAL_RANGES: Ranges = [Node::GLOBAL_RANGE, Node::GLOBAL_RANGE, Node::GLOBAL_RANGE];

    fn new_leaf(ranges: Ranges, on: bool) -> Self {
        Self {
            ranges,
            node_data: NodeData::Leaf { on },
        }
    }

    fn new_global_leaf(on: bool) -> Self {
        Self::new_leaf(Self::GLOBAL_RANGES, on)
    }

    fn new_parent<const CHILDREN: usize>(
        ranges: &Ranges,
        on: bool,
        reboot_step: &RebootStep,
        results: [RCRR; 3_usize],
    ) -> Self {
        let mut node_data: NodeData = NodeData::default_parent::<CHILDREN>();

        // Call an internal function to prevent this code from being duplicated
        fn new_parent_internal(
            ranges: &Ranges,
            on: bool,
            reboot_step: &RebootStep,
            results: [RCRR; 3_usize],
            node_data: &mut NodeData,
        ) {
            let children: &mut [Node] = match node_data {
                NodeData::Parent { children } => children.as_mut_slice(),
                _ => unreachable!(),
            };

            let mut child_index: usize = 0_usize;

            for octant_index in 0_usize..8_usize {
                let mut child_ranges: Ranges = Default::default();

                if (0_usize..3_usize)
                    .try_fold((), |_, component_index| {
                        let is_after_middle: bool =
                            (octant_index & (1_usize << component_index)) != 0_usize;
                        let result: RCRR = results[component_index];
                        let node_range: Range<i32> = ranges[component_index].clone();

                        if is_after_middle
                            && result == RCRR::RightContainsLeft
                            && node_range.len() == 1_usize
                        {
                            None
                        } else {
                            let reboot_step_range: Range<i32> =
                                reboot_step.ranges[component_index].clone();
                            let middle: i32 = match result {
                                RCRR::RightContainsLeft => node_range.end,
                                RCRR::RightStartBisectsLeft => reboot_step_range.start,
                                RCRR::RightEndBisectsLeft => reboot_step_range.end,
                                RCRR::LeftContainsRight => {
                                    let node_range_middle: i32 =
                                        ((node_range.start as i64 + node_range.end as i64) / 2_i64)
                                            as i32;

                                    if node_range_middle + reboot_step_range.start.saturating_neg()
                                        <= reboot_step_range.end - node_range_middle
                                    {
                                        reboot_step_range.start
                                    } else {
                                        reboot_step_range.end
                                    }
                                }
                                _ => unreachable!(),
                            };

                            child_ranges[component_index] = if is_after_middle {
                                middle..node_range.end
                            } else {
                                node_range.start..middle
                            };

                            Some(())
                        }
                    })
                    .is_some()
                {
                    children[child_index] = Node {
                        ranges: child_ranges,
                        node_data: NodeData::Leaf { on },
                    };
                    child_index += 1_usize;
                }
            }
        }

        new_parent_internal(ranges, on, reboot_step, results, &mut node_data);

        Self {
            ranges: ranges.clone(),
            node_data,
        }
    }

    fn is_on(&self) -> Option<bool> {
        match &self.node_data {
            NodeData::Leaf { on } => Some(*on),
            NodeData::Parent { children } => children
                .as_slice()
                .iter()
                .try_fold(None, |expected, child| {
                    let child_is_on: bool = child.is_on()?;

                    if let Some(expected) = expected {
                        if expected == child_is_on {
                            Some(Some(expected))
                        } else {
                            None
                        }
                    } else {
                        Some(Some(child_is_on))
                    }
                })
                .map(Option::unwrap),
        }
    }

    fn on_cubes(&self) -> usize {
        match &self.node_data {
            NodeData::Leaf { on } => {
                if *on {
                    self.cubes()
                } else {
                    0_usize
                }
            }
            NodeData::Parent { children } => children.as_slice().iter().map(Self::on_cubes).sum(),
        }
    }

    fn cubes(&self) -> usize {
        self.ranges.iter().map(Range::len).product()
    }

    fn execute(&mut self, reboot_step: &RebootStep) {
        match &mut self.node_data {
            NodeData::Leaf { on } => {
                if *on != reboot_step.on {
                    let mut results: [RCRR; 3_usize] = [RCRR::Disjoint; 3_usize];

                    for (result, (self_range, reboot_step_range)) in results.iter_mut().zip(
                        self.ranges
                            .iter()
                            .cloned()
                            .zip(reboot_step.ranges.iter().cloned()),
                    ) {
                        *result = RCRR::new(self_range, reboot_step_range);
                    }

                    if results
                        .iter()
                        .copied()
                        .all(|result| result != RCRR::Disjoint)
                    {
                        if results
                            .iter()
                            .copied()
                            .filter(|result| *result == RCRR::RightContainsLeft)
                            .count()
                            == 3_usize
                        {
                            *on = reboot_step.on;
                        } else {
                            match results
                                .iter()
                                .copied()
                                .enumerate()
                                .filter(|(index, result)| {
                                    *result == RCRR::RightContainsLeft
                                        && self.ranges[*index].len() == 1_usize
                                })
                                .count()
                            {
                                0_usize => {
                                    *self = Self::new_parent::<8_usize>(
                                        &self.ranges,
                                        *on,
                                        reboot_step,
                                        results,
                                    );
                                }
                                1_usize => {
                                    *self = Self::new_parent::<4_usize>(
                                        &self.ranges,
                                        *on,
                                        reboot_step,
                                        results,
                                    );
                                }
                                2_usize => {
                                    *self = Self::new_parent::<2_usize>(
                                        &self.ranges,
                                        *on,
                                        reboot_step,
                                        results,
                                    );
                                }
                                _ => unreachable!(),
                            }

                            // Now iterate over the children
                            self.execute(reboot_step);
                        }
                    }
                }
            }
            NodeData::Parent { children } => {
                for child in children.as_mut_slice().iter_mut() {
                    child.execute(reboot_step);
                }

                if let Some(on) = self.is_on() {
                    // Simplify self
                    *self = Self::new_leaf(self.ranges.clone(), on);
                }
            }
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            node_data: NodeData::Leaf { on: false },
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<RebootStep>);

macro_rules! reboot_steps {
    [ $( $on:ident => x=$x_range:expr,y=$y_range:expr,z=$z_range:expr),* ;$add_one:expr] => {
        [ $(
            RebootStep {
                ranges: [
                    $x_range.start..($x_range.end + $add_one as i32),
                    $y_range.start..($y_range.end + $add_one as i32),
                    $z_range.start..($z_range.end + $add_one as i32),
                ],
                on: Solution::$on(),
            },
        )* ]
    }
}

#[cfg(test)]
macro_rules! solution{
    [ $( $on:ident => x=$x_range:expr,y=$y_range:expr,z=$z_range:expr, )* ] => {
        Solution(reboot_steps![ $( $on => x=$x_range,y=$y_range,z=$z_range ),* ; true].into())
    }
}

impl Solution {
    const INIT_PROC_RANGE: Range<i32> = -50_i32..51_i32;
    const REBOOT_STEPS_TO_ISOLATE_INIT_PROC_REGION: &[RebootStep] = &reboot_steps![
        off =>
            x=Node::GLOBAL_RANGE.start..Solution::INIT_PROC_RANGE.start,
            y=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            z=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
        off =>
            x=Solution::INIT_PROC_RANGE.end..Node::GLOBAL_RANGE.end,
            y=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            z=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
        off =>
            x=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            y=Node::GLOBAL_RANGE.start..Solution::INIT_PROC_RANGE.start,
            z=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
        off =>
            x=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            y=Solution::INIT_PROC_RANGE.end..Node::GLOBAL_RANGE.end,
            z=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
        off =>
            x=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            y=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            z=Node::GLOBAL_RANGE.start..Solution::INIT_PROC_RANGE.start,
        off =>
            x=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            y=Node::GLOBAL_RANGE.start..Node::GLOBAL_RANGE.end,
            z=Solution::INIT_PROC_RANGE.end..Node::GLOBAL_RANGE.end;
        false
    ];

    #[cfg(test)]
    #[inline(always)]
    const fn on() -> bool {
        true
    }

    #[inline(always)]
    const fn off() -> bool {
        false
    }

    fn count_on_cubes_in_init_proc_region(&self) -> usize {
        let mut node: Node = self.node();

        for reboot_step in Self::REBOOT_STEPS_TO_ISOLATE_INIT_PROC_REGION.iter() {
            node.execute(reboot_step);
        }

        node.on_cubes()
    }

    fn count_on_cubes(&self) -> usize {
        self.node().on_cubes()
    }

    fn node(&self) -> Node {
        let mut node: Node = Node::new_global_leaf(false);

        for reboot_step in self.0.iter() {
            node.execute(reboot_step);
        }

        node
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(RebootStep::parse), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_on_cubes_in_init_proc_region());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_on_cubes());
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

    const SOLUTION_STRS: &[&str] = &[
        "\
        on x=10..12,y=10..12,z=10..12\n\
        on x=11..13,y=11..13,z=11..13\n\
        off x=9..11,y=9..11,z=9..11\n\
        on x=10..10,y=10..10,z=10..10\n",
        "\
        on x=-20..26,y=-36..17,z=-47..7\n\
        on x=-20..33,y=-21..23,z=-26..28\n\
        on x=-22..28,y=-29..23,z=-38..16\n\
        on x=-46..7,y=-6..46,z=-50..-1\n\
        on x=-49..1,y=-3..46,z=-24..28\n\
        on x=2..47,y=-22..22,z=-23..27\n\
        on x=-27..23,y=-28..26,z=-21..29\n\
        on x=-39..5,y=-6..47,z=-3..44\n\
        on x=-30..21,y=-8..43,z=-13..34\n\
        on x=-22..26,y=-27..20,z=-29..19\n\
        off x=-48..-32,y=26..41,z=-47..-37\n\
        on x=-12..35,y=6..50,z=-50..-2\n\
        off x=-48..-32,y=-32..-16,z=-15..-5\n\
        on x=-18..26,y=-33..15,z=-7..46\n\
        off x=-40..-22,y=-38..-28,z=23..41\n\
        on x=-16..35,y=-41..10,z=-47..6\n\
        off x=-32..-23,y=11..30,z=-14..3\n\
        on x=-49..-5,y=-3..45,z=-29..18\n\
        off x=18..30,y=-20..-8,z=-3..13\n\
        on x=-41..9,y=-7..43,z=-33..15\n\
        on x=-54112..-39298,y=-85059..-49293,z=-27449..7877\n\
        on x=967..23432,y=45373..81175,z=27513..53682\n",
        "\
        on x=-5..47,y=-31..22,z=-19..33\n\
        on x=-44..5,y=-27..21,z=-14..35\n\
        on x=-49..-1,y=-11..42,z=-10..38\n\
        on x=-20..34,y=-40..6,z=-44..1\n\
        off x=26..39,y=40..50,z=-2..11\n\
        on x=-41..5,y=-41..6,z=-36..8\n\
        off x=-43..-33,y=-45..-28,z=7..25\n\
        on x=-33..15,y=-32..19,z=-34..11\n\
        off x=35..47,y=-46..-34,z=-11..5\n\
        on x=-14..36,y=-6..44,z=-16..29\n\
        on x=-57795..-6158,y=29564..72030,z=20435..90618\n\
        on x=36731..105352,y=-21140..28532,z=16094..90401\n\
        on x=30999..107136,y=-53464..15513,z=8553..71215\n\
        on x=13528..83982,y=-99403..-27377,z=-24141..23996\n\
        on x=-72682..-12347,y=18159..111354,z=7391..80950\n\
        on x=-1060..80757,y=-65301..-20884,z=-103788..-16709\n\
        on x=-83015..-9461,y=-72160..-8347,z=-81239..-26856\n\
        on x=-52752..22273,y=-49450..9096,z=54442..119054\n\
        on x=-29982..40483,y=-108474..-28371,z=-24328..38471\n\
        on x=-4958..62750,y=40422..118853,z=-7672..65583\n\
        on x=55694..108686,y=-43367..46958,z=-26781..48729\n\
        on x=-98497..-18186,y=-63569..3412,z=1232..88485\n\
        on x=-726..56291,y=-62629..13224,z=18033..85226\n\
        on x=-110886..-34664,y=-81338..-8658,z=8914..63723\n\
        on x=-55829..24974,y=-16897..54165,z=-121762..-28058\n\
        on x=-65152..-11147,y=22489..91432,z=-58782..1780\n\
        on x=-120100..-32970,y=-46592..27473,z=-11695..61039\n\
        on x=-18631..37533,y=-124565..-50804,z=-35667..28308\n\
        on x=-57817..18248,y=49321..117703,z=5745..55881\n\
        on x=14781..98692,y=-1341..70827,z=15753..70151\n\
        on x=-34419..55919,y=-19626..40991,z=39015..114138\n\
        on x=-60785..11593,y=-56135..2999,z=-95368..-26915\n\
        on x=-32178..58085,y=17647..101866,z=-91405..-8878\n\
        on x=-53655..12091,y=50097..105568,z=-75335..-4862\n\
        on x=-111166..-40997,y=-71714..2688,z=5609..50954\n\
        on x=-16602..70118,y=-98693..-44401,z=5197..76897\n\
        on x=16383..101554,y=4615..83635,z=-44907..18747\n\
        off x=-95822..-15171,y=-19987..48940,z=10804..104439\n\
        on x=-89813..-14614,y=16069..88491,z=-3297..45228\n\
        on x=41075..99376,y=-20427..49978,z=-52012..13762\n\
        on x=-21330..50085,y=-17944..62733,z=-112280..-30197\n\
        on x=-16478..35915,y=36008..118594,z=-7885..47086\n\
        off x=-98156..-27851,y=-49952..43171,z=-99005..-8456\n\
        off x=2032..69770,y=-71013..4824,z=7471..94418\n\
        on x=43670..120875,y=-42068..12382,z=-24787..38892\n\
        off x=37514..111226,y=-45862..25743,z=-16714..54663\n\
        off x=25699..97951,y=-30668..59918,z=-15349..69697\n\
        off x=-44271..17935,y=-9516..60759,z=49131..112598\n\
        on x=-61695..-5813,y=40978..94975,z=8655..80240\n\
        off x=-101086..-9439,y=-7088..67543,z=33935..83858\n\
        off x=18020..114017,y=-48931..32606,z=21474..89843\n\
        off x=-77139..10506,y=-89994..-18797,z=-80..59318\n\
        off x=8476..79288,y=-75520..11602,z=-96624..-24783\n\
        on x=-47488..-1262,y=24338..100707,z=16292..72967\n\
        off x=-84341..13987,y=2429..92914,z=-90671..-1318\n\
        off x=-37810..49457,y=-71013..-7894,z=-105357..-13188\n\
        off x=-27365..46395,y=31009..98017,z=15428..76570\n\
        off x=-70369..-16548,y=22648..78696,z=-1892..86821\n\
        on x=-53470..21291,y=-120233..-33476,z=-44150..38147\n\
        off x=-93533..-4276,y=-16170..68771,z=-104985..-24507\n",
    ];

    const SOLUTION_FNS: &[fn() -> Solution] = &[
        || {
            solution![
                on => x=10..12,y=10..12,z=10..12,
                on => x=11..13,y=11..13,z=11..13,
                off => x=9..11,y=9..11,z=9..11,
                on => x=10..10,y=10..10,z=10..10,
            ]
        },
        || {
            solution![
                on => x=-20..26,y=-36..17,z=-47..7,
                on => x=-20..33,y=-21..23,z=-26..28,
                on => x=-22..28,y=-29..23,z=-38..16,
                on => x=-46..7,y=-6..46,z=-50..-1,
                on => x=-49..1,y=-3..46,z=-24..28,
                on => x=2..47,y=-22..22,z=-23..27,
                on => x=-27..23,y=-28..26,z=-21..29,
                on => x=-39..5,y=-6..47,z=-3..44,
                on => x=-30..21,y=-8..43,z=-13..34,
                on => x=-22..26,y=-27..20,z=-29..19,
                off => x=-48..-32,y=26..41,z=-47..-37,
                on => x=-12..35,y=6..50,z=-50..-2,
                off => x=-48..-32,y=-32..-16,z=-15..-5,
                on => x=-18..26,y=-33..15,z=-7..46,
                off => x=-40..-22,y=-38..-28,z=23..41,
                on => x=-16..35,y=-41..10,z=-47..6,
                off => x=-32..-23,y=11..30,z=-14..3,
                on => x=-49..-5,y=-3..45,z=-29..18,
                off => x=18..30,y=-20..-8,z=-3..13,
                on => x=-41..9,y=-7..43,z=-33..15,
                on => x=-54112..-39298,y=-85059..-49293,z=-27449..7877,
                on => x=967..23432,y=45373..81175,z=27513..53682,
            ]
        },
        || {
            solution![
                on => x=-5..47,y=-31..22,z=-19..33,
                on => x=-44..5,y=-27..21,z=-14..35,
                on => x=-49..-1,y=-11..42,z=-10..38,
                on => x=-20..34,y=-40..6,z=-44..1,
                off => x=26..39,y=40..50,z=-2..11,
                on => x=-41..5,y=-41..6,z=-36..8,
                off => x=-43..-33,y=-45..-28,z=7..25,
                on => x=-33..15,y=-32..19,z=-34..11,
                off => x=35..47,y=-46..-34,z=-11..5,
                on => x=-14..36,y=-6..44,z=-16..29,
                on => x=-57795..-6158,y=29564..72030,z=20435..90618,
                on => x=36731..105352,y=-21140..28532,z=16094..90401,
                on => x=30999..107136,y=-53464..15513,z=8553..71215,
                on => x=13528..83982,y=-99403..-27377,z=-24141..23996,
                on => x=-72682..-12347,y=18159..111354,z=7391..80950,
                on => x=-1060..80757,y=-65301..-20884,z=-103788..-16709,
                on => x=-83015..-9461,y=-72160..-8347,z=-81239..-26856,
                on => x=-52752..22273,y=-49450..9096,z=54442..119054,
                on => x=-29982..40483,y=-108474..-28371,z=-24328..38471,
                on => x=-4958..62750,y=40422..118853,z=-7672..65583,
                on => x=55694..108686,y=-43367..46958,z=-26781..48729,
                on => x=-98497..-18186,y=-63569..3412,z=1232..88485,
                on => x=-726..56291,y=-62629..13224,z=18033..85226,
                on => x=-110886..-34664,y=-81338..-8658,z=8914..63723,
                on => x=-55829..24974,y=-16897..54165,z=-121762..-28058,
                on => x=-65152..-11147,y=22489..91432,z=-58782..1780,
                on => x=-120100..-32970,y=-46592..27473,z=-11695..61039,
                on => x=-18631..37533,y=-124565..-50804,z=-35667..28308,
                on => x=-57817..18248,y=49321..117703,z=5745..55881,
                on => x=14781..98692,y=-1341..70827,z=15753..70151,
                on => x=-34419..55919,y=-19626..40991,z=39015..114138,
                on => x=-60785..11593,y=-56135..2999,z=-95368..-26915,
                on => x=-32178..58085,y=17647..101866,z=-91405..-8878,
                on => x=-53655..12091,y=50097..105568,z=-75335..-4862,
                on => x=-111166..-40997,y=-71714..2688,z=5609..50954,
                on => x=-16602..70118,y=-98693..-44401,z=5197..76897,
                on => x=16383..101554,y=4615..83635,z=-44907..18747,
                off => x=-95822..-15171,y=-19987..48940,z=10804..104439,
                on => x=-89813..-14614,y=16069..88491,z=-3297..45228,
                on => x=41075..99376,y=-20427..49978,z=-52012..13762,
                on => x=-21330..50085,y=-17944..62733,z=-112280..-30197,
                on => x=-16478..35915,y=36008..118594,z=-7885..47086,
                off => x=-98156..-27851,y=-49952..43171,z=-99005..-8456,
                off => x=2032..69770,y=-71013..4824,z=7471..94418,
                on => x=43670..120875,y=-42068..12382,z=-24787..38892,
                off => x=37514..111226,y=-45862..25743,z=-16714..54663,
                off => x=25699..97951,y=-30668..59918,z=-15349..69697,
                off => x=-44271..17935,y=-9516..60759,z=49131..112598,
                on => x=-61695..-5813,y=40978..94975,z=8655..80240,
                off => x=-101086..-9439,y=-7088..67543,z=33935..83858,
                off => x=18020..114017,y=-48931..32606,z=21474..89843,
                off => x=-77139..10506,y=-89994..-18797,z=-80..59318,
                off => x=8476..79288,y=-75520..11602,z=-96624..-24783,
                on => x=-47488..-1262,y=24338..100707,z=16292..72967,
                off => x=-84341..13987,y=2429..92914,z=-90671..-1318,
                off => x=-37810..49457,y=-71013..-7894,z=-105357..-13188,
                off => x=-27365..46395,y=31009..98017,z=15428..76570,
                off => x=-70369..-16548,y=22648..78696,z=-1892..86821,
                on => x=-53470..21291,y=-120233..-33476,z=-44150..38147,
                off => x=-93533..-4276,y=-16170..68771,z=-104985..-24507,
            ]
        },
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            SOLUTION_FNS
                .iter()
                .map(|solution_fn| solution_fn())
                .collect()
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
    fn test_node_execute_and_on_cubes() {
        let mut node: Node = Node::new_global_leaf(false);

        node.execute(&solution(0_usize).0[0_usize]);

        assert_eq!(node.on_cubes(), 27_usize);
    }

    #[test]
    fn test_solution_count_on_cubes_in_init_proc_region() {
        for (index, cubes) in [39_usize, 590_784_usize, 474_140_usize]
            .into_iter()
            .enumerate()
        {
            assert_eq!(solution(index).count_on_cubes_in_init_proc_region(), cubes);
        }
    }

    #[test]
    fn test_solution_count_on_cubes() {
        assert_eq!(
            solution(2_usize).count_on_cubes(),
            2_758_514_936_282_235_usize
        );
    }
}
