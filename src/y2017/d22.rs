use {
    crate::*,
    glam::IVec2,
    nom::{
        combinator::{map, map_opt},
        error::Error,
        Err, IResult,
    },
    std::collections::HashSet,
};

/* --- Day 22: Sporifica Virus ---

Diagnostics indicate that the local grid computing cluster has been contaminated with the Sporifica Virus. The grid computing cluster is a seemingly-infinite two-dimensional grid of compute nodes. Each node is either clean or infected by the virus.

To prevent overloading the nodes (which would render them useless to the virus) or detection by system administrators, exactly one virus carrier moves through the network, infecting or cleaning nodes as it moves. The virus carrier is always located on a single node in the network (the current node) and keeps track of the direction it is facing.

To avoid detection, the virus carrier works in bursts; in each burst, it wakes up, does some work, and goes back to sleep. The following steps are all executed in order one time each burst:

    If the current node is infected, it turns to its right. Otherwise, it turns to its left. (Turning is done in-place; the current node does not change.)
    If the current node is clean, it becomes infected. Otherwise, it becomes cleaned. (This is done after the node is considered for the purposes of changing direction.)
    The virus carrier moves forward one node in the direction it is facing.

Diagnostics have also provided a map of the node infection status (your puzzle input). Clean nodes are shown as .; infected nodes are shown as #. This map only shows the center of the grid; there are many more nodes beyond those shown, but none of them are currently infected.

The virus carrier begins in the middle of the map facing up.

For example, suppose you are given a map like this:

..#
#..
...

Then, the middle of the infinite grid looks like this, with the virus carrier's position marked with [ ]:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . # . . .
. . . #[.]. . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

The virus carrier is on a clean node, so it turns left, infects the node, and moves left:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . # . . .
. . .[#]# . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

The virus carrier is on an infected node, so it turns right, cleans the node, and moves up:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . .[.]. # . . .
. . . . # . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

Four times in a row, the virus carrier finds a clean, infects it, turns left, and moves forward, ending in the same place and still facing up:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . #[#]. # . . .
. . # # # . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

Now on the same node as before, it sees an infection, which causes it to turn right, clean the node, and move forward:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . # .[.]# . . .
. . # # # . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

After the above actions, a total of 7 bursts of activity had taken place. Of them, 5 bursts of activity caused an infection.

After a total of 70, the grid looks like this, with the virus carrier facing up:

. . . . . # # . .
. . . . # . . # .
. . . # . . . . #
. . # . #[.]. . #
. . # . # . . # .
. . . . . # # . .
. . . . . . . . .
. . . . . . . . .

By this time, 41 bursts of activity caused an infection (though most of those nodes have since been cleaned).

After a total of 10000 bursts of activity, 5587 bursts will have caused an infection.

Given your actual map, after 10000 bursts of activity, how many bursts cause a node to become infected? (Do not count nodes that begin infected.)

--- Part Two ---

As you go to remove the virus from the infected nodes, it evolves to resist your attempt.

Now, before it infects a clean node, it will weaken it to disable your defenses. If it encounters an infected node, it will instead flag the node to be cleaned in the future. So:

    Clean nodes become weakened.
    Weakened nodes become infected.
    Infected nodes become flagged.
    Flagged nodes become clean.

Every node is always in exactly one of the above states.

The virus carrier still functions in a similar way, but now uses the following logic during its bursts of action:

    Decide which way to turn based on the current node:
        If it is clean, it turns left.
        If it is weakened, it does not turn, and will continue moving in the same direction.
        If it is infected, it turns right.
        If it is flagged, it reverses direction, and will go back the way it came.
    Modify the state of the current node, as described above.
    The virus carrier moves forward one node in the direction it is facing.

Start with the same map (still using . for clean and # for infected) and still with the virus carrier starting in the middle and facing up.

Using the same initial state as the previous example, and drawing weakened as W and flagged as F, the middle of the infinite grid looks like this, with the virus carrier's position again marked with [ ]:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . # . . .
. . . #[.]. . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

This is the same as before, since no initial nodes are weakened or flagged. The virus carrier is on a clean node, so it still turns left, instead weakens the node, and moves left:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . # . . .
. . .[#]W . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

The virus carrier is on an infected node, so it still turns right, instead flags the node, and moves up:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . .[.]. # . . .
. . . F W . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

This process repeats three more times, ending on the previously-flagged node and facing right:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . W W . # . . .
. . W[F]W . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

Finding a flagged node, it reverses direction and cleans the node:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . W W . # . . .
. .[W]. W . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

The weakened node becomes infected, and it continues in the same direction:

. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . W W . # . . .
.[.]# . W . . . .
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .

Of the first 100 bursts, 26 will result in infection. Unfortunately, another feature of this evolved virus is speed; of the first 10000000 bursts, 2511944 will result in infection.

Given your actual map, after 10000000 bursts of activity, how many bursts cause a node to become infected? (Do not count nodes that begin infected.) */

trait ComputeNodeTrait {
    const CLEAN: Self;
    const INFECTED: Self;

    fn is_infected(self) -> bool;
}

define_cell! {
    #[repr(u8)]
    #[cfg_attr(test, derive(Debug))]
    #[derive(Clone, Copy, PartialEq)]
    enum ComputeNode {
        Clean = CLEAN = b'.',
        Weakened = WEAKENED = b'W',
        Infected = INFECTED = b'#',
        Flagged = FLAGGED = b'F',
    }
}

impl ComputeNode {
    fn is_infected(self) -> bool {
        self == Self::Infected
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct ComputingCluster {
    grid: Grid2D<ComputeNode>,
    pos: IVec2,
    local_to_world: IVec2,
    dir: Direction,
}

impl ComputingCluster {
    fn simple_burst_rules(dir: Direction, compute_node: ComputeNode) -> (Direction, ComputeNode) {
        match compute_node {
            ComputeNode::Clean => (dir.prev(), ComputeNode::Infected),
            ComputeNode::Weakened => unimplemented!(),
            ComputeNode::Infected => (dir.next(), ComputeNode::Clean),
            ComputeNode::Flagged => unimplemented!(),
        }
    }

    fn complex_burst_rules(dir: Direction, compute_node: ComputeNode) -> (Direction, ComputeNode) {
        match compute_node {
            ComputeNode::Clean => (dir.prev(), ComputeNode::Weakened),
            ComputeNode::Weakened => (dir, ComputeNode::Infected),
            ComputeNode::Infected => (dir.next(), ComputeNode::Flagged),
            ComputeNode::Flagged => (dir.rev(), ComputeNode::Clean),
        }
    }

    fn burst<B: Fn(Direction, ComputeNode) -> (Direction, ComputeNode)>(
        &mut self,
        burst_rules: B,
    ) -> Option<IVec2> {
        let compute_node: &mut ComputeNode = self.grid.get_mut(self.pos).unwrap();

        let (next_dir, next_compute_node): (Direction, ComputeNode) =
            burst_rules(self.dir, *compute_node);

        self.dir = next_dir;
        *compute_node = next_compute_node;

        self.pos += self.dir.vec();

        if !self.grid.contains(self.pos) {
            self.grid.double_dimensions(ComputeNode::Clean);

            match self.dir {
                Direction::North => {
                    // The grid currently looks like this:
                    // #.
                    // ..
                    // And it needs to look like this:
                    // ..
                    // #.
                    // or this:
                    // ..
                    // .#
                    // Easiest way to fix this is to rotate all the cells by half the length of the
                    // cells slice, arriving at the first option
                    let cells: &mut [ComputeNode] = self.grid.cells_mut();
                    let cells_len: usize = cells.len();

                    cells.rotate_right(cells_len / 2_usize);

                    let half_col_len: i32 = self.grid.dimensions().y / 2_i32;

                    self.pos.y += half_col_len;
                    self.local_to_world.y -= half_col_len;
                }
                Direction::West => {
                    // The grid currently looks like this:
                    // #.
                    // ..
                    // And it needs to look like this:
                    // .#
                    // ..
                    // or this:
                    // ..
                    // .#
                    // Easiest way to fix this is to rotate all the cells by half the length of a
                    // row, arriving at the first option.
                    let half_row_len: i32 = self.grid.dimensions().x / 2_i32;

                    self.grid.cells_mut().rotate_right(half_row_len as usize);
                    self.pos.x += half_row_len;
                    self.local_to_world.x -= half_row_len;
                }
                _ => (),
            }
        }

        next_compute_node
            .is_infected()
            .then(|| self.pos + self.local_to_world + self.dir.rev().vec())
    }

    fn infections_after_bursts<B: Fn(Direction, ComputeNode) -> (Direction, ComputeNode)>(
        &mut self,
        burst_rules: B,
        bursts: usize,
        only_distinct_compute_nodes: bool,
        only_new_compute_nodes: bool,
    ) -> usize {
        let old_compute_nodes: HashSet<IVec2> = if only_new_compute_nodes {
            self.grid
                .cells()
                .iter()
                .enumerate()
                .filter_map(|(index, compute_node)| {
                    compute_node
                        .is_infected()
                        .then(|| self.grid.pos_from_index(index) + self.local_to_world)
                })
                .collect()
        } else {
            HashSet::new()
        };

        let mut distinct_compute_nodes: HashSet<IVec2> = HashSet::new();
        let mut infections: usize = 0_usize;

        for _ in 0_usize..bursts {
            if let Some(compute_node_pos) = self.burst(&burst_rules) {
                if !old_compute_nodes.contains(&compute_node_pos) && !only_distinct_compute_nodes
                    || !distinct_compute_nodes.contains(&compute_node_pos)
                {
                    infections += 1_usize;

                    if only_distinct_compute_nodes {
                        distinct_compute_nodes.insert(compute_node_pos);
                    }
                }
            }
        }

        infections
    }
}

impl Parse for ComputingCluster {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map_opt(Grid2D::parse, |grid| {
            let two: IVec2 = 2_i32 * IVec2::ONE;

            grid.dimensions()
                .rem_euclid(two)
                .cmpeq(IVec2::ONE)
                .all()
                .then(|| {
                    let pos: IVec2 = grid.dimensions().div_euclid(two);
                    let local_to_world: IVec2 = IVec2::ZERO;
                    let dir: Direction = Direction::North;

                    Self {
                        grid,
                        pos,
                        local_to_world,
                        dir,
                    }
                })
        })(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(ComputingCluster);

impl Solution {
    const MANY_SIMPLE_BURSTS: usize = 10000_usize;
    const MANY_COMPLEX_BURSTS: usize = 10000000_usize;
    const ONLY_DISTINCT_COMPUTE_NODES: bool = false;
    const ONLY_NEW_COMPUTE_NODES: bool = true;

    fn infections_after_bursts_internal<
        B: Fn(Direction, ComputeNode) -> (Direction, ComputeNode),
    >(
        &self,
        burst_rules: B,
        bursts: usize,
        only_distinct_compute_nodes: bool,
        only_new_compute_nodes: bool,
    ) -> usize {
        let mut computing_cluster: ComputingCluster = self.0.clone();

        computing_cluster.infections_after_bursts(
            burst_rules,
            bursts,
            only_distinct_compute_nodes,
            only_new_compute_nodes,
        )
    }

    fn infections_after_simple_bursts(&self, bursts: usize) -> usize {
        self.infections_after_bursts_internal(
            ComputingCluster::simple_burst_rules,
            bursts,
            Self::ONLY_DISTINCT_COMPUTE_NODES,
            Self::ONLY_NEW_COMPUTE_NODES,
        )
    }

    fn infections_after_complex_bursts(&self, bursts: usize) -> usize {
        self.infections_after_bursts_internal(
            ComputingCluster::complex_burst_rules,
            bursts,
            Self::ONLY_DISTINCT_COMPUTE_NODES,
            Self::ONLY_NEW_COMPUTE_NODES,
        )
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(ComputingCluster::parse, Self)(input)
    }
}

impl RunQuestions for Solution {
    /// Neat 2D turing machine
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.infections_after_simple_bursts(Self::MANY_SIMPLE_BURSTS));
    }

    /// Simple extension.
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.infections_after_complex_bursts(Self::MANY_COMPLEX_BURSTS));
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
        ..#\n\
        #..\n\
        ...\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            use ComputeNode::{Clean as C, Infected as I};

            vec![Solution(ComputingCluster {
                grid: Grid2D::try_from_cells_and_width(vec![C, C, I, I, C, C, C, C, C], 3_usize)
                    .unwrap(),
                pos: IVec2::ONE,
                local_to_world: IVec2::ZERO,
                dir: Direction::North,
            })]
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
    fn test_burst() {
        for (index, expected_computing_clusters) in [&[
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..#\n\
                        ##.\n\
                        ...\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(0_i32, 1_i32),
                    local_to_world: IVec2::ZERO,
                    dir: Direction::West,
                },
                Some(IVec2::ONE),
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..#\n\
                        .#.\n\
                        ...\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(0_i32, 0_i32),
                    local_to_world: IVec2::ZERO,
                    dir: Direction::North,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ...#.#\n\
                        ....#.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(2_i32, 0_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::West,
                },
                Some(IVec2::ZERO),
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..##.#\n\
                        ....#.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(2_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::South,
                },
                Some(IVec2::new(-1_i32, 0_i32)),
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..##.#\n\
                        ..#.#.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(3_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::East,
                },
                Some(IVec2::new(-1_i32, 1_i32)),
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..##.#\n\
                        ..###.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(3_i32, 0_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::North,
                },
                Some(IVec2::new(0_i32, 1_i32)),
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..#..#\n\
                        ..###.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(4_i32, 0_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::East,
                },
                None,
            ),
        ]]
        .into_iter()
        .enumerate()
        {
            let mut real_computing_cluster: ComputingCluster = solution(index).0.clone();

            for (expected_computing_cluster, infected_pos) in expected_computing_clusters {
                assert_eq!(
                    real_computing_cluster.burst(ComputingCluster::simple_burst_rules),
                    *infected_pos
                );
                assert_eq!(real_computing_cluster, *expected_computing_cluster);
            }
        }

        for (index, expected_computing_clusters) in [&[
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..#\n\
                        #W.\n\
                        ...\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(0_i32, 1_i32),
                    local_to_world: IVec2::ZERO,
                    dir: Direction::West,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..#\n\
                        FW.\n\
                        ...\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(0_i32, 0_i32),
                    local_to_world: IVec2::ZERO,
                    dir: Direction::North,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ...W.#\n\
                        ...FW.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(2_i32, 0_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::West,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..WW.#\n\
                        ...FW.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(2_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::South,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..WW.#\n\
                        ..WFW.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(3_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::East,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..WW.#\n\
                        ..W.W.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(2_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::West,
                },
                None,
            ),
            (
                ComputingCluster {
                    grid: Grid2D::parse(
                        "\
                        ..WW.#\n\
                        ..#.W.\n\
                        ......\n\
                        ......\n\
                        ......\n\
                        ......\n",
                    )
                    .unwrap()
                    .1,
                    pos: IVec2::new(1_i32, 1_i32),
                    local_to_world: IVec2::new(-3_i32, 0_i32),
                    dir: Direction::West,
                },
                Some(IVec2::new(-1_i32, 1_i32)),
            ),
        ]]
        .into_iter()
        .enumerate()
        {
            let mut real_computing_cluster: ComputingCluster = solution(index).0.clone();

            for (expected_computing_cluster, infected_pos) in expected_computing_clusters {
                assert_eq!(
                    real_computing_cluster.burst(ComputingCluster::complex_burst_rules),
                    *infected_pos
                );
                assert_eq!(real_computing_cluster, *expected_computing_cluster);
            }
        }
    }

    #[test]
    fn test_infections_after_simple_bursts() {
        for (index, bursts_and_infections) in [&[
            (70_usize, 41_usize),
            (Solution::MANY_SIMPLE_BURSTS, 5587_usize),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (bursts, infections) in bursts_and_infections {
                assert_eq!(
                    solution.infections_after_simple_bursts(*bursts),
                    *infections
                );
            }
        }
    }

    #[test]
    fn test_infections_after_complex_bursts() {
        for (index, bursts_and_infections) in [&[
            (100_usize, 26_usize),
            (Solution::MANY_COMPLEX_BURSTS, 2511944_usize),
        ]]
        .into_iter()
        .enumerate()
        {
            let solution: &Solution = solution(index);

            for (bursts, infections) in bursts_and_infections {
                assert_eq!(
                    solution.infections_after_complex_bursts(*bursts),
                    *infections
                );
            }
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
