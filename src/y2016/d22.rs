use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        character::complete::{digit1, line_ending, space1},
        combinator::{iterator, map, opt, success, verify},
        error::Error,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    num::Integer,
    std::collections::{HashMap, VecDeque},
    strum::IntoEnumIterator,
};

/* --- Day 22: Grid Computing ---

You gain access to a massive storage cluster arranged in a grid; each storage node is only connected to the four nodes directly adjacent to it (three if the node is on an edge, two if it's in a corner).

You can directly access data only on node /dev/grid/node-x0-y0, but you can perform some limited actions on the other nodes:

    You can get the disk usage of all nodes (via df). The result of doing this is in your puzzle input.
    You can instruct a node to move (not copy) all of its data to an adjacent node (if the destination node has enough space to receive the data). The sending node is left empty after this operation.

Nodes are named by their position: the node named node-x10-y10 is adjacent to nodes node-x9-y10, node-x11-y10, node-x10-y9, and node-x10-y11.

Before you begin, you need to understand the arrangement of data on these nodes. Even though you can only move data between directly connected nodes, you're going to need to rearrange a lot of the data to get access to the data you need. Therefore, you need to work out how you might be able to shift data around.

To do this, you'd like to count the number of viable pairs of nodes. A viable pair is any two nodes (A,B), regardless of whether they are directly connected, such that:

    Node A is not empty (its Used is not zero).
    Nodes A and B are not the same node.
    The data on node A (its Used) would fit on node B (its Avail).

How many viable pairs of nodes are there?

--- Part Two ---

Now that you have a better understanding of the grid, it's time to get to work.

Your goal is to gain access to the data which begins in the node with y=0 and the highest x (that is, the node in the top-right corner).

For example, suppose you have the following grid:

Filesystem            Size  Used  Avail  Use%
/dev/grid/node-x0-y0   10T    8T     2T   80%
/dev/grid/node-x0-y1   11T    6T     5T   54%
/dev/grid/node-x0-y2   32T   28T     4T   87%
/dev/grid/node-x1-y0    9T    7T     2T   77%
/dev/grid/node-x1-y1    8T    0T     8T    0%
/dev/grid/node-x1-y2   11T    7T     4T   63%
/dev/grid/node-x2-y0   10T    6T     4T   60%
/dev/grid/node-x2-y1    9T    8T     1T   88%
/dev/grid/node-x2-y2    9T    6T     3T   66%

In this example, you have a storage grid 3 nodes wide and 3 nodes tall. The node you can access directly, node-x0-y0, is almost full. The node containing the data you want to access, node-x2-y0 (because it has y=0 and the highest x value), contains 6 terabytes of data - enough to fit on your node, if only you could make enough space to move it there.

Fortunately, node-x1-y1 looks like it has enough free space to enable you to move some of this data around. In fact, it seems like all of the nodes have enough space to hold any node's data (except node-x0-y2, which is much larger, very full, and not moving any time soon). So, initially, the grid's capacities and connections look like this:

( 8T/10T) --  7T/ 9T -- [ 6T/10T]
    |           |           |
  6T/11T  --  0T/ 8T --   8T/ 9T
    |           |           |
 28T/32T  --  7T/11T --   6T/ 9T

The node you can access directly is in parentheses; the data you want starts in the node marked by square brackets.

In this example, most of the nodes are interchangable: they're full enough that no other node's data would fit, but small enough that their data could be moved around. Let's draw these nodes as .. The exceptions are the empty node, which we'll draw as _, and the very large, very full node, which we'll draw as #. Let's also draw the goal data as G. Then, it looks like this:

(.) .  G
 .  _  .
 #  .  .

The goal is to move the data in the top right, G, to the node in parentheses. To do this, we can issue some commands to the grid and rearrange the data:

    Move data from node-y0-x1 to node-y1-x1, leaving node node-y0-x1 empty:

    (.) _  G
     .  .  .
     #  .  .

    Move the goal data from node-y0-x2 to node-y0-x1:

    (.) G  _
     .  .  .
     #  .  .

    At this point, we're quite close. However, we have no deletion command, so we have to move some more data around. So, next, we move the data from node-y1-x2 to node-y0-x2:

    (.) G  .
     .  .  _
     #  .  .

    Move the data from node-y1-x1 to node-y1-x2:

    (.) G  .
     .  _  .
     #  .  .

    Move the data from node-y1-x0 to node-y1-x1:

    (.) G  .
     _  .  .
     #  .  .

    Next, we can free up space on our node by moving the data from node-y0-x0 to node-y1-x0:

    (_) G  .
     .  .  .
     #  .  .

    Finally, we can access the goal data by moving the it from node-y0-x1 to node-y0-x0:

    (G) _  .
     .  .  .
     #  .  .

So, after 7 steps, we've accessed the data we want. Unfortunately, each of these moves takes time, and we need to be efficient:

What is the fewest number of steps required to move your goal data to node-x0-y0? */

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default, Clone, Copy)]
struct ServerNode {
    size: u16,
    used: u16,
}

impl ServerNode {
    fn available(self) -> u16 {
        self.size - self.used
    }
}

impl Parse for ServerNode {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((Solution::parse_terabytes, space1, Solution::parse_terabytes)),
            |(size, _, used)| Self { size, used },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug))]
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
struct GraphNode {
    goal_data: IVec2,
    empty_node: IVec2,
}

impl GraphNode {
    const INVALID: Self = Self {
        goal_data: IVec2::NEG_ONE,
        empty_node: IVec2::NEG_ONE,
    };

    fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }

    fn dist_empty_to_goal(&self) -> i32 {
        manhattan_distance_2d(self.goal_data, self.empty_node)
    }

    fn dist_goal_to_access(&self) -> i32 {
        manhattan_magnitude_2d(self.goal_data)
    }

    fn heuristic(&self) -> i32 {
        self.dist_goal_to_access() + self.dist_empty_to_goal() - 1_i32
    }
}

define_cell! {
    #[repr(u8)]
    enum ServerNodeCell {
        Occupied = OCCUPIED = b'.',
        Locked = LOCKED = b'#',
        Empty = EMPTY = b'_',
        Goal = GOAL = b'G',
    }
}

impl Default for ServerNodeCell {
    fn default() -> Self {
        Self::Occupied
    }
}

struct GraphNodeData {
    parent: GraphNode,
    cost: i32,
}

struct MoveSequenceFinder<'s> {
    simple_grid: &'s SimpleGrid,
    node_data_map: HashMap<GraphNode, GraphNodeData>,
}

impl<'s> WeightedGraphSearch for MoveSequenceFinder<'s> {
    type Vertex = GraphNode;
    type Cost = i32;

    fn start(&self) -> &Self::Vertex {
        &self.simple_grid.start
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        vertex.goal_data == IVec2::ZERO
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<GraphNode> = VecDeque::new();
        let mut vertex: GraphNode = *vertex;

        while vertex != self.simple_grid.start {
            path.push_front(vertex);
            vertex = self.node_data_map.get(&vertex).unwrap().parent;
            assert!(vertex.is_valid());
        }

        path.push_front(vertex);
        path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.node_data_map
            .get(vertex)
            .map_or(i32::MAX, |node_data| node_data.cost)
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        vertex.heuristic()
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        neighbors.clear();

        neighbors.extend(
            (vertex.dist_empty_to_goal() == 1_i32)
                .then(|| GraphNode {
                    goal_data: vertex.empty_node,
                    empty_node: vertex.goal_data,
                })
                .into_iter()
                .chain(Direction::iter().filter_map(|dir| {
                    let neighbor_empty_node: IVec2 = vertex.empty_node + dir.vec();

                    (neighbor_empty_node != vertex.goal_data
                        && self.simple_grid.grid.contains(neighbor_empty_node)
                        && !self.simple_grid.is_locked_node
                            [self.simple_grid.grid.index_from_pos(neighbor_empty_node)])
                    .then(|| GraphNode {
                        goal_data: vertex.goal_data,
                        empty_node: neighbor_empty_node,
                    })
                }))
                .map(|node| OpenSetElement(node, 1_i32)),
        );
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        self.node_data_map.insert(
            *to,
            GraphNodeData {
                parent: *from,
                cost,
            },
        );
    }

    fn reset(&mut self) {
        self.node_data_map.clear();
        self.node_data_map.insert(
            self.simple_grid.start,
            GraphNodeData {
                parent: GraphNode::INVALID,
                cost: 0_i32,
            },
        );
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
struct SimpleGrid {
    start: GraphNode,
    is_locked_node: BitVec,
    grid: Grid2D<()>,
}

impl SimpleGrid {
    fn node_string(&self, node: GraphNode) -> String {
        let mut grid: Grid2D<ServerNodeCell> = Grid2D::default(self.grid.dimensions());

        for index in self.is_locked_node.iter_ones() {
            grid.cells_mut()[index] = ServerNodeCell::Locked;
        }

        *grid.get_mut(node.empty_node).unwrap() = ServerNodeCell::Empty;
        *grid.get_mut(node.goal_data).unwrap() = ServerNodeCell::Goal;

        grid.into()
    }

    fn try_move_sequence(&self) -> Option<Vec<GraphNode>> {
        let mut move_sequence_finder: MoveSequenceFinder = MoveSequenceFinder {
            simple_grid: self,
            node_data_map: HashMap::new(),
        };

        move_sequence_finder.run_a_star()
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Grid2D<ServerNode>);

impl Solution {
    fn parse_terabytes<'i>(input: &'i str) -> IResult<&'i str, u16> {
        terminated(parse_integer, tag("T"))(input)
    }

    fn parse_pos_and_node<'i>(input: &'i str) -> IResult<&'i str, (IVec2, ServerNode)> {
        let (input, (_, x, _, y, _, node, _)): (&str, (_, i32, _, i32, _, ServerNode, _)) =
            tuple((
                tag("/dev/grid/node-x"),
                parse_integer,
                tag("-y"),
                parse_integer,
                space1,
                ServerNode::parse,
                space1,
            ))(input)?;
        let input = tuple((
            verify(Self::parse_terabytes, |available| {
                *available == node.available()
            }),
            space1,
            digit1,
            tag("%"),
            opt(line_ending),
        ))(input)?
        .0;

        Ok((input, (IVec2::new(x, y), node)))
    }

    fn iter_viable_node_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0
            .cells()
            .iter()
            .enumerate()
            .filter(|(_, node)| node.used != 0_u16)
            .flat_map(|(index_a, node_a)| {
                self.0
                    .cells()
                    .iter()
                    .enumerate()
                    .filter_map(move |(index_b, node_b)| {
                        (index_b != index_a && node_a.used <= node_b.available())
                            .then_some((index_a, index_b))
                    })
            })
    }

    fn viable_node_pair_count(&self) -> usize {
        self.iter_viable_node_pairs().count()
    }

    fn try_simple_grid(&self) -> Option<SimpleGrid> {
        let mut empty_node: Option<IVec2> = None;
        let mut size_frequencies: HashMap<u16, usize> = HashMap::new();

        for (index, server_node) in self.0.cells().iter().enumerate() {
            if server_node.size > 0_u16 && server_node.used == 0_u16 {
                empty_node.is_none().then_some(())?;
                empty_node = Some(self.0.pos_from_index(index));
            }

            if let Some(frequency) = size_frequencies.get_mut(&server_node.size) {
                *frequency += 1_usize;
            } else {
                size_frequencies.insert(server_node.size, 1_usize);
            }
        }

        let goal_data: IVec2 = self.0.max_dimensions() * IVec2::X;
        let empty_node: IVec2 = empty_node?;
        let start: GraphNode = GraphNode {
            goal_data,
            empty_node,
        };

        let mut size_frequencies: Vec<(u16, usize)> = size_frequencies.into_iter().collect();

        size_frequencies.sort_by(|(size_a, frequency_a), (size_b, frequency_b)| {
            frequency_a.cmp(frequency_b).then(size_a.cmp(size_b))
        });

        let median_size_index: usize =
            size_frequencies.len() / 2_usize + size_frequencies.len().is_odd() as usize;
        let median_size: u16 = size_frequencies[median_size_index].0;
        let used_threshold: u16 = median_size * 2_u16;
        let is_locked_node: BitVec = self
            .0
            .cells()
            .iter()
            .map(|server_node| server_node.size == 0_u16 || server_node.used > used_threshold)
            .collect();
        let grid: Grid2D<()> = Grid2D::empty(self.0.dimensions());

        Some(SimpleGrid {
            start,
            is_locked_node,
            grid,
        })
    }

    fn try_move_sequence(&self) -> Option<Vec<GraphNode>> {
        self.try_simple_grid()
            .as_ref()
            .map(SimpleGrid::try_move_sequence)
            .flatten()
    }

    fn try_steps_to_access_goal_data(&self) -> Option<usize> {
        self.try_move_sequence()
            .map(|move_sequence| move_sequence.len() - 1_usize)
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let input: &str = tuple((
            tag("root@ebhq-gridcenter# df -h"),
            line_ending,
            tag("Filesystem"),
            space1,
            tag("Size"),
            space1,
            tag("Used"),
            space1,
            tag("Avail"),
            space1,
            tag("Use%"),
            line_ending,
        ))(input)?
        .0;

        let mut pos_and_node_iter = iterator(input, Self::parse_pos_and_node);

        let (min, max): (IVec2, IVec2) = pos_and_node_iter
            .fold((IVec2::MAX, IVec2::MIN), |(min, max), (pos, _)| {
                (min.min(pos), max.max(pos))
            });

        pos_and_node_iter.finish()?;

        verify(success(()), |_| min.cmple(max).all())(input)?;

        let offset: IVec2 = -min;
        let dimensions: IVec2 = (max - min) + IVec2::ONE;

        let mut grid: Grid2D<ServerNode> = Grid2D::default(dimensions);
        let mut pos_and_node_iter = iterator(input, Self::parse_pos_and_node);

        for (pos, node) in &mut pos_and_node_iter {
            *grid.get_mut(pos + offset).unwrap() = node;
        }

        let input: &str = pos_and_node_iter.finish()?.0;

        Ok((input, Self(grid)))
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.viable_node_pair_count());
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if !args.verbose {
            dbg!(self.try_steps_to_access_goal_data());
        } else if let Some(simple_grid) = self.try_simple_grid() {
            if let Some(move_sequence) = simple_grid.try_move_sequence() {
                dbg!(move_sequence.len() - 1_usize);

                for (index, node) in move_sequence.into_iter().enumerate() {
                    println!("Node {index}:\n{}\n", simple_grid.node_string(node));
                }
            } else {
                eprintln!("Failed to find move sequence!");
            }
        } else {
            eprintln!("Failed to convert to simple grid!");
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

    const SOLUTION_STR: &'static str = "\
        root@ebhq-gridcenter# df -h\n\
        Filesystem            Size  Used  Avail  Use%\n\
        /dev/grid/node-x0-y0   10T    8T     2T   80%\n\
        /dev/grid/node-x0-y1   11T    6T     5T   54%\n\
        /dev/grid/node-x0-y2   32T   28T     4T   87%\n\
        /dev/grid/node-x1-y0    9T    7T     2T   77%\n\
        /dev/grid/node-x1-y1    8T    0T     8T    0%\n\
        /dev/grid/node-x1-y2   11T    7T     4T   63%\n\
        /dev/grid/node-x2-y0   10T    6T     4T   60%\n\
        /dev/grid/node-x2-y1    9T    8T     1T   88%\n\
        /dev/grid/node-x2-y2    9T    6T     3T   66%\n";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(
                Grid2D::try_from_cells_and_width(
                    [
                        (10, 8),
                        (9, 7),
                        (10, 6),
                        (11, 6),
                        (8, 0),
                        (9, 8),
                        (32, 28),
                        (11, 7),
                        (9, 6),
                    ]
                    .into_iter()
                    .map(|(size, used)| ServerNode { size, used })
                    .collect(),
                    3_usize,
                )
                .unwrap(),
            )
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_try_simple_grid() {
        assert_eq!(
            solution().try_simple_grid(),
            Some(SimpleGrid {
                start: GraphNode {
                    goal_data: IVec2::new(2_i32, 0_i32),
                    empty_node: IVec2::ONE
                },
                is_locked_node: [false, false, false, false, false, false, true, false, false]
                    .into_iter()
                    .collect(),
                grid: Grid2D::empty(3_i32 * IVec2::ONE)
            })
        )
    }

    #[test]
    fn test_try_steps_to_access_goal_data() {
        assert_eq!(solution().try_steps_to_access_goal_data(), Some(7_usize));
    }
}
