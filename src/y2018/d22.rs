use {
    crate::*,
    glam::{BVec2, IVec2},
    nom::{
        bytes::complete::tag, character::complete::line_ending, combinator::map, error::Error,
        sequence::tuple, Err, IResult,
    },
    std::{
        collections::VecDeque,
        ops::{Add, Range, Sub},
    },
    strum::{EnumCount, EnumIter, IntoEnumIterator},
};

/* --- Day 22: Mode Maze ---

This is it, your final stop: the year -483. It's snowing and dark outside; the only light you can see is coming from a small cottage in the distance. You make your way there and knock on the door.

A portly man with a large, white beard answers the door and invites you inside. For someone living near the North Pole in -483, he must not get many visitors, but he doesn't act surprised to see you. Instead, he offers you some milk and cookies.

After talking for a while, he asks a favor of you. His friend hasn't come back in a few hours, and he's not sure where he is. Scanning the region briefly, you discover one life signal in a cave system nearby; his friend must have taken shelter there. The man asks if you can go there to retrieve his friend.

The cave is divided into square regions which are either dominantly rocky, narrow, or wet (called its type). Each region occupies exactly one coordinate in X,Y format where X and Y are integers and zero or greater. (Adjacent regions can be the same type.)

The scan (your puzzle input) is not very detailed: it only reveals the depth of the cave system and the coordinates of the target. However, it does not reveal the type of each region. The mouth of the cave is at 0,0.

The man explains that due to the unusual geology in the area, there is a method to determine any region's type based on its erosion level. The erosion level of a region can be determined from its geologic index. The geologic index can be determined using the first rule that applies from the list below:

    The region at 0,0 (the mouth of the cave) has a geologic index of 0.
    The region at the coordinates of the target has a geologic index of 0.
    If the region's Y coordinate is 0, the geologic index is its X coordinate times 16807.
    If the region's X coordinate is 0, the geologic index is its Y coordinate times 48271.
    Otherwise, the region's geologic index is the result of multiplying the erosion levels of the regions at X-1,Y and X,Y-1.

A region's erosion level is its geologic index plus the cave system's depth, all modulo 20183. Then:

    If the erosion level modulo 3 is 0, the region's type is rocky.
    If the erosion level modulo 3 is 1, the region's type is wet.
    If the erosion level modulo 3 is 2, the region's type is narrow.

For example, suppose the cave system's depth is 510 and the target's coordinates are 10,10. Using % to represent the modulo operator, the cavern would look as follows:

    At 0,0, the geologic index is 0. The erosion level is (0 + 510) % 20183 = 510. The type is 510 % 3 = 0, rocky.
    At 1,0, because the Y coordinate is 0, the geologic index is 1 * 16807 = 16807. The erosion level is (16807 + 510) % 20183 = 17317. The type is 17317 % 3 = 1, wet.
    At 0,1, because the X coordinate is 0, the geologic index is 1 * 48271 = 48271. The erosion level is (48271 + 510) % 20183 = 8415. The type is 8415 % 3 = 0, rocky.
    At 1,1, neither coordinate is 0 and it is not the coordinate of the target, so the geologic index is the erosion level of 0,1 (8415) times the erosion level of 1,0 (17317), 8415 * 17317 = 145722555. The erosion level is (145722555 + 510) % 20183 = 1805. The type is 1805 % 3 = 2, narrow.
    At 10,10, because they are the target's coordinates, the geologic index is 0. The erosion level is (0 + 510) % 20183 = 510. The type is 510 % 3 = 0, rocky.

Drawing this same cave system with rocky as ., wet as =, narrow as |, the mouth as M, the target as T, with 0,0 in the top-left corner, X increasing to the right, and Y increasing downward, the top-left corner of the map looks like this:

M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Before you go in, you should determine the risk level of the area. For the rectangle that has a top-left corner of region 0,0 and a bottom-right corner of the region containing the target, add up the risk level of each individual region: 0 for rocky regions, 1 for wet regions, and 2 for narrow regions.

In the cave system above, because the mouth is at 0,0 and the target is at 10,10, adding up the risk level of all regions with an X coordinate from 0 to 10 and a Y coordinate from 0 to 10, this total is 114.

What is the total risk level for the smallest rectangle that includes 0,0 and the target's coordinates?

--- Part Two ---

Okay, it's time to go rescue the man's friend.

As you leave, he hands you some tools: a torch and some climbing gear. You can't equip both tools at once, but you can choose to use neither.

Tools can only be used in certain regions:

    In rocky regions, you can use the climbing gear or the torch. You cannot use neither (you'll likely slip and fall).
    In wet regions, you can use the climbing gear or neither tool. You cannot use the torch (if it gets wet, you won't have a light source).
    In narrow regions, you can use the torch or neither tool. You cannot use the climbing gear (it's too bulky to fit).

You start at 0,0 (the mouth of the cave) with the torch equipped and must reach the target coordinates as quickly as possible. The regions with negative X or Y are solid rock and cannot be traversed. The fastest route might involve entering regions beyond the X or Y coordinate of the target.

You can move to an adjacent region (up, down, left, or right; never diagonally) if your currently equipped tool allows you to enter that region. Moving to an adjacent region takes one minute. (For example, if you have the torch equipped, you can move between rocky and narrow regions, but cannot enter wet regions.)

You can change your currently equipped tool or put both away if your new equipment would be valid for your current region. Switching to using the climbing gear, torch, or neither always takes seven minutes, regardless of which tools you start with. (For example, if you are in a rocky region, you can switch from the torch to the climbing gear, but you cannot switch to neither.)

Finally, once you reach the target, you need the torch equipped before you can find him in the dark. The target is always in a rocky region, so if you arrive there with climbing gear equipped, you will need to spend seven minutes switching to your torch.

For example, using the same cave system as above, starting in the top left corner (0,0) and moving to the bottom right corner (the target, 10,10) as quickly as possible, one possible route is as follows, with your current position marked X:

Initially:
X=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Down:
M=.|=.|.|=.|=|=.
X|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Right:
M=.|=.|.|=.|=|=.
.X=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Switch from using the torch to neither tool:
M=.|=.|.|=.|=|=.
.X=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Right 3:
M=.|=.|.|=.|=|=.
.|=|X|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Switch from using neither tool to the climbing gear:
M=.|=.|.|=.|=|=.
.|=|X|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Down 7:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..X==..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Right:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..=X=..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Down 3:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||.X.|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Right:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||..X|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Down:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.X..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Right 4:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===T===||
=|||...|==..|=.|
=.=|=.=..=X||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Up 2:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===X===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

Switch from using the climbing gear to the torch:
M=.|=.|.|=.|=|=.
.|=|=|||..|.=...
.==|....||=..|==
=.|....|.==.|==.
=|..==...=.|==..
=||.=.=||=|=..|=
|.=.===|||..=..|
|..==||=.|==|===
.=..===..=|.|||.
.======|||=|=.|=
.===|=|===X===||
=|||...|==..|=.|
=.=|=.=..=.||==|
||=|=...|==.=|==
|=.=||===.|||===
||.|==.|.|.||=||

This is tied with other routes as the fastest way to reach the target: 45 minutes. In it, 21 minutes are spent switching tools (three times, seven minutes each) and the remaining 24 minutes are spent moving.

What is the fewest number of minutes you can take to reach the target? */

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, PartialEq)]
    enum Region {
        Rocky = ROCKY = b'.',
        Wet = WET = b'=',
        Narrow = NARROW = b'|',
        Mouth = MOUTH = b'M',
        Target = TARGET = b'T',
        Visited = VISITED = b'o',
        ToolChange = TOOL_CHANGE = b'x',
    }
}

impl Region {
    fn from_risk_level(risk_level: u16) -> Self {
        match risk_level {
            0_u16 => Self::Rocky,
            1_u16 => Self::Wet,
            2_u16 => Self::Narrow,
            _ => unreachable!(),
        }
    }

    fn allows_tool(self, tool: Tool) -> bool {
        matches!(
            (self, tool),
            (Self::Rocky, Tool::ClimbingGear | Tool::Torch)
                | (Self::Wet, Tool::ClimbingGear | Tool::Neither)
                | (Self::Narrow, Tool::Torch | Tool::Neither)
        )
    }
}

#[derive(Clone, Copy, EnumCount, Eq, Hash, EnumIter, PartialEq)]
enum Tool {
    Torch,
    ClimbingGear,
    Neither,
}

#[derive(Clone, Copy)]
enum VertexDelta {
    DirectionNorth,
    DirectionEast,
    DirectionSouth,
    DirectionWest,
    ToolTorch,
    ToolClimbingGear,
    ToolNeither,
}

impl VertexDelta {
    fn try_direction(self) -> Option<Direction> {
        match self {
            Self::DirectionNorth => Some(Direction::North),
            Self::DirectionEast => Some(Direction::East),
            Self::DirectionSouth => Some(Direction::South),
            Self::DirectionWest => Some(Direction::West),
            _ => None,
        }
    }

    fn try_tool(self) -> Option<Tool> {
        match self {
            Self::ToolTorch => Some(Tool::Torch),
            Self::ToolClimbingGear => Some(Tool::ClimbingGear),
            Self::ToolNeither => Some(Tool::Neither),
            _ => None,
        }
    }
}

impl From<Direction> for VertexDelta {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => Self::DirectionNorth,
            Direction::East => Self::DirectionEast,
            Direction::South => Self::DirectionSouth,
            Direction::West => Self::DirectionWest,
        }
    }
}

impl From<Tool> for VertexDelta {
    fn from(value: Tool) -> Self {
        match value {
            Tool::Torch => Self::ToolTorch,
            Tool::ClimbingGear => Self::ToolClimbingGear,
            Tool::Neither => Self::ToolNeither,
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Vertex {
    pos: IVec2,
    tool: Tool,
}

impl Add<VertexDelta> for Vertex {
    type Output = Self;

    fn add(self, rhs: VertexDelta) -> Self::Output {
        match (rhs.try_direction(), rhs.try_tool()) {
            (Some(direction), None) => Self {
                pos: self.pos + direction.vec(),
                ..self
            },
            (None, Some(tool)) => Self { tool, ..self },
            _ => unreachable!(),
        }
    }
}

impl Sub for Vertex {
    type Output = VertexDelta;

    fn sub(self, rhs: Self) -> Self::Output {
        let pos_delta: IVec2 = self.pos - rhs.pos;

        Direction::try_from(pos_delta).map_or_else(|_| self.tool.into(), VertexDelta::from)
    }
}

#[derive(Clone)]
struct RegionData {
    erosion_level: u16,
    tool_dists: [u16; Tool::COUNT],
    tool_prev_verts: [Option<VertexDelta>; Tool::COUNT],
}

impl RegionData {
    fn region(&self) -> Region {
        Region::from_risk_level(Solution::risk_level(self.erosion_level))
    }
}

impl Default for RegionData {
    fn default() -> Self {
        Self {
            erosion_level: 0_u16,
            tool_dists: [u16::MAX; Tool::COUNT],
            tool_prev_verts: [None; Tool::COUNT],
        }
    }
}

trait ErosionLevel: Default {
    fn get(&self) -> u16;
    fn set(&mut self, erosion_level: u16);
}

impl ErosionLevel for u16 {
    fn get(&self) -> u16 {
        *self
    }

    fn set(&mut self, erosion_level: u16) {
        *self = erosion_level;
    }
}

impl ErosionLevel for RegionData {
    fn get(&self) -> u16 {
        self.erosion_level
    }

    fn set(&mut self, erosion_level: u16) {
        self.erosion_level = erosion_level;
    }
}

struct PathToTargetFinder {
    region_data_grid: Grid2D<RegionData>,
    depth: u16,
    mouth: Vertex,
    target: Vertex,
}

impl PathToTargetFinder {
    fn dist_to_target(&self) -> i32 {
        self.region_data_grid
            .get(self.target.pos)
            .unwrap()
            .tool_dists[self.target.tool as usize] as i32
    }

    fn try_dist_to_target(&mut self) -> Option<i32> {
        self.run_a_star().map(|_| self.dist_to_target())
    }

    fn try_dist_to_target_and_grid_string(&mut self) -> Option<(i32, String)> {
        self.run_a_star().map(|path| {
            let mut region_grid: Grid2D<Region> = Grid2D::try_from_cells_and_dimensions(
                self.region_data_grid
                    .cells()
                    .iter()
                    .map(|region_data| {
                        Region::from_risk_level(Solution::risk_level(region_data.erosion_level))
                    })
                    .collect(),
                self.region_data_grid.dimensions(),
            )
            .unwrap();

            for vertex in path {
                let region: &mut Region = region_grid.get_mut(vertex.pos).unwrap();

                *region = if *region == Region::Visited {
                    Region::ToolChange
                } else {
                    Region::Visited
                };
            }

            (self.dist_to_target(), region_grid.into())
        })
    }
}

impl WeightedGraphSearch for PathToTargetFinder {
    type Vertex = Vertex;

    type Cost = i32;

    fn start(&self) -> &Self::Vertex {
        &self.mouth
    }

    fn is_end(&self, vertex: &Self::Vertex) -> bool {
        *vertex == self.target
    }

    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        let mut path: VecDeque<Vertex> = VecDeque::new();
        let mut vertex: Vertex = *vertex;

        while {
            path.push_front(vertex);

            vertex != self.mouth
        } {
            vertex = vertex
                + self
                    .region_data_grid
                    .get(vertex.pos)
                    .unwrap()
                    .tool_prev_verts[vertex.tool as usize]
                    .unwrap();
        }

        path.into()
    }

    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost {
        self.region_data_grid.get(vertex.pos).unwrap().tool_dists[vertex.tool as usize] as i32
    }

    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost {
        manhattan_distance_2d(vertex.pos, self.target.pos) * Solution::POS_CHANGE_COST
            + (vertex.tool != self.target.tool) as i32 * Solution::TOOL_CHANGE_COST
    }

    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    ) {
        let region: Region = self.region_data_grid.get(vertex.pos).unwrap().region();

        neighbors.clear();
        neighbors.extend(
            Direction::iter()
                .filter_map(|direction| {
                    let neighbor_pos: IVec2 = vertex.pos + direction.vec();

                    self.region_data_grid
                        .get(neighbor_pos)
                        .map_or(false, |region_data| {
                            region_data.region().allows_tool(vertex.tool)
                        })
                        .then(|| {
                            OpenSetElement(
                                Vertex {
                                    pos: neighbor_pos,
                                    tool: vertex.tool,
                                },
                                Solution::POS_CHANGE_COST,
                            )
                        })
                })
                .chain(Tool::iter().filter_map(|tool| {
                    (tool != vertex.tool && region.allows_tool(tool)).then(|| {
                        OpenSetElement(
                            Vertex {
                                pos: vertex.pos,
                                tool,
                            },
                            Solution::TOOL_CHANGE_COST,
                        )
                    })
                })),
        );
    }

    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        _heuristic: Self::Cost,
    ) {
        let region_data: &mut RegionData = self.region_data_grid.get_mut(to.pos).unwrap();
        let tool_index: usize = to.tool as usize;

        region_data.tool_dists[tool_index] = cost as u16;
        region_data.tool_prev_verts[tool_index] = Some(*from - *to);

        let prev_dimensions: IVec2 = self.region_data_grid.dimensions();

        match to.pos.cmpeq(self.region_data_grid.max_dimensions()) {
            BVec2 { x: true, y: true } => {
                self.region_data_grid
                    .double_dimensions(RegionData::default());

                let dimensions: IVec2 = self.region_data_grid.dimensions();

                Solution::initialize_erosion_levels(
                    &mut self.region_data_grid,
                    self.depth,
                    prev_dimensions * IVec2::X..prev_dimensions * IVec2::new(2_i32, 1_i32),
                    self.target.pos,
                );
                Solution::initialize_erosion_levels(
                    &mut self.region_data_grid,
                    self.depth,
                    prev_dimensions * IVec2::Y..dimensions,
                    self.target.pos,
                );
            }
            BVec2 { x: false, y: true } => {
                self.region_data_grid.double_rows(RegionData::default());

                let dimensions: IVec2 = self.region_data_grid.dimensions();

                Solution::initialize_erosion_levels(
                    &mut self.region_data_grid,
                    self.depth,
                    prev_dimensions * IVec2::Y..dimensions,
                    self.target.pos,
                );
            }
            BVec2 { x: true, y: false } => {
                self.region_data_grid.double_cols(RegionData::default());

                let dimensions: IVec2 = self.region_data_grid.dimensions();

                Solution::initialize_erosion_levels(
                    &mut self.region_data_grid,
                    self.depth,
                    prev_dimensions * IVec2::X..dimensions,
                    self.target.pos,
                );
            }
            _ => (),
        }
    }

    fn reset(&mut self) {
        for region_data in self.region_data_grid.cells_mut() {
            region_data.tool_dists.fill(u16::MAX);
            region_data.tool_prev_verts.fill(None);
        }

        self.region_data_grid
            .get_mut(self.mouth.pos)
            .unwrap()
            .tool_dists[self.mouth.tool as usize] = 0_u16;
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    depth: u16,
    target: IVec2,
}

impl Solution {
    const EROSION_LEVEL_MOD: u16 = 20183_u16;
    const GEOLOGIC_INDEX_Y_EQ_0_X_SCALAR: u16 = 16807_u16 % Self::EROSION_LEVEL_MOD;
    const GEOLOGIC_INDEX_X_EQ_0_Y_SCALAR: u16 = 48271_u16 % Self::EROSION_LEVEL_MOD;
    const POS_CHANGE_COST: i32 = 1_i32;
    const TOOL_CHANGE_COST: i32 = 7_i32;

    fn product(a: u16, b: u16) -> u16 {
        ((a as u32 * b as u32) % Self::EROSION_LEVEL_MOD as u32) as u16
    }

    fn erosion_level(depth: u16, geologic_index: u16) -> u16 {
        (geologic_index + depth) % Self::EROSION_LEVEL_MOD
    }

    fn risk_level(erosion_level: u16) -> u16 {
        erosion_level % 3_u16
    }

    fn iter_risk_levels(erosion_level_grid: &Grid2D<u16>) -> impl Iterator<Item = u16> + '_ {
        erosion_level_grid
            .cells()
            .iter()
            .copied()
            .map(Self::risk_level)
    }

    fn sum_risk_levels(erosion_level_grid: &Grid2D<u16>) -> u16 {
        Self::iter_risk_levels(erosion_level_grid).sum()
    }

    fn grid_string(erosion_level_grid: &Grid2D<u16>) -> String {
        let mut region_grid: Grid2D<Region> = Grid2D::try_from_cells_and_dimensions(
            Self::iter_risk_levels(erosion_level_grid)
                .map(Region::from_risk_level)
                .collect(),
            erosion_level_grid.dimensions(),
        )
        .unwrap();

        *region_grid.get_mut(IVec2::ZERO).unwrap() = Region::Mouth;
        *region_grid.get_mut(region_grid.max_dimensions()).unwrap() = Region::Target;

        region_grid.into()
    }

    fn initialize_erosion_levels<E: ErosionLevel>(
        erosion_level_grid: &mut Grid2D<E>,
        depth: u16,
        range: Range<IVec2>,
        target: IVec2,
    ) {
        if range.start == IVec2::ZERO {
            erosion_level_grid
                .get_mut(IVec2::ZERO)
                .unwrap()
                .set(Self::erosion_level(depth, 0_u16))
        }

        if range.start.y == 0_i32 {
            for x in 1_i32..range.end.x {
                erosion_level_grid
                    .get_mut((x, 0_i32).into())
                    .unwrap()
                    .set(Self::erosion_level(
                        depth,
                        Self::product(x as u16, Self::GEOLOGIC_INDEX_Y_EQ_0_X_SCALAR),
                    ));
            }
        }

        if range.start.x == 0_i32 {
            for y in 1_i32..range.end.y {
                erosion_level_grid
                    .get_mut((0_i32, y).into())
                    .unwrap()
                    .set(Self::erosion_level(
                        depth,
                        Self::product(y as u16, Self::GEOLOGIC_INDEX_X_EQ_0_Y_SCALAR),
                    ));
            }
        }

        for pos in grid_2d_iter_positions(range.start.max(IVec2::ONE)..range.end) {
            let erosion_level: u16 = Self::erosion_level(
                depth,
                Self::product(
                    erosion_level_grid.get(pos - IVec2::X).unwrap().get(),
                    erosion_level_grid.get(pos - IVec2::Y).unwrap().get(),
                ),
            );

            erosion_level_grid.get_mut(pos).unwrap().set(erosion_level);
        }

        if target.cmpge(range.start).all() {
            erosion_level_grid
                .get_mut(target)
                .unwrap()
                .set(Self::erosion_level(depth, 0_u16));
        }
    }

    fn erosion_level_grid(&self) -> Grid2D<u16> {
        let dimensions: IVec2 = self.target + IVec2::ONE;
        let mut erosion_level_grid: Grid2D<u16> = Grid2D::default(dimensions);

        Self::initialize_erosion_levels(
            &mut erosion_level_grid,
            self.depth,
            IVec2::ZERO..dimensions,
            self.target,
        );

        erosion_level_grid
    }

    fn total_risk_level(&self) -> u16 {
        Self::sum_risk_levels(&self.erosion_level_grid())
    }

    fn path_to_target_finder(&self, erosion_level_grid: &Grid2D<u16>) -> PathToTargetFinder {
        PathToTargetFinder {
            region_data_grid: Grid2D::try_from_cells_and_dimensions(
                erosion_level_grid
                    .cells()
                    .iter()
                    .copied()
                    .map(|erosion_level| RegionData {
                        erosion_level,
                        ..RegionData::default()
                    })
                    .collect(),
                erosion_level_grid.dimensions(),
            )
            .unwrap(),
            depth: self.depth,
            mouth: Vertex {
                pos: IVec2::ZERO,
                tool: Tool::Torch,
            },
            target: Vertex {
                pos: self.target,
                tool: Tool::Torch,
            },
        }
    }

    fn try_dist_to_target(&self) -> Option<i32> {
        self.path_to_target_finder(&self.erosion_level_grid())
            .try_dist_to_target()
    }

    fn try_dist_to_target_and_grid_string(&self) -> Option<(i32, String)> {
        self.path_to_target_finder(&self.erosion_level_grid())
            .try_dist_to_target_and_grid_string()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                tag("depth: "),
                parse_integer,
                line_ending,
                tag("target: "),
                parse_integer,
                tag(","),
                parse_integer,
            )),
            |(_, depth, _, _, x, _, y)| Self {
                depth,
                target: (x, y).into(),
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// ring algebra!
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let erosion_level_grid: Grid2D<u16> = self.erosion_level_grid();

            dbg!(Self::sum_risk_levels(&erosion_level_grid));
            println!("{}", Self::grid_string(&erosion_level_grid));
        } else {
            dbg!(self.total_risk_level());
        }
    }

    /// Vertical growth is definitely more than it needs to be, but this was a really cool problem.
    /// 3D search problem while being easy to print in 2D.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some((dist_to_target, grid_string)) = self.try_dist_to_target_and_grid_string() {
                dbg!(dist_to_target);
                println!("{grid_string}");
            } else {
                eprintln!("Failed to find a path to the target.");
            }
        } else {
            dbg!(self.try_dist_to_target());
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

    const SOLUTION_STRS: &'static [&'static str] = &["depth: 510\ntarget: 10,10"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution {
                depth: 510_u16,
                target: (10_i32, 10_i32).into(),
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
    fn test_grid_string() {
        for (index, grid_string) in ["\
            M=.|=.|.|=.\n\
            .|=|=|||..|\n\
            .==|....||=\n\
            =.|....|.==\n\
            =|..==...=.\n\
            =||.=.=||=|\n\
            |.=.===|||.\n\
            |..==||=.|=\n\
            .=..===..=|\n\
            .======|||=\n\
            .===|=|===T\n"]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                Solution::grid_string(&solution(index).erosion_level_grid()),
                grid_string
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
