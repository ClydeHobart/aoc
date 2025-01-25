use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec2,
    nom::{
        branch::alt,
        bytes::complete::{is_a, tag},
        character::complete::one_of,
        combinator::{all_consuming, map, map_opt, success, verify},
        error::Error,
        multi::{many0, separated_list1},
        sequence::{delimited, tuple},
        Err, IResult,
    },
    num::Integer,
    std::cmp::Ordering,
};

/* --- Day 20: A Regular Map ---

While you were learning about instruction pointers, the Elves made considerable progress. When you look up, you discover that the North Pole base construction project has completely surrounded you.

The area you are in is made up entirely of rooms and doors. The rooms are arranged in a grid, and rooms only connect to adjacent rooms when a door is present between them.

For example, drawing rooms as ., walls as #, doors as | or -, your current position as X, and where north is up, the area you're in might look like this:

#####
#.|.#
#-###
#.|X#
#####

You get the attention of a passing construction Elf and ask for a map. "I don't have time to draw out a map of this place - it's huge. Instead, I can give you directions to every room in the facility!" He writes down some directions on a piece of parchment and runs off. In the example above, the instructions might have been ^WNE$, a regular expression or "regex" (your puzzle input).

The regex matches routes (like WNE for "west, north, east") that will take you from your current room through various doors in the facility. In aggregate, the routes will take you through every door in the facility at least once; mapping out all of these routes will let you build a proper map and find your way around.

^ and $ are at the beginning and end of your regex; these just mean that the regex doesn't match anything outside the routes it describes. (Specifically, ^ matches the start of the route, and $ matches the end of it.) These characters will not appear elsewhere in the regex.

The rest of the regex matches various sequences of the characters N (north), S (south), E (east), and W (west). In the example above, ^WNE$ matches only one route, WNE, which means you can move west, then north, then east from your current position. Sequences of letters like this always match that exact route in the same order.

Sometimes, the route can branch. A branch is given by a list of options separated by pipes (|) and wrapped in parentheses. So, ^N(E|W)N$ contains a branch: after going north, you must choose to go either east or west before finishing your route by going north again. By tracing out the possible routes after branching, you can determine where the doors are and, therefore, where the rooms are in the facility.

For example, consider this regex: ^ENWWW(NEEE|SSE(EE|N))$

This regex begins with ENWWW, which means that from your current position, all routes must begin by moving east, north, and then west three times, in that order. After this, there is a branch. Before you consider the branch, this is what you know about the map so far, with doors you aren't sure about marked with a ?:

#?#?#?#?#
?.|.|.|.?
#?#?#?#-#
    ?X|.?
    #?#?#

After this point, there is (NEEE|SSE(EE|N)). This gives you exactly two options: NEEE and SSE(EE|N). By following NEEE, the map now looks like this:

#?#?#?#?#
?.|.|.|.?
#-#?#?#?#
?.|.|.|.?
#?#?#?#-#
    ?X|.?
    #?#?#

Now, only SSE(EE|N) remains. Because it is in the same parenthesized group as NEEE, it starts from the same room NEEE started in. It states that starting from that point, there exist doors which will allow you to move south twice, then east; this ends up at another branch. After that, you can either move east twice or north once. This information fills in the rest of the doors:

#?#?#?#?#
?.|.|.|.?
#-#?#?#?#
?.|.|.|.?
#-#?#?#-#
?.?.?X|.?
#-#-#?#?#
?.|.|.|.?
#?#?#?#?#

Once you've followed all possible routes, you know the remaining unknown parts are all walls, producing a finished map of the facility:

#########
#.|.|.|.#
#-#######
#.|.|.|.#
#-#####-#
#.#.#X|.#
#-#-#####
#.|.|.|.#
#########

Sometimes, a list of options can have an empty option, like (NEWS|WNSE|). This means that routes at this point could effectively skip the options in parentheses and move on immediately. For example, consider this regex and the corresponding map:

^ENNWSWW(NEWS|)SSSEEN(WNSE|)EE(SWEN|)NNN$

###########
#.|.#.|.#.#
#-###-#-#-#
#.|.|.#.#.#
#-#####-#-#
#.#.#X|.#.#
#-#-#####-#
#.#.|.|.|.#
#-###-###-#
#.|.|.#.|.#
###########

This regex has one main route which, at three locations, can optionally include additional detours and be valid: (NEWS|), (WNSE|), and (SWEN|). Regardless of which option is taken, the route continues from the position it is left at after taking those steps. So, for example, this regex matches all of the following routes (and more that aren't listed here):

    ENNWSWWSSSEENEENNN
    ENNWSWWNEWSSSSEENEENNN
    ENNWSWWNEWSSSSEENEESWENNNN
    ENNWSWWSSSEENWNSEEENNN

By following the various routes the regex matches, a full map of all of the doors and rooms in the facility can be assembled.

To get a sense for the size of this facility, you'd like to determine which room is furthest from you: specifically, you would like to find the room for which the shortest path to that room would require passing through the most doors.

    In the first example (^WNE$), this would be the north-east corner 3 doors away.
    In the second example (^ENWWW(NEEE|SSE(EE|N))$), this would be the south-east corner 10 doors away.
    In the third example (^ENNWSWW(NEWS|)SSSEEN(WNSE|)EE(SWEN|)NNN$), this would be the north-east corner 18 doors away.

Here are a few more examples:

Regex: ^ESSWWN(E|NNENN(EESS(WNSE|)SSS|WWWSSSSE(SW|NNNE)))$
Furthest room requires passing 23 doors

#############
#.|.|.|.|.|.#
#-#####-###-#
#.#.|.#.#.#.#
#-#-###-#-#-#
#.#.#.|.#.|.#
#-#-#-#####-#
#.#.#.#X|.#.#
#-#-#-###-#-#
#.|.#.|.#.#.#
###-#-###-#-#
#.|.#.|.|.#.#
#############

Regex: ^WSSEESWWWNW(S|NENNEEEENN(ESSSSW(NWSW|SSEN)|WSWWN(E|WWS(E|SS))))$
Furthest room requires passing 31 doors

###############
#.|.|.|.#.|.|.#
#-###-###-#-#-#
#.|.#.|.|.#.#.#
#-#########-#-#
#.#.|.|.|.|.#.#
#-#-#########-#
#.#.#.|X#.|.#.#
###-#-###-#-#-#
#.|.#.#.|.#.|.#
#-###-#####-###
#.|.#.|.|.#.#.#
#-#-#####-#-#-#
#.#.|.|.|.#.|.#
###############

What is the largest number of doors you would be required to pass through to reach a room? That is, find the room for which the shortest path from your starting location to that room would require passing through the most doors; what is the fewest doors you can pass through to reach it?

--- Part Two ---

Okay, so the facility is big.

How many rooms have a shortest path from your current location that pass through at least 1000 doors? */

#[derive(Clone, Copy)]
struct PosAndDist {
    pos: IVec2,
    dist: usize,
}

impl PosAndDist {
    fn sortable_index(&self) -> u64 {
        sortable_index_from_pos(self.pos)
    }

    fn cmp_without_dist(a: &Self, b: &Self) -> Ordering {
        a.sortable_index().cmp(&b.sortable_index())
    }

    fn cmp_with_dist(a: &Self, b: &Self) -> Ordering {
        Self::cmp_without_dist(a, b).then_with(|| a.dist.cmp(&b.dist))
    }
}

trait ProcessNodeFunc
where
    Self: FnMut(&PosAndDist, &PosAndDist),
{
}

impl<T: FnMut(&PosAndDist, &PosAndDist)> ProcessNodeFunc for T {}

#[derive(Default)]
struct ProcessLayer<'i> {
    set_and_vec: SetAndVec<PosAndDist>,
    input: Option<&'i str>,
}

impl<'i> ProcessLayer<'i> {
    fn clear(&mut self) {
        self.set_and_vec.clear();
        self.input = None;
    }
}

// Use `NoDropVec` so that we don't drop those allocations each time we pop.
#[derive(Default)]
struct ProcessStack<'i>(NoDropVec<ProcessLayer<'i>>);

impl<'i> ProcessStack<'i> {
    fn push_layer(&mut self) {
        self.0.push();

        if let Some(prev_index) = self.0.len().checked_sub(2_usize) {
            let curr_index: usize = prev_index + 1_usize;
            let (prev_slice, curr_slice): (&mut [ProcessLayer], &mut [ProcessLayer]) =
                self.0.split_at_mut(curr_index);
            let prev: &mut ProcessLayer = &mut prev_slice[prev_index];
            let curr: &mut ProcessLayer = &mut curr_slice[0_usize];

            curr.clear();
            curr.set_and_vec
                .extend_vec(prev.set_and_vec.set_as_slice().iter().copied());
        }
    }

    fn pop_layer(&mut self) {
        if let Some(prev_index) = self.0.len().checked_sub(2_usize) {
            let curr_index: usize = prev_index + 1_usize;
            let (prev_slice, curr_slice): (&mut [ProcessLayer], &mut [ProcessLayer]) =
                self.0.split_at_mut(curr_index);
            let prev: &mut ProcessLayer = &mut prev_slice[prev_index];
            let curr: &mut ProcessLayer = &mut curr_slice[0_usize];

            prev.set_and_vec
                .extend_vec(curr.set_and_vec.vec_as_slice().iter().copied());
            curr.clear();
        }

        self.0.pop();
    }

    fn process_directions<F: ProcessNodeFunc, I: IntoIterator<Item = Direction>>(
        &mut self,
        visit_door: &mut F,
        iter: I,
    ) {
        let curr: &mut [PosAndDist] = self.0.last_mut().unwrap().set_and_vec.vec_as_slice_mut();

        for delta in iter.into_iter().map(Direction::vec) {
            for curr_node in curr.iter_mut() {
                let curr_node_value: PosAndDist = *curr_node;
                let next_node: PosAndDist = PosAndDist {
                    pos: curr_node_value.pos + delta,
                    dist: curr_node_value.dist + 1_usize,
                };

                visit_door(&curr_node_value, &next_node);
                *curr_node = next_node;
            }
        }
    }

    fn pre_process_valid_sequence(&mut self) {
        // All positions are currently in the vec at the top layer. We need to move them to the set
        // (which is currently empty) to safely call `process_valid_sequence`.
        self.0
            .last_mut()
            .unwrap()
            .set_and_vec
            .extend_set_with_vec_by_key(PosAndDist::sortable_index);
    }

    fn post_process_valid_sequence(&mut self) {
        // The set has the contents from before, and the vec has the contents from after,
        // potentially with duplicates. We no longer need the before, but we do want the after, and
        // we need to de-dup it.
        let curr: &mut SetAndVec<PosAndDist> = &mut self.0.last_mut().unwrap().set_and_vec;

        curr.clear_set();

        // First sort the vec using the distance, meaning smaller distances will be earlier in the
        // list.
        curr.vec_as_slice_mut().sort_by(PosAndDist::cmp_with_dist);

        // Then when extending the vec, only consider the position. The smaller distance will be
        // first still, keeping only those when there are duplicates.
        curr.extend_set_with_vec_by_key(PosAndDist::sortable_index);
        curr.extend_vec_front_with_set();
    }

    /// # Parameters
    /// * `self`: It is expected that the caller has the input positions in the set at the top of
    /// the stack. The output positions will be in the vec at the top of the stack.
    fn process_valid_sequence<'a, F: ProcessNodeFunc>(
        &'a mut self,
        input: &'i str,
        visit_door: &'a mut F,
    ) -> IResult<&'i str, ()>
    where
        'i: 'a,
    {
        self.push_layer();
        self.0.last_mut().unwrap().input = Some(input);

        while {
            let input: &str = self.0.last().unwrap().input.unwrap();

            if input.is_empty() {
                false
            } else if let Ok((new_input, directions)) = Solution::parse_directions(input) {
                self.0.last_mut().unwrap().input = Some(new_input);
                self.process_directions(
                    visit_door,
                    directions.chars().map(|direction| {
                        Direction::from(Solution::DIRECTION_STR.find(direction).unwrap() as u8)
                    }),
                );

                true
            } else if let Ok((new_input, branches)) = Solution::parse_branches(input) {
                self.0.last_mut().unwrap().input = Some(new_input);
                self.pre_process_valid_sequence();
                self.push_layer();
                self.0.last_mut().unwrap().input = Some(tag("(")(branches)?.0);

                true
            } else if let Ok((new_input, _)) = Solution::parse_tag("|")(input) {
                self.pop_layer();
                self.push_layer();
                self.0.last_mut().unwrap().input = Some(new_input);

                true
            } else if let Ok((_, _)) = all_consuming(Solution::parse_tag(")"))(input) {
                self.pop_layer();
                self.post_process_valid_sequence();

                true
            } else {
                false
            }
        } {}

        let input: &str = self.0.last_mut().unwrap().input.unwrap();

        self.pop_layer();

        Ok((input, ()))
    }

    fn process<'a, F: ProcessNodeFunc>(
        &'a mut self,
        visit_door: &'a mut F,
        input: &'i str,
        initial_pos: IVec2,
    ) -> IResult<&'i str, &'a [PosAndDist]>
    where
        'i: 'a,
    {
        self.0.clear();

        let root: &mut SetAndVec<PosAndDist> = &mut self.0.push().set_and_vec;

        root.clear();
        root.push_vec(PosAndDist {
            pos: initial_pos,
            dist: 0_usize,
        });

        self.pre_process_valid_sequence();
        // let input: &'i str = self.process_valid_sequence(visit_door)(input)?.0;
        let input: &'i str = self.process_valid_sequence(input, visit_door)?.0;
        self.post_process_valid_sequence();

        Ok((input, self.0.last().unwrap().set_and_vec.vec_as_slice()))
    }
}

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Default)]
    enum Cell {
        Room = ROOM = b'.',
        #[default]
        Wall = WALL = b'#',
        VerticalDoor = VERTICAL_DOOR = b'|',
        HorizontalDoor = HORIZONTAL_DOOR = b'-',
        Start = START = b'X',
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    dist_grid: Grid2D<u16>,
    start: IVec2,
    doors: BitVec,
    max_dist: usize,
}

impl Solution {
    const DIRECTION_STR: &'static str = "NESW";
    const MIN_DIST: u16 = 1000_u16;

    fn parse_tag<'i>(tag_str: &'i str) -> impl FnMut(&'i str) -> IResult<&'i str, &'i str> {
        tag(tag_str)
    }

    fn parse_directions<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        is_a(Self::DIRECTION_STR)(input)
    }

    fn parse_branches<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        let remaining: &'i str = tuple((
            tag("("),
            separated_list1(tag("|"), map(Self::parse_valid_sequence, |_| ())),
            tag(")"),
        ))(input)?
        .0;

        Ok((remaining, &input[..input.len() - remaining.len()]))
    }

    fn parse_valid_sequence<'i>(input: &'i str) -> IResult<&'i str, &'i str> {
        let mut depth: usize = 0_usize;

        let remaining: &'i str = many0(alt((
            map(Self::parse_directions, |_| ()),
            map(
                verify(
                    map(one_of("(|)"), |c: char| match c {
                        '(' => {
                            depth += 1_usize;

                            true
                        }
                        '|' => depth > 0_usize,
                        ')' => depth.checked_sub(1_usize).map_or(false, |new_depth| {
                            depth = new_depth;

                            true
                        }),
                        _ => unreachable!(),
                    }),
                    |&is_verified| is_verified,
                ),
                |_| (),
            ),
        )))(input)?
        .0;

        verify(success(()), |_| depth == 0_usize)("depth was not 0")?;

        Ok((remaining, &input[..input.len() - remaining.len()]))
    }

    fn new(dimensions: IVec2, start: IVec2) -> Self {
        let max_dist: usize = 0_usize;
        let area: usize = (dimensions.x * dimensions.y) as usize;
        let doors: BitVec = bitvec![0; area * 2_usize];
        let dist_grid: Grid2D<u16> =
            Grid2D::try_from_cells_and_dimensions(vec![u16::MAX; area], dimensions).unwrap();

        Self {
            start,
            max_dist,
            doors,
            dist_grid,
        }
    }

    fn index(&self, a: IVec2, b: IVec2) -> usize {
        let min: IVec2 = a.min(b);
        let max: IVec2 = a.max(b);
        let delta: IVec2 = max - min;

        assert_eq!(delta.min_element(), 0_i32);
        assert_eq!(delta.max_element(), 1_i32);

        grid_2d_try_index_from_pos_and_dimensions(min, self.dist_grid.dimensions()).unwrap()
            + (delta.y == 1_i32) as usize * self.doors.len() / 2_usize
    }

    fn local_pos_to_grid_pos(pos: IVec2) -> IVec2 {
        2_i32 * pos + IVec2::ONE
    }

    fn grid(&self) -> Grid2D<Cell> {
        let grid_dimensions: IVec2 = Self::local_pos_to_grid_pos(self.dist_grid.dimensions());
        let mut grid: Grid2D<Cell> = Grid2D::default(grid_dimensions);

        for pos in (0_i32..grid_dimensions.y)
            .filter(|y| y.is_odd())
            .flat_map(|y| {
                (0_i32..grid_dimensions.x)
                    .filter(|x| x.is_odd())
                    .map(move |x| IVec2::new(x, y))
            })
        {
            *grid.get_mut(pos).unwrap() = Cell::Room;
        }

        *grid
            .get_mut(Self::local_pos_to_grid_pos(self.start))
            .unwrap() = Cell::Start;

        let half_doors_len: usize = self.doors.len() / 2_usize;

        for (index, delta, cell) in self.doors[..half_doors_len]
            .iter_ones()
            .map(|vertical_door_index| (vertical_door_index, IVec2::X, Cell::VerticalDoor))
            .chain(
                self.doors[half_doors_len..]
                    .iter_ones()
                    .map(|horizontal_door_index| {
                        (horizontal_door_index, IVec2::Y, Cell::HorizontalDoor)
                    }),
            )
        {
            *grid
                .get_mut(
                    Self::local_pos_to_grid_pos(grid_2d_pos_from_index_and_dimensions(
                        index,
                        self.dist_grid.dimensions(),
                    )) + delta,
                )
                .unwrap() = cell;
        }

        grid
    }

    fn grid_string(&self) -> String {
        self.grid().into()
    }

    fn rooms_with_min_dist(&self, min_dist: u16) -> usize {
        self.dist_grid
            .cells()
            .iter()
            .copied()
            .filter(|dist| (min_dist..u16::MAX).contains(dist))
            .count()
    }

    fn set_door(&mut self, a: IVec2, b: IVec2, value: bool) {
        let index: usize = self.index(a, b);

        self.doors.set(index, value);
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        delimited(
            tag("^"),
            map_opt(Self::parse_valid_sequence, |input: &'i str| {
                let mut position_stack: ProcessStack = ProcessStack::default();
                let mut initial_pos: IVec2 = IVec2::ZERO;
                let mut min: IVec2 = initial_pos;
                let mut max: IVec2 = initial_pos;

                position_stack
                    .process(
                        &mut |_, node| {
                            min = min.min(node.pos);
                            max = max.max(node.pos);
                        },
                        input,
                        initial_pos,
                    )
                    .ok()?;

                let dimensions: IVec2 = max - min + IVec2::ONE;

                // While we don't use `SmallPos` to store positions more compactly, it is a
                // convenient way of checking if we can store the distance in a `u16`. In theory
                // this also excludes us from using dimensions like (512x128), but I'm fine with
                // excluding those.
                SmallPos::are_dimensions_valid(dimensions).then_some(())?;

                let offset: IVec2 = -min;

                initial_pos += offset;

                let mut solution: Self = Self::new(max - min + IVec2::ONE, initial_pos);

                *solution.dist_grid.get_mut(solution.start).unwrap() = 0_u16;

                position_stack
                    .process(
                        &mut |prev_node, curr_node| {
                            solution.set_door(prev_node.pos, curr_node.pos, true);

                            let dist: &mut u16 = solution.dist_grid.get_mut(curr_node.pos).unwrap();

                            *dist = (*dist).min(curr_node.dist as u16);
                        },
                        input,
                        initial_pos,
                    )
                    .unwrap();
                solution.max_dist = solution
                    .dist_grid
                    .cells()
                    .iter()
                    .copied()
                    .filter(|&dist| dist < u16::MAX)
                    .max()
                    .unwrap() as usize;

                Some(solution)
            }),
            tag("$"),
        )(input)
    }
}

impl RunQuestions for Solution {
    /// Had to restructure things to not be recursive, which was muckied by keeping track of all the
    /// strings and what was expected vs not expected in the various scenarios.
    fn q1_internal(&mut self, args: &QuestionArgs) {
        dbg!(self.max_dist);

        if args.verbose {
            println!("{}", self.grid_string());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.rooms_with_min_dist(Self::MIN_DIST));
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
        "^WNE$",
        "^ENWWW(NEEE|SSE(EE|N))$",
        "^ENNWSWW(NEWS|)SSSEEN(WNSE|)EE(SWEN|)NNN$",
        "^ESSWWN(E|NNENN(EESS(WNSE|)SSS|WWWSSSSE(SW|NNNE)))$",
        "^WSSEESWWWNW(S|NENNEEEENN(ESSSSW(NWSW|SSEN)|WSWWN(E|WWS(E|SS))))$",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    start: (1_i32, 1_i32).into(),
                    max_dist: 3_usize,
                    doors: bitvec![1, 0, 1, 0, 1, 0, 0, 0,],
                    dist_grid: Grid2D::try_from_cells_and_dimensions(
                        vec![2, 3, 1, 0],
                        (2_i32, 2_i32).into(),
                    )
                    .unwrap(),
                },
                Solution {
                    start: (2_i32, 2_i32).into(),
                    max_dist: 10_usize,
                    doors: bitvec![
                        1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1,
                        1, 0, 0, 0, 0, 0, 0,
                    ],
                    dist_grid: Grid2D::try_from_cells_and_dimensions(
                        vec![6, 7, 8, 9, 5, 4, 3, 2, 6, 9, 0, 1, 7, 8, 9, 10],
                        (4_i32, 4_i32).into(),
                    )
                    .unwrap(),
                },
                Solution {
                    start: (2_i32, 2_i32).into(),
                    max_dist: 18_usize,
                    doors: bitvec![
                        1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                        1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                    ],
                    dist_grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            8, 9, 4, 3, 18, 7, 6, 5, 2, 17, 8, 15, 0, 1, 16, 9, 14, 13, 14, 15, 10,
                            11, 12, 17, 16,
                        ],
                        (5_i32, 5_i32).into(),
                    )
                    .unwrap(),
                },
                Solution {
                    start: (3_i32, 3_i32).into(),
                    max_dist: 23_usize,
                    doors: bitvec![
                        1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
                        0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                        1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                    ],
                    dist_grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            14, 13, 12, 11, 12, 13, 15, 22, 23, 10, 17, 14, 16, 21, 8, 9, 16, 15,
                            17, 20, 7, 0, 1, 16, 18, 19, 6, 7, 2, 17, 21, 20, 5, 4, 3, 18,
                        ],
                        (6_i32, 6_i32).into(),
                    )
                    .unwrap(),
                },
                Solution {
                    start: (3_i32, 3_i32).into(),
                    max_dist: 31_usize,
                    doors: bitvec![
                        1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                        1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                        0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                        1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    dist_grid: Grid2D::try_from_cells_and_dimensions(
                        vec![
                            28, 27, 26, 27, 22, 21, 22, 29, 30, 25, 24, 23, 20, 23, 30, 15, 16, 17,
                            18, 19, 24, 31, 14, 1, 0, 29, 28, 25, 12, 13, 2, 31, 30, 27, 26, 11,
                            10, 3, 4, 5, 28, 31, 12, 9, 8, 7, 6, 29, 30,
                        ],
                        (7_i32, 7_i32).into(),
                    )
                    .unwrap(),
                },
            ]
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
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
