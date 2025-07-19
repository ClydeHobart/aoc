use {
    crate::*,
    bitvec::prelude::*,
    glam::I8Vec4,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space0},
        combinator::{map, success, verify},
        error::Error,
        multi::separated_list1,
        sequence::preceded,
        Err, IResult,
    },
};

/* --- Day 25: Four-Dimensional Adventure ---

The reindeer's symptoms are getting worse, and neither you nor the white-bearded man have a solution. At least the reindeer has a warm place to rest: a small bed near where you're sitting.

As you reach down, the reindeer looks up at you, accidentally bumping a button on your wrist-mounted device with its nose in the process - a button labeled "help".

"Hello, and welcome to the Time Travel Support Hotline! If you are lost in time and space, press 1. If you are trapped in a time paradox, press 2. If you need help caring for a sick reindeer, press 3. If you--"

Beep.

A few seconds later, you hear a new voice. "Hello; please state the nature of your reindeer." You try to describe the situation.

"Just a moment, I think I can remotely run a diagnostic scan." A beam of light projects from the device and sweeps over the reindeer a few times.

"Okay, it looks like your reindeer is very low on magical energy; it should fully recover if we can fix that. Let me check your timeline for a source.... Got one. There's actually a powerful source of magical energy about 1000 years forward from you, and at roughly your position, too! It looks like... hot chocolate? Anyway, you should be able to travel there to pick some up; just don't forget a mug! Is there anything else I can help you with today?"

You explain that your device isn't capable of going forward in time. "I... see. That's tricky. Well, according to this information, your device should have the necessary hardware to open a small portal and send some hot chocolate back to you. You'll need a list of fixed points in spacetime; I'm transmitting it to you now."

"You just need to align your device to the constellations of fixed points so that it can lock on to the destination and open the portal. Let me look up how much hot chocolate that breed of reindeer needs."

"It says here that your particular reindeer is-- this can't be right, it says there's only one like that in the universe! But THAT means that you're--" You disconnect the call.

The list of fixed points in spacetime (your puzzle input) is a set of four-dimensional coordinates. To align your device, acquire the hot chocolate, and save the reindeer, you just need to find the number of constellations of points in the list.

Two points are in the same constellation if their manhattan distance apart is no more than 3 or if they can form a chain of points, each a manhattan distance no more than 3 from the last, between the two of them. (That is, if a point is close enough to a constellation, it "joins" that constellation.) For example:

 0,0,0,0
 3,0,0,0
 0,3,0,0
 0,0,3,0
 0,0,0,3
 0,0,0,6
 9,0,0,0
12,0,0,0

In the above list, the first six points form a single constellation: 0,0,0,0 is exactly distance 3 from the next four, and the point at 0,0,0,6 is connected to the others by being 3 away from 0,0,0,3, which is already in the constellation. The bottom two points, 9,0,0,0 and 12,0,0,0 are in a separate constellation because no point is close enough to connect them to the first constellation. So, in the above list, the number of constellations is 2. (If a point at 6,0,0,0 were present, it would connect 3,0,0,0 and 9,0,0,0, merging all of the points into a single giant constellation instead.)

In this example, the number of constellations is 4:

-1,2,2,0
0,0,2,-2
0,0,0,-2
-1,2,0,0
-2,-2,-2,2
3,0,2,-1
-1,3,2,2
-1,0,-1,0
0,2,1,-2
3,0,0,0

In this one, it's 3:

1,-1,0,1
2,0,-1,0
3,2,-1,0
0,0,3,1
0,0,-1,-1
2,3,-2,0
-2,2,0,0
2,-2,0,-1
1,-1,0,-1
3,2,0,2

Finally, in this one, it's 8:

1,-1,-1,-2
-2,-2,0,1
0,2,1,3
-2,3,-2,1
0,2,3,-2
-1,-1,1,-2
0,-2,-1,0
-2,2,3,-1
1,2,2,0
-1,-2,0,-2

The portly man nervously strokes his white beard. It's time to get that hot chocolate.

How many constellations are formed by the fixed points in spacetime? */

struct ConstellationFinder<'s> {
    solution: &'s Solution,
    neighbors: BitVec,
    constellations: Vec<BitVec>,
    start: usize,
    visited_points: BitVec,
}

impl<'s> BreadthFirstSearch for ConstellationFinder<'s> {
    type Vertex = usize;

    fn start(&self) -> &Self::Vertex {
        &self.start
    }

    fn is_end(&self, _vertex: &Self::Vertex) -> bool {
        false
    }

    fn path_to(&self, _vertex: &Self::Vertex) -> Vec<Self::Vertex> {
        unreachable!()
    }

    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>) {
        neighbors.clear();
        neighbors.extend(
            self.solution
                .neighbors_for_point(&self.neighbors, *vertex)
                .iter_ones()
                .filter(|&point_index| !self.visited_points[point_index]),
        );
    }

    fn update_parent(&mut self, _from: &Self::Vertex, &to: &Self::Vertex) {
        self.constellations.last_mut().unwrap().set(to, true);
        self.visited_points.set(to, true);
    }

    fn reset(&mut self) {
        self.constellations
            .push(bitvec![0; self.solution.points_len()]);

        let start: usize = self.start;

        self.update_parent(&start, &start);
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<I8Vec4>);

impl Solution {
    const MAX_CONSTELLATION_MANHATTAN_DISTANCE: i8 = 3_i8;

    fn points(&self) -> &[I8Vec4] {
        &self.0
    }

    fn points_len(&self) -> usize {
        self.points().len()
    }

    fn populate_neighbors_for_point<T: BitStore>(
        &self,
        point: I8Vec4,
        neighbors: &mut BitSlice<T>,
    ) {
        for (neighbor_point, mut is_neighbor) in
            self.points().iter().copied().zip(neighbors.iter_mut())
        {
            is_neighbor.set(
                neighbor_point != point
                    && point.manhattan_distance(neighbor_point)
                        <= Self::MAX_CONSTELLATION_MANHATTAN_DISTANCE,
            );
        }
    }

    fn neighbors(&self) -> BitVec {
        let points_len: usize = self.points_len();

        let mut neighbors: BitVec = bitvec![0; points_len * points_len];

        for (point, neighbors_for_point) in self
            .points()
            .iter()
            .copied()
            .zip(neighbors.chunks_exact_mut(points_len))
        {
            self.populate_neighbors_for_point(point, neighbors_for_point);
        }

        neighbors
    }

    fn neighbors_for_point<'n, T: BitStore>(
        &self,
        neighbors: &'n BitSlice<T>,
        point_index: usize,
    ) -> &'n BitSlice<T> {
        let points_len: usize = self.points_len();
        let neighbors_start: usize = point_index * points_len;
        let neighbors_end: usize = neighbors_start + points_len;

        &neighbors[neighbors_start..neighbors_end]
    }

    fn constellation_finder(&self) -> ConstellationFinder {
        ConstellationFinder {
            solution: self,
            neighbors: self.neighbors(),
            constellations: Vec::new(),
            start: 0_usize,
            visited_points: bitvec![0; self.points_len()],
        }
    }

    fn constellations(&self) -> Vec<BitVec> {
        let mut constellation_finder: ConstellationFinder = self.constellation_finder();
        let mut state: BreadthFirstSearchState<usize> = BreadthFirstSearchState::default();

        let points_len: usize = self.points_len();

        while constellation_finder.start < points_len {
            constellation_finder.run_internal(&mut state);
            constellation_finder.start = constellation_finder.visited_points.leading_ones();
        }

        constellation_finder.constellations
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        let (input, mut points): (&str, Vec<I8Vec4>) = separated_list1(
            line_ending,
            preceded(
                space0,
                map(
                    parse_separated_array(parse_integer, tag(",")),
                    I8Vec4::from_array,
                ),
            ),
        )(input)?;

        let old_points_len: usize = points.len();

        points.sort_by_key(|point| sortable_index_from_pos_4d(*point));
        points.dedup();

        let new_points_len: usize = points.len();

        verify(success(()), |_| new_points_len == old_points_len)("duplicate points were present")?;

        Ok((input, Self(points)))
    }
}

impl RunQuestions for Solution {
    /// The most time-consuming component was just setting up the test cases.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.constellations().len());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {}
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
        " \
        0,0,0,0\n \
        3,0,0,0\n \
        0,3,0,0\n \
        0,0,3,0\n \
        0,0,0,3\n \
        0,0,0,6\n \
        9,0,0,0\n\
        12,0,0,0\n",
        "\
        -1,2,2,0\n\
        0,0,2,-2\n\
        0,0,0,-2\n\
        -1,2,0,0\n\
        -2,-2,-2,2\n\
        3,0,2,-1\n\
        -1,3,2,2\n\
        -1,0,-1,0\n\
        0,2,1,-2\n\
        3,0,0,0\n",
        "\
        1,-1,0,1\n\
        2,0,-1,0\n\
        3,2,-1,0\n\
        0,0,3,1\n\
        0,0,-1,-1\n\
        2,3,-2,0\n\
        -2,2,0,0\n\
        2,-2,0,-1\n\
        1,-1,0,-1\n\
        3,2,0,2\n",
        "\
        1,-1,-1,-2\n\
        -2,-2,0,1\n\
        0,2,1,3\n\
        -2,3,-2,1\n\
        0,2,3,-2\n\
        -1,-1,1,-2\n\
        0,-2,-1,0\n\
        -2,2,3,-1\n\
        1,2,2,0\n\
        -1,-2,0,-2\n",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution(vec![
                    (0, 0, 0, 0).into(),
                    (3, 0, 0, 0).into(),
                    (9, 0, 0, 0).into(),
                    (12, 0, 0, 0).into(),
                    (0, 3, 0, 0).into(),
                    (0, 0, 3, 0).into(),
                    (0, 0, 0, 3).into(),
                    (0, 0, 0, 6).into(),
                ]),
                Solution(vec![
                    (0, 0, 0, -2).into(),
                    (0, 2, 1, -2).into(),
                    (0, 0, 2, -2).into(),
                    (3, 0, 2, -1).into(),
                    (-1, 0, -1, 0).into(),
                    (3, 0, 0, 0).into(),
                    (-1, 2, 0, 0).into(),
                    (-1, 2, 2, 0).into(),
                    (-2, -2, -2, 2).into(),
                    (-1, 3, 2, 2).into(),
                ]),
                Solution(vec![
                    (0, 0, -1, -1).into(),
                    (2, -2, 0, -1).into(),
                    (1, -1, 0, -1).into(),
                    (2, 3, -2, 0).into(),
                    (2, 0, -1, 0).into(),
                    (3, 2, -1, 0).into(),
                    (-2, 2, 0, 0).into(),
                    (1, -1, 0, 1).into(),
                    (0, 0, 3, 1).into(),
                    (3, 2, 0, 2).into(),
                ]),
                Solution(vec![
                    (1, -1, -1, -2).into(),
                    (-1, -2, 0, -2).into(),
                    (-1, -1, 1, -2).into(),
                    (0, 2, 3, -2).into(),
                    (-2, 2, 3, -1).into(),
                    (0, -2, -1, 0).into(),
                    (1, 2, 2, 0).into(),
                    (-2, 3, -2, 1).into(),
                    (-2, -2, 0, 1).into(),
                    (0, 2, 1, 3).into(),
                ]),
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
    fn test_neighbors() {
        for (index, neighbors) in [
            [
                bitvec![0, 1, 0, 0, 1, 1, 1, 0],
                bitvec![1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 1, 0, 0, 0, 0],
                bitvec![0, 0, 1, 0, 0, 0, 0, 0],
                bitvec![1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![1, 0, 0, 0, 0, 0, 0, 1],
                bitvec![0, 0, 0, 0, 0, 0, 1, 0],
            ]
            .into_iter()
            .flat_map(BitVec::into_iter)
            .collect::<BitVec>(),
            [
                bitvec![0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
            .into_iter()
            .flat_map(BitVec::into_iter)
            .collect::<BitVec>(),
            [
                bitvec![0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                bitvec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                bitvec![0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
            .into_iter()
            .flat_map(BitVec::into_iter)
            .collect::<BitVec>(),
            [
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            .into_iter()
            .flat_map(BitVec::into_iter)
            .collect::<BitVec>(),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).neighbors(), neighbors);
        }
    }

    #[test]
    fn test_constellations() {
        for (index, constellations) in [
            vec![
                bitvec![1, 1, 0, 0, 1, 1, 1, 1],
                bitvec![0, 0, 1, 1, 0, 0, 0, 0],
            ],
            vec![
                bitvec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            vec![
                bitvec![1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            vec![
                bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(solution(index).constellations(), constellations);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
