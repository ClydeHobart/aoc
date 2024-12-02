use {
    crate::*,
    nom::{bytes::complete::tag, combinator::map, error::Error, sequence::tuple, Err, IResult},
    std::iter::repeat_with,
};

/* --- Day 9: Marble Mania ---

You talk to the Elves while you wait for your navigation system to initialize. To pass the time, they introduce you to their favorite marble game.

The Elves play this game by taking turns arranging the marbles in a circle according to very particular rules. The marbles are numbered starting with 0 and increasing by 1 until every marble has a number.

First, the marble numbered 0 is placed in the circle. At this point, while it contains only a single marble, it is still a circle: the marble is both clockwise from itself and counter-clockwise from itself. This marble is designated the current marble.

Then, each Elf takes a turn placing the lowest-numbered remaining marble into the circle between the marbles that are 1 and 2 marbles clockwise of the current marble. (When the circle is large enough, this means that there is one marble between the marble that was just placed and the current marble.) The marble that was just placed then becomes the current marble.

However, if the marble that is about to be placed has a number which is a multiple of 23, something entirely different happens. First, the current player keeps the marble they would have placed, adding it to their score. In addition, the marble 7 marbles counter-clockwise from the current marble is removed from the circle and also added to the current player's score. The marble located immediately clockwise of the marble that was removed becomes the new current marble.

For example, suppose there are 9 players. After the marble with value 0 is placed in the middle, each player (shown in square brackets) takes a turn. The result of each of those turns would produce circles of marbles like this, where clockwise is to the right and the resulting current marble is in parentheses:

[-] (0)
[1]  0 (1)
[2]  0 (2) 1
[3]  0  2  1 (3)
[4]  0 (4) 2  1  3
[5]  0  4  2 (5) 1  3
[6]  0  4  2  5  1 (6) 3
[7]  0  4  2  5  1  6  3 (7)
[8]  0 (8) 4  2  5  1  6  3  7
[9]  0  8  4 (9) 2  5  1  6  3  7
[1]  0  8  4  9  2(10) 5  1  6  3  7
[2]  0  8  4  9  2 10  5(11) 1  6  3  7
[3]  0  8  4  9  2 10  5 11  1(12) 6  3  7
[4]  0  8  4  9  2 10  5 11  1 12  6(13) 3  7
[5]  0  8  4  9  2 10  5 11  1 12  6 13  3(14) 7
[6]  0  8  4  9  2 10  5 11  1 12  6 13  3 14  7(15)
[7]  0(16) 8  4  9  2 10  5 11  1 12  6 13  3 14  7 15
[8]  0 16  8(17) 4  9  2 10  5 11  1 12  6 13  3 14  7 15
[9]  0 16  8 17  4(18) 9  2 10  5 11  1 12  6 13  3 14  7 15
[1]  0 16  8 17  4 18  9(19) 2 10  5 11  1 12  6 13  3 14  7 15
[2]  0 16  8 17  4 18  9 19  2(20)10  5 11  1 12  6 13  3 14  7 15
[3]  0 16  8 17  4 18  9 19  2 20 10(21) 5 11  1 12  6 13  3 14  7 15
[4]  0 16  8 17  4 18  9 19  2 20 10 21  5(22)11  1 12  6 13  3 14  7 15
[5]  0 16  8 17  4 18(19) 2 20 10 21  5 22 11  1 12  6 13  3 14  7 15
[6]  0 16  8 17  4 18 19  2(24)20 10 21  5 22 11  1 12  6 13  3 14  7 15
[7]  0 16  8 17  4 18 19  2 24 20(25)10 21  5 22 11  1 12  6 13  3 14  7 15

The goal is to be the player with the highest score after the last marble is used up. Assuming the example above ends after the marble numbered 25, the winning score is 23+9=32 (because player 5 kept marble 23 and removed marble 9, while no other player got any points in this very short example game).

Here are a few more examples:

    10 players; last marble is worth 1618 points: high score is 8317
    13 players; last marble is worth 7999 points: high score is 146373
    17 players; last marble is worth 1104 points: high score is 2764
    21 players; last marble is worth 6111 points: high score is 54718
    30 players; last marble is worth 5807 points: high score is 37305

What is the winning Elf's score?

--- Part Two ---

Amused by the speed of your answer, the Elves are curious:

What would the new winning Elf's score be if the number of the last marble were 100 times larger? */

type NodeIndexRaw = u32;
type NodeIndex = TableIndex<NodeIndexRaw>;

#[derive(Default)]
struct Node {
    marble: u32,
    prev_node_index: NodeIndex,
    next_node_index: NodeIndex,
}

struct MarbleGame {
    nodes: Vec<Node>,
    curr_node_index: NodeIndex,
    next_marble: u32,
    scores: Vec<u32>,
}

impl MarbleGame {
    const MARBLE_REMOVAL_MULTIPLE: u32 = 23_u32;

    fn new() -> Self {
        Self {
            nodes: vec![Node {
                marble: 0_u32,
                prev_node_index: 0_usize.into(),
                next_node_index: 0_usize.into(),
            }],
            curr_node_index: 0_usize.into(),
            next_marble: 1_u32,
            scores: Vec::new(),
        }
    }

    fn initialize(&mut self, players: u32) {
        self.scores.clear();
        self.scores.resize(players as usize, 0_u32);
    }

    fn iter_nodes(
        nodes: &[Node],
        curr_node_index: NodeIndex,
        clockwise: bool,
    ) -> impl Iterator<Item = NodeIndex> + '_ {
        let mut next_node_index: NodeIndex = curr_node_index;

        repeat_with(move || {
            let curr_node_index: NodeIndex = next_node_index;

            let node: &Node = &nodes[curr_node_index.get()];

            next_node_index = if clockwise {
                node.next_node_index
            } else {
                node.prev_node_index
            };

            curr_node_index
        })
    }

    fn traverse(&self, clockwise: bool, steps: u32) -> NodeIndex {
        Self::iter_nodes(&self.nodes, self.curr_node_index, clockwise)
            .skip(steps as usize)
            .next()
            .unwrap()
    }

    fn place_marble(&mut self) {
        if self.next_marble % Self::MARBLE_REMOVAL_MULTIPLE == 0_u32 {
            let player_index: usize = self.next_marble as usize % self.scores.len();

            self.scores[player_index] += self.next_marble;

            let removal_node_index: NodeIndex = self.traverse(false, 7_u32);
            let removal_node: &mut Node = &mut self.nodes[removal_node_index.get()];
            let prev_node_index: NodeIndex = removal_node.prev_node_index;
            let next_node_index: NodeIndex = removal_node.next_node_index;

            self.scores[player_index] += removal_node.marble;
            *removal_node = Node::default();

            self.nodes[prev_node_index.get()].next_node_index = next_node_index;
            self.nodes[next_node_index.get()].prev_node_index = prev_node_index;
            self.curr_node_index = next_node_index;
        } else {
            let prev_node_index: NodeIndex = self.traverse(true, 1_u32);
            let next_node_index: NodeIndex = self.nodes[prev_node_index.get()].next_node_index;
            let curr_node_index: NodeIndex = self.nodes.len().into();

            self.nodes[prev_node_index.get()].next_node_index = curr_node_index;
            self.nodes[next_node_index.get()].prev_node_index = curr_node_index;
            self.nodes.push(Node {
                marble: self.next_marble,
                prev_node_index,
                next_node_index,
            });
            self.curr_node_index = curr_node_index;
        }

        self.next_marble += 1_u32;
    }

    fn place_marbles(&mut self, last_marble: u32) {
        while self.next_marble <= last_marble {
            self.place_marble();
        }
    }

    fn try_high_score(&self) -> Option<u32> {
        self.scores.iter().map(|score| *score).max()
    }

    #[cfg(test)]
    fn iter_full_circle(&self) -> impl Iterator<Item = u32> + '_ {
        Self::iter_nodes(&self.nodes, 0_usize.into(), true)
            .take(
                self.nodes.len()
                    - ((self.next_marble - 1_u32) / Self::MARBLE_REMOVAL_MULTIPLE) as usize,
            )
            .map(|node_index| self.nodes[node_index.get()].marble)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    players: u32,
    last_marble: u32,
}

impl Solution {
    const EXTRA_HIGH_SCORE_MULTIPLIER: u32 = 100_u32;

    fn high_score(&self) -> u32 {
        let mut marble_game: MarbleGame = MarbleGame::new();

        marble_game.initialize(self.players);
        marble_game.place_marbles(self.last_marble);

        marble_game.try_high_score().unwrap()
    }

    fn extra_high_score(&self) -> u32 {
        let mut marble_game: MarbleGame = MarbleGame::new();

        marble_game.initialize(self.players);
        marble_game.place_marbles(self.last_marble * Self::EXTRA_HIGH_SCORE_MULTIPLIER);

        marble_game.try_high_score().unwrap()
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                parse_integer,
                tag(" players; last marble is worth "),
                parse_integer,
                tag(" points"),
            )),
            |(players, _, last_marble, _)| Self {
                players,
                last_marble,
            },
        )(input)
    }
}

impl RunQuestions for Solution {
    /// I feel like part 2 is just going to be "okay, now do it 40 million more times".
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.high_score());
    }

    /// Pretty much lol
    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.extra_high_score());
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
        "10 players; last marble is worth 1618 points",
        "13 players; last marble is worth 7999 points",
        "17 players; last marble is worth 1104 points",
        "21 players; last marble is worth 6111 points",
        "30 players; last marble is worth 5807 points",
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![
                Solution {
                    players: 10_u32,
                    last_marble: 1618_u32,
                },
                Solution {
                    players: 13_u32,
                    last_marble: 7999_u32,
                },
                Solution {
                    players: 17_u32,
                    last_marble: 1104_u32,
                },
                Solution {
                    players: 21_u32,
                    last_marble: 6111_u32,
                },
                Solution {
                    players: 30_u32,
                    last_marble: 5807_u32,
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
    fn test_place_marble_and_iter_full_circle() {
        let mut marble_game: MarbleGame = MarbleGame::new();

        marble_game.initialize(9_u32);

        for full_circle in [
            vec![0_u32, 1_u32],
            vec![0_u32, 2_u32, 1_u32],
            vec![0_u32, 2_u32, 1_u32, 3_u32],
            vec![0_u32, 4_u32, 2_u32, 1_u32, 3_u32],
            vec![0_u32, 4_u32, 2_u32, 5_u32, 1_u32, 3_u32],
            vec![0_u32, 4_u32, 2_u32, 5_u32, 1_u32, 6_u32, 3_u32],
            vec![0_u32, 4_u32, 2_u32, 5_u32, 1_u32, 6_u32, 3_u32, 7_u32],
            vec![
                0_u32, 8_u32, 4_u32, 2_u32, 5_u32, 1_u32, 6_u32, 3_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 5_u32, 1_u32, 6_u32, 3_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 1_u32, 6_u32, 3_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 6_u32, 3_u32,
                7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 12_u32, 6_u32,
                3_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 12_u32, 6_u32,
                13_u32, 3_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 12_u32, 6_u32,
                13_u32, 3_u32, 14_u32, 7_u32,
            ],
            vec![
                0_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 12_u32, 6_u32,
                13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32, 12_u32,
                6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32, 1_u32,
                12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 9_u32, 2_u32, 10_u32, 5_u32, 11_u32,
                1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 9_u32, 19_u32, 2_u32, 10_u32, 5_u32,
                11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 9_u32, 19_u32, 2_u32, 20_u32, 10_u32,
                5_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 9_u32, 19_u32, 2_u32, 20_u32, 10_u32,
                21_u32, 5_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 9_u32, 19_u32, 2_u32, 20_u32, 10_u32,
                21_u32, 5_u32, 22_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32,
                15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 19_u32, 2_u32, 20_u32, 10_u32, 21_u32,
                5_u32, 22_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32, 15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 19_u32, 2_u32, 24_u32, 20_u32, 10_u32,
                21_u32, 5_u32, 22_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32, 7_u32,
                15_u32,
            ],
            vec![
                0_u32, 16_u32, 8_u32, 17_u32, 4_u32, 18_u32, 19_u32, 2_u32, 24_u32, 20_u32, 25_u32,
                10_u32, 21_u32, 5_u32, 22_u32, 11_u32, 1_u32, 12_u32, 6_u32, 13_u32, 3_u32, 14_u32,
                7_u32, 15_u32,
            ],
        ] {
            marble_game.place_marble();

            assert_eq!(
                marble_game.iter_full_circle().collect::<Vec<u32>>(),
                full_circle
            );
        }
    }

    #[test]
    fn test_high_score() {
        for (index, high_score) in [8317_u32, 146373_u32, 2764_u32, 54718_u32, 37305_u32]
            .into_iter()
            .enumerate()
        {
            assert_eq!(solution(index).high_score(), high_score);
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
