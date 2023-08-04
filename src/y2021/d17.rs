use {
    crate::*,
    glam::IVec2,
    nom::{
        bytes::complete::tag,
        combinator::map,
        error::Error,
        sequence::{preceded, separated_pair, tuple},
        Err, IResult,
    },
    std::{collections::HashSet, ops::Range},
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct XVelocityData {
    velocities: Vec<Option<i32>>,
    x_delta_ends: Vec<usize>,
}

impl XVelocityData {
    fn push(&mut self, x_delta: i32, time: i32, velocity: i32) {
        let x_delta: usize = x_delta as usize;

        while self.x_delta_ends.get(x_delta).is_none() {
            self.x_delta_ends.push(self.velocities.len());
        }

        let velocity_index: usize = self.x_delta_start(x_delta) + time as usize;

        while self.velocities.get(velocity_index).is_none() {
            self.velocities.push(None);
        }

        *self.velocities.get_mut(velocity_index).unwrap() = Some(velocity);
        *self.x_delta_ends.get_mut(x_delta).unwrap() = self.velocities.len();
    }

    fn iter_velocities(&self, x_delta: i32, time: i32) -> impl Iterator<Item = i32> {
        self.velocity(x_delta, time).copied().into_iter()
    }

    fn iter_velocities_unbounded(&self, x_delta: i32, time: i32) -> impl Iterator<Item = i32> + '_ {
        (0_i32..=time)
            .into_iter()
            .flat_map(move |time| self.iter_velocities(x_delta, time))
    }

    fn velocity(&self, x_delta: i32, time: i32) -> Option<&i32> {
        self.velocities(x_delta).and_then(|velocities| {
            let time: usize = time as usize;

            velocities.get(time as usize).and_then(Option::as_ref)
        })
    }

    fn velocities(&self, x_delta: i32) -> Option<&[Option<i32>]> {
        let x_delta: usize = x_delta as usize;

        self.x_delta_ends.get(x_delta).map(|x_delta_end| {
            let x_delta_start: usize = self.x_delta_start(x_delta);

            &self.velocities[x_delta_start..*x_delta_end]
        })
    }

    fn x_delta_start(&self, x_delta: usize) -> usize {
        x_delta
            .checked_sub(1_usize)
            .map_or(0_usize, |prev_x_delta| self.x_delta_ends[prev_x_delta])
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
struct XVelocityDataPair {
    bounded: XVelocityData,
    unbounded: XVelocityData,
}

impl XVelocityDataPair {
    fn iter_velocities(&self, x_delta: i32, time: i32) -> impl Iterator<Item = i32> + '_ {
        self.bounded
            .iter_velocities(x_delta, time)
            .chain(self.unbounded.iter_velocities_unbounded(x_delta, time))
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    x_bounds: Range<i32>,
    y_bounds: Range<i32>,
}

impl Solution {
    fn compute_highest_trajectory_y(&self) -> i32 {
        self.compute_trajectories()
            .into_iter()
            .map(|initial_velocity| {
                if initial_velocity.y <= 0_i32 {
                    0_i32
                } else {
                    Self::triangle_number(initial_velocity.y)
                }
            })
            .max()
            .unwrap_or_default()
    }

    fn count_trajectories(&self) -> usize {
        self.compute_trajectories().len()
    }

    fn compute_trajectories(&self) -> HashSet<IVec2> {
        let x_velocity_data_pair: XVelocityDataPair = self.compute_x_velocity_data_pair();
        let mut trajectories: HashSet<IVec2> = HashSet::new();

        for y_pos in self.y_bounds.clone() {
            let mut time: i32 = 1_i32;
            let should_loop = if y_pos == 0_i32 {
                |time: i32, _y_pos: i32| time <= 1_i32
            } else {
                |time: i32, y_pos: i32| time <= 2_i32 * y_pos.abs()
            };

            while should_loop(time, y_pos) {
                if let Some(y_velocity) = self.try_compute_y_velocity(y_pos, time) {
                    for x_delta in 0_i32..self.x_bounds.len() as i32 {
                        for x_velocity in x_velocity_data_pair.iter_velocities(x_delta, time) {
                            trajectories.insert(IVec2::new(x_velocity, y_velocity));
                        }
                    }
                }

                time += 1_i32;
            }
        }

        trajectories
    }

    fn compute_x_velocity_data_pair(&self) -> XVelocityDataPair {
        let mut x_velocity_data_pair: XVelocityDataPair = XVelocityDataPair::default();

        for x_pos in self.x_bounds.clone() {
            let x_pos_abs: i32 = x_pos.abs();
            let mut time: i32 = 1_i32;

            while Self::triangle_number(time) <= x_pos_abs {
                self.try_push_x_velocity(x_pos, time, &mut x_velocity_data_pair);

                time += 1_i32;
            }
        }

        x_velocity_data_pair
    }

    fn try_push_x_velocity(
        &self,
        x_pos: i32,
        time: i32,
        x_velocity_data_pair: &mut XVelocityDataPair,
    ) {
        // The distance a specific velocity will travel is the difference of two triangle numbers.
        // At time 1, the probe will have traveled distance `|v|`. At time 2, the probe will have
        // traveled distance `|v| + |v| - 1`. At time `|v|`, the probe will have traveled
        // ```
        // sum_{i = 0}^{|v| - 1}(|v| - i) ==
        // |v| * |v| - sum_{i = 0}^{|v| - 1}(i) ==
        // |v| * |v| - sum_{i = 1}^{|v| - 1}(i) ==
        // |v| * |v| - |v| * (|v| - 1) / 2 == // Here, I'm substituting for the formula for a
        //                                    // triangle number, `T_{|v| - 1}`
        // |v| * (2 * |v| - |v| + 1) / 2 ==
        // |v| * (|v| + 1) / 2 ==
        // `T_{|v|}`
        // ```
        // After time `|v|`, the traveled distance will remain `T_{|V|}`. At time `t`, where
        // `t <= |v|`, we can construct the following formula for the traveled distance:
        // ```
        // T_{|v|} - T_{|v| - t} ==
        // |v| * (|v| + 1) / 2 - (|v| - t) * (|v| - t + 1) / 2 ==
        // ((|v| ^ 2 + |v|) - (|v| ^ 2 - 2 * |v| * t + |v| + t ^ 2 - t)) / 2 ==
        // (2 * |v| * t - t ^ 2 + t) / 2
        // ```
        // In this case, `|v|` is unknown, but `(2 * |v| * t - t ^ 2 + t) / 2` needs to equal
        // `|x_pos|`. Solving for this:
        // ```
        // (2 * |v| * t - t ^ 2 + t) / 2 == |x_pos|
        // 2 * |v| * t == 2 * |x_pos| + t ^ 2 - t
        // |v| == (2 * |x_pos| + t ^ 2 - t) / (2 * t)
        // ```
        // Because of the discreet nature of this problem, there is only a valid velocity if the
        // right expression is an integer.
        let speed: f32 =
            (2_i32 * x_pos.abs() + (time * time) - time) as f32 / (2_i32 * time) as f32;
        let speed_round: f32 = speed.round();

        if speed == 0.0_f32 || ((speed - speed_round) / speed).abs() < f32::EPSILON {
            let speed: i32 = speed_round as i32;
            let x_velocity_data: &mut XVelocityData = if speed == time {
                &mut x_velocity_data_pair.unbounded
            } else {
                &mut x_velocity_data_pair.bounded
            };
            let x_delta: i32 = x_pos - self.x_bounds.start;
            let velocity: i32 = x_pos.signum() * speed;

            x_velocity_data.push(x_delta, time, velocity);
        }
    }

    fn try_compute_y_velocity(&self, y_pos: i32, time: i32) -> Option<i32> {
        // At time 1, the y position will be `v`. At time 2, the y position will be `v + (v - 1)`.
        // At time 3, the y position will be `v + (v - 1) + (v - 2)`. At time `t`, the y position
        // will be `t * v - sum_{i = 0}^{t - 1} == t * v - T_{t - 1}`, where `T_{t - 1}` is the
        // triangle number of `t - 1`. In this case, `v` is unknown, but `t * v - T_{t - 1}` needs
        // to equal `y_pos`. Solving for this:
        // ```
        // t * v - T_{t - 1} == y_pos
        // v == (y_pos + T_{t - 1}) / t
        // ```
        // Because of the discreet nature of this problem, there is only a valid velocity if the
        // right expression is an integer.
        let velocity: f32 = (y_pos + Self::triangle_number(time - 1_i32)) as f32 / time as f32;
        let velocity_round: f32 = velocity.round();

        if velocity == 0.0_f32 || ((velocity - velocity_round) / velocity).abs() < f32::EPSILON {
            Some(velocity_round as i32)
        } else {
            None
        }
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        preceded(
            tag("target area: "),
            map(
                separated_pair(
                    |input| Self::parse_1d_bounds(input, "x"),
                    tag(", "),
                    |input| Self::parse_1d_bounds(input, "y"),
                ),
                |(x_bounds, y_bounds)| Self { x_bounds, y_bounds },
            ),
        )(input)
    }

    fn parse_1d_bounds<'i>(input: &'i str, bound: &str) -> IResult<&'i str, Range<i32>> {
        preceded(
            tuple((tag(bound), tag("="))),
            map(
                separated_pair(
                    parse_integer::<i32>,
                    tag(".."),
                    map(parse_integer::<i32>, |end: i32| end + 1_i32),
                ),
                |(start, end)| start..end,
            ),
        )(input)
    }

    fn triangle_number(n: i32) -> i32 {
        n * (n + 1_i32) / 2_i32
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.compute_highest_trajectory_y());
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_trajectories());
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
        nom::{
            character::complete::multispace1, combinator::opt, multi::many0, sequence::terminated,
        },
        std::sync::OnceLock,
    };

    const SOLUTION_STR: &str = "target area: x=20..30, y=-10..-5";
    const TRAJECTORIES_STR: &str = "\
        23,-10  25,-9   27,-5   29,-6   22,-6   21,-7   9,0     27,-7   24,-5\n\
        25,-7   26,-6   25,-5   6,8     11,-2   20,-5   29,-10  6,3     28,-7\n\
        8,0     30,-6   29,-8   20,-10  6,7     6,4     6,1     14,-4   21,-6\n\
        26,-10  7,-1    7,7     8,-1    21,-9   6,2     20,-7   30,-10  14,-3\n\
        20,-8   13,-2   7,3     28,-8   29,-9   15,-3   22,-5   26,-8   25,-8\n\
        25,-6   15,-4   9,-2    15,-2   12,-2   28,-9   12,-3   24,-6   23,-7\n\
        25,-10  7,8     11,-3   26,-7   7,1     23,-9   6,0     22,-10  27,-6\n\
        8,1     22,-8   13,-4   7,6     28,-6   11,-4   12,-4   26,-9   7,4\n\
        24,-10  23,-8   30,-8   7,0     9,-1    10,-1   26,-5   22,-9   6,5\n\
        7,5     23,-6   28,-10  10,-2   11,-1   20,-9   14,-2   29,-7   13,-3\n\
        23,-5   24,-8   27,-9   30,-7   28,-5   21,-10  7,9     6,6     21,-5\n\
        27,-10  7,2     30,-9   21,-8   22,-7   24,-9   20,-6   6,9     29,-5\n\
        8,-2    27,-8   30,-5   24,-7";

    fn solution() -> &'static Solution {
        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| Solution {
            x_bounds: 20_i32..31_i32,
            y_bounds: -10..-4_i32,
        })
    }

    fn trajectories() -> &'static Vec<IVec2> {
        static ONCE_LOCK: OnceLock<Vec<IVec2>> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| sort_trajectories(parse_trajectories(TRAJECTORIES_STR).unwrap().1))
    }

    fn parse_trajectory<'i>(input: &'i str) -> IResult<&'i str, IVec2> {
        map(
            separated_pair(parse_integer::<i32>, tag(","), parse_integer::<i32>),
            |(x, y)| IVec2::new(x, y),
        )(input)
    }

    fn parse_trajectories<'i>(input: &'i str) -> IResult<&'i str, Vec<IVec2>> {
        many0(terminated(parse_trajectory, opt(multispace1)))(input)
    }

    fn sort_trajectories(mut trajectories: Vec<IVec2>) -> Vec<IVec2> {
        trajectories.sort_by(|a, b| a.x.cmp(&b.x).then_with(|| a.y.cmp(&b.y)));

        trajectories
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_compute_x_velocity_data_pair() {
        use Option::{None as N, Some as S};

        assert_eq!(
            Solution {
                x_bounds: 2..7,
                y_bounds: -6..-1
            }
            .compute_x_velocity_data_pair(),
            XVelocityDataPair {
                bounded: XVelocityData {
                    velocities: vec![N, S(2), N, S(3), N, S(4), N, S(5), S(3), N, S(6),],
                    x_delta_ends: vec![2, 4, 6, 9, 11]
                },
                unbounded: XVelocityData {
                    velocities: vec![N, N, S(2), N, N, N, S(3)],
                    x_delta_ends: vec![0, 3, 3, 3, 7]
                }
            }
        );
    }

    #[test]
    fn test_compute_trajectories() {
        assert_eq!(
            &sort_trajectories(solution().compute_trajectories().into_iter().collect()),
            trajectories()
        );
    }
}
