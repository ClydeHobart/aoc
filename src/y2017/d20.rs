use {
    crate::*,
    bitvec::prelude::*,
    glam::IVec3,
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space0},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    num::integer::Roots,
    std::{cmp::Ordering, collections::BTreeMap, fmt::Debug},
};

/* --- Day 20: Particle Swarm ---

Suddenly, the GPU contacts you, asking for help. Someone has asked it to simulate too many particles, and it won't be able to finish them all in time to render the next frame at this rate.

It transmits to you a buffer (your puzzle input) listing each particle in order (starting with particle 0, then particle 1, particle 2, and so on). For each particle, it provides the X, Y, and Z coordinates for the particle's position (p), velocity (v), and acceleration (a), each in the format <X,Y,Z>.

Each tick, all particles are updated simultaneously. A particle's properties are updated in the following order:

    Increase the X velocity by the X acceleration.
    Increase the Y velocity by the Y acceleration.
    Increase the Z velocity by the Z acceleration.
    Increase the X position by the X velocity.
    Increase the Y position by the Y velocity.
    Increase the Z position by the Z velocity.

Because of seemingly tenuous rationale involving z-buffering, the GPU would like to know which particle will stay closest to position <0,0,0> in the long term. Measure this using the Manhattan distance, which in this situation is simply the sum of the absolute values of a particle's X, Y, and Z position.

For example, suppose you are only given two particles, both of which stay entirely on the X-axis (for simplicity). Drawing the current states of particles 0 and 1 (in that order) with an adjacent a number line and diagram of current X positions (marked in parentheses), the following would take place:

p=< 3,0,0>, v=< 2,0,0>, a=<-1,0,0>    -4 -3 -2 -1  0  1  2  3  4
p=< 4,0,0>, v=< 0,0,0>, a=<-2,0,0>                         (0)(1)

p=< 4,0,0>, v=< 1,0,0>, a=<-1,0,0>    -4 -3 -2 -1  0  1  2  3  4
p=< 2,0,0>, v=<-2,0,0>, a=<-2,0,0>                      (1)   (0)

p=< 4,0,0>, v=< 0,0,0>, a=<-1,0,0>    -4 -3 -2 -1  0  1  2  3  4
p=<-2,0,0>, v=<-4,0,0>, a=<-2,0,0>          (1)               (0)

p=< 3,0,0>, v=<-1,0,0>, a=<-1,0,0>    -4 -3 -2 -1  0  1  2  3  4
p=<-8,0,0>, v=<-6,0,0>, a=<-2,0,0>                         (0)

At this point, particle 1 will never be closer to <0,0,0> than particle 0, and so, in the long run, particle 0 will stay closest.

Which particle will stay closest to position <0,0,0> in the long term?

--- Part Two ---

To simplify the problem further, the GPU would like to remove any particles that collide. Particles collide if their positions ever exactly match. Because particles are updated simultaneously, more than two particles can collide at the same time and place. Once particles collide, they are removed and cannot collide with anything else after that tick.

For example:

p=<-6,0,0>, v=< 3,0,0>, a=< 0,0,0>
p=<-4,0,0>, v=< 2,0,0>, a=< 0,0,0>    -6 -5 -4 -3 -2 -1  0  1  2  3
p=<-2,0,0>, v=< 1,0,0>, a=< 0,0,0>    (0)   (1)   (2)            (3)
p=< 3,0,0>, v=<-1,0,0>, a=< 0,0,0>

p=<-3,0,0>, v=< 3,0,0>, a=< 0,0,0>
p=<-2,0,0>, v=< 2,0,0>, a=< 0,0,0>    -6 -5 -4 -3 -2 -1  0  1  2  3
p=<-1,0,0>, v=< 1,0,0>, a=< 0,0,0>             (0)(1)(2)      (3)
p=< 2,0,0>, v=<-1,0,0>, a=< 0,0,0>

p=< 0,0,0>, v=< 3,0,0>, a=< 0,0,0>
p=< 0,0,0>, v=< 2,0,0>, a=< 0,0,0>    -6 -5 -4 -3 -2 -1  0  1  2  3
p=< 0,0,0>, v=< 1,0,0>, a=< 0,0,0>                       X (3)
p=< 1,0,0>, v=<-1,0,0>, a=< 0,0,0>

------destroyed by collision------
------destroyed by collision------    -6 -5 -4 -3 -2 -1  0  1  2  3
------destroyed by collision------                      (3)
p=< 0,0,0>, v=<-1,0,0>, a=< 0,0,0>

In this example, particles 0, 1, and 2 are simultaneously destroyed at the time and place marked X. On the next tick, particle 3 passes through unharmed.

How many particles are left after all collisions are resolved? */

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Collision {
    time: i32,
    pos: IVec3,
}

impl Collision {
    fn new(time: i32, pos: IVec3) -> Self {
        Self { time, pos }
    }
}

impl Ord for Collision {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .cmp(&other.time)
            .then_with(|| self.pos.to_array().cmp(&other.pos.to_array()))
    }
}

impl PartialOrd for Collision {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A representation of a particle in 3D space.
///
/// Here, kinematics equations are slightly different due to integer discretization:
/// * Acceleration: `a` (constant)
/// * Velocity: `v(t) = v_0 + a * t` (linear function of time)
/// * Position: `p(t) = p_0 + v_0 * t + p_a(t)`, where `p_a(t)` is the acceleration's contribution
/// to the position. Using
/// `v_a(t) = a * t` as an intermediate quantity, we can define `p_a(t)` as
/// `/sum_{i=0}^t(v_a(i))`. Let's examine this quantity in the early ticks, :
///
/// t | v_a(t) | p_a(t)
/// --+--------+-------
/// 0 | 0      | 0
/// 1 | a      | a
/// 2 | 2 * a  | (1 + 2) * a = 3 * a
/// 3 | 3 * a  | (1 + 2 + 3) * a = 6 * a
/// 4 | 4 * a  | (1 + 2 + 3 + 4) * a = 10 * a
///
/// At this point, the pattern is clear: `p_a(t)` is `triangular_number(t) * a`, which can be
/// simplified as `t * (t + 1) * a / 2`. This yields the final position equation:
///
/// `p(t) = p_0 + v_0 * t + t * (t + 1) * a / 2`
///
/// Rearranging this as a quadratic function `p(t) = l * t ^ 2 + m * t + n`, we get constants:
/// * `l = a / 2`
/// * `m = v_0 + a / 2`
/// * `n = p_0`
#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone)]
struct Particle {
    pos: IVec3,
    vel: IVec3,
    acc: IVec3,
}

impl Particle {
    fn parse_component<'i>(
        tag_str: &'static str,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, IVec3> {
        map(
            tuple((
                tag(tag_str),
                tag("=<"),
                space0,
                parse_integer,
                tag(","),
                parse_integer,
                tag(","),
                parse_integer,
                tag(">"),
            )),
            |(_, _, _, x, _, y, _, z, _)| IVec3::new(x, y, z),
        )
    }

    /// To compute the collision, it's easiest to use `p(t) = p_0 + v_0 * t + t * (t + 1) * a / 2`,
    /// since `t * (t + 1)` guarantees that `t * (t + 1) * a / 2` is an integer.
    fn pos_at_time(&self, time: i32) -> IVec3 {
        self.pos + self.vel * time + time * (time + 1_i32) * self.acc / 2_i32
    }

    fn try_compute_collision_time_for_numerator_and_denominator(
        numerator: i32,
        denominator: i32,
    ) -> Option<i32> {
        (denominator != 0_i32).then_some(())?;

        // find time, verify is non-negative
        (numerator % denominator == 0_i32).then_some(())?;

        let time: i32 = numerator / denominator;

        (time >= 0_i32).then_some(time)
    }

    fn try_compute_collision_time_candidates_for_axis(
        &self,
        other: &Self,
        axis: usize,
    ) -> Option<[Option<i32>; 2_usize]> {
        let diff_pos: i32 = self.pos.to_array()[axis] - other.pos.to_array()[axis];
        let diff_vel: i32 = self.vel.to_array()[axis] - other.vel.to_array()[axis];
        let diff_acc: i32 = self.acc.to_array()[axis] - other.acc.to_array()[axis];

        // If the acceleration is zero, compute the time via linear, not quadratic.
        if diff_acc == 0_i32 {
            Some([
                Self::try_compute_collision_time_for_numerator_and_denominator(-diff_pos, diff_vel),
                None,
            ])
        } else {
            let non_sqrt_numerator: i32 = -2_i32 * diff_vel - diff_acc;
            let sq: i32 = non_sqrt_numerator * non_sqrt_numerator - 8_i32 * diff_acc * diff_pos;

            (sq >= 0_i32).then_some(())?;

            let sqrt: i32 = sq.sqrt();

            (sqrt * sqrt == sq).then_some(())?;

            let denominator: i32 = 2_i32 * diff_acc;

            Some([
                Self::try_compute_collision_time_for_numerator_and_denominator(
                    non_sqrt_numerator + sqrt,
                    denominator,
                ),
                Self::try_compute_collision_time_for_numerator_and_denominator(
                    non_sqrt_numerator - sqrt,
                    denominator,
                ),
            ])
        }
    }

    /// Computes the collision between two particles, if there is one. Collisions can only happen in
    /// the present or the future.
    ///
    /// Consider the function for position:
    ///
    /// `p(t) = (a / 2) * t ^ 2 + ((2 * v_0 + a) / 2) * t + p_0`
    ///
    /// To find when two particles collide (in a single axis), set two copies of this equal to each
    /// other, but with distinct constants:
    ///
    /// ```ignore
    /// (a_1 / 2) * t ^ 2 + ((2 * v_1 + a_1) / 2) * t + p_1 =
    ///     (a_2 / 2) * t ^ 2 + ((2 * v_2 + a_2) / 2) * t + p_1
    /// ```
    ///
    /// Rearranging this as a quadratic function `d(t) = l * t ^ 2 + m * t + n`, we get constants:
    /// * `l = (a_1 - a_2) / 2`
    /// * `m = (2 * (v_1 - v_2) + a_1 - a_2) / 2`
    /// * `n = p_1 - p_2`
    ///
    /// Defining `d_a = a_1 - a_2`, `d_v = v_1 - v_2`, and `d_p = p_1 - p_2`, this becomes
    /// * `l = d_a / 2`
    /// * `m = (2 * d_v + d_a) / 2`
    /// * `n = d_p`
    ///
    /// If `d_a` is zero:
    ///
    /// This simplifies to `d(t) = d_v * t + d_p`, which yields a collision at time `-d_p / d_v`
    /// (assuming `d_v` isn't zero).
    ///
    /// If `d_a` isn't zero:
    ///
    /// This yields a collision at time
    ///
    /// `(-m +- sqrt(m ^ 2 - 4 * l * n)) / (2 * l)`
    ///
    /// This simplifies to
    ///
    /// `(-(2 * d_v + d_a) / 2 +- sqrt(((2 * d_v + d_a) / 2) ^ 2 - 2 * d_a * d_p)) / d_a`
    ///
    /// The division in the computation of `m` could prove to be an issue if it goes into non-
    /// integer territory.
    ///
    /// ```ignore
    /// (-(2 * d_v + d_a) +- 2 * sqrt(((2 * d_v + d_a) / 2) ^ 2 - 2 * d_a * d_p)) / (2 * d_a)
    /// = (-(2 * d_v + d_a) +- sqrt((2 * d_v + d_a) ^ 2 - 8 * d_a * d_p)) / (2 * d_a)
    /// ```
    ///
    /// There is no collision in any of the following cases:
    /// * the value inside `sqrt` is negative
    /// * the value inside isn't a square number
    /// * the denominator is 0
    /// * the numerator isn't divisible by the denominator
    /// * the whole expression is negative
    ///
    /// Once we've found the time, compute the position for both, and verify they're equal. They'll
    /// have the same in the initial axis, but not necessarily in the other two. We try to find a
    /// time for all axes, in case the trajectories in a single axis always match.
    fn try_compute_collision(&self, other: &Self) -> Option<Collision> {
        (0_usize..3_usize)
            .filter_map(|axis| self.try_compute_collision_time_candidates_for_axis(other, axis))
            .flatten()
            .flatten()
            .filter_map(|time| {
                let self_pos: IVec3 = self.pos_at_time(time);
                let other_pos: IVec3 = other.pos_at_time(time);

                (self_pos == other_pos).then_some(Collision::new(time, self_pos))
            })
            .next()
    }
}

impl Parse for Particle {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((
                Self::parse_component("p"),
                tag(", "),
                Self::parse_component("v"),
                tag(", "),
                Self::parse_component("a"),
            )),
            |(pos, _, vel, _, acc)| Self { pos, vel, acc },
        )(input)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Particle>);

impl Solution {
    fn long_term_closest_particle_index(&self) -> usize {
        self.0
            .iter()
            .enumerate()
            .map(|(index, particle)| (index, manhattan_magnitude_3d(&particle.acc)))
            .min_by_key(|(_, acc_manhattan_mag)| *acc_manhattan_mag)
            .unwrap()
            .0
    }

    fn compute_collisions(&self) -> BTreeMap<Collision, Vec<usize>> {
        let mut collision_to_particle_indices: BTreeMap<Collision, Vec<usize>> = BTreeMap::new();

        for (index_a, particle_a) in self.0.iter().enumerate() {
            for (index_b, particle_b) in self.0.iter().enumerate().skip(index_a + 1_usize) {
                if let Some(collision) = particle_a.try_compute_collision(particle_b) {
                    if !collision_to_particle_indices.contains_key(&collision) {
                        collision_to_particle_indices.insert(collision, Vec::new());
                    }

                    let particles: &mut Vec<usize> =
                        collision_to_particle_indices.get_mut(&collision).unwrap();

                    if !particles.contains(&index_a) {
                        particles.push(index_a);
                    }

                    if !particles.contains(&index_b) {
                        particles.push(index_b);
                    }
                }
            }
        }

        let mut has_collided: BitVec = bitvec![0; self.0.len()];
        let mut false_collisions: Vec<Collision> = Vec::new();

        for (collision, particle_indices) in collision_to_particle_indices.iter_mut() {
            for particles_index_index in (0_usize..particle_indices.len()).rev() {
                let particle_index: usize = particle_indices[particles_index_index];

                if has_collided[particle_index] {
                    particle_indices.remove(particles_index_index);
                }
            }

            if particle_indices.len() < 2_usize {
                false_collisions.push(*collision);
            } else {
                particle_indices.sort();

                for particle_index in particle_indices {
                    has_collided.set(*particle_index, true);
                }
            }
        }

        for false_collision in false_collisions {
            collision_to_particle_indices.remove(&false_collision);
        }

        collision_to_particle_indices
    }

    fn particles_remaining_after_collisions(
        &self,
        collisions: &BTreeMap<Collision, Vec<usize>>,
    ) -> usize {
        self.0.len()
            - collisions
                .iter()
                .map(|(_, particle_indices)| particle_indices.len())
                .sum::<usize>()
    }

    fn particles_remaining_after_all_collisions(&self) -> usize {
        self.particles_remaining_after_collisions(&self.compute_collisions())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Particle::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    /// My original attempt here was to figure out the point in time when all particles are speeding
    /// up away from the origin (pos dot vel and vel dot acc are both positive). Unfortunately, this
    /// wasn't correct, as there can be cases where there's a point that's closer to the origin but
    /// speeding up faster away than some other point. It was at this point that I realized the true
    /// solution is just which ever point is accelerating the least, because all ther points will
    /// be moving faster than it at some point, which means after enough time they will be further
    /// from the origin than the one that accelerates the least.
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.long_term_closest_particle_index());
    }

    /// This part 2 seems not super related computationally to part 1. I don't know of a sure fire
    /// way to know when all collisions have occurred. My plan is just to run N steps after each
    /// step a collision is found, and hope that I've found an N that will yield the correct answer
    /// for my sample set without being prohibitively time consuming.
    ///
    /// Edit: Well, that wasn't correct, so back to the drawing board. Updated strat: for each pair
    /// of particles, compute the collision between them if it exists. If it does, keep track of it
    /// in a map of collision to list of particles that collide. Sort this map by the time of
    /// collision. Iterate over the values in this map. For each particle that collides, check if it
    /// has already collided. If it has, remove it from the new collision. If the resulting list
    /// (after checking all particles) is has less than two particles, remove it from the map,
    /// otherwise sort the particles in its list (this sorting likely isn't necessary, as elements
    /// should be added in increasing order). From here, computing the number of remaining particles
    /// is just the original number of particles minus the sum of lengths of these lists. I had some
    /// bugs in my collision detection/computation code. Thankfully, the lists of particles aren't
    /// randomly permutated, meaning all collisions are from runs of consecutive particles, which
    /// helped to identify edge cases my code wasn't catching.
    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let collisions: BTreeMap<Collision, Vec<usize>> = self.compute_collisions();

            dbg!(&collisions);
            dbg!(self.particles_remaining_after_collisions(&collisions));
        } else {
            dbg!(self.particles_remaining_after_all_collisions());
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

    const SOLUTION_STRS: &'static [&'static str] = &["\
        p=< 3,0,0>, v=< 2,0,0>, a=<-1,0,0>\n\
        p=< 4,0,0>, v=< 0,0,0>, a=<-2,0,0>\n"];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCK: OnceLock<Vec<Solution>> = OnceLock::new();

        &ONCE_LOCK.get_or_init(|| {
            vec![Solution(vec![
                Particle {
                    pos: 3_i32 * IVec3::X,
                    vel: 2_i32 * IVec3::X,
                    acc: -1_i32 * IVec3::X,
                },
                Particle {
                    pos: 4_i32 * IVec3::X,
                    vel: 0_i32 * IVec3::X,
                    acc: -2_i32 * IVec3::X,
                },
            ])]
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
    fn test_long_term_closest_particle_index() {
        for (index, long_term_closest_particle_index) in [0_usize].into_iter().enumerate() {
            assert_eq!(
                solution(index).long_term_closest_particle_index(),
                long_term_closest_particle_index
            );
        }
    }

    #[test]
    fn test_input() {
        // let args: Args = Args::parse(module_path!()).unwrap().1;

        // Solution::both(&args);
    }
}
