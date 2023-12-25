use {
    crate::*,
    glam::{DMat4, DVec2, DVec3, DVec4, Vec3Swizzles, Vec4Swizzles},
    nom::{
        bytes::complete::tag,
        character::complete::{line_ending, space1},
        combinator::{map, opt},
        error::Error,
        multi::many0,
        sequence::{terminated, tuple},
        Err, IResult,
    },
};

#[cfg_attr(test, derive(Debug, PartialEq))]
struct XYIntersection {
    pos: DVec2,
    t: f64,
    u: f64,
}

impl XYIntersection {
    fn filter_t_and_u(is_a_in_past: bool, is_b_in_past: bool) -> impl Fn(&Self) -> bool {
        move |xy_intersection: &Self| {
            ((xy_intersection.t < 0.0_f64) == is_a_in_past)
                && ((xy_intersection.u < 0.0_f64) == is_b_in_past)
        }
    }

    #[cfg(test)]
    fn filter_pos(pos: DVec2) -> impl Fn(&XYIntersection) -> bool {
        move |xy_intersection: &XYIntersection| xy_intersection.pos.abs_diff_eq(pos, 0.001_f64)
    }

    fn filter_range(min: f64, max: f64, is_in_range: bool) -> impl Fn(&XYIntersection) -> bool {
        move |xy_intersection: &XYIntersection| {
            (xy_intersection.pos.cmpge(min * DVec2::ONE).all()
                && xy_intersection.pos.cmple(max * DVec2::ONE).all())
                == is_in_range
        }
    }
}

#[cfg_attr(test, derive(Clone, PartialEq))]
#[derive(Debug)]
struct Hailstone {
    pos: DVec3,
    vel: DVec3,
}

impl Hailstone {
    fn parse_dvec3<'i>(input: &'i str) -> IResult<&'i str, DVec3> {
        map(
            tuple((
                parse_integer::<i64>,
                tag(","),
                space1,
                parse_integer::<i64>,
                tag(","),
                space1,
                parse_integer::<i64>,
            )),
            |(x, _, _, y, _, _, z)| DVec3::new(x as f64, y as f64, z as f64),
        )(input)
    }

    fn as_dprojectile2(&self) -> DProjectile2 {
        DProjectile2 {
            pos: self.pos.xy(),
            vel: self.vel.xy(),
        }
    }

    fn sum_pos(&self) -> i64 {
        let pos: DVec3 = self.pos.round();

        pos.x as i64 + pos.y as i64 + pos.z as i64
    }
}

impl Parse for Hailstone {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(
            tuple((Self::parse_dvec3, tag(" @"), space1, Self::parse_dvec3)),
            |(pos, _, _, vel)| Self { pos, vel },
        )(input)
    }
}

struct DProjectile2 {
    pos: DVec2,
    vel: DVec2,
}

impl DProjectile2 {
    /// Computes the interpolation constants, `(t, u)`, for two `DProjectile2`s.
    ///
    /// Use the following definitions:
    ///
    /// * `x1 = self.pos.x`
    /// * `y1 = self.pos.y`
    /// * `vx1 = self.vel.x`
    /// * `vy1 = self.vel.y`
    /// * `x2 = x1 + vx1` (the `x` position after 1 time unit)
    /// * `y2 = y1 + vy1` (the `y` position after 1 time unit)
    /// * `x3 = other.pos.x`
    /// * `y3 = other.pos.y`
    /// * `vx2 = other.vel.x`
    /// * `vy2 = other.vel.y`
    /// * `x4 = x3 + vx2` (the `x` position after 1 time unit)
    /// * `y4 = y3 + vy2` (the `y` position after 1 time unit)
    ///
    /// With these definitions, consider the following equations for `t` and `u` from
    /// [Wikipedia][wiki]:
    ///
    /// ```
    /// t = (x1 - x3)(y3 - y4) - (y1 - y3)(x3 - x4)
    ///     ---------------------------------------
    ///     (x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)
    ///
    /// u = (x1 - x3)(y1 - y2) - (y1 - y3)(x1 - x2)
    ///     ---------------------------------------
    ///     (x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)
    /// ```
    ///
    /// Substituting for the velocity definitions:
    ///
    /// ```
    /// t = (x1 - x3)(-vy2) - (y1 - y3)(-vx2)
    ///     ---------------------------------
    ///     (-vx1)(-vy2) - (-vy1)(-vx2)
    ///
    /// u = (x1 - x3)(-vy1) - (y1 - y3)(-vx1)
    ///     ---------------------------------
    ///     (-vx1)(-vy2) - (-vy1)(-vx2)
    /// ```
    ///
    /// Simplifying some of the negatives:
    ///
    /// ```
    /// t = vy2 * (x3 - x1) - vx2 * (y3 - y1)
    ///     ---------------------------------
    ///     vx1 * vy2 - vy1 * vx2
    ///
    /// u = vy1 * (x3 - x1) - vx1 * (y3 - y1)
    ///     ---------------------------------
    ///     vx1 * vy2 - vy1 * vx2
    /// ```
    ///
    /// Note that the denominator is the same for both `t` and `u`. If this is zero, the two
    /// `DProjectile2's trajectories are parellel, and there is no intersection. Otherwise, the
    /// intersection is `(x1 + t * vx1, y1 + t * vy1) = (x3 + u * vx2, y3 + u * vy2)`.
    ///
    /// [wiki]: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
    fn compute_xy_intersection_t_and_u(&self, other: &Self) -> Option<(f64, f64)> {
        let denominator: f64 = self.vel.x * other.vel.y - self.vel.y * other.vel.x;

        (denominator != 0.0_f64).then(|| {
            let x3_minus_x1: f64 = other.pos.x - self.pos.x;
            let y3_minus_y1: f64 = other.pos.y - self.pos.y;
            let t: f64 = (other.vel.y * x3_minus_x1 - other.vel.x * y3_minus_y1) / denominator;
            let u: f64 = (self.vel.y * x3_minus_x1 - self.vel.x * y3_minus_y1) / denominator;

            (t, u)
        })
    }

    fn compute_xy_intersection(&self, other: &Self) -> Option<XYIntersection> {
        self.compute_xy_intersection_t_and_u(other)
            .map(|(t, u)| XYIntersection {
                pos: self.pos + t * self.vel,
                t,
                u,
            })
    }

    /// Given this pair of `DProjectile2`s, compute the coefficients for an equation that can be
    /// used to solve for the position and velocity of a rock that can hit all existing
    /// `DProjectile2`s.
    ///
    /// Consider the equations for `t` and `u` on `Self::compute_xy_intersection_t_and_u`. Consider
    /// these values for `DProjectile2` from `Hailstone` `a` and the `DProjectile2` from thrown
    /// rock, `r`. Because these two projectiles are supposed to intersect at the same time value,
    /// we can set these two equations equal to each other to get a valid equation.
    ///
    /// Use the following substitutions/definitions:
    ///
    /// * `xa = a.pos.x = x1`
    /// * `ya = a.pos.y = y1`
    /// * `vxa = a.vel.x = vx1`
    /// * `vya = a.vel.y = vy1`
    /// * `xr = r.pos.x = x3`
    /// * `yr = r.pos.x = y3`
    /// * `vxr = r.vel.x = vx2`
    /// * `vyr = r.vel.y = vy2`
    ///
    /// ```
    /// vyr * (xr - xa) - vxr * (yr - ya) = vya * (xa - xr) - vxa * (ya - yr)
    /// ```
    ///
    /// Simplify, keeping in mind that for the sake of solving for the thrown rock, the
    /// values associated with `a` are constant.
    ///
    /// ```
    /// vyr * xr - xa * vyr - vxr * yr + ya * vxr - (vya * xa - vya * xr - vxa * ya + vxa * yr) = 0
    /// = vyr * xr - vxr * yr + ya * vxr - xa * vyr + vya * xr - vxa * yr + vxa * ya - vya * xa
    /// ```
    ///
    /// Call the last equation above "rock coefficients in terms of `a`". This alone can't be used
    /// to produce the coefficients we're looking for, since the two products on the left each
    /// contain two unknowns. Critically, however, they don't include any constants from `a`.
    /// Similarly, the two products on the right contain only constants. Now we're ready to use
    /// `self` and `other` (keep in mind `other` is also a `Hailstone`, not the thrown stone).
    ///
    /// Use the following definitions:
    ///
    /// * `xa = self.pos.x`
    /// * `ya = self.pos.y`
    /// * `vxa = self.vel.x`
    /// * `vya = self.vel.y`
    /// * `xb = other.pos.x`
    /// * `yb = other.pos.y`
    /// * `vxb = other.vel.x`
    /// * `vyb = other.vel.y`
    /// * `ta = vxa * ya - vya * xa`
    /// * `tb = vxb * yb - vyb * xb`
    /// * `xs = xa - xb`
    /// * `ys = ya - yb`
    /// * `vxs = vxa - vxb`
    /// * `vys = vya - vyb`
    /// * `ts = ta - tb`
    ///
    /// Consider the difference between "rock coefficients in terms of `a`" and "rock coefficients
    /// in terms of `b`". Convince yourself that this is equal to the following:
    ///
    /// ```
    /// ys * vxr - xs * vyr + vys * xr - vxs * yr + ts = 0
    /// ys * vxr - xs * vyr + vys * xr - vxs * yr = -ts
    /// vys * xr - vxs * yr + ys * vxr - xs * vyr = -ts
    /// ```
    ///
    /// If we use four different pairs of `Hailstone`s, this will produce four linear equations with
    /// four unknowns, which we can solve using matrix manipulation.
    fn compute_thrown_rock_matrix_equation_row(&self, other: &Self) -> (DVec4, f64) {
        let ta: f64 = self.vel.x * self.pos.y - self.vel.y * self.pos.x;
        let tb: f64 = other.vel.x * other.pos.y - other.vel.y * other.pos.x;
        let pos_s: DVec2 = self.pos - other.pos;
        let vel_s: DVec2 = self.vel - other.vel;

        (DVec4::new(vel_s.y, -vel_s.x, pos_s.y, -pos_s.x), tb - ta)
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution(Vec<Hailstone>);

impl Solution {
    const MIN: f64 = 200000000000000.0_f64;
    const MAX: f64 = 400000000000000.0_f64;

    fn iter_hailstone_pairs(&self) -> impl Iterator<Item = (&Hailstone, &Hailstone)> {
        self.0.iter().enumerate().flat_map(|(index, hailstone_a)| {
            self.0[index + 1_usize..]
                .iter()
                .map(move |hailstone_b| (hailstone_a, hailstone_b))
        })
    }

    fn iter_xy_intersections(&self) -> impl Iterator<Item = Option<XYIntersection>> + '_ {
        self.iter_hailstone_pairs()
            .map(|(hailstone_a, hailstone_b)| {
                hailstone_a
                    .as_dprojectile2()
                    .compute_xy_intersection(&hailstone_b.as_dprojectile2())
            })
    }

    fn count_xy_intersections_within_range(&self, min: f64, max: f64) -> usize {
        self.iter_xy_intersections()
            .flatten()
            .filter(XYIntersection::filter_t_and_u(false, false))
            .filter(XYIntersection::filter_range(min, max, true))
            .count()
    }

    fn iter_thrown_rock_matrix_equation_rows(&self) -> impl Iterator<Item = (DVec4, f64)> + '_ {
        self.iter_hailstone_pairs()
            .map(|(hailstone_a, hailstone_b)| {
                hailstone_a
                    .as_dprojectile2()
                    .compute_thrown_rock_matrix_equation_row(&hailstone_b.as_dprojectile2())
            })
    }

    fn compute_thrown_rock_matrix_equation(&self) -> Option<(DMat4, DVec4)> {
        let mut cols: [DVec4; 4_usize] = Default::default();
        let mut col: [f64; 4_usize] = Default::default();

        (self
            .iter_thrown_rock_matrix_equation_rows()
            .take(4_usize)
            .enumerate()
            .map(|(index, (row, col_value))| {
                cols[index] = row;
                col[index] = col_value;
            })
            .count()
            == 4_usize)
            .then(|| {
                (
                    DMat4::from_cols(cols[0_usize], cols[1_usize], cols[2_usize], cols[3_usize])
                        .transpose(),
                    DVec4::from(col),
                )
            })
    }

    fn compute_thrown_rock_dprojectile2(&self) -> Option<DProjectile2> {
        self.compute_thrown_rock_matrix_equation()
            .map(|(mat, col)| {
                let thrown_rock_col: DVec4 = mat.inverse() * col;

                DProjectile2 {
                    pos: thrown_rock_col.xy(),
                    vel: thrown_rock_col.zw(),
                }
            })
    }

    fn compute_thrown_rock_hailstone(&self) -> Option<Hailstone> {
        self.compute_thrown_rock_dprojectile2().map(|dprojectile2| {
            let compute_intersection_and_time_with_hailstone = |hailstone: &Hailstone| {
                let t: f64 = hailstone
                    .as_dprojectile2()
                    .compute_xy_intersection_t_and_u(&dprojectile2)
                    .unwrap()
                    .0;

                (hailstone.pos + t * hailstone.vel, t)
            };

            let hailstone_a: &Hailstone = &self.0[0_usize];
            let hailstone_b: &Hailstone = &self.0[1_usize];
            let (intersection_a, time_a): (DVec3, f64) =
                compute_intersection_and_time_with_hailstone(hailstone_a);
            let (intersection_b, time_b): (DVec3, f64) =
                compute_intersection_and_time_with_hailstone(hailstone_b);

            let vel: DVec3 = (intersection_a - intersection_b) / (time_a - time_b);
            let pos: DVec3 = intersection_a - time_a * vel;

            Hailstone {
                pos: pos.round(),
                vel: vel.round(),
            }
        })
    }

    fn compute_thrown_rock_hailstone_pos_sum(&self) -> Option<i64> {
        self.compute_thrown_rock_hailstone()
            .map(|hailstone| hailstone.sum_pos())
    }
}

impl Parse for Solution {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        map(many0(terminated(Hailstone::parse, opt(line_ending))), Self)(input)
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.count_xy_intersections_within_range(Self::MIN, Self::MAX));
    }

    fn q2_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            if let Some(hailstone) = self.compute_thrown_rock_hailstone() {
                dbg!(hailstone.sum_pos(), hailstone);
            } else {
                eprintln!("Failed to compute thrown rock hailstone.");
            }
        } else {
            dbg!(self.compute_thrown_rock_hailstone_pos_sum());
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
        19, 13, 30 @ -2,  1, -2\n\
        18, 19, 22 @ -1, -1, -2\n\
        20, 25, 34 @ -2, -2, -4\n\
        12, 31, 28 @ -1, -2, -1\n\
        20, 19, 15 @  1, -5, -3\n";

    const MIN: f64 = 7.0_f64;
    const MAX: f64 = 27.0_f64;

    fn solution() -> &'static Solution {
        macro_rules! hailstones {
            [ $( $px:expr, $py:expr, $pz:expr, $vx:expr, $vy:expr, $vz:expr; )* ] => { vec![ $(
                Hailstone {
                    pos: DVec3::new($px as f64, $py as f64, $pz as f64),
                    vel: DVec3::new($vx as f64, $vy as f64, $vz as f64),
                },
            )* ] };
        }

        static ONCE_LOCK: OnceLock<Solution> = OnceLock::new();

        ONCE_LOCK.get_or_init(|| {
            Solution(hailstones![
                19, 13, 30, -2,  1, -2;
                18, 19, 22, -1, -1, -2;
                20, 25, 34, -2, -2, -4;
                12, 31, 28, -1, -2, -1;
                20, 19, 15,  1, -5, -3;
            ])
        })
    }

    #[test]
    fn test_try_from_str() {
        assert_eq!(Solution::try_from(SOLUTION_STR).as_ref(), Ok(solution()));
    }

    #[test]
    fn test_iter_future_xy_intersections() {
        enum XYIntersectionExpectation {
            Parallel,
            PastForA,
            PastForB,
            PastForBoth,
            OutsideTestArea(DVec2),
            InsideTestArea(DVec2),
        }

        use XYIntersectionExpectation::*;

        impl XYIntersectionExpectation {
            fn evaluate(self, xy_intersection: Option<XYIntersection>) -> bool {
                match self {
                    XYIntersectionExpectation::Parallel => xy_intersection.is_none(),
                    XYIntersectionExpectation::PastForA => xy_intersection
                        .filter(XYIntersection::filter_t_and_u(true, false))
                        .is_some(),
                    XYIntersectionExpectation::PastForB => xy_intersection
                        .filter(XYIntersection::filter_t_and_u(false, true))
                        .is_some(),
                    XYIntersectionExpectation::PastForBoth => xy_intersection
                        .filter(XYIntersection::filter_t_and_u(true, true))
                        .is_some(),
                    XYIntersectionExpectation::OutsideTestArea(pos) => xy_intersection
                        .filter(XYIntersection::filter_t_and_u(false, false))
                        .filter(XYIntersection::filter_range(MIN, MAX, false))
                        .filter(XYIntersection::filter_pos(pos))
                        .is_some(),
                    XYIntersectionExpectation::InsideTestArea(pos) => xy_intersection
                        .filter(XYIntersection::filter_t_and_u(false, false))
                        .filter(XYIntersection::filter_range(MIN, MAX, true))
                        .filter(XYIntersection::filter_pos(pos))
                        .is_some(),
                }
            }
        }

        for (xy_intersection, xy_intersection_expectation) in
            solution().iter_xy_intersections().zip([
                InsideTestArea(DVec2::new(14.333_f64, 15.333_f64)),
                InsideTestArea(DVec2::new(11.667_f64, 16.667_f64)),
                OutsideTestArea(DVec2::new(6.2_f64, 19.4_f64)),
                PastForA,
                Parallel,
                OutsideTestArea(DVec2::new(-6.0_f64, -5.0_f64)),
                PastForBoth,
                OutsideTestArea(DVec2::new(-2.0_f64, 3.0_f64)),
                PastForB,
                PastForBoth,
            ])
        {
            assert!(xy_intersection_expectation.evaluate(xy_intersection));
        }
    }

    #[test]
    fn test_count_xy_intersections_within_range() {
        assert_eq!(
            solution().count_xy_intersections_within_range(MIN, MAX),
            2_usize
        );
    }

    #[test]
    fn test_compute_thrown_rock_hailstone() {
        assert_eq!(
            solution().compute_thrown_rock_hailstone(),
            Some(Hailstone {
                pos: DVec3::new(24.0_f64, 13.0_f64, 10.0_f64),
                vel: DVec3::new(-3.0_f64, 1.0_f64, 2.0_f64)
            })
        );
    }

    #[test]
    fn test_compute_thrown_rock_hailstone_pos_sum() {
        assert_eq!(
            solution().compute_thrown_rock_hailstone_pos_sum(),
            Some(47_i64)
        );
    }
}
