use {
    crate::*,
    glam::IVec3,
    nom::{
        bytes::complete::tag,
        character::complete::{digit1, line_ending},
        combinator::{cut, map, map_opt, map_res, opt},
        error::Error,
        multi::many0_count,
        sequence::{terminated, tuple},
        Err, IResult,
    },
    num::{Num, NumCast, ToPrimitive},
    std::{
        cell::{Ref, RefCell, RefMut},
        cmp::Ordering,
        collections::{HashMap, HashSet},
        ops::Range,
        str::FromStr,
    },
};

mod ivec3_util {
    #![allow(dead_code)]

    use super::*;

    pub const ROTATE_ABOUT_IDENTITY_TAU_OVER_3: IMat3 =
        IMat3::from_cols(IVec3::Y, IVec3::Z, IVec3::X);
    pub const ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
    );

    pub const ROTATE_ABOUT_X_TAU_OVER_4: IMat3 = IMat3::from_cols(IVec3::X, IVec3::Z, IVec3::NEG_Y);
    pub const ROTATE_ABOUT_X_TAU_OVER_2: IMat3 =
        imat3_const_mul(&ROTATE_ABOUT_X_TAU_OVER_4, &ROTATE_ABOUT_X_TAU_OVER_4);
    pub const ROTATE_ABOUT_X_3_TAU_OVER_4: IMat3 =
        imat3_const_mul(&ROTATE_ABOUT_X_TAU_OVER_4, &ROTATE_ABOUT_X_TAU_OVER_2);

    pub const ROTATE_ABOUT_Y_TAU_OVER_4: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_TAU_OVER_4,
            &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        ),
    );
    pub const ROTATE_ABOUT_Y_TAU_OVER_2: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_TAU_OVER_2,
            &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        ),
    );
    pub const ROTATE_ABOUT_Y_3_TAU_OVER_4: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_3_TAU_OVER_4,
            &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        ),
    );

    pub const ROTATE_ABOUT_Z_TAU_OVER_4: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_TAU_OVER_4,
            &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        ),
    );
    pub const ROTATE_ABOUT_Z_TAU_OVER_2: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_TAU_OVER_2,
            &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        ),
    );
    pub const ROTATE_ABOUT_Z_3_TAU_OVER_4: IMat3 = imat3_const_mul(
        &ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
        &imat3_const_mul(
            &ROTATE_ABOUT_X_3_TAU_OVER_4,
            &ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
        ),
    );

    pub fn greatest_mag_index(delta: IVec3) -> usize {
        let abs: [i32; 3_usize] = delta.abs().into();

        (0_usize..3_usize)
            .into_iter()
            .max_by_key(|index| abs[*index])
            .unwrap()
    }

    pub fn orientation(delta: IVec3) -> IMat3 {
        let greatest_mag_index: usize = greatest_mag_index(delta);
        let rotate_greatest_mag_along_x: IMat3 = match greatest_mag_index {
            0_usize => IMat3::IDENTITY,
            1_usize => ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
            2_usize => ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
            _ => unreachable!(),
        };
        let x_is_negative: bool = delta[greatest_mag_index] < 0;
        let rotate_greatest_mag_along_positive_x: IMat3 = if x_is_negative {
            ROTATE_ABOUT_Z_TAU_OVER_2
        } else {
            IMat3::IDENTITY
        };
        let y_ordering: Ordering = delta[(greatest_mag_index + 1_usize) % 3_usize].cmp(&0_i32);
        let rotate_second_greatest_mat_along_positive_y: IMat3 = match (
            if x_is_negative {
                y_ordering.reverse()
            } else {
                y_ordering
            },
            delta[(greatest_mag_index + 2_usize) % 3_usize].cmp(&0_i32),
        ) {
            (Ordering::Less, Ordering::Greater) => ROTATE_ABOUT_X_3_TAU_OVER_4,
            (Ordering::Less, _) => ROTATE_ABOUT_X_TAU_OVER_2,
            (Ordering::Equal, Ordering::Less) => ROTATE_ABOUT_X_TAU_OVER_4,
            (Ordering::Equal, Ordering::Equal) => IMat3::IDENTITY,
            (Ordering::Equal, Ordering::Greater) => ROTATE_ABOUT_X_3_TAU_OVER_4,
            (Ordering::Greater, Ordering::Less) => ROTATE_ABOUT_X_TAU_OVER_4,
            (Ordering::Greater, _) => IMat3::IDENTITY,
        };

        // Apply the three in the correct order
        rotate_second_greatest_mat_along_positive_y
            * rotate_greatest_mag_along_positive_x
            * rotate_greatest_mag_along_x
    }

    #[cfg(test)]
    mod tests {
        use {
            super::*,
            glam::{Mat3, Vec3},
            rand::prelude::*,
            std::{f32::consts::TAU, ops::RangeInclusive},
        };

        #[test]
        fn test_constants() {
            const ITERATIONS: usize = 1000_usize;
            const RANGE: RangeInclusive<i32> = -1000_i32..=1000_i32;
            let mut rng: ThreadRng = rand::thread_rng();

            for (index, (imat3, mat3)) in [
                (
                    ROTATE_ABOUT_IDENTITY_TAU_OVER_3,
                    Mat3::from_axis_angle(Vec3::ONE.normalize(), TAU / 3.0_f32),
                ),
                (
                    ROTATE_ABOUT_IDENTITY_2_TAU_OVER_3,
                    Mat3::from_axis_angle(Vec3::ONE.normalize(), 2.0_f32 * TAU / 3.0_f32),
                ),
                (
                    ROTATE_ABOUT_X_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::X, TAU / 4.0_f32),
                ),
                (
                    ROTATE_ABOUT_X_TAU_OVER_2,
                    Mat3::from_axis_angle(Vec3::X, TAU / 2.0_f32),
                ),
                (
                    ROTATE_ABOUT_X_3_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::X, 3.0_f32 * TAU / 4.0_f32),
                ),
                (
                    ROTATE_ABOUT_Y_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::Y, TAU / 4.0_f32),
                ),
                (
                    ROTATE_ABOUT_Y_TAU_OVER_2,
                    Mat3::from_axis_angle(Vec3::Y, TAU / 2.0_f32),
                ),
                (
                    ROTATE_ABOUT_Y_3_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::Y, 3.0_f32 * TAU / 4.0_f32),
                ),
                (
                    ROTATE_ABOUT_Z_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::Z, TAU / 4.0_f32),
                ),
                (
                    ROTATE_ABOUT_Z_TAU_OVER_2,
                    Mat3::from_axis_angle(Vec3::Z, TAU / 2.0_f32),
                ),
                (
                    ROTATE_ABOUT_Z_3_TAU_OVER_4,
                    Mat3::from_axis_angle(Vec3::Z, 3.0_f32 * TAU / 4.0_f32),
                ),
            ]
            .into_iter()
            .enumerate()
            {
                for _ in 0_usize..ITERATIONS {
                    let ivec3: IVec3 = IVec3::new(
                        rng.gen_range(RANGE),
                        rng.gen_range(RANGE),
                        rng.gen_range(RANGE),
                    );

                    assert_eq!(
                        imat3 * ivec3,
                        (mat3 * ivec3.as_vec3()).round().as_ivec3(),
                        "\nindex: {index},\nimat3: {imat3:#?},\nmat3: {mat3:#?},\nivec3: {ivec3}"
                    );
                }
            }
        }
    }
}

/// An orientation-agnostic delta from one Beacon to another.
#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct NormDelta<T: Num + Ord + PartialOrd = u16>([T; 3_usize]);

impl<T: Num + NumCast + Ord + PartialOrd> NormDelta<T> {
    fn can_construct_orientation(self) -> bool {
        self.0[0_usize] != T::zero() && self.0[1_usize] != T::zero()
    }

    fn try_mirror_image(self) -> Option<Self> {
        let [x, y, z]: [T; 3_usize] = self.0;

        if z != T::zero() && y != z {
            Some(Self([x, z, y]))
        } else {
            None
        }
    }

    fn try_from<U: Num + ToPrimitive + Ord + PartialOrd>(norm_delta: NormDelta<U>) -> Option<Self> {
        let [x, y, z]: [U; 3_usize] = norm_delta.0;

        Some(Self([
            <T as NumCast>::from(x)?,
            <T as NumCast>::from(y)?,
            <T as NumCast>::from(z)?,
        ]))
    }
}

impl<T: Num + NumCast + Ord + PartialOrd> NormDelta<T> {
    /// Construct an orientation-agnostic representation of the delta between two Beacons.
    ///
    /// Note that this is not necessarily commutative (i.e., some deltas are chiral).
    ///
    /// This will only return `None` if the greatest-magnitude component of the difference is
    /// greater than `u16::MAX`.
    fn try_from_start_and_end(start_beacon: IVec3, end_beacon: IVec3) -> Option<Self> {
        Self::try_from_delta(end_beacon - start_beacon)
    }

    /// Try to construct an orientation-agnostic representation of a three-dimensional vector.
    ///
    /// This will only return `None` if the greatest-magnitude component is greater than `u16::MAX`.
    fn try_from_delta(delta: IVec3) -> Option<Self> {
        NormDelta((ivec3_util::orientation(delta) * delta).to_array()).try_into()
    }
}

impl<T: Num + Ord + PartialOrd + ToPrimitive> NormDelta<T> {
    fn try_into<U: Num + NumCast + Ord + PartialOrd>(self) -> Option<NormDelta<U>> {
        NormDelta::try_from(self)
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Key {
    /// Index of the owning Scanner.
    scanner: u16,

    /// An orientation-agnostic delta between two Beacons owned by the Scanner.
    norm_delta: NormDelta,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct BeaconPair {
    /// Starting Beacon index relative to the slice of `Solution::beacons` for the owning Scanner.
    start: u16,

    /// Ending Beacon index relative to the slice of `Solution::beacons` for the owning Scanner.
    end: u16,
}

impl BeaconPair {
    fn flip(self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }

    fn norm(self) -> Self {
        if self.start > self.end {
            self.flip()
        } else {
            self
        }
    }
}

#[derive(Clone, Copy)]
struct BeaconPairLink {
    beacon_pair: BeaconPair,

    /// The next `PairData` index relative to `DeltaPairMap::pair_data_vec` for the owning
    /// `Scanner`, or `u32::MAX` if there is none.
    next: u32,
}

struct BeaconPairIter<'l> {
    beacon_pair_links: &'l [BeaconPairLink],
    next: u32,
}

impl<'l> Iterator for BeaconPairIter<'l> {
    type Item = BeaconPair;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX {
            None
        } else {
            let beacon_pair_link: BeaconPairLink = self.beacon_pair_links[self.next as usize];

            self.next = beacon_pair_link.next;

            Some(beacon_pair_link.beacon_pair)
        }
    }
}

struct ScannerData {
    norm_delta_range: Range<u32>,
    beacons: usize,
}

#[derive(Default)]
struct NormDeltaData {
    norm_deltas: Vec<NormDelta>,
    scanner_datas: Vec<ScannerData>,
    beacon_pairs: Vec<BeaconPair>,
    beacon_pair_ranges: HashMap<Key, Range<u32>>,
}

impl NormDeltaData {
    fn new(solution: &Solution) -> Self {
        let scanners_len: usize = solution.scanners.len();

        let mut norm_delta_data: Self = Self::default();

        let Self {
            norm_deltas,
            scanner_datas,
            beacon_pairs,
            beacon_pair_ranges,
        } = &mut norm_delta_data;
        let mut norm_delta_beacon_pairs: Vec<(NormDelta, BeaconPair)> = Vec::new();

        for scanner in 0_usize..scanners_len {
            let norm_delta_start: u32 = norm_deltas.len() as u32;
            let beacons: &[IVec3] = solution.beacons(scanner);
            let beacons_len: usize = beacons.len();
            let scanner: u16 = scanner as u16;

            for start_beacon_index in 0_usize..beacons_len.saturating_sub(1_usize) {
                let start_beacon_ivec3: IVec3 = beacons[start_beacon_index];

                for end_beacon_index in start_beacon_index + 1_usize..beacons_len {
                    let end_beacon_ivec3: IVec3 = beacons[end_beacon_index];
                    let norm_delta: NormDelta =
                        NormDelta::try_from_start_and_end(start_beacon_ivec3, end_beacon_ivec3)
                            .unwrap();

                    norm_deltas.push(norm_delta);
                    norm_delta_beacon_pairs.push((
                        norm_delta,
                        BeaconPair {
                            start: start_beacon_index as u16,
                            end: end_beacon_index as u16,
                        },
                    ));
                }
            }

            let norm_delta_end: u32 = norm_deltas.len() as u32;

            scanner_datas.push(ScannerData {
                norm_delta_range: norm_delta_start..norm_delta_end,
                beacons: beacons_len,
            });
            norm_delta_beacon_pairs.sort();

            let mut norm_delta_beacon_pair_start: usize = 0_usize;

            while norm_delta_beacon_pair_start < norm_delta_beacon_pairs.len() {
                let start_norm_delta: NormDelta =
                    norm_delta_beacon_pairs[norm_delta_beacon_pair_start].0;
                let norm_delta_beacon_pair_end: usize = norm_delta_beacon_pairs
                    [norm_delta_beacon_pair_start..]
                    .iter()
                    .position(|(norm_delta, _)| *norm_delta != start_norm_delta)
                    .map_or(norm_delta_beacon_pairs.len(), |offset| {
                        norm_delta_beacon_pair_start + offset
                    });
                let beacon_pair_start: u32 = beacon_pairs.len() as u32;

                beacon_pairs.extend(
                    norm_delta_beacon_pairs
                        [norm_delta_beacon_pair_start..norm_delta_beacon_pair_end]
                        .iter()
                        .map(|(_, beacon_pair)| *beacon_pair),
                );

                let beacon_pair_end: u32 = beacon_pairs.len() as u32;

                beacon_pair_ranges.insert(
                    Key {
                        scanner,
                        norm_delta: start_norm_delta,
                    },
                    beacon_pair_start..beacon_pair_end,
                );
                norm_delta_beacon_pair_start = norm_delta_beacon_pair_end;
            }

            norm_delta_beacon_pairs.clear();
        }

        norm_delta_data
    }

    fn key(&self, scanner: usize, start_beacon: usize, end_beacon: usize) -> Key {
        let scanner_data: &ScannerData = &self.scanner_datas[scanner];
        let norm_deltas: &[NormDelta] =
            &self.norm_deltas[scanner_data.norm_delta_range.as_range_usize()];
        let row_offset: usize = if start_beacon == 0_usize {
            0_usize
        } else {
            let beacons_minus_1: usize = scanner_data.beacons - 1_usize;

            triangle_number(beacons_minus_1) - triangle_number(beacons_minus_1 - start_beacon)
        };
        let col_offset: usize = end_beacon - start_beacon - 1_usize;
        let norm_delta: NormDelta = norm_deltas[row_offset + col_offset];

        Key {
            scanner: scanner as u16,
            norm_delta,
        }
    }

    fn beacon_pairs_for_scanner(&self, scanner: usize) -> impl Iterator<Item = BeaconPair> {
        let beacons: u16 = self.scanner_datas[scanner].beacons as u16;

        (0_u16..beacons.saturating_sub(1_u16)).flat_map(move |start| {
            (start + 1_u16..beacons).map(move |end| BeaconPair { start, end })
        })
    }

    fn beacon_pairs_for_key(&self, key: Key) -> &[BeaconPair] {
        self.beacon_pair_ranges
            .get(&key)
            .map_or(&self.beacon_pairs[..0], |beacon_pair_range| {
                &self.beacon_pairs[beacon_pair_range.as_range_usize()]
            })
    }

    fn beacon_pairs_for_key_with_mirror(&self, key: Key) -> impl Iterator<Item = BeaconPair> + '_ {
        let mirror_norm_delta_beacon_pairs: &[BeaconPair] =
            if let Some(norm_delta) = key.norm_delta.try_mirror_image() {
                self.beacon_pairs_for_key(Key { norm_delta, ..key })
            } else {
                &[]
            };

        self.beacon_pairs_for_key(key).iter().copied().chain(
            mirror_norm_delta_beacon_pairs
                .iter()
                .copied()
                .map(BeaconPair::flip),
        )
    }
}

#[derive(Clone)]
struct Transformation {
    orientation: IMat3,
    translation: IVec3,
}

impl Transformation {
    const INVALID: Self = Transformation {
        orientation: IMat3::ZERO,
        translation: IVec3::ZERO,
    };

    fn is_valid(&self) -> bool {
        self.orientation != Self::INVALID.orientation
    }

    fn transform(&self, position: IVec3) -> IVec3 {
        (self.orientation * position) + self.translation
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            orientation: self.orientation * rhs.orientation,
            translation: (self.orientation * rhs.translation) + self.translation,
        }
    }

    fn try_from_beacons(
        start_a: IVec3,
        end_a: IVec3,
        start_b: IVec3,
        end_b: IVec3,
    ) -> Option<Self> {
        Self::try_compute_orientation_from_deltas(end_a - start_a, end_b - start_b).map(
            |orientation| Self {
                orientation,
                translation: start_b - (orientation * start_a),
            },
        )
    }

    fn try_compute_orientation_from_deltas(start_vec: IVec3, end_vec: IVec3) -> Option<IMat3> {
        let start_orientation: IMat3 = ivec3_util::orientation(start_vec);
        let end_orientation: IMat3 = ivec3_util::orientation(end_vec);
        let start_delta: NormDelta<i32> = NormDelta((start_orientation * start_vec).to_array());
        let end_delta: NormDelta<i32> = NormDelta((end_orientation * end_vec).to_array());

        if start_delta != end_delta || !start_delta.can_construct_orientation() {
            None
        } else {
            Some(imat3_const_inverse(&end_orientation) * start_orientation)
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Solution {
    beacons: Vec<IVec3>,
    scanners: Vec<Range<usize>>,

    #[cfg(test)]
    minimum_overlapping_beacons: usize,
}

impl Solution {
    const MINIMUM_OVERLAPPING_BEACONS: usize = 12_usize;

    fn try_compute_all_beacons(&self) -> Option<Vec<IVec3>> {
        self.try_compute_all_beacons_internal(|x| x)
            .map(Self::sort_ivec3s)
    }

    fn try_compute_all_beacons_count(&self) -> Option<usize> {
        self.try_compute_all_beacons_internal(|_| ())
            .map(|all_beacons| all_beacons.len())
    }

    fn try_compute_largest_manhattan_distance(&self) -> Option<i32> {
        self.try_compute_transformations()
            .as_ref()
            .map(|transformations| {
                (0_usize..transformations.len().saturating_sub(1_usize))
                    .flat_map(|scanner_a| {
                        let translation_a: IVec3 = transformations[scanner_a].translation;

                        (scanner_a..transformations.len()).map(move |scanner_b| {
                            (transformations[scanner_b].translation - translation_a)
                                .abs()
                                .to_array()
                                .into_iter()
                                .sum()
                        })
                    })
                    .max()
                    .unwrap_or_default()
            })
    }

    fn try_compute_all_beacons_internal<T, F: Fn(IVec3) -> T>(&self, f: F) -> Option<Vec<T>> {
        self.try_compute_transformations()
            .map(|transformations: Vec<Transformation>| {
                let mut present_beacons: HashSet<IVec3> = HashSet::new();

                for scanner in 0_usize..self.scanners.len() {
                    let scanner_beacons: &[IVec3] = self.beacons(scanner);
                    let transformation: &Transformation = &transformations[scanner];

                    for beacon in scanner_beacons {
                        present_beacons.insert(transformation.transform(*beacon));
                    }
                }

                present_beacons.iter().copied().map(f).collect()
            })
    }

    fn try_compute_transformations(&self) -> Option<Vec<Transformation>> {
        let scanners_len: usize = self.scanners.len();
        let mut transformations: Vec<Transformation> = Vec::with_capacity(scanners_len);
        let mut invalid_scanners: usize = 0_usize;

        if scanners_len > 0_usize {
            for _ in 0_usize..scanners_len {
                transformations.push(Transformation::INVALID);
            }

            transformations[0_usize] = Transformation {
                orientation: IMat3::IDENTITY,
                translation: IVec3::ZERO,
            };
            invalid_scanners = scanners_len - 1_usize;

            let norm_delta_data: NormDeltaData = NormDeltaData::new(self);
            let mut mapped_scanner_beacon_trios: HashSet<(bool, BeaconPair)> = HashSet::new();

            'while_loop: while invalid_scanners > 0_usize {
                for mapped_scanner in 0_usize..scanners_len {
                    if transformations[mapped_scanner].is_valid() {
                        for unmapped_scanner in 0_usize..scanners_len {
                            if !transformations[unmapped_scanner].is_valid() {
                                if let Some(transformation) = self.find_transformation(
                                    &norm_delta_data,
                                    mapped_scanner,
                                    unmapped_scanner,
                                    &mut mapped_scanner_beacon_trios,
                                ) {
                                    transformations[unmapped_scanner] =
                                        transformations[mapped_scanner].mul(&transformation);
                                    invalid_scanners -= 1_usize;

                                    continue 'while_loop;
                                }
                            }
                        }
                    }
                }

                break;
            }
        }

        if invalid_scanners == 0_usize {
            Some(transformations)
        } else {
            None
        }
    }

    fn beacons(&self, scanner: usize) -> &[IVec3] {
        &self.beacons[self.scanners[scanner].clone()]
    }

    fn find_transformation(
        &self,
        norm_delta_data: &NormDeltaData,
        mapped_scanner: usize,
        unmapped_scanner: usize,
        mapped_scanner_beacon_trios: &mut HashSet<(bool, BeaconPair)>,
    ) -> Option<Transformation> {
        mapped_scanner_beacon_trios.clear();

        if self.scanners[mapped_scanner].len() < self.minimum_overlapping_beacons()
            || self.scanners[unmapped_scanner].len() < self.minimum_overlapping_beacons()
        {
            None
        } else {
            let mapped_beacons: &[IVec3] = self.beacons(mapped_scanner);
            let unmapped_beacons: &[IVec3] = self.beacons(unmapped_scanner);

            #[derive(Default)]
            struct BeaconIndexSet {
                array: [u16; Solution::MINIMUM_OVERLAPPING_BEACONS],
                len: usize,
            }

            impl BeaconIndexSet {
                fn is_full(&self) -> bool {
                    self.len == self.array.len()
                }

                fn insert(&mut self, value: u16) {
                    match self.array[..self.len].binary_search(&value) {
                        Ok(_) => (),
                        Result::Err(index) => {
                            assert!(!self.is_full());
                            self.array[self.len] = value;
                            self.len += 1_usize;
                            self.array[index..].rotate_right(1_usize);
                        }
                    }
                }

                fn clear(&mut self) {
                    self.len = 0_usize;
                }
            }

            let mut mapped_beacon_indices: BeaconIndexSet = BeaconIndexSet::default();

            norm_delta_data
                .beacon_pairs_for_scanner(mapped_scanner)
                .find_map(|mapped_beacon_pair| {
                    let mapped_key: Key = norm_delta_data.key(
                        mapped_scanner,
                        mapped_beacon_pair.start as usize,
                        mapped_beacon_pair.end as usize,
                    );

                    if !mapped_key.norm_delta.can_construct_orientation() {
                        None
                    } else {
                        norm_delta_data
                            .beacon_pairs_for_key_with_mirror(Key {
                                scanner: unmapped_scanner as u16,
                                ..mapped_key
                            })
                            .find_map(|unmapped_beacon_pair| {
                                // Use the transformation that maps into the already mapped space
                                let transformation: Transformation =
                                    Transformation::try_from_beacons(
                                        unmapped_beacons[unmapped_beacon_pair.start as usize],
                                        unmapped_beacons[unmapped_beacon_pair.end as usize],
                                        mapped_beacons[mapped_beacon_pair.start as usize],
                                        mapped_beacons[mapped_beacon_pair.end as usize],
                                    )
                                    .unwrap();

                                mapped_scanner_beacon_trios.insert((true, mapped_beacon_pair));
                                mapped_scanner_beacon_trios
                                    .insert((false, unmapped_beacon_pair.norm()));
                                mapped_beacon_indices.clear();
                                mapped_beacon_indices.insert(mapped_beacon_pair.start);
                                mapped_beacon_indices.insert(mapped_beacon_pair.end);

                                let transformation: Option<Transformation> = norm_delta_data
                                    .beacon_pairs_for_scanner(mapped_scanner)
                                    .find_map(|mapped_beacon_pair| {
                                        let mapped_key: Key = norm_delta_data.key(
                                            mapped_scanner,
                                            mapped_beacon_pair.start as usize,
                                            mapped_beacon_pair.end as usize,
                                        );

                                        norm_delta_data
                                            .beacon_pairs_for_key_with_mirror(Key {
                                                scanner: unmapped_scanner as u16,
                                                ..mapped_key
                                            })
                                            .find_map(|unmapped_beacon_pair| {
                                                if transformation.transform(
                                                    unmapped_beacons
                                                        [unmapped_beacon_pair.start as usize],
                                                ) != mapped_beacons
                                                    [mapped_beacon_pair.start as usize]
                                                    || transformation.transform(
                                                        unmapped_beacons
                                                            [unmapped_beacon_pair.end as usize],
                                                    ) != mapped_beacons
                                                        [mapped_beacon_pair.end as usize]
                                                {
                                                    None
                                                } else {
                                                    mapped_scanner_beacon_trios
                                                        .insert((true, mapped_beacon_pair));
                                                    mapped_scanner_beacon_trios.insert((
                                                        false,
                                                        unmapped_beacon_pair.norm(),
                                                    ));
                                                    mapped_beacon_indices
                                                        .insert(mapped_beacon_pair.start);

                                                    if !mapped_beacon_indices.is_full() {
                                                        mapped_beacon_indices
                                                            .insert(mapped_beacon_pair.end);
                                                    }

                                                    if mapped_beacon_indices.is_full() {
                                                        Some(transformation.clone())
                                                    } else {
                                                        None
                                                    }
                                                }
                                            })
                                    });

                                mapped_scanner_beacon_trios.clear();

                                transformation
                            })
                    }
                })
        }
    }

    #[inline(always)]
    fn minimum_overlapping_beacons(&self) -> usize {
        #[cfg(test)]
        let minimum_overlapping_beacons: usize = self.minimum_overlapping_beacons;

        #[cfg(not(test))]
        let minimum_overlapping_beacons: usize = Self::MINIMUM_OVERLAPPING_BEACONS;

        minimum_overlapping_beacons
    }

    fn sort_ivec3s(mut ivec3s: Vec<IVec3>) -> Vec<IVec3> {
        ivec3s.sort_by(Self::cmp_ivec3);

        ivec3s
    }

    fn cmp_ivec3(a: &IVec3, b: &IVec3) -> Ordering {
        a.as_ref().cmp(b.as_ref())
    }

    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        #[derive(Default)]
        struct ParseState {
            solution: Solution,
            start: usize,
        }

        let parse_state: RefCell<ParseState> = RefCell::new(ParseState::default());
        let (input, _) = cut(map_opt(
            many0_count(tuple((
                tag("--- scanner "),
                map_opt(map_res(digit1, usize::from_str), |scanner: usize| {
                    if scanner == parse_state.borrow().solution.scanners.len() {
                        Some(scanner)
                    } else {
                        None
                    }
                }),
                tag(" ---"),
                line_ending,
                many0_count(map(
                    tuple((
                        terminated(parse_integer::<i32>, tag(",")),
                        terminated(parse_integer::<i32>, tag(",")),
                        terminated(parse_integer::<i32>, opt(line_ending)),
                    )),
                    |(x, y, z)| {
                        parse_state
                            .borrow_mut()
                            .solution
                            .beacons
                            .push(IVec3 { x, y, z })
                    },
                )),
                cut(map_opt(opt(line_ending), |_| -> Option<()> {
                    let mut parse_state: RefMut<ParseState> = parse_state.borrow_mut();
                    let end: usize = parse_state.solution.beacons.len();
                    let scanner: Range<usize> = parse_state.start..end;

                    if scanner.len().saturating_sub(1_usize) <= u16::MAX as usize {
                        parse_state.solution.scanners.push(scanner);
                        parse_state.start = end;

                        Some(())
                    } else {
                        None
                    }
                })),
            ))),
            |_| -> Option<()> {
                let parse_state: Ref<ParseState> = parse_state.borrow();

                const MAX_SCANNER_OR_BEACONS_LEN: usize = u8::MAX as usize + 1_usize;

                if parse_state.solution.scanners.len() <= MAX_SCANNER_OR_BEACONS_LEN
                    && (0_usize..parse_state.solution.scanners.len())
                        .map(|scanner| parse_state.solution.beacons(scanner).len())
                        .max()
                        .unwrap_or_default()
                        <= MAX_SCANNER_OR_BEACONS_LEN
                {
                    Some(())
                } else {
                    None
                }
            },
        ))(input)?;

        Ok((input, parse_state.into_inner().solution))
    }

    #[cfg(test)]
    fn compute_overlap(
        &self,
        transformations: &Vec<Transformation>,
        scanner_a: usize,
        scanner_b: usize,
    ) -> Vec<IVec3> {
        let beacons_a: &[IVec3] = self.beacons(scanner_a);
        let beacons_b: &[IVec3] = self.beacons(scanner_b);
        let transformation_a: &Transformation = &transformations[scanner_a];
        let transformation_b: &Transformation = &transformations[scanner_b];
        let present_in_a: HashSet<IVec3> = beacons_a
            .iter()
            .map(|beacon| transformation_a.transform(*beacon))
            .collect();

        let mut overlap: Vec<IVec3> = beacons_b
            .iter()
            .map(|beacon| transformation_b.transform(*beacon))
            .filter(|beacon| present_in_a.contains(beacon))
            .collect();

        overlap.sort_by(Self::cmp_ivec3);

        overlap
    }

    #[allow(dead_code)]
    #[cfg(test)]
    fn try_compute_all_overlap_beacons<T, F: Fn(IVec3) -> T>(&self, f: F) -> Option<Vec<T>> {
        self.try_compute_transformations()
            .map(|transformations: Vec<Transformation>| {
                let mut present_beacons: HashMap<IVec3, u32> = HashMap::new();

                for scanner in 0_usize..self.scanners.len() {
                    let scanner_beacons: &[IVec3] = self.beacons(scanner);
                    let transformation: &Transformation = &transformations[scanner];

                    for beacon in scanner_beacons {
                        let beacon: IVec3 = transformation.transform(*beacon);

                        if let Some(count) = present_beacons.get_mut(&beacon) {
                            *count += 1_u32;
                        } else {
                            present_beacons.insert(beacon, 1_u32);
                        }
                    }
                }

                present_beacons
                    .iter()
                    .filter(|(_, count)| **count > 1_u32)
                    .map(|(beacon, _)| f(*beacon))
                    .collect()
            })
    }

    #[allow(dead_code)]
    #[cfg(test)]
    fn try_compute_all_beacons_with_sources(&self) -> Option<HashMap<IVec3, Vec<usize>>> {
        self.try_compute_transformations()
            .map(|transformations: Vec<Transformation>| {
                let mut present_beacons: HashMap<IVec3, Vec<usize>> = HashMap::new();

                for scanner in 0_usize..self.scanners.len() {
                    let scanner_beacons: &[IVec3] = self.beacons(scanner);
                    let transformation: &Transformation = &transformations[scanner];

                    for beacon in scanner_beacons {
                        let beacon: IVec3 = transformation.transform(*beacon);

                        if let Some(sources) = present_beacons.get_mut(&beacon) {
                            sources.push(scanner);
                        } else {
                            present_beacons.insert(beacon, vec![scanner]);
                        }
                    }
                }

                present_beacons
            })
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self {
            beacons: Default::default(),
            scanners: Default::default(),
            #[cfg(test)]
            minimum_overlapping_beacons: Self::MINIMUM_OVERLAPPING_BEACONS,
        }
    }
}

impl RunQuestions for Solution {
    fn q1_internal(&mut self, args: &QuestionArgs) {
        if args.verbose {
            let all_beacons: Option<Vec<IVec3>> = self.try_compute_all_beacons();

            dbg!(all_beacons.as_ref().map(Vec::len));
            dbg!(all_beacons);
        } else {
            dbg!(self.try_compute_all_beacons_count());
        }
    }

    fn q2_internal(&mut self, _args: &QuestionArgs) {
        dbg!(self.try_compute_largest_manhattan_distance());
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

    const SOLUTION_ARGS: &[(usize, &str)] = &[
        (
            Solution::MINIMUM_OVERLAPPING_BEACONS,
            "--- scanner 0 ---\n\
            404,-588,-901\n\
            528,-643,409\n\
            -838,591,734\n\
            390,-675,-793\n\
            -537,-823,-458\n\
            -485,-357,347\n\
            -345,-311,381\n\
            -661,-816,-575\n\
            -876,649,763\n\
            -618,-824,-621\n\
            553,345,-567\n\
            474,580,667\n\
            -447,-329,318\n\
            -584,868,-557\n\
            544,-627,-890\n\
            564,392,-477\n\
            455,729,728\n\
            -892,524,684\n\
            -689,845,-530\n\
            423,-701,434\n\
            7,-33,-71\n\
            630,319,-379\n\
            443,580,662\n\
            -789,900,-551\n\
            459,-707,401\n\
            \n\
            --- scanner 1 ---\n\
            686,422,578\n\
            605,423,415\n\
            515,917,-361\n\
            -336,658,858\n\
            95,138,22\n\
            -476,619,847\n\
            -340,-569,-846\n\
            567,-361,727\n\
            -460,603,-452\n\
            669,-402,600\n\
            729,430,532\n\
            -500,-761,534\n\
            -322,571,750\n\
            -466,-666,-811\n\
            -429,-592,574\n\
            -355,545,-477\n\
            703,-491,-529\n\
            -328,-685,520\n\
            413,935,-424\n\
            -391,539,-444\n\
            586,-435,557\n\
            -364,-763,-893\n\
            807,-499,-711\n\
            755,-354,-619\n\
            553,889,-390\n\
            \n\
            --- scanner 2 ---\n\
            649,640,665\n\
            682,-795,504\n\
            -784,533,-524\n\
            -644,584,-595\n\
            -588,-843,648\n\
            -30,6,44\n\
            -674,560,763\n\
            500,723,-460\n\
            609,671,-379\n\
            -555,-800,653\n\
            -675,-892,-343\n\
            697,-426,-610\n\
            578,704,681\n\
            493,664,-388\n\
            -671,-858,530\n\
            -667,343,800\n\
            571,-461,-707\n\
            -138,-166,112\n\
            -889,563,-600\n\
            646,-828,498\n\
            640,759,510\n\
            -630,509,768\n\
            -681,-892,-333\n\
            673,-379,-804\n\
            -742,-814,-386\n\
            577,-820,562\n\
            \n\
            --- scanner 3 ---\n\
            -589,542,597\n\
            605,-692,669\n\
            -500,565,-823\n\
            -660,373,557\n\
            -458,-679,-417\n\
            -488,449,543\n\
            -626,468,-788\n\
            338,-750,-386\n\
            528,-832,-391\n\
            562,-778,733\n\
            -938,-730,414\n\
            543,643,-506\n\
            -524,371,-870\n\
            407,773,750\n\
            -104,29,83\n\
            378,-903,-323\n\
            -778,-728,485\n\
            426,699,580\n\
            -438,-605,-362\n\
            -469,-447,-387\n\
            509,732,623\n\
            647,635,-688\n\
            -868,-804,481\n\
            614,-800,639\n\
            595,780,-596\n\
            \n\
            --- scanner 4 ---\n\
            727,592,562\n\
            -293,-554,779\n\
            441,611,-461\n\
            -714,465,-776\n\
            -743,427,-804\n\
            -660,-479,-426\n\
            832,-632,460\n\
            927,-485,-438\n\
            408,393,-506\n\
            466,436,-512\n\
            110,16,151\n\
            -258,-428,682\n\
            -393,719,612\n\
            -211,-452,876\n\
            808,-476,-593\n\
            -575,615,604\n\
            -485,667,467\n\
            -680,325,-822\n\
            -627,-443,-432\n\
            872,-547,-609\n\
            833,512,582\n\
            807,604,487\n\
            839,-516,451\n\
            891,-625,532\n\
            -652,-548,-490\n\
            30,-46,-14",
        ),
        (
            6_usize,
            "--- scanner 0 ---\n\
            -1,-1,1\n\
            -2,-2,2\n\
            -3,-3,3\n\
            -2,-3,1\n\
            5,6,-4\n\
            8,0,7\n\
            \n\
            --- scanner 1 ---\n\
            1,-1,1\n\
            2,-2,2\n\
            3,-3,3\n\
            2,-1,3\n\
            -5,4,-6\n\
            -8,-7,0\n\
            \n\
            --- scanner 2 ---\n\
            -1,-1,-1\n\
            -2,-2,-2\n\
            -3,-3,-3\n\
            -1,-3,-2\n\
            4,6,5\n\
            -7,0,8\n\
            \n\
            --- scanner 3 ---\n\
            1,1,-1\n\
            2,2,-2\n\
            3,3,-3\n\
            1,3,-2\n\
            -4,-6,5\n\
            7,0,8\n\
            \n\
            --- scanner 4 ---\n\
            1,1,1\n\
            2,2,2\n\
            3,3,3\n\
            3,1,2\n\
            -6,-4,-5\n\
            0,7,-8",
        ),
    ];

    macro_rules! solution {
        [ $minimum_overlapping_beacons:expr, $( [ $( ($x:expr, $y:expr, $z:expr), )* ], )* ] => { {
            let beacons: Vec<IVec3> = vec![ $( $(
                IVec3::new($x, $y, $z),
            )* )* ];
            let mut scanners: Vec<Range<usize>> = Vec::new();
            let mut start: usize = 0_usize;
            const fn gulp(_x: i32) -> () {}

            for beacons_slice in [ $(
                [ $( gulp($x), )* ].as_slice(),
            )* ] {
                let end: usize = start + beacons_slice.len();

                scanners.push(start..end);
                start = end;
            }

            Solution {
                beacons,
                scanners,

                #[cfg(test)]
                minimum_overlapping_beacons: $minimum_overlapping_beacons,
            }
        } }
    }

    const SOLUTION_FNS: &[fn() -> Solution] = &[
        || {
            solution![
                Solution::MINIMUM_OVERLAPPING_BEACONS,
                [
                    (404, -588, -901),
                    (528, -643, 409),
                    (-838, 591, 734),
                    (390, -675, -793),
                    (-537, -823, -458),
                    (-485, -357, 347),
                    (-345, -311, 381),
                    (-661, -816, -575),
                    (-876, 649, 763),
                    (-618, -824, -621),
                    (553, 345, -567),
                    (474, 580, 667),
                    (-447, -329, 318),
                    (-584, 868, -557),
                    (544, -627, -890),
                    (564, 392, -477),
                    (455, 729, 728),
                    (-892, 524, 684),
                    (-689, 845, -530),
                    (423, -701, 434),
                    (7, -33, -71),
                    (630, 319, -379),
                    (443, 580, 662),
                    (-789, 900, -551),
                    (459, -707, 401),
                ],
                [
                    (686, 422, 578),
                    (605, 423, 415),
                    (515, 917, -361),
                    (-336, 658, 858),
                    (95, 138, 22),
                    (-476, 619, 847),
                    (-340, -569, -846),
                    (567, -361, 727),
                    (-460, 603, -452),
                    (669, -402, 600),
                    (729, 430, 532),
                    (-500, -761, 534),
                    (-322, 571, 750),
                    (-466, -666, -811),
                    (-429, -592, 574),
                    (-355, 545, -477),
                    (703, -491, -529),
                    (-328, -685, 520),
                    (413, 935, -424),
                    (-391, 539, -444),
                    (586, -435, 557),
                    (-364, -763, -893),
                    (807, -499, -711),
                    (755, -354, -619),
                    (553, 889, -390),
                ],
                [
                    (649, 640, 665),
                    (682, -795, 504),
                    (-784, 533, -524),
                    (-644, 584, -595),
                    (-588, -843, 648),
                    (-30, 6, 44),
                    (-674, 560, 763),
                    (500, 723, -460),
                    (609, 671, -379),
                    (-555, -800, 653),
                    (-675, -892, -343),
                    (697, -426, -610),
                    (578, 704, 681),
                    (493, 664, -388),
                    (-671, -858, 530),
                    (-667, 343, 800),
                    (571, -461, -707),
                    (-138, -166, 112),
                    (-889, 563, -600),
                    (646, -828, 498),
                    (640, 759, 510),
                    (-630, 509, 768),
                    (-681, -892, -333),
                    (673, -379, -804),
                    (-742, -814, -386),
                    (577, -820, 562),
                ],
                [
                    (-589, 542, 597),
                    (605, -692, 669),
                    (-500, 565, -823),
                    (-660, 373, 557),
                    (-458, -679, -417),
                    (-488, 449, 543),
                    (-626, 468, -788),
                    (338, -750, -386),
                    (528, -832, -391),
                    (562, -778, 733),
                    (-938, -730, 414),
                    (543, 643, -506),
                    (-524, 371, -870),
                    (407, 773, 750),
                    (-104, 29, 83),
                    (378, -903, -323),
                    (-778, -728, 485),
                    (426, 699, 580),
                    (-438, -605, -362),
                    (-469, -447, -387),
                    (509, 732, 623),
                    (647, 635, -688),
                    (-868, -804, 481),
                    (614, -800, 639),
                    (595, 780, -596),
                ],
                [
                    (727, 592, 562),
                    (-293, -554, 779),
                    (441, 611, -461),
                    (-714, 465, -776),
                    (-743, 427, -804),
                    (-660, -479, -426),
                    (832, -632, 460),
                    (927, -485, -438),
                    (408, 393, -506),
                    (466, 436, -512),
                    (110, 16, 151),
                    (-258, -428, 682),
                    (-393, 719, 612),
                    (-211, -452, 876),
                    (808, -476, -593),
                    (-575, 615, 604),
                    (-485, 667, 467),
                    (-680, 325, -822),
                    (-627, -443, -432),
                    (872, -547, -609),
                    (833, 512, 582),
                    (807, 604, 487),
                    (839, -516, 451),
                    (891, -625, 532),
                    (-652, -548, -490),
                    (30, -46, -14),
                ],
            ]
        },
        || {
            solution![
                6_usize,
                [
                    (-1, -1, 1),
                    (-2, -2, 2),
                    (-3, -3, 3),
                    (-2, -3, 1),
                    (5, 6, -4),
                    (8, 0, 7),
                ],
                [
                    (1, -1, 1),
                    (2, -2, 2),
                    (3, -3, 3),
                    (2, -1, 3),
                    (-5, 4, -6),
                    (-8, -7, 0),
                ],
                [
                    (-1, -1, -1),
                    (-2, -2, -2),
                    (-3, -3, -3),
                    (-1, -3, -2),
                    (4, 6, 5),
                    (-7, 0, 8),
                ],
                [
                    (1, 1, -1),
                    (2, 2, -2),
                    (3, 3, -3),
                    (1, 3, -2),
                    (-4, -6, 5),
                    (7, 0, 8),
                ],
                [
                    (1, 1, 1),
                    (2, 2, 2),
                    (3, 3, 3),
                    (3, 1, 2),
                    (-6, -4, -5),
                    (0, 7, -8),
                ],
            ]
        },
    ];

    fn solution(index: usize) -> &'static Solution {
        static ONCE_LOCKS: OnceLock<Vec<OnceLock<Solution>>> = OnceLock::new();

        ONCE_LOCKS.get_or_init(|| {
            let mut once_locks: Vec<OnceLock<Solution>> = Vec::with_capacity(SOLUTION_FNS.len());

            for _ in 0_usize..SOLUTION_FNS.len() {
                once_locks.push(OnceLock::new());
            }

            once_locks
        })[index]
            .get_or_init(SOLUTION_FNS[index])
    }

    fn as_sorted_ivec3s(arrays: &[[i32; 3_usize]]) -> Vec<IVec3> {
        Solution::sort_ivec3s(arrays.iter().copied().map(IVec3::from).collect())
    }

    #[test]
    fn test_try_from_str() {
        for (index, (minimum_overlapping_beacons, solution_str)) in
            SOLUTION_ARGS.iter().copied().enumerate()
        {
            assert_eq!(
                Solution::try_from(solution_str)
                    .map(|mut solution| {
                        solution.minimum_overlapping_beacons = minimum_overlapping_beacons;
                        solution
                    })
                    .as_ref(),
                Ok(solution(index))
            );
        }
    }

    #[test]
    fn test_delta_new() {
        let solution: &Solution = solution(1_usize);
        let beacons: &[IVec3] = &solution.beacons[solution.scanners[0_usize].clone()];

        for (real_delta, expected_delta) in (0_usize..beacons.len() - 1_usize)
            .flat_map(|beacon_a_index| {
                (beacon_a_index + 1_usize..beacons.len())
                    .map(move |beacon_b_index| (beacon_a_index, beacon_b_index))
            })
            .map(|(beacon_a_index, beacon_b_index)| {
                NormDelta::try_from_start_and_end(beacons[beacon_a_index], beacons[beacon_b_index])
            })
            .zip([
                [1, 1, 1],
                [2, 2, 2],
                [2, 1, 0],
                [7, 6, 5],
                [9, 1, 6],
                [1, 1, 1],
                [1, 1, 0],
                [8, 7, 6],
                [10, 2, 5],
                [2, 1, 0],
                [9, 8, 7],
                [11, 3, 4],
                [9, 7, 5],
                [10, 3, 6],
                [11, 6, 3],
            ])
        {
            assert_eq!(real_delta, Some(NormDelta(expected_delta)));
        }
    }

    #[test]
    fn test_transformation_transform() {
        let start_a: IVec3 = IVec3::new(1, 1, 1);
        let end_a: IVec3 = IVec3::new(3, 4, 2);
        let start_b: IVec3 = IVec3::new(-1, 1, -1);
        let end_b: IVec3 = IVec3::new(-4, 0, 1);
        let transformation: Transformation =
            Transformation::try_from_beacons(start_a, end_a, start_b, end_b).unwrap();

        assert_eq!(transformation.transform(start_a), start_b);
        assert_eq!(transformation.transform(end_a), end_b);
    }

    #[test]
    fn test_compute_overlap() {
        {
            let solution: &Solution = solution(1_usize);
            let transformations: Vec<Transformation> =
                solution.try_compute_transformations().unwrap();
            let beacons: Vec<IVec3> = Solution::sort_ivec3s(solution.beacons(0_usize).into());

            for scanner_a in 0_usize..solution.scanners.len() - 1_usize {
                for scanner_b in scanner_a + 1_usize..solution.scanners.len() {
                    assert_eq!(
                        solution.compute_overlap(&transformations, scanner_a, scanner_b),
                        beacons
                    );
                }
            }
        }

        {
            let solution: &Solution = solution(0_usize);
            let transformations: Vec<Transformation> =
                solution.try_compute_transformations().unwrap();

            for (scanner_a, scanner_b, arrays) in [
                (
                    0_usize,
                    1_usize,
                    [
                        [-618, -824, -621],
                        [-537, -823, -458],
                        [-447, -329, 318],
                        [404, -588, -901],
                        [544, -627, -890],
                        [528, -643, 409],
                        [-661, -816, -575],
                        [390, -675, -793],
                        [423, -701, 434],
                        [-345, -311, 381],
                        [459, -707, 401],
                        [-485, -357, 347],
                    ]
                    .as_ref(),
                ),
                (
                    1_usize,
                    4_usize,
                    [
                        [459, -707, 401],
                        [-739, -1745, 668],
                        [-485, -357, 347],
                        [432, -2009, 850],
                        [528, -643, 409],
                        [423, -701, 434],
                        [-345, -311, 381],
                        [408, -1815, 803],
                        [534, -1912, 768],
                        [-687, -1600, 576],
                        [-447, -329, 318],
                        [-635, -1737, 486],
                    ]
                    .as_ref(),
                ),
            ] {
                assert_eq!(
                    solution.compute_overlap(&transformations, scanner_a, scanner_b),
                    as_sorted_ivec3s(arrays)
                )
            }
        }
    }

    #[test]
    fn test_try_compute_all_beacons() {
        const SOLUTION_0_ARRAYS: &[[i32; 3_usize]] = &[
            [-892, 524, 684],
            [-876, 649, 763],
            [-838, 591, 734],
            [-789, 900, -551],
            [-739, -1745, 668],
            [-706, -3180, -659],
            [-697, -3072, -689],
            [-689, 845, -530],
            [-687, -1600, 576],
            [-661, -816, -575],
            [-654, -3158, -753],
            [-635, -1737, 486],
            [-631, -672, 1502],
            [-624, -1620, 1868],
            [-620, -3212, 371],
            [-618, -824, -621],
            [-612, -1695, 1788],
            [-601, -1648, -643],
            [-584, 868, -557],
            [-537, -823, -458],
            [-532, -1715, 1894],
            [-518, -1681, -600],
            [-499, -1607, -770],
            [-485, -357, 347],
            [-470, -3283, 303],
            [-456, -621, 1527],
            [-447, -329, 318],
            [-430, -3130, 366],
            [-413, -627, 1469],
            [-345, -311, 381],
            [-36, -1284, 1171],
            [-27, -1108, -65],
            [7, -33, -71],
            [12, -2351, -103],
            [26, -1119, 1091],
            [346, -2985, 342],
            [366, -3059, 397],
            [377, -2827, 367],
            [390, -675, -793],
            [396, -1931, -563],
            [404, -588, -901],
            [408, -1815, 803],
            [423, -701, 434],
            [432, -2009, 850],
            [443, 580, 662],
            [455, 729, 728],
            [456, -540, 1869],
            [459, -707, 401],
            [465, -695, 1988],
            [474, 580, 667],
            [496, -1584, 1900],
            [497, -1838, -617],
            [527, -524, 1933],
            [528, -643, 409],
            [534, -1912, 768],
            [544, -627, -890],
            [553, 345, -567],
            [564, 392, -477],
            [568, -2007, -577],
            [605, -1665, 1952],
            [612, -1593, 1893],
            [630, 319, -379],
            [686, -3108, -505],
            [776, -3184, -501],
            [846, -3110, -434],
            [1135, -1161, 1235],
            [1243, -1093, 1063],
            [1660, -552, 429],
            [1693, -557, 386],
            [1735, -437, 1738],
            [1749, -1800, 1813],
            [1772, -405, 1572],
            [1776, -675, 371],
            [1779, -442, 1789],
            [1780, -1548, 337],
            [1786, -1538, 337],
            [1847, -1591, 415],
            [1889, -1729, 1762],
            [1994, -1805, 1792],
        ];

        for (solution_index, arrays) in [
            (0_usize, SOLUTION_0_ARRAYS),
            (
                1_usize,
                [
                    [-1, -1, 1],
                    [-2, -2, 2],
                    [-3, -3, 3],
                    [-2, -3, 1],
                    [5, 6, -4],
                    [8, 0, 7],
                ]
                .as_ref(),
            ),
        ] {
            let solution: &Solution = solution(solution_index);

            assert_eq!(solution.try_compute_all_beacons_count(), Some(arrays.len()));
            assert_eq!(
                solution.try_compute_all_beacons(),
                Some(as_sorted_ivec3s(arrays))
            );
        }
    }

    #[test]
    fn test_try_compute_largest_manhattan_distance() {
        assert_eq!(
            solution(0_usize).try_compute_largest_manhattan_distance(),
            Some(3621_i32)
        );
    }
}
