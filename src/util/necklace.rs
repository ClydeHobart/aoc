use {crate::*, arrayvec::ArrayVec, bitvec::prelude::*, std::iter::from_fn};

const fn max_beads() -> usize {
    let mut beads: usize = 0_usize;
    let mut next_beads: usize;

    while {
        next_beads = beads + 1_usize;

        bits_per_field(next_beads) * next_beads <= BIT_FIELD_ARRAY_BITS
    } {
        beads = next_beads;
    }

    beads
}

pub const MAX_BEADS: usize = max_beads();

type BeadBitArr = BitArr!(for MAX_BEADS, in u16);

#[derive(Clone, Default)]
struct DistinctBeadNecklaceIteratorStackFrame {
    bit_field_array: BitFieldArray,
    absent_beads: BeadBitArr,
    remaining_absent_beads: BeadBitArr,
}

impl DistinctBeadNecklaceIteratorStackFrame {
    fn new(beads: usize) -> Self {
        let mut frame: Self = Default::default();

        frame.absent_beads[..beads].fill(true);
        frame.remaining_absent_beads[..beads].fill(true);

        frame
    }

    fn yields_valid_turnover_necklaces(&self, beads: usize, bits_per_bead: usize) -> bool {
        match beads {
            0_usize..=2_usize => true,
            3_usize => {
                self.absent_beads[..3_usize].count_zeros() < 2_usize
                    || self
                        .bit_field_array
                        .field(1_usize, bits_per_bead)
                        .load::<usize>()
                        == 1_usize
            }
            _ => {
                self.absent_beads[..beads].count_zeros() < 3_usize ||
                    // If everyone's present, assume it's valid, as we would've blocked it earlier
                    // otherwise
                    !self.absent_beads.any() ||
                    self.bit_field_array.field(1_usize, bits_per_bead).load::<usize>() <
                    beads - 1_usize - self.absent_beads[..beads].trailing_zeros()
            }
        }
    }

    fn try_next(
        &mut self,
        beads: usize,
        bits_per_bead: usize,
        bead: usize,
        is_turnover_necklace: bool,
    ) -> Option<Self> {
        assert!(self.absent_beads[bead]);
        assert!(self.remaining_absent_beads[bead]);

        let mut next: Self = self.clone();

        self.remaining_absent_beads.set(bead, false);
        next.bit_field_array
            .field_mut(self.absent_beads[..beads].count_zeros(), bits_per_bead)
            .store(bead);
        next.absent_beads.set(bead, false);
        next.remaining_absent_beads = next.absent_beads;

        (!is_turnover_necklace || next.yields_valid_turnover_necklaces(beads, bits_per_bead))
            .then_some(next)
    }
}

pub fn iter_distinct_bead_necklaces(
    beads: usize,
    is_turnover_necklace: bool,
) -> impl Iterator<Item = BitFieldArray> {
    let bits_per_bead: usize = bits_per_field(beads);

    let mut stack: ArrayVec<DistinctBeadNecklaceIteratorStackFrame, MAX_BEADS> = ArrayVec::new();

    if beads > 0_usize {
        stack.push(
            DistinctBeadNecklaceIteratorStackFrame::new(beads)
                .try_next(beads, bits_per_bead, 0_usize, is_turnover_necklace)
                .unwrap(),
        );
    }

    (beads > 0_usize)
        .then(|| {
            from_fn(move || {
                while (1_usize..beads).contains(&stack.len()) {
                    let mut curr_frame: DistinctBeadNecklaceIteratorStackFrame =
                        stack.pop().unwrap();

                    if curr_frame.remaining_absent_beads.any() {
                        let next_frame: Option<DistinctBeadNecklaceIteratorStackFrame> = curr_frame
                            .try_next(
                                beads,
                                bits_per_bead,
                                curr_frame.remaining_absent_beads.leading_zeros(),
                                is_turnover_necklace,
                            );

                        stack.push(curr_frame);

                        if let Some(next_frame) = next_frame {
                            stack.push(next_frame);
                        }
                    }
                }

                (stack.len() == beads).then(|| stack.pop().unwrap().bit_field_array)
            })
        })
        .into_iter()
        .flatten()
        .chain((beads == 0_usize).then(Default::default))
}

pub fn iter_beads(
    necklace: &BitFieldArray,
    beads: usize,
    bits_per_bead: usize,
) -> impl Iterator<Item = usize> + '_ {
    (0_usize..beads).map(move |bead| necklace.field(bead, bits_per_bead).load())
}

pub fn beads_from_necklace(
    necklace: &BitFieldArray,
    beads: usize,
    bits_per_bead: usize,
) -> ArrayVec<u8, MAX_BEADS> {
    iter_beads(necklace, beads, bits_per_bead)
        .map(|bead| bead as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_distinct_bead_necklaces_count(beads: usize) -> usize {
        factorial(beads.saturating_sub(1_usize))
    }

    fn expected_distinct_bead_turnover_necklaces_count(beads: usize) -> usize {
        (expected_distinct_bead_necklaces_count(beads) / 2_usize).max(1_usize)
    }

    #[test]
    fn test_iter_distinct_bead_necklaces() {
        for beads in 0_usize..=8_usize {
            let bits_per_bead: usize = bits_per_field(beads);

            assert_eq!(
                iter_distinct_bead_necklaces(beads, false).count(),
                expected_distinct_bead_necklaces_count(beads),
                "beads: {beads}, no turnovers, necklaces: {:?}",
                iter_distinct_bead_necklaces(beads, false)
                    .map(|necklace| beads_from_necklace(&necklace, beads, bits_per_bead))
                    .collect::<Vec<ArrayVec<u8, MAX_BEADS>>>()
            );
            assert_eq!(
                iter_distinct_bead_necklaces(beads, true).count(),
                expected_distinct_bead_turnover_necklaces_count(beads),
                "beads: {beads}, yes turnovers, necklaces: {:?}",
                iter_distinct_bead_necklaces(beads, true)
                    .map(|necklace| beads_from_necklace(&necklace, beads, bits_per_bead))
                    .collect::<Vec<ArrayVec<u8, MAX_BEADS>>>()
            );
        }
    }
}
