use {
    crate::{CliqueIterator, CliqueState, UserCliqueIteratorTrait},
    arrayvec::ArrayVec,
    bitvec::prelude::*,
};

type SspBitArrayRaw = u32;

const MAX_SSP_SET_CARDINALITY: usize = SspBitArrayRaw::BITS as usize;

pub type SspBitArray = BitArr!(for MAX_SSP_SET_CARDINALITY, in SspBitArrayRaw);
pub type SspInputArrayVec = ArrayVec<i32, MAX_SSP_SET_CARDINALITY>;

#[derive(Clone, Default)]
pub struct SspUserCliqueState {
    subset_sum: i32,
}

pub type SspCliqueState = CliqueState<SspBitArray, SspUserCliqueState>;

pub struct SspInput {
    /// Integer input values, sorted in increasing order.
    pub values: SspInputArrayVec,
    pub target_sum: i32,
}

impl SspInput {
    fn is_valid(&self) -> bool {
        let mut values_clone: SspInputArrayVec = self.values.clone();

        values_clone.sort();

        values_clone == self.values
    }
}

impl<'i> UserCliqueIteratorTrait<SspBitArrayRaw> for &'i SspInput {
    type BitArray = SspBitArray;

    type UserCliqueState = SspUserCliqueState;

    fn vertex_count(&self) -> usize {
        self.values.len()
    }

    fn integrate_vertex(
        &self,
        vertex_index: usize,
        _clique: &Self::BitArray,
        inout_user_clique_state: &mut Self::UserCliqueState,
    ) {
        inout_user_clique_state.subset_sum += self.values[vertex_index];
    }

    fn is_clique_state_valid(
        &self,
        _clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
    ) -> bool {
        user_clique_state.subset_sum <= self.target_sum
    }

    fn visit_clique(
        &mut self,
        _clique_state: &crate::CliqueState<Self::BitArray, Self::UserCliqueState>,
    ) {
    }

    fn get_neighbors(
        &self,
        vertex_index: usize,
        _clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
        out_neighbors: &mut Self::BitArray,
    ) {
        out_neighbors.fill(false);

        let target_delta: i32 = self.target_sum - user_clique_state.subset_sum;

        self.values
            .iter()
            .copied()
            .enumerate()
            .skip(vertex_index + 1_usize)
            .try_for_each(|(neighbor_vertex_index, value)| {
                (value <= target_delta).then(|| {
                    out_neighbors.set(neighbor_vertex_index, true);
                })
            });
    }

    fn should_visit_neighbors(
        &self,
        _clique: &Self::BitArray,
        _user_clique_state: &Self::UserCliqueState,
        neighbors: &Self::BitArray,
    ) -> bool {
        !neighbors.is_empty()
    }
}

pub type SspCliqueIterator<'i> =
    CliqueIterator<SspBitArray, SspUserCliqueState, &'i SspInput, SspBitArrayRaw, Lsb0>;

pub struct SspSubsetWithSumIterator<'i> {
    pub target_sum: i32,
    pub clique_iterator: SspCliqueIterator<'i>,
}

impl<'i> Iterator for SspSubsetWithSumIterator<'i> {
    type Item = SspBitArray;

    fn next(&mut self) -> Option<Self::Item> {
        self.clique_iterator.find_map(|clique_state| {
            (clique_state.user_clique_state.subset_sum == self.target_sum)
                .then_some(clique_state.clique)
        })
    }
}

impl<'i> IntoIterator for &'i SspInput {
    type Item = SspBitArray;

    type IntoIter = SspSubsetWithSumIterator<'i>;

    fn into_iter(self) -> Self::IntoIter {
        assert!(self.is_valid());

        let target_sum: i32 = self.target_sum;

        Self::IntoIter {
            target_sum,
            clique_iterator: self.iter(),
        }
    }
}
