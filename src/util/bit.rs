use {
    bitvec::{
        prelude::*,
        view::{BitView, BitViewSized},
    },
    std::{
        mem::size_of,
        ops::{Deref, DerefMut, Index, IndexMut, Range},
    },
};

pub trait BitArrayArguments {
    type Array;
    type Ordering;
}

impl<A: BitViewSized, O: BitOrder> BitArrayArguments for BitArray<A, O> {
    type Array = A;
    type Ordering = O;
}

#[macro_export]
macro_rules! const_bitarr_lsb0_get {
    ($bitarr:expr, $bitarr_type:ty; $index:expr) => {{
        type Store = <<$bitarr_type as BitArrayArguments>::Array as bitvec::view::BitView>::Store;

        let block_index: usize = $index / Store::BITS as usize;
        let bit_index: usize = $index % Store::BITS as usize;

        ($bitarr.data[block_index] & ((1 as Store) << bit_index)) != 0 as Store
    }};
}

#[macro_export]
macro_rules! const_bitarr_lsb0_set {
    ($bitarr:expr, $bitarr_type:ty; $index:expr, $value:expr) => {{
        type Store = <<$bitarr_type as BitArrayArguments>::Array as bitvec::view::BitView>::Store;

        let block_index: usize = $index / Store::BITS as usize;
        let bit_index: usize = $index % Store::BITS as usize;

        if $value {
            $bitarr.data[block_index] |= (1 as Store) << bit_index
        } else {
            $bitarr.data[block_index] &= !((1 as Store) << bit_index)
        }
    }};
}

pub type BitFieldArrayRaw = [u64; 1_usize];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BitFieldArray<A: BitViewSized = BitFieldArrayRaw>(pub BitArray<A, Lsb0>);

impl<A: BitViewSized> BitFieldArray<A> {
    pub fn field(
        &self,
        field_index: usize,
        bits_per_field: usize,
    ) -> &BitSlice<<A as BitView>::Store> {
        self.fields(field_index..field_index + 1_usize, bits_per_field)
    }

    pub fn fields(
        &self,
        field_range: Range<usize>,
        bits_per_field: usize,
    ) -> &BitSlice<<A as BitView>::Store> {
        &self.0[field_range.start * bits_per_field..field_range.end * bits_per_field]
    }

    pub fn field_mut(
        &mut self,
        field_index: usize,
        bits_per_field: usize,
    ) -> &mut BitSlice<<A as BitView>::Store> {
        self.fields_mut(field_index..field_index + 1_usize, bits_per_field)
    }

    pub fn fields_mut(
        &mut self,
        field_range: Range<usize>,
        bits_per_field: usize,
    ) -> &mut BitSlice<<A as BitView>::Store> {
        &mut self.0[field_range.start * bits_per_field..field_range.end * bits_per_field]
    }
}

pub const BIT_FIELD_ARRAY_BITS: usize = size_of::<BitFieldArray>() * u8::BITS as usize;

// 4 bits per field means 16 values in the [0,16) range in a `BitFieldArrayRaw`
const BIT_FIELD_ARRAY_SIZED_BITS_PER_FIELD: usize = 4_usize;

#[derive(Default)]
pub struct BitFieldArraySized<
    A: BitViewSized = BitFieldArrayRaw,
    const BITS_PER_FIELD: usize = BIT_FIELD_ARRAY_SIZED_BITS_PER_FIELD,
>(BitFieldArray<A>);

impl<A: BitViewSized, const BITS_PER_FIELD: usize> Deref for BitFieldArraySized<A, BITS_PER_FIELD> {
    type Target = BitFieldArray<A>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A: BitViewSized, const BITS_PER_FIELD: usize> DerefMut
    for BitFieldArraySized<A, BITS_PER_FIELD>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<A: BitViewSized, const BITS_PER_FIELD: usize> Index<usize>
    for BitFieldArraySized<A, BITS_PER_FIELD>
{
    type Output = BitSlice<<A as BitView>::Store>;

    fn index(&self, index: usize) -> &Self::Output {
        &self[index..index + 1_usize]
    }
}

impl<A: BitViewSized, const BITS_PER_FIELD: usize> Index<Range<usize>>
    for BitFieldArraySized<A, BITS_PER_FIELD>
{
    type Output = BitSlice<<A as BitView>::Store>;

    fn index(&self, index: Range<usize>) -> &Self::Output {
        self.fields(index, BITS_PER_FIELD)
    }
}

impl<A: BitViewSized, const BITS_PER_FIELD: usize> IndexMut<usize>
    for BitFieldArraySized<A, BITS_PER_FIELD>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self[index..index + 1_usize]
    }
}

impl<A: BitViewSized, const BITS_PER_FIELD: usize> IndexMut<Range<usize>>
    for BitFieldArraySized<A, BITS_PER_FIELD>
{
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        self.fields_mut(index, BITS_PER_FIELD)
    }
}

pub const fn bits_to_store(value: usize) -> usize {
    match value.count_ones() {
        0_u32 => 1_usize,
        _ => value.ilog2() as usize + 1_usize,
    }
}

pub const fn bits_per_field(value_count: usize) -> usize {
    if value_count == 0_usize {
        0_usize
    } else {
        bits_to_store(value_count - 1_usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_to_store() {
        assert_eq!(bits_to_store(0b_0_usize), 1_usize);
        assert_eq!(bits_to_store(0b_1_usize), 1_usize);
        assert_eq!(bits_to_store(0b_10_usize), 2_usize);
        assert_eq!(bits_to_store(0b_11_usize), 2_usize);
        assert_eq!(bits_to_store(0b_100_usize), 3_usize);
        assert_eq!(bits_to_store(0b_101_usize), 3_usize);
        assert_eq!(bits_to_store(0b_110_usize), 3_usize);
        assert_eq!(bits_to_store(0b_111_usize), 3_usize);
        assert_eq!(bits_to_store(0b_1000_usize), 4_usize);
    }
}
