use {
    crate::*,
    num::{FromPrimitive, NumCast, PrimInt},
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        hash::Hash,
    },
};

pub trait IndexRawConstsTrait {
    const INVALID: Self;
}

macro_rules! impl_table_index_const_trait {
    ( $( $index:ty, )* ) => { $(
        impl IndexRawConstsTrait for $index {
            const INVALID: Self = !0;
        }
    )* };
}

impl_table_index_const_trait!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,);

define_super_trait! {
    pub trait IndexRawTrait where Self: Debug + Default + Hash + PrimInt + NumCast + FromPrimitive + IndexRawConstsTrait {}
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Index<IndexRaw: IndexRawTrait>(IndexRaw);

impl<IndexRaw: IndexRawTrait> Index<IndexRaw> {
    pub const INVALID: Self = Self::invalid();

    pub const fn invalid() -> Self {
        Self(IndexRaw::INVALID)
    }

    pub const fn new_raw(index: IndexRaw) -> Self {
        Self(index)
    }

    pub fn new(index: usize) -> Self {
        Self(IndexRaw::from_usize(index).unwrap())
    }

    pub fn is_valid(self) -> bool {
        self != Self::invalid()
    }

    pub fn get(self) -> usize {
        assert!(self.is_valid());

        self.0.to_usize().unwrap()
    }

    pub fn opt(self) -> Option<Self> {
        self.is_valid().then_some(self)
    }
}

pub trait IndexTrait
where
    Self: Clone
        + Copy
        + Debug
        + Default
        + Eq
        + From<usize>
        + From<Option<Self>>
        + Into<usize>
        + Hash
        + Ord
        + PartialEq
        + PartialOrd
        + Sized,
{
    type Raw: IndexRawTrait;

    const INVALID: Self;

    fn invalid() -> Self;

    fn new_raw(index: Self::Raw) -> Self;

    fn new(index: usize) -> Self;

    fn is_valid(self) -> bool;

    fn get(self) -> usize;

    fn opt(self) -> Option<Self>;
}

impl<IndexRaw: IndexRawTrait> IndexTrait for Index<IndexRaw> {
    type Raw = IndexRaw;

    const INVALID: Self = Index::INVALID;

    fn invalid() -> Self {
        Index::invalid()
    }

    fn new_raw(index: Self::Raw) -> Self {
        Index::new_raw(index)
    }

    fn new(index: usize) -> Self {
        Index::new(index)
    }

    fn is_valid(self) -> bool {
        Index::is_valid(self)
    }

    fn get(self) -> usize {
        Index::get(self)
    }

    fn opt(self) -> Option<Self> {
        Index::opt(self)
    }
}

impl<IndexRaw: IndexRawTrait> Debug for Index<IndexRaw> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.is_valid() {
            f.write_fmt(format_args!("{:?}", self.0))
        } else {
            f.write_str("<invalid>")
        }
    }
}

impl<IndexRaw: IndexRawTrait> Default for Index<IndexRaw> {
    fn default() -> Self {
        Self::invalid()
    }
}

impl<IndexRaw: IndexRawTrait> From<usize> for Index<IndexRaw> {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl<IndexRaw: IndexRawTrait> From<Option<Index<IndexRaw>>> for Index<IndexRaw> {
    fn from(value: Option<Index<IndexRaw>>) -> Self {
        value.unwrap_or_default()
    }
}

impl<IndexRaw: IndexRawTrait> From<Index<IndexRaw>> for usize {
    fn from(value: Index<IndexRaw>) -> Self {
        value.get()
    }
}
