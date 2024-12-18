use {
    crate::*,
    num::{FromPrimitive, NumCast, PrimInt},
    std::fmt::{Debug, Formatter, Result as FmtResult},
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
    pub trait IndexRawTrait where Self: PrimInt + NumCast + FromPrimitive + IndexRawConstsTrait {}
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Index<IndexRaw: IndexRawTrait>(IndexRaw);

impl<IndexRaw: IndexRawTrait> Index<IndexRaw> {
    pub const INVALID: Self = Self::invalid();

    pub const fn invalid() -> Self {
        Self(IndexRaw::INVALID)
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

impl<IndexRaw: IndexRawTrait + Debug> Debug for Index<IndexRaw> {
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

impl<IndexRaw: IndexRawTrait> From<Index<IndexRaw>> for usize {
    fn from(value: Index<IndexRaw>) -> Self {
        value.get()
    }
}
