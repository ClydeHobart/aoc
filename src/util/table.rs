use {
    crate::*,
    num::{FromPrimitive, NumCast, PrimInt},
    std::mem::transmute,
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        marker::PhantomData,
    },
};

define_super_trait! {
    pub trait TableIdTrait where Self: Clone + Eq + Ord + PartialEq + PartialOrd {}
}

pub trait TableIndexConstTrait {
    const INVALID: Self;
}

macro_rules! impl_table_index_const_trait {
    ( $( $index:ty, )* ) => { $(
        impl TableIndexConstTrait for $index {
            const INVALID: Self = !0;
        }
    )* };
}

impl_table_index_const_trait!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,);

define_super_trait! {
    pub trait TableIndexTrait where Self: PrimInt + NumCast + FromPrimitive + TableIndexConstTrait {}
}

#[cfg_attr(test, derive(PartialEq))]
pub struct TableElement<Id, Data> {
    pub id: Id,
    pub data: Data,
}

impl<Id: Debug, Data: Debug> Debug for TableElement<Id, Data> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        (&self.id, &self.data).fmt(f)
    }
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TableIndex<Index: TableIndexTrait>(Index);

impl<Index: TableIndexTrait> TableIndex<Index> {
    pub const INVALID: Self = Self::invalid();

    pub const fn invalid() -> Self {
        Self(Index::INVALID)
    }

    pub fn new(index: usize) -> Self {
        Self(Index::from_usize(index).unwrap())
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

impl<Index: TableIndexTrait + Debug> Debug for TableIndex<Index> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.is_valid() {
            f.write_fmt(format_args!("{:?}", self.0))
        } else {
            f.write_str("<invalid>")
        }
    }
}

impl<Index: TableIndexTrait> Default for TableIndex<Index> {
    fn default() -> Self {
        Self::invalid()
    }
}

impl<Index: TableIndexTrait> From<usize> for TableIndex<Index> {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl<Index: TableIndexTrait> From<TableIndex<Index>> for usize {
    fn from(value: TableIndex<Index>) -> Self {
        value.get()
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Default)]
pub struct Table<Id: TableIdTrait, Data, Index: TableIndexTrait = usize> {
    table: Vec<TableElement<Id, Data>>,
    _index: PhantomData<Index>,
}

impl<Id: TableIdTrait, Data, Index: TableIndexTrait> Table<Id, Data, Index> {
    pub fn new() -> Self
    where
        Self: Default,
    {
        Self::default()
    }

    pub fn as_slice(&self) -> &[TableElement<Id, Data>] {
        &self.table
    }

    /// It is the user's responsibility to ensure either the IDs aren't modified, or that other
    /// consumers don't care about the IDs being modified.
    pub fn as_slice_mut(&mut self) -> &mut [TableElement<Id, Data>] {
        &mut self.table
    }

    pub fn find_or_add_index(&mut self, id: &Id) -> TableIndex<Index>
    where
        Data: Default,
    {
        let mut index: TableIndex<Index> = self.find_index(id);

        if !index.is_valid() {
            index = self.table.len().into();
            self.table.push(TableElement {
                id: id.clone(),
                data: Data::default(),
            });
        }

        index
    }

    pub fn find_index(&self, id: &Id) -> TableIndex<Index> {
        self.table
            .iter()
            .position(|table_element| table_element.id == *id)
            .map_or_else(TableIndex::default, TableIndex::new)
    }
}

impl<Id: TableIdTrait + Debug, Data: Debug, Index: TableIndexTrait + Debug> Debug
    for Table<Id, Data, Index>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("Table")?;
        f.debug_list()
            .entries(self.table.iter().enumerate())
            .finish()
    }
}

impl<Id: TableIdTrait, Data, Index: TableIndexTrait> TryFrom<Vec<TableElement<Id, Data>>>
    for Table<Id, Data, Index>
{
    type Error = Box<String>;

    fn try_from(table: Vec<TableElement<Id, Data>>) -> Result<Self, Self::Error> {
        let mut ids: Vec<Id> = table
            .iter()
            .map(|table_element| table_element.id.clone())
            .collect();

        ids.sort();
        ids.dedup();

        if ids.len() != table.len() {
            Err(
                format!("`Table::try_from` failed because there were duplicate IDs present.")
                    .into(),
            )
        } else {
            Ok(Self {
                table,
                _index: PhantomData,
            })
        }
    }
}

pub type IdList<Id, Index = usize> = Table<Id, (), Index>;

impl<Id: TableIdTrait, Index: TableIndexTrait> IdList<Id, Index> {
    pub fn as_id_slice(&self) -> &[Id] {
        // Why can't you play nice, rust compiler?
        // use {
        //     static_assertions::const_assert_eq,
        //     std::mem::{align_of, size_of},
        // };
        // const_assert_eq!(size_of::<Id>(), size_of::<TableElement<Id, ()>>());
        // const_assert_eq!(align_of::<Id>(), align_of::<TableElement<Id, ()>>());
        unsafe { transmute(self.as_slice()) }
    }
}

impl<Id: TableIdTrait, Index: TableIndexTrait> TryFrom<Vec<Id>> for IdList<Id, Index> {
    type Error = Box<String>;

    fn try_from(ids: Vec<Id>) -> Result<Self, Self::Error> {
        IdList::try_from(
            ids.into_iter()
                .map(|id| TableElement { id, data: () })
                .collect::<Vec<TableElement<Id, ()>>>(),
        )
    }
}
