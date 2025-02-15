use {
    crate::*,
    std::mem::{swap, transmute},
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        marker::PhantomData,
    },
};

define_super_trait! {
    pub trait TableIdTrait where Self: Clone + Eq + Ord + PartialEq + PartialOrd {}
}

#[derive(Clone, PartialEq)]
pub struct TableElement<Id, Data> {
    pub id: Id,
    pub data: Data,
}

impl<Id: TableIdTrait, Data> TableElement<Id, Data> {
    fn id_for_sort_key(&self) -> Id {
        self.id.clone()
    }
}

impl<Id: Debug, Data: Debug> Debug for TableElement<Id, Data> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        (&self.id, &self.data).fmt(f)
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Default)]
pub struct Table<Id: TableIdTrait, Data, IndexRaw: IndexRawTrait = usize> {
    table: Vec<TableElement<Id, Data>>,
    _index: PhantomData<IndexRaw>,
}

impl<Id: TableIdTrait, Data, IndexRaw: IndexRawTrait> Table<Id, Data, IndexRaw> {
    pub fn new() -> Self
    where
        Self: Default,
    {
        Self::default()
    }

    pub fn as_slice(&self) -> &[TableElement<Id, Data>] {
        &self.table
    }

    pub fn find_index(&self, id: &Id) -> Index<IndexRaw> {
        self.table
            .iter()
            .position(|table_element| table_element.id == *id)
            .map_or_else(Index::default, Index::new)
    }

    pub fn find_index_binary_search(&self, id: &Id) -> Index<IndexRaw> {
        self.table
            .binary_search_by_key(id, TableElement::id_for_sort_key)
            .ok()
            .map_or_else(Index::default, Index::new)
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }

    /// It is the user's responsibility to ensure either the IDs aren't modified, or that other
    /// consumers don't care about the IDs being modified.
    pub fn as_slice_mut(&mut self) -> &mut [TableElement<Id, Data>] {
        &mut self.table
    }

    pub fn sort_by_id(&mut self) {
        self.table.sort_by_key(TableElement::id_for_sort_key)
    }

    pub fn insert(&mut self, id: Id, data: Data) -> Option<Data> {
        if let Some(index) = self.find_index(&id).opt() {
            let mut data: Data = data;

            swap(&mut data, &mut self.table[index.get()].data);

            Some(data)
        } else {
            self.table.push(TableElement { id, data });

            None
        }
    }

    pub fn insert_binary_search(&mut self, id: Id, data: Data) -> Option<Data> {
        match self
            .table
            .binary_search_by_key(&id, TableElement::id_for_sort_key)
        {
            Ok(index) => {
                let mut data: Data = data;

                swap(&mut data, &mut self.table[index].data);

                Some(data)
            }
            Err(index) => {
                self.table.insert(index, TableElement { id, data });

                None
            }
        }
    }

    pub fn find_or_add_index(&mut self, id: &Id) -> Index<IndexRaw>
    where
        Data: Default,
    {
        let mut index: Index<IndexRaw> = self.find_index(id);

        if !index.is_valid() {
            index = self.table.len().into();
            self.table.push(TableElement {
                id: id.clone(),
                data: Data::default(),
            });
        }

        index
    }

    pub fn find_or_add_index_binary_search(&mut self, id: &Id) -> Index<IndexRaw>
    where
        Data: Default,
    {
        match self
            .table
            .binary_search_by_key(id, TableElement::id_for_sort_key)
        {
            Ok(index) => index.into(),
            Err(index) => {
                self.table.insert(
                    index,
                    TableElement {
                        id: id.clone(),
                        data: Data::default(),
                    },
                );

                index.into()
            }
        }
    }

    pub fn remove_by_index(&mut self, index: Index<IndexRaw>) -> TableElement<Id, Data> {
        self.table.remove(index.get())
    }

    pub fn remove_by_id(&mut self, id: &Id) -> Option<Data> {
        self.find_index(id)
            .opt()
            .map(|index| self.remove_by_index(index).data)
    }

    pub fn remove_by_id_binary_search(&mut self, id: &Id) -> Option<Data> {
        self.find_index_binary_search(id)
            .opt()
            .map(|index| self.remove_by_index(index).data)
    }
}

impl<Id: TableIdTrait + Debug, Data: Debug, Index: IndexRawTrait + Debug> Debug
    for Table<Id, Data, Index>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("Table")?;
        f.debug_list()
            .entries(self.table.iter().enumerate())
            .finish()
    }
}

impl<Id: TableIdTrait, Data, Index: IndexRawTrait> TryFrom<Vec<TableElement<Id, Data>>>
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

impl<Id: TableIdTrait, Index: IndexRawTrait> IdList<Id, Index> {
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

impl<Id: TableIdTrait, Index: IndexRawTrait> TryFrom<Vec<Id>> for IdList<Id, Index> {
    type Error = Box<String>;

    fn try_from(ids: Vec<Id>) -> Result<Self, Self::Error> {
        IdList::try_from(
            ids.into_iter()
                .map(|id| TableElement { id, data: () })
                .collect::<Vec<TableElement<Id, ()>>>(),
        )
    }
}
