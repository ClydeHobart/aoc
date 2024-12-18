use {
    crate::*,
    nom::{
        error::{Error, ErrorKind},
        Err, IResult, Parser,
    },
    std::{
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        iter::{from_fn, FromIterator},
        mem::take,
    },
};

define_super_trait! {
    pub trait LinkedListDataTrait where Self: Default {}
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone)]

pub struct LinkedListNode<Data: LinkedListDataTrait, IndexRaw: IndexRawTrait> {
    pub data: Data,
    pub prev: Index<IndexRaw>,
    pub next: Index<IndexRaw>,
}

impl<Data: LinkedListDataTrait, IndexRaw: IndexRawTrait> LinkedListNode<Data, IndexRaw> {
    fn new(data: Data) -> Self {
        Self {
            data,
            prev: Index::invalid(),
            next: Index::invalid(),
        }
    }

    fn get_prev(&self) -> Index<IndexRaw> {
        self.prev
    }

    fn get_next(&self) -> Index<IndexRaw> {
        self.next
    }
}

impl<Data: LinkedListDataTrait, Index: IndexRawTrait> Default for LinkedListNode<Data, Index> {
    fn default() -> Self {
        Self::new(Data::default())
    }
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Clone, Default)]
pub struct LinkedList<Data: LinkedListDataTrait, IndexRaw: IndexRawTrait> {
    nodes: Vec<LinkedListNode<Data, IndexRaw>>,
    head: Index<IndexRaw>,
    tail: Index<IndexRaw>,
    vacant: Vec<Index<IndexRaw>>,
}

impl<Data: LinkedListDataTrait, IndexRaw: IndexRawTrait> LinkedList<Data, IndexRaw> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head: Index::invalid(),
            tail: Index::invalid(),
            vacant: Vec::new(),
        }
    }

    pub fn parse_with_parser<'i, F: Parser<&'i str, Data, Error<&'i str>>>(
        mut f: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        move |mut input: &'i str| {
            let mut linked_list: Self = Self::new();

            loop {
                let len: usize = input.len();

                match f.parse(input) {
                    Err(Err::Error(_)) => return Ok((input, linked_list)),
                    Err(error) => return Err(error),
                    Ok((next_input, data)) => {
                        if next_input.len() == len {
                            return Err(Err::Error(Error::new(input, ErrorKind::Many0)));
                        }

                        input = next_input;
                        linked_list.push_back(data);
                    }
                }
            }
        }
    }

    pub fn vacancy(&self) -> usize {
        self.vacant.len()
    }

    pub fn len(&self) -> usize {
        self.nodes.len() - self.vacancy()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }

    pub fn get(&self, index: usize) -> Option<&Data> {
        self.nodes.get(index).map(|node| &node.data)
    }

    pub fn get_head(&self) -> Option<usize> {
        self.head.opt().map(Index::get)
    }

    pub fn get_tail(&self) -> Option<usize> {
        self.tail.opt().map(Index::get)
    }

    pub fn get_prev(&self, index: usize) -> Option<usize> {
        self.nodes
            .get(index)
            .and_then(|node| node.prev.opt())
            .map(usize::from)
    }

    pub fn get_next(&self, index: usize) -> Option<usize> {
        self.nodes
            .get(index)
            .and_then(|node| node.next.opt())
            .map(usize::from)
    }

    fn iter_data_generic<F: Fn(&LinkedListNode<Data, IndexRaw>) -> Index<IndexRaw>>(
        &self,
        mut curr_node_index: Index<IndexRaw>,
        get_next_node_index: F,
    ) -> impl Iterator<Item = &Data> {
        from_fn(move || {
            curr_node_index.is_valid().then(|| {
                let curr_node: &LinkedListNode<Data, IndexRaw> = &self.nodes[curr_node_index.get()];

                curr_node_index = get_next_node_index(curr_node);

                &curr_node.data
            })
        })
    }

    pub fn iter_data(&self) -> impl Iterator<Item = &Data> {
        self.iter_data_generic(self.head, LinkedListNode::get_next)
    }

    pub fn iter_data_rev(&self) -> impl Iterator<Item = &Data> {
        self.iter_data_generic(self.tail, LinkedListNode::get_prev)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.vacant.clear();
        self.head = Index::invalid();
        self.tail = Index::invalid();
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Data> {
        self.nodes.get_mut(index).map(|node| &mut node.data)
    }

    fn get_new_index(&mut self) -> Index<IndexRaw> {
        self.vacant.pop().unwrap_or_else(|| {
            let index: Index<IndexRaw> = self.nodes.len().into();

            self.nodes.push(LinkedListNode::default());

            index
        })
    }

    fn push_first(&mut self, data: Data) -> usize {
        let curr: Index<IndexRaw> = self.get_new_index();

        self.nodes[curr.get()] = LinkedListNode::new(data);
        self.head = curr;
        self.tail = curr;

        curr.get()
    }

    pub fn push_front(&mut self, data: Data) -> usize {
        if self.is_empty() {
            self.push_first(data)
        } else {
            let new_head: Index<IndexRaw> = self.get_new_index();

            self.nodes[self.head.get()].prev = new_head;
            self.nodes[new_head.get()] = LinkedListNode {
                data,
                prev: Index::invalid(),
                next: self.head,
            };
            self.head = new_head;

            new_head.get()
        }
    }

    pub fn push_back(&mut self, data: Data) -> usize {
        if self.is_empty() {
            self.push_first(data)
        } else {
            let new_tail: Index<IndexRaw> = self.get_new_index();

            self.nodes[self.tail.get()].next = new_tail;
            self.nodes[new_tail.get()] = LinkedListNode {
                data,
                prev: self.tail,
                next: Index::invalid(),
            };
            self.tail = new_tail;

            new_tail.get()
        }
    }

    pub fn insert(&mut self, data: Data, index: usize) -> usize {
        if index == self.len() {
            self.push_back(data)
        } else if index == self.head.get() {
            self.push_front(data)
        } else {
            let curr: Index<IndexRaw> = self.get_new_index();
            let next: Index<IndexRaw> = index.into();
            let next_node: &mut LinkedListNode<Data, IndexRaw> = &mut self.nodes[next.get()];
            let prev: Index<IndexRaw> = next_node.prev;

            next_node.prev = curr;
            self.nodes[prev.get()].next = curr;
            self.nodes[curr.get()] = LinkedListNode { data, prev, next };

            curr.get()
        }
    }

    fn pop_last(&mut self) -> Data {
        let data: Data = take(&mut self.nodes[self.head.get()]).data;

        self.vacant.push(self.head);
        self.head = Index::invalid();
        self.tail = Index::invalid();

        data
    }

    pub fn pop_front(&mut self) -> Option<Data> {
        match self.len() {
            0_usize => None,
            1_usize => Some(self.pop_last()),
            _ => {
                let old_head_node: &mut LinkedListNode<Data, IndexRaw> =
                    &mut self.nodes[self.head.get()];
                let new_head: Index<IndexRaw> = old_head_node.next;
                let data: Data = take(old_head_node).data;

                self.vacant.push(self.head);
                self.nodes[new_head.get()].prev = Index::invalid();
                self.head = new_head;

                Some(data)
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<Data> {
        match self.len() {
            0_usize => None,
            1_usize => Some(self.pop_last()),
            _ => {
                let old_tail_node: &mut LinkedListNode<Data, IndexRaw> =
                    &mut self.nodes[self.tail.get()];
                let new_tail: Index<IndexRaw> = old_tail_node.prev;
                let data: Data = take(old_tail_node).data;

                self.vacant.push(self.tail);
                self.nodes[new_tail.get()].next = Index::invalid();
                self.tail = new_tail;

                Some(data)
            }
        }
    }

    pub fn remove(&mut self, index: usize) -> Data {
        let curr: Index<IndexRaw> = index.into();

        if curr == self.head {
            self.pop_front().unwrap()
        } else if curr == self.tail {
            self.pop_back().unwrap()
        } else {
            let curr_node: &mut LinkedListNode<Data, IndexRaw> = &mut self.nodes[index];
            let prev: Index<IndexRaw> = curr_node.prev;
            let next: Index<IndexRaw> = curr_node.next;
            let data: Data = take(curr_node).data;

            self.vacant.push(curr);
            self.nodes[prev.get()].next = next;
            self.nodes[next.get()].prev = prev;

            data
        }
    }
}

impl<Data: LinkedListDataTrait + Debug, Index: IndexRawTrait> Debug for LinkedList<Data, Index> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut debug_list: DebugList = f.debug_list();

        for data in self.iter_data() {
            debug_list.entry(data);
        }

        debug_list.finish()
    }
}

impl<Data: LinkedListDataTrait, Index: IndexRawTrait> From<Vec<Data>> for LinkedList<Data, Index> {
    fn from(value: Vec<Data>) -> Self {
        value.into_iter().collect()
    }
}

impl<Data: LinkedListDataTrait, Index: IndexRawTrait> FromIterator<Data>
    for LinkedList<Data, Index>
{
    fn from_iter<T: IntoIterator<Item = Data>>(iter: T) -> Self {
        let mut linked_list: Self = Self::new();

        for data in iter {
            linked_list.push_back(data);
        }

        linked_list
    }
}

impl<Data: LinkedListDataTrait + Parse, Index: IndexRawTrait> Parse for LinkedList<Data, Index> {
    fn parse<'i>(input: &'i str) -> IResult<&'i str, Self> {
        Self::parse_with_parser(Data::parse)(input)
    }
}
