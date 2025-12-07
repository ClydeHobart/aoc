use {
    crate::*,
    static_assertions::const_assert,
    std::{
        cmp::Ordering,
        fmt::{Debug, DebugList, Formatter, Result as FmtResult},
        iter::Peekable,
        mem::{swap, take, MaybeUninit},
        num::NonZeroUsize,
    },
};

define_super_trait! {
    pub trait LinkedTrieNodeKey where Self: PartialEq {}
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LinkedTrieNodeKeyValueState {
    #[default]
    Neither,
    Key,
    KeyAndValue,
}

impl LinkedTrieNodeKeyValueState {
    pub const fn do_all_values_have_keys() -> bool {
        true
    }

    pub fn has_key(self) -> bool {
        matches!(self, Self::Key | Self::KeyAndValue)
    }

    pub fn has_value(self) -> bool {
        matches!(self, Self::KeyAndValue)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinkedTrieNodeState<Index: IndexTrait> {
    parent_node_index: Index,
    child_node_index: Index,
    prev_sibling_node_index: Index,
    next_sibling_node_index: Index,
    key_value_state: LinkedTrieNodeKeyValueState,
}

impl<Index: IndexTrait> LinkedTrieNodeState<Index> {
    pub fn has_vertical_linkage(&self) -> bool {
        self.parent_node_index.is_valid() || self.child_node_index.is_valid()
    }

    pub fn has_horizontal_linkage(&self) -> bool {
        self.prev_sibling_node_index.is_valid() || self.next_sibling_node_index.is_valid()
    }

    pub fn has_key_or_value(&self) -> bool {
        const_assert!(LinkedTrieNodeKeyValueState::do_all_values_have_keys());

        self.key_value_state.has_key()
    }

    pub fn is_default(&self) -> bool {
        !self.has_vertical_linkage() && !self.has_horizontal_linkage() && !self.has_key_or_value()
    }
}

#[derive(Debug, PartialEq)]
pub struct LinkedTrieKeyValuePair<'l, Key, Value> {
    pub key: &'l Key,
    pub value: Option<&'l Value>,
}

#[derive(Debug, PartialEq)]
pub struct LinkedTrieNode<'l, Key, Value, Index: IndexTrait> {
    pub kvp: LinkedTrieKeyValuePair<'l, Key, Value>,
    pub state: &'l LinkedTrieNodeState<Index>,
}

#[derive(PartialEq)]
pub struct LinkedTrieVisitState<'l, Key, Value> {
    pub depth: usize,
    pub kvp: LinkedTrieKeyValuePair<'l, Key, Value>,
}

impl<'l, Key: Debug, Value: Debug> Debug for LinkedTrieVisitState<'l, Key, Value> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if f.alternate() {
            f.write_fmt(format_args!("{0:1$}", "", 4_usize * self.depth))?;
        } else {
            f.write_fmt(format_args!("{}, ", self.depth))?;
        }

        f.write_fmt(format_args!("{:?}: {:?}", &self.kvp.key, &self.kvp.value))
    }
}

#[derive(Debug, PartialEq)]
pub struct LinkedTrieVisitPairState<'l, Key, Value> {
    pub depth: usize,
    pub kvp_a: Option<LinkedTrieKeyValuePair<'l, Key, Value>>,
    pub kvp_b: Option<LinkedTrieKeyValuePair<'l, Key, Value>>,
}

pub const fn linked_trie_does_node_state_have_key_iff_node_key_is_initialized() -> bool {
    true
}

pub const fn linked_trie_does_node_state_have_value_iff_node_value_is_initialized() -> bool {
    true
}

#[derive(Clone, Copy)]
struct LinkedTrieVacantState<Index: IndexTrait> {
    count: NonZeroUsize,
    node_index: Index,
}

/// A Trie implemented as a linked list of nodes, with nodes referencing other nodes via indices.
///
/// # Invariants
///
/// 1. Any node that is initialized has a valid key.
/// 2. A node has the corresponding bit in `occupied` set if and only if it is initialized.
pub struct LinkedTrie<Key: LinkedTrieNodeKey, Value, Index: IndexTrait = crate::util::Index<u32>> {
    node_states: Vec<LinkedTrieNodeState<Index>>,
    node_keys: Vec<MaybeUninit<Key>>,
    node_values: Vec<MaybeUninit<Value>>,
    root_node_index: Index,
    vacant_state: Option<LinkedTrieVacantState<Index>>,
}

impl<Key: LinkedTrieNodeKey, Value, Index: IndexTrait> LinkedTrie<Key, Value, Index> {
    pub fn new() -> Self {
        Self {
            node_states: Vec::new(),
            node_keys: Vec::new(),
            node_values: Vec::new(),
            root_node_index: Index::invalid(),
            vacant_state: None,
        }
    }

    pub fn vacancy(&self) -> usize {
        self.get_vacant_count_and_node_index()
            .map(|(vacancy, _)| vacancy)
            .unwrap_or_default()
    }

    pub fn capacity(&self) -> usize {
        self.node_states.len()
    }

    pub fn len(&self) -> usize {
        self.capacity() - self.vacancy()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }

    pub fn root_node_index(&self) -> Index {
        self.root_node_index
    }

    pub fn does_node_have_key(&self, node_index: Index) -> bool {
        self.try_get_node_state(node_index)
            .map_or(false, |node_state| node_state.key_value_state.has_key())
    }

    pub fn does_node_have_value(&self, node_index: Index) -> bool {
        self.try_get_node_state(node_index)
            .map_or(false, |node_state| node_state.key_value_state.has_value())
    }

    pub fn get_key_value_pair(
        &self,
        node_index: Index,
        node_state: &LinkedTrieNodeState<Index>,
    ) -> LinkedTrieKeyValuePair<Key, Value> {
        let node_index: usize = node_index.get();

        LinkedTrieKeyValuePair {
            // SAFETY: Invariant
            key: unsafe {
                assert!(node_state.key_value_state.has_key());

                const_assert!(linked_trie_does_node_state_have_key_iff_node_key_is_initialized());

                self.node_keys[node_index].assume_init_ref()
            },
            value: node_state
                .key_value_state
                .has_value()
                // SAFETY: Invariant
                .then(|| unsafe {
                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );

                    self.node_values[node_index].assume_init_ref()
                }),
        }
    }

    /// Retrieves a node for a node index that may be invalid (as in `!node_index.is_valid()`), but
    /// corresponds to an occupied node when valid.
    ///
    /// # Assertions
    ///
    /// * This asserts if `node_index` is valid but not occupied.
    pub fn try_get_node(&self, node_index: Index) -> Option<LinkedTrieNode<Key, Value, Index>> {
        self.try_get_node_state(node_index)
            .map(|node_state| LinkedTrieNode {
                kvp: self.get_key_value_pair(node_index, node_state),
                state: node_state,
            })
    }

    pub fn visit_pair<'v, 'l: 'v, F: FnMut(LinkedTrieVisitPairState<'l, Key, Value>) -> bool>(
        depth: usize,
        a: &'l Self,
        node_index_a: Index,
        b: &'l Self,
        node_index_b: Index,
        visitor: &'v mut F,
    ) -> bool {
        let node_a: Option<LinkedTrieNode<Key, Value, Index>> = a.try_get_node(node_index_a);
        let node_b: Option<LinkedTrieNode<Key, Value, Index>> = b.try_get_node(node_index_b);

        if node_a.is_none() && node_b.is_none() {
            // Keep visiting.
            true
        } else {
            let get_kvp_child_node_index_and_sibling_node_index = |node: Option<
                LinkedTrieNode<'l, Key, Value, Index>,
            >|
             -> (
                Option<LinkedTrieKeyValuePair<'l, Key, Value>>,
                Index,
                Index,
            ) {
                node.map(|node| {
                    (
                        Some(node.kvp),
                        node.state.child_node_index,
                        node.state.next_sibling_node_index,
                    )
                })
                .unwrap_or_default()
            };
            let (kvp_a, child_node_index_a, sibling_node_index_a): (
                Option<LinkedTrieKeyValuePair<'l, Key, Value>>,
                Index,
                Index,
            ) = get_kvp_child_node_index_and_sibling_node_index(node_a);
            let (kvp_b, child_node_index_b, sibling_node_index_b): (
                Option<LinkedTrieKeyValuePair<'l, Key, Value>>,
                Index,
                Index,
            ) = get_kvp_child_node_index_and_sibling_node_index(node_b);

            visitor(LinkedTrieVisitPairState {
                depth,
                kvp_a,
                kvp_b,
            }) && Self::visit_pair(
                depth + 1_usize,
                a,
                child_node_index_a,
                b,
                child_node_index_b,
                visitor,
            ) && Self::visit_pair(
                depth,
                a,
                sibling_node_index_a,
                b,
                sibling_node_index_b,
                visitor,
            )
        }
    }

    pub fn visit<'v, 'l: 'v, F: FnMut(LinkedTrieVisitState<'l, Key, Value>) -> bool>(
        &'l self,
        depth: usize,
        node_index: Index,
        visitor: &'v mut F,
    ) -> bool {
        Self::visit_pair(
            depth,
            self,
            node_index,
            self,
            Index::invalid(),
            &mut |visit_pair_state| {
                visit_pair_state.kvp_a.is_none()
                    || visitor(LinkedTrieVisitState {
                        depth: visit_pair_state.depth,
                        kvp: visit_pair_state.kvp_a.unwrap(),
                    })
            },
        )
    }

    pub fn contains_keys_sorted_by<
        I: IntoIterator<Item = Key>,
        F: FnMut(&Key, &Key) -> Ordering,
    >(
        &self,
        keys: I,
        mut compare: F,
    ) -> bool {
        let mut keys: Peekable<_> = keys.into_iter().peekable();

        self.find_node_sorted_by(&mut keys, &mut compare)
            .map_or(false, |node_index| self.does_node_have_value(node_index))
    }

    pub fn contains_keys_sorted_by_other_key<
        K: Ord,
        I: IntoIterator<Item = Key>,
        F: FnMut(&Key) -> K,
    >(
        &self,
        keys: I,
        mut f: F,
    ) -> bool {
        self.contains_keys_sorted_by(keys, |key_a, key_b| f(key_a).cmp(&f(key_b)))
    }

    pub fn contains_keys_sorted<I: IntoIterator<Item = Key>>(&self, keys: I) -> bool
    where
        Key: Ord,
    {
        self.contains_keys_sorted_by(keys, Key::cmp)
    }

    pub fn contains_keys<I: IntoIterator<Item = Key>>(&self, keys: I) -> bool {
        self.contains_keys_sorted_by(keys, Self::no_compare)
    }

    pub fn find_child_node_index_sorted_by<F: FnMut(&Key, &Key) -> Ordering>(
        &self,
        key: &Key,
        parent_node_index: Index,
        mut compare: F,
    ) -> Option<Index> {
        let mut child_node_index: Index = self.get_child_node_index(parent_node_index);

        while self
            .try_get_node(child_node_index)
            .map_or(false, |child_node| {
                match compare(key, &child_node.kvp.key) {
                    Ordering::Less => {
                        child_node_index = Index::invalid();

                        false
                    }
                    Ordering::Equal => false,
                    Ordering::Greater => {
                        child_node_index = child_node.state.next_sibling_node_index;

                        true
                    }
                }
            })
        {}

        child_node_index.opt()
    }

    pub fn find_child_node_index_sorted_by_other_key<K: Ord, F: FnMut(&Key) -> K>(
        &self,
        key: &Key,
        parent_node_index: Index,
        mut f: F,
    ) -> Option<Index> {
        self.find_child_node_index_sorted_by(key, parent_node_index, |key_a, key_b| {
            f(key_a).cmp(&f(key_b))
        })
    }

    pub fn find_child_node_index_sorted(&self, key: &Key, parent_node_index: Index) -> Option<Index>
    where
        Key: Ord,
    {
        self.find_child_node_index_sorted_by(key, parent_node_index, Key::cmp)
    }

    pub fn find_child_node_index(&self, key: &Key, parent_node_index: Index) -> Option<Index> {
        self.find_child_node_index_sorted_by(key, parent_node_index, Self::no_compare)
    }

    pub fn insert_sorted_by<I: IntoIterator<Item = Key>, F: FnMut(&Key, &Key) -> Ordering>(
        &mut self,
        keys: I,
        mut value: Value,
        mut compare: F,
    ) -> Option<Value> {
        let mut prev_value: Option<Value> = None;
        let mut keys: Peekable<_> = keys.into_iter().peekable();

        match self.find_node_sorted_by(&mut keys, &mut compare) {
            Ok(node_index) => {
                if self.does_node_have_value(node_index) {
                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );

                    swap(
                        // SAFETY: This node has a value.
                        unsafe { self.node_values[node_index.get()].assume_init_mut() },
                        &mut value,
                    );

                    prev_value = Some(value);
                } else {
                    self.node_values[node_index.get()].write(value);
                    self.node_states[node_index.get()].key_value_state =
                        LinkedTrieNodeKeyValueState::KeyAndValue;

                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );
                }
            }
            Err(mut curr_node_index) => {
                while let Some(key) = keys.next() {
                    let next_node_index: Index = self.new_node_index();

                    self.node_keys[next_node_index.get()].write(key);
                    self.node_states[next_node_index.get()].key_value_state =
                        LinkedTrieNodeKeyValueState::Key;
                    self.insert_absent_key(curr_node_index, next_node_index, &mut compare);
                    curr_node_index = next_node_index;
                }

                if curr_node_index.is_valid() {
                    self.node_values[curr_node_index.get()].write(value);
                    self.node_states[curr_node_index.get()].key_value_state =
                        LinkedTrieNodeKeyValueState::KeyAndValue;

                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );
                }
            }
        }

        prev_value
    }

    pub fn insert_sorted_by_other_key<K: Ord, I: IntoIterator<Item = Key>, F: FnMut(&Key) -> K>(
        &mut self,
        keys: I,
        value: Value,
        mut f: F,
    ) -> Option<Value> {
        self.insert_sorted_by(keys, value, |key_a, key_b| f(key_a).cmp(&f(key_b)))
    }

    pub fn insert_sorted<I: IntoIterator<Item = Key>>(
        &mut self,
        keys: I,
        value: Value,
    ) -> Option<Value>
    where
        Key: Ord,
    {
        self.insert_sorted_by(keys, value, Key::cmp)
    }

    pub fn insert<I: IntoIterator<Item = Key>>(&mut self, keys: I, value: Value) -> Option<Value> {
        self.insert_sorted_by(keys, value, Self::no_compare)
    }

    pub fn remove_sorted_by<I: IntoIterator<Item = Key>, F: FnMut(&Key, &Key) -> Ordering>(
        &mut self,
        keys: I,
        mut compare: F,
    ) -> Option<Value> {
        let mut keys: Peekable<_> = keys.into_iter().peekable();

        self.find_node_sorted_by(&mut keys, &mut compare)
            .ok()
            .and_then(|node_index| {
                let node_state: &mut LinkedTrieNodeState<Index> =
                    &mut self.node_states[node_index.get()];

                let value: Option<Value> = node_state.key_value_state.has_value().then(|| {
                    let mut maybe_uninit_value: MaybeUninit<Value> = MaybeUninit::uninit();

                    swap(
                        &mut self.node_values[node_index.get()],
                        &mut maybe_uninit_value,
                    );

                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );

                    let value: Value = unsafe { maybe_uninit_value.assume_init() };

                    const_assert!(LinkedTrieNodeKeyValueState::do_all_values_have_keys());

                    node_state.key_value_state = LinkedTrieNodeKeyValueState::Key;

                    value
                });

                if !node_state.child_node_index.is_valid() {
                    self.remove_node(node_index);
                }

                value
            })
    }

    pub fn remove_sorted_by_other_key<K: Ord, I: IntoIterator<Item = Key>, F: FnMut(&Key) -> K>(
        &mut self,
        keys: I,
        mut f: F,
    ) -> Option<Value> {
        self.remove_sorted_by(keys, |key_a, key_b| f(key_a).cmp(&f(key_b)))
    }

    pub fn remove_sorted<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> Option<Value>
    where
        Key: Ord,
    {
        self.remove_sorted_by(keys, Key::cmp)
    }

    pub fn remove<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> Option<Value> {
        self.remove_sorted_by(keys, Self::no_compare)
    }

    fn no_compare(target_key: &Key, curr_key: &Key) -> Ordering {
        if target_key == curr_key {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }

    fn update_node_state_index(
        expected_node_index: Index,
        source_node_index: Index,
        target_node_index: &mut Index,
    ) {
        assert!(*target_node_index == expected_node_index);
        *target_node_index = source_node_index;
    }

    fn try_get_node_state(&self, node_index: Index) -> Option<&LinkedTrieNodeState<Index>> {
        node_index.is_valid().then(|| {
            let node_index: usize = node_index.get();

            assert!(node_index < self.node_states.len());

            &self.node_states[node_index]
        })
    }

    /// Finds the node matching the keys provided.
    ///
    /// Upon success, the index to the node is returned, and `keys` is exhausted.
    ///
    /// Upon failure, the index to the last matching node is returned (invalid if there is no
    /// matching node) and `keys` still holds the remaining keys.
    fn find_node_sorted_by<I: Iterator<Item = Key>, F: FnMut(&Key, &Key) -> Ordering>(
        &self,
        keys: &mut Peekable<I>,
        compare: &mut F,
    ) -> Result<Index, Index> {
        let mut node_index: Index = Index::invalid();

        while let Some(next_node_index) = keys.peek().and_then(|key| {
            self.find_child_node_index_sorted_by(key, node_index, |key_a, key_b| {
                compare(key_a, key_b)
            })
        }) {
            keys.next();
            node_index = next_node_index;
        }

        if keys.peek().is_none() {
            Ok(node_index)
        } else {
            Err(node_index)
        }
    }

    fn get_vacant_count_and_node_index(&self) -> Option<(usize, Index)> {
        self.vacant_state
            .as_ref()
            .map(|vacant_state| (vacant_state.count.get(), vacant_state.node_index))
    }

    fn set_vacant_count_and_node_index(&mut self, count: usize, node_index: Index) {
        let count: Option<NonZeroUsize> = NonZeroUsize::new(count);
        let node_index: Option<Index> = node_index.opt();

        assert_eq!(count.is_some(), node_index.is_some());

        self.vacant_state = count
            .zip(node_index)
            .map(|(count, node_index)| LinkedTrieVacantState { count, node_index });
    }

    fn get_child_node_index(&self, parent_node_index: Index) -> Index {
        if parent_node_index.is_valid() {
            self.node_states[parent_node_index.get()].child_node_index
        } else {
            self.root_node_index
        }
    }

    fn get_child_node_index_mut(&mut self, parent_node_index: Index) -> &mut Index {
        if parent_node_index.is_valid() {
            &mut self.node_states[parent_node_index.get()].child_node_index
        } else {
            &mut self.root_node_index
        }
    }

    fn new_node_index(&mut self) -> Index {
        if let Some((vacant_count, curr_vacant_node_index)) = self.get_vacant_count_and_node_index()
        {
            let curr_vacant_node_state: &mut LinkedTrieNodeState<Index> =
                &mut self.node_states[curr_vacant_node_index.get()];

            assert!(!curr_vacant_node_state.has_vertical_linkage());
            assert!(!curr_vacant_node_state.has_key_or_value());

            let next_vacant_node_index: Index = curr_vacant_node_state.next_sibling_node_index;

            curr_vacant_node_state.next_sibling_node_index = Index::invalid();

            assert!(!curr_vacant_node_state.has_horizontal_linkage());

            if next_vacant_node_index.is_valid() {
                let next_vacant_node_state: &mut LinkedTrieNodeState<Index> =
                    &mut self.node_states[next_vacant_node_index.get()];

                assert!(!next_vacant_node_state.has_vertical_linkage());
                assert!(!next_vacant_node_state.has_key_or_value());

                next_vacant_node_state.prev_sibling_node_index = Index::invalid();
            }

            self.set_vacant_count_and_node_index(vacant_count - 1_usize, next_vacant_node_index);

            curr_vacant_node_index
        } else {
            let new_node_index: Index = self.node_keys.len().into();

            self.node_keys.push(MaybeUninit::uninit());
            self.node_values.push(MaybeUninit::uninit());
            self.node_states.push(Default::default());

            new_node_index
        }
    }

    fn insert_absent_key<F: FnMut(&Key, &Key) -> Ordering>(
        &mut self,
        parent_node_index: Index,
        key_node_index: Index,
        compare: &mut F,
    ) {
        let key: &Key = self.try_get_node(key_node_index).unwrap().kvp.key;

        let mut prev_sibling_node_index: Index = Index::invalid();
        let mut next_sibling_node_index: Index = self.get_child_node_index(parent_node_index);
        let mut next_sibling_node: Option<LinkedTrieNode<Key, Value, Index>> =
            self.try_get_node(next_sibling_node_index);

        while {
            next_sibling_node
                .as_ref()
                .map_or(false, |next_sibling_node| {
                    let ordering: Ordering = compare(key, next_sibling_node.kvp.key);

                    assert!(ordering.is_ne());

                    ordering.is_gt()
                })
        } {
            prev_sibling_node_index = next_sibling_node_index;
            next_sibling_node_index = next_sibling_node
                .as_ref()
                .unwrap()
                .state
                .next_sibling_node_index;
            next_sibling_node = self.try_get_node(next_sibling_node_index);
        }

        let key_node_state: &mut LinkedTrieNodeState<Index> =
            &mut self.node_states[key_node_index.get()];

        key_node_state.parent_node_index = parent_node_index;
        key_node_state.prev_sibling_node_index = prev_sibling_node_index;
        key_node_state.next_sibling_node_index = next_sibling_node_index;

        if prev_sibling_node_index.is_valid() {
            Self::update_node_state_index(
                next_sibling_node_index,
                key_node_index,
                &mut self.node_states[prev_sibling_node_index.get()].next_sibling_node_index,
            );
        } else {
            Self::update_node_state_index(
                next_sibling_node_index,
                key_node_index,
                self.get_child_node_index_mut(parent_node_index),
            );
        }

        if next_sibling_node_index.is_valid() {
            Self::update_node_state_index(
                prev_sibling_node_index,
                key_node_index,
                &mut self.node_states[next_sibling_node_index.get()].prev_sibling_node_index,
            );
        }
    }

    fn remove_node(&mut self, node_index: Index) {
        let node_state: &mut LinkedTrieNodeState<Index> = &mut self.node_states[node_index.get()];

        assert!(!node_state.child_node_index.is_valid());
        assert!(!node_state.key_value_state.has_value());

        if node_state.key_value_state.has_key() {
            const_assert!(linked_trie_does_node_state_have_key_iff_node_key_is_initialized());

            unsafe { self.node_keys[node_index.get()].assume_init_drop() };

            node_state.key_value_state = LinkedTrieNodeKeyValueState::Neither;
        }

        let node_state: LinkedTrieNodeState<Index> = take(node_state);

        let vacant_count: usize = self
            .get_vacant_count_and_node_index()
            .map(|(vacant_count, vacant_node_index)| {
                Self::update_node_state_index(
                    Index::invalid(),
                    node_index,
                    &mut self.node_states[vacant_node_index.get()].prev_sibling_node_index,
                );
                Self::update_node_state_index(
                    Index::invalid(),
                    vacant_node_index,
                    &mut self.node_states[node_index.get()].next_sibling_node_index,
                );

                vacant_count
            })
            .unwrap_or_default();

        self.set_vacant_count_and_node_index(vacant_count + 1_usize, node_index);

        if node_state.has_horizontal_linkage() {
            if node_state.prev_sibling_node_index.is_valid() {
                Self::update_node_state_index(
                    node_index,
                    node_state.next_sibling_node_index,
                    &mut self.node_states[node_state.prev_sibling_node_index.get()]
                        .next_sibling_node_index,
                );
            } else {
                Self::update_node_state_index(
                    node_index,
                    node_state.next_sibling_node_index,
                    self.get_child_node_index_mut(node_state.parent_node_index),
                );
            }

            if node_state.next_sibling_node_index.is_valid() {
                Self::update_node_state_index(
                    node_index,
                    node_state.prev_sibling_node_index,
                    &mut self.node_states[node_state.next_sibling_node_index.get()]
                        .prev_sibling_node_index,
                );
            }
        } else {
            Self::update_node_state_index(
                node_index,
                Index::invalid(),
                self.get_child_node_index_mut(node_state.parent_node_index),
            );

            if node_state.parent_node_index.is_valid()
                && !self.does_node_have_value(node_state.parent_node_index)
            {
                self.remove_node(node_state.parent_node_index);
            }
        }
    }
}

impl<Key: LinkedTrieNodeKey + Clone, Value: Clone, Index: IndexTrait> Clone
    for LinkedTrie<Key, Value, Index>
{
    fn clone(&self) -> Self {
        fn clone_maybe_uninit_slice<
            T: Clone,
            Index: IndexTrait,
            F: Fn(LinkedTrieNodeKeyValueState) -> bool,
        >(
            node_elements: &[MaybeUninit<T>],
            node_states: &[LinkedTrieNodeState<Index>],
            node_has_element: F,
        ) -> Vec<MaybeUninit<T>> {
            node_elements
                .iter()
                .zip(node_states.iter())
                .map(|(node_element, node_state)| {
                    if node_has_element(node_state.key_value_state) {
                        const_assert!(
                            linked_trie_does_node_state_have_key_iff_node_key_is_initialized()
                        );
                        const_assert!(
                            linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                        );

                        MaybeUninit::new(unsafe { node_element.assume_init_ref() }.clone())
                    } else {
                        MaybeUninit::uninit()
                    }
                })
                .collect()
        }

        Self {
            node_states: self.node_states.clone(),
            node_keys: clone_maybe_uninit_slice(
                &self.node_keys,
                &self.node_states,
                LinkedTrieNodeKeyValueState::has_key,
            ),
            node_values: clone_maybe_uninit_slice(
                &self.node_values,
                &self.node_states,
                LinkedTrieNodeKeyValueState::has_value,
            ),
            root_node_index: self.root_node_index.clone(),
            vacant_state: self.vacant_state.clone(),
        }
    }
}

impl<Key: LinkedTrieNodeKey, Value, Index: IndexTrait> Default for LinkedTrie<Key, Value, Index> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Key: LinkedTrieNodeKey, Value: PartialEq, Index: IndexTrait> PartialEq
    for LinkedTrie<Key, Value, Index>
{
    fn eq(&self, other: &Self) -> bool {
        let mut self_eq_other: bool = true;

        Self::visit_pair(
            0_usize,
            self,
            self.root_node_index,
            other,
            other.root_node_index,
            &mut |visit_pair_state| {
                self_eq_other = visit_pair_state.kvp_a == visit_pair_state.kvp_b;

                self_eq_other
            },
        );

        self_eq_other
    }
}

impl<Key: LinkedTrieNodeKey, Value, Index: IndexTrait> Drop for LinkedTrie<Key, Value, Index> {
    fn drop(&mut self) {
        for (node_index, node_state) in self.node_states.iter_mut().enumerate() {
            if node_state.key_value_state.has_key() {
                const_assert!(linked_trie_does_node_state_have_key_iff_node_key_is_initialized());

                // SAFETY: See const assert above.
                unsafe {
                    self.node_keys[node_index].assume_init_drop();
                }

                const_assert!(LinkedTrieNodeKeyValueState::do_all_values_have_keys());

                if node_state.key_value_state.has_value() {
                    const_assert!(
                        linked_trie_does_node_state_have_value_iff_node_value_is_initialized()
                    );

                    // SAFETY: See const assert above.
                    unsafe {
                        self.node_keys[node_index].assume_init_drop();
                    }

                    // The node no longer has a value.
                    node_state.key_value_state = LinkedTrieNodeKeyValueState::Key;
                }

                // The node no longer has either.
                node_state.key_value_state = LinkedTrieNodeKeyValueState::Neither;
            }
        }
    }
}

impl<Key: LinkedTrieNodeKey + Debug, Value: Debug, Index: IndexTrait> Debug
    for LinkedTrie<Key, Value, Index>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut debug_list: DebugList = f.debug_list();

        self.visit(0_usize, self.root_node_index, &mut |visit_state| {
            debug_list.entry(&visit_state);

            // Keep visiting
            true
        });

        debug_list.finish()
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        std::{str::Chars, sync::OnceLock},
    };

    const WORDS: &'static [&'static str] = &["cat", "cob", "cod", "code", "cog", "do", "dog"];

    macro_rules! visit_state {
        ($depth:expr, $key:expr) => {
            LinkedTrieVisitState {
                depth: $depth,
                kvp: LinkedTrieKeyValuePair {
                    key: &$key,
                    value: None,
                },
            }
        };
        ($depth:expr, $key:expr, $value:expr) => {
            LinkedTrieVisitState {
                depth: $depth,
                kvp: LinkedTrieKeyValuePair {
                    key: &$key,
                    value: Some(&$value),
                },
            }
        };
    }

    macro_rules! visit_states { [ $( ( $( $token:expr ),+ ) ),* $(,)? ] => { [ $(
        visit_state!( $( $token ),+ ),
    )* ] } }

    const VISIT_STATES: &'static [LinkedTrieVisitState<'static, char, i32>] = &visit_states![
        (0_usize, 'c'),
        (1_usize, 'a'),
        (2_usize, 't', 0_i32),
        (1_usize, 'o'),
        (2_usize, 'b', 1_i32),
        (2_usize, 'd', 2_i32),
        (3_usize, 'e', 3_i32),
        (2_usize, 'g', 4_i32),
        (0_usize, 'd'),
        (1_usize, 'o', 5_i32),
        (2_usize, 'g', 6_i32),
    ];

    /// # Words:
    /// * cat
    /// * cob
    /// * cod
    /// * code
    /// * cog
    /// * do
    /// * dog
    ///
    /// # Structure
    /// * 0: c
    ///     * 1: a
    ///         * 2: t (0)
    ///     * 3: o
    ///         * 4: b (1)
    ///         * 5: d (2)
    ///             * 6: e (3)
    ///         * 7: g (4)
    /// * 8: d
    ///     * 9: o (5)
    ///         * 10: g (6)
    fn manual_linked_trie() -> &'static LinkedTrie<char, i32> {
        static LINKED_TRIE: OnceLock<LinkedTrie<char, i32>> = OnceLock::new();

        LINKED_TRIE.get_or_init(|| LinkedTrie {
            node_states: vec![
                // 0: c
                LinkedTrieNodeState {
                    child_node_index: 1_usize.into(),
                    next_sibling_node_index: 8_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::Key,
                    ..Default::default()
                },
                // 1: a
                LinkedTrieNodeState {
                    parent_node_index: 0_usize.into(),
                    child_node_index: 2_usize.into(),
                    next_sibling_node_index: 3_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::Key,
                    ..Default::default()
                },
                // 2: t (0)
                LinkedTrieNodeState {
                    parent_node_index: 1_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
                // 3: o
                LinkedTrieNodeState {
                    parent_node_index: 0_usize.into(),
                    child_node_index: 4_usize.into(),
                    prev_sibling_node_index: 1_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::Key,
                    ..Default::default()
                },
                // 4: b (1)
                LinkedTrieNodeState {
                    parent_node_index: 3_usize.into(),
                    next_sibling_node_index: 5_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
                // 5: d (2)
                LinkedTrieNodeState {
                    parent_node_index: 3_usize.into(),
                    child_node_index: 6_usize.into(),
                    prev_sibling_node_index: 4_usize.into(),
                    next_sibling_node_index: 7_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                },
                // 6: e (3)
                LinkedTrieNodeState {
                    parent_node_index: 5_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
                // 7: g (4)
                LinkedTrieNodeState {
                    parent_node_index: 3_usize.into(),
                    prev_sibling_node_index: 5_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
                // 8: d
                LinkedTrieNodeState {
                    child_node_index: 9_usize.into(),
                    prev_sibling_node_index: 0_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::Key,
                    ..Default::default()
                },
                // 9: o (5)
                LinkedTrieNodeState {
                    parent_node_index: 8_usize.into(),
                    child_node_index: 10_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
                // 10: g (6)
                LinkedTrieNodeState {
                    parent_node_index: 9_usize.into(),
                    key_value_state: LinkedTrieNodeKeyValueState::KeyAndValue,
                    ..Default::default()
                },
            ],
            node_keys: "catobdegdog".chars().map(MaybeUninit::new).collect(),
            node_values: [
                None,
                None,
                Some(0_i32),
                None,
                Some(1_i32),
                Some(2_i32),
                Some(3_i32),
                Some(4_i32),
                None,
                Some(5_i32),
                Some(6_i32),
            ]
            .into_iter()
            .map(|value| value.map_or_else(MaybeUninit::uninit, MaybeUninit::new))
            .collect(),
            root_node_index: 0_usize.into(),
            vacant_state: None,
        })
    }

    fn insert_linked_trie() -> &'static LinkedTrie<char, i32> {
        static LINKED_TRIE: OnceLock<LinkedTrie<char, i32>> = OnceLock::new();

        LINKED_TRIE.get_or_init(|| {
            let mut linked_trie: LinkedTrie<char, i32> = LinkedTrie::new();

            for (value, keys) in WORDS.iter().copied().enumerate() {
                linked_trie.insert_sorted(keys.chars(), value as i32);
            }

            linked_trie
        })
    }

    fn visit_states(linked_trie: &LinkedTrie<char, i32>) -> Vec<LinkedTrieVisitState<char, i32>> {
        let mut visit_states: Vec<LinkedTrieVisitState<char, i32>> = Vec::new();

        linked_trie.visit(0_usize, linked_trie.root_node_index(), &mut |visit_state| {
            visit_states.push(visit_state);

            true
        });

        visit_states
    }

    #[test]
    fn test_visit() {
        // Use the manually constructed `LinkedTrie` since `eq` makes use of `visit`.
        assert_eq!(visit_states(manual_linked_trie()), VISIT_STATES);
    }

    #[test]
    fn test_insert() {
        let insert_sorted_fwd_linked_trie: &LinkedTrie<char, i32> = insert_linked_trie();

        assert_eq!(insert_sorted_fwd_linked_trie, manual_linked_trie());

        let mut insert_sorted_rev_linked_trie: LinkedTrie<char, i32> = LinkedTrie::new();

        for (value, keys) in WORDS.iter().copied().enumerate().rev() {
            insert_sorted_rev_linked_trie.insert_sorted(keys.chars(), value as i32);
        }

        assert_eq!(&insert_sorted_rev_linked_trie, manual_linked_trie());
        assert_eq!(
            &insert_sorted_rev_linked_trie,
            insert_sorted_fwd_linked_trie
        );

        let mut insert_fwd_linked_trie: LinkedTrie<char, i32> = LinkedTrie::new();

        for (value, keys) in WORDS.iter().copied().enumerate() {
            insert_fwd_linked_trie.insert(keys.chars(), value as i32);
        }

        assert_eq!(&insert_fwd_linked_trie, manual_linked_trie());
        assert_eq!(&insert_fwd_linked_trie, insert_sorted_fwd_linked_trie);
        assert_eq!(insert_fwd_linked_trie, insert_sorted_rev_linked_trie);

        let mut insert_rev_linked_trie: LinkedTrie<char, i32> = LinkedTrie::new();

        for (value, keys) in WORDS.iter().copied().enumerate().rev() {
            insert_rev_linked_trie.insert(keys.chars(), value as i32);
        }

        assert_ne!(&insert_rev_linked_trie, manual_linked_trie());
        assert_ne!(insert_rev_linked_trie, insert_fwd_linked_trie);
        assert_ne!(&insert_rev_linked_trie, insert_sorted_fwd_linked_trie);
        assert_ne!(insert_rev_linked_trie, insert_sorted_rev_linked_trie);

        assert_eq!(
            visit_states(&insert_rev_linked_trie),
            visit_states![
                (0_usize, 'd'),
                (1_usize, 'o', 5_i32),
                (2_usize, 'g', 6_i32),
                (0_usize, 'c'),
                (1_usize, 'o'),
                (2_usize, 'g', 4_i32),
                (2_usize, 'd', 2_i32),
                (3_usize, 'e', 3_i32),
                (2_usize, 'b', 1_i32),
                (1_usize, 'a'),
                (2_usize, 't', 0_i32),
            ]
        );
    }

    #[test]
    fn test_remove() {
        fn assert_size_metrics(linked_trie: &LinkedTrie<char, i32>, len: usize, capacity: usize) {
            assert_eq!(linked_trie.len(), len);
            assert_eq!(linked_trie.capacity(), capacity);
            assert_eq!(linked_trie.vacancy(), capacity - len);
        }

        fn remove_assert_did_not_contain<
            'c,
            F: Fn(&mut LinkedTrie<char, i32>, Chars<'c>) -> Option<i32>,
        >(
            remove: F,
            word: &'c str,
            linked_trie: &mut LinkedTrie<char, i32>,
        ) {
            assert_eq!(
                remove(linked_trie, word.chars()),
                None,
                "word == {word},\nlinked_trie == {linked_trie:#?}",
            );
        }

        fn remove_assert_did_contain<
            'c,
            F: Fn(&mut LinkedTrie<char, i32>, Chars<'c>) -> Option<i32>,
        >(
            remove: F,
            word: &'c str,
            value: i32,
            linked_trie: &mut LinkedTrie<char, i32>,
        ) {
            assert_eq!(
                remove(linked_trie, word.chars()),
                Some(value),
                "word == {word},\nvalue == {value},\nlinked_trie == {linked_trie:#?}",
            );
        }

        fn remove_assert_size_metrics_and_contents<
            'c,
            F: Fn(&mut LinkedTrie<char, i32>, Chars<'c>) -> Option<i32>,
        >(
            remove: F,
            word: &'c str,
            value: i32,
            len: usize,
            capacity: usize,
            expected_visit_states: &[LinkedTrieVisitState<char, i32>],
            linked_trie: &mut LinkedTrie<char, i32>,
        ) {
            remove_assert_did_contain(&remove, word, value, linked_trie);
            remove_assert_did_not_contain(&remove, word, linked_trie);
            assert_size_metrics(linked_trie, len, capacity);
            assert_eq!(visit_states(linked_trie), expected_visit_states);
        }

        fn test_remove_internal<'c, F: Fn(&mut LinkedTrie<char, i32>, Chars<'c>) -> Option<i32>>(
            remove: F,
        ) {
            let mut linked_trie: LinkedTrie<char, i32> = insert_linked_trie().clone();

            assert_size_metrics(&linked_trie, 11_usize, 11_usize);
            remove_assert_did_not_contain(&remove, "coder", &mut linked_trie);
            remove_assert_did_not_contain(&remove, "con", &mut linked_trie);
            remove_assert_did_not_contain(&remove, "cig", &mut linked_trie);
            remove_assert_size_metrics_and_contents(
                &remove,
                "cob",
                1_i32,
                10_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'a'),
                    (2_usize, 't', 0_i32),
                    (1_usize, 'o'),
                    (2_usize, 'd', 2_i32),
                    (3_usize, 'e', 3_i32),
                    (2_usize, 'g', 4_i32),
                    (0_usize, 'd'),
                    (1_usize, 'o', 5_i32),
                    (2_usize, 'g', 6_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "cog",
                4_i32,
                9_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'a'),
                    (2_usize, 't', 0_i32),
                    (1_usize, 'o'),
                    (2_usize, 'd', 2_i32),
                    (3_usize, 'e', 3_i32),
                    (0_usize, 'd'),
                    (1_usize, 'o', 5_i32),
                    (2_usize, 'g', 6_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "cod",
                2_i32,
                9_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'a'),
                    (2_usize, 't', 0_i32),
                    (1_usize, 'o'),
                    (2_usize, 'd'),
                    (3_usize, 'e', 3_i32),
                    (0_usize, 'd'),
                    (1_usize, 'o', 5_i32),
                    (2_usize, 'g', 6_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "cat",
                0_i32,
                7_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'o'),
                    (2_usize, 'd'),
                    (3_usize, 'e', 3_i32),
                    (0_usize, 'd'),
                    (1_usize, 'o', 5_i32),
                    (2_usize, 'g', 6_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "do",
                5_i32,
                7_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'o'),
                    (2_usize, 'd'),
                    (3_usize, 'e', 3_i32),
                    (0_usize, 'd'),
                    (1_usize, 'o'),
                    (2_usize, 'g', 6_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "dog",
                6_i32,
                4_usize,
                11_usize,
                &visit_states![
                    (0_usize, 'c'),
                    (1_usize, 'o'),
                    (2_usize, 'd'),
                    (3_usize, 'e', 3_i32),
                ],
                &mut linked_trie,
            );
            remove_assert_size_metrics_and_contents(
                &remove,
                "code",
                3_i32,
                0_usize,
                11_usize,
                &visit_states![],
                &mut linked_trie,
            );
        }

        test_remove_internal(LinkedTrie::remove);
        test_remove_internal(LinkedTrie::remove_sorted);
    }
}
