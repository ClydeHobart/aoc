use std::cmp::Ordering;

pub struct SetAndVec<T> {
    data: Vec<T>,
    set_len: usize,
}

impl<T> SetAndVec<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set_as_slice(&self) -> &[T] {
        &self.data[..self.set_len]
    }

    pub fn vec_as_slice(&self) -> &[T] {
        &self.data[self.set_len..]
    }

    /// # Safety
    ///
    /// It is the caller's responsibility to ensure that the ordering is maintained.
    pub unsafe fn set_as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data[..self.set_len]
    }

    pub fn vec_as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data[self.set_len..]
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.set_len = 0_usize;
    }

    pub fn clear_set(&mut self) {
        self.data.rotate_left(self.set_len);
        self.data.truncate(self.data.len() - self.set_len);
        self.set_len = 0_usize;
    }

    pub fn insert_set(&mut self, value: T) -> bool
    where
        T: Ord,
    {
        self.insert_set_by(value, T::cmp)
    }

    pub fn insert_set_by<F: FnMut(&T, &T) -> Ordering>(
        &mut self,
        value: T,
        mut compare: F,
    ) -> bool {
        if let Err(index) = self
            .set_as_slice()
            .binary_search_by(|probe| compare(probe, &value))
        {
            self.data.insert(index, value);
            self.set_len += 1_usize;

            true
        } else {
            false
        }
    }

    pub fn insert_set_by_key<K: Ord, F: FnMut(&T) -> K>(&mut self, value: T, mut f: F) -> bool {
        self.insert_set_by(value, |a, b| f(a).cmp(&f(b)))
    }

    pub fn extend_set<I: IntoIterator<Item = T>>(&mut self, iter: I)
    where
        T: Ord,
    {
        self.extend_set_by(iter, T::cmp);
    }

    pub fn extend_set_by<I: IntoIterator<Item = T>, F: FnMut(&T, &T) -> Ordering>(
        &mut self,
        iter: I,
        mut compare: F,
    ) where
        T: Ord,
    {
        for value in iter {
            self.insert_set_by(value, &mut compare);
        }
    }

    pub fn extend_set_by_key<I: IntoIterator<Item = T>, K: Ord, F: FnMut(&T) -> K>(
        &mut self,
        iter: I,
        mut f: F,
    ) where
        T: Ord,
    {
        self.extend_set_by(iter, |a, b| f(a).cmp(&f(b)));
    }

    pub fn clear_vec(&mut self) {
        self.data.truncate(self.set_len);
    }

    pub fn push_vec(&mut self, value: T) {
        self.data.push(value);
    }

    pub fn extend_vec<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push_vec(value);
        }
    }

    pub fn pop_vec(&mut self) -> Option<T> {
        (self.data.len() - self.set_len > 0_usize)
            .then(|| self.data.pop())
            .flatten()
    }

    pub fn extend_set_with_vec(&mut self)
    where
        T: Ord,
    {
        self.extend_set_with_vec_by(T::cmp)
    }

    pub fn extend_set_with_vec_by<F: FnMut(&T, &T) -> Ordering>(&mut self, mut compare: F) {
        self.data.sort_by(|a, b| compare(a, b));
        self.data.dedup_by(|a, b| compare(a, b).is_eq());
        self.set_len = self.data.len();
    }

    pub fn extend_set_with_vec_by_key<K: Ord, F: FnMut(&T) -> K>(&mut self, mut f: F) {
        self.extend_set_with_vec_by(|a, b| f(a).cmp(&f(b)))
    }

    pub fn extend_vec_front_with_set(&mut self) {
        self.set_len = 0_usize;
    }
}

impl<T> Default for SetAndVec<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            set_len: Default::default(),
        }
    }
}
