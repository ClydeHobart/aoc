use std::ops::{Deref, DerefMut};

pub struct NoDropVec<T> {
    vec: Vec<T>,
    len: usize,
}

impl<T> NoDropVec<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.vec[..self.len]
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.vec[..self.len]
    }

    pub fn clear(&mut self) {
        self.len = 0_usize;
    }

    pub fn push(&mut self) -> &mut T
    where
        T: Default,
    {
        if self.vec.len() == self.len {
            self.vec.push(T::default());
        }

        self.len += 1_usize;

        &mut self.vec[self.len - 1_usize]
    }

    pub fn pop(&mut self) {
        assert!(self.len > 0_usize);

        self.len -= 1_usize;
    }
}

impl<T> Deref for NoDropVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for NoDropVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T> Default for NoDropVec<T> {
    fn default() -> Self {
        Self {
            vec: Default::default(),
            len: Default::default(),
        }
    }
}
