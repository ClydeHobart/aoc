use {
    crate::*,
    bitvec::prelude::*,
    num::{NumCast, PrimInt},
    std::{
        fmt::Debug,
        iter::{Iterator, Peekable},
        mem::{transmute, MaybeUninit},
        ops::Range,
    },
};

define_super_trait! {
    pub trait RangeIntTrait where Self : PrimInt + Debug + Default {}
}

fn bits<I: RangeIntTrait>() -> u32 {
    I::zero().count_zeros()
}

fn is_signed<I: RangeIntTrait>() -> bool {
    I::min_value() != I::zero()
}

struct RangeDChildrenIter<I: RangeIntTrait, const D: usize> {
    children_per_component: [[RangeD<I, 1_usize>; 2_usize]; D],
    curr_child: RangeD<I, D>,
    last_child_index: u8,
    curr_child_index: u8,
}

impl<I: RangeIntTrait, const D: usize> Iterator for RangeDChildrenIter<I, D> {
    type Item = RangeD<I, D>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.curr_child_index <= self.last_child_index).then(|| {
            let next: Self::Item = self.curr_child.clone();
            let prev_child_index: u8 = self.curr_child_index;

            while {
                self.curr_child_index += 1_u8;

                self.curr_child_index <= self.last_child_index
                    && (self.curr_child_index & !self.last_child_index) != 0_u8
            } {}

            if self.curr_child_index <= self.last_child_index {
                for index in (self.curr_child_index ^ prev_child_index)
                    .view_bits::<Lsb0>()
                    .iter_ones()
                {
                    self.curr_child.set_component(
                        index,
                        self.children_per_component[index]
                            [self.curr_child_index.view_bits::<Lsb0>()[index] as usize],
                    );
                }
            }

            next
        })
    }
}

#[derive(Clone, Copy, Default)]
struct RangeToRange1Iter<I: RangeIntTrait> {
    start: I,
    end: I,
}

impl<I: RangeIntTrait> RangeToRange1Iter<I> {
    fn step_fits_in_i(&self) -> bool {
        !is_signed::<I>() || self.start != I::min_value() || self.end < I::zero()
    }

    fn len_exponent(&self) -> [u8; 1_usize] {
        let start_trailing_zeros: u32 = self.start.trailing_zeros();

        [if !self.step_fits_in_i() {
            bits::<I>() - 1_u32
        } else if self.start == I::zero()
            || self
                .start
                .checked_add(&(I::one() << start_trailing_zeros as usize))
                .map_or(true, |sum| sum > self.end)
        {
            bits::<I>() - (self.end - self.start).leading_zeros() - 1_u32
        } else {
            start_trailing_zeros
        } as u8]
    }
}

impl<I: RangeIntTrait> Iterator for RangeToRange1Iter<I> {
    type Item = RangeD<I, 1_usize>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.start < self.end).then(|| {
            let next: Self::Item = RangeD {
                start: [self.start],
                len_exponent: self.len_exponent(),
            };

            self.start = if !self.step_fits_in_i() {
                I::zero()
            } else {
                self.start + (I::one() << next.len_exponent[0_usize] as usize)
            };

            next
        })
    }
}

impl<I: RangeIntTrait> From<Range<I>> for RangeToRange1Iter<I> {
    fn from(Range { start, end }: Range<I>) -> Self {
        Self { start, end }
    }
}

struct RangeToRangeDIter<I: RangeIntTrait, const D: usize> {
    range: Range<[I; D]>,
    range_1_iters: [Peekable<RangeToRange1Iter<I>>; D],
}

impl<I: RangeIntTrait, const D: usize> Iterator for RangeToRangeDIter<I, D> {
    type Item = RangeD<I, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.range_1_iters
            .last_mut()
            .unwrap()
            .peek()
            .is_some()
            .then(|| {
                (!(0_usize..D)
                    .try_fold(false, |carry, index| {
                        if carry {
                            // Consume the value we've been peeking.
                            self.range_1_iters[index].next();
                        }

                        if self.range_1_iters[index].peek().is_some() {
                            Some(false)
                        } else {
                            let range: Range<I> = self.range.start[index]..self.range.end[index];

                            if range.is_empty() {
                                // Poison the last range 1 iter.
                                *self.range_1_iters.last_mut().unwrap() =
                                    RangeToRange1Iter::from(I::zero()..I::zero()).peekable();

                                None
                            } else {
                                self.range_1_iters[index] =
                                    RangeToRange1Iter::from(range).peekable();

                                Some(true)
                            }
                        }
                    })
                    .unwrap_or(true))
                .then(|| {
                    let mut next: Self::Item = Self::Item::default();

                    for index in 0_usize..D {
                        next.set_component(
                            index,
                            if index == 0_usize {
                                self.range_1_iters[index].next().unwrap()
                            } else {
                                self.range_1_iters[index].peek().unwrap().clone()
                            },
                        );
                    }

                    next
                })
            })
            .flatten()
    }
}

impl<I: RangeIntTrait, const D: usize, A> From<Range<A>> for RangeToRangeDIter<I, D>
where
    [I; D]: From<A>,
{
    fn from(Range { start, end }: Range<A>) -> Self {
        let range: Range<[I; D]> = start.into()..end.into();

        let mut maybe_uninit_range_1_iters: MaybeUninit<[Peekable<RangeToRange1Iter<I>>; D]> =
            MaybeUninit::uninit();

        {
            // SAFETY: We're transmuting from a maybe uninit array of peekables to an array of maybe
            // uninit peekables, which doesn't make any assumptions about initialization state.
            let maybe_uninit_range_1_iters: &mut [MaybeUninit<Peekable<RangeToRange1Iter<I>>>; D] =
                unsafe { transmute(&mut maybe_uninit_range_1_iters) };

            for index in 0_usize..maybe_uninit_range_1_iters.len() {
                maybe_uninit_range_1_iters[index].write(
                    RangeToRange1Iter::from(range.start[index]..range.end[index]).peekable(),
                );
            }
        }

        // SAFETY: All elements of maybe_uninit_range_1_iters were just initialized.
        let range_1_iters: [Peekable<RangeToRange1Iter<I>>; D] =
            unsafe { maybe_uninit_range_1_iters.assume_init() };

        Self {
            range,
            range_1_iters,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RangeD<I: RangeIntTrait, const D: usize> {
    pub start: [I; D],
    pub len_exponent: [u8; D],
}

impl<I: RangeIntTrait, const D: usize> RangeD<I, D> {
    pub fn all_regions() -> Self {
        Self {
            start: [I::min_value(); D],
            len_exponent: [bits::<I>() as u8; D],
        }
    }

    pub fn pos_region() -> Self {
        if is_signed::<I>() {
            Self {
                start: [I::min_value(); D],
                len_exponent: [bits::<I>() as u8 - 1_u8; D],
            }
        } else {
            Self::all_regions()
        }
    }

    fn get_component(&self, index: usize) -> RangeD<I, 1_usize> {
        RangeD {
            start: [self.start[index]],
            len_exponent: [self.len_exponent[index]],
        }
    }

    fn try_intersect_internal<J: PrimInt>(&self, other: &Self) -> Option<Self> {
        let mut intersection: Self = self.clone();

        for index in 0_usize..D {
            let self_component: RangeD<I, 1_usize> = self.get_component(index);
            let other_component: RangeD<I, 1_usize> = other.get_component(index);

            intersection.set_component(
                index,
                self_component.try_intersect_1d::<J>(&other_component)?,
            );
        }

        Some(intersection)
    }

    pub fn try_intersect(&self, other: &Self) -> Option<Self> {
        if bits::<I>() == u64::BITS {
            self.try_intersect_internal::<i128>(other)
        } else {
            self.try_intersect_internal::<i64>(other)
        }
    }

    pub fn len_product(&self) -> usize {
        self.len_exponent
            .iter()
            .map(|len_exponent| 1_usize << *len_exponent as u32)
            .product()
    }

    fn set_component(&mut self, index: usize, component: RangeD<I, 1_usize>) {
        self.start[index] = component.start[0_usize];
        self.len_exponent[index] = component.len_exponent[0_usize];
    }

    const fn children_len() -> usize {
        1_usize << D
    }

    fn iter_children(&self) -> impl Iterator<Item = Self> {
        let mut children_per_component: [[RangeD<I, 1_usize>; 2_usize]; D] =
            LargeArrayDefault::large_array_default();
        let mut children_len_per_component: [usize; D] = [0_usize; D];

        for (index, (component_children, component_children_len)) in children_per_component
            .iter_mut()
            .zip(children_len_per_component.iter_mut())
            .enumerate()
        {
            self.get_component(index)
                .compute_children(component_children, component_children_len);
        }

        let mut curr_child: Self = Self::default();

        for (index, component_children) in children_per_component.iter().enumerate() {
            curr_child.set_component(index, component_children[0_usize]);
        }

        let last_child_index: u8 = children_len_per_component.iter().enumerate().fold(
            0_u8,
            |last_child_index, (index, children_len)| {
                last_child_index | (((*children_len == 2_usize) as u8) << index)
            },
        );
        let curr_child_index: u8 = 0_u8;

        RangeDChildrenIter {
            children_per_component,
            curr_child,
            last_child_index,
            curr_child_index,
        }
    }

    pub fn iter_from_start_and_end<A>(range: Range<A>) -> impl Iterator<Item = Self>
    where
        [I; D]: From<A>,
    {
        RangeToRangeDIter::from(range)
    }
}

impl<I: RangeIntTrait> RangeD<I, 1_usize> {
    fn try_intersect_1d<J: PrimInt>(&self, other: &Self) -> Option<Self> {
        let self_start: J = <J as NumCast>::from(self.start[0_usize]).unwrap();
        let other_start: J = <J as NumCast>::from(other.start[0_usize]).unwrap();
        let self_len: J = J::one() << self.len_exponent[0_usize] as usize;
        let other_len: J = J::one() << other.len_exponent[0_usize] as usize;
        let self_end: J = self_start + self_len;
        let other_end: J = other_start + other_len;

        (self_start < other_end && other_start < self_end).then(|| {
            let intersection_start: J = self_start.max(other_start);
            let intersection_end: J = self_end.min(other_end);
            let intersection_len: J = intersection_end - intersection_start;

            Self {
                start: [<I as NumCast>::from(intersection_start).unwrap()],
                len_exponent: [intersection_len.trailing_zeros() as u8],
            }
        })
    }

    fn compute_children(&self, children: &mut [Self], children_len: &mut usize) {
        assert_eq!(children.len(), Self::children_len());

        *children_len = 0_usize;

        let len_exponent: u8 = self.len_exponent[0_usize];

        if let Some(len_exponent) = len_exponent.checked_sub(1_u8) {
            let children_array: [Self; 2_usize] = [
                Self {
                    start: self.start,
                    len_exponent: [len_exponent],
                },
                Self {
                    start: [self.start[0_usize] + (I::one() << len_exponent as usize)],
                    len_exponent: [len_exponent],
                },
            ];
            children[..children_array.len()].clone_from_slice(&children_array);
            *children_len = children_array.len();
        } else {
            children[0_usize] = *self;
            *children_len = 1_usize;
        }
    }

    pub fn iter_from_start_and_end_1d(range: Range<I>) -> impl Iterator<Item = Self> {
        RangeToRange1Iter::from(range)
    }
}

impl<I: RangeIntTrait> Copy for RangeD<I, 1_usize> {}

impl<I: RangeIntTrait, const D: usize> Default for RangeD<I, D> {
    fn default() -> Self {
        Self::all_regions()
    }
}

impl<I: RangeIntTrait, const D: usize, A: From<[I; D]>> From<RangeD<I, D>> for Range<A> {
    fn from(
        RangeD {
            start,
            len_exponent,
        }: RangeD<I, D>,
    ) -> Self {
        let mut end: [I; D] = start.clone();

        for (len_exponent, end) in len_exponent.into_iter().zip(end.iter_mut()) {
            *end = *end + (I::one() << (len_exponent as usize));
        }

        start.into()..end.into()
    }
}

impl<I: RangeIntTrait, const D: usize, A> TryFrom<Range<A>> for RangeD<I, D>
where
    [I; D]: From<A>,
{
    type Error = ();

    fn try_from(Range { start, end }: Range<A>) -> Result<Self, Self::Error> {
        let start: [I; D] = start.into();
        let end: [I; D] = end.into();

        let iter_len = || {
            start
                .iter()
                .zip(end.iter())
                .map(|(start, end)| *end - *start)
        };

        start
            .into_iter()
            .zip(end.into_iter())
            .all(|(start, end)| start < end)
            .then(|| {
                start
                    .iter()
                    .zip(iter_len())
                    .all(|(start, len)| {
                        len.count_ones() == 1_u32 && len.trailing_zeros() <= start.trailing_zeros()
                    })
                    .then(|| {
                        let mut len_exponent: [u8; D] = LargeArrayDefault::large_array_default();

                        for (len, len_exponent) in iter_len().zip(len_exponent.iter_mut()) {
                            *len_exponent = len.trailing_zeros() as u8;
                        }

                        Self {
                            start,
                            len_exponent,
                        }
                    })
            })
            .flatten()
            .ok_or(())
    }
}

pub type Range1<I = i32> = RangeD<I, 1_usize>;
pub type Range2<I = i32> = RangeD<I, 2_usize>;
pub type Range3<I = i32> = RangeD<I, 3_usize>;

pub trait RegionTreeValue
where
    Self: Clone,
{
    fn insert_value_into_leaf_with_matching_range(&mut self, other: &Self);

    fn should_convert_leaf_to_parent(&self, other: &Self) -> bool;

    fn get_leaf<const D: usize, I: RangeIntTrait>(
        &self,
        range: &RangeD<I, D>,
        child_range: &RangeD<I, D>,
    ) -> Self;

    fn try_convert_parent_to_leaf<'a, I>(iter: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a;
}

#[derive(Debug)]
enum RegionTreeNode<I: RangeIntTrait, const D: usize, T: RegionTreeValue> {
    Leaf(T),
    Parent(Box<[RegionTree<I, D, T>]>),
}

impl<const D: usize, I: RangeIntTrait, T: RegionTreeValue> RegionTreeNode<I, D, T> {
    fn get_leaf(&self) -> Option<&T> {
        match self {
            RegionTreeNode::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct RegionTree<I: RangeIntTrait, const D: usize, T: RegionTreeValue> {
    range: RangeD<I, D>,
    node: RegionTreeNode<I, D, T>,
}

impl<const D: usize, I: RangeIntTrait, T: RegionTreeValue> RegionTree<I, D, T> {
    pub fn new(range: RangeD<I, D>, value: T) -> Self {
        Self {
            range,
            node: RegionTreeNode::Leaf(value),
        }
    }

    fn get_leaf(&self) -> Option<&T> {
        self.node.get_leaf()
    }

    fn iter_surface_leaves(children: &[Self]) -> impl Iterator<Item = &T> {
        children.iter().filter_map(Self::get_leaf)
    }

    pub fn insert(&mut self, range: &RangeD<I, D>, value: &T) {
        if let Some(intersection) = self.range.try_intersect(range) {
            match &mut self.node {
                RegionTreeNode::Leaf(self_value) => {
                    if intersection == self.range {
                        self_value.insert_value_into_leaf_with_matching_range(value);
                    } else if self_value.should_convert_leaf_to_parent(value) {
                        let boxed_children: Box<[Self]> = self
                            .range
                            .iter_children()
                            .map(|child_range| {
                                let child_value: T = self_value.get_leaf(&self.range, &child_range);

                                Self::new(child_range, child_value)
                            })
                            .collect::<Vec<Self>>()
                            .into_boxed_slice();

                        self.node = RegionTreeNode::Parent(boxed_children);

                        self.insert(&intersection, value);
                    }
                }
                RegionTreeNode::Parent(children) => {
                    for child in children.iter_mut() {
                        child.insert(&intersection, value);
                    }

                    if let Some(leaf) = (Self::iter_surface_leaves(children).count()
                        == children.len())
                    .then(|| T::try_convert_parent_to_leaf(Self::iter_surface_leaves(children)))
                    .flatten()
                    {
                        self.node = RegionTreeNode::Leaf(leaf);
                    }
                }
            }
        }
    }

    fn visit_all_leaves_internal<
        P: Fn(&RangeD<I, D>) -> bool,
        F: FnMut(&RangeD<I, D>, &T) -> bool,
    >(
        &self,
        parent_predicate: &P,
        leaf_visitor: &mut F,
    ) -> bool {
        match &self.node {
            RegionTreeNode::Leaf(value) => leaf_visitor(&self.range, value),
            RegionTreeNode::Parent(children) => {
                !parent_predicate(&self.range)
                    || children.iter().all(|child| {
                        child.visit_all_leaves_internal(parent_predicate, leaf_visitor)
                    })
            }
        }
    }

    pub fn visit_all_leaves<P: Fn(&RangeD<I, D>) -> bool, F: FnMut(&RangeD<I, D>, &T) -> bool>(
        &self,
        parent_predicate: P,
        leaf_visitor: F,
    ) {
        let mut leaf_visitor: F = leaf_visitor;

        self.visit_all_leaves_internal(&parent_predicate, &mut leaf_visitor);
    }
}

pub type BinaryTree<T, I = i32> = RegionTree<I, 1_usize, T>;
pub type QuadTree<T, I = i32> = RegionTree<I, 2_usize, T>;
pub type OctTree<T, I = i32> = RegionTree<I, 3_usize, T>;

#[cfg(test)]
mod tests {
    use {
        super::*,
        glam::{
            i16::{I16Vec2, I16Vec3, I16Vec4},
            i32::{IVec2, IVec3, IVec4},
            i64::{I64Vec2, I64Vec3, I64Vec4},
            u16::{U16Vec2, U16Vec3, U16Vec4},
            u32::{UVec2, UVec3, UVec4},
            u64::{U64Vec2, U64Vec3, U64Vec4},
        },
        rand::prelude::*,
        std::{cmp::Ordering, sync::OnceLock},
    };

    const RANGE: Range<i32> = -8_i32..8_i32;
    const RAND_TEST_COUNT: usize = 10_usize;

    #[derive(Debug, Default)]
    struct ValidTestData {
        len_exponent: u8,
        children: [Range<i32>; 2_usize],
        children_len: usize,
    }

    impl ValidTestData {
        fn children(&self) -> &[Range<i32>] {
            &self.children[..self.children_len]
        }
    }

    #[derive(Debug, Default)]
    struct Test {
        range: Range<i32>,
        valid_test_data: Option<Box<ValidTestData>>,
    }

    #[derive(Default)]
    struct Tests {
        valid_unsigned_tests: Vec<Test>,
        valid_signed_tests: Vec<Test>,
        invalid_unsigned_tests: Vec<Test>,
        invalid_signed_tests: Vec<Test>,
    }

    impl Tests {
        fn new() -> Self {
            let mut tests: Self = Self::default();

            for start in RANGE {
                for end in (start + 1_i32)..RANGE.end {
                    let range: Range<i32> = start..end;
                    let delta: i32 = end - start;
                    let valid_test_data: Option<Box<ValidTestData>> =
                        (delta.count_ones() == 1_u32 && start % delta == 0_i32).then(|| {
                            let children_delta: i32 = (delta / 2_i32).max(1_i32);
                            let len_exponent: u8 = delta.trailing_zeros() as u8;
                            let mut children: [Range<i32>; 2_usize] =
                                LargeArrayDefault::large_array_default();
                            let children_len: usize = (len_exponent > 0_u8) as usize + 1_usize;

                            for (index, child) in children[..children_len].iter_mut().enumerate() {
                                let offset: i32 = index as i32 * children_delta;
                                let child_start = start + offset;
                                let child_end = child_start + children_delta;

                                *child = child_start..child_end;
                            }

                            Box::new(ValidTestData {
                                len_exponent,
                                children,
                                children_len,
                            })
                        });
                    let test: Test = Test {
                        range,
                        valid_test_data,
                    };

                    match (test.valid_test_data.is_some(), start >= 0_i32) {
                        (true, true) => &mut tests.valid_unsigned_tests,
                        (true, false) => &mut tests.valid_signed_tests,
                        (false, true) => &mut tests.invalid_unsigned_tests,
                        (false, false) => &mut tests.invalid_signed_tests,
                    }
                    .push(test);
                }
            }

            tests
        }

        fn get() -> &'static Self {
            static ONCE_LOCK: OnceLock<Tests> = OnceLock::new();

            ONCE_LOCK.get_or_init(Self::new)
        }

        fn random_test<I: RangeIntTrait>(&self, valid: bool, thread_rng: &mut ThreadRng) -> &Test {
            match (valid, !is_signed::<I>()) {
                (true, true) => &self.valid_unsigned_tests,
                (true, false) => {
                    if thread_rng.gen::<bool>() {
                        &self.valid_unsigned_tests
                    } else {
                        &self.valid_signed_tests
                    }
                }
                (false, true) => &self.invalid_unsigned_tests,
                (false, false) => {
                    if thread_rng.gen::<bool>() {
                        &self.invalid_unsigned_tests
                    } else {
                        &self.invalid_signed_tests
                    }
                }
            }
            .choose(thread_rng)
            .unwrap()
        }
    }

    struct TestAndDirection {
        test: &'static Test,
        direction: Ordering,
    }

    impl TestAndDirection {
        fn range(&self) -> Range<i32> {
            let range: Range<i32> = self.test.range.clone();

            match self.direction {
                Ordering::Less => range.end..range.start,
                Ordering::Equal => range.start..range.start,
                Ordering::Greater => range,
            }
        }
    }

    #[derive(Clone)]
    struct RangeParams {
        valid: bool,
        direction: Ordering,
    }

    impl Default for RangeParams {
        fn default() -> Self {
            Self {
                valid: true,
                direction: Ordering::Greater,
            }
        }
    }

    fn glam_vec_range_from_range_i32s<I, const D: usize, V, Iter>(ranges: Iter) -> Range<V>
    where
        I: RangeIntTrait,
        V: Copy + From<[I; D]>,
        Iter: Iterator<Item = Range<i32>>,
    {
        let mut start: [I; D] = [I::zero(); D];
        let mut end: [I; D] = [I::zero(); D];
        let mut len: usize = 0_usize;

        for range in ranges {
            start[len] = <I as NumCast>::from(range.start).unwrap();
            end[len] = <I as NumCast>::from(range.end).unwrap();
            len += 1_usize;
        }

        assert_eq!(len, D);

        V::from(start)..V::from(end)
    }

    fn rand_glam_vec_range_and_expectation<I, const D: usize, V>(
        range_params: [RangeParams; D],
        thread_rng: &mut ThreadRng,
    ) -> (Range<V>, Result<RangeD<I, D>, ()>)
    where
        I: RangeIntTrait,
        V: Copy + From<[I; D]>,
        [I; D]: From<V>,
    {
        let mut tests_and_directions: Vec<TestAndDirection> = Vec::new();

        let is_valid: bool = range_params
            .iter()
            .all(|range_params| range_params.valid && range_params.direction.is_gt());

        tests_and_directions.extend(range_params.into_iter().map(|range_params| {
            TestAndDirection {
                test: Tests::get().random_test::<I>(range_params.valid, thread_rng),
                direction: range_params.direction,
            }
        }));
        tests_and_directions.shuffle(thread_rng);

        let glam_vec_range: Range<V> = glam_vec_range_from_range_i32s(
            tests_and_directions
                .iter()
                .map(|tests_and_direction| tests_and_direction.range()),
        );
        let expectation: Result<RangeD<I, D>, ()> = is_valid
            .then(|| {
                let start: [I; D] = glam_vec_range.start.into();
                let mut len_exponent: [u8; D] = [0_u8; D];

                for (src_len_exponent, dst_len_exponent) in tests_and_directions
                    .into_iter()
                    .map(|test_and_direction| {
                        test_and_direction
                            .test
                            .valid_test_data
                            .as_ref()
                            .unwrap()
                            .len_exponent
                    })
                    .zip(len_exponent.iter_mut())
                {
                    *dst_len_exponent = src_len_exponent;
                }

                RangeD {
                    start,
                    len_exponent,
                }
            })
            .ok_or(());

        (glam_vec_range, expectation)
    }

    fn test_range_d_try_from_glam_vec_range_generic_test<I, const D: usize, V>(
        source_range_params: RangeParams,
        thread_rng: &mut ThreadRng,
    ) where
        I: RangeIntTrait,
        V: Copy + Debug + From<[I; D]>,
        [I; D]: From<V> + Default,
    {
        for negative_range_count in 1_usize..D {
            for _ in 0_usize..RAND_TEST_COUNT {
                let mut range_params: [RangeParams; D] = LargeArrayDefault::large_array_default();

                range_params[..negative_range_count].fill(source_range_params.clone());

                let (glam_vec_range, expectation): (Range<V>, Result<RangeD<I, D>, ()>) =
                    rand_glam_vec_range_and_expectation(range_params, thread_rng);

                let reality: Result<RangeD<I, D>, ()> = glam_vec_range.clone().try_into();

                assert_eq!(reality, expectation, "glam_vec_range: {glam_vec_range:#?}");
            }
        }
    }

    fn test_range_d_try_from_glam_vec_range_generic_suite<I, const D: usize, V>(
        thread_rng: &mut ThreadRng,
    ) where
        I: RangeIntTrait,
        V: Copy + Debug + From<[I; D]>,
        [I; D]: From<V> + Default,
    {
        // Validate negative ranges fail.
        test_range_d_try_from_glam_vec_range_generic_test(
            RangeParams {
                valid: true,
                direction: Ordering::Less,
            },
            thread_rng,
        );

        // Validate empty ranges fail.
        test_range_d_try_from_glam_vec_range_generic_test(
            RangeParams {
                valid: true,
                direction: Ordering::Equal,
            },
            thread_rng,
        );

        // Validate invalid ranges fail.
        test_range_d_try_from_glam_vec_range_generic_test(
            RangeParams {
                valid: false,
                direction: Ordering::Greater,
            },
            thread_rng,
        );

        // Validate valid ranges succeed.
        test_range_d_try_from_glam_vec_range_generic_test(RangeParams::default(), thread_rng);
    }

    fn rand_range_d_and_children<I, const D: usize, V>(
        thread_rng: &mut ThreadRng,
    ) -> (RangeD<I, D>, Vec<RangeD<I, D>>)
    where
        I: RangeIntTrait,
        V: Copy + From<[I; D]>,
        [I; D]: From<V>,
    {
        let tests: Vec<&Test> = (0_usize..D)
            .into_iter()
            .map(|_| Tests::get().random_test::<I>(true, thread_rng))
            .collect();
        let range: RangeD<I, D> = glam_vec_range_from_range_i32s::<I, D, V, _>(
            tests.iter().map(|test| test.range.clone()),
        )
        .try_into()
        .unwrap();
        let children: Vec<RangeD<I, D>> = (0_usize..(1_usize << D))
            .filter_map(|child_index| {
                let child_index_bitslice: &BitSlice = child_index.view_bits::<Lsb0>();

                child_index_bitslice
                    .iter_ones()
                    .all(|component_index| {
                        tests[component_index]
                            .valid_test_data
                            .as_ref()
                            .unwrap()
                            .children_len
                            == 2_usize
                    })
                    .then(|| {
                        glam_vec_range_from_range_i32s::<I, D, V, _>((0_usize..D).map(
                            |component_index| {
                                tests[component_index]
                                    .valid_test_data
                                    .as_ref()
                                    .unwrap()
                                    .children()
                                    [child_index_bitslice[component_index] as usize]
                                    .clone()
                            },
                        ))
                        .try_into()
                        .unwrap()
                    })
            })
            .collect();

        (range, children)
    }

    fn test_range_d_iter_children_generic_test<I, const D: usize, V>(thread_rng: &mut ThreadRng)
    where
        I: RangeIntTrait,
        V: Copy + Debug + From<[I; D]>,
        [I; D]: From<V> + Default,
    {
        for _ in 0_usize..RAND_TEST_COUNT {
            let (range_d, children): (RangeD<I, D>, Vec<RangeD<I, D>>) =
                rand_range_d_and_children::<I, D, V>(thread_rng);

            assert_eq!(
                range_d.iter_children().collect::<Vec<RangeD<I, D>>>(),
                children
            );
        }
    }

    fn midpoint<I: RangeIntTrait>(Range { start, end }: Range<I>) -> I {
        assert!(start <= end);

        match (start <= I::zero(), end >= I::zero()) {
            (true, true) => I::zero(),
            (true, false) => {
                if start == I::min_value() {
                    start
                } else {
                    I::zero() - midpoint(I::zero() - end..I::zero() - start)
                }
            }
            (false, true) => {
                let power: I =
                    I::one() << (bits::<I>() - (start ^ end).leading_zeros() - 1_u32) as usize;

                if start % power == I::zero() {
                    start
                } else {
                    ((start / power) + I::one()) * power
                }
            }
            (false, false) => unimplemented!(),
        }
    }

    fn range_1s_from_range_i<I: RangeIntTrait>(
        Range { mut start, end }: Range<I>,
    ) -> Vec<RangeD<I, 1_usize>> {
        let mut range_1s: Vec<RangeD<I, 1_usize>> = Vec::new();
        let midpoint: I = midpoint(start..end);

        while start != midpoint {
            let len_exponent: u8 = start.trailing_zeros() as u8;

            range_1s.push(RangeD {
                start: [start],
                len_exponent: [len_exponent],
            });

            start = if (RangeToRange1Iter {
                start,
                end: midpoint,
            })
            .step_fits_in_i()
            {
                start + (I::one() << len_exponent as usize)
            } else {
                I::zero()
            };
        }

        while start != end {
            let len_exponent: u8 = (bits::<I>() - (end - start).leading_zeros() - 1_u32) as u8;

            range_1s.push(RangeD {
                start: [start],
                len_exponent: [len_exponent],
            });

            start = start + (I::one() << len_exponent as usize);
        }

        range_1s
    }

    fn test_range_to_range_1_iter_generic_test_count<I: RangeIntTrait>(
        start: i32,
        end: i32,
        count: usize,
    ) {
        assert_eq_break(
            RangeToRange1Iter::from(
                <I as NumCast>::from(start).unwrap()..<I as NumCast>::from(end).unwrap(),
            )
            .count(),
            count,
        );
    }

    fn test_range_to_range_1_iter_generic_test<I: RangeIntTrait>() {
        let endpoints: Vec<I> = is_signed::<I>()
            .then_some([I::min_value()])
            .into_iter()
            .flatten()
            .chain(
                is_signed::<I>()
                    .then_some(-16_i32..=-1_i32)
                    .into_iter()
                    .flatten()
                    .chain(0_i32..=16_i32)
                    .map(|i| <I as NumCast>::from(i).unwrap()),
            )
            .chain([I::max_value()])
            .collect();

        // Necessary to not have the closure move the full `Vec`.
        let endpoints: &[I] = &endpoints;

        for i in 0_i32..=16_i32 {
            let i_count_ones: usize = i.count_ones() as usize;

            test_range_to_range_1_iter_generic_test_count::<I>(i, 0_i32, 0_usize);
            test_range_to_range_1_iter_generic_test_count::<I>(i, i, 0_usize);
            test_range_to_range_1_iter_generic_test_count::<I>(0_i32, i, i_count_ones);
            test_range_to_range_1_iter_generic_test_count::<I>(16_i32 - i, 16_i32, i_count_ones);
            test_range_to_range_1_iter_generic_test_count::<I>(
                i,
                32_i32 - i,
                if i == 0_i32 {
                    1_usize
                } else {
                    2_usize * (16_i32 - i).count_ones() as usize
                },
            );
        }

        for range in (0_usize..endpoints.len() - 1_usize)
            .into_iter()
            .flat_map(|a| {
                (a + 1_usize..endpoints.len())
                    .into_iter()
                    .map(move |b| endpoints[a]..endpoints[b])
            })
        {
            assert_eq!(
                RangeToRange1Iter::from(range.clone()).collect::<Vec<RangeD<I, 1_usize>>>(),
                range_1s_from_range_i(range)
            );
        }
    }

    fn range_to_range_d_iter_rand_range_end(thread_rng: &mut ThreadRng) -> i32 {
        const RANGES: &'static [i32] = &[1_i32, 3_i32, 7_i32];

        *RANGES.choose(thread_rng).unwrap()
    }

    fn range_to_range_d_iter_rand_glam_vec_range<I, const D: usize, V>(
        thread_rng: &mut ThreadRng,
        invalid_components: usize,
    ) -> Range<V>
    where
        I: RangeIntTrait,
        V: Copy + Debug + From<[I; D]>,
        [I; D]: Clone + From<V> + Default,
    {
        let mut ranges: [Range<i32>; D] = unsafe { MaybeUninit::zeroed().assume_init() };

        for (index, range) in ranges.iter_mut().enumerate() {
            range.start = index as i32 * 8_i32;

            if index >= invalid_components {
                range.end = range.start + range_to_range_d_iter_rand_range_end(thread_rng);
            } else {
                range.end = range.start;
            }
        }

        glam_vec_range_from_range_i32s(ranges.into_iter())
    }

    fn range_ds_from_component_to_range_1s<I: RangeIntTrait, const D: usize>(
        component_to_range_1s: &[Vec<RangeD<I, 1_usize>>],
        component_index: usize,
        range_d: &mut RangeD<I, D>,
        range_ds: &mut Vec<RangeD<I, D>>,
    ) {
        for range_1 in &component_to_range_1s[component_index] {
            range_d.set_component(component_index, range_1.clone());

            if component_index == 0_usize {
                range_ds.push(range_d.clone());
            } else {
                range_ds_from_component_to_range_1s(
                    component_to_range_1s,
                    component_index - 1_usize,
                    range_d,
                    range_ds,
                );
            }
        }
    }

    fn range_ds_from_glam_vec_range<I, const D: usize, V>(
        Range { start, end }: Range<V>,
    ) -> Vec<RangeD<I, D>>
    where
        I: RangeIntTrait,
        V: Copy,
        [I; D]: From<V>,
    {
        let start: [I; D] = start.into();
        let end: [I; D] = end.into();

        if start
            .iter()
            .zip(end.iter())
            .any(|(start, end)| *end <= *start)
        {
            Vec::new()
        } else {
            let component_to_range_1s: Vec<Vec<RangeD<I, 1_usize>>> = start
                .into_iter()
                .zip(end.into_iter())
                .map(|(start, end)| range_1s_from_range_i(start..end))
                .collect();
            let mut range_d: RangeD<I, D> = RangeD::default();
            let mut range_ds: Vec<RangeD<I, D>> = Vec::new();

            range_ds_from_component_to_range_1s(
                &component_to_range_1s,
                D - 1_usize,
                &mut range_d,
                &mut range_ds,
            );

            range_ds
        }
    }

    fn test_range_to_range_d_iter_generic_test<I, const D: usize, V>(thread_rng: &mut ThreadRng)
    where
        I: RangeIntTrait,
        V: Copy + Debug + From<[I; D]>,
        [I; D]: From<V> + Default,
    {
        for invalid_components in 0_usize..=D {
            for _ in 0_usize..RAND_TEST_COUNT {
                let glam_vec_range: Range<V> = range_to_range_d_iter_rand_glam_vec_range::<I, D, V>(
                    thread_rng,
                    invalid_components,
                );

                assert_eq_break(
                    RangeToRangeDIter::from(glam_vec_range.clone()).collect::<Vec<RangeD<I, D>>>(),
                    range_ds_from_glam_vec_range(glam_vec_range),
                );
            }
        }
    }

    #[test]
    fn test_range_d_try_from_glam_vec_range() {
        let mut thread_rng: ThreadRng = thread_rng();

        // i16
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<i16, 2_usize, I16Vec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i16, 3_usize, I16Vec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i16, 4_usize, I16Vec4>(
                &mut thread_rng,
            );
        }

        // i32
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<i32, 2_usize, IVec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i32, 3_usize, IVec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i32, 4_usize, IVec4>(
                &mut thread_rng,
            );
        }

        // i64
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<i64, 2_usize, I64Vec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i64, 3_usize, I64Vec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<i64, 4_usize, I64Vec4>(
                &mut thread_rng,
            );
        }

        // u16
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<u16, 2_usize, U16Vec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u16, 3_usize, U16Vec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u16, 4_usize, U16Vec4>(
                &mut thread_rng,
            );
        }

        // u32
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<u32, 2_usize, UVec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u32, 3_usize, UVec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u32, 4_usize, UVec4>(
                &mut thread_rng,
            );
        }

        // u64
        {
            test_range_d_try_from_glam_vec_range_generic_suite::<u64, 2_usize, U64Vec2>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u64, 3_usize, U64Vec3>(
                &mut thread_rng,
            );
            test_range_d_try_from_glam_vec_range_generic_suite::<u64, 4_usize, U64Vec4>(
                &mut thread_rng,
            );
        }
    }

    fn range_2_from_ranges(range_0: Range<i32>, range_1: Range<i32>) -> Range2 {
        Range2::try_from(
            IVec2::new(range_0.start, range_1.start)..IVec2::new(range_0.end, range_1.end),
        )
        .unwrap()
    }

    #[test]
    fn test_range_d_iter_children() {
        assert_eq!(
            range_2_from_ranges(0_i32..1024_i32, 0_i32..1024_i32)
                .iter_children()
                .collect::<Vec<Range2>>(),
            vec![
                range_2_from_ranges(0_i32..512_i32, 0_i32..512_i32),
                range_2_from_ranges(512_i32..1024_i32, 0_i32..512_i32),
                range_2_from_ranges(0_i32..512_i32, 512_i32..1024_i32),
                range_2_from_ranges(512_i32..1024_i32, 512_i32..1024_i32),
            ]
        );

        let mut thread_rng: ThreadRng = thread_rng();

        // i16
        {
            test_range_d_iter_children_generic_test::<i16, 2_usize, I16Vec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i16, 3_usize, I16Vec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i16, 4_usize, I16Vec4>(&mut thread_rng);
        }

        // i32
        {
            test_range_d_iter_children_generic_test::<i32, 2_usize, IVec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i32, 3_usize, IVec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i32, 4_usize, IVec4>(&mut thread_rng);
        }

        // i64
        {
            test_range_d_iter_children_generic_test::<i64, 2_usize, I64Vec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i64, 3_usize, I64Vec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<i64, 4_usize, I64Vec4>(&mut thread_rng);
        }

        // u16
        {
            test_range_d_iter_children_generic_test::<u16, 2_usize, U16Vec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u16, 3_usize, U16Vec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u16, 4_usize, U16Vec4>(&mut thread_rng);
        }

        // u32
        {
            test_range_d_iter_children_generic_test::<u32, 2_usize, UVec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u32, 3_usize, UVec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u32, 4_usize, UVec4>(&mut thread_rng);
        }

        // u64
        {
            test_range_d_iter_children_generic_test::<u64, 2_usize, U64Vec2>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u64, 3_usize, U64Vec3>(&mut thread_rng);
            test_range_d_iter_children_generic_test::<u64, 4_usize, U64Vec4>(&mut thread_rng);
        }
    }

    #[test]
    fn test_midpoint() {
        // u8 cases
        assert_eq!(midpoint(0_u8..0_u8), 0_u8);
        assert_eq!(midpoint(0_u8..7_u8), 0_u8);
        assert_eq!(midpoint(1_u8..8_u8), 8_u8);
        assert_eq!(midpoint(1_u8..15_u8), 8_u8);
        assert_eq!(midpoint(2_u8..3_u8), 2_u8);
        assert_eq!(midpoint(4_u8..6_u8), 4_u8);
        assert_eq!(midpoint(9_u8..15_u8), 12_u8);
        assert_eq!(midpoint(13_u8..15_u8), 14_u8);
        assert_eq!(midpoint(u8::MIN..u8::MAX), 0_u8);

        // strictly negative i8 cases
        assert_eq!(midpoint(-15_i8..-13_i8), -14_i8);
        assert_eq!(midpoint(-15_i8..-9_i8), -12_i8);
        assert_eq!(midpoint(-15_i8..-1_i8), -8_i8);
        assert_eq!(midpoint(-8_i8..-1_i8), -8_i8);
        assert_eq!(midpoint(-7_i8..-1_i8), -4_i8);
        assert_eq!(midpoint(-6_i8..-4_i8), -4_i8);
        assert_eq!(midpoint(-3_i8..-2_i8), -2_i8);

        // other i8 cases
        assert_eq!(midpoint(-7_i8..0_i8), 0_i8);
        assert_eq!(midpoint(-7_i8..1_i8), 0_i8);
        assert_eq!(midpoint(-7_i8..7_i8), 0_i8);
        assert_eq!(midpoint(-1_i8..7_i8), 0_i8);
        assert_eq!(midpoint(i8::MIN..0_i8), 0_i8);
        assert_eq!(midpoint(i8::MIN..i8::MAX), 0_i8);
        assert_eq!(midpoint(0_i8..0_i8), 0_i8);
        assert_eq!(midpoint(0_i8..i8::MAX), 0_i8);
    }

    #[test]
    fn test_range_to_range_1_iter() {
        test_range_to_range_1_iter_generic_test::<u16>();
        test_range_to_range_1_iter_generic_test::<u32>();
        test_range_to_range_1_iter_generic_test::<u64>();
        test_range_to_range_1_iter_generic_test::<i16>();
        test_range_to_range_1_iter_generic_test::<i32>();
        test_range_to_range_1_iter_generic_test::<i64>();
    }

    #[test]
    fn test_range_to_range_d_iter() {
        let mut thread_rng: ThreadRng = thread_rng();

        // i16
        {
            test_range_to_range_d_iter_generic_test::<i16, 2_usize, I16Vec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i16, 3_usize, I16Vec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i16, 4_usize, I16Vec4>(&mut thread_rng);
        }

        // i32
        {
            test_range_to_range_d_iter_generic_test::<i32, 2_usize, IVec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i32, 3_usize, IVec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i32, 4_usize, IVec4>(&mut thread_rng);
        }

        // i64
        {
            test_range_to_range_d_iter_generic_test::<i64, 2_usize, I64Vec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i64, 3_usize, I64Vec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<i64, 4_usize, I64Vec4>(&mut thread_rng);
        }

        // u16
        {
            test_range_to_range_d_iter_generic_test::<u16, 2_usize, U16Vec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u16, 3_usize, U16Vec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u16, 4_usize, U16Vec4>(&mut thread_rng);
        }

        // u32
        {
            test_range_to_range_d_iter_generic_test::<u32, 2_usize, UVec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u32, 3_usize, UVec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u32, 4_usize, UVec4>(&mut thread_rng);
        }

        // u64
        {
            test_range_to_range_d_iter_generic_test::<u64, 2_usize, U64Vec2>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u64, 3_usize, U64Vec3>(&mut thread_rng);
            test_range_to_range_d_iter_generic_test::<u64, 4_usize, U64Vec4>(&mut thread_rng);
        }
    }
}
