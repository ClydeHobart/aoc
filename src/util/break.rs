use {
    num::PrimInt,
    std::{fmt::Debug, hint::black_box},
};

pub fn debug_break() {
    black_box(());
}

pub fn cond_debug_break(cond: bool) {
    if cond {
        debug_break();
    }
}

pub fn add_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_add(&b) {
        Some(sum) => sum,
        None => {
            debug_break();

            panic!();
        }
    }
}

pub fn sub_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_sub(&b) {
        Some(difference) => difference,
        None => {
            debug_break();

            panic!();
        }
    }
}

pub fn mul_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_mul(&b) {
        Some(product) => product,
        None => {
            debug_break();

            panic!();
        }
    }
}

pub fn div_break_on_overflow<I: PrimInt>(a: I, b: I) -> I {
    match a.checked_div(&b) {
        Some(quotient) => quotient,
        None => {
            debug_break();

            panic!();
        }
    }
}

pub fn assert_eq_break<T: Debug + PartialEq>(left: T, right: T) {
    if left != right {
        debug_break();

        assert_eq!(left, right);
    }
}

pub fn slice_assert_eq_break<T: Debug + PartialEq>(left: &[T], right: &[T]) {
    assert_eq_break(left.len(), right.len());

    for (_index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq_break(left, right);
    }
}

pub fn map_break<T>(value: T) -> T {
    debug_break();

    value
}

pub trait UnwrapBreak<T> {
    fn unwrap_break(self) -> T;
}

impl<T> UnwrapBreak<T> for Option<T> {
    fn unwrap_break(self) -> T {
        if self.is_none() {
            debug_break();

            panic!()
        } else {
            self.unwrap()
        }
    }
}
