pub use {grid_2d::*, grid_3d::*, grid_4d::*};

use {
    super::*,
    glam::{
        I16Vec2, I16Vec3, I16Vec4, I64Vec2, I64Vec3, I64Vec4, I8Vec2, I8Vec3, I8Vec4, IVec2, IVec3,
        IVec4,
    },
    std::ops::{Index, Sub},
};

mod grid_2d;
mod grid_3d;
mod grid_4d;

#[derive(Debug)]
pub enum CellIterFromRangeError {
    PositionsIdentical,
    PositionsNotAligned,
}

pub trait Manhattan
where
    Self: Copy + Index<usize> + Sub<Self, Output = Self> + Sized,
    <Self as Index<usize>>::Output: Sized,
{
    fn manhattan_magnitude(&self) -> <Self as Index<usize>>::Output;

    fn manhattan_distance(self, other: Self) -> <Self as Index<usize>>::Output {
        (self - other).manhattan_magnitude()
    }
}

macro_rules! impl_manhattan {
    ( $( $glam_vec:ty ),* $(,)? ) => { $(
        impl Manhattan for $glam_vec {
            fn manhattan_magnitude(&self) -> <Self as Index<usize>>::Output {
                self.abs().element_sum()
            }
        }
    )* };
}

impl_manhattan!(
    I16Vec2, I16Vec3, I16Vec4, I64Vec2, I64Vec3, I64Vec4, I8Vec2, I8Vec3, I8Vec4, IVec2, IVec3,
    IVec4,
);
