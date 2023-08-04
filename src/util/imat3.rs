use {
    glam::IVec3,
    std::ops::{Add, Mul},
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct IMat3 {
    pub x_axis: IVec3,
    pub y_axis: IVec3,
    pub z_axis: IVec3,
}

impl IMat3 {
    /// A 3x3 matrix with all elements set to `0`.
    pub const ZERO: Self = Self {
        x_axis: IVec3::ZERO,
        y_axis: IVec3::ZERO,
        z_axis: IVec3::ZERO,
    };

    /// A 3x3 identity matrix, where all diagonal elements are `1`, and all off-diagonal elements
    /// are `0`.
    pub const IDENTITY: Self = Self {
        x_axis: IVec3::X,
        y_axis: IVec3::Y,
        z_axis: IVec3::Z,
    };

    #[inline(always)]
    pub const fn from_cols(x_axis: IVec3, y_axis: IVec3, z_axis: IVec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    #[inline(always)]
    pub const fn from_rows(x_axis: IVec3, y_axis: IVec3, z_axis: IVec3) -> Self {
        Self {
            x_axis: IVec3::new(x_axis.x, y_axis.x, z_axis.x),
            y_axis: IVec3::new(x_axis.y, y_axis.y, z_axis.y),
            z_axis: IVec3::new(x_axis.z, y_axis.z, z_axis.z),
        }
    }

    /// Transforms a 3D vector.
    #[inline]
    pub fn mul_ivec3(&self, rhs: IVec3) -> IVec3 {
        let mut res = self.x_axis.mul(rhs.x);
        res = res.add(self.y_axis.mul(rhs.y));
        res = res.add(self.z_axis.mul(rhs.z));
        res
    }

    /// Multiplies two 3x3 matrices.
    #[inline]
    pub fn mul_imat3(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_ivec3(rhs.x_axis),
            self.mul_ivec3(rhs.y_axis),
            self.mul_ivec3(rhs.z_axis),
        )
    }
}

impl Default for IMat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul<IVec3> for IMat3 {
    type Output = IVec3;

    fn mul(self, rhs: IVec3) -> Self::Output {
        self.mul_ivec3(rhs)
    }
}

impl Mul<IMat3> for IMat3 {
    type Output = Self;

    fn mul(self, rhs: IMat3) -> Self::Output {
        self.mul_imat3(&rhs)
    }
}
