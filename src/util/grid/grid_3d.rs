use {
    super::*,
    glam::IVec3,
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        ops::{Range, RangeInclusive},
    },
};

pub struct Grid3D<T> {
    cells: Vec<T>,

    /// Should only contain unsigned values, but is signed for ease of use for iterating
    dimensions: IVec3,
}

pub fn manhattan_distance_3d(a: &IVec3, b: &IVec3) -> i32 {
    let abs_diff: IVec3 = (*b - *a).abs();

    abs_diff.x + abs_diff.y + abs_diff.z
}

pub fn sanitize_dir_3d(dir: &mut IVec3) {
    let abs: IVec3 = dir.abs();

    if abs.x + abs.y + abs.z != 1_i32 {
        *dir = if abs == IVec3::ZERO {
            IVec3::X
        } else {
            const POS_AND_NEG_AXES: [IVec3; 6_usize] = [
                IVec3::X,
                IVec3::NEG_X,
                IVec3::Y,
                IVec3::NEG_Y,
                IVec3::Z,
                IVec3::NEG_Z,
            ];

            *POS_AND_NEG_AXES
                .iter()
                .min_by_key(|axis| manhattan_distance_3d(*axis, dir))
                .unwrap()
        }
    }
}

impl<T> Grid3D<T> {
    pub fn empty(dimensions: IVec3) -> Self {
        Self {
            cells: Vec::new(),
            dimensions,
        }
    }

    pub fn allocate(dimensions: IVec3) -> Self {
        Self {
            cells: Vec::with_capacity((dimensions.x * dimensions.y * dimensions.z) as usize),
            dimensions,
        }
    }

    #[inline(always)]
    pub fn cells(&self) -> &[T] {
        &self.cells
    }

    #[inline(always)]
    pub fn cells_mut(&mut self) -> &mut [T] {
        &mut self.cells
    }

    #[inline(always)]
    pub fn dimensions(&self) -> &IVec3 {
        &self.dimensions
    }

    #[inline(always)]
    pub fn contains(&self, pos: &IVec3) -> bool {
        pos.cmpge(IVec3::ZERO).all() && pos.cmplt(self.dimensions).all()
    }

    pub fn is_border(&self, pos: &IVec3) -> bool {
        self.contains(pos)
            && (pos.cmpeq(IVec3::ZERO).any() || pos.cmpeq(self.max_dimensions()).any())
    }

    pub fn index_from_pos(&self, pos: &IVec3) -> usize {
        let [width, height, _] = self.width_height_depth();

        pos.z as usize * width * height + pos.y as usize * width + pos.x as usize
    }

    pub fn try_index_from_pos(&self, pos: &IVec3) -> Option<usize> {
        if self.contains(pos) {
            Some(self.index_from_pos(pos))
        } else {
            None
        }
    }

    pub fn pos_from_index(&self, mut index: usize) -> IVec3 {
        let [width, height, _] = self.width_height_depth();
        let width_height_product: usize = width * height;
        let z: i32 = (index / width_height_product) as i32;

        index %= width_height_product;

        let y: i32 = (index / width) as i32;

        index %= width;

        let x: i32 = index as i32;

        IVec3 { x, y, z }
    }

    #[inline(always)]
    pub fn max_dimensions(&self) -> IVec3 {
        self.dimensions - IVec3::ONE
    }

    pub fn get(&self, pos: &IVec3) -> Option<&T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &self.cells[index])
    }

    pub fn get_mut(&mut self, pos: &IVec3) -> Option<&mut T> {
        self.try_index_from_pos(pos)
            .map(|index: usize| &mut self.cells[index])
    }

    pub fn rem_euclid(&self, pos: &IVec3) -> IVec3 {
        IVec3::new(
            pos.x.rem_euclid(self.dimensions.x),
            pos.y.rem_euclid(self.dimensions.y),
            pos.z.rem_euclid(self.dimensions.z),
        )
    }

    pub fn resize_layers<F: FnMut() -> T>(&mut self, new_layer_len: usize, f: F) {
        self.dimensions.z = new_layer_len as i32;
        self.cells.resize_with(
            (self.dimensions.x * self.dimensions.y * self.dimensions.z) as usize,
            f,
        );
    }

    pub fn reserve_layers(&mut self, additional_layers: usize) {
        self.cells
            .reserve((self.dimensions.x * self.dimensions.y) as usize * additional_layers);
    }

    #[inline(always)]
    fn width_height_depth(&self) -> [usize; 3_usize] {
        [
            self.dimensions.x as usize,
            self.dimensions.y as usize,
            self.dimensions.z as usize,
        ]
    }
}

impl<T: Clone> Clone for Grid3D<T> {
    fn clone(&self) -> Self {
        Self {
            cells: self.cells.clone(),
            dimensions: self.dimensions,
        }
    }
}

impl<T: Debug> Debug for Grid3D<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let y_layers: Vec<&[T]> = self
            .cells
            .chunks_exact(self.dimensions.x as usize)
            .collect();
        let z_layers: Vec<&[&[T]]> = y_layers.chunks_exact(self.dimensions.y as usize).collect();

        f.write_str("Grid3D")?;
        f.debug_list().entries(z_layers.iter()).finish()
    }
}

impl<T: Default> Grid3D<T> {
    pub fn default(dimensions: IVec3) -> Self {
        let capacity: usize = (dimensions.x * dimensions.y * dimensions.z) as usize;
        let mut cells: Vec<T> = Vec::with_capacity(capacity);

        cells.resize_with(capacity, T::default);

        Self { cells, dimensions }
    }
}

impl<T> Default for Grid3D<T> {
    fn default() -> Self {
        Self::empty(IVec3::ZERO)
    }
}

impl<T: PartialEq> PartialEq for Grid3D<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.cells == other.cells
    }
}

pub struct CellIter3D {
    curr: IVec3,
    end: IVec3,
    dir: IVec3,
}

impl CellIter3D {
    pub fn until_boundary<T>(grid: &Grid3D<T>, curr: IVec3, mut dir: IVec3) -> Self {
        sanitize_dir_3d(&mut dir);

        let end: IVec3 =
            (curr + dir * *grid.dimensions()).clamp(IVec3::ZERO, grid.max_dimensions()) + dir;

        Self { curr, end, dir }
    }

    pub fn until_boundary_from_dimensions(dimensions: &IVec3, curr: IVec3, mut dir: IVec3) -> Self {
        sanitize_dir_3d(&mut dir);

        let end: IVec3 =
            (curr + dir * *dimensions).clamp(IVec3::ZERO, *dimensions - IVec3::ONE) + dir;

        Self { curr, end, dir }
    }
}

impl Iterator for CellIter3D {
    type Item = IVec3;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr != self.end {
            let prev: IVec3 = self.curr;

            self.curr += self.dir;

            Some(prev)
        } else {
            None
        }
    }
}

impl TryFrom<Range<IVec3>> for CellIter3D {
    type Error = CellIterFromRangeError;

    fn try_from(Range { start, end }: Range<IVec3>) -> Result<Self, Self::Error> {
        use CellIterFromRangeError::*;

        let delta: IVec3 = end - start;

        if delta == IVec3::ZERO {
            Err(PositionsIdentical)
        } else if delta.cmpeq(IVec3::ZERO).bitmask().count_ones() != 2_u32 {
            Err(PositionsNotAligned)
        } else {
            let dir: IVec3 = delta / manhattan_distance_3d(&start, &end);

            Ok(Self {
                curr: start,
                end,
                dir,
            })
        }
    }
}

impl TryFrom<RangeInclusive<IVec3>> for CellIter3D {
    type Error = CellIterFromRangeError;

    fn try_from(range_inclusive: RangeInclusive<IVec3>) -> Result<Self, Self::Error> {
        use CellIterFromRangeError::*;

        let curr: IVec3 = *range_inclusive.start();
        let end: IVec3 = *range_inclusive.end();
        let delta: IVec3 = end - curr;

        if delta == IVec3::ZERO {
            Err(PositionsIdentical)
        } else if delta.cmpeq(IVec3::ZERO).bitmask().count_ones() != 2_u32 {
            Err(PositionsNotAligned)
        } else {
            let dir: IVec3 = delta / manhattan_distance_3d(&curr, &end);

            Ok(Self {
                curr,
                end: end + dir,
                dir,
            })
        }
    }
}
