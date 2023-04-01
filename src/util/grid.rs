pub use {grid_2d::*, grid_3d::*};

mod grid_2d;
mod grid_3d;

#[derive(Debug)]
pub enum CellIterFromRangeError {
    PositionsIdentical,
    PositionsNotAligned,
}
