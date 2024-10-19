pub use Pixel::{Dark as D, Light as L};

use {super::*, crate::define_cell};

define_cell! {
    #[repr(u8)]
    #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
    pub enum Pixel {
        #[default]
        Dark = DARK = b'.',
        Light = LIGHT = b'#',
    }
}

impl Pixel {
    pub fn is_light(self) -> bool {
        matches!(self, Self::Light)
    }
}

impl From<bool> for Pixel {
    fn from(value: bool) -> Self {
        if value {
            Self::Light
        } else {
            Self::Dark
        }
    }
}

impl From<Pixel> for bool {
    fn from(value: Pixel) -> Self {
        value.is_light()
    }
}
