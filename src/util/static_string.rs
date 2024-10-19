use {
    super::*,
    nom::bytes::complete::take_while_m_n,
    std::{
        fmt::{Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
    },
};

type StaticStringLen = u8;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StaticString<const N: usize> {
    bytes: [u8; N],
    len: StaticStringLen,
}

impl<const N: usize> StaticString<N> {
    pub fn as_str(&self) -> &str {
        // SAFETY: This always having valid UTF8 bytes is an invariant of the type
        unsafe { from_utf8_unchecked(&self.bytes[..self.len as usize]) }
    }

    pub fn parse_char1<'i, F: Fn(char) -> bool>(
        min: usize,
        f: F,
    ) -> impl FnMut(&'i str) -> IResult<&'i str, Self> {
        map_res(take_while_m_n(min, N, f), Self::try_from)
    }
}

impl<const N: usize> Debug for StaticString<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(&self.as_str())
    }
}

impl<const N: usize> Default for StaticString<N> {
    fn default() -> Self {
        Self {
            bytes: LargeArrayDefault::large_array_default(),
            len: 0 as StaticStringLen,
        }
    }
}

impl<const N: usize> TryFrom<&str> for StaticString<N> {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        (value.len() <= min(StaticStringLen::MAX as usize, N))
            .then(|| {
                let mut bytes: [u8; N] = LargeArrayDefault::large_array_default();

                bytes[..value.len()].copy_from_slice(value.as_bytes());

                Self {
                    bytes,
                    len: value.len() as StaticStringLen,
                }
            })
            .ok_or(())
    }
}
