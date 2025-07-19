use {super::*, std::mem::transmute};

pub fn sortable_index_from_pos_4d(pos: I8Vec4) -> u32 {
    const MSB_MASK: u32 = 1_u32 << (u8::BITS - 1_u32);
    const TOGGLE_MASK: u32 = MSB_MASK << (u8::BITS * 0_u32)
        | MSB_MASK << (u8::BITS * 1_u32)
        | MSB_MASK << (u8::BITS * 2_u32)
        | MSB_MASK << (u8::BITS * 3_u32);

    // SAFETY: Trivial.
    u32::from_le_bytes(unsafe { transmute::<[i8; 4_usize], [u8; 4_usize]>(pos.to_array()) })
        ^ TOGGLE_MASK
}
