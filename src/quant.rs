use half::f16;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const TYPE_SIZE: usize = 2 + 32;
}
