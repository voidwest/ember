use half::f16;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; 32],
}
