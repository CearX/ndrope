use digit_layout::{DigitLayout, types};
use ndarray_layout::ArrayLayout;

pub trait PosTy {
    fn dt() -> DigitLayout;
    fn from_usize(p: usize) -> Self;
}

impl PosTy for u32 {
    fn dt() -> DigitLayout {
        types::U32
    }
    fn from_usize(p: usize) -> Self {
        p as _
    }
}

impl PosTy for u64 {
    fn dt() -> DigitLayout {
        types::U64
    }
    fn from_usize(p: usize) -> Self {
        p as _
    }
}

pub fn pos_nd_default<U: PosTy + Clone>(grid: Vec<usize>) -> (Box<[U]>, DigitLayout) {
    assert!(!grid.is_empty(), "grid must not be empty");
    let dim = grid.len();
    let total_size: usize = grid.iter().product();
    let mut pos = vec![U::from_usize(0); total_size * dim];

    let mut strides = vec![1; dim];
    for i in (0..dim - 1).rev() {
        strides[i] = strides[i + 1] * grid[i + 1];
    }

    for idx in 0..total_size {
        let mut remainder = idx;
        for d in 0..dim {
            pos[idx * dim + d] = U::from_usize(remainder / strides[d]);
            remainder %= strides[d];
        }
    }

    (
        pos.into(),
        U::dt(),
        // ArrayLayout::<dim>::new(&[], &[], 0),
    )
}

#[test]
fn test_pos_ids_nd() {
    let grid = vec![2, 2, 3, 4];
    let len = grid.len();
    let (pos, _pos_dt) = pos_nd_default::<u64>(grid);
    let pos = pos.chunks(len).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &pos {
        println!("pos_ids: {:?}", chunk);
    }

    // todo nd layout
}

pub fn pos_2d_qwen2vl_vit<U: PosTy + Clone>(
    [h, w]: [usize; 2],
    d_patch: usize,
) -> (Box<[U]>, DigitLayout, ArrayLayout<2>) {
    let h = h / d_patch;
    let w = w / d_patch;
    let mut pos = vec![U::from_usize(0); h * w * 2];

    let mut ptr = 0;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    pos[ptr * 2] = U::from_usize(y + dy);
                    pos[ptr * 2 + 1] = U::from_usize(x + dx);
                    ptr += 1;
                }
            }
        }
    }
    (
        pos.into(),
        U::dt(),
        ArrayLayout::<2>::new(&[h * w, 2], &[2, 1], 0),
    )
}

pub fn pos_3d_qwen2vl_llm<U: PosTy + Clone>() {
    todo!()
}

#[test]
fn test_pos_qwen2vl() {
    let (pos, pos_dt, pos_layout) = pos_2d_qwen2vl_vit::<u64>([336, 476], 14);
    println!("pos_dt: {:?}", pos_dt);
    println!("pos_shape: {:?}", pos_layout.shape());
    println!("{:?}", pos);
}

#[test]
fn test_qwen2vl_2d_mrope_f16_u64() {
    use crate::rope_m;
    use crate::sin_cos::sin_cos_default;
    use half::f16;

    let nh = 16;
    let mid = 816;
    let dh = 80;
    let size = std::mem::size_of::<f16>();
    let mut data: Vec<f16> = (0..(nh * mid * dh))
        .map(|i| f16::from_f32(i as f32))
        .collect(); // x1设为递增序列
    let x1 =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, size * data.len()) };
    let dt = digit_layout::types::F16;
    let shape = [nh, mid, dh];
    let strides = [
        (mid * dh * size) as isize,
        (dh * size) as isize,
        (1 * size) as isize,
    ];
    let offset = 0;
    let grid = [24, 34];
    let rope_section = None;

    let (pos, pos_dt, pos_layout) = pos_2d_qwen2vl_vit::<u64>([336, 476], 14);

    let theta = f16::from_f32(10000.0);
    let (sin, sin_dt, sin_layout, cos, cos_dt, cos_layout) =
        sin_cos_default::<f16>(&shape, &grid, rope_section.clone(), theta);

    rope_m(
        &x1,
        dt,
        &shape,
        &strides,
        offset,
        &grid,
        rope_section,
        pos,
        pos_dt,
        pos_layout,
        sin,
        sin_dt,
        sin_layout,
        cos,
        cos_dt,
        cos_layout,
    );

    let out = unsafe {
        std::slice::from_raw_parts_mut(x1.as_mut_ptr() as *mut f16, nh * mid * dh * size)
    };
    let start = 1145;
    let end = start + 20;
    println!("x {}:{}  {:?}", start, end, &out[start..end]);
    // output(f32):
    // x 1145:1165  [275.9079, 626.51355, 833.59015, 956.6164, 1030.9128, 1076.5735, 1105.0979, 1123.2014, 1134.8898, 1142.5938, 1147.8093, 1151.464,
    // 1154.1375, 1156.1931, 1157.8593, 1569.198, 1598.7628, 1506.0979, 1405.6301, 1326.8085]

    // crate::permute::test_permute_nm(&shape, None);
}
