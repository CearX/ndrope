use digit_layout::{DigitLayout, types};
use ndarray_layout::ArrayLayout;
use tensor::Tensor;

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

pub fn pos_nd<U: PosTy + Clone>(grid: Vec<usize>) -> Tensor<Box<[U]>, 2> {
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

    let dt = U::dt();
    let layout = ArrayLayout::<2>::new(&[total_size, dim], &[dim as isize, 1], 0);
    let data = pos.into();
    Tensor::from_raw_parts(dt, layout, data)
}

#[test]
fn test_pos_nd() {
    let grid = vec![2, 2, 3, 4];
    let len = grid.len();
    let pos = pos_nd::<u64>(grid);
    let (pos, pos_dt, pos_layout) = (pos.get(), pos.dt(), pos.layout());
    println!("pos_dt: {pos_dt}");
    println!("pos_shape: {:?}", pos_layout.shape());
    println!("pos_strides: {:?}", pos_layout.strides());
    println!("pos_offset: {:?}", pos_layout.offset());
    let pos = pos.chunks(len).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &pos {
        println!("pos_ids: {chunk:?}");
    }
}

pub fn pos_2d_qwen2vl_vit<U: PosTy + Clone>(
    [h, w]: [usize; 2],
    d_patch: usize,
) -> Tensor<Box<[U]>, 2> {
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

    let dt = U::dt();
    let layout = ArrayLayout::<2>::new(&[h * w, 2], &[2, 1], 0);
    let data = pos.into();
    Tensor::from_raw_parts(dt, layout, data)
}

// pub fn pos_3d_qwen2vl_llm<U: PosTy + Clone>() {
//     todo!()
// }

#[test]
fn test_pos_2d_qwen2vl_vit() {
    let tensor = pos_2d_qwen2vl_vit::<u64>([336, 476], 14);
    let (pos, pos_dt, pos_layout) = (tensor.get(), tensor.dt(), tensor.layout());
    println!("pos_dt: {pos_dt}");
    println!("pos_shape: {:?}", pos_layout.shape());
    println!("pos: {pos:?}");
}
