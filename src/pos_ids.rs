use any_tensor::Tensor;
use digit_layout::{DigitLayout, types};
use ndarray_layout::ArrayLayout;

/// 用于计算泛型 `pos_ids` 的trait
pub trait PosTy {
    /// 返回类型布局
    fn dt() -> DigitLayout;
    /// 从 usize 转换为具体类型
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

/// 计算 n 维的 `pos_ids`
/// ### 兼容数据类型：`u32`, `u64`
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

/// 计算 qwen2vl-vit 的 2 维 `pos_ids`
///
/// ### pos_ids按如下方式排列：
/// - 每组包含 2 * 2 的小 patch，按行优先遍历
/// - 如 `grid(h, w) = [4, 4]`, 则生成的 `pos_ids` 为：
///   - [(0, 0), (0, 1), (1, 0), (1, 1)],
///   - [(0, 2), (0, 3), (1, 2), (1, 3)],
///   - [(2, 0), (2, 1), (3, 0), (3, 1)],
///   - [(2, 2), (2, 3), (3, 2), (3, 3)]。
/// - 见测试函数 test_pos_2d_qwen2vl_vit_u32()
///
/// ### 兼容数据类型：`u32`, `u64`
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
fn test_pos_2d_qwen2vl_vit_u64() {
    let tensor = pos_2d_qwen2vl_vit::<u64>([336, 476], 14);
    let (pos, pos_dt, pos_layout) = (tensor.get(), tensor.dt(), tensor.layout());
    println!("pos_dt: {pos_dt}");
    println!("pos_shape: {:?}", pos_layout.shape());
    println!("pos: {pos:?}");
}

#[test]
fn test_pos_2d_qwen2vl_vit_u32() {
    let tensor = pos_2d_qwen2vl_vit::<u32>([4 * 14, 4 * 14], 14);
    let (pos, pos_dt, pos_layout) = (tensor.get(), tensor.dt(), tensor.layout());
    println!("pos_dt: {pos_dt}");
    println!("pos_shape: {:?}", pos_layout.shape());
    println!("pos: {pos:?}");
}
