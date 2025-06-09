#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]
#![allow(rustdoc::broken_intra_doc_links)] // cargo doc 公式识别问题

/// 验证 `rope_nd` 和 `rope_m` 两种实现的转换
mod permute;
/// `pos_ids` 的多种计算方法
pub mod pos_ids;
/// `sin_cos` 的多种计算方法
pub mod sin_cos;

use any_tensor::Tensor;
use digit_layout::{DigitLayout, types};
use half::f16;
use ndarray_layout::ArrayLayout;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use pos_ids::{PosTy, pos_nd};
use sin_cos::{Float, sin_cos_nd};

/// 把原始数据包装为张量
pub fn tensor(
    x: &mut [u8],
    dt: DigitLayout,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
) -> Tensor<&mut [u8], 3> {
    assert_eq!(shape.len(), 3);
    assert_eq!(strides.len(), 3);
    Tensor::from_raw_parts(
        dt,
        ArrayLayout::<3>::new(
            &[shape[0], shape[1], shape[2]],
            &[strides[0], strides[1], strides[2]],
            offset as isize,
        ),
        x,
    )
}

struct Scheme {
    nh: usize,
    dh: usize,
    mid: usize,
    n: usize,
    s_x_0: isize,
    s_x_1: isize,
    s_pos_0: isize,
    s_pos_1: isize,
    s_sin_0: isize,
    s_sin_1: isize,
    s_cos_0: isize,
    s_cos_1: isize,

    x: *mut u8,
    pos: *const u8,
    sin: *const u8,
    cos: *const u8,
    rope_section: *const u8,
}
trait Pos: Copy {
    fn pos(&self) -> usize;
}

impl Pos for u32 {
    fn pos(&self) -> usize {
        *self as _
    }
}
impl Pos for u64 {
    fn pos(&self) -> usize {
        *self as _
    }
}

/// 计算时 f16 转 f32; f32, f64不变。
/// 张量类型为 f16 时，sin_cos 为 f32，提高精度。
trait Data: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy {
    type ComputeType: Data + Display;
    fn to_compute(self) -> Self::ComputeType;
    fn from_compute(val: Self::ComputeType) -> Self;
}

impl Data for f16 {
    type ComputeType = f32;
    fn to_compute(self) -> Self::ComputeType {
        self.to_f32()
    }
    fn from_compute(val: Self::ComputeType) -> Self {
        f16::from_f32(val)
    }
}

impl Data for f32 {
    type ComputeType = f32;
    fn to_compute(self) -> Self::ComputeType {
        self
    }
    fn from_compute(val: Self::ComputeType) -> Self {
        val
    }
}

impl Data for f64 {
    type ComputeType = f64;
    fn to_compute(self) -> Self::ComputeType {
        self
    }
    fn from_compute(val: Self::ComputeType) -> Self {
        val
    }
}

impl Scheme {
    /// ### `rope_m``
    /// - `RoPE` 论文中原版实现，每两个相邻元素一组，在本项目中称为 `rope_nd`。
    /// #### 类型支持
    /// - `x`: `f16`, `f32`, `f64`
    /// - `sin`, `cos`: `f16`, `f32`, `f64`
    /// - `pos`: `u32`, `u64`
    /// - f16 的张量计算时需要传 f32 的 sin_cos 提高精度
    ///
    /// #### `rope_section`
    /// - 可设置来分配各个维度的权重
    fn calculate_nd<T: Data + Display, U: Pos>(&self) {
        let &Self {
            nh,
            dh,
            mid,
            n,
            s_sin_0,
            s_sin_1,
            s_cos_0,
            s_cos_1,
            s_pos_0,
            s_pos_1,
            s_x_0,
            s_x_1,
            x,
            pos,
            sin,
            cos,
            rope_section,
        } = self;

        let x = x.cast::<[T; 2]>();
        let pos = pos.cast::<U>();
        let sin = sin.cast::<T::ComputeType>();
        let cos = cos.cast::<T::ComputeType>();
        let rope_section = rope_section.cast::<u32>();

        let dh = dh / 2;
        let s_x_2 = size_of::<[T; 2]>() as isize;
        for i in 0..nh * mid * dh {
            let i0 = (i / (mid * dh)) as isize;
            let i1 = ((i / dh) % (mid)) as isize;
            let i2 = (i % dh) as isize;
            let x = unsafe { &mut *x.byte_offset(i0 * s_x_0 + i1 * s_x_1 + i2 * s_x_2) };

            // 根据rope_section计算i3和i4
            let mut i3 = 0;
            let mut remaining = i2 as u32;
            while i3 < n && remaining >= unsafe { *rope_section.add(i3) } {
                remaining -= unsafe { *rope_section.add(i3) };
                i3 += 1;
            }
            let i4 = remaining as isize;

            let pos = unsafe {
                pos.byte_offset(i1 * s_pos_0 + i3 as isize * s_pos_1)
                    .read()
                    .pos()
            } as isize;
            let sin = unsafe { sin.byte_offset(pos * s_sin_0 + i4 * s_sin_1).read() };
            let cos = unsafe { cos.byte_offset(pos * s_cos_0 + i4 * s_cos_1).read() };

            let [a, b] = *x;
            let [a, b] = [a.to_compute(), b.to_compute()];
            let [res1, res2] = [a * cos - b * sin, a * sin + b * cos];

            *x = [T::from_compute(res1), T::from_compute(res2)];
        }
    }

    /// ### `rope_m``
    /// - huggingface 中实现，相距 d/2 的每两个元素一组, 在本项目中称为 rope_m。
    /// #### 类型支持
    /// - `x`: `f16`, `f32`, `f64`
    /// - `sin`, `cos`: `f16`, `f32`, `f64`
    /// - `pos`: `u32`, `u64`
    /// - f16 的张量计算时需要传 f32 的 sin_cos 提高精度
    ///
    /// #### `rope_section`
    /// - 可设置来分配各个维度的权重
    fn calculate_m<T: Data + Display, U: Pos>(&self) {
        let &Self {
            nh,
            dh,
            mid,
            n,
            s_sin_0,
            s_sin_1,
            s_cos_0,
            s_cos_1,
            s_pos_0,
            s_pos_1,
            s_x_0,
            s_x_1,
            x,
            pos,
            sin,
            cos,
            rope_section,
        } = self;

        let x = x.cast::<T>();
        let pos = pos.cast::<U>();
        let sin = sin.cast::<T::ComputeType>();
        let cos = cos.cast::<T::ComputeType>();
        let rope_section = rope_section.cast::<u32>();

        let dh = dh / 2;
        let s_x_2 = size_of::<T>() as isize;
        for i in 0..nh * mid * dh {
            let i0 = (i / (mid * dh)) as isize;
            let i1 = ((i / dh) % (mid)) as isize;
            let i2 = (i % dh) as isize;
            let x1 = unsafe { &mut *x.byte_offset(i0 * s_x_0 + i1 * s_x_1 + i2 * s_x_2) };
            let x2 = unsafe {
                &mut *x.byte_offset(i0 * s_x_0 + i1 * s_x_1 + (i2 + dh as isize) * s_x_2)
            };

            // 根据rope_section计算i3和i4
            let mut i3 = 0;
            let mut remaining = i2 as u32;
            while i3 < n && remaining >= unsafe { *rope_section.add(i3) } {
                remaining -= unsafe { *rope_section.add(i3) };
                i3 += 1;
            }
            let i4 = remaining as isize;

            let pos = unsafe {
                pos.byte_offset(i1 * s_pos_0 + i3 as isize * s_pos_1)
                    .read()
                    .pos()
            } as isize;
            let sin = unsafe { sin.byte_offset(pos * s_sin_0 + i4 * s_sin_1).read() };
            let cos = unsafe { cos.byte_offset(pos * s_cos_0 + i4 * s_cos_1).read() };

            let [a, b] = [x1.to_compute(), x2.to_compute()];
            let [res1, res2] = [a * cos - b * sin, a * sin + b * cos];

            *x1 = T::from_compute(res1);
            *x2 = T::from_compute(res2);
        }
    }
}

/// 取出 `tensor` 包装的参数，调用对应的 `calculate` 函数
fn rope<T, U>(
    x: Tensor<&mut [u8], 3>,
    pos: Tensor<Box<[U]>, 2>,
    sin: Tensor<Box<[T]>, 2>,
    cos: Tensor<Box<[T]>, 2>,
    grid: &[usize],
    rope_section: Option<Vec<usize>>,
    is_nd: bool,
) where
    U: PosTy + Clone,
    T: Float,
{
    let (x, dt, shape, strides, offset) = (x.get(), x.dt(), x.shape(), x.strides(), x.offset());
    assert_eq!(shape.len(), 3);
    assert_eq!(strides.len(), 3);
    let nh = shape[0];
    let mid = shape[1];
    let dh = shape[2];
    assert_eq!(grid.iter().product::<usize>(), mid);

    // 如果 rope_section 为 None，则每个维度均分dh/2
    let rope_section = rope_section.unwrap_or_else(|| {
        let dims = grid.len();
        assert_eq!((dh / 2) % dims, 0);
        vec![(dh / 2) / dims; dims]
    });
    assert_eq!(rope_section.len(), grid.len());
    assert_eq!(dh / 2, rope_section.iter().sum());

    let (pos, pos_dt, pos_layout) = (pos.get(), pos.dt(), pos.layout());
    let (sin, sin_dt, sin_layout) = (sin.get(), sin.dt(), sin.layout());
    let (cos, cos_dt, cos_layout) = (cos.get(), cos.dt(), cos.layout());

    if let types::F16 = dt {
        // f16的张量计算时需要传f32的sin_cos提高精度
        assert_eq!(sin_dt, types::F32);
        assert_eq!(cos_dt, types::F32);
    } else {
        assert_eq!(sin_dt, dt);
        assert_eq!(cos_dt, dt);
    }

    let rope_section = rope_section.iter().map(|&x| x as u32).collect::<Vec<_>>();

    // 创建 Scheme 实例
    let scheme = Scheme {
        nh,
        dh,
        mid,
        n: rope_section.len(),
        rope_section: rope_section.as_ptr() as *const u8,
        s_x_0: strides[0],
        s_x_1: strides[1],
        s_pos_0: pos_layout.strides()[0] * size_of::<U>() as isize,
        s_pos_1: pos_layout.strides()[1] * size_of::<U>() as isize,
        s_sin_0: sin_layout.strides()[0] * size_of::<T>() as isize,
        s_sin_1: sin_layout.strides()[1] * size_of::<T>() as isize,
        s_cos_0: cos_layout.strides()[0] * size_of::<T>() as isize,
        s_cos_1: cos_layout.strides()[1] * size_of::<T>() as isize,
        x: unsafe { (*x).as_ptr().byte_offset(offset) } as *mut u8,
        pos: pos.as_ptr() as *const u8,
        sin: sin.as_ptr() as *const u8,
        cos: cos.as_ptr() as *const u8,
    };

    // 根据 is_nd 和 dt 调用不同的计算方法
    if is_nd {
        match (dt, pos_dt) {
            (types::F16, types::U32) => scheme.calculate_nd::<f16, u32>(),
            (types::F32, types::U32) => scheme.calculate_nd::<f32, u32>(),
            (types::F64, types::U32) => scheme.calculate_nd::<f64, u32>(),
            (types::F16, types::U64) => scheme.calculate_nd::<f16, u64>(),
            (types::F32, types::U64) => scheme.calculate_nd::<f32, u64>(),
            (types::F64, types::U64) => scheme.calculate_nd::<f64, u64>(),
            _ => todo!(),
        };
    } else {
        match (dt, pos_dt) {
            (types::F16, types::U32) => scheme.calculate_m::<f16, u32>(),
            (types::F32, types::U32) => scheme.calculate_m::<f32, u32>(),
            (types::F64, types::U32) => scheme.calculate_m::<f64, u32>(),
            (types::F16, types::U64) => scheme.calculate_m::<f16, u64>(),
            (types::F32, types::U64) => scheme.calculate_m::<f32, u64>(),
            (types::F64, types::U64) => scheme.calculate_m::<f64, u64>(),
            _ => todo!(),
        };
    };
}

/// 调用 `rope_nd` 的接口
pub fn rope_nd<T, U>(
    x: Tensor<&mut [u8], 3>,
    pos: Tensor<Box<[U]>, 2>,
    sin: Tensor<Box<[T]>, 2>,
    cos: Tensor<Box<[T]>, 2>,
    grid: &[usize],
    rope_section: Option<Vec<usize>>,
) where
    U: PosTy + Clone,
    T: Float,
{
    rope(x, pos, sin, cos, grid, rope_section, true);
}

/// 调用 `rope_m` 的接口
pub fn rope_m<T, U>(
    x: Tensor<&mut [u8], 3>,
    pos: Tensor<Box<[U]>, 2>,
    sin: Tensor<Box<[T]>, 2>,
    cos: Tensor<Box<[T]>, 2>,
    grid: &[usize],
    rope_section: Option<Vec<usize>>,
) where
    U: PosTy + Clone,
    T: Float,
{
    rope(x, pos, sin, cos, grid, rope_section, false);
}

/// `rope_nd` 和 `rope_m` 的泛型测试函数
/// ### 泛型支持
/// - `data`: `f16`, `f32`, `f64`
/// - `sin`, `cos`: `f16`, `f32`, `f64`
/// - `pos`: `u32`, `u64`
/// - f16 的张量计算时需要传 f32 的 sin_cos 提高精度
pub fn test_rope_nm<T, U, S>(
    data: Option<Vec<T>>,
    shape: [usize; 3],
    grid: Vec<usize>,
    rope_section: Option<Vec<usize>>,
    is_pos_nd: bool,
    is_nd: bool,
) -> Vec<T>
where
    U: PosTy + Clone,
    T: Float + std::ops::Neg<Output = T> + std::ops::Div<Output = T> + std::ops::Mul<Output = T>,
    S: Float + std::ops::Neg<Output = S> + std::ops::Div<Output = S> + std::ops::Mul<Output = S>,
{
    let nh = shape[0];
    let mid = shape[1];
    let dh = shape[2];
    let size = std::mem::size_of::<T>();
    let mut data = data.unwrap_or_else(|| {
        (0..(nh * mid * dh))
            .map(|i| T::from_usize(i))
            .collect::<Vec<T>>() // x1设为递增序列
    });
    let x1 =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, size * data.len()) };
    let dt = T::dt();
    let shape = [nh, mid, dh];
    let strides = [
        (mid * dh * size) as isize,
        (dh * size) as isize,
        size as isize,
    ];
    let offset = 0;
    let x = tensor(x1, dt, shape.to_vec(), strides.to_vec(), offset);

    let pos = if is_pos_nd {
        pos_nd::<U>(grid.clone())
    } else {
        assert_eq!(grid.len(), 2);
        let d_patch = 14;
        let h = grid[0] * d_patch;
        let w = grid[1] * d_patch;
        pos_ids::pos_2d_qwen2vl_vit::<U>([h, w], d_patch)
    };

    let [sin, cos] = sin_cos_nd::<S>(&shape, &grid, rope_section.clone(), S::from_f32(10000.0));

    if is_nd {
        rope_nd(x, pos, sin, cos, &grid, rope_section);
    } else {
        rope_m(x, pos, sin, cos, &grid, rope_section);
    }

    let x1 = unsafe { std::slice::from_raw_parts_mut(x1.as_mut_ptr() as *mut T, nh * mid * dh) };
    x1.to_vec()
}

#[test]
fn test_nd_f16_u32_u64() {
    let shape = [1, 2, 4]; // [nh, seq, dh]
    let grid = [2];
    let is_pos_nd = true;
    let is_nd = true;

    // data 默认递增初始化
    // f16的张量计算时需要传f32的sin_cos提高精度
    let x_f16_u32 =
        test_rope_nm::<f16, u32, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f16_u64 =
        test_rope_nm::<f16, u64, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    let ans = [
        0.0, 1.0, 2.0, 3.0, -2.046875, 6.0664063, 5.9296875, 7.0585938,
    ]
    .iter()
    .map(|&x| f16::from_f32(x))
    .collect::<Vec<_>>();
    assert_eq!(x_f16_u32, ans);
    assert_eq!(x_f16_u64, ans);
}

#[test]
fn test_nd_f32_u32_u64() {
    let shape = [1, 2, 4]; // [nh, seq, dh]
    let grid = [2];
    let is_pos_nd = true;
    let is_nd = true;

    // data 默认递增初始化
    let x_f32_u32 =
        test_rope_nm::<f32, u32, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f32_u64 =
        test_rope_nm::<f32, u64, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    let ans = [
        0.0, 1.0, 2.0, 3.0, -2.0461454, 6.067395, 5.9297013, 7.059649,
    ];
    assert_eq!(x_f32_u32, ans);
    assert_eq!(x_f32_u64, ans);
}

#[test]
fn test_nd_f64_u32_u64() {
    let shape = [1, 2, 4]; // [nh, seq, dh]
    let grid = [2];
    let is_pos_nd = true;
    let is_nd = true;

    // data 默认递增初始化
    let x_f64_u32 =
        test_rope_nm::<f64, u32, f64>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f64_u64 =
        test_rope_nm::<f64, u64, f64>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    let ans = [
        0.0,
        1.0,
        2.0,
        3.0,
        -2.0461457005669237,
        6.067395468572284,
        5.9297011691608255,
        7.059649002921657,
    ];
    assert_eq!(x_f64_u32, ans);
    assert_eq!(x_f64_u64, ans);
}

#[test]
fn test_m_2d_qwen2vl_f16_u32_u64() {
    let shape = [16, 816, 80]; // [nh, seq, dh]
    let grid = [24, 34];
    let is_pos_nd = false; // 使用qw2en2vl的pos_ids
    let is_nd = false;

    // f16的张量计算时需要传f32的sin_cos提高精度
    let x_f16_u32 =
        test_rope_nm::<f16, u32, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f16_u64 =
        test_rope_nm::<f16, u64, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    // f16容量有限，递增初始化会溢出, 只看看部分值
    let start = 1145;
    let end = start + 20;
    let ans = [
        276.0, 626.5, 833.5, 956.5, 1031.0, 1077.0, 1105.0, 1123.0, 1135.0, 1143.0, 1148.0, 1151.0,
        1154.0, 1156.0, 1158.0, 1569.0, 1599.0, 1506.0, 1406.0, 1327.0,
    ]
    .iter()
    .map(|&x| f16::from_f32(x))
    .collect::<Vec<_>>();
    assert_eq!(x_f16_u32[start..end], ans);
    assert_eq!(x_f16_u64[start..end], ans);
}

#[test]
fn test_m_2d_qwen2vl_f32_u32_u64() {
    let shape = [16, 816, 80]; // [nh, seq, dh]
    let grid = [24, 34];
    let is_pos_nd = false; // 使用qw2en2vl的pos_ids
    let is_nd = false;

    let x_f32_u32 =
        test_rope_nm::<f32, u32, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f32_u64 =
        test_rope_nm::<f32, u64, f32>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    let start = 1145;
    let end = start + 20;
    let ans = [
        275.9079, 626.51355, 833.59015, 956.6164, 1030.9128, 1076.5735, 1105.0979, 1123.2014,
        1134.8898, 1142.5938, 1147.8093, 1151.464, 1154.1375, 1156.1931, 1157.8593, 1569.198,
        1598.7628, 1506.0979, 1405.6301, 1326.8085,
    ];
    assert_eq!(x_f32_u32[start..end], ans);
    assert_eq!(x_f32_u64[start..end], ans);
}

#[test]
fn test_m_2d_qwen2vl_f64_u32_u64() {
    let shape = [16, 816, 80]; // [nh, seq, dh]
    let grid = [24, 34];
    let is_pos_nd = false; // 使用qw2en2vl的pos_ids
    let is_nd = false;

    let x_f64_u32 =
        test_rope_nm::<f64, u32, f64>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);
    let x_f64_u64 =
        test_rope_nm::<f64, u64, f64>(None, shape, grid.to_vec(), None, is_pos_nd, is_nd);

    let start = 1145;
    let end = start + 20;
    let ans = [
        275.90794809846454,
        626.5134874971391,
        833.5901795130417,
        956.6163510896864,
        1030.912878552941,
        1076.5734532149459,
        1105.0978574227604,
        1123.201454822187,
        1134.8896511868547,
        1142.593819779052,
        1147.8092530822926,
        1151.4639771070692,
        1154.1374963230123,
        1156.1931415493236,
        1157.8593039794746,
        1569.1981777918863,
        1598.762818931396,
        1506.0979613104353,
        1405.6301400223888,
        1326.8085402953352,
    ];
    assert_eq!(x_f64_u32[start..end], ans);
    assert_eq!(x_f64_u64[start..end], ans);
}
