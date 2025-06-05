pub mod permute;
pub mod pos_ids;
pub mod sin_cos;

use digit_layout::{DigitLayout, types};
use half::f16;
use ndarray_layout::ArrayLayout;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};
use tensor::Tensor;

use pos_ids::PosTy;
use sin_cos::Float;

pub fn tensor<'a>(
    x: &'a mut [u8],
    dt: DigitLayout,
    shape: [usize; 3],
    strides: [isize; 3],
    offset: isize,
) -> Tensor<&'a mut [u8], 3> {
    Tensor::from_raw_parts(dt, ArrayLayout::<3>::new(&shape, &strides, offset), x)
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

/// 计算时f16转f32; f32, f64不变;
/// 张量类型为f16时，sin_cos为f32，提高精度;
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

        let x = x.cast::<[f32; 2]>();
        let pos = pos.cast::<u32>();
        let sin = sin.cast::<f32>();
        let cos = cos.cast::<f32>();
        let rope_section = rope_section.cast::<u32>();

        let dh = dh / 2;
        let s_x_2 = size_of::<[f32; 2]>() as isize;
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

            let pos =
                unsafe { pos.byte_offset(i1 * s_pos_0 + i3 as isize * s_pos_1).read() } as isize;
            let sin = unsafe { sin.byte_offset(pos * s_sin_0 + i4 * s_sin_1).read() };
            let cos = unsafe { cos.byte_offset(pos * s_cos_0 + i4 * s_cos_1).read() };

            let [a, b] = *x;
            *x = [a * cos - b * sin, a * sin + b * cos];
        }
    }

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

    // 根据 is_nd和dt 调用不同的计算方法
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

// #[test]
// fn test_n() {
//     let shape = [1, 2, 4]; // [nh, seq, dh]
//     let nh = shape[0];
//     let dh = shape[shape.len() - 1];
//     let mid: usize = shape.iter().product::<usize>() / (nh * dh);

//     // -------nd--------
//     let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
//     let x = rope_nd(x, &shape, None, None);

//     let x = x.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in &x {
//         println!("{:?}", chunk);
//     }
// }

// #[test]
// fn test_m() {
//     let shape = [1, 2, 4]; // [nh, seq, dh]
//     let nh = shape[0];
//     let dh = shape[shape.len() - 1];
//     let mid: usize = shape.iter().product::<usize>() / (nh * dh);

//     // -------m--------
//     let x1: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列
//     let x1 = rope_m(x1, &shape, None, None);

//     let x1 = x1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in &x1 {
//         println!("{:?}", chunk);
//     }
// }

// #[test]
// fn test_nm() {
//     let shape = [1, 2, 4]; // [nh, seq, dh]
//     let nh = shape[0];
//     let dh = shape[shape.len() - 1];
//     let mid: usize = shape.iter().product::<usize>() / (nh * dh);

//     // -------nd--------
//     let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
//     let x = rope_nd(x, &shape, None, None);

//     let x = x.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in &x {
//         println!("{:?}", chunk);
//     }

//     // -------m--------
//     let x1: Vec<f32> = vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0];
//     let x1 = rope_m(x1, &shape, None, None);

//     let x1 = x1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in &x1 {
//         println!("{:?}", chunk);
//     }
// }
