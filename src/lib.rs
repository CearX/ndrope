pub mod permute;
pub mod pos_ids;

use digit_layout::{DigitLayout, types};
use half::f16;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use pos_ids::build_pos_ids_nd;

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

trait Data: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy {}
impl<T> Data for T where T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy {}

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
        let sin = sin.cast::<T>();
        let cos = cos.cast::<T>();
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

            let [a, b] = [*x1, *x2];
            [*x1, *x2] = [a * cos - b * sin, a * sin + b * cos];
        }
    }
}

fn build_sin_cos_table(
    row_max: usize,
    col_max: usize,
    theta: f32,
    f: impl Fn(f32, f32) -> f32,
) -> [Vec<f32>; 2] {
    let size = row_max * col_max;
    let mut sin = vec![0.; size];
    let mut cos = vec![0.; size];
    for i in 0..size {
        let pos = (i / col_max) as f32;
        let idx = (i % col_max) as f32;
        let theta = theta.powf(-(idx / col_max as f32));

        let (sin_, cos_) = f(theta, pos).sin_cos();

        sin[i] = sin_;
        cos[i] = cos_;
    }
    [sin, cos]
}

pub fn rope(
    x: &&mut [u8],
    dt: digit_layout::DigitLayout,
    offset: usize,
    shape_o: &[usize],
    strides: &[isize],
    shape: &[usize],
    pos_ids: Option<Vec<u32>>,
    rope_section: Option<Vec<usize>>,
    is_nd: bool,
) {
    let nh = shape[0];
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / (nh * dh);
    let mid_dims = &shape[1..shape.len() - 1];

    assert_eq!(shape_o.len(), 3);
    assert_eq!(strides.len(), 3);

    // 如果 rope_section 为 None，则每个维度均分dh/2
    let rope_section = rope_section.unwrap_or_else(|| {
        assert_eq!((dh / 2) % mid_dims.len(), 0);
        vec![(dh / 2) / mid_dims.len(); mid_dims.len()]
    });
    assert_eq!(rope_section.len(), mid_dims.len());
    assert_eq!(dh / 2, rope_section.iter().sum());

    // 位置编码, 例如3维时，pos_ids: [h*w*t, 3]
    let pos = pos_ids.unwrap_or_else(|| build_pos_ids_nd(mid_dims.to_vec()));

    // sin/cos: [row_max, col_max]
    let row_max = mid_dims.iter().max().unwrap();
    let col_max = rope_section.iter().max().unwrap();
    let theta = 10000.0;
    let [sin, cos] = build_sin_cos_table(*row_max, *col_max, theta, |theta, pos| theta * pos);

    let rope_section = rope_section.iter().map(|&x| x as u32).collect::<Vec<_>>();

    let size = match dt {
        types::F16 => size_of::<f16>(),
        types::F32 => size_of::<f32>(),
        types::F64 => size_of::<f64>(),
        _ => todo!(),
    };

    // 创建 Scheme 实例
    let scheme = Scheme {
        nh,
        dh,
        mid,
        n: rope_section.len(),
        rope_section: rope_section.as_ptr() as *const u8,
        // s_x_0: (mid * dh) as isize * size as isize,
        // s_x_1: dh as isize * size as isize,
        s_x_0: strides[0],
        s_x_1: strides[1],
        s_pos_0: (mid_dims.len()) as isize * size_of::<u32>() as isize,
        s_pos_1: size_of::<u32>() as isize,
        s_sin_0: (col_max * size) as isize,
        s_sin_1: size as isize,
        s_cos_0: (col_max * size) as isize,
        s_cos_1: size as isize,
        x: unsafe { (*x).as_ptr().byte_offset(offset as isize) } as *mut u8,
        pos: pos.as_ptr() as *const u8,
        sin: sin.as_ptr() as *const u8,
        cos: cos.as_ptr() as *const u8,
    };

    // 根据 is_nd和dt 调用不同的计算方法
    if is_nd {
        match (dt, types::U32) {
            (types::F16, types::U32) => scheme.calculate_nd::<f16, u32>(),
            (types::F32, types::U32) => scheme.calculate_nd::<f32, u32>(),
            _ => todo!(),
        };
    } else {
        match (dt, types::U32) {
            (types::F16, types::U32) => scheme.calculate_m::<f16, u32>(),
            (types::F32, types::U32) => scheme.calculate_m::<f32, u32>(),
            _ => todo!(),
        };
    };
}

pub fn rope_nd(
    x: &&mut [u8],
    dt: DigitLayout,
    offset: usize,
    shape_o: &[usize],
    strides: &[isize],
    shape: &[usize],
    pos_ids: Option<Vec<u32>>,
    rope_section: Option<Vec<usize>>,
) {
    rope(
        x,
        dt,
        offset,
        shape_o,
        strides,
        shape,
        pos_ids,
        rope_section,
        true,
    );
}

pub fn rope_m(
    x: &&mut [u8],
    dt: DigitLayout,
    offset: usize,
    shape_o: &[usize],
    strides: &[isize],
    shape: &[usize],
    pos_ids: Option<Vec<u32>>,
    rope_section: Option<Vec<usize>>,
) {
    rope(
        x,
        dt,
        offset,
        shape_o,
        strides,
        shape,
        pos_ids,
        rope_section,
        false,
    );
}

// 1.no return
// 2.grid
// 3.层次： cal -> rope(scheme) -> rope_nd/m

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
