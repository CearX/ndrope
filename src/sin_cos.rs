use digit_layout::{DigitLayout, types};
use half::f16;
use ndarray_layout::ArrayLayout;
use tensor::Tensor;

pub trait Float: Copy {
    fn dt() -> DigitLayout;
    fn from_usize(n: usize) -> Self;
    fn from_f32(n: f32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
}

impl Float for f16 {
    fn dt() -> DigitLayout {
        types::F16
    }
    fn from_usize(n: usize) -> Self {
        f16::from_f32(n as f32)
    }
    fn from_f32(n: f32) -> Self {
        f16::from_f32(n)
    }

    fn powf(self, n: Self) -> Self {
        f16::from_f32(f32::from(self).powf(f32::from(n)))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = f32::from(self).sin_cos();
        (f16::from_f32(sin), f16::from_f32(cos))
    }
}

impl Float for f32 {
    fn dt() -> DigitLayout {
        types::F32
    }
    fn from_usize(n: usize) -> Self {
        n as f32
    }

    fn from_f32(n: f32) -> Self {
        n
    }

    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }

    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }
}

impl Float for f64 {
    fn dt() -> DigitLayout {
        types::F64
    }
    fn from_usize(n: usize) -> Self {
        n as f64
    }

    fn from_f32(n: f32) -> Self {
        n as f64
    }

    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }

    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }
}

fn build_sin_cos_table<T>(
    row_max: usize,
    col_max: usize,
    theta: T,
    f: impl Fn(T, T) -> T,
) -> [Box<[T]>; 2]
where
    T: Float + std::ops::Neg<Output = T> + std::ops::Div<Output = T>,
{
    let size = row_max * col_max;
    let mut sin = vec![T::from_f32(0.); size];
    let mut cos = vec![T::from_f32(0.); size];
    for i in 0..row_max * col_max {
        let pos = T::from_usize(i / col_max);
        let idx = T::from_usize(i % col_max);
        let theta = theta.powf(-(idx / T::from_usize(col_max)));

        let (sin_, cos_) = f(theta, pos).sin_cos();

        sin[i] = sin_;
        cos[i] = cos_;
    }
    [sin.into(), cos.into()]
}

pub fn sin_cos_nd<T>(
    shape: &[usize],
    grid: &[usize],
    rope_section: Option<Vec<usize>>,
    theta: T,
) -> [Tensor<Box<[T]>, 2>; 2]
where
    T: Float + std::ops::Neg<Output = T> + std::ops::Div<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(shape.len(), 3);
    let _nh = shape[0];
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

    let row_max = *(grid.iter().max().unwrap());
    let col_max = *(rope_section.iter().max().unwrap());

    let [sin, cos] = build_sin_cos_table(row_max, col_max, theta, |theta, pos| theta * pos);
    let layout = ArrayLayout::<2>::new(&[row_max, col_max], &[col_max as isize, 1], 0);

    let sin = Tensor::from_raw_parts(T::dt(), layout.clone(), sin);
    let cos = Tensor::from_raw_parts(T::dt(), layout.clone(), cos);
    [sin, cos]
}

#[test]
fn test_sin_cos_nd_f16() {
    let shape = [2, 4, 8];
    let grid = [2, 2];
    let rope_section = Some(vec![2, 2]);
    let theta = f16::from_f32(10000.0);
    let [sin, cos] = sin_cos_nd::<f16>(&shape, &grid, rope_section, theta);
    assert_eq!(sin.shape(), &[2, 2]);
    assert_eq!(cos.shape(), &[2, 2]);
}

#[test]
fn test_sin_cos_nd_f32() {
    let shape = [2, 4, 8];
    let grid = [2, 2];
    let rope_section = Some(vec![2, 2]);
    let theta = 10000.0;
    let [sin, cos] = sin_cos_nd::<f32>(&shape, &grid, rope_section, theta);
    assert_eq!(sin.shape(), &[2, 2]);
    assert_eq!(cos.shape(), &[2, 2]);
}

#[test]
fn test_sin_cos_nd_f64() {
    let shape = [2, 4, 8];
    let grid = [2, 2];
    let rope_section = Some(vec![2, 2]);
    let theta = 10000.0;
    let [sin, cos] = sin_cos_nd::<f64>(&shape, &grid, rope_section, theta);
    assert_eq!(sin.shape(), &[2, 2]);
    assert_eq!(cos.shape(), &[2, 2]);
}

// /// normal sin_cos
// /// 3dmrope_llm
// /// todo
// pub fn sin_cos<T>(
//     nctx: usize,
//     dh: usize,
//     theta: T,
// ) -> [Tensor<Box<[T]>, 2>; 2]
// where
//     T: Float + std::ops::Neg<Output = T> + std::ops::Div<Output = T> + std::ops::Mul<Output = T>,
// {
//     let rol_max = nctx;
//     let col_max = dh / 2;
//     let [sin, cos] = build_sin_cos_table(row_max, col_max, theta, |theta, pos| theta * pos);
//     let layout = ArrayLayout::<2>::new(&[row_max, col_max], &[col_max as isize, 1], 0);

//     let sin = Tensor::from_raw_parts(T::dt(), layout.clone(), sin);
//     let cos = Tensor::from_raw_parts(T::dt(), layout.clone(), cos);
//     [sin, cos]
// }
