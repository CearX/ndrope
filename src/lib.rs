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

impl Scheme {
    fn calculate_nd(&self) {
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

    fn calculate_m(&self) {
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

        let x = x.cast::<f32>();
        let pos = pos.cast::<u32>();
        let sin = sin.cast::<f32>();
        let cos = cos.cast::<f32>();
        let rope_section = rope_section.cast::<u32>();

        let dh = dh / 2;
        let s_x_2 = size_of::<f32>() as isize;
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

            let pos =
                unsafe { pos.byte_offset(i1 * s_pos_0 + i3 as isize * s_pos_1).read() } as isize;
            let sin = unsafe { sin.byte_offset(pos * s_sin_0 + i4 * s_sin_1).read() };
            let cos = unsafe { cos.byte_offset(pos * s_cos_0 + i4 * s_cos_1).read() };

            let [a, b] = [*x1, *x2];
            [*x1, *x2] = [a * cos - b * sin, a * sin + b * cos];
        }
    }
}

fn generate_nd(
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

fn build_pos_ids_nd(shape: Vec<usize>) -> Vec<u32> {
    assert!(!shape.is_empty(), "shape must not be empty");
    let dim = shape.len();
    let total_size: usize = shape.iter().product();
    let mut pos = vec![0; total_size * dim];

    let mut strides = vec![1; dim];
    for i in (0..dim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for idx in 0..total_size {
        let mut remainder = idx;
        for d in 0..dim {
            pos[idx * dim + d] = (remainder / strides[d]) as u32;
            remainder %= strides[d];
        }
    }

    pos
}

#[test]
fn test_pos_ids_nd() {
    let mid_dims = vec![2, 2, 3, 4];
    let len = mid_dims.len();
    let pos = build_pos_ids_nd(mid_dims);
    let pos = pos.chunks(len).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &pos {
        println!("pos_ids: {:?}", chunk);
    }
}

pub fn rope(
    mut x: Vec<f32>,
    rope_section: Option<Vec<usize>>,
    nh: usize,
    shape: &[usize],
    is_nd: bool,
) -> Vec<f32> {
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;
    let mid_dims = &shape[..shape.len() - 1];

    // 如果 rope_section 为 None，则每个维度均分dh/2
    let rope_section = rope_section.unwrap_or_else(|| {
        assert_eq!((dh / 2) % mid_dims.len(), 0);
        vec![(dh / 2) / mid_dims.len(); mid_dims.len()]
    });
    assert_eq!(rope_section.len(), mid_dims.len());
    assert_eq!(dh / 2, rope_section.iter().sum());

    // 生成位置编码和 sin/cos; 例如3维时，pos_ids: [h*w*t, 3], sin/cos: [row_max, col_max]
    let pos = build_pos_ids_nd(mid_dims.to_vec());
    let row_max = mid_dims.iter().max().unwrap();
    let col_max = rope_section.iter().max().unwrap();
    let theta = 10000.0;
    let [sin, cos] = generate_nd(*row_max, *col_max, theta, |theta, pos| theta * pos);

    let rope_section = rope_section.iter().map(|&x| x as u32).collect::<Vec<_>>();

    // 创建 Scheme 实例
    let scheme = Scheme {
        nh,
        dh,
        mid,
        n: rope_section.len(),
        rope_section: rope_section.as_ptr() as *const u8,
        s_x_0: (mid * dh) as isize * size_of::<f32>() as isize,
        s_x_1: dh as isize * size_of::<f32>() as isize,
        s_pos_0: (shape.len() - 1) as isize * size_of::<u32>() as isize,
        s_pos_1: size_of::<u32>() as isize,
        s_sin_0: (col_max * size_of::<f32>()) as isize,
        s_sin_1: size_of::<f32>() as isize,
        s_cos_0: (col_max * size_of::<f32>()) as isize,
        s_cos_1: size_of::<f32>() as isize,
        x: x.as_mut_ptr() as *mut u8,
        pos: pos.as_ptr() as *const u8,
        sin: sin.as_ptr() as *const u8,
        cos: cos.as_ptr() as *const u8,
    };

    // 根据 is_nd 调用不同的计算方法
    if is_nd {
        scheme.calculate_nd();
    } else {
        scheme.calculate_m();
    }

    x
}

pub fn rope_nd(
    x: Vec<f32>,
    rope_section: Option<Vec<usize>>,
    nh: usize,
    shape: &[usize],
) -> Vec<f32> {
    rope(x, rope_section, nh, shape, true)
}

pub fn rope_m(
    x: Vec<f32>,
    rope_section: Option<Vec<usize>>,
    nh: usize,
    shape: &[usize],
) -> Vec<f32> {
    rope(x, rope_section, nh, shape, false)
}

#[test]
fn test_n() {
    let shape = [2, 4];
    let nh = 1;
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;

    // -------nd--------
    let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
    let x = rope_nd(x, None, nh, &shape);

    let x = x.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &x {
        println!("{:?}", chunk);
    }
}

#[test]
fn test_m() {
    let shape = [2, 4];
    let nh = 1;
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;

    // -------m--------
    let x1: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列
    let x1 = rope_m(x1, None, nh, &shape);

    let x1 = x1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &x1 {
        println!("{:?}", chunk);
    }
}

#[test]
fn test_nm() {
    let shape = [2, 4];
    let nh = 1;
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;

    // -------nd--------
    let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
    let x = rope_nd(x, None, nh, &shape);

    let x = x.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &x {
        println!("{:?}", chunk);
    }

    // -------m--------
    let x1: Vec<f32> = vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0];
    let x1 = rope_m(x1, None, nh, &shape);

    let x1 = x1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in &x1 {
        println!("{:?}", chunk);
    }
}

pub fn test_permute_nm(nh: usize, shape: Vec<usize>, rope_section: Option<Vec<usize>>) {
    use ndarray::{Array2, Array3};

    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;
    let dim = nh * dh;

    // -------nd--------
    let x = vec![1.0f32; nh * mid * dh]; // x设为全1
    // let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
    let wq = (0..(nh * dh * nh * dh))
        .map(|i| i as f32)
        .collect::<Vec<_>>();

    // q = x @ wq.T;
    let x = Array2::from_shape_vec((mid, dim), x).unwrap();
    let wq = Array2::from_shape_vec((dim, dim), wq).unwrap();
    let q = x.dot(&wq.t()).into_raw_vec_and_offset().0;

    let r_q = rope_nd(q, rope_section.clone(), nh, &shape);

    println!("r_q:");
    let data = &r_q.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in data {
        println!("{:?}", chunk);
    }

    // -------m--------
    let x1 = vec![1.0f32; nh * mid * dh]; // x1设为全1
    // let x1: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列

    // q1 = x @ wq1.T;
    let x1 = Array2::from_shape_vec((mid, dim), x1).unwrap();
    let permute = wq
        .to_shape((nh, dim / nh / 2, 2, dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3]);
    let wq1 = permute.to_shape((dim, dim)).unwrap();
    let q1 = x1.dot(&wq1.t()).into_raw_vec_and_offset().0;

    let r_q1 = rope_m(q1, rope_section.clone(), nh, &shape);

    println!("r_q1:");
    let data = &r_q1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
    for chunk in data {
        println!("{:?}", chunk);
    }

    // ---permute_back---
    let r_q1 = Array3::from_shape_vec((nh, mid, dh), r_q1)
        .unwrap()
        .to_shape((nh, mid, 2, dh / 2))
        .unwrap()
        .permuted_axes([0, 1, 3, 2])
        .to_shape((mid, dim))
        .unwrap()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;

    assert_eq!(r_q, r_q1);
}

#[test]
fn test_permute() {
    let nh = 1;
    // let nh = 16;
    let shape = vec![2, 4]; // [h, dh]
    // let shape = vec![2, 4, 8]; // [h, w, dh]
    // let shape = vec![2, 4, 8, 12]; // [h, w, t, dh]
    // let shape = vec![2, 4, 8, 12, 16]; // [h, w, t, e, dh]
    let rope_section = None;
    test_permute_nm(nh, shape, rope_section);
}

#[test]
fn test_section() {
    let nh = 2;
    let shape = vec![8, 2, 4, 16]; // [t, h, w, dh], 可以改变维度顺序，会体现在pos_ids上
    let rope_section = Some(vec![2, 2, 4]); // 可以手动设置各个维度的大小, 不设置则默认均分
    test_permute_nm(nh, shape, rope_section);
}
