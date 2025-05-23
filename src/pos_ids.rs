pub fn build_pos_ids_nd(shape: Vec<usize>) -> Vec<u32> {
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

pub fn pos_qwen2vl([h, w]: [usize; 2], d_patch: usize) -> Vec<u32> {
    let h = h / d_patch;
    let w = w / d_patch;
    let mut ans = vec![0; h * w * 2];

    let mut ptr = 0;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    ans[ptr * 2] = (y + dy) as u32;
                    ans[ptr * 2 + 1] = (x + dx) as u32;
                    ptr += 1;
                }
            }
        }
    }
    ans
}

#[test]
fn test_pos_qwen2vl() {
    let pos_ids = pos_qwen2vl([336, 476], 14);
    println!("pos_ids_len: {}", pos_ids.len());
    println!("{:?}", pos_ids);
}

#[test]
fn test_qwen2vl_2d_mrope_f32() {
    use crate::rope_m;
    let shape = [16, 24, 34, 80];
    let nh = shape[0];
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / (nh * dh);

    let pos_ids = pos_qwen2vl([336, 476], 14);

    // -------LM-in----------
    let size = std::mem::size_of::<f32>();
    let mut data: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列
    let len = data.len() * size;
    let x1 = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len) };

    let dt = digit_layout::types::F32;
    let offset = 0;
    let shape_o = [nh, mid, dh];
    let strides = [
        (mid * dh * size) as isize,
        (dh * size) as isize,
        (1 * size) as isize,
    ];
    let pos_ids = Some(pos_ids);
    let rope_section = None;

    // -------m--------
    rope_m(
        &x1,
        dt,
        offset,
        &shape_o,
        &strides,
        &shape,
        pos_ids,
        rope_section,
    );

    let out = unsafe { std::slice::from_raw_parts_mut(x1.as_mut_ptr() as *mut f32, nh * mid * dh) };
    let start = 1145;
    let end = start + 20;
    println!("x {}:{}  {:?}", start, end, &out[start..end]);
    // output(f32):
    // x 1145:1165  [275.9079, 626.51355, 833.59015, 956.6164, 1030.9128, 1076.5735, 1105.0979, 1123.2014, 1134.8898, 1142.5938, 1147.8093, 1151.464,
    // 1154.1375, 1156.1931, 1157.8593, 1569.198, 1598.7628, 1506.0979, 1405.6301, 1326.8085]

    // crate::permute::test_permute_nm(&shape, None);
}
