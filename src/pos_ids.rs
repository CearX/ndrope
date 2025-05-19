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
fn test_qwen2vl_2d_mrope() {
    use crate::rope_m;
    let shape = [24, 34, 80];
    let nh = 16;
    let dh = shape[shape.len() - 1];
    let mid: usize = shape.iter().product::<usize>() / dh;

    let pos_ids = pos_qwen2vl([336, 476], 14);

    // -------m--------
    let x1: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列
    let x1 = rope_m(x1, Some(pos_ids), None, nh, &shape);

    let start = 1145;
    let end = start + 20;
    println!("x {}:{}  {:?}", start, end, &x1[start..end]);
}
