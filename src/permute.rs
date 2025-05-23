// use crate::{rope_m, rope_nd};
// use ndarray::{Array2, Array3};

// fn test_permute_nm(shape: &[usize], rope_section: Option<Vec<usize>>) {
//     let nh = shape[0];
//     let dh = shape[shape.len() - 1];
//     let mid: usize = shape.iter().product::<usize>() / (nh * dh);
//     let dim = nh * dh;

//     // -------nd--------
//     let x = vec![1.0f32; nh * mid * dh]; // x设为全1
//     // let x: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x设为递增序列
//     let wq = (0..(nh * dh * nh * dh))
//         .map(|i| i as f32)
//         .collect::<Vec<_>>();

//     // q = x @ wq.T;
//     let x = Array2::from_shape_vec((mid, dim), x).unwrap();
//     let wq = Array2::from_shape_vec((dim, dim), wq).unwrap();
//     let q = x.dot(&wq.t()).into_raw_vec_and_offset().0;

//     let r_q = rope_nd(q, shape, None, rope_section.clone());

//     println!("r_q:");
//     let data = &r_q.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in data {
//         println!("{:?}", chunk);
//     }

//     // -------m--------
//     let x1 = vec![1.0f32; nh * mid * dh]; // x1设为全1
//     // let x1: Vec<f32> = (0..(nh * mid * dh)).map(|i| i as f32).collect(); // x1设为递增序列

//     // q1 = x @ wq1.T;
//     let x1 = Array2::from_shape_vec((mid, dim), x1).unwrap();
//     let permute = wq
//         .to_shape((nh, dim / nh / 2, 2, dim))
//         .unwrap()
//         .permuted_axes([0, 2, 1, 3]);
//     let wq1 = permute.to_shape((dim, dim)).unwrap();
//     let q1 = x1.dot(&wq1.t()).into_raw_vec_and_offset().0;

//     let r_q1 = rope_m(q1, shape, None, rope_section.clone());

//     println!("r_q1:");
//     let data = &r_q1.chunks(dh).map(|x| x.to_vec()).collect::<Vec<_>>();
//     for chunk in data {
//         println!("{:?}", chunk);
//     }

//     // ---permute_back---
//     let r_q1 = Array3::from_shape_vec((nh, mid, dh), r_q1)
//         .unwrap()
//         .to_shape((nh, mid, 2, dh / 2))
//         .unwrap()
//         .permuted_axes([0, 1, 3, 2])
//         .to_shape((mid, dim))
//         .unwrap()
//         .to_owned()
//         .into_raw_vec_and_offset()
//         .0;

//     assert_eq!(r_q, r_q1);
// }

// #[test]
// fn test_permute() {
//     let shape = [1, 2, 4]; // [nh, seq, dh]
//     // let shape = [2, 2, 4, 8]; // [nh, h, w, dh]
//     // let shape = [2, 2, 4, 8, 12]; // [nh, h, w, t, dh]
//     // let shape = [2, 2, 4, 8, 12, 16]; // [nh, h, w, t, e, dh]
//     let rope_section = None;
//     test_permute_nm(&shape, rope_section);
// }

// #[test]
// fn test_section() {
//     let shape = [2, 8, 2, 4, 16]; // [nh, t, h, w, dh], 可以改变维度顺序，会体现在pos_ids上
//     let rope_section = Some(vec![2, 2, 4]); // 可以手动设置各个维度的大小, 不设置则默认均分
//     test_permute_nm(&shape, rope_section);
// }
