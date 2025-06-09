# ndrope

[![CI](https://github.com/CearX/ndrope/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/CearX/ndrope/actions)
[![Latest version](https://img.shields.io/crates/v/ndrope.svg)](https://crates.io/crates/ndrope)
[![Documentation](https://docs.rs/ndrope/badge.svg)](https://docs.rs/ndrope)
[![license](https://img.shields.io/github/license/CearX/ndrope)](https://mit-license.org/)
[![codecov](https://codecov.io/github/CearX/ndrope/branch/main/graph/badge.svg)](https://codecov.io/github/CearX/ndrope)
![GitHub repo size](https://img.shields.io/github/repo-size/CearX/ndrope)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/CearX/ndrope)

[![GitHub Issues](https://img.shields.io/github/issues/CearX/ndrope)](https://github.com/CearX/ndrope/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/CearX/ndrope)](https://github.com/CearX/ndrope/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/CearX/ndrope)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/CearX/ndrope)

*ndrope* 是一个用于计算 n 维 `RoPE` 的 crate, 它提供了 n 维 `RoPE` 的两种实现，支持多种 `pos_ids` 和 `sin_cos_table` 计算方法，以及多种数据类型的计算，可以满足多种模型的 `RoPE` 计算需求。

## 主要功能特点

### `RoPE` 的两种实现

`RoPE (Rotary Position Embedding)`, 即**旋转位置编码**算子，常见有两种实现，主要区别在于分组。

- `rope_nd`

  - `RoPE` 论文中原版实现，每两个相邻元素一组，在本项目中称为 `rope_nd`。对于一个位置为 $m$ 的 token 的嵌入向量 $x$ 中的每两个相邻元素，即第 $2i$ 和 $2i+1$ 个元素，其旋转后的嵌入向量 $y$ 可以表示为：

$$
y[2i] = x[2i] \cdot \cos(m\cdot \theta_i) - x[2i+1] \cdot \sin(m\cdot \theta_i)
$$
$$
y[2i+1] = x[2i] \cdot \sin(m\cdot \theta_i) + x[2i+1] \cdot \cos(m\cdot \theta_i)
$$

- `rope_m`

  - `huggingface` 中实现，相距 `d/2` 的每两个元素一组, 在本项目中称为 `rope_m`。对于一个位置为 $m$ 的 token 的嵌入向量 $x$ 中的每两个距离 $dh/2$ 的元素对，即第 $i$ 和 $i+dh/2$ 个元素，其旋转后的嵌入向量 $y$ 可以表示为：

$$
y[i] = x[i] \cdot \cos(m\cdot \theta_i) - x[i+dh/2] \cdot \sin(m\cdot \theta_i)
$$
$$
y[i+dh/2] = x[i] \cdot \sin(m\cdot \theta_i) + x[i+dh/2] \cdot \cos(m\cdot \theta_i)
$$

其中：

$$
\theta_{i} = \text{base}^{\frac{-2i}{d}}
$$

- `base`：控制旋转速率的超参数，常用 `base` = 10000。
- `d`：嵌入向量长度，须为2的倍数。
- `i`：嵌入维度的双步长索引，满足 $i \in [0, ..., d/2 - 1]$。
- `m`：位置

#### 两种实现的转换

这两种实现可以通过变换模型权重的方式转换，在测试代码中可以验证。

---

### n 维 `RoPE`

不同维度的 `RoPE` 适用于不同任务：

- **1维 RoPE**: 序列任务，例如文本处理 (seq);
- **2维 RoPE**: 图像任务，例如高度 (h) 和宽度 (w);
- **3维 RoPE**: 视频任务，例如高度 (h)、宽度 (w) 和时间 (t);
- ...

#### 计算规则

- n 个维度需要分配到 `dh` 上。
- 支持设置 `rope_section` 分配各维度权重；如果未设置，则默认均分。

#### 示例

以下为 3 维 `RoPE` 的计算示例：

![RoPE N-d Diagram](https://raw.githubusercontent.com/limefax/rope-nd/main/rope_nd.svg)

- `dh = 12`，分配到 h、w、t 三个维度。
- `p` 中分别是 h、w、t 的 `pos_ids`。
- h、w、t 均分嵌入向量 `x` 的 `dh`。
- 两两一组计算 `RoPE`。

---

### `pos_ids`

#### 支持多种 `pos_ids` 计算方法

- **pos_nd**: 计算 n 维的 `pos_ids`;
- **pos_2d_qwen2vl_vit**: 计算 `qwen2vl-vit` 的 2 维 `pos_ids`;
- **pos_3d_qwen2vl_llm**: *待实现*;

#### `pos_ids` 兼容数据类型

- **u32**;
- **u64**;

### `sin_cos`

#### 支持多种 `sin_cos_table` 计算方法

- **sin_cos**: 普通 `sin_cos` 方法;
- **sin_cos_nd**: n 维 `RoPE` 的 `sin_cos` 方法;

#### `sin_cos` 兼容数据类型

- **f16**;
- **f32**;
- **f64**;

### `RoPE` 计算

#### 支持多种模型

- **1维 RoPE**: llama;
- **2维 RoPE**: qwen2vl-vit;
- **3维 RoPE**: *待实现*: qwen2vl-llm;

#### `RoPE`兼容数据类型

- **f16**: **f16** 的张量计算时会转为 **f32**，需要传入 **f32** 的 `sin_cos` 来提高精度;
- **f32**;
- **f64**;

## 使用示例

以下为使用 `rope_nd` 方法计算 1 维 `RoPE`的计算示例，张量数据类型为 `f32`, `pos_ids` 数据类型为 `u32`：

```rust
use ndrope::{pos_ids::pos_nd, sin_cos::sin_cos_nd, rope_nd, tensor};
use digit_layout::types;

let shape = [1, 2, 4]; // [nh, seq, dh]
let size = std::mem::size_of::<f32>();
let mut data = (0..(shape[0] * shape[1] * shape[2]))
    .map(|i| i as f32)
    .collect::<Vec<f32>>(); // x1设为递增序列
let x1 =
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, size * data.len()) };
let dt = types::F32;
let strides = [
    (shape[1] * shape[2] * size) as isize,
    (shape[2] * size) as isize,
    size as isize,
];
let offset = 0;
let x = tensor(x1, dt, shape.to_vec(), strides.to_vec(), offset);

let grid = [2];
let rope_section = None;

let pos = pos_nd::<u32>(grid.to_vec());
let [sin, cos] = sin_cos_nd::<f32>(&shape, &grid, rope_section.clone(), 10000.0);
rope_nd(x, pos, sin, cos, &grid, rope_section);

let out = unsafe { std::slice::from_raw_parts_mut(x1.as_mut_ptr() as *mut f32, data.len()) };
let ans = [
    0.0, 1.0, 2.0, 3.0, -2.0461454, 6.067395, 5.9297013, 7.059649,
];
assert_eq!(out, ans);
```
