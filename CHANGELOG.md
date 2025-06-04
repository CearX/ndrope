# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

## [0.1.0] - 2025-6-4

### Added

- 创建更新日志

### Changed

- 使用新的api, 支持使用 digit-layout 和 ndarray-layout 描述张量，解耦 pos 和 sin_cos 逻辑
- 重构代码，独立 sin_cos 代码
- 支持调用函数计算 u32 和 u64 的 pos
- 支持调用函数计算 f16, f32, f64 的 sin_cos
- 计算 f16 张量时会转为 f32 来计算，并使用 f32 的 sin_cos 来提高精度

### Removed

- 移除旧api, 不再使用vec

## [0.0.2] - 2025-5-28

### Added

- 支持传入pos_ids

### Changed

- 重构代码，独立 permute 和 pos_ids 代码

## [0.0.1] - 2025-5-21

### Added

- n维rope的两种实现
- n维rope的两种实现可以相互转换
- 支持设置rope_section, 不设置则默认均分