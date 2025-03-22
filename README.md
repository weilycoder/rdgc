# RDGC Data Generator for Contests

[![Release](https://img.shields.io/github/v/release/weilycoder/rdgc)](https://github.com/weilycoder/rdgc/releases/)
[![github](https://img.shields.io/badge/github-rdgc-blue?logo=github)](https://github.com/weilycoder/rdgc)
[![Test PyPI](https://img.shields.io/badge/Test_PyPI-rdgc-blue?logo=pypi)](https://test.pypi.org/project/rdgc/)
[![sata-license](https://img.shields.io/badge/License-SATA-green)](https://github.com/zTrix/sata-license)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-raw/weilycoder/rdgc)](https://github.com/weilycoder/rdgc/issues)
![Stars](https://img.shields.io/github/stars/weilycoder/rdgc)
![Forks](https://img.shields.io/github/forks/weilycoder/rdgc)

由于我现在常用的数据生成器在 API 设计上令人迷惑，我决定自己编写一个生成器。

## features

目前计划支持的特性：

+ [x] 生成随机图
  + [x] 随机图
  + [x] 空图（null graph）
  + [x] 完全图（complete graph）
  + [x] 竞赛图（tournament graph）
  + [x] 随机树（随机父亲）
  + [x] 随机树（随机连边）
  + [x] 二叉树（binary tree）
  + [x] 链图（chain graph）
  + [x] 菊花图（star graph）
  + [x] 圈图（cycle graph）
  + [x] 轮图（wheel graph）
  + [x] 从度数序列随机生成图（蒙特卡洛方法）
  + [x] 生成 $k$-正则图（$k$-regular Graph）
+ [ ] 生成特定类型的数字
+ [x] 生成特定类型的数字序列
  + [x] 生成给定递推函数的序列（递归实现）
+ [ ] 调用 std 生成 ans 文件
+ [x] 杂项
  + [x] 从命令行参数获取随机种子
  + [x] 批量将 CRLF 格式转换为 LF 格式

## Semantic Versioning

原则上，我是支持 [语义化版本](https://semver.org/lang/zh-CN/) 的，但是由于现在项目随时可能出现 Bug 和 idea，并可能由于使 API 更可读而重构，因此将项目版本号维持在 `0.x.x` 以说明处于快速开发阶段。

计划在完成大部分 idea 后进行一次重构并将版本号改为 `1.x.x`。

目前尽量遵循以下规范：

+ 若引入了重要的数据生成 API（被记录在 README 文件），则将次版本号递增；
+ 若加入了小的 API 或 Bug 修复，将修订号递增。

当然，暂时更可能看我心情。

## License

项目没有额外声明的部分使用 [SATA](https://github.com/zTrix/sata-license) 许可证，如果你认为我的项目有用，请考虑为我点赞。
