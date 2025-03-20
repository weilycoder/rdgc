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
+ [ ] 生成特定类型的数字
+ [x] 生成特定类型的数字序列
  + [x] 生成给定递推函数的序列（递归实现）
+ [ ] 调用 std 生成 ans 文件

## License

项目没有额外声明的部分使用 [SATA](https://github.com/zTrix/sata-license) 许可证，如果你认为我的项目有用，请考虑为我点赞。
