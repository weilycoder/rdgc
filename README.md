# RDGC Data Generator for Contests

[![Release](https://img.shields.io/github/v/release/weilycoder/rdgc)](https://github.com/weilycoder/rdgc/releases/)
![Last Commit](https://img.shields.io/github/last-commit/weilycoder/rdgc)
[![github](https://img.shields.io/badge/github-rdgc-blue?logo=github)](https://github.com/weilycoder/rdgc)
[![PyPI](https://img.shields.io/badge/PyPI-rdgc-blue?logo=pypi)](https://pypi.org/project/rdgc/)
[![sata-license](https://img.shields.io/badge/License-SATA-green)](https://github.com/zTrix/sata-license)

[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-raw/weilycoder/rdgc)](https://github.com/weilycoder/rdgc/issues)
![GitHub Downloads](https://img.shields.io/github/downloads/weilycoder/rdgc/total)
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
  + [x] 生成 $k$-正则图（ $k$-regular Graph）
+ [x] 生成特定类型的数字
  + [x] 支持从范围中随机选择质数。
+ [x] 生成特定类型的数字序列
  + [x] 生成给定递推函数的序列（递归实现）
+ [ ] 调用 std 生成 ans 文件
+ [x] 杂项
  + [x] 从命令行参数获取随机种子
  + [x] 批量将 CRLF 格式转换为 LF 格式

## PyPI

项目上传了 [PyPI](https://pypi.org/project/rdgc/)，因此你可以使用：

```bash
pip install rdgc
```

来下载最新版。

项目也将同时上传到 [Test PyPI](https://test.pypi.org/project/rdgc/)，因此也可以：

```bash
pip install -i https://test.pypi.org/simple/ rdgc
```

当然，后者不是推荐行为。

## Semantic Versioning

原则上，我是支持 [语义化版本](https://semver.org/lang/zh-CN/) 的，但是由于现在项目随时可能出现 Bug 和 idea，并可能由于使 API 更可读而重构，因此将项目版本号维持在 `0.x.x` 以说明处于快速开发阶段。

计划在完成大部分 idea 后进行一次重构并将版本号改为 `1.x.x`。

目前尽量遵循以下规范：

+ 若引入了重要的数据生成 API（被记录在 README 文件），则将次版本号递增；
+ 若加入了小的 API 或 Bug 修复，将修订号递增。

当然，暂时更可能看我心情。

## License

项目没有额外声明的部分使用 [SATA](https://github.com/zTrix/sata-license) 许可证，如果你认为我的项目有用，请考虑在 [Github](https://github.com/weilycoder/rdgc) 为我点赞。

项目的 API 设计参考了 [CYaRon](https://github.com/luogu-dev/cyaron)，如果有人认为我需要为此修改许可证，请联系我；后者的推荐方式是在 [issue](https://github.com/weilycoder/rdgc/issues) 中说明。

## Pylint

项目使用 Pylint 检查 Python 语法。

全局禁用的检查包括：

+ line-too-long (C0301)
+ too-many-lines (C0302)
+ unnecessary-lambda-assignment (C3001)
+ too-many-public-methods (R0904)
+ too-many-return-statements (R0911)
+ too-many-arguments (R0913)
+ too-many-locals (R0914)
+ keyword-arg-before-vararg (W1113)

在测试脚本中禁用了所有出现的警告，包括但不限于注释缺失。

在项目主文件中禁用的警告使用具体名称而非代码。
