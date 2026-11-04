# Deep Learning Notes

个人深度学习学习笔记，记录从 PyTorch 基础、卷积神经网络（CNN）到 Stanford CS224N 的学习过程。

本仓库以概念理解、公式推导、代码阅读和实验方法为主。内容持续更新，属于个人学习记录，不是官方教材或完整课程答案；笔记中可能存在疏漏，欢迎交流与指正。

## 学习路线与导航

| 模块 | 主要内容 | 笔记入口 |
| --- | --- | --- |
| 01 · PyTorch 基础 | 张量、梯度与自动求导、优化器、MLP、训练与评估、模型保存及实验分析 | [A-NOTE](01-pytorchbasics/A-NOTE.md) |
| 02 · CNN | 卷积、通道与输出尺寸、ReLU、池化、小型 CNN 和错误分析 | [B-Notes](02-CNN/B-Notes.md) |
| 03 · CS224N | 词向量、神经网络、依存句法分析，以及后续 NLP 课程内容 | [课程说明](03-CS224N/README.md) |

### CS224N 笔记

- [Lecture 1 & 2 · Word Vectors](03-CS224N/notes_lec1%20lec2/notes1%262_Word%20Vectors.md)
- [Lecture 3 · Neural Networks](03-CS224N/notes_lec3/notes3_Neuralnets.md)
- [Lecture 4 · Dependency Parsing](03-CS224N/notes_lec4/notes4_dependency%20parsing.md)

以上导航表示当前已有笔记，不代表对应内容或课程作业已经全部完成。

## 目录结构

```text
.
├── README.md
├── 01-pytorchbasics/
│   ├── A-NOTE.md
│   └── assets/
├── 02-CNN/
│   ├── B-Notes.md
│   └── assets/
└── 03-CS224N/
    ├── README.md
    ├── notes_lec1 lec2/
    │   ├── notes1&2_Word Vectors.md
    │   └── assets/
    ├── notes_lec3/
    │   ├── notes3_Neuralnets.md
    │   └── assets/
    ├── notes_lec4/
    │   └── notes4_dependency parsing.md
    └── references/
```

目录树仅展示主要入口；各模块可能包含其他辅助文件和图片。

## 阅读说明

- 推荐按 **PyTorch 基础 → CNN → CS224N** 的顺序阅读，也可以通过导航直接进入感兴趣的主题。
- 笔记使用 Markdown，并尽量采用 GitHub 与 Obsidian 均可阅读的图片相对路径和数学公式格式。
- 下载或移动笔记时，请同时保留对应的 `assets` 文件夹及相对目录结构。
- 跨笔记链接依赖仓库中的文件位置；调整目录或标题后，需要同步检查链接。
- 部分笔记保留了指向个人电脑的 `file:///` 链接。这些链接不能供 GitHub 访客访问，相关 notebook 暂不纳入本仓库。

## 学习资料与致谢

- 深度学习实现与练习主要使用 [PyTorch](https://pytorch.org/)。
- CS224N 部分的课程版本与资料说明见 [课程 README](03-CS224N/README.md)。
- 课程官网：[Stanford CS224N — Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)。

个人笔记与课程原始资料应分别看待。引用或收录的讲义、图片和作业材料，其权利归原作者或相应权利人所有；本仓库不额外授予这些第三方材料的使用许可。
