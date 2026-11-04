****
# 句法结构Syntactic Structure

**成分结构constituency structure**：organize words into nested constituents
重点是分组和嵌套

**依存结构dependency structure**：showmwhich words depend on other words
重点是词与词的连接。
如cuddly cat, 中心词cat，依存词cuddly

为什么连接位置影响理解？
介词短语(prepositional phrase,PP)附着**歧义**attachment ambiguity
例句: Scientists count whales from space

# 依存语法与树库Dependency Grammar and Treebanks
依存关系记录中心词head，依存词dependent，以及依存关系

依存树库为句子提供人工标注gold annotations，用于训练和评估。

依存关系判断线索：

| 线索                         | 判断时看什么？                                        |
| -------------------------- | ---------------------------------------------- |
| 词对搭配（bilexical affinities） | 两个词在这种语法关系中是否常见？例如 _read_ 和 _books_ 容易形成动宾关系。  |
| 依存距离（dependency distance）  | 候选中心词离当前词**有多远**？很多依存关系发生在近处，距离可以作为线索。         |
| 中间成分（intervening material） | 两个**词之间**有没有标点或其他动词，提示它们可能分属不同结构？              |
| 中心词的配价（valency）            | 这个中心词通常**和哪些成分搭配**？例如 _give_ 常带「给什么」和「给谁」两个成分。 |
依存树**约束**：
每个词有且仅有一个中心词；
不能有环。

**投射性projectivity**
词序排列。交叉弧

# 基于转移的依存句法分析Transition-based Dependency Parsing
把建树拆成对状态（σ，β，A）及（stack栈，buffer缓冲区，依存边集合）的一系列操作。
shift把缓冲区最左边的词移到栈最右边

| 动作            | 加哪条边？                 | 哪个词离开栈？ |
| ------------- | --------------------- | ------- |
| 左弧（Left-Arc）  | w_j -> w_i，右边的词指向左边的词 | 左边的w_i  |
| 右弧（Right-Arc） | w_i -> w_j，左边的词指向右边的词 | 右边的w_j  |
评估指标

| 指标                                           | 一个词怎样才算对？   |         课件例子 |
| -------------------------------------------- | ----------- | -----------: |
| **无标签附着准确率（Unlabeled Attachment Score，UAS）** | 中心词选对       | \(4/5=80\%\) |
| **带标签附着准确率（Labeled Attachment Score，LAS）**   | 中心词和关系标签都选对 | \(2/5=40\%\) |
# 神经依存句法分析Neural Dependency Parsing
**输入**
取词，查词嵌入表
找到对应的词嵌入向量，按固定顺序拼接，得到输入x

**前馈网络预测动作**
把拼接后的输入x转化为relu隐藏向量，再softmax得到概率向量y，再算交叉熵损失。

**基于图的神经分析器**给候选中心词与依存词的连接打分，再按全句的树约束选边