****
# Intro ：为什么要研究word representation
wordnet语义上存在的问题:
词义有细微差别nuance；词义不断更新。
**regard words as discrete symbols**离散符号

**one-hot vectors**
例：motel=[`0 0 0 0 0 0 0 0 0 0 1 0 0 0 0`] ; hotel=[`0 0 0 0 0 0 0 1 0 0 0 0 0 0 0`] 但两者独立(orthogonal), 即one-hot编码无法察觉语义相似性similarity。

# skip-gram模型

分布式语义distributional semantics ：要关联上下文语境**context**，判断词义的含义用法。
**word vector=word embedding(词嵌入)**
**similarity**: vector dot product词向量点积

word2vec是一组方法，这里先介绍**skip-gram模型**：用context word预测center word
窗口产生4个训练词对-->窗口继续滑动，下一个位置的词成为center word-->产生新词对
![示意图](./assets/Pasted%20image%2020261101113127.png)![示意图](./assets/Pasted%20image%2020261101113149.png)
**likelihood**
$$
L(\theta)=\prod_{t=1}^{T}\prod_{\substack{-m\leq j\leq m\\j\neq 0}}P(w_{t+j}\mid w_t;\theta) $$
T指语料长度，m是窗口半径，θ是全部模型参数。

**objective function**目标/损失函数：平均负对数似然
$$
J(\theta)=-\frac{1}{T}\log L(\theta) =-\frac{1}{T}\sum_{t=1}^{T}\sum_{\substack{-m\leq j\leq m\\j\neq 0}}\log P(w_{t+j}\mid w_t;\theta) 
$$
$$ \min_{\theta}J(\theta)\quad\Longleftrightarrow\quad\max_{\theta}L(\theta) $$

那如何计算P（..|..,..）?
**softmax**
两套向量： ![示意图](./assets/Pasted%20image%2020261101120347.png)
1. 对于center word **c**和context word **w** ，点积计算分数![示意图](./assets/Pasted%20image%2020261101120606.png)（标量，并非概率，理解为二者的匹配程度）
2. 代入softmax，对算出来的分数取指数，得出概率。

$$ P(o\mid c) = \frac{\exp(\mathbf{u}_o^{\top}\mathbf{v}_c)} {\sum_{w\in\mathcal{V}}\exp(\mathbf{u}_w^{\top}\mathbf{v}_c)} $$

整个词表大小为V，每个词有2套向量，每个向量有d个分量，则共有**2Vd**个可学习参数。

**梯度下降（gradient descent)公式推导**

概率P对v_c求导
![示意图](./assets/8fc41d4cead546bf99277cd7b8a7ce1.jpg)

**optimization：更新参数**
$$ \theta^{\text{new}} = \theta^{\text{old}} - \alpha\nabla_{\theta}J(\theta) $$
![示意图](./assets/df06d17ad5caf967fb6bbe9a614b930.jpg)
**algorithm**
	`while True:
		`theta_grad=evaluate_gradient(J,corpus,theta)` 要def这个函数，并非自带
		`theta=theta_alpha * theta_grad`

But!! 如果每次更新都计算整个语料库的损失和梯度，太麻烦。因此采用**随机梯度下降(stochastic gradient descent, SGD)**
**algorithm**
	`while True:`
		`window=sample_window(corpus)`
		`theta_grad=evaluate_gradient(J,windows,theta) #corpus变windows采样窗口
		`theta=theta-alpha*theta_grad`
skip-gram没有区分context word具体在中心词的哪边、偏移位置如何

# 负采样negative sampling

| 模型                                  | 输入     | 预测目标 |
| ----------------------------------- | ------ | ---- |
| Skip-gram，SG                        | 中心词    | 上下文词 |
| 连续词袋模型，Continuous Bag of Words，CBOW | 周围上下文词 | 中心词  |

由于完整的softmax遍历整个词表计算成本较高，采用skip-gram+negative sampling
$$\sigma(s)=\frac{1}{1+e^{-s}} $$
把点积分数转换为二分类概率![示意图](./assets/Pasted%20image%2020261101172133.png)
正样本的s(即uT·v)越大，σ越小，损失越小；反之...

对一个词对，损失
$$ l=-\log\sigma(\mathbf u_o^\top\mathbf v_c)$$
对真实context word **o**和K个负样本，损失函数可写成
$$ J_{\mathrm{neg}} = -\log\sigma(\mathbf u_o^\top\mathbf v_c) - \sum_{k=1}^{K} \log\sigma(-\mathbf u_{n_k}^\top\mathbf v_c) $$
**负样本项**前加**负号**，保持趋势变化（输入与输出的正/负相关性？）的一致性。

**稀疏梯度sparse gradient**
一次负采样训练只涉及少量词向量，未参与本次损失计算的词向量，其梯度为 0，因此整个词向量参数的梯度大部分为 0。

**正负样本关系**
	m：窗口半径
	K：每个正样本词对抽取的负样本数
一个完整窗口最多包含**1** center word+**2m** context word ； 抽取**2mK**个负样本词
**向量角色与正负标签**是两个不同概念。
对每个真实的中心词与上下文词组成的正样本，保留中心词，从噪声分布中抽取K个词，构造负样本词对。训练**提高**正样本的匹配分数，**降低**负样本的匹配分数，从而避免每次遍历整个词表。

# 共现矩阵co-occurrence matrix
通过统计词在指定上下文中的共同出现次数，将词的**使用环境表示为数值向量**。
X_ij=词j出现在词i的上下文窗口中的次数。对称性X_ij=X_ji
词与词的窗口反映局部关系，词与文档的共现偏向主题关系。

如果把共现矩阵的每一行当作词向量，存在**维度高**和**稀疏**的问题。因此需要**降维**。
## 奇异值分解singular value decomposition SVD
![示意图](./assets/Pasted%20image%2020261101210930.png) 保留最大的k个奇异值后-->  ![示意图](./assets/Pasted%20image%2020261101210939.png)
![示意图](./assets/Pasted%20image%2020261101210815.png)
k：选择保留的维度
截断 SVD 保留最大的k个奇异值及相应向量，用低秩矩阵近似共现矩阵，并从中提取低维词表示。重建误差最小不等于所有语义任务效果最优。
低维词表示的行对应词，列对应降维后得到的特征方向；这些特征通常综合多个原始上下文维度。重建矩阵仍保留原共现矩阵的形状。
![示意图](./assets/Pasted%20image%2020261102103940.png)
the等词出现频繁产生干扰，因此做SVD之前**处理原始计数**，可用以下方法：

| 方法            | 作用                |
| ------------- | ----------------- |
| 对计数取对数        | 压缩大计数与小计数之间的差距    |
| min(X_ij,t)   | 将超过阈值 \(t\) 的计数截断 |
| 忽略部分功能词       | 减少这些高频词的影响        |
| 近处词权重大、远处词权重小 | 区分不同距离的上下文        |
| 使用相关系数等统计量    | 不只关注原始共现次数        |
## GloVe
思想：比较两个词与同一上下文词的共现概率，可以突出这两个词之间的差别。
![示意图](./assets/Pasted%20image%2020261101211916.png)
比较不同中心词下，某一个context word出现的概率。比值若接近1就不一定

log-bilinear relation把概率比值关联到向量差
$$ \mathbf w_x^\top\mathbf w_a \approx \log P(x\mid a) $$
$$ \boxed{ \mathbf w_x^\top(\mathbf w_a-\mathbf w_b) \approx \log\frac{P(x\mid a)}{P(x\mid b)} } $$
**损失**计算：加权平方误差，模拟给出的拟合值-统计得到的目标值
![示意图](./assets/Pasted%20image%2020261101212545.png)调整词向量和偏置以减小损失。参数如下
![示意图](./assets/Pasted%20image%2020261101212912.png)![示意图](./assets/Pasted%20image%2020261101213023.png)
GloVe利用全局词共现统计学习词向量，通过最小化加权平方误差，使两套词向量的点积与偏置之和拟合对数共现次数，希望用向量差表达共现概率比值所体现的语义关系。

# 词向量的评价

| 类型                         | 具体做什么           | 优点与局限                            |
| -------------------------- | --------------- | -------------------------------- |
| 内在评价（intrinsic evaluation） | 直接测试词相似度、词类比等性质 | 较快，方便分析，但不一定代表实际任务效果。直接测试表示的特定性质 |
| 外在评价（extrinsic evaluation） | 把词向量放进真实任务，检查结果 | 更直接体现用途，但耗时，也受其他模块影响。通过下游任务表现来检验 |
**内在评价**
· 词类比analogies
![示意图](./assets/Pasted%20image%2020261102111552.png)向量线性加减，返回相似度最高的候选词。
· 与人工相似度评分比较
比较一致性而非分数；spearman系数衡量排序关系
**外在评价**
在同一套任务系统中替换词向量，检查实体识别表现是否改善。
比较时应控制变量
# 一词多义ambiguity
**多原型词表示(multiple word prototypyes)** 根据上下文聚类
**线性叠加(linear superposition)** 词义向量加权组合

# 词向量分类
**NER**(NAMED ENTITY RECOGNITION)是识别并分类文本中实体的任务。词的类别可能依赖上下文。
**拼接（concatenation）** 窗口内按顺序拼接每个词向量的分量![示意图](./assets/Pasted%20image%2020261102120505.png)
**多类别分类**
传入原始的logits和整数类别标签
softmax把各类分数分配概率
$$ P(y=c\mid\mathbf x) = \frac{\exp(\mathbf W_c\mathbf x)} {\sum_{j=1}^{C}\exp(\mathbf W_j\mathbf x)} $$
	1. 二分类1/0，加入隐藏层
	![示意图](./assets/Pasted%20image%2020261102121416.png)
	2. 计算分类分数 $\[ s=\mathbf u^\top\mathbf h \]$
	3. sigmoid函数得到概率 $\[ \hat p=\sigma(s)=\frac{1}{1+e^{-s}} \]$

**交叉熵cross entropy**
![示意图](./assets/Pasted%20image%2020261102122041.png)p(c)是目标分布，q(c)是模型预测概率，C是类别数
如果one-hot目标p(y)=1，则H=-log q(y)
`torch.nn.CrossEntropyLoss()`创建交叉熵损失对象

**神经网络**
单个神经元
$$
a_i=f\left(\sum_{j=1}^{D}W_{ij}x_j+b_i\right) 
$$
每个输入分量乘权重，连加，再加偏置，通过激活函数得到输出