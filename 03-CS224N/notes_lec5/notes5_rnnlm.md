****
# 正则化regularization
有时出现过拟合overfitting，需要提高泛化能力generalization，即：在没见过的、来自目标数据分布的样本上仍然表现良好。

## L2正则化
在原先更新参数的基础上加入**参数惩罚**，约束参数使其不要变太大
![420](assets/Pasted%20image%2020261106134527.png)

λ：正则化强度。人为设置的超参数(hyperparameter)
θ_k：第k个参数

对参数求导：

$$
 \frac{\partial J}{\partial\theta_k} = \frac{\partial}{\partial\theta_k} \left(\frac1N\sum_i\ell_i\right) + 2\lambda\theta_k 
$$

新增项让梯度下降更新时有把参数拉向0的趋势，倾向于获得权重较小的模型。
λ太小，约束作用弱；λ太大，模型可能为了缩小参数而牺牲预测能力。

## 随机失活Dropout
训练时，随机把部分神经元传出的数值暂时置为0。（暂时屏蔽，并非删除，还能恢复。）

激活值activation：神经元计算后输出的数值。如经过relu得到的结果
掩码mask：1/0标记。0表示屏蔽。1保留。
丢弃概率p：每个位置有p概率被置零。人工设置超参数。

为什么能改善泛化：随机屏蔽部分激活值，使神经元的**依赖**变得不可靠，从而促使模型利用更多种特征组合。

缩放：使加权求和部分的平均数值规模一致。
课件约定训练时以p概率置零（dropout），测试时的输入全部保留**不置0**（无dropout），权重相应乘1-p 。

# 补充
## 向量化vectorization
把原本逐个执行的同类计算，组织成向量/矩阵运算。能够加速，减少循环开销。

例：N=500个输入，d=300维，C=5个类别分数。
W是  5×300 的权重矩阵，X 是 300乘500 的输入矩阵，结果是 5乘500，每一列对应一个输入的五个类别分数。这些分数再交给softmax转成概率。

## 参数初始化parameter initialization
给模型的权重和偏置设定起始值。但要打破对称性

**Xavier初始化**的方差公式，描述初始化时从哪个分布抽样，即抽出来的值通常离0有多远。

$$
 \operatorname{Var}(W_i)=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}
$$

W_i是权重矩阵的一个标量元素。n_in，n_out是该层的输入/输出维数。

## 优化器optimizer
**随机梯度下降stochastic gradient descent**

$$
 w_{\text{new}}=w-\eta g
$$

**自适应adaptive优化器**

Adagrad, RMSprop, Adam等


# language model: n-gram
根据给定的前文预测下一个词
![577](assets/Pasted%20image%2020261107191619.png)
概率链式法则把序列概率拆成逐词条件概率。

**Markov assumption**
n-gram预测下一个词时，只看它前面最近的n-1个词。如4gram，只看预测词的前三个词，更早的词怎么变都不影响概率。
![](assets/Pasted%20image%2020261107200124.png)

**计数法**预测概率
$$
 \widehat P(w\mid h) = \frac{\operatorname{count}(h\,w)} {\operatorname{count}(h)}，eg: \widehat P(\text{books}\mid\text{students opened their})=\frac{400}{1000}=0.4
$$

**问题**：
1. 分子或分母为0，数据稀疏data sparsity（n一般不大于5）

| 发生了什么                                     | 课件给出的处理办法                                        |
| ----------------------------------------- | ------------------------------------------------ |
| 前文见过，但它后面从未接过某个候选词。即分子为 0                 | **平滑（smoothing）**：给候选词的计数加一点，使没见过的接法也有少量概率。      |
| `students opened their` 这段前文本身就没见过。即分母为 0 | **回退（backoff）**：改用更短的前文，例如 `opened their`，重新查计数。 |

2. 模型需要保存语料中观察到的词串计数，较长的词串可能让计数表变大。

generate text：按概率采样sampling预测下一个词，生成后文。

**固定窗口的神经语言模型**
窗口内的词，按顺序拼接词向量；经过隐藏层的激活函数，得到隐藏表示h；再权重偏置计算出一个原始分数；softmax计算某个词出现在窗口词之后的概率。
![407](assets/Pasted%20image%2020261107201216.png)

优点：该方法解决了n-gram的计数稀疏问题，具体来说，即使训练语料里从没出现过 `the students opened their` 这个完整窗口，只要四个词都在词表里，模型仍然能分别查到它们的向量！

缺点：窗口外的词仍然看不到(如proctor)；窗口的不同位置使用W的不同部分。

对比

| 模型              | 给模型什么                                | 要预测什么                                                 | 怎么算                                     |
| --------------- | ------------------------------------ | ----------------------------------------------------- | --------------------------------------- |
| **Skip-gram**   | 一个中心词，比如 `opened`                    | 窗口里的某个邻词，比如左边的 `students` 或右边的 `their`。每个邻词各形成一个训练样本。 | 用中心词向量和候选上下文词向量计算匹配分数                   |
| **5-gram 语言模型** | 前面四个词 `the students opened their`    | 紧接着的词，比如 `books`                                      | **计数法**：查语料里这四个词出现了多少次，其中多少次紧接着 `books` |
| **固定窗口神经语言模型**  | 同样是前面四个词 `the students opened their` | 同样是紧接着的词，比如 `books`                                   | 查四个词的词向量，拼接后送入**神经网络**，再用 softmax 算概率   |

# 循环神经网络RNN：Recurrent neural networks
## 介绍模型
循环使用同一套权重，重复更新状态
![](assets/Pasted%20image%2020261107204322.png)
其中
$$
h^{(t)}=\sigma\!\left(W_hh^{(t-1)}+W_e e^{(t)}+b_1\right)
$$

激活函数σ内，第一项W_h处理上一步的状态，W_e处理当前词的向量

**advantage**
	由于**权重共享**，句子变长时，计算步数增加但参数数量不随之增加。
	能处理任何长度的输入
	处理输入的过程对称
	往后计算能保存很久以前的信息

**disadvantage**
	n+1步要用到第n步，不能同时计算，算的慢
	实际上追溯很久以前的信息比较困难

## 训练模型
**交叉熵损失**

$$
 J^{(t)}(\theta) =-\sum_{w\in V} y_w^{(t)}\log \hat y_w^{(t)} =-\log \hat y_{x_{t+1}}^{(t)}
$$

每一步模型给**正确**后文词的概率，取负对数得损失。
![](assets/Pasted%20image%2020261107210741.png)

**链式法则叠加贡献**
沿每条路径乘局部导数，各条路径贡献相加，**随时间反向传播BPTT**。
有时为节省训练开销会**截断truncation**。

例：
![](assets/Pasted%20image%2020261107213440.png)

训练时，无论算损失还是喂给下一步（teacher forcing），都只关注**正确词**。哪怕模型给错误词分配的概率更大。

## 生成文本
rollout：用上一步概率分布抽到的词喂给下一步。即重复采样，把结果送回模型。

## 评价模型
**困惑度perplexity,PPL**: 面对一段真实文本，模型给每个下一个真实词的概率是多少。跟计算CE的是同一个东西。

$$
 \mathrm{PPL} =\left(\frac{1}{p_1p_2\cdots p_T}\right)^{1/T}
$$

其中，p_t表示第t步给真实词的概率。T是参与评价的预测步数。
例

![](assets/Pasted%20image%2020261107215555.png)

跟交叉熵联系：
$$
 \mathrm{PPL} =\exp\left(\frac1T\sum_{t=1}^{T}-\log p_t\right) =\exp(J). 
$$

正相关：在同一份测试文本、同一种切分方式下，**平均交叉熵越低，困惑度也越低**。

Q：有了损失的概念为何还要引入PPL？区别是？
A：可以直接1/PPL换回一个代表整段文本的概率，几何平均。

# RNN问题：梯度消失与梯度爆炸vanishing and exploding gradient
**为什么出问题**

反向传播时按照链式法则连乘局部导数，梯度大小随传播距离快速缩小或放大。

$$
 \frac{\partial J^{(i)}}{\partial h^{(j)}} = \frac{\partial J^{(i)}}{\partial h^{(i)}}\,W_h^{\,i-j}. 
$$

矩阵W反复相乘。看特征值绝对值。

梯度消失让RNN难以学习**长距离依赖（long-distance dependency）**。模型更容易根据近处的预测误差调整参数而非远处。

**处理方法**
**梯度裁剪（gradient clipping）**：按范数(norm)裁剪。向量长度超过阈值就按比例缩小整个梯度向量。

$$
 g_{\mathrm{clip}}=\frac{\tau}{\|g\|}\,g \qquad \text{当 }\|g\|>\tau. 
$$

# ending
语言模型（language model，LM)在本讲中指根据前文给下一个词分配概率的模型。
循环神经网络（recurrent neural network，RNN）是一类网络结构，它按顺序读入数据，每一步用同一套权重更新隐藏状态。把 RNN 的隐藏状态接到词表上的概率输出，就得到本讲的RNN 语言模型（RNN-LM）。

序列标注sequence tagging对每个输入词输出一个标签。句子分类sentence classification给整句话一个标签。






