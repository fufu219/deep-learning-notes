**recap**
language model：a system that predicts the next word
rnn（recurrent neural network） 任意长度/按顺序输入，每一步权重相同。但rnn不是lm
# vanishing & exploding
$$
 h^{(t)}=\sigma\!\left(W_hh^{(t-1)}+W_xx^{(t)}+b_1\right). 
$$

梯度消失（vanishing gradient）是指梯度在跨多个时间步反向传播时，因局部导数连续相乘而衰减到很小。较晚的损失因此难以影响较早时间步对输入的处理，使普通 RNN 难以学习长距离依赖（long-distance dependency）

反之，若连乘持续放大梯度，则可能出现梯度爆炸（exploding gradient）。对应处理当打是梯度裁剪clipping。
# 长短期记忆网络Long Short-Term Memory,LSTM
**更新公式**

$$
 \begin{aligned}{细胞状态} c^{(t)} &= \underbrace{f^{(t)}\odot c^{(t-1)}}_{\text{保留的旧记忆}} + \underbrace{i^{(t)}\odot \tilde c^{(t)}}_{\text{写入的新内容}},\\ {隐藏状态}h^{(t)} &= \underbrace{o^{(t)}\odot \tanh(c^{(t)})}_{\text{从更新后的记忆中读出}}. \end{aligned} 
$$

f(t)为遗忘门，控制旧记忆保留多少。i(t)为输入门，控制候选细胞内容(candidate cell content)，决定写入的强度。
三个门怎么算得 ![358](assets/Pasted%20image%2020261120190653.png)

**如何缓解梯度消失问题**
反向传播时只看c(t)细胞状态这条更新路径。当遗忘门接近1，就信息更容易传到后面，即后面的损失更容易把学习信号传到前面。更好追溯。

# 其它rnn用途
**1 序列标注sequence tagging**
给每个输入一个标签，如词性POS、实体命名识别NER
使用各个位置的隐藏状态，分别预测相应位置的标签。

**2 句子编码sentence encoding**
用一个向量表示一句话：将变长的句子输入序列映射为固定维度向量，供句子级任务使用。
取最后一个隐藏状态/平均池化/最大池化

**3 条件文本生成conditional text generation**
预测下一个词时同时依赖**额外输入**和此前的词
可用于语音识别，机器翻译，摘要等任务

# 双向与多层rnn:bidirectional and multi-layer RNNs
## BiRNN
因为一个词放在不同句子里，由于前文不同，算出的隐藏状态也不同。

$$
 {前向}\overrightarrow h^{(t)} = \mathrm{RNN}_{\mathrm{FW}} \left(\overrightarrow h^{(t-1)},x^{(t)}\right)  {后向}\overleftarrow h^{(t)} = \mathrm{RNN}_{\mathrm{BW}} \left(\overleftarrow h^{(t+1)},x^{(t)}\right). 
$$

两个方向各自有可训练参数。

拼接词向量，使该位置的表示同时利用左右两侧的上下文。

$$
 h^{(t)} = \left[ \overrightarrow h^{(t)}; \overleftarrow h^{(t)} \right].
$$

## multi-layer RNN/stacked RNN
将下层各时间步的隐藏状态作为上层对应时间步的输入，并在每层保留独立的时间递推。
最佳层数取决于实验任务和设置。并非越多越好。
![334](assets/Pasted%20image%2020261123225859.png)

# machine translation机器翻译，MT
把source language转换成target language，需要同时满足fidelity准确性和fluency流畅性

统计机器翻译SMT从数据中学习概率模型。

| 部分                                   | 主要作用                     | 学习所需的数据                    |
| ------------------------------------ | ------------------------ | -------------------------- |
| **翻译模型（translation model）** (P(x\|y) | 衡量候选译文与源句之间的翻译对应关系，支持忠实性 | 平行语料（parallel data）        |
| **语言模型（language model）** P(y)        | 衡量目标语言句子的概率，支持流畅性        | 目标语言单语语料（monolingual data） |

## neural machine translation神经机器翻译NMT
采用经典结构seq2seq，把一个输入序列转换成一个输出序列。

| 组成               | 任务             | 输入与输出                |
| ---------------- | -------------- | -------------------- |
| **编码器（encoder）** | 读取源句，计算源句的向量表示 | 源语言词序列 → 编码状态        |
| **解码器（decoder）** | 根据源句信息，逐词生成译文  | 编码状态与此前的目标词 → 下一个目标词 |

RNN 编码器读取源句并形成编码状态，解码器以该状态为初始条件，根据此前的目标词逐步预测下一个词，直到生成结束符号。

解码器每次预测都依赖**源句和此前的目标词**，所以是**条件语言模型**