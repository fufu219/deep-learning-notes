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

**2 句子编码sentence encoding**
用一个向量表示一句话：将变长的句子输入序列映射为固定维度向量，供句子级任务使用。
取最后一个隐藏状态/平均池化/最大池化

**3 条件文本生成conditional text generation**
预测下一个词时同时依赖**额外输入**和此前的词
可用于语音识别，机器翻译，摘要等任务