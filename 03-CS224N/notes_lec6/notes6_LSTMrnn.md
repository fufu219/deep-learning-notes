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

## 统计机器翻译SMT
从数据中学习概率模型。

| 部分                                  | 主要作用                     | 学习所需的数据                    |
| ----------------------------------- | ------------------------ | -------------------------- |
| **翻译模型（translation model）** P(x\|y) | 衡量候选译文与源句之间的翻译对应关系，支持忠实性 | 平行语料（parallel data）        |
| **语言模型（language model）** P(y)       | 衡量目标语言句子的概率，支持流畅性        | 目标语言单语语料（monolingual data） |

平行语料parallel corpus指**互为翻译的文本对**组成的数据集
## neural machine translation神经机器翻译NMT!!
采用经典结构seq2seq，把一个输入序列转换成一个输出序列。

| 组成               | 任务             | 输入与输出                |
| ---------------- | -------------- | -------------------- |
| **编码器（encoder）** | 读取源句，计算源句的向量表示 | 源语言词序列 → 编码状态        |
| **解码器（decoder）** | 根据源句信息，逐词生成译文  | 编码状态与此前的目标词 → 下一个目标词 |

RNN 编码器读取源句并形成编码状态，解码器以该状态为初始条件，根据此前的目标词逐步预测下一个词，直到生成结束符号。

$$
 P(y\mid x) = P(y_1\mid x) P(y_2\mid y_1,x) \cdots P(y_T\mid y_1,\ldots,y_{T-1},x).
$$

x是完整源句，y_t是第t个目标词，T是目标序列长度。每一项都是在源句和已有目标词已知时，下一个正确目标词的概率。相乘即整段目标序列的概率。

解码器每次预测都依赖**源句和此前的目标词**，所以属于**条件语言模型**。

损失计算同样是“模型给正确词的概率”取负对数。计算损失后，反向传播从这些预测出发，经过解码器，再沿编码状态回到编码器。这样，损失会同时推动解码器学习“怎样生成正确译文”，以及编码器学习“怎样表示源句，才能帮助翻译”。

这就是**端到端训练（end-to-end training）**。编码器、解码器以及词嵌入、输出层等可训练参数，都通过**同一个翻译目标**联合更新。
编码器与解码器的结构也能用于摘要、对话、代码生成等序列转换任务。

So，监督式 seq2seq 训练用**正确目标**前缀预测下一个目标词，以逐步交叉熵构成训练损失，并将梯度通过解码器传播到编码器，联合学习整个翻译系统。

code
	`import torch.nn as nn`
	
	定义模型组件
	source_embedding = nn.Embedding(source_vocab_size, embedding_size)
	target_embedding = nn.Embedding(target_vocab_size, embedding_size)
	
	encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
	decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
	output_layer = nn.Linear(hidden_size, target_vocab_size)
	
	前向计算
	source_vectors = source_embedding(source_ids)
	_, encoder_state = encoder(source_vectors)
	
	target_vectors = target_embedding(target_input_ids)
	decoder_outputs, _ = decoder(target_vectors, encoder_state)
	
	logits = output_layer(decoder_outputs)


**多层结构与编码瓶颈bottleneck**
在这种基本结构中，源句信息需要经由编码器最后传给解码器的固定大小状态，支持后续所有词的生成。源句越长、内容越复杂，让这个状态保留翻译所需的全部信息就越困难。

# 整理

|概念|它描述什么|在当前模型中的关系|
|---|---|---|
|**机器翻译 MT**|要完成的任务|将源语言文本转换成目标语言文本|
|**神经机器翻译 NMT**|用神经网络完成翻译的方法类别|本讲正在介绍的方法|
|**seq2seq／encoder–decoder**|组织输入到输出的整体结构|先编码源句，再解码译文，也可用于摘要等任务|
|**编码器、解码器**|系统中两个组件的职责|分别负责读取输入、生成输出|
|**RNN、LSTM**|可以用来构建组件的网络类型|本讲使用它们实现编码器和解码器|
|**双向、多层**|网络可以采用的结构设计|编码器或解码器可根据任务采用适合的设计|
|**端到端训练**|怎样优化整个系统|最终翻译损失联合更新相关模块|
|**平行语料**|训练数据的组织形式|提供源句与正确译文|

# 检查理解
1. 编码器读完整个源句后，传给解码器的编码状态起什么作用？
	编码器的最终状态 初始化 解码器的状态。
2. 某个目标词预测错误产生的损失，能否更新编码器的参数？请说明梯度通过什么连接传回编码器。
	可以。梯度经过解码器，沿（作为初始状态）传入的编码状态，回到编码器的计算和参数。


