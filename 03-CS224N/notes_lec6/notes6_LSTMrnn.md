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


