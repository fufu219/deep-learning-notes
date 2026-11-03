****
# Intro

**隐藏层与概率**
m表示隐藏单元个数

| 运算                | 输出形状 | 这一步在做什么                        |
| ----------------- | ---- | ------------------------------ |
| z=Wx+b            | m×1  | W的每一行和 x 做一次点积，再加上对应的偏置        |
| h=f(z)            | m×1  | 激活函数（activation function）逐元素作用 |
| s=u^T* h          | 标量   | 对 m个隐藏层输出加权求和，得到地点类别的分数        |
| σ(s)=1/(1+e^(-3)) | 标量   | 用 sigmoid 把分数映射成中心词为地点的预测概率    |

**non linearities非线性的重要性**
如果隐藏层之间只有矩阵乘法和偏置，相当于只有一个仿射变换affine transformation

**参数更新**
$$
 \theta^{\mathrm{new}} = \theta^{\mathrm{old}} - \alpha\nabla_\theta J(\theta^{\mathrm{old}}). 
$$

# 矩阵微积分：梯度、雅可比矩阵与链式法则
**梯度gradient**
偏导数partial derivative：某个输入的微小变化怎么影响输出？

**Jacob矩阵**
![329](Pasted%20image%2020261103214452.png)

**chain rule链式法则**

