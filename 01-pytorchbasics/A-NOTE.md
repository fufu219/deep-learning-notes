[A.ipynb](file:///D:%5CCode%5Clearn_pytorch%5CA.ipynb)

# 张量基本概念：维度与形状

张量（Tensor）：用于存放数值，可以是一个数、一串数或多维数组。

维度数（ndim）：张量有几个轴。

形状（shape）：每个轴上分别有多少个元素。

元素总数（numel）：形状中各个数字相乘；标量包含一个元素。

| 例子            | 形状            | 维度数 | 元素总数 |
| ------------- | ------------- | --- | ---- |
| 一个数 `3.0`     | `[]`          | 0   | 1    |
| 一串数 `[1,2,3]` | `[3]`         | 1   | 3    |
| 两行三列的数        | `[2,3]`       | 2   | 6    |
| 一张灰度图片        | `[1,28,28]`   | 3   | 784  |
| 一批8张灰度图片      | `[8,1,28,28]` | 4   | 6272 |

维度数不等于元素数量：`[8]` 是一维张量，包含8个元素。

形状中的数字表示各轴的长度，其含义取决于数据约定。图片批次通常使用 `[B,C,H,W]`，分别表示批次大小、通道数、高度、宽度。

**索引**

整数索引在指定轴上选出一项，该轴不再保留；这一项内部的其他轴保留。

| 表达式 | 含义 | 结果形状 |
|---|---|---|
| `images` | 一批8张图片 | `[8,1,28,28]` |
| `images[2]` | 当前批次中的第三张图片 | `[1,28,28]` |
| `images[2,0]` | 第三张图片的第一个通道 | `[28,28]` |
| `images[2,0,0]` | 该通道的第一行 | `[28]` |
| `images[2,0,0,0]` | 该行的第一个像素 | `[]` |

`images.shape[0]` 读取形状的第一个数字，本例为8。

`images[0]` 取出第一张图片，结果形状为 `[1,28,28]`。

**图片与标签**

一个批次中，图片与标签按位置一一对应。

| 内容     | 图片形状          | 标签形状  |
| ------ | ------------- | ----- |
| 8个样本   | `[8,1,28,28]` | `[8]` |
| 其中一个样本 | `[1,28,28]`   | `[]`  |

图片需要很多像素表示内容；整图分类的标签只需要一个类别编号。

# 损失函数L

均方误差

![示意图](assets/Pasted%20image%2020261016153155.png)

L(w):  将输出，输出，b三个值固定后，损失取决于w.

# 梯度

梯度公式：$dL/dw$

![示意图](assets/Pasted%20image%2020261016155024.png)

（学习率为正）

![示意图](assets/Pasted%20image%2020261016154333.png)

Δw必须比较小，即学习率较小

| 当前梯度dL/dw | 更新公式产生的动作           | 直观含义                 |
| --------- | ------------------- | -------------------- |
| 正数        | 减去正数，w减小,Δw<0,则ΔL<0 | 当前附近，减小 $w$ 能让损失下降 |
| 负数        | 减去负数，w增大            | 当前附近，增大 $w$ 能让损失下降 |

计算梯度与更新参数是**两个**步骤

更新后需要**重新计算**预测和损失L

**w迭代流程** ：当前w -> 当前梯度dL(w)/dw -> Δw=-α · (dL/dw) -> 新w -> 重新算L

**改变学习率α，不会直接改变L或当前梯度，而是先影响Δw**

在当前梯度非零时，沿负梯度方向走足够小的一步，可以降低损失；学习率过大则可能使损失增加。**跨过最低点本身不代表更新失败！可以跨！**，只是要比较更新前后的损失。

![示意图](assets/Pasted%20image%2020261016194729.png)

# pytorch自动求导

**前向传播forward pass**

从参数和输入，计算预测y_pred和损失Loss：预测得咋样？

**反向传播**

![示意图](assets/Pasted%20image%2020261016204948.png)

从损失L开始，沿计算关系往回求导：损失对前面的参数有多敏感？

**Code**

`requires_grad=True`要求 PyTorch 记录涉及它的相关运算，以便之后计算梯度

`w.requires_grad`查看这个张量是否需要追踪梯度

`loss.backward()`从损失开始反向计算，并把参数梯度累积到.grad

`w.grad`保存损失对参数w的梯度（None是尚未保存梯度而不是梯度为0的意思）

**对于两个参数w,b**

```python
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
prediction = w * 2 + b
loss = (prediction - 8) ** 2
```

**`loss.backward()`** **!!!把w和b两个参数的梯度分别存入w.grad和b.grad**

**PS：**自动求导不等于参数更新，暂不涉及学习率。

# 优化器与训练循环

`optimizer = torch.optim.SGD([w, b], lr=0.1)` **创建优化器**

`optimizer.zero_grad()` 清除已有的梯度(因为pytorch默认把新算出来的梯度加到已有的梯度上而非自动覆盖)

`optimizer.step()` 优化器工作，更新参数

**单次流程**

![示意图|304](assets/Pasted%20image%2020261017000033.png)

并存入w.grad，b.grad

`optimizer.step()`更新参数

**循环训练**

参数与优化器在循环外创建；

重复执行“清除梯度->前向算pred和loss->反向backward->参数更新”

**多样本时**

1.输入x,y -> 2.设置w,b -> 3.创建优化器 -> 4.计算预测值（多个）、MSE（损失求平均）

![示意图](assets/Pasted%20image%2020261018165229.png)

-> 5.开始循环更新参数

其中，5.的每个循环内

`optimizer.zero_grad()               `# 清除旧梯度

`prediction = w * x + b              `# 计算预测

`loss = ((prediction - y) ** 2).mean()` # 计算**平均**损失

`loss.backward()                     `# 计算梯度，存入 w.grad、b.grad

`optimizer.step()                    `# 读取梯度，结合**学习率**更新参数，即

![示意图](assets/Pasted%20image%2020261018174523.png)

**PS**

损失取**平均**，且对平均损失**执行backward()**

循环的每次更新都使用**全部样本**

**可视化**

```python
x_values = x.tolist()
y_values = y.tolist()
prediction_values=final_prediction.tolist()
```

**tolist()**把张量中的数值取出来转换成普通python列表

# 图片与标签

**加载**

| 写法                               | 作用                                                                  |
| -------------------------------- | ------------------------------------------------------------------- |
| `datasets.FashionMNIST(...)`     | 创建一个用于访问 FashionMNIST 数据的对象                                         |
| `root=...`                       | 指定数据存放的根目录，与 Conda 环境位置无关                                           |
| `train=True`                     | 选择官方提供的训练部分                                                         |
| `download=True`                  | 数据不存在时尝试下载                                                          |
| `transform=ToTensor()`           | 每次取出图片时，把图片转换成张量                                                    |
| `len(training_data)`             | 查询这个数据集中有多少个样本                                                      |
| `vmin=0, vmax=1`                 | 把 0 对应为黑色、1 对应为白色                                                   |
| `image, label =training_data[0]` | **解包赋值**，取出第一个样本的图片和标签                                              |
| `training_data.classes[label]`   | classes是training_data的属性，保存了**类别名称列表**，指的是['T-shirt','Trouser',...] |

`image.shape()` ---> `torch.Size([1, 28, 28])` :  一个灰度通道表示每个像素只需要1个数记录亮暗

`plt.imshow(image[0], cmap="gray", vmin=0, vmax=1)`image[0]表示取出第一个维度中的第0项，保留这一项里面的内容，而每个通道里都存着28* 28的像素表，所以取出的结果的**形状**是[28,28]

**batch迭代**

`train_loader = DataLoader(training_data,batch_size=4,shuffle=True)`

`images, labels = next(iter(train_loader))` ：iter()是创建迭代器；next()是取出下一个批次

# 全连接神经网络

**flatten**

把每个样本展开，表示为一串输入特征，但在批次内部保留样本间的分隔。

`flat_images = images.reshape(batch_size, 784)` 重新组织形状为[B , 784]

**全连接层 fully connected layer**

每个输出都使用全部输入，且每个输出都有自己的一套权重和偏置。

全连接层改变特征数量，保留样本数量。

· `small_layer = nn.Linear(in_features=3, out_features=2)` **创建层**，准备好参数；这里的 `nn` 是 PyTorch 提供神经网络组件的模块，`nn.Linear` 就是其中的全连接层。  
· `output = small_layer(sample)` **使用层**，像调用函数一样按照当前参数和输入计算输出  

输入`[[1.0,2.0,3.0]]` （2个方括号！）表示一个样本

每个样本3个特征，其形状为[1,3]，经过使用层后形状变为[1,2]

即特征数从in_features 变成 out_features 。

![示意图](assets/Pasted%20image%2020261018204820.png)

以此为例，每一个输出需要in_features个权重。

所以**权重的形状**是[out_features , in_features]。

**非线性激活函数：ReLU**

**Q：为什么不能单纯叠加全连接层？** 无法表达更复杂的**非线性关系**。

$ReLU(z) = max(0 , z)$

**MLP多层感知机**

· `__init__()` : **准备层**创建并保存模型需要的层  
· `forward()`   : **使用层**规定输入依次怎样经过这些层，最后返回什么  

全连接层+relu组装成小型神经网络

![示意图](assets/Pasted%20image%2020261018215215.png)

**流程**：

· `model=SmallMLP()`创建具体模型，执行`__init__()`准备好各层及参数  
· `outputs=model(images)` 把图片传入模型，执行forward()里的计算，返回结果  

| 组件     | 怎样计算           | 在当前网络中的作用          |
| ------ | -------------- | ------------------ |
| `fc1`  | 全连接计算：784 → 32 | 产生隐藏层特征            |
| `relu` | 将负数置零          | 对隐藏层特征进行激活         |
| `fc2`  | 全连接计算：32 → 10  | 输出层，产生最终的 10 个类别分数 |

![示意图](assets/Pasted%20image%2020261021204026.png)

784个输入--->**隐藏层hidden layer（16个中间特征，经ReLU激活）**--->10个输出

以上只是前向计算，还没有训练。创建层时参数随机初始化，因此输出暂时没有可靠的分类意义。

# 分类输出与损失

**原始分数logits**

未经softmax转换的原始类别分数。e.g. MLP输出[B , 10]，10个数字就是logits，可正可负可0。并非概率**probabilities概率**

`logits.argmax()` 找出最大logits对应的类别，返回的是**索引**，即类别编号。

**softmax**

把同一个样本的类别分数转换为概率，**形状不变**

**计算公式**

![示意图](assets/Pasted%20image%2020261020154206.png)

每个e_logits除以总和。 e.g.

![示意图](assets/Pasted%20image%2020261020154232.png)

每个样本内概率和为1

`example_probs = torch.softmax(example_logits, dim=1)`

`example_probs.sum(dim=1)`-->返回1

**Q : 对形状为 `[4,10]` 的 logits，执行 `argmax(dim=1)` 后，形状是？** [4]。变成一维了。

**交叉熵损失cross-entropy loss**

1. 基础计算公式：

![示意图](assets/Pasted%20image%2020261020160700.png)

（！！此处p_y指模型落在真实类别上的概率，不一定是它选的最大的概率）。模型给真实类别的概率越高，L越小。多个样本时取平均值

2. 创建计算工具： `loss_fn=nn.CrossEntropyLoss()` 创建交叉熵损失计算工具

其中

![示意图](assets/Pasted%20image%2020261023141330.png)

默认取一个批次里的平均。避免仅仅因为批次里的样本更多，损失和梯度的规模就随之成倍增加。

3. code

`loss=loss_fn(example_logits,examples_labels)` 输入logits和标签去算`nn.CrossEntropyLoss(reduction=='sum')` 求和`nn.CrossEntropyLoss(reduction=='none')`  保留每个样本的损失

**不用先手动做softmax！CrossEntropyLoss已经包含了**

**准确率accuracy**

准确率关注最终类别是否正确；交叉熵还反映模型给真实类别分配的概率。

`labels.shape[0]` 指样本数。因为labels的形状是[B]

**PS**：  torch.long 是 PyTorch 的一种**整数数据类型**，也叫 torch.int64，用 64 位存储整数。把真实类别标签[0,2]保存为整数张量

# 训练

#### 一个批次

`optimizer=torch.optim.SGD(model.parameters(),lr=0.01)` model.parameters()把所有参数交给优化器管理

`loss.backward()` 只是计算梯度；`optimizer.step()` 才是根据梯度和学习率**更新**

输出层10个logits对应10个类别，10套参数。所以**偏置数量等于输出特征数**

|`fc2` 的参数|数量或形状|原因|
|---|---|---|
|权重 `fc2.weight`|`[10,32]`|10 个输出，每个使用 32 个权重|
|偏置 `fc2.bias`|`[10]`|10 个输出，每个使用 1 个偏置|

在fc2层，如何把32个特征变为10个？

![示意图](assets/Pasted%20image%2020261021204553.png)

**PS** 交叉熵损失减小不意味着准确率增加。交叉熵看“模型给真实类别的概率多大”；准确率只看最高分对应的类别对不对。

#### 完整训练

1个**epoch**（训练轮次，遍历训练集）=n个**iteration**（处理一个批次并更新一次）

`total_loss += loss.item() * current_batch_size` 因为crossentropyloss默认给出当前批次的**平均损失**，而不同批次的样本数不同，因此要乘size。

`def train_one_epoch(model, data_loader, loss_fn, optimizer)` 封装一轮训练，接收四个参数，返回本轮的**平均**损失与准确率。

画图时，根据loss_history和accuracy_history来画

参数在批次之间持续更新，不是重新创建模型。

训练**过程**记录的指标，要与**最终模型**重新评估的指标进行区分。

外层循环控制epoch，内层循环处理batch。

函数内统计量每轮重置，函数外列表用 `.append()` 保存各轮结果。

# 验证与测试validation and testing

**数据集类型**

| 数据                | 作用                          | 是否用于反向传播和参数更新 |
| ----------------- | --------------------------- | ------------- |
| 训练集training set   | 让模型**学习参数**                 | 是             |
| 验证集validation set | 训练期间**检查表现**，选择**超参数或模型版本** | 否             |
| 测试集test set       | **模型选择完成后**，**评估最终**表现      | 否             |

**code**

`from torch.utils.data import random_split` 随机划分数据，得到互不重叠的子集。

![示意图](assets/Pasted%20image%2020261024123617.png)

generator随机数生成器；

manual_seed42无特殊含义，相同条件下用相同种子重新开始，重复得到相同的随机结果，方便比较实验；

`train=False` 表示加载官方测试部分。已有完整文件时使用本地数据；若缺少测试文件，会尝试下载。

**评估evaluation：验证+测试**

`model.train()` 和 `model.eval()` 设置层的运行模式，本身不执行训练或评估。

`with torch.no_grad()`

`no_grad` 暂时**关闭反向求导记录**，但仍然计算预测和损失。↓

| 代码情况                   | 是否记录求导信息  | 是否计算梯度  |
| ---------------------- | --------- | ------- |
| 正常前向，但不调用 `backward()` | **通常会记录** | 不会      |
| 正常前向，再调用 `backward()`  | 会记录       | 会       |
| 在 `no_grad()` 内前向      | **不记录**   | 不执行反向求导 |

**PS**

1. 数据集要先划分再训练
2. 验证集和测试集通常不打乱顺序, shuffle=False。
3. 验证和测试都使用evaluation()函数

# 保存、重新加载、独立推理

· 模型结构architecture ：规定怎样计算  
· 模型参数parameters ： 决定使用哪些权重与偏置  

**保存**

| 写法                                   | 作用                          |
| ------------------------------------ | --------------------------- |
| `save_path.parent`                   | 文件所在的文件夹，即 `checkpoints`    |
| `mkdir(parents=True, exist_ok=True)` | 创建所需目录；目录已存在也可以继续           |
| `model.state_dict()`                 | 取得模型的**状态字典**，本模型中包含各层权重与偏置 |
| `torch.save(..., save_path)`         | 把这些**数值写入文件**               |
| `.exists()`                          | 检查文件是否存在                    |

**加载**

1. 加载时先**创建相同的模型结构** :未接入字典参数文件之前，模型里的参数是还没训练的
2. `save_weights=torch.load(...)` 从文件中**读取参数**字典：是之前**训练过**的参数
3. `loaded_model.load_state_dict(saved_weights)` 把字典**参数装进新模型**，替换随机初始化的数值

不需要重新训练，但要使用相同的输入处理方式，eval()和no_grad()

**推理inference**

不依赖之前的训练过程，只用模型结构，保存的参数和待预测图片，计算结果。不用优化器和backward()。

`state_dict()` 把模型中的参数等状态单独列出来，并不保存 `forward()`、预处理代码或评估模式，需要另行提供。

加载时参数名称和形状要匹配。

预测时真实标签仅用于检查结果。

# 基础实验与错误分析

超参数hyperparameter ：learning_rate, batch_size....

`def run_experiment(learning_rate,num_epoch=5)`

![示意图](assets/Pasted%20image%2020261024182247.png)

基础对比实验一次只改变**一个**超参数，其他条件尽量保持一致。

比较学习率时，应从**相同初始参数**出发，避免把之前的训练成果混入比较。

根据**验证集比较**实验，暂不使用测试集调整选择。

实验结论要**限定**在**当前**模型、数据、训练轮数和随机条件下。

# 完整实验梳理！！！

**数据保留/清零**

一次实验中，**参数**跨batch&epoch持续保留；**统计数据**按需要清零。

| 何时清零             |             清什么             |             为什么             |
| -------------------- |:------------------------------:|:------------------------------:|
| 每个批次开始前       |             旧梯度             | 本次只使用当前批次计算出的梯度 |
| 每个 epoch 开始时    |  本轮损失总和、正确数、样本数  |         单独统计这一轮         |
| 一个新对比实验开始时 | 新建模型、优化器，清空历史列表 |       从独立起点开始实验       |

## 一、数据准备

**加载并划分数据**（ToTensor转成张量）

1. 加载full_train_data，random_split划分成训练集train_data和验证集val_data；
2. 加载test_data
3. 三个数据集分别创建DataLoader对象,以便`for images, labels in train_loader`
    1. `train_loader=DataLoader(train_data,batch_size=64,shuffle=True)`
    2. `val_loader=DataLoader(...shuffle=False)`
    3. `test_loader=DataLoader(...shuffle=False)`

**形状**

images : [B,1,28,28] ; labels : [B]

`images.shape[0]`表示batch_size

## 二、实验准备

**定义模型**

SmallMLP ： 图片、展平、Linear(784,32)、ReLU、Linear(32,10)

`from torch import nn`

`model=SmallMLP()`

**损失函数**

`loss_fn=nn.CrossEntropyLoss()`

**优化器**

`optimizer=torch.optim.SGD(model.parameters(),lr=0.01)`

## 三、训练函数

**一个batch**

`optimizer.zero_grad()`

`logits=model(images)`

`loss=loss_fn(logits,labels)` 要根据真实label和模型logits来算损失。

`loss.backward()`

`optimizer.step()` 用梯度实际修改参数。

**一个epoch** ：训练+统计

自定义train_one_epoch

```python
def train_one_epoch(model, data_loader, loss_fn, optimizer):
    model.train()  # 切换训练模式
    # 统计数据初始化
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # batch内循环
    for images, labels in data_loader: 
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        # 统计数据累加
        total_loss += loss.item() * batch_size  #loss是平均值所以乘batch_size!!
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return (total_loss / total_samples,total_correct / total_samples)
```

## 四、评估函数

自定义evaluate函数

```python
def evaluate(model, data_loader, loss_fn):
    model.eval()  # 评估模式

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.shape[0]

            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return (total_loss / total_samples,total_correct / total_samples)
```

与train_one_epoch相比：model.eval()切换模式；没有backward（）和optimizer.step()更新参数

## 五、多轮训练与验证

组合四、五部分

1. 初始化

```python
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
```

2. 循环并累加指标

```python
for epoch in range(5):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer)

    val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
```

## 六、模型选择与测试

1. 验证集选择版本

**循环开始前**设置`best_val_loss = float("inf")` 正无穷。后续用最小的loss覆盖

**每轮验证后**执行⬇

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_path)
```

2. 恢复选中的参数

`model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))`

3. 测试集最终检查

`test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)`

## 七、保存与加载

`torch.save(model.state_dict(), save_path)`保存模型状态（参数fc1.weight等），未保存计算代码

重启kernel后

```python
loaded_model = SmallMLP()
saved_weights = torch.load(save_path,map_location="cpu",weights_only=True)
loaded_model.load_state_dict(saved_weights) 
loaded_model.eval()
```

| 代码区分                   | 作用          |
| ---------------------- | ----------- |
| `torch.load(...)`      | 从文件读出参数字典   |
| `load_state_dict(...)` | 把字典中的参数装入模型 |

## 八、独立推理

输入图片。输出预测。真实标签只用于检查对比预测标签

```python
image, true_label = test_data[12]
input_batch = image.reshape(1, 1, 28, 28)
loaded_model.eval()

with torch.no_grad():
    logits = loaded_model(input_batch)
    predicted_label = logits.argmax(dim=1).item()
```

此处注意输入图片的预处理方式需与训练时一致。

# 九、对比实验与错误分析

控制条件只改变学习率，重复完整训练流程。

**错误图片**

`logits=model(images)`

`predictions=logits.argmax(dim=1)`

`wrong_mask=predictions!=labels`

**分析指标**

|现象|能说明什么|
|---|---|
|训练、验证损失整体下降|两部分数据上的表现都在改善|
|训练持续改善，验证连续恶化|提示可能过拟合|
|验证只有一轮明显退步|证据不足，可能存在更新波动|
|损失下降、准确率不变|真实类别概率可能改善，但最终预测编号没变|

# 十、提纲

```text
数据准备
    加载原始数据
    划分训练、验证、测试
    分别创建DataLoader

定义工具
    模型类SmallMLP
    train_one_epoch函数
    evaluate函数

开始一次实验
    创建模型
    创建损失函数和优化器
    创建历史记录

重复多个epoch
    训练一轮，更新参数
    验证一次，不更新参数
    记录指标
    必要时保存验证表现最好的参数

实验结束
    根据验证结果确定模型
    加载选中的参数
    最终测试

部署到新的运行会话
    重建相同结构
    加载参数
    输入经过相同预处理的图片
    独立预测
```
