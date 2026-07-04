我们没法像实现各种传统的算法（比如冒泡算法、排序算法）那样，通过定义一个函数就能做到实现图像分类。即，我们没法靠硬编码来给图像分类。

因此，我们需要从数据驱动的视角来看待这个问题。设计以下三个步骤：

1. Collect a dataset of images and labels.
2. Use Machine Learning algorithm to train a classifier.
    - 本质上就是定义一个函数，这个函数接收训练数据中的图像以及对应的标签，构建一个能把图像和标签关联起来的模型
    - ![](附件/Pasted%20image%2020260704212153.png)
3. Evaluate the classifier on new images.
    - 要实现一个predict函数，它接收模型还有不属于训练图像的测试图像，让模型去预测这个标签，并将结果作为输出返回
    - ![](附件/Pasted%20image%2020260704212806.png)
## 1. Nearest Neighbour Classifier

### 1.1 Nearest Neighbour Classifier的原理
要构建最近邻分类器，要先准备好数据和预测函数。

训练函数(train)就是把所有数据和标签都记下来，所以训练函数基本就是把所有东西都存在内存里，基本上没干别的。

接着，预测函数会去找最像的那张图片。

说白了，train函数就是建立了一个表格，把所有的数据和对应的标签都列出来。到了预测和测试的时候，predict函数会试着找出最接近的那张图，然后把那张图的标签输出出来。

![](附件/Pasted%20image%2020260704213356.png)

### 1.2 实现方式

#### 方法一：
![](附件/Pasted%20image%2020260704213621.png)
#### 具体实现形式：
```Python
import numpy as np

class NearestNeighbour:
    def __init__(self): # 类的初始化函数
        pass
        
    def train(self, X, y): # X是训练数据，y是数据的类别
        """X is N*D where each row is an example. y is 1 dimension of size N"""
        # the nearest neighbour classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        """X is N*D where each row is an example we wish to predict label for"""
        num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
    
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance: sum of absolute value differences
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)

            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
    
        return Ypred
```

!!! explanation "上面那段代码的详解"
    1. 首先定义了`NearestNeighbour`这个类，这个类里面包含`train()`和`predict()`这两个函数
        - 之后我们可以创建一个`nn = NearestNeighbour()`
        - `__init__(self)`是类的初始化函数，每当执行类似`nn = NearestNeighbour()`的语句时，Python会自动调用这个初始化函数
        - 初始化函数里面只写了个`pass`说明这个分类器初始化时什么都不用干
    2. 定义训练函数`train()`，其中输入的X表示好几张图片，y表示每张图片的类别，例如`X[0]`是第一张图片，`y[0]`是第一张图片的标签。
	    - X是一个N行D列的矩阵，N表示样本的总数，D表示每个样本的维度。如果有5000张图，那么N就是5000。X的每一行都是一个样本，例如`X[0, :]`就是第一张训练的图片，`X[1, :]`就是第二张训练的图片。
        - y是一个一维数组，长度为N。每一个 `y[i]` 都是第 `i` 张图片的标签。
        - `self.Xtr = X`这句话把训练数据`X`存到对象自己的变量`self.Xtr`里面。`nn.train(X_train, y_train)`调用后，分类器对象 `nn` 里面就保存了`nn.Xtr = X_train`
    3. 定义预测函数`predict()`，其中输入的X表示测试的样本。`X.shape` 表示数组的形状，如果`X.shape = (10000, 3072)`，那么`X.shape[0] = 10000, X.shape[1] = 3072`
        - 所以`num_test = X.shape[0]`意思是测试样本的数量
        - `Ypred = np.zeros(num_test, dtype=self.ytr.dtype)`表示创建一个数组，用来保存每一张测试图片的预测标签。
            - `np.zeros(num_test)`表示创建一个长度为`num_test`的数组，里面全是0
            - `dtype=self.ytr.dtype`意思是让`Ypred` 的数据类型和训练标签 `self.ytr` 的数据类型一致。
        - `distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)`最后得到的是一个一维数组，它的长度等于训练图片数量。
            - `self.Xtr - X[i, :]`是把当前测试图片和所有训练图片相减。
            - numpy 会自动把 `X[i, :]` 和 `self.Xtr` 的每一行相减。`np.abs(self.Xtr - X[i, :])`表示取绝对值。
            - `distances[0]`表示当前测试图片和第0张训练图片的距离，`distances[1]`表示当前测试图片和第1张训练图片之间的距离
        - `min_index = np.argmin(distances)`表示找到距离最小的训练样本下标。
        - `Ypred[i] = self.ytr[min_index]`表示当前测试图片的预测标签 = 距离它最近的训练图片的标签

#### 方法二：KNN（K Nearest Neighbour）

方法一有一个问题，即只寻找一个最近的邻居，有可能会被单个奇怪的点影响。

下图的这些点就是训练数据，这6个区域就是通过最近邻算法算出来的几个区域。如果某一个测试样本落在了某个区域内，就表示这个区域内的某个训练数据是这个测试数据的最近邻居。比如某块区域背景是蓝色，那么一个测试点落进去，就会被预测成蓝色类。
![](附件/Pasted%20image%2020260704234651.png)

但是用K=1（即只找一个最近邻居的话），可能会导致背景颜色出现很多杂碎的区域，就像这幅图中绿色的区域内有一个很小的黄色的区域。这个黄色的点可能只是一个偶然的异常点或者噪声点。但是K=1会非常相信它，因为只要某个测试数据算出来离这个异常点的距离比别的距离都近（也就是这个测试数据落在了中间那个黄色的小区域内），就会将这个测试数据输出为黄色（但是它大概率是绿色的）。

为了解决这个问题，我们引入了KNN，也就是寻找K个最近邻居。比如K=3，就是预测的时候不再只看最近的哪一个点，而是看最近的三个点，综合三个点分析，来确定这个测试数据到底属于哪个类。比如某个测试点的3个最近邻居分别为{绿色，绿色，黄色}，那么这个时候黄色很有可能就是一个异常点，我们将这个测试数据归类为绿色。

![](附件/Pasted%20image%2020260704235605.png)

### 1.3 计算距离的方法
下面两幅图定义了两种计算距离的方法
![](附件/Pasted%20image%2020260704235803.png)

1. **L1：曼哈顿距离**
2. **L2：欧几里得距离**

!!! example "L1 和 L2 的区别"
    L1 是：
    
    ```
    |差值1| + |差值2| + |差值3| + ...
    ```

    L2 是：

    ```
    sqrt(差值1^2 + 差值2^2 + 差值3^2 + ...)
    ```

    假设有两个差值向量：

    ```
    A = [5, 0], B = [3, 4]
    ```

    它们的 L1 距离：

    ```
    A: |5-0| + |0-0| = 5, B: |3-0| + |4-0| = 7
    ```

    所以按 L1 看，A 更近。

    它们的 L2 距离：

    ```
    A: sqrt((5-0)^2 + (0-0)^2) = 5, B: sqrt((3-0)^2 + (4-0)^2) = 5
    ```
    
    所以按 L2 看，A 和 B 一样远。

**L2 距离只关心两个点之间的“直线距离”，不太在乎坐标轴怎么转。**  

**L1 距离很在乎坐标轴的方向，所以特征轴本身有意义时，L1 更符合“每个特征单独比较”的想法。**

假设点 $A = (1, 0)$，那么它到原点的 `L1` 距离为 $d1 = |1 - 0| + |0 - 0| = 1$，L2 距离为 $d2 = \sqrt{(1 - 0)^2 + (0 - 0)^2} = 1$。现在我们把这个点绕原点旋转 $45°$，即$A' = (\frac{\sqrt2}{2}, \frac{\sqrt2}{2})$。计算一下就会知道，旋转之后，L1距离改变了，但是L2距离没变。

这说明了L1对坐标轴的方向很敏感。在特征值非常有意义的情况下（每一个坐标维度本身都有清楚的含义），比如一个人的数据是`x = [身高, 体重]`时，用L1会比L2作为计算距离的函数会好很多。
![](附件/Pasted%20image%2020260705001626.png)

---
## 2. Linear Classifier

### 2.1 线性分类器的原理

我们需要找到找到一些参数W或者权重，把输入图像映射到输出数值上。

以下图为例，已知有10个不同的输出类别，那我们就需要输出一个 $10 \times 1$ 的向量。因为 $X$ 是一个 $3072 \times 1$ 的矩阵，所以 $W$ 为 $10 \times 3072$ 的矩阵，其中 $b$ 为偏置项。
![](附件/Pasted%20image%2020260705003813.png)

!!! example "以一个只有4个像素值的图片作为例子"
    输入的图像只有4个像素值，将其转换为一个 $4\times 1$ 的列向量。数据集中一共有3个类，因此 $W$ 是一个 $3 \times 4$ 的矩阵。
    ![](附件/Pasted%20image%2020260705004018.png)


### 2.2 Linear Classifier - Choose a good $W$

#### 2.2.1 调整权重 $W$ 的方式

1. 定义一个损失函数（目标函数），用来衡量这个分类器的表现到底有多差。
2. 定义好损失函数之后，还要找一个办法来高效的调整 $W$ 的值，来最小化损失函数。

!!! example "举个例子"
    ![](附件/Pasted%20image%2020260705012149.png)
#### 2.2.2 Softmax Classifier (Multinomial Logistic Regression)

$$s=f(x_i\ ;\ W)$$

由于 $f(x_i\ ;\ W)$ 的大小没有上下界，数值往往很难控制（如下图所示，输出的frog为-1.7）.为了将这个线性函数转化成评分函数，最好的方法就是从概率出发，计算出属于某个类别 $K$ 的概率。

$$Softmax\ Function:\ P(Y = k|X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$$

虽然用到了概率。但是我们发现，这个最终的结果并不是正确的结果，因为这幅图是一只猫，但是这个分类器计算出来猫的概率仅为13%，而车的概率为87%。此时我们需要==用损失函数L来调整权重==。此处我们用==极大似然估计==来计算，其实就是==将正确类别的概率取对数之后再取负值==。

$$L_i = -\log P(Y = y_i|X = x_i)$$
![](附件/Pasted%20image%2020260705012121.png)


### 2.3 线性分类器的缺点

下图的三种情况，都没法用一条线将两个类分开，这时就没法用线性分类

![](附件/Pasted%20image%2020260705005202.png)

---

## 3. 两个课后问答

![](附件/Pasted%20image%2020260705012645.png)

### 3.1 Q1
对第 $i$ 个样本，假设真实类别是 $y_i$​，每一类的 score 是：

$$s1,s2,…,sC$$

$softmax$ 把 $score$ 变成概率：

$$p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$​​

正确类别的概率就是：

$$p_{y_i} = \frac{e^{s_{y_i}}}{\sum_k e^{s_k}}$$​​​

$softmax\  loss$ 是：

$$L_i = -\log p_{y_i}$$​​

也就是：

$$L_i = -\log \left( \frac{e^{s_{y_i}}}{\sum_k e^{s_k}} \right)$$

因为 

$$p_i\in(0,\ 1)$$

所以

$$L_i = -\log p_{y_i}\in(0,\ +∞)$$

---

## 3.2 Q2

题目说：At initialization all $s_j$​ will be approximately equal. 即模型刚初始化的时候，所有类别的分数差不多一样。


比如有 $C$ 个类别：

$$s1≈s2≈⋯≈s_C$$

假设它们完全相等：

$$s1=s2=⋯=s_C$$


因为所有类别 $score$ 一样，所以每个类别的概率一样。一共有 C 类，所以每一类概率都是：

$$p_j = \frac{1}{C}$$

正确类别的概率也是：

$$p_j = \frac{1}{C}$$

所以 loss 是：

$$L_i = -\log p_{y_i}=-\log\frac{1}{C}=\log C$$​​
#### 以 $CIFAR-10为例$
$CIFAR-10$ 有 10 个类别，所以 softmax loss 大概是：

$$L_i = -\log 0.1$$

如果用自然对数，则：

$$ln10≈2.302$$
