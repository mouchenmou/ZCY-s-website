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

### 1.4 确定最优超参数的方法
比如 KNN 里面的：

$$K=1,3,5,7$$

到底选哪个？  

还有距离函数选 L1 还是 L2。这些都叫**超参数**。

#### 1.4.1 Choose hyper parameters that works best on the ==trainning data==
方法一的原理：==在训练集上选表现最好的超参数==。也就是说，哪个 $K$ 在训练集上的准确度最高，就选哪个。这个是大错特错的方法，因为你拿的是训练集中的数据，而不是测试集中的数据，因此 $K=1$ 的时候，每一个测试集中的图片都能精准找到自身，标签当然也是一模一样，因此准确率永远是100%。

但是这不代表在新的数据上也能完美表现，这个前面也说过了，不然我们为什么要引入 $K$ 呢。

#### 1.4.2 Choose hyper parameters that works best on ==test data==
方法二的原理：把数据分成==训练数据==和==测试数据==，然后试不同的 $K$ ，看哪个在 $test$ 上的效果最好，就选哪个。这个方法看起来很合理，但其实问题也很大。原因是：==测试集==应该==只用来做最终评估==，而==不能拿来调参数==。

可以把训练集理解成平时的练习题，测试集理解成期末考试。如果每次都拿期末考试的卷子来当测试集，那就不再是一个公平的评估了，因为你已经提前看过了测试集，并通过这些测试集调出了一个超参数。因此测试集在这种方法下，==不再代表真正没见过的新数据==，所以这种方法不可取。

#### 1.4.3 Split data into train, val; choose hyper parameters on val and evaluate on test

方法三的原理：图里把数据分成三部分：
```
train / validation / test
```

作用分别是：

1. train：训练集
    - 用来训练模型。
    - 对于 KNN 来说，就是把训练图片和标签记下来。
2. validation：验证集
    - 用来选择超参数。
    - 比如可以将训练集用 `K=1,3,5,7,9` 分别做一次，然后在 validation set 上测试准确率。取准确率最高的那个 $K$ 作为我们想要的 $K$。
3. test：测试集
    - 等我们已经确定好 $K$ 以及其它的超参数之后，再拿 test set 测试一次，作为最终结果
    - 所以 test set的作用是模拟真正的新数据，给模型做最终考试。

所以方法三的本质是，将训练数据一分为二，一部分用来训练，一部分用来评估，并选出最优的超参数。选定好最优的超参数之后，再用真正的测试数据做最终评估。

#### 1.4.4 Cross-Validation: Split data into folds, try each fold as validation and average the results

$Cross-validation$ 即**交叉验证**

方法四的原理：假如我们没有很多数据，没办法单独拿出一大块数据做validation的话。我们就进行多次训练。假设训练数据中只有5个数据，$\{fold1，fold2，fold3，fold4，fold5\}$。那么做**5-fold cross-validation**，那就相当于每次拿 1 个做 validation，剩下 4 个做 train。
![](附件/Pasted%20image%2020260705231746.png)

这种方法适用于小数据集，但是在深度学习里面不太常用。因为深度学习训练一次就特别贵，多次训练的开销会特别大。

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

#### 2.2.2 SVM Classifier (Multiclass SVM Loss)

SVM 的全称是 **Support Vector Machine**。它不是直接输出概率，而是希望：

> 正确类别的分数不仅要比错误类别高，而且要至少高出一个安全间隔（margin）。

假设第 $i$ 个样本是 $x_i$，真实类别是 $y_i$，线性分类器输出每一类的分数：

$$s=f(x_i;W)$$

其中：

$$s_j=f(x_i;W)_j$$

表示第 $j$ 个类别的分数，而正确类别的分数是：

$$s_{y_i}=f(x_i;W)_{y_i}$$

##### 2.2.2.1 Multiclass SVM Loss 的公式

对单个样本，SVM loss 定义为：

$$L_i=\sum_{j\neq y_i}\max(0,\ s_j-s_{y_i}+\Delta)$$

其中 $\Delta$ 是 margin，通常取：

$$\Delta=1$$

所以也可以写成：

$$L_i=\sum_{j\neq y_i}\max(0,\ s_j-s_{y_i}+1)$$

这个公式的意思是：对每一个错误类别 $j$，都检查它的分数 $s_j$ 有没有离正确类别分数 $s_{y_i}$ 足够远。

如果：

$$s_{y_i}\ge s_j+\Delta$$

说明正确类别已经比这个错误类别至少高出 margin，这个错误类别就不会产生损失：

$$\max(0,\ s_j-s_{y_i}+\Delta)=0$$

如果：

$$s_{y_i}<s_j+\Delta$$

说明正确类别虽然可能比错误类别高，但是高得还不够；或者干脆比错误类别低。此时就会产生损失：

$$s_j-s_{y_i}+\Delta>0$$

也就是说，SVM 不满足于“正确类别分数最高”，它还要求“正确类别分数要明显高”。

##### 2.2.2.2 举个例子

假设有 3 个类别：cat、car、frog。某张图片真实类别是 cat，分类器输出的分数是：

$$s=[13,\ -7,\ 11]$$

其中：

$$s_{cat}=13,\quad s_{car}=-7,\quad s_{frog}=11$$

取 $\Delta=10$。

对于 car：

$$\max(0,\ s_{car}-s_{cat}+10)=\max(0,\ -7-13+10)=0$$

因为 cat 的分数已经比 car 高很多，所以 car 不产生损失。

对于 frog：

$$\max(0,\ s_{frog}-s_{cat}+10)=\max(0,\ 11-13+10)=8$$

虽然 cat 的分数 $13$ 比 frog 的分数 $11$ 高，但是只高了 $2$，没有达到 margin $10$，所以仍然会产生损失。

因此这个样本的 SVM loss 是：

$$L_i=0+8=8$$

##### 2.2.2.3 对整个训练集的 Loss

对整个训练集，我们会把所有样本的 loss 取平均：

$$L=\frac{1}{N}\sum_i L_i$$

实际训练时还会加上正则化项：

$$L=\frac{1}{N}\sum_i\sum_{j\neq y_i}\max(0,\ f(x_i;W)_j-f(x_i;W)_{y_i}+1)+\alpha R(W)$$

其中：

- 第一项是 data loss，用来衡量分类结果是否正确。
- 第二项是 regularization loss，用来防止 $W$ 变得过大，降低过拟合。
- $\alpha$ 控制正则化项的重要程度。

##### 2.2.2.4 为什么叫 Hinge Loss

SVM loss 里面最关键的部分是：

$$\max(0,\ s_j-s_{y_i}+\Delta)$$

这个函数长得像一个“折页”：

- 当 $s_j-s_{y_i}+\Delta\le 0$ 时，loss 为 0。
- 当 $s_j-s_{y_i}+\Delta>0$ 时，loss 按线性方式增加。

所以它也叫 **hinge loss**。

从图像上看，它不是一条光滑曲线，而是由几段直线拼起来的。即，SVM loss 具有 **piecewise-linear** 的结构，也就是分段线性结构。

这件事很重要，因为它说明：

1. SVM loss 大部分地方可以求梯度。
2. 在折点处严格来说不可导，因为 $\max(0,\cdot)$ 在拐角处没有唯一梯度。
3. 实际训练时，我们通常使用 **subgradient**，也就是次梯度。

在这门课里，很多时候会把 gradient 和 subgradient 混着说，因为对实现梯度下降来说，它们的作用很接近。

##### 2.2.2.5 SVM Loss 的梯度直觉

对某个样本 $x_i$，如果某个错误类别 $j$ 违反了 margin：

$$s_j-s_{y_i}+\Delta>0$$

那么这个错误类别对 loss 有贡献。为了降低 loss，我们希望：

- 降低错误类别 $j$ 的分数 $s_j$。
- 提高正确类别 $y_i$ 的分数 $s_{y_i}$。

因为：

$$s_j=w_j^Tx_i$$

所以如果错误类别 $j$ 违反了 margin，那么它对应的权重梯度是：

$$\nabla_{w_j}L_i=x_i$$

直观理解：梯度下降会让 $w_j$ 往远离 $x_i$ 的方向走，从而降低错误类别的分数。

对于正确类别 $y_i$，假设一共有 $k$ 个错误类别违反了 margin，那么：

$$\nabla_{w_{y_i}}L_i=-k x_i$$

直观理解：如果有很多错误类别都离正确类别太近，那么正确类别的权重就要更用力地朝 $x_i$ 的方向调整，让正确类别的分数升上去。

##### 2.2.2.6 SVM 和 Softmax 的区别

SVM 和 Softmax 都可以用来训练线性分类器，但它们看问题的角度不同：

| 方法 | 关注点 | 输出含义 | Loss 的想法 |
|---|---|---|---|
| SVM | margin | 分数，不是概率 | 正确类分数要比错误类至少高出 $\Delta$ |
| Softmax | probability | 概率分布 | 正确类概率越接近 1 越好 |

#### 2.2.3 Softmax Classifier (Multinomial Logistic Regression)

$$s=f(x_i\ ;\ W)$$

由于 $f(x_i\ ;\ W)$ 的大小没有上下界，数值往往很难控制（如下图所示，输出的frog为-1.7）.为了将这个线性函数转化成评分函数，最好的方法就是从概率出发，计算出属于某个类别 $K$ 的概率。

$$Softmax\ Function:\ P(Y = k|X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$$

虽然用到了概率。但是我们发现，这个最终的结果并不是正确的结果，因为这幅图是一只猫，但是这个分类器计算出来猫的概率仅为13%，而车的概率为87%。此时我们需要==用损失函数L来调整权重==。此处我们用==极大似然估计==来计算，其实就是==将正确类别的概率取对数之后再取负值==。

$$L_i = -\log P(Y = y_i|X = x_i)$$

![](附件/Pasted%20image%2020260705012121.png)

### 2.3 线性分类器的缺点

下图的三种情况，都没法用一条线将两个类分开，这时就没法用线性分类

![](附件/Pasted%20image%2020260705005202.png)

### 2.4 从三个视角理解 $f(x,W)=Wx+b$
![](附件/Pasted%20image%2020260705234147.png)
#### 2.4.1 Algebraic Viewpoint：代数角度
就是从公式的角度入手（其实上面都讲过了）：

$$f(x,W)=Wx+b$$

比如 $x$ 是 $3072 \times 1$ 的图片如果有 10 个类别，那么 $W$ 可以看成 $10\times 3072$ 的矩阵，$b$ 为 $10\times1$ 的矩阵。最终得到一个 $10\times 1$ 的矩阵，也就是 $10$ 个类别的分数。

也就是说，每个类别都有一套权重。输入图片和这一套权重做加权求和，得到这个类别的分数。

比如：

$$s_{cat}​=w_{cat}^T​x+b_{cat}​$$

$$s_{car}=w_{car}^Tx+b_{car}$$

最终得到的 $10\times1$ 的矩阵中哪个分数最大，模型就将那个类别作为预测的输出结果。

#### 2.4.2 Visual Viewpoint：视觉角度

##### 2.4.2.1 原理：W可以看成模糊的模版图

我们以 $CIFAR-10$ 为例，一张图片的规格为:

$$32\times32 \times 3$$

拉平成向量以后即为 $3072$。所以输入图片是 

$$x \in \mathbb{R}^{3072}$$

如果有 10 个类别，那么 $W$ 的形状是：

$$10 \times 3072$$

也就是说，$W$ 里面有 10 行：

$$W= \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_{10} \end{bmatrix}$$​​​

每一行 $w_j$​ 都负责一个类别。

比如：
```
w_airplane：判断像不像飞机
w_car：判断像不像车
w_cat：判断像不像猫
w_dog：判断像不像狗
```

对于第 $j$ 个类别，它的分数是：

$$s_j = w_j^T x + b_j$$​

也就是说，输入图片 $x$ 和类别模板 $w_j$​ 做匹配，匹配得越好，这个类别分数越高。

##### 2.4.2.2 One template per class. 
意思是：线性分类起给每个类别都学了一张模版图。

那么问题来了，模版是何意味？

因为 $w_j$ 也是 3072 维，而图片 $x$ 也是 3072 维。既然 $x$ 可以 reshape 成 $32\times32 \times 3$ 的规格，那么 $w_j$​ 也可以 reshape 成 $32 \times 32 \times 3$ 的规格。所以，每一个类别的权重  $w_j$​ 都可以当成一张图片来看。

比如一个模型训练完之后，学出了以下模版：

```
猫模板
狗模板
车模板
飞机模板
船模板
```

当输入图片和猫模板比较像时：

$$w_{\text{cat}}^T x$$

就比较大。

于是猫的 score 就比较高。

##### 2.4.2.3点积 $w_j^T x$ 怎么理解成“匹配”？

假设图片只有 3 个像素：

$$x = [x_1, x_2, x_3]$$

某个类别的模板是：

$$w = [w_1, w_2, w_3]$$

那么分数是：

$$w^T x = w_1x_1 + w_2x_2 + w_3x_3$$

如果某个位置的权重 $w_1$​ 是正的，而且图片那个位置 $x_1$​ 很亮，那么 $w_1x_1$ ​就会让这个类别的分数变大。说明：

> 这个位置亮起来，有利于判断成这个类别。

#### 2.4.3 Geometric ViewPoint：用超平面切空间
##### 2.4.3.1 原理

现在换一个视角，假设每个样本只有两个特征：

$$x = (x_1, x_2)$$

这样我们可以把每个样本画成二维平面上的一个点。当然真实图片是 3072 维，画不出来。但二维情况比较容易理解。

二分类时，线性分类器就是画一条线

假设只有猫和狗这两个类别


线性分类器会算两个分数：

$$s_{cat}=w_{cat}^Tx+b_{cat}$$
$$s_{\text{dog}} = w_{\text{cat}}^T x + b_{dog}$$


如果：

$$s_{\text{cat}} > s_{\text{dog}}$$​

就预测猫。

如果：

$$s_{\text{dog}} > s_{\text{cat}}$$​

就预测狗。

猫和狗的分界线就是两个分数相等的地方，即：

$$s_{\text{cat}} = s_{\text{dog}}$$​

代入：

$$w_{\text{cat}}^T x + b_{\text{cat}} = w_{\text{dog}}^T x + b_{\text{dog}}$$​

整理得：

$$(w_{\text{cat}}-w_{\text{dog}})^T x + (b_{\text{cat}}-b_{\text{dog}}) =0$$

这是一个线性方程。

在二维平面里，它就是一条直线。

所以二分类时，线性分类器的几何意义就是：

画一条直线，把猫和狗分开。

##### 2.4.3.2 多分类时，就是很多条线一起切空间

如果有 10 个类别，就有 10 个分数：

$$s1,s2,…,s10$$

模型最后选择分数最大的类别：

$$\hat{y} = \arg\max_j s_j$$

也就是说，哪个类别的分数最高，就预测成哪个类别。几何上，每两个类别之间都会有一个分界面。在二维是直线，在三维是平面。在 $3072$ 维这种高维空间里，就叫超平面。


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
