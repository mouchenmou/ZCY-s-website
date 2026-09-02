# 1. K-Nearest Neighbors（KNN）

$k = 1$ 的 KNN 叫做 nearest neighbor classifier。给定一个新点 $x$，它会在训练集中找到离 $x$ 最近的那个训练样本，然后输出那个训练样本的标签。

如果两个点分别是：

$$
x = (x_1, x_2, \dots, x_D), \quad x_n = (x_{n1}, x_{n2}, \dots, x_{nD})
$$

那么它们的欧几里得距离可以写成：

$$
\|x - x_n\|_2 = \sqrt{\sum_{d=1}^{D}(x_d - x_{nd})^2}
$$

所以 $k=1$ 的预测过程可以理解成：

$$
i(x) = \arg\min_n \|x_n - x\|_2
$$

$$
g_\mathcal{D}(x) = y_{i(x)}
$$

其中 $\mathcal{D}$ 表示训练数据集，$x_n$ 是第 $n$ 个训练样本，$y_n$ 是对应的标签。

---

# 2. Training vs Inference

## 2.1 Training

Training：**用数据构建模型的过程**。

比如线性模型训练时，会通过数据学到一组权重 $w$。训练结束之后，模型里面的参数就固定下来了。

但是对 KNN 来说，它只是把训练数据记住，而非用 loss 函数进行学习：

```python
knn.fit(X_train, y_train)
```

## 2.2 Inference

Inference：**用已经构建好的模型做预测的过程**。

对于 KNN，inference 才是真正干活的地方。因为每来一个新样本，它都要：

1. 把这个新样本和训练集中的每个点算距离。
2. 找到最近的 $k$ 个邻居。
3. 根据邻居的标签投票或者取平均。

---

# 3. Parametric vs Non-Parametric Models

## 3.1 Parametric Model（参数模型）

Parametric model family：**模型参数的数量是固定的，不随着训练数据数量变多而变多**。

比如线性回归：

$$
g_w(x) = w_0 + \sum_{d=1}^{D} w_d x_d
$$

这里参数就是：

$$
w_0, w_1, \dots, w_D
$$

只要特征维度 $D$ 固定，不管训练数据有 $100$ 个还是 $100000$ 个，参数数量都是 $D+1$ 个。

## 3.2 Non-Parametric Model（非参数模型）

Non-parametric model family：**模型需要保存的信息会随着训练数据一起增长**。

比如 KNN，它的参数其实就是整个训练集：

$$
\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}
$$

当 $N$ 变大时，KNN 需要保存的数据也变大。

!!! explanation "non-parametric 不是没有参数"
    non-parametric 不是说模型完全没有任何可以调的东西，而是说模型复杂度不会被一个固定维度的参数向量限制住。

    对 KNN 来说，训练集越大，它能形成的决策边界就越复杂。它的模型容量本质上跟数据量绑定在一起。

---

# 4. Generalization（泛化）

Generalization：模型在新的、没有见过的数据上也能表现得很好。

!!! explanation "为什么需要泛化"
    因为**每个训练样本最近的邻居就是它自己**，所以我们用 $k=1$ 的 KNN 在训练集上做预测，那么得到的训练准确率就是 $100\%$。这就像把考试原题和答案提前发给大家，然后考试时还考原题。所有人都能考满分，但这不能说明大家真的学会了。
    
    **在见过的数据上表现好，不能证明模型会处理没见过的数据**。这就是为什么机器学习里面很关心 generalization。



---

# 5. Train / Validation / Test Split

## 5.1 Train-Test Split

为了评估 generalization，最基本的方法是 train-test split：

1. Shuffle training data。
2. 把数据分成两部分：
    - training part：通常比较大，比如 $80\%$，用来训练模型。
    - testing part：通常比较小，比如 $20\%$，用来评估泛化能力。

test dataset 只能在模型开发完成之后用一次。如果我们拿 test data 来调模型，那 test data 就不再是没见过的数据了。

## 5.2 Validation Split

问题来了：如果 test set 不能用来调模型，那我们怎么选择模型设计和超参数？

答案是再切一个 validation set。

数据可以分成：training data, validation data 和 testing data

三者的作用分别是：

1. train：用来训练模型。
2. validation：用来调模型设计和选择超参数。
3. test：最后只用一次，用来评估最终模型的 generalization。

!!! explanation "参数和超参数"
    参数是模型从数据里学出来的，超参数是训练开始前或者训练过程中由人设置的。
    
    参数会通过 training data 和 backprop自动更新，比如 $W$ 和 $b$。超参数不会被学出来，而是我们认为规定的，比如 $η$

!!! explanation "我的疑问"
    ### 我的疑问1
    什么叫 validation data 用来调整模型和选择超参数？
    ### 解析
    这里的**调模型设计**和**选超参数**很类似，它们都是：**你做出多个候选模型，然后看哪个在 validation set 上表现最好。**
	
	比如选超参数。假设网络结构完全不变，只改 $learning rate$：
	
	$$10^{-2},\quad 10^{-3},\quad 10^{-4}$$
	
	你分别训练三个模型，然后比较 validation accuracy，哪个最好就选哪个。这叫 **hyperparameter tuning**。
	
	而**调模型设计**更广一点，比如你连网络长什么样都在改：
	
	- 2 层还是 5 层
	- hidden dimension 用 128 还是 512
	- 用 ReLU 还是 GELU
	- 要不要加 dropout
	- 要不要加 BatchNorm
	- CNN 还是 Transformer
	### 我的疑问2
	为什么还要搞一个 validation 出来？训练的时候用 training data，测试的时候用 testing data 但是不用它来更新参数不就好了？
	### 解析
    如果用 testing data 来调整的话，testing data 就相当于已经参与了模型优化的整个过程。这就没法泛化了。
---

# 6. Overfitting vs Underfitting
我们通过 KNN 来了解 Overfitting 和 Undercutting。

KNN 里面的 $k$ 是一个非常重要的超参数。

如果不是只看最近的一个邻居，而是让最近的 $k$ 个邻居一起投票，那么 $k$ 的大小会直接影响决策边界的复杂程度。

## 6.1 $k$ 太小：Overfitting

当 $k$ 很小，比如 $k=1$，模型会非常相信每一个训练点。

如果训练集中有一个异常点，$k=1$ 也会认真地给它划出一小块区域。这样模型的边界会变得很碎，甚至会把噪声也当成规律。

这就是 overfitting（过拟合）：

$$
\text{训练集表现很好，但新数据表现不好}
$$

## 6.2 $k$ 太大：Underfitting

当 $k$ 很大时，模型会把很多邻居混在一起投票，决策边界会变得非常平滑。

如果 $k$ 大到离谱，模型几乎不会响应数据中的局部结构。它可能连训练集本身都拟合不好。往大了点想，如果原先只有 100 个图片，你将 K 设成 101，那不炸了吗。

这就是 underfitting（欠拟合）：

$$
\text{训练集和验证集表现都不好}
$$

## 6.3 用 validation set 选择 $k$

我们不能靠肉眼看边界来选 $k$，更标准的方法是：

1. 对每一个候选的 $k$ 都训练一个 KNN。
2. 在 validation set 上计算 accuracy。
3. 选择 validation accuracy 最高的 $k$。
4. 最后在 test set 上评估一次。

如果画出 training accuracy 和 validation accuracy，通常会看到：

1. $k$ 太小时，training accuracy 很高，validation accuracy 没那么高，这是 overfitting。
2. $k$ 太大时，training accuracy 和 validation accuracy 都下降，这是 underfitting。
3. validation curve 的最高点，就是我们要选的 $k$。


---

# 7. Hyperparameters（超参数）

## 7.1 什么是 hyperparameter

Hyperparameter 是在训练过程中保持固定、由我们提前选择的量。

比如在带正则化的模型中：

$$
\hat{w} = \arg\min_{w \in \Theta} \text{Error}[h_w; \mathcal{D}_{training}] + \lambda \text{Reg}[w]
$$

这里的 $\lambda$ 就是一个 hyperparameter，因为它不是模型从数据里自己学出来的，而是我们在训练前指定的。

KNN 里面的 $k$ 也是 hyperparameter。

## 7.2 Parameter 和 Hyperparameter 的区别

Parameter 是模型在 training 的时候从数据里面学出来的。

Hyperparameter 是 training 开始之前由我们设定的。

比如：

1. 线性回归里的 $w$ 是 parameter。
2. 正则化强度 $\lambda$ 是 hyperparameter。
3. KNN 里的 $k$ 是 hyperparameter。
4. KNN 没有真正意义上的 learned parameters，这一点比较特殊。



---

# 8. Feature Engineering（特征工程）

Feature engineering：为了让模型更有效地学习，对数据和特征做一些准备和变换。

KNN 特别依赖 feature engineering，因为它完全建立在 distance 上。如果特征的单位和尺度不合适，距离就会被某些特征主导。

## 8.1 为什么 KNN 需要标准化

以企鹅数据为例：

1. flipper length 大概在 $172$ 到 $231$ mm。
2. bill length 大概在 $32$ 到 $60$ mm。

这两个特征的数字范围差很多。因为 KNN 用距离来判断相似性，所以 flipper length 会因为数值范围更大而主导距离。

但这不一定说明 flipper length 就更重要，它只是单位和尺度更大。

## 8.2 Standardization

Standardization（标准化）会把每个 feature 重新缩放成：

$$
\text{mean} = 0, \quad \text{standard deviation} = 1
$$

常见写法是：

$$
x'_d = \frac{x_d - \mu_d}{\sigma_d}
$$

其中 $\mu_d$ 是第 $d$ 个特征在训练集上的均值，$\sigma_d$ 是第 $d$ 个特征在训练集上的标准差。

标准化之后，不同特征就能在距离计算里更公平地发挥作用。

!!! warning "标准化时容易犯的错"
    $\mu_d$ 和 $\sigma_d$ 应该只用 training set 来算。

    validation set 和 test set 只能使用 training set 算出来的均值和标准差进行变换。否则就相当于提前偷看了 validation/test 的统计信息。

---

# 9. KNN for Regression

前面 KNN 解决的是 classification 问题。但 KNN for regression 是用 KNN 来**预测连续数值**，而不是预测类别。

KNN 可以做 classification 和 regression 的区别：

1. classification：让最近的 $k$ 个邻居投票。
2. regression：对最近的 $k$ 个邻居的数值取平均。

KNN regression 可以写成：

$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$

其中 $N_k(x)$ 表示离 $x$ 最近的 $k$ 个训练样本下标集合。

## 9.1 KNN Regression 里的 $k$

KNN regression 里面也会遇到 overfitting 和 underfitting。

1. $k=1$：预测函数会变成很锯齿状的 step function，它会追着每一个训练点跑，容易 overfit。
2. $k$ 很大，比如 $k=200$：预测函数会接近一条很平的线，模型对数据变化不敏感，容易 underfit。

所以 classification 和 regression 虽然输出形式不同，但对 $k$ 的理解是一致的：

$$
k \text{ 越小，模型越复杂；} \quad k \text{ 越大，模型越平滑}
$$

---

# 10. K-Means Clustering

## 10.1 从 supervised 到 unsupervised

前面的问题都是 supervised learning，因为每只企鹅都有 species label。

现在换一个问题：如果研究员只给我们企鹅的 bill length 和 flipper length，但是不给 species label，我们还能不能发现数据里有没有自然形成的群体？

这就是 unsupervised learning（无监督学习）。

K-means clustering 要解决的就是 clustering（聚类）问题：**在没有标签的情况下，把相似的数据点分到同一组**。

## 10.2 K-Means 的输入和输出

输入是一个数据集：

$$
\mathcal{D} = \{x_1, x_2, \dots, x_N\}, \quad x_n \in \mathbb{R}^D
$$

输出包括两部分：

1. $K$ 个 cluster centers：

$$
\mu_k \in \mathbb{R}^D
$$

2. 每个数据点的 assignment：

$$
z_n \in \{1, 2, \dots, K\}
$$

$z_n$ 表示第 $n$ 个点被分到了哪个 cluster。

这里的 $\mu_k$ 是 K-means 学出来的 model parameters，而 $K$ 是我们提前指定的 hyperparameter。

## 10.3 K-Means 的目标函数

K-means 希望每个点都离自己所属的 cluster center 尽可能近。

所以目标函数是：

$$
\arg\min_{\mu, z} \sum_{n=1}^{N} \|x_n - \mu_{z_n}\|_2^2
$$

!!! explanation "这个目标函数在说什么"
    对每个数据点 $x_n$，它都会被分配到某个 cluster，那个 cluster 的中心是 $\mu_{z_n}$。

    $\|x_n - \mu_{z_n}\|_2^2$ 就是这个点到自己所属中心的平方距离。

    K-means 要做的是同时找到：

    1. 每个 cluster 的中心 $\mu_1, \dots, \mu_K$
    2. 每个点应该属于哪个 cluster，也就是 $z_1, \dots, z_N$

    让所有点到所属中心的总距离最小。

## 10.4. Lloyd's Algorithm

直接同时优化 $\mu$ 和 $z$ 比较麻烦，所以 K-means 通常用 Lloyd's algorithm。

### 10.4.1 Initialization

先随机选择 $K$ 个点作为初始 cluster centers：

$$
\mu_1, \mu_2, \dots, \mu_K
$$

因为初始化是随机的，所以 K-means 每次运行可能会得到不同结果。

### 10.4.2 Update Assignments

固定 cluster centers $\mu$，然后把每个点分配给离它最近的中心。

也就是更新 $z$：

$$
z = \arg\min_z \sum_{n=1}^{N} \|x_n - \mu_{z_n}\|_2^2
$$

直观理解就是：每个点看看自己离哪个中心最近，就加入哪个 cluster。

### 10.4.3 Update Centers

固定 assignments $z$，然后重新计算每个 cluster 的中心。

新的中心就是这个 cluster 里面所有点的平均值。

如果第 $k$ 个 cluster 里的点的下标集合是：

$$
C_k = \{n : z_n = k\}
$$

那么新的中心是：

$$
\mu_k = \frac{1}{|C_k|} \sum_{n \in C_k} x_n
$$



### 10.4.4 迭代直到收敛

Lloyd's algorithm 的完整过程就是：

1. 随机初始化 $K$ 个中心。
2. 把每个点分配给最近的中心。
3. 对每个 cluster 重新计算平均值作为新中心。
4. 重复 2 和 3，直到结果不再变化。


## 10.5 K-Means 的收敛和局限

### 10.5.1 K-Means 一定会收敛

Lloyd's algorithm always converges, but only to a local optimum.

原因是：

1. assignment step 会让每个点选择最近的中心，所以目标函数不会变大。
2. center update step 会把中心放到当前 cluster 的均值位置，所以目标函数也不会变大。
3. 可能的 assignments 数量是有限的。

因此它不可能无限下降下去，最终会停下来。

### 10.5.2 但它只保证 local optimum

K-means 的问题是：它不保证找到全局最优解。

不同的随机初始化可能会带来不同的最终 clustering。也就是说，算法一定会停，但停下来的位置不一定是最好的位置。

所以 `sklearn` 里面通常会运行多次 K-means，然后保留 objective 最小的那次结果。

---

## 10.6 如何选择 Cluster 的数量 $K$

K-means 里面的 $K$ 是 hyperparameter，需要我们自己选。

常见方法有：

1. 使用 domain understanding。比如企鹅数据里，如果我们知道大概有三种 species，那可以先试 $K=3$。
2. 使用 elbow method。

### 10.6.1 Elbow Method

Elbow method：尝试不同的 $K$，看 K-means objective 的下降曲线。

目标函数还是：

$$
\sum_{n=1}^{N} \|x_n - \mu_{z_n}\|_2^2
$$

当 $K$ 增大时，cluster 更多，每个点离自己中心一般会更近，所以 objective 通常会下降。

但是下降到某个位置之后，继续增大 $K$ 带来的改进会变得很小。这个下降开始变慢的拐点，就叫 elbow。

![](附件/Pasted%20image%2020260902002118.png)

!!! explanation "为什么不能无脑让 K 越大越好"
    如果 $K=N$，每个点都可以自己成为一个 cluster center，那么 objective 可以变成 $0$。

    但这显然没什么意义，因为它只是把每个点单独分出来，并没有帮我们发现数据里的自然结构。

    所以选择 $K$ 不是让 objective 越小越好，而是要在解释能力和模型复杂度之间做权衡。

---

## 10.7 Clusters Are Not Labels

K-means 在企鹅数据上可能会找到三个 cluster，而且这三个 cluster 可能和三种 species 很接近。

但要注意：**clusters are not labels**。

K-means 从来没有见过 species label，它只是发现了在 bill length 和 flipper length 这两个特征空间中，有一些点彼此靠得比较近。

所以我们不能直接说：

$$
\text{cluster} = \text{species}
$$

更合理的说法是：

$$
\text{cluster 和 species 在这个特征空间里恰好比较一致}
$$

如果换一组 feature，cluster 可能会完全不一样。

!!! warning "解释 cluster 要小心"
    聚类结果需要谨慎解释。

    一个 cluster 到底对应什么语义，不能只靠算法自己告诉我们。我们通常还需要 domain knowledge，或者拿一些真实 labels 来检查。

    所以 K-means 可以帮助我们发现数据结构，但不能自动赋予这些结构真实世界里的含义。
