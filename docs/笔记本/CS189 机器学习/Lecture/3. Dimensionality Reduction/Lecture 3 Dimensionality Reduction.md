## 1. Principal Component Analysis (PCA)
主成分分析，也就是分析数据里面主要的那些成分，主要的方向：

$$把一个高维的数据，投影到一个更低维的空间，同时尽量保留原数据中的重要信息$$

- 降维：我们会把高维数据投影到一个能较好描述原始数据的低维空间中
- Parametirc method：模型会有一组固定数量的参数
    - 像 KNN 这种就是 non-parametric method，因为它没有固定的参数，输入图像 X 的数量就是参数，因此参数的数量会随着 X 的数量的改变而改变
    - Parametric method 就是去学 $w,b$ 这些东西，它们的数量是固定的


## 1.1 Dimensionality Reduction

Dimensionality Reduction 跟 Clustering 一样，都是 Unsupervised Learning

### 1.1.1 降维的原因

1. 为了更好的可视化。像1维、2维、3维还能画出来，但是到了4维就没办法可视化了，因此我们只能降维。
2. 像 KNN，K-means 这类，在太高的维度反而会表现得很差
    - 因为它们非常依赖距离，如果维度很高，比如1000维的话，那么距离公式就是 $∥x−y∥^2=j=\sum^{1000}_{j=1}​(x_j​−y_j​)^2$，然而这里面很多维度可能只是噪声，但是它们都被贡献到了距离里面
    - 因此遇到这种情况，降维是一个比较好的选择

### 1.2 Intrinsic Dimension and Matrix Rank

### 1.2.1 Matirx Rank

矩阵的秩：矩阵列空间的维数，也就是最多有多少列是线性无关的。

对于

$$X\in \mathbb{R}^{n\times d}$$

有：

$$\operatorname{rank}(X)\leq \min(n,d)$$

如果达到这个上界，就称这个矩阵是 **full rank**。

### 1.2.2 Intrinsic Dimension

Intrinsic Dimension：完整描述这组数据所需要的最少变量

它和 rank 最大的区别就是 Intrinsic Dimension 允许变量之间存在非线性关系，而 rank 只考虑线性关系。

!!! example "举个例子"
    以下面这幅图为例：
    ![](附件/Pasted%20image%2020260905201136.png)
    图中一共有 5 个变量，但是 perimeter 这个变量可以由 width 和 height 线性组合出来，即：
    $$周长=2*(长+宽度)$$
    
    因此，我们只需要 4 个维度就能表示出这五个变量，所以它的 $rank=4$
    
    然而，intrinsic dimension 允许非线性关系，由于 diagonal、area、perimeter 都能由 height 和 width 表示出来，即：
    
    $$\begin{aligned}
    &diagonal = \sqrt{height^2z+width^2}
    \\& perimeter=2(height+width)
    \\& area=height\times width
    \end{aligned}$$
    
    因此，只需要 height 和 width 这两个维度就能把它们都表示出来了，所以它的 $intrinsic dimension=2$。

### 1.3 降维的原因 plus 版本
Real data is rarely exactly low rank.

明明刚才上面那个例子，我们可以看出，虽然有5个变量，但是 $rank=4$。按理来说，在更宽泛的情况下，一个维度很大很大的矩阵，它的 rank 也应该是一个比较小的数，但是为什么这里说真实的数据往往不是 low rank 呢？

原因是，我们现实中会有很多误差，也就是噪声，比如说我们刚刚说的：

$$周长=2(长+宽)$$

这只是理论上的，现实中我们去测量的时候，肯定会有误差的：

$周长 =2(长+宽)+ϵ$

这要 $ϵ \ne 0$，那么原先的 $rank=4$ 就得变成 $rank=5$ 了，这也就是我们现实数据中的噪声。

这种数据虽然不是严格的低 rank 数据，但是往往可以被低维空间很好地近似。

而且研究表明，PCA发现，将维度压缩到 rank=2 都能解释 86% 左右的数据 spread. 这样印证了 datasets are nearly full rank but are often accurately approximated by a lower dimensional sub-space

---

## 2. Deriving PCA

我们刚刚讲了降维的好处，那么我们应该如何找到一个低维表示，使它尽可能准确地近似原始数据呢？

### 2.1 Dimensionality reduction as Matrix Factorization（因式分解）

先规定：假设有 n 个 data point，每个 data point 有 d 个 feature，那么:

$$X\in\mathbb R^{n\times d}$$

然后 PCA 的目标被写成：

$$X\approx ZW$$

其中：

$$Z\in\mathbb R^{n\times k}, W\in\mathbb R^{k\times d}$$

而且 k<d。因为原来每个点有 d 个 feature，现在只用 k 个数表示，所以这里的 Z 就是**低维坐标**。也就是说，我们把每个数据的 d 个 feature 压缩成了 k 个 feature，这样就实现了降维。

![](附件/Pasted%20image%2020260905233344.png)

### 2.2 什么时候 $ZW$ 能完全还原 $X$，什么时候只能近似？

能不能完全还原 $X$，取决于 $k$ 和 $rank(X)$ 的大小：

- $k>rank(X)$：能完全还原
- $k<rank(X)$：只能近似

!!! example "举个例子"
    ![](附件/Pasted%20image%2020260905233902.png)
    这道题选 B 和 D。
    
    A 得出的矩阵最大的秩只能是2，C 得出的最大的秩只能是1

### 2.3 objective function
如何判断我们这个 Z 和 W 构建的好不好，就要用到损失函数：

$$Loss(Z,W)=\frac{1}{n}​\sum^n_i​∥X_i​−Z_i​W∥^2$$


### 2.4 Why we subtract the mean?

$ZW$ 能表示出来的所有点，天然都只能落在由 $W$ 的行向量张成的那个子空间里，而这个子空间一定经过原点。

然而，真实的数据 $X$ 不一定围绕原点分布。比如一堆点可能都集中在 (10,10)  附近，那么最合适的低维直线也应该经过这堆数据的中心附近，而不是被强迫经过 (0,0)。

所以 PCA 先对每个 feature 减去它自己的均值，也就是做 centering：

$$x_i \leftarrow x_i-\mu$$

这样一来，数据的均值就被移动到了原点：

$$\mu \rightarrow 0$$

于是：**原本应该经过数据均值的最佳低维子空间，现在就等价于经过原点的低维子空间。**

![](附件/Pasted%20image%2020260906002200.png)

!!! explanation "解析"
	在 PCA 中，我们写：
	
	$$
	X \approx ZW
	$$
	
	对第 $i$ 个数据点来说，它的重建结果是：
	
	$$
	Z_iW
	$$
	
	为什么说这些重建出来的点一定落在一个**经过原点的子空间**里？
	
	先看最简单的情况：$k=1$。
	
	这时：
	
	$$
	W=w\in\mathbb{R}^{1\times d}
	$$
	
	而第 $i$ 个数据点的低维坐标只是一个标量：
	
	$$
	z_i\in\mathbb{R}
	$$
	
	所以重建结果为：
	
	$$
	Z_iW=z_iw
	$$
	
	当 $z_i$ 取不同的值时，我们得到：
	
	$$
	\cdots,-2w,-w,0,w,2w,\cdots
	$$
	
	这些点全部落在由 $w$ 决定的一条直线上。
	
	而当：
	
	$$
	z_i=0
	$$
	
	时：
	
	$$
	z_iw=0
	$$
	
	所以这条直线一定经过原点。
	
	如果 $k=2$，那么：
	
	$$
	Z_i=[z_{i1},z_{i2}]
	$$
	
	并且：
	
	$$
	W=
	\begin{bmatrix}
	w_1\\
	w_2
	\end{bmatrix}
	$$
	
	因此：
	
	$$
	Z_iW=z_{i1}w_1+z_{i2}w_2
	$$
	
	也就是说，所有重建出来的点都是 $w_1$ 和 $w_2$ 的线性组合，因此它们都落在：
	
	$$
	\operatorname{span}(w_1,w_2)
	$$
	
	这个二维子空间中。
	
	而这个子空间一定包含：
	
	$$
	0w_1+0w_2=0
	$$
	
	所以它也一定经过原点。
	
	因此一般来说：
	
	$$
	\boxed{ZW\text{ 描述的是一个经过原点的线性子空间}}
	$$
	
	但真实数据不一定分布在原点附近。比如数据可能都集中在 $(10,10)$ 附近，那么最适合这些数据的直线可能经过 $(10,10)$，而不是 $(0,0)$。
	
	所以 PCA 会先对数据做 centering：
	
	$$
	x_i\leftarrow x_i-\mu
	$$
	
	把数据的均值移动到原点，然后再去寻找这个经过原点的低维子空间。

### 2.5 The factorization is not unique
 $X=ZW$ 这个分解不是唯一的，对于任意可逆的（invertible）$k\times k$ 的矩阵 $A$，都有：

  $$ZW=ZA^{-1}AW$$

虽然 reconstruction 完全没变，但是 $Z$ 和 $W$ 本身可以完全不一样

我们甚至可以把 $Z$ 放大 $10$ 倍，把 $W$ 缩小 $10$ 倍，那么 objective 也完全不变：

$$(10Z)(\frac{1}{10}W)=ZW$$

为了解决这个困扰，我们**规定 $W$ 为正交矩阵**，这个规定有以下两个好处：

1. 消除 scale ambiguous
    - 因为原来的 $ZW$ 你可以随意的将 $W$ 放大缩小，只要 $Z$ 跟着反向变就行，这样的话就没有统一的标准。
2. 让这些 bias 变得规范、好解释、好计算
    - 规定了 $WW^T=I$，相当于要求每个 basis 的长度都是 1，而且彼此互相垂直，表达不同的方向。

### 2.6 Summary

因此，我们最终规定：

$$(Z^*,W^*) =\min_{Z,W} \frac1n\sum_{i=1}^n \|X_i-Z_iW\|^2 \quad \text{subject to }WW^\top=I $$

---
## 3. The basic stationary point method:

我们求函数的最小值，就是高中的无脑法：

1. 对参数求导
2. 令导数为 0，解出 stationary point (驻点)
3. 如果需要的话再验证一下二阶导是不是正数。

### 3.1 具体做法：

设

$$X\in\mathbb R^{n\times d},\ Z\in\mathbb R^{n\times k},\ W\in\mathbb R^{k\times d}$$

并且有约束

$$WW^\top=I_k$$

PCA 的 loss 是：

$${Loss}(Z,W) = \frac1n \|X-ZW\|_F^2$$

把 loss 展开：

$$
\begin{aligned}
\|X-ZW\|_F^2 &= \operatorname{tr}\left( (X-ZW)(X-ZW)^\top \right) \\
&= \operatorname{tr}(XX^\top) - \operatorname{tr}(ZWX^\top) - \operatorname{tr}(XW^\top Z^\top)+\operatorname{tr}(ZWW^\top Z^\top)\\
&= \operatorname{tr}(XX^\top) - 2\operatorname{tr}(ZWX^\top) + \operatorname{tr}(ZWW^\top Z^\top)\\
&= \operatorname{tr}(XX^\top) - 2\operatorname{tr}(ZWX^\top) + \operatorname{tr}(ZZ^\top)
\end{aligned}
$$

!!! explanation "解释几个问题"
    ### 1. $tr$ 
    $tr$ 是把一个方阵的**主对角线的元素都加起来**：
    
    对于一个方阵 
    
    $$A= \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22} \end{bmatrix}$$
    
    我们可以得到 $AA^T$：
    
    $$AA^T=\begin{bmatrix} a^2+b^2 & ac+bd\\ ac+bd & c^2+d^2 \end{bmatrix}$$
    
    取 trace 可以得到：
    
    $$\mathrm{tr}(AA^T) = a^2+b^2+c^2+d^2$$
    
    ### 2. 两个化简是如何得到的
    
    1. $ZWX^T$ 和 $XW^TZ^T$ 两个矩阵互为对方的转置，因此它们的对角线上元素相同，trace 也就相同，所以：$\operatorname{tr}(ZWX^\top) + \operatorname{tr}(XW^\top Z^\top)=2\operatorname{tr}(ZWX^\top)$
    2. 因为 $W$ 是正交矩阵，所以 $WW^T=I$，即 $\operatorname{tr}(ZWW^\top Z^\top)=\operatorname{tr}(Z Z^\top)$

因此：

$$Loss(Z,W)=\frac{1}{n}[tr(XX^T)-2tr(ZWX^T)+tr(ZZ^T)]$$

其中 $\operatorname{tr}(XX^\top)$ 跟 $Z,W$ 都无关，可以记成常数 $C$。所以 loss 函数可以写成：

$$Loss(Z,W)=C+\frac{1}{n}​[−2tr(ZWX^⊤)+tr(ZZ^⊤)]$$





















这节课主要讲 **PCA（Principal Component Analysis，主成分分析）**。

PCA 要解决的问题是：原始数据维度很高，但是这些高维特征里面可能有很多冗余信息。我们希望把数据投影到一个低维空间里，同时尽可能保留原始数据中的主要信息。

例如 lecture 里面的 congressional votes demo：

1. 原始数据是一个 $441\times 41$ 的矩阵。
    - $441$ 表示 441 个 legislators。
    - $41$ 表示 41 次投票记录。
    - 每个元素是 $0/1$，表示某个人对某个议案的投票结果。
2. PCA 把它降到 $441\times 2$。
    - 每个 legislator 变成二维平面上的一个点。
    - 这里选择 $2$ 维是一个 hyperparameter decision。
3. 很有意思的是，算法没有拿到 party label，但是第一维坐标的符号和党派标签高度一致。

![](附件/Lecture3_vote_pca_demo-05.png)

!!! explanation "为什么这很神奇"
    PCA 只看投票矩阵本身，也就是每个人在不同议案上的投票模式。
    
    它没有被告诉谁是民主党，谁是共和党。但是如果投票模式本身就主要沿着“党派差异”这个方向变化，那么 PCA 找到的第一主成分就很可能对应这个方向。
    
    所以 PCA 不是在做 supervised classification，而是在 unsupervised 地找数据里面最大的变化方向。

## 1.1 为什么需要降维

High-dimensional data 会带来几个问题：

1. 很难可视化。
    - 二维、三维还能画图。
    - 四维以上就很难直接看。
2. 有些模型在高维空间里面表现会变差。
    - 例如 KNN 和 K-means 都依赖距离。
    - 维度很高时，距离会变得不那么有区分度。
3. 很多数据本身虽然维度很高，但真正变化的自由度可能很少。

Dimensionality reduction 的目标可以写成：

$$
\text{high-dimensional data}\rightarrow \text{low-dimensional representation}
$$

并且这个 low-dimensional representation 要尽量 preserve information。

---

# 2. Rank 和 Intrinsic Dimension

## 2.1 Matrix Rank（矩阵的秩）

矩阵的 rank 可以从两个角度理解：

1. columns 张成的空间的维度。
2. 最大的 linearly independent columns 的数量。

如果一个 column 能由其他 columns 线性组合得到，那么它就是 redundant 的。

对于一个矩阵：

$$
X\in\mathbb{R}^{n\times d}
$$

它的 rank 满足：

$$
\operatorname{rank}(X)\leq \min(n,d)
$$

如果达到了这个上界，就叫 full rank。

## 2.2 Rectangle 例子

假设我们有很多 rectangles，每一行是一个 rectangle，每一列是它的属性：

$$
\text{width},\quad \text{height},\quad \text{perimeter},\quad \text{area},\quad \text{diagonal}
$$

其中：

$$
\text{perimeter}=2w+2h
$$

所以 perimeter 这一列可以由 width 和 height 线性组合得到，它是 redundant column。

但是：

$$
\text{area}=wh
$$

$$
\text{diagonal}=\sqrt{w^2+h^2}
$$

这两个不是 width 和 height 的线性组合。因此从 matrix rank 的角度看，这个数据矩阵的 rank 是 $4$，不是 $2$。

## 2.3 Intrinsic Dimension（内在维度）

Intrinsic dimension 指的是：**真正描述这个数据所需要的最少变量数量**。

对于 rectangle 例子，其实只需要两个变量：

$$
w,\quad h
$$

因为其他所有属性都可以由 $w$ 和 $h$ 决定：

$$
\text{perimeter}=2w+2h
$$

$$
\text{area}=wh
$$

$$
\text{diagonal}=\sqrt{w^2+h^2}
$$

因此：

$$
\text{intrinsic dimension}=2
$$

![](附件/Lecture3_intrinsic_dimension-12.png)

!!! explanation "rank 和 intrinsic dimension 的区别"
    rank 只允许看 **linear relationship（线性关系）**。
    
    intrinsic dimension 允许看 **nonlinear relationship（非线性关系）**。
    
    在 rectangle 例子里，area 和 diagonal 虽然不是 width、height 的线性组合，但它们仍然完全由 width 和 height 决定。所以 rank 是 $4$，但是 intrinsic dimension 是 $2$。

## 2.4 真实数据通常不会严格 low rank

真实数据里面经常有 noise。

例如原本 perimeter 应该满足：

$$
\text{perimeter}=2w+2h
$$

但如果 perimeter 是用卷尺测出来的，就会有测量误差。这样精确的线性关系会被破坏，矩阵 rank 可能直接变回 full rank。

所以真实数据通常不是 exactly low rank，而是：

$$
\text{full rank but approximately low-dimensional}
$$

!!! warning "不要把 full rank 理解成不能降维"
    full rank 只说明从严格线性代数角度看，没有哪一列可以被完全丢掉。
    
    但是 PCA 关心的是近似：能不能用一个低维子空间把数据的大部分变化解释掉。
    
    所以一个矩阵可以 full rank，同时又非常适合被一个低维空间近似。

---

# 3. 把降维写成 Matrix Factorization

## 3.1 输入数据矩阵

假设有 $n$ 个数据点，每个数据点有 $d$ 个 features：

$$
X\in\mathbb{R}^{n\times d}
$$

第 $i$ 个数据点是第 $i$ 行：

$$
X_i\in\mathbb{R}^{1\times d}
$$

lecture 里面默认 $X$ 已经 centered，也就是每一列都减去了自己的均值。

$$
\frac{1}{n}\sum_{i=1}^{n}X_i=0
$$

## 3.2 降维作为矩阵分解

PCA 希望找到两个矩阵：

$$
Z\in\mathbb{R}^{n\times k}
$$

$$
W\in\mathbb{R}^{k\times d}
$$

使得：

$$
X\approx ZW
$$

![](附件/Lecture3_matrix_factorization-17.png)

其中 $k<d$，所以 $Z$ 是低维表示，$W$ 是把低维表示还原回原空间的 basis。

可以把它理解成：

$$
\underbrace{X}_{n\times d}
\approx
\underbrace{Z}_{n\times k}
\underbrace{W}_{k\times d}
$$

## 3.3 $Z$ 和 $W$ 分别是什么

$Z$ 的每一行表示一个数据点在低维空间里的坐标：

$$
Z_i\in\mathbb{R}^{1\times k}
$$

$W$ 的每一行表示原空间中的一个 basis direction：

$$
W_j\in\mathbb{R}^{1\times d}
$$

所以第 $i$ 个数据点的重建结果是：

$$
Z_iW
$$

!!! explanation "直观理解 ZW"
    $W$ 的每一行是一种“方向”或者“模板”。
    
    $Z_i$ 告诉我们：第 $i$ 个数据点要按什么比例混合这些方向。
    
    因此 $Z_iW$ 就是用 $k$ 个主要方向重新拼回一个 $d$ 维数据点。

## 3.4 为什么低维分解只能近似

矩阵乘积的 rank 满足：

$$
\operatorname{rank}(ZW)\leq \min(\operatorname{rank}(Z),\operatorname{rank}(W))\leq k
$$

如果原始数据 $X$ 的 rank 大于 $k$，那么 $ZW$ 的 rank 不可能等于 $X$ 的 rank。

所以当我们选择：

$$
k<\operatorname{rank}(X)
$$

时，$X\approx ZW$ 一定只是 approximation。

---

# 4. PCA 的 Objective Function

## 4.1 Reconstruction Error

PCA 要选择最好的 $Z$ 和 $W$。这里“最好”的意思是：重建误差最小。

对于第 $i$ 个数据点：

$$
X_i
$$

它的重建结果是：

$$
Z_iW
$$

所以 reconstruction error 是：

$$
\|X_i-Z_iW\|_2^2
$$

整体 loss 是平均 squared error：

$$
\operatorname{Loss}(Z,W)=\frac{1}{n}\sum_{i=1}^{n}\|X_i-Z_iW\|_2^2
$$

于是 PCA 的优化问题可以写成：

$$
Z^*,W^*
=
\arg\min_{Z\in\mathbb{R}^{n\times k},\ W\in\mathbb{R}^{k\times d}}
\frac{1}{n}\sum_{i=1}^{n}\|X_i-Z_iW\|_2^2
$$

## 4.2 为什么要先 subtract mean

模型：

$$
X\approx ZW
$$

表示所有重建点都落在一个经过 origin 的 subspace 上。

但是 origin 是人为选出来的，真实数据没有理由天然围绕 origin 分布。最好的低维子空间应该经过数据的 mean。

所以 PCA 会先做 centering：

$$
X_i\leftarrow X_i-\bar{x}
$$

这样数据均值变成 $0$，最好的 subspace 就可以被看成经过 origin。

!!! explanation "为什么 centered 以后可以过原点"
    原始数据如果集中在某个不经过原点的位置，那么用一个必须经过原点的 subspace 去拟合它，会浪费很多能力在“追位置”上。
    
    减去均值之后，数据云被平移到原点附近。此时 PCA 只需要关心数据主要沿哪些方向展开。

## 4.3 Factorization 不唯一

如果 $A$ 是任意 invertible 的 $k\times k$ 矩阵，那么：

$$
ZW=ZA^{-1}AW
$$

也就是说，同一个重建结果可以对应很多组不同的 $Z$ 和 $W$。

为了去掉这种 scale 和 basis 的 ambiguity，PCA 要求 $W$ 的行向量是 orthonormal 的：

$$
WW^T=I
$$

这包含两层意思：

1. 每一行长度为 $1$：

$$
W_iW_i^T=1
$$

2. 不同行互相垂直：

$$
W_iW_j^T=0,\quad i\neq j
$$

!!! explanation "为什么这样不会损失表达能力"
    一个 $k$ 维 subspace 可以有很多组 basis。
    
    即使用一组乱七八糟的 basis 能表示这个 subspace，我们也总能通过 Gram-Schmidt 或 SVD 找到一组 orthonormal basis 表示同一个 subspace。
    
    所以要求 $W$ 的行向量正交归一，只是在规范化表示方式，不是在缩小可表示的 subspace。

---

# 5. 先看 Rank-1 PCA

为了推导简单，先令：

$$
k=1
$$

这时只有一个 principal direction：

$$
w\in\mathbb{R}^{1\times d}
$$

并且它是 unit vector：

$$
ww^T=1
$$

每个数据点只有一个 coefficient：

$$
z_i\in\mathbb{R}
$$

因此：

$$
X_i\approx z_iw
$$

loss 变成：

$$
\operatorname{Loss}(z,w)
=
\frac{1}{n}\sum_{i=1}^{n}\|X_i-z_iw\|_2^2
$$

## 5.1 展开 loss

对单个数据点：

$$
\|X_i-z_iw\|_2^2
=
(X_i-z_iw)(X_i-z_iw)^T
$$

展开：

$$
\|X_i-z_iw\|_2^2
=
X_iX_i^T-2z_iX_iw^T+z_i^2ww^T
$$

因为：

$$
ww^T=1
$$

所以：

$$
\|X_i-z_iw\|_2^2
=
X_iX_i^T-2z_iX_iw^T+z_i^2
$$

其中 $X_iX_i^T$ 和 $z,w$ 无关，可以看成常数项。

所以 loss 可以写成：

$$
\operatorname{Loss}(z,w)
=
C+\frac{1}{n}\sum_{i=1}^{n}
\left(-2z_iX_iw^T+z_i^2\right)
$$

## 5.2 固定 $w$，求最优 $z_i$

对 $z_i$ 求导：

$$
\frac{\partial}{\partial z_i}\operatorname{Loss}(z,w)
=
\frac{1}{n}\left(-2X_iw^T+2z_i\right)
$$

令导数等于 $0$：

$$
\frac{1}{n}\left(-2X_iw^T+2z_i\right)=0
$$

得到：

$$
z_i^*=X_iw^T
$$

也就是说，最优的 $z_i$ 就是 $X_i$ 在方向 $w$ 上的 projection。

!!! explanation "为什么 z_i 是 projection"
    $w$ 是一个 unit direction。
    
    $X_iw^T$ 就是 $X_i$ 和 $w$ 的点积，表示 $X_i$ 在 $w$ 方向上的坐标。
    
    所以 rank-1 PCA 的重建过程就是：先把每个点投影到一条线上，再从这条线投影回原空间。

## 5.3 把 $z_i^*$ 代回 loss

将：

$$
z_i^*=X_iw^T
$$

代回：

$$
\operatorname{Loss}(z,w)
=
C+\frac{1}{n}\sum_{i=1}^{n}
\left(-2z_iX_iw^T+z_i^2\right)
$$

得到：

$$
\operatorname{Loss}(w)
=
C-\frac{1}{n}\sum_{i=1}^{n}(X_iw^T)^2
$$

把它写成矩阵形式：

$$
\operatorname{Loss}(w)
=
C-w\Sigma w^T
$$

其中：

$$
\Sigma=\frac{1}{n}X^TX
$$

!!! explanation "这个式子在说什么"
    $C$ 是固定常数，所以最小化 loss 等价于最大化：
    
    $$
    w\Sigma w^T
    $$
    
    也就是说，PCA 找到的方向不是随便找一条线，而是找让投影后 spread 最大的方向。
    
    这就是常说的：PCA finds the direction of maximum variance。

---

# 6. 用 Lagrange Multiplier 求 $w$

## 6.1 为什么必须加 constraint

现在我们要最小化：

$$
\operatorname{Loss}(w)=C-w\Sigma w^T
$$

如果不限制 $w$ 的长度，那么可以把 $w$ 放大很多倍，使得：

$$
w\Sigma w^T
$$

变得越来越大，于是 loss 会越来越小，甚至没有下界。

所以必须加约束：

$$
ww^T=1
$$

这表示 $w$ 是 unit vector。

## 6.2 Lagrangian

构造 Lagrangian：

$$
\mathcal{L}(w,\lambda)
=
C-w\Sigma w^T+\lambda(ww^T-1)
$$

然后分别对 $w$ 和 $\lambda$ 求 stationary point。

对 $\lambda$ 求导会把 constraint 找回来：

$$
\frac{\partial}{\partial\lambda}\mathcal{L}(w,\lambda)
=ww^T-1=0
$$

所以：

$$
ww^T=1
$$

## 6.3 对 $w$ 求导

因为：

$$
\Sigma=\frac{1}{n}X^TX
$$

所以 $\Sigma$ 是 symmetric matrix。

对 $w\Sigma w^T$ 求导：

$$
\nabla_w(w\Sigma w^T)=2\Sigma w^T
$$

对 $ww^T$ 求导：

$$
\nabla_w(ww^T)=2w^T
$$

因此：

$$
\nabla_w\mathcal{L}(w,\lambda)
=
-2\Sigma w^T+2\lambda w^T
$$

令它等于 $0$：

$$
-2\Sigma w^T+2\lambda w^T=0
$$

得到：

$$
\Sigma w^T=\lambda w^T
$$

这就是 eigenvector equation。

!!! explanation "为什么突然出现 eigenvector"
    特征向量的定义就是：
    
    $$
    Av=\lambda v
    $$
    
    这里令：
    
    $$
    A=\Sigma,\quad v=w^T
    $$
    
    就得到：
    
    $$
    \Sigma w^T=\lambda w^T
    $$
    
    所以最优方向 $w$ 必须是 $\Sigma$ 的 eigenvector。

## 6.4 为什么取最大 eigenvalue

把 eigenvector equation 代回：

$$
\operatorname{Loss}(w)=C-w\Sigma w^T
$$

由于：

$$
\Sigma w^T=\lambda w^T
$$

所以：

$$
w\Sigma w^T=w(\lambda w^T)=\lambda ww^T
$$

又因为：

$$
ww^T=1
$$

因此：

$$
w\Sigma w^T=\lambda
$$

loss 变成：

$$
\operatorname{Loss}(w)=C-\lambda
$$

要让 loss 最小，就要让 $\lambda$ 最大。

所以：

$$
w^*=\text{eigenvector of }\Sigma\text{ with the largest eigenvalue}
$$

这就是 first principal component。

!!! definition "First Principal Component"
    第一主成分是：
    
    $$
    \Sigma=\frac{1}{n}X^TX
    $$
    
    对应最大 eigenvalue 的 unit eigenvector。

---

# 7. 多个 Principal Components

如果我们要降到 $k$ 维，就需要 $k$ 个 principal components。

第一个 principal component 是最大 eigenvalue 对应的 eigenvector。

第二个 principal component 要满足两个条件：

1. 它也要让 reconstruction error 尽可能小。
2. 它要和第一个 principal component 垂直。

也就是：

$$
w_2w_1^T=0
$$

继续用 Lagrange multiplier 推导，可以得到：

$$
\Sigma w_2^T=\lambda_2 w_2^T
$$

并且 $w_2$ 是第二大 eigenvalue 对应的 eigenvector。

推广到 $k$ 维：

$$
W=
\begin{bmatrix}
- & w_1 & -\\
- & w_2 & -\\
& \vdots &\\
- & w_k & -
\end{bmatrix}
$$

其中 $w_1,\dots,w_k$ 是 $\Sigma$ 的前 $k$ 个 eigenvectors，按 eigenvalue 从大到小排列。

低维坐标是：

$$
Z=XW^T
$$

!!! explanation "为什么 Z = XW^T"
    $W$ 的每一行都是一个 principal direction。
    
    对某个数据点 $X_i$，它在第 $j$ 个主成分上的坐标是：
    
    $$
    X_iw_j^T
    $$
    
    把所有 $j=1,\dots,k$ 的坐标拼起来，就是 $Z_i$。
    
    所以整体矩阵形式就是：
    
    $$
    Z=XW^T
    $$

---

# 8. 用 SVD 计算 PCA

直接构造：

$$
X^TX
$$

再做 eigendecomposition 是正确的，但是可能很贵。

实际中通常使用 SVD：

$$
X=USV^T
$$

在代码里可以写成：

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
```

其中 $V$ 的 columns 就是 $X^TX$ 的 eigenvectors。

因此 PCA 的 principal components 可以从 $V^T$ 里面拿到：

```python
W = Vt[:k, :]
Z = X @ W.T
```

!!! explanation "为什么 SVD 和 PCA 连在一起"
    如果：
    
    $$
    X=USV^T
    $$
    
    那么：
    
    $$
    X^TX=(USV^T)^T(USV^T)
    $$
    
    $$
    =VS^TU^TUSV^T
    $$
    
    因为 $U^TU=I$，所以：
    
    $$
    X^TX=VS^2V^T
    $$
    
    这正好就是 $X^TX$ 的 eigendecomposition。
    
    所以 $V$ 的 columns 是 eigenvectors，$S^2$ 对应 eigenvalues。

---

# 9. Explained Variance Ratio

PCA 每个 principal component 能解释多少数据 spread，可以用 explained variance ratio 表示。

如果 SVD 中的 singular values 是：

$$
S_{11},S_{22},\dots,S_{dd}
$$

那么第 $i$ 个 principal component 的 explained variance ratio 是：

$$
\frac{S_{ii}^2}{\sum_{j=1}^{d}S_{jj}^2}
$$

在 congressional votes demo 里：

$$
\text{PC1}:80.3\%
$$

$$
\text{PC2}:5.3\%
$$

两者合起来：

$$
85.6\%
$$

这说明只用二维就能解释数据中大部分 spread。

![](附件/Lecture3_explained_variance-37.png)

!!! explanation "为什么看 explained variance ratio"
    如果前几个 principal components 的 explained variance ratio 很高，说明数据主要变化确实集中在少数几个方向上。
    
    如果每个 component 都只能解释一点点，那么强行降到很低维就会丢掉很多信息。

---

# 10. 总结

PCA 的核心流程如下：

1. 先 center data：

$$
X_i\leftarrow X_i-\bar{x}
$$

2. 把降维写成矩阵分解：

$$
X\approx ZW
$$

3. 用 reconstruction error 定义目标函数：

$$
\operatorname{Loss}(Z,W)
=
\frac{1}{n}\sum_{i=1}^{n}\|X_i-Z_iW\|_2^2
$$

4. 固定 $w$ 时，最优 low-dimensional coordinate 是 projection：

$$
z_i^*=X_iw^T
$$

5. 最优 direction 会变成 eigenvector problem：

$$
\Sigma w^T=\lambda w^T
$$

6. principal components 是 $\Sigma=\frac{1}{n}X^TX$ 的前 $k$ 个 eigenvectors。

7. 低维坐标是：

$$
Z=XW^T
$$

!!! warning "PCA 到底在最大化什么"
    从 reconstruction error 的角度看，PCA 是在找能最小化重建误差的低维子空间。
    
    从 projection 的角度看，PCA 是在找让投影后 variance 最大的方向。
    
    这两个说法是等价的：保留的 spread 越多，丢掉的 reconstruction error 就越少。
