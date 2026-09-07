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

### 3.1.1 Loss 对 Z 求导

首先需要知道这两个公式：

$$公式1：∇_Z\ ​tr(AZ)=A^⊤\ ​​$$

$$公式2：\ ∇_Z\ ​tr(ZZ^⊤)=2Z$$
以及一个定理：

$$tr(AB)=tr(BA)$$

!!! explanation "证明公式1"

	假设
	
	$$A\in\mathbb R^{k\times n},\qquad Z\in\mathbb R^{n\times k}$$
	
	那么
	
	$$AZ\in\mathbb R^{k\times k}$$
	
	它的 trace 是对角线元素之和：
	
	$$\operatorname{tr}(AZ) = \sum_{i=1}^k (AZ)_{ii}$$
	
	而
	
	$$(AZ)_{ii} = \sum_{j=1}^n A_{ij}Z_{ji}$$
	
	所以
	
	$$\operatorname{tr}(AZ) = \sum_{i=1}^k\sum_{j=1}^n A_{ij}Z_{ji}$$
	
	现在我们对某一个元素 $Z_{pq}$ 求导。
	
	在上面的双重求和里，只有当 $j=p, i=q$ 时，才会出现 $Z_{pq}$。对应的那一项就是：
	
	$$A_{qp}Z_{pq}$$
	
	所以：
	
	$$\frac{\partial}{\partial Z_{pq}} \operatorname{tr}(AZ) = A_{qp}$$
	
	而梯度矩阵 $\nabla_Z$ 的第 $(p,q)$ 个元素就是这个导数：
	
	$$[\operatorname{tr}(AZ)]_{pq} = A_{qp}$$
	
	这恰好就是 $A^\top$ 的第 $(p,q)$ 个元素，因此：
	
	$$∇_Z\ ​tr(AZ)=A^⊤$$
	


!!! explanation "证明公式2"

	$tr(ZZ^T)$ 就是 $Z$ 里面所有元素的平方和，即：
	
	$$\operatorname{tr}(ZZ^\top) = \sum_{i,j} Z_{ij}^2$$
	
	因此我们对 $Z_{pq}$ 求导，得到的就是 $2Z_{pq}$，对其它的也是一样，因此：
	
	$公式2：\ ∇_Z\ ​tr(ZZ^⊤)=2Z$

!!! explanation "证明 $tr(AB)=tr(BA)$"

	假设：
	
	$$A\in\mathbb R^{m\times n},\ \  B\in\mathbb R^{n\times m}$$
	
	那么：
	
	$$AB\in\mathbb R^{m\times m}$$
	
	
	$$BA\in\mathbb R^{n\times n}$$
	
	
	虽然 AB 和 BA 的尺寸不同，但 trace 相同：
	
	$$\operatorname{tr}(AB) = \sum_{i=1}^{m}(AB)_{ii}$$
	
	
	$$(AB)_{ii} = \sum_{j=1}^{n}A_{ij}B_{ji}$$
	
	所以：
	
	$$\operatorname{tr}(AB) = \sum_{i=1}^{m}\sum_{j=1}^{n}A_{ij}B_{ji}$$
	
	再看：
	
	$$\operatorname{tr}(BA) = \sum_{j=1}^{n}(BA)_{jj}$$
	
	而：
	
	$$(BA)_{jj} = \sum_{i=1}^{m}B_{ji}A_{ij}$$
	
	所以：
	
	$$\operatorname{tr}(BA) = \sum_{j=1}^{n}\sum_{i=1}^{m}B_{ji}A_{ij}$$
	
	因为标量乘法满足：
	
	$$A_{ij}B_{ji}=B_{ji}A_{ij}$$
	
	而且双重求和换顺序不影响结果，所以：
	
	$$tr(AB)=tr(BA)​$$

### 求解 loss 函数对 Z 的导数

首先，根据 $tr(AB)=tr(BA)$ 可以得出：

$$tr(ZWX^T)=tr(WX^TZ)$$

再根据 $∇_Z\ ​tr(AZ)=A^⊤$，可以得出：

$$∇_Z\ ​tr(WX^TZ)=XW^T$$

最后根据 $\ ∇_Z\ ​tr(ZZ^⊤)=2Z$，得出：

$$∇_Z\ ​Loss(Z,W)=\frac{1}{n}[−2XW^⊤+2Z]$$

令梯度为 0，可以得到 $Z^*=XW^T$

