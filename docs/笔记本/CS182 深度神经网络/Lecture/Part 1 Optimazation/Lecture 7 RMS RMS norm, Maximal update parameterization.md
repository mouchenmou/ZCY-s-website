## 0. 这节课的各种 norm
由于这节课的各种 norm，vector norm 倒还好，matrix norm 的那一大堆我实在记不住，所以在这里列一个清单，顺便附上字面意思，这样子好记一点
## 0.1 Matrix $2$-norm/Spectral norm

Matrix $2$-norm **不是把矩阵元素平方求和**。它的 2 是说：**输入和输出都用 vector $L_2$ norm 来量。**

定义：

$$ \|W\|_2 = \max_{x\neq0} \frac{\|Wx\|_2}{\|x\|_2} $$

表示**这个 matrix 最多能把一个向量的 $L_2$ 长度放大多少倍？**

Spectral norm 和 matrix $2$-norm 是**同一个东西**。那么为什么叫 spectral？

因为它和矩阵的 singular values 有关：

$$\|W\|_2 = \sigma_{\max}(W) $$

“spectral” 这个词本来就和 eigenvalues / singular values 这种“谱”有关。**spectral norm = 看矩阵最强的那个伸缩方向。**因此：

$$\boxed{ \text{spectral norm} = \text{最大 singular value} = \text{最大 }L_2\text{ 放大倍数} }$$

## 0.2 RMS norm

RMS 全称：**Root Mean Square**。它是针对 vector 的 norm

逐字就是：

- Square：平方
- Mean：平均
- Root：开根号

所以：

$$\|x\|_{\mathrm{RMS}} = \sqrt{ \frac1d\sum_i x_i^2 } $$

字面意思就是：**平方 → 求平均 → 开根号**。

## 0.3 RMS-to-RMS norm

**输入 RMS $\rightarrow$ 输出 RMS**

定义：

$$\|W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}} = \max_{x\neq0} \frac{ \|Wx\|_{\mathrm{RMS}} }{ \|x\|_{\mathrm{RMS}} } $$


## 0.4 Frobenius norm

**把矩阵所有元素摊平，当成一个大 vector，然后算 $L_2$ norm。**

$$
 \|G\|_F = \sqrt{ \sum_{i,j}G_{ij}^2 } $$

## 最容易记的总表

|Norm|字面/直观意思|它在问什么|
|---|---|---|
|$L_p$ norm|第 $p$ 种长度定义|vector 有多大|
|$L_2$ norm|$p=2$|整个 vector 总长度|
|$L_\infty$ norm|$p\to\infty$|最大 coordinate 有多大|
|matrix $2$-norm|输入输出都用 $L_2$|matrix 最大能放大 $L_2$ 多少倍|
|spectral norm|“谱”上的 norm|最大 singular value|
|RMS norm|Root Mean Square|平均一个 coordinate 有多大|
|RMS-to-RMS norm|RMS 输入到 RMS 输出|matrix 最大能放大 RMS 多少倍|
|Frobenius norm|人名|矩阵所有元素整体有多大|

---

# 1. Local Linear Perspective：Optimizer 到底在做什么？

假设模型参数是 $\theta$，当前 loss 为：

$$
L(\theta)
$$

我们把参数稍微改变：

$$
\theta \rightarrow \theta+\Delta\theta
$$

其中 $\Delta\theta$ 就是这一步的 parameter update。

如果 $\Delta\theta$ 足够小，可以使用一阶 Taylor approximation：

$$
L(\theta+\Delta\theta)
\approx
L(\theta)
+
\langle \nabla_\theta L,\Delta\theta\rangle
$$

因此：

$$
\Delta L
\approx
\langle \nabla_\theta L,\Delta\theta\rangle
$$

这里：

$$
\boxed{
\langle \nabla_\theta L,\Delta\theta\rangle
}
$$

表示这次 parameter update 对 loss 造成的**近似变化量**。

如果：

$$
\langle \nabla_\theta L,\Delta\theta\rangle<0
$$

说明 loss 会下降。

所以 optimizer 想做的事情就是：

$$
\min_{\Delta\theta}
\langle \nabla_\theta L,\Delta\theta\rangle
$$

但是不能让 $\Delta\theta$ 无限大，否则这个内积可以无限变小，而且 Taylor approximation 也会失效。

因此必须限制：

$$
\|\Delta\theta\|\le\eta
$$

这里的 $\eta$ 在这个 constrained optimization 视角里，更适合理解为：

> **一次 update 最大允许有多大，也就是 step-size budget / radius。**

它和传统意义的 learning rate 有联系，但此时先不要把它直接理解成普通 GD 公式里的 learning rate。

---

# 2. Vector Norm：什么叫“Update 不能太大”？

一般的 $L_p$ norm 定义为：

$$
\|x\|_p
=
\left(
\sum_i |x_i|^p
\right)^{1/p}
$$

不同 norm 对“向量有多大”的定义不同，因此会导出不同的 update。

## 2.1 $L_2$ Norm

$$
\boxed{
\|x\|_2
=
\sqrt{
\sum_i x_i^2
}
}
$$

它就是向量的普通欧氏长度。

例如：

$$
x=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

则：

$$
\|x\|_2=5
$$

因此：

$$
\|\Delta\theta\|_2\le\eta
$$

表示：

> 整个 update 向量的欧氏长度不能超过 $\eta$。

## 2.2 $L_\infty$ Norm

$L_\infty$ norm 可以从 $L_p$ norm 的极限得到：

$$
\|x\|_\infty
=
\lim_{p\to\infty}
\left(
\sum_i|x_i|^p
\right)^{1/p}
$$

最终：

$$
\boxed{
\|x\|_\infty
=
\max_i|x_i|
}
$$

原因是 $p$ 越大，最大的分量的 $p$ 次方越占主导。

例如：

$$
x=
\begin{bmatrix}
1\\
2\\
10
\end{bmatrix}
$$

当 $p$ 很大时：

$$
1^p+2^p+10^p
\approx
10^p
$$

因此：

$$
\|x\|_\infty=10
$$

所以：

$$
\boxed{
\|\Delta\theta\|_\infty\le\eta
}
$$

等价于：

$$
|\Delta\theta_i|\le\eta,\quad \forall i
$$

也就是：

> 每一个参数在这一步最多只能改变 $\eta$。

## 2.3 $L_\infty$ Constraint $\rightarrow$ Sign SGD

考虑：

$$
\min_{\Delta\theta}
\langle\nabla L,\Delta\theta\rangle
$$

subject to：

$$
\|\Delta\theta\|_\infty\le\eta
$$

因为：

$$
|\Delta\theta_i|\le\eta
$$

所以每一个参数都可以独立地在：

$$
[-\eta,\eta]
$$

之间选择。

假设：

$$
\nabla L=
\begin{bmatrix}
2\\
-3\\
0.5
\end{bmatrix}
$$

则：

$$
\langle\nabla L,\Delta\theta\rangle
=
2\Delta\theta_1
-3\Delta\theta_2
+0.5\Delta\theta_3
$$

为了让它最小：

$$
\Delta\theta_1=-\eta,\qquad
\Delta\theta_2=+\eta,\qquad
\Delta\theta_3=-\eta
$$

因此：

$$
\boxed{
\Delta\theta
=
-\eta\operatorname{sign}(\nabla L)
}
$$

这就是 Sign SGD。

### 什么是 sign？

符号函数：

$$
\operatorname{sign}(x)
=
\begin{cases}
1,&x>0\\
0,&x=0\\
-1,&x<0
\end{cases}
$$

Sign SGD 的特点是：

> 不看 gradient magnitude，只看正负号。


## 2.4 $L_2$ Constraint 与 Gradient Descent

现在换成：

$$
\|\Delta\theta\|_2\le\eta
$$

优化问题为：

$$
\min_{\Delta\theta}
\langle\nabla L,\Delta\theta\rangle
$$

subject to：

$$
\|\Delta\theta\|_2\le\eta
$$

根据：

$$
a^\top b
=
\|a\|_2\|b\|_2\cos\phi
$$

要让内积最小，需要：

$$
\cos\phi=-1
$$

也就是 $\Delta\theta$ 与 gradient 完全反向。

因此：

$$
\boxed{
\Delta\theta
=
-\eta
\frac{\nabla L}{\|\nabla L\|_2}
}
$$

这里 gradient 被归一化，所以：

$$
\|\Delta\theta\|_2=\eta
$$

也就是说：

> 不管 gradient 本身有多大，一步的总长度固定为 $\eta$。

### 2.4.1 从 $L_2$ Regularization 得到普通 Gradient Descent

考虑：

$$
\min_{\Delta\theta}
\left[
\langle\nabla L,\Delta\theta\rangle
+
\lambda\|\Delta\theta\|_2^2
\right]
$$

对 $\Delta\theta$ 求导：

$$
\nabla L+2\lambda\Delta\theta=0
$$

因此：

$$
\Delta\theta
=
-\frac{1}{2\lambda}\nabla L
$$

如果定义：

$$
\alpha=\frac{1}{2\lambda}
$$

就得到：

$$
\boxed{
\Delta\theta=-\alpha\nabla L
}
$$

这就是 ordinary Gradient Descent。

---

# 3. Matrix Parameterization 与 Spectral Norm

神经网络参数天然通常是矩阵。

例如线性层：

$$
h=Wx
$$

其中：

$$
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}
$$

于是可以直接在 matrix space 中定义 update：

$$
\theta\rightarrow W,\qquad
\Delta\theta\rightarrow\Delta W,\qquad
\nabla_\theta L\rightarrow\nabla_WL
$$

优化问题变为：

$$
\min_{\Delta W}
\langle\nabla_WL,\Delta W\rangle
$$

subject to 某种 matrix norm constraint。

## 3.1 Matrix Inner Product

对于矩阵 $A,B$：

$$
\boxed{
\langle A,B\rangle
=
\sum_{i,j}A_{ij}B_{ij}
}
$$

也可以写成：

$$
\boxed{
\langle A,B\rangle
=
\operatorname{tr}(AB^\top)
}
$$

因此：

$$
\langle\nabla_WL,\Delta W\rangle
$$

表示：

> 每一个 weight gradient 乘以对应的 weight update，再全部加起来，也就是 loss 的一阶近似变化量。

即：

$$
L(W+\Delta W)
\approx
L(W)
+
\langle\nabla_WL,\Delta W\rangle
$$

## 3.2 Spectral Norm：矩阵的 $2$-Norm

对于矩阵 $W$：

$$
\boxed{
\|W\|_2
}
$$

叫做 **matrix induced $2$-norm**，也叫 **spectral norm**。

它定义为：

$$
\boxed{
\|W\|_2
=
\max_{x\neq0}
\frac{\|Wx\|_2}{\|x\|_2}
}
$$

等价于：

$$
\boxed{
\|W\|_2
=
\max_{\|x\|_2=1}
\|Wx\|_2
}
$$

这里的 $x$ 是一个单独的向量，不是整个 batch 的 data matrix。

它的含义是：

> 对所有单位向量 $x$，矩阵 $W$ 最多能把它的长度放大多少倍。

### 3.2.1 为什么下标是 2？

因为定义中输入和输出都使用 vector $L_2$ norm：

$$
\frac{\|Wx\|_2}{\|x\|_2}
$$

所以叫 matrix $2$-norm。

### 3.2.2 Spectral Norm 与 Singular Value

如果：

$$
W=U\Sigma V^\top
$$

则：

$$
\boxed{
\|W\|_2
=
\sigma_{\max}(W)
}
$$

因此 spectral norm 可以理解成：

> 矩阵在所有方向中的最大伸缩倍率。

注意：

> Spectral norm 本身只表示 $\|W\|_2=\sigma_{\max}(W)$。

后面出现的：

$$
\Delta W^*=-\eta UV^\top
$$

不是 spectral norm 的定义，而是使用 spectral norm 作为 constraint 后求出来的最优 update。

## 3.3 Spectral-Norm Constraint

考虑：

$$
\min_{\Delta W}
\langle\nabla_WL,\Delta W\rangle
$$

subject to：

$$
\|\Delta W\|_2\le\eta
$$

假设 gradient 的 SVD：

$$
\nabla_WL
=
U\Sigma V^\top
$$

其中：

$$
\Sigma
=
\operatorname{diag}(\sigma_1,\sigma_2,\ldots)
$$

### 3.3.1 为什么要引入 $B$？

定义：

$$
B=U^\top\Delta W V
$$

所以：

$$
\Delta W=UBV^\top
$$

这只是把 $\Delta W$ 换到 gradient 的 singular-vector coordinate system 中，并不是说 $B$ 本身是 diagonal matrix。

### 3.3.2 为什么 $\langle U\Sigma V^\top,UBV^\top\rangle=\langle\Sigma,B\rangle$？

根据：

$$
\langle A,C\rangle
=
\operatorname{tr}(AC^\top)
$$

有：

$$
\langle U\Sigma V^\top,UBV^\top\rangle
=
\operatorname{tr}
\left[
(U\Sigma V^\top)(UBV^\top)^\top
\right]
$$

因为：

$$
(UBV^\top)^\top
=
VB^\top U^\top
$$

所以：

$$
=
\operatorname{tr}
\left[
U\Sigma V^\top VB^\top U^\top
\right]
$$

利用：

$$
V^\top V=I
$$

得到：

$$
=
\operatorname{tr}
\left[
U\Sigma B^\top U^\top
\right]
$$

再利用 trace 的循环性质：

$$
\operatorname{tr}(ABC)=\operatorname{tr}(BCA)
$$

于是：

$$
=
\operatorname{tr}
\left[
\Sigma B^\top U^\top U
\right]
$$

因为：

$$
U^\top U=I
$$

所以：

$$
=
\operatorname{tr}(\Sigma B^\top)
=
\langle\Sigma,B\rangle
$$

因此：

$$
\boxed{
\langle U\Sigma V^\top,UBV^\top\rangle
=
\langle\Sigma,B\rangle
}
$$

## 3.4 为什么最优 Update 是 $-\eta UV^\top$？

如果：

$$
\Sigma=
\begin{bmatrix}
\sigma_1&0\\
0&\sigma_2
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

那么：

$$
\langle\Sigma,B\rangle
=
\sigma_1a+\sigma_2d
$$

我们希望这个量尽可能小，同时：

$$
\|B\|_2\le\eta
$$

由于：

$$
\sigma_i\ge0
$$

最优时可以取：

$$
a=-\eta,\qquad d=-\eta
$$

即：

$$
B^*=-\eta I
$$

然后：

$$
\Delta W^*
=
UB^*V^\top
$$

于是：

$$
\boxed{
\Delta W^*
=
-\eta UV^\top
}
$$

这里要注意：

- $\nabla_WL=U\Sigma V^\top$ 是已经给定的 gradient matrix；
- $\Delta W$ 是我们还没有确定、正在优化的 update matrix；
- 我们不是人为修改一个固定矩阵的奇异值，而是在所有满足 constraint 的候选 $\Delta W$ 中寻找最优的那个。

## 3.5 $U\Sigma V^\top\rightarrow UV^\top$ 的意义

gradient：

$$
G=U\Sigma V^\top
$$

如果直接用 gradient descent：

$$
\Delta W\propto-U\Sigma V^\top
$$

那么大的 singular value 会严重主导 update。

例如：

$$
\Sigma=
\begin{bmatrix}
100&0\\
0&0.01
\end{bmatrix}
$$

两个 singular directions 的 update strength 差了：

$$
10000
$$

倍。

而：

$$
UV^\top
=
UIV^\top
$$

相当于：

$$
\sigma_i\rightarrow1
$$

因此不同 singular directions 的 scale 被拉平。

### 3.5.1 Condition Number

对于 full-rank matrix：

$$
\kappa
=
\frac{\sigma_{\max}}{\sigma_{\min}}
$$

例如：

$$
\Sigma=
\begin{bmatrix}
100&0\\
0&0.01
\end{bmatrix}
$$

则：

$$
\kappa=10000
$$

而 $UV^\top$ 的非零 singular values 都是 1，所以在相应非零 singular subspace 上：

$$
\kappa=1
$$

表示各方向 scale 最均衡。

---

# 4. Xavier Initialization 与 RMS Norm

到这里老师已经建立：

$$
\boxed{
\text{Choose a norm}
\rightarrow
\text{Get an optimizer}
}
$$

接下来真正的问题是：

> **对于 deep neural network，到底什么 norm 才最自然？**

这就是为什么老师重新回到 Xavier initialization。

Xavier 的作用不是这节课的终点，而是为了说明：

> 神经网络里我们真正关心的是“每个 activation 的典型 scale 是否保持稳定”。

这会自然引出 RMS norm。

## 4.1 Xavier Initialization

考虑一层：

$$
h=Wx
$$

第 $j$ 个输出：

$$
h_j
=
\sum_{i=1}^{d_{\text{in}}}
W_{ji}x_i
$$

假设：

$$
\mathbb E[W_{ji}]=0
$$

$$
\mathbb E[x_i]=0
$$

并假设各项近似独立。

### 4.1.1 为什么方差可以相加？

一般情况下：

$$
\operatorname{Var}(X+Y)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y)
+
2\operatorname{Cov}(X,Y)
$$

如果 $X,Y$ 独立，则：

$$
\operatorname{Cov}(X,Y)=0
$$

因此：

$$
\operatorname{Var}(X+Y)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y)
$$

所以：

$$
\operatorname{Var}(h_j)
=
\sum_i
\operatorname{Var}(W_{ji}x_i)
$$

### 4.1.2 为什么 $\operatorname{Var}(W_{ji}x_i)$ 等于方差的乘积？

如果 $W_{ji}$ 和 $x_i$ 独立且均值为 0：

$$
\operatorname{Var}(W_{ji}x_i)
=
\operatorname{Var}(W_{ji})
\operatorname{Var}(x_i)
$$

设：

$$
\operatorname{Var}(W_{ji})=v_w
$$

$$
\operatorname{Var}(x_i)=v_x
$$

一共有 $d_{\text{in}}$ 项：

$$
\operatorname{Var}(h_j)
=
d_{\text{in}}v_wv_x
$$

为了让 forward signal 的 scale 保持稳定，希望：

$$
\operatorname{Var}(h_j)
\approx
v_x
$$

因此：

$$
d_{\text{in}}v_wv_x
\approx
v_x
$$

得到：

$$
v_w
\approx
\frac{1}{d_{\text{in}}}
$$

所以：

$$
\boxed{
\operatorname{Std}(W_{ji})
\approx
\frac{1}{\sqrt{d_{\text{in}}}}
}
$$

经典 Xavier 还会同时兼顾 backward flow，因此常见形式是：

$$
\operatorname{Var}(W_{ji})
=
\frac{2}{d_{\text{in}}+d_{\text{out}}}
$$

因此：

$$
\operatorname{Std}(W_{ji})
=
\sqrt{
\frac{2}{d_{\text{in}}+d_{\text{out}}}
}
$$

## 4.2 为什么 Xavier 会引出 RMS Norm？

Xavier 想维持的不是整个向量的总长度，而是：

> 每个 activation 的典型 magnitude。

普通 $L_2$ norm 会随着维度增加而自然变大。

因此定义 RMS norm：

$$
\boxed{
\|x\|_{\mathrm{RMS}}
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}x_i^2
}
}
$$

也就是：

$$
\boxed{
\|x\|_{\mathrm{RMS}}
=
\frac{1}{\sqrt d}\|x\|_2
}
$$

它衡量的是：

> 一个典型 coordinate 的 magnitude。

如果所有元素都是 1，那么无论 $d=10$ 还是 $d=10000$：

$$
\|x\|_{\mathrm{RMS}}=1
$$

因此 RMS norm 比普通 $L_2$ norm 更适合描述 neural network 中 activation 的典型 scale。

## 4.3 RMS-to-RMS Matrix Norm

对于：

$$
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}
$$

定义：

$$
\boxed{
\|W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
=
\max_{x\neq0}
\frac{
\|Wx\|_{\mathrm{RMS}}
}{
\|x\|_{\mathrm{RMS}}
}
}
$$

它表示：

> 一个 matrix 最多能把输入的 RMS scale 放大多少倍。

因为：

$$
\|x\|_{\mathrm{RMS}}
=
\frac{\|x\|_2}{\sqrt{d_{\text{in}}}}
$$

而：

$$
\|Wx\|_{\mathrm{RMS}}
=
\frac{\|Wx\|_2}{\sqrt{d_{\text{out}}}}
$$

所以：

$$
\frac{
\|Wx\|_{\mathrm{RMS}}
}{
\|x\|_{\mathrm{RMS}}
}
=
\sqrt{
\frac{d_{\text{in}}}{d_{\text{out}}}
}
\frac{
\|Wx\|_2
}{
\|x\|_2
}
$$

取最大值：

$$
\boxed{
\|W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
=
\sqrt{
\frac{d_{\text{in}}}{d_{\text{out}}}
}
\|W\|_2
}
$$

所以：

> **RMS-to-RMS matrix norm 本质上就是 spectral norm 乘上一个由 fan-in / fan-out 决定的 scaling factor。**

这是本节课最重要的公式之一。

---

# 5. Maximal Update Parameterization

如果要求：

$$
\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
\le\eta
$$

则：

$$
\sqrt{
\frac{d_{\text{in}}}{d_{\text{out}}}
}
\|\Delta W\|_2
\le\eta
$$

所以：

$$
\boxed{
\|\Delta W\|_2
\le
\eta
\sqrt{
\frac{d_{\text{out}}}{d_{\text{in}}}
}
}
$$

这意味着：

> 即使整个 network 只使用一个全局 $\eta$，不同 layer 因为 $d_{\text{in}}$ 和 $d_{\text{out}}$ 不同，实际允许的 update scale 也会不同。

也就是说：

$$
\text{effective update scale}
\propto
\sqrt{
\frac{d_{\text{out}}}{d_{\text{in}}}
}
$$

这就是课件所说的 maximal update parameterization 的 essential idea：

> **同一个 global hyperparameter，通过参数化方式让不同宽度的 layer 自动得到合适的 update scale。**

## 5.1 从 RMS-to-RMS Norm 到 Muon

定义：

$$
\gamma
=
\eta
\sqrt{
\frac{d_{\text{out}}}{d_{\text{in}}}
}
$$

那么：

$$
\|\Delta W\|_2\le\gamma
$$

根据前面的 spectral-norm constrained solution：

$$
\Delta W^*
=
-\gamma UV^\top
$$

因此：

$$
\boxed{
\Delta W^*
=
-\eta
\sqrt{
\frac{d_{\text{out}}}{d_{\text{in}}}
}
UV^\top
}
$$

Muon 的核心思想之一就是：

$$
G=U\Sigma V^\top
$$

不直接保留原始 singular values $\Sigma$，而希望得到：

$$
UV^\top
$$

即：

$$
\Sigma\rightarrow I
$$

从而让不同 singular directions 的 update 更均衡。

---

# 6. Muon 与 Newton–Schulz

## 6.1 Muon 是什么？

**Muon 全称 Momentum Orthogonalized by Newton–Schulz，是一个 optimizer。**

它和 SGD、Adam 属于同一类概念：

> Muon 决定在训练神经网络时，拿到 gradient 以后，应该怎样更新参数。

假设某一层参数矩阵的 gradient 是：

$$
G=\nabla_WL
$$

对 $G$ 做 SVD：

$$
G=U\Sigma V^\top
$$

其中 $\Sigma$ 中存放 singular values：

$$
\Sigma=
\operatorname{diag}(\sigma_1,\sigma_2,\ldots)
$$

如果直接使用普通 gradient descent，那么 update direction 与：

$$
-U\Sigma V^\top
$$

成正比。

问题是：

> 如果不同 singular values 差别很大，那么 update 会被最大的 singular directions 强烈支配。

例如：

$$
\Sigma=
\begin{bmatrix}
100&0\\
0&0.01
\end{bmatrix}
$$

那么两个 singular directions 的 scale 相差：

$$
\frac{100}{0.01}=10000
$$

倍。

Muon 的核心思想之一就是：

> **尽量把 gradient matrix 的不同 singular directions 拉到更接近的 scale。**

也就是希望：

$$
G
=
U\Sigma V^\top
$$

变成接近：

$$
UV^\top
$$

因为：

$$
UV^\top
=
UIV^\top
$$

相当于：

$$
\Sigma\rightarrow I
$$

也就是让非零 singular values 尽量靠近 1。

所以可以暂时把 Muon 理解成：

> **Muon 是一个 optimizer，它会对 gradient matrix 做近似 orthogonalization / singular-value flattening，使不同 singular directions 的 update 更均衡。**

## 6.2 Newton–Schulz 是什么？

**Newton–Schulz 的作用是**帮助 Muon 在不显式做完整 SVD 的情况下，近似得到 $UV^\top$。**

理论上：

$$
G=U\Sigma V^\top
$$

如果我们想得到：

$$
UV^\top
$$

最直接的方法就是：

$$
G
\rightarrow
\text{SVD}
\rightarrow
U,\Sigma,V
\rightarrow
UV^\top
$$

但是问题是：

> SVD 本身是比较昂贵的 matrix decomposition。

深度学习训练中：

- matrix 很大；
- layer 很多；
- optimizer 每一步都要运行。

所以如果每一个 training step、每一层都完整做一次 SVD，计算成本会很高。

因此 Muon 希望：

> 不显式求出 $U,\Sigma,V$，而是直接从 $G$ 本身出发，用便宜的 matrix multiplication 近似得到 $UV^\top$。

这就是 Newton–Schulz 出现的原因。


## 6.3 Newton–Schulz 如何把 Singular Values 推向 1？

老师使用的简化 polynomial 是：

$$
p(x)
=
\frac32x-\frac12x^3
$$

对于 matrix，使用：

$$
\boxed{
p(G)
=
\frac32G
-
\frac12GG^\top G
}
$$

然后反复迭代：

$$
\boxed{
G_{k+1}
=
\frac32G_k
-
\frac12G_kG_k^\top G_k
}
$$

### 6.3.1 为什么这个操作会修改 Singular Values？

假设：

$$
G=U\Sigma V^\top
$$

那么：

$$
G^\top
=
V\Sigma^\top U^\top
$$

所以：

$$
GG^\top G
=
(U\Sigma V^\top)
(V\Sigma^\top U^\top)
(U\Sigma V^\top)
$$

利用：

$$
V^\top V=I
$$

以及：

$$
U^\top U=I
$$

得到：

$$
GG^\top G
=
U\Sigma\Sigma^\top\Sigma V^\top
$$

对于每一个 singular value 来说，这相当于：

$$
\sigma_i
\rightarrow
\sigma_i^3
$$

因此：

$$
p(G)
=
\frac32G-\frac12GG^\top G
$$

就等价于：

$$
p(G)
=
U
\left(
\frac32\Sigma-\frac12\Sigma^3
\right)
V^\top
$$

也就是：

$$
\boxed{
p(G)
=
U\,p(\Sigma)\,V^\top
}
$$

所以：

> **直接对 matrix $G$ 做这个 polynomial operation，会保持 singular vectors，同时把 singular values 按 $p(\sigma)$ 修改。**

---

### 6.4.2 为什么 Singular Values 会向 1 靠近？

对于：

$$
p(x)
=
\frac32x-\frac12x^3
$$

如果：

$$
0<x<1
$$

那么：

$$
p(x)>x
$$

并且迭代会逐渐靠近 1。

![](附件/Pasted%20image%2020260929175808.png)

## 6.5 为什么 Newton–Schulz 之前必须 Normalize？

Newton–Schulz 并不是对任意大的 singular value 都稳定。

例如：

$$
p(2)
=
\frac32\times2-\frac12\times2^3
=
3-4
=
-1
$$

而：

$$
p(3)
=
\frac32\times3-\frac12\times3^3
=
4.5-13.5
=
-9
$$

可以看到：

> 如果 singular value 太大，iteration 可能直接变得非常不稳定。

所以在 Newton–Schulz 之前，需要先把 gradient matrix 缩小，让 singular values 落到比较安全的范围。

这就引出了 Frobenius norm。

## 6.6 Frobenius Norm 是什么？

对于矩阵：

$$
G=
\begin{bmatrix}
g_{11}&g_{12}&\cdots\\
g_{21}&g_{22}&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix}
$$

Frobenius norm 定义为：

$$
\boxed{
\|G\|_F
=
\sqrt{
\sum_{i,j}G_{ij}^2
}
}
$$

直观上就是：

> **把整个矩阵摊平成一个长向量，然后计算这个长向量的普通 $L_2$ norm。**

例如：

$$
G=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
$$

那么：

$$
\|G\|_F
=
\sqrt{
1^2+2^2+3^2+4^2
}
=
\sqrt{30}
$$

所以 Frobenius norm 衡量的是：

> **整个矩阵所有元素的总体 magnitude。**


### 6.6.1 为什么 $\|G\|_F=\sqrt{\sum_i\sigma_i^2}$？

这是 Frobenius norm 和 singular values 之间的一个重要性质。

首先：

$$
\|G\|_F^2
=
\sum_{i,j}G_{ij}^2
$$

而所有 matrix entries 的平方和，也可以写成：

$$
\boxed{
\|G\|_F^2
=
\operatorname{tr}(G^\top G)
}
$$

为什么？

因为 $G^\top G$ 的对角元素分别是 $G$ 每一列元素的平方和。

把这些对角元素全部加起来：

$$
\operatorname{tr}(G^\top G)
$$

正好就是：

$$
\sum_{i,j}G_{ij}^2
$$

现在对 $G$ 做 SVD：

$$
G=U\Sigma V^\top
$$

那么：

$$
G^\top
=
V\Sigma^\top U^\top
$$

于是：

$$
G^\top G
=
V\Sigma^\top U^\top U\Sigma V^\top
$$

因为：

$$
U^\top U=I
$$

所以：

$$
G^\top G
=
V\Sigma^\top\Sigma V^\top
$$

因此：

$$
\operatorname{tr}(G^\top G)
=
\operatorname{tr}
\left(
V\Sigma^\top\Sigma V^\top
\right)
$$

利用 trace 的循环性质：

$$
\operatorname{tr}(ABC)
=
\operatorname{tr}(BCA)
$$

得到：

$$
=
\operatorname{tr}
\left(
\Sigma^\top\Sigma V^\top V
\right)
$$

因为：

$$
V^\top V=I
$$

所以：

$$
=
\operatorname{tr}
\left(
\Sigma^\top\Sigma
\right)
$$

而 $\Sigma^\top\Sigma$ 的对角线就是：

$$
\sigma_1^2,\sigma_2^2,\ldots
$$

因此：

$$
\operatorname{tr}
\left(
\Sigma^\top\Sigma
\right)
=
\sum_i\sigma_i^2
$$

所以：

$$
\boxed{
\|G\|_F^2
=
\sum_i\sigma_i^2
}
$$

最后两边开根号：

$$
\boxed{
\|G\|_F
=
\sqrt{
\sum_i\sigma_i^2
}
}
$$

因此 Frobenius norm 有两种等价理解：

$$
\boxed{
\text{所有 matrix entries 平方和开根号}
}
$$

和：

$$
\boxed{
\text{所有 singular values 平方和开根号}
}
$$


### 6.6.2 关键性质：为什么 $\|G\|_2\le\|G\|_F$？

我们已经知道：

$$
\|G\|_2
=
\sigma_{\max}(G)
$$

同时：

$$
\|G\|_F
=
\sqrt{
\sigma_1^2+\sigma_2^2+\cdots
}
$$

假设最大的 singular value 是：

$$
\sigma_{\max}
$$

显然：

$$
\sigma_1^2+\sigma_2^2+\cdots
\ge
\sigma_{\max}^2
$$

所以：

$$
\sqrt{
\sigma_1^2+\sigma_2^2+\cdots
}
\ge
\sigma_{\max}
$$

因此：

$$
\boxed{
\|G\|_2
\le
\|G\|_F
}
$$


---

### 6.6.3 为什么 Frobenius Norm 可以用来 Normalize？

Newton–Schulz 希望 singular values 不要太大。

但我们又不想先去计算：

$$
\sigma_{\max}
$$

因为那通常又需要 SVD 或类似的昂贵计算。

所以直接使用容易计算的 Frobenius norm：

$$
G_0
=
\frac{G}{\|G\|_F}
$$

矩阵整体除以一个常数后，它的所有 singular values 也会除以同一个常数：

$$
\sigma_i(G_0)
=
\frac{
\sigma_i(G)
}{
\|G\|_F
}
$$

而我们已经知道：

$$
\sigma_{\max}(G)
=
\|G\|_2
\le
\|G\|_F
$$

所以：

$$
\frac{
\sigma_{\max}(G)
}{
\|G\|_F
}
\le1
$$

因此：

$$
\boxed{
\sigma_{\max}(G_0)\le1
}
$$

既然最大的 singular value 都不超过 1，那么其他 singular values 当然也都不超过 1。

所以：

> **Frobenius normalization 的作用，就是不用显式知道 singular values，也能保证最大的 singular value 不超过 1。**

---

### 6.6.4 为什么我们不需要先知道 Singular Values？

原始 singular values 的确是 matrix 本身的性质，我们不能手动指定。

但是：

$$
\|G\|_F
=
\sqrt{
\sum_{i,j}G_{ij}^2
}
$$

可以直接从 matrix entries 算出来。

不需要做 SVD。

而由于：

$$
\sigma_{\max}(G)
\le
\|G\|_F
$$

所以即使我们不知道：

$$
\sigma_{\max}
$$

具体是多少，也可以通过：

$$
G_0
=
\frac{G}{\|G\|_F}
$$

保证：

$$
\sigma_{\max}(G_0)\le1
$$

---

# 7. 本节课的完整主线

这节课更符合标题的逻辑是：

$$
\boxed{
\text{Local linear perspective}
\rightarrow
\text{Choose a norm}
\rightarrow
\text{Get an optimizer}
}
$$

例如：

$$
L_\infty
\rightarrow
\text{Sign SGD}
$$

以及：

$$
L_2
\rightarrow
\text{Gradient Descent geometry}
$$

再把 parameter 从 vector 换成 matrix：

$$
W
$$

引入 spectral norm：

$$
\boxed{
\|W\|_2
=
\sigma_{\max}(W)
}
$$

在 spectral-norm constraint 下得到：

$$
\boxed{
\Delta W^*
=
-\eta UV^\top
}
$$

然后真正进入本节课标题的重点：

$$
\boxed{
\text{Xavier}
\rightarrow
\text{RMS norm}
\rightarrow
\text{RMS-to-RMS norm}
}
$$

其中：

$$
\|x\|_{\mathrm{RMS}}
=
\frac{\|x\|_2}{\sqrt d}
$$

以及：

$$
\boxed{
\|W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
=
\sqrt{
\frac{d_{\text{in}}}{d_{\text{out}}}
}
\|W\|_2
}
$$

从而：

$$
\boxed{
\|\Delta W\|_2
\le
\eta
\sqrt{
\frac{d_{\text{out}}}{d_{\text{in}}}
}
}
$$

这说明：

> 同一个 global hyperparameter $\eta$，会因为 layer shape 不同而自然产生不同的 effective update scale。

这就是 maximal update parameterization 的核心直觉。

最后：

$$
\boxed{
\mu\text{P}
\rightarrow
\text{Muon}
}
$$

Muon 希望：

$$
G=U\Sigma V^\top
$$

变成：

$$
UV^\top
$$

也就是：

$$
\Sigma\rightarrow I
$$

为了避免显式 SVD，使用 Newton–Schulz：

$$
\boxed{
G_{k+1}
=
\frac32G_k
-
\frac12G_kG_k^\top G_k
}
$$

并在迭代前先做：

$$
\boxed{
G_0
=
\frac{G}{\|G\|_F}
}
$$

把 singular values 放进稳定范围。

---

# 8. 最后一句总结

这节课真正想表达的是：

$$
\boxed{
\text{对 DNN 来说，RMS scale 比单纯的 Euclidean scale 更自然。}
}
$$

因此从 RMS-to-RMS norm 出发，可以得到：

$$
\boxed{
\text{layer-width-aware update scaling}
}
$$

这就是 maximal update parameterization 的核心思想之一。

而 Muon 则进一步把这种 matrix update geometry 变成实际 optimizer：

$$
\boxed{
U\Sigma V^\top
\rightarrow
UV^\top
}
$$

从而减少 singular-value imbalance，同时用 Newton–Schulz 避免每一步显式做昂贵的 SVD。




