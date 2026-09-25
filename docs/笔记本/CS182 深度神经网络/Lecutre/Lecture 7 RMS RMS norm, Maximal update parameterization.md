
## 0. 这节课到底在讲什么

这节课的核心问题是：

> 已知当前的 gradient，我们应该如何选择 parameter update，才能让 loss 尽可能快地下降？

老师给出的统一视角是：

$$
\boxed{\text{Choose a norm} \rightarrow \text{Choose a step size} \rightarrow \text{Get an optimizer}}
$$

也就是说，很多 optimizer 的区别，本质上来自：

$$我们如何定义“一次 update 有多大”。$$

课程从参数向量上的 $L_\infty$ norm、$L_2$ norm 出发，再推广到矩阵参数上的 spectral norm，最后引出 Shampoo、RMS-to-RMS norm 和 Muon。

---

# 1. Local Linear Perspective：先把 Loss 局部线性化

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

因此 optimizer 想做的事情就是：

$$
\min_{\Delta\theta}
\langle \nabla_\theta L,\Delta\theta\rangle
$$

但是不能让 $\Delta\theta$ 无限大，否则可以直接让这个内积趋向 $-\infty$，而且 Taylor approximation 也会失效。

所以必须增加一个约束：

$$
\|\Delta\theta\|\le \eta
$$

这里的 $\eta$ 在这个 constrained optimization 视角中，首先表示：

$$一次 update 最大允许有多大$$

它和传统公式中的 learning rate 有联系，但这里更适合理解为 **step-size budget / radius**。

---

# 2. Vector Norm

一般的 $L_p$ norm 定义为：

$$
\|x\|_p
=
\left(
\sum_i |x_i|^p
\right)^{1/p}
$$

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

---

# 3. $L_\infty$ Constraint $\rightarrow$ Sign SGD

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

## 3.1 什么是 sign？

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

例如：

$$
\operatorname{sign}
\begin{pmatrix}
2\\
-5\\
0.1
\end{pmatrix}
=
\begin{pmatrix}
1\\
-1\\
1
\end{pmatrix}
$$

Sign SGD 的特点：

> 不看 gradient magnitude，只看正负号。

---

# 4. $L_2$ Constraint

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

---

# 5. 从 $L_2$ Regularization 得到普通 Gradient Descent

随后考虑：

$$
\min_{\Delta\theta}
\left[
\langle\nabla L,\Delta\theta\rangle
+
\lambda\|\Delta\theta\|_2^2
\right]
$$

第一项希望 loss 降得更多。

第二项：

$$
\lambda\|\Delta\theta\|_2^2
$$

惩罚过大的 update。

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

# 6. 为什么突然从 Vector 变成 Matrix？

神经网络参数天然通常是矩阵。

例如线性层：

$$
h=Wx
$$

其中：

$$
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}
$$


与其把所有参数摊平成一个 vector，不如直接在 matrix space 中定义 update，于是：

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

---

# 7. Matrix Inner Product

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

就是每一个 weight gradient 乘以对应的 weight update，再全部加起来。它依然表示 loss 的一阶近似变化量：

$$
L(W+\Delta W)
\approx
L(W)
+
\langle\nabla_WL,\Delta W\rangle
$$

---

# 8. Spectral Norm

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

含义：

> 对所有单位向量 $x$，矩阵 $W$ 最多能把它的长度放大多少倍。

## 8.1 为什么下标是 2？

因为定义中输入和输出都使用 vector $L_2$ norm：

$$
\frac{\|Wx\|_2}{\|x\|_2}
$$

所以叫 matrix $2$-norm。

## 8.2 Spectral Norm 与 Singular Value

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

也就是最大的 singular value。

因此 spectral norm 可以理解成：

> 矩阵在所有方向中的最大伸缩倍率。

---

# 9. Spectral-Norm Constraint

现在考虑：

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

## 9.1 换到 Singular-Vector Coordinate System

定义：

$$
B=U^\top\Delta W V
$$

所以：

$$
\Delta W=UBV^\top
$$

这只是换坐标系，并不是假设 $B$ 是 diagonal matrix。

利用 matrix inner product：

$$
\langle U\Sigma V^\top,UBV^\top\rangle
=
\langle\Sigma,B\rangle
$$

因为：

$$
V^\top V=I,\qquad U^\top U=I
$$

以及 trace 的循环性质。

如果：

$$
\Sigma=
\begin{bmatrix}
\sigma_1&0\\
0&\sigma_2
\end{bmatrix}
$$

而：

$$
B=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

则：

$$
\langle\Sigma,B\rangle
=
\sigma_1a+\sigma_2d
$$

非对角元素因为和 $\Sigma$ 的 0 相乘，不直接进入 objective。

---

# 10. 为什么最优 Update 是 $-\eta UV^\top$？

因为：

$$
\|B\|_2\le\eta
$$

而我们希望：

$$
\sigma_1a+\sigma_2d
$$

尽可能小。

由于：

$$
\sigma_i\ge0
$$

因此最优时希望：

$$
a=-\eta,\qquad d=-\eta
$$

于是可以取：

$$
B^*=-\eta I
$$

然后：

$$
\Delta W^*=UB^*V^\top
$$

因此：

$$
\boxed{
\Delta W^*
=
-\eta UV^\top
}
$$

注意：

> $UV^\top$ 并不是 spectral norm 的定义。

spectral norm 本身只是：

$$
\|W\|_2=\sigma_{\max}(W)
$$

而：

$$
\Delta W^*=-\eta UV^\top
$$

是**在 spectral norm constraint 下求出来的最优 update**。

---

# 11. $U\Sigma V^\top\rightarrow UV^\top$ 的意义

gradient：

$$
G=U\Sigma V^\top
$$

其中：

$$
\Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\ldots)
$$

如果直接用 gradient descent：

$$
\Delta W\propto-U\Sigma V^\top
$$

那么大的 singular value 对 update 影响特别大。

例如：

$$
\Sigma=
\begin{bmatrix}
100&0\\
0&0.01
\end{bmatrix}
$$

两个方向的 update strength 差了：

$$
10000
$$

倍。

而：

$$
UV^\top=UIV^\top
$$

相当于：

$$
\sigma_i\rightarrow1
$$

因此各个 singular direction 的 scale 被拉平。

---

# 12. Condition Number

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

说明不同方向尺度极度不均匀。

而 $UV^\top$ 的非零 singular values 都是 1，所以在相应非零 singular subspace 上：

$$
\kappa=1
$$

这表示各方向 scale 最均衡。

---

# 13. Xavier Initialization

对 deep neural network 来说，什么 norm 才最自然？

回顾 Xavier initialization。

考虑：

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

## 13.1 为什么方差可以相加？

一般：

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

所以：

$$
\operatorname{Var}(X+Y)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y)
$$

因此：

$$
\operatorname{Var}(h_j)
=
\sum_i
\operatorname{Var}(W_{ji}x_i)
$$

## 13.2 继续推导

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

这解释了：

> layer 越宽，每个 individual weight 初始化时应该越小。

经典 Xavier 还会同时考虑 backward flow，因此常见形式是：

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

---

# 14. RMS Norm

Xavier 想维持的是：

> 每个 activation 的典型 magnitude。

普通 $L_2$ norm 会随着维度增加而增加。

所以定义 RMS norm：

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

如果所有元素都是 1：

$$
\|x\|_{\mathrm{RMS}}=1
$$

无论 $d=10$ 还是 $d=10000$。

---

# 15. RMS-to-RMS Matrix Norm

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

所以 RMS-to-RMS matrix norm 本质上就是：

> spectral norm × 一个与 layer shape 有关的 scaling factor。

---

# 16. RMS-to-RMS Constraint 带来的 Layer-Specific Scaling

如果：

$$
\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
\le\eta
$$

那么：

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

因此即使全网络只设置一个 $\eta$，不同 layer 因为 $d_{\text{in}}$ 和 $d_{\text{out}}$ 不同，实际允许的 spectral-norm update 也不同。

---

# 17. Muon 的核心 Update

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

不直接使用原始 $\Sigma$，而希望得到近似：

$$
UV^\top
$$

即：

$$
\Sigma\rightarrow I
$$

从而让不同 singular directions 的 update 更均衡。

---

# 18. 为什么不能每一步直接算 SVD？

理论上：

$$
G=U\Sigma V^\top
$$

我们只需要：

$$
UV^\top
$$

但问题是：

> 要得到 $U$ 和 $V$，通常还是需要先做 SVD。

也就是：

$$
G
\rightarrow
\text{SVD}
\rightarrow
U,\Sigma,V
\rightarrow
UV^\top
$$

SVD 是比较昂贵的 matrix decomposition。

深度学习训练中：

- matrix 很大；
- layer 很多；
- optimizer 每一步都要执行。

所以每一步对每层做完整 SVD 成本过高。

因此 Muon 想：

> 不显式求 $U$ 和 $V$，能不能直接从 $G$ 本身近似算出 $UV^\top$？

这就引出了 Newton–Schulz iteration。

---

# 19. Newton–Schulz：不做 SVD，直接修改 Singular Values

目标：

$$
G=U\Sigma V^\top
$$

希望：

$$
G\rightarrow UV^\top
$$

也就是：

$$
\Sigma\rightarrow I
$$

即：

$$
\sigma_i\rightarrow1
$$

我们考虑：

$$
p(x)
=
\frac32x-\frac12x^3
$$

对于 matrix：

$$
\boxed{
p(G)
=
\frac32G
-
\frac12GG^\top G
}
$$

因为：

$$
G=U\Sigma V^\top
$$

所以：

$$
GG^\top G
=
U\Sigma^3V^\top
$$

于是：

$$
p(G)
=
U
\left(
\frac32\Sigma-\frac12\Sigma^3
\right)
V^\top
$$

即：

$$
\boxed{
p(G)
=
U\,p(\Sigma)\,V^\top
}
$$

关键意义：

> 不需要显式知道 $U,\Sigma,V$，直接对 $G$ 做这个 matrix polynomial，就等价于保持 singular vectors，同时把每个 singular value $\sigma_i$ 替换成 $p(\sigma_i)$。

---

# 20. Singular Value 如何靠近 1？

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

则反复迭代：

$$
x
\rightarrow
p(x)
\rightarrow
p(p(x))
\rightarrow
\cdots
$$

会向 1 靠近。

例如：

$$
0.5
\rightarrow
0.6875
\rightarrow
\cdots
\rightarrow1
$$

因此 matrix iteration：

$$
G_{k+1}
=
\frac32G_k
-
\frac12G_kG_k^\top G_k
$$

会让非零 singular values 逐渐趋向 1。

最终：

$$
G_k
\approx
UIV^\top
=
UV^\top
$$

![](附件/Pasted%20image%2020260924221757.png)

---

# 21. 为什么 Newton–Schulz 之前必须 Normalize？

如果 singular value 太大，这个 iteration 会不稳定。

例如：

$$
x=2
$$

则：

$$
p(2)=3-4=-1
$$

如果：

$$
x=3
$$

则：

$$
p(3)=4.5-13.5=-9
$$

所以不能直接把任意大的 singular values 扔进去。

---

# 22. Frobenius Norm

矩阵的 Frobenius norm：

$$
\boxed{
\|G\|_F
=
\sqrt{
\sum_{i,j}G_{ij}^2
}
}
$$

它也满足：

$$
\|G\|_F
=
\sqrt{
\sum_i\sigma_i^2
}
$$

关键性质：

$$
\boxed{
\|G\|_2\le\|G\|_F
}
$$

而：

$$
\|G\|_2
=
\sigma_{\max}(G)
$$

所以：

$$
\sigma_{\max}(G)
\le
\|G\|_F
$$

于是定义：

$$
G_0
=
\frac{G}{\|G\|_F}
$$

那么：

$$
\sigma_i(G_0)
=
\frac{\sigma_i(G)}{\|G\|_F}
$$

因此：

$$
\sigma_{\max}(G_0)\le1
$$

所以所有 singular values 都被压到 $[0,1]$ 内。

最关键的是：

> 我们不需要知道 singular values 是多少。

因为 $\|G\|_F$ 可以直接从 matrix entries 计算：

$$
\sqrt{\sum_{i,j}G_{ij}^2}
$$

无需 SVD。

---

# 23. 一个额外的重要问题：如果归一化后 singular value 太小怎么办？

如果矩阵很大，Frobenius norm 可能很大。

例如有很多 singular values：

$$
\sigma_1,\sigma_2,\ldots,\sigma_r
$$

则：

$$
\|G\|_F
=
\sqrt{
\sum_i\sigma_i^2
}
$$

所以归一化后某些 singular values 可能非常小：

$$
\tilde\sigma_i
=
\frac{\sigma_i}{\|G\|_F}
\ll1
$$

当 $x$ 很小时：

$$
p(x)
=
\frac32x-\frac12x^3
\approx
\frac32x
$$

所以：

$$
x\rightarrow1.5x
$$

收敛可能比较慢。

例如：

$$
0.01
\rightarrow
0.015
\rightarrow
0.0225
\rightarrow\cdots
$$

因此 Frobenius normalization 主要解决的是：

> 防止最大的 singular value 太大，导致 iteration 不稳定。

它并不能保证：

> 最小 singular value 一开始就足够大。

Muon 实际只做有限次数 iteration，因此并不要求所有 singular values 数学上精确变成 1。

另外，如果：

$$
\sigma_i=0
$$

那么：

$$
p(0)=0
$$

所以真正为 0 的 singular value 不会被变成 1。

---

# 24. 整节课最终逻辑

整节 Lecture 7 可以压缩成下面这条链：

$$
L(\theta+\Delta\theta)
\approx
L(\theta)
+
\langle\nabla L,\Delta\theta\rangle
$$

因此：

$$
\text{optimizer}
=
\text{寻找一个小的 }\Delta\theta\text{ 让 loss 降最多}
$$

然后：

$$
L_\infty\text{ constraint}
$$

得到：

$$
\boxed{
\Delta\theta
=
-\eta\operatorname{sign}(\nabla L)
}
$$

即 Sign SGD。

$L_2$ geometry 对应 negative-gradient direction，并与普通 Gradient Descent 联系起来：

$$
\boxed{
\Delta\theta
=
-\alpha\nabla L
}
$$

参数从 vector 变成 matrix，使用 spectral norm：

$$
\|W\|_2
=
\sigma_{\max}(W)
$$

得到：

$$
\boxed{
\Delta W^*
=
-\eta UV^\top
}
$$

神经网络中考虑 Xavier 所保持的 activation scale，引入 RMS norm：

$$
\|x\|_{\mathrm{RMS}}
=
\frac{\|x\|_2}{\sqrt d}
$$

进而：

$$
\|W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}
=
\sqrt{
\frac{d_{\text{in}}}{d_{\text{out}}}
}
\|W\|_2
$$

最终 Muon 希望：

$$
G
=
U\Sigma V^\top
$$

变成：

$$
UV^\top
$$

也就是：

$$
\Sigma\rightarrow I
$$

但显式 SVD 太贵，于是使用 Newton–Schulz：

$$
\boxed{
G_{k+1}
=
\frac32G_k
-
\frac12G_kG_k^\top G_k
}
$$

在迭代前先：

$$
G_0
=
\frac{G}{\|G\|_F}
$$

把 singular values 放进安全范围。

---

# 25. 一句话总结

$$
\boxed{
\text{不同的 norm 定义了不同的 update geometry，从而导出不同的 optimizer。}
}
$$

而 Muon 的核心思想可以理解为：

> 把 gradient matrix 的 singular values 拉平，让不同 singular directions 的更新更均衡，同时用 Newton–Schulz 避免每一步显式做昂贵的 SVD。