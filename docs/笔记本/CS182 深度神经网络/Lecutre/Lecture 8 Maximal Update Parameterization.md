# 0. 前言
## 为什么这一讲选择 RMS 作为基础？

这节课讨论的是 **network width 改变时，feature 和 update 的尺度会不会失控**。这里选择 RMS，而不是直接用 $L_2$ / spectral norm，最核心的原因是：

> **RMS 会把“维度变多”本身带来的 $\sqrt d$ 放大去掉，更直接地衡量每个 coordinate 的典型大小。**

例如，如果

$$
x=(1,1,\dots,1)\in\mathbb R^d,
$$

那么

$$
\|x\|_2=\sqrt d,
$$

所以随着 width $d$ 增大，$L_2$ norm 会自然变大。

但这不代表每个 feature coordinate 真的“爆炸”了，因为每个 coordinate 仍然只是 $1$。

RMS 是

$$
\|x\|_{\mathrm{RMS}}
=
\frac{\|x\|_2}{\sqrt d}
=1.
$$

所以 RMS 更适合回答：

> **网络变宽以后，每个 neuron / feature coordinate 的典型 magnitude 有没有改变？**

同理，对 matrix，RMS-to-RMS induced norm 是

$$
\|W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}\|W\|_2.
$$

它衡量的是：

> **输入 vector 的典型 coordinate 大小时，经过 $W$ 后，输出 vector 的典型 coordinate magnitude 最多会被放大多少倍。**

这就是为什么 spectral norm 本身可能随着 layer shape 改变而变大，但并不一定意味着网络真的“爆炸”。Spectral norm 衡量的是总 $L_2$ 长度的最大放大倍数，而输出维度增加时，总 $L_2$ 长度本来就可能因为 coordinate 数量变多而增长。

因此这一讲选择 RMS / RMS-to-RMS 作为 width scaling 的基准：

$$
\boxed{
\text{RMS 关注每个 coordinate 的典型 scale，而不是维度增加带来的总长度增长}
}
$$

这也是为什么 $\mu$P 会用

$$
\|h_l\|_{\mathrm{RMS}}=\Theta(1)
$$

和

$$
\|\Delta h_l\|_{\mathrm{RMS}}=\Theta(1)
$$

作为核心目标：它关心的是网络 width 改变后，**每个 feature coordinate 的量级和每次 feature update 的量级是否仍然保持稳定**。

---

## 1. $\mu$P 的第一层意思：同一个“预算”，不同 layer 自动得到不同 spectral scale

从 RMS-to-RMS constraint 出发：

$$
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
\le \gamma
}
$$

而我们已经知道：

$$
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\|\Delta W\|_2.
$$

所以：

$$
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\|\Delta W\|_2
\le \gamma
$$

等价于

$$
\boxed{
\|\Delta W\|_2
\le
\gamma
\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}
}
$$

注意：

- $\gamma$ 是 **RMS-to-RMS budget**；
- 它不是 spectral norm budget；
- 不同 layer 的 $d_{\mathrm{in}},d_{\mathrm{out}}$ 不同，所以同一个 $\gamma$ 会对应不同的 spectral-norm scale。

这就是课件说的：可以跨 layer 共享一个 hyperparameter，但实际效果会由于 fan-in / fan-out 而自动 layer-specific。

---

## 2. SignSGD / Adam：为什么 learning rate 要按 $1/d_{\mathrm{in}}$ 缩放？

Lecture 8 p.17–21 用一个非常具体的例子说明 $\mu$P 的 width scaling。

SignSGD（以及把 Adam 简化成 elementwise sign update 的视角）写成：

$$
W_{t+1}
=
W_t-\eta\,\operatorname{sign}(\nabla_W L).
$$

定义：

$$
S=\operatorname{sign}(\nabla_WL),
$$

所以：

$$
\boxed{
\Delta W=-\eta S
}
$$

目标是控制：

$$
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
\le \gamma
}
$$

### 2.1 Batch size $=1$ 时为什么 gradient 是 rank 1？

对一个 linear layer：

$$
y=Wx.
$$

单个 sample 的 weight gradient 是：

$$
\nabla_WL
=
\frac{\partial L}{\partial y}x^\top.
$$

令：

$$
u=\frac{\partial L}{\partial y},
\qquad
v=x,
$$

就得到：

$$
\boxed{
\nabla_WL=uv^\top
}
$$

它是一个 outer product，所以 rank 最多为 1；非零时 rank 就是 1。

这里要特别注意：

> batch size $=1$ 并不是因为 gradient matrix 只有一行，所以 rank $\le1$。
>
> gradient matrix 仍然可以是 $d_{\mathrm{out}}\times d_{\mathrm{in}}$；rank 1 来自 **单样本 gradient 的 outer-product 结构**。

### 2.2 和 SVD 的关系：为什么可以写成 $\sigma uv^\top$？

任意 matrix 都可以写 SVD：

$$
\nabla_WL=U\Sigma V^\top.
$$

如果 rank $=1$，只有一个非零 singular value，因此 reduced SVD 可以写成：

$$
U\in\mathbb R^{d_{\mathrm{out}}\times1},
$$

$$
\Sigma=[\sigma]\in\mathbb R^{1\times1},
$$

$$
V\in\mathbb R^{d_{\mathrm{in}}\times1},
\qquad
V^\top\in\mathbb R^{1\times d_{\mathrm{in}}}.
$$

于是：

$$
\nabla_WL
=
\sigma uv^\top.
$$

如果把 scalar $\sigma$ 吸收到 $u$ 或 $v$ 中，也可以简写成：

$$
\nabla_WL=uv^\top.
$$

### 2.3 为什么取 elementwise sign 以后，$S$ 仍然 rank 1？

不能因为“$S$ 和 gradient 的 nonzero pattern 一样”就断定 rank 一样。一般情况下，elementwise sign 完全可能改变 rank。

这里之所以仍然 rank 1，需要证明。

从：

$$
\nabla_WL=\sigma uv^\top
$$

出发，第 $(i,j)$ 个元素是：

$$
(\nabla_WL)_{ij}=\sigma u_i v_j.
$$

逐元素取 sign：

$$
S_{ij}
=
\operatorname{sign}(\sigma u_iv_j).
$$

因为 singular value $\sigma>0$，所以：

$$
\operatorname{sign}(\sigma)=1.
$$

同时：

$$
\operatorname{sign}(ab)
=
\operatorname{sign}(a)\operatorname{sign}(b).
$$

因此：

$$
S_{ij}
=
\operatorname{sign}(u_i)
\operatorname{sign}(v_j).
$$

把整个 matrix 写出来：

$$
\boxed{
S
=
\operatorname{sign}(u)
\operatorname{sign}(v)^\top
}
$$

它仍然是一个 outer product，所以：

$$
\boxed{
\operatorname{rank}(S)=1
}
$$

### 2.4 为什么 rank 1 时 spectral norm = Frobenius norm？

rank $=1$ 表示只有一个非零 singular value $\sigma_1$。

Spectral norm：

$$
\|S\|_2=\sigma_1.
$$

Frobenius norm：

$$
\|S\|_F
=
\sqrt{\sigma_1^2+0+0+\cdots}
=
\sigma_1.
$$

所以：

$$
\boxed{
\|S\|_2=\|S\|_F
}
$$

而 $S$ 的 shape 是：

$$
d_{\mathrm{out}}\times d_{\mathrm{in}},
$$

每个非零元素都是 $\pm1$，所以：

$$
\|S\|_F^2
=
\sum_{i,j}S_{ij}^2
=
d_{\mathrm{out}}d_{\mathrm{in}}.
$$

因此：

$$
\boxed{
\|S\|_2
=
\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}
}
$$

### 2.5 换成 RMS-to-RMS norm

利用：

$$
\|S\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\|S\|_2,
$$

得到：

$$
\|S\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}
=
d_{\mathrm{in}}.
$$

所以：

$$
\boxed{
\|S\|_{\mathrm{RMS}\to\mathrm{RMS}}=d_{\mathrm{in}}
}
$$

### 2.6 为什么 $\|\Delta W\|=\eta\|S\|$？

这里用的是 norm 的 homogeneous property：

$$
\boxed{
\|cA\|=|c|\|A\|
}
$$

其中：

- $c$ 是 scalar；
- $A$ 是 vector 或 matrix；
- 左右两边必须使用同一种 norm。

在这里：

$$
c=-\eta,
\qquad
A=S.
$$

因此更完整地写：

$$
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\|-\eta S\|_{\mathrm{RMS}\to\mathrm{RMS}}
$$

$$
=
|{-\eta}|\,
\|S\|_{\mathrm{RMS}\to\mathrm{RMS}}
$$

learning rate $\eta>0$，所以：

$$
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\eta d_{\mathrm{in}}
}
$$

这就是整个推导最重要的结果。

它说明：

> 即使 raw learning rate $\eta$ 完全不变，只要 $d_{\mathrm{in}}$ 改变，update 的有效 RMS-to-RMS 大小也会改变。

如果希望：

$$
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
\le \gamma,
$$

就需要：

$$
\eta d_{\mathrm{in}}
\le \gamma.
$$

所以：

$$
\boxed{
\eta
\le
\frac{\gamma}{d_{\mathrm{in}}}
}
$$

从 scaling 角度：

$$
\boxed{
\eta\propto\frac1{d_{\mathrm{in}}}
}
$$

这就是 Lecture 8 用 SignSGD / Adam 举例说明的 $\mu$P essence。

### 2.7 Batch size 增大时的 caveat

课件 p.21 特别提醒：batch size 增大后，gradient 不再一定 rank 1。

这时 Frobenius norm 一般只给 spectral norm 一个 upper bound：

$$
\|G\|_2\le\|G\|_F.
$$

所以用 Frobenius norm 推出来的 learning-rate bound 会更 conservative。

### 2.8 这一段要掌握到什么程度？

必须会重建的逻辑链：

$$
\Delta W=-\eta S
$$

$$
S\text{ rank 1}
\Rightarrow
\|S\|_2=\|S\|_F
$$

$$
\Rightarrow
\|S\|_2=\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}
$$

$$
\Rightarrow
\|S\|_{\mathrm{RMS}\to\mathrm{RMS}}=d_{\mathrm{in}}
$$

$$
\Rightarrow
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\eta d_{\mathrm{in}}
}
$$

$$
\Rightarrow
\boxed{
\eta\propto\frac1{d_{\mathrm{in}}}
}
$$

每一个 trace identity 不一定需要死记，但上面这条因果链最好能够自己重新写出来。

---

## 3. $\mu$P 到底想控制什么？Feature Learning 的两个目标

Lecture 8 p.25 重新把问题拉回 feature learning。

一层 hidden feature 记为：

$$
h_l.
$$

忽略 bias 和 nonlinearity 时：

$$
\boxed{
h_l=W_lh_{l-1}
}
$$

课件提出两个 desiderata：

$$
\boxed{
\|h_l\|_{\mathrm{RMS}}=\Theta(1)
}
$$

以及

$$
\boxed{
\|\Delta h_l\|_{\mathrm{RMS}}=\Theta(1)
}
$$

第一条：网络变宽时，feature 本身不要爆炸或消失。

第二条：网络变宽时，一次训练造成的 feature change 也不要爆炸或消失；否则宽网络可能越来越“学不动 feature”。

所以可以粗略理解为：

$$
\boxed{
\text{Xavier：主要关心 forward scale}
}
$$

$$
\boxed{
\mu P：forward scale + training update scale 都要稳定
}
$$

---

## 4. 最后几页：老师的方法 vs 另一种 induced-norm 方法

这部分最容易混乱，因为两种讲法的目标相同，但证明路径不同。

### 4.1 目标 A：为什么希望 $\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)$？

#### 老师在 p.26 的方法：用 top singular-vector alignment 得到等号

先由：

$$
\|h\|_{\mathrm{RMS}}
=
\frac{\|h\|_2}{\sqrt d}
$$

如果：

$$
\|h_{l-1}\|_{\mathrm{RMS}}=\Theta(1),
$$

且

$$
h_{l-1}\in\mathbb R^{d_{\mathrm{in}}},
$$

那么：

$$
\boxed{
\|h_{l-1}\|_2
=
\Theta(\sqrt{d_{\mathrm{in}}})
}
$$

同理，若：

$$
\|h_l\|_{\mathrm{RMS}}=\Theta(1),
$$

则：

$$
\boxed{
\|h_l\|_2
=
\Theta(\sqrt{d_{\mathrm{out}}})
}
$$

课件再加一个关键条件：

> $h_{l-1}$ 与 $W_l$ 的 top singular vector 对齐（课件写 “can show this is the case”）。

这样：

$$
\boxed{
\|W_lh_{l-1}\|_2
=
\|W_l\|_2
\|h_{l-1}\|_2
}
$$

因为：

$$
h_l=W_lh_{l-1},
$$

所以：

$$
\Theta(\sqrt{d_{\mathrm{out}}})
=
\|W_l\|_2
\Theta(\sqrt{d_{\mathrm{in}}}).
$$

因此：

$$
\boxed{
\|W_l\|_2
=
\Theta\left(
\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}
\right)
}
$$

再利用：

$$
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\|W_l\|_2,
$$

得到：

$$
\boxed{
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\Theta(1)
}
$$

这就是老师的证明路径：

$$
\text{feature RMS }\Theta(1)
\Rightarrow
\text{feature L2 scale}
\Rightarrow
\text{top-singular alignment}
\Rightarrow
\|W_l\|_2\text{ scaling}
\Rightarrow
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1).
$$

#### 另一种更直观但更弱的方法：直接用 induced norm upper bound

RMS-to-RMS norm 的定义直接给：

$$
\boxed{
\|W_lh_{l-1}\|_{\mathrm{RMS}}
\le
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
\|h_{l-1}\|_{\mathrm{RMS}}
}
$$

如果：

$$
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)
$$

和

$$
\|h_{l-1}\|_{\mathrm{RMS}}=\Theta(1),
$$

那么只能得到：

$$
\|h_l\|_{\mathrm{RMS}}=O(1).
$$

也就是可以保证“不爆炸”，但光靠这个 upper bound 不能证明它一定不会趋近于 0。

所以：

- 老师的方法更强，因为加入 top singular-vector alignment 后可以用等号做 scaling 推导；
- induced-norm 方法更直观，但严格来说主要给 upper bound。

---

### 4.2 目标 B：为什么还要 $\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)$？

Lecture 8 p.27 提出第二个 condition：

$$
\boxed{
\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\Theta(1)
}
$$

利用 RMS-to-RMS 和 spectral norm 的关系：

$$
\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\sqrt{\frac{d_{\mathrm{in}}}{d_{\mathrm{out}}}}
\|\Delta W_l\|_2,
$$

所以等价于：

$$
\boxed{
\|\Delta W_l\|_2
=
\Theta\left(
\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}
\right)
}
$$

这和 $W_l$ 本身的 spectral-norm scaling 是同一个量级。

---

### 4.3 老师课件中的 $\Delta h_l$ 公式

课件写：

$$
\boxed{
\Delta h_l
=
\Delta W_lh_{l-1}
+
W_l\Delta h_{l-1}
}
$$

这应该理解为 **first-order / differential view**。

它的两项分别是：

$$
\Delta W_lh_{l-1}
$$

当前这一层 weight update 对 feature 的直接作用；

以及：

$$
W_l\Delta h_{l-1}
$$

上一层 feature change 经过当前 $W_l$ 向后传播。

对 L2 norm 用 triangle inequality：

$$
\|a+b\|_2
\le
\|a\|_2+\|b\|_2,
$$

得到：

$$
\|\Delta h_l\|_2
\le
\|\Delta W_lh_{l-1}\|_2
+
\|W_l\Delta h_{l-1}\|_2.
$$

再用 spectral norm 的 induced property：

$$
\|Ax\|_2
\le
\|A\|_2\|x\|_2,
$$

所以：

$$
\boxed{
\|\Delta h_l\|_2
\le
\|\Delta W_l\|_2\|h_{l-1}\|_2
+
\|W_l\|_2\|\Delta h_{l-1}\|_2
}
$$

假设上一层已经满足 feature-learning scaling：

$$
\|h_{l-1}\|_2
=
\Theta(\sqrt{d_{\mathrm{in}}}),
$$

$$
\|\Delta h_{l-1}\|_2
=
\Theta(\sqrt{d_{\mathrm{in}}}),
$$

并且：

$$
\|W_l\|_2,
\|\Delta W_l\|_2
=
\Theta\left(
\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}
\right),
$$

那么右边每一项都是：

$$
\Theta\left(
\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}
\right)
\Theta(\sqrt{d_{\mathrm{in}}})
=
\Theta(\sqrt{d_{\mathrm{out}}}).
$$

因此由这个 upper bound 至少严格得到：

$$
\boxed{
\|\Delta h_l\|_2
=
O(\sqrt{d_{\mathrm{out}}})
}
$$

也就是：

$$
\boxed{
\|\Delta h_l\|_{\mathrm{RMS}}
=
O(1)
}
$$

课件的目标是更强的：

$$
\|\Delta h_l\|_{\mathrm{RMS}}=\Theta(1).
$$

从上面这些 inequality 本身主要能控制“不爆炸”；要得到严格 lower bound / non-vanishing，还需要额外结构或不发生严重 cancellation 的假设。

---

### 4.4 如果不用 first-order approximation，精确展开是什么？

这是课件没有展开、但为了避免混淆必须记住的补充。

原来：

$$
h_l=W_lh_{l-1}.
$$

训练后：

$$
W_l\to W_l+\Delta W_l,
$$

$$
h_{l-1}\to h_{l-1}+\Delta h_{l-1}.
$$

所以精确地：

$$
h_l+\Delta h_l
=
(W_l+\Delta W_l)
(h_{l-1}+\Delta h_{l-1}).
$$

展开：

$$
h_l+\Delta h_l
=
W_lh_{l-1}
+
\Delta W_lh_{l-1}
+
W_l\Delta h_{l-1}
+
\Delta W_l\Delta h_{l-1}.
$$

减掉：

$$
h_l=W_lh_{l-1},
$$

得到精确式：

$$
\boxed{
\Delta h_l
=
\Delta W_lh_{l-1}
+
W_l\Delta h_{l-1}
+
\Delta W_l\Delta h_{l-1}
}
$$

最后一个：

$$
\Delta W_l\Delta h_{l-1}
$$

是二阶 cross term。

如果采用 small-update / first-order approximation，就忽略它：

$$
\boxed{
\Delta h_l
\approx
\Delta W_lh_{l-1}
+
W_l\Delta h_{l-1}
}
$$

这和 Lecture 7 一开始 local linearization 的思想完全一致：只保留一阶变化，忽略二阶小量。

因此：

- **老师课件**：直接使用 first-order 两项形式；
- **精确有限差分展开**：会多出 $\Delta W_l\Delta h_{l-1}$；
- 两者并不矛盾，区别在于是否忽略二阶项。

---

## 5. 把 Lecture 8 的 $\mu$P 逻辑完整串起来

最重要的不是单独背公式，而是能说出下面这条主线：

网络变宽时，我们希望：

$$
\boxed{
\|h_l\|_{\mathrm{RMS}}=\Theta(1)
}
$$

也希望：

$$
\boxed{
\|\Delta h_l\|_{\mathrm{RMS}}=\Theta(1)
}
$$

于是希望 matrix 本身和 matrix update 都有稳定的 effective scale：

$$
\boxed{
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)
}
$$

$$
\boxed{
\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)
}
$$

对于 Lecture 8 分析的 SignSGD / Adam 简化情形：

$$
\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\eta d_{\mathrm{in}}.
$$

为了让它保持 constant order：

$$
\eta d_{\mathrm{in}}=\Theta(1),
$$

所以：

$$
\boxed{
\eta
=
\Theta\left(\frac1{d_{\mathrm{in}}}\right)
}
$$

这就是 Lecture 8 最具体地展示的 $\mu$P scaling rule。

---

## 6. 考试 / 复习时建议掌握的层级

### 必须真正理解

1. $\mu$P 的目的：网络 width 改变时，feature 和 feature update 的 scale 不要被 width 搞乱。
2. 两个 desiderata：

$$
\|h_l\|_{\mathrm{RMS}}=\Theta(1),
\qquad
\|\Delta h_l\|_{\mathrm{RMS}}=\Theta(1).
$$

3. 为什么希望：

$$
\|W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1),
\qquad
\|\Delta W_l\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1).
$$

4. SignSGD / Adam 简化分析的主推导：

$$
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
=
\eta d_{\mathrm{in}}
}
$$

$$
\boxed{
\eta\propto\frac1{d_{\mathrm{in}}}
}
$$

### 最好能自己重新推出来

- rank 1 gradient 为什么能写成 $uv^\top$；
- 为什么 sign 后仍 rank 1；
- rank 1 时为什么 spectral norm = Frobenius norm；
- 为什么 $\|S\|_F=\sqrt{d_{\mathrm{out}}d_{\mathrm{in}}}$；
- RMS-to-RMS 与 spectral norm 的换算；
- p.26 老师从 feature RMS 反推出 $\|W_l\|_2$ scaling 的过程。

### 不需要死记

- 每一个中间 trace identity；
- 每一个 $\Theta/O$ 的形式化定义证明；
- 只要能在需要时从定义重新构造即可。

---

## 7. 一句话总结

$$
\boxed{
\mu P
\text{ 的核心不是“某个固定 learning rate 公式”，而是规定 width 改变时各个量如何缩放，使 network 的 forward scale 和 learning dynamics 都保持稳定。}
}
$$

Lecture 8 中最具体的例子就是：

$$
\boxed{
\text{SignSGD / Adam simplified analysis:}
\qquad
\eta\propto\frac1{d_{\mathrm{in}}}
}
$$

以维持：

$$
\boxed{
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}=\Theta(1)
}
$$
