# Position Encoding、MLA、GPT-3、BERT、GLU 与 MoE

> **范围：课程笔记 PDF 第 1–18 页。第 19 页以及之后的所有内容都没有收录。** 这份笔记按第一次上课的顺序展开，不是考前提纲。原讲义图已裁剪放进同目录的 `附件` 文件夹；公式和数字例子中，未注明为老师原图的部分是为教学补充的示例。
>
> **统一矩阵约定**：一个 token 是一行，维度 $1\times D$；$n$ 个 token 组成 $X\in\mathbb R^{n\times D}$。全文不随意切换成列向量。涉及单个二维旋转向量时，会特别注明用列向量推导，并写回行向量形式。

## 1. Position Encoding（第 1–5 页）

### 1.1 先明确 Self-Attention 算出了什么

输入 $n$ 个 token，每个 token 的特征维度为 $D$：

$$
X=\begin{bmatrix}x_0\\x_1\\\vdots\\x_{n-1}\end{bmatrix}\in\mathbb R^{n\times D},\qquad x_i\in\mathbb R^{1\times D}.
$$

对于**一个 Attention Head**，可以自由设定 Query/Key 的维度 $d_k$ 与 Value 的维度 $d_v$：

$$
W_Q,W_K\in\mathbb R^{D\times d_k},\qquad W_V\in\mathbb R^{D\times d_v},
$$

$$
Q=XW_Q\in\mathbb R^{n\times d_k},\quad
K=XW_K\in\mathbb R^{n\times d_k},\quad
V=XW_V\in\mathbb R^{n\times d_v}.
$$

先算 $QK^\top$（大小为 $n\times n$）。第 $t$ 行第 $i$ 列的 Score 是当前 token $t$ 对 token $i$ 的匹配分数。对每一行做 Softmax，再乘 $V$：

$$
A=\operatorname{softmax}_{\text{row}}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)\in\mathbb R^{n\times n},\qquad
O=AV\in\mathbb R^{n\times d_v}.
$$

**$A$ 是注意力权重矩阵；$O$ 才是每个 token 聚合其他 token 信息后的特征表示。** 第 $t$ 行有：

$$
o_t=\sum_{i=0}^{n-1}a_{ti}v_i\in\mathbb R^{1\times d_v}.
$$

如果用 $H$ 个 Heads，分别计算 $O^{(1)},\ldots,O^{(H)}$，再沿特征维度拼接：

$$
O_{\mathrm{concat}}\in\mathbb R^{n\times(Hd_v)}.
$$

再通过输出投影 $W_O\in\mathbb R^{Hd_v\times D}$ 得到 $n\times D$，这样才能直接与原输入做残差相加。常见配置 $Hd_v=D$ 只是设计选择，**不是** Self-Attention 或 Multi-Head Attention 的数学硬性要求。例如 $D=512,H=8,d_v=32$，拼接是 $n\times256$，照样可以用 $256\times512$ 的 $W_O$ 投影回 $n\times512$。

现在有个重要问题：原始的 $QK^\top$ 只体现 token 特征匹配。如果把输入 token 整体重新排列，没有位置机制的普通 Self-Attention 会相应重排输出（Permutation Equivariance）。它并没有独立、明确地表示“这是第几个词”。

### 1.2 Absolute Position Encoding：直接告诉模型绝对位置

最直接的办法是在计算 Q、K、V 之前，把位置向量加到 Token Embedding 上：

$$
Z=X+P,\qquad P\in\mathbb R^{n\times D},\qquad
Q=ZW_Q,\ K=ZW_K,\ V=ZW_V.
$$

所以**位置编码先加入输入，再进入 Attention**；不是 Attention 结束后附加某个标签。

原始 Transformer 的经典 Sinusoidal（正弦/余弦）编码是：

$$
\boxed{PE(t,2m)=\sin(\omega_mt),\qquad PE(t,2m+1)=\cos(\omega_mt)}
$$

$$
\omega_m=10000^{-2m/D},\qquad m=0,1,\ldots,\frac D2-1.
$$

- $t$：token 的**位置**，一般从 0 开始；
- $D$：整个位置向量的维度；
- $2m,2m+1$：相邻的两列特征；
- $\omega_m$：这一对维度使用的角频率；不同维度对使用不同的频率。

例如 $D=4$ 时，$\omega_0=1,\omega_1=0.01$，位置 $t$ 的向量是：

$$
PE(t)=[\sin t,\ \cos t,\ \sin(0.01t),\ \cos(0.01t)].
$$

这不是四个随意的数：高频分量变化快，低频分量变化慢，组合起来可以让模型在不同尺度上区分位置。固定 Sinusoidal Encoding **本身无需训练**。另一类 *Learned Absolute Position Embedding* 则直接设一个可训练查找表 $E_{\rm pos}$，令 $P_t=E_{\rm pos}[t]$。

> 老师第 1 页写了 “No longer used”，应理解为：**经典固定绝对 Sin/Cos 编码已经不是许多现代大语言模型的主流配置**。不能推导成所有模型都不用绝对位置编码。

### 1.3 NoPE：没有显式位置编码，Causal Attention 为什么仍可能感知位置？

**NoPE = No Position Embedding。** 它不是先算出一个位置编码再加到 Attention Output 上，而是**根本不显式加 $P$**。教授想证明的是：*即使不加 $P$，Causal Attention 本身也可能让输出携带位置相关信息。*

先区分两个目的：

- **Causal Mask 的原始用途**：在 GPT 预测下一个词时，禁止某个位置看到它后面的词，以免训练时偷看答案。
- **它顺带带来的效果**：不同位置能看到的历史 token 数量不同，因此有可能成为一种隐式位置信号。

下面**完整算出 Attention 矩阵**，而不是一开始就扔出 $1/(t+1)$。

假设序列有四个 token，按位置 0、1、2、3 排列：

$$
[\mathrm{START},A,B,C].
$$

首先正常算 $S=QK^\top/\sqrt{d_k}$。为了专门观察 Mask 的作用，构造一个特殊例子，**假设每个可见位置的 Score 都是 0**：

$$
S=\begin{bmatrix}
0&0&0&0\\
0&0&0&0\\
0&0&0&0\\
0&0&0&0
\end{bmatrix}.
$$

Causal Mask 规定位置 $t$ 只能关注位置 $0,1,\ldots,t$。给未来位置加上 $-\infty$：

$$
M=\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix},\qquad \widetilde S=S+M.
$$

因为 $e^{-\infty}=0$，每一行 Softmax 只在允许的位置归一化：

$$
\boxed{
A=\operatorname{softmax}_{\rm row}(\widetilde S)=
\begin{bmatrix}
1&0&0&0\\
\frac12&\frac12&0&0\\
\frac13&\frac13&\frac13&0\\
\frac14&\frac14&\frac14&\frac14
\end{bmatrix}}
$$

**这时才出现 $1/(t+1)$！** 位置 $t$ 有 $t+1$ 个可见 token，且例子中它们的 Score 相等，因此每个可见 token 的 Attention Weight 都是 $1/(t+1)$。特别地，针对 START（第 0 列）：

$$
a_{t0}=\frac{1}{t+1}.
$$

但别忘了：**$A$ 不是最终 Attention Output**。还需要乘以 $V$：

$$
O=AV.
$$

教授又构造了一个特殊的一维 Value：

$$
V=\begin{bmatrix}1\\0\\0\\0\end{bmatrix}
\quad\text{（只有 START 的 Value 为 1，其余为 0）}.
$$

直接矩阵相乘：

$$
O=
\begin{bmatrix}
1&0&0&0\\\frac12&\frac12&0&0\\\frac13&\frac13&\frac13&0\\\frac14&\frac14&\frac14&\frac14
\end{bmatrix}
\begin{bmatrix}1\\0\\0\\0\end{bmatrix}
=\boxed{\begin{bmatrix}1\\\frac12\\\frac13\\\frac14\end{bmatrix}}.
$$

**为什么要刻意设 $V=[1,0,0,0]^\top$？** 因为教授想把权重矩阵第 0 列的 $1,1/2,1/3,1/4$ 提取成**最后输出的数值**。如果所有 Value 都设成 1，那么每行权重加起来都是 1，最后四个位置的输出全都是 1，就无法靠这个输出区分位置。

做个对照：若没有 Causal Mask，且所有 Score 仍是 0，那么四行都变为 $[1/4,1/4,1/4,1/4]$；乘以同一个 $V$ 后，输出全为 $1/4$。**两次计算中，输入 Q、K、V 没改，唯一差别是 Causal Mask。**

所以教授真正想说明的是：

$$
\boxed{\text{Causal Mask 的可见范围随位置变化}\ \Longrightarrow\ \text{网络有可能从输出中学到位置特征。}}
$$

一定不要把例子误认为真实模型的规则：**真实模型不会手动把 START 的 Value 固定为 1、其余固定为 0**，真实 Attention Weight 也不会永远是 $1/(t+1)$。这只是一个“可以构造出位置信息”的证明，并不能保证 NoPE 在所有任务上足够好。

### 1.4 Learned Relative Position Encoding：关心相距几格，而非各在第几格

绝对位置告诉模型“位置 100”和“位置 98”。但是很多关系更关心它们相距 2 个 token。老师第 3 页示范的一种方式，是为相对位移 $t-i$ 学一个 Bias：

$$
\boxed{s'_{ti}=\frac{q_tk_i^\top}{\sqrt{d_k}}+b_{t-i}}.
$$

这里 $q_t,k_i$ 都是**行向量**，所以点积是 $q_tk_i^\top$；$b_{t-i}$ 是标量。位置 (5,3) 和 (100,98) 都使用同一个 $b_2$，这就是**针对相同相对距离共享参数**。

完整的 Attention 权重为：

$$
a_{ti}=\frac{\exp(s'_{ti})}{\sum_{j\in\mathrm{visible}(t)}\exp(s'_{tj})}.
$$

分数里同时包含**内容匹配分数**和**位置 Bias**。它不是必须偏爱附近 token：$b_1,b_2,\ldots$ 都是学习出来的。不同方法也可以使用相对位置向量而不仅仅是标量 Bias；这页笔记展示的是简单 Bias 形式。

### 1.5 RoPE：不是给 $X$ 加向量，而是逐 Token 旋转 Q、K

**RoPE = Rotary Position Embedding（旋转位置编码）。** 它在计算 Q、K 后，按位置旋转各 token 的 Q、K，然后正常做 $Q'K'^\top$。这和“直接给 Token Embedding 加位置向量”不同。

先看二维列向量（**仅本小段为方便使用旋转矩阵，临时改成列向量**）：

$$
R_t=\begin{bmatrix}\cos(t\omega)&-\sin(t\omega)\\\sin(t\omega)&\cos(t\omega)\end{bmatrix},
\qquad q_t^{\prime\,\mathrm{col}}=R_t q_t^{\mathrm{col}}.
$$

二维旋转满足：

$$
R_t^\top R_i=R_{i-t}.
$$

**现在回到全篇统一的行向量表示**。假设三个 token，每个 Head 的 Q、K 为 $3\times2$：

$$
Q=\begin{bmatrix}q_0\\q_1\\q_2\end{bmatrix},\qquad
K=\begin{bmatrix}k_0\\k_1\\k_2\end{bmatrix},\qquad q_t,k_i\in\mathbb R^{1\times2}.
$$

第 0、1、2 行分别使用 $R_0,R_1,R_2$；**按行右乘转置**（对应前面的二维列向量左乘 $R_t$）：

$$
Q'=\begin{bmatrix}q_0R_0^\top\\q_1R_1^\top\\q_2R_2^\top\end{bmatrix},\qquad
K'=\begin{bmatrix}k_0R_0^\top\\k_1R_1^\top\\k_2R_2^\top\end{bmatrix}.
$$

接下来像普通 Attention 一样进行**矩阵乘法**：

$$
\boxed{
S'=Q'K'^\top=
\begin{bmatrix}
q_0R_0^\top R_0k_0^\top&q_0R_0^\top R_1k_1^\top&q_0R_0^\top R_2k_2^\top\\
q_1R_1^\top R_0k_0^\top&q_1R_1^\top R_1k_1^\top&q_1R_1^\top R_2k_2^\top\\
q_2R_2^\top R_0k_0^\top&q_2R_2^\top R_1k_1^\top&q_2R_2^\top R_2k_2^\top
\end{bmatrix}}
$$

现在才逐项用 $R_t^\top R_i=R_{i-t}$ 化简。第 $t$ 行第 $i$ 列为：

$$
\boxed{s'_{ti}=q_tR_{i-t}k_i^\top}.
$$

**这就解释了为什么绝对位置旋转会得到相对位置：** 对 Q 按 $t$ 旋转、对 K 按 $i$ 旋转，做点积后自然出现两者的角度差 $i-t$。注意相对位置相同，不代表最终 Score 一定相等，因为 $q_t,k_i$ 的内容不同。

一个数值例子：令 $\omega=\pi/2$（相邻 token 相差旋转 $90^\circ$），

$$
R_0=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
R_1=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad
R_2=\begin{bmatrix}-1&0\\0&-1\end{bmatrix}.
$$

$$
Q=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix},\qquad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.
$$

逐行右乘对应的 $R_t^\top$，得到：

$$
Q'=\begin{bmatrix}1&0\\-1&1\\0&-1\end{bmatrix},\qquad
K'=\begin{bmatrix}1&0\\-1&0\\-1&-1\end{bmatrix}.
$$

于是：

$$
S'=Q'K'^\top
=\begin{bmatrix}1&0\\-1&1\\0&-1\end{bmatrix}
\begin{bmatrix}1&-1&-1\\0&0&-1\end{bmatrix}
=\boxed{\begin{bmatrix}1&-1&-1\\-1&1&0\\0&0&1\end{bmatrix}}.
$$

这个结果只是**未缩放的 Score**。真实 Attention 后面还要除以 $\sqrt{d_k}$，必要时加 Causal Mask，再做 Softmax、乘 $V$。

**为什么不能直接写 $R_1Q(R_2K)^\top$？** 因为 RoPE 的 $R_t$ 是作用于**某个 token 的特征维度**的 $2\times2$ 矩阵，不是同一矩阵左乘整个 $n\times2$ 的 Q；不同的行还要用**不同的** $R_t$。若一定要写成一次大矩阵乘法，可以先把各 token 的二维列向量堆叠成长度 $2n$ 的列向量，再用 $\operatorname{diag}(R_0,\ldots,R_{n-1})$ 这种块对角矩阵旋转，最后 reshape 回 $n\times2$。

真实 RoPE 通常在每个 Head 的高维 Q、K 上**每两维分成一组**，每组使用不同频率 $\omega_m$ 的二维旋转；第 4–5 页的 $M_j$ 就是这些 $2\times2$ 小旋转矩阵。频率在此经典设计中是预先给定的，Q、K 的投影权重仍然需要训练。

> **记忆位置编码这一节**：Absolute 是在 Attention 前给 $X$ 加 $P$；Relative Bias 是给 Score 加 $b_{t-i}$；RoPE 是给每个位置的 Q、K 旋转；NoPE 则不显式加入位置向量，而可能利用 Causal Mask 的结构学到位置。Causal Mask 的首要目的是**防止未来信息泄露**，并不是专门为位置编码发明的。所以使用 RoPE 的自回归 GPT 仍然需要 Causal Mask。

---

## 2. MLA 与 Decoupled RoPE（第 6–7 页）

### 2.1 先解释 KV Cache：为什么优先压缩 K、V，不是 Q？

MLA = **Multi-head Latent Attention**，核心目标之一是**减少 GPT 逐 Token 生成时的 KV Cache**。这里不先解释 KV Cache，就无法理解为什么只强调压缩 KV。

假设已有三个 token，马上要处理新 token $x_3$。在一个使用 Causal Attention 的 Head 中，完整 Score 矩阵本来是：

$$
S=\begin{bmatrix}
s_{00}&-\infty&-\infty&-\infty\\
s_{10}&s_{11}&-\infty&-\infty\\
s_{20}&s_{21}&s_{22}&-\infty\\
s_{30}&s_{31}&s_{32}&s_{33}
\end{bmatrix}.
$$

前面三个 token 的输出过去已经算过。**增量解码时，只需要算新 token 对应的最后一行**：

$$
s_3=\frac{q_3K_{0:3}^\top}{\sqrt{d_k}}\in\mathbb R^{1\times4},\qquad
o_3=\operatorname{softmax}(s_3)V_{0:3}.
$$

当前 Query $q_3$ 要和**所有历史 Key**比较，并用 Attention Weight 加权**所有历史 Value**。但不需要过去的 $q_0,q_1,q_2$：它们对应的历史输出不必重新计算。因此生成时要长期保存的是每层历史 $K,V$，而不是历史 $Q$。

假设有 $H=32$ 个 Heads，各 Head $d_k=d_v=128$。每个 token、每层的普通 KV Cache 需要：

$$
32\times128\ \text{个 K 数值}+32\times128\ \text{个 V 数值}=8192\ \text{个数值}.
$$

随着已生成 token 数 $T$ 增大，缓存大小近似按 $T$ 增长。所以 MLA **优先压缩需要长期保留的历史 KV**。压缩 Q 也有可能（实际 MLA 有 Query 的低秩投影变体），但是压缩 Q 并不能直接消除历史 KV Cache，因为历史 Q 原本就不缓存。

### 2.2 MLA 压缩了什么？Weight Absorption 又是什么？

假设 Attention 之后第 $i$ 个 token 的特征是 $x_i\in\mathbb R^{1\times D}$。先投影到共享的低维表示：

$$
c_i=x_iW_{\rm down}\in\mathbb R^{1\times d_c},\qquad d_c\ll Hd_k,Hd_v.
$$

所有 token 组成 $C\in\mathbb R^{n\times d_c}$。对于某个 Head，完整的 Key 和 Value 可由 $C$ 线性恢复：

$$
K=CW_{UK},\quad W_{UK}\in\mathbb R^{d_c\times d_k};\qquad
V=CW_{UV},\quad W_{UV}\in\mathbb R^{d_c\times d_v}.
$$

最直观的办法是缓存 $C$，需要时把 $K,V$ 算回来。但这样每次生成新 token，都要对很多旧的 $c_i$ 重建 $k_i,v_i$。于是考虑**改变矩阵乘法顺序**。

普通 Score：

$$
S=QK^\top=Q(CW_{UK})^\top=QW_{UK}^\top C^\top.
$$

把固定投影 $W_{UK}^\top$ **吸收到 Query 一侧**：

$$
\boxed{\widetilde Q=QW_{UK}^\top\in\mathbb R^{n\times d_c},\qquad S=\widetilde QC^\top.}
$$

关注尺寸：$[n\times d_k][d_k\times d_c][d_c\times n]\to[n\times n]$。所以 **不用显式构造完整 $K$，直接让转换过的 Query 与压缩的历史 $C$ 做点积，Score 仍完全一样**。这叫 *Weight Absorption（权重吸收）*，是矩阵结合律的等价变换，并不是靠近似魔法。

Value 端也能这样重排（暂时忽略多 Head 的输出投影）：

$$
O=AV=A(CW_{UV})=(AC)W_{UV}.
$$

因此大部分计算可以直接基于压缩表示 $C$ 完成，缓存主要存储 $C$。这才是老师第 6 页 “never compute $k$ explicitly” 的意思。

### 2.3 为什么 RoPE 会破坏刚才的权重吸收？

没有 RoPE 时，当前 Query 对所有历史 token 共用同一个 $\widetilde q_t=q_tW_{UK}^\top$：

$$
s_{ti}=\widetilde q_t c_i^\top.
$$

有 RoPE 后，Content Key 要根据**Key 自己的位置 $i$**旋转，Query 也按位置 $t$ 旋转。用统一的行向量记号写：

$$
q_t'=q_tR_t^\top,\qquad k_i'=c_iW_{UK}R_i^\top.
$$

二者点积：

$$
\begin{aligned}
s_{ti}'&=q_t'(k_i')^\top\\
&=q_tR_t^\top R_iW_{UK}^\top c_i^\top\\
&=\underbrace{\big(q_tR_t^\top R_iW_{UK}^\top\big)}_{\text{每个历史位置 }i\text{ 都不同}}c_i^\top.
\end{aligned}
$$

现在无法只算**一次** $\widetilde q_t$ 就与所有 $c_i$ 相乘，因为表达式中出现了每个历史位置各自的 $R_i$。如果依旧为了省缓存而不保存 $K$，就得针对每个 $(t,i)$ 处理位置相关变换，失去刚才简单、统一的权重吸收优势。

直接看一个数值矛盾：若压缩的历史表示为 $C=[1,2,3]^\top$，对某个 Query，旋转后希望三个 Score 恰好是 $[-3,2,9]$，不可能找到一个**单独的标量** $\widetilde q$ 让 $\widetilde q[1,2,3]=[-3,2,9]$。原因并不是矩阵乘法出错，而是**每个位置用的旋转不同**。这是一个便于理解的低维例子；一般形式由上面的 $R_i$ 说明。

### 2.4 Decoupled RoPE：分开算 Content Score 与 Position Score，再相加

**Decoupled = 解耦**。解决思路不是丢掉 RoPE，而是把本来耦合在一起的两件事拆开：

1. **内容分支 Content**：不用 RoPE，保留 MLA 的压缩和 Weight Absorption。
2. **位置分支 Positional / RoPE**：单独做 RoPE，并只保留较小的位置 Key；在 MLA 的设计中，位置 Key 可以跨 Heads 共享。

对于一个 Head，把 Query/Key 的特征在**特征列方向**拼在一起（不是把 token 分成两组）：

$$
q_t'=[\ q_t^C\;\mid\;q_t^RR_t^\top\ ],\qquad
k_i'=[\ k_i^C\;\mid\;k_i^RR_i^\top\ ].
$$

两个等长拼接向量的点积，会拆成两段点积之和：

$$
\boxed{
s_{ti}=(q_t^C)(k_i^C)^\top
+ (q_t^RR_t^\top)(k_i^RR_i^\top)^\top.
}
$$

如果 $k_i^C=c_iW_{UK}$，第一部分可继续吸收权重：

$$
(q_t^C)(k_i^C)^\top=(q_t^CW_{UK}^\top)c_i^\top.
$$

第二部分保留真正的位置旋转。于是：

$$
\boxed{
s_{ti}=
\underbrace{(q_t^CW_{UK}^\top)c_i^\top}_{\text{Content：可直接读压缩缓存 }c_i}
+
\underbrace{(q_t^RR_t^\top)(k_i^RR_i^\top)^\top}_{\text{Position：使用 RoPE}}.
}
$$

**具体怎么计算？** 以 3 个 token 为例：假设无旋转 Content 分支通过 MLA 得到

$$
S_C=\begin{bmatrix}3&6&9\\3&6&9\\3&6&9\end{bmatrix}.
$$

位置分支令所有原始二维位置向量都是 $[1,0]$，各 token 按 $0^\circ,90^\circ,180^\circ$ 旋转：

$$
Q_R'=K_R'=\begin{bmatrix}1&0\\0&1\\-1&0\end{bmatrix}.
$$

所以：

$$
S_R=Q_R'(K_R')^\top
=\begin{bmatrix}1&0&-1\\0&1&0\\-1&0&1\end{bmatrix}.
$$

把两个 Score 矩阵相加（**不是**把两组 Attention 权重分别 Softmax 后再相加）：

$$
\boxed{S=S_C+S_R=\begin{bmatrix}4&6&8\\3&7&9\\2&6&10\end{bmatrix}.}
$$

之后再做必要的缩放、加 Causal Mask，并对总 Score **统一做一次 Softmax**：

$$
A=\operatorname{softmax}_{\rm row}\!\left(\frac{S}{\sqrt{d_{C}+d_R}}+M\right).
$$

这里 $d_C+d_R$ 表示本示例拼接后的 Head Key 总维度；实际 MLA 实现的具体缩放约定以模型实现为准。

**到底缓存什么？** 主要保存每个历史 token 的压缩 $c_i$，再加上较小的旋转位置 Key。**不缓存历史 Q**。这让内容端可以继续使用 Weight Absorption，而位置端又有 RoPE 的相对距离信息。这就是第 6–7 页的两个加号、两条 Key 分支背后的全部原因。

---
## 3. GPT-3：Decoder-Only Transformer（第 8–11 页）

### 3.1 为什么叫 Decoder-Only？先从原始 Transformer 的 Encoder / Decoder 讲起

**你从英文直译的理解就是对的：Decoder-Only = 没有 Encoder、只有 Decoder 风格的 Transformer。** 但要知道“Decoder”最初从哪里来的，否则这个名字很空洞。

原始 Transformer（2017）的任务之一是机器翻译，例如输入 `I love you.`，输出 `Je t'aime.`。它有两大部分：

- **Encoder（编码器）**：接收整句英文，通过多层 *双向* Self-Attention + FFN，得到表示英文上下文的向量矩阵 $H$。
- **Decoder（解码器）**：已经生成了一部分法语，需要继续生成下一个法语 token。它先用 **Causal Self-Attention** 处理已生成的法语，再通过 **Cross-Attention** 读取 Encoder 的英文表示，最后经过 FFN。

这三个 Attention 名字有什么差别？从 Q、K、V 的来源看就懂了：

| 机制 | Q 从哪里来 | K、V 从哪里来 | 能否看未来 |
|---|---|---|---|
| Encoder 的 Self-Attention | 英文输入 $H$ | 同一段英文输入 $H$ | 通常可双向关注 |
| Decoder 的 Causal Self-Attention | 已生成法语 $D$ | 同一段已生成法语 $D$ | 不行，必须遮住未来 |
| Encoder–Decoder Cross-Attention | Decoder 当前隐藏表示 $D$ | **Encoder 的输出 $H$** | 可以读取整个**源语言**表示 |

以 Cross-Attention 为例，假设 Decoder 表示 $D\in\mathbb R^{m\times d}$、Encoder 输出 $H\in\mathbb R^{n\times d}$：

$$
Q=DW_Q\in\mathbb R^{m\times d_k},\qquad
K=HW_K\in\mathbb R^{n\times d_k},\qquad
V=HW_V\in\mathbb R^{n\times d_v}.
$$

所以 $QK^\top$ 是 $m\times n$，表示目标语言的每个位置要关注源语言的哪些位置。**Self-Attention 是同一个序列内部交换信息；Cross-Attention 是从另一段表示中读取信息。**

但是 GPT 的基础预训练任务是：**已知前面的 Token，预测下一个 Token**。它不需要一个单独的 Encoder 先处理另一段源序列。因此 GPT 沿用原始 Decoder 的自回归核心，却**删去读取独立 Encoder 输出的 Cross-Attention**。

一个 GPT Decoder-style Transformer Block 主要是：

$$
\boxed{\text{Causal Self-Attention}\ +\ \text{FFN}\ +\ \text{Residual / Norm}.}
$$

重复许多层，再接语言建模输出头。这就叫 **Decoder-Only Transformer**。它确实“只有 Decoder”，但也不是将原始 Encoder–Decoder Transformer 的完整 Decoder（含 Cross-Attention）原封不动地搬进来。

**为什么又叫 Masked / Causal Attention？** 两个词指向相同的限制，但强调点不同：

- *Causal*：当前 token 只能依赖当前及之前的 token，不能依赖未来；
- *Masked*：实现方式是在 Attention Score 里给未来位置加 $-\infty$，使 Softmax 后的权重为 0。

$$
\operatorname{CausalAttention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

**GPT 的 Causal Mask 与 BERT 的 `[MASK]` 不是一回事。** 前者是在 Attention 矩阵上遮挡未来位置，后者是把输入文本的个别 token 换成一个特殊占位符，让模型猜回原词。

### 3.2 GPT 到底训练什么？为什么训练可以并行而生成要逐个来？

老师第 8 页的图（橙色为多层 Masked Attention，右边为每个位置的词表概率）：

![](Pasted%20image%2020261121150101.png)

假设训练文本经过 Tokenizer 后是：

$$
[\mathrm{BOS},\ \text{I},\ \text{love},\ \text{machine},\ \text{learning}],
$$

其中 BOS（或 START）是人为定义的序列起始特殊 Token。给定每个位置已经看到的内容，模型要预测**下一个**正确 Token：

| 模型当前已知的上下文 | 正确下一个 Token |
|---|---|
| `[BOS]` | I |
| `[BOS] I` | love |
| `[BOS] I love` | machine |
| `[BOS] I love machine` | learning |

对于位置 $t$，最后一层输出一个 $1\times D$ 的隐藏向量 $h_t$；经词表输出层得到 $1\times|\mathcal V|$ 的 logits，再经过 Softmax 得到所有候选 token 的概率：

$$
z_t=h_tW_{\rm vocab}+b\in\mathbb R^{1\times|\mathcal V|},\qquad
p_t=\operatorname{softmax}(z_t).
$$

如果正确答案对应 token $x_{t+1}$，训练 Loss 是：

$$
\boxed{L=-\sum_t\log p_t(x_{t+1}).}
$$

例如正确答案是 `learning`，模型只给了它 $0.3$ 的概率，则该位置的 Loss 是 $-\log 0.3$。整个序列的 Loss 用所有有效预测位置的 Loss 合并（实践中常求平均）。经过 Backpropagation，不只是最后的输出层被更新，**各层 Attention 的 $W_Q,W_K,W_V,W_O$、FFN、Embedding 等都会被更新**。

**为什么训练时能一次性输入整句？** 因为训练文本的正确 token 已经全部知道，可以用 Teacher Forcing：一次前向传播并行计算每个位置的下一词预测；Causal Mask 保证前面位置不会偷看到后面的真实 token。**为什么推理时不能这样？** 因为未来 token 根本还没有生成，必须先生成一个、加入上下文、再算下一个。这就叫 **Autoregressive Generation（自回归生成）**。

推理时在每一层保存历史 KV Cache，就不用为了新 token 重新计算全部历史的 Q、K、V；每次新 token 的 Query 只需对历史和当前的 K、V 做一次 Attention。这也解释了第 2 节 MLA 为什么专门优化 KV Cache。

### 3.3 Greedy、Beam Search、Top-k Sampling：从 0 解释名字及完整选择过程

GPT 每一步输出的是**概率分布**，并没有自动决定最终选哪个 token。根据概率分布选 token 的规则，叫 **Decoding Strategy（解码策略）**。

**Greedy（贪心）**：每一步都选当前概率最大的 Token。例如 $P(A)=0.6,P(B)=0.4$，它直接选 A。问题是当前一步最高，不代表整段生成序列的**联合概率**最高。

**Beam Search（束搜索）**：Beam 可以理解成一条**正在生成的候选序列/路线**，Search 就是在这些候选路线里搜索。所谓 **Beam Width = 2**，不是 Attention 的维度，而是规定：**每一轮最多同时保留 2 条得分最高的候选路线**。

现在完整算一次（假设只需要生成两个 Token）：

1. 第一步模型输出 $P(A)=0.6,\ P(B)=0.4$。因为 Beam Width = 2，先**同时保留 A 和 B**，而不是像 Greedy 一样立即丢弃 B。
2. 第二步**分别**在两条路线后继续生成。假设：
   $$
   P(C\mid A)=0.5,\ P(D\mid A)=0.3,\ P(E\mid A)=0.2;
   $$
   $$
   P(C\mid B)=0.9,\ P(D\mid B)=0.1.
   $$
3. 得到所有候选序列和**整个序列的概率**（不是第二步单独的概率）：

   | 候选路线 | 计算 | 累计概率 |
   |---|---|---:|
   | B → C | $0.4\times0.9$ | **0.36** |
   | A → C | $0.6\times0.5$ | **0.30** |
   | A → D | $0.6\times0.3$ | 0.18 |
   | A → E | $0.6\times0.2$ | 0.12 |
   | B → D | $0.4\times0.1$ | 0.04 |

4. 再按累计概率排序，只保留 **B → C**、**A → C** 两条。若继续生成第三个 token，就从这两条再扩展、再筛选。

这样 Beam Search 就有机会找到 Greedy 因第一步贪心而错过的 B → C（$0.36>0.30$）。它是近似搜索，不保证找到全局最优；实际实现常用**累计 log probability**（乘积变成求和），还需要合理处理 EOS（序列结束符）、不同长度评分等。

**Top-k Sampling（Top-k 随机采样）**完全不是 Beam Search。假设某一步输出：

$$
P(A)=0.4,\ P(B)=0.3,\ P(C)=0.2,\ P(D)=0.1.
$$

若 $k=3$，先只保留**概率最高的三个 token** A、B、C，把 D 排除；再将前三个概率按总和 $0.9$ **重新归一化**：

$$
P'(A)=\frac{0.4}{0.9},\quad P'(B)=\frac{0.3}{0.9},\quad P'(C)=\frac{0.2}{0.9}.
$$

然后**按这个新分布随机采样**一个 token。A 最可能被选中，但不保证一定是 A。

| | Greedy | Beam Search | Top-k Sampling |
|---|---|---|---|
| 每一步干什么 | 直接选最大概率 | 扩展多条路线，保留累计得分最高的若干条 | 截断为最可能的 k 个 token，再随机采样 |
| 是否同时保留多条生成路线 | 否 | **是** | 通常否 |
| Width / k 的含义 | 无 | **保留多少条候选序列** | **允许随机选择多少个 Token** |
| 是否随机 | 否 | 典型 Beam Search 否 | 是 |

### 3.4 GPT-3 规模参数（老师第 10–11 页）

老师此处谈的是 **GPT-3 175B（2020）**，不是所有 GPT 模型的统一配置：

| 项目 | GPT-3 175B |
|---|---:|
| Transformer Decoder-style Layers | 96 |
| 每层 Attention Heads | 96 |
| 每个 Head 的 Q/K/V 维度 | 128 |
| Hidden Size / Embedding Dimension | 12,288 |
| 最大上下文窗口 | 2,048 Tokens |
| 总参数量 | 175B = 1750 亿 |
| 训练文本量 | 300B = 3000 亿 Tokens |
| 词表规模 | 大约 50,000 |

维度上：$96\text{ Heads}\times128\text{ 维/Head}=12,288$，恰好等于这个版本的 Hidden Dimension。这是常见的 Multi-Head 设计配置，不是所有 Head 拼接维度**必须**等于输入维度的数学规定。

---
## 4. BERT：它是怎样训练 Transformer 的？（第 12–14 页）

### 4.1 BERT 不是一种 Self-Attention 训练算法，而是完整模型

**这一段是理解老师第 12 页之前必须补上的基础。** 听到 “BERT 预训练 Self-Attention”，很容易误以为 BERT 就是一套单独训练 $W_Q,W_K,W_V$ 的方法。事实上需要区分三个层次：

| 层次 | 到底是什么 | 有没有需要训练的参数 |
|---|---|---|
| **Self-Attention** | 一种计算机制：$QK^\top\to\operatorname{softmax}\to AV$，使 token 从其他 token 聚合信息 | 有，$W_Q,W_K,W_V$ 等 |
| **Transformer Encoder Block** | 一个完整神经网络模块：**双向** Self-Attention、FFN、Residual、Norm | 有，上面各模块的权重都要学 |
| **BERT** | 把**多层 Transformer Encoder Block** 堆起来、再用特定预训练任务学习参数的整个模型 | 有，包括 Embedding、各层 Attention/FFN、预测头等 |

所以，说“BERT 的预训练会训练 Self-Attention 参数”，**没问题**；但说“BERT 就是训练 Self-Attention 的方法”，**不准确**，因为训练的是整个模型，而不只是 Self-Attention。

BERT = **Bidirectional Encoder Representations from Transformers**：

- **Bidirectional**：处理一个 token 时，可以关注左、右两侧的其他 token；
- **Encoder**：它使用的是 Transformer **Encoder-style Blocks**，而不是 GPT 那样的 Decoder-only 自回归结构；
- **Representations**：预训练希望学到每个 token 的、有上下文信息的向量表示；
- **from Transformers**：这些表示是多层 Transformer 的计算结果。

老师第 12 页原图中橙色区域的 Attention 连线能到达前后两侧 token，和 GPT 第 8 页的因果遮挡不同：

![](Pasted%20image%2020261121150201.png)

**那 BERT 到底怎样训练？** 先用一个完整例子。原始文本是 `I love machine learning.`，预处理随机选中了 `machine` 这个位置，训练输入变为：

```text
[CLS] I love [MASK] learning . [SEP]
```

将这些 token（包含两个特殊 Token）转成 Embedding，加入位置等信息：

$$
X\in\mathbb R^{n\times D}.
$$

送进 **多层双向 Transformer Encoder**：

$$
H=\operatorname{BERT}_{\theta}(X)
=\begin{bmatrix}h_{\rm CLS}\\h_{\rm I}\\h_{\rm love}\\h_{\rm MASK}\\h_{\rm learning}\\h_{.}\\h_{\rm SEP}\end{bmatrix}\in\mathbb R^{7\times D}.
$$

这里 $H$ 的**每一行仍对应输入的一个 token 位置**，不过现在已经融合了上下文信息。为了猜 `machine`，我们取的是被选中位置的**输出向量 $h_{\rm MASK}\in\mathbb R^{1\times D}$**，不是随便把所有输出拼成一个巨大向量：

$$
z_{\rm MASK}=h_{\rm MASK}W_{\rm vocab}+b\in\mathbb R^{1\times|\mathcal V|},\qquad
p=\operatorname{softmax}(z_{\rm MASK}).
$$

词表 $\mathcal V$ 里的每个 token 都有一个概率。假设模型给正确答案 `machine` 的概率只有 $0.3$：

$$
L_{\rm MLM}=-\log P(\text{machine}\mid\text{带缺口的输入})=-\log 0.3.
$$

然后 **Backpropagation** 会把这个 Loss 的梯度从预测头传回 Transformer Encoder 各层和 Embedding。更新的参数包括：

$$
\boxed{W_Q,W_K,W_V,W_O,\ \text{FFN 权重},\ \text{Embedding},\ \text{预测头参数},\ldots}
$$

**这才是“BERT 训练 Self-Attention”的正确含义：Self-Attention 在整个 BERT 网络里面，和其他可训练模块一起被优化。** 和 GPT 相比，两者都是通过预测 Token 进行自监督学习，但 GPT 通常预测**下一个 Token**，BERT 的经典任务是根据**左右两侧上下文**预测被选中的原始 Token。

### 4.2 输入为什么要人为加 `[CLS]`、`[SEP]`？它们到底在哪里出现？

这两个 Token 在正式进入 Transformer 前，**就由数据处理程序放好**，不是 Self-Attention 在运行过程中自己产生的。

普通输入例如：

```text
I love cats .
```

为了配合 BERT 的输入格式，可以写成：

```text
[CLS] I love cats . [SEP]
```

如果需要明确指定**两段输入**（常叫 Segment A、Segment B），例如：

```text
[CLS] I love cats . [SEP] They are cute . [SEP]
```

- **`[CLS]` = Classification**：放在整个输入最前面，是一个*专门预留的输出位置*，以后可以从它的最终隐藏向量预测整个输入的类别；
- **`[SEP]` = Separator**：标记**数据预处理选定的两段输入的分界线**，或者标记一段输入的结束。它不自动禁止不同段之间相互做 Attention。

这跟普通句号、换行到底有什么不同？**不是说 BERT 看不懂句号。** 举个真正能看出区别的例子，原始文本完全相同：

```text
I went shopping. I bought a book. The book was interesting.
```

你可能将其分为：

```text
[CLS] I went shopping. [SEP] I bought a book. The book was interesting. [SEP]
```

也可能将其分为：

```text
[CLS] I went shopping. I bought a book. [SEP] The book was interesting. [SEP]
```

两者句号一模一样，但**我们指定的 Segment A、B 不同**。`[SEP]` 是为了显式传达我们选定的分段边界，而不是替代英语标点。

**那用“换行 + 两个空格”不行吗？** 完全可以**设计另一套输入规范**，但 BERT 原始 Tokenizer 处理普通空格和换行时通常把它们当分词分隔，不能依赖“两个空格”在 Token ID 序列中原样保留。`[SEP]` 则是词表里有专门 Token ID 的特殊 Token，会留下一个可学习的 Embedding。你也可以设计一个 Tokenizer，让它把某种换行格式显式转换成特殊 Token；只是**原始 BERT 采用了 `[SEP]`**。

`[SEP]` 的位置由预处理决定，**不是**永远固定在第几个位置；但它本身的 Embedding 是训练出来的。`[CLS]` 同理：它作为输入中的第一个 Token，其 Embedding 也会随训练更新。

BERT 输入向量通常还会合并三类信息：

$$
\boxed{X=E_{\rm token}+E_{\rm position}+E_{\rm segment}.}
$$

- Token Embedding：这个 Token 是 `cat`、`[CLS]`、`[SEP]` 还是别的；
- Position Embedding：在整个输入中的位置；
- Segment Embedding：这个位置属于 A 段还是 B 段。**Segment 与自然语言中“用句号划分的单句”不是同一概念**，一段可以有多句。

### 4.3 `[CLS]` 为什么能用于分类？标签在训练前就给好了吗？

先解释 `[CLS]` 的计算。假设输入 5 个 token：

```text
[CLS] I love cats [SEP]
```

经过 BERT 某一层 Self-Attention，Attention 权重矩阵是 $5\times5$。第 0 行对应 `[CLS]` 发出的 Query：

$$
A=\begin{bmatrix}
a_{00}&a_{01}&a_{02}&a_{03}&a_{04}\\
a_{10}&a_{11}&a_{12}&a_{13}&a_{14}\\
\vdots&\vdots&\vdots&\vdots&\vdots\\
a_{40}&a_{41}&a_{42}&a_{43}&a_{44}
\end{bmatrix}.
$$

第 0 行乘以 V，得到当前层 `[CLS]` 的新特征：

$$
\boxed{o_{\rm CLS}=a_{00}v_{\rm CLS}+a_{01}v_{\rm I}+a_{02}v_{\rm love}+a_{03}v_{\rm cats}+a_{04}v_{\rm SEP}.}
$$

BERT 是双向 Self-Attention，`[CLS]` 可以看到整个输入。经多层 Transformer 后得到 $h_{\rm CLS}\in\mathbb R^{1\times D}$。**它不是因为名字里有 Classification 就天生会分类**，而是因为我们**把它作为固定的输出位置，接上分类头，再用标签训练**，才学会提供有用的整体表示。

假设情感分类任务的标签是 `Positive` 或 `Negative`。最简单的分类头是：

$$
z=h_{\rm CLS}W_{\rm cls}+b,\quad W_{\rm cls}\in\mathbb R^{D\times2},\qquad p=\operatorname{softmax}(z).
$$

若 $p=[0.8,0.2]$，正确 Label 是 Positive，Loss 就是 $-\log0.8$。训练时反向传播更新分类头，也可进一步更新 BERT 全部或部分参数。原始 BERT 的 NSP 头还可经过一个专门的 Pooler（如 Dense + Tanh）再做分类，但“从 `[CLS]` 的最终表示预测整体类别”这件事没变。

**分类的 Label 到底哪里来？** 要分两类任务：

1. **预训练的 Next Sentence Prediction（NSP）**：我们提前决定哪两段构成训练样本。如果 B 真的是语料中 A 紧接着的下一段，自动生成 `IsNext=1`；如果 B 是随机构造的不相邻段，自动生成 `NotNext=0`。这是**利用原始文本自动产生监督信号**，不需要人工给每一对句子打标签。
2. **下游情感分类的 Fine-Tuning**：使用一批已经带 `Positive/Negative` 等标签的训练数据（通常由人工或其他标注流程提供）。模型通过 Loss 学会分类。真正推理时只输入文本，没有正确标签让它偷看；它要自己输出预测的类别。

所以 **`[CLS]` 是输入中的特殊 Token；Label 是训练任务准备好的正确答案**。这两者不是同一个东西。

### 4.4 MLM：随机选中 15%，再按 80% / 10% / 10% 处理

MLM = **Masked Language Modeling（掩码语言建模）**。老师第 12–13 页核心预训练任务就是它。操作顺序不要颠倒：

**先**从原始文本随机选中大约 $15\%$ 的 Token 位置作为预测目标；**再**对这些已选中的位置，按下面概率处理：

| 对**选中的位置** | 大约占选中位置的比例 | 输入给 BERT | 正确预测目标 |
|---|---:|---|---|
| 替换为 `[MASK]` | 80% | `The cat is [MASK]` | 原来的 `sleeping` |
| 替换成一个随机词 | 10% | `The cat is running` | 仍是原来的 `sleeping` |
| 保留原词 | 10% | `The cat is sleeping` | **仍是** `sleeping` |

举例：原始句子 `The cat is sleeping on the sofa.`，预处理选中了 `sleeping` 这个位置。注意：无论它最后呈现为 `[MASK]`、`running` 还是原词 `sleeping`，**Loss 对照的永远是原来的 `sleeping`**。它要学的是“恢复被选中位置原本的词”，不是“预测你替换后的那个词”。

**为什么不把选中的 15% 全部改成 `[MASK]`？** 因为 `[MASK]` 是专门为预训练引入的符号。很多下游任务（情感分类、实体识别等）输入都是正常句子，不会有 `[MASK]`。如果模型在预训练时只见过这种被挖洞的输入，就可能过于依赖特殊占位符，造成预训练和真实使用的输入差异。因此原始 BERT 让少部分选中位置变成随机词，或者**保持原词不变**，使模型也处理非 `[MASK]` 的普通输入。

**但保留原词不是直接给答案了吗？** 是的，这种情况更容易；它占比很小，目的不是增加难度，而是让被监督的位置不总是带着 `[MASK]`。如果总共有 100 个 Token，期望约 15 个被选中预测，其中约 12 个换成 `[MASK]`、1.5 个换随机词、1.5 个保留原词；实际整数数量随随机抽样变化。**没被选中的 85% 仍然参与 Attention 提供上下文，但不针对这些位置计算该任务的 MLM Loss。**

再次注意：**MLM 的 `[MASK]` 是输入序列中一个特殊 Token，不等于 GPT 的 Causal Attention Mask。** BERT 的双向 Self-Attention 可以在左右两个方向看上下文；它只是将被选中的某些输入位置替换或保留，以此构造预测任务。

### 4.5 NSP 与 Fine-Tuning：老师第 13–14 页图怎样读？

NSP = **Next Sentence Prediction**。目标是判断：指定的 Sentence/Segment B 是否紧接着 A 出现在原始文本中。例如：

```text
[CLS] I bought a book . [SEP] I read it yesterday . [SEP]
```

如果这两段在语料里相邻，训练标签是 `IsNext`；如果 B 被替换成随机选来的其他段，标签是 `NotNext`。这个任务用 `[CLS]` 的最终输出做二分类，典型计算流程是：

$$
\boxed{h_{\rm CLS}\ \longrightarrow\ \text{分类头}\ \longrightarrow\ \operatorname{softmax}\ \longrightarrow\ P(\mathrm{IsNext}),P(\mathrm{NotNext}).}
$$

> **原笔记的一处需要明确纠正的地方**：第 13 页最后写“Predict this on the sep token”。原始 BERT 的 NSP **使用 `[CLS]` 的最终表示**做句子对分类；`[SEP]` 负责分隔两个输入 Segment。不能因为 `[SEP]` 放在句子中间就把它当作 NSP 的预测位置。

原始 BERT 的预训练结合两种自监督任务：**MLM 预测选中位置的原词，NSP 根据句子对判断是否相邻**。两种任务会共同更新 BERT 的参数；NSP 并不是所有后续 BERT 变体都必需采用的任务。

完成通用预训练后，可以进行 **Fine-Tuning（针对特定任务继续训练）**：不是从零重训，而是在预训练模型上增加任务头，用新的带标签样本继续优化。老师第 14 页两张图如下：

![](Pasted%20image%2020261121150301.png)

- **图 (a)：Sentence Classification（整句分类）**。例如判断一条评论是 Positive 还是 Negative。只要一个整句结果，常取 $h_{\rm CLS}$ 接分类头。训练前准备好每条评论的标签；推理时不用给标签。
- **图 (b)：Token Classification（逐词分类）**。例如 NER（Named Entity Recognition，命名实体识别），要判断 `Zara` 是 PERSON，`Victoria` 是 LOCATION 等。于是分别取对应 Token 的 $h_i$，每个 Token 输出自己的分类结果，并不只用 `[CLS]`。

现在能把整个 BERT 串起来了：

$$
\boxed{\text{原始文本}\to\text{Tokenizer / 特殊 Token / Embedding}\to\text{多层 Encoder}\to\text{各位置输出 }H\to\text{任务预测头}\to\text{Loss}\to\text{反向传播更新整个模型}}.
$$

**GPT 与 BERT 的共同点**：都通过训练 Token 预测任务来训练整套 Transformer，包括 Self-Attention、FFN 和 Embedding。**主要不同点**：GPT 用 Causal Self-Attention 预测下一个 Token；BERT 用双向 Encoder Self-Attention，在经典预训练中通过 MLM 等任务学习上下文表示，再用 Fine-Tuning 适配具体任务。BERT **不是**一种只训练 Attention、不训练其他参数的方法。

---
## 5. Gated Linear Units 与 SwiGLU（第 15 页）

### 5.1 先理解普通 FFN：它在 Transformer 的哪个位置？

一个常见 Transformer Block 主要是 Self-Attention + FFN（Feed-Forward Network，前馈网络；也经常叫 MLP）。Self-Attention 让**不同 Token 之间**交换信息；FFN 则对每个 token 的特征**分别**做非线性变换，所有 token **共享同一套 FFN 参数**。

假设 Self-Attention 和相应残差 / Norm 后：

$$
U\in\mathbb R^{n\times D},\quad D=512.
$$

一个典型的两层 FFN 可以是 $512\to2048\to512$：

$$
\operatorname{FFN}(U)=\operatorname{ReLU}(UW_1+b_1)W_2+b_2,
$$

$$
W_1\in\mathbb R^{512\times2048},\quad W_2\in\mathbb R^{2048\times512}.
$$

矩阵维度：

$$
[n\times512]\to[n\times2048]\to[n\times512].
$$

原来的 FFN 只通过一个隐藏分支做激活。第 15 页的 GLU / SwiGLU 是想在 FFN 内加入**另一个分支来动态控制特征**；这和后面第 16–18 页的 MoE 是两条不同的 FFN 改造路线。

### 5.2 GLU：为什么要有两条分支？

GLU = **Gated Linear Unit（门控线性单元）**。老师的写法是：

$$
\boxed{\operatorname{GLU}(x)=(xW+b)\odot\sigma(xV+c)},
$$

这里 $x\in\mathbb R^{1\times D}$ 是一个 token 的**行向量**；$W,V\in\mathbb R^{D\times d_f}$；两个分支输出都为 $1\times d_f$。$\odot$ 是**逐元素相乘**，不是矩阵乘法：

$$
[2,3]\odot[0.8,0.1]=[1.6,0.3].
$$

两条分支分别扮演：

- **Value / Content 分支 $xW+b$**：提出一组要输出的特征；
- **Gate 分支 $\sigma(xV+c)$**：产生一组在 $0$ 与 $1$ 之间的控制系数。

当某个 Gate 接近 1，就较多保留这一维特征；接近 0，就显著抑制它。它不是一个人工设置的二值开关，而是通过训练由当前 $x$ 决定的连续可微函数。

接入完整 Transformer FFN 时，通常还需要一个 **Down Projection** 把中间维度从 $d_f$ 降回 $D$，这样可以做残差连接。例如：

$$
Y=\big[(UW+b)\odot\sigma(UV+c)\big]W_{\rm down},
\quad W_{\rm down}\in\mathbb R^{d_f\times D},\quad Y\in\mathbb R^{n\times D}.
$$

### 5.3 SwiGLU：把 Sigmoid Gate 换成 Swish / SiLU

老师给出的变体是：

$$
\boxed{\operatorname{SwiGLU}(x)=(xW+b)\odot\operatorname{Swish}(xV+c).}
$$

Swish 的一种定义是：

$$
\operatorname{Swish}_{\beta}(z)=z\,\sigma(\beta z),\qquad
\sigma(z)=\frac1{1+e^{-z}}.
$$

当 $\beta=1$，就得到常说的 **SiLU**：

$$
\boxed{\operatorname{SiLU}(z)=z\,\sigma(z).}
$$

与 $\operatorname{ReLU}(z)=\max(0,z)$ 不同，SiLU 在负数区间不一定是 0。例如：

$$
\operatorname{ReLU}(-1)=0,\qquad \operatorname{SiLU}(-1)=-\frac1{1+e}\approx-0.269.
$$

SwiGLU 常见完整 FFN 形式为（省略 Bias）：

$$
\boxed{\operatorname{FFN}_{\rm SwiGLU}(U)=\big[\operatorname{SiLU}(UW_{\rm gate})\odot(UW_{\rm up})\big]W_{\rm down}},
$$

$$
W_{\rm gate},W_{\rm up}\in\mathbb R^{D\times d_f},\qquad
W_{\rm down}\in\mathbb R^{d_f\times D}.
$$

维度完全可以手算：

$$
\underbrace{[n\times D][D\times d_f]}_{\text{Gate 分支 }n\times d_f}\ \odot\ 
\underbrace{[n\times D][D\times d_f]}_{\text{Up 分支 }n\times d_f}
\to[n\times d_f]\to[n\times D].
$$

图中红框是 Gated MLP：有两条分支，其中一条经过非线性，两者在 $\odot$ 处逐元素相乘，再输出；两侧 H3、Mamba 示意别的序列模块也可能采用门控机制。**这张图不是说 GLU 必须有 SSM 或卷积**。

![](Pasted%20image%2020261121150401.png)

老师笔记里标记 $\beta$ 为 *Learnable*，意思是 Swish **允许**把 $\beta$ 设为可学习参数；但很多现代 SwiGLU 直接使用固定 $\beta=1$ 的 SiLU。Sigmoid Gate 和 Swish Gate 也不要误认成完全一样：Swish 的值不局限在 $[0,1]$。

---

## 6. Mixture of Experts（MoE，第 16–18 页）

### 6.1 MoE 到底接在 Transformer 的哪里？先从“只有一个 FFN”讲起

MoE = **Mixture of Experts（混合专家）**。这里的 *Expert* 通常就是一套**独立参数的完整 FFN**，不是把一个 FFN 的矩阵任意切几份；*Mixture* 表示按输入选择和组合不同 Expert 的结果。

普通 Transformer Block 的一段流程是：

$$
X\ (n\times D)\ \longrightarrow\ \text{Self-Attention + Residual / Norm}\ \longrightarrow\ U\ (n\times D)
\ \longrightarrow\ \boxed{\text{同一个 FFN 处理所有 Token}}\ \longrightarrow\ M\ (n\times D).
$$

假设 $D=512$、$n=3$：

$$
U=\begin{bmatrix}u_1\\u_2\\u_3\end{bmatrix}\in\mathbb R^{3\times512},\qquad u_i\in\mathbb R^{1\times512}.
$$

**普通 FFN** 对三个 token 分别计算 $\operatorname{FFN}(u_1),\operatorname{FFN}(u_2),\operatorname{FFN}(u_3)$，但是用**同一套** $W_1,W_2$。可以设计成 $512\to2048\to512$，每个 Token 的最终输出都是 $1\times512$。

现在 MoE 把**一个共享 FFN**替换成：

$$
\boxed{\text{Router + 多套不同参数的 FFN（Experts）+ 选中结果的合并}}.
$$

例如放四套：$E_1,E_2,E_3,E_4$，每套都接受 $1\times512$，也输出 $1\times512$，但有自己的权重。Token 1 可能运行 $E_2$，Token 2 可能运行 $E_1$；如果选择 Top-2，一个 Token 也可以运行两套 FFN 并将输出加权相加。**是每个 Token 单独选 Expert，不是整句话只进入一个 Expert。**

最重要的接线关系：**Attention 先完成；它的 $n\times D$ 输出 $U$ 进入 MoE；Router 为每个 Token 选中 Expert；Expert 输出返回这个 Token 的原位置；MoE 最后仍输出 $n\times D$；再接残差 / Norm。** MoE 不是接在 Attention 的每个 Head 里面，也不是在 FFN 后额外叠一层更大的 FFN。

### 6.2 Router 真正怎么算？从输入矩阵算到 Expert 输出并接回残差

这一节用**完整矩阵**演示，统一采用“每个 Token 是一行”的约定，不会把 $1\times D$ 突然写成 $D\times1$。

为了手算，假设有 **2 个 Token、每个 Token 为 2 维、3 个 Experts**。Attention（及相应残差/Norm）之后：

$$
U=\begin{bmatrix}1&2\\3&1\end{bmatrix}\in\mathbb R^{2\times2}.
$$

Router 是一个小的可训练投影，它的任务仅仅是**给每个 Token 的三个 Expert 打分**。所以其权重矩阵有 **3 列**，不需要保持 Token 的 2 维：

$$
W_r=\begin{bmatrix}1&0&-1\\0&1&-1\end{bmatrix}\in\mathbb R^{2\times3}.
$$

**第一步：计算 Router Scores。**

$$
R=UW_r
=\begin{bmatrix}1&2\\3&1\end{bmatrix}
\begin{bmatrix}1&0&-1\\0&1&-1\end{bmatrix}
=\boxed{\begin{bmatrix}1&2&-3\\3&1&-4\end{bmatrix}}
\in\mathbb R^{2\times3}.
$$

第一行是 Token 1 对 E1、E2、E3 的三个分数；第二行对应 Token 2。**Router 输出 $n\times N_{\rm experts}$，不是最后给 Transformer 的 Token 特征矩阵。**

**第二步：对每一行 Softmax，变成 Routing Weights。**

$$
P=\operatorname{softmax}_{\rm row}(R)
\approx\begin{bmatrix}0.268&0.727&0.005\\0.880&0.119&0.001\end{bmatrix}.
$$

**第三步：选 Top-1 Expert。** 这里 Top-1 表示“每个 Token 只选得分最高的 **1 套 FFN**”。Token 1 选择 E2（0.727 最大）；Token 2 选择 E1（0.880 最大）。注意：**输入 Expert 的不是 Router 输出的那三个分数，而是原来的完整 Token $u_i$！**

**第四步：真正执行 FFN。** 为了手算，假设：

$$
E_1(u)=\operatorname{ReLU}\!\left(u\begin{bmatrix}2&0\\0&1\end{bmatrix}\right),\quad
E_2(u)=\operatorname{ReLU}\!\left(u\begin{bmatrix}1&0\\0&3\end{bmatrix}\right).
$$

这两条是用简化的矩阵表现不同 Expert 的不同参数；真实 Expert 往往是完整的升维—非线性—降维 FFN。按刚才路由结果：

$$
E_2(u_1)=[1,2]\begin{bmatrix}1&0\\0&3\end{bmatrix}=[1,6],
$$

$$
E_1(u_2)=[3,1]\begin{bmatrix}2&0\\0&1\end{bmatrix}=[6,1].
$$

**第五步：用路由权重缩放对应 Expert 的输出**（对应老师第 17 页 “multiply output of expert by routing probability”）：

$$
m_1\approx0.727[1,6]=[0.727,4.362],
$$

$$
m_2\approx0.880[6,1]=[5.280,0.880].
$$

**第六步：按 Token 原来的行号放回 $n\times D$ 矩阵**，不是把不同 Token 相加，也不是把 Expert 输出沿特征维度拼到一起：

$$
\boxed{M=\begin{bmatrix}0.727&4.362\\5.280&0.880\end{bmatrix}\in\mathbb R^{2\times2}.}
$$

最后接回原来的 Residual + Norm，例如老师第 16 页图中的 Post-Norm 写法：

$$
Y=\operatorname{Norm}(U+M),\qquad
U+M=\begin{bmatrix}1.727&6.362\\8.280&1.880\end{bmatrix}.
$$

**维度全过程**：

$$
\boxed{U:[n\times D]\ \xrightarrow{W_r:[D\times E]}\ R:[n\times E]\ \xrightarrow{\text{Routing + Experts}}\ M:[n\times D]\ \xrightarrow{\text{Residual / Norm}}\ Y:[n\times D].}
$$

**这里的 E 是 Expert 的数量，不是 Token Feature Dimension！** 例如 $D=512,E=4$ 时：一个 Token 是 $1\times512$；Router 权重是 $512\times4$；Router 对它给出四个 Score，是 $1\times4$；选中的 Expert 仍接收 $1\times512$ 并输出 $1\times512$；最后 MoE 输出仍是 $1\times512$。中间分数维度变小，不代表整个 MoE 把 Token 特征永久缩成四维。

### 6.3 Top-k 不是 Multi-Head Attention；Expert 输出如何“混合”？

Top-k Routing 的 $k$ 是**每个 Token 要调用多少个 Expert**。比如某个 Token 的 Routing Weights 是：

$$
p=[0.1,0.4,0.3,0.2].
$$

- Top-1：只执行 $E_2(u)$，可以得到 $0.4E_2(u)$；
- Top-2：执行 $E_2(u)$ 与 $E_3(u)$，再合并，例如 $0.4E_2(u)+0.3E_3(u)$。

有些实现会把选中两个 Expert 的权重重新归一化，变成 $(0.4/0.7)E_2(u)+(0.3/0.7)E_3(u)$；具体使用哪种要看模型设计。**无论哪种，Expert 输出维度必须可以彼此相加，并能接回 Transformer。**

一般写作：

$$
\boxed{m_i=\sum_{j\in\operatorname{TopK}(p_i)}\alpha_{ij}E_j(u_i),\qquad m_i\in\mathbb R^{1\times D}.}
$$

与 Multi-Head Attention 的区别很大：多个 Head 通常**各自运行并沿特征方向 Concat**，最后用 $W_O$ 投影；MoE 的多个 Expert 则是 **由 Router 为每个 token 选出少数几个，输出在同一 $D$ 维空间中按权重求和**。不要把“多个 Head”和“多个 Expert”理解成同一种划分。

### 6.4 为什么说 MoE 高效？比较基准必须说清楚

**不能说“在一个单 FFN 上改成 64 个 Expert，Top-8 之后就一定更省计算”。那是错误的比较。**

假设原始单个 FFN 有 $P$ 个参数，每处理一个 Token 的 FFN 计算量约为 $F$。现在造 64 个**与原 FFN 同等大小**的 Expert，每个 Token 选 8 个：

| 比较对象 | FFN 总参数量 | 每 Token 专家部分的计算量（粗略） |
|---|---:|---:|
| 原始：仅 **1 个** FFN | $P$ | $F$ |
| 扩成 **64 个** FFN、**全部执行** | $64P$ | $64F$ |
| MoE：**64 个** Expert、**Top-8** | $64P$ | $8F$，外加 Router 等开销 |

所以在这个假设下，**MoE 相比原来的单个 FFN，计算是增加了，不是减少了**；它比“同样拥有 64 个 Expert、但每次全部执行”的密集方案便宜得多。

MoE 的真实目标是：

$$
\boxed{\text{让模型拥有大量总参数，但每个 Token 只激活其中一部分参数。}}
$$

术语：

- **Total Parameters**：模型总共有多少可训练参数；
- **Active Parameters**：在处理某个 Token 时参与主要计算的参数。MoE 中两个数量可以差距很大。

这并**不**代表总计算量按 $8/64=1/8$ 缩小：还有 Attention、Router、共享层、Expert 的调度和跨设备通信；也**不**表示一定比一个小 Dense FFN 更快。优势是同等大的模型容量不再需要每个 Token 执行全部专家参数，属于**参数规模 / 激活计算量的折中**。

### 6.5 Switch Transformer、Router 训练、OLMoE：老师第 16–18 页

先看老师第 16 页 Switch Transformer 结构图：

![](Pasted%20image%2020261121150501.png)

它沿着这条路径读：

$$
\text{Token}\to\text{Self-Attention}\to\text{Add + Norm}\to\boxed{\text{Switch FFN：Router 选择 Expert}}\to\text{按权重缩放}\to\text{Add + Norm}.
$$

图中两个 Token `More`、`Parameters` 各自先通过同一层 Self-Attention，然后走自己的 Router。`More` 选 FFN 2，权重约 0.65；`Parameters` 选 FFN 1，权重约 0.8。它们算出的 FFN 输出分别回到原 Token 的位置。**Switch Transformer 的关键特点是 Top-1 Routing：每个 Token 选择一个 Expert。** 这就是 Switch（切换/选择）这个名字的由来。

**Router 和 Expert 如何学会选择？** 一开始 Router 只是一套参数 $W_r$，并没有人提前指定“E1 必须负责数学、E2 必须负责英语”。语言模型仍然通过 Next-Token Loss 等任务训练；对于被选中专家，任务 Loss 可沿其输出反向更新 Expert 参数，也可以沿连续的 Routing Weight 更新 Router 参数。Top-k 里的离散索引选择本身并不是普通的可微算子，因此具体实现会对梯度和负载分配做相应设计。

如果 Router 越来越爱选某个 Expert，容易出现：某些 Expert 处理大量 Token，而其他 Expert 很少获得训练机会。这是 **Load Imbalance（负载不均衡）**。第 17 页写的 *Balanced routing at initialization* 只是初始阶段的一个要求；训练过程中也需要防止负载集中。常见做法是为专家负载增加一个辅助 Loss：

$$
\boxed{L_{\rm total}=L_{\rm task}+\lambda L_{\rm balance}.}
$$

$L_{\rm task}$ 负责模型任务表现；$L_{\rm balance}$ 鼓励更多 Expert 获得较均衡的处理机会。不同 MoE 还可能有每个专家的容量限制与 Dropless（不因容量上限丢掉 Token 的专家计算）路由等设计。

第 18 页的 OLMoE 是把这种稀疏 Expert 结构用于语言模型：左边 Dense Model 的每一层是普通 FFN；右边在 FFN 位置换成 MoE，**Attention 并未因此消失**。

![](Pasted%20image%2020261121150601.png)

老师附上的 OLMoE-1B-7B 表格包含以下具体配置：

![](Pasted%20image%2020261121150701.png)

| 老师图中的 OLMoE 配置 | 含义 |
|---|---|
| **64 Experts** | 总共 64 套专家 FFN |
| **Top-8** | 每个 Token 在该专家层激活 8 套 Expert |
| **6.9B Total** | 模型总参数约 69 亿 |
| **1.3B Active** | 每个 Token 约 13 亿参数参与模型计算，**包含非 Expert 的部分**，不等于单独 8 套 Expert 的参数和 |
| **No shared Expert** | 这个配置不另设每次都调用的共享专家 |
| **Dropless, Token-choice routing** | 由 Token 选择 Expert，避免直接丢弃超出容量的 Token |
| **Load Balancing Loss 0.01** | 辅助优化专家负载分配 |
| **Router z-loss 0.001** | 抑制过大的 Router Logits、帮助训练稳定的另一项辅助损失 |

到这里就可以把整个 MoE 过程总结为一句**具体的计算流程**：

$$
\boxed{
U:[n\times D]\ \longrightarrow\ R=UW_r:[n\times E]\ \longrightarrow\ \text{每行 Top-k 选专家}\ \longrightarrow\ \text{各专家执行完整 }1\times D\text{ Token}\ \longrightarrow\ \text{按权重求和、放回原行}\ \longrightarrow\ M:[n\times D]\ \longrightarrow\ \text{Residual / Norm}.
}
$$

**不要再混淆两个“为什么”**：*MoE 为什么能工作*，答案是 Router 为不同 Token 动态选 FFN 并汇合输出；*MoE 为什么值得使用*，答案是能把**总参数容量**扩大得比**每 Token 激活的计算量**更快，而不是凭空让原来单 FFN 的计算变少。

---

> **本笔记到此结束：只覆盖 PDF 第 1–18 页。第 19 页及之后的 Qwen Next、Gated DeltaNet、GPT 发展史及后续架构图均未纳入。**
