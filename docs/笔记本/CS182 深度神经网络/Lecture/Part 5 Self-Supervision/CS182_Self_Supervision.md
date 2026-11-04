---
title: "CS182 · Self-Supervised Learning 复习笔记"
subtitle: "Lecture 13/14 · Pages 17–22｜含课堂图与疑难推导"
date: "2026-11-04"
---

# CS182 · Self-Supervised Learning（自监督学习）

> **范围**：`13_rnn_fa26(2).pdf` 第 **17–22** 页。第 13–16 页的 Kalman Filtering 是本节的引入；Lecture 15 讲 State Space Models，不属于本节。  
> **标记说明**：正文依照老师笔记的顺序；标为 **[补充]** 的 SVD 推导、例子、Masked Loss 与稀疏惩罚，是为了回答学习过程中遇到的问题，**不是老师笔记原本逐项讲解的内容**。

## 1. Self-Supervision：标签可以从数据中获得（P17）

- **监督学习**：训练集给出输入 $x$ 与人工提供的标签 $y$，最小化预测与标签的误差。
- **自监督学习**：不依赖人工标签，而是**从原始数据构造训练目标**（老师写的是 *Make your own labels (from data)*）。它仍然有 Loss、反向传播与训练目标。
- 老师的示例思路：**Next-thing prediction（预测下一个元素）**。

例如句子 `I love playing basketball`：用 `I love playing` 预测 `basketball`，真实的 `basketball` 就来自训练文本，不需要人工打标签。

$$
\hat x_{t+1}=f_\theta(x_1,\ldots,x_t),\qquad
\mathcal L=\ell(\hat x_{t+1},x_{t+1}).
$$

**易错点**：真实的 $x_{t+1}$ 是**计算 Loss 的目标**；在这个因果预测任务中，不能提前把它作为同一步预测的输入。

**与 Kalman Filtering 的联系**：真实隐状态 $z_t$ 可能不可观察，但可以利用可观察的数据设计预测／重建目标，让模型间接学习潜在状态。这对应老师的 *partial pattern / scaffolding* 提示。要注意具体预测任务的输入、目标和时间关系，避免把待预测的量提前泄漏给模型。


## 2. PCA：用 SVD 找到主要方向（P18–19）

### 2.1 先把所有维度固定下来

设有 $m$ 个样本，每个样本 $n$ 个特征，降维后留下 $k$ 维，且 $k\le\min(m,n)$。

$$
X\in\mathbb R^{m\times n}
$$

**约定：每行是一个样本；PCA 前先将每个特征减去其训练集均值**。下面的 $X$ 均指中心化后的矩阵。

设 $q=\min(m,n)$，使用截短尺寸的 SVD 表示：

$$
\underbrace{X}_{m\times n}=
\underbrace{U}_{m\times q}
\underbrace{\Sigma}_{q\times q}
\underbrace{V^\top}_{q\times n}.
$$

这里 $U=[u_1,\ldots,u_q]$，$V=[v_1,\ldots,v_q]$，$\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_q)$，并按 $\sigma_1\ge\cdots\ge\sigma_q\ge0$ 排序。

| 对象 | 尺寸 | 作用 |
|:--|:--|:--|
| $u_i$ | $m\times1$ | **左奇异向量**，位于样本空间 |
| $v_i$ | $n\times1$ | **右奇异向量**，位于特征空间 |
| $\sigma_i$ | 标量 | 对应方向的奇异值（非负） |

**[补充：右奇异向量是什么？]** 它是 $X^\top X$ 的单位特征向量，满足

$$
X^\top Xv_i=\sigma_i^2v_i,\qquad Xv_i=\sigma_i u_i\quad (\sigma_i>0).
$$

因此 $v_i$ 描述原始 **$n$ 维特征空间**中的一个特殊方向；$\sigma_i$ 对应该方向的伸缩强度。在 PCA 中，最大的奇异值所对应的方向承载最多的样本方差（前提：数据已中心化）。“右”来自它们组成 $V$，出现在分解式的右侧 $V^\top$。

### 2.2 PCA 的真正降维公式：注意 $X$ 和 $x$ 的区别

取前 $k$ 个右奇异向量：

$$
V_k=[v_1,\ldots,v_k]\in\mathbb R^{n\times k}.
$$

**整个数据矩阵**的降维：

$$
\boxed{\underbrace{Z}_{m\times k}=
\underbrace{X}_{m\times n}
\underbrace{V_k}_{n\times k}}
$$

**单个样本作为列向量** $x\in\mathbb R^{n\times1}$ 时，才写成：

$$
\boxed{\underbrace{z}_{k\times1}=
\underbrace{V_k^\top}_{k\times n}
\underbrace{x}_{n\times1}}
$$

> **之前遇到的维度错误**：不能把 $m\times n$ 的整批数据 $X$ 直接放进 $V_k^\top X$。整批样本应写 $Z=XV_k$；$V_k^\top x$ 只对应单个 **列向量**样本。若原始数据未中心化，还要先减均值，重建后再加回均值。

### 2.3 为什么 PCA 还能重建？

$$
\boxed{\hat X=ZV_k^\top=XV_kV_k^\top}
$$

尺寸：$(m\times k)(k\times n)=m\times n$。

**区别**：$Z$ 是真正降维后的矩阵（$m\times k$）；$\hat X$ 是回到原始形状的**近似重建**（$m\times n$），二者不能混淆。

## 3. SVD 的求和展开与 Eckart–Young（P19）

### 3.1 为什么 $X=\sum_i\sigma_i u_iv_i^\top$？[补充：你专门追问的推导]

对一般的 $m\times n$ 矩阵，先算：

$$
U\Sigma=
\underbrace{\begin{bmatrix}
\sigma_1u_1 & \sigma_2u_2 & \cdots & \sigma_qu_q
\end{bmatrix}}_{m\times q}.
$$

而

$$
V^\top=
\underbrace{\begin{bmatrix}
v_1^\top\\v_2^\top\\\vdots\\v_q^\top
\end{bmatrix}}_{q\times n}.
$$

两者相乘，就是**第一块的第 $i$ 列乘以第二块的第 $i$ 行，再全部相加**：

$$
\begin{aligned}
X&=(U\Sigma)V^\top\\
 &=\underbrace{\sigma_1u_1v_1^\top}_{m\times n}
  +\underbrace{\sigma_2u_2v_2^\top}_{m\times n}
  +\cdots+
  \underbrace{\sigma_qu_qv_q^\top}_{m\times n}\\
&=\boxed{\sum_{i=1}^{q}\sigma_i u_iv_i^\top}.
\end{aligned}
$$

维度检查：$u_i$ 是 $m\times1$，$v_i^\top$ 是 $1\times n$，所以每项都是 $m\times n$，而且单项的秩最多为 1（$\sigma_i>0$ 时恰为 1）。

**[补充：不是只对对角矩阵成立]** 例如：

$$
X=\begin{bmatrix}4&3\\0&5\end{bmatrix}
=\underbrace{\begin{bmatrix}2&4\\2&4\end{bmatrix}}_{\sigma_1u_1v_1^\top}
+\underbrace{\begin{bmatrix}2&-1\\-2&1\end{bmatrix}}_{\sigma_2u_2v_2^\top}.
$$

这就是之前算过的非对角例子：$\sigma_1=2\sqrt{10}$、$\sigma_2=\sqrt{10}$，$U,V$ 也都不是单位矩阵。

### 3.2 Rank-$k$ 与 $\arg\min$ 到底什么意思？

**Rank（秩）**：矩阵线性独立的行或列的最大数量。一个 $uv^\top$ 的外积矩阵秩至多为 1，所以保留 $k$ 项后的矩阵秩至多为 $k$。

**只保留前 $k$ 项**：

$$
\boxed{\hat X_k=\sum_{i=1}^{k}\sigma_i u_iv_i^\top}
$$

$\hat X_k$ 的形状**仍是 $m\times n$**；改变的是 Rank，不是行列数。

老师写的 Eckart–Young 定理：

$$
\boxed{\hat X_k\in
\arg\min_{\operatorname{rank}(Y)\le k}
\|X-Y\|_F^2}
$$

逐词翻译：在**所有与 $X$ 同尺寸且秩不超过 $k$ 的候选矩阵 $Y$**中，找一个使得 $\|X-Y\|_F^2$（每个位置的误差平方和）最小的矩阵。这里用 $\in\arg\min$，因为出现并列奇异值时最优解可能不唯一。

**为什么选最大的前 $k$ 个？** 定理保证截断 SVD 是最优低秩近似；平方误差恰等于**丢弃奇异值的平方和**：

$$
\boxed{\|X-\hat X_k\|_F^2=\sum_{i=k+1}^{q}\sigma_i^2.}
$$

沿用上例，如果 $k=1$，丢弃 $\sigma_2=\sqrt{10}$，则

$$
\hat X_1=\begin{bmatrix}2&4\\2&4\end{bmatrix},\qquad
\|X-\hat X_1\|_F^2=10.
$$

> **重点**：“误差等于 10”能直接计算；“没有其他 Rank-1 矩阵能让误差小于 10”才是 Eckart–Young 定理额外保证的结论。

对已中心化的 $X$，PCA 重建 $XV_kV_k^\top$ 与截断 SVD 的 $\hat X_k$ 相同。这就解释了为什么 PCA 选择前 $k$ 个右奇异向量。

![老师第 19 页：Eckart–Young → 深度学习中的编码与重建](images/lecture_p19_pca_to_ae.png){width=84%}

*图 2｜Lecture 13/14，第 19 页原图。*

## 4. Autoencoder（自编码器）：让神经网络学压缩与重建（P19–20）

老师的问题：**“But what if we wanted to do this DL style?”** —— 不用 SVD，能不能直接训练 Encoder + Decoder？

一个最简单的**线性** Autoencoder（先忽略 bias）：

$$
\underbrace{Z}_{m\times k}=\underbrace{X}_{m\times n}\underbrace{W_1}_{n\times k},
\qquad
\underbrace{\hat X}_{m\times n}=\underbrace{Z}_{m\times k}\underbrace{W_2}_{k\times n}.
$$

因此

$$
\boxed{\hat X=XW_1W_2},\qquad
\boxed{\mathcal L_{\rm recon}=\|X-\hat X\|_F^2}.
$$

- **Encoder**：$n\to k$，学习低维潜在表示（latent representation）$Z$。
- **Bottleneck**：中间维度 $k<n$，要求压缩信息。
- **Decoder**：$k\to n$，重建输入。
- **训练**：Forward → Reconstruction Loss → Backpropagation 更新 $W_1,W_2$。不要求 $W_2=W_1^\top$。
- 可以加多层网络和非线性激活；线性 AE 在适当条件下可达到与 PCA 相同的最佳重建子空间，但权重矩阵不必逐项等于主成分。

**为什么属于自监督？** 老师第 20 页原话：**“Labels are $x_i$ itself.”** 输入是 $x_i$，监督目标就是 $x_i$；并不需要人工写“这是狗”。


*图 3｜Lecture 13/14，第 20 页原图；课堂示例中 100×100 图片展平为 10,000 维，经 128 维瓶颈重建。*

## 5. 三种 Autoencoder 变体（P21–22）

### 5.1 Denoising Autoencoder：输入加噪，目标仍干净（P21）

$$
\tilde X=X+\epsilon,\qquad
Z=f_\theta(\tilde X),\quad
\hat X=g_\phi(Z),\quad
\boxed{\mathcal L=\|X-\hat X\|_F^2}.
$$

**区别：输入是 $\tilde X$，Target 是原始 $X$。** 复制噪声无法得到低 Loss，网络需要学习有利于去噪的结构。老师写有 *Use data augmentation*。

### 5.2 Masked Autoencoder：遮住一部分，让模型补全（P21）

设 $M\in\{0,1\}^{m\times n}$，1 表示保留，0 表示遮挡，$\odot$ 表示逐元素乘法：

$$
\tilde X=M\odot X,\qquad
\hat X=g_\phi(f_\theta(\tilde X)).
$$

- 简化的完整重建 Loss：$\|X-\hat X\|_F^2$。
- **[补充]** 一种常见 MAE 做法是**只对遮挡位置算误差**：

$$
\mathcal L_{\rm masked}
=\frac{\|(1-M)\odot(X-\hat X)\|_F^2}{\sum_{i,j}(1-M_{ij})},
$$

分母假设至少遮挡一个元素。实际图像 MAE 常按 patch 遮挡，具体实现可能不等于逐像素置零。




### 5.3 Sparse Autoencoder：候选特征很多，但单个输入只激活少量（P22）

**先分清两个词：**

- **Bottleneck**：限制中间层**总共多少个维度**，比如 $1000\to100\to1000$。
- **Sparse**：中间层可以很宽（例如 $1000\to2000\to1000$），但**一个样本只有少量维度明显激活**。它不要求一定是宽层。

例如某个输入的 latent vector：

$$
\text{普通 }z=[0.6,0.8,0.3,0.9,0.5],\qquad
\text{较稀疏 }z=[0,0.8,0,0,0.5].
$$

**[补充：如何训练]** 常见方式是在重建损失外加入 L1 激活惩罚：

$$
\boxed{\mathcal L
=\underbrace{\|X-\hat X\|_F^2}_{\text{必须重建准确}}
+\lambda\underbrace{\|Z\|_1}_{\text{鼓励激活变少}}},\qquad
\|Z\|_1=\sum_{i,j}|Z_{ij}|.
$$

同样的重建误差下，更少／更弱的激活会降低第二项；但**不能全部变为零**，否则不同输入无法被 Decoder 有效区分，重建损失通常会变大。$\lambda$ 控制两种目标的权衡。L1 鼓励稀疏，**不保证激活一定恰好为零**。

**[补充：为什么有意义？这也是之前追问的重点]**

可以把 2000 个神经元想象成一个包含 2000 种**候选特征**的词典。每张图片只挑几个相关特征进行组合，例如某些单元偏向响应边缘、纹理或形状。这样既拥有大的特征表达空间，又希望每个输入使用相对少的特征。潜在好处是**选择性、组合性、某些特征更便于解释**；但不是理论保证：**稀疏不等于每个神经元必然对应一个可读的概念**，稀疏过强也可能让重建变差。


## 6. 最终对照与易错点

| 方法 | 输入 | Target / 目标 | 核心 |
|:--|:--|:--|:--|
| PCA | 已中心化的 $X$ | 最优低秩重建 | SVD 取最大奇异值对应方向 |
| Vanilla AE | $X$ | $X$ | 学 Encoder / Decoder |
| Denoising AE | 加噪后的 $\tilde X$ | 干净的 $X$ | 通过去噪学习表示 |
| Masked AE | 被遮挡的输入 | $X$（常只算遮挡处） | 补全缺失信息 |
| Sparse AE | $X$ | $X$ + 稀疏约束 | 少量激活组合特征 |

**复习时最容易混淆的 6 点**：

1. **自监督不是没有 Loss**：而是 Target 来自数据本身。
2. **$X$ 和 $x$ 不能混用**：$X:m\times n$ → $Z=XV_k$；单列样本 $x:n\times1$ → $z=V_k^\top x$。
3. **右奇异向量 $v_i$ 是 $n$ 维的方向**；左奇异向量 $u_i$ 是 $m$ 维的方向；奇异值 $\sigma_i$ 是数。
4. **$\hat X_k$ 和 $Z$ 不一样**：$\hat X_k:m\times n$ 是低秩重建；$Z:m\times k$ 才是降维后的表示。
5. **Eckart–Young 的“最优”限定条件**：在秩不超过 $k$ 的所有同尺寸矩阵中，Frobenius 平方误差最小。
6. **Sparse 不等于 Bottleneck**：Sparse 限制单个输入实际激活的特征；Bottleneck 限制特征总维度。

**一页记忆公式**：

$$
\begin{gathered}
X=U\Sigma V^\top=\sum_{i=1}^{q}\sigma_i u_iv_i^\top,\qquad
Z=XV_k,\quad \hat X=ZV_k^\top,\\
\hat X_k=\sum_{i=1}^{k}\sigma_i u_iv_i^\top,\quad
\|X-\hat X_k\|_F^2=\sum_{i=k+1}^{q}\sigma_i^2,\\
\hat X=g_\phi(f_\theta(X)),\qquad
\mathcal L_{\rm AE}=\|X-\hat X\|_F^2.
\end{gathered}
$$

> **范围提醒**：P17–22 是这次 Self-Supervision 的全部主要课堂内容；SVD 计算步骤、矩阵维度细节、L1 Sparse Loss 和几种方法的解释性讨论，是为了补足学习过程中遇到的问题而加的复习材料。
