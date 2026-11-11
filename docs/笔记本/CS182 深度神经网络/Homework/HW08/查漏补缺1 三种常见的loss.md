
# 三种常见 Loss Function

## 1. L1 Loss（平均绝对误差）

### 1.1 名称来源

**L1 Loss** 的名字来自数学中的 **L1 Norm（L1 范数）**。

对于向量：

$$
v=(v_1,v_2,\ldots,v_n)
$$

其 L1 Norm 定义为：

$$
\boxed{\|v\|_1=\sum_{i=1}^{n}|v_i|}
$$

即把向量中所有元素取绝对值，然后求和。

例如：

$$
v=(2,-3,4)
$$

那么：

$$
\|v\|_1=|2|+|-3|+|4|=9
$$

### 1.2 Loss 公式

假设：

- $y_i$：真实值（Ground Truth）
- $\hat y_i$：模型预测值（Prediction）
- $n$：参与计算的元素数量

PyTorch 中默认的 `nn.L1Loss` 计算：

$$
\boxed{
L_{\mathrm{L1}}=\frac1n\sum_{i=1}^{n}|y_i-\hat y_i|
}
$$

这个公式也叫：

**MAE（Mean Absolute Error，平均绝对误差）**

注意：

- L1 Norm 本身是绝对值之和。
- PyTorch 默认的 L1Loss 是绝对误差的平均值。

### 1.3 计算实例

真实值：

$$
y=(1,2,3)
$$

预测值：

$$
\hat y=(2,4,6)
$$

因此：

$$
L_{\mathrm{L1}}
=\frac{|1-2|+|2-4|+|3-6|}{3}
$$

$$
=\frac{1+2+3}{3}=\boxed{2}
$$

### 1.4 特点

- 误差大小与惩罚成线性关系。
- 相比 MSE，对特别大的误差不那么敏感。
- 经常用于回归任务（Regression）。

PyTorch：

```python
import torch.nn as nn

loss_fn = nn.L1Loss()
loss = loss_fn(y_pred, y_true)
```

---

## 2. MSE Loss（均方误差）

### 2.1 名称来源

**MSE = Mean Squared Error**

三个单词分别表示：

- Mean：平均值
- Squared：平方的
- Error：误差

因此 MSE 就是：

**把每个预测误差平方，再求平均。**

### 2.2 Loss 公式

$$
\boxed{
L_{\mathrm{MSE}}
=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
$$

其中：

- $y_i$：真实值
- $\hat y_i$：预测值
- $n$：元素数量

### 2.3 与 L2 Norm 的关系

对于误差向量：

$$
e=y-\hat y
$$

L2 Norm（L2 范数）定义为：

$$
\|e\|_2=\sqrt{\sum_i e_i^2}
$$

因此：

$$
\boxed{
\|e\|_2^2=\sum_i e_i^2
}
$$

这就是 **Squared L2 Norm（L2 范数的平方）**。

MSE 则是在平方误差之和的基础上取平均：

$$
\boxed{
L_{\mathrm{MSE}}=\frac1n\|y-\hat y\|_2^2
}
$$

注意区分：

- L2 Norm：先平方求和，再开根号。
- Squared L2 Norm：平方求和，不开根号。
- MSE：平方求和，再取平均。

### 2.4 计算实例

继续使用：

$$
y=(1,2,3),\qquad \hat y=(2,4,6)
$$

则：

$$
L_{\mathrm{MSE}}
=\frac{(1-2)^2+(2-4)^2+(3-6)^2}{3}
$$

$$
=\frac{1+4+9}{3}
=\boxed{\frac{14}{3}\approx4.67}
$$

### 2.5 特点

MSE 对较大的误差惩罚更强，因为误差会被平方。

例如：

| 绝对误差 | L1 惩罚 | 平方误差 |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 5 | 5 | 25 |
| 10 | 10 | 100 |

因此：

- MSE 对较大的预测误差更加敏感。
- 常用于 Regression、Autoencoder Reconstruction 等任务。
- 对异常值（Outliers）通常比 L1 更敏感。

PyTorch：

```python
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)
```

### 2.6 CS182 HW08 Q3 中的 MSE

题目中的 Reconstruction Loss 是：

$$
L=\frac1n\|X-W_2W_1X\|_F^2
$$

这里：

$$
\hat X=W_2W_1X
$$

Frobenius Norm（Frobenius 范数）是矩阵中所有元素平方和的平方根。

因此：

$$
\|X-\hat X\|_F^2
=\sum_{i=1}^{n}\sum_{j=1}^{m}(x_{j,i}-\hat x_{j,i})^2
$$

其中：

- $n$：样本数量
- $m$：每个样本的特征维度

注意：PyTorch 默认 `nn.MSELoss()` 对全部 $nm$ 个元素取平均，而题目的公式只除以 $n$。

所以若要严格匹配题目的 Reconstruction Loss，需要将默认 MSE 结果乘以 $m$，或者先对每个样本的特征误差求和，再对样本取平均。

---

## 3. Cross-Entropy Loss（交叉熵损失）

### 3.1 名称来源

**Cross-Entropy = 交叉熵**

这个概念来自 Information Theory（信息论）。

先理解两个术语：

**Entropy（熵）**

用于衡量一个概率分布的不确定性。

对于概率分布 $p$：

$$
H(p)=-\sum_{i=1}^{C}p_i\log p_i
$$

**Cross-Entropy（交叉熵）**

使用两个概率分布计算：

$$
\boxed{
H(p,q)=-\sum_{i=1}^{C}p_i\log q_i
}
$$

其中：

- $p_i$：真实概率分布
- $q_i$：模型预测概率分布
- $C$：类别数量
- $\log$：自然对数

因此，Cross-Entropy Loss 常用于衡量模型的预测概率分布与真实标签是否匹配。

### 3.2 分类任务中的 Cross-Entropy

假设我们要识别一张图片是猫还是狗。

真实类别是猫，所以真实标签用 One-Hot Encoding（独热编码）表示：

$$
p=(1,0)
$$

模型预测概率：

$$
q=(0.8,0.2)
$$

分别代表：

- 猫：80%
- 狗：20%

代入 Cross-Entropy：

$$
L=-[1\log(0.8)+0\log(0.2)]
$$

得到：

$$
\boxed{L=-\log(0.8)\approx0.223}
$$

如果模型只给正确类别 10% 的概率：

$$
q=(0.1,0.9)
$$

则：

$$
L=-\log(0.1)\approx2.303
$$

可以发现：

**模型给真实类别分配的概率越小，Cross-Entropy Loss 越大。**

### 3.3 为什么 One-Hot 标签可以简化公式？

假设总共有 $C$ 个类别，真实类别的编号是 $c$。

One-Hot 标签只有：

$$
p_c=1
$$

其余类别：

$$
p_i=0\quad(i\ne c)
$$

因此：

$$
L=-\sum_{i=1}^{C}p_i\log q_i
$$

可以直接简化成：

$$
\boxed{L=-\log q_c}
$$

也就是：

**只需要取模型给正确类别分配的概率，然后取负对数。**

### 3.4 PyTorch 实现

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

这里：

- `logits`：模型输出的原始分类分数。
- `labels`：真实类别的整数编号，例如 0、1、2。

注意：`nn.CrossEntropyLoss()` 通常直接接收 Logits，不需要提前手动计算 Softmax。

PyTorch 内部会通过 LogSoftmax 和 NLLLoss 等价地完成计算。

### 3.5 特点

- 主要用于 Classification（分类）任务。
- 根据预测的类别概率计算 Loss。
- 对模型给真实类别分配很低概率的情况施加较大惩罚。
- 常见于图像分类、语言模型等任务。

---

## 4. 三种 Loss 的对比

| Loss | 英文名称 | 计算方式 | 常见用途 |
|---|---|---|---|
| L1 | L1 / Mean Absolute Error | 误差绝对值的平均 | Regression |
| MSE | Mean Squared Error | 误差平方的平均 | Regression、Reconstruction |
| Cross-Entropy | Cross-Entropy | 真实分布与预测分布的交叉熵 | Classification |

### 三个核心公式

**L1 Loss**

$$
\boxed{L_{\mathrm{L1}}=\frac1n\sum_i|y_i-\hat y_i|}
$$

**MSE Loss**

$$
\boxed{L_{\mathrm{MSE}}=\frac1n\sum_i(y_i-\hat y_i)^2}
$$

**Cross-Entropy Loss**

$$
\boxed{L_{\mathrm{CE}}=-\sum_i p_i\log q_i}
$$

对于 One-Hot 分类标签：

$$
L_{\mathrm{CE}}=-\log q_{\mathrm{true}}
$$

### 如何根据题目选择 Loss？

1. 如果公式包含绝对误差 $|y-\hat y|$，通常对应 L1 Loss。
2. 如果公式包含平方误差 $(y-\hat y)^2$，通常对应 MSE Loss。
3. 如果涉及类别概率、Softmax、$-\log p$，通常对应 Cross-Entropy Loss。

### 重要补充：Loss 和 Regularization

在 CS182 HW08 Q3 中，完整目标函数是：

$$
L_{\mathrm{total}}=
L_{\mathrm{reconstruction}}
+
\lambda\|W_1\|_F^2
+
\lambda\|W_2\|_F^2
$$

其中：

- Reconstruction Loss：使用平方误差。
- Regularization Loss：对权重大小施加惩罚。
- $\lambda$：控制正则化强度。

**MSELoss 本身不包含 Weight Decay。**

Weight Decay 是额外的正则化机制，需要通过优化器设置或显式加入 Loss 来实现。
