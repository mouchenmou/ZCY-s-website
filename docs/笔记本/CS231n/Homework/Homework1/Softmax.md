## Softmax Loss 与梯度推导

对于第 $i$ 个训练样本，输入为 $X_i$，权重矩阵为 $W$。

首先计算各个类别的分数：

$$
s_i = X_iW
$$

其中，第 $j$ 个类别的分数为：

$$
s_{ij}=X_iW_j
$$

### 1. Softmax 概率

Softmax 将各类别的分数转换为概率：

$$
p_{ij}
=
\frac{e^{s_{ij}}}
{\sum_{k=1}^{C}e^{s_{ik}}}
$$

其中，$C$ 表示类别总数。

假设第 $i$ 个样本的正确类别为 $y_i$，则正确类别的预测概率为：

$$
p_{i,y_i}
=
\frac{e^{s_{i,y_i}}}
{\sum_{k=1}^{C}e^{s_{ik}}}
$$

### 2. 单个样本的交叉熵损失


$$
L_i
=
-\log e^{s_{i,y_i}}
+
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
$$

也就是：

$$
L_i
=
-s_{i,y_i}
+
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
$$

### 3. Loss 对 score 的梯度

现在求第 $i$ 个样本的损失 $L_i$ 对第 $j$ 个类别分数 $s_{ij}$ 的导数：

$$
\frac{\partial L_i}{\partial s_{ij}}
$$

损失由两部分组成：

$$
L_i
=
-s_{i,y_i}
+
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
$$

首先看第一部分：

$$
\frac{\partial(-s_{i,y_i})}{\partial s_{ij}}
=
\begin{cases}
-1, & j=y_i \\
0, & j\neq y_i
\end{cases}
$$

再看第二部分：

$$
\frac{\partial}{\partial s_{ij}}
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
$$

根据链式法则：

$$
\frac{\partial}{\partial s_{ij}}
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
=
\frac{1}
{\sum_{k=1}^{C}e^{s_{ik}}}
\cdot e^{s_{ij}}
$$

因此：

$$
\frac{\partial}{\partial s_{ij}}
\log
\left(
\sum_{k=1}^{C}e^{s_{ik}}
\right)
=
\frac{e^{s_{ij}}}
{\sum_{k=1}^{C}e^{s_{ik}}}
=
p_{ij}
$$

把两部分相加，得到：

$$
\frac{\partial L_i}{\partial s_{ij}}
=
\begin{cases}
p_{ij}-1, & j=y_i \\
p_{ij}, & j\neq y_i
\end{cases}
$$

也可以统一写成：

$$
\frac{\partial L_i}{\partial s_{ij}}
=
p_{ij}-\mathbf{1}(j=y_i)
$$

其中，$\mathbf{1}(j=y_i)$ 是指示函数：

$$
\mathbf{1}(j=y_i)
=
\begin{cases}
1, & j=y_i \\
0, & j\neq y_i
\end{cases}
$$
​​
### 4. Loss 对权重矩阵 $W$ 的梯度

由于：

$$S=XW$$

矩阵乘法的反向传播公式为：

$$dW=X^TdS$$

这里的 `dW` 和 `dS` 是代码中的梯度变量名，分别表示：

$$dW\equiv\frac{\partial L}{\partial W},\ dS≡\frac{∂L}{∂s}$$

因此：

$$\frac{\partial L}{\partial W} = X^T\frac{\partial L}{\partial S}​$$



### 5. 对所有训练样本取平均

假设训练样本数量为 $N$，所有样本的平均损失为：

$$L = \frac{1}{N} \sum_{i=1}^{N}L_i$$​

所以梯度也需要除以样本数量：

$$dW = \frac{1}{N}X^TdS$$

代码中对应：

```
dW = np.dot(X.T, dS) / num_train
```

### 6. 加入正则化

加入 L2 正则化后，总损失为：

$$L = \frac{1}{N} \sum_{i=1}^{N} -\log p_{i,y_i} + \lambda\sum_{a,b}W_{ab}^{2}$$​
总梯度为：

$$\frac{\partial L}{\partial W}=\frac{1}{N}X^T\frac{\partial L}{\partial S}+2\lambda W$$

```python 
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dS = np.zeros_like(np.dot(X, W))
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

  
        loss -= logp[y[i]]  # negative log probability is the loss
        p[y[i]]-=1
        dS[i]=p

  
    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW=np.dot(X.T, dS)/num_train + 2*reg*W
```
