## 1. Depthwise Convolution（逐通道卷积）

### 1.1 普通卷积（Standard Convolution）

对于普通二维卷积：

- 输入通道数：$C_{\text{in}}$
- 输出通道数：$C_{\text{out}}$
- 卷积核大小：$k\times k$
    

每一个输出 Channel 都依赖全部输入 Channels。

权重数量：

$$  
N_{\text{weights}}=k^2C_{\text{in}}C_{\text{out}}  
$$

如果每个输出 Channel 还有一个 Bias：

$$  
N_{\text{parameters}}=(k^2C_{\text{in}}+1)C_{\text{out}}  
$$


### 1.2 Depthwise Convolution
**每个输入 Channel 分别使用自己的卷积核，不在这一操作中混合不同 Channel 的信息。**

假设：

- 输入 Channel = 3
- 每个 Channel 使用一个 $5\times5$ Kernel

那么：

$$  
N_{\text{weights}}=5^2\times3=75  
$$

若用普通卷积且 $c_{out} =3$ 的话需要 225 个权重，因此用 Depthwise Convolution 的话参数量明显减少。

---

## 2. Why Might Depth Hurt Optimization?

为什么网络越深，反而可能越难训练？

### 2.1 Degradation Problem（深度退化问题）
![](附件/Pasted%20image%2020261014003103.png)

更深的网络不一定拥有更低的训练 Loss。

这并不意味着更深的网络表达能力一定更弱，而是可能因为网络太深，优化变得更加困难。

### 2.2 Vanishing Gradients（梯度消失）

反向传播依赖 Chain Rule（链式法则）。

对于：

$$  
y=f_3(f_2(f_1(x)))  
$$

梯度：

$$
f_3'(f_2(f_1(x)))f_2'(f_1(x))f_1'(x)  
$$

如果每层的局部导数都是 0.5，那么经过 $L$ 层：

$$  
\frac{dy}{dx}=0.5^L  
$$

例如：

$$  
0.5^{10}\approx0.000977  
$$

$$  
0.5^{100}\approx7.89\times10^{-31}  
$$

当网络很深时，梯度可能迅速趋近于 0，导致前面层的参数难以有效更新。

实际网络中还可能出现梯度爆炸，因此并不是所有深层网络的梯度都必然消失。

### 2.3 Shattered Gradients（梯度碎片化）

![](附件/Pasted%20image%2020261014003324.png)

Shattered Gradients 描述的是：

**深层网络的梯度可能随着输入的微小变化而剧烈变化，邻近输入位置的梯度相关性迅速下降。**

上图中的三个函数图：

- Figure a：浅层网络，梯度变化相对平滑。
    
- Figure b：24 个隐藏层的深层网络，梯度变化十分杂乱。
    
- Figure c：使用 Autocorrelation 展示不同网络深度下的梯度相关性。
    

!!! explanation "**Autocorrelation（自相关）**"

	用于衡量同一个信号在相隔一定距离的两个位置上有多相似。
	
	对于梯度函数：
	
	$$  
	g(x)=\frac{df(x)}{dx}  
	$$
	
	如果 $g(x)$ 与 $g(x+\Delta x)$ 比较相似，那么对应的梯度相关性较高。
	
	深层网络中，即使 $\Delta x$ 很小，梯度之间的相关性也可能迅速下降。

因此：

- Vanishing Gradients：梯度数值可能趋近于 0。
    
- Shattered Gradients：梯度不一定小，但随输入变化可能变得高度不规则。
    

老师使用这些现象说明深层网络为什么难以优化。

---

## 3. ResNet（Residual Network，残差网络）
![](附件/Pasted%20image%2020261014003840.png)
### 3.1 Residual Block

ResNet 最核心的结构：

$$  
\boxed{y=x+F(x)}  
$$

其中：

- $x$：输入。
    
- $F(x)$：残差分支经过若干网络层得到的输出。
    
- $y$：残差分支与输入相加后的结果。
    

这种连接称为 Residual Connection 或 Identity Skip Connection。

### 3.2 为什么 ResNet 更容易训练？

**原因一：容易表达 Identity Mapping（恒等映射）。**

普通网络如果不需要改变输入，必须学到：

$$  
F(x)=x  
$$

而对于 ResNet，只需要让残差分支输出：

$$  
F(x)=0  
$$

就有：

$$  
y=x  
$$

因此残差分支只需要学习相对于输入的变化。

**原因二：改善梯度传播。**

对于：

$$  
y=x+F(x)  
$$

求导：

$$  
\frac{dy}{dx}=1+\frac{dF(x)}{dx}  
$$

多维情况下：

$$  
\frac{\partial y}{\partial x}=I+J_F(x)  
$$

其中：

- $I$：Identity Matrix（单位矩阵）。
    
- $J_F$：$F$ 的 Jacobian（雅可比矩阵）。
    

额外的 Identity Term 提供了直接的梯度传播路径，减轻必须经过所有非线性变换的困难。

不过，Residual Connection 并不能数学上保证梯度永远不会消失。

### 3.3 ResNet 的不同路径

![](附件/Pasted%20image%2020261014004024.png)

上图展示了 4 个 Residual Blocks。

对于每一个 Block，计算图中都存在：

1. Residual Branch
    
2. Skip Branch
    

从路径组合角度，可以展开成：

$$  
2^4=16  
$$

条不同的分支路径。

一般情况下，$n$ 个残差块有：

$$  
2^n  
$$

种分支路径组合。

这些路径长度不同，有些路径可以绕过较多网络层，因此提供较短的信息传播通道。

注意：这不代表实际独立训练了 $2^n$ 个神经网络。

### 3.4 Residual Block 的结构顺序

![](附件/Pasted%20image%2020261014004530.png)

示了不同的 Residual Block 结构，包括：

- Linear → ReLU → Addition
- Linear → ReLU → Linear → Addition
- Pre-activation 风格的结构
    

这里的 Linear 泛指可学习的线性变换，在 CNN 中可以是 Convolution。

不同结构中，Activation Function 和 Addition 的顺序会影响网络的传播性质。

### 3.5 Residual Addition 的尺度问题

对于：

$$  
x_{l+1}=x_l+F_l(x_l)  
$$

如果假设 $x_l$ 和 $F_l(x_l)$：

- 均值为 0。
- 方差相同。
- 互不相关。
    

那么：

$$
\operatorname{Var}(x_l)+\operatorname{Var}(F_l(x_l))  
$$

因此：

$$  
\operatorname{Var}(x_{l+1})=2\operatorname{Var}(x_l)  
$$

我们可以执行这种缩放：

$$  
x_{l+1}=\frac{x_l+F_l(x_l)}{\sqrt{2}}  
$$

在上述假设成立时，这样可以保持方差不变。这是一种示意性的方差控制策略，并不是所有 ResNet 的标准公式。

---
## 4. Normalization（归一化）

![](附件/Pasted%20image%2020261014010607.png)

Normalization：对神经网络中的激活值进行标准化，控制它们的数值尺度，改善训练稳定性。

不同 Normalization 方法之间最重要的区别是：

**究竟选择哪些数字放在一起计算 Mean（均值）和 Variance（方差）？**

### 4.1 Batch Normalization（批归一化）

#### 4.1.1 基本概念

Batch 指一次训练中输入的一批样本。

BatchNorm 的做法：对于同一个 Feature，将 Batch 中不同样本的对应激活值放在一起计算均值和方差。

假设某一层有 3 个 Features，当前 Batch 有 4 个 Samples：

$$  
X=  
\begin{bmatrix}  
1&2&3\\  
4&5&6\\  
7&8&9\\  
10&11&12  
\end{bmatrix}  
$$

其中：

- 每一行代表一个 Sample。
- 每一列代表一个 Feature。
    

BatchNorm 对每一列分别计算：

- Feature 1：$[1,4,7,10]$
- Feature 2：$[2,5,8,11]$
- Feature 3：$[3,6,9,12]$

因此，BatchNorm 是**同一特征跨不同样本进行归一化**，而不是每个样本独立归一化。

#### 4.1.2 数学公式

设 Batch 大小为 $m$，某个 Feature 的激活值为：

$$  
h_1,h_2,\ldots,h_m  
$$

第一步，计算 Mean：

$$  
\mu_B=\frac{1}{m}\sum_{i=1}^{m}h_i  
$$

第二步，计算 Variance：

$$  
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(h_i-\mu_B)^2  
$$

第三步，Normalization：

$$  
\boxed{  
\hat h_i=\frac{h_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}  
}  
$$

其中 $\epsilon$ 是一个很小的正数，用来保证数值稳定性。

#### 4.1.3 Learnable Scale and Shift

标准化之后，BatchNorm 还会进行：

$$  
\boxed{y_i=\gamma\hat h_i+\beta}  
$$

其中：

- $\gamma$：Scale（缩放参数）。
- $\beta$：Shift（平移参数）。
    

两者都是 Learnable Parameters。

为什么需要这两个参数？

因为我们不希望神经网络的特征永远受到固定的标准化尺度限制。通过 $\gamma$ 和 $\beta$，网络能够学习适合任务的输出尺度与偏移。

通常初始化：

$$  
\gamma=1,\qquad\beta=0  
$$

若一层有 $D$ 个 Features：

$$  
N_{\text{parameters}}=2D  
$$

若有 $K$ 层，每层有 $D$ 个 Features：

$$  
\boxed{N_{\text{parameters}}=2KD}  
$$

#### 4.1.4 CNN 中的 BatchNorm

对于 CNN，输入通常是四维张量：

$$  
X\in\mathbb R^{B\times C\times H\times W}  
$$

其中：

- $B$：Batch Size。
- $C$：Channels。
- $H$：Height。
- $W$：Width。

可以将四维张量理解成：**$B$ 张三维图片，每张图片包含 $C$ 个二维 Feature Maps。**

CNN 中的 BatchNorm 对每个 Channel 单独归一化。

对于固定的 Channel $c$，使用整个 Batch 中这个 Channel 的全部空间位置计算 Mean 和 Variance。

即统计维度：

$$  
\boxed{B,H,W}  
$$

例如：

$$  
X\in\mathbb R^{2\times3\times2\times2}  
$$

有两张图片，每张图片包含 3 个 Channel，每个 Channel 是 $2\times2$。

计算 Channel 1 的 Mean 时：

- 图片 1 的 Channel 1 提供 4 个数。
- 图片 2 的 Channel 1 提供 4 个数。
- 一共使用 8 个数计算均值和方差。
    

其他 Channel 分别计算。

对于每个 Channel，都有独立的 $\gamma$ 和 $\beta$。

所以：

$$  
\boxed{N_{\text{parameters}}=2C}  
$$

对于 $K$ 层、每层 $C$ 个 Channels 的网络，若每层都采用这种 BatchNorm：

$$  
\boxed{N_{\text{parameters}}=2KC}  
$$

#### 4.1.5 BatchNorm 的优点与局限

优点：

- Stabilize Forward Propagation：稳定前向传播。
- Higher Learning Rate：有助于使用较大的学习率。
- Regularization：Batch Statistics 的随机变化可能产生一定正则化效果。
- 减少 Loss 和 Gradient 的变化。

局限：

- 依赖 Batch Statistics。
- 不同样本可能相互影响归一化结果。
- Batch Size 较小时，统计量可能不稳定。
- 训练与测试阶段需要采用不同的统计方式。
    

### 4.2 Layer Normalization（层归一化）

#### 4.2.1 名称与含义

**LayerNorm 对每个 Sample 在当前层选定的特征集合独立进行 Normalization。**(这里说的层是整个神经网络中的各个层，而不是每一层中的那些channels)

与 BatchNorm 最大的区别是：

- BatchNorm：同一个 Feature，跨不同 Samples 统计。
- LayerNorm：同一个 Sample，跨当前层选定的 Features 统计。
    

例如：

$$  
X=  
\begin{bmatrix}  
1&2&3\\
4&5&6\\
7&8&9  
\end{bmatrix}  
$$

每一行是一个 Sample，每一列是一个 Feature。

LayerNorm 分别对：

$$  
[1,2,3]  
$$

$$  
[4,5,6]  
$$

$$  
[7,8,9]  
$$

计算均值和方差。

不同 Samples 互不影响。

#### 4.2.2 CNN 中的 LayerNorm

对于：

$$  
X\in\mathbb R^{B\times C\times H\times W}  
$$

按照本节课堂图示，LayerNorm 对每个 Sample 内的所有 Channels 和空间位置进行统计。

即：

$$  
\boxed{C,H,W}  
$$

例如：

$$  
X\in\mathbb R^{32\times64\times16\times16}  
$$

表示 Batch 内有 32 张图片。

每张图片有：

$$  
64\times16\times16=16384  
$$

个 Feature Values。

因此：

- 图片 1 使用自己的 16384 个值计算。
    
- 图片 2 使用自己的 16384 个值计算。
    
- 依此类推。
    

不同图片之间互不参与彼此的统计。

注意：这里描述的是讲义所展示的 CNN LayerNorm 统计方式；实际实现中的统计维度由具体归一化配置决定。

### 4.3 Other Normalization Methods（其他归一化方法）

#### 4.3.1 Ghost Batch Normalization

GhostNorm 仍然基于 BatchNorm，但只使用 Batch 的一部分样本计算统计量。

例如：

$$  
B=32  
$$

将一个 Batch 分成 4 个子 Batch，每组 8 张图片。

每组分别执行 BatchNorm。

因此：

**GhostNorm = 使用较小子 Batch 计算统计量的 BatchNorm。**

#### 4.3.2 Group Normalization

Group 指“组”。

GroupNorm 将 Channels 划分成若干组，对每个组分别归一化。

例如：

$$  
C=8,\qquad G=2  
$$

则每组有 4 个 Channels：

$$  
G_1={C_1,C_2,C_3,C_4}  
$$

$$  
G_2={C_5,C_6,C_7,C_8}  
$$

对于每个 Sample：

- Group 1 的所有 Channel 和空间位置一起统计。
    
- Group 2 独立统计。
    
- 不同 Samples 之间不混合。
    

统计范围为：

$$  
\boxed{\text{同一样本中的组内 Channels、Height、Width}}  
$$

#### 4.3.3 Instance Normalization

Instance 指单个数据实例。

在 CNN 中，InstanceNorm 对每个 Sample 的每个 Channel 分别进行 Normalization。

例如某张图片有 3 个 Channels：

- Channel 1 独立计算 Mean 和 Variance。
    
- Channel 2 独立计算。
    
- Channel 3 独立计算。
    

不同 Samples 不混合，不同 Channels 也不混合。

归一化范围：

$$  
\boxed{H,W}  
$$

#### 4.3.4 各种 Normalization 对比

假设 CNN 输入为：

$$  
X\in\mathbb R^{B\times C\times H\times W}  
$$

|Method|参与同一次统计的维度|是否跨 Samples|
|---|---|---|
|BatchNorm|$B,H,W$（固定 Channel）|是|
|GhostNorm|子 Batch 内的 $B,H,W$|是|
|LayerNorm|$C,H,W$（本课 CNN 示例）|否|
|GroupNorm|同一组内的 $C,H,W$|否|
|InstanceNorm|$H,W$（固定 Sample 和 Channel）|否|

GroupNorm 有一个重要关系：

当：

$$  
G=1  
$$

所有 Channels 属于同一组，其统计范围与本课示例中的 LayerNorm 相同。

当：

$$  
G=C  
$$

每个 Channel 单独成组，其统计范围与 InstanceNorm 相同。

### 4.4 Where to Put Normalization?（归一化放在哪里？）

老师第 18 页提出：

> Where do you put norm block?

即：Normalization 应该放在网络结构的什么位置？

一种常见方式是 **Pre-Norm**，即在一个网络 Block 的开始处进行 Normalization。

例如 Residual Block 可以采用：

$$  
y=x+F(\operatorname{Norm}(x))  
$$

这种结构使输入先经过 Normalization，再进入残差分支中的后续网络层。

实际结构可能包含多个 Linear / Convolution 和 Activation。

### 4.5 Normalization at Test Time（测试阶段）

#### 4.5.1 BatchNorm

训练阶段，BatchNorm 利用当前 Mini-batch 的统计量进行归一化，并通常维护：

- Running Mean
    
- Running Variance
    

测试阶段通常使用训练过程中积累的 Running Statistics，而不再使用当前测试 Batch 的统计量。

原因是测试时 Batch Size 可能发生变化，甚至只有一个 Sample。

#### 4.5.2 LayerNorm

LayerNorm 根据当前 Sample 自身的特征计算统计量，因此一般不需要像 BatchNorm 一样切换为预先保存的 Batch Statistics。

**核心区别：BatchNorm 通常依赖训练时积累的统计量进行推理，而 LayerNorm 可以直接基于当前样本计算。**

---

## 5. Dimension Mismatch in ResNet（残差连接的维度不匹配）

ResNet 的基本公式是：

$$  
y=x+F(x)  
$$

但进行 Addition 时，两条路径的 Tensor Shape 必须兼容。

假设输入：

$$  
x\in\mathbb R^{100\times100\times32}  
$$

而 Residual Branch 输出：

$$  
F(x)\in\mathbb R^{50\times50\times64}  
$$

显然无法直接相加。

一种常见的解决方法是 **Projection Shortcut（投影捷径连接）**。

通过 $1\times1$ Convolution 将 Skip Branch 的维度转换为目标维度。

例如使用：

$$  
\operatorname{Conv}_{1\times1,\ s=2}  
$$

将：

$$  
100\times100\times32  
$$

转换为：

$$  
50\times50\times64  
$$

令：

$$  
P(x)=\operatorname{Conv}_{1\times1,\ s=2}(x)  
$$

最终：

$$  
\boxed{y=P(x)+F(x)}  
$$

这样两条路径的维度一致，就可以执行 Addition。

---

## 6. U-Net

### 6.1 为什么需要 U-Net？

传统 CNN 常用于 Image Classification（图像分类）。

输入一张图片，输出一个类别。

但 Semantic Segmentation（语义分割）需要对每个像素进行分类。

例如输入：

$$  
256\times256\times3  
$$

对于二分类分割任务，可以输出：

$$  
256\times256\times1  
$$

其中每个位置代表对应像素属于目标类别的预测概率。

### 6.2 Downsampling 的问题

CNN 通过 Stride 或 Pooling 不断缩小 Feature Maps：

$$  
256\times256  
\rightarrow128\times128  
\rightarrow64\times64  
\rightarrow32\times32  
$$

这样可以扩大感受野，让网络提取更高层的特征。

但同时也可能丢失精确的空间细节，例如物体边界信息。

课堂笔记：

> Fundamental tension between getting big picture things and getting details.

即：

**获取整体语义信息与保留局部空间细节之间存在权衡。**

### 6.3 Encoder（编码器）

Encoder 负责特征提取和 Downsampling。

例如：

$$  
256\times256\times64  
$$

经过逐层 Downsampling：

$$  
128\times128\times128  
$$

$$  
64\times64\times256  
$$

$$  
32\times32\times512  
$$

通常：

- Height、Width 减小。
    
- Channels 增加。
    
- 感受野逐渐扩大。
    
- 特征逐渐包含更多高级语义信息。
    

### 6.4 Decoder（解码器）

Decoder 负责 Upsampling，将低分辨率 Feature Maps 逐步恢复成高分辨率表示。

例如：

$$  
32\times32\times512  
\rightarrow64\times64\times256  
\rightarrow128\times128\times128  
\rightarrow256\times256\times64  
$$

最后可以通过卷积产生像素级预测。

但仅靠 Upsampling 无法保证恢复 Downsampling 丢失的全部细节。

这就是 U-Net 需要 Skip Connections 的原因。

### 6.5 Skip Connections & Concatenation

U-Net 会保存 Encoder 中较高分辨率的 Feature Maps。

Decoder 在 Upsampling 后，将自己的 Feature Map 与 Encoder 中对应空间分辨率的 Feature Map 合并。

经典 U-Net 使用的是：

**Concatenation（拼接）**

而不是 Addition（相加）。

例如：

Encoder 保存：

$$  
X_{\text{skip}}\in\mathbb R^{64\times64\times256}  
$$

Decoder Upsampling 后：

$$  
X_{\text{decoder}}\in\mathbb R^{64\times64\times128}  
$$

沿 Channel 维度 Concatenate：

$$  
X_{\text{concat}}\in\mathbb R^{64\times64\times384}  
$$

因为：

$$  
256+128=384  
$$

拼接之后，通常继续经过 Convolution 融合这些特征。

因此，Decoder 的典型流程是：

$$  
\boxed{  
\text{Upsampling}  
\rightarrow  
\text{Concatenation}  
\rightarrow  
\text{Convolution}  
}  
$$


### 6.6 U-Net 与 ResNet 的 Skip Connections

| 比较项目 | ResNet | U-Net |
| --- | --- | --- |
| 主要目的 | 改善深层网络优化 | 融合高层语义与局部细节 |
| 合并方式 | Addition（相加） | Concatenation（拼接） |
| 典型位置 | Residual Block 内 | Encoder 和 Decoder 之间 |

老师特别指出：

> Does not replace block-by-block skip connections.

意思是 U-Net 的跨层连接并不取代 ResNet 的 Residual Connections，两者可以同时存在。

---

## 7. Upsampling（上采样）

### 7.1 Downsampling 回顾

- Stride / Subsampling：按照一定步长采样。
- Max Pooling：局部取最大值。
- Mean Pooling：局部取平均值。
    

### 7.2 Max Unpooling

假设输入：

$$  
X=  
\begin{bmatrix}  
1&3\\  
4&2  
\end{bmatrix}  
$$

Max Pooling：

$$  
y=4  
$$

如果保留了最大值原来所在的位置，Max Unpooling 可以将它放回对应位置：

$$  
\begin{bmatrix}  
0&0\\  
4&0  
\end{bmatrix}  
$$

但是它不能恢复原来的其他数值。

因此 Max Unpooling 不是 Max Pooling 的真正逆运算。

### 7.3 Repetition / Nearest-neighbor Upsampling

一种简单的 Upsampling 方法是重复原来的数值。

例如：

$$  
\begin{bmatrix}  
2&4\\  
4&8  
\end{bmatrix}  
$$

放大后：

$$  
\begin{bmatrix}  
2&2&4&4\\  
2&2&4&4\\  
4&4&8&8\\  
4&4&8&8  
\end{bmatrix}  
$$

特点：

- 增加空间尺寸。
    
- 不需要可学习参数。
    
- 通过复制现有数值扩大 Feature Map。
    
- 不能保证恢复丢失的原始细节。
    

### 7.4 Transposed Convolution（转置卷积）

#### 7.4.1 基本思想

普通卷积可以将多个局部输入通过加权求和合成为一个输出。

Transposed Convolution 则可以让一个输入值通过多个 Kernel 权重，对多个输出位置产生贡献。

如果不同输入的贡献落在同一个输出位置，就将它们相加。

Transposed Convolution 是一种带有可学习权重的操作，可用于 Upsampling。

#### 7.4.2 为什么叫 Transposed？

普通卷积可以写成矩阵乘法：

$$  
h=Wx  
$$

这里的 $W$ 是由卷积核及其连接结构构造的大矩阵。

Transposed Convolution 对应：

$$  
z=W^\top h  
$$

其中 $W^\top$ 是矩阵 $W$ 的转置。

注意：

$$  
W^\top\neq W^{-1}  
$$

所以 Transposed Convolution 并不是普通卷积的逆运算，也不能保证恢复原始输入。

#### 7.4.3 Lecture 11 的四张图

![](附件/Pasted%20image%2020261014011751.png)
老师使用 a、b、c、d 四张图展示两种操作。

- **a 图**：普通卷积的节点连接。
- **b 图**：普通卷积对应的稀疏权重矩阵。
- **c 图**：Transposed Convolution 的节点连接。
- **d 图**：Transposed Convolution 对应的转置矩阵。
    

在 c 图中：

- 边界输入可能只连接部分输出。
    
- 中间输入可能连接更多输出。
    
- 不同输入可能对同一个输出产生贡献。
    

对于一个具有三个共享权重 $a,b,c$ 的示意性连接结构，可以写成：

$$  
h_1=ax_1  
$$

$$  
h_2=bx_1+cx_2  
$$

$$  
h_3=ax_2  
$$

$$  
h_4=bx_2+cx_3  
$$

$$  
h_5=ax_3  
$$

$$  
h_6=bx_3+cx_4  
$$

$$  
h_7=ax_4  
$$

$$  
h_8=bx_4  
$$

对应矩阵：

$$
\begin{bmatrix}  
a&0&0&0\\  
b&c&0&0\\  
0&a&0&0\\  
0&b&c&0\\  
0&0&a&0\\  
0&0&b&c\\  
0&0&0&a\\  
0&0&0&b\\  
\end{bmatrix}  
\begin{bmatrix}  
x_1\  
x_2\  
x_3\  
x_4  
\end{bmatrix}  
$$

因此：

$$  
\mathbb R^4\rightarrow\mathbb R^8  
$$

矩阵中：

- 非零元素表示存在连接及其对应权重。
    
- 零元素表示对应输入输出之间没有直接连接。
    

注意：这里的矩阵是帮助理解图中连接与转置关系的简化示意，权重字母不是老师原图明确给出的具体数值。


#### 7.4.4 Transposed Convolution 与 Nearest-neighbor Upsampling

| 比较项目 | Nearest-neighbor Upsampling | Transposed Convolution |
| --- | --- | --- |
| 是否可以放大尺寸 | 是 | 是 |
| 是否具有可学习参数 | 否 | 是 |
| 计算方式 | 复制输入值 | 通过卷积权重形成输出贡献 |
| 能否保证恢复原始细节 | 否 | 否 |

在 U-Net 中，Decoder 可以使用 Transposed Convolution 进行 Upsampling，再与 Encoder 保存的 Feature Maps 拼接。

