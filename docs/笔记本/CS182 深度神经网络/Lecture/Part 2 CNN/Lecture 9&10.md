## 1. Why CNN?

Convolutional Neural Network 主要利用了图像数据的三个结构性特点。
### 1.1 Locality（局部性）

图像中相邻像素通常具有较强的关联性。

MLP（Multi-Layer Perceptron）采用全连接结构，每个输出神经元都可以连接所有输入，因此无法直接利用图像的局部结构。

CNN 使用小尺寸卷积核（Small Filters / Kernels），只连接输入中的局部区域。它能够：

- 减少参数数量。
- 利用图像的空间结构。
- 逐层提取局部特征。

### 1.2 Symmetry, Invariance and Equivariance

1. **Invariance（不变性）**
	
	输入图像发生某种变换，但网络输出保持不变。
	
	例如：猫在图片中向右平移，分类结果仍然是猫。
	
	$$  
	f(T(x))=f(x)  
	$$
	
	其中 $T$ 表示对输入进行变换。

2. **Equivariance（等变性）**

	输入发生变换，输出也发生对应的变换。
	
	$$  
	f(T(x))=T(f(x))  
	$$
	
	例如：输入图片向右平移，卷积得到的 Feature Map 也相应向右平移。
	
	标准卷积具有平移等变性（忽略边界效应等因素），但并不自动保证分类结果具有完全的平移不变性。
	
	实现或增强这些性质的常见方法：
	
	1. **Weight Sharing（权重共享）**：不同空间位置使用同一组卷积核权重。
	2. **Data Augmentation（数据增强）**：通过平移、旋转、裁剪等方法生成更多训练样本。
	    

### 1.3 Hierarchical Structure（层级结构）

复杂物体由简单的局部特征组成。

CNN 通过堆叠多个卷积层，逐渐提取更复杂的特征。

例如：

- 浅层：边缘、颜色、纹理。
- 中间层：角点、局部形状、物体组成部分。
- 深层：更加复杂的物体特征。
    

因此CNN还有以下特点：

- **Depth（深度）**：堆叠多个网络层。
- **Multiresolution（多分辨率）**：在不同空间尺度上处理信息。

---

## 2. Convolution（卷积运算）

### 2.1 1D Convolution（一维卷积）

连续时间卷积：

$$  
y(t)=\int_{-\infty}^{\infty}x(\tau)h(t-\tau)d\tau  
$$

离散时间卷积：

$$  
y[t]=\sum_{\tau=-\infty}^{\infty}x[\tau]h[t-\tau]  
$$

对于有限长度的卷积核，只需要对有限范围求和。

**注意：CNN 中通常使用 Cross-correlation（互相关）运算。**与数学意义上的严格卷积不同，CNN 的计算通常不需要 flip (翻转) Kernel。即：

> CNNs: No flip.

例如一个长度为 3 的 Kernel：

$$  
w=[w_1,w_2,w_3]  
$$

对应计算：

$$  
h_i=a\left(\beta+\sum_{j=1}^{3}w_jx_{i+j-2}\right)  
$$

其中：

- $w_j$：卷积核权重。
- $\beta$：Bias（偏置）。
- $a(\cdot)$：Activation Function（激活函数）。

### 2.2 具体计算例子

假设：

$$  
x=[1,2,3,4,5]  
$$

卷积核：

$$  
w=[2,1,-1]  
$$

Stride = 1，Padding = 0，Bias = 0。

第一个输出：

$$  
y_1=2(1)+1(2)-1(3)=1  
$$

第二个输出：

$$  
y_2=2(2)+1(3)-1(4)=3  
$$

第三个输出：

$$  
y_3=2(3)+1(4)-1(5)=5  
$$

因此：

$$  
y=[1,3,5]  
$$

注意：

**三个位置使用的是同一组权重 $[2,1,-1]$，这就是 Weight Sharing。**

虽然卷积核被使用了三次，但并没有产生三套独立参数。

### 2.3 2D Convolution（二维卷积）

二维卷积将一维滑动扩展到两个空间方向：

- Horizontal（水平）
    
- Vertical（垂直）
    

假设输入图像为 $X$，卷积核为 $K$。

对于单通道图像，一个输出位置对应的 Pre-activation 为：

$$  
z_{i,j}=\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}K_{u,v}X_{i+u,j+v}+b  
$$

这里采用无 Padding、Stride = 1、Dilation = 1 的索引方式。

重要术语：

- **Patch**：输入图像中的局部区域。
- **Pointwise Multiplication**：对应元素逐个相乘，并非矩阵乘法。
- **Pre-activation**：应用激活函数之前的值。
- **Activation**：经过激活函数之后的输出。
- **Feature Map**：卷积计算得到的特征图。

例如：

$$  
z=Wx+b  
$$

$$  
h=\operatorname{ReLU}(z)  
$$

其中 $z$ 是 Pre-activation，$h$ 是 Activation。

---

## 3. Receptive Field（感受野）

Receptive Field 指某一层的一个输出位置能够受到输入中多大区域的影响。

### 3.1 单层卷积

一个 Kernel Size 为 3 的一维卷积，其单个输出位置依赖连续 3 个输入位置。

因此：

$$  
R=3  
$$

对于二维 $3\times3$ 卷积，其感受野为 $3\times3$。

### 3.2 多层卷积

假设连续堆叠两层 Kernel Size 为 3 的卷积，Stride = 1，Dilation = 1。

第一层输出：

$$  
h_i=f(x_i,x_{i+1},x_{i+2})  
$$

第二层输出：

$$  
z_i=g(h_i,h_{i+1},h_{i+2})  
$$

由于三个 $h$ 又分别依赖相邻的输入位置，因此 $z_i$ 最终受到 5 个原始输入位置的影响。

所以两层的感受野是 5，而不是 3。

对于连续 $L$ 层、相同 Kernel Size $k$、Stride = 1、Dilation = 1 的卷积：

$$  
\boxed{R=1+L(k-1)}  
$$

例如 5 层 $3\times3$ 卷积：

$$  
R=1+5(3-1)=11  
$$

最终理论感受野为 $11\times11$。

**核心结论：增加网络深度可以扩大感受野，使深层特征能够利用更广范围的信息。**

### 3.3 Channels as Feature Detectors

> Channel as feature detector at the scale of its receptive field.

---

## 4. Multi-channel Convolution（多通道卷积）

### 4.1 RGB Input

一张 RGB 图像有三个 Channel：

$$  
X\in\mathbb R^{H\times W\times3}  
$$

分别对应 Red、Green、Blue。

例如：

$$  
X\in\mathbb R^{224\times224\times3}  
$$

对于一个 $3\times3$ 的卷积核，如果输入有 3 个 Channel，那么生成单个输出 Channel 所需要的 Kernel 权重形状为：

$$  
3\times3\times3  
$$

即 27 个 Weight。

### 4.2 单个输出 Channel 的计算

对于具有 $C_{\text{in}}$ 个输入 Channel 的图像：

$$
\sum_{c=1}^{C_{\text{in}}}  
\sum_{u=0}^{k-1}  
\sum_{v=0}^{k-1}  
K_{u,v,c}X_{i+u,j+v,c}+b  
$$

即：

1. 对每个输入 Channel 分别进行局部加权求和。
2. 将所有 Channel 的结果相加。
3. 加上一个 Bias。

这样得到一个二维的输出 Feature Map。

### 4.3 多个输出 Channel

如果希望生成多个输出 Channel，需要使用多组不同的 Kernel。

例如：

$$  
X\in\mathbb R^{32\times32\times3}  
$$

采用：

- Kernel Size = 3
- Stride = 1
- Padding = 0
- Output Channels = 10
    

则输出尺寸为：

$$  
30\times30\times10  
$$

因为 10 组不同的 Kernel 分别产生 10 张二维 Feature Map。

### 4.4 参数量计算

对于普通二维卷积：

- 输入 Channel 数量：$C_{\text{in}}$
- 输出 Channel 数量：$C_{\text{out}}$
- Kernel Size：$k\times k$
    

权重数量：

$$  
\boxed{N_{\text{weights}}=k^2C_{\text{in}}C_{\text{out}}}  
$$

如果每个输出 Channel 都有一个 Bias：

$$  
N_{\text{bias}}=C_{\text{out}}  
$$

总参数量：

$$
(k^2C_{\text{in}}+1)C_{\text{out}}  
$$

注意：**卷积层的参数量与输入图像的空间尺寸 $H,W$ 没有直接关系。**

因为所有空间位置共享同一组 Kernel 权重。

---

## 5. 1×1 Convolution

### 5.1 单输入 Channel、单输出 Channel

对于 $1\times1$ 卷积：

$$  
y_{i,j}=wx_{i,j}+b  
$$

参数数量：

- Weight：1
- Bias：1

总计 2 个参数。
### 5.2 多输入和输出 Channel

对于：

$$  
C_{\text{in}}=64,\quad C_{\text{out}}=32  
$$

一个 $1\times1$ 卷积在每个像素位置将 64 维特征映射为 32 维特征。

$$  
y_{i,j}=Wx_{i,j}+b  
$$

其中：

$$  
W\in\mathbb R^{32\times64}  
$$

参数量：

$$  
N=(64+1)\times32=2080  
$$

如果输入尺寸是：

$$  
32\times32\times64  
$$

那么采用 Stride = 1 的 $1\times1$ 卷积后可以得到：

$$  
32\times32\times32  
$$

因此：

**1×1 Convolution 可以在保持空间尺寸不变的情况下改变 Channel 数量。**

它主要对同一空间位置的不同 Channel 进行线性组合，不直接聚合相邻空间位置的信息。

---

## 6. Stride、Dilation、Padding

### 6.1 Stride（步长）

Stride 决定卷积核每次移动多少个输入位置。

- Stride = 1：每次移动 1 格。
- Stride = 2：每次移动 2 格。
- Stride = 3：每次移动 3 格。

Stride 增大通常会减小输出 Feature Map 的空间尺寸。这是一种 Downsampling（下采样）方式。

### 6.2 Dilation（空洞卷积）

Dilation 决定卷积核中相邻权重所对应输入位置之间的距离。

假设一维 Kernel：

$$  
w=[w_1,w_2,w_3]  
$$

Dilation = 1：

$$  
y=w_1x_1+w_2x_2+w_3x_3  
$$

Dilation = 2：

$$  
y=w_1x_1+w_2x_3+w_3x_5  
$$

Dilation = 3：

$$  
y=w_1x_1+w_2x_4+w_3x_7  
$$

注意：

$$  
\boxed{\text{跳过的元素数}=d-1}  
$$

因此 Dilation = 2 表示相邻权重之间跳过 1 个元素。

![](附件/Pasted%20image%2020261013190043.png)

对于 Kernel Size $k$ 和 Dilation $d$：

$$  
\boxed{k_{\text{effective}}=d(k-1)+1}  
$$

其中 $k_{\text{effective}}$ 是 Effective Kernel Size（有效卷积核尺寸）。

例如：

$$  
k=3,\quad d=2  
$$

则：

$$  
k_{\text{effective}}=2(3-1)+1=5  
$$

因此一个 $3\times3$ Kernel 的空间覆盖范围变为 $5\times5$。

但是仍然只有 9 个空间权重，而不是 25 个。

**Dilation 的作用：在不增加卷积核权重数量的情况下，扩大感受野。**

### 6.3 Padding（填充）

Padding 是在输入图像边缘补充额外像素。

课上提到的4种方式：

- Zero Padding：补 0。
- Mirror Padding：镜像填充。
- Circular Padding：循环填充。
- Average Padding：用平均值进行填充。

基本用的都是 Zero Padding。

常见的两个概念：

1. **Valid Convolution**

	不进行 Padding，仅在卷积核完全落在输入范围内的位置计算输出。
	
	通常会减小空间尺寸。

2. **Same Padding**

	选择适当的 Padding，使 Stride = 1 时输出尺寸与输入相同。
	
	注意：Same Padding 描述的是输出尺寸目标，Zero Padding 描述的是具体填充值，两者并非同一概念。

### 6.4 输出尺寸公式

对于一维卷积：

- $W$：输入长度。
- $k$：Kernel Size。
- $p$：每侧 Padding 数量。
- $s$：Stride。
- $d$：Dilation。
- 
输出长度：

$$
\left\lfloor  
\frac{W+2p-d(k-1)-1}{s}+1  
\right\rfloor  

$$

由于：

$$  
k_{\text{effective}}=d(k-1)+1  
$$

因此公式可以写成：

$$
\left\lfloor  
\frac{W+2p-k_{\text{effective}}}{s}  
\right\rfloor+1  
$$

理解公式：

1. $W+2p$：Padding 后输入的总长度。
2. $k_{\text{effective}}$：Kernel 实际覆盖的长度。
3. 两者相减：Kernel 最多能够移动的距离。
4. 除以 $s$ 并向下取整：最多能够移动的次数。
5. 最后 $+1$：加上 Kernel 初始位置产生的输出。
    
---

## 7. Pooling（池化）

Pooling 用于汇聚局部信息，通常同时降低 Feature Map 的空间分辨率。

### 7.1 Stride / Subsampling

按照固定间隔直接选取输入值。例如每隔两个位置选择一次。

### 7.2 Max Pooling（最大池化）

选择局部区域中的最大值。

例如：

$$  
X=  
\begin{bmatrix}  
1&3&4&2  
\end{bmatrix}  
$$

则：

$$  
\operatorname{MaxPool}(X)=4  
$$

### 7.3 Mean Pooling（平均池化）

计算局部区域的平均值：

$$
\frac{1+3+4+2}{4}  
=2.5  
$$

标准 Max/Mean Pooling 不包含可学习权重。

与普通卷积不同，Pooling 通常通过固定的聚合规则产生输出。

---

## 8. Backpropagation with Weight Sharing

CNN 的核心特点之一是多个输出位置使用同一组权重。

因此反向传播时，某个共享参数的梯度需要累加所有使用该参数的位置所贡献的梯度。

假设：

$$  
\hat y_i=ax_i  
$$

其中 $a$ 是共享参数。

定义 Loss：

$$  
L=\sum_i(ax_i-y_i)^2  
$$

对 $a$ 求导：

$$
\sum_i\frac{\partial}{\partial a}(ax_i-y_i)^2  
$$

得到：

$$
2\sum_i(ax_i-y_i)x_i  
$$

也就是说：

$$
\sum_i\frac{\partial L_i}{\partial a}  
$$

**同一个 Weight 在多个位置被重复使用，因此它的最终梯度是各个位置梯度贡献的总和。**

CNN 中卷积核的反向传播也遵循这一原则。

---

## 9. Classic CNN Architectures

### 9.1 AlexNet

![](附件/Pasted%20image%2020261013191755.png)

AlexNet 是经典的图像分类卷积神经网络。它的特征包括：

- 多层 Convolution。
- Max Pooling。
- Fully Connected Layers。
- Softmax。

AlexNet 使用的训练方法包括：

- Data Augmentation。
- Weight Decay。（跟正常的gradient descent加一个regularization本质没区别）
- Dropout。

### 9.2 VGGNet
![](附件/Pasted%20image%2020261013191905.png)
VGGNet 的典型特点是使用大量较小的 $3\times3$ 卷积核堆叠成更深的网络。

通过反复堆叠卷积层逐步扩大感受野，并通过 Pooling 降低空间分辨率。

例如两个连续的 $3\times3$ 卷积，在 Stride = 1、Dilation = 1 时具有 $5\times5$ 的理论感受野。

若各层输入输出 Channel 均为 $C$：

两个 $3\times3$ 卷积的权重数量：

$$  
2\times9C^2=18C^2  
$$

一个 $5\times5$ 卷积的权重数量：

$$  
25C^2  
$$

因此在这一特定条件下，堆叠两个 $3\times3$ 卷积可以使用更少的权重，并允许在两层之间引入额外的非线性激活。

---

## 10. Data Augmentation（数据增强）

Data Augmentation 指通过对训练数据进行变换，增加训练样本的多样性。

常见方法：

- Random Crop：随机裁剪。
- Horizontal Flip：水平翻转。
- Rotation：旋转。
- Color Jitter：颜色扰动。
- Random Erasing / Cutout：随机遮挡部分区域。
    

主要目的：

- 改善 Generalization（泛化能力）。
- 缓解 Overfitting（过拟合）。
- 增强对输入变化的鲁棒性。
    

---

## 11. Dropout（随机失活）

Dropout 是一种 Regularization（正则化）方法。我们可以将它描述为：

> Data augmentation applied inside our network during training.

与直接改变输入图片的数据增强不同，Dropout 在训练时随机关闭网络内部的部分激活值。

### 11.1 Basic Dropout

假设一个神经元原始输出为：

$$  
h_i  
$$

Keep Probability（保留概率）为 $p$。

最基础的 Dropout 定义为：

$$  
\tilde h_i=  
\begin{cases}  
0 & \text{概率 }1-p\  
h_i & \text{概率 }p  
\end{cases}  
$$

也就是说：

- 概率 $1-p$：关闭神经元。
    
- 概率 $p$：保留神经元。
    

由于部分神经元被随机关闭，训练时模型不能过度依赖某一个神经元。

这有助于促使网络学习更稳健的特征表示。

### 11.2 为什么需要 Scaling？

假设：

$$  
h_i=10,\quad p=0.5  
$$

如果不做 Scaling：

$$  
\tilde h_i=  
\begin{cases}  
0 & 50%\  
10 & 50%  
\end{cases}  
$$

则输出的期望：

$$
0.5(0)+0.5(10)=5  
$$

原始激活是 10，Dropout 后的期望却变成了 5。

为了保持平均输出不变，可以对激活进行缩放。

### 11.3 Inverted Dropout

PyTorch 使用 Inverted Dropout。

训练阶段：


$$
\tilde{h}_i =
\begin{cases}
0 & \text{概率 } 1-p \\
\frac{h_i}{p} & \text{概率 } p
\end{cases}
$$
  

测试阶段：

$$  
\boxed{\tilde h_i=h_i}  
$$

其中 $p$ 表示保留概率。

为什么训练时要除以 $p$？

因为：

$$  
\begin{aligned}  
\mathbb E[\tilde h_i]  
&=(1-p)\times0+p\times\frac{h_i}{p}\  =h_i  
\end{aligned}  
$$

因此训练阶段经过 Dropout 后，激活值的期望与原始激活相同。

例如：

$$  
h_i=10,\quad p=0.5  
$$

训练阶段：


$$
\tilde{h}_i =
\begin{cases}
0, & \text{概率 } 50\% \\
20, & \text{概率 } 50\%
\end{cases}
$$
  

期望：

$$
0.5(0)+0.5(20)=10  
$$

测试阶段直接输出：

$$  
\tilde h_i=10  
$$

**Inverted Dropout 的优点：测试时不需要额外缩放激活值。**

注意：保持激活的期望不变，不代表训练与测试时整个网络的预测完全相同。