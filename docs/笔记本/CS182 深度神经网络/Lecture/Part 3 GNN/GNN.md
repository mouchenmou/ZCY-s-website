## 1. Introduction to Graph Neural Networks

### 1.1 Graph（图）的基本概念

Graph Neural Network（GNN，图神经网络）是一种专门处理 Graph-structured Data（图结构数据）的神经网络。

一个 Graph 由两部分构成：

- **Node / Vertex（节点 / 顶点）**：表示一个对象。
- **Edge（边）**：表示两个节点之间的关系。

Graph 可以表示为：

$$  
G=(V,E)  
$$

常见 Graph-structured Data：

- Social Networks：用户为 Nodes，社交关系为 Edges。
- Molecular Graphs：原子为 Nodes，化学键为 Edges。
- Robot Networks：机器人为 Nodes，通信或交互关系为 Edges。
- Knowledge Graphs：实体为 Nodes，实体之间的关系为 Edges。
### 1.2 Node Features and Embeddings

Graph 中的每个 Node 通常都携带一个特征向量。例如，一个机器人节点可以用位置和速度表示：

$$  
h_i=  
\begin{bmatrix}  
x_i\  
y_i\  
v_i  
\end{bmatrix}  
$$

其中：

- $x_i,y_i$：机器人位置。
- $v_i$：机器人速度。
    

这个向量称为 Node Feature Vector（节点特征向量）。经过神经网络处理后，得到的隐藏特征表示通常称为 Node Embedding（节点嵌入）。因此，GNN 的输入不仅包含 Graph 的连接结构，还包含各个 Node 的 Features。

$$  
\boxed{\text{GNN Input}=\text{Graph Structure}+\text{Node Features}}  
$$

---

## 2. Generalizing CNNs to Graphs

### 2.1 CNN 与 GNN 的联系

CNN 主要处理具有规则 Grid Structure（网格结构）的数据，例如图像。

图像中的每个 Pixel 都有确定的空间位置，其邻居通常位于上、下、左、右及对角方向。

例如一个 $3\times3$ Kernel：

$$  
K=  
\begin{bmatrix}  
w_{NW}&w_N&w_{NE}\  
w_W&w_{\text{self}}&w_E\  
w_{SW}&w_S&w_{SE}  
\end{bmatrix}  
$$

CNN 在每个位置对邻近 Pixel 进行局部加权求和。

我们可以将：

- 每个 Pixel 看成一个 Node。
- 相邻 Pixels 之间的关系看成 Edge。
- Kernel Weights 看成不同邻居关系对应的权重。

因此，CNN 可以被理解为一种特殊的、在规则 Grid Graph 上执行的局部信息聚合操作。

### 2.2 Edge Labels（边标签）

在普通 CNN 中，不同方向的 Neighbor 具有不同含义：

- N：North，上方。
- S：South，下方。
- E：East，右方。
- W：West，左方。
- NE、NW、SE、SW：四个对角方向。
- Self：当前节点自己。

这些方向可以被视为 Edge Labels。

例如：

$$  
z_i=  
W_{\text{self}}h_i+  
\sum_{r\in\mathcal R}W_rh_{j_r}+b  
$$

其中：

- $h_i$：当前 Node 的 Feature。
    
- $h_{j_r}$：具有关系 $r$ 的 Neighbor Feature。
    
- $W_r$：关系 $r$ 对应的 Weight Matrix。
    
- $\mathcal R$：所有邻居关系类型的集合。
    
- $b$：Bias。
    

对于普通单通道 CNN，$W_r$ 可以退化为标量权重。

**核心思想：CNN 的卷积可以看成 Node 从 Neighbor 接收信息，再将这些信息进行加权聚合。**

### 2.3 Weight Sharing

CNN 中，同一个 Kernel 在不同像素位置重复使用。

GNN 也可以采用 Weight Sharing：

- 不同 Nodes 使用相同的 Message Function。
    
- 相同类型的 Edges 使用相同的权重。
    
- 不需要为每一个 Node 单独学习一套参数。
    

因此，GNN 能够将 CNN 的局部处理思想推广到不规则 Graph。

### 2.4 CNN 中哪些技术可以推广到 GNN？

课堂笔记指出：

比较容易推广的技术：

- Weight Sharing。
- Additional Feature Channels。
- Network Depth。
- Nonlinear Activation。
- Residual Connections。
- Dropout。

需要进一步讨论的技术：

- Normalization。
- Pooling / Downsampling。
- Upsampling。

这是因为 Graph 不一定具有规则的二维空间结构，也不一定具有固定的 Node 数量。

---

## 3. Message Passing（消息传递）

Message Passing 是 GNN 最核心的计算机制。

基本思想是：

**每个 Node 从与自己相连的 Neighbor Nodes 收集信息，再根据这些信息更新自己的 Feature。**

### 3.1 Neighbor Set

设当前 Node 为 $i$。

它的 Neighbor Set（邻居集合）表示为：

$$  
\mathcal N(i)  
$$

例如：

$$  
A-B-C  
$$

则：

$$  
\mathcal N(B)={A,C}  
$$

Node B 可以从 A 和 C 收集信息。

### 3.2 Message Passing 的基本步骤

Message Passing 通常分成三个阶段：

1. Message：生成消息。
    
2. Aggregation：聚合消息。
    
3. Update：更新节点。
    

#### Step 1：Message Function

Node $j$ 向 Node $i$ 发送消息：

$$  
m_{j\rightarrow i}=g_\theta(h_j)  
$$

其中：

- $h_j$：发送节点的 Feature。
    
- $g_\theta$：具有可学习参数的函数。
    
- $m_{j\rightarrow i}$：生成的消息。
    

最简单的 Message Function 可以是 Linear Transformation：

$$  
m_{j\rightarrow i}=Wh_j  
$$

更复杂时可以使用 MLP。

#### Step 2：Aggregation Function

Node $i$ 收集所有 Neighbors 的消息：

$$  
m_i=\sum_{j\in\mathcal N(i)}m_{j\rightarrow i}  
$$

常见 Aggregation Functions：

- Sum：求和。
    
- Mean：求平均值。
    
- Max：取最大值。
    
- Min：取最小值。
    

#### Step 3：Update Function

利用原来的 Node Feature 和聚合后的消息，生成新的 Feature。

$$  
h_i^{(l+1)}=  
f_\theta(h_i^{(l)},m_i^{(l)})  
$$

其中：

- $l$：当前 GNN Layer。
    
- $h_i^{(l)}$：当前层 Node $i$ 的 Feature。
    
- $h_i^{(l+1)}$：下一层 Node $i$ 的 Feature。
    

### 3.3 一个简单的 GNN 更新公式

一种简单设计是：

$$
\sigma\left(  
W_{\text{self}}h_i^{(l)}  
+  
\sum_{j\in\mathcal N(i)}  
W_{\text{neighbor}}h_j^{(l)}  
+b  
\right)  
$$

其中：

- $W_{\text{self}}$：处理当前 Node 的权重。
- $W_{\text{neighbor}}$：处理 Neighbor Features 的权重。
- $\sigma$：Activation Function，例如 ReLU。
- $b$：Bias。

不同 Neighbor 通常共享 $W_{\text{neighbor}}$。

### 3.4 Message Passing 计算例子

假设 Graph：

$$  
A-B-C  
$$

三个 Nodes 的 Features：

$$  
h_A=2,\qquad h_B=3,\qquad h_C=5  
$$

使用简单更新规则：

$$  
h_i^{\text{new}}=h_i+\sum_{j\in\mathcal N(i)}h_j  
$$

对于 Node A：

$$  
h_A^{\text{new}}=2+3=5  
$$

对于 Node B：

$$  
h_B^{\text{new}}=3+2+5=10  
$$

对于 Node C：

$$  
h_C^{\text{new}}=5+3=8  
$$

因此：

$$  
H_{\text{new}}=[5,10,8]  
$$

这里所有 Nodes 都使用上一层的 Features 同时进行更新，而不是按照 A、B、C 的顺序依次更新。

### 3.5 GNN 的 Receptive Field

在常规的一跳 Message Passing 中：

- 1 层 GNN：接收 1-hop Neighbors 的信息。
- 2 层 GNN：可以间接接收 2-hop Neighbors 的信息。
- 3 层 GNN：可以间接接收 3-hop Neighbors 的信息。
    

其中 Hop 指沿 Graph 的 Edge 移动一步。

因此：

**堆叠更多 Message Passing Layers，可以让 Node 利用更远距离的 Graph 信息。**

这与 CNN 中增加 Depth 可以扩大 Receptive Field 的思想类似。

---

## 4. Permutation Invariance（排列不变性）

### 4.1 为什么需要 Permutation Invariance？

Permutation 指重新排列顺序。

一般 Graph 中，一个 Node 的 Neighbors 没有天然的固定顺序。

例如：

$$  
\mathcal N(A)={B,C,D}  
$$

与：

$$  
\mathcal N(A)={D,B,C}  
$$

表示相同的邻居集合。

因此，改变 Neighbor 的排列顺序，不应该改变聚合后的信息。

这种性质称为：

**Permutation Invariance（排列不变性）。**

### 4.2 为什么不能直接拼接 Neighbor Features？

假设：

$$  
h_B=2,\qquad h_C=5,\qquad h_D=8  
$$

将 Features 拼成向量：

$$  
x=[2,5,8]  
$$

使用位置相关的 Linear Weights：

$$  
y=w_1x_1+w_2x_2+w_3x_3  
$$

若：

$$  
w=[1,2,3]  
$$

则：

$$  
y=1(2)+2(5)+3(8)=36  
$$

交换 Neighbor 顺序：

$$  
x'=[8,2,5]  
$$

则：

$$  
y'=1(8)+2(2)+3(5)=27  
$$

虽然 Neighbors 完全相同，但结果发生了变化。

因此，一般不能直接把无序邻居按照任意顺序拼接，再使用不同的位置权重。

### 4.3 Permutation-invariant Aggregation

Sum：

$$  
2+5+8=8+2+5  
$$

Mean：

$$  
\frac{2+5+8}{3}=\frac{8+2+5}{3}  
$$

Max：

$$  
\max(2,5,8)=\max(8,2,5)  
$$

这些操作都不会受到排列顺序的影响。

因此可以作为 Message Aggregation Functions。

### 4.4 Permutation Invariance 与 Equivariance

这里还需要区分两个概念。

**Permutation Invariance**

重新排列输入元素，最终输出不变。

例如 Graph-level Prediction 中，对 Node Embeddings 求和：

$$  
h_G=\sum_i h_i  
$$

改变节点编号顺序不会改变 $h_G$。

**Permutation Equivariance**

重新排列输入 Nodes，输出 Nodes 也按照对应顺序重新排列。

对于 Node-level GNN，通常要求满足这种性质。

即：重新编号不应该改变节点本身的预测，只应该改变结果的排列顺序。

---

## 5. Learnable Message Functions & Over-smoothing

### 5.1 Learnable Message Functions

简单的 GNN 使用：

$$  
m_{j\rightarrow i}=Wh_j  
$$

更一般地，Message Function 可以同时考虑发送者和接收者：

$$  
m_{j\rightarrow i}=g_\theta(h_i,h_j)  
$$

其中 $g_\theta$ 可以是一个 MLP。

这样不同 Neighbor 发出的消息可以依赖其与接收节点的关系。

### 5.2 Attention-style Aggregation

可以通过一个 Learnable Score 衡量不同 Neighbors 的重要程度。

首先计算：

$$  
s_{ij}=g_\theta(h_i,h_j)  
$$

再通过 Softmax：

$$
\frac{\exp(s_{ij})}  
{\sum_{k\in\mathcal N(i)}\exp(s_{ik})}  
$$

其中：

$$  
\sum_{j\in\mathcal N(i)}\alpha_{ij}=1  
$$

然后进行 Weighted Aggregation：

$$  
m_i=\sum_{j\in\mathcal N(i)}\alpha_{ij}Wh_j  
$$

不同 Neighbors 可以具有不同的权重。

这种思路与 Transformer 中的 Attention Mechanism 存在联系：

> Transformers can be thought of as GNNs on a complete graph.

Complete Graph（完全图）指任意两个不同 Nodes 之间都存在 Edge。

将 Transformer 中的 Tokens 看作 Nodes，可以把 Self-attention 理解成一种在 Token 之间交换信息的机制。

这是理解上的类比，并不意味着 Transformer 与所有 GNN 架构完全相同。

### 5.3 Nonlinearity（非线性）

如果 GNN 只是不断执行线性信息聚合，那么模型的表达能力可能受到限制。

例如：

$$  
H^{(l+1)}=AH^{(l)}W_l  
$$

如果不引入非线性：

$$  
H^{(2)}=A^2H^{(0)}W_0W_1  
$$

多个 Linear Transformations 可以在一定条件下合并。

因此 GNN 通常需要：

- Nonlinear Activation，例如 ReLU。
    
- MLP。
    
- Nonlinear Message Functions。
    
- Residual Connections。
    

### 5.4 Over-smoothing（过度平滑）

Over-smoothing 指：

**随着 Message Passing 不断进行，不同 Nodes 的 Embeddings 可能变得越来越相似。**

例如两个相连 Nodes：

$$  
h_A=0,\qquad h_B=10  
$$

若每次都将自己的信息与邻居取平均：

$$  
h_A^{\text{new}}=\frac{0+10}{2}=5  
$$

$$  
h_B^{\text{new}}=\frac{10+0}{2}=5  
$$

更新后两个 Nodes 的 Features 完全相同。

如果一个 Graph 中的不同 Nodes 最终都变得非常相似，就会降低模型区分 Nodes 的能力。

因此 GNN 的深度并非越深越好。

---

## 6. Normalization in GNN

CNN 中常见的输入形状：

$$  
X\in\mathbb R^{B\times C\times H\times W}  
$$

但 GNN 的单个 Graph 可以表示为：

$$  
H\in\mathbb R^{N\times C}  
$$

其中：

- $N$：Node 数量。
    
- $C$：Node Feature Dimension。
    

不同 Graph 的 Node 数量可能不同。

例如一个 Batch 中：

- Graph 1 有 5 个 Nodes。
    
- Graph 2 有 12 个 Nodes。
    
- Graph 3 有 8 个 Nodes。
    

因此 GNN 中的 Normalization 需要考虑不规则的节点数量。

### 6.1 BatchNorm

一种常见的 GNN BatchNorm 方法是：

对于固定 Feature Channel，在 Batch 内所有 Nodes 上计算统计量。

例如：

$$  
H=  
\begin{bmatrix}  
1&10\\  
3&20\\  
5&30  
\end{bmatrix}  
$$

第一列：

$$  
[1,3,5]  
$$

一起进行 Normalization。

第二列：

$$  
[10,20,30]  
$$

单独进行 Normalization。

如果一个 Batch 包含多个 Graph，常见实现会对 Batch 中的所有 Nodes 共同统计。

### 6.2 LayerNorm

课堂介绍的一种简单方式是：

**对每个 Node 的 Embedding 独立进行 Normalization。**

例如：

$$  
h_A=[1,3,5]  
$$

$$  
h_B=[10,20,30]  
$$

则：

- Node A 对自己的三个 Feature 进行 Normalization。
    
- Node B 对自己的三个 Feature 进行 Normalization。
    

不同 Nodes 不参与彼此的统计。

这就是把 LayerNorm 应用到每个 Node Embedding 上的一种方式。

实际 GNN 中还有其他 Normalization 变体。

---

## 7. Node-level and Graph-level Tasks

### 7.1 Node-level Prediction（节点级预测）

目标是预测某个 Node 的属性或 Label。

例如：

- 预测社交网络中某个用户的类别。
    
- 预测某个分子中原子的性质。
    
- 预测多机器人系统中每个机器人的状态或行为类别。
    

经过多层 GNN：

$$  
h_i^{(L)}  
$$

然后使用分类器：

$$  
\hat y_i=\operatorname{MLP}(h_i^{(L)})  
$$

每个 Node 有自己的预测。

### 7.2 Graph-level Prediction（图级预测）

Graph-level Prediction 的目标是预测整个 Graph 的属性。

例如预测整个分子的性质。

首先获得所有 Node Embeddings：

$$  
h_1^{(L)},h_2^{(L)},\ldots,h_N^{(L)}  
$$

然后使用 Graph-level Aggregation：

$$  
h_G=\sum_{i=1}^{N}h_i^{(L)}  
$$

最后：

$$  
\hat y_G=\operatorname{MLP}(h_G)  
$$

这种将整个 Graph 汇总为一个向量的操作常称为 Readout 或 Global Pooling。

### 7.3 Held-out Nodes（留出节点）

**Approach 1：Remove Nodes**

直接从 Graph 中移除留出的 Nodes。

问题：

- 会改变 Graph Structure。
- 可能破坏 Connectivity。
- 影响其他 Nodes 的 Message Passing。
    

**Approach 2：Mask Node Features**

保留 Nodes 和 Edges，但隐藏相应的 Node Features。

例如：

$$  
h_i^{(0)}=m  
$$

其中 $m$ 是一个 Mask Vector，可以设计为可学习参数。

这种方法保留 Graph Structure，让被 Mask 的 Node 仍然能够从其他 Nodes 接收消息。

训练时还可以采用 Random Masking：

1. 随机选择部分有 Label 的 Nodes。
    
2. 将其输入 Features 替换为 Mask。
    
3. 仍然要求模型预测这些 Nodes 的 Labels。
    
4. 对这些预测计算 Loss。
    

这样可以训练模型根据 Graph Context 推断缺失的信息。

注意，这里描述的是课堂讨论的 Masking 方案，并不代表所有 Node-level GNN 任务都必须这样处理测试节点。

---

## 8. Graph Pooling & Downsampling

### 8.1 为什么需要 Graph Pooling？

CNN 可以通过 Stride 或 Pooling 减少空间分辨率：

$$  
256\times256\rightarrow128\times128\rightarrow64\times64  
$$

但 Graph 没有规则的二维网格，因此不能直接使用相同的 Pooling 方式。

GNN 中 Pooling 的一个重要目的，是将较大的 Graph 压缩成较小的 Graph，同时尽可能保留重要结构与特征。

这样可以：

- 减少 Nodes。
    
- 构建 Hierarchical Representations。
    
- 更快聚合大范围的 Graph 信息。
    

### 8.2 Clustering（聚类）

Clustering 是将相似或联系密切的 Nodes 划分到同一个 Group。

例如：

$$  
V={1,2,3,4,5,6}  
$$

划分为两个 Clusters：

$$  
G_1={1,2,3}  
$$

$$  
G_2={4,5,6}  
$$

这样原来的 6 个 Nodes 可以聚合成 2 个新的 Nodes。

### 8.3 K-means Clustering

课堂回顾了经典 K-means。

给定 Cluster 数量 $K$：

1. 初始化 $K$ 个 Cluster Centers。
    
2. 根据距离将每个数据点分配给某个 Cluster。
    
3. 根据分配结果重新计算 Centers。
    
4. 重复直到满足停止条件。
    

一个困难是：

**Hard Assignment（硬分配）通常包含离散的 Cluster 选择操作，不能直接通过普通 Backpropagation 对分配过程进行端到端优化。**

因此老师引入 DiffPool。

### 8.4 Over-smoothing 与 Over-squashing

**Over-smoothing**

反复聚合邻居信息，使不同 Node Embeddings 越来越相似。

**Over-squashing**

大量远距离节点的信息需要经过有限维度的 Node Embeddings 进行传递，导致信息被压缩，部分重要信息可能丢失。

两者区别：

- Over-smoothing：不同 Nodes 的表示趋于相似。
    
- Over-squashing：大量信息被压缩进有限维向量，难以完整保留。
    

老师还提出一种简单的扩大信息传播范围的方法：

增加一个 Global Node，与所有 Nodes 相连。

这样不同 Nodes 可以通过 Global Node 更快交换信息。

但这并不自动解决所有信息压缩问题。

---

## 9. DiffPool（Differentiable Pooling）

DiffPool 是一种可以通过 Backpropagation 训练的 Graph Pooling 方法。

核心思想：

**让 GNN 学习每个原始 Node 属于不同 Clusters 的 Soft Assignment（软分配）。**

### 9.1 Soft Assignment

假设原来有 3 个 Nodes：

$$  
A,B,C  
$$

希望 Pooling 后得到 2 个 Clusters：

$$  
G_1,G_2  
$$

使用一个 Assignment Matrix：

$$  
S=  
\begin{bmatrix}  
0.9&0.1\  
0.8&0.2\  
0.2&0.8  
\end{bmatrix}  
$$

含义：

- Node A：90% 分配给 Cluster 1，10% 分配给 Cluster 2。
    
- Node B：80% 分配给 Cluster 1，20% 分配给 Cluster 2。
    
- Node C：20% 分配给 Cluster 1，80% 分配给 Cluster 2。
    

每一行的和为 1。

一般情况下：

$$  
\boxed{S\in\mathbb R^{N\times K}}  
$$

其中：

- $N$：原始 Node 数量。
    
- $K$：新的 Cluster 数量。
    
- $S_{ik}$：Node $i$ 分配给 Cluster $k$ 的权重。
    

### 9.2 如何学习 Assignment Matrix？

老师提出：

> Don't know what to do? Make it learnable.

使用一个专门的 GNN，为每个 Node 生成 $K$ 个 Scores。

然后经过 Softmax，得到属于各个 Clusters 的概率。

$$  
\boxed{  
S=\operatorname{Softmax}(\operatorname{GNN}_{\text{pool}}(A,H))  
}  
$$

其中 Softmax 沿 Cluster 维度进行，使每个 Node 对各 Clusters 的分配权重之和为 1。

这个专门预测 Assignments 的 GNN，可以与负责学习 Node Embeddings 的 GNN 使用不同参数。

### 9.3 生成新的 Node Embeddings

假设原始 Node Features：

$$  
H=  
\begin{bmatrix}  
2\  
4\  
8  
\end{bmatrix}  
$$

Assignment Matrix：

$$  
S=  
\begin{bmatrix}  
0.9&0.1\  
0.8&0.2\  
0.2&0.8  
\end{bmatrix}  
$$

Cluster 1：

$$  
h_{G_1}=0.9(2)+0.8(4)+0.2(8)=6.6  
$$

Cluster 2：

$$  
h_{G_2}=0.1(2)+0.2(4)+0.8(8)=7.4  
$$

写成矩阵形式：

$$  
\boxed{H'=S^\top H}  
$$

维度：

$$  
S^\top\in\mathbb R^{K\times N}  
$$

$$  
H\in\mathbb R^{N\times C}  
$$

因此：

$$  
\boxed{H'\in\mathbb R^{K\times C}}  
$$

也就是说，原来 $N$ 个 Node Embeddings，被聚合成了 $K$ 个 Cluster Embeddings。

这里采用的是 Weighted Sum（加权求和）。

### 9.4 生成新的 Adjacency Matrix

原始 Graph 的连接关系由 Adjacency Matrix 表示：

$$  
A\in\mathbb R^{N\times N}  
$$

其中 $A_{ij}$ 表示 Node $i$ 和 Node $j$ 之间的连接关系或边权重。

经过 Pooling 后，需要生成新的 Graph Connectivity。

DiffPool 使用：

$$  
\boxed{A'=S^\top AS}  
$$

维度：

$$  
(K\times N)(N\times N)(N\times K)  
$$

最终：

$$  
\boxed{A'\in\mathbb R^{K\times K}}  
$$

其中 $A'_{pq}$ 表示 Cluster $p$ 和 Cluster $q$ 之间聚合后的连接强度。

新的 Adjacency Matrix 通常具有连续权重，不一定只有 0 和 1。

### 9.5 DiffPool 的完整流程

原始输入：

$$  
H\in\mathbb R^{N\times C}  
$$

$$  
A\in\mathbb R^{N\times N}  
$$

第一步：使用 GNN 预测 Assignment Matrix。

$$  
S\in\mathbb R^{N\times K}  
$$

第二步：计算新的 Node Embeddings。

$$  
\boxed{H'=S^\top H}  
$$

第三步：计算新的 Adjacency Matrix。

$$  
\boxed{A'=S^\top AS}  
$$

最终得到一个新的 Graph：

$$  
H'\in\mathbb R^{K\times C}  
$$

$$  
A'\in\mathbb R^{K\times K}  
$$

当 $K<N$ 时，完成 Graph Downsampling。

整个过程中的 Assignment Matrix 由神经网络生成，可以通过 Backpropagation 进行端到端训练。

---

## 10. Auxiliary Losses in DiffPool

Auxiliary Loss（辅助损失）是指在原本任务 Loss 之外，额外加入的训练目标。

Lecture 12 第 12 页介绍了两个用于 DiffPool 的辅助损失。

### 10.1 Link Prediction Loss

目标：

**鼓励 Cluster Assignment 保留原 Graph 的拓扑连接关系。**

例如两个 Nodes 在原 Graph 中相互连接，那么希望它们的 Cluster Assignments 也具有较高相似性。

一种损失函数：

$$  
\boxed{  
L_{\text{LP}}=|A-SS^\top|_F  
}  
$$

其中：

- $A$：原始 Adjacency Matrix。
    
- $S$：Assignment Matrix。
    
- $|\cdot|_F$：Frobenius Norm。
    

注意：

$$  
(SS^\top)_{ij}=S_i^\top S_j  
$$

即两个 Node 的 Assignment Vectors 的内积。

如果两个 Nodes 的 Assignment Vectors 较相似，其内积通常较大。

因此，最小化 $L_{\text{LP}}$ 可以鼓励 Soft Assignments 反映原始 Graph 的连接结构。

### 10.2 Entropy Loss

目标：

**鼓励 Soft Assignment 接近 Hard Assignment。**

例如：

$$  
S_i=[0.5,0.5]  
$$

说明该 Node 对两个 Clusters 没有明显偏好。

而：

$$  
S_i=[0.99,0.01]  
$$

说明它几乎完全属于第一个 Cluster。

使用 Entropy：

$$  
H(p)=-\sum_{k=1}^{K}p_k\log p_k  
$$

对所有 Nodes 求平均：

$$  
\boxed{  
L_{\text{ent}}=\frac1N\sum_{i=1}^{N}H(S_i)  
}  
$$

当概率比较均匀时，Entropy 较高。

当概率集中在某个 Cluster 时，Entropy 较低。

因此最小化 Entropy Loss，会鼓励模型产生更加明确的 Cluster Assignments。

### 10.3 Total Loss

DiffPool 可以将主要任务 Loss 与辅助 Loss 结合：

$$
L_{\text{task}}  
+  
\lambda_{\text{LP}}L_{\text{LP}}  
+  
\lambda_{\text{ent}}L_{\text{ent}}   
$$

其中：

- $L_{\text{task}}$：原本任务的 Loss。
    
- $L_{\text{LP}}$：Link Prediction Loss。
    
- $L_{\text{ent}}$：Entropy Loss。
    
- $\lambda_{\text{LP}},\lambda_{\text{ent}}$：控制辅助损失强度的超参数。
    