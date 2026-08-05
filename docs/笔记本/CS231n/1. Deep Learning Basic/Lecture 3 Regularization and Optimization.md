# 1. Regularization（正则化）

![](附件/Pasted%20image%2020260706142140.png)

## 1.1 Regularization 是干什么的

将 $Loss$ 函数分为 $Data loss$ 和 $Regularization$ 两部分

如果某个模型为了把训练集拟合得特别好，搞出一堆很大的权重，比如：

$$W=[100,-250,80,\dots]$$

那它的 $R(W)$ 会很大，整体 $loss$ 就会被惩罚。

因此 $R(W)$ 的作用是防止 **$Overfitting$**（过拟合）。

!!! explanation "过拟合和欠拟合"
    假设两个学生A和B，A看到题目后专门总结规律，遇到新的题目他也能秒杀。B把所有的练习题都背下来了，但是遇到了新的题照样不会做。这就是机器学习里的区别：
    
    $$训练集表现好 \ne 真的学会规律$$
    
    训练集表现好，有可能只是模型把训练集背下来了，不一定是真的学会了。
    
    过拟合就是：模型把训练数据里面的偶然细节、噪声、特殊情况也当成了规律。
    
    一种理解方式：
    
    - 过拟合：一个学生拿到的资料太少了，没法学会一个东西，就只能记题目。
    - 欠拟合：一个学生拿到的资料太多了，这个东西学不完了，脑容量不够。
    
    因此过拟合的话，可以增加训练数据，减小网络容量。
    
    欠拟合的话，可以增大网络容量，提高训练的时间。

!!! example "看个例子"
    以下图为例。$f_1$ 不太注重细节，$f_2$ 过于注重细节。虽然在训练数据（蓝色点点）上，$f_2$ 的表现比 $f_1$ 好。但是在测试数据（白色点点）上，明显是 $f_1$ 优于 $f_2$。这就是正则化在起作用。
    ![](附件/Pasted%20image%2020260706144755.png)


## 1.2 Regularization strength: $\lambda$

$\lambda$ 是正则化强度，也就是另一个超参数，因此我们在也要在训练时确定一个最优的 $\lambda$

## 1.3 常见的正则化函数


L1：把所有权重的绝对值加起来

$$L1:\ R(W)=\sum_k\sum_l |W_{k,\ l}|$$​
L2：把所有权重平方再加起来。

$$L_2:\ R(W)=\sum_k\sum_l W_{k,\ l}^2$$

---

# 2. Optimization（优化）

## 2.1 Optimization的目标

我们现在有一个损失函数 $loss=L(W)$

训练模型就是要让它变小：

$$\min_W L(W)$$

所以 $optimization$ 的目标是：**调整 $W$，让 $loss$ 越来越小。**


## 2.2 Optimization的原理

!!! explanation "一维的情况"
    先看一维的情况，即只有一个变量。比如 $f(x)=x^2$

    它的导数是：$f'(x)=2x$

    导数告诉我们 $x$ 往右动一点，函数值是上升还是下降，上升/下降有多快。

    - $f′(3)=6$，导数是正的。想让函数变小，就应该往左走。
    - $f′(-3)=-6$，导数是负的。想让函数变小，就应该往右走。

    可以规定梯度下降公式为:
    
    $$x_{new}​=x−ηf'(x)$$
    
    带入上面那两个例子看看:
    
    - $x=3$ 时 $f'(3)=6$，说明 $x$ 向右 $f(x)$ 走会增大，因此 $x$ 要向左走，此时 $x_{new}​=3−6η<3$，符合 $x$ 向左走
    - $x=-3$ 时 $f'(-3)=-6$，说明 $x$ 向右走 $f(x)$ 会减小，因此 $x$ 要向右走，此时 $x_{new}​=-3-（-6）η》3$，符合 $x$ 向右走。
    
    因此**负梯度（negative gradient）就是我们要更新的方向**。

!!! explanation "多维的情况"
    神经网络里面不是只有一个 xxx，而是有一大堆参数：

    $$W=[w_1,w_2,w_3,\dots]$$
    
    即：
    
    $$loss函数：L(W)=L(w_1,\ w_2,\ w_3,\ ...)$$
    
    所以每个方向都要算一个导数。这些导数组合在一起，就叫 **gradient（梯度）**：

    $$\nabla L(W)= \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \frac{\partial L}{\partial w_3}, \dots \right]$$
    
    通过上式可以看出，$梯度=每个参数分别对loss的影响$。
    
    以三维的情况为例，假设 $∇L(W)=[3,−2,5]$，意思是：

    $$\frac{\partial L}{\partial w_1}=3$$

    说明 $w_1$​ 增大，$loss$ 会变大，所以 $w_1$​ 应该减小。
    
    $$\frac{\partial L}{\partial w_2}=-2$$

    说明 $w_2$​ 增大，$loss$ 会变小，所以 $w_1$​ 应该增大。
    
    $$\frac{\partial L}{\partial w_3}=-5$$

    说明 $w_3$​ 增大，$loss$ 会变小，所以 $w_3$​ 应该增大。
    
    我们要往梯度的反方向走，即 $−∇L(W)=[−3,2,−5]$，因此要这么更新：
    
    $$W_{new}=W-η\times[3,-2,5]=W+η\times[−3,2,−5]$$

上面两个例子说明了，我们想让 $loss$ 变小，就要知道**参数 $W$ 往哪个方向改，$loss$ 下降最快**。这个方向就是**负梯度方向**

```python
while True:
    weight_grad = evaluate_gradient(loss_fun, data, weights)
    weights+ = -step_size * weights_grad # perform parameter update
```

上面这个方法不太好，会一直持续下去。我们可以提前设置好迭代次数，或者等 $weights$ 改动量非常小了之后停止循环。

---
## 2.3 Stochastic Gradient Descent (SGD): 随机梯度下降

当我们的数据集很大很大的时候，每一次梯度更新都需要用到所有的 $training \ data（不包括validation \ data）$，这样既浪费时间，又会造成巨大的消耗。因此，每一次的梯度更新，我们都随机选取 $training \ data$ 中的一部分 $data$ 进行更新。

所以完整的训练过程中，几乎所有的 $training\ data$ 还是都会被用上的，只不过不是每一次更新都被用上。

```python
while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights+ = -step_size * weights_grad # perform parameter update
```

---

## 2.4 梯度下降和随机梯度下降存在的一些问题

### 2.4.1 Problem 1：不同方向的变化速度不一样

问题描述：$loss$ 在不同方向上的变化速度可能差很多。那就有可能出现 Very slow progress along shallow dimension, jitter along steep direction. 

#### 解析：
比如有两个参数 $w_1,w_2$。$loss$ 对 $w_1$ 的变化很慢，对 $w_2$ 的变化很快。这个时候损失函数的形状就像一个很窄很长的山谷。这里的每个圆环都代表都代表 $loss$ 相等的一个环（有点像等高线），如下图所示：

注意：这是一个二维图！！！就跟xy坐标图一样的，不是三维的！！！不要把它想当成等高线。当成等高线就理解不了 $w_2$ 为何会反复横跳了。

![](附件/Pasted%20image%2020260706194533.png)

以 $L(w1​,w2​)=0.0001(w_1-20)^2​+(w_2-5)^2​$ 为例。这个情况就是 $loss$ 对 $w_2$ 很敏感，对 $w_1$ 不敏感。

$$由于\ W_{new}←W−η∇L$$

这个时候会出现一个问题，如果我们的学习率 $η$ 比较大的话，会出现 **$w_2$ 来回跳动的情况**：

因为参数 $W$ 的改动是找到使得 $loss$ 下降最快（即$|\frac{\partial L}{\partial w}|$ 最大）的方向来改动的 。$w_2$ 稍微变化一点点 $\frac{\partial L}{\partial w_2}$ 就会很大。

$$假定某一时刻\ \ w_2=4，我们最终希望得到的结果是w_1=20,\ w_2=5$$

$$对 L(w_2)=(w_2-5)^2 进行求导$$

$$解得 L'(w_2)=-2$$

$$ \begin{aligned} 更新：w_{2new} &= w_2 - \eta \times L'(w_2) \\ &= 4 - \eta \times (-2) \\ &= 4 + 2\eta \end{aligned} $$

若 $\eta=0.9$（也就是 $\eta$ 比较大的情况），更新后的 $w_{2new}=5.8>5$，也就是 $w_2$ 跳太高了，那么下一次更新的时候 **$w_2$ 又会跳回小于5的值**。而 $w_1$ 的变化速度非常缓慢，只要 $\eta$ 不太大的情况下，基本不会出现横跳的情况，所以 $w_1$ 是缓慢稳步前行的。

![](附件/Pasted%20image%2020260706202601.png)

---

### 2.4.2 Problem 2：local minima 和 saddle point

问题描述：$loss$ 函数里面可能有 **local minima（局部最小值）** 或者 **saddle point（鞍点）**。
![](附件/Pasted%20image%2020260706203320.png)
#### 2.4.2.1 Local minima：局部最小值

局部最小值就是：站在这个点附近看，它确实是最低的；但是放到整个函数上看，它不一定是全局最低点。

也就是说：

局部最小值不等于全局最小值。

如果模型走到一个局部最小值，梯度可能就接近 0 了。梯度下降看到梯度为 0。

但实际上它可能只是掉进了一个小坑。

#### 2.4.2.2 Saddle point：鞍点

鞍点比局部最小值更常见，尤其是在高维空间里。

鞍点的特点是：

- 从某些方向看，它像最低点
- 从另一些方向看，它又像最高点
- 但是这个点附近的梯度可能也是 0
![](附件/Pasted%20image%2020260706204121.png)

有研究证明：Saddle points much more common in high dimension.

神经网络的参数不是二维、三维，而是几百万甚至几亿维。维度越高，某个点在所有方向上都是局部最小值的概率反而很低；更常见的是某些方向能下降、某些方向不能下降的鞍点。

---

### 2.4.3 Problem 3：minibatch 会带来噪声

问题描述：SGD 不是每次都用完整训练集算梯度，而是随机抽一个 minibatch。

这样做速度更快，但是也带来一个问题：每个 minibatch 算出来的梯度都只是对真实梯度的估计。

也就是说：

$$
minibatch\ gradient \approx full\ gradient
$$

它不是完全准确的。

!!! example "为什么会有噪声？"
    假设训练集里有 50000 张图片，我们每次只抽 256 张。
    
    这 256 张图片大体上能代表整个训练集，但不可能每次都完全代表。
    
    某一次 minibatch 里面猫比较多，下一次车比较多，那么这两次算出来的梯度方向就可能不太一样。
    
    所以 SGD 的路线不会是一条很平滑的下降曲线，而是会有一些抖动。

这种噪声有时候也不是纯坏事。它可能帮助模型从很浅的局部最小值或鞍点附近抖出来。但是噪声太大时，训练就会不稳定。

---

## 2.5 SGD + Momentum

为了解决 SGD 的这些问题，需要引入 **Momentum（动量）**。

普通 SGD 的更新方式是：

$$
W_{t+1}=W_t-\eta \nabla L(W_t)
$$

也就是每一步只看当前这一刻的梯度。

!!! explanation "Momentum"
    假设我们在山谷里下坡。
    
    普通 SGD 每一步都只看脚下这一点的坡度，所以很容易左右摇摆。
    
    Momentum 会记住之前大致往哪里走。如果最近几步都在往某个方向走，那就继续沿着这个方向多走一点。
    
    这样可以：
    
    - 减少陡峭方向上的来回震荡
    - 加快平缓方向上的前进速度
    - 在一些很浅的坑或者鞍点附近更容易冲出去

Momentum 会维护一个速度：

$$
v_{t+1}=\rho v_t-\eta \nabla L(W_t)
$$

然后用这个速度更新参数：

$$
W_{t+1}=W_t+v_{t+1}
$$

其中 $\rho$ 可以理解为“保留多少历史速度”。课件里说通常取：

$$
\rho=0.9\quad 或者\quad 0.99
$$

如果 $\rho$ 越大，说明越相信过去的方向；如果 $\rho$ 越小，说明越相信当前的梯度。

```python
v = rho * v - learning_rate * dw
w += v
```

!!! explanation "SGD vs. SGD+Momentum"
    ![](附件/Pasted%20image%2020260706210110.png)

!!! explanation "SGD+Momentum的两种等价形式"
    ![](附件/Pasted%20image%2020260706224144.png)

---

## 2.6 RMSProp

Momentum 解决的是方向惯性的问题，但还有一个问题：不同参数方向的坡度大小差别可能很大。

RMSProp 的方法是，==给每个参数单独调学习率==。

也就是说，不再让所有参数都用同一个更新幅度，而是根据每个方向过去的梯度大小来缩放它。

!!! explanation "如何理解学习率不同"
    # Momentum 和 RMSProp 的一个误区：更新量不同 ≠ 学习率不同

    ## 误区
    
    我一开始以为：
    
    $$v_{t+1}=\rho v_t-\eta \nabla L(W_t)$$
    
    里面每个参数的梯度本来就不同，比如：
    
    $$\frac{\partial L}{\partial w_1} \neq \frac{\partial L}{\partial w_2}$$
    
    每个参数的更新量也不同，所以我误以为Momentum 其实也相当于给每个参数单独设置了学习率。
    
    这个理解是不准确的。
    
    ---
    
    ## 正确理解
    
    **每个参数更新量不同，不等于每个参数学习率不同。**
    
    在 Momentum 里面：
    
    $$v_{t+1}=\rho v_t-\eta \nabla L(W_t)$$
    
    拆开看：
    
    $$v_{1,t+1}=\rho v_{1,t}-\eta \frac{\partial L}{\partial w_1}$$

    $$v_{2,t+1}=\rho v_{2,t}-\eta \frac{\partial L}{\partial w_2}$$
    
    虽然：$\frac{\partial L}{\partial w_1}$ 和 $\frac{\partial L}{\partial w_2}$ 可能不同，所以 $w_1,w_2$ 的更新量不同。

    但是它们前面乘的学习率都是 $\eta$

    所以 Momentum 不是给每个参数单独调学习率。

    它只是**用同一个学习率 $\eta$，乘上每个参数自己的梯度**，再加上历史速度项。

    ---

    ## Momentum 主要解决什么？

    Momentum 的作用是累积历史更新方向：

    $$v_{t+1}=\rho v_t-\eta g_t$$

    $$W_{t+1}=W_t+v_{t+1}$$

    其中：

    $$g_t=\nabla L(W_t)$$

    Momentum 更像是：
    
    > 如果某个方向连续几次都朝同一个方向走，就加速；如果某个方向来回震荡，就互相抵消。
    
    所以 Momentum 主要缓解的是 $\text{zigzag 震荡}$ ，但它没有给每个参数设置不同的有效学习率。
    
    ---
    
    ## RMSProp

    RMSProp 是：
    
	$$s_{t+1}=\rho s_t+(1-\rho)g_t^2$$
	
	$$W_{t+1}=W_t-\eta\frac{g_t}{\sqrt{s_{t+1}}+\epsilon}$$
	
	拆成每个参数：
	
	$$
	w_{1,t+1}=w_{1,t}-\eta\frac{g_1}{\sqrt{s_1}+\epsilon}
	$$
	
	$$
	w_{2,t+1}=w_{2,t}-\eta\frac{g_2}{\sqrt{s_2}+\epsilon}
	$$
	
	这里关键是：
	
	$$
	s_1 \neq s_2
	$$
	
	其中：$s_1$ 记录的是 $w_1$ 这个方向过去梯度平方的大小。$s_2$ 记录的是 $w_2$ 这个方向过去梯度平方的大小。
	
	因此 $w_1$ 的实际有效学习率是：
	
	$$
	\eta_1=\frac{\eta}{\sqrt{s_1}+\epsilon}
	$$
	
	$w_2$ 的实际有效学习率是：
	
	$$
	\eta_2=\frac{\eta}{\sqrt{s_2}+\epsilon}
	$$
	
	所以 RMSProp 才是真的给每个参数分配了不同的有效学习率。


---
### 2.6.1 RMSProp的原理

RMSProp 会记录每个维度历史梯度平方的平均值：

$$
cache_{new}=\rho \times cache+(1-\rho)\times(dw)^2
$$

然后更新时除以这个平方平均值：

$$
W=W-\eta \frac{dw}{\sqrt{cache_{new}}+\epsilon}
$$

这里的 $\epsilon$ 是一个很小的数，主要是为了防止分母为 0。$dw=∇f(w)$

```python
cache = decay_rate * cache + (1 - decay_rate) * dw ** 2
w += -learning_rate * dw / (np.sqrt(cache) + eps)
```

### 2.6.2 RMSProp 为什么有用？

如果某个方向的梯度一直很大，说明这个方向很陡。RMSProp 会让这个方向的 $cache$ 变大，于是更新步长被除小。

如果某个方向的梯度一直很小，说明这个方向比较平。RMSProp 会让这个方向的 $cache$ 比较小，于是这个方向相对可以走快一点。

总结为：

> Progress along “steep” directions is damped; progress along “flat” directions is accelerated.

翻译一下就是：

- 陡峭方向：更新被压住，不要乱撞
- 平缓方向：更新被放大，走得快一点

---

## 2.7 Adam

Adam 可以理解成：

$$
Adam \approx Momentum + RMSProp
$$

也就是说，它一方面像 Momentum 一样记录梯度的一阶矩，也就是大致的方向；另一方面像 RMSProp 一样记录梯度平方的二阶矩，用来做自适应学习率。

### 2.7.1 Adam 的更新公式

Adam 会维护两个量：

1. $m$：梯度的滑动平均，可以理解成 Momentum 里的速度方向。
2. $v$：梯度平方的滑动平均，可以理解成 RMSProp 里的 cache。

公式大致是：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

其中 $g_t$ 就是当前梯度。

然后用它们来更新参数：

$$
W_{t+1}=W_t-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

### 2.7.2 Bias correction：偏差修正



因为一开始 $m$ 和 $v$ 都初始化为 0，所以训练刚开始的时候，$m$ 和 $v$ 会偏小。

比如第一步：

$$
m_1=\beta_1\cdot0+(1-\beta_1)g_1
$$

如果 $\beta_1=0.9$，那么：

$$
m_1=0.1g_1
$$

这个值明显比真实梯度小很多。

所以 Adam 会做修正：

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

最后用修正后的 $\hat{m}_t,\hat{v}_t$ 更新：

$$
W_{t+1}=W_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

课件里给的经验默认值是：

$$
\beta_1=0.9,\quad \beta_2=0.999,\quad learning\ rate=10^{-3}\ 或\ 5\times10^{-4}
$$

!!! note "实践中的 Adam"
    很多时候 Adam 是一个很好用的默认选择。
    
    如果不知道先用什么 optimizer，可以先试 Adam 或 AdamW。它不一定永远最好，但通常很稳。

---

## 2.8 AdamW：带 Weight Decay 的 Adam

AdamW 是 Adam 的一个变体，重点在于它处理正则化的方式和普通 Adam 不一样。

前面讲过 L2 regularization 会在 loss 里面加入：

$$
\lambda ||W||^2
$$

如果直接把这一项加进 loss，再用 Adam 优化，那么这个 L2 项也会参与 Adam 的 moment 计算。

也就是说，普通 Adam 会把 L2 正则化也混进 $m$ 和 $v$ 里面。

AdamW 的做法是：把 weight decay 和梯度更新解耦。

!!! explanation "AdamW 的直觉"
    普通 Adam 像是先把“数据梯度”和“正则化梯度”揉在一起，再拿去算动量和自适应学习率。
    
    AdamW 则是先根据数据梯度做 Adam 更新，然后额外把权重往 0 拉一点。
    
    所以 weight decay 更像一个单独的“缩小权重”的操作，而不是混在梯度里面。

可以粗略理解为：

$$
W \leftarrow W-\eta\cdot AdamUpdate(W)
$$

然后再做：

$$
W \leftarrow W-\eta\lambda W
$$

这就是 decoupled weight decay 的思想。

当然这种AdamW比一定会比Adam好，在深度学习中就是把每一种方式都跑一遍，看哪种更强。

---

## 2.9 Learning Rate Schedules：学习率怎么变

不管是 SGD、SGD + Momentum、RMSProp、Adam 还是 AdamW，都有一个共同的超参数：

$$
learning\ rate
$$

也就是学习率。

学习率太大，可能直接越过最低点，甚至让 loss 爆炸；学习率太小，训练会非常慢。

所以实际训练中，学习率通常不是一直不变，而是会随时间调整。

### 2.9.1 Step decay

训练到某些固定 epoch 时，把学习率突然变小。

比如：

> 在第 30、60、90 个 epoch 后，把 learning rate 乘以 0.1。

也就是：

$$
\eta \leftarrow 0.1\eta
$$

!!! explanation "为什么后面要降低学习率？"
    训练刚开始时，我们离最优点很远，所以可以走大步。
    
    训练到后面时，我们已经接近比较好的区域了，这时候还走大步就容易来回震荡。
    
    所以后期要把学习率调小，让模型更细致地靠近最低点。

### 2.9.2 Cosine decay

Cosine decay 会让学习率按照余弦曲线慢慢下降。

它不像 step decay 那样突然跳变，而是比较平滑地从初始学习率降到较小的值。

直观上就是：

> 前期下降慢一点，中间下降快一点，后期再慢慢贴近 0。

这种方式在很多深度学习训练里都很常见。

### 2.9.3 Linear decay

Linear decay 就是线性下降。

如果总共有 $T$ 个 epoch，那么学习率可以从初始值 $\eta_0$ 一路线性降到 0：

$$
\eta_t=\eta_0\left(1-\frac{t}{T}\right)
$$

这种方法的好处是简单、直观。

### 2.9.4 Inverse sqrt decay

Inverse sqrt decay 是 Transformer 相关模型里常见的一类学习率策略。

它大致让学习率和训练步数的平方根成反比：

$$
\eta_t \propto \frac{1}{\sqrt{t}}
$$

也就是说，训练越往后，学习率越小，但不会像线性下降那样很快降到 0。

---

## 2.10 Linear Warmup

它的意思是：训练刚开始时，不要立刻使用很大的学习率，而是先从 0 开始，慢慢把学习率升到目标值。

比如前 5000 次 iteration：

$$
\eta_t=\eta_{max}\frac{t}{5000}
$$

等 warmup 结束后，再按照正常的学习率 schedule 下降。

!!! explanation "为什么需要 warmup？"
    训练刚开始时，模型参数还很乱，梯度也可能不稳定。
    
    如果一上来就用很大的学习率，很容易一步迈太大，导致 loss explode。
    
    warmup 相当于先小步试探一下，让训练稳定起来，再逐渐加速。

一个经验规则：

> 如果 batch size 增大 $N$ 倍，那么初始 learning rate 也可以增大 $N$ 倍。

这背后的直觉是：batch size 越大，梯度估计越稳定，所以可以承受更大的学习率。

---

## 2.11 First-Order Optimization 和 Second-Order Optimization

### 2.11.1 First-Order Optimization：一阶优化

前面讲的 SGD、Momentum、RMSProp、Adam 基本都属于一阶优化。

一阶优化只用梯度信息：

$$
\nabla L(W)
$$

也就是说，它只知道当前位置的斜率，然后用一条直线近似当前位置附近的 loss：

$$
L(W+\Delta W)\approx L(W)+\nabla L(W)^T\Delta W
$$

然后沿着让这个近似函数下降的方向走。

即：

> 我只知道脚下的坡往哪边斜，所以我就往下坡方向走。

### 2.11.2 Second-Order Optimization：二阶优化

二阶优化不只看梯度，还看 Hessian 矩阵。

Hessian 描述的是曲率，也就是函数弯曲的程度：

$$
H=\nabla^2 L(W)
$$

二阶 Taylor 展开大概是：

$$
L(W+\Delta W)\approx L(W)+\nabla L(W)^T\Delta W+\frac{1}{2}\Delta W^TH\Delta W
$$

也就是说，一阶方法只用直线近似，二阶方法会用一个二次函数去近似。

Newton update 的形式是：

$$
W_{new}=W-H^{-1}\nabla L(W)
$$

### 2.11.3 为什么深度学习里不常用完整二阶优化？

问题在于 Hessian 太大了。

如果模型有 $N$ 个参数，那么 Hessian 是一个：

$$
N\times N
$$

的矩阵。

所以它有：

$$
O(N^2)
$$

个元素。

而求逆的复杂度大概是：

$$
O(N^3)
$$

现在的神经网络参数量动不动就是几千万、几亿。这个时候完整 Hessian 根本存不下，更别说求逆了。

所以深度学习里大部分时候还是用一阶优化方法。

!!! note "实践总结"
    
    - Adam 或 AdamW 是很多情况下不错的默认选择，即使用固定 learning rate 也经常能工作。
    - SGD + Momentum 有时候能超过 Adam，但是通常需要更仔细地调 learning rate 和 schedule。
    - 如果真的能做 full batch update，而且目标函数比较确定，可以考虑一阶以外的方法，比如二阶优化。
