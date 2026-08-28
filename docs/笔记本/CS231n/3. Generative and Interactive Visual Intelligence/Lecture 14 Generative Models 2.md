![](附件/Pasted%20image%2020260822152700.png)
上节课我们讲了 Explicit density，这节课将讨论 Implicit density

# 1. Generative Adversarial Networks (GANs)

假设真实世界里的图片来自某个真实分布：

$$x\sim p_{\text{data}}(x)$$

我们虽然有很多训练图片，但我们并不知道那个真正的 $p_{\text{data}}(x)$ 长什么样。

GAN 的目标就是训练一个 $P_{G}(x)$，使得：

$$\boxed{p_G(x)\approx p_{\text{data}}(x)}$$

这里 $p_G$ 是生成器产生的数据所形成的分布。

!!! warning "我的疑惑"
    ### 我的疑惑1
    这个 $p(x)$ 到底是什么东西，为什么会算不出来？
    ### 解析：
    这里的 $p(x)$ 指的是真实数据分布在某个具体图片 $x$ 处的概率密度。

    问题是，我们根本没有这个分布的公式。我们只有一堆训练样本：

    $$x_1,x_2,\dots,x_N$$

    我们并不知道一个可以让你丢进去任意一张图片，然后直接告诉你这张图片有多像真实世界中的人脸的显式函数：

    $$p_{\text{data}}(x)=?$$

    所以 $p(x)$ 算不出来的原因是：我们只有这个分布产生出来的样本，但不知道分布本身的解析形式
    
    ### 我的疑惑2
    G 不是用来生成图片的吗？这跟训练 $p_G(x)$ 使它约等于 $p(x)$ 有任何关联吗？
    ### 解析：
    $$z\sim p(z),\ x_{fake}=G(z)$$
    
    这些生成的 $x_{fake}$ 就服从 Generator distribution，而我们的目标是：
    
    $$p_G=p_{data}$$
    
    举个例子，假设 z 只可能取两个值，且它俩的概率都是 0.5：

    $$p(z_1)=\;p(z_2)=0.5$$



    如果当前 Generator 是：$G(z_1)=\text{猫}$，$G(z_2)=\text{猫}$。那么你不断采样 z，最后生成结果就是：

    $$100\%\text{ 猫}$$
    
    此时，$p_G$ 大概就是全部概率集中在猫那里。

    现在训练 G，参数变了，变成：

     $$G(z_1)=\text{猫} ,\ G(z_2)=\text{狗}$$

    那么生成结果的整体分布立刻变成：

    $$ 50\%\text{ 猫},\quad 50\%\text{ 狗}$$

    也就是说 $G\text{ 一变}$，$p_G\text{ 也会跟着变}$

    实际训练的是：$G_\theta$，随着参数 $\theta$ 改变，$G_\theta(z)$ 生成的图片发生变化，于是由这些生成图片诱导出来的分布 $p_G(x)$ 也发生变化。
    
    因此，GAN 让 $p_G\approx p_{\text{data}}$ 意思是把 G 训练到一个状态，使得它生成出来的样本整体看起来和真实数据来自同一个分布。
## 1.1 Generator

GAN 首先定义一个简单的 latent distribution：

$$z\sim p(z)$$

一般可以设成：

$$p(z)=\mathcal N(0,I)$$

然后让一个神经网络 $G(z)$ 把这个随机向量 $z$ 变成一张图片：

$$x_{\text{fake}}=G(z)$$

所以整个流程就是：

$$z∼N(0,I)\rightarrow G(z)\rightarrow image$$

因此，生成器的任务是把从 $p(z)$ 里抽出来的 latent vector 转换成生成数据，而这些生成数据共同形成 $p_G$。

## 1.2 Discriminator

我们没办法让 $G(z)$ 和某一张训练图片做 MSE，因为 $z$ 根本没有对应的 ground- truth 图片。因此没法像 supervised learning 那用标签来训练。那么我们该如何训练 Generator 呢？

为此，引入了 Discriminator（判别器）：

给判别器输入一张图片，判别器输出一个数：

$$D(x)\in [0,1]$$

若 $D(x)=1$，这张图为 real；若 $D(x)=0$，这张图为 Generator 造出来的。

![](附件/Pasted%20image%2020260826181836.png)

## 1.3 Adversarial
为什么叫 Adversarial？

因为这两个网络的目标是相反的：

- Discriminator 希望 $D(x_{real})\rightarrow1$，$D(G(z))\rightarrow0$
- Generator 希望 $D(G(z)\rightarrow 0$
    - 虽然 Generator 生成的图片是假的，但是想要骗 D，让 D 认为它是真的。

因此 Discriminator 和 Generator 是在互相对抗的，所以叫 Adversarial。

## 1.4 GAN 的 Training object

$$
\min_{G}\max_{D}\left[
\mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big]
+
\mathbb{E}_{z\sim p(z)}\big[\log\big(1-D(G(z))\big)\big]
\right]
$$

!!! warning "我的疑问"
    ### 我的疑问
    为什么这里面一个是 $log\ D(x)$ 一个是 $log(1-D(G(z)))$
    ### 解答：
    
    对于真实样本，我们希望:
    
    $$\begin{aligned}
    &D(x)\rightarrow 1
    \\ &D(x) 越大，logD(x) 越接近于0，
    \\&这是单调递增的
    \end{aligned}$$
    
    对于 G 生成的样本，我们希望：
    
	 $$D(G(z))\rightarrow0$$
	 
    此时为了让它跟 $logD(x)$ 的增长方向相同，我们就将它变成了 $log(1-D(G(z)))$
### 1.4.1 固定 G，只看 D

先看第一项：

$$log\ D(x)$$

这里的 $x\sim p_{data}$ 是真实的图片，Discriminator 希望它大一点，最理想的情况就是 $D(x)=1$

再看第二项：

$$log\ (1-D(G(z)))$$

这里的 $G(z)$ 是 generator 生成的假的图片，Discriminator 希望它小一点，最理想的情况就是 $D(G(z))=0$ 

### 1.4.2 固定 D，只看 G

G 能控制的只有 $G(z)$，第一项 $log \ D(x)$ 跟 $G$ 完全没关系 ，$G$ 只跟 $\min_{G}{E}_{z\sim p(z)}\big[\log\big(1-D(G(z))\big)\big]$ 有关。

由于 $G$ 想骗过 $D$，让 $D$ 以为 $G$ 生成的图片也是真实的图片，因此这里希望 $G(z)\rightarrow 1$

$G$ 希望 $G(z) \rightarrow 1$，$D 希望 G(z) \rightarrow 0$，这里就体现出了 Adversarial

### 1.4.3 更新方式

因为 D 需要 maximizes，G 需要 minimize，所以 D 是 gradient ascent，G 是 minimize：

$$\begin{aligned}
&D=D+α_D\frac{​∂V}{∂D}
\\& G=G-α_G\frac{​∂V}{∂G}​​
\end{aligned}$$


### 1.4.4 $\min_{G}\max_{D}\left[\mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big]+\mathbb{E}_{z\sim p(z)}\big[\log\big(1-D(G(z))\big)\big]\right]$ 的问题

原本 Generator 最小化：

$$\log(1-D(G(z)))$$

但是训练刚开始时，G 很差，所以 D 很容易发现假图：

$$G(z)\approx0$$

但是这里有一个问题：

$$
\begin{aligned}
&如果使用 \log(1-D(G(z))) 的话，刚开始训练时\ Generator\ gradient\ 接近于0，
\\&这被称为\ gradient\ saturation
\end{aligned}$$

所以实际常把 Generator loss 改成：

$$\boxed{ L_G=-\log D(G(z)) }$$

而不是：

$$\log(1-D(G(z)))$$

这两个目标都希望：

$$D(G(z))\rightarrow1$$

但前者在训练初期 gradient 更强。

---