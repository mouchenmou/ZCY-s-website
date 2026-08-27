![](附件/Pasted%20image%2020260822152700.png)
上节课我们讲了 Explicit density，这节课将讨论 Implicit density

# 1. Generative Adversarial Networks (GANs)

假设真实世界里的图片来自某个真实分布：

$$x\sim p_{\text{data}}(x)$$

我们虽然有很多训练图片，但我们并不知道那个真正的 $p_{\text{data}}(x)$ 长什么样。

GAN 的目标就是训练一个 $P_{G}(x)$，使得：

$$\boxed{p_G(x)\approx p_{\text{data}}(x)}$$

这里 $p_G$ 是生成器产生的数据所形成的分布。

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

### 1.4.1 固定 G，只看 D

先看第一项：

$$log\ D(x)$$

这里的 $x\sim p_{data}$ 是真实的图片，Discriminator 希望它大一点，最理想的情况就是 $D(x)=1$

再看第二项：

$$log\ (1-D(G(z)))$$

这里的 $G(z)$ 是 generator 生成的假的图片，Discriminator 希望它小一点，最理想的情况就是 $D(G(z))=0$ 

### 1.4.2 固定 D，只看 G


---