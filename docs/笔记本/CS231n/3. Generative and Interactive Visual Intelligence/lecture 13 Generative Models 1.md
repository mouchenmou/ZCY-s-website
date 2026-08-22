# 1. Supervised vs Unsupervised Learning
![](附件/Pasted%20image%2020260819213829.png)

---
# 2. Generative vs Discriminative Models
## 2.1 Discriminative Model
这就是前面的课一直在学的监督学习。比如：给定一张图，不同的 label 存在竞争关系，也就是：

$$\sum_{y}p(y|x) = 1$$

每一张图片输入之后，都会输出各个类别的概率，不同的图片之间互相不影响。因此，**不同的图片之间不存在竞争关系。**

但是，判别式模型有一个很重要的问题，比如说我们的 class 里面只有猫和狗两个类别，那么世界上所有的图片输入进去之后，只会输出这张图片是猫的概率和狗的概率，而且它们俩加起来等于1。万一我们输入了一张猴子的图片，模型也只会输出这张图是猫的概率和狗的概率。

## 2.2 Generative Model
生成式模型学习的时候是**没有标签 y 的，它直接学 $p(x)$。**

与判别式模型不同，**生成式模型的所有图片一起竞争 probability mass !!!**

比如我们有四张图：

- $x_1 = cat$
- $x_2=dog$
- $x_3=monkey$
- $x_4=abstract\ noise$

$$那么:\ p(x_1)+p(x_2)+p(x_3)+p(x_4)=1$$

!!! warning "我的疑惑"
    ### 我的疑惑
    连标签都没有，这个概率到底是个什么东西？
    ### 解析：
    $p(x)$ 的定义：$p(x)$ 给每一个可能的数据 $x$ 分配一个数，数越大表示这个 $x$ 在模型所学习的数据分布下有多可能出现。
    
    它不是 $p(cat|x)$ ，不是在问这张图片是猫的概率是多少，而是在问**这样的像素组合在训练数据所代表的图片世界里，是常见的还是罕见的。**
    
    假设有一张正常的狗图片(令它为 $A$)和一张噪声图片(令它为 $B$)，生成式模型可能会给出 $p(dog)>p(x_{noise})$，它的意思不是图 A 的类别概率比 B 更高，而是 A 比 B 更符合模型学到的数据分布。

## 2.3 Conditional Generative Model

有条件的生成式模型学的是 $p(x|y)$，即在给定条件 $y$ 下，哪些 $x$ 比较合理

例如： $y=cat$，图片 A 是一张猫图，图片 B 是一张狗图，那么：

$$p(X_A|y)>p(X_B|y)$$

如果换成 $y=dog$，就反过来了。

!!! warning "Conditional Generative Model vs Discriminative Model"
    Conditional Generative Model 和 Discriminative Model 正好反一下：
    
    - Discriminative Model 是给定一张图片(即一个 $x$)，然后预测每个 $y$ 的概率(每个 $y$ 都是这个模型里面的类别)。
    - Conditional Generative Model 是给定一个类别 $y$，然后预测各个 $x$ 在条件 $y$ 下的概率。

---

# 3. Taxonomy of Generative Models
Generative Models 可以分成两大类：Explicit density 和 Implicit density：

1. Explicit density：模型可以直接计算 $p(x)$
    - 比如，$x=一张照片$。那么将 $x$ 输入到模型中能够输出 $p(x)$
2. Implicit density：模型不能直接计算 $p(x)$，但是可以按照学到的分布生成样本。
    - 比如，你让这个模型画一只猫，模型能画出来，这就是 sample from $p(x)$。
    - 你给它一张猫的图片，问它这张图片出现的概率是多少，模型回答不出来。

![](附件/Pasted%20image%2020260820145002.png)


## 3.1 Autoregressive Models

### 3.1.1 Maximum Likelihood Estimation

我们需要找到一个 explicit function：$p(x) = f(x,W)$

给定一个训练集：

$$x_1,x_2,…,x_N$$

我们希望训练之后，这些真实的训练数据出现的概率都很大，所以希望 $p_{\theta}(x_1),p_{\theta}(x_2),…,p_{\theta}(x_N)$ 都尽可能的大。因此我们用到了**极大似然估计。**

!!! explanation "极大似然估计"
    这里的极大似然估计就是概率论与数理统计中学过的那个。
    
    假设这 $N$ 个样本互相独立，那么整批训练数据同时出现的概率是：
    
	$$L(W)=\prod_{i=1}^N p_W(x_i)$$
    
    这个 $L(W)$ 就叫似然，极大似然估计就是：
    
    $$W^{*}=\arg\max_{W}\prod_{i=1}^{N} p_{W}(x_i)$$
    
    由于一大堆概率相乘算起来很坐牢，所以我们选择**取对数**：
    
    $$W^{*}=\arg\max_{W}\sum_{i=1}^{N} log\ p_{W}(x_i)$$
    
    由于 $log$ 函数是单调递增的，所以上面那两个 $W^*$ 是一样的。

### 3.1.2 How to compute the $p(x)$

刚刚讲了极大似然估计，但是还不知道如何计算 $p(x)$，我们可以根据概率的链式法则来计算 $p(x)$：

假设序列是

$$x=(x_1​,x_2​,…,xT_​)$$

比如：

$$x_1=I,\ x_2​=love,\ x_3​=黄杰曦$$

那整句话的概率即为 $p(x_1​,x_2​,x_3​)$

根据概率的 chain rule，可以拆成：

$$p(x_1​,x_2​,x_3​)=p(x_1​)p(x_2​∣x_1​)p(x_3​∣x_1​,x_2​)$$

推广到一般形式即为：

所以一般形式就是：

$$p(x_1,x_2,...,x_T)=\prod^T_{t=1}p(x_t|x_1,x_2,...x_{t-1})$$

### 3.1.3 如何实现 Autoregressive Model

RNN 和 Transformer 都能够实现 Autoregression Model。

!!! explanation "使用RNN"
    RNN 会把前面的信息不断地压缩进 hidden state。
    
    先给 RNN 一个初始 hidden state $h_0$，通常还会有一个特殊的开始符号`<BOS>`。
    
    第一步：预测 $x_1$：
    
    $$h_0+BOS\rightarrow p(x_1)$$
    
    第二步：把 $x_1$ 输入 RNN，让 RNN更新 hidden state：
    
    $$h_1=RNN(h_0,x_1)$$
    
    用 $h_1$ 来预测 $p(x_2|x_1)$
    
    第三步：把 $x_2$ 输入 RNN：
    
    $$h_2=RNN(h_1, x_2)$$

    用 $h_2$ 来预测 $p(x_3|x_1,x_2)$
    
    以此类推，RNN 的整个执行过程就是：
    
    $$\begin{aligned}
    & BOS\rightarrow p(x_1)
    \\ &x_1,h_0\rightarrow h_1\rightarrow p(x_2|x_1)
    \\ &x_2,h_1 \rightarrow h_2\rightarrow p(x_3|x_1,x_2)
    \\ &x_3,h_2 \rightarrow h_3\rightarrow p(x_4|x_1,x_2,x_3)
    \end{aligned}$$


!!! explanation "使用 Transformer
    Transformer 不需要像 RNN 那样把信息一步步塞进 hidden state。若要计算 $p(x_3|x_1,x_2)$，Transformer 会直接把 $x_1,x_2$ 都拿来，通过 masked self- attention 计算出 $x_3$ 的概率：
    
    $$[x_1, x_2] \xrightarrow{\text{Transformer}} p(x_3 \mid x_1, x_2)$$

    同理，计算 $p(x_4|x_1,x_2,x_3)$ 即为：
    
    $$[x_1, x_2,x_3] \xrightarrow{\text{Transformer}} p(x_4 \mid x_1, x_2,x_3)$$

    由于此处用的是 masked self- attention，所以能保证模型看不到未来的 token。

## 3.2 Variational Autoencoders (VAEs) 

PixelRNN / PixelCNN 这种 autoregressive model，可以把 $p(x)$ 明确地写出来。比如：

$$p_W​(x)=\prod^T_{t=1} ​p_W​(x_t​∣x_1​,x_2,…,x_{t−1}​)$$

于是就可以做极大似然估计。

然而，VAE define an **intractable density(难以处理的密度)** that we cannot explicitly compute or optimize.

它的意思是，$p(x)$ 理论上存在，但是直接计算起来太难了。



!!! warning "为什么 VAE 的 $p(x)$ 难计算？"
    VAE 里面会引入一个 latent variable：
    
    $$z$$
    
    可以把 $z$ 理解成一个隐藏层，即：
    
    $$z\rightarrow x$$
    
    于是 VAE 里的 $p(x) 需要把所有可能的 $z$ 都考虑进去：

    $$p(x)=∫p(x,z)dz$$

    也可以写成：

    $$p(x)=∫p(x|z)p(z)dz​$$
    
    如果 $z$ 是一个连续高维向量，比如 $z \in R^{128}$。那么会非常非常难算。

因此我们不直接最大化 $log\ p(x)$，而是最大化它的一个 lower bound，即：

$$lower\ bound\le log\ p(x)$$
### 3.2.1 (Non-Variational) Autoencoders

在讲 Variational Autoencoders 之前，先来看一下普通的 Autoencoders。

普通的 Autoencoders 可以看成一种 self-supervised  learning。因为它的训练的时候：

$$x \xrightarrow{\text{Encoder}} z\xrightarrow{\text{Decoder}}\hat{x} $$

然后拿 $\hat{x}$ 和原始输入 $x$ 来计算 loss：

$$loss = ||\hat{x}-x||^2$$

这里不需要人工标签，因为它的 target 就是原始数据本身。

当 Autoencoders 训练完之后，丢掉 Decoder，接上我们真正要用的 classifier 或者别的东西

### 3.2.2 Variational Autoencoder

先理解 Variational 是什么：

Variational 在这里不是“变化的”，“随机的”那个意思。在 Variational Autoencoder 里，它指的是**用变分法来做近似推断**：

我们刚刚讲了 VAE 里面有一个很难直接计算的东西 $P(z|x)$，由于它的真实分布太难计算了，我们另外定义一个比较简单的分布 $q(z|x)$，我们调整这个 $q$ 让它尽量接近真正的 $p(z|x)$。即 $q(z∣x)≈p(z∣x)$。

不断调整一个候选分布 $q$，去逼近目标分布的这套数学方法，就叫 **variational inference**，所以 Variational Autoencoder 中 Variational 的意思就是：**用variational inference 来近似难算的后验分布**

!!! explanation "贝叶斯公式"
    这里我们需要用到贝叶斯公式：

    $$p_{\theta}(z|x)=\frac{p_{\theta}(x|z)p(z)}{p_{\theta}(x)}$$
    
    这个式子是这么来的：
    
    $$p(x,z)=p(z)p(x|z)$$
    
    同时也可以反过来写，没区别的：
    
    $$p(z,x)=p(x)p(z|x)$$
    
    因此可以得出 $p(x)p(z|x)=p(z)p(x|z)$，将 $p(x)$ 除到右边，就得到了：
    
    $$p_{\theta}(z|x)=\frac{p_{\theta}(x|z)p(z)}{p_{\theta}(x)}$$
    
我们将贝叶斯公式稍微改动一下，改成：

$$p_{\theta}(x)=\frac{p_{\theta}(x|z)p(z)}{p_{\theta}(z|x)}$$

我们前面已经说了 $p_{\theta}(z|x)$ 很难求出来，因此需要训练另一个网络，构造一个：

$$q_ϕ​(z∣x)$$

让它们两个近似相等：

$$q_ϕ​(z|x)≈p_{\theta}(z|x)$$

![](附件/Pasted%20image%2020260821094318.png)

!!! warning "我的疑惑"
    ### 我的疑惑1
    为什么 $p_{\theta}(x|z)和p_{\theta}(z)$ 好算，而 $p_{\theta}(z|x)$ 难算？
    ### 解答
    $p(z)$ 好算是因为我们自己规定了 $p(z)$ 的分布，通常设成 $p(z)=N(0,I)$。因此，给定一个 $z$，我们就可以通过高斯分布计算出来。
    
    $p_{\theta}(x|z)$ 好算，是因为它是 Decoder 明确建模的分布。比如，上面那幅图中就规定了 $p_{\theta}(x|z)=N(u_{x|z},σ^2)$。
    
    因此 $p_{\theta}(x|z)和p_{\theta}(z)$ 都是我们已经规定好分布的，当然能计算。
    
    $p_{\theta}(z|x)$ 难算，是因为根据 Bayes 公式：
    
    $$p_{\theta}(z|x)=\frac{p_{\theta}(x|z)p(z)}{p_{\theta}(x)}$$
    
    其中：
    
    $$p_θ​(x)=∫p_θ​(x|z)p(z)dz$$
    
    需要对所有可能的 $z$ 做高维度积分，所以 $p_{\theta}(x)$ 很难计算。又因为 $p_{\theta}(x)$ 是 $p_{\theta}(z|x)$ 的分母，所以 p_{\theta}(z|x) 不好计算。
    
    因此，我们才用 $q_ϕ​(z|x)$ 来估计 $p_{\theta}(z|x)$
    
    ---
    
    ### 我的疑惑2
    为什么 Variational Autoencoder 需要这么麻烦的计算，而普通的 Autoencoder（ Non-Variational Autoencoder）却不需要那么麻烦，普通的那个不也是 encoder 生成一个类似 z 的东西交给 decoder 吗 ？
    ### 解答：
    ![](附件/Pasted%20image%2020260821095956.png)


### 3.2.3 $D_{kl}$

在讲 3.2.4 Variational Autoencoders (ELBO) 之前，需要进入一个概念： $D_{kl}$

**KL divergence（KL 散度）是用来衡量两个概率分布有多不一样的。**

比如我们有两个关于 z 的分布：$q(z)\ 和\ p(z)$。我们想知道它们俩的差别，这时候就能用到：

$$D_{KL}​(q∥p)$$

它的具体定义如下：

$$D_{KL}​(q∥p)=\sum q(z)\ log\frac{q(z)}{p(z)}=E_{z∼q}​[log\frac{q(z)}{p(z)}​]​$$

上面那个是离散的，如果连续的话，就是：

$$D_{KL}​(q∥p)=∫q(z)\ log\frac{q(z)}{p(z)}​dz=E_{z∼q}​[log\frac{q(z)}{p(z)}​]​$$

意思是：按照 $q(z)$ 这个分布去看各种可能的 $z$，对每个 $z$ 比较 $q(z)$ 和 $p(z)$ 的大小，然后**把这些比较结果全都加起来之后再取平均**

!!! explanation "$D_{KL}​(q∥p)\ge 0$"
    对于任意$D_{KL}$，都有 $D_{KL}​(q∥p)\ge 0$。
    
    证明：
    
    我们以离散的情况为例：
    
    $$
    \begin{aligned}
    \\&D_{KL}​(q∥p)=\sum q(z)\ log\frac{q(z)}{p(z)}=-\sum q(z)\ log\frac{p(z)}{q(z)}
    \\&\because\ log\ x \le x-1
    \\&\therefore\ -log\ x \ge 1-x
    \end{aligned}$$
    
    令 $x= \frac{p(z)}{q(z)}$，可得：
    
    $$-log \frac{p(z)}{q(z)}\ge1-\frac{p(z)}{q(z)}$$
    
    左右两边乘上 $q(z)$ 得：
    
    $$-q(z)\ log\frac{p(z)}{q(z)}\ge q(z)-p(z)$$
    对所有 $z$ 求和，得到：
    
    $$D_{KL}(p||q)\ge \sum_zq(z)-\sum_zp(z)$$
    
     $$\begin{aligned}
     &\because\sum_z q(z)=\sum_z p(z)=1
     \\& \therefore D_{KL}(p||q) \ge1-1=0
     \end{aligned}$$

### 3.2.4 Variational Autoencoders (ELBO)



$$\begin{aligned}
&log\ 𝑝_𝜃(𝑥) = log\frac{\ 𝑝_𝜃(𝑥|𝑧)𝑝(𝑧)}{𝑝_𝜃(𝑧| 𝑥)} =log\frac{\ 𝑝_𝜃(𝑥|𝑧)𝑝(𝑧)q_ϕ​(z|x)}{𝑝_𝜃(𝑧| 𝑥)q_ϕ​(z|x)} 
\\ &\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =log\ p_θ​(x|z)−log\ \frac{p(z)}{q_ϕ​(z|x)}​+log\ \frac{p_θ​(z|x)}{q_ϕ​(z|x)​}
\end{aligned}$$

**我们对左右两边同时求 $z$ 的平均**：

- 等号左边：$log\ 𝑝_𝜃(𝑥)$ 中没有任何跟 $z$ 有关的数，因此左边求完平均之后仍然是 $log\ 𝑝_𝜃(𝑥)$
- 等号右边：$E_z​[log\ p_θ​(x|z)]−E_z​[\frac{q_ϕ​(z|x)​}{log\ p(z)}]+E_z​[\frac{q_ϕ​(z|x)​}{log\ p_θ​(z|x)}]$

左右两边对 $z$ 求平均之后，等式变为：

$$
\begin{align*}
\log p_\theta(x)
&= \mathbb{E}_{z}\left[\log p_\theta(x|z)\right]
- \mathbb{E}_{z}\left[\log \frac{q_\phi(z|x)}{p(z)}\right]
+ \mathbb{E}_{z}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right] \\
&= \mathbb{E}_{z \sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right]
- D_{\text{KL}}\big(q_\phi(z|x),\, p(z)\big)
+ \textcolor{red}{D_{\text{KL}}\big(q_\phi(z|x),\, p_\theta(z|x)\big)}
\end{align*}$$

因为 ${D_{\text{KL}}\big(q_\phi(z|x),\, p_\theta(z|x)\big)}\ge0$，所以可以得出下面这个结论：

$$
\log p_\theta(x)
\ge
\mathbb{E}_{z \sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big]
- D_{KL}\big(q_\phi(z|x),\, p(z)\big)
$$

### 3.2.5 Variational Autoencoders: Training

3.2.4 中已经得出了 $\log p_\theta(x) \ge\mathbb{E}_{z \sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{KL}\big(q_\phi(z|x),\, p(z)\big)$

那么我们训练的时候，就要最大化这个下界 $\mathbb{E}_{z \sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{KL}\big(q_\phi(z|x),\, p(z)\big)$

![](附件/Pasted%20image%2020260822003658.png)

普通 Autoencoder ：输入图片 x，Encoder 直接输出一个确定的 z。

但 VAE 不是直接输出一个确定的 $z$，而是输出 $u_{z|x}$ 和 $σ_{z|x}$

用这两个数来定义一个高斯分布：

$$q_ϕ​(z|x)=N(μ_{z|x},σ^2_{z|x})$$
因此，此处的 $z$ 并不是普通的 Autoencoder 那样输出的一个固定值，而是说这个 $z$ 大致落在 $q_ϕ​(z|x)$ 这个高斯分布里。

!!! explanation "两个 loss"
    $\mathbb{E}_{z \sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{KL}\big(q_\phi(z|x),\, p(z)\big)$ 中有两项：
    
    - 第一项；根据 $z$ 重建出来的 $\hat{x}$ 要尽量接近原来的 $x$，所以它叫 reconstruction.
    - 第二项：$q_ϕ​(z|x)$ 和 $p(z)$ 差距越小，$D_{KL}\big(q_\phi(z|x),\, p(z)\big)$ 就越小，确定的下界就越大。因此我们希望 $q_ϕ​(z|x)$ 尽可能接近 $p(z)$ 
    
    这两个 loss 会打架：
    
    - 对于第一项 Reconstruction loss：我们希望对于一张图片 $x$，最好不要随机，直接给一个确定的 $z$，使得 Decoder 更容易重建。因此 reconstruction 希望 $σ_{z|x}=0$。由于每张图片都有自己独特的 $z$，因此我们 希望每张图片的 $u_{z|x}$ 都不一样
    - 对于第二项 Prior loss：我们希望 $q_ϕ​(z|x)$ 尽可能的接近提前规定好的 prior：$p(z)=N(0,1)$。因此希望 $σ_{z|x}=1$，$u_{z|x}=0$
    
    因此这两个 loss 会相互拉扯，VAE 最终就是在这两个目标之间找到一个平衡。