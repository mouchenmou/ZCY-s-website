# 1. Wake Word Detector

这节课一开始用语音助手的 wake word detection 作为例子。

Wake word 就是能触发语音助手开始认真听的词，比如：

$$
\text{Hey Siri}, \quad \text{Alexa}, \quad \text{OK Google}
$$

语音助手平时并不是完全不工作，而是一直在每一小段声音上跑一个很小的 detector，判断这一秒声音里面有没有 wake word。

这个问题可以理解成：

$$
\text{每一段声音} \rightarrow \text{是否包含 wake word}
$$

所以它是一个 supervised binary classification problem。

## 1.1 这个问题里的随机变量

对于每一段声音，比如每 $1$ 秒，我们可以定义三个随机变量：

$$
Z \in \{0,1\}, \quad X \in \mathbb{R}^D, \quad Y \in \{0,1\}
$$

其中：

1. $Z$：真实世界里有没有人说 wake word。
    - $Z=1$：真的说了。
    - $Z=0$：没有说。
2. $X$：麦克风收集到的声音特征向量。
3. $Y$：detector 有没有触发。
    - $Y=1$：detector 认为有 wake word。
    - $Y=0$：detector 认为没有。

我们要关注的是 $Z$ 和 $Y$ 之间的关系，也就是：

$$
p(Z,Y)
$$

根据 product rule，可以写成：

$$
p(Z,Y)=p(Z)p(Y\mid Z)
$$

这里：

1. $p(Z)$ 是 prior，也就是 wake word 本身出现的概率。
2. $p(Y\mid Z)$ 是 likelihood，也就是在真实情况给定之后，detector 做出某种判断的概率。

## 1.2 Detection Rate 和 False Alarm Rate

Wake word detector 主要有两个重要指标：

### 1.2.1 Detection Rate

Detection rate 也叫 True Positive Rate 或者 Recall：

$$
p(Y=1\mid Z=1)
$$

它表示：如果真的有人说了 wake word，那么 detector 能不能把它检测出来。

### 1.2.2 False Alarm Rate

False alarm rate 也叫 False Positive Rate：

$$
p(Y=1\mid Z=0)
$$

它表示：如果没有人说 wake word，detector 会不会误触发。


---

# 2. Joint Probability Distribution

假设 $X$ 和 $Y$ 是离散随机变量：

$$
X \in \{x_1,\dots,x_L\}, \quad Y \in \{y_1,\dots,y_M\}
$$

那么 joint probability distribution 是：

$$
p(X=x_i,Y=y_j)
$$

它表示两个事件同时发生的概率。

---

# 3. Marginalization

Marginalization 也叫 sum rule：把 joint distribution 里面某个变量 sum out 掉，得到只关于另一个变量的分布。

比如从 $p(X,Y)$ 得到 $p(X)$：

$$
p(X=x_i)=\sum_{j=1}^{M}p(X=x_i,Y=y_j)
$$

!!! example "举个例子"
    假设我们有一张 joint probability table：

    $$
    \begin{array}{c|cc}
    &Y=0&Y=1\\
    \hline
    X=0&0.2&0.1\\
    X=1&0.25&0.45
    \end{array}
    $$

    那么：

    $$
    p(X=0)=0.2+0.1=0.3
    $$

    $$
    p(X=1)=0.25+0.45=0.7
    $$

    同理也可以把 $X$ sum out，得到 $p(Y)$。

---

# 4. Conditional Probability

Conditional probability 表示在一个事件已经发生的前提下，另一个事件发生的概率。

定义是：

$$
p(Y=y_j\mid X=x_i)=\frac{p(X=x_i,Y=y_j)}{p(X=x_i)}
$$

---

# 5. Product Rule

从 conditional probability 的定义出发：

$$
p(Y=y_j\mid X=x_i)=\frac{p(X=x_i,Y=y_j)}{p(X=x_i)}
$$

两边乘上 $p(X=x_i)$，可以得到 product rule：

$$
p(X=x_i,Y=y_j)=p(Y=y_j\mid X=x_i)p(X=x_i)
$$

Product rule 也可以理解成 probability 的 chain rule。

比如三个变量的时候：

$$
p(X,Y,Z)=p(Z\mid X,Y)p(X\mid Y)p(Y)
$$


---

# 6. Bayes' Theorem

Bayes' theorem 可以把 conditional probability 的方向反过来。

$$
p(A\mid B)=\frac{p(B\mid A)p(A)}{p(B)}
$$

其中分母可以通过 marginalization 展开：

$$
p(B)=\sum_{A'}p(B\mid A')p(A')
$$

所以也可以写成：

$$
p(A\mid B)=
\frac{p(B\mid A)p(A)}
{\sum_{A'}p(B\mid A')p(A')}
$$


---

# 7. Independence 和 IID

## 7.1 Independence

如果两个随机变量 $X$ 和 $Y$ independent，那么它们的 joint distribution 可以拆成：

$$
X\perp Y \quad \Longrightarrow \quad p(X,Y)=p(X)p(Y)
$$

## 7.2 IID

IID 是 Independent and Identically Distributed。

对于数据：

$$
(X_1,Y_1), (X_2,Y_2), \dots, (X_N,Y_N)
$$

IID assumption 表示：

1. Independent：每个样本之间相互独立。
2. Identically Distributed：每个样本都来自同一个分布。

于是整个数据集的概率可以写成：

$$
p(X_1,Y_1,\dots,X_N,Y_N)=\prod_{n=1}^{N}p(X_n,Y_n)
$$

!!! explanation "IID 中 Independent 和 Identically Distributed 到底是什么意思"

	假设数据集是：
	
	$$(X_1,Y_1),(X_2,Y_2),\dots,(X_N,Y_N)$$
	
	这里的 $(X_n,Y_n)$ 表示**第 $n$ 个样本**。
	
	例如在猫狗分类中：
	
	- $X_n$：第 $n$ 张图片
	- $Y_n$：第 $n$ 张图片对应的标签
	    
	
	所以 $(X_n,Y_n)$ 是一个完整的数据样本。

---

# 8. Wake Word Detector 的 Bayes 分析

现在回到 wake word detector。

假设 wake word 很少出现：

$$
p(Z=1)=0.0001
$$

也就是 $0.01\%$ 的声音片段里真的有 wake word。

于是：

$$
p(Z=0)=0.9999
$$

假设 detector 很强：

$$
p(Y=1\mid Z=1)=0.99
$$

也就是真的说了 wake word 时，它有 $99\%$ 的概率能检测出来。

同时 false positive rate 很小：

$$
p(Y=1\mid Z=0)=0.001
$$

也就是没有 wake word 时，只有 $0.1\%$ 的概率误触发。

现在问题是：如果 detector 触发了，也就是 $Y=1$，那么真的说了 wake word 的概率是多少？

也就是求：

$$
p(Z=1\mid Y=1)
$$

根据 Bayes' theorem：

$$
p(Z=1\mid Y=1)
=
\frac{p(Y=1\mid Z=1)p(Z=1)}
{p(Y=1)}
$$

分母用 sum rule 展开：

$$
p(Y=1)=p(Y=1\mid Z=1)p(Z=1)+p(Y=1\mid Z=0)p(Z=0)
$$

所以：

$$
p(Z=1\mid Y=1)
=
\frac{0.99\cdot 0.0001}
{0.99\cdot 0.0001+0.001\cdot 0.9999}
\approx 0.09
$$

也就是：

$$
p(Z=1\mid Y=1)\approx 9\%
$$

!!! explanation "为什么 detector 看起来很准，但 precision 只有 9%"
    关键原因是 wake word 是 rare event。

    虽然 false positive rate 只有 $0.1\%$，但是没有 wake word 的片段太多了。

    只要背景样本的数量特别大，一个很小的误报率也会产生很多 false alarms。

## 8.1 用 counts 重新理解

想象一天里有 $1,000,000$ 个声音片段。

因为：

$$
p(Z=1)=0.0001
$$

所以真的有 wake word 的片段大约是：

$$
1,000,000 \times 0.0001 = 100
$$

没有 wake word 的片段大约是：

$$
999,900
$$

detector 能抓住 $99\%$ 的真实 wake word：

$$
100\times 0.99=99
$$

但是它会在 $0.1\%$ 的非 wake word 片段上误触发：

$$
999,900\times 0.001\approx 999
$$

所以 detector 一共触发：

$$
99+999=1098
$$

其中真正的 wake word 只有 $99$ 个：

$$
\frac{99}{1098}\approx 9\%
$$

这就是 base rate 的影响。

---

# 9. Expectation

Expectation 是 random variable 在某个概率分布下的加权平均。

对于离散随机变量 $X$：

$$
\mathbb{E}[X]=\sum_x x p(x)
$$

Conditional expectation 是在 conditional distribution 下做加权平均：

$$
\mathbb{E}[X\mid y]=\sum_x x p(x\mid y)
$$

## 9.1 Functions of Random Variables

如果：

$$
Y=f(X)
$$

那么 $Y$ 也是一个 random variable。

它的 expectation 是：

$$
\mathbb{E}[f(X)]=\sum_x f(x)p(x)
$$

如果函数依赖多个 random variables：

$$
f(X,Y)
$$

那么 expectation 要对 joint distribution 做：

$$
\mathbb{E}[f(X,Y)]=\sum_x\sum_y f(x,y)p(x,y)
$$

## 9.2 Linearity of Expectation

对于任意两个随机变量 $X,Y$：

$$
\mathbb{E}[aX+bY+c]=a\mathbb{E}[X]+b\mathbb{E}[Y]+c
$$

这个性质不需要 $X$ 和 $Y$ independent。

!!! explanation "为什么不需要 independent"
    因为 linearity of expectation 只是在把求和拆开：

    $$
    \sum_x\sum_y (ax+by+c)p(x,y)
    $$

    它不要求：

    $$
    p(x,y)=p(x)p(y)
    $$

    所以就算 $X$ 和 $Y$ highly correlated，这个性质也依然成立。

---

# 10. Variance 和 Covariance

## 10.1 Variance

Variance 描述一个 random variable 的 spread，也就是它围绕均值波动得有多大。

定义是：

$$
\operatorname{var}(X)=\mathbb{E}\left[(X-\mathbb{E}[X])^2\right]
$$

利用 expectation 的线性性质，可以化简成：

$$
\operatorname{var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2
$$

如果做线性变换：

$$
Y=aX+b
$$

那么：

$$
\operatorname{var}(aX+b)=a^2\operatorname{var}(X)
$$

## 10.2 Covariance

Covariance 描述两个 random variables 一起变化的程度。

定义是：

$$
\operatorname{cov}(X,Y)=
\mathbb{E}\left[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\right]
$$

也可以写成：

$$
\operatorname{cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]
$$

对于 random vectors $\mathbf{x}$ 和 $\mathbf{y}$，covariance 是矩阵：

$$
\operatorname{cov}(\mathbf{x},\mathbf{y})
=
\mathbb{E}\left[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{y}-\mathbb{E}[\mathbf{y}])^\top\right]
$$

也可以写成：

$$
\operatorname{cov}(\mathbf{x},\mathbf{y})
=
\mathbb{E}[\mathbf{x}\mathbf{y}^\top]-\mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{y}]^\top
$$

---

# 11. False Alarm 的 Expected Cost

假设每次 detector 触发之后，设备都要调用一个更大的 speech model。

这个大模型可能很贵，因为它会消耗：

1. energy：本地计算或者上传到 cloud。
2. money：在 datacenter 里面运行模型。

设每次调用大模型的 cost 是 $C$。

如果每一段声音是 $1$ 秒，那么一天大约有：

$$
24\times 60\times 60=86400
$$

个声音片段。

定义 indicator variable：

$$
I_t=
\begin{cases}
1,& \text{第 }t\text{ 个片段是 false alarm}\\
0,& \text{otherwise}
\end{cases}
$$

那么一天的 false alarm 总成本是：

$$
\text{Total Cost}=\sum_{t=1}^{86400} C I_t
$$

取 expectation：

$$
\mathbb{E}[\text{Total Cost}]
=
\mathbb{E}\left[\sum_{t=1}^{86400} C I_t\right]
$$

利用 linearity of expectation：

$$
\mathbb{E}[\text{Total Cost}]
=
C\sum_{t=1}^{86400}\mathbb{E}[I_t]
$$

因为 $I_t$ 是 indicator：

$$
\mathbb{E}[I_t]=p(Z=0,Y=1)
$$

而：

$$
p(Z=0,Y=1)=p(Y=1\mid Z=0)p(Z=0)
$$

代入前面的数字：

$$
p(Z=0,Y=1)=0.001\cdot 0.9999
$$

所以：

$$
\mathbb{E}[\text{Total Cost}]
=86400\cdot C\cdot 0.001\cdot 0.9999
\approx 86.4C
$$

---

# 12. Modeling Distributions

很多机器学习模型其实都在建模数据的 probability distribution。

比如 wake word detector 不只是输出：

$$
\text{yes or no}
$$

更好的方式是输出：

$$
p(\text{this sound is a wake word})
$$

也就是对不确定性进行建模。

常见的 distribution modeling 方法有：

1. Tabular representations。
2. Classic probability models。
3. Nonparametric models，比如 histograms 和 kernel density estimation。

## 12.1 Tabular Representation

对于离散变量，我们可以直接用表格表示 joint distribution。

例如：

$$
\begin{array}{c|cc}
&y_1&y_2\\
\hline
x_1&w_{1,1}&w_{1,2}\\
x_2&w_{2,1}&w_{2,2}\\
x_3&w_{3,1}&w_{3,2}
\end{array}
$$

这里 $w_{i,j}$ 就是：

$$
p_w(X=x_i,Y=y_j)=w_{i,j}
$$

为了让它成为合法的 probability model，需要满足：

$$
w_{i,j}\geq 0
$$

以及：

$$
\sum_{i=1}^{L}\sum_{j=1}^{M}w_{i,j}=1
$$

!!! warning "tabular representation 的问题"
    对离散变量来说，表格表示是 complete 的：它可以表达任何离散 joint distribution。

    但是它通常不 efficient。

    如果变量很多，每个变量又有很多取值，那么表格大小会爆炸。

    这就是为什么我们经常需要更有结构的 probability model。


## 12.2 Bernoulli Distribution

Bernoulli distribution 用来描述 binary random variable：

$$
X\in\{0,1\}
$$

记作：

$$
X\sim \operatorname{Bern}(\mu)
$$

它的 probability mass function 是：

$$
p(x\mid \mu)=\mu^x(1-\mu)^{1-x}
$$

其中：

$$
0\leq \mu \leq 1
$$

如果 $x=1$：

$$
p(X=1\mid \mu)=\mu
$$

如果 $x=0$：

$$
p(X=0\mid \mu)=1-\mu
$$

### 12.2.1 Bernoulli 的 expectation

$$
\mathbb{E}[X]=0\cdot (1-\mu)+1\cdot \mu=\mu
$$

### 12.2.2 Bernoulli 的 variance

因为 $X$ 只会取 $0$ 或 $1$，所以：

$$
X^2=X
$$

因此：

$$
\mathbb{E}[X^2]=\mathbb{E}[X]=\mu
$$

所以：

$$
\operatorname{var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2
=\mu-\mu^2
=\mu(1-\mu)
$$

!!! explanation "Bernoulli 在 classification 里面很常见"
    对 binary classification 来说，label 经常就是 $0$ 或 $1$。

    所以我们可以把 label 看成 Bernoulli random variable，而模型要学的是：

    $$
    p(Y=1\mid X=x)
    $$

## 12.3 Continuous Random Variables

对于 continuous random variable，我们不用 probability mass function，而是用 probability density function：

$$
p(x)
$$

区间概率通过积分得到：

$$
P(a\leq X\leq b)=\int_a^b p(x)\,dx
$$

Density 需要满足：

$$
p(x)\geq 0
$$

以及：

$$
\int p(x)\,dx=1
$$

!!! warning "density 不是 probability"
    对连续变量来说，$p(x)$ 是 density，不是 $X=x$ 的概率。

    所以 $p(x)$ 可以大于 $1$。

    同时：

    $$
    P(X=x)=0
    $$

    真正有意义的是区间概率：

    $$
    P(a\leq X\leq b)
    $$

### 12.3.1 离散规则到连续规则

连续情况下，很多规则还是一样，只是把 summation 换成 integral。

1. Sum Rule:

$$
p(x)=\int p(x,y)\,dy
$$
2. Product Rule:

$$
p(x,y)=p(y\mid x)p(x)
$$

3. Expectation:

$$
\mathbb{E}[f(X)]=\int f(x)p(x)\,dx
$$

### 12.3.2 Density Estimation

Density estimation：给定 observations，也就是数据：

$$
\mathcal{D}=\{x_1,\dots,x_N\}
$$

我们希望推断出数据背后的 probability distribution：

$$
p(X)
$$

问题是，可能有无穷多个 density function 都能解释这些数据。

所以选择哪个 function family，本身就是一个 modeling decision，也是一种 inductive bias。

!!! explanation "density estimation 和机器学习的关系"
    很多机器学习模型都可以看成是在做 density estimation，也就是估计 $P(x)$。或者至少是在估计某种 conditional distribution，也就是 supervised learning，估计 $p(Y\mid X)$。

## 12.4 Empirical Probability Distribution

假设我们做了 $N$ 次试验，观察到：

$$
(X_1,Y_1),\dots,(X_N,Y_N)
$$

令 $n_{ij}$ 表示满足下面条件的样本数量：

$$
X=x_i,\quad Y=y_j
$$

那么 empirical probability distribution 可以定义成：

$$
\hat{p}(X=x_i,Y=y_j)=\frac{n_{ij}}{N}
$$

Frequentist view 认为，当 $N\rightarrow\infty$ 时，根据 law of large numbers：

$$
\hat{p}(X=x_i,Y=y_j)\rightarrow p(X=x_i,Y=y_j)
$$

也就是 empirical distribution 会收敛到真实分布。

!!! warning "empirical distribution 的局限"
    如果 $N$ 很小，那么直接用 frequency 估计概率会非常不稳定。

    如果变量很多，或者变量取值很多，很多格子可能根本没有观察到样本。

    如果变量是 continuous 的，直接数每个取值出现了多少次就更不现实。

---

# 13. Estimating Parameters of a Distribution

给定一个 parametric distribution：

$$
p(x\mid w)
$$

我们希望根据数据：

$$
\mathcal{D}=\{x_1,\dots,x_N\}
$$

找到最好的参数 $w$。

!!! warning "我的疑问"
    ### 我的疑问1
    什么是 parametric distribution？
    ### 解析
    **parameters of a distribution** 就是**一个概率分布里面，用来决定这个分布具体长什么样的那些数。**比如正态分布：
	
	$$X \sim \mathcal N(\mu,\sigma^2)$$
	
	这里的：
	
    - $\mu$：均值，决定分布中心在哪里
	- $\sigma^2$：方差，决定分布有多宽
	
	所以 $\mu$ 和 $\sigma^2$ 就是这个 distribution 的 **parameters**。
	### 我的疑问2
	在我的认知里，$p(x|w)$ 表示的是 $w$ 发生的前提下， $x$ 发生的概率。但是这里的 $w$ 明明是参数，为什么还可以写成 $p(x|w)$。
	### 解析
	$p(x∣w)$ 在这里表示：**在参数 $w$ 已经给定的情况下，$x$ 的概率是多少。**
	
	

这节课讲两种方法：

1. Maximum Likelihood Estimation (MLE)
2. Maximum A Posteriori (MAP)

## 13.1 MLE

MLE 的目标是选择让 observed data 最可能出现的参数：

$$
w^*_{\text{MLE}}=\arg\max_w p(\mathcal{D}\mid w)
$$

这是 frequentist 的做法。

意思是，我们已经观察到一组数据 $\mathcal D$。然后假设这些数据是由某个参数化分布 $p(x\mid w)$ 产生的。MLE 要做的，就是在所有可能的参数 $w$ 里，找到那个最能让这组已观察数据出现的参数。
## 13.2 MAP

MAP chooses the parameter value $w$ that has the highest posterior probability after observing the data.

$$
w^*_{\text{MAP}}=\arg\max_w p(w\mid \mathcal{D})
$$

根据 Bayes：

$$
p(w\mid \mathcal{D})=
\frac{p(\mathcal{D}\mid w)p(w)}{p(\mathcal{D})}
$$

因为 $p(\mathcal{D})$ 和 $w$ 无关，所以优化时可以忽略：

$$
w^*_{\text{MAP}}=
\arg\max_w p(\mathcal{D}\mid w)p(w)
$$


---

# 14. MLE and Likelihood Function

Likelihood function 描述的是：在某个参数 $w$ 下，观察到当前数据集 $\mathcal{D}$ 的概率。

如果假设每个样本都是 IID：

$$
x_n\sim p(x\mid w)
$$

那么：

$$
p(\mathcal{D}\mid w)
=
\prod_{n=1}^{N}p(x_n\mid w)
$$

这个函数叫 likelihood function。

!!! warning "likelihood 不是参数的概率"
    $p(\mathcal{D}\mid w)$ 是在固定 $w$ 的情况下，数据出现的概率。

    当我们把它看成 $w$ 的函数并最大化时，它叫 likelihood。

    但它本身不是 $p(w)$。

## 14.1. Log Likelihood

实际计算中通常不直接最大化 likelihood，而是最大化 log likelihood：

$$
\log p(\mathcal{D}\mid w)
=
\sum_{n=1}^{N}\log p(x_n\mid w)
$$

由于 $\log$ 是 monotonic increasing function，所以：

$$
\arg\max_w p(\mathcal{D}\mid w)
=
\arg\max_w \log p(\mathcal{D}\mid w)
$$

如果机器学习里更喜欢最小化 loss，也可以写成：

$$
w^*_{\text{ML}}=
\arg\min_w -\log p(\mathcal{D}\mid w)
$$

这个 $-\log p(\mathcal{D}\mid w)$ 就是 negative log likelihood，也经常被叫做 log loss。

!!! explanation "为什么 MLE 很重要"
    很多模型的训练目标都可以理解成 maximum likelihood。

    比如 logistic regression 是在最大化 label 的 conditional likelihood。

    大语言模型预训练也可以理解成最大化每个 token 在前文条件下出现的 likelihood：

    $$
    \sum_{\text{doc}}\sum_{\text{word}}\log p(\text{word}\mid \text{words before it},w)
    $$

## 14.2 Bernoulli Samples 的 MLE

假设我们观察到 $N$ 个 IID Bernoulli samples：

$$
\mathcal{D}=\{x_1,\dots,x_N\}
$$

其中：

$$
x_n\in\{0,1\}
$$

并且：

$$
x_n\sim \operatorname{Bern}(\mu)
$$

### 14.2.1 Modeling

Bernoulli 的概率是：

$$
p(x_n\mid \mu)=\mu^{x_n}(1-\mu)^{1-x_n}
$$

所以 log likelihood 是：

$$
\log p(\mathcal{D}\mid \mu)
=
\sum_{n=1}^{N}\log\left(\mu^{x_n}(1-\mu)^{1-x_n}\right)
$$

展开：

$$
\log p(\mathcal{D}\mid \mu)
=
\sum_{n=1}^{N}
\left[
x_n\log\mu+(1-x_n)\log(1-\mu)
\right]
$$

因为 $x_n$ 只能是 $0$ 或 $1$，所以我们只需要数：

1. $n_1$：观测到 $1$ 的次数。
2. $n_0$：观测到 $0$ 的次数。

于是：

$$
\log p(\mathcal{D}\mid \mu)
=
n_1\log\mu+n_0\log(1-\mu)
$$

$n_0$ 和 $n_1$ 叫 sufficient statistics（充分统计量），因为对于估计 $\mu$ 来说，原始数据的顺序已经不重要了，只需要知道 $0$ 和 $1$ 各出现了多少次。

### 14.2.2 Optimization

现在最大化：

$$
\arg\max_\mu
\left[
n_1\log\mu+n_0\log(1-\mu)
\right]
$$

对 $\mu$ 求导：

$$
\frac{\partial}{\partial \mu}
\left[
n_1\log\mu+n_0\log(1-\mu)
\right]
=
\frac{n_1}{\mu}
-
\frac{n_0}{1-\mu}
$$

令导数为 $0$：

$$
\frac{n_1}{\mu}
-
\frac{n_0}{1-\mu}
=0
$$

移项：

$$
n_1(1-\mu)=n_0\mu
$$

展开：

$$
n_1-n_1\mu=n_0\mu
$$

所以：

$$
n_1=(n_0+n_1)\mu
$$

得到：

$$
\mu_{\text{ML}}
=
\frac{n_1}{n_0+n_1}
=
\frac{n_1}{N}
$$

也就是说，Bernoulli MLE 就是 observed frequency。

!!! explanation "为什么 Bernoulli MLE 就是 counting"
    如果 $10$ 次投硬币里面有 $7$ 次 heads，那么 MLE 会估计：

    $$
    \mu_{\text{ML}}=\frac{7}{10}
    $$

    因为 $\mu=0.7$ 会让这组观察结果在 Bernoulli model 下最合理。

## 14.3 MLE 在 Rare Events 上的问题

MLE 很直观，但在 rare event 上会有问题。

比如投一个硬币 $10$ 次，结果 $0$ 次 heads。

MLE 会给：

$$
\mu_{\text{ML}}=\frac{0}{10}=0
$$

这表示模型认为：

$$
p(\text{heads})=0
$$

但是这很可能不合理。它只是因为数据太少，没有观察到 heads。

### 14.3.1 Wake Word 里的 rare event 问题

假设真实 wake word rate 是：

$$
0.0001
$$

如果我们观察 $N=10000$ 个片段，那么一次 wake word 都没看到的概率是：

$$
(1-0.0001)^{10000}\approx 0.37
$$

也就是说，大约三分之一的时候，我们会观察到：

$$
n_1=0
$$

于是 MLE 会估计：

$$
\mu_{\text{ML}}=0
$$

这等于说 wake word 永远不会发生。这个结论虽然是 MLE 的正确结果，但不是一个好的现实判断。

!!! explanation "MLE 的问题在哪里"
    MLE 只相信当前数据。

    如果数据太少，或者某个事件太稀有，那么 MLE 可能会给出过于极端的概率，比如 $0$ 或 $1$。

    这时我们需要把一些先验知识放进去，也就是 MAP。

---

# 15. Parameter as a Random Variable

MLE 把参数 $\mu$ 当成一个 unknown constant（固定但不知道的常数）。

Bayesian view 会把 $\mu$ 本身也当成 random variable，并且给它一个 distribution。

根据 Bayes' theorem：

$$
p(\mu\mid \mathcal{D})
=
\frac{p(\mathcal{D}\mid \mu)p(\mu)}{p(\mathcal{D})}
$$

这里：

1. $p(\mu)$ 是 prior：看到数据之前，我们认为 $\mu$ 可能是什么。
2. $p(\mathcal{D}\mid \mu)$ 是 likelihood：这就是 MLE 最大化的那个东西。
3. $p(\mu\mid \mathcal{D})$ 是 posterior：看到数据之后，我们对 $\mu$ 的更新后的信念。

MAP estimate 是让 posterior 最大的参数：

$$
\mu_{\text{MAP}}
=
\arg\max_\mu p(\mu\mid \mathcal{D})
$$

因为 $p(\mathcal{D})$ 与 $\mu$ 无关，所以也可以最大化：

$$
\mu_{\text{MAP}}
=
\arg\max_\mu
\left[
\log p(\mathcal{D}\mid \mu)+\log p(\mu)
\right]
$$

---

# 16. Beta Distribution

对于 Bernoulli parameter：

$$
0\leq \mu\leq 1
$$

我们需要一个定义在 $[0,1]$ 上的 prior distribution。

常用的是 Beta distribution：

$$
X\sim \operatorname{Beta}(a,b)
$$

它的 density 是：

$$
p(X;a,b)=\frac{1}{B(a,b)}X^{a-1}(1-X)^{b-1}
$$

其中 $B(a,b)$ 是 normalizer，保证整个 density 积分等于 $1$。

## 16.1 Beta Distribution 的形状

不同的 $a,b$ 表示不同的 prior belief：

1. $a=b=1$：所有 $[0,1]$ 具有相同的 probability density。
2. $a=b=2$：更相信 $X$ 靠近 $0.5$。
3. $a,b$ 越大，distribution 越集中在 $\frac{a}{a+b}$ 附近。

!!! explanation "为什么 Beta 适合做 Bernoulli 的 prior"
    Bernoulli 的参数 $\mu$ 本来就是一个概率，所以它必须在 $[0,1]$ 之间。

    Beta distribution 也定义在 $[0,1]$ 上，而且形状很灵活。

    所以它很适合表示我们对 $\mu$ 的 prior belief。

---

# 17. Bernoulli + Beta 的 Posterior

我们要计算：

$$
p(\mu\mid \mathcal{D})
\propto
p(\mathcal{D}\mid \mu)p(\mu)
$$

Bernoulli likelihood 是：

$$
p(\mathcal{D}\mid \mu)
=
\mu^{n_1}(1-\mu)^{n_0}
$$

Beta prior 去掉 normalizer 后是：

$$
p(\mu)\propto \mu^{a-1}(1-\mu)^{b-1}
$$

相乘：

$$
p(\mu\mid \mathcal{D})
\propto
\mu^{n_1}(1-\mu)^{n_0}
\mu^{a-1}(1-\mu)^{b-1}
$$

整理指数：

$$
p(\mu\mid \mathcal{D})
\propto
\mu^{n_1+a-1}(1-\mu)^{n_0+b-1}
$$

这还是一个 Beta distribution。

新的参数是：

$$
a'=n_1+a,\quad b'=n_0+b
$$

!!! explanation "conjugate prior"
    如果一个 prior 和 likelihood 结合之后，posterior 还是同一种 distribution family，那么这个 prior 就叫 conjugate prior。

    Beta distribution 是 Bernoulli likelihood 的 conjugate prior。

    好处是计算 posterior 很方便，因为更新前后形式不变。

---

# 18 Bernoulli Likelihood + Beta Prior 下的 MAP

!!! note "关于 $p(\mu)$ 的记号"

	因为 $\mu$ 是连续变量，所以这里的 $p(\mu)$ 和 $p(\mu\mid\mathcal D)$ 严格来说都是 **probability density**，而不是某个具体取值的 probability。

机器学习中通常仍然习惯用 $p(\mu)$ 来表示 density。

这里讨论的是一个**具体的 MAP 情况**：

- likelihood 选择 Bernoulli distribution；
- prior 选择 Beta distribution。

也就是说，我们假设：

$$
X \mid \mu \sim \mathrm{Bernoulli}(\mu)
$$

同时给 Bernoulli 的未知参数 $\mu$ 一个 Beta prior：

$$
\mu \sim \mathrm{Beta}(a,b)
$$

因此：

$$
p(\mu)\propto \mu^{a-1}(1-\mu)^{b-1}
$$

MAP 要寻找 posterior 最大的位置：

$$
\mu_{\mathrm{MAP}}
=
\arg\max_{\mu} p(\mu\mid\mathcal D)
$$

根据 Bayes' theorem：

$$
p(\mu\mid\mathcal D)
\propto
p(\mathcal D\mid\mu)p(\mu)
$$

所以 MAP 等价于最大化：

$$
p(\mathcal D\mid\mu)p(\mu)
$$

或者最大化它的 log：

$$
\log p(\mathcal D\mid\mu)+\log p(\mu)
$$

注意：

MAP 本身并不要求 prior 一定是 Beta distribution。

这里只是针对：

$$
\text{Bernoulli likelihood} + \text{Beta prior}
$$

这个具体情况进行推导。


## 18.1 Computing the MAP

Bernoulli likelihood 是：

$$
p(\mathcal D\mid\mu)
=
\mu^{n_1}(1-\mu)^{n_0}
$$

其中：

- $n_1$：数据中 $1$ 的数量，也就是 successes 的数量；
- $n_0$：数据中 $0$ 的数量，也就是 failures 的数量；
- $N=n_0+n_1$：总样本数。

Beta prior 是：

$$
p(\mu)
\propto
\mu^{a-1}(1-\mu)^{b-1}
$$

因此：

$$
p(\mathcal D\mid\mu)p(\mu)
\propto
\mu^{n_1}(1-\mu)^{n_0}
\mu^{a-1}(1-\mu)^{b-1}
$$

整理指数：

$$
p(\mathcal D\mid\mu)p(\mu)
\propto
\mu^{n_1+a-1}
(1-\mu)^{n_0+b-1}
$$

取 log：

$$
\log p(\mathcal D\mid\mu)
+
\log p(\mu)
=
(n_1+a-1)\log\mu
+
(n_0+b-1)\log(1-\mu)
$$

这个形式和前面 Bernoulli MLE 的形式几乎完全一样。

Bernoulli MLE 得到：

$$
\mu_{\mathrm{ML}}
=
\frac{n_1}{n_0+n_1}
$$

现在相当于把：

$$
n_1
\leftarrow
n_1+a-1
$$

以及：

$$
n_0
\leftarrow
n_0+b-1
$$

代入 MLE 的形式。

因此：

$$
\mu_{\mathrm{MAP}}
=
\frac{n_1+a-1}
{n_0+n_1+a+b-2}
$$

因为：

$$
N=n_0+n_1
$$

所以：

$$
\boxed{
\mu_{\mathrm{MAP}}
=
\frac{n_1+a-1}
{N+a+b-2}
}
$$

## 18.2 Prior as Pseudo-Counts

在 Bernoulli likelihood + Beta prior 的情况下：

$$
\mu_{\mathrm{MAP}}
=
\frac{n_1+a-1}
{N+a+b-2}
$$

可以把 Beta prior 中的：

$$
a-1
$$

理解成在真正观察数据之前，额外加入的 imaginary successes。

同样：

$$
b-1
$$

可以理解成额外加入的 imaginary failures。

因此：

$$
a-1
=
\text{pseudo-count of successes}
$$

$$
b-1
=
\text{pseudo-count of failures}
$$

所以 MAP 的公式可以理解成：

$$
\mu_{\mathrm{MAP}}
=
\frac{
\text{real successes}
+
\text{pseudo successes}
}{
\text{real observations}
+
\text{pseudo observations}
}
$$


