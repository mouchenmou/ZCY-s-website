# CS182 Lecture 21 — RLVR, RLHF and DPO

> 依据老师的 [Lecture 21 Notes](CS182%20深度神经网络/Lecture/Part%209%20RL/附件/21_fa26.pdf)（第 4 页为空白）。本节从基础 Reinforcement Learning 推导到语言模型的两条后训练路线：**可验证奖励（RLVR / ScaleRL）**与**人类偏好（RLHF / DPO）**。下文的手算例子、Policy 网络构造和中间数学推导是为理解原稿补充的内容，不是论文的实验数据。全文 $\log$ 表示自然对数。

## 1. 为什么语言模型需要 Post-training

### 1.1 Pre-training（预训练）与 SFT（监督微调）

**Pre-training（预训练）**让语言模型学会 Next-token Prediction（下一个 Token 预测）。Token 是模型处理文本的基本单位；预测下一个 Token 本质上是 Multi-class Classification（多类别分类），类别是词表中的各个 Token。模型对每个 Token 计算概率，概率之和为 1。

例如输入 `The capital of France is`，训练文本中的下一个 Token 为 `Paris`。模型需要提高该 Token 的概率。对于真实文本序列 $y_1,\ldots,y_T$，训练的 Negative Log-Likelihood（负对数似然，NLL）为

$$
\mathcal L_{\mathrm{pre}}=-\sum_{t=1}^{T}\log P_\theta(y_t\mid y_{<t}).
$$

预训练数据包含各种行为（正确解释、错误解释、小说、论坛等）。学会续写文本（Intelligent Autocomplete）不等于学会按用户要求回答（Intelligent Assistant）。因此需要 **Post-training（后训练）**，提高好回答的概率、降低坏回答的概率。

**SFT（Supervised Fine-Tuning，监督微调）**是后训练的一种方式。外部数据提供问题 $x$ 和高质量示范回答 $y^*$，模型计算该回答的概率并学习模仿它：

$$
\mathcal L_{\mathrm{SFT}}=-\log\pi_\theta(y^*\mid x).
$$

这里 $\theta$ 是模型参数，$\pi_\theta(y\mid x)$ 是该模型给回答 $y$ 分配的概率。取对数把多个 Token 的概率连乘变成加法；加负号把最大化概率变成最小化 Loss。SFT 使用已知目标文本的前缀进行预测（Teacher Forcing），不必先随机生成出示范答案。

### 1.2 Domain Shift 与 Chicken-and-Egg Problem（鸡生蛋问题）

**Domain Shift（领域偏移）/ Distribution Shift（分布偏移）**：训练时的数据分布与实际使用时的数据分布不同。例如，预训练时学习网页文本，实际使用时要求进行复杂数学推理。SGD（Stochastic Gradient Descent，随机梯度下降）通过训练数据中的规律更新权重；目标领域与训练数据差异很大时，原有能力不一定够用。

![](Lecture21%20p01%20motivation.png)

**model shapes the input** 涉及两种依赖关系：

- **Autoregressive Generation（自回归生成）**：模型刚生成的 Token 会成为下一步输入的前文。如果前面写错了，后面会在这种前文上继续预测。预训练通常用人写的前文，实际生成用模型自己的前文，这会造成输入分布差异（也称 Exposure Bias，暴露偏差）。
- 在下面讲的 **Reinforcement Learning（强化学习，RL）** 中，外界提供题目，模型自己尝试作答，外界再评价这些答案。模型当前的能力决定哪些回答更容易被采样到，也就影响后续训练能获得哪些反馈。如果一道题几乎从未答对，就难以从正确回答中学习；这又使模型更难提升该能力。

第二层就是 Chicken-and-Egg Problem 的一种具体表现。模型不是凭空创造所有训练数据：问题和评价规则仍可来自外界，模型生成的是待评价的回答。老师最后一句强调训练进展受模型目前能够生成的内容影响。

### 1.3 Feedback（反馈）的来源

SFT 需要完整的示范答案。很多任务却更容易评价答案而不是撰写标准示范。**Reward（奖励）**就是对一次回答的评价分数，强化学习利用它提高今后的期望得分。

| 反馈来源 | 训练信号 | 示例 |
|---|---|---|
| Human Feedback（人类反馈） | 人类给分或比较两个回答 | 哪个解释更清楚？ |
| Verifiable Rewards（可验证奖励） | 程序依据规则打分 | 数学标准答案、形式化证明检查、代码测试用例 |
| AI Feedback（AI 反馈） | 另一个 AI 模型给分或比较 | 评价回答是否符合要求 |

Good examples → GD，Bad examples → GA：若使用普通 NLL $\ell=-\log\pi_\theta(y\mid x)$，GD（Gradient Descent，梯度下降）鼓励好回答，GA（Gradient Ascent，梯度上升）抑制坏回答。直接对坏样本做 GA 可能不稳定，后面会讲更系统的训练目标。

**两条路线**：自动验证得到奖励 → RLVR；人类比较得到偏好标签 → 经典 RLHF 或 DPO。它们都在后训练语言模型，而不是三种不同的网络结构。

## 2. Reinforcement Learning（RL）：从 Policy 到 REINFORCE

### 2.1 Policy（策略）究竟是什么？

本课从这里开始仍然是在训练语言模型。强化学习是训练方法，Policy 是需要训练的策略函数；此处 Policy 由语言模型神经网络实现。

一般的随机策略（Stochastic Policy）记作 $\pi_\theta(a\mid s)$：输入 State（状态）$s$，输出各个 Action（动作）$a$ 的概率。语言模型的状态是 Prompt（输入问题）与已生成前文，动作是下一个 Token。

$$
(x,y_{<t})\xrightarrow{\text{Transformer }\theta}\text{Logits}\xrightarrow{\mathrm{Softmax}}\pi_\theta(\cdot\mid x,y_{<t})\xrightarrow{\mathrm{Sampling}}y_t.
$$

| 对象                        | 含义                                  |
| ------------------------- | ----------------------------------- |
| $\theta$                  | 神经网络中连续的可训练权重                       |
| $\pi_\theta(\cdot\mid s)$ | 在固定上下文下，各个动作的概率；概率值可随 $\theta$ 连续变化 |
| $y_t$                     | 实际抽出的离散 Token                       |
|                           |                                     |

**离散概率分布（Discrete Probability Distribution）** 指被抽取的结果是离散的，不是说概率或权重只能取离散数。固定一个回答，$\pi_\theta(y\mid x)$ 关于 $\theta$ 可以求导；对随机抽出的离散 Token 本身不能直接使用普通反传。

一个 Rollout（完整采样轨迹）是从问题开始生成直到回答结束的整个过程。一个回答 $y=(y_1,\ldots,y_T)$ 的概率为

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}),\qquad
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

这里完整回答的概率包含结束 Token（EOS，End of Sequence）等停止规则。回答不同，所经过的前文也可能不同。

**Parallel Rollouts（并行采样）** 指同一个 Policy、同一个问题、参数暂时不变，随机生成 $G$ 个回答。每个回答都有自己的 Reward。随机抽样的是回答，不是不同的神经网络。例如同题答 4 次可能得到 $8,7,7,7$。

### 2.2 基础 RL 训练流程与 REINFORCE 推导

最简单的 On-policy RL（同策略强化学习）：

1. 外界提供问题 $x$，当前模型 $\pi_{\theta_0}$ 生成一批回答，外界给每个回答评分。
2. 用当前模型计算这些回答的概率梯度，更新参数 $\theta_0\to\theta_1$。
3. 使用 $\pi_{\theta_1}$ **重新生成**一批回答，再评分、更新 $\theta_1\to\theta_2$。

这种每次更新后重新采样、生成和训练使用同一策略的流程，完全不需要新旧策略的概率比 $\rho$。第 3 章 ScaleRL 才引入它。

**REINFORCE（一种 Policy Gradient，策略梯度算法）** 解决的问题：如何用生成后的奖励训练模型？

固定问题 $x$，为了简洁先省略条件 $x$。老师使用 $P_\theta(y)$ 表示 Policy（等价于 $\pi_\theta(y\mid x)$），$F(y)$ 表示奖励。答对得 1，答错得 0。评分与已经给定的回答有关，求导时视为固定。

我们真正要最大化的是 Expected Reward（期望奖励）：

$$
\boxed{J(\theta)=\mathbb E_{y\sim P_\theta}[F(y)]=\sum_y P_\theta(y)F(y).}
$$

$J$ 是由神经网络给出的概率和奖励构成的标量目标，不是另一个神经网络。$\sum_y$ 遍历所有可能的回答。若模型只允许答 $8$ 和 $7$，$p_8=0.25,p_7=0.75$，则 $J=0.25\times1+0.75\times0=0.25$。

直接求导可得

$$
\nabla_\theta J=\sum_y F(y)\nabla_\theta P_\theta(y).
$$

注意 $\nabla_\theta P_\theta(y)$ **已经表示概率对模型权重的梯度**。实际神经网络用 Backpropagation（反向传播）和 Chain Rule（链式法则）计算它，第 2.4 节给出完整数值例子。

困难在于真实语言模型的回答空间太大，无法枚举全部回答；Reward 和离散 Sampling 也不能当成普通可微层直接反传。使用 **Log-Derivative Trick（对数求导技巧）**：

$$
\nabla_\theta\log P_\theta(y)=\frac{1}{P_\theta(y)}\nabla_\theta P_\theta(y)
\quad\Longrightarrow\quad
\nabla_\theta P_\theta(y)=P_\theta(y)\nabla_\theta\log P_\theta(y).
$$

代回上式，就可以把梯度改写成期望：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_y P_\theta(y)\big[F(y)\nabla_\theta\log P_\theta(y)\big]\\
&=\boxed{\mathbb E_{y\sim P_\theta}\big[F(y)\nabla_\theta\log P_\theta(y)\big]}. 
\end{aligned}
$$

现在才使用 **Monte Carlo Sampling（蒙特卡洛采样）**。独立抽取 $G$ 个回答 $y_1,\ldots,y_G\sim P_\theta$，用样本平均近似期望：

$$
\boxed{\widehat g=\frac1G\sum_{i=1}^G F(y_i)\nabla_\theta\log P_\theta(y_i)\approx\nabla_\theta J.}
$$


### 2.3 Advantage（优势）与 Surrogate Objective（代理目标）

只用 $F=0,1$ 时，错误答案的直接梯度项为零。我们希望错误回答也提供负的更新信号，因此引入 Baseline（基准）$b$：

$$
A(y)=F(y)-b.
$$

本课后面采用 Group-relative Advantage（组内相对优势）。例如同一道题抽四次，奖励为 $[1,0,0,0]$，则

$$
b=\bar r=\frac{1+0+0+0}{4}=0.25,\qquad
\hat A=[0.75,-0.25,-0.25,-0.25].
$$

用一个不依赖当前采样结果的固定基准时，减去它不改变真实的期望梯度，因为

$$
\begin{aligned}
\mathbb E_{y\sim P_\theta}[\nabla_\theta\log P_\theta(y)]
&=\sum_y P_\theta(y)\nabla_\theta\log P_\theta(y)\\
&=\nabla_\theta\sum_yP_\theta(y)=\nabla_\theta 1=0.
\end{aligned}
$$

因此 $\mathbb E[F\nabla\log P]=\mathbb E[(F-b)\nabla\log P]$。合适的基准还可以降低梯度估计的方差（Variance Reduction）。组内均值包含当前样本自身，因此有限样本下不严格满足独立基准的无偏性条件；ScaleRL 使用这种组内估计并做归一化。

用 Advantage 的样本梯度估计是

$$
\boxed{\hat g=\frac1G\sum_i\hat A_i\nabla_\theta\log\pi_\theta(y_i\mid x).}
$$



实际代码用自动微分（Automatic Differentiation）计算上面的梯度。为此，在已经推导出梯度后，另行构造 **Surrogate Objective（代理目标）**：

$$
S(\theta)=\frac1G\sum_i\operatorname{sg}(\hat A_i)\log\pi_\theta(y_i\mid x),\qquad \nabla_\theta S=\hat g.
$$

$\operatorname{sg}$ 表示 Stop Gradient（停止梯度），本次反传把已计算的优势固定。$S$ 和 $J$ 的函数值不同：$J$ 是期望奖励，$S$ 是为获得所需梯度而构造的标量。不能把 $S$ 或 $A\log P$ 直接加到参数上，必须对它求导。做梯度下降时使用 $\mathcal L=-S$，效果与对 $S$ 做梯度上升等价。

!!! warning "我的问题"

    ### 我的问题

    为什么 Surrogate Objective（代理目标函数）里面直接就有 $\log$？

    $$
    S(\theta)=
    \frac1G\sum_{i=1}^{G}
    \operatorname{sg}(\hat A_i)
    \log P_\theta(y_i)
    $$

    原本的目标函数 $J(\theta)$ 里面不是没有 $\log$ 吗？不是在对 $J(\theta)$ 求导之后，才通过 Log-Derivative Trick 引入 $\log$ 的吗？

    为什么这里还没有求导，就已经出现了 $\log$？

    ### 解析

    首先必须明确，Surrogate Objective $S(\theta)$ 和原始目标函数 $J(\theta)$ 是两个不同的函数。

    **1. 原始目标函数 $J(\theta)$**

    使用 Advantage 表达原始目标函数：

    $$
    J(\theta)=\sum_y P_\theta(y)A(y)+b
    $$

    此时确实没有 $\log$。

    将 Advantage $A(y)$ 和 Baseline $b$ 视为常数，对模型参数 $\theta$ 求导：

    $$
    \begin{aligned}
    \nabla_\theta J(\theta)
    &=\sum_y A(y)\nabla_\theta P_\theta(y)\\
    &=\sum_y P_\theta(y)A(y)
    \frac{\nabla_\theta P_\theta(y)}{P_\theta(y)}\\
    &=\sum_y P_\theta(y)A(y)
    \nabla_\theta\log P_\theta(y)\\
    &=\mathbb E_{y\sim P_\theta}
    \left[A(y)\nabla_\theta\log P_\theta(y)\right]
    \end{aligned}
    $$

    再通过 Monte Carlo Sampling（蒙特卡洛采样），使用实际采样到的 $G$ 个回答估计梯度：

    $$
    \boxed{
    \nabla_\theta J(\theta)\approx\hat g
    =
    \frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)
    }
    $$

    到这里，REINFORCE 的数学推导已经结束。我们已经知道应该使用什么梯度来更新神经网络的参数。

    如果手动计算梯度，可以直接计算 $\hat g$，然后通过 Gradient Ascent（梯度上升）更新参数：

    $$
    \theta_{\mathrm{new}}=\theta+\eta\hat g
    $$

    整个过程并不需要 Surrogate Objective。

    **2. 为什么还要构造 Surrogate Objective？**

    实际使用 PyTorch 时，我们通常不会手动计算每个参数的梯度，而是提供一个标量目标函数，通过 Automatic Differentiation（自动微分）计算梯度。

    现在我们已经知道，希望自动微分程序计算出的梯度为：

    $$
    \hat g=
    \frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)
    $$

    因此，我们需要反过来构造一个函数，使得对这个函数求导之后，恰好能够得到 $\hat g$。

    这个函数就是 Surrogate Objective（代理目标函数）：

    $$
    \boxed{
    S(\theta)=
    \frac1G\sum_{i=1}^{G}
    \operatorname{sg}(\hat A_i)\log P_\theta(y_i)
    }
    $$

    其中 $\operatorname{sg}$ 表示 Stop Gradient，即反向传播时将已经计算好的 Advantage 视为常数。

    现在对 $S(\theta)$ 求导：

    $$
    \begin{aligned}
    \nabla_\theta S(\theta)
    &=
    \frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)\\
    &=
    \frac1G\sum_{i=1}^{G}
    \hat A_i
    \frac{\nabla_\theta P_\theta(y_i)}
    {P_\theta(y_i)}\\
    &=\hat g
    \end{aligned}
    $$

    可以看到，对 $S(\theta)$ 求导之后，恰好得到了之前通过 REINFORCE 推导出来的梯度估计。

    注意，对 $\log P_\theta(y_i)$ 求导产生的分母 $P_\theta(y_i)$ 仍然存在，并没有被省略。

    
    
### 2.4 完整计算例子：Forward → Backward → Update

!!! example "完整例子"

    **1. Forward Propagation（前向传播）**

    假设我们训练一个语言模型回答 $5+3=?$，为了方便计算，假设模型只有两个可能的输出：$8$ 和 $7$。

    题目经过神经网络前面的所有层，得到一个 feature $h$。为了方便手算，我们固定 $h$，只训练最后一个 Linear Layer。

    模型的可训练参数为：

    $$
    \theta=
    \begin{bmatrix}
    w_8\\
    w_7
    \end{bmatrix}
    $$

    Linear Layer 首先计算两个答案对应的 Logits：

    $$
    \begin{bmatrix}
    s_8\\
    s_7
    \end{bmatrix}
    =
    \begin{bmatrix}
    w_8\\
    w_7
    \end{bmatrix}h
    =
    \begin{bmatrix}
    w_8h\\
    w_7h
    \end{bmatrix}
    $$

    接着通过 Softmax 计算两个答案的概率：

    $$
    \begin{aligned}
    p_8&=\frac{e^{s_8}}{e^{s_8}+e^{s_7}}\\
    p_7&=\frac{e^{s_7}}{e^{s_8}+e^{s_7}}
    \end{aligned}
    $$

    因此，这个模型的 Policy 就是：

    $$
    P_\theta(\cdot\mid x)
    =
    \operatorname{Softmax}
    \left(
    \begin{bmatrix}
    w_8\\
    w_7
    \end{bmatrix}h
    \right)
    =
    \begin{bmatrix}
    p_8\\
    p_7
    \end{bmatrix}
    $$

    Policy 是一个由参数 $\theta$ 控制的函数，它输出各个离散答案的概率。参数 $\theta$ 和概率值本身都可以连续变化，而根据概率采样得到的答案是离散的。

    注意，虽然 $s_8$ 只依赖于 $w_8$，但是经过 Softmax 后，$p_8$ 的分母中同时包含 $s_8$ 和 $s_7$，因此 $p_8$ 同时依赖于两个权重。$p_7$ 也是如此。

    现在假设初始参数为：

    $$
    h=2,\qquad
    w_8=\frac{\log 2}{2},\qquad
    w_7=\frac{\log 6}{2}
    $$

    代入 Linear Layer：

    $$
    \begin{aligned}
    s_8&=w_8h=\frac{\log 2}{2}\times2=\log 2\\
    s_7&=w_7h=\frac{\log 6}{2}\times2=\log 6
    \end{aligned}
    $$

    再代入 Softmax：

    $$
    \begin{aligned}
    p_8
    &=\frac{e^{\log 2}}{e^{\log 2}+e^{\log 6}}
    =\frac{2}{2+6}=0.25\\
    p_7
    &=\frac{e^{\log 6}}{e^{\log 2}+e^{\log 6}}
    =\frac{6}{2+6}=0.75
    \end{aligned}
    $$

    所以当前 Policy 为：

    $$
    \boxed{
    P_\theta(\cdot\mid x)=
    \begin{bmatrix}
    0.25\\
    0.75
    \end{bmatrix}
    }
    $$

    **2. Sampling（采样）与 Advantage 的计算**

    使用当前 Policy 随机采样 $G=4$ 次，采样过程中不更新模型参数。

    假设采样结果为：

    | Sample | Response | Reward |
    | --- | --- | --- |
    | 1 | 8 | 1 |
    | 2 | 7 | 0 |
    | 3 | 7 | 0 |
    | 4 | 7 | 0 |

    首先计算 Baseline（基准）：

    $$
    b=\frac1G\sum_{i=1}^{G}r_i
    $$

    代入数字：

    $$
    b=\frac{1+0+0+0}{4}=0.25
    $$

    然后计算每个样本的 Advantage：

    $$
    \hat A_i=r_i-b
    $$

    因此：

    $$
    \hat A=
    \begin{bmatrix}
    1-0.25\\
    0-0.25\\
    0-0.25\\
    0-0.25
    \end{bmatrix}
    =
    \begin{bmatrix}
    0.75\\
    -0.25\\
    -0.25\\
    -0.25
    \end{bmatrix}
    $$

    为了方便书写，将两个答案对应的 Advantage 分别记为：

    $$
    A_8=0.75,\qquad A_7=-0.25
    $$

    在接下来的这一次参数更新过程中，将 $A_8,A_7,b$ 全部视为常数。

    **3. 计算 Objective Function（目标函数）并推导 Policy Gradient**

    我们真正想最大化的是 Expected Reward（期望奖励）：

    $$
    J(\theta)
    =
    \mathbb E_{y\sim P_\theta}[r(y)]
    =
    \sum_yP_\theta(y)r(y)
    $$

    由于只有两个可能的答案：

    $$
    J(\theta)=p_8r_8+p_7r_7
    $$

    根据 Advantage 的定义：

    $$
    A(y)=r(y)-b
    $$

    因此：

    $$
    r(y)=A(y)+b
    $$

    代入 $J(\theta)$：

    $$
    \begin{aligned}
    J(\theta)
    &=p_8(A_8+b)+p_7(A_7+b)\\
    &=p_8A_8+p_7A_7+b(p_8+p_7)\\
    &=p_8A_8+p_7A_7+b
    \end{aligned}
    $$

    最后一行是因为 $p_8+p_7=1$。

    代入当前的概率和 Advantage：

    $$
    \begin{aligned}
    J(\theta)
    &=0.25\times0.75+0.75\times(-0.25)+0.25\\
    &=0.1875-0.1875+0.25\\
    &=0.25
    \end{aligned}
    $$

    接下来对模型参数 $\theta$ 求导。由于本次求导时 Advantage 和 Baseline 都被视为常数：

    $$
    \begin{aligned}
    \nabla_\theta J(\theta)
    &=\nabla_\theta(p_8A_8+p_7A_7+b)\\
    &=A_8\nabla_\theta p_8+A_7\nabla_\theta p_7
    \end{aligned}
    $$

    推广到所有可能的回答：

    $$
    \nabla_\theta J(\theta)
    =
    \sum_y A(y)\nabla_\theta P_\theta(y)
    $$

    由于真实语言模型无法枚举所有可能的回答，因此使用 Log-Derivative Trick，将其改写成可以通过采样估计的形式。

    首先在每一项中同时乘除 $P_\theta(y)$：

    $$
    \begin{aligned}
    \nabla_\theta J(\theta)
    &=\sum_yA(y)\nabla_\theta P_\theta(y)\\
    &=\sum_yP_\theta(y)A(y)
    \frac{\nabla_\theta P_\theta(y)}{P_\theta(y)}
    \end{aligned}
    $$

    根据链式法则：

    $$
    \boxed{
    \nabla_\theta\log P_\theta(y)
    =
    \frac{\nabla_\theta P_\theta(y)}{P_\theta(y)}
    }
    $$

    因此：

    $$
    \begin{aligned}
    \nabla_\theta J(\theta)
    &=\sum_yP_\theta(y)A(y)
    \nabla_\theta\log P_\theta(y)\\
    &=\mathbb E_{y\sim P_\theta}
    \left[
    A(y)\nabla_\theta\log P_\theta(y)
    \right]
    \end{aligned}
    $$

    然后使用 Monte Carlo Sampling，通过实际采样到的 $G$ 个回答估计期望：

    $$
    \boxed{
    \nabla_\theta J(\theta)
    \approx
    \hat g
    =
    \frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)
    }
    $$

    注意，外面的 $P_\theta(y)$ 不再显式出现，是因为我们已经按照 $P_\theta$ 的概率进行采样，样本的出现频率体现了这一概率权重。

    在实际手算梯度时，再将 Log Probability 的导数展开：

    $$
    \boxed{
    \hat g=
    \frac1G\sum_{i=1}^{G}
    \hat A_i
    \frac{\nabla_\theta P_\theta(y_i)}
    {P_\theta(y_i)}
    }
    $$

    代入采样结果 $8,7,7,7$：

    $$
    \begin{aligned}
    \hat g=\frac14\Bigg[
    &0.75\frac{\nabla_\theta p_8}{p_8}\\
    &-0.25\frac{\nabla_\theta p_7}{p_7}\\
    &-0.25\frac{\nabla_\theta p_7}{p_7}\\
    &-0.25\frac{\nabla_\theta p_7}{p_7}
    \Bigg]
    \end{aligned}
    $$

    到这里，我们还无法计算出具体的梯度，因为还不知道：

    $$
    \nabla_\theta p_8,\qquad
    \nabla_\theta p_7
    $$

    这两个量需要根据神经网络的具体结构，通过 Backpropagation 计算出来。

    这里使用同组样本的平均奖励作为 Baseline。由于该 Baseline 包含当前样本自身的奖励，因此有限样本下的梯度估计并不严格无偏，但不影响本例对具体更新过程的展示。

    **4. Backpropagation（反向传播）**

    前向传播中，模型参数与概率的关系为：

    $$
    (w_8,w_7)
    \longrightarrow
    (s_8,s_7)
    \longrightarrow
    (p_8,p_7)
    $$

    因此，首先计算概率对 Logits 的导数，再计算 Logits 对模型参数的导数，最后利用 Chain Rule 将它们连接起来。

    **(1) 计算概率 $p$ 对 Logits $s$ 的导数**

    回顾 Softmax 的计算公式：

    $$
    p_8=\frac{e^{s_8}}{e^{s_8}+e^{s_7}},
    \qquad
    p_7=\frac{e^{s_7}}{e^{s_8}+e^{s_7}}
    $$

    为了方便求导，将共同的分母记为：

    $$
    D=e^{s_8}+e^{s_7}
    $$

    因此：

    $$
    p_8=\frac{e^{s_8}}D,\qquad
    p_7=\frac{e^{s_7}}D
    $$

    使用 Quotient Rule（商的求导法则）：

    $$
    \left(\frac{f}{g}\right)'
    =
    \frac{f'g-fg'}{g^2}
    $$

    首先计算 $p_8$ 对 $s_8$ 的导数：

    $$
    \begin{aligned}
    \frac{\partial p_8}{\partial s_8}
    &=\frac{e^{s_8}D-e^{s_8}e^{s_8}}{D^2}\\
    &=\frac{e^{s_8}(e^{s_8}+e^{s_7})-e^{2s_8}}{D^2}\\
    &=\frac{e^{s_8}e^{s_7}}{D^2}\\
    &=\frac{e^{s_8}}D\frac{e^{s_7}}D\\
    &=p_8p_7
    \end{aligned}
    $$

    再计算 $p_8$ 对 $s_7$ 的导数：

    $$
    \begin{aligned}
    \frac{\partial p_8}{\partial s_7}
    &=\frac{0\times D-e^{s_8}e^{s_7}}{D^2}\\
    &=-\frac{e^{s_8}e^{s_7}}{D^2}\\
    &=-p_8p_7
    \end{aligned}
    $$

    同样，计算 $p_7$ 对 $s_8$ 和 $s_7$ 的导数：

    $$
    \begin{aligned}
    \frac{\partial p_7}{\partial s_8}
    &=\frac{0\times D-e^{s_7}e^{s_8}}{D^2}\\
    &=-p_8p_7\\
    \frac{\partial p_7}{\partial s_7}
    &=\frac{e^{s_7}D-e^{s_7}e^{s_7}}{D^2}\\
    &=\frac{e^{s_7}e^{s_8}}{D^2}\\
    &=p_8p_7
    \end{aligned}
    $$

    将四个导数整理成矩阵：

    $$
    \boxed{
    \begin{bmatrix}
    \frac{\partial p_8}{\partial s_8}&
    \frac{\partial p_8}{\partial s_7}\\
    \frac{\partial p_7}{\partial s_8}&
    \frac{\partial p_7}{\partial s_7}
    \end{bmatrix}
    =
    \begin{bmatrix}
    p_8p_7&-p_8p_7\\
    -p_8p_7&p_8p_7
    \end{bmatrix}
    }
    $$

    代入 $p_8=0.25,p_7=0.75$：

    $$
    \begin{bmatrix}
    0.1875&-0.1875\\
    -0.1875&0.1875
    \end{bmatrix}
    $$

    **(2) 计算 Logits $s$ 对权重 $w$ 的导数**

    回顾 Linear Layer 的计算：

    $$
    s_8=w_8h,\qquad s_7=w_7h
    $$

    因此：

    $$
    \begin{aligned}
    \frac{\partial s_8}{\partial w_8}&=h,
    &
    \frac{\partial s_8}{\partial w_7}&=0\\
    \frac{\partial s_7}{\partial w_8}&=0,
    &
    \frac{\partial s_7}{\partial w_7}&=h
    \end{aligned}
    $$

    出现 $0$ 是因为 $s_8$ 不依赖于 $w_7$，而 $s_7$ 不依赖于 $w_8$。

    整理成矩阵：

    $$
    \begin{bmatrix}
    \frac{\partial s_8}{\partial w_8}&
    \frac{\partial s_8}{\partial w_7}\\
    \frac{\partial s_7}{\partial w_8}&
    \frac{\partial s_7}{\partial w_7}
    \end{bmatrix}
    =
    \begin{bmatrix}
    h&0\\
    0&h
    \end{bmatrix}
    $$

    代入 $h=2$：

    $$
    \begin{bmatrix}
    2&0\\
    0&2
    \end{bmatrix}
    $$

    **(3) 使用 Chain Rule 计算概率 $p$ 对权重 $w$ 的导数**

    由于每个概率都依赖于两个 Logits，因此对每个权重求导时，需要考虑经过两个 Logits 的路径。

    $$
    \begin{aligned}
    \frac{\partial p_8}{\partial w_8}
    &=
    \frac{\partial p_8}{\partial s_8}
    \frac{\partial s_8}{\partial w_8}
    +
    \frac{\partial p_8}{\partial s_7}
    \frac{\partial s_7}{\partial w_8}\\
    &=p_8p_7h+(-p_8p_7)\times0\\
    &=hp_8p_7
    \end{aligned}
    $$

    $$
    \begin{aligned}
    \frac{\partial p_8}{\partial w_7}
    &=
    \frac{\partial p_8}{\partial s_8}
    \frac{\partial s_8}{\partial w_7}
    +
    \frac{\partial p_8}{\partial s_7}
    \frac{\partial s_7}{\partial w_7}\\
    &=p_8p_7\times0+(-p_8p_7)h\\
    &=-hp_8p_7
    \end{aligned}
    $$

    $$
    \begin{aligned}
    \frac{\partial p_7}{\partial w_8}
    &=
    \frac{\partial p_7}{\partial s_8}
    \frac{\partial s_8}{\partial w_8}
    +
    \frac{\partial p_7}{\partial s_7}
    \frac{\partial s_7}{\partial w_8}\\
    &=(-p_8p_7)h+p_8p_7\times0\\
    &=-hp_8p_7
    \end{aligned}
    $$

    $$
    \begin{aligned}
    \frac{\partial p_7}{\partial w_7}
    &=
    \frac{\partial p_7}{\partial s_8}
    \frac{\partial s_8}{\partial w_7}
    +
    \frac{\partial p_7}{\partial s_7}
    \frac{\partial s_7}{\partial w_7}\\
    &=(-p_8p_7)\times0+p_8p_7h\\
    &=hp_8p_7
    \end{aligned}
    $$

    由于模型参数为 $\theta=[w_8,w_7]^\top$，所以：

    $$
    \boxed{
    \nabla_\theta p_8=
    \begin{bmatrix}
    hp_8p_7\\
    -hp_8p_7
    \end{bmatrix},
    \qquad
    \nabla_\theta p_7=
    \begin{bmatrix}
    -hp_8p_7\\
    hp_8p_7
    \end{bmatrix}
    }
    $$

    代入数字：

    $$
    hp_8p_7=2\times0.25\times0.75=0.375
    $$

    得到：

    $$
    \boxed{
    \nabla_\theta p_8=
    \begin{bmatrix}
    0.375\\
    -0.375
    \end{bmatrix},
    \qquad
    \nabla_\theta p_7=
    \begin{bmatrix}
    -0.375\\
    0.375
    \end{bmatrix}
    }
    $$

    这两个向量就是 REINFORCE 公式中的 $\nabla_\theta P_\theta(y)$ 在本例中的具体计算结果。

    **(4) 计算最终的 Policy Gradient**

    回到前面已经推导出来的采样梯度：

    $$
    \begin{aligned}
    \hat g
    &=\frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)\\
    &=\frac1G\sum_{i=1}^{G}
    \hat A_i
    \frac{\nabla_\theta P_\theta(y_i)}
    {P_\theta(y_i)}
    \end{aligned}
    $$

    由于采样结果为一个 $8$ 和三个 $7$：

    $$
    \hat g
    =
    \frac14
    \left[
    A_8\frac{\nabla_\theta p_8}{p_8}
    +
    3A_7\frac{\nabla_\theta p_7}{p_7}
    \right]
    $$

    现在所有未知量都已经计算出来，可以直接代入：

    $$
    \begin{aligned}
    \hat g=\frac14\Bigg[
    &0.75
    \frac{
    \begin{bmatrix}
    0.375\\
    -0.375
    \end{bmatrix}
    }{0.25}\\
    &+3(-0.25)
    \frac{
    \begin{bmatrix}
    -0.375\\
    0.375
    \end{bmatrix}
    }{0.75}
    \Bigg]
    \end{aligned}
    $$

    先计算分母：

    $$
    \hat g=
    \frac14
    \left[
    0.75
    \begin{bmatrix}
    1.5\\
    -1.5
    \end{bmatrix}
    +
    3(-0.25)
    \begin{bmatrix}
    -0.5\\
    0.5
    \end{bmatrix}
    \right]
    $$

    再乘上 Advantage：

    $$
    \hat g=
    \frac14
    \left[
    \begin{bmatrix}
    1.125\\
    -1.125
    \end{bmatrix}
    +
    3
    \begin{bmatrix}
    0.125\\
    -0.125
    \end{bmatrix}
    \right]
    $$

    最后对四条样本取平均：

    $$
    \boxed{
    \hat g=
    \frac14
    \begin{bmatrix}
    1.5\\
    -1.5
    \end{bmatrix}
    =
    \begin{bmatrix}
    0.375\\
    -0.375
    \end{bmatrix}
    }
    $$

    虽然四条样本的 Advantage 相加为零：

    $$
    0.75-0.25-0.25-0.25=0
    $$

    但是每条样本的 Advantage 需要先乘上各自的 Log Probability Gradient，再对梯度取平均。

    | Response | Advantage | 对 $w_8$ 的梯度贡献 | 对 $w_7$ 的梯度贡献 |
    | --- | --- | --- | --- |
    | 8 | 0.75 | 1.125 | -1.125 |
    | 7 | -0.25 | 0.125 | -0.125 |
    | 7 | -0.25 | 0.125 | -0.125 |
    | 7 | -0.25 | 0.125 | -0.125 |
    | Mean | 0 | 0.375 | -0.375 |

    因此，虽然 Advantage 的平均值为零，但是最终的 Policy Gradient 并不为零。

    **5. Parameter Update（参数更新）**

    由于我们的目标是最大化 Expected Reward，因此使用 Gradient Ascent（梯度上升）：

    $$
    \boxed{
    \theta_{\mathrm{new}}
    =
    \theta+\eta\hat g
    }
    $$

    其中 $\eta$ 是 Learning Rate。

    取 $\eta=0.1$，代入原始权重和刚刚计算出的梯度：

    $$
    \begin{aligned}
    \theta_{\mathrm{new}}
    &=
    \begin{bmatrix}
    w_8\\
    w_7
    \end{bmatrix}
    +
    0.1
    \begin{bmatrix}
    0.375\\
    -0.375
    \end{bmatrix}\\
    &=
    \begin{bmatrix}
    \frac{\log2}{2}+0.0375\\
    \frac{\log6}{2}-0.0375
    \end{bmatrix}
    \end{aligned}
    $$

    至此，完成一次参数更新。

    使用新的权重重新进行 Forward Propagation。

    首先计算新的 Logits：

    $$
    \begin{aligned}
    s_{8,\mathrm{new}}
    &=w_{8,\mathrm{new}}h\\
    &=\left(\frac{\log2}{2}+0.0375\right)\times2\\
    &=\log2+0.075\\
    s_{7,\mathrm{new}}
    &=w_{7,\mathrm{new}}h\\
    &=\left(\frac{\log6}{2}-0.0375\right)\times2\\
    &=\log6-0.075
    \end{aligned}
    $$

    再通过 Softmax 计算新的概率：

    $$
    \begin{aligned}
    p_{8,\mathrm{new}}
    &=
    \frac{e^{s_{8,\mathrm{new}}}}
    {e^{s_{8,\mathrm{new}}}+e^{s_{7,\mathrm{new}}}}\\
    &=
    \frac{2e^{0.075}}
    {2e^{0.075}+6e^{-0.075}}\\
    &\approx0.2792
    \end{aligned}
    $$

    $$
    \begin{aligned}
    p_{7,\mathrm{new}}
    &=
    \frac{e^{s_{7,\mathrm{new}}}}
    {e^{s_{8,\mathrm{new}}}+e^{s_{7,\mathrm{new}}}}\\
    &=
    \frac{6e^{-0.075}}
    {2e^{0.075}+6e^{-0.075}}\\
    &\approx0.7208
    \end{aligned}
    $$

    最后，使用新的概率重新计算原始目标函数 $J(\theta)$。这里仍然使用原来的固定 $A_8,A_7,b$ 表达期望奖励：

    $$
    \begin{aligned}
    J(\theta_{\mathrm{new}})
    &=p_{8,\mathrm{new}}A_8+p_{7,\mathrm{new}}A_7+b\\
    &=0.2792\times0.75+0.7208\times(-0.25)+0.25\\
    &=0.2792
    \end{aligned}
    $$

    对比参数更新前后：

    | | Before Update | After Update |
    | --- | --- | --- |
    | $p_8$ | 0.25 | 0.2792 |
    | $p_7$ | 0.75 | 0.7208 |
    | $J(\theta)$ | 0.25 | 0.2792 |

    可以看到，梯度更新直接作用于神经网络的参数 $\theta$，新的 Policy 概率则是通过更新后的参数重新执行 Forward Propagation 得到的。

    **6. Surrogate Objective（代理目标函数）**

    上面已经直接通过 REINFORCE 计算出了参数梯度：

    $$
    \hat g=
    \frac1G\sum_{i=1}^{G}
    \hat A_i
    \frac{\nabla_\theta P_\theta(y_i)}
    {P_\theta(y_i)}
    $$

    但是实际使用 PyTorch 时，我们一般不会手动编写所有求导过程，而是提供一个标量函数，然后调用 backward() 自动求导。

    因此，可以另外构造一个 Surrogate Objective：

    $$
    S(\theta)=
    \frac1G\sum_{i=1}^{G}
    \operatorname{sg}(\hat A_i)
    \log P_\theta(y_i)
    $$

    其中 $\operatorname{sg}$ 表示 Stop Gradient，即求导时将 Advantage 视为常数。

    对这个函数求导：

    $$
    \begin{aligned}
    \nabla_\theta S(\theta)
    &=\frac1G\sum_{i=1}^{G}
    \hat A_i\nabla_\theta\log P_\theta(y_i)\\
    &=\frac1G\sum_{i=1}^{G}
    \hat A_i
    \frac{\nabla_\theta P_\theta(y_i)}
    {P_\theta(y_i)}\\
    &=\hat g
    \end{aligned}
    $$

    因此，对 $S(\theta)$ 求导，恰好能够得到前面手动推导的 Policy Gradient。

    注意区分：

    - $J(\theta)$：真正希望最大化的 Expected Reward，原始公式中没有 log。
    - $\hat g$：使用采样得到的参数梯度估计。
    - $S(\theta)$：为了方便 Automatic Differentiation 而构造的代理目标函数。

    $S(\theta)$ 不是原来的 $J(\theta)$，而是在推导出 Policy Gradient 之后，为方便实际编程而另外构造的函数。
## 3. RLVR（可验证奖励）与 ScaleRL（大规模强化学习）

### 3.1 RLVR：Reinforcement Learning with Verifiable Rewards

RLVR 指奖励来自能够自动检查输出的 Verifier（验证器）。它不是一个新的网络结构，也不等于某个具体的梯度算法。

例如数学题 $5+3=?$。外界有标准答案 $8$，或者外界有计算程序可以独立算出 $8$。语言模型生成完整回答，验证器核对结果：答 8 得 1，答 7 得 0。**如果没有标准答案也没有可靠的验证方法，就不能凭空设置正确性奖励。**

SFT 与 RLVR 的区别：SFT 直接提高已提供示范回答的概率；RLVR 让模型自己生成回答，再根据验证结果更新。数学题可能有多种解题过程，代码题可以用测试用例验证不同程序，形式化证明可以由证明检查器验证。我们不需要提前写出所有正确推理过程。

老师第 2 页的 Data Generation: **Chain-of-Thought（CoT，思维链）**：模型生成中间推理与最终答案。仅验证最终答案不能保证中间每一步正确；终局奖励如何归因给各个 Token 是 Credit Assignment（信用分配）问题。增加同题 Rollouts 有机会采到正确答案：单次成功概率 $p$，独立采样 $G$ 次至少成功一次的概率为 $1-(1-p)^G$。

### 3.2 ScaleRL：Scaling RL Compute（扩大强化学习计算规模）

老师第 5–6 页引用 ScaleRL 论文（Khatri et al., 2025）。研究重点是 **Post-training 中的 RL 计算量**，不是增加 Pre-training 的数据量。更长的训练、更多生成计算等都可能增加 RL compute；具体方案需要提高效率并保持稳定。

![](Lecture21%20p05%20scaling.png)

- Figure 1：横轴 GPU Hours（GPU 数量乘运行小时数，图中单位为千 GPU 小时），纵轴 Pass Rate（通过率），两轴为对数刻度。左图是验证集，右图是 AIME-24 数学评测。Dense（稠密模型）与 MoE（Mixture of Experts，混合专家模型）是不同模型结构；MoE 每个 Token 只选择部分专家参与计算。图中的实际训练点与拟合、外推曲线要区分。

![](Lecture21%20p05%20recipes.png)

- Figure 2：在同一实验设置中比较 ScaleRL、MiniMax、Magistral、Qwen2.5（DAPO）、DeepSeek（GRPO）的计算效率及通过率趋势。$A$ 是拟合的性能上限参数，$B$ 控制拟合增长形状，此处 $A$ 不是 Advantage。曲线反映特定实验而非所有任务的理论上限。

接下来四小节都是 **ScaleRL 的训练设计**。先解释为什么出现 $\rho$，再计算 Advantage、截断和完整目标。

#### 3.2.1 Importance Sampling（重要性采样）：为什么需要 $\rho$？

基础 RL 可以每次更新后重新生成，生成策略与训练策略相同。但语言模型生成长回答很贵，为节省计算可以先生成一大批，再分成若干 Mini-batch 更新，或复用同一批回答多次；大规模系统还可能异步生成，生成端和训练端使用不同版本或计算实现。

| 时间 | 生成端 | 实际用来训练的模型 |
|---|---|---|
| 先生成一大批回答 | $\pi_{\theta_0}$ | 尚未更新 |
| 用第一小批更新 | 数据来自 $\pi_{\theta_0}$ | $\theta_0\to\theta_1$ |
| 继续用第二小批（即使之前没用过） | 数据仍来自 $\pi_{\theta_0}$ | 当前 $\theta_1$ |

生成端记作 $\pi_{\mathrm{gen}}^{\theta_{\mathrm{old}}}$，训练端记作 $\pi_{\mathrm{train}}^\theta$。这时数据来自不同于当前训练策略的分布，即 Off-policy Data（异策略数据）。**不是两个模型一起更新：旧生成策略只说明这些回答当时怎么被抽出来，更新的是当前模型。** 若重新采样且两端计算完全一致，概率比恒为 1。

为什么旧样本不能直接代替新样本？继续用 $5+3$：

| 回答 | 旧概率 | 旧样本出现次数（假设恰好 100 次） | 新概率 | Reward |
|---|---:|---:|---:|---:|
| 8 | 0.2 | 20 | 0.8 | 1 |
| 7 | 0.8 | 80 | 0.2 | 0 |

Reward 不变；改变的是出现频率。用旧样本直接算平均奖励得到 $0.2$，但新模型的真实期望是 $0.8$。

定义 **Importance Sampling Ratio（重要性采样比值）**：

$$
\rho(y)=\frac{\pi_{\mathrm{train}}(y\mid x)}{\pi_{\mathrm{gen}}(y\mid x)}.
$$

本例 $\rho(8)=4,\rho(7)=0.25$，即把旧样本的 20 份按 80 份计算、80 份按 20 份计算。新期望的估计为

$$
\widehat J_{\mathrm{new}}=\frac1G\sum_i\rho(y_i)r_i
=\frac{20\cdot4\cdot1+80\cdot0.25\cdot0}{100}=0.8.
$$

这个例子的样本频率恰好等于旧概率，因此结果完全准确。一般有限采样有误差。其精确恒等式（要求旧分布覆盖新分布的支持集）是

$$
\begin{aligned}
\mathbb E_{y\sim\pi_{\mathrm{gen}}}[\rho(y)f(y)]
&=\sum_y\pi_{\mathrm{gen}}(y)\frac{\pi_{\mathrm{train}}(y)}{\pi_{\mathrm{gen}}(y)}f(y)\\
&=\mathbb E_{y\sim\pi_{\mathrm{train}}}[f(y)].
\end{aligned}
$$

对需要的策略梯度使用同样的权重，固定基准 $b$ 时有

$$
\nabla_\theta J=\mathbb E_{y\sim\pi_{\mathrm{gen}}}\big[\rho(y)(r(y)-b)\nabla_\theta\log\pi_{\mathrm{train}}(y)\big].
$$

**为什么是 $\rho(r-b)$ 而不是 $\rho r-b$？** 假设选新期望 $b=0.8$ 为基准。正确回答有 20 份，要按 80 份的贡献计算，每一份都要减一次基准。因此正确回答的总系数是 $20\times4(1-0.8)=16$。若用 $20(4\times1-0.8)=64$，就只减了 20 次基准而不是 80 次。错误回答也要按 20 份计算：$80\times0.25(0-0.8)=-16$。所以 $\rho$ 必须乘整份训练贡献。它不改变单个回答的奖励。

以上是**完整回答级别**的精确重要性采样。下面 ScaleRL 实际用的是逐 Token、截断后的权重，不能把它当作完整序列的精确无偏估计。

#### 3.2.2 Advantage Normalization（优势归一化）与重新计算时机

ScaleRL 实际不使用上面教学例子的 $E_{\mathrm{new}}$ 作基准，而使用已经生成的同题回答组的平均奖励：

$$
\bar r(x)=\frac1G\sum_{j=1}^G r_j,\qquad
\hat A_i=r_i-\bar r(x),\qquad
\boxed{\hat A_i^{\mathrm{norm}}=\frac{\hat A_i}{\hat A_{\mathrm{std}}}.}
$$

$\hat A_{\mathrm{std}}$ 是整个 Batch（批次）的 Advantage 的标准差（Standard Deviation）。先每道题内部减去各自的组均值，再把批次中不同题目的优势放在一起计算共同的标准差。公式使用总体标准差作手算示范：$s=\sqrt{\frac1N\sum_{k=1}^N(a_k-\bar a)^2}$。

延续 20 对、80 错的旧组，假设 Batch 只有这一题：

$$
\bar r=0.2,\quad \hat A_+=1-0.2=0.8,\quad \hat A_-=0-0.2=-0.2,
$$

$$
\hat A_{\mathrm{std}}=\sqrt{\frac{20(0.8)^2+80(-0.2)^2}{100}}=0.4,\qquad
\hat A_+^{\mathrm{norm}}=2,\quad\hat A_-^{\mathrm{norm}}=-0.5.
$$

优势归一化保留正负方向，调整不同批次的数值尺度。**同一批数据下，更新参数不会重新计算 Advantage。**

| 量 | 刚生成 20 对 / 80 错时 | 模型更新后仍使用旧组 | 重新采样，恰好 80 对 / 20 错 |
|---|---:|---:|---:|
| 当前 Policy 正确概率 | 0.2 | 0.8 | 0.8 |
| 组均值 | 0.2 | 0.2 | 0.8 |
| 正确/错误原始优势 | 0.8 / -0.2 | 0.8 / -0.2 | 0.2 / -0.8 |
| 归一化优势（单题 Batch） | 2 / -0.5 | 2 / -0.5 | 0.5 / -2 |
| 正确/错误概率比 | 1 / 1 | 4 / 0.25 | 1 / 1 |

新组标准差恰好也是 0.4，因为成功和失败比例互换，并非总不变。如果用于标准差统计的 Batch 改变，分母也可能改变；只更新参数不会自动修改已经算好的组优势。**重新生成数据时才按新组重新计算。**

#### 3.2.3 Token-level Importance Ratio、Truncation 与 Stop Gradient

真实回答 $y_i$ 有多个 Token。ScaleRL 给每个已生成的 Token 计算当前模型与生成模型的条件概率比：

$$
\boxed{\rho_{i,t}=\frac{\pi_{\mathrm{train}}^\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{gen}}^{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})}.}
$$

分子、分母都评价同一个 Token 和同一个前文，不需要让新模型重新随机生成出它。保存生成时的概率，再用当前模型前向计算指定 Token 的概率即可。完整回答的概率比是逐 Token 比值的乘积；ScaleRL 实际使用单个 Token 的比值。

**Truncation（截断）**：使用 $\min(\rho_{i,t},\epsilon)$ 限制权重上限，防止一个很大的比值导致更新不稳定。例如旧概率 0.001、新概率 0.1，$\rho=100$；假设上限 $\epsilon=2$，则权重为 2。这里是单边上限，不是标准 PPO 的双边 Clipping（裁剪）目标。

**Stop Gradient（停止梯度，sg）**：前向使用权重的数值，反向把它视为常数。例如 $f=\theta^2(3\theta)$ 的普通导数是 $9\theta^2$；若 $f=\operatorname{sg}(\theta^2)(3\theta)$，则反向只通过 $3\theta$，导数为 $3\theta^2$。ScaleRL 对截断比值使用 sg，让梯度从后面的当前 Policy 对数概率传回参数，而不是对比值再额外求导。

#### 3.2.4 ScaleRL 的完整 Objective（目标函数）

![](Lecture21%20p06%20objective.png)

老师第 6 页的目标如下（补全了 Token 的条件上下文）：

$$
\begin{aligned}
\mathcal J_{\mathrm{ScaleRL}}(\theta)=\mathbb E_{\substack{x\sim D\\\{y_i\}_{i=1}^{G}\sim\pi_{\mathrm{gen}}^{\theta_{\mathrm{old}}}(\cdot\mid x)}}\Bigg[
\frac{1}{\sum_{g=1}^G|y_g|}
\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}
&\operatorname{sg}(\min(\rho_{i,t},\epsilon))\\
&\cdot\hat A_i^{\mathrm{norm}}\log\pi_{\mathrm{train}}^\theta(y_{i,t}\mid x,y_{i,<t})\Bigg].
\end{aligned}
$$

| 符号 | 含义 |
|---|---|
| $x\sim D$ | 从题库抽取题目 |
| $G,i$ | 同一道题生成 $G$ 份回答，$i$ 为回答编号 |
| $t,|y_i|$ | 当前 Token 的位置、回答的 Token 长度 |
| $\sum_g|y_g|$ | 同题所有回答的 Token 总数 |
| $\rho_{i,t}$ | 当前 Token 的重要性采样权重 |
| $\hat A_i^{\mathrm{norm}}$ | 整条回答的归一化优势，同一回答所有 Token 共用 |

先在一道题内部把所有 Token 项求和、除以该题的总 Token 数，再对题目取平均。若两份回答长度为 2 和 8，本题共有 10 项；不是先分别给两个回答取平均，因此长回答贡献更多 Token 项。此式是 **Surrogate Objective**，不是最初定义的真实 Expected Reward。

固定本批数据，对每一道题求导，得到用于更新同一个当前模型的向量：

$$
g_x=\frac1{\sum_g|y_g|}\sum_{i,t}\min(\rho_{i,t},\epsilon)\hat A_i^{\mathrm{norm}}\nabla_\theta\log\pi_{\mathrm{train}}^\theta(y_{i,t}\mid x,y_{i,<t}).
$$

对各题的向量平均，再进行梯度上升；代码中可使用负目标进行梯度下降。旧生成模型不参加反向传播。

**Prompt Filtering（题目筛选）**：老师还写出两个条件：

$$
0<\operatorname{mean}(\{r_j\}_{j=1}^G)<1,\qquad\operatorname{pass\_rate}(x)<0.9.
$$

第一个要求本组既有对又有错，全对或全错使每个优势都等于 0；第二个根据历史通过率避免继续消耗计算量在已经很容易答对的题上。Imperfect Prompts 指模型尚未稳定掌握的题目，不是题目自身有错误。如果没有有效组，需跳过或补充采样，避免归一化除以零。

**完整训练顺序**：选题 → 生成回答并保存旧概率 → 验证器打分 → 筛选 → 计算组优势与 Batch 标准差 → 当前模型前向计算概率和 $\rho$ → 截断、sg、构造目标 → 反传更新当前模型。换新回答组时再重新计算该组的优势。

## 4. Human Preference-based Post-training（基于人类偏好的后训练）：RLHF 与 DPO

### 4.1 RLHF：Reinforcement Learning from Human Feedback

在无法用唯一答案自动判分的任务中，可以让人比较同一道题的两个回答，得到 **Preference Data（偏好数据）**：$(x,y_w,y_l)$。$w$ 是 Winner（较受偏好回答），$l$ 是 Loser（较不受偏好回答）。例如对“向小学生解释重力”，人类可选择解释更清楚的那个，而不必提供两个答案的绝对分数。

经典 RLHF 分成两个训练阶段：**先用偏好数据训练 Reward Model（奖励模型），然后固定奖励模型，用其分数对语言模型进行 RL。** 不要把这两个训练对象混淆。

#### 4.1.1 Reward Model 与 Bradley–Terry Model

奖励模型输入 $(x,y)$，输出一个实数 $r_\phi(x,y)$，$\phi$ 是奖励模型自身的参数，与回答模型的 $\theta$ 不同。如何仅凭人选出的 Winner/Loser 训练评分者？老师使用 **Bradley–Terry Model（成对偏好概率模型）**：

$$
P(y_w\succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)),\qquad \sigma(z)=\frac1{1+e^{-z}}.
$$

Sigmoid（S 形函数）将分数差转成 0 到 1 之间的偏好预测概率。分数差为 0 时概率 0.5；差为 2 时约 0.881；差为 -2 时约 0.119。

标注已经告诉我们 $y_w$ 获胜，所以希望该偏好概率大。使用 Maximum Likelihood（最大似然）训练，取负对数得到 Binary Cross-Entropy（此处是二分类交叉熵在胜者标签为 1 时的形式）：

$$
\boxed{\mathcal L_{\mathrm{RM}}(\phi)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\big[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big].}
$$

例如 $r_w=2,r_l=0$ 时 Loss $=-\log0.881\approx0.127$；若分数相反，则 Loss $\approx2.127$。梯度下降更新 $\phi$，使人类更喜欢的回答得到相对更高分。这个阶段是监督学习，不是用 RL 更新奖励模型。奖励只需比较相对大小；模型给出的分数不一定完全代表真实的人类偏好。

#### 4.1.2 Reference Policy（参考策略）与 KL Divergence（KL 散度）

奖励模型训练好后，我们通常从经过 Pre-training 和 SFT 的语言模型开始做 RL。这个起点复制成两份：正在训练的 $\pi_\theta$ 和固定的 $\pi_{\mathrm{ref}}$。参考模型在本阶段不更新，只提供原来的输出概率。

**KL（Kullback–Leibler，两位研究者的姓）Divergence（散度）**衡量两个概率分布在指定方向上的差异。为什么定义成这个公式？先从负对数损失开始：对某次观察到的结果 $y$，用 $Q$ 评价的损失为 $-\log Q(y)$。取 log 将独立结果的概率连乘变成可相加的损失，加负号使给实际结果的概率越高、损失越小。

假设结果按照 $P$ 出现，比较用 $Q$ 和用 $P$ 评价同一个结果的损失差，再按**实际出现概率 $P(y)$**求平均：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\mathbb E_{y\sim P}[(-\log Q(y))-(-\log P(y))]\\
&=\boxed{\sum_yP(y)\log\frac{P(y)}{Q(y)}}.
\end{aligned}
$$

例如 $P=(0.8,0.2),Q=(0.5,0.5)$，假设 100 次结果为 80 个甲、20 个乙。

| 结果 | $-\log P(y)$ | $-\log Q(y)$ | 单次损失差 $\log(P/Q)$ |
|---|---:|---:|---:|
| 甲 | 0.223 | 0.693 | 0.470 |
| 乙 | 1.609 | 0.693 | -0.916 |

因此 $D_{\mathrm{KL}}=\frac{80(0.470)+20(-0.916)}{100}\approx0.193$。外层乘 $P$ 是把单次差值变成平均差值；并非再次计算已经平均过的损失。KL 也等于 Cross-Entropy（交叉熵）$H(P,Q)$ 减 Entropy（熵）$H(P)$。

KL 散度非负，且为 0 当且仅当两个分布相同。简单证明：利用 $\log u\le u-1$，令 $u=Q(y)/P(y)$，得到 $P(y)\log(P(y)/Q(y))\ge P(y)-Q(y)$，求和后右边为 $1-1=0$（先假设概率为正）。但交换 $P,Q$ 一般不相等，因此 KL 不是对称距离。

在 RLHF 中令 $P=\pi_\theta(\cdot\mid x),Q=\pi_{\mathrm{ref}}(\cdot\mid x)$，KL 就衡量当前模型相对于固定参考模型的分布偏离。KL 本身不是一个神经网络，也不负责评价回答正确性。

#### 4.1.3 RLHF 的第二阶段：训练语言模型

奖励模型可能学到不准确的评分规律，例如喜欢很长的回答。模型只追求奖励时可能利用这一漏洞（Reward Hacking，奖励投机），破坏原本的行为。因此在提高奖励的同时，加入 **KL Regularization（KL 正则化）**，惩罚偏离参考模型。

老师第 7 页的目标是

$$
\boxed{\max_{\pi_\theta}\ \mathbb E_x\left[\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}r_\phi(x,y)-\beta D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))\right].}
$$

$\beta>0$ 为正则化系数，越大越重视保持参考行为。当前 Policy 生成回答 → 固定 Reward Model 给分 → 当前模型与固定 Reference Policy 计算/估计 KL → 结合奖励和惩罚构造 RL 目标 → 更新当前模型 $\theta$。**RL 训练的是语言模型，不是奖励模型。** 经典方案可使用 PPO（Proximal Policy Optimization，近端策略优化），其细节不在本页 Notes 中。

| 对象 | 作用 | 是否在该 RL 阶段更新 |
|---|---|---|
| 当前 Policy $\pi_\theta$ | 生成回答、计算当前概率 | 是 |
| Reward Model $r_\phi$ | 对问题和回答评分 | 否 |
| Reference Policy $\pi_{\mathrm{ref}}$ | 固定训练起点、计算 KL 所需的概率 | 否 |
| 旧生成 Policy $\pi_{\mathrm{gen}}$（若采用异策略数据） | 记录该批数据的采样概率，用于重要性采样 | 生成端刷新时可变化，不等于固定参考策略 |

### 4.2 DPO：Direct Preference Optimization（直接偏好优化）

DPO 与经典 RLHF 使用同类 Preference Data，但直接训练语言模型，不先训练独立奖励模型，也不必执行标准离线训练中的在线 RL 循环。当前模型和固定参考模型分别对数据中已经给定的 $y_w,y_l$ 计算概率，然后通过 DPO Loss 更新当前模型。

![](Lecture21%20p08%20pipeline.png)

*原图左侧的 RLHF 是奖励模型提供分数给语言模型的 RL；箭头不能理解成用 RL 更新奖励模型。*

老师第 8–9 页先提出两个朴素方法：

- 直接最大化 $\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)$。问题：只降低 loser 的概率就能让目标无限增大，不保证 winner 概率提高；可能出现 Degenerate Generation（退化生成）。
- SFT on Winners：仅使用 winner 做 SFT，损失为 $-\log\pi_\theta(y_w\mid x)$。问题：没有直接利用 loser 的对比信息。

![](Lecture21%20p08%20degeneration.png)

*老师第 8 页所示的重复文本退化例子。它说明朴素目标可能带来不良行为，不应从图片单独推断具体因果。*

**DPO Loss**（老师第 9 页）：

$$
\boxed{\mathcal L_{\mathrm{DPO}}=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\left[\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right]\right)\right].}
$$

用一个数值例子从内往外算。数据是 $x=5+3=?$，winner 为 8，loser 为 7；假设只有两个输出：

| 回答 | 当前 Policy | 固定 Reference Policy |
|---|---:|---:|
| $y_w=8$ | 0.4 | 0.2 |
| $y_l=7$ | 0.6 | 0.8 |

取 $\beta=1$，Sigmoid 内的值为

$$
z=\beta\left(\log\frac{0.4}{0.2}-\log\frac{0.6}{0.8}\right)=\log2-\log0.75\approx0.981.
$$

因此偏好预测概率为 $\sigma(z)\approx0.727$，单条数据的 Loss 为 $-\log0.727\approx0.318$。这个 0.727 是预测人类偏好的概率，不是生成 8 的概率（生成概率仍为 0.4）。若当前模型与参考模型完全相同，两个概率比都是 1，Loss 为 $-\log0.5\approx0.693$。

固定参考模型时，降低 Loss 会提高 $\pi_\theta(y_w\mid x)/\pi_\theta(y_l\mid x)$，即鼓励 winner 相对于 loser 的概率提高。Sigmoid 单调递增，且 $-\log$ 单调递减，因此上述方向可以直接从公式看出。不能保证每次参数更新都使两个绝对概率严格一升一降。

#### 4.2.1 为什么 DPO 可以省掉 Reward Model？（数学来源）

这是原稿公式背后的推导。对固定 Prompt $x$，省略条件 $x$，展开经典 RLHF 的目标：

$$
\max_\pi\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)},\qquad\sum_y\pi(y)=1.
$$

使用 Lagrange Multiplier（拉格朗日乘子）$\lambda$ 处理概率归一化约束：

$$
\mathcal F=\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+\lambda(\sum_y\pi(y)-1).
$$

对任一个 $\pi(y)$ 求导，利用 $(p\log p)'=\log p+1$，令导数为 0：

$$
r(y)-\beta\left[\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1\right]+\lambda=0.
$$

整理并取指数，利用概率和等于 1，得到理想最优策略：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta},\qquad Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
$$

反过来表达 Reward：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

同一道题的两个回答共享 $Z(x)$，所以 Bradley–Terry Model 中的奖励差为

$$
r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

将理想策略用待训练的 $\pi_\theta$ 参数化，把这条奖励差代入偏好概率与负对数似然，就得到上面的 DPO Loss。**用当前 Policy 和 Reference Policy 的概率比表示隐式评分，省去了单独的 Reward Model。** 这是在 KL 正则化目标和 Bradley–Terry 偏好模型等假设下的推导，不代表实际的在线 RLHF 与离线 DPO 在任何情况下都完全等价。

#### 4.2.2 DPO Gradient（梯度）：外面的期望如何处理？

标准离线 DPO 从固定偏好数据集 $\mathcal D$ 取数据。若有 $N$ 条样本，目标为 $\mathcal L=\frac1N\sum_i\ell_i(\theta)$，所以 $\nabla_\theta\mathcal L=\frac1N\sum_i\nabla_\theta\ell_i$。这里期望的采样分布不随 $\theta$ 改变，直接对各条 Loss 求导取平均即可。不像 REINFORCE 那样需要对采样分布求导、再使用 Log-Derivative Trick。

先定义老师第 9 页的 Implicit Reward（隐式奖励）：

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},\qquad z=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l).
$$

单条损失为 $\ell=-\log\sigma(z)$。使用 Chain Rule：

$$
\nabla_\theta\ell=\frac{d\ell}{dz}\nabla_\theta z.
$$

先求外层，利用 $\sigma'(z)=\sigma(z)(1-\sigma(z))$：

$$
\frac{d\ell}{dz}=-\frac{\sigma'(z)}{\sigma(z)}=-(1-\sigma(z))=-\sigma(-z).
$$

参考模型固定，所以 $\nabla_\theta\log\pi_{\mathrm{ref}}=0$。再求内层：

$$
\begin{aligned}
\nabla_\theta z&=\beta[\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)]\\
&=\beta\left[\frac{\nabla_\theta\pi_\theta(y_w\mid x)}{\pi_\theta(y_w\mid x)}-\frac{\nabla_\theta\pi_\theta(y_l\mid x)}{\pi_\theta(y_l\mid x)}\right].
\end{aligned}
$$

这些概率关于参数的梯度用语言模型普通反向传播计算（与 2.4 的 Softmax 求导相同原理）。长回答的 log probability 是各 Token log probability 之和。已有回答直接用于计算，不需要反传通过离散 Sampling。

乘上外层导数，再对数据集取期望：

$$
\boxed{\nabla_\theta\mathcal L_{\mathrm{DPO}}=-\beta\mathbb E_{\mathcal D}\Big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].}
$$

![](Lecture21%20p09%20dpo%20gradient.png)

含义：当 winner 的隐式奖励低于 loser 时，前面的 Sigmoid 权重较大（例如差为 -2 时权重 0.881）；当模型已明显偏好 winner 时权重较小（差为 2 时权重 0.119）。最后按 Gradient Descent 更新 $\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L_{\mathrm{DPO}}$，只更新当前语言模型。

### 4.3 RLHF vs. DPO：优缺点与 A/B/C 的例子

| | 经典在线 RLHF | 标准离线 DPO |
|---|---|---|
| 共同起点 | 人类偏好数据 | 人类偏好数据 |
| 训练方式 | 先训练 Reward Model，再在线生成、评分、RL 更新 | 直接在偏好对上计算 Loss 并反传 |
| 计算开销 | 需要奖励模型和在线生成，流程复杂 | 无独立奖励模型与在线 RL 循环，较简单 |
| 新回答反馈 | 能对当前生成的回答评分 | 固定数据没有未标注回答的新偏好反馈 |

**DPO 的关键局限（Offline Data Coverage，离线数据覆盖）**：如果同一问题在整个输出空间有 A、B、C 三种回答，概率分别为 0.3、0.5、0.2，但人类偏好数据只有 A 优于 B，那么 DPO 直接优化 A 相对于 B 的概率比。C 并非不在 Policy 中，而是没有关于 C 的偏好标签。C 的概率仍可能因共享参数和 Softmax 归一化而间接变化，但模型不会自动获知 C 是否比 A 更好。

在线 RLHF 可采到 C 并由固定 Reward Model 评分。假设评分 $r(A)=2,r(B)=0,r(C)=5$，C 就能获得直接训练信号，倾向于提高生成 C 的概率（不是提高固定评分者的分数）。但 Reward Model 可能误判，且 KL 等因素也会影响更新。

所以标准离线 DPO 的弱点不是数学上无法改变未标注回答的概率，而是**缺乏对这些回答的直接新反馈与在线探索（Online Exploration）**。Online DPO 等扩展可重新生成、标注偏好对；两种算法并非谁必然完胜谁。

---

**原稿与补充资料**：第 1–2 页对应 1、2.1、3.1；第 3 页对应 2.2–2.4；第 5–6 页对应 3.2；第 7 页对应 4.1；第 8–9 页对应 4.2–4.3。老师原始 PDF 与截图均在附件中。

- [ScaleRL 原论文](https://arxiv.org/html/2510.13786v1)：训练计算规模、重要性权重、优势归一化与筛选。
- [InstructGPT 原论文](https://arxiv.org/html/2203.02155v1)：经典 SFT、奖励模型、RLHF 流程。
- [DPO 原论文](https://arxiv.org/html/2305.18290v3)：DPO 损失、最优策略推导与梯度。
