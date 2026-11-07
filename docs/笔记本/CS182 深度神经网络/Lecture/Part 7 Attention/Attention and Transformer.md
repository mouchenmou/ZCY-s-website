---
tags:
  - CS182
  - DeepLearning
  - Attention
  - Transformer
  - Lecture18
aliases:
  - CS182 Lecture 18
---

# CS182 · Lecture 18：Attention、Transformer、Embedding 与位置编码

> **对应材料**：课程手写 notes `18_fa26.pdf`，共 10 页。P1 为目录；P2–P4 为 Transformer 架构、GPT 和翻译；P5–P7 为 Embedding / De-embedding 及数值尺度；P8–P10 为位置编码、相对位置偏置和 RoPE。
>
> **笔记原则**：先解释术语和用途，再给一个能算的例子，最后才给公式。明确区分「课上 notes 写了什么」和「为了理解额外补充什么」。
>
> **图片说明**：本笔记所有插图均已保存在同级 `附件` 文件夹；图片是依照讲义和公开示意图重绘的中文教学图，而不是照搬网页的外链图片。打开 Markdown 不需要联网。

## 0. 先有一张全局地图：这节课究竟在解决什么问题？

你已经理解过 CS231n 形式的 Attention：给定 Query（Q），对 Key（K）打分，用分数的权重汇总 Value（V）。**但 Attention 只产生向量，不直接预测单词。**

整个语言模型的路径应当分清楚：

```text
文字 → Tokenizer 得到 Token ID
         ↓
Embedding：每个 ID 查出一个可学习向量
         ↓
位置相关信息（原始 Transformer 加位置向量；RoPE 在 Q/K 上操作）
         ↓
Transformer Block 第 1 层：Attention + 其他模块
         ↓
Transformer Block 第 2 层、……、第 N 层
         ↓
最终隐藏向量 h_t
         ↓
输出线性层（De-embedding / LM head）→ 词表大小的 logits
         ↓
Softmax：下一个 Token 的概率分布 → 选择一个 Token
```

**先认识这些新名词**：

| 词 | 首次出现时应理解为 | 别误解成 |
|---|---|---|
| Token | Tokenizer 切分出的文本单位，可以是词、词的一部分、标点等 | 一定是完整英文单词 |
| Token ID | 词表中某个 Token 的编号 | 这个词的「语义大小」 |
| Vocabulary / 词表 | 模型允许直接使用的一套 Token ID 集合 | 当前输入句子的 Token 数量 |
| Embedding | 将 Token ID 查成一个可学习向量 | 已经理解了整句语义 |
| Attention | 根据 Query 对 Key 的匹配，汇总 Value 的运算 | 预测单词或翻译的完整模型 |
| Transformer Block | Attention、FFN、残差、归一化等组成的一层处理单元 | 只有一条 Attention 公式 |
| Hidden state（隐藏向量） | 某个位置在某一层处理后的特征向量 | 新生成的 Token |
| Logits | 每个候选 Token 的未归一化分数 | 已经是概率 |
| Causal Mask | 限制某位置不能关注同一自回归序列里更后面的 Token | 所有 Attention 天生都会加的规则 |

---

# 第一部分｜先补 Attention 的地基
## 1. Q、K、V 到底是什么？分别从哪里来？

**定义**：

- **Query（查询向量，Q）**：当前输出位置用来匹配信息的特征，直觉是「这个位置要找什么」。
- **Key（键向量，K）**：各信息来源用于被匹配的特征，直觉是「我能不能被这个 Query 找到」。
- **Value（值向量，V）**：真正参与加权汇总的信息。

三者都**不是三种单词**；都是数值向量，是从输入特征经过不同可学习矩阵变换出来的。直觉上的「找什么」「提供什么」只是类比，不代表神经网络人为定义了对应语义。

### 1.1 为什么 CS231n 只写了 $W_K,W_V$，Transformer 却还有 $W_Q$？

CS231n 的一般 Attention 可以先有一组**独立给定的 Q**：

$$
K=XW_K,\qquad V=XW_V,\qquad C=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

这里 **Q 不必来自 X**，因此 Q 和 X 的行数可以不同。例如 Q 有 2 行，X 有 3 行。

而**标准 Self-Attention** 将 Q、K、V 都从同一组当前隐藏向量 $X$ 投影得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

写 $W_Q$ 的含义是让网络能学习 Query 的特征空间。**投影矩阵改变的是每行特征的内容或维度，不会因为右乘投影矩阵就改变行数。**

> 不要把「一般 Attention」和「标准 Transformer Self-Attention」的特殊前提混淆。一般情况下 $N_Q$ **可以不等于** $N_X$；同源 Self-Attention 中它们才自然相等。

### 1.2 精确看一遍维度

采用**行向量约定**：每个 Token / Query 占矩阵一行。

$$
Q\in\mathbb R^{N_Q\times d_k},\quad K\in\mathbb R^{N_X\times d_k},\quad V\in\mathbb R^{N_X\times d_v}.
$$

那么：

$$
\underbrace{QK^T}_{N_Q\times N_X}
\xrightarrow{\text{逐行 Softmax}}
\underbrace{A}_{N_Q\times N_X}
\xrightarrow{\times V}
\underbrace{C}_{N_Q\times d_v}.
$$

**这里的两个数量，功能不同**：

- $N_Q$：我要得到多少个查询位置的**输出**；
- $N_X$：每个 Query 有多少个 Key/Value 信息来源可供关注。

注意 $d_k$（Q/K 的特征维度）必须相等才能点积，但 $N_Q$ 和 $N_X$ 不要求相等。

## 2. 你之前最困惑的地方：$A$ 的一行和 $C=AV$ 的一行分别是什么？

例如 2 个 Query、3 个输入，给出一个**纯教学用的假设权重矩阵**：

$$
A=\begin{bmatrix}0.1&0.7&0.2\\0.6&0.1&0.3\end{bmatrix},\qquad
V=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}.
$$

第 1 行 $[0.1,0.7,0.2]$ 的意思是：**第一个 Query 对 3 个 Value 分别给了 10%、70%、20% 的权重**，并不是 3 个候选单词的预测概率。

乘上 V：

$$
\begin{aligned}
c_1&=0.1v_1+0.7v_2+0.2v_3=[3.2,4.2],\\
c_2&=0.6v_1+0.1v_2+0.3v_3=[2.4,3.4].
\end{aligned}
$$

![](附件/Pasted%20image%2020261106210102.png)

**结论**：$A$ 的每行是一个「从谁取多少信息」的权重分布；$C$ 的每行是某个 Query **实际取得的混合信息向量**。这一步没有创建新单词，更没有翻译。

**特别提醒**：这些数字只用于解释矩阵乘法，不表示真实模型一定关注哪个词。

## 3. 为什么标准 Self-Attention 有多少 Token 就有多少 Q？

设 `The cat eats` 的表示为：

$$X\in\mathbb R^{3\times512},\qquad W_Q\in\mathbb R^{512\times64}.$$

于是 $Q=XW_Q\in\mathbb R^{3\times64}$。一共有 **3 个 Query**，每个 64 维；K、V 同理。

原因是这个模块要**更新三个位置的表示**：The 要一个新向量，cat 要一个新向量，eats 也要一个新向量。它们分别使用自己的 Query 向其他位置取信息。

但这只是标准 Self-Attention 的设计，**不是 Attention 数学上永远要求 Q 与 X 行数相同**。Cross-Attention 的 Q 和 K/V 来自不同序列时，数量完全可能不同。

## 4. Q/K/V 到第二层时为什么还表示原先那个 Token？

这个地方很容易误会：「第一层 $c_i$ 已经混了其他词，第二层的 Q/K/V 还怎么代表原来那个词？」

准确说法是：**每个向量始终对应一个“位置”，但它的内容会随着层数不断更新**。

- 初始 $x_2^{(0)}$：第 2 个位置 `cat` 的 Embedding（以及模型采用的位置信息）；
- 第 1 层 Attention：第 2 个位置从允许关注的位置获取信息；
- 第 1 层完整 Block：输出更新后的 $x_2^{(1)}$；
- 第 2 层：重新计算 $q_2^{(2)}=x_2^{(1)}W_Q^{(2)}$、$k_2^{(2)}$、$v_2^{(2)}$。

所以第二层 Q/K/V **不是重新从原始单词取三个固定向量**，而是从上一层的**上下文隐藏表示**投影出来的。每一层一般使用自己的投影矩阵参数。

---

# 第二部分｜P2–P4：完整 Transformer、GPT 与翻译

## 5. Transformer 不是一条 Attention 公式：先认识一个 Block

老师 P2 图中 `Multi-Head Attention`、`Add & Norm`、`Feed Forward` 几个方框，各自是什么？

### 5.1 Multi-Head Attention（多头注意力）

**定义**：同时运行多组 Attention，各组有不同的 $W_Q,W_K,W_V$；最后把各组输出拼接并投影回模型需要的维度。

两个 Head 的例子：

$$
C^{(1)}=\operatorname{Attention}(XW_Q^{(1)},XW_K^{(1)},XW_V^{(1)}),
$$
$$
C^{(2)}=\operatorname{Attention}(XW_Q^{(2)},XW_K^{(2)},XW_V^{(2)}),
$$
$$
O=\operatorname{Concat}(C^{(1)},C^{(2)})W_O.
$$

这里的 `Concat` 是**拼接**：把两个输出向量首尾接起来，形成一个更长的特征向量。$W_O$ 是可学习的**输出投影矩阵**，用于将拼接后的特征变换为 Block 所需的维度。

它不是「一个 Token 生出更多 Token」，Head 的数目通常不会改变序列的行数。不同头可以学习不同的信息匹配模式，但没有人规定某头必须识别语法或主语。

### 5.2 Add（残差连接）

**定义**：将模块输入绕过某个运算，再和模块输出相加，典型形式：

$$y=x+f(x).$$

例如原先 $x=[1,2]$，Attention 分支的输出 $f(x)=[0.3,-0.4]$，则 $y=[1.3,1.6]$。

**用途**：让后续模块既能利用原始信息，又能利用本模块新算的信息；同时有利于深层网络训练时传播梯度。要求两边维度匹配，所以多头结果之后常有输出投影。

### 5.3 Norm（归一化层）

**定义**：通过某种规则调节向量的数值尺度，使网络训练更稳定。这里常见的是 **LayerNorm** 或某些架构里的 **RMSNorm**。

不要把它们混同：

- **LayerNorm**：对一个位置的特征先减去平均值，再按标准差缩放，通常还有可学习的缩放与偏移；
- **RMSNorm**：主要用均方根 RMS 来调整尺度，通常没有 LayerNorm 的减均值步骤；
- **RMS 值**：只是一个数值指标，不等同于一个完整网络层（后面的 P6 先算 RMS 值）。

老师的原始架构图是 **Add & Norm**，对应经典 Post-LN 的摆放；老师旁注特别说现代模型常将 Norm 放到 Attention 之前，即 **Pre-Norm**。**先理解功能，别把两种顺序硬混成一个公式。**

### 5.4 FFN（Feed Forward Network，逐位置前馈网络）

**定义**：最简单可以看作两个线性层加一个非线性激活函数：

$$\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2.$$

- Linear / 线性层：矩阵乘法（可加偏置），改变向量的特征；
- Activation / 激活函数 $\sigma$：如 ReLU，使整个操作不再仅是一次线性变换；
- **逐位置**：同一套 FFN 参数分别处理每一个 Token 位置，**不会直接像 Attention 那样在两个位置之间交换信息**。

因此一个好记的分工是：**Attention 负责「不同位置之间取信息」，FFN 负责「在每个位置内部加工信息」**。

### 5.5 一个 Block 的整体输出是什么？

简化（示意结构，不固定 Pre-Norm/Post-Norm 位置）为：

$$\boxed{X^{(\ell-1)}\xrightarrow{\text{Attention + 残差/Norm}}\text{中间表示}\xrightarrow{\text{FFN + 残差/Norm}}X^{(\ell)}}$$

![](附件/Pasted%20image%2020261106210108.png)

**注意**：你学的 $C=AV$ **还只是 Attention Head 的输出**。它之后通常还经过多个 Head 的拼接、输出投影、残差、归一化、FFN 等，才得到一整个 Block 的输出 $X^{(\ell)}$。

## 6. P2：原始 Transformer 的 Encoder 和 Decoder 是什么？

先看下面这张中文架构重绘图；然后对照老师 P2 的原图（左边 Encoder，右边 Decoder）。

![](附件/Pasted%20image%2020261106210101.png)

### 6.1 Encoder（编码器）做什么？

**定义**：读取一整段已经给定的输入序列，为每一个输入位置生成包含上下文信息的向量。

比如翻译英语 `I love cats` 为法语：

1. 输入的英文 Token 先通过 Embedding；
2. 加入位置信息（原始 Transformer 使用位置编码）；
3. 经过 **Encoder Block**：Self-Attention → Add & Norm → FFN → Add & Norm；
4. 同样的 Block 堆叠多层（P2 标的 `N×`），每层通常有不同的可学习参数；
5. 得到 $H_{\text{enc}}=[h_1,h_2,h_3]^T$。

此时 $h_1,h_2,h_3$ **仍是英文三个位置的向量**，不是三个法文词。Encoder 也不是必须直接给出某句「语义解释文本」。

Encoder 的 Self-Attention **通常不需要因果 Mask**：因为整个英文句子已经给定，处理 `I` 时可以参考后面的 `cats`。这与「生成时不能偷看目标法文」没有冲突。

### 6.2 Decoder（解码器）做什么？

**定义**：在自回归生成架构中，根据已知信息和已经生成的目标 Token，预测**下一个目标 Token**。

原始 Transformer Decoder Block 比 Encoder Block 多一种 Attention：

1. **Masked Self-Attention**：让已经生成的法文之间交换信息；因果 Mask 防止它们看到后面尚未生成的法文；
2. **Cross-Attention**：法文侧当前表示提出 Query，从英文 Encoder 提供的 Key/Value 中取信息；
3. **FFN**：逐位置加工；中间照样有残差、归一化等。

这里先定义 **Cross-Attention（交叉注意力）**：**Q 来自序列 A，K/V 来自另一组序列 B**。与 Self-Attention（Q/K/V 同源）相比，主要区别是信息来源不同，数学运算仍是 $\operatorname{softmax}(QK^T/\sqrt d)V$。

翻译时可以写成：

$$
Q=H_{\text{dec}}W_Q,\qquad K=H_{\text{enc}}W_K,\qquad V=H_{\text{enc}}W_V.
$$

若法文侧当前有 2 个 Query 位置、英文有 3 个 Token，则 Attention 矩阵可为 $2\times3$：**没有要求两边 Token 数一样**。若不特意设计源端限制，Cross-Attention 可让每个法文位置看完整英文。

## 7. P2 的 `Outputs (shifted right)` 究竟是什么意思？

**定义**：训练时把正确目标序列向右错开一个位置，作为 Decoder 的输入；每个位置要预测的标签则是下一个 Token。

例如目标法文简化为 `J' aime les chats <end>`（这里只拿词语作示意，真实 Tokenizer 可能拆得不同）：

| 当前位置给 Decoder 的输入 | 要预测的下一个 Token |
|---|---|
| `<start>` | `J'` |
| `J'` | `aime` |
| `aime` | `les` |
| `les` | `chats` |
| `chats` | `<end>` |

整个训练序列右移一格后，**所有位置可以在一次前向传播中并行计算损失**；但必须使用因果 Mask，不能让第一个位置看见后面的法文输入而作弊。

**推理（inference，实际生成）**时，正确法文还未知，只能从 `<start>` 开始逐个生成、把生成的 Token 接上、再继续预测。

### 一个容易误会的词：未来

Causal Mask 中的「未来」，是指**当前预测所不允许提前读取的后续目标 Token**，不是说现实世界未来发生的事件，也不是所有排在右边的输入词都必须被遮住。英文原句在翻译时**已经完整给定**。

## 8. 三种 Attention，到底什么时候 Mask？

![](附件/Pasted%20image%2020261106210104.png)

| 原始 Encoder–Decoder 的模块 | Q 来自 | K/V 来自 | 因果 Mask？ | 原因 |
|---|---|---|---|---|
| Encoder Self-Attention | 已知英文 | 已知英文 | **通常没有** | 整个输入句子已知，可以双向处理 |
| Decoder Self-Attention | 已有法文 | 已有法文 | **有** | 防止偷看后面的法文目标 |
| Decoder Cross-Attention | 已有法文 | Encoder 英文表示 | **通常没有** | 全部英文源句已知 |

Mask 实现上是在 Softmax 前给**不允许关注**的位置加 $-\infty$（实践中使用足够小的数等效实现），从而使其 Softmax 权重变为 0：

$$A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right),\qquad M_{t,i}=\begin{cases}0&\text{允许}\\-\infty&\text{不允许}\end{cases}.$$

> **绝对不要说「任何 Transformer 都天然有因果 Mask」**。Mask 是为任务和架构选用的规则；也还有 padding mask 等其他目的的 Mask，这里先讨论因果 Mask。

## 9. P3：为什么 GPT 只有 Decoder，也能翻译？

**Decoder-only** 不等于把 P2 右半边原封不动裁下来：GPT 主要保留 **带因果 Mask 的 Self-Attention、FFN、残差、归一化、重复堆叠与输出层**，没有原始 Encoder–Decoder 中那条从独立 Encoder 读取信息的 Cross-Attention。

### 9.1 GPT 的核心训练目标是「续写下一个 Token」

$$P(x_{t+1}\mid x_1,x_2,\dots,x_t).$$

比如现有输入是 `The cat eats`：

1. Embedding 得到 3 个向量；
2. 经过很多 Transformer Block（每层不断更新 3 个位置的向量）；
3. 最终得到 $h_1,h_2,h_3$；
4. **取最后一个位置 $h_3$**，它是 `eats` 位置最终获得的上下文表示；
5. $h_3$ 经过输出线性层 → Softmax，可能预测 `fish`；
6. 将生成的 `fish` 接到输入末尾，再继续生成。

![](附件/Pasted%20image%2020261106210103.png)

**关键区别**：Attention 输出 $C$ 和最终隐藏向量 $h_3$ 都不是字符串。真正形成「词表上的预测」是在**输出线性层 + Softmax**，而不是 Q/K/V 算出来那一刻。

### 9.2 那 GPT 为什么也能做英译法？

输入一条指令作为前缀，例如：

```text
Translate English to French: I love cats.
French:
```

GPT 把「翻译指令 + 完整英文 + 已输出的法文」视为**同一条 Token 序列的前缀**，继续逐 Token 预测后续的法文。它无需单独 Encoder，就可以通过 Self-Attention 读取前面的整句英文。

**你追问过的细节**：既然英文已经给定，为什么 GPT 的英文 Token 内部也要 Causal Mask？

- 从**翻译任务本身**看，英文内部并没有必须遮住后词的理由；
- 但标准 GPT 采用**统一的 Causal Self-Attention 规则**来处理整个序列，这是预训练和架构选择；
- 英文前面的 Token 虽看不到后面的英文，但开始生成法文的那些位置**可以看到全部前缀英文**；
- 也存在 **Prefix-LM** 等不同设计，使完整输入前缀内部双向关注，而输出部分仍因果生成。不要因此说 GPT 架构是翻译唯一做法。

| 比较 | 原始 Encoder–Decoder | 标准 GPT（Decoder-only） |
|---|---|---|
| 输入英文 | 单独 Encoder，通常双向 Self-Attention | 放在整个序列的前缀中，仍采用 Causal Self-Attention |
| 生成法文 | Decoder 使用自身因果 Self-Attention + 对英文的 Cross-Attention | 在同一序列中用因果 Self-Attention 读取英文前缀并生成法文 |
| 生成法文时是否逐 Token | 是 | 是 |
| 翻译一定需要 Encoder 吗 | 不一定 | 不需要独立 Encoder |

**本质**：两者是不同**模型架构**，翻译、问答、续写才是**任务**；不要强行一对一对应。

## 10. P4：双语翻译图到底在说明什么？

P4 图的英文侧是标准 Encoder（双向 Self-Attention），法文侧是 Decoder：

- 英文句子进入 Encoder 后，所有英文 Token 互相处理并得到 Encoder 输出；
- 法文 Decoder 的 Masked Self-Attention 只能看此前已生成的法文；
- 法文 Decoder 的 Cross-Attention 用法文 Query 读取英文 Encoder 的 Key/Value；
- 法文端最后的输出层预测**下一个法文 Token**，反复进行。

**不要误读两处图示**：上图里看到整个英文句子，不代表模型已经生成了整句法文；下图训练时右侧同时出现目标位置，不代表推理时可以偷看未来答案。

---
# 第三部分｜P5：Embedding Layer 与 De-embedding

## 11. 先区别三个数字：词表大小、这次输入长度、Embedding 维度

老师在 P5 的假设：

- **词表大小**（Vocabulary Size）：$V_{\text{vocab}}=32000$，意思是 Tokenizer 词表中有约 32,000 个可直接编号的 Token；
- **Embedding 维度**：$d_{\text{model}}=1000$，意思是用一个 1,000 维向量表示每个 Token 的初始特征；
- **序列长度** $N$：这次**实际输入**有多少个 Token，可以是 3,000，下次是 4,000，不会因此改变词表规模或 Embedding 参数的维度。

**你以前问的那个问题**：「输入 3,000 个 Token，下次 4,000 个，那 $W$ 不是要变吗？」

**不需要。**我们用一个固定的词表查向量；输入 3,000 个 Token 就查 3,000 次，输出 $3000\times1000$；输入 4,000 个就查 4,000 次，输出 $4000\times1000$。同一个 Embedding 表没有换。

![](附件/Pasted%20image%2020261106210105.png)

### 11.1 词表 32,000 是怎么确定的？

**定义 Tokenizer（分词器）**：一套把文本转换成 Token 序列、再转换成 ID 序列的规则。

不是老师「算出英语恰好有 32,000 个单词」。常见做法是**设计时指定词表规模**，使用 BPE 等方法从语料里学习常见文本片段；生僻词可以拆成多个子词或更小单位。

例如 `unbelievable` 不一定必须单独拥有一个 Token，它可能被分成几段。**Token 不必等于完整英文单词**，所以有限词表也可以处理大量不同字符串。不同模型会采用不同的词表大小；32,000 只是老师举例。

### 11.2 One-hot（独热向量）到底是什么？

**定义**：一个长度为 $V_{\text{vocab}}$ 的向量，**恰好一个位置是 1，其余全为 0**。这个 1 的位置用来标明 Token ID。

词表只有 4 个 Token 时：

$$
\text{cat}\mapsto[1,0,0,0]^T,\quad
\text{dog}\mapsto[0,1,0,0]^T,\quad
\text{fish}\mapsto[0,0,1,0]^T.
$$

词表 32,000 个 Token 时，每个 One-hot 就是 32,000 维，规则**完全一样**。它是一种**身份表示**，不是预先含有动物、语法等语义特征的向量。

## 12. 为什么还要从 32,000 维变成 1,000 维？是不是浪费计算？

这两个维度不是同一种含义：

- 32,000 维 One-hot 只是**编号**：告诉模型这次出现的是词表里的哪一种 Token；
- 1,000 维 Embedding 是**模型学习出来的特征表示**，供后面的 Q/K/V、FFN 等模块运算。

数学上当然可以把隐藏维度设成 32,000，但那样后续很多线性层也要处理极宽的向量，计算和参数开销可能很大。用较小的 $d_{\text{model}}$ 是**架构设计选择**，不是 One-hot 的数学定理。

### 12.1 老师 P5 的矩阵乘法实际上只是「查表」

老师用**列向量**记法（这一节与前面 $X$ 的行向量记法不同，千万别混）：

$$
\underbrace{W}_{1000\times32000}
\underbrace{x}_{32000\times1}
=\underbrace{y}_{1000\times1}.
$$

$x$ 是 One-hot，只有第 $j$ 位是 1：

$$Wx=0\cdot W_{:,1}+\cdots+1\cdot W_{:,j}+\cdots+0\cdot W_{:,32000}=W_{:,j}.$$

也就是说，**只选出 $W$ 的第 $j$ 列**。真正写代码时，可以直接用 Token ID 索引 Embedding 表，**不需要真的构造一个巨大的 One-hot 并做稠密矩阵乘法**。

你以前质疑：「先映射到 1,000 维，这一步不是也很麻烦吗？」——这里恰好因为 One-hot 只有一个非零位置，**可直接查表**，所以不是每个 Token 做 32,000×1,000 次真实乘法。

**重要维度约定**：老师采用列向量时 $W\in\mathbb R^{1000\times32000}$（每列是一个 Token 的 Embedding）；如果我们采用常见代码里的行向量查找表，则存储视角常写作 $E\in\mathbb R^{32000\times1000}$（每行一个 Token）。它们本质互为转置；不是两次不同变换。

## 13. P5 下半页：De-embedding 是什么？

**定义**：把 Transformer 最后某个位置的隐藏向量，转换为词表中**所有候选 Token 的分数**。也称输出线性层或（在语言模型中）LM Head。

假设最后一个位置的隐藏向量是 $h\in\mathbb R^{1000}$（这里改用**列向量**方便和老师 P7 对接）：

$$
\underbrace{z}_{32000\times1}
=\underbrace{W_{\text{out}}}_{32000\times1000}
\underbrace{h}_{1000\times1}.
$$

得到 $z_1,\ldots,z_{32000}$，它们叫 **logits（未归一化分数）**。每个分数对应词表里的一个 Token，不是说这个位置突然拥有了 32,000 个输入 Token。

接着对**这 32,000 个候选分数**做 Softmax：

$$
P(\text{next token}=j)=\frac{e^{z_j}}{\sum_{k=1}^{32000}e^{z_k}}.
$$

模型再根据概率分布选一个 Token，加入序列并继续生成。

**两个 Softmax，千万不要混**：

| Softmax | 对谁归一化 | 输出代表什么 |
|---|---|---|
| Attention 内部 Softmax | 某个 Query 对所有 Key 的分数 | 对各信息来源的**关注权重** |
| 最终 LM Head Softmax | 下一步词表里所有 Token 的 logits | **下一个 Token 的预测概率** |

这就是此前「明明 Attention 只算现有 Token 的相互影响，为什么能预测新词」的完整答案。

---

# 第四部分｜P6–P7：RMS 数值尺度与 $1/\sqrt{1000}$ 的推导

## 14. P6 开头：RMS（Root Mean Square，均方根）的定义

**RMS 是一个数，不是网络层。** 对一个含 $n$ 个数字的向量：

$$
\boxed{\operatorname{RMS}(x)=\sqrt{\frac1n\sum_{i=1}^n x_i^2}}.
$$

步骤是「平方 → 取平均 → 开平方」。例如：

$$
\operatorname{RMS}([1,-2,2])=\sqrt{\frac{1+4+4}{3}}=\sqrt3.
$$

它衡量一组数的**典型绝对大小**，正负号不会互相抵消。

**区分另一个概念**：$L_2$ 范数（欧几里得长度）定义为 $\|x\|_2=\sqrt{\sum_i x_i^2}$。二者关系为：

$$\operatorname{RMS}(x)=\frac{\|x\|_2}{\sqrt n}.$$

因此「RMS 为 1」和「$L_2$ 范数为 1」完全不是同一个条件。

## 15. P6 第一行为什么出现 $1/\sqrt{32000}$？

输入的 One-hot 只有一个 1 和 31,999 个 0：

$$
\operatorname{RMS}(x)
=\sqrt{\frac{1^2+0^2+\cdots+0^2}{32000}}
=\boxed{\frac{1}{\sqrt{32000}}}.
$$

**这只是算 One-hot 向量的 RMS，不是某个 Transformer 特殊公式。**

老师随后写：

> We want RMS norm for input to attention heads to be 1.

正确理解：老师希望**送到后续 Attention 的 Embedding 表示具有合适尺度**，例如 RMS 约为 1。不要误读成原始 One-hot 必须 RMS 为 1，更不要误读成 $Wx$ 这个向量要变成数字 1。

## 16. P6：怎么从 $y=Wx$ 推到「W 每一列的 RMS 约为 1」？

仍使用老师的**列向量**记号：

$$W\in\mathbb R^{1000\times32000},\qquad x\in\mathbb R^{32000},\qquad y=Wx\in\mathbb R^{1000}.$$

若 $x$ 的第 $j$ 位是 1，则 $y=W_{:,j}$：输出就是 **W 的第 $j$ 列**。

所以，如果希望**每一种 Token** 查出的 1,000 维向量 RMS 都约为 1，等价于希望 W 的**每一列** RMS 约为 1：

$$
\operatorname{RMS}(W_{:,j})
=\sqrt{\frac1{1000}\sum_{i=1}^{1000}W_{ij}^2}\approx1.
$$

平方移项：

$$
\boxed{\sum_{i=1}^{1000}W_{ij}^2\approx1000.}
$$

换句话说，此时每列的 $L_2$ 范数约为 $\sqrt{1000}$，**不是 1**。

### 16.1 你追问：「老师有讲具体怎么做到它等于 1 吗？」

**按 P6 纸面内容：老师只提出目标与逻辑，没有写具体参数约束或初始化算法。**以下是额外补充，不应误记成课上规定。

如果已经有一列非零随机权重 $w$，数学上可以直接缩放：

$$
\boxed{w'=\frac{w}{\operatorname{RMS}(w)}}
\quad\Longrightarrow\quad
\operatorname{RMS}(w')=1.
$$

这是**一种可以做到的方法**；但实际模型未必在训练时强制每列永远 RMS=1，也可能通过初始化选择、网络中的归一化层等机制控制数值尺度。要区分「期望规模」「某种数学可行做法」与「具体模型实现」。

### 16.2 RMSNorm 是另外一个名字，别混

如果真的使用 **RMSNorm 层**，常见形式是：

$$\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_{i=1}^d x_i^2+\epsilon}}.$$

其中 $\epsilon$ 是防止除零的小正数，$\gamma$ 是可学习缩放参数，$\odot$ 表示逐元素相乘。P6 的重点是**RMS 的尺度分析**，并没有在图中完整推导此层的实现；不要把「老师希望 RMS=1」写成「老师强制插入了 RMSNorm 层」。

## 17. P7：为什么 De-embedding 权重每个元素的典型大小是 $1/\sqrt{1000}$？

这次一定从公式推出来，而且说清楚**需要的假设**。

### 17.1 先确认 P7 的每个符号

老师写：

$$\boxed{\beta=Q\alpha}$$

- $\alpha\in\mathbb R^{1000}$：经过前面网络得到的隐藏向量；
- $\operatorname{RMS}(\alpha)\approx1$：老师采用的尺度假设；
- $Q\in\mathbb R^{32000\times1000}$：**这里是输出 De-embedding 矩阵，完全不是 Attention 的 Query**；
- $\beta\in\mathbb R^{32000}$：词表中所有候选 Token 的 logits。

老师担心：每个 $\beta_i$ 都是 1,000 项乘积的**求和**，如果各项不控制大小，logits 可能过大，导致 Softmax 特别尖锐。

### 17.2 单独展开一个 Logit

$$
\beta_i=\sum_{j=1}^{1000}Q_{ij}\alpha_j
=Q_{i1}\alpha_1+Q_{i2}\alpha_2+\cdots+Q_{i,1000}\alpha_{1000}.
$$

P6 的输入是 One-hot，只取 W 的一列；这里的 $\alpha$ 通常是**稠密向量**，所以真的有很多项参加求和。这正是两页结论看起来不同的原因。

### 17.3 先把 RMS=1 改写成平方和

按定义：

$$\operatorname{RMS}(\alpha)=1
\Longrightarrow \frac1{1000}\sum_{j=1}^{1000}\alpha_j^2=1
\Longrightarrow\boxed{\sum_{j=1}^{1000}\alpha_j^2=1000}.$$

这个等式是精确的（如果 RMS 恰为 1）。但**光有这个条件，仍不能保证任意 Q 乘完后 $\beta_i$ 小**。必须对 Q 的正负分布、相关性等加一些假设。

### 17.4 用方差真正推导 $1/\sqrt{1000}$（本笔记补充的完整数学解释）

**方差（Variance）先定义**：对随机变量 $Z$，

$$\operatorname{Var}(Z)=\mathbb E[(Z-\mathbb E[Z])^2].$$

它衡量数值围绕平均值的波动规模；标准差是方差开平方。

为了分析**初始化时**的典型数值，我们把 Q 这一行的权重视为随机变量，假设：

1. $\mathbb E[Q_{ij}]=0$（平均为 0，正负可以抵消）；
2. 不同 $j$ 的 $Q_{ij}$ 相互独立；
3. 每个权重的方差相同：$\operatorname{Var}(Q_{ij})=\sigma^2$；
4. 把 $\alpha$ 当成一个暂时固定的向量，并假设 $\operatorname{RMS}(\alpha)=1$。

在这些假设下，单个 Logit 的期望：

$$
\mathbb E[\beta_i\mid\alpha]
=\sum_j\alpha_j\mathbb E[Q_{ij}]=0.
$$

因为独立随机变量求和时**方差相加**（常数倍会把方差乘上常数的平方），所以：

$$
\begin{aligned}
\operatorname{Var}(\beta_i\mid\alpha)
&=\operatorname{Var}\left(\sum_{j=1}^{1000}Q_{ij}\alpha_j\right)\\
&=\sum_{j=1}^{1000}\alpha_j^2\operatorname{Var}(Q_{ij})\\
&=\sigma^2\sum_{j=1}^{1000}\alpha_j^2\\
&=1000\sigma^2.
\end{aligned}
$$

现在老师的目标可以具体化为：**希望输出 Logit 的标准差（典型波动规模）约为 1**，即方差约为 1：

$$1000\sigma^2\approx1
\Longrightarrow \sigma^2\approx\frac1{1000}
\Longrightarrow \boxed{\sigma\approx\frac1{\sqrt{1000}}}.$$

所以在这种典型**零均值随机初始化的尺度分析**里，每个 $Q_{ij}$ 的**标准差（典型数值大小）**应约为：

$$\boxed{|Q_{ij}|\text{ 的典型量级}\sim\frac1{\sqrt{1000}}.}$$

**注意不是全部权重都固定取正的 $1/\sqrt{1000}$！** 如果 $\alpha$ 的所有分量都为 1，而 Q 每一项也都取正的 $1/\sqrt{1000}$，则 $\beta_i=1000/\sqrt{1000}=\sqrt{1000}$，反而很大。这正说明正负抵消/随机性假设不可省略。

### 17.5 老师笔记里还有一种快速思路

设 Q 某行的**平方和**约为 1：

$$\sum_{j=1}^{1000}Q_{ij}^2\approx1.$$

如果各项平方大小差不多，平均每项约为 $1/1000$，因此：

$$Q_{ij}^2\sim\frac1{1000}\quad\Longrightarrow\quad |Q_{ij}|\sim\frac1{\sqrt{1000}}.$$

这种写法直观地体现「Q 一行的 $L_2$ 范数约为 1」，但要从它进一步推出 Logit 的典型大小，也需要类似上面的随机方向、独立或不相关假设。

**千万注意：P6 要求列向量的 RMS 约为 1；P7 推的是行向量的 $L_2$ 范数约为 1。不是一个条件。**

| | P6：Embedding | P7：De-embedding |
|---|---|---|
| 矩阵大小（老师列向量约定） | $1000\times32000$ | $32000\times1000$ |
| 输入 | One-hot：只有一个非零元素 | 稠密的 1,000 维隐藏向量 |
| 运算本质 | **取出一列** | **1,000 项乘积相加** |
| 期望的尺度 | 每个输出向量 RMS $\approx1$ | 每个输出 Logit 的典型大小 $\approx1$ |
| 权重典型尺度 | 每列 RMS $\approx1$（列的 $L_2$ 约 $\sqrt{1000}$） | 每行 $L_2\approx1$（元素典型量级 $\sim1/\sqrt{1000}$） |

> **课内外边界**：P6 老师写了目标 RMS、One-hot 选择一列、列尺度结论；P7 老师给出行平方和与 $1/\sqrt{1000}$ 的近似启发式。这里补上的零均值、独立、方差推导是为了说明**为什么这种尺度有意义**，不是声称老师在 P7 逐行写出了这些概率论假设。

---
# 第五部分｜P8–P10：Positional Encoding、Relative Bias、RoPE

## 18. P8：Attention 已经有 Q/K/V，为什么还要位置信息？

回到原始 Attention：

$$s_{t,i}=\frac{q_t^Tk_i}{\sqrt{d_k}}.$$

- $t$ 是**当前 Query 所在位置**；
- $i$ 是**被关注的 Key 所在位置**；
- $s_{t,i}$ 是第 $t$ 个 Query 对第 $i$ 个 Key 的匹配分数；
- $d_k$ 是 Q/K 的特征维度，**不是 Token 个数**；
- $\sqrt{d_k}$ 是缩放因子，用于控制较高特征维度时点积的典型大小。

如果 Q/K 只从词的内容生成，且我们没加入位置相关信息，那么公式并没有直接读取位置编号：

`cat chases dog` 和 `dog chases cat` 的三个内容向量只是交换了排列。如果 **没有位置机制、也没有 Mask**，中间 `chases` 的 Query 对这些 Key 的得分会跟着排列交换，Value 加权和的结果不因排列顺序改变。这种 Attention 无法从结果里单独知道 `cat` 在左还是在右。

> 这个结论有前提：「没有显式位置相关机制、没有其他破坏对称性的结构」。对 GPT 那样采用 Causal Mask 的模型，Mask 本身会带来一定的顺序结构，所以不能不加前提地说「任何无位置编码模型都完全分不出顺序」。

**Positional Encoding（位置编码）的定义**：让模型在处理 Token 时利用它们**所在位置或相对距离**的信息。P8–P10 介绍了四种设计思路。

## 19. 方法一：原始正弦位置编码（Sinusoidal Positional Encoding）

**它要解决的问题**：两个内容相同的 Token 出现在不同位置时，不能只凭同一套内容 Embedding 认为它们处于相同环境。

**做法**：给位置 $t$ 构造一个位置向量 $p_t$，加到 Token Embedding $x_t$ 上：

$$\boxed{x'_t=x_t+p_t}.$$

例如 $x_t=[0.1,0.2,0.3,0.4]$，第 2 个位置的 $p_2$ 假设为 $[0.3,0,-0.1,0.2]$（仅示意），则输入 Attention 的向量为 $[0.4,0.2,0.2,0.6]$。

那么 $p_t$ 怎么构造？原始 Transformer 使用不同频率的正弦和余弦（假设位置维度为 $d_{\text{model}}$，$r$ 表示第几个二维特征对）：

$$
\boxed{\begin{aligned}
PE(t,2r)&=\sin\left(\frac{t}{10000^{2r/d_{\text{model}}}}\right),\\
PE(t,2r+1)&=\cos\left(\frac{t}{10000^{2r/d_{\text{model}}}}\right).
\end{aligned}}
$$

**不需要背这个公式才能理解 P8。**你只需要知道：不同位置产生有规律的不同向量，而且不需要为每个位置单独存储一组随机可训练参数。

**与课上备注的关系**：P8 老师旁边写了 `No longer used`，应读成「这种原始固定正弦加法编码已不是许多现代 LLM 的主流选择」，**不是**「正弦余弦已经彻底消失」。例如 P10 的 RoPE 仍然用 sin/cos 构造旋转。

## 20. 方法二：NoPE（No Positional Encoding）

**定义**：不额外加入显式的绝对位置向量或相对位置偏置，让网络从其他结构（如 Causal Mask）及训练中利用可能的顺序信号。

公式依然从 $Q=XW_Q,K=XW_K,V=XW_V$ 出发，不额外加 $p_t$ 或 $b_{t-i}$。

**容易误解**：NoPE 不是「自然语言不需要顺序」，也不是「模型绝对不可能得到任何顺序信息」。它是一个可尝试的架构选择，效果依赖于具体任务和模型结构。

## 21. 方法三：Learned Relative Positional Encoding（可学习相对位置偏置）

### 21.1 先定义「相对位置」

假设 `The cat eats fish` 四个 Token 的位置分别为 $1,2,3,4$。

当前在处理位置 $t=3$ 的 `eats`：

| 被关注的 Key | Key 位置 $i$ | $t-i$ | 解释 |
|---|---:|---:|---|
| The | 1 | 2 | 在 Query 前两格 |
| cat | 2 | 1 | 在 Query 前一格 |
| eats | 3 | 0 | 同一个位置 |
| fish | 4 | −1 | 在 Query 后一格 |

**相对位置**就是位置差 $t-i$，而不是「这个词是第几个绝对位置」。正负号区分方向。若是标准 GPT 因果 Mask，最后一行（未来的 fish）会被禁止关注；这里先使用一般 Attention 讲原理。

### 21.2 先定义 Bias（偏置）

**Bias 在这里就是一个加在 Attention Score 上的数字**。对每一种相对距离 $\Delta=t-i$，我们关联一个可学习参数 $b_\Delta$。

比如：

- $b_0$：关注自己时加的分；
- $b_1$：关注前一个位置时加的分；
- $b_2$：关注前两个位置时加的分。

这些数字**不是我们手工规定的语法规则**，而是网络在训练中学习的；不同头也可以使用不同的偏置设置。

### 21.3 把 Bias 加到原本的 Attention Score

原本：

$$s_{t,i}=\frac{q_t^Tk_i}{\sqrt{d_k}}.$$

现在：

$$\boxed{s'_{t,i}=s_{t,i}+b_{t-i}}.$$

再对当前 Query 的整行分数做 Softmax：

$$
\boxed{p_{t,i}=\frac{\exp(s_{t,i}+b_{t-i})}{\sum_j\exp(s_{t,j}+b_{t-j})}}.
$$

**每个下标都要能解释**：分子是「当前 Query t 对 Key i」的权重；分母是对「同一个 Query t 的所有允许 Key j」进行求和。因此分母里的 Key 下标必须变成 $j$，相对偏置也写为 $b_{t-j}$。老师 P9 手写式子可以按这一严格记号理解。

### 21.4 用三个数字检验它怎么改变 A

现在处理 `eats`（$t=3$），暂不考虑 future 的 `fish`。假设三个内容 Score 全为 0，未加 Bias 时：

$$\operatorname{softmax}([0,0,0])=[1/3,1/3,1/3].$$

假设模型学到：$b_2=0$、$b_1=\ln2$、$b_0=0$。这意味着只有「前一个位置」得到额外的 $\ln2$ 分（注意**完全是假设的参数**）。则：

$$s'=[0,\ln2,0]$$

$$
\begin{aligned}
p'&=\operatorname{softmax}([0,\ln2,0])\\
&=\frac{[1,2,1]}{1+2+1}\\
&=\boxed{[0.25,0.50,0.25]}.
\end{aligned}
$$

![](附件/Pasted%20image%2020261106210106.png)

原本三个位置的 Value 各占 $1/3$，现在中间那个 `cat` 的 Value 占了 $1/2$，所以 **Attention Matrix A 变了，随后 $C=AV$ 也变了**。

这就是 P9 老师用醒目颜色写的 `This changes the attention matrix!`，以及 `learn these!` 指的**学习 $b$ 这些偏置**。

**补充：Mask 与 Relative Bias 不冲突。**对于禁止关注的位置，仍然要先后应用适当的 Mask，使其最后权重为 0。位置 Bias 的存在不能让 Token 偷看后面的目标答案。

## 22. 方法四：RoPE（Rotary Positional Embedding，旋转位置编码）

### 22.1 它与 P9 的区别是什么？

P9：**直接给分数加一个位置相关数字**。

$$s'_{t,i}=s_{t,i}+b_{t-i}.$$

P10 的 RoPE：**先根据位置旋转 Q 和 K，然后再点积**：

$$\boxed{s'_{t,i}=\frac{(R_tq_t)^T(R_ik_i)}{\sqrt{d_k}}}.$$

注意：RoPE 不是在最后预测 Token 时改变词表，也不是转动 $V$；它是在 Attention **计算 Score 的阶段**给 Q/K 注入位置相关变化。

### 22.2 什么叫「二维旋转矩阵」？

**定义**：在平面上将向量逆时针旋转角度 $\theta$ 的线性变换：

$$
\boxed{R(\theta)=\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}}.
$$

先不考虑 Token。向量 $q=[1,0]^T$ 原本向右：

$$
R(90^\circ)\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0&-1\\1&0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0\\1\end{bmatrix}.
$$

也就是逆时针转 90° 后向上。**向量长度不变，只是方向改变。**

### 22.3 怎样把「位置」变成「旋转角度」？

RoPE 把位置 $t$ 和角度联系起来。二维最简单的设定是：

$$\theta_t=t\omega,\qquad R_t=R(t\omega).$$

其中 $\omega$ 是每往后一个位置增加的角度（更准确说是每位置的旋转频率），我们先假设 $\omega=30^\circ$。

- 第 1 个位置的向量旋转 30°；
- 第 2 个位置旋转 60°；
- 第 3 个位置旋转 90°。

![](附件/Pasted%20image%2020261106210107.png)

此时第 $t$ 个 Query 和第 $i$ 个 Key 变成：

$$q'_t=R_tq_t,\qquad k'_i=R_ik_i.$$

**为什么这样有相对位置信息？**因为两个旋转角度相减：

$$t\omega-i\omega=(t-i)\omega.$$

比如位置 3 和 1 相距 2，位置 4 和 2 也相距 2；相对角度都是 60°。但只是直观上像「夹角」还不够，下面要用矩阵严格推出。

### 22.4 老师 P10 最核心的等式：逐步推导

老师 P10 写了：

$$\langle R_tq_t,R_ik_i\rangle.$$

这里 $\langle a,b\rangle$ 表示内积，在二维实向量里就是普通**点积**：

$$\langle a,b\rangle=a^Tb.$$

所以：

$$\begin{aligned}
\langle R_tq_t,R_ik_i\rangle
&=(R_tq_t)^T(R_ik_i)\\
&=q_t^TR_t^TR_ik_i.
\end{aligned}$$

**为什么第二行成立？**因为 $(AB)^T=B^TA^T$，转置乘积时**顺序反转**。

接下来需要两个旋转矩阵的基础性质：

**性质 A：旋转矩阵的转置等于反方向旋转。**

$$R(\theta)^T=R(-\theta).$$

可以直接核对：

$$
R(\theta)^T
=\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}
=R(-\theta),
$$

这里用了 $\cos(-\theta)=\cos\theta$ 和 $\sin(-\theta)=-\sin\theta$。

**性质 B：连续旋转两个角度，相当于旋转角度之和。**

$$R(\alpha)R(\beta)=R(\alpha+\beta).$$

可以理解为先转 $\beta$，再转 $\alpha$，方向最后一共转了 $\alpha+\beta$；也可以通过矩阵相乘和三角函数和角公式验证。

现在代入 $R_t=R(t\omega)$、$R_i=R(i\omega)$：

$$
\begin{aligned}
R_t^TR_i
&=R(t\omega)^TR(i\omega)\\
&=R(-t\omega)R(i\omega)\\
&=R((i-t)\omega).
\end{aligned}
$$

因此：

$$
\boxed{\begin{aligned}
\langle R_tq_t,R_ik_i\rangle
&=q_t^TR_t^TR_ik_i\\
&=q_t^TR((i-t)\omega)k_i.
\end{aligned}}
$$

**关键结论**：最后的相对旋转矩阵只依赖两个位置之差 $i-t$。虽然 Q 和 K 分别按照自己的**绝对位置**旋转，但点积运算中就出现了**相对位置**。

这里符号是 $i-t$，P9 使用了 $t-i$，二者只差一个负号，源自约定/展开顺序；只要公式里的定义始终一致，不是算法错误。

**不要过度理解成「只要相距相同，Score 就必然相同」**：最后还有 $q_t$ 与 $k_i$ 自身的内容。准确说法是：**当内容固定时，位置的作用通过相对位移进入点积**。

### 22.5 实际的 Q/K 有很多维，为什么老师只写 $d=2$？

因为二维最容易展示旋转。对高维（例如 64 维）Q/K，可以按照实现约定把特征分成 32 个二维分量对：

$$[q_1,q_2],\ [q_3,q_4],\ \ldots,\ [q_{63},q_{64}].$$

对每一对分别用二维旋转矩阵处理；不同二维对通常对应**不同旋转频率**。于是完整 $R_t$ 可以看成由许多二维旋转块构成的矩阵。无论采用哪种等价的维度排列实现，核心性质仍是**相对旋转依赖位置差**。

RoPE **不增加 Token 数**，也**不改变 Q/K 的向量维度**，只改变用于算 Score 的数值。后续仍然是：

$$
A=\operatorname{softmax}\left(\frac{Q'K'^T}{\sqrt{d_k}}+M\right),\quad C=AV.
$$

这里 $Q',K'$ 是已经施加 RoPE 的 Query/Key。**V 通常不施加同样的 RoPE 旋转**；Causal Mask 也不会因为有了 RoPE 而消失。

## 23. 四种位置方案放在一起，再也不要混

| 方法 | 新概念/直觉 | 具体在何处动手脚？ | 影响 |
|---|---|---|---|
| 原始 Sinusoidal | 每个绝对位置有一个固定 sin/cos 向量 | Attention 前将 $p_t$ 加到输入表示上 | Q/K/V 等后续结果都可能受到影响 |
| NoPE | 不显式额外提供位置参数 | 不加位置向量/偏置 | 仍可能从 Mask 等结构获得一定顺序信号 |
| Learned Relative Bias | 距离 $t-i$ 对应一个可学习数 $b_{t-i}$ | **Score** 上直接加偏置 | 改变 Attention Matrix A |
| RoPE | Q/K 随位置旋转 | 计算 Score 前旋转 **Q 和 K** | 通过点积体现相对位置 |

---

# 第六部分｜把整堂课连成一条线（最重要的一页）

## 24. 用 GPT 预测 `The cat eats` 后面的 Token，完整走一次

现在假设词表固定有 32,000 种 Token，模型隐藏维度 1,000，输入恰好 3 个 Token。

**Step 1：Token ID → Embedding**

输入序列长度 $N=3$，得到：

$$X^{(0)}\in\mathbb R^{3\times1000}.$$

其中 **3 是本次输入 Token 数**；32,000 是**整个模型词表大小**；1,000 是**每个 Token 向量的特征维度**。三者千万不要混！

**Step 2：位置机制 + 第一层 Attention**

从 $X^{(0)}$ 产生 Q/K/V；若采用 RoPE，就对 Q/K 旋转；若采用 GPT 因果 Mask，就禁掉未来位置。计算 Attention：

$$A^{(1)}=\operatorname{softmax}\left(\frac{Q^{(1)}K^{(1)T}}{\sqrt{d_k}}+M\right),\quad C^{(1)}=A^{(1)}V^{(1)}.$$

这是 3 个位置的信息汇总。**这里并没有预测第 4 个 Token**。

**Step 3：完整第 1 个 Block + 后续 Block**

多头拼接与输出投影、Add/Norm、FFN 等得到 $X^{(1)}$；第 2 层以 $X^{(1)}$ 为输入重新计算新的 Q/K/V……直到第 N 层，得到：

$$H=\begin{bmatrix}h_1\\h_2\\h_3\end{bmatrix}\in\mathbb R^{3\times1000}.$$

三行**依然对应 The、cat、eats 的三个位置**。$h_3$ 不是 `eats` 这个字符串，而是 `eats` 位置的最终上下文特征。

**Step 4：预测真正开始的地方**

取 $h_3$，用输出线性层得到词表大小的分数，再 Softmax：

$$z=h_3W_{\text{out}}\in\mathbb R^{32000},\qquad p=\operatorname{softmax}(z).$$

注意上面是**行向量**形式，因此 $W_{\text{out}}\in\mathbb R^{1000\times32000}$；在 P7 的老师**列向量**形式里，矩阵会转置成 $32000\times1000$。两种写法都正确，记号选哪一种必须前后一致。

假设模型最后选出了 Token `fish`，才把它**追加**到上下文：

```text
The cat eats  →  预测 fish  →  The cat eats fish
```

然后利用新的前缀继续预测。这就叫 **Autoregressive Generation（自回归生成）**：根据已有 Token，逐步生成下一 Token。实际可用 KV Cache 减少重复计算历史 Key/Value，但那是进一步的推理优化，不属于本节核心。

## 25. 一份针对我们之前所有疑问的纠错清单

| 容易混淆的说法 | 正确说法 |
|---|---|
| Q 必须有和 X 一样多的行 | 一般 Attention 不要求；**Self-Attention 同源投影**时才一样多 |
| 给 Q 乘 $W_Q$ 会改变 Query 数量 | 右乘投影矩阵通常只改变**每行特征维度/内容**，行数不变 |
| Q/K/V 是三个分别代表一个单词的对象 | 它们是从**当前层隐藏表示**投影的三种特征，不是三个新词 |
| $A$ 的一行已经包含需要的信息 | $A$ 只是**权重**；$C=AV$ 的一行才是信息汇总后的向量 |
| Attention 一算完就能预测新 Token | 还要经历完整 Block、多层处理、LM Head 与最终 Softmax |
| 第二层还把原始 Embedding 直接当 Q/K/V | 第二层从**第一层更新后的隐藏表示**重新投影 |
| Transformer 的所有 Attention 都有 Causal Mask | 不对。原始 Encoder Self-Attention、Cross-Attention 通常没有；自回归 Decoder Self-Attention 有 |
| GPT 因为不翻译，所以不用 Encoder | GPT **也可以翻译**；是否使用 Encoder 是架构选择，而非任务规定 |
| 词表有 32,000 意味着一次输入 32,000 个 Token | 词表大小与**本次输入序列长度**是独立数字 |
| 32,000 维 One-hot → 1,000 维要做巨大的稠密乘法 | 实际 Embedding **按 ID 查表**即可 |
| P6 和 P7 都是让权重 RMS=1 | P6 针对列 Embedding 的 RMS；P7 讨论输出矩阵一行的**平方和 / L2 尺度** |
| 每个 P7 权重都应该等于正的 $1/\sqrt{1000}$ | 应理解为**零均值、近似独立权重初始化下的典型大小** |
| Relative Bias 和 RoPE 是同一个东西 | 一个是**Score 加数**，一个是**旋转 Q/K 再点积** |
| RoPE 替代了因果 Mask | 没有。位置机制与 Mask 解决不同问题 |

## 26. 自测题：应该可以不看答案先做

**题 1**：词表 32,000；当前输入 7 个 Token；Embedding 维度 1,000。输入 Embedding 后的矩阵形状？如果下次输入变成 11 个 Token，Embedding 参数需要变吗？

> [!note]- 参考答案
> 输出形状 $7\times1000$；下次是 $11\times1000$；固定词表对应的 Embedding 参数大小不变。老师列向量记法 $W$ 为 $1000\times32000$，查表记法为 $32000\times1000$。

**题 2**：给定 $Q:2\times64$、$K:5\times64$、$V:5\times128$。$A$ 和 $C$ 各是什么形状？

> [!note]- 参考答案
> $A:2\times5$；$C=AV:2\times128$。2 个 Query、5 个信息来源，最终每个 Query 得到一个 128 维信息向量。

**题 3**：为什么 `The cat eats` 的 $C=AV$ 不是对 `fish` 的预测？

> [!note]- 参考答案
> $C$ 的每行仍是已有位置的信息向量，不是对词表概率。要经过一个或多个完整 Block 及输出层，取最后位置隐藏向量，投影为词表 logits，再 Softmax 才得到预测分布。

**题 4**：在原始英译法 Encoder–Decoder 里，Encoder 的英文 Self-Attention 和 Decoder 的法文 Self-Attention 哪个有因果 Mask？

> [!note]- 参考答案
> Encoder 英文 Self-Attention 通常没有；Decoder 自回归法文 Self-Attention 有。法文→英文的 Cross-Attention 通常不需要因果 Mask，因为英文源句已完整给定。

**题 5**：如果 $x$ 是 32,000 维 One-hot，它的 RMS 是多少？若 Embedding 输出 1,000 维且 RMS 恰为 1，输出向量平方和是多少？

> [!note]- 参考答案
> $1/\sqrt{32000}$；输出平方和为 $1000$，因为 $\operatorname{RMS}(y)^2=\sum_i y_i^2/1000=1$。

**题 6**：在 P7 的随机初始化假设下，若输入 $\alpha$ RMS=1，输出 $\beta_i=\sum_{j=1}^{1000}Q_{ij}\alpha_j$，且 $Q_{ij}$ 相互独立、均值为 0、方差为 $\sigma^2$，求 $\operatorname{Var}(\beta_i\mid\alpha)$。要求它约为 1，应选 $\sigma$ 多大？

> [!note]- 参考答案
> $\operatorname{Var}(\beta_i\mid\alpha)=\sigma^2\sum_j\alpha_j^2=1000\sigma^2$；令其约为 1 得 $\sigma\approx1/\sqrt{1000}$。

**题 7**：在 P9 里 $b_{t-i}$ 为什么能够改变 $C=AV$？

> [!note]- 参考答案
> 它先改变 Score，经 Softmax 改变 A 中的关注权重，A 变了自然会改变 $C=AV$ 的各行加权和。

**题 8**：在 P10 里为什么 $R_t^TR_i$ 会只依赖相对位置？

> [!note]- 参考答案
> $R_t^TR_i=R(-t\omega)R(i\omega)=R((i-t)\omega)$，依靠旋转矩阵转置表示逆旋转，以及两个连续旋转角度可相加。

---

# 参考来源与范围说明

1. **本课程主资料**：CS182 Lecture 18，用户提供的手写 `18_fa26.pdf`，P1–P10。笔记按照该文件的主题次序组织；P6–P7 关于「老师是否具体规定 W 的初始化/约束」已在正文明确区分。
2. **Transformer 原始论文**：Vaswani et al., *Attention Is All You Need* (2017)，https://arxiv.org/abs/1706.03762 。用于核对原始 Encoder–Decoder、Masked Attention 和固定正弦位置编码的背景。
3. **通俗架构补充阅读**：Jay Alammar, *The Illustrated Transformer*，https://jalammar.github.io/illustrated-transformer/ 。适合结合图理解 Encoder、Decoder、Multi-Head、FFN 和 Cross-Attention。
4. **公开 Transformer 示意图参考**：dvgodoy, *Transformer, full architecture*，Wikimedia Commons，https://commons.wikimedia.org/wiki/File:Transformer,_full_architecture.png ，CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ 。本笔记为避免图片依赖网络，采用**根据通用架构概念重新绘制的中文图**，不是该图片本身的复制。

> 本文为学习辅助笔记，不是老师原文逐字转录。尤其是 WordPiece/BPE 类分词说明、RMSNorm 层的完整式子、P7 的方差推导、Prefix-LM 与高维 RoPE 说明，都属于**为了补齐理解而增加的背景/推导**，不应记成老师在对应手写页上已经详细讲过的内容。
