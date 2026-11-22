# SuperMimic 阅读记录（完整版·第 1 天）

**日期：2026-11-01**  
**论文：** *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*（arXiv:2610.28480v1）  
**目前真实进度：** 主要逐句讨论了 **Abstract**；为了判断能不能迁移到 GALATEA，又提前查看/讨论了 SuperMimic 的训练原理、§3.1–3.3、Figure 2 和 Appendix D 的 Algorithm A.1。**尚未按顺序正式阅读 Introduction / Method 全文，也未做复现实验。**  
**笔记目标：** 保留「我当时哪里不懂 → 英文怎么翻译 → 具体是什么意思 → 与研究有什么关系」，而不仅仅记录最后的结论。

> **标记**：`[论文原意]` = 能从 SuperMimic / GALATEA 原文核实；`[理解]` = 为了学习所做的解释、简化例子；`[Idea]` = 我与学长讨论的研究设想，尚未验证；`[待查]` = 不能仅凭目前阅读下结论。以下“我提出的问题”是按已有讨论记录重组的提问要点，**不是每一句都逐字还原我的聊天原话**。

---

# 第一部分：Abstract 逐句翻译、词汇和我遇到的问题

## 句 1：研究背景——为什么要追踪大量人类动作？

**英文关键词：** `Tracking large human-motion datasets`、`dominant route`、`general`、`expressive`、`humanoid whole-body control`、`scaling across many motions`。

**先翻译句意：** 利用大量人类动作数据进行动作追踪，已经成为实现通用、表现丰富的人形机器人全身控制的一条主流路线；然而，要让系统扩展到很多不同动作，训练成本非常高。

**当时的问题与解释：**

1. **Tracking 是什么？** 这里不是计算机视觉中“检测并跟踪一个人”的意思，而是 **Motion Tracking / Reference Tracking（运动轨迹跟踪）**：给机器人一条参考动作轨迹，让机器人通过关节控制尽量重现它。例：目标轨迹要求下一时刻抬高左腿、保持上身平衡，Policy 要决定实际关节控制指令。
2. **human-motion datasets 是视频吗？** 可以是人体动作捕捉（mocap）等数据构成的参考运动集合。它们描述目标姿态/运动，**不等于机器人每一时刻的标准电机控制指令**。
3. **dominant route 是什么？** `dominant` = 占主导地位的；`route` = 路线、途径。合起来是“主流技术路线”，不是“道路”。
4. **general 和 expressive 有什么区别？** `general` = 通用、覆盖不同动作；`expressive` = 动作丰富、有表现力，而不是只会单一行走。
5. **humanoid whole-body control 是什么？** `humanoid` = 人形机器人；`whole-body` = 全身；`control` = 控制。指协调腿、手臂、躯干等多个部位，而非只控制一只手。
6. **scaling 为什么贵？** 动作数量/多样性增加后，传统 RL 需要在仿真器里进行大量尝试、收集训练数据，计算开销随任务覆盖增加。

**与 GALATEA 的联系：** SuperMimic 的“动作追踪”与 GALATEA 的 HOI（Human–Object Interaction，人—物交互）参考动作追踪在“跟踪目标轨迹”上相似；但 SuperMimic 主要是人形全身动作，GALATEA 涉及灵巧手—物体复杂接触，**不能把两者看成同一个动力学难度**。

## 句 2：过去的两种 Model-Free RL 路线

**英文关键词：** `Current pipelines`、`model-free reinforcement learning`、`per-motion specialists`、`lossy distillation`、`brute-forcing`、`single policy`、`cluster-scale compute`。

**先翻译句意：** 现有流程通常采用无模型强化学习：或者训练针对不同动作的专门策略，再通过有信息损失的蒸馏整合；或者投入集群级计算资源，硬是把一个统一策略训练出来。

**当时逐项不理解的地方：**

1. **Pipeline 是什么？** 流程、处理管线。这里指完成训练的一整套步骤，而不是单个神经网络。
2. **Model-free RL（无模型强化学习）是“不用神经网络”吗？** **不是。** 它完全可以使用 Policy MLP、Critic MLP。`model-free` 是指策略优化时不依赖一个显式学习的环境动力学模型来预测下一状态并通过其反传梯度；比如 PPO 在仿真中采样，然后根据奖励相关的 Advantage 更新动作概率。
3. **per-motion specialists 是什么？** `per-motion` = 每个动作/动作片段专属；`specialists` = 专家。例：一个 Policy 专门模仿动作 A，另一个专门模仿动作 B。注意这里是 SuperMimic Abstract 在概括先前的人形机器人做法，**不要自动等同于 GALATEA 按物体类别划分的 Expert**。
4. **distillation（蒸馏）是什么？** 让一个 Student/Unified Policy 学习多个 Expert/Teacher Policies 的行为，把多个策略整合到一个策略里。
5. **为什么叫 lossy distillation？** `lossy` = 有损的、有信息损失的。Student 不一定能完整保留每个 Expert 的能力；整合后可能在某些动作上性能下降。**并非说每次蒸馏必定失败**。
6. **brute-force / brute-forcing 是什么意思？** “蛮力解决”“靠大量资源硬推”。这里是以大量训练计算量直接学习单一通用策略，不是某个专门的网络层或优化公式。
7. **cluster-scale compute 是什么？** `cluster` = 计算集群；`scale` = 量级；`compute` = 算力/计算资源。意为动用大量 GPU 的计算集群训练。
8. **我当时重点追问：增加 GPU 难道会改变神经网络结构吗？** **不会自动改变。** 128 张 GPU 并不等于网络自动多出 128 层。更多 GPU 可以并行运行仿真器、收集更多数据、训练更快，或支持更大的训练配置。这里作者批评的是对庞大算力的依赖，而不是说“使用集群就是采用不同的 Policy 结构”。
9. **“一个 Policy”与“很多 Expert”怎么比较？** 一种方法先训练多个专门 Policy 再蒸馏；另一种一开始就训练能覆盖大量参考动作的通用 Policy。前者涉及蒸馏，后者可能需要大规模训练。

**后续发现的另一个易混淆点：** GALATEA 的 SAPG 内部有并行探索分支；GALATEA 的不同物体类别又会训练 Expert Policies 并蒸馏。**这两个“多个 Policy / Expert”不是同一层面的概念**（见第四部分）。

## 句 3：SuperMimic 的核心方法——amortize、logged transitions

**英文关键句（今天明确逐词追问过）：**

> SuperMimic amortizes learning across all motions with one differentiable world model of their shared dynamics, fit from logged transitions of a standard rigid-body simulator.

**先完整翻译：** SuperMimic 利用一个可微分的世界模型，学习不同动作共有的动力学规律，并把这种学习所付出的成本分摊到所有动作上；该世界模型由标准刚体物理仿真器记录的状态转移数据拟合训练得到。

**当时的问题与词汇拆解：**

1. **amortize 是什么？** 原意“摊销、分摊成本”；在这里是“把一次学到的可复用知识推广到许多动作上”。**不是**“计算每个动作平均费用”这么机械地理解。
2. **amortizes learning across all motions 怎么理解？** 不同动作虽然目标轨迹不同，但都由同一个机器人身体及其动力学规律实现。World Model 不必为每段动作重复从零学会“关节如何运动、重力如何作用”等通用规律。共享模型可以供多种动作的策略训练使用。
3. **shared dynamics 是什么？** `dynamics` = 动力学/状态演化规律；`shared` = 共有的。例：左手抬起与向前迈步是两种动作，但同一个机器人质量、关节动力学、接触等规律贯穿其中。不是说这些参考轨迹本身一样。
4. **differentiable world model 是什么？** `differentiable` = 可微分的；`world model` = 世界模型。可以把它看成一个学习物理变化的神经网络：输入当前状态、动作（还有环境参数），预测下一个状态。因为计算路径可导，训练 Policy 时能将梯度从预测后果沿模型传回 Action。
5. **logged 是“登录”吗？** **不是！** 这里 `log` 作动词 = 记录；`logged` = 已记录下来的。
6. **logged transitions 是什么？** `transition` = 状态转移。一条训练记录可以写成 `(s_t, a_t, s_{t+1})`：之前处于哪个状态、执行了什么动作、下一状态怎样。是**仿真器产生并保存的数据**，不是 World Model 自己生成的标准答案。
7. **fit 是“合适”吗？** 这里 `fit from ...` = 用数据拟合、训练得到；不是形容词“合身”。
8. **standard rigid-body simulator 是什么？** `rigid-body` = 刚体（在动力学模拟时物体视为不发生可观形变）；`simulator` = 仿真器，例如论文实验使用 MuJoCo Warp。它根据当前物理状态和控制动作计算之后的运动。
9. **仿真器为什么能给 World Model 提供“标准答案”？** 因为仿真器自己有动力学求解过程：接收 `s_t` 与 `a_t`，计算出 `s_{t+1}`。对 World Model 的监督学习来说，这个模拟结果就是标签；它是“仿真真值”，**不等于现实世界无误差的真值**。
10. **World Model 与 Policy 是同一个 MLP 吗？** 不是。World Model 学习预测环境；Policy 决定选择动作。两者权重不同，任务不同。

**一个最简例子：**

- 当前：杯子高度 3 cm，机械手处于某个关节状态，执行动作 `a_t`。
- 仿真器模拟：下一时刻杯子高度 4 cm；保存为训练标签。
- World Model 预测：下一时刻 3.5 cm。
- 用两者误差训练 World Model，例如高度这一维 `L=(3.5−4)^2`。
- 真实模型预测的是多维状态，还可能包含其它 head；不能把例子的单一“杯子高度”当成论文的完整模型。

## 句 4：world model 和 tracking policy 一起训练；BPTT 与 analytic gradients

**英文关键词：** `train together in one loop`、`backpropagation-through-time`、`short imagined rollouts`、`analytic gradients`。

**先翻译句意：** World Model 与运动追踪 Policy 在同一个训练循环中共同改进；通过对短时虚拟预测轨迹进行时间反向传播，为 Policy 计算解析梯度。

**当时的问题与解释：**

1. **train together 是不是把两个网络拼成一个网络一起反传？** 不是。它们有各自的参数和优化目标，并在同一训练循环中**交替更新**。
2. **具体交替顺序？** `当前 Policy 在仿真器采样 → 数据存入 Replay Buffer → 用监督学习更新 World Model → 固定 World Model 参数，通过想象轨迹和 BPTT 更新 Policy → 再采样`。
3. **谁先开始？World Model 是不是原本就预训练好了？** 主算法不以提前训练好的 World Model 为先决条件。初始 Policy 虽然还不会模仿，也可以输出动作以供数据收集；**先用 Policy 采样 ≠ 先把 Policy 训练成功**。初始数据预热的精确工程实现要查代码，不能擅自断言。
4. **imagined rollout 是什么意思？** `imagined` = 在模型内部“想象/预测”的；`rollout` = 按时间向前逐步展开的状态—动作序列。不是播放一个真实视频，而是 Model 连续预测：`预测下一状态 → Policy 再给动作 → 模型再预测……`。
5. **short 是多短？** 论文 §3.3 的 Policy imagined rollout 使用 **H=32 步**；World Model 的监督训练用 **k=8 步** 的记录窗口。两者不是同一个长度。
6. **BPTT 是什么？** Backpropagation Through Time，沿时间展开的计算图反向传播。把多步 World Model 和 Policy 运算连接起来，最终将累计目标的梯度反传回 Policy 参数。它不是另一个 Policy。
7. **analytic gradients 是什么意思？** `analytic` 在这里强调通过模型的可微计算图/雅可比求得的梯度（链式法则），相对于 PPO 用采样动作概率构造的随机梯度估计。**不是数学上完全没有近似**；梯度依赖学习到的 World Model，模型预测有误时梯度也可能有偏差。
8. **为什么不直接对 PhysX/MuJoCo 求导？** 普通刚体仿真器不一定提供训练所需的完整可微动力学路径，尤其离散接触时可能不光滑。于是用可微学习模型提供预测及所需导数。
9. **如果 World Model 冻结了，为什么梯度还穿得过去？** `冻结参数 ω` 只是不更新 World Model 权重；仍可计算 World Model 输出对输入 Action 的导数。比如固定函数 `h=2a`，仍有 `dh/da=2`。

**特别注意：** “冻结 World Model”只发生在优化 Policy 的那一步；下次数据收集和 World Model 训练时，它仍会更新。

## 句 5：为什么训练更高效？

**英文关键词：** `Fast imagined rollouts`、`analytic gradients`、`policy learning efficient`。

**先翻译句意：** 快速的模型内部预测轨迹和解析梯度，使 Policy 的学习更高效。

**解释：** 效率优势有两个不同来源：(1) 学习到的 World Model 可以比频繁调用完整物理仿真器更快地展开短期未来；(2) 通过 BPTT 获得的路径梯度可能比从采样动作估计的策略梯度方差更低、更省样本。论文后文对这些因素做了分析。

**尚不能推出：** 任何灵巧手 Contact-rich 环境只要换 World Model 就必然更快。复杂接触可能使预测及梯度失真，需要做对照实验。

## 句 6–8：3 小时、128 GPUs、GPU-hours 和失败动作数量

**英文关键词：** `surpasses the return`、`state-of-the-art baseline`、`GPU-hours`、`benchmark motions`、`failures`。

**先翻译句意：** SuperMimic 在一张 GPU 上训练 3 小时，其回报超过一个使用 128 张 GPU 训练 7 天的先进动作追踪基线。它的 GPU 总计算时长减少超过 1000 倍；在 129,514 条基准动作中，SuperMimic 失败 25 条，而基线失败 179 条。用两张 GPU 训练 9.8 小时后，SuperMimic 在该测试集上实现零失败。

**当时的问题与解释：**

1. **return 是什么？** 这里不是函数 `return`，而是强化学习的**回报**：一段轨迹上多步 Reward 的汇总（按具体定义可以折扣/归一化）。不是单步 Reward，也不直接等于“抓取成功率”。
2. **state-of-the-art 是什么意思？** 当时最先进/具有代表性的高性能方法；论文具体对比的是 SONIC，不能把这句话理解为超过所有既有方法和任何任务。
3. **baseline 是什么？** 基线方法，作为对照比较的已有方案。
4. **GPU-hours 怎么算？** `GPU 数量 × 训练小时数` 是粗略资源计量。128 张 GPU × 168 小时 ≈ 21,504 GPU-hours；单 GPU 3 小时 ≈ 3 GPU-hours。论文报告超过 1000× 的算力成本减少。**GPU-hours 不是现实时间，也不是 GPU 运算性能完全等价的保证**；论文比较还存在显卡型号差异。
5. **7 days on 128 GPUs 是什么意思？** 每张 GPU 参与约 7 天的集群训练，不是“7 天里轮流使用了 128 张卡”。
6. **129,514 benchmark motions 是什么？** 由大量参考动作片段组成的评测集合，**不是 129,514 种物体或 129,514 次灵巧手抓取**。
7. **25 failures、179 failures、zero failures 指什么？** 在论文动作追踪评测标准下，未通过某些参考动作的数量；不能直接解读为真实机器人抓取失败次数。
8. **Single GPU 和 two GPUs 为什么不矛盾？** Abstract 报告两个不同算力配置/训练时长对应的不同检查点，不能混为同一个结果。

## 句 9：数据规模扩大了，World Model 为什么还能稳定？

**英文关键词：** `From 1 to 264 hours of motion`、`prediction error stays stable`、`GPU-hours to a target return grow slowly`。

**先翻译句意：** 当参考运动数据从 1 小时增加到 264 小时，World Model 的预测误差保持相对稳定；达到某个指定回报水平所需的 GPU-hours 随数据规模增加得较慢。

**问题与解释：**

1. **1 to 264 hours of motion 指训练了 264 小时吗？** **不是。** 这里是参考动作数据集的**动作总时长**，不是 GPU 训练用时。
2. **prediction error 是什么？** 模型预测下一状态/多步状态与仿真器记录状态之间的差距；不是单独的 Policy 成功率。
3. **stays stable 是完全不变吗？** 论文表达的是随训练动作库扩大，误差没有明显随之恶化；不是数学上严格恒定。
4. **target return 是什么？** 预先选定的目标回报阈值。比较不同规模数据达到同样指定性能需要多少计算。
5. **grow slowly 表示什么？** 训练成本随动作数量增加相对缓慢；这是规模扩展性（scalability）的证据，不代表完全零增量成本。

## 句 10：真实 Unitree G1 迁移

**英文关键词：** `transfers to a physical Unitree G1`、`diverse and acrobatic motions`。

**先翻译句意：** 训练得到的 Policy 能够迁移至真实的 Unitree G1 人形机器人，并执行多样化且具有较高动态性的动作。

**词汇：** `transfer` = 从模拟/训练环境转移到新的系统或真实硬件；`physical` = 真实实体；`acrobatic` = 杂技式/高动态的。

**边界：** 真实机器人迁移是论文的一个实验结果，不能据此推断同样方法能无需调整地在复杂灵巧手抓取上真实部署。

## 句 11：scaling laws 是什么？

**英文关键词：** `removes compute as the bottleneck`、`practical to measure`、`motion-tracking scaling laws`。

**先翻译句意：** SuperMimic 希望让计算资源不再是扩展人形机器人全身控制的主要瓶颈，使研究者能够实际研究动作追踪系统的规模规律。

**当时相关的问题：**

1. **bottleneck 是什么？** 瓶颈、限制系统发展的关键因素。这里主要指训练所需 GPU 算力巨大。
2. **scaling laws 是什么？** 研究例如动作数据量、模型规模、计算资源与最终性能之间的经验关系。不是机器人必须遵守的某个物理定律。
3. **作者已经证明了完整 scaling laws 吗？** Abstract 的措辞是“使测量规模规律变得可行”，并不等同于已经给出了所有规模变量的确定数学定律。

### Abstract 中值得反复看的英文速查表

| 英文 | 本文语境下的中文 | 易错之处 |
|---|---|---|
| tracking | 参考运动轨迹跟踪 | 不是视觉目标检测 |
| dominant route | 主流路线 | route 不是道路 |
| expressive | 动作丰富、有表现力 | 不是“表达情感” |
| pipeline | 一套流程、训练管线 | 不是单个网络 |
| model-free RL | 无显式环境模型的强化学习 | 不是不用神经网络 |
| per-motion specialist | 单动作/动作片段专家策略 | 不是 SAPG 内部采样分支 |
| lossy distillation | 有损的策略蒸馏 | 不是一定会失败 |
| brute-forcing | 靠大量算力硬推 | 不代表改网络架构 |
| cluster-scale compute | 集群级算力 | 不等于网络层数变多 |
| amortize | 摊销、分摊、复用学习成本 | 不是“均摊 GPU 钱”而已 |
| shared dynamics | 共用的动力学规律 | 不是各个动作完全相同 |
| differentiable | 可微分 | 不等于没有模型预测误差 |
| rigid-body simulator | 刚体物理仿真器 | 不是 World Model 本身 |
| logged | 记录下来的 | 不是登录 |
| transition | 状态转移 | 不是任务转型 |
| fit | 拟合/训练 | 不是“合适的” |
| imagined rollout | 模型内预测展开的运动轨迹 | 不是直接播放/生成视频 |
| BPTT | 时间反向传播 | 不是另一种 Policy |
| analytic gradients | 通过可微计算图求得的梯度 | 并非代表对真实环境零误差 |
| return | 累计回报 | 不是程序 `return` |
| baseline | 对照方法、基线 | 不是标签 |
| GPU-hours | GPU 数 × 小时的总资源量 | 不是自然经过的时间 |
| benchmark motions | 基准评测的参考动作 | 不是物体数量 |
| prediction error | 预测误差 | 不是任务失败率 |
| scaling laws | 规模规律 | 不是动力学定律 |

---

# 第二部分：为理解论文而补课——GALATEA / PPO / Policy / Reward

> 这一部分不是 SuperMimic Abstract 的逐句翻译，而是**今天为了理解迁移方案而补的基础知识**，保留今天真实出现过的一连串困惑。

## Q1. $o_t^π=[q_t,a_{t-1},x_t^w,x_t^{tip},\tilde x_t^o,e_t^{HOI},d_t^{ref},φ(M^o)]$ 中，哪个才是 Policy？

**我的疑问：** GALATEA 有这么多输入，Policy 究竟在哪里？

**结论：** 整串 `o_t^π` 是 **Observation（观测向量）**，不是 Policy。真正的 Policy 是 MLP：

```text
Observation o_t^π ──> Policy π_θ（MLP）──> Action a_t
```

- `q_t`：关节配置；`a_{t-1}`：前一步动作；`x_t^w`：手腕位姿；`x_t^{tip}`：指尖位置；`\tilde x_t^o`：带噪声的物体位姿；`e_t^{HOI}`：相对参考 HOI 的运动误差；`d_t^{ref}`：参考指尖到物体距离；`φ(M^o)`：物体几何编码。
- Policy 的网络权重是 `θ`；最终实际想通过 RL 更新的就是 `θ`。
- GALATEA 机械臂/灵巧手的 Action 可理解为 29 维（7 维机械臂 + 22 维手部），具体是关节控制目标，**不是在论文给定的每个参考帧中直接抄出标准动作**。

## Q2. 已经有 HOI 视频/参考轨迹，为什么不能直接监督训练 Action？

参考轨迹告诉机器人“希望手或物体运动到哪里”，但不直接告诉控制器“为了实际达到这个运动，所有 29 个关节应该在这一刻输出多少控制命令”。物理作用、接触、惯性等使这个映射不直接等于参考姿态。

**Reward 的作用：** 在仿真器执行动作后，给出动作追踪、物体跟踪、靠近物体、举升、正则化等评价信号，帮助 Policy 学习如何实现参考轨迹。Reward 不是正确 Action 标签。

## Q3. PPO 到底如何用 Reward 更新 MLP？

**最开始的困惑：** “最终分数不就是所有奖励加起来吗？那当然应该从总分一路对 g 求导啊！”

**一步一步的区分：**

1. Policy 给出某个动作的概率/概率密度，采样一个 Action。
2. 仿真器执行一批动作，得到各步 Reward。
3. Critic/回报估计帮助构造 `Advantage A_t`：这次动作相对于预期是更好还是更差。
4. 为了教学先看简化版 Actor Loss：`L_t = −A_t log π_θ(a_t|o_t)`。
5. 这条式子的反向传播路径是 **Policy 输出的动作 log probability → Policy 权重 θ**。已采样的动作与 Advantage 被当作当前更新的给定数据；不需要对仿真器和奖励门控函数求导。

**反例追问：** “`−log P(A)` 是建立在这个动作好的基础上吧？如果 A 是坏动作呢？”

- **好动作（`A_t>0`）**：例 `A_t=+3`，`L=−3log p`；把该动作概率 `p` 增大，就降低 Loss，因此优化倾向于增加其概率。
- **坏动作（`A_t<0`）**：例 `A_t=−3`，`L=+3log p`；减小概率能够降低 Loss，因此优化倾向于减少其概率。
- `A_t=0`：该样本在简化公式中的 Actor 梯度为零。
- `A_t` 是 **Advantage**，**不是动作 A 的名称**；负的 Advantage 不代表 Reward 必须是负数。可以是“获得 3 分但原本预期 5 分”。

**另一次追问：** “所以 Loss 就是各种 Policy 的加起来？”

**纠正：** 不是很多不同 Policy，而是**同一个 Policy 在很多时间步、很多环境/轨迹采到的样本 Loss 求平均/聚合**。例如 `L_actor≈(L_1+L_2+L_3)/3`。

**特别强调：** `L=−A_t log π_θ(a_t|o_t)` 是**基础 Policy Gradient 的教学式子**，不是完整 PPO。PPO 还有 `r_t(θ)=π_θ(a_t|o_t)/π_old(a_t|o_t)` 与 Clipping；Critic 也有独立的价值预测 Loss。GALATEA 使用 SAPG（基于 PPO 的并行探索/经验聚合方案），不是把每个独立 Expert 的 Loss 简单相加。

## Q4. 如果总 Reward 由很多个 r 组成，真的不能对 g 求导吗？

**当时的推理：** 假设 `r = g·R + 其它项`，则 `∂r/∂g=R`。这一步数学上**没有错**！

问题是更新 `θ` 必须看整个优化算法的梯度路径：

- **PPO / SAPG：** 在 Actor 更新时，Reward 及其派生的 Advantage 作为采样数据，不沿 `Reward → g → 物理状态 → Action` 路径求导。
- **SuperMimic-style BPTT：** 如果用可微 World Model 预测状态并直接构造累计预测 Reward 目标，那么梯度可能经过 `Reward → g → 预测状态 → Action → θ`。如果 `g` 对相应状态量的导数为 0，**这条奖励路径**传不回有效梯度。

**不是说“总 Reward 对 g 没有导数”**，而是两种方法**选择的 Actor 梯度估计方式不同**。

## Q5. ReLU 的梯度有时也是 0，为什么硬阈值 Reward 更值得注意？

**我的原问题：** “梯度是 0 的话不更新不就好了？ReLU 不也是这样？非线性更多反而好吗？”

- ReLU：`max(0,x)` 在负半轴为零梯度，但在正半轴有梯度；其它神经元和其它样本仍可学习。
- 硬阶跃 `I[h>5cm]`：在阈值两侧局部几乎处处导数为 0；即使 3cm 没成功、6cm 已成功，**来自该阶跃项的局部梯度**都为 0。
- 所以并非“有零梯度就不允许”，而是要看这个关键奖励项是否告诉 Policy **怎样微调动作可得到更好的结果**。
- 简单使用 `ReLU(h−5cm)` 也不是万能的：低于 5cm 时仍是零梯度。可能尝试有界连续斜坡、smoothstep 或其他设计，但必须考虑原奖励的语义与量纲。
- **其他连续 Reward 和 Critic 仍可能传递梯度；不能断言 SuperMimic 的整个 Policy 因某个硬门控就完全无法训练。**

---

# 第三部分：World Model 怎么训练？它冻结时 Policy 为什么还能训练？

## Q6. World Model 的输入和监督标签是什么？

**我的原问题要点：** “仿真器凭什么给 World Model 标准答案？仿真器总得有输入才能有输出。”

**答案：** 当前 Policy 先根据 Observation 输出 Action；物理仿真器根据完整状态 `s_t`、执行动作 `a_t`（还有场景参数/随机因素）计算下一状态 `s_{t+1}`，并记录数据。World Model 使用这些记录做监督学习。

```text
Policy πθ 当前给出 Action a_t
       |
       v
物理仿真器（s_t, a_t）──────> s_{t+1} 〔监督标签〕
       |                                  ^
       |                                  |
       +--> 记录 (s_t, a_t, s_{t+1}) ----+
                |
                v
         World Model fω(s_t,a_t) = 预测的 ŝ_{t+1}
                |
                v
         L_WM = 预测值与标签的误差
```

**别混淆：** 参考视频里的目标位置不是监督 World Model 下一状态的标签。目标轨迹给控制任务提供“想要怎样动”的目标；仿真器状态转移告诉模型“按照当前控制实际上怎样动了”。

**论文更完整的情况：** World Model 条件化于完整状态、动作和模拟环境 context `ξ`，并涉及 reward、observation、termination 的预测/解析计算；§3.2 给出 **k=8 步**训练窗口和多项 Loss，不只是例子里的一维 MSE。

## Q7. World Model 和 Policy 都是 RL 训练的吗？

**不是。**

| | World Model | Policy |
|---|---|---|
| 目标 | 预测状态转移/相关输出 | 选择能取得更高回报的控制动作 |
| 典型输入 | 状态、动作、仿真条件 | Observation、Goal |
| “标准答案” | 仿真器实际产生的状态等记录 | **没有逐步正确动作标签** |
| 训练方法 | 监督学习 | Model-Based RL（通过 World Model 的 BPTT） |
| 更新参数 | `ω` | `θ` |

**记忆点：** World Model 学的是“这个动作会发生什么”，Policy 学的是“为了完成目标应该做什么”。只学好预测模型，不代表机器人自动学会抓取。

## Q8. 如果优化 Policy 时冻结 World Model，梯度为什么能穿过去？

**最简单反例：** 假设 World Model 是已经确定、参数不更新的函数 `h=2a`。即使函数系数 2 不变，仍有 `dh/da=2`。因此 Policy 改变 Action，预测高度就改变，我们可以算出 Loss 对 Action、进一步对 Policy 权重的导数。

**冻结权重 `ω` ≠ 对输入 `a` 禁止求导。** 在 PyTorch 里更要区分：冻结模型参数可以不更新 `ω`，但在 Policy 更新阶段如果把整个 World Model 前向计算包进 `torch.no_grad()`，就可能意外切断我们正需要的路径梯度。

**示意链式法则：**

`∂L/∂θ = (∂L/∂r)(∂r/∂g)(∂g/∂h)(∂h/∂a)(∂a/∂θ)`。

如果 `g` 是针对高度的硬阶跃，`∂g/∂h=0`（除了阈值处不可导），那这一项对 Policy 的 BPTT 梯度为 0。改 Gate 是 **Policy 优化阶段**的研究问题，而不是 World Model 监督学习标签的问题。

## Q9. SuperMimic 最开始先训练 Policy 还是 World Model？World Model 是预训练的吗？

**论文 Figure 2 / §3.1 / Algorithm A.1 的循环：**

1. **Collect**：当前 Policy 在 MuJoCo Warp 仿真器中运行，加动作探索噪声；记录状态转移到 Replay Buffer。
2. **Fit World Model**：从 Buffer 抽已记录窗口，利用监督误差更新 `ω`。
3. **Improve Policy**：从 Buffer 取起点，经冻结的 World Model 想象 H 步，优化预测回报并用 Critic 估计更远未来，更新 `θ`。
4. 重复；World Model 和 Policy 在整个训练期间共同进步。

**关键纠正：** “第一步由 Policy 收集数据”不等于“先把 Policy 用 RL 训练好了”。初期 Policy 可以很差甚至接近随机，但产生的状态转移仍可用于动力学监督学习。

**[待查]** Algorithm A.1 介绍了每轮顺序，但仅凭它还不能确认第一次更新前 Buffer 的初始预填充细节，以及是否有工程实现上的额外 Warm-up；不要把我们为理解而写的示意流程误当成代码细节。

---

# 第四部分：将 SuperMimic 迁移到 GALATEA——与学长实际讨论的问题

## Q10. 学长说“直接用 SuperMimic 的 model-based RL 替换 SAPG”，究竟替换什么？

**[Idea] 核心是替换 GALATEA 的 Expert Policy 优化算法，而不是推翻整条 HOI 数据管线。**

GALATEA 的上游（视频生成、三维 HOI 恢复、参考轨迹准备/增强）可以先保留。原来使用 SAPG 在 PhysX 中训练 Policy；拟尝试用可微 World Model 和 BPTT 训练同类 Policy。能否实现、性能能否维持，需要实验而不是仅凭理论断言。

## Q11. 两种“多个 Expert”为什么让我与学长误会？

**学长先说的：** SAPG 内部不同探索设置的多个并行 PPO 分支/采样来源，它们在训练过程中聚合经验/梯度以改善共享主 Policy。

**我当时说的：** GALATEA Method 的 **Policy Distillation** 部分：不同物体类别各自训练 Expert Policies，最后通过 **Behavior Cloning（BC，行为克隆）+ DAgger** 蒸馏成一个 Unified Policy。

两者不是同一种“多个 Policy”。我已经向学长澄清，他回复“这个肯定先这样搞”，并补充说**这样训练难度低一点**。

## Q12. 两种可行研究路线

**路线 A——学长倾向先做的低难度方案：**

```text
GALATEA 原来的参考 HOI 轨迹
           |
           +--> 物体类别/Expert A：用 SuperMimic-style MBRL 训练
           |
           +--> 物体类别/Expert B：用 SuperMimic-style MBRL 训练
           |
           +--> 其它 Expert（成功后再扩展）
                              |
                              v
                    BC + DAgger 蒸馏
                              |
                              v
                        Unified Policy
```

- “保留多个 Expert”指保留 GALATEA **跨类别专家训练与最终蒸馏的结构**。
- 不代表保留原 SAPG 算法；恰恰是把每个 Expert 的 SAPG 替换成可微 World Model + BPTT。
- 每个 Expert 是否需要独立 World Model？**不一定**。可以先单类建模，后续探索共享 World Model，这属于尚未决策的工程设计。

**路线 B——更激进、更难：** 一个共享 World Model + 一个直接覆盖所有物体、所有参考 HOI 的 Unified Policy，尝试不再蒸馏。由于物体几何、质量、摩擦和接触模式跨度大，训练和模型泛化问题更复杂。

## Q13. 为什么学长说路线 A 训练难度更低？

Expert A 不必在初期同时学会杯子、瓶子、工具等所有不同形状与接触技能；每个阶段的任务范围较窄，也更容易定位是 World Model 预测不准、Reward 设计不合适，还是策略优化不稳定。**这是合理的研究直觉，并非已经被实验保证。**

## Q14. GALATEA 的 Lift Gate 与 Lift Reward 是一回事吗？

**不是。** 两个函数今天容易被混淆，要分别记：

**Lift Gate（对跟踪奖励的门控，简写示意）：**

`g_t^L = 1 − I[参考举升高度 > 5cm 且 实际举升高度 ≤ 5cm]`。

参考已经要求举起物体、实际没有成功举起时，该项变 0，用于抑制某些跟踪奖励。**原门控多数情况下是 1，不是“高度低就一律 0”**。

**Lift Reward（额外成功举升奖励）：**

`r_t^lift = 3 · I[参考举升高度 > 5cm] · I[实际举升高度 > 5cm]`。

这两个都可能有硬阈值，但位置、语义不同。不能说“把 Lift Gate 平滑了，就等于把 Lift Reward 平滑了”。

## Q15. 为什么说 GALATEA 的 Reward 对 PPO 没问题，换 BPTT 就有问题？

原 PPO 不要求沿 PhysX/Reward 反传动作梯度，硬阈值可以当作采样后得到的奖励数值。SuperMimic 式 BPTT 希望通过 World Model 的可微预测状态计算 Reward 梯度。硬门槛从这条路径得到的局部梯度通常为零。

**但是重要修正：**

- 并不是“只要任何一个 Reward 里出现 g，整套 BPTT 就失败”。其它连续奖励和 Critic 可以提供梯度。
- 不是只要把 g 改成“非线性”就好了；ReLU 在某些区域同样为零梯度。
- 不能不经实验就宣布修改一定改善成功率；还可能带来 Reward Hacking、错误优化方向或者改变原任务目标。

**我先前的一个 Idea（待验证）：** 当参考高度超过 5cm 时，尝试让实际举升 **5–8cm** 区间的 Lift Gate 逐步从 0 变成 1（线性 / smoothstep），其余按逻辑处理。**注意这种 5–8cm 平滑仍不自动给低于 5cm 的动作提供 Lift Gate 梯度。** 是否应该更早给出 progress reward，要另外设计消融。另一想法是接触前基于距离变化设计 approach progress reward，属于独立 Idea。

## Q16. Contact-rich 为什么是最大的不确定性？

学长明确说：“就是看看 Contact rich 的环境里能不能做 model base rl。”

- `Contact-rich` = 大量频繁且复杂的接触、分离、碰撞、摩擦、滑动等。
- SuperMimic 的人形动作追踪已经涉及脚—地等接触，不能说它完全没有接触。
- GALATEA 的灵巧手抓物涉及多个指尖与物体、滑动、抓握稳定性、物体不同形状/摩擦，World Model 的预测更难。
- **即使预测状态的均方误差看起来很小，也不能保证预测状态对 Action 的导数方向准确**；而 BPTT 直接依赖这个导数。

**要验证的不是一句“可以做”，而是：** World Model 预测精度、接触切换处梯度质量、BPTT 训练稳定性，以及与 SAPG 的效率/效果对照。

## Q17. 当前最适合动手做的最小实验是什么？

**[Idea] 实验路线：**

1. 先从 GALATEA 的一个物体类别 / 一个 Expert / 少量参考 HOI 轨迹开始；尽量固定物体、环境与评估协议。
2. 用原 SAPG 训练结果作为对照（Baseline）。
3. 从同样的 PhysX 仿真环境收集状态转移，训练可微 World Model；关注接触前、接触中、抓稳/滑落阶段的误差。
4. 用冻结权重的 World Model 展开短时间预测轨迹，尝试 BPTT 优化同一个任务的 Policy；必要时探索连续型 Reward / Gate。
5. 对比成功率、Reference Tracking 指标、Wall-clock Time、GPU-hours、样本效率与训练稳定性。
6. 单 Expert 可行之后再推广到多个 Expert，最后沿用 BC + DAgger；避免一开始把多个研究变量同时改变。

**研究设计提醒：** 如果改了优化算法，又改了 Reward，又改了训练专家数量，最后效率变好/变差都很难判断是哪项造成的。应保留一组逐项改动的消融实验。

---

# 第五部分：明天阅读论文时的具体问题清单

## A. 明天先把 Abstract 再看一遍，能自己回答这 12 题吗？

- [ ] `tracking human-motion datasets` 中 tracking 的目标到底是什么？
- [ ] `dominant route`、`expressive` 各怎么翻译？
- [ ] `model-free RL` 为什么不是“完全没有模型/神经网络”？
- [ ] `per-motion specialists` 与 `lossy distillation` 组成什么训练流程？
- [ ] 为什么 `brute-forcing with cluster-scale compute` 不代表网络结构改变？
- [ ] `amortize learning across all motions` 如何理解“复用共享动力学”？
- [ ] `logged transitions` 的三个最基本部分是什么，谁产生标签？
- [ ] `differentiable world model` 为什么比不可微黑盒仿真器更适合 BPTT？
- [ ] `imagined rollouts` 究竟是怎样一帧一帧展开的？
- [ ] `analytic gradients` 和 PPO 的动作 log-probability 梯度有什么区别？
- [ ] `return` 与 `reward`、`GPU-hours` 与真实时间如何区分？
- [ ] `prediction error stays stable` 和 `scaling laws` 证明了什么，没证明什么？

## B. 然后阅读 §1 Introduction

- [ ] 理解已有大型 Mocap/Motion Tracking 流程为什么贵：重复采样物理后果与并行算力成本。
- [ ] 分清论文对先前方法的比较，不要将 SuperMimic 的 per-motion experts 自动当作 GALATEA 的 per-object experts。
- [ ] 理解论文声明的适用范围：人形机器人全身动作追踪；哪些结论尚未覆盖灵巧手—物体交互？

## C. §3.1 / Figure 2 / Algorithm A.1

- [ ] 当前 Policy 初始如何输出动作？动作噪声在哪里注入？
- [ ] Replay Buffer 具体存了哪些量？训练初期怎样初始化/预热？
- [ ] 为什么顺序是 Collect → Fit World Model → Improve Policy？
- [ ] 32-step imagined rollout 为什么从 Buffer 中某个真实状态开始？

## D. §3.2 / Eq. (1)

- [ ] World Model 的完整输入是 `s_t,a_t,ξ` 中的哪些维度？
- [ ] 预测 State / Reward / Observation / Termination 的哪些 head 是可学习、哪些解析计算？
- [ ] `k=8` 多步监督训练如何避免多步误差积累？
- [ ] 论文的硬接触终止条件为什么采用可微的 termination probability？

## E. §3.3 / Eq. (2)–(3)

- [ ] `H=32` 的 predicted reward sum 和 Critic bootstrap 如何组成 Policy 目标？
- [ ] BPTT 求的是哪一条对 θ 的梯度？在哪些地方冻结 ω？
- [ ] 训练 World Model 的监督 Loss 与训练 Policy 的 RL Loss 如何严格区分？
- [ ] 在模型很不准时，Policy 为什么可能利用错误的预测梯度？论文如何缓解？

## F. 针对 GALATEA 的迁移问题

- [ ] 原始 GALATEA 的 Lift Gate、Lift Reward、Contact Gate、Near Reward 精确公式和用途分别是什么？
- [ ] 每个奖励项哪些输入可导、哪些是硬判定？是否真的必须修改？
- [ ] PhysX 中需要额外记录哪些接触/速度/物体参数，才能训练灵巧手 World Model？
- [ ] 单 Expert 应先训练独立 World Model 吗？多个 Expert 能否共享？
- [ ] 接触复杂性会不会让 BPTT 的梯度不可靠？怎样设计量化实验？
- [ ] 与学长进一步确认最低可接受的实验指标和计算预算。

---

# 第六部分：阶段性结论（供明天快速回忆）

1. **论文的主张：** 用一个共享的可微 World Model，把不同动作需要重复探索的动力学知识复用起来；借助 imagined rollouts + BPTT 高效训练人形机器人追踪 Policy。
2. **World Model 训练：** 在仿真器中收集 `(s_t,a_t,s_{t+1})` 等状态转移，用监督学习更新模型。
3. **Policy 训练：** 优化多步预测 Reward + Critic 组成的目标；冻结 World Model 的**参数**，但通过其输入导数反向传播到 Policy。
4. **与 PPO 不同：** PPO 从采样后得到的 Advantage 和动作概率构造策略梯度；SuperMimic 式 BPTT 直接利用可微预测轨迹的路径梯度。
5. **与学长的初步共识：** 先保留 GALATEA 按物体类别训练 Expert、最终 BC + DAgger 蒸馏的总体结构，把 Expert 训练里的 SAPG 换成 SuperMimic-style MBRL；这样比直接端到端训练所有物体的 Unified Policy 更容易开始。
6. **研究风险：** Contact-rich 预测误差和梯度偏差、硬门控 Reward 的有效梯度、训练稳定性、训练效率是否真的提高。**这些仍是开放研究问题，不是已证明的结论。**
7. **真实阅读进度：** 只是 Abstract 与为理解方案而提前讨论的原理；明天还需要认真读 §1、§3，并在读论文时随时补充新的问题。
