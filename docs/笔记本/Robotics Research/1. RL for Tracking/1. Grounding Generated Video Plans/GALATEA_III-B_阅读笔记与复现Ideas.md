# GALATEA — III-B 阅读笔记与复现改进 Ideas

> 论文：*Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers*  
> 范围：III-B.1 **Reference Augmentation and Preprocessing**；III-B.2 **RL Formulation and Training Recipe**；III-B.3 **Policy Distillation**。  
> 标记：**[论文]** 为论文明确给出的内容；**[讨论]** 为阅读时的解释或推理；**[待确认]** 需要查看代码或进一步询问；**[Idea]** 为计划尝试的改进，并非已证实有效。  
> 注：文末附录保留 III-A 的 Idea 1，便于跟此前的 Ideas 编号对应。

---

# Part I. 阅读过程中的问题与解答

## 1. Reference Augmentation and Preprocessing

### Q1. III-A 已经重建了三维轨迹，为什么还需要 Reference Augmentation？

**[论文]** III-A 从生成视频重建三维 MANO 手部与物体轨迹。III-B 对每条原始轨迹采样 **5 个增强版本**，扩充训练中接近物体及接触后操作的轨迹变化。

**[讨论]** 如果只训练固定轨迹，Policy 看到的参考状态有限；增强轨迹有助于提高训练覆盖度。不过增强不等于保证物理可执行，所以论文还会筛选增强候选轨迹。

### Q2. $T_t^O$ 是什么？是不是从上一帧到这一帧的相对运动？

**不是。** $T_t^O\in SE(3)$ 是物体在第 $t$ 帧的三维位姿（旋转 + 平移），不是 $T_{t-1}\to T_t$ 的相对变换。同理 $T_t^H$ 表示手的位姿。

### Q3. 为什么接触前只扰动手，接触后手和物体一起扰动？

**[论文]** 增强公式：

$$
\widetilde T_t^X=GC_t^X T_t^X,\qquad X\in\{H,O\}
$$

$$
(C_t^H,C_t^O)=
\begin{cases}
(\operatorname{Exp}(\alpha_t\xi_0),I),&t<t_c,\\
(\operatorname{Exp}(\beta_t\xi_e),\operatorname{Exp}(\beta_t\xi_e)),&t\ge t_c.
\end{cases}
$$

- $t_c$：第一次接触的参考帧。
- $G$：对整段轨迹施加的 yaw 变换。
- $\xi_0,\xi_e$：采样的刚体运动扰动（Twist）；$\operatorname{Exp}$ 把 Twist 映射为 $SE(3)$ 变换。
- **接触前**：只改变手的接近路径，物体不附加该时变扰动。
- **接触后**：手和物体使用同一个时变扰动，维持参考轨迹中的手物相对几何关系。

### Q4. $\alpha_t,\beta_t$ 的作用是什么？

$$
\alpha_t=1-\frac{t}{t_c},\qquad
\beta_t=\frac{t-t_c}{T-t_c}
$$

- 接触前，$\alpha_t$ 逐步减小到 0：接近接触时扰动消失。
- 接触后，$\beta_t$ 从 0 逐步增加到 1：接触后的共同扰动平滑增大。
- **[论文]** 扰动幅度设置为原轨迹 clearance 和 motion range 的约 30%。

### Q5. 为什么增强后还要 IK 和插值？

**[论文]** 对增强轨迹用 IK 求解 FR3 机械臂关节轨迹，再根据关节速度限制插值、放慢整段轨迹。还会筛选增强候选轨迹是否容易执行。

**[讨论]** 这里解决的是运动学可达性和速度问题，不能据此推断所有接触都满足严格无穿透约束。

---

## 2. Actor Observation

**[论文]** Actor 的观测：

$$
 o_t^\pi=[q_t,a_{t-1},x_t^w,x_t^{tip},\widetilde x_t^o,e_t^{HOI},d_t^{ref},\phi(M^o)]
$$

### Q6. $q_t$ 指什么？

实际机器人当前的 **29 个关节角度**：FR3 机械臂 7 DoF + Sharpa Wave Hand 22 DoF。不是 MANO 人手模型的参数。

### Q7. $a_{t-1}$ 为什么放进 Observation？

它是上一时刻网络输出的 **Raw Action**，提供最近控制行为的信息。要区分：

- $a_{t-1}$：上一时刻原始动作输出；
- $q_t$：当前实际关节角度；
- $q_t^{PD}$：映射和 EMA 平滑后的 PD 位置目标。

### Q8. Wrist Frame 和 FR3 Base Frame 有什么区别？

**[论文]** Wrist quantities 在 **FR3 Base Frame** 表示；指尖和带噪物体位姿在 **Wrist Frame** 下表示。

即使手指尚未接触物体，也能定义物体相对于手腕的位姿。例如：

$$
 p^W=(R_W^B)^T(p^B-p_W^B)
$$

这里不仅涉及平移，还必须考虑手腕朝向；不能把完整的坐标系转换简单理解成相减。

### Q9. $e_t^{HOI}$ 究竟包含什么？

**[论文]** 包含当前的手腕、MANO 关键点和物体位姿追踪误差。论文在该处没有逐个列清 Observation 各维度的展开方式。

**[待确认]** 它是否已经显式包含全部五个指尖的三维追踪误差？此问题关系到 **Idea 2** 是否重复已有特征。

### Q10. $d_t^{ref}$ 是当前机器人指尖到物体的距离吗？

**不是。**

- $d_t^{ref}$：参考轨迹中 **5 个参考人手指尖** 到参考物体表面的距离；
- $d_i$：Reward 中使用的 **当前仿真机器人指尖** 到物体的距离。

前者帮助 Actor 理解参考动作目前应处于接近还是接触阶段；后者反映机器人实际执行状况。

### Q11. $\phi(M^o)$ 有什么用？

**[论文]** 它是通过 BPS（Basis Point Set）得到的物体形状编码，让 Actor 可以根据不同物体的三维几何形状采取不同动作。

### Q12. Actor、Critic、Isaac Gym Simulator 有什么区别？

| 模块 | 作用 |
|---|---|
| **Actor** | 观察 $o_t^\pi$，输出控制动作 $a_t$ |
| **Critic** | 估计长期价值，辅助 PPO 更新 Actor |
| **Simulator** | 执行物理运动、计算接触并提供训练状态与 Reward |

**[论文]** 使用 *Asymmetric Actor–Critic*：Actor 不获得精确的关节速度、接触力、物体质量、质心及 retargeted hand-joint target；Critic 可使用额外的 Privileged Simulator State，例如无噪声物体位姿和关节速度。

---

## 3. Action Mapping、EMA 和 PD Control

### Q13. Actor 输出什么？

$$
 a_t=[a_t^A,a_t^H]\in[-1,1]^{29}
$$

其中机械臂 7 维、灵巧手 22 维。

### Q14. Arm Action 和 Hand Action 的区别是什么？

**Arm 使用相对当前位置的增量：**

$$
\bar q_t^A=q_t^A+(0.03\,\mathrm{rad})a_t^A
$$

**Hand 将归一化动作映射到各关节限位内的绝对目标位置。**

需要区分：两种动作的**解释方式不同**，但最终都变成具体的关节位置目标送往后续控制模块。

### Q15. EMA 是什么？为什么需要？

$$
q_t^{X,PD}=\alpha_X\bar q_t^X+(1-\alpha_X)q_{t-1}^{X,PD}
$$

$$
\alpha_A=0.20,\qquad\alpha_H=0.10
$$

EMA 平滑相邻关节位置目标，减少控制指令突变；平滑后的目标发给 **30 Hz Position PD Controller**。

**完整流程：**

`Actor Raw Action → Action Mapping → Joint Targets → EMA → PD Controller → Isaac Gym 物理运动`

---

## 4. Reward

**[论文]** 总奖励：

$$
\boxed{r_t=r_t^H+r_t^O+r_t^{near}+r_t^{multi}+r_t^{lift}+r_t^{eff}-r_t^{reg}}
$$

| 项 | 含义 |
|---|---|
| $r_t^H$ | Hand Tracking Reward |
| $r_t^O$ | Object Tracking Reward |
| $r_t^{near}$ | Fingertip Approach / Near Reward |
| $r_t^{multi}$ | Multi-contact Reward |
| $r_t^{lift}$ | Lift Reward |
| $r_t^{eff}$ | Mechanical-power Efficiency Reward |
| $r_t^{reg}$ | Action / Velocity / Torque Regularization |

### Q16. $\kappa_\alpha(e)=\exp(-\alpha e)$ 是什么？

把误差转换成正向奖励：误差为 0 时输出 1；误差增加时奖励下降。$\alpha$ 越大，对误差越敏感。

### Q17. 为什么 Contact 阶段降低 Hand Keypoint Tracking 权重？

**[论文]**

$$
\rho_t=\begin{cases}
2,&\text{free space}\\
0.75,&\text{contact}
\end{cases}
$$

**[讨论]** 接触后如果过度强制机器人追踪有误差的 MANO 手型，可能与实际物理接触冲突。降低追踪权重，可能让 Policy 更有余地调整实际抓握动作。论文给出了参数，但这里的完整因果解释仍属于合理推理。

### Q18. Hand Tracking Reward 的 Lift Gate $g_t^L$ 为什么存在？

**[论文]**

$$
g_t^L=1-\mathbb I[\hat h_t-h_0>0.05\land h_t-h_0\le0.05]
$$

- $\hat h_t$：参考物体高度；
- $h_t$：当前仿真物体高度；
- $h_0$：物体初始放置高度。

当**参考物体已举升超过 5 cm，但实际物体举升不超过 5 cm**，$g_t^L=0$，关闭整个 Hand Tracking Reward。这样可以避免机器人只追踪空手上举动作，却没有真正举起物体。

**发现的问题：** 一旦实际举升跨过 5 cm 阈值，门控发生 $0\to1$ 的硬切换，奖励不连续。见 **Idea 3**。

### Q19. Object Tracking Reward 的 Contact Gate 怎么工作？

$$
g_t^C=\mathbb I[c_t^{ref}>0\lor c_t>0]
$$

参考轨迹出现接触，或者当前仿真已有接触，就激活 Object Tracking Reward。

### Q20. 为什么 Near Reward 存在 Plateau（饱和平台）？

**[论文]**

$$
r_t^{near}=0.5g_t^{ref,3}\sum_i\exp\left(-\left(\frac{[d_i-0.015]_+}{0.020}\right)^2\right)
$$

其中 $[x]_+=\max(x,0)$，$g_t^{ref,3}=\mathbb I[c_t^{ref}\ge3]$。

当 $d_i\le0.015\,\mathrm m$（1.5 cm），每个指尖的指数因子都为 1。

这意味着指尖从 **0.5 cm 后退到 1 cm**，Near Reward 也不会下降。它奖励“足够靠近”，却不直接刻画“此刻是否正向物体靠近”。见 **Idea 4**。

### Q21. 距离不超过 1.5 cm 就算接触吗？

**不是。** 实际仿真接触数量定义为：

$$
c_t=\sum_i\mathbb I[F_i>0.1\,\mathrm N\land d_i\le0.015\,\mathrm m]
$$

同时要求接触力 $F_i>0.1\,\mathrm N$ 和距离 $d_i\le1.5\,\mathrm{cm}$。仅仅靠近不等于实际接触。

### Q22. Multi-contact Reward 怎么工作？

$$
r_t^{multi}=g_t^{ref,+}
\begin{cases}
1.25,&c_t=4\\
2.5,&c_t=5\\
0,&\text{otherwise}
\end{cases}
$$

其中 $g_t^{ref,+}=\mathbb I[c_t^{ref}>0]$。参考阶段要求接触时，奖励机器人形成 4 指或 5 指的实际接触。

### Q23. Lift Reward 和 $g_t^L$ 有什么区别？

$$
r_t^{lift}=3\mathbb I[\hat h_t-h_0>0.05]\mathbb I[h_t-h_0>0.05]
$$

- **$g_t^L$**：决定 Hand Tracking Reward 是否被关掉。
- **$r_t^{lift}$**：参考需要举升且实际完成大于 5 cm 举升时，提供额外奖励 3。

两者独立，且都涉及 5 cm 硬阈值。即使平滑 $g_t^L$，$r_t^{lift}$ 仍有阶跃。

### Q24. Efficiency Reward 是不是等于节省电量？

$$
r_t^{eff}=0.5\exp(-10P_t),\qquad P_t=\sum_{k\in H}|\tau_{t,k}\dot q_{t,k}|
$$

这是手部关节机械功率的惩罚性激励，不是完整的电气能耗模型。即使关节不动、$\dot q\approx0$，保持夹持力仍可能耗电。

### Q25. Regularization 的完整公式在哪里？

**[论文]** 描述了：

- 连续 Arm / Hand Raw Action 差异，权重 $0.05/0.025$；
- 关节速度，以对应速度限值的 15% 归一化，权重 $0.10/0.04$；
- 超过 20% effort limit 的归一化力矩，权重 $0.2$。

**[待确认]** 论文没有在该节完整给出正则项的逐项数学形式，不能直接认定实现一定采用平方 $L_2$ 惩罚；复现时查代码。

---

## 5. PPO、SAPG、Domain Randomization

### Q26. PPO 是否只会选择当前 Reward 最高的动作？

**不是。** PPO 优化长期预期回报。Actor 学习 $\pi_\theta(a\mid o)$，Critic 辅助估计未来累计回报，使用 Advantage 和 PPO 更新目标进行训练。

**与 Q-learning 的区别：** Q-learning 主要学习 $Q(s,a)$；PPO 直接优化参数化策略，并利用 Critic 估计价值。

### Q27. SAPG 的多个 PPO Agent 是不是多个不同的 Expert Policy？

**不是。**

**[论文]** SAPG 将并行 Rollout 分配给使用不同探索设置的 PPO Agent，再聚合经验以更新共享 Policy。

- SAPG 中的 Agent：围绕**同一个策略学习目标**进行不同探索，帮助训练一个 Policy。
- 策略蒸馏中的 Expert：**分别训练完成的不同 Policy**，覆盖不同的轨迹集合。

**[待确认]** “不同探索设置”在具体实现中是什么参数？调整高斯动作方差只是我们此前举的例子，不能直接当作论文的实现细节。

### Q28. SAPG Agent 与 Expert 的层级关系？

```text
多组 PPO Agents（不同探索设置）
           ↓ SAPG / shared policy update
       Expert Policy A

多组 PPO Agents（不同探索设置）
           ↓ SAPG / shared policy update
       Expert Policy B

Expert A + Expert B + ...
           ↓ Behavior Cloning + DAgger
       Unified Student Policy
```

简记：**SAPG 训练 Expert；Policy Distillation 整合 Experts。**

### Q29. Domain Randomization 随机化哪些参数？

**[论文]** 每个 Episode 随机化 Arm / Hand PD Gains、物体质量、物体 / 桌面 / 指尖摩擦；还包含逐 Step 的 Actuation Latency 随机化。Actor 的 Object Pose 也被添加噪声。

目的：提升训练后的策略对 Sim-to-real 差异的适应能力。

### Q30. 为什么 Actor 看到带噪 Object Pose，而 Critic 看到干净的真值？

真实部署中的视觉物体位姿估计会有误差；训练时让 Actor 使用带噪观测可提高鲁棒性。Critic 使用仿真器提供的精确状态来辅助学习。这里“干净”指仿真真值，**不是 III-A 的视频重建 Ground Truth**。

---

## 6. Episode Initialization and Termination

### Q31. Episode 必须从第一帧或者 $t_c$ 之前开始吗？

**不是。**

**[论文]** 在训练时抽取参考帧，并将机械臂与物体初始化到该帧对应的状态。

**[学长解释]** 实际训练会从整段 HOI 参考轨迹随机选择 Reset 帧，以免初始状态过多集中在开始阶段。

### Q32. 为什么 Reset 时将 22 个手指关节张开？

**[论文]** 保留 Retargeted Wrist Pose，但把 22 个 Hand Joints 重置为 Open Configuration。

原因：视频三维重建与后续机器人重定向的参考手型可能导致 Robot Hand Links 出现在物体或桌子内部（穿模），造成较大的 **Depenetration Impulse（解穿透冲量）**，使仿真不稳定。

### Q33. III-A 都做完 3D Reconstruction 了，为什么仍会穿模？

- **Ground Truth (GT)**：用于评估或测量的真实参考状态。
- **Reconstruction**：从 RGB 视频推断出来的三维状态，仍然存在误差。
- III-A 的 Contact Loss 有助于改善接触几何，但不是严格无穿透物理约束。
- MANO 人手与 Sharpa Wave Hand 的几何形状也不同，重定向后不一定能够保持严格接触。

**结论：** 可作为 **RL Tracking Reference** 的轨迹，不等于能安全地直接设置成 **Simulator Initial State** 的轨迹。

### Q34. 从接触前开始 Reset，后续追踪到接触阶段，不还是可能穿模？

**是，仍然可能。** 区别在于：

1. **直接 Reset 到重叠姿态**：仿真一开始两个刚体就可能相互穿透，发生强烈的解穿透反应。
2. **从接触前逐步执行 Policy**：机器人随 Action/PD Control 逐步靠近物体，PhysX 在仿真每一步处理碰撞；Policy 也可能学到不完全贴合错误参考位置的合理接触动作。

但并不保证后者绝对无穿透；数值求解或高速接触仍可能产生穿透。

还需要区分：

- **PhysX** 每步处理接触和碰撞。
- **SAPG/PPO** 在收集 Rollout 后根据 Reward 更新网络参数，**不会在每次接触瞬间更新网络**。
- Reward 中**没有论文明确列出的独立 Penetration Penalty**，不能说作者直接靠某一项惩罚穿透。

### Q35. 随机 Reset 到物体已经悬空、手指却张开的帧，不会掉落吗？

**[学长解释]** 这是可能发生的。作者曾尝试对物体施加**短暂的虚拟向上力**抵消重力，让物体短时间悬浮；后来认为不加这个虚拟力效果也不错，就没有继续采用。网络在一定程度上能学会迅速收紧手指，减少掉落，但训练过程仍可能失败。

**[讨论]** 简单限制 $t_{init}\le t_c$ 可以回避部分悬空状态，却会使训练偏重轨迹前半段，不利于学习后续操作。见 **Idea 5**。

### Q36. 为什么 Evaluation SR 可能高于训练中统计的 SR？

**[讨论]** 一个合理可能性是 Training 的随机 Reset 包含不利的悬空/张开状态，而 Evaluation 的初始状态分布不同，因而更容易。**[待确认]** 仍需核查评估的实际 Reset 规则和成功率定义，不能笼统认定所有实验如此。

### Q37. Episode 如何结束？

**[论文]** 成功 Episode 在参考轨迹末尾结束；如果手部或物体追踪失败，或者关节速度超过阈值，则可能提前终止。

---

## 7. Policy Distillation

### Q38. 为什么不直接在全部参考轨迹上训练一个 Policy？

**[论文]** 直接训练完整参考集在有限算力下困难；逐轨迹训练独立 Policy 又难扩展。因此采用少量 **Multi-skill Experts** 负责不同的轨迹组，再蒸馏到统一 Policy。

### Q39. 每个 Expert 负责多少条轨迹？怎样训练？

**[论文]** 每个物体类别**最多两个** Multi-skill Experts，每个 Expert 覆盖约 **40 条原始轨迹**（增强后约 200 条）。

训练方式二选一：

1. 直接在该物体类别的轨迹上从头训练。
2. 当上面效果不佳时，先在全部多物体数据上训练，再针对目标类别 Fine-tune。

注意：具体类别与轨迹划分、每个类别采用哪种训练路径，需要进一步查实现。

### Q40. Behavior Cloning（BC）是什么？

把 Expert 在某一 Observation 下采取的 Action 当作监督标签，让 Student 学习模仿。

常见示意损失：

$$
L_{BC}=\mathbb E_{(o,a^E)}\left[\|\pi_S(o)-a^E\|_2^2\right]
$$

**注意：此平方误差仅为教学示例。** 论文在该节没有明确给出 BC 的确切 Loss。

### Q41. 为什么 BC 后还要 DAgger？

BC 的训练数据主要来自 Expert 的状态分布；Student 真正执行时可能犯错，进入数据中很少出现的状态，即 **Distribution Shift**。

DAgger 的基本思想：收集 Student 实际访问到的状态，让 Expert 为这些状态给出动作标签，再聚合数据继续训练 Student，以减轻误差累积。

**[待确认]** GALATEA 的 DAgger 采样比例、专家标签的产生方式和迭代细节，需要查代码。

### Q42. 为什么必须蒸馏成单个 Policy？能不能部署多个 Expert 再自动选择？

**可以，不是必须蒸馏。** 我们讨论过另一个方案：保留多个 Expert，用 **Policy Selector / Gating Network** 根据任务、物体、Observation 等信息选择其中一个。

- **Hard Gating**：每次选择一个 Expert 输出 Action。
- **Soft Gating**：按权重融合多个 Expert Action（接触操作中直接平均动作可能不稳定）。

潜在好处：保留各 Expert 的能力，避免部分蒸馏损失。主要问题：选择器如何训练、同物体类别多 Expert 怎样区分、切换时如何避免 Action 突变，以及模型存储与运行成本。见 **Idea 6**。

---

# Part II. 复现时想尝试的改进

> 原则：每个 Idea 都是**待验证假设**。第一次实验尽量仅改变一个因素，控制随机种子、训练步数、评估场景和成功率统计方式；除了 SR，也观察收敛速度、接触行为及失败类型。

## Idea 2 — Actor Observation 显式加入 Fingertip Tracking Error

**对应：** III-B.2 Observation。  
**原始疑问：** $e_t^{HOI}$ 虽然已有 MANO Keypoint Errors，但是否充分、显式地向 Actor 提供每个指尖的跟踪偏差？

可尝试增加：

$$
e_{t,i}^{tip}=p_{t,i}^{tip}-\widehat p_{t,i}^{tip},\qquad i=1,\dots,5
$$

每个指尖 3 维，共 15 维（具体坐标系必须统一）。

- **潜在收益：** 让 Actor 更直接地感知每个指尖到参考指尖的位置误差。
- **主要风险：** 可能与现有 $e_t^{HOI}$ 完全或部分重复，增加冗余特征却不带来收益。
- **第一步：** 查代码中 $e_t^{HOI}$ 的确切展开维度；如已有对应信息，可改为测试特征编码方式，而非盲目增加特征。
- **消融：** Baseline / Baseline + explicit fingertip errors。

## Idea 3 — Smooth Lift Gate（平滑举升门控）

**对应：** III-B.2 Hand Tracking Reward。  
**原始问题：** 当参考物体已举升超过 5 cm 时，原 $g_t^L$ 会随实际高度越过 5 cm 由 0 突变到 1，导致 Dense Hand Tracking Reward 不连续。

**你提出的修改：** 参考举升超过 5 cm 时，让实际物体举升 **≤5 cm 时 $g=0$，5–8 cm 平滑增加，≥8 cm 时 $g=1$**；参考还没举升超过 5 cm 时仍保持 $g=1$。

设：

$$
z=\frac{h_t-h_0-0.05}{0.03}
$$

当 $\widehat h_t-h_0>0.05$：

$$
g_t^{L,\mathrm{smooth}}=
\begin{cases}
0,&z\le0\\
3z^2-2z^3,&0<z<1\\
1,&z\ge1
\end{cases}
$$

若参考未要求超过 5 cm 的举升，则 $g_t^{L,\mathrm{smooth}}=1$。

- **备选：** 线性 Ramp $g=\operatorname{clip}(z,0,1)$。
- **潜在收益：** 缓解 Hand Tracking Reward 在阈值附近的突然切换。
- **主要风险：** 改变激励强度，可能延后对抬升的奖励；$r_t^{lift}$ 仍在 5 cm 发生阶跃。
- **消融：** Binary baseline / Linear ramp / Smoothstep。
- **观察指标：** SR、举升高度随时间变化、门控阈值附近的奖励和动作稳定性。

## Idea 4 — Progress-based Fingertip Approach Reward

**对应：** III-B.2 Near Reward。  
**原始问题：** $d_i\le1.5$ cm 后 Near Reward 饱和，无法识别指尖在近距离范围内逐步靠近还是远离。

定义指尖距离进展：

$$
\Delta d_{t,i}=d_{t-1,i}-d_{t,i}
$$

**候选 A：只奖励接近**

$$
r_t^{progress}=\lambda\sum_i\max(0,\Delta d_{t,i})
$$

**候选 B：接近奖励、远离惩罚**

$$
r_t^{progress}=\lambda\sum_i(d_{t-1,i}-d_{t,i})
$$

可考虑只在 **pre-contact** 阶段启用，接触后减少或停止该项，以避免干扰调整抓型及稳定操作。

- **潜在收益：** 显式鼓励指尖逐步接近物体，而不仅是停在 Near Reward 的平台内。
- **主要风险：** 有些合理动作必须临时后退；只奖励正向进展可能被反复靠近/后退利用；还需考虑不同指尖应对齐参考接触时序。
- **消融：** Original near / Progress only / Original near + progress；可进一步比较 signed / nonnegative progress。
- **观察指标：** 接触前耗时、接触成功率、动作抖动、最终抓取 SR。

## Idea 5 — Contact-aware / Lift-aware Episode Initialization

**对应：** III-B.2 Episode Initialization。  
**原始问题：** 从整段 HOI 轨迹随机 Reset，如果采到物体已悬空的帧，但 22 个手指关节又张开，物体可能立刻掉落。

**学长反馈：** 已尝试过短暂施加虚拟向上力抵消重力；最终认为没有虚拟力效果也不错，因而未采用。Policy 可能学会快速收紧手指；训练中仍可能掉落。

| 方案 | 初始化策略 | 主要取舍 |
|---|---|---|
| **Baseline** | 全轨迹随机采样，手指张开 | 保留全阶段覆盖，但包含困难的悬空初始状态 |
| **Variant A** | 只从尚未悬空的帧采样 | 降低悬空掉落，但减少中后期训练覆盖 |
| **Variant B** | 全轨迹采样，悬空时加短暂虚拟支撑 | 允许后期 Reset，但加入额外非真实物理力 |
| **Variant C** | 全轨迹采样，按接触/举升阶段调整采样权重 | 在覆盖与困难度之间折中 |

- **注意：** $t_{init}\le t_c$ 只是初期想到的简化限制，并不是唯一方案，也未必最好。
- **需要核对：** 采样分布、虚拟升力原型是否保留、Evaluation 的初始条件和成功率定义。
- **消融指标：** Training SR、Evaluation SR、按阶段划分的失败率、学习曲线、接触后的恢复能力。

## Idea 6 — 用 Expert Selection 替代单一 Policy Distillation

**对应：** III-B.3 Policy Distillation。  
**原始疑问：** 多个 Expert 都已训练好了，为什么一定要 BC + DAgger 合并成一个 Policy？是否可以部署时自动选择？

**候选架构：**

$$
k_t=g(o_t,\mathrm{task}),\qquad a_t=\pi_{k_t}(o_t)
$$

其中 $g$ 是 Gating Network / Selector，$\pi_{k_t}$ 是被选中的 Expert。

- **潜在收益：** 保留各 Expert 的独立性能，避免蒸馏时的能力损失；增添专家也更直接。
- **主要风险：** 如何在同一物体类别的不同 Expert 间选择？如何控制切换频率和 Action 突变？额外模型的存储/推理成本如何？
- **优先比较：**
  1. GALATEA 原 BC + DAgger Student。
  2. Oracle / 已知任务标签选择 Expert（先测专家选择的性能上限；注意不能用于真实未知标签部署）。
  3. Learned Gating Network 自动选择 Expert。
- **评价：** 任务 SR、类别泛化、推理延迟、模型存储、切换失败率。

---

# Part III. 复现前优先核查的代码与实验细节

| 核查内容 | 目的 |
|---|---|
| $e_t^{HOI}$ 的具体维度与坐标系 | 判断 Idea 2 是否增加了新信息 |
| Hand Tracking 与 Object Tracking Reward 实现 | 核对 Gate、Keypoint 误差以及接触阶段系数 |
| $r_t^{reg}$ 的具体表达式 | 补齐论文未展开的正则项 |
| Near Reward 使用的距离计算及接触判定 | 验证 Idea 4 的阈值和物理语义 |
| Episode Reset 的帧采样规则 | 确立 Idea 5 的 Baseline |
| 张开手型、悬空物体和虚拟升力的实现 | 验证学长说明与当前代码一致 |
| Evaluation Reset 与 Success Rate 定义 | 避免比较不同初始化难度造成的假提升 |
| SAPG 的 Exploration Settings | 不将教学示例误写成实际实现 |
| 专家轨迹划分与专家模型对应关系 | 为 Idea 6 设计公平的 Expert Selector |
| BC Loss、DAgger 数据收集与更新细节 | 完整复现 Policy Distillation |

## 改进优先级建议

1. **先核对代码：** Idea 2、5、6 的有效性强烈依赖实现细节。
2. **优先做 Reward 消融：** Idea 3（Smooth Lift Gate）和 Idea 4（Progress-based Near Reward）改动相对局部，适合先做单因素实验。
3. **再做初始化策略比较：** Idea 5 要确保各阶段的训练覆盖和评估口径可比。
4. **最后尝试架构替代：** Idea 6 牵涉多个 Experts 的集成、选择和切换，工作量较大。

**科研记录原则：** 所有改动先提出假设，再通过消融验证；不能将“感觉更合理”直接等同于成功率一定更高。

---

# 附录：III-A 的 Idea 1（为保持编号完整）

## Idea 1 — Temporal Loss 去掉一阶时间差分惩罚

**对应：** III-A Joint Hand–Object Optimization（不属于 III-B）。

原论文 Temporal Loss 同时惩罚一阶、二阶时间差分，鼓励轨迹平滑。

**提出的问题：** 对合法的手/物体运动速度进行直接惩罚，可能过度抑制正常运动。

**候选修改：** 删除一阶差分项，只保留二阶差分项，比较是否能保留合理运动速度，同时抑制不自然的快速加速度变化。

**必须验证：** Reconstruction 定量误差、视频重建观感、接触几何、下游 RL 成功率。不能预设该修改优于原论文。
