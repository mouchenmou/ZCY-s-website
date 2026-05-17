---
comment: True
counter: True
---

# 第七章 Laplace 变换

!!! abstract
    本章研究 Laplace 变换的定义、性质与逆变换。核心思想是将时域中的微分/积分运算转化为复频域中的代数运算，从而方便求解微分方程和积分方程。

## 7.1 定义与存在性

### 7.1.1 定义

设 $f(t)$ 定义在 $[0, +\infty)$ 上，$s = \sigma + i\omega \in \mathbb{C}$，若积分收敛，则

$$\mathcal{L}[f(t)] = F(s) = \int_0^{+\infty} f(t)e^{-st}\,dt$$

- $F(s)$ 称为 **像函数** （image function）
- $f(t)$ 称为 **原像函数** （original function）
- 逆变换：$f(t) = \mathcal{L}^{-1}[F(s)]$

### 7.1.2 存在定理

**充分条件** ：若 $f(t)$ 满足

1. 在任意有限区间上 **分段连续**
2. **指数阶增长** ：$|f(t)| \leq Me^{ct}$（$M > 0, c \geq 0$）

则 $F(s)$ 在半平面 $\text{Re}(s) > c$ 上存在且 **解析** 。

### 7.1.3 常用 Laplace 变换表

| 原函数 $f(t)$ | 像函数 $F(s)$ | 收敛域 |
|--------------|-------------|--------|
| $u(t)$（单位阶跃） | $\frac{1}{s}$ | $\text{Re}(s) > 0$ |
| $e^{kt}$ | $\frac{1}{s - k}$ | $\text{Re}(s) > k$ |
| $t^n \cdot u(t)$ | $\frac{n!}{s^{n+1}}$ | $\text{Re}(s) > 0$ |
| $t^a$ | $\frac{\Gamma(a+1)}{s^{a+1}}$ | $\text{Re}(s) > 0$ |
| $\sin kt$ | $\frac{k}{s^2 + k^2}$ | $\text{Re}(s) > 0$ |
| $\cos kt$ | $\frac{s}{s^2 + k^2}$ | $\text{Re}(s) > 0$ |
| $\text{sh}\,wt$ | $\frac{w}{s^2 - w^2}$ | $\text{Re}(s) > |w|$ |
| $\text{ch}\,wt$ | $\frac{s}{s^2 - w^2}$ | $\text{Re}(s) > |w|$ |

---

## 7.2 Laplace 变换的性质

### 7.2.1 线性性

$$\mathcal{L}[a_1 f_1(t) + a_2 f_2(t)] = a_1 F_1(s) + a_2 F_2(s)$$

### 7.2.2 时移性

$$\mathcal{L}[f(t - \tau)] = e^{-s\tau}F(s) \quad (\tau > 0)$$

即：时间延迟 $\tau$ $\Rightarrow$ 乘以因子 $e^{-s\tau}$。

### 7.2.3 频移性

$$\mathcal{L}[e^{s_0 t}f(t)] = F(s - s_0)$$

即：时域乘以指数 $\Rightarrow$ 频域平移。

### 7.2.4 微分性质（Differentiation）

| 方向 | 公式 |
|------|------|
| 象原函数 | $\mathcal{L}[f'(t)] = sF(s) - f(0^+)$ |
| 象原函数（$n$ 阶） | $\mathcal{L}[f^{(n)}(t)] = s^n F(s) - s^{n-1}f(0) - \cdots - f^{(n-1)}(0)$ |
| 象函数 | $\mathcal{L}[(-t)^n f(t)] = F^{(n)}(s)$ |

### 7.2.5 积分性质

| 方向 | 公式 |
|------|------|
| 象原函数 | $\mathcal{L}\left[\int_0^t f(\tau)d\tau\right] = \frac{1}{s}F(s)$ |
| 象原函数（$n$ 次） | $\mathcal{L}\left[\int_0^t \cdots \int_0^t f(t)dt^n\right] = \frac{1}{s^n}F(s)$ |
| 象函数 | $\mathcal{L}\left[\frac{f(t)}{t}\right] = \int_s^{+\infty} F(u)\,du$ |

### 7.2.6 极限性质

| 定理 | 公式 | 条件 |
|------|------|------|
| 初值定理 | $f(0^+) = \lim\limits_{s \to \infty} sF(s)$ | 极限存在 |
| 终值定理 | $f(+\infty) = \lim\limits_{s \to 0} sF(s)$ | $sF(s)$ 的奇点都在 $\text{Re}(s) < 0$ 半平面 |

### 7.2.7 卷积性质

**卷积定义** ：

$$[f_1 * f_2](t) = \int_0^t f_1(\tau)f_2(t - \tau)\,d\tau$$

**卷积定理** ：

$$\mathcal{L}[f_1(t) * f_2(t)] = F_1(s) \cdot F_2(s)$$

$$\mathcal{L}^{-1}[F_1(s) \cdot F_2(s)] = f_1(t) * f_2(t) \quad \text{（更常用）}$$

!!! example "利用卷积求逆变换"
    已知 $\mathcal{L}[f(t)] = \frac{1}{(s^2 + 4s + 13)^2}$，求 $f(t)$。

    $$\frac{1}{(s^2 + 4s + 13)^2} = \frac{1}{9} \cdot \frac{3}{(s+2)^2 + 3^2} \cdot \frac{3}{(s+2)^2 + 3^2}$$

    $$\mathcal{L}^{-1}\left[\frac{3}{(s+2)^2 + 3^2}\right] = e^{-2t}\sin 3t$$

    $$f(t) = \frac{1}{9}(e^{-2t}\sin 3t) * (e^{-2t}\sin 3t) = \frac{1}{54}e^{-2t}(\sin 3t - 3t\cos 3t)$$

---

## 7.3 Laplace 逆变换

### 7.3.1 反演积分（Bromwich 积分）

$$f(t) = \frac{1}{2\pi i}\int_{\beta - i\infty}^{\beta + i\infty} F(s)e^{st}\,ds \quad (t > 0)$$

其中 $\beta$ 大于 $F(s)$ 所有奇点的实部。

### 7.3.2 用留数计算逆变换

**定理** ：若 $F(s)$ 只有有限个奇点 $s_1, \ldots, s_n$（均在 $\text{Re}(s) = \beta$ 左侧），且 $\lim\limits_{s \to \infty} F(s) = 0$，则

$$f(t) = \sum_{k=1}^{n} \text{Res}[F(s)e^{st}, s_k]$$

### 7.3.3 两种方法

- **部分分式分解** ：$F(s)$ 为有理函数时，分解为简单分式后查表
- **留数法** ：通用方法，尤其适合高阶极点

!!! example "部分分式法"
    $F(s) = \frac{1}{s^2(s+1)} = \frac{1}{s^2} - \frac{1}{s} + \frac{1}{s+1}$

    $$f(t) = t - 1 + e^{-t}$$

!!! example "留数法"
    $F(s) = \frac{1}{s(s-1)^2}$，$s = 0$ 为一阶极点，$s = 1$ 为二阶极点：

    $$f(t) = \text{Res}[F(s)e^{st}, 0] + \text{Res}[F(s)e^{st}, 1]$$

    $$= \left.\frac{e^{st}}{(s-1)^2}\right|_{s=0} + \lim_{s \to 1}\frac{d}{ds}\left[\frac{e^{st}}{s}\right]$$

    $$= 1 + (te^t - e^t) = 1 + e^t(t - 1)$$

---

## 7.4 $\delta(t)$ 函数

### 7.4.1 定义

$$\delta(t) = \lim_{\tau \to 0} \sigma_\tau(t), \quad \sigma_\tau(t) = \begin{cases} \frac{1}{\tau}, & 0 < t < \tau \\ 0, & \text{其他} \end{cases}$$

满足归一化：$\int_{-\infty}^{+\infty} \delta(t)\,dt = 1$

### 7.4.2 Laplace 变换

$$\mathcal{L}[\delta(t)] = 1$$

---

## 7.5 用 Laplace 变换解微分方程

### 7.5.1 求解步骤

1. 对方程两边取 Laplace 变换（利用微分性质处理导数项）
2. 代入初始条件，解出 $X(s)$
3. 求 $X(s)$ 的逆变换得到 $x(t)$

### 7.5.2 核心思路

时域中的 **微分运算** $\xrightarrow{\mathcal{L}}$ 复频域中的 **代数运算**

| 时域 | 频域 |
|------|------|
| $x(t)$ | $X(s)$ |
| $x'(t)$ | $sX(s) - x(0)$ |
| $x''(t)$ | $s^2 X(s) - sx(0) - x'(0)$ |
| $x^{(n)}(t)$ | $s^n X(s) - s^{n-1}x(0) - \cdots - x^{(n-1)}(0)$ |

!!! example "解 $x'' - 2x' + 2x = 2e^t\cos t$，$x(0) = x'(0) = 0$"

    取 Laplace 变换：

    $$(s^2 - 2s + 2)X(s) = \frac{2(s-1)}{(s-1)^2 + 1}$$

    $$X(s) = \frac{2(s-1)}{[(s-1)^2 + 1]^2}$$

    用频移性质（$a = 1$）：

    $$x(t) = e^t \mathcal{L}^{-1}\left[\frac{2s}{(s^2+1)^2}\right]$$

    注意到 $\frac{2s}{(s^2+1)^2} = -\frac{d}{ds}\left(\frac{1}{s^2+1}\right)$，由象函数微分性质：

    $$\mathcal{L}^{-1}\left[\frac{2s}{(s^2+1)^2}\right] = t\sin t$$

    $$x(t) = t e^t \sin t$$

### 7.5.3 解积分方程

Volterra 型积分方程中的卷积项直接利用卷积定理处理。

!!! example "解 $y(t) + \int_0^t y(t-u)e^u du = 2t - 3$"

    取 Laplace 变换，左边积分项为 $Y(s) \cdot \frac{1}{s-1}$：

    $$Y(s) + Y(s) \cdot \frac{1}{s-1} = \frac{2}{s^2} - \frac{3}{s}$$

    $$Y(s) \cdot \frac{s}{s-1} = \frac{2-3s}{s^2}$$

    $$Y(s) = \frac{(s-1)(2-3s)}{s^3} = -\frac{2}{s^3} + \frac{5}{s^2} - \frac{3}{s}$$

    $$y(t) = -t^2 + 5t - 3$$
