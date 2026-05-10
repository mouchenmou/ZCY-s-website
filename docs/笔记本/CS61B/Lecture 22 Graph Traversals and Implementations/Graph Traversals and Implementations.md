# Shortest Paths

---

## 1. 回顾：DFS vs BFS 找路径

之前学过的两种图遍历方式，在**找路径**这个问题上的对比：

| 考量 | DFS | BFS |
|------|-----|-----|
| 正确性 | ✓ 都能找到路径 | ✓ 都能找到路径 |
| 输出质量 | 随便一条路径 | **最短路径**（边数最少），2合1 deal |
| 时间效率 | $O(V+E)$，两者差不多 | $O(V+E)$ |
| 空间效率 | 深瘦图差（调用栈很深） | 宽胖图差（队列会很大） |

!!! warning "BFS 用于地图导航的问题"
    BFS 返回的是**边数最少**的路径，但在实际地图中，我们关心的是**总距离最短**。
    
    比如从起点 $s$ 到终点 $t$，BFS 可能选了一条有 3 条边但总距离 190 的路，而实际最优路径虽然边数多但总距离只有 110。
    
    **核心问题：** BFS 没有考虑边的权重（edge weights）！

---

## 2. Dijkstra's Algorithm

### 2.1 问题定义

**Single Source Single Target Shortest Paths：** 给定一个带权图，找从源点 $s$ 到目标 $t$ 的最短路径。

**Single Source Shortest Paths：** 找从 $s$ 到**每个可达顶点**的最短路径。

!!! example "示例"
    找从镇 0 到镇 5 的最短路径：
    
    - 最佳路径：$0 \to 2 \to 4 \to 5$，总长度 9 英里
    - 注意：路径 $0 \to 2 \to 5$ 虽然只经过 3 个镇，但总长度是 16 英里，不是最短的
    
    观察：最短路径一定**没有环**（假设边权重非负）。

### 2.2 Shortest Paths Tree (SPT)

对于 Single Source Shortest Paths 问题，解的结构是一棵树：

- 如果图有 $V$ 个顶点且连通，SPT 有 **$V-1$ 条边**。
- 每个顶点（除了源点）恰好有一条输入边。
- 可以看作是从 $s$ 到所有顶点的最短路径的并集。

### 2.3 核心概念：Edge Relaxation（边松弛）

这是整个算法的灵魂操作。

**松弛一条边 $p \to q$（权重为 $w$）：**

```
如果 distTo[p] + w < distTo[q]:
    distTo[q] = distTo[p] + w
    edgeTo[q] = p
```

意思就是：如果通过 $p$ 走到 $q$ 比目前已知的到 $q$ 的路径更短，就**更新** $q$ 的距离和前驱。

!!! tip "手动模拟找 SPT"
    以源点 A 为例，手动标记每个点到 A 的最短距离：
    
    ```
        A──(5)──B
        │       │
       (1)     (2)
        │       │
        C──(5)──D
    ```
    
    从 A 出发：
    - A 到 A：0
    - A 到 C：1（直接走 A→C）
    - A 到 B：2（走 A→C→B，而不是直接 A→B 的 5）
    - A 到 D：4（走 A→C→B→D）
    
    所以 SPT 的边是：A→C, C→B, B→D。

### 2.4 从错误算法到正确算法

#### 坏算法 #1：直接 DFS

按 DFS 顺序加边到 SPT 里，**不管距离**。

结果：可能选到很长的路径。比如 A→B 直接加了 5 的距离，但其实走 A→C→B 只需要 2。

#### 坏算法 #2：DFS + 松弛

DFS 遍历，但遇到更短的路径时就更新（松弛）。

问题：DFS 的遍历顺序是"深度优先"，可能先走了很远的分支，后来才发现近的路径，但已经来不及回头了。

#### 正确算法：Dijkstra（Best First Search）

**核心思想：每次都先处理"目前已知距离源点最近"的那个顶点。**

```
A 的距离是 0（最小），先处理 A：
  A→B 是 5，< ∞，更新
  A→C 是 1，< ∞，更新

C 的距离是 1（现在最小），处理 C：
  C→B 是 1+1=2，比 5 好，更新！
  C→D 是 1+5=6，< ∞，更新

B 的距离是 2（现在最小），处理 B：
  B→A 是 2+3=5，不比 0 好
  B→D 是 2+2=4，比 6 好，更新！

D 的距离是 4，处理 D：没有更好的了
```

### 2.5 Dijkstra 算法伪代码

```java
// 初始化
PQ.add(source, 0)
for (其他顶点 v) PQ.add(v, infinity)

// 主循环
while (PQ 不为空) {
    p = PQ.removeSmallest()  // 取出最近的顶点
    // 松弛 p 的所有出边
    for (每条边 p→q，权重 w) {
        if (distTo[p] + w < distTo[q]) {
            distTo[q] = distTo[p] + w
            edgeTo[q] = p
            PQ.changePriority(q, distTo[q])
        }
    }
}
```

**关键不变量：**
- `edgeTo[v]` 是 v 的当前最优前驱
- `distTo[v]` 是已知的从源点到 v 的最优距离
- PQ 按 `distTo` 排序存储所有未访问的顶点

**重要性质：**
- 总是按从源点的**总距离从小到大**访问顶点
- 对已访问（白色）顶点的松弛**总是失败**（不会再更新）

---

## 3. Dijkstra 的正确性与运行时间

### 3.1 为什么 Dijkstra 是正确的？

**定理：** 如果所有边权重非负，Dijkstra 保证返回正确结果。

**证明思路（归纳法）：**

- **Base case：** `distTo[source] = 0`，显然最优。
- **Inductive step：** 假设之前出队的所有顶点的 `distTo` 都是最优的。
  - 现在出队的是 `distTo` 最小的顶点 $v_1$。
  - 对于任何还没出队的顶点 $p$，有 `distTo[p]` ≥ `distTo[v1]`。
  - 任何经过 $p$ 再到 $v_1$ 的路径长度至少是 `distTo[p] + w` ≥ `distTo[v1]`（因为 $w \ge 0$）。
  - 所以 $v_1$ 的 `distTo` 不可能再被改善。
  - 由归纳法，对所有顶点成立。

### 3.2 负权重边的问题

!!! danger "Dijkstra 在有负权重边时会失败！"
    **原因：** 已访问（白色）顶点的松弛**可能成功**。
    
    **例子：**
    ```
    顶点 33 → 34（权重 1）
    顶点 34 → 14（权重 -6）
    ```
    即使 34 出队时的 `distTo` 比 14 大，它仍然可以通过负权重边 -6 来更新已经访问过的顶点 14 的 `distTo`。这破坏了"已访问顶点的 distTo 不再改变"这个关键性质。
    
    所以 Dijkstra **不能处理负权重边**。

### 3.3 时间复杂度

假设使用**二叉堆**实现的优先队列：

| PQ 操作 | 次数 | 每次代价 | 总代价 |
|---------|------|----------|--------|
| `add` | $V$ | $O(\log V)$ | $O(V \log V)$ |
| `removeSmallest` | $V$ | $O(\log V)$ | $O(V \log V)$ |
| `changePriority` | $E$ | $O(\log V)$ | $O(E \log V)$ |

**总时间复杂度：** $O(E \log V)$（假设 $E \ge V$）

---

## 4. A* 算法

### 4.1 Dijkstra 用于导航的问题

Dijkstra 会**均匀地向四周扩散**。如果用来做地图导航（比如从 Denver 到 NYC），它会探索 Denver 周围几乎所有地方，包括完全反方向的区域。

但实际上我们心里有数：NYC 在东边，应该**优先向东探索**。

### 4.2 A* 的核心思想

**简单想法：** 访问顶点的优先级不是只看 `d(source, v)`，而是看：

$$\text{priority} = d(source, v) + h(v, goal)$$

其中 $h(v, goal)$ 是从 $v$ 到目标的**估计距离**（heuristic）。

- 和 Dijkstra 的区别：Dijkstra 只看已经走了多远，A* 还**预估还要走多远**。
- 例子：Henderson 比 Englewood 离 Denver 更远，但 Henderson 更靠东，所以整体上可能更接近 NYC。A* 会优先考虑 Henderson。

### 4.3 A* 伪代码

```java
// 初始化：优先级 = distTo + h(v, goal)
PQ.add(source, 0 + h(source, goal))
for (其他顶点 v) PQ.add(v, infinity + h(v, goal))

while (PQ 不为空) {
    v = PQ.removeSmallest()  // 按 d + h 排序
    if (v == goal) return 找到路径!
    // 松弛 v 的所有出边
    for (每条边 v→w，权重 weight) {
        if (distTo[v] + weight < distTo[w]) {
            distTo[w] = distTo[v] + weight
            edgeTo[w] = v
            PQ.changePriority(w, distTo[w] + h(w, goal))  // 注意这里加了 h
        }
    }
}
```

### 4.4 启发函数的例子

对于地图导航，可以用**直线距离（as-the-crow-flies distance）**作为启发函数：

```java
/** h(v, goal) 在算法运行过程中不变 */
public double h(Vertex v, Vertex goal) {
    return computeLineDistance(v.latLong, goal.latLong);
}
```

**不需要完美！** 启发函数就是个估计，不用精确。

---

## 5. 启发函数的质量（CS188 预览）

### 5.1 极端情况分析

| 设置 | 效果 |
|------|------|
| $h(v, goal) = 0$ | 退化为 **Dijkstra**（没有方向性） |
| $h(v, goal) = 10000$（常数） | 还是 Dijkstra（每个点都加同一个数，相对顺序不变） |

### 5.2 坏的启发函数

!!! danger "启发函数高估会导致错误结果"
    如果你"觉得"伊利诺伊州和印第安纳州是荒郊野岭，给它们设 $h = 100000$：
    
    - 算法会**绕开这些州**，即使穿过它们可能是最短路径
    - **结果错误！**

### 5.3 保证正确性的条件

要让 A* 给出正确结果，启发函数需要满足：

- **Admissible（可接纳）：** $h(v, goal) \le$ 从 $v$ 到 $goal$ 的**真实**最短距离。
  - 简单说：**永远不要高估**。
  
- **Consistent（一致）：** 对于每个邻居 $w$：
  $$h(v, goal) \le dist(v, w) + h(w, goal)$$
  - 简单说：从 $v$ 到目标的估计距离，不超过走到邻居的实际距离加上邻居到目标的估计距离。

!!! tip
    这部分是 AI 课程 CS188 的内容，CS61B 不要求掌握。
    
    你只需要知道：**启发函数的选择很重要，选错了 A* 会给出错误答案。**
    
    所有 consistent 的启发函数一定是 admissible 的。

---

## 6. 总结：图问题全景

| 问题 | 描述 | 算法 | 时间复杂度 |
|------|------|------|-----------|
| **找路径** | 从 $s$ 到每个可达顶点的一条路径 | DFS | $O(V+E)$ |
| **无权最短路径** | 从 $s$ 到每个可达顶点的最短路径（边数最少） | BFS | $O(V+E)$ |
| **有权最短路径（全目标）** | 从 $s$ 到每个顶点的最短加权路径 | Dijkstra | $O(E \log V)$ |
| **有权最短路径（单目标）** | 从 $s$ 到某个特定 $t$ 的最短加权路径 | A* | 取决于启发函数 |

**关键洞察：**
- 当边权重都相等时，Dijkstra 就是 BFS。
- 当 $h(v, goal) = 0$ 时，A* 就是 Dijkstra。
- A* 通过启发函数提供了**方向性**，在单目标问题中通常比 Dijkstra 快很多。
