# Lecture 24: Minimum Spanning Trees

!!! abstract
    本章研究最小生成树（MST）问题。核心内容包括 Cut Property、Prim 算法和 Kruskal 算法，以及它们与最短路径算法的对比。

## 1. Warm-up：检测无向图中的环

**问题：** 给定一个无向图，判断它是否包含环。

### Approach 1：DFS

从任意顶点（比如 0）出发做 DFS：
- 标记访问过的顶点
- 如果遇到已标记的顶点，说明有环
- **注意：** 不要把自己来的那个父节点当成环（回溯边）

```
1 —— 2 —— 3
|         |
0 —— 5 —— 4 —— 6
```

从 0 出发 DFS，走到 1→2→3→4→5，发现 5 已标记，**但 5 是 0 的邻居且是来的方向**，不算环。

**时间复杂度：** $O(V + E)$，更紧的界是 $O(V)$（因为一旦找到环就可以停止）。

### Approach 2：并查集（WeightedQuickUnionUF）

- 遍历每条边
- 检查两个端点是否 already connected
  - 如果**没有**：union 它们
  - 如果**已经**连通：说明有环

```python
uf = WeightedQuickUnionUF(G.V())
for each edge (v, w) in G:
    if uf.connected(v, w):
        return True  # 有环
    uf.union(v, w)
return False  # 无环
```

**时间复杂度：** $O(V + E \cdot \alpha(V))$（带路径压缩的并查集，$\alpha(V)$ 是逆 Ackermann 函数）。

---

## 2. 生成树（Spanning Tree）

### 定义

给定一个无向图 $G$，**生成树** $T$ 是 $G$ 的一个子图，满足：
1. **连通**（Connected）
2. **无环**（Acyclic）
3. **包含所有顶点**（Includes all vertices）

前两个性质保证了它是**树**，第三个性质保证了它是**生成**的。

```
原图 G：                生成树 T：
    A                       A
   /|\\                    / |
  B | D                  B  C  D
   \\|/                    |
    C                      E
```

**最小生成树（MST）：** 总权重最小的生成树。

### MST vs SPT

!!! warning "MST ≠ 最短路径树（SPT）"
    - **SPT 依赖于源点：** 它告诉你从某个特定的 $s$ 出发到所有其他点的最短路径
    - **MST 没有源点：** 它关心的是整体权重最小

    **例子：**
    ```
        A──(2)──B
        |       |
       (2)     (2)
        |       |
        C──(3)──D
    ```
    - MST：选三条权重为 2 的边，总权重 = 6
    - SPT from A：A→B(2), A→C(2), C→D(3) 或 B→D(2)，总权重可能不同
    - 当源点为 B 时，SPT 恰好等于 MST，但其他源点不一定

    **结论：** MST 有时恰好是某个源点的 SPT，但两者解决的是完全不同的问题。

### MST 应用
- 老式手写识别
- 医学影像（如癌细胞核排列分析）
- 网络设计（最小成本连通所有节点）

---

## 3. Cut Property（割性质）

### 定义

- **Cut（割）：** 将图的顶点分成两个**非空集合**
- **Crossing Edge（横跨边）：** 连接两个集合中顶点的边

```
Set 1: {2, 3, 5, 6}       Set 2: {0, 1, 4, 7}

Crossing edges: 0-2, 1-3, 1-5, 2-7, 4-5, 4-7, 6-0, 6-4
```

### Cut Property 定理

> **给定任意一个割，权重最小的横跨边一定在 MST 中。**

（假设边权重互不相同）

!!! proof "Cut Property 证明（反证法）"
    假设权重最小的横跨边 $e$ **不在** MST 中。
    
    1. 把 $e$ 加到 MST 中，**一定会形成一个环**
    2. 这个环中一定还有**另一条横跨边** $f$（因为环要从 Set 1 走到 Set 2 再走回来）
    3. 去掉 $f$、加上 $e$，得到一棵新的生成树，且总权重更小（因为 $w(e) < w(f)$）
    4. **矛盾！** 原来的树不是 MST
    
    所以 $e$ **必须在** MST 中。

---

## 4. 通用 MST 算法

基于 Cut Property，可以得到一个通用的 MST 构造方法：

1. 初始时 MST 中没有任何边
2. 找一个**没有任何横跨边在 MST 中**的割
3. 把权重最小的横跨边加入 MST
4. 重复直到有 $V-1$ 条边

**问题：** 如何有效地找到这样的割？这就是 Prim 和 Kruskal 算法要解决的。

---

## 5. Prim 算法

### 算法描述

1. 从任意起始节点开始
2. 重复地选择**一端在 MST 内、一端在 MST 外**的权重最小的边，加入 MST
3. 重复直到有 $V-1$ 条边

### 为什么 Prim 有效？

Prim 是通用算法的特例。每次选择的割是：
- **Set 1：** 已经连到起始点的所有顶点
- **Set 2：** 其余顶点

所有 MST 内的边都在 Set 1 内部（没有横跨边），所以没有黑色边是横跨边。按权重递增考虑，最小横跨边一定属于 MST。

### Prim vs Dijkstra（核心区别）

|  | Dijkstra | Prim |
|------|----------|------|
| **访问顺序** | 按距离**源点**的距离从小到大 | 按距离**当前 MST** 的距离从小到大 |
| **松弛标准** | `distTo[p] + w < distTo[q]` | `e.weight() < distTo[w]` |
| **`distTo[w]` 含义** | 从源点到 $w$ 的最短距离 | 从 MST 到 $w$ 的最小边权重 |

!!! explanation "直观理解"
    - **Dijkstra** 想的是："我离**起点**有多近？"
    - **Prim** 想的是："我离**这棵树**有多近？"
    
    两者的代码结构几乎一样，唯一的区别在于 `distTo` 的更新方式。

### Prim 算法实现

```java
public class PrimMST {
    private Edge[] edgeTo;       // 连接到 MST 的最优边
    private double[] distTo;     // 到 MST 的最小边权重
    private boolean[] marked;    // 是否已在 MST 中
    private SpecialPQ<Double> fringe;  // 按 distTo 排序的优先队列

    public PrimMST(EdgeWeightedGraph G) {
        edgeTo = new Edge[G.V()];
        distTo = new double[G.V()];
        marked = new boolean[G.V()];
        fringe = new SpecialPQ<>(G.V());

        distTo[s] = 0.0;
        // 其余 distTo 设为无穷大
        setDistancesToInfinityExceptS(s);
        insertAllVertices(fringe);

        while (!fringe.isEmpty()) {
            int v = fringe.delMin();  // 取出离树最近的顶点
            scan(G, v);
        }
    }

    private void scan(EdgeWeightedGraph G, int v) {
        marked[v] = true;
        for (Edge e : G.adj(v)) {
            int w = e.other(v);
            if (marked[w]) { continue; }  // 已在 MST 中，跳过
            if (e.weight() < distTo[w]) { // 找到更近的边
                distTo[w] = e.weight();
                edgeTo[w] = e;
                fringe.decreasePriority(w, distTo[w]);
            }
        }
    }
}
```

**关键不变量：** fringe（优先队列）始终按当前已知的**到 MST 的最短距离**排序。

### 时间复杂度

与 Dijkstra 完全相同：

| PQ 操作 | 次数 | 每次代价 | 总代价 |
|---------|------|----------|--------|
| `add` | $V$ | $O(\log V)$ | $O(V \log V)$ |
| `delMin` | $V$ | $O(\log V)$ | $O(V \log V)$ |
| `decreasePriority` | $E$ | $O(\log V)$ | $O(E \log V)$ |

**总时间复杂度：** $O(E \log V)$（假设 $E > V$）

---

## 6. Kruskal 算法

### 算法描述

1. 初始所有边为灰色（未选择）
2. 按权重**从小到大**考虑每条边
3. 如果加入这条边**不会形成环**，就把它加入 MST（标记为黑色）
4. 重复直到有 $V-1$ 条边

### 为什么 Kruskal 有效？

也是通用算法的特例。假设加入边 $e = v \to w$：
- **Set 1：** 与 $v$ 连通的所有顶点
- **Set 2：** 其余顶点

因为不允许环，所以没有黑色边是横跨边。按权重递增考虑，当前最小横跨边一定属于 MST。

### Kruskal 算法实现

```java
public class KruskalMST {
    private List<Edge> mst = new ArrayList<>();

    public KruskalMST(EdgeWeightedGraph G) {
        MinPQ<Edge> pq = new MinPQ<>();
        for (Edge e : G.edges()) {
            pq.insert(e);  // 所有边加入优先队列
        }
        
        WeightedQuickUnionPC uf = new WeightedQuickUnionPC(G.V());
        
        while (!pq.isEmpty() && mst.size() < G.V() - 1) {
            Edge e = pq.delMin();
            int v = e.from();
            int w = e.to();
            if (!uf.connected(v, w)) {  // 不连通才加
                uf.union(v, w);
                mst.add(e);
            }
        }
    }
}
```

### 时间复杂度

| 操作 | 次数 | 每次代价 | 总代价 |
|------|------|----------|--------|
| `insert` | $E$ | $O(\log E)$ | $O(E \log E)$ |
| `delMin` | $E$ | $O(\log E)$ | $O(E \log E)$ |
| `union` | $V$ | $O(\log^* V)$ | $O(V \log^* V)$ |
| `isConnected` | $E$ | $O(\log^* V)$ | $O(E \log^* V)$ |

**总时间复杂度：** $O(E \log E)$

!!! tip "预处理优化"
    如果使用**预先排序**的边列表（而非优先队列），可以用 bottom-up heapification 在 $O(E)$ 时间内排序，总复杂度降到 $O(E \log^* V)$。

---

## 7. 算法对比总结

| 问题 | 算法 | 时间复杂度 | 备注 |
|------|------|-----------|------|
| 最短路径 | Dijkstra | $O(E \log V)$ | 不能处理负权边 |
| MST | Prim | $O(E \log V)$ | 类似 Dijkstra |
| MST | Kruskal | $O(E \log E)$ | 用并查集 |
| MST | Kruskal（预排序） | $O(E \log^* V)$ | 用并查集 |

### Prim vs Kruskal 直观对比

```
Prim: 从一个点开始"长"出一棵树，每次加离树最近的边
Kruskal: 从所有边里挑最小的，一段一段地拼接森林

Prim 的 MST 始终是一棵树
Kruskal 的中间结果是一个森林（多棵树），最后才合并成一棵
```

---

## 8. MST 算法前沿（拓展）

最优比较型 MST 算法的发展：

| 年份 | 最坏情况复杂度 | 发现者 |
|------|---------------|--------|
| 1975 | $E \log \log V$ | Yao |
| 1984 | $E \log^* V$ | Fredman-Tarjan |
| 1986 | $E \log (\log^* V)$ | Gabow-Galil-Spencer-Tarjan |
| 1997 | $E \alpha(V) \log \alpha(V)$ | Chazelle |
| 2000 | $E \alpha(V)$ | Chazelle |
| 2002 | 最优 | Pettie-Ramachandran |
