---
comment: True
counter: True
---


# 基础数据结构

!!! abstract
    「数据结构基础」课程 树、堆、线段树、并查集、图 部分内容复习与总结

## 1. 树
### 1.1 基础概念
- 树包括根节点（root），0 个或多个非空子树 $T_1,\cdots, T_k$ 与根通过一条有向边连接
- 每棵子树的根叫做 root 的儿子（children），root 是每棵子树根的父亲（parent）
- 一棵树是 $N$ 个节点和 $N-1$ 条边的集合
- 没有儿子的节点称为树叶（leaf）
- 有相同父亲的节点称为兄弟（siblings）
- 节点的儿子数量称为这个节点的度（degree）
- 一棵树的度是这棵树里所有节点度的最大值
- 从节点 $n_1$ 到 $n_k$ 的路径是唯一的，其长度是路径上边的数量
- 对于节点 $n_i$，其深度为从根到 $n_i$ 的唯一路径的长度
- 对于节点 $n_i$，其高度为从 $n_i$ 到一个叶节点的最长长度
- 根的高度称为这棵树的高度/深度
- 一个节点的祖先（ancestors）是从根到这个节点的路径上的所有节点
- 一个节点的后裔（descendants）是这个节点子树中的所有节点

#### 1.1.1 树的表示
- 列表表示
    - 子节点个数未知，不易表示
- FirstChild-NextSibling 表示法
    - 记录第一个子节点和下一个兄弟节点
    - 因为一棵树的儿子顺序不定，所以一棵树的表示方式不唯一
    ```c 
    struct TreeNode {
        ElementType Element;
        PtrToNode FirstChild;
        PtrToNode NextSibling;FirstChildFirfsads
    };
    typedef struct TreeNode *PtrToNode;
    ```

### 1.2 二叉树
- 二叉树（binary tree）是每个节点最多有两个儿子的树
- 每棵树都可以用二叉树来表示
    - 即通过 FirstChild-NextSibling 表示法
    - 将 FirstChild 视为左儿子，NextSibling 视为右儿子
- 完全二叉树（complete binary tree）是所有叶节点都在相邻的两层上的二叉树
    - 除了最后一层，每一层都是满的
    - 最后一层的节点都靠左排列
- 第 $i$ 层的节点数最多为 $2^{i-1}$
- 深度为 $k$ 的二叉树最多有 $2^k - 1$ 个节点
- $n_0$ 表示叶节点数，$n_2$ 表示度为 2 的节点数，则 $n_0 = n_2 + 1$
- 二叉树可以通过数组来表示
    - 根为 tree[1]
    - 节点 tree[i] 的左儿子为 tree[2i]，右儿子为 tree[2i+1]
    - 完全二叉树的数组中元素全部分布在 1 ~ n 中
- 表达式树（expression tree）
    - 二叉树的一种，每个节点都是一个运算符，叶节点是操作数
    - 用于表示算术表达式

#### 1.2.1 二叉树的遍历
- 先序遍历（preorder traversal）
    - 根 -> 左 -> 右
- 后序遍历（postorder traversal）
    - 左 -> 右 -> 根
- 中序遍历（inorder traversal）
    - 左 -> 根 -> 右
    - 只适用于二叉树
    - 迭代写法
        ```c
        void iter_inorder(tree_ptr tree) {
            Stack S = create_stack();
            for (;;) {
                for (; tree; tree = tree->left)
                    push(S, tree);
                tree = top(S); pop(S);
                if (!tree) break;
                visit(tree->element);
                tree = tree->right;
            }
        }
        ```
- 层序遍历（level order traversal）
    - 从上到下，从左到右
    ```c 
    void levelorder(tree_ptr tree) {
        enqueue(tree);
        while (queue is not empty) {
            visit(T = dequeue());
            for (each child C of T) 
                enqueue(C);
        }
    }
    ```

#### 1.2.2 线索二叉树（Threaded Binary Tree）
- 如果一个节点的左儿子为空，那么它的左指针指向它的遍历前驱节点
- 如果一个节点的右儿子为空，那么它的右指针指向它的遍历后继节点
- 一定有一个 head node，左儿子为根，右儿子为自身
 Homework4中就有一个post-order threaded binary tree，可以作为参考。

### 1.3 二叉搜索树
- 二叉搜索树（binary search tree）是一种二叉树，非空情况下服从以下性质：
    - 所有节点都有不同的 key（一个整数）
    - 一个节点的左子树的所有节点的 key 都小于这个节点的 key
    - 一个节点的右子树的所有节点的 key 都大于这个节点的 key
    - 左子树和右子树也都是二叉搜索树
- 二叉搜索树的中序遍历是有序的
- 查找
    - 从根节点开始，如果 key 小于当前节点的 key，就往左子树找，否则往右子树找
    - 直到找到 key 相等的节点，或者找到空节点
    - 时间复杂度 $O(h)$，$h$ 为树的高度，即 $O(\log n)$
- 查询最小值：遍历到最左边的节点
- 查询最大值：遍历到最右边的节点
- 插入
    - 从根节点开始，如果 key 小于当前节点的 key，就在左子树递归插入，大于就在右子树递归插入
    - 直到找到空节点，然后插入
    - 或者找到相同的 key，忽略
    - 时间复杂度 $O(h)$，$h$ 为树的高度，即 $O(\log n)$
- 删除
    - 删除叶节点：直接删除即可
    - 删除只有一个儿子的节点：直接删除，然后把儿子接上
    - 删除有两个儿子的节点
        - 将该节点替换为左子树的最大值，或右子树的最小值
        - 递归删除左子树的最大值，或右子树的最小值
    - 也可以使用懒惰方法（删除不多的情况下），仅找到并标记删除，访问到时忽略这个节点

    ??? success "代码"
        ```c 
        SearchTree Delete(ElementType X, SearchTree T) {
            Position TmpCell;
            if (T == NULL) Error("not found");
            else if (X < T->Element) T->Left = Delete(X, T->Left);
            else if (x > T->Element) T->Right = Delete(X, T->Right);
            else {
                if (T->Left && T->Right) {
                    TmpCell = FindMin(T->Right);
                    T->Element = TmpCell->Element;
                    T->Right = Delete(T->Element, T->Right);
                } else {
                    TmpCell = T;
                    if (T->Left == NULL) T = T->Right;
                    else if (T->Right == NULL) T = T->Left;
                    free(TmpCell);
                }
            }
            return T;
        }
        ```


---

## 2. 堆
- 堆（heap）也称作优先队列（priority queue），支持插入值、找最小/大值，删除最小/大值
- 二叉堆（binary heap）是一种完全二叉树，满足以下性质：
    - 任意节点的值都不大于（或不小于）其父节点的值（即最大堆或最小堆）
- 插入
    - 先放到最后一个位置，然后和父节点比较，不满足条件则和父节点交换，直到满足
    ```c 
    void Insert(ElementType X, MaxHeap H) {
        if (IsFull(H)) {
            Error("Full Heap");
            return;
        }
        int i = ++H->Size;
        for (; H->Elements[i/2] < X; i /= 2) {
            H->Elements[i] = H->Elements[i/2];
        }
        H->Elements[i] = X;
    }
    ```
- 删除
    - 将最后一个叶节点放到根节点，然后和两个儿子比较，不满足则和最大（或最小）的儿子交换，直到满足
    ```c
    ElementType DeleteMax(MaxHeap H) {
        if (IsEmpty(H)) {
            Error("Empty Heap");
            return H->Elements[0];
        }
        int i, Child;
        ElementType MaxElement, LastElement;
        MaxElement = H->Elements[1];
        LastElement = H->Elements[H->Size--];
        for (i = 1; i * 2 <= H->size; i = Child) {
            Child = i * 2;
            if (Child != H->Size && H->Elements[Child + 1] > H->Elements[Child]) {
                Child++;
            }
            if (LastElement < H->Elements[Child]) {
                H->Elements[i] = H->Elements[Child];
            } else {
                break;
            }
        }
        H->Elements[i] = LastElement;
        return MaxElement;
    }
    ```
- PercolateUp：自下而上堆化（即这个元素可能会向上移动）
- PercolateDown：自上而下堆化（即这个元素可能会向下移动）
- 有了 Percolate 操作可以实现某个元素的增加减少，或者删除非最大最小的元素
- 从 $N$ 个元素的序列中构建一个堆
    - 一个个插入需要 $O(N\log N)$ 的复杂度
    - 可以直接将这个序列当作二叉树，然后从最后一个非叶节点开始，进行 PercolateDown，复杂度为 $O(N)$

---
## 3. 线段树

我之所以把线段树跟别的树分开，是因为线段树是今年新增的内容，我还不是很熟悉，所以单开一段，把内容写的详细一点。

### 3.1 动机

给定一个数组 `A`，需要**频繁**计算任意区间 `[L, R]` 的和（或最值等聚合操作）。

暴力做法每次遍历区间，复杂度 $O(N)$，作为频繁操作太慢。

线段树可以在 $O(\log N)$ 内完成**区间查询**和**单点更新**。

### 3.2 结构

线段树是一棵**满二叉树**，可以用数组存储：

```
数组 A:    [7, 2, 5, 3, 8]
            ↓
线段树:
                [0,4] = 25
           ──────────┐
        [0,2] = 14   [3,4] = 11
       ─────┴───     ────┴──
     [0,1]=9 [2]=5   [3]=3 [4]=8
     ┌──
    [0]=7 [1]=2
```

- 根节点 `tree[1]` 表示区间 `[0, n-1]`
- 节点 `tree[node]` 的左儿子是 `tree[2*node]`，右儿子是 `tree[2*node+1]`
- 叶子节点 `tree[node]` 存储 `A[start]`（单个元素）
- 内部节点存储其两个儿子的 **聚合结果** （比如和、最大值、最小值、gcd等等）
- 空间复杂度 $O(N)$，实际开 `4N` 是安全的（N 不是 2 的幂时，数组最大索引可能接近 4N）

### 3.3 建树

从根节点开始，递归地分割区间，到叶子节点时赋值为 `A[start]`，回溯时合并两个儿子的值。

```c
void Build(int node, int start, int end) {
    // 叶子节点
    if (start == end) {
        tree[node] = A[start];
        return;
    }
    int mid = (start + end) / 2;
    Build(2 * node, start, mid);           // 左儿子
    Build(2 * node + 1, mid + 1, end);     // 右儿子
    tree[node] = tree[2 * node] + tree[2 * node + 1];  // 合并
}
```

时间复杂度：$O(N)$（只跑一次）

!!! question "为什么 Build 的返回值是 void？"
    因为线段树用 **数组存储** ，根节点永远在 `tree[1]`。子节点位置由索引自动算出（`2*node` 和 `2*node+1`），不需要返回任何东西。这和指针实现的树不同——指针需要返回新节点地址才能"挂"上去，数组的索引本身就是"天然指针"。

### 3.4 区间查询

查询区间 `[L, R]` 时，从根节点往下走，有三种情况：

| 情况 | 条件 | 操作 |
|------|------|------|
| 完全无交集 | `R < start || end < L` | 返回 0（或单位元） |
| 完全包含 | `L <= start && end <= R` | 直接返回 `tree[node]` |
| 部分重叠 | 其他 | 递归查询左右儿子，合并结果 |

```c
int Query(int node, int start, int end, int L, int R) {
    // 完全无交集
    if (R < start || end < L) return 0;
    // 完全包含
    if (L <= start && end <= R) return tree[node];
    // 部分重叠
    int mid = (start + end) / 2;
    int left_sum = Query(2 * node, start, mid, L, R);
    int right_sum = Query(2 * node + 1, mid + 1, end, L, R);
    return left_sum + right_sum;
}
```

时间复杂度：$O(\log N)$

!!! question "为什么每一层最多用到 4 个元素？"
    任意查询区间 `[L, R]` 在线段树上分解时，每一层最多只有 **4 个节点被访问**：2 个在左边界附近的部分重叠节点 + 2 个在右边界附近的部分重叠节点。中间的全包含节点直接返回，不用往下递归。所以每层工作量是常数，总层数 $\log N$，总体 $O(\log N)$。

### 3.5 单点更新

将 `A[idx]` 更新为 `val`：从根节点找到对应的叶子节点，更新后回溯时重新计算每个祖先节点的值。

```c
void Update(int node, int start, int end, int idx, int val) {
    if (start == end) {       // 找到叶子节点
        tree[node] = val;
        return;
    }
    int mid = (start + end) / 2;
    if (idx <= mid)
        Update(2 * node, start, mid, idx, val);      // 往左走
    else
        Update(2 * node + 1, mid + 1, end, idx, val); // 往右走
    tree[node] = tree[2 * node] + tree[2 * node + 1];  // 回溯更新
}
```

时间复杂度：$O(\log N)$

### 3.6 注意事项

- 不限于求和，适用于 **任何满足结合律的区间聚合操作** ：`min`、`max`、`gcd`、`xor` 等
- 对于 **区间更新** （如给 `[L, R]` 所有元素加 10），需要用到 **lazy propagation**（延迟标记），这里就不讲了。。。
---
## 4. 并查集
- 并查集（disjoint set）是一种数据结构，支持两种操作：
    - 合并两个集合
    - 查询两个元素是否属于同一个集合
- 用树（森林）的方式表示连接情况，一棵树表示一个集合
    - 合并则将一棵树的根节点作为另一棵树的根节点的儿子
    - 查询是否属于同一个集合则判断两个元素的根节点是否相同
- 用数组记录即可，例如 S[i] 值表示 i 这个节点的父节点
- 查找：依次根据数组值向上递归查找根节点即可
- 合并：先查找两个元素的根节点，然后将其中一个根节点的父节点设置为另一个根节点即可

### 按大小合并
- 正常方式的查找合并可能会造成一个长链的情况
- 按大小合并（union-by-size）始终将小的树合并到大的树上，这样可以减少树的高度
- 记录则对于每个根，另 S[root] = -size，size 表示这个树的大小（初始为 -1）
- 通过按大小合并的带有 $N$ 个节点的树最高为 $\lfloor \log_2N\rfloor+1$

### 路径压缩
在查找的同时将路径上的所有节点的父节点都设置为根节点，这样可以减少树的高度

```c 
int find(int x) { 
    return ufs[x] == x ? x : ufs[x] = find(ufs[x]);
}
```

---

## 图
### 基础概念
- 有限的边集 $E=E(G)$ 和点集 $V=V(G)$
- 无向图（undirected graph）：边没有方向
- 有向图（directed graph）：边有方向
- 不考虑自环（self loop）和多重边（multigraph）
- 完全图（complete graph）：任意两个点之间都有边
- $v_i$ 和 $v_j$ 之间有一条无向边，则称：
    - $v_i$ 和 $v_j$ 相邻（adjacent）
    - $(v_i, v_j)$ is incident on $v_i$ and $v_j$
- 有一条从 $v_i$ 到 $v_j$ 的有向边，则称：
    - $v_i$ is adjacent **to** $v_j$
    - $v_j$ is adjacent **from** $v_i$
    - $<v_i, v_j>$ is incident on $v_i$ and $v_j$
- 子图（subgraph）：边集和点集都是原图的子集
- 路径、路径长度都不必多说
    - 简单路径（simple path）：路径上的点不重复
- 回路（cycle）：路径的起点和终点相同
- 对于无向图：
    - 如果存在一条从 $v_i$ 到 $v_j$ 的路径，则称 $v_i$ 和 $v_j$ 连通（connected）
    - 如果任意一对点都连通，则称这个无向图是连通的（connected graph）
    - 无向图的连通组件（connected component of an undirected G）：无向图的极大连通子图
        1. 子图 (Subgraph)：它是原图 $G$ 的一部分，由原图中的部分顶点（Vertices）和部分边（Edges）组成。
        2. 连通 (Connected)：在这个子图内部， **任意两个顶点之间都至少存在一条路径可以互相到达** 。也就是说，这个子图内部是四通八达的，没有任何节点是被完全孤立在这个子图之外的。
        3. 极大 (Maximal)："极大"的意思是，你不能再从原图中拉任何一个其他顶点加入到这个子图中了。如果你强行拉一个外面的顶点进来，这个子图就不再"连通"了（因为那个外面的顶点和现在的子图没有任何边相连）。它已经"尽可能大"地把所有能连通的节点都圈在一起了。
        - 也就是说，==如果图G本身是连通的图的话，那么这幅图的connected component只有一个，就是它自己==
- 树是联通的无环图
- DAG（directed acyclic graph）：有向无环图
- 对于有向图：
    - 如果存在一条从 $v_i$ 到 $v_j$ 的路径，则称 $v_i$ 可达（reachable）$v_j$
    - 如果任意一对点都可达，则称这个有向图是强连通的（strongly connected）
    - 强连通分量：有向图的极大强连通子图
    - 如果有向图不是强连通的，但它的基础图（边去掉方向得到的无向图）是连通的，则称这个有向图是弱连通的（weakly connected）
    - 强连通组件（Strongly connected component）：有向图的极大连通子图
        - 对比一下无向图的极大连通子图，很好理解，HW8中有这个概念，可以辅助理解。
- 度数：
    - 对于无向图，节点 $v$ 的度数 $\mathrm{degree}(v)$ 是与 $v$ 相邻的节点数
    - 对于有向图，节点 $v$ 的入度数 $\mathrm{indegree}(v)$ 是以 $v$ 为终点的边数，出度数 $\mathrm{outdegree}(v)$ 是以 $v$ 为起点的边数
    - 边数等于所有节点的度数之和除以 2

#### 图的表示
- 邻接矩阵：
    - 如果存在一条从 $v_i$ 到 $v_j$ 的边，则 $a_{ij}=1$，否则 $a_{ij}=0$
        - 如果说这个图是一个无向图，设这个邻接矩阵为A，则$A=A^T$，且对角线上的值全为0
    - 对于非稠密图来说非常浪费空间
- 邻接表：
    - 对于每个节点，存储一个链表来记录所有和它相临的节点
    - 对于无向图会将每个边存储两次
    - 对于有向图无法遍历入度，需要额外存储一个逆邻接表
- 带权图：
    - 邻接矩阵：$a_{ij}$ 存储边 $(v_i, v_j)$ 的权重
    - 邻接表：每个链表节点存储边的权重

### 拓扑排序
- AOV 网路：定点表示活动，边表示活动之间的先后关系
    - 可以实现的 AOV 网路一定是 DAG（有向无环图）
- 如果存在一条从 $i$ 到 $j$ 的路径，则称 $i$ 是 $j$ 的前驱（predecessor），$j$ 是 $i$ 的后继（successor）
- 如果存在一条边 $<i, j>$，则称 $i$ 是 $j$ 的直接（immediate）前驱，$j$ 是 $i$ 的直接后继
- 拓扑排序（topological order）是一个图的点集的线性序列，满足：
    - 对于任意一对点 $i, j$，如果 $i$ 是 $j$ 的前驱，则序列中 $i$ 在 $j$ 的前面

```c 
void topsort(Graph G) {
    Queue Q = CreateQueue(); 
    int cnt = 0; Vertex V, W;
    for (each vertex V)
        if (indegree[V] == 0) Enqueue(Q, V);
    while (!IsEmpty(Q)) {
        V = Dequeue(Q); 
        TopNum[V] = ++cnt;
        for (each W adjacent to V)
            if (--indegree[W] == 0) Enqueue(Q, W);
    }
    if (cnt != NumVertex) Error("Graph has a cycle");
    free(Q);
}
```

### 最短路算法

**单源最短路（Single-Source Shortest Path）：** 给定一个带权有向图 $G = (V, E)$ 和源点 $s$，求从 $s$ 到图中每个顶点的最短路径（加权路径长度 $\sum_{e \in P} c(e)$）。

#### 无权图的最短路

无权图（或所有边权重相等）的最短路用 **BFS** 即可（CS61B中有讲为什么BSF在无权途中返回最短的路径，并且这个想想也很简单）：

```c
void Unweighted(Table T) {
    Queue Q = CreateQueue();
    Enqueue(S, Q);
    while (!IsEmpty(Q)) {
        V = Dequeue(Q);
        T[V].Known = true;
        for (each W adjacent to V)
            if (T[W].Dist == Infinity) {
                T[W].Dist = T[V].Dist + 1;
                T[W].Path = V;
                Enqueue(W, Q);
            }
    }
}
```

- `T[i].Dist`：从 $s$ 到 $v_i$ 的距离（初始化为 $\infty$，$s$ 为 0）
- `T[i].Known`：是否已处理
- `T[i].Path`：记录路径前驱
- 时间复杂度：$O(|V| + |E|)$

#### Dijkstra 算法（带权最短路）

设 $S = \{s$ 和已找到最短路径的顶点$\}$。对任意 $u \notin S$，定义 `distance[u]` 为只经过 $S$ 中顶点从 $s$ 到 $u$ 的最短路径长度。

**贪心策略：** 每次选 `distance` 最小的 $u \notin S$ 加入 $S$，然后更新其邻居。

**正确性证明：** 假设最短路径经过某个不在 $S$ 中的顶点 $w$，那么该路径上必然有第一个不在 $S$ 中的顶点，其距离一定比 $u$ 更小——与 $u$ 是最小矛盾。

```c
void Dijkstra(Table T) {
    for (;;) {
        V = smallest unknown distance vertex;
        if (V == NotAVertex) break;
        T[V].Known = true;
        for (each W adjacent to V)
            if (!T[W].Known)
                if (T[V].Dist + Cvw < T[W].Dist) {
                    T[W].Dist = T[V].Dist + Cvw;
                    T[W].Path = V;
                }
    }
}
```

**两种实现方式：**

| 实现 | 找最小顶点 | Decrease | 总复杂度 | 适用场景 |
|------|-----------|----------|---------|---------|
| 1 | 线性扫描 $O(|V|)$ | $O(1)$ | $O(|V|^2 + |E|)$ | 稠密图 |
| 2 | 优先队列 $O(\log|V|)$ | DecreaseKey $O(\log|V|)$ 或直接插入 | $O(|E|\log|V|)$ | 稀疏图 |

- **方法 2 的 DecreaseKey 实现：** 用堆维护未访问顶点，更新时调用 DecreaseKey（需维护顶点在堆中的位置）
- **方法 2 的插入实现：** 不修改堆内元素，直接将更新后的 `{W, newDist}` 插入堆，出队时跳过已访问的顶点。需要 $O(|E|)$ 空间。

**注意：Dijkstra 不能处理负权边！**

#### 带负权边的图

**不能简单给每条边加常数**（因为边数不同的路径受影响不同，最短路径可能改变）。

正确做法：**SPFA 算法**（队列优化的 Bellman-Ford）

```c
void WeightedNegative(Table T) {
    Queue Q = CreateQueue();
    Enqueue(S, Q);
    while (!IsEmpty(Q)) {
        V = Dequeue(Q);
        for (each W adjacent to V)
            if (T[V].Dist + Cvw < T[W].Dist) {
                T[W].Dist = T[V].Dist + Cvw;
                T[W].Path = V;
                if (W is not already in Q)
                    Enqueue(W, Q);
            }
    }
}
```

- 与 BFS 的区别：同一顶点可能**多次入队**（因为距离可能被多次更新）
- 每个顶点最多出队 $|V|$ 次
- 时间复杂度：$O(|V| \times |E|)$
- **负环会导致无限循环**（没有最短路径）

#### 有向无环图（DAG）

如果图是 DAG，可以按**拓扑排序**的顺序处理顶点：

```c
// 拓扑排序后依次松弛
for (each vertex v in topological order) {
    for (each W adjacent to v)
        if (dist[v] + Cvw < dist[W]) {
            dist[W] = dist[v] + Cvw;
            path[W] = v;
        }
}
```

- 时间复杂度：$O(|V| + |E|)$，**不需要优先队列**
- 原理：拓扑序保证处理 $v$ 时，所有可能的入边起点都已经处理完

#### AOE 网络（Activity On Edge）

AOE 网络用于**项目调度**：边表示活动（持续时间 = 边权），顶点表示事件的完成。

- **EC[v]**（最早完成时间）：从起点正向计算，$EC[w] = \max_{\langle v,w \rangle \in E} \{ EC[v] + C_{v,w} \}$
- **LC[v]**（最晚完成时间）：从终点反向计算，$LC[v] = \min_{\langle v,w \rangle \in E} \{ LC[w] - C_{v,w} \}$
- 边 $\langle v,w \rangle$ 的**松弛时间（slack time）**：$LC[w] - EC[v] - C_{v,w}$
- **关键路径（critical path）**：由松弛时间全为 0 的边构成的路径

#### 全源最短路（All-Pairs Shortest Path）

求任意两点之间的最短路径：

| 方法 | 思路 | 复杂度 | 适用 |
|------|------|--------|------|
| 单源重复 $|V|$ 次 | 对每个顶点跑一次单源最短路 | $O(|V|^3)$ | 稀疏图 |
| Floyd-Warshall（Ch.10） | 动态规划 | $O(|V|^3)$ | 稠密图 |

### 网络流
- 最大流：给定一个有向图，求从源点到汇点的最大流量
- 建立残差图（residual graph），设 $f$ 是图 $G=(V, E)$ 的一个流量，则残差图的边权为：

$$
c_f(u, v) = \begin{cases}
    c(u, v) - f(u, v) & \text{if } (u, v) \in E\\
    f(v, u) & \text{if } (v, u) \in E\\
    0 & \text{otherwise}
\end{cases}
$$

- 找残差图中从源点到汇点的一条简单路径，称为增广路（augmenting path）
- 增广路的流量为增广路上的最小边权，创建这样一条流量，更新残差图，直到找不到增广路为止

### 最小生成树
- 最小生成树（minimum spanning tree）是给定一个连通图，找一个总边权最小的生成树
    - 生成树为图的一个子图，包含所有点，且任意两点之间有且仅有一条边
- 连通图一定存在一个最小生成树
- 向生成树中添加一个额外的边，则一定会存在一个环
- Prim 算法
    ```text
    T = 最小权边
    for i = 1..n-2
        e = 与 T 中的点相连且加入 T 后不会形成环的最小权边
        将 e 加入 T
    T 即为最小生成树
    ```
- Kruskal 算法
    ```text
    T = 空图
    for i = 1..n-1
        e = 加入 T 中不会形成环的最小权边
        将 e 加入 T
    T 即为最小生成树
    ```
    - 可以通过并查集来维护
        1. 建立并查集
        2. 将边按照权值从小到大排序
        3. 从小到大遍历所有边，若两个端点不在同一集合，则将两个端点合并，将边加入 T；在同一集合中则忽略

### 深度优先搜索
- 深度优先搜索（depth-first search，DFS）是一种用于遍历或搜索树或图的算法
- 是先序遍历的一种推广
    ```c 
    void dfs(Vertex V) {
        Visited[V] = true;
        for (each W adjacent to V)
            if (!Visited[W]) dfs(W);
    }
    ```
- 双连通性
    - 如果去掉一个点，剩下的图不再是连通的，则称这个点为割点（cut vertex），或 articulation point
    - 如果图 $G$ 是连通的，并且没有 articulation point，则称 $G$ 是双连通的
    - 双连通分量（biconnected component）是极大的双连通子图
    - Tarjan 算法
        - dfn[x] 表示 x 第一次被访问的时间戳（dfs 时第几个被访问，从 0 开始）
        - 追溯值 low[x]
            - 初始 low[x] = dfn[x]
            - 对于从 x 到 y 的边，如果 <x, y\> 在搜索树上，则 low[x] = min(low[x], low[y])
            - 如果 <x, y\> 不在搜索树上，则 low[x] = min(low[x], dfn[y])
        - 找割点
            - 如果 u 是 root，则 u 是割点当且仅当 u 有至少两个儿子
            - 如果 u 不是 root 则 u 是割点当且仅当存在一个儿子 v，满足 dfn[u] <= low[v]
- 欧拉回路与欧拉路径
    - 欧拉回路（Euler circuit）为包含所有边的简单环，欧拉路径（Euler path）为包含所有边的简单路径
    - 无向图
        - 无向图 G 有欧拉回路当且仅当 G 是连通的且每个顶点的度数都是偶数
        - 无向图 G 有欧拉路径当且仅当 G 是连通的且有且仅有两个顶点的度数是奇数
    - 有向图
        - 有向图 G 有欧拉回路当且仅当 G 是弱连通的且每个顶点的出度等于入度
        - 有向图 G 有欧拉路径当且仅当 G 是弱连通的且有且仅有一个顶点的出度比入度大 1，有且仅有一个顶点的入度比出度大 1，其余顶点的出度等于入度
    - dfs 即可

    ??? success "代码"
        ```c
        void dfs(int x) {
            for (int y = 1; y <= maxn; ++y) {
                if (G[x][y]) {
                    G[x][y] = 0;
                    G[y][x] = 0;
                    dfs(y);
                }
            }
            ans[++ansi] = x;
            return;
        }
        for (int i = 1; i <= maxn; ++i) {
            if (deg[i] % 2) {
                cnt++;
                if (!root) root = i;
            }
        }
        if (!root) {
            for (int i = 1; i <= maxn; ++i) {
                if (deg[i]) {
                    root = i; break;
                }
            }
        }
        if (cnt && cnt != 2) {
            printf("No Solution\n");
            return 0;
        }
        dfs(root);
        ```
