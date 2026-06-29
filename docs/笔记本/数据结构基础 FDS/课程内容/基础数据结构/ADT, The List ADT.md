
## 1. 抽象数据类型（ADT）

### 1.1 Data Type

数据类型（Data Type）由两部分组成：

$$
\text{Data Type} = \{ \text{Objects} \} \cup \{ \text{Operations} \}
$$

也就是说，一个数据类型不仅要说明它能表示哪些对象，还要说明“这些对象上可以做哪些操作”。

!!! example "int 类型"
    `int` 可以理解为：

    $$
    \text{int} = \{0, \pm 1, \pm 2, \cdots, \text{INT\_MAX}, \text{INT\_MIN}\}
    \cup
    \{+, -, \times, \div, \%, \cdots\}
    $$

    前半部分是 `int` 类型可以表示的对象，后半部分是 `int` 类型支持的操作。

### 1.2 ADT 的核心思想

ADT是一种数据类型，它把下面两件事分开：

1. **Specification（规格说明）**

    说明对象是什么、对象上可以做什么操作。

2. **Representation / Implementation（表示与实现）**


ADT 关心的是==这个东西应该表现得像什么==，而不是一开始就规定==它必须怎么实现。

!!! note "ADT 的意义"
    ADT 的好处是可以把接口和实现分离。只要外部使用者遵守同一套操作接口，底层实现可以从数组换成链表，也可以从指针链表换成 cursor 实现，而使用者不需要关心这些内部细节。

---

## 2. The List ADT

### 2.1 List 的对象

线性表（List）的对象是一串有序元素：

$$
(item_0, item_1, \cdots, item_{N-1})
$$

其中 $N$ 是 list 的长度。

### 2.2 List 支持的操作

List ADT 常见操作包括：

1. 求 list 的长度 $N$。
2. 打印 list 中的所有元素。
3. 创建一个空 list。
4. 找到第 $k$ 个元素，其中 $0 \le k < N$。
5. 在第 $k$ 个元素后插入一个新元素，其中 $0 \le k < N$。
6. 删除某个元素。
7. 找到当前元素的下一个元素。
8. 找到当前元素的前一个元素。

!!! question "为什么是在第 $k$ 个元素后出入一个新元素，而不是在它之前?"
    # List ADT 中为什么定义为 Insert After？
    在 List ADT 的操作定义中，经常会看到：Insert a new item after the $k_{th}$ item. 即在第 k 个元素之后插入一个新元素。
    
    那么为什么是insert after而不是insert before呢？
    
    如果我们遇到的是一个单向链表。那么在某个节点前面插入会非常难实现。因为单向链表只知道当前节点本身，探测它的后驱节点很容易，但是探测它的前驱节点很难。

---

## 3. List 的数组实现

数组实现的思想很直接：

$$
array[i] = item_i
$$

这是一种顺序映射（sequential mapping），元素在逻辑上相邻，在物理内存中也相邻。

### 3.1 优点

查找第 $k$ 个元素很快：

$$
\text{Find\_Kth} = O(1)
$$

因为数组可以通过下标直接访问。

### 3.2 缺点

数组实现有两个主要问题：

1. `MaxSize` 需要提前估计。

    如果估计太小，后面可能放不下；如果估计太大，又会浪费空间。

2. 插入和删除不仅是 $O(N)$，还会产生大量元素移动。

    例如在中间插入一个元素，后面的元素都要整体后移；删除中间元素时，后面的元素也要整体前移。

因此，数组适合频繁按下标访问的场景，但不适合频繁在中间插入和删除。

---

## 4. List 的链表实现

### 4.1 单链表节点

linked list：每个节点自己保存数据，并通过指针指向下一个节点。

```c
typedef struct list_node {
    char data[5];
    struct list_node *next;
} *list_ptr;

list_ptr ptr;
```

其中：

- `data` 保存节点中的元素。
- `next` 指向下一个节点。
- `ptr` 是头指针，指向整个链表的第一个节点。

!!! note "链表节点的位置"
    链表节点通常由 `malloc` 动态申请，因此每次运行时，节点在内存中的地址可能不同。链表关心的是指针连接关系，而不是节点地址必须连续。

### 4.2 创建两个节点并连接
![](附件/Pasted%20image%2020260623122303.png)

```c
list_ptr N1, N2;

N1 = (list_ptr)malloc(sizeof(struct list_node));
N2 = (list_ptr)malloc(sizeof(struct list_node));

strcpy(N1->data, "ZHAO");
strcpy(N2->data, "QIAN");

N1->next = N2;
N2->next = NULL;

ptr = N1;
```

逻辑结构是：

```text
ptr -> ZHAO -> QIAN -> NULL
```

---

## 5. 单链表的插入

假设现在有链表：
![](附件/Pasted%20image%2020260623122901.png)
如果要在节点 $a_i$ 后面插入新节点 `temp`（存放元素 $b$），正确顺序是：

```c
temp->next = node->next;
node->next = temp;
```

插入后：
![](附件/Pasted%20image%2020260623122951.png)
### 5.1 为什么顺序不能反过来？

如果先执行：

```c
node->next = temp;
```

那么 `node` 原本指向的 `ai+1` 就丢失了。之后再写：

```c
temp->next = node->next;
```

此时 `node->next` 已经是 `temp`，于是会得到：

```text
temp->next = temp
```

也就是 `temp` 指向自己，原来的后半段链表被断开。

!!! warning "单链表插入的指针顺序"
    插入节点时，必须先让新节点接住原来的后继节点，再让前一个节点指向新节点。

    口诀是：**先接后面，再接前面**。

### 5.2 插入第一个节点怎么办？

如果不用特殊技巧，插入第一个节点需要单独处理头指针：

```c
temp->next = ptr;
ptr = temp;
```

但这样会让插入首节点和插入普通节点变成两套逻辑。因此实际实现中经常加入 dummy head node。

---

## 6. 单链表的删除

假设要删除节点 `node`，并且已知它的前驱节点是 `pre`：
![](附件/Pasted%20image%2020260623123403.png)

正确步骤是：

```c
pre->next = node->next;
free(node);
```

删除后：
![](附件/Pasted%20image%2020260623123453.png)

### Question 删除第一个节点怎么办？

如果不用 dummy head node，删除第一个节点也要单独修改头指针：

```c
list_ptr old = ptr;
ptr = ptr->next;
free(old);
```

### 给链表加一个 dummy head node。

dummy head node 不存真正的数据，只是放在第一个真实节点之前。

有了 dummy head node 之后，真正的第一个节点也有了前驱节点，于是插入、删除首节点就可以和普通节点一样处理。

---

## 7. 双向循环链表(Double Linked Circular Lists)

### 7.1 为什么需要双向链表？

单链表从某个节点找到下一个节点很容易，但找前一个节点很麻烦。

例如链表：

```text
1 -> 2 -> 3 -> ... -> m
```

如果你已经走到了第 $m$ 个节点，想找它的前一个节点 $m-1$，单链表没有反向指针，只能从第一个节点重新走一遍。

这在删除当前节点、向前遍历等操作中都很不方便。

### 7.2 双向循环链表的节点

双向链表节点包含三个部分：

```c
typedef struct node {
    struct node *llink;
    element item;
    struct node *rlink;
} *node_ptr;
```
![](附件/Pasted%20image%2020260623123702.png)

其中：

- `llink` 指向左边，也就是前驱节点。
- `item` 保存数据。
- `rlink` 指向右边，也就是后继节点。

对于任意节点 `ptr`，理想情况下有：

```c
ptr == ptr->llink->rlink;
ptr == ptr->rlink->llink;
```

也就是说，从一个节点先向左再向右，或者先向右再向左，应该回到自己。

### 7.3 双向循环链表的头结点

使用的是带头结点的双向循环链表。

如果有三个元素：

![](附件/Pasted%20image%2020260623124253.png)

这里的 `H` 是 head node，不存实际数据。最后一个节点的 `rlink` 指回 `H`，`H` 的 `llink` 指向最后一个节点。

空表时也不是 `NULL`，而是：

```text
H <-> H
```

也就是：

```c
H->llink = H;
H->rlink = H;
```

### 7.4 双向循环链表的好处

- 双向链表解决了找前驱困难的问题；
- 循环链表和头结点则减少了边界情况，让空表、首节点、尾节点的处理更统一。

---

## 8. List ADT 的两个应用

### 8.1 Polynomial ADT

多项式 ADT 的对象可以写成：

$$
P(x) = a_1x^{e_1} + a_2x^{e_2} + \cdots + a_nx^{e_n}
$$

也可以看成一组有序对：

$$
\langle e_i, a_i \rangle
$$

其中：

- $a_i$ 是系数（coefficient）。
- $e_i$ 是指数（exponent）。
- $e_i$ 是非负整数。

多项式 ADT 常见操作包括：

1. 求多项式的 degree，也就是 $\max\{e_i\}$。
2. 多项式加法。
3. 多项式减法。
4. 多项式乘法。
5. 多项式求导。

#### 8.1.1 表示方法 1：数组表示

一种表示方式是用数组下标表示指数：

```c
typedef struct {
    int CoeffArray[MaxDegree + 1];
    int HighPower;
} *Polynomial;
```


例如 `CoeffArray[i]` 表示 $x^i$ 的系数。

!!! note "HighPower的作用"
    HighPower表示的是最高的指数。对于 $P(x)=3x^5+2x^2−7$，HighPower就是5。
    
    ### Question：
    我们都已经有数组的总长度了，为什么还要HighPower？
    ### Answer：
    对于一个`CoeffArray[1000]`来说，如果没有HighPower的话，我们找最高次项，必须从下标为999开始一直往前找，直到找到第一个非零系数，才能够得知最高次项是多少。如果有了HighPower，那么求degree就可以直接返回HighPower。
    

这种表示对加法、乘法等操作比较容易写，但在稀疏多项式上会浪费空间和时间。

!!! example "数组表示的低效情况"

    $$
    P_1(x) = 10x^{1000} + 5x^{14} + 1
    $$

    $$
    P_2(x) = 3x^{1990} - 2x^{1492} + 11x + 5
    $$

    它们的最高次数很高，但实际非零项很少。如果用数组表示，就要为大量系数为 0 的项保留空间。

    对两个 degree 分别为 $N_1$ 和 $N_2$ 的多项式做乘法，朴素复杂度是：

    $$
    O(N_1N_2)
    $$

    但在这种稀疏多项式里，很多计算都浪费在 0 系数上。

#### 8.1.2 表示方法 2：链表表示

对于稀疏多项式，可以把每一项表示成一个节点：

```text
Coefficient | Exponent | Next
```

对应的声明是：

```c
typedef struct poly_node *poly_ptr;

struct poly_node {
    int Coefficient;
    int Exponent;
    poly_ptr Next;
};

typedef poly_ptr a;    /* nodes sorted by exponent */
```

链表中的节点按指数排序，只保存非零项。

例如：

```text
a -> [am-1, em-1] -> ... -> [a0, e0] -> NULL
```

### 总结

如果多项式很稠密，数组表示简单直接；如果多项式很稀疏，链表表示通常更合适，因为它只存真正存在的项。

### 8.2 Multilists

假设校内有 40000 个学生，2500 门课程，需要打印每门课的学生名单，也需要打印每个学生注册的课程列表。

#### 8.2.1 表示方法 1：二维数组

可以用一个二维数组：

```c
int Array[40000][2500];
```

其中：

$$
Array[i][j] =
\begin{cases}
1, & \text{student } i \text{ registered for course } j \\
0, & \text{otherwise}
\end{cases}
$$

问题是这个矩阵通常非常稀疏，大量位置都是 0。

#### 8.2.2 表示方法 2：Multilist

Multilist 的思想是：一个注册关系同时属于两条链。

例如学生 $S_i$ 注册了课程 $C_j$，那么这个节点既可以挂在学生 $S_i$ 的课程链表上，也可以挂在课程 $C_j$ 的学生链表上。

这样：

1. 想打印某门课的学生名单，就沿着这门课对应的链走。
2. 想打印某个学生的选课列表，就沿着这个学生对应的链走。

![](附件/Pasted%20image%2020260623131908.png)


---

## 9. Cursor Implementation of Linked Lists

### 9.1 为什么需要 cursor 实现？


普通链表需要两个能力：

1. 数据存放在一组结构体中，每个结构体包含数据和指向下一个结构体的指针。
2. 可以通过 `malloc` 从系统全局内存申请新结构体，也可以通过 `free` 释放结构体。

Cursor 实现用数组下标模拟指针，用一个数组 `CursorSpace` 模拟内存池。

### 9.2 CursorSpace 的结构

每个数组元素包含：

```text
Element | Next
```

其中：

- `Element` 保存数据。
- `Next` 不是真正的指针，而是下一个节点在数组中的下标。

![](附件/Pasted%20image%2020260623132740.png)

可以把 `CursorSpace[i].Next` 理解为“下一个节点的 cursor”。

!!! note "接口保持一致"
    课件强调：cursor implementation 的接口和 pointer implementation 是一样的。也就是说，外部看起来仍然是在使用链表，只是内部不用真正的指针。

### 9.3 空闲链表

Cursor 实现通常把 `CursorSpace[0]` 当成空闲链表的头结点。

初始化时，可以让所有空位置串成一条 free list：

```text
0 -> 1 -> 2 -> 3 -> ... -> S-1 -> 0
```

这里的 `0` 不是普通数据节点，而是管理空闲空间的头结点。

### 9.4 Cursor 版 malloc

初始状态：
![](附件/Pasted%20image%2020260623133330.png)
从空闲链表中取出一个节点，等价于“申请内存”：

```c
p = CursorSpace[0].Next;
CursorSpace[0].Next = CursorSpace[p].Next;
```

最终得到：![](附件/Pasted%20image%2020260623133603.png)

含义是：

1. `p` 取得当前空闲链表的第一个可用节点。
2. 空闲链表头结点跳过 `p`，指向下一个空闲节点。
3. `p` 就可以拿去作为新链表节点使用。



!!! warning "Cursor malloc 的边界情况"
    如果 `CursorSpace[0].Next == 0`，说明 free list 已经空了，此时没有可分配的节点。

### 9.5 Cursor 版 free

初始状态（`p`是即将被释放的节点）：
![](附件/Pasted%20image%2020260623134120.png)

释放节点 `p`，等价于把它插回空闲链表的开头：

```c
CursorSpace[p].Next = CursorSpace[0].Next;
CursorSpace[0].Next = p;
```

最终得到：![](附件/Pasted%20image%2020260623134230.png)
含义是：

1. 先让 `p` 指向原来的第一个空闲节点。
2. 再让空闲链表头结点指向 `p`。
3. `p` 就重新变成可分配空间。

### 9.6 Cursor 实现的特点

Cursor 实现通常更快一些，因为它不需要频繁调用系统级的内存管理例程 `malloc` 和 `free`。

但是它也有明显限制：

1. `CursorSpace` 大小通常需要预先确定。
2. 数组空间用完后无法像系统堆内存那样自然扩展。
3. 实现上需要自己维护 free list，写错指针下标时也可能破坏结构。

### 总结
Cursor implementation 的本质是用数组下标模拟指针，用 `CursorSpace[0]` 管理空闲节点。它保留了链表接口，但把动态内存管理变成了手写的数组空间管理。

---

## 10. 本节总结

1. ADT 的核心是把“对象与操作的规格说明”和“对象表示与操作实现”分开。
2. List ADT 表示一组有序元素，支持查找、插入、删除、前驱后继等操作。
3. 数组实现支持 $O(1)$ 的按下标访问，但插入删除需要移动大量元素。
4. 单链表插入删除节点本身是 $O(1)$，但通常需要已知前驱节点。
5. dummy head node 可以统一首节点和普通节点的插入删除逻辑。
6. 双向循环链表方便找前驱，也减少了头尾边界情况。
7. 多项式和 multilist 都体现了 List ADT 在稀疏数据表示中的作用。
8. Cursor 实现用数组下标模拟指针，可以避免频繁系统内存管理，但需要提前管理固定大小的空间。
