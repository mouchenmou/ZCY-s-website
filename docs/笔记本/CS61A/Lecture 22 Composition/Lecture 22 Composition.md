## 1. Composition

Composition的含义：**一个对象可以把另一个对象作为自己的组成部分。**

![](附件/Pasted%20image%2020260716122035.png)

---

## 2. Linked List

Linked list is a recursive data structure。

定义：

> A linked list is either empty or a first value and the rest of the linked list.

也就是说，一个链表要么是空的，要么由两部分组成：

1. `first`：当前这个位置的值。
2. `rest`：剩下的链表。

例如：

```python
Link(3, Link(4, Link(5, Link.empty)))
```

可以理解成：

```text
3 -> 4 -> 5 -> empty
```

`rest` 不是一个普通的 Python list，而是另一个 `Link` 实例。

---

## 3. `Link` Class

```python
class Link:
    """A linked list."""

    empty = ()

    def __init__(self, first, rest=empty):
        assert rest is Link.empty or isinstance(rest, Link)
        self.first = first
        self.rest = rest
```

`isinstance(rest, Link)` 会查看 `rest` 是否是 `Link` 这个类别的

`Link.empty` 是一个 class attribute，用来表示空链表。

它不是真的 `Link` 实例，而是一个特殊的标记。这样我们可以写：

```python
Link(5, Link.empty)
```

表示一个只含有 `5` 的链表。


```python
def __init__(self, first, rest=empty):
```

如果没有传入 `rest`，默认就是空链表。

所以：

```python
Link(5)
```

等价于：

```python
Link(5, Link.empty)
```

```python
assert rest is Link.empty or isinstance(rest, Link)
```

这句保证 `rest` 只能是：

1. 空链表 `Link.empty`
2. 另一个 `Link` 实例

!!! explanation "`isinstance(rest, Link)`"
    `isinstance(x, Class)` 用来判断 `x` 是否是 `Class` 的实例。

    ```python
    isinstance(Link(3), Link)
    # True

    isinstance(3, Link)
    # False
    ```

---

## 4. 链表的结构

若创建以下链表：

```python
s = Link(3, Link(4, Link(5)))
```

则：

```python
s.first
# 3

s.rest.first
# 4

s.rest.rest.first
# 5

s.rest.rest.rest is Link.empty
# True
```


---

## 5. Link List Processing
递归在链表处理中非常常见。

Python 内置了 Range，Map，Filter等功能，但是这些功能没法用到我们创建的Linked Lists上，毕竟Python也不知道要通过 `s.rest` 去查找下一个元素，不过我们可以手动实现这些功能。

![](附件/Pasted%20image%2020260716203242.png)

```Python
def square(x):
    return x*x

def odd(x):
    return x%2==1

def range_link(start, end):
    """Return a Link containing consecutive integers from start to end
    
    >>> range_link(3, 6)
    Link(3, Link(4, Link(5)))
    """
    
    if start >= end:
        return Link.empty
    else:
        return Link(start, range_Link(start+1, end))
    
    
def map_link(f, s):
"""Return a Link that contains f(x) for each x in Link s.

>>> map_link(square, range_link(3, 6))
Link(9, Link(16, Link(25)))
"""

    if s is Link.empty :
        return s
    else :
        return Link(f(s.first), map_link(f, s.rest))

def filter_link(f, s):
"""Return a Link that contains only the elements x of s for which f(x) is a true value.

>>> filter_link(odd, range_link(3, 6))
Link(3, Link(5))
"""
    if s is Link.empty:
        return s
    filter_rest = filter_link(f, s.rest)
    if f(s.first):
        return Link(s.first, filter_rest)
    else:
        return filter_rest
```


!!! explanation "Another form of`filter_link(f, s)`"
    Apart from what teacher said in the video, I came up another form of the implementation of the filter_link function, and it is also avaliable.
    ```
    def filter_link(f, s):
        if s is Link.empty:
            return s
        if f(s.first):
            return Link(s.first, filter_link(f, s.rest))
        else:
            return filter_link(f, s.rest)
    ```
---
## 6. Linked Lists Mutation

A linked list instance is an object, and all instances of a user-defined class can be changed or mutated.

![](Pasted%20image%2020260716211805.png)

Initially, `t.rest` points to the node containing `3`. After `t.rest = s`, it points back to the node containing `5`, so the linked list forms a cycle.

---

## 7. Linked List Mutation Example

### 7.1 Adding to an Ordered List

Assuming that there exists an ordered Linked List. The add function is designed to add a new item into the list in the appropriate position so that everything stays ordered from least to greatest. However, if the item is alread in the Link List, then we don't modify s, but still return it.

```Python
def add(s, v):
   """Add v to an ordered list s with no repeats, return modified s.
   (Note: If v is alread in s, then don't modify s, but still return it.)
   """
    # 情况 0：s 不能为空链表
    assert s is not Link.empty
    # 情况 1：v 比当前节点小
    if s.first > v:
        s.first, s.rest = v, Link(s.first, s.rest)
    # 情况 2：v 比当前节点大，并且当前节点已经是最后一个节点
    elif s.first < v and empty(s.rest):
        s.rest = Link(v)
    # 情况 3：v 比当前节点大，并且后面还有节点
    elif s.first < v:
        add(s.rest, v)
    return s
```

!!! explanation "我的疑问"
   ```
    s.first, s.rest = v, Link(s.first, s.rest)
   ```
       疑问一：这段话中我感觉是先改动了 `s.first` 再改动 `s.rest`，明明已经将`s.first` 改成了 `v` 为什么后面那个 `s.rest=Link(s.first, s.rest)` 使用的还是原先的 's.first'。
       
    解答：Python 的这种多重赋值会先把右边全部算完，再统一赋值给左边。
       
    疑问二：为什么这里要写 `Link(s.first, s.rest)` 而不是直接写 `s.rest=s`
       
    解答：
    
    假设原来：

	```
	s = Link(3, Link(5))
	v = 2
	```
	
	目标是得到：
	
	```
	2 -> 3 -> 5
	```
	
	如果写：
	
	```
	s.rest = s
	```
	
	那就会让 `s` 的 `rest` 指向自己：
	
	```
	┌───────┐
	↓       │
	3 ──────┘
	```


!!! warning "我的错误写法"
    我一开始是这么写的，但是这个写法有问题
	```Python
	def add(s, v):
	if(s.first==v):
	    return s
	elif(s.first<v):
	    return Link(s.first, add(s.rest, v))
	else:
	    return Link(v, s)
	```
	问题是我这个做法没有处理空的链表
	比如：

	```
	s = Link(1, Link(3, Link(5)))
	add(s, 6)
	```
	
	执行过程：
	
	```
	1 < 6，递归处理 Link(3, Link(5))
	3 < 6，递归处理 Link(5)
	5 < 6，递归处理 Link.empty
	```
	
	最后会调用：
	
	```
	add(Link.empty, 6)
	```
	
	然后第一句就访问：
	
	```
	s.first
	```
	
	但 `Link.empty` 没有 `first`，所以会报错。
	
	要修复，至少需要加：
	
	```
	if s is Link.empty:
	    return Link(v)
	```
	
	于是可以写成：
	
	```
	def add(s, v):
	    if s is Link.empty:
	        return Link(v)
	    if s.first == v:
	        return s
	    elif s.first < v:
	        return Link(s.first, add(s.rest, v))
	    else:
	        return Link(v, s)
	```
	
	这个版本在结果上是正确的。

---

## 8. Tree
The main difference between tree and list is that a tree has multiple trees as branches, whereas a linked list has only one linked list as the rest of the list.

### 8.1 Tree Class

A Tree has a label and a list of branches; each branch is a Tree

```Python
class Tree:
    def __init__(self, label, branches=[]):
        self.label=label
        for branch in branches:
            assert isinstance(branch, Tree)
        self.branches=list(branches)
```

我们使用Class来表示Tree跟用Data Abstraction来表示Tree有点区别
![](附件/Pasted%20image%2020260716232917.png)

当我们使用Class Tree来表示树时，树的各个部分已经变成了对象的属性，所以不用专门写 `label(Tree)`、`branches(tree)` 这样的选择函数。

也就是说，创建了一棵树之后，可以直接写下面的两句话来取得树根的标签和所有分支：

```Python
t.label
t.branches
```

但是Data Abstraction不一样，Python本身不知道 `tree[0]` 表示label，`tree(1:)` 表示分支，所以还要定义

```Python
def label(tree):
    return tree[0]

def branches(tree):
    return tree[1:]
```

!!! warning "我的误解"
    我之前以为Data Abstraction就已经是一个类了，但其实**`tree(label, branches=[])` 就只是定义了一个普通函数**。


### 8.2 Tree Mutation

### Example：Pruning trees

Pruning trees就是把一棵树中的某些子树删掉，又叫剪枝

方法是：先把当前这个树节点的分支都给修剪了，然后再递归地修剪剩下的分支。

这样子说过来实在是太抽象了，举个例子就好理解了：

假设我们要把所有root label为 n 的子树(下图表示要删除root label为1的子树)：
![](附件/Pasted%20image%2020260716235644.png)

方法如下：

```Python
def prune(t, n):
    t.branches = [b, for b in t.branches if b.label !=n]
    for b in t.branches:
        prune(b, n)
```