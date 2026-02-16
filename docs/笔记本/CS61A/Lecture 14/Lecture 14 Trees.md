## 1. Implementing the Tree Abstraction
![](附件/截屏2026-02-10%2015.28.25.png)

!!! example
    ![](附件/Pasted%20image%2020260210175824.png)
    
    ## 报错原因：
    ### 1. 发生了什么？（代码拆解）
    这里的参数分别对应：
    
    - `label` = `1`
    - `branches` = `[5]` （这是一个包含整数 `5` 的列表）
    
    接着，`tree` 函数内部开始运行（看右边代码）：
    
    - 你的 `branches` 是 `[5]`。
    - 循环第一次取出的 `branch` 是 **`5`**（整数）。
    - 检查 `is_tree(5)`。
    - 看 `is_tree` 的定义：`if type(tree) != list... return False`。
    - **整数 5 不是列表，所以它不是树。**
    - **报错：AssertionError: branches must be trees**。
    
    ---
    
    ### 2. 为什么我一开始感觉没错？
    我最初认为只要传进去一个数组就是正确的，没看清楚`is_tree`函数的定义。
    - **5** 只是一个数字（Number）。
    - **[5]** 才是代表“节点值为 5 的叶子节点树”（Tree）。
    - 
    `tree` 函数的第二个参数必须是 **“由树组成的列表” (List of Trees)**，而不是“由数字组成的列表”。
    
    ---
    ### 3. 正确写法
    要把那个 `5` 包装成一棵树。
    
    **方法一：使用 `tree` 构造函数（推荐）**
    
    - 解析：`tree(5)` 会返回 `[5]`，这才是树。`is_tree([5])` 是 True。
    
    - **方法二：直接手写列表结构（不推荐，但这解释了原理）**
    
    - 解析：这里 `branches` 是 `[[5]]`。循环取出的 `branch` 是 `[5]`。`[5]` 是一个合法的树结构。

---

## 2. Tree Processing Uses Recursion

Processing a leaf is often the base case of a tree processing function. The tree recursion case typically makes a recursive call on each branch, then aggregates.

```Python
    def count_leaves(t):
        if is_leaf(t):
            return 1
        else:
            return sum[count_leaves(b) for b in branches(t)]
```

---

## 3. Creating Trees

A function that creates a tree from another tree is typically also recursive.

```python
    def increment_leaves(t):
        """ Return a tree like t but within leaf labels incremented. """
        if is_leaf(t):
            return tree(label(t)+1)
        else:
            bs = [increment_leaves(b) for b in branches(t)]
            return tree(lable(t), bs)
            
            
    def increment(t):
        """ Return a tree like t but with all labels incremented. """
        return tree(label(t)+1, [increment(b) for b in branches(t)])
```

---

## 4. Count Paths that have a Total Label sum
```Python
    def count_paths(t, total):
        """ Return the number of paths from the root to any node in tree t for which the labels along the path sum to total.
        >>> count_paths(t, 3)
        2
        >>> count_paths(t, 4)
        2
        >>> count_paths(t, 5)
        0
        >>> count_paths(t, 6)
        1
        >>> count_paths(t, 7)
        2
        """
        if _________:
            found = _________
        else:
            _________
        return found + _________([__________ for b in branches(t)])
```

### Answer:
```Python
    def count_paths(t, total):
        if label(t) == total:
            found = 1
        else:
            found = 0
        return found + sum([count_paths(b, total-label(t)) for b in branches(t)])
```

!!! warning "关于sum函数的一些内容"
    之前对sum函数的理解很不到位，一旦把sum和list放在一起就晕头转向，所以跟Gemini交流了一下内容：
     1. `sum([1, 2, 3])`（最简单的开胃菜）：
        - Python 在后台其实只做了一件事：
            - **它把列表的方括号 `[]` 拆掉，然后在每个数字中间填上 `+` 号。**
            - 输入：`[1, 2, 3]`
            - 计算：`1 + 2 + 3`
            - 结果：`6`
    2. `sum([[1], [2, 3], [4]])`  ==会报错!!!==
        - 当你写 `sum([A, B, C])` 时，Python 实际执行的是： `0 + A + B + C`
        - 注意那个开头的 **`0`**。`sum` 函数默认总是从整数 `0` 开始累加的。
        - 所以，当你运行 `sum([[1], [2, 3], [4]])` 时，Python 实际上在做： **`0 + [1] + [2, 3] + [4]`**
        - 应该修正为 `sum([[1], [2, 3], [4]], [])`
            - 记住，第二个参数是起始值，如果我们不设置的话就默认为0