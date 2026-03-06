**Definition:** A _nested list of numbers_ is a list that contains numbers and lists. It may contain only numbers, only lists, or a mixture of both. The lists must also be _nested lists of numbers_. For example: `[1, [2, [3]], 4]`, `[1, 2, 3]`, and `[[1, 2], [3, 4]]` are all _nested lists of numbers_.

Write a function `deep_map` that takes two arguments: a nested list of numbers `s` and a one-argument function `f`. It modifies `s` **in place** by applying `f` to each number within `s` and replacing the number with the result of calling `f` on that number.

`deep_map` returns `None` and should not create any new lists.

> **Hint:** `type(a) == list` will evaluate to `True` if `a` is a list.

```Python
    def deep_map(f, s): 
        """Replace all non-list elements x with f(x) in the nested list s.
        
        >>> six = [1, 2, [3, [4], 5], 6] 
        >>> deep_map(lambda x: x * x, six) 
        >>> six
        [1, 4, [9, [16], 25], 36]
        >>> # Check that you're not making new lists 
        >>> s = [3, [1, [4, [1]]]] 
        >>> s1 = s[1] 
        >>> s2 = s1[1]
        >>> s3 = s2[1]
        >>> deep_map(lambda x: x + 1, s) 
        >>> s
        4, [2, [5, [2]]]]
        
```
需要注意的是，这个函数返回的是None，而不是s，我在做的过程中犯了几次错误，并且这几个错误比的逻辑比较难理解，下面挨个展开。

### 错误一：
```python
    for i in range(len(s)):
        if type(s[i])==list :
            s[i] = deep_map(f, s[i])
        else:
            s[i] = f(s[i])
    return None
```
### 终端的错误结果：
![](附件/截屏2026-02-24%2017.18.59.png)

---

### 错误二：
```python
    for i in range(len(s)):
        if type(s[i])==list :
            s[i] = deep_map(f, s[i])
        else:
            s[i] = f(s[i])
    return s
```
### 终端的错误结果：
![](附件/截屏2026-02-24%2017.20.32.png)
### 出错原因：
![](附件/截屏2026-02-24%2017.27.36%201.png)


---
### 正确答案：
```Python
    for i in range(len(s)):
        if type(s[i])==list :
            deep_map(f, s[i])
        else:
            s[i] = f(s[i])

return None
```