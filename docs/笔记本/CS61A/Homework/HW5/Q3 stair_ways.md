Imagine that you want to go up a staircase that has `n` steps, where `n` is a positive integer. You can take either **one** or **two steps** each time you move.

Write a generator function `stair_ways` that yields all the different ways you can climb the staircase.

Each "way" of climbing a staircase can be represented by a list of 1s and 2s, where each number indicates whether you take one step or two steps at a time.

For example, for a staircase with 3 steps, there are three ways to climb it:

- You can take one step each time: `[1, 1, 1]`.
- You can take two steps then one step: `[2, 1]`.
- You can take one step then two steps: `[1, 2].`.

Therefore, `stair_ways(3)` should yield `[1, 1, 1]`, `[2, 1]`, and `[1, 2]`. These can be yielded in any order.

```
def stair_ways(n):
    """
    Yield all the ways to climb a set of n stairs taking
    1 or 2 steps at a time.

    >>> list(stair_ways(0))
    [[]]
    >>> s_w = stair_ways(4)
    >>> sorted([next(s_w) for _ in range(5)])
    [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2]]
    >>> list(s_w) # Ensure you're not yielding extra
    []
    """
    "*** YOUR CODE HERE ***"
```

### 第一版错误代码：
```Python 
    if (n==0):
        yield []
    if(n==1):
        yield [1]
    else:
        yield [1]+stair_ways(n-1)
        yield [2]+stair_ways(n-1)
```
`TypeError: can only concatenate list (not "generator") to list` （只能把列表和列表拼接，不能把列表和“生成器”拼接）。
### 1. 为什么报错？

当你写下 `stair_ways(n-1)` 的时候，你以为它会返回一个列表（比如 `[1, 1, 1]`），但因为它是一个包含 `yield` 的生成器函数，所以它返回的是一台“生成器机器”**。

你试图用 `[1] + 机器`，Python 当然会崩溃。

**解决办法：** 我们需要用 `for` 循环启动机器，把机器里面吐出来的每一条“路径（列表）”拿出来，然后再和 `[1]` 拼接。

---

### 第二版错误代码：
```Python
    if(n==0):
        yield []
    if(n==1):
        yield [1]
    else:
        for element in stair_ways(n-1):
            yield element + [1] 
        for element in stair_ways(n-2):
            yield element + [2]
```

1. **进入函数：** `n = 0`。
2. **第 1 行：** 遇到 `if (n == 0):`。0 等于 0，条件成立！
3. **第 2 行：** 执行 `yield []`。很好，我们成功向外吐出了一个空列表 `[]`。
4. **关键点来了：** 在普通函数里，遇到 `return` 函数就结束了。 **但是 `yield` 不会结束函数！**  它只是暂停一下交出数据，下次恢复时，它会继续往下走!
5. **第 3 行：** Python 继续往下看，遇到了新的条件分支 `if (n == 1):`。0 不等于 1，条件不成立。
6. **第 4 行：** 因为上面的 `if` 不成立，Python 理所当然地走进了 `else:` 分支。
7. **第 5 行：** 准备执行 `for element in stair_ways(n-1):`。因为 `n=0`，所以这里变成了调用 `stair_ways(-1)`！

---
### 正确的代码：
```Python
    if(n==0):
        yield []
    elif(n==1):
        yield [1]
    else:
        for element in stair_ways(n-1):
            yield element + [1] 
        for element in stair_ways(n-2):
            yield element + [2]
```