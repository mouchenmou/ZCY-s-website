这一节课讲的内容跟数据结构学的算时间复杂度差不多。

## 1. 计算斐波那契数列的效率
```python
def fib(n):
    if n==0 or n==1 :
        return n
    else:
        return fib(n-1)+fib(n-2)
        
def count(f):

    def counted(n):
        counted.call_count+=1
        return f(n)
        
    counted.call_count=0
    return counted
```

### 我的疑问

我感觉写了上面这段代码之后，没法做到记录函数总共执行了多少次，比如我这么写：

```Python
fib_new=count(fib)
fib_new(5)
print(fib_new.call_count)
```
执行了这段语句之后，确实如我所说，没法做到真正计算 `fib(5)` 真正执行的次数，而 `print(fib_new.call_count)` 确实也只会打印出 `1`。

但是，执行下面这段语句，就能正确输出 `fib(5)` 真正执行的次数了 ：

```Python
fib=count(fib)
fib(5)
print(fib.call_count)
```

这段代码我原先也有一个错误的理解：我认为执行过程是这样的：

1. 执行了 `fib=count(fib)` 之后，左边的 `fib` 变成了 `counted` 函数，右边那个括号里的 `fib` 还是原先老版的 `fib` 函数。
2. 然后执行 `fib(5)` 等价于执行 `counted(5)`
3. 执行 `counted(5)` 中返回的是 `f(5)`，而此时，`f` 函数在第一步中被复制成了 原先的`fib`函数，因此会返回 `fib(5)`。
4. 而 `fib(5)` 会返回 `fib(4)+fib(3)`，就这样一直下去，我觉得这里的 `fib` 函数执行的都是老版的 `fib` 函数，而不是改版后的 `counted` 函数，所以我觉得执行完 `fib(5)`这句话之后，`counted.call_count+=1` 只被执行了一次，之后的递归中执行的 `fib` 函数都是老版的 `fib` 函数，跟 `counted` 函数都没关系了

我理解的步骤1，2，3都是正确的，但是==步骤4我的理解是错误的==

应该这样子来看这个问题：

`fib` 最开始是被定义为求斐波那契数列的函数，然后 `f` 函数是 `count` 函数的参数。

- 执行了 `fib=count(fib)` 之后，等号左侧的`fib` 函数便不再是原来的求斐波那契数列的函数，而是被赋值为了 `counted` 函数。
- 而 `count(fib)` 中的 `f` 函数被赋值为了原始的 `fib` 函数，即求斐波那契数列的那个函数

因此，接下来执行 `fib(5)` 之后，其实是调用了 `counted(5)`， `counted(5)` 内会返回 `f(5)`，此时的`f` 函数是老版的 `fib` 函数，在内部会返回 `fib(4)+fib(3)`。然而，此时的`fib` 函数已经不再是原先求斐波那契的函数了，而是被赋值为了 `counted` 函数，因此还会接着执行 `counted(3)` 和 `counted(4)`，这也是为什么 `counted.call_count` 能够计算出斐波那契数列一共执行了多少次。

真正的解释应该是看==这个函数在哪个环境中==：

- 在Global Frame中，`fib` 函数被赋值为了 `counted` 函数。
- 执行了 `fib(5)` 之后，会进入Counted Frame。
- `fib(5)` 返回了 `f(5)`，`f(5)` 会进入Fib Frame中，需要注意的是，这里的 Fib Frame是老版的fib函数内部的环境。
- 进入了Fib Frame环境之后，会执行 `fib(n-1)+fib(n-2)`。重点来了！！！`fib` 函数在Fib Frame中并没有被定义，因此会去Global Frame中找。在Global Frame中找到了 `fib` 函数，它被赋值为 `counted` 函数了，因此 `fib(n-1)+fib(n-2)`调用的不再是老版的 `fib` 函数，而是调用 `counted` 函数

![](附件/Pasted%20image%2020260717225727.png)

---
## 2. Memoization
思想：Remember the results that have been computed before

```python
def memo(f):
    cache = P{}
    def memoized(n):
        if n not in cache:
            cache[n] = f(n)
        return cache[n]
    return memoized
```


这里有点太难理解了，实在是不好解释清楚。总之严格按照函数所在的环境去分析，是可以分析出来的。
![](附件/Pasted%20image%2020260718000150.png)

---

## 3. Exponention
