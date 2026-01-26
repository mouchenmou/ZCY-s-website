```Python 
    def cascade(n):
        if(n<10):
            print(n)
        else:
            print(n)
            cascade(n//10)
            print(n)

```

### 上述代码输入 `cascade(123456)` 后的结果如下：
![](附件/Pasted%20image%2020260125194528.png)

---

## 1. 斐波那契数列就是树形递归中的一种

```python
    def fib(n):
        if(n==0):
            return 0
        elif(n==1):
            return 1
        else:
            return fib(n-2)+ fib(n-1)
```

![](附件/Pasted%20image%2020260125203505.png)

- However, this is not an efficient way to compute Fibonacci numbers, in particular because there's a great amount of repetition within this tree recursive computation.
- Take `fib(5)` as an example, we can see that the `fib(3)` computation is repeated twice.
- 更快的实现方式将在几周之后的课堂上实现。

---

<span id="partition"></span>
## 2. Example: Counting Partitions（计算分区）
The number of partitions of a positive integer n, using parts up to size m, is the number of ways in which n can be expressed as the sum of positive integer parts up to m in increasing order.

### Here is an example:
### 一定要理解这个递归，非常重要！！！
![](附件/Pasted%20image%2020260125204440.png)
```Python
    def count_partitions(n, m):
        if(n==0):
            return 1
        elif(n<0):
            return 0
        elif(m==0):
            return 0
            return 0
        else:
            with_m=count_partitions(n-m, m)
            without_m=count_partitions(n, m-1)
            return with_m+without_m
```


