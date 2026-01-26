### 这道题跟Q4是同一道题的两种实现方式，一种是正向递归一种是反向递归

Write a **recursive** function `count_dollars_upward` that is just like `count_dollars` except it uses `next_larger_dollar`, which returns the next larger dollar bill value from the input (e.g. `next_larger_dollar(5)` is `10`). _The function will return `None` if the next dollar bill value does not exist._

> **Important:** Use recursion; the tests will fail if you use loops.

```
def next_larger_dollar(bill):
    """Returns the next larger bill in order."""
    if bill == 1:
        return 5
    elif bill == 5:
        return 10
    elif bill == 10:
        return 20
    elif bill == 20:
        return 50
    elif bill == 50:
        return 100

def count_dollars_upward(total):
    """Return the number of ways to make change using bills.

    >>> count_dollars_upward(15)  # 15 $1 bills, 10 $1 & 1 $5 bills, ... 1 $5 & 1 $10 bills
    6
    >>> count_dollars_upward(10)  # 10 $1 bills, 5 $1 & 1 $5 bills, 2 $5 bills, 10 $1 bills
    4
    >>> count_dollars_upward(20)  # 20 $1 bills, 15 $1 & $5 bills, ... 1 $20 bill
    10
    >>> count_dollars_upward(45)  # How many ways to make change for 45 dollars?
    44
    >>> count_dollars_upward(100) # How many ways to make change for 100 dollars?
    344
    >>> count_dollars_upward(200) # How many ways to make change for 200 dollars?
    3274
    >>> from construct_check import check
    >>> # ban iteration
    >>> check(HW_SOURCE_FILE, 'count_dollars_upward', ['While', 'For'])
    True
    """
    "*** YOUR CODE HERE ***"
```

---
### 一开始的错误解法：
```python
    def calculate(t, bill):
        if (t<0  or bill==None ):
            return 0
        elif(t<5):
            return 1
        else:
            last=calculate(t-bill,bill)
            not_last=calculate(t,next_larger_dollar(bill))
            return last+not_last
    return calculate(total, 1)
```

## ：为什么 `t < 5` 不能直接返回 `1`？

### 错误举例：

假设我们输入的`total=6`，按理来说应该只有两种方法，那我们将`total=6`带入到上面的代码中看看结果如何。
$calculate(6, 1) = calculate(5, 1)+calculate(6, 5)$
$calculate(5, 1)+calculate(6, 5)= calculate(4, 1)+calculate(5, 5)+calculate(1, 5)+calculate(6, 10)$

- $calculate(4, 1)=1$
- $calculate(5, 5)会被拆解成(0, 5)和(5, 10)$
    - $calculate(0, 5)直接返回1$
    - $calculate(5, 10)$会被拆解成$(-5, 10)$和$(5,20)$，$-5$会返回$0$，$(5, 20)$一直递归到`bill`超过`100`也会返回`0`，因此$calculate(5, 10)=0$
- $calculate(1, 5)也返回1$

这样子算下来答案就是3了。

但是按照这道题的思路，我们是将`bill`从小到大进行递归的，所以当`bill=5`的时候，我们只能考虑`bill=5`或者比`5`大的时候的情况而不能考虑`bill=1`的情况，所以`calculate(1, 5)`应该返回的是`0`而不是`1`。

### 错误核心：
- 为什么会想到t<5时返回1是因为上一道题中从上往下递归，递归到最下面之后是只有bill=1，其它bill都不能存在的情况，这个时候可以有无数个1，而上一个bill是5，所以在5以内都是一种情况。
- 至于为什么在这里不适用，是因为这道题中bill=1是最基础的情况，bill=1时允许bill从1到100不等，而bill=5时只允许bill从5到100不等，bill=100时只允许bill=100。比方我们递归一个东西递归到calculate(4,10)时，在上一道题中bill可以为1，5，10所以在上一道题时这个值存在一种情况。但是在这道题中，bill只能等于10或者比10大，所以存在0种情况使得4将10分掉。
- 至于为什么t=0时返回1，是因为t能正好把当前的bill分完，比方说calculate(20, 20)，此时的bill只能是20或者比20大的数，total=20正好能把bill=20给分成一整份，bill>20时都没法分，所以返回1


---

### 正确解法：
```python
    def calculate(t, bill):
        if (t<0  or bill==None ):
            return 0
        elif(t==0):
            return 1
        else:
            last=calculate(t-bill,bill)
            not_last=calculate(t,next_larger_dollar(bill))
            return last+not_last
    return calculate(total, 1)
```
