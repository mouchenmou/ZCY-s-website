### 这道题的递归方式跟Lecture 10中的[count partitions](<../../Lecture 10/Lecture 10 Tree Recursion.md#partition>)这个例子的递归方式类似。
Given a positive integer `total`, a set of dollar bills makes change for `total` if the sum of the values of the dollar bills is `total`. Here we will use standard US dollar bill values: 1, 5, 10, 20, 50, and 100. For example, the following sets make change for `15`:

- 15 1-dollar bills
- 10 1-dollar, 1 5-dollar bills
- 5 1-dollar, 2 5-dollar bills
- 5 1-dollar, 1 10-dollar bills
- 3 5-dollar bills
- 1 5-dollar, 1 10-dollar bills

Thus, there are 6 ways to make change for `15`. Write a **recursive** function `count_dollars` that takes a positive integer `total` and returns the number of ways to make change for `total` using 1, 5, 10, 20, 50, and 100 dollar bills.

Use `next_smaller_dollar` in your solution: `next_smaller_dollar` will return the next smaller dollar bill value from the input (e.g. `next_smaller_dollar(5)` is `1`). _The function will return `None` if the next dollar bill value does not exist._

> **Important:** Use recursion; the tests will fail if you use loops.
> 
> **Hint:** Refer to the [implementation](https://www.composingprograms.com/pages/17-recursive-functions.html#example-partitions) of `count_partitions` for an example of how to count the ways to sum up to a final value with smaller parts. If you need to keep track of more than one value across recursive calls, consider writing a helper function.

```
def next_smaller_dollar(bill):
    """Returns the next smaller bill in order."""
    if bill == 100:
        return 50
    if bill == 50:
        return 20
    if bill == 20:
        return 10
    elif bill == 10:
        return 5
    elif bill == 5:
        return 1

def count_dollars(total):
    """Return the number of ways to make change.

    >>> count_dollars(15)  # 15 $1 bills, 10 $1 & 1 $5 bills, ... 1 $5 & 1 $10 bills
    6
    >>> count_dollars(10)  # 10 $1 bills, 5 $1 & 1 $5 bills, 2 $5 bills, 10 $1 bills
    4
    >>> count_dollars(20)  # 20 $1 bills, 15 $1 & $5 bills, ... 1 $20 bill
    10
    >>> count_dollars(45)  # How many ways to make change for 45 dollars?
    44
    >>> count_dollars(100) # How many ways to make change for 100 dollars?
    344
    >>> count_dollars(200) # How many ways to make change for 200 dollars?
    3274
    >>> from construct_check import check
    >>> # ban iteration
    >>> check(HW_SOURCE_FILE, 'count_dollars', ['While', 'For'])
    True
    """
    "*** YOUR CODE HERE ***"
    
```

!!! Thinking "我的个人思考"
     我一开始就感觉这道题的递归方式应该跟Lecture 10中的count partitions是一样的。但是我一开始一直用错了方式。像Lecture 10中的那个例子 `count_partitioon(6, 4)`，老师使用的分成两种大情况来看，第一种是最大部分为4的情况、第二种是最大部分小于4的情况。最大部分为4的时候。但是这种方式需要传入两个参数`m、n`，也就是这道题中的` total``和``max_bill` 。可是这道题给出的代码只能传入一个参数，而这道题用了一个很聪明的解法解决了这个问题。
     
     我原来见过的`higher_order_function`都是下面这种形式的：
     
    ```python
        def f1(argument):
             def f2(...)
                 ...
                 return ...
            return f1
                
    ```
    
    但是这道题的解法并非`return f1`，而是给`f2`带上了参数并返回，完美解决了我的疑惑。
    
    

---
### 我的解法如下：

```python
     "*** YOUR CODE HERE ***"
     def calculate(total, bill):
        if(total<0 or bill==None):
            return 0
        elif (total<5):
            return 1
        else:
            last=calculate(total-bill,bill)
            not_last=calculate(total, next_smaller_dollar(bill))
            return last+not_last
        return calculate(total,100)
```
#### Tip：使用total<5在这种递归情况下恰好能成功，但是反过来递归就失败了

---
### Gemini解法：
```python
    def count_dollars(total):
        def calculate(t, bill):
            # 1. 基础情况：必须判断递归中的变量 t
            if t == 0:
                return 1
            # 如果金额变负数，或者已经没有更小的面值了，此路不通
            if t < 0 or bill is None:
                return 0
                
            # 2. 递归逻辑
            # 选择 A: 用一张面额为 bill 的纸币
            use_it = calculate(t - bill, bill)
            # 选择 B: 彻底不用这个面额，换更小的面额试一下
            skip_it = calculate(t, next_smaller_dollar(bill))
            
            return use_it + skip_it
            
        # 注意这里！要调用内部的辅助函数，而不是 count_dollars 自己
        return calculate(total, 100)
```