## 1. What Would Python Print?
==The print function returns None.== It also displays its arguments(separated by spaces) when it is called.

![](附件/Pasted%20image%2020260123002837.png)

!!! example "一个棘手的问题"
    ### 看明白下面这幅图！！！如果又不会了就自己画一下或者回看Lecture 8（B站上CS自学社区的24spring）
    ![](Pasted%20image%2020260123204234.png)

---

## 2. Function Decorators

### One Example
```Python
    def traced1(fn):
        def traced(x):
            print('Calling', fn, 'on argument ' , x)
            return fn(x)
        return traced
        
    def square(x):
        return x*x
        
    def sum_squares_up_to(n):
        k=1
        total=0
        while k<=n:
            total=total + square(k)
            k=k+1
        return total
```

在终端中输入`sum_squares_up_to(5)`后会得到以下结果：

![](附件/Pasted%20image%2020260123231908.png)

### 装饰函数的原理：

- If I trace a function, that's called the decorator. What you get back is a decorated function, and it is bound to the name of the function that's given.
-  So this is identical to first defining the function and then rebinding the name of the function to a traced version of that function. 

![](附件/Pasted%20image%2020260123234319.png)