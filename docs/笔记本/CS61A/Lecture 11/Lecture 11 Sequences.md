## 1. Working with Lists
```Python
    digits = [1, 8, 2, 8]
    
    ## The number of elements
    len(digits)
    
    ## An element selected by its index
    digits[3]
    getitem(digits, 3)
    
    ## Concatenation and repetition
    [2, 7] + digits*2
    add([2, 7], mul(digits, 2))
    >>> [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]
```

---

## 2. For Statement
```Python
    def count(s, value):
        total = 0
        for element in s:
            if element == value
                total +=1
        return total
```

### Sequence Unpacking in For Statement :
![](附件/Pasted%20image%2020260203014902.png)

---

## 3. Range
![](附件/Pasted%20image%2020260203015343.png)


!!! example
    ```Python
    def sum_below(n);
        total=0
        for i in range(n):
            total=total+i
        return total
    ```
    #### 上述代码的输出结果如下：
    ![](附件/Pasted%20image%2020260203015908.png)
    

### If we just want to do something in a fixed number of times
```Python
    def cheer():
        for _ in range(3):
            print('Go bears!')
```
#### 上述代码的输出结果如下：
![](附件/Pasted%20image%2020260203020115.png)
我们可以使用一个 ==下划线字符== 或 ==空格== 来让其他程序员知道这里没有东西。我们也可以把这个下划线改成 `x` 或者其它东西，这样并不会有任何影响，用下划线和空格只是为了让人看得懂。

---

## 4. List Comprehensions

![](附件/Pasted%20image%2020260203020721.png)

!!! example
    ```
        def division():
            return [1] + [x for x in range(2, n) if n%x==0]
    ```
    #### 输入与输出结果如下：
    ![](附件/Pasted%20image%2020260203020937.png)
    
