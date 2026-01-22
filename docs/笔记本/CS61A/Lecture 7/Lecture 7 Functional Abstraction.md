## 1. Environment Diagrams with Lambda
A lambda function's parent is the current frame in which the lambda expression is evaluated.
!!! example 
    ![](附件/Pasted%20image%2020260122191031.png)
    这道题中，`f`函数会先在自己的当前帧找变量，由于当前帧中没有，且f函数的`parent`是`global`，f所以会去`global`中寻找`a`和`y`，由于`global`中的`a=1`，所以此时`lambda y: a + y` 这个式子中的`a=1`，由用户`g`函数需要接收一个变量`y`，此时的代码为`f(lambda y : a + y)(a)`，因此接受的`y`值为`a`，由于`f`的`parent`是`global`，所以`y`接受的`a`是1而不是2，因此`g(y)`的值为`1+1=2`。然后再执行`f`内部的代码，由于`f`内部的`parent`为`f`，所以`lambda y: a * g(y)`接受的`a`为`2`。所以这道题的输出结果为`4`。

---

## 2. Functional Abstractions
Functional abstraction is giving a name to some computational process and then referring to that process as a whole without worrying about its implementation details.（函数抽象就是给某个计算过程起个名字，然后整个过程都用这个名字来引用，而不用担心具体的实现细节。）

!!! example 
    ```Python
        def square:
            return mul(x, y)
            
        def sum_square(x, y)
            return square(x) + square(y)
    ```
    
    ### What does sum_square need to know about square?
        
        1. Square takes one argument. Yes(如果不知道的话就没法调用)
        2. Square has the intrinsic name square. No(固有名称只是为了让人类能够检查函数的名称而存在，只要它绑定到当前环境中的名称为square，任何带有任何固有名称的函数都可行。)
            - ```Python
                def f(x):  # 固有名称是 'f'
                    return x * x
                    
                # 把函数 f 赋值给变量名 square
                square = f 
                
                def sum_square(x, y):
                    return square(x) + square(y) # sum_square 在这里调用了 square
                ```
                在这个例子中：
                - `sum_square` 内部写的是 `square(x)`。
                - 当程序运行到这一行时，它会在当前环境中找名为 `square` 的东西。
                - 它找到了之前赋值给 `square` 的那个函数对象。
                - 哪怕那个函数对象的“真名”（固有名称）其实是 `f`，`sum_square` 也能运行得很好。
        3. Square computes the square of a number. Yes
        4. Square computes the square by calling mul. No
    




