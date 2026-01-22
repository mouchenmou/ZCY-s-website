## Environment Diagrams with Lambda
A lambda function's parent is the current frame in which the lambda expression is evaluated.
!!! example 
    ![](附件/Pasted%20image%2020260122191031.png)
    这道题中，`f`函数会先在自己的当前帧找变量，由于当前帧中没有，且f函数的`parent`是`global`，f所以会去`global`中寻找`a`和`y`，由于`global`中的`a=1`，所以此时`lambda y: a + y` 这个式子中的`a=1`，由用户`g`函数需要接收一个变量`y`，此时的代码为`f(lambda y : a + y)(a)`，因此接受的`y`值为`a`，由于`f`的`parent`是`global`，所以`y`接受的`a`是1而不是2，因此`g(y)`的值为`1+1=2`。然后再执行`f`内部的代码，由于`f`内部的`parent`为`f`，所以`lambda y: a * g(y)`接受的`a`为`2`。所以这道题的输出结果为`4`。

