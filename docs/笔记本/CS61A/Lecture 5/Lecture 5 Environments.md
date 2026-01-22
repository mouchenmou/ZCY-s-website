
## How to draw an environment diagram

#### When a function is defined:
- Create a function value:  `func <name>(<formal parameters>  [parent = <parent>]`
- Its parent is the current frame.
    - f1: make_adder                 func adder(k)  [parent=f1]
- Bind  `<name>` to the function value in the current frame.


---

## Local Name
### Local Names are not visible to Other (Non-Nested）Functions
![](附件/Pasted%20image%2020260119153311.png)
- An environment is a sequence of frames.
- The environment created by calling a top-level function (no def within def) consists of one local frame, followed by the global frame.

---

## Lambda Expression

!!! definition "Lambda"
    使用赋值语句将一个接受名为x的参数的函数绑定到名为square的名称上：
    ```Python
        square = lambda x: x*x
        print(square)
        print(square(5))
    ```
    
    ### 上述代码的输出结果如下：
    ![](附件/Pasted%20image%2020260119161020.png)


- 需要注意的是lambda是不会有return这个关键字的，我们只需要在冒号后直接写下返回表达式
- lambda表达式创建函数，但它们总是创建简单的函数，只是评估单个表达式而已
![](附件/Pasted%20image%2020260119161316.png)
---

## Lambda Expression VS. Def Statement

### 1. 共同点：
1. Both create a function with the same domain, range, and behaviour.
2. Both functions have as their parent the frame in which they were defined.
3. Both bind that function to the name square.

### 2. 不同点
- Only the def statement gives the function an intrinsic name.

```Python
    square1 = lambda x: x*x
    def square2(x):
        return x*x
        
    print(square1)
    print(square2)
```
### 上述代码的输出结果如下：

![](附件/Pasted%20image%2020260119162505.png)

---

## Currying

### Function currying
柯里化是将多参数函数住哪换位一个单参数高阶函数的行为，该函数返回一个接受其余参数的函数。

```Python
    def curry2(f):
        def g(x):
            def h(y):
                return f(x, y)
            return h
        return g
        
    from operator import add
    m=curry2(add)
    m_add=m(2)
    m_add(3)
    print(m_add(3))
```
上述代码的输出结果为5

