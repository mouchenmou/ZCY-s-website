## Logical Operators

### 1. To evaluate the expression `<left>` and `<right>`:

1. Evaluate the subexpression `<left>`.
2.  If the result is a false value v, then the expression evaluates to v.
3. Otherwise, the expression evaluates to the value of the subexpression `<right>`.

### 举个例子：

- **`2 and 3` 返回的值是3** ：因为`2`是true，而规则2讨论的是左值为false的情况，故我们考虑规则3。规则3说如果左值为true的话就取右值，由于右值3也是true，所以`2 and 3` 返回的值是3。
- **`0 and 2` 返回的值是0** ：因为0是false，符合规则2，因此取左值，答案为0。

### 2. To evaluate the expression `<left> or <right>`:

1. Evaluate the subexpression `<left>`.
2. If the result is a true value v, then the expression evaluates to v.
3. Otherwise, the expression evaluates to the value of the subexpression `<right>`.

### 举个例子：

- **`0 or 3` returns the value 0** : Since 0 is false and comfores to Rule 2, the returned value is determined by the left value, which is 0.
- **`1 or 3` returns the value 3** : Since 1 is false and comfores to Rule 3, the returned value is determined by the right value, which is 3.

---

### 断言语句

比方说我们设计一个计算正多边形面积的函数，通过输入的边长来计算这个正多边形的面积。但是设计时会出现一个问题，那就是输入的数为负数时，同样也能输出一个面积数，为了解决这个问题，我们需要引入 **断言语句**。

断言语句以关键字 `assert` 开头，后面是一个表达式，这是一个布尔上下文。
- 如果该表达式的值为假，那么每当用户运行python程序时，都会打印出一个错误消息。

```Python
    assert 3>2, "Math is broken"
    assert 2>3, "That is false"
```
### 执行完上述代码后输出结果如下：
![](附件/Pasted%20image%2020260118225416.png)

### 下面设计一个计算圆面积的函数：
```Python
    from math import pi
    def area_square(n):
        assert n>0, "A length must be positive"
        return pi*n*n
```


---

## Generalization

我们可以通过generalization来设计一个函数，这个函数既能够求前n项的和，也能够求前n项的立方和：

```Python
    def identity(k):
        return k
        
    def cube(k):
        return pow(k, 3)
        
    def summation(n, term):
        total=0
        k=1
        while(k<=n):
            total=total+term(k)
            k=k+1
        print(total)
        return total
```

### 上述代码的输出结果：
![](附件/Pasted%20image%2020260118232146.png)

---

## Functions as return values

### make_adder returns a function, adder just returns a number
```python
    def make_adder(n):
         def adder(k):
            return n+k
        return adder
```

adder可以使用它的形式参数以及make_adder的形式参数，即周围函数。
![](附件/Pasted%20image%2020260118233653.png)
![](附件/Pasted%20image%2020260118233924.png)