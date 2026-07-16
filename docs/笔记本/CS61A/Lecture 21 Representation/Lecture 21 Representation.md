## 1. Expressions, Values, and Types

Python 执行一个 expression（表达式）时，会得到一个 value（值），每个值都有自己的 type（类型）。

在面向对象编程中，class 本身就是一种 type，而调用 class 会创建这个 type 的 instance：

```python
class Letter:
    pass

letter = Letter()

type(letter)
# <class '__main__.Letter'>

isinstance(letter, Letter)
# True
```

Lecture 18 到 Lecture 20 主要在讨论对象里面保存什么属性、如何查找属性以及类之间如何继承。这节课进一步讨论：**当一个对象需要被显示出来时，Python 应该用什么字符串来表示它？**

---

## 2. String Representations

在 Python 中，每个对象都有两种 string representation（字符串表示）：

1. `str`：给人看
2. `repr`：给 Python interpreter 看

大部分情况它们都是相同的，但也不是完全相同。

!!! example "Fraction 的两种表示"
    ```python
    from fractions import Fraction

    half = Fraction(1, 2)

    str(half)
    # '1/2'

    repr(half)
    # 'Fraction(1, 2)'
    ```

    `1/2` 很适合人阅读，但如果把它交给 Python 求值，Python 会把它理解成除法表达式，结果是 `0.5`。

    `Fraction(1, 2)` 则清楚地写出了这个对象的类型以及构造它所需的参数，更接近 Python interpreter 能理解的形式。

### `repr` 的理想性质

一个设计良好的 `repr` 通常希望满足：

```python
eval(repr(object))
```

可以创建出一个与原对象等价的新对象。

```python
half = Fraction(1, 2)
repr(half)
# 'Fraction(1, 2)'

eval(repr(half))
# Fraction(1, 2)
```

!!! warning "这只是设计目标，不是强制保证"
    并不是所有对象的 `repr` 都能交给 `eval`。例如一个普通用户自定义对象如果没有设置自己的 representation，通常会显示：

    ```python
    <__main__.Letter object at 0x...>
    ```

    这种字符串包含对象的类型和内存信息，但不是一个可以直接重新创建对象的 Python expression。

---

## 3. Python 在什么时候使用 `str` 和 `repr`

### 在交互式解释器中直接输入对象

Python shell 需要显示 expression 的 value，因此会使用这个对象的 `repr`：

```python
>>> half
Fraction(1, 2)
```

### 使用 `print`

`print` 的目标是给人看，因此会使用对象的 `str`：

```python
>>> print(half)
1/2
```


!!! explanation "如果类没有定义自己的 `__str__`"
    Python 会退回去使用 `__repr__`。因此只定义 `__repr__` 时，直接在 shell 中查看对象和 `print(object)` 都可能得到同一种结果。

---

## 4. `repr`、`str` 与对应的特殊方法

!!! explanation "我的疑问"
    ### 我的疑问：`
    repr` 和 `__repr__` 明明是两个东西，为什么会联系在一起？

    解答：

    `repr` 和 `__repr__` 的确不是同一个东西。

	1. `repr` 是 Python 已经提供好的 built-in function，可以直接写 `repr(x)`。
	2. `__repr__` 是定义在 class 中的 special method，用来告诉 Python 这个 class 的对象应该怎样被正式表示。
	
	它们之所以有联系，是因为 **Python 语言本身提前规定了这套对应关系**：
	
	```text
	用户调用 repr(x)
	        ↓
	Python 查看 x 所属的 class
	        ↓
	在这个 class 中寻找 __repr__
	        ↓
	调用 __repr__，并把 x 传给 self
	```
	
	因此，`repr(x)` 的行为更接近：
	
	```python
	type(x).__repr__(x)
	```
	
	同理，Python 也提前规定了很多类似的对应关系：
	
	```text
	repr(x)  -> 寻找 __repr__
	str(x)   -> 寻找 __str__
	len(x)   -> 寻找 __len__
	x + y    -> 寻找 __add__
	x == y   -> 寻找 __eq__
	```
	
	双下划线是在提醒程序员：这不是一个随便起名的普通方法，而是 Python 语言规定的特殊接口。因此这类方法也叫 special method、magic method 或 dunder method。

!!! explanation "可以把它理解成按钮和接口"
    `repr(x)` 是所有对象共用的“正式显示”按钮，`__repr__` 是每个 class 为这个按钮提供的接口。

    用户只需要按下同一个按钮：

    ```python
    repr(x)
    ```

    至于最后怎样显示，由 `x` 所属的 class 中定义的 `__repr__` 决定。

### 如果只定义普通的 `repr` 方法会怎样？

我们当然可以自己定义一个名字叫 `repr` 的普通方法：

```python
class Student:
    def repr(self):
        return 'ordinary repr'
```

但它只是一个普通 method，只能由我们主动调用：

```python
tom = Student()

tom.repr()
# 'ordinary repr'
```

Python 内置的 `repr(tom)` 不会调用它，因为 Python 规定自己寻找的接口叫 `__repr__`，而不是 `repr`。

```python
class Student:
    def repr(self):
        return 'ordinary repr'

    def __repr__(self):
        return 'special repr'
```

现在两种调用会得到不同结果：

```python
tom = Student()

tom.repr()
# 'ordinary repr'

repr(tom)
# 'special repr'
```

!!! warning "一个 class 中不能用两个同名方法分别表示 `repr` 和 `str`"
    如果写成：

    ```python
    class Fraction:
        def repr(self):
            return 'Fraction(1, 2)'

        def repr(self):
            return '1/2'
    ```

    后面的 `repr` 会覆盖前面的 `repr`，最后 class 中只剩下返回 `'1/2'` 的版本。

    如果只是想定义两个普通方法，至少要使用两个不同的名字：

    ```python
    def repr(self):
        return 'Fraction(1, 2)'

    def str(self):
        return '1/2'
    ```

    但这样只能手动调用 `half.repr()` 和 `half.str()`。如果想接入 Python 内置的 `repr(half)` 和 `str(half)`，就必须实现 Python 规定的 `__repr__` 和 `__str__`。

---

## 5. Polymorphic Functions

Polymorphic function（多态函数）指的是：**同一个 function 可以处理许多不同类型的对象，具体行为由对象所属的 class 决定。**

`repr` 和 `str` 都是 polymorphic functions：

```python
repr(123)
# '123'

repr([1, 2])
# '[1, 2]'

repr('hello')
# "'hello'"

repr(Fraction(1, 2))
# 'Fraction(1, 2)'
```

---

### 6. 为什么 `repr(x)` 使用 `type(x).__repr__(x)`？

如果让我们自己近似实现 built-in function `repr`，下面哪一种写法正确？
![](附件/Pasted%20image%2020260715212103.png)

正确答案是：

```python
def repr(x):
    return type(x).__repr__(x)
```

以 `half` 为例：

```python
type(half)
# Fraction
```

所以：

```python
type(half).__repr__(half)
```

就是：

```python
Fraction.__repr__(half)
```

它表达的意思是：使用 `Fraction` class 规定的 `__repr__`，显示 `half` 这个对象。

这里必须先说明一个容易漏掉的前提：**正常情况下，instance `x` 自己身上并没有 `__repr__`。**

例如：

```python
class A:
    def __repr__(self):
        return 'class repr'


x = A()
```

但是Python 允许我们在创建 instance 以后，随时给它添加新的 instance attribute：

```python
x.name = 'Tom'
```

因此，程序员也可以手动给 `x` 添加一个名字叫 `__repr__` 的 instance attribute：

```python
x.__repr__ = lambda: 'instance repr'
```

执行完这一行之后，同时存在两个不同的 `__repr__`：

```python
A.__repr__
# 定义在 class A 中

x.__repr__
# 手动添加在 instance x 上
```

直接执行普通的 dot expression 时，会先找到 `x` 自己的 instance attribute：

```python
x.__repr__()
# 'instance repr'
```

但是 built-in function `repr` 会忽略这个手动添加的 `x.__repr__`，直接使用 class `A` 中定义的 `A.__repr__`：

```python
repr(x)
# 'class repr'
```

所以：

> An instance attribute called `__repr__` is ignored! Only class attributes are found.

专门讨论的是这种不常见的情况：程序员执行了 `x.__repr__ = ...`，手动给某一个 instance 添加了自己的 `__repr__`。

如果程序员从来没有写过：

```python
x.__repr__ = ...
```

那么 `x` 自己身上就没有 `__repr__`。这时 `x.__repr__` 和 `repr(x)` 最后找到的都是 class 中定义的 `A.__repr__`。

正因为 Python 必须保证 `repr(x)` 忽略手动添加的 instance attribute，所以在Class中，它不能简单写成 `x.__repr__()`，而要近似写成：

```python
type(x).__repr__(x)
```

---

## 7. Interface

### 7.1 接口的定义

在一段程序中，好几个类可以定义一些**名称相同、用途相同的方法**。

那么这些方法就可以称为一个接口，可以称为一个接口。

例如：

```
class Dog:
    def speak(self):
        return "汪汪"


class Cat:
    def speak(self):
        return "喵喵"
```

`Dog` 和 `Cat` 中都有一个叫作 `speak` 的方法。

虽然它们返回的结果不同：

```
Dog().speak()   # "汪汪"
Cat().speak()   # "喵喵"
```

但它们的用途相同，因此，可以认为 `Dog` 和 `Cat` 都实现了 `speak` 这个接口。

不过需要注意：

> 不同类中这个方法的具体代码不需要完全相同，只需要它们表示的功能相同。

## 7.2 Python 内置的接口

有些接口是 Python 提前规定好的。

例如：

```
__init__
__str__
__repr__
__len__
```

这些方法名有特殊含义，Python 会在特定情况下自动调用它们。

## 7.3 我们也可以自己定义接口

我们刚刚在7.3中就自己定义了speak这个接口。
