
来源：[CS231n Python Numpy Tutorial](https://cs231n.github.io/python-numpy-tutorial/)

> 这份笔记是对原英文教程的中文整理版，覆盖原页面的主要知识点、章节顺序、关键代码和学习提醒；不是逐字逐句的全文翻译。

## 目录

- [[#Jupyter 和 Colab Notebook]]
- [[#Python]]
  - [[#Python 版本]]
  - [[#基本数据类型]]
  - [[#容器]]
  - [[#函数]]
  - [[#类]]
- [[#NumPy]]
  - [[#数组]]
  - [[#数组索引]]
  - [[#数据类型]]
  - [[#数组数学]]
  - [[#广播]]
- [[#SciPy]]
- [[#Matplotlib]]
- [[#补充说明：为什么有些输出不是固定的]]

## Jupyter 和 Colab Notebook

CS231n 的作业主要使用 Python。Python 本身是通用编程语言；配合 `numpy`、`scipy`、`matplotlib` 这些库之后，它会变成非常适合科学计算和机器学习实验的环境。

Notebook 的意义：

- Jupyter Notebook 可以在本地浏览器里编写和运行 Python 代码。
- 它适合把代码分块运行、快速实验、观察中间结果。
- Colab 是 Google 提供的云端 Jupyter 环境，免安装、方便分享，并且通常可以使用 GPU / TPU 等硬件加速资源。

运行方式：

- 推荐：点击原网页顶部的 `Open in Colab`，直接在 Colab 中运行。
- 本地运行：先按课程说明配置虚拟环境，然后安装 Jupyter：

```bash
pip install notebook
jupyter notebook
```

正常情况下，浏览器会打开类似 `http://localhost:8888` 的本地页面，然后可以选择对应的 `.ipynb` 文件运行。

## Python

Python 是一种高级、动态类型、多范式编程语言。它的代码可读性很强，很多时候接近伪代码，但表达能力很强。

快速排序的 Python 写法可以非常紧凑：

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))
# [1, 1, 2, 3, 6, 8, 10]
```

### Python 版本

Python 2 已停止官方支持。CS231n 课程代码使用 Python 3。学习和写作业时应确认当前环境使用的是 Python 3：

```bash
python --version
```

### 基本数据类型

Python 常用基础类型包括：

- 数字：整数、浮点数、复数。
- 布尔值：`True`、`False`。
- 字符串：单引号或双引号都可以。

#### 数字

```python
x = 3
print(type(x))  # <class 'int'>
print(x + 1)    # 4
print(x - 1)    # 2
print(x * 2)    # 6
print(x ** 2)   # 9

x += 1
print(x)        # 4

x *= 2
print(x)        # 8

y = 2.5
print(type(y))  # <class 'float'>
print(y, y + 1, y * 2, y ** 2) # 2.5 3.5 5.0 6.25
```

注意：Python 没有 `x++` 或 `x--` 这样的自增、自减运算符。

#### 布尔值

Python 使用英文单词表示逻辑运算：

```python
t = True
f = False

print(type(t))  # <class 'bool'>
print(t and f)  # False，逻辑与
print(t or f)   # True，逻辑或
print(not t)    # False，逻辑非
print(t != f)   # True，可以理解为异或效果
```

#### 字符串

```python
hello = "hello"
world = "world"

print(hello)                   # hello
print(len(hello))              # 字符串长度；输出：5
print(hello + " " + world)     # 字符串拼接；输出：hello world
print("%s %s %d" % (hello, world, 12)) # hello world 12
```

常见字符串方法：

```python
s = "hello"

print(s.capitalize())          # 首字母大写；输出：Hello
print(s.upper())               # 全部大写；输出：HELLO
print(s.rjust(7))              # 右对齐并补空格；输出："  hello"
print(s.center(7))             # 居中并补空格；输出：" hello "
print(s.replace("l", "(ell)")) # 替换子串；输出：he(ell)(ell)o
print("  world ".strip())      # 去掉首尾空白；输出：world
```

## 容器

Python 内置了几种非常常用的容器类型：

- `list`：列表。
- `dict`：字典，键值对。
- `set`：集合，不重复、无序。
- `tuple`：元组，不可变、有序。

### 列表

列表类似其他语言中的数组，但长度可变，元素类型也可以不同。

```python
xs = [3, 1, 2]
print(xs, xs[2])  # [3, 1, 2] 2
print(xs[-1])     # 负索引从末尾开始

xs[2] = "foo"
xs.append("bar")
print(xs)         # [3, 1, 'foo', 'bar']

x = xs.pop()
print(x, xs)      # bar [3, 1, 'foo']
```

#### 切片

切片用于取出子列表，语法通常是 `start:end`，左闭右开。

```python
nums = list(range(5))  # [0, 1, 2, 3, 4]

print(nums[2:4])   # [2, 3]
print(nums[2:])    # [2, 3, 4]
print(nums[:2])    # [0, 1]
print(nums[:])     # [0, 1, 2, 3, 4]
print(nums[:-1])   # [0, 1, 2, 3]

nums[2:4] = [8, 9]
print(nums)        # [0, 1, 8, 9, 4]
```

切片在 NumPy 数组中也会频繁出现。

#### 循环

```python
animals = ["cat", "dog", "monkey"]

for animal in animals:
    print(animal)

# 输出：
# cat
# dog
# monkey
```

如果需要同时拿到索引和值，可以用 `enumerate`：

```python
for idx, animal in enumerate(animals):
    print("#%d: %s" % (idx + 1, animal))

# 输出：
# #1: cat
# #2: dog
# #3: monkey
```

#### 列表推导式

列表推导式可以用更简洁的方式生成列表。

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)  # [0, 1, 4, 9, 16]
```

也可以加入条件：

```python
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # [0, 4, 16]
```

### 字典

字典保存键值对，类似 Java 的 `Map` 或 JavaScript 的对象。

```python
d = {"cat": "cute", "dog": "furry"}

print(d["cat"])          # cute
print("cat" in d)        # True

d["fish"] = "wet"
print(d["fish"])         # wet

print(d.get("monkey", "N/A"))  # 键不存在时返回默认值；输出：N/A
print(d.get("fish", "N/A"))    # wet

del d["fish"]
print(d.get("fish", "N/A"))    # N/A
```

遍历字典：

```python
d = {"person": 2, "cat": 4, "spider": 8}

for animal in d:
    legs = d[animal]
    print("A %s has %d legs" % (animal, legs))

# 输出：
# A person has 2 legs
# A cat has 4 legs
# A spider has 8 legs
```

同时遍历键和值：

```python
for animal, legs in d.items():
    print("A %s has %d legs" % (animal, legs))

# 输出：
# A person has 2 legs
# A cat has 4 legs
# A spider has 8 legs
```

字典推导式：

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # {0: 0, 2: 4, 4: 16}
```

### 集合

集合是无序且元素不重复的容器。

```python
animals = {"cat", "dog"}

print("cat" in animals)   # True
print("fish" in animals)  # False

animals.add("fish")
print("fish" in animals)  # True
print(len(animals))       # 3

animals.add("cat")        # 已存在的元素不会重复加入
print(len(animals))       # 3

animals.remove("cat")
print(len(animals))       # 2
```

遍历集合的语法和列表类似，但集合没有固定顺序。

```python
animals = {"cat", "dog", "fish"}

for idx, animal in enumerate(animals):
    print("#%d: %s" % (idx + 1, animal))

# 可能输出如下；集合是无序的，因此顺序不保证固定：
# #1: fish
# #2: dog
# #3: cat
```

集合推导式：

```python
from math import sqrt

nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # {0, 1, 2, 3, 4, 5}
```

### 元组

元组是不可变的有序值序列。它和列表很像，但不可修改。重要区别是：元组可以作为字典的键，也可以作为集合的元素；列表不可以。

```python
d = {(x, x + 1): x for x in range(10)}
t = (5, 6)

print(type(t))    # <class 'tuple'>
print(d[t])       # 5
print(d[(1, 2)])  # 1
```

## 函数

Python 使用 `def` 定义函数。

```python
def sign(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"

for x in [-1, 0, 1]:
    print(sign(x))

# 输出：
# negative
# zero
# positive
```

函数也可以使用带默认值的关键字参数：

```python
def hello(name, loud=False):
    if loud:
        print("HELLO, %s!" % name.upper())
    else:
        print("Hello, %s" % name)

hello("Bob")              # Hello, Bob
hello("Fred", loud=True)  # HELLO, FRED!
```

## 类

Python 定义类的语法比较直接。构造函数写成 `__init__`，实例方法的第一个参数通常命名为 `self`。

```python
class Greeter(object):
    def __init__(self, name):
        self.name = name

    def greet(self, loud=False):
        if loud:
            print("HELLO, %s!" % self.name.upper())
        else:
            print("Hello, %s" % self.name)

g = Greeter("Fred")
g.greet()           # Hello, Fred
g.greet(loud=True)  # HELLO, FRED!
```

## NumPy

NumPy 是 Python 科学计算的核心库。它提供了高性能的多维数组对象，以及一系列处理这些数组的工具。机器学习中的向量、矩阵、图像张量等都经常用 NumPy 表示。

### 数组

NumPy 数组可以理解为同类型数据组成的网格：

- 数组元素类型通常一致。
- 数组用非负整数元组索引。
- 维度数量称为数组的 `rank`。
- `shape` 是一个元组，表示每个维度的大小。

从 Python 列表创建数组：

```python
import numpy as np

a = np.array([1, 2, 3])
print(type(a))   # <class 'numpy.ndarray'>
print(a.shape)   # (3,)
print(a[0], a[1], a[2]) # 1 2 3

a[0] = 5
print(a)         # [5 2 3]

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)   # (2, 3)
print(b[0, 0], b[0, 1], b[1, 0]) # 1 2 4
```

常见数组创建函数：

```python
import numpy as np

print(np.zeros((2, 2)))
# [[0. 0.]
#  [0. 0.]]

print(np.ones((1, 2)))
# [[1. 1.]]

print(np.full((2, 2), 7))
# [[7 7]
#  [7 7]]

print(np.eye(2))
# [[1. 0.]
#  [0. 1.]]

print(np.random.random((2, 2)))
# 随机数每次不同，可能类似：
# [[0.91940167 0.08143941]
#  [0.68744134 0.87236687]]
```

### 数组索引

NumPy 支持多种索引方式。

#### 切片索引

数组也可以切片；多维数组需要给每个维度指定切片。

```python
import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

b = a[:2, 1:3]
print(b)
# [[2 3]
#  [6 7]]
```

重要注意点：NumPy 的切片通常是原数组的视图，不是独立副本。修改切片可能会修改原数组。

```python
print(a[0, 1])  # 2
b[0, 0] = 77
print(a[0, 1])  # 77
```

#### 整数索引和切片混用

整数索引和切片索引混用时，结果可能降低维度；只用切片通常会保留维度。

```python
import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

row_r1 = a[1, :]
row_r2 = a[1:2, :]

print(row_r1, row_r1.shape)  # [5 6 7 8] (4,)
print(row_r2, row_r2.shape)  # [[5 6 7 8]] (1, 4)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # [ 2  6 10] (3,)
print(col_r2, col_r2.shape)
# [[ 2]
#  [ 6]
#  [10]] (3, 1)
```

#### 整数数组索引

整数数组索引可以从原数组中按指定位置构造出新数组。

```python
import numpy as np

a = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

print(a[[0, 1, 2], [0, 1, 0]])              # [1 4 5]
print(np.array([a[0, 0], a[1, 1], a[2, 0]])) # [1 4 5]
print(a[[0, 0], [1, 1]])                     # [2 2]
print(np.array([a[0, 1], a[0, 1]]))          # [2 2]
```

一个常见技巧：从矩阵每一行选择或修改一个元素。

```python
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])

b = np.array([0, 2, 0, 1])

print(a[np.arange(4), b])  # [ 1  6  7 11]

a[np.arange(4), b] += 10
print(a)
# [[11  2  3]
#  [ 4  5 16]
#  [17  8  9]
#  [10 21 12]]
```

#### 布尔数组索引

布尔索引用于筛选满足条件的元素。

```python
import numpy as np

a = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

bool_idx = (a > 2)
print(bool_idx)
# [[False False]
#  [ True  True]
#  [ True  True]]

print(a[bool_idx])  # [3 4 5 6]
print(a[a > 2])     # [3 4 5 6]
```

### 数据类型

每个 NumPy 数组中的元素都有统一的数据类型。创建数组时，NumPy 会尝试自动推断类型；也可以用 `dtype` 显式指定。

```python
import numpy as np

x = np.array([1, 2])
print(x.dtype)  # int64，具体名称可能因系统而略有不同

x = np.array([1.0, 2.0])
print(x.dtype)  # float64

x = np.array([1, 2], dtype=np.int64)
print(x.dtype)  # int64
```

### 数组数学

NumPy 的基本数学运算通常是逐元素进行的，可以使用运算符，也可以使用 NumPy 函数。

```python
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))
# 两者输出：
# [[ 6.  8.]
#  [10. 12.]]

print(x - y)
print(np.subtract(x, y))
# 两者输出：
# [[-4. -4.]
#  [-4. -4.]]

print(x * y)
print(np.multiply(x, y))
# 两者输出：
# [[ 5. 12.]
#  [21. 32.]]

print(x / y)
print(np.divide(x, y))
# 两者输出：
# [[0.2        0.33333333]
#  [0.42857143 0.5       ]]

print(np.sqrt(x))
# [[1.         1.41421356]
#  [1.73205081 2.        ]]
```

注意：和 MATLAB 不同，NumPy 中 `*` 是逐元素乘法，不是矩阵乘法。矩阵乘法、向量内积等可以使用 `dot`。

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))
print(x.dot(v))
print(np.dot(x, v))
print(x.dot(y))
print(np.dot(x, y))
```

`sum` 是非常常用的聚合函数：

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])

print(np.sum(x))          # 所有元素求和：10
print(np.sum(x, axis=0))  # 每一列求和：[4 6]
print(np.sum(x, axis=1))  # 每一行求和：[3 7]
```

矩阵转置可以使用 `.T`：

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

print(x.T)
# [[1 3]
#  [2 4]]

v = np.array([1, 2, 3])
print(v)    # [1 2 3]
print(v.T)  # [1 2 3]，一维数组转置后形状不变
```

### 广播

广播是 NumPy 的重要机制：它允许不同形状的数组在算术运算中自动扩展到兼容形状。

典型例子：给矩阵的每一行都加上同一个向量。

#### 显式循环写法

```python
import numpy as np

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])
v = np.array([1, 0, 1])
y = np.empty_like(x)

for i in range(4):
    y[i, :] = x[i, :] + v

print(y)
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
```

这种写法直观，但当矩阵很大时，Python 层循环会比较慢。

#### 用 tile 手动复制

```python
import numpy as np

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])
v = np.array([1, 0, 1])

vv = np.tile(v, (4, 1))
print(vv)
# [[1 0 1]
#  [1 0 1]
#  [1 0 1]
#  [1 0 1]]

y = x + vv
print(y)
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
```

这种写法减少了 Python 循环，但会显式创建多个 `v` 的副本。

#### 广播写法

```python
import numpy as np

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])
v = np.array([1, 0, 1])

y = x + v
print(y)
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
```

这里 `x` 的形状是 `(4, 3)`，`v` 的形状是 `(3,)`。NumPy 会把 `v` 当作每一行都相同的 `(4, 3)` 数组来参与逐元素加法，但不会真的复制出完整矩阵。

#### 广播规则

对两个数组做广播时，大致遵循这些规则：

1. 如果两个数组的维度数不同，就在低维数组的形状前面补 `1`，直到两个数组维度数一致。
2. 对某个维度来说，如果两个数组在该维度大小相同，或者其中一个大小为 `1`，就认为该维度兼容。
3. 所有维度都兼容时，两个数组可以广播。
4. 广播后的形状是两个输入形状在每个维度上的较大值。
5. 原来大小为 `1` 的维度，会表现得像沿该维度复制到了需要的大小。

广播示例：

```python
import numpy as np

v = np.array([1, 2, 3])
w = np.array([4, 5])

print(np.reshape(v, (3, 1)) * w)
# [[ 4  5]
#  [ 8 10]
#  [12 15]]

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print(x + v)
# [[2 4 6]
#  [5 7 9]]

print((x.T + w).T)
# [[ 5  6  7]
#  [ 9 10 11]]

print(x + np.reshape(w, (2, 1)))
# [[ 5  6  7]
#  [ 9 10 11]]

print(x * 2)
# [[ 2  4  6]
#  [ 8 10 12]]
```

广播通常能让代码更短、更快，因此在 NumPy 中应尽量熟练使用。

## SciPy

NumPy 提供多维数组和基本计算工具；SciPy 在此基础上提供大量面向科学和工程应用的函数。

### 图像操作

原教程示例展示了如何：

- 从磁盘读取图片为 NumPy 数组。
- 通过缩放不同颜色通道给图片加色调。
- 调整图片大小。
- 把 NumPy 数组保存回图片文件。

概念上可以理解为：一张彩色图片通常是形状类似 `(height, width, 3)` 的数组，最后一个维度对应 RGB 三个通道。

示意代码：

```python
# 旧版教程中常见写法使用 scipy.misc.imread / imresize / imsave。
# 现代环境中这些接口可能已被移除，通常可以改用 imageio、Pillow 或 skimage。

img_tinted = img * [1, 0.95, 0.9]
```

如果原图数组形状是 `(400, 248, 3)`，读取图片时常见输出类似：

```text
uint8 (400, 248, 3)
```

这里 `[1, 0.95, 0.9]` 会通过广播作用到 RGB 三个通道：红色不变，绿色和蓝色略微降低。

### MATLAB 文件

SciPy 可以读写 MATLAB 文件：

- `scipy.io.loadmat`：读取 `.mat` 文件。
- `scipy.io.savemat`：保存 `.mat` 文件。

### 点之间的距离

`scipy.spatial.distance` 提供了计算点集之间距离的函数。

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

x = np.array([
    [0, 1],
    [1, 0],
    [2, 0],
])

print(x)
# [[0 1]
#  [1 0]
#  [2 0]]

d = squareform(pdist(x, "euclidean"))
print(d)
# [[0.         1.41421356 2.23606798]
#  [1.41421356 0.         1.        ]
#  [2.23606798 1.         0.        ]]
```

含义：

- `pdist(x, "euclidean")` 计算 `x` 中所有行向量两两之间的欧氏距离。
- `squareform` 把压缩形式的距离结果转换成方阵。
- 如果要计算两个不同点集之间的两两距离，可以使用 `scipy.spatial.distance.cdist`。

## Matplotlib

Matplotlib 是 Python 中常用的绘图库。原教程重点介绍 `matplotlib.pyplot`，它提供了类似 MATLAB 的绘图接口。

### 基本绘图

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

`plt.show()` 用于显示图像。

### 多条曲线、标题、图例和坐标轴标签

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title("Sine and Cosine")
plt.legend(["Sine", "Cosine"])
plt.show()
```

### 子图

`subplot` 可以在同一个 figure 中放置多个图。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title("Sine")

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title("Cosine")

plt.show()
```

### 显示图像

`imshow` 可以显示图片数组。

```python
import numpy as np
import matplotlib.pyplot as plt

# 假设 img 是形状为 (height, width, channels) 的图像数组
img_tinted = img * [1, 0.95, 0.9]

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))

plt.show()
```

注意：`imshow` 处理非 `uint8` 图像数组时可能出现显示异常，因此经常需要显式转换为 `np.uint8`。

## 学习重点总结

- Python 的基础语法很简洁，CS231n 中要熟练使用列表、字典、函数、类和推导式。
- NumPy 的核心是 `ndarray`：理解 `shape`、维度、索引、切片和数据类型非常重要。
- NumPy 切片通常返回视图，修改切片可能影响原数组。
- `*` 是逐元素乘法；矩阵乘法要用 `dot` 或现代 Python 中的 `@`。
- `axis=0` 通常沿行方向压缩、得到每列结果；`axis=1` 通常沿列方向压缩、得到每行结果。
- 广播是写出高效向量化代码的关键，要重点掌握形状兼容规则。
- SciPy 补充科学计算工具，例如距离计算、MATLAB 文件读写、图像处理等。
- Matplotlib 用于可视化数据、曲线、子图和图像。

## 补充说明：为什么有些输出不是固定的

- `np.random.random((2, 2))` 会生成随机数，所以每次运行的具体数字都不同。
- `set` 是无序集合，所以遍历集合时，输出元素的顺序不能依赖。
- `dict` 在现代 Python 中通常保持插入顺序，但理解字典时最好重点关注键值对应关系，而不是把它当成普通列表。
- Matplotlib 的 `plt.show()` 不是打印文本，而是在 notebook 或图形窗口里显示图像，因此笔记里用文字说明图像结果。




