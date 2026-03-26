What makes methods different from functions is that a function is typically defined in the global frame, and it can be called on any value, whereas a method is specific to a particular object.
![](附件/Pasted%20image%2020260308191908.png)

---

## 1. Classes
![](附件/Pasted%20image%2020260308192539.png)
![](附件/Pasted%20image%2020260308192935.png)
### 我的疑惑：
deposit这个函数明明需要传入两个参数，为什么`a.deposit(10)`的语法是合法的？
### Gemini的解答：
简单来说，没有语法错误是因为：在 Python 中，当你通过对象实例（如这里的 `a`）去调用类的方法时，**Python 会在后台自动将该实例本身作为第一个参数传递过去**。

详细的参数对应过程如下：

- **定义方法时**：`def deposit(self, amount):` 确实定义了两个形参。`self` 代表对象实例本身，`amount` 代表要操作的金额。
- **调用方法时**：当你执行 `a.deposit(10)` 时，Python 会自动把它在底层转换为 `Account.deposit(a, 10)` 的形式。
- **参数的自动绑定**：
    - 实例 **`a`** 被自动隐式地传给了第一个参数 **`self`**。
    - 数字 **`10`** 被显式地传给了第二个参数 **`amount`**。
正因为 Python 帮你代劳了传递 `self` 的工作，所以你在调用时只需要传入 `amount` 这一个参数就可以了。

其实，图片中间那个灰色的气泡提示框正好也在解释这个魔法："self is the instance of the Account class on which deposit was invoked: a.deposit(10)"（`self` 就是调用了 `deposit` 方法的那个 `Account` 类的实例对象）

### 我的疑惑：
那这个跟java不是完全两样了嘛，按理来说`Account.deposit(a, 10)`在Java中应该是static的类才能实现的，而`a.deposit(10)`应该是在non-static类中才能实现的。

### Gemini的解答：
实际上，Python 和 Java 在底层的逻辑是有异曲同工之妙的，只是 **语言的设计哲学和表现形式** 不同。

在 Java 中，你的理解完全正确：
- `a.deposit(10)` 调用的是非静态实例方法。
- `Account.deposit(a, 10)` 调用的是静态方法，需要传入一个 `Account` 类型的实例。

而在 Python 中，之所以看起来“完全两样”，是因为 Python 处理实例引用的方式与 Java 不同：

### 1. Java 喜欢“隐式” (`this`)

在 Java 中定义非静态方法时，你不需要在参数列表里写代表实例本身的参数，比如 `void deposit(int amount)`。 但这并不意味着实例没有被传进去。Java 编译器在底层悄悄地把调用该方法的对象传了进去，并把它绑定到了一个隐藏的关键字 **`this`** 上。你可以直接写 `this.balance = this.balance + amount;`。

### 2. Python 崇尚“显式” (`self`)

Python 有一个核心设计理念叫做 “Explicit is better than implicit”（显式优于隐式）。 Python 的设计者认为，与其像 Java 那样凭空变出一个隐藏的 `this`，不如要求程序员 **在定义方法时，强行把代表实例本身的参数明明白白地写在第一个位置** 。这就是为什么 Python 的实例方法总是带有一个 `self` 参数：`def deposit(self, amount):`。

### 3. Python 的语法糖转换

虽然 Python 要求 **定义时显式** （必须写 `self`），但在 **调用时却是隐式** 的。 为了方便，Python 允许你像 Java 一样使用点号语法来调用实例方法：`a.deposit(10)`。 当你这么写的时候，Python 解释器在后台做了一个“语法糖”转换，直接把它等价翻译成了： `Account.deposit(a, 10)`

所以在 Python 里，`Account.deposit(a, 10)` **并不是** 在调用像 Java 那样的静态方法。它其实就是拿类里面定义的那个普通函数，手动把实例 `a` 塞给了第一个参数 `self`，把 `10` 塞给了第二个参数 `amount`。

---

## Creating Instance
!!! Explanation "Object Construction"
    ![](附件/Pasted%20image%2020260308201219.png)
    ### 我的疑惑：
    在C++中，定义了一个类之后，构造函数都是以这个类来命名的。但是这幅图中定义了Account这个类之后，类里面却没有Account这个构造函数，反而是__init__成为了构造函数，这样子名字对不上号为什么还能成功执行`a = Account('Alan')`?
    
    ### Gemini的解答：
    #### 1. C++ / Java：名字绑定 (Name Binding)
    
    在 C++ 里，语言硬性规定了**“用类名做函数名，这就代表它是构造函数”**。
    
    #### 2. Python：魔法方法 (Magic Methods / Dunder Methods)
    
    Python 没有采用“函数名等于类名”的规则，而是引入了一套“双下划线魔法方法”的机制。Python 的设计者认为，类的初始化、打印、相加等都属于“系统级的特殊行为”，应该用一套统一的特殊命名规则来接管。
    
    当你执行 `a = Account('John')` 时，并不是因为类里面有一个叫 `Account` 的函数，而是 Python 在底层帮你完成了一次**自动映射（Mapping）**：
    
    1. **触发实例化：** 你在外部调用了类本身，也就是 `Account('John')`。
    2. **底层拦截与分配：** Python 拦截到了这个动作。它在内存里建好了一个空对象，然后自动去这个类里面寻找一个叫 `__init__` 的方法，并把 `'John'` 作为参数传进去。
    
    #### 为什么 Python 要这么设计？
    核心是为了保持语言底层的一致性。在 Python 中，所有控制类底层行为的方法都是双下划线开头的（被称为 Magic Methods）。比如：

    - 想要定义对象如何初始化，重写 `__init__`
    - 想要定义 `print(a)` 时打印出什么文本，重写 `__str__`
    - 想要让两个账户对象能直接用加号合并（`a + b`），重写 `__add__`
    通过这种设计，类名就仅仅是一个“蓝图”的名字，而所有具体的“动作规则”都归一化到了双下划线方法中。这让代码的命名空间变得极其干净，不需要你去比对方法名和类名是否一致。
    
    ---
    
    ### 我的疑惑：
    所以说在Python的类中，只能有一个函数带有__这样的下划线，并且带有这样的下划线的函数就是构造函数是吗？
    Gemini的解答：
    不是的哦！这里有两个非常常见的误解，我来帮你理清一下：
    
    ### 1. 类中可以有**很多个**带有双下划线的方法
    
    在 Python 中，带有前后双下划线的方法统称为魔法方法（Magic Methods）或 Dunder Methods（Double Under的缩写）。Python 内部定义了数十个这样的方法，你可以在一个类里重写其中的任意多个，而不是只能有一个。
    
    它们的作用是接管这个类的各种基础行为，不仅仅是初始化。比如：
    
    - `__init__(self)`：负责对象的初始化。
    - `__str__(self)`：负责定义当你 `print(这个对象)` 时，屏幕上应该输出什么字符串。
    - `__len__(self)`：负责定义当你对这个对象调用 `len()` 时，返回什么长度。
    - `__add__(self, other)`：负责定义当两个对象用 `+` 号相加时，应该发生什么。
    你可以把它们想象成 Python 给你留好的“系统接口”，你想自定义哪个行为，就去实现对应的双下划线方法。
    ### 2. 带有双下划线的函数并不都是构造函数
    
    就像上面列举的，除了 `__init__`（和真正负责分配内存的 `__new__`）之外，其他的双下划线方法和“构造对象”**毫无关系**。它们只是用来定义对象在被打印、被相加、被测量长度时的表现。
    除此之外，还有一种情况：如果一个函数或变量只有前面有两个下划线，后面没有（比如 `__my_secret_func`），这在 Python 里代表私有方法/私有变量（Private），类似于 C++ 里的 `private` 关键字，意思是只能在类内部使用，不希望被外部直接调用。

!!! Explanation "Instance Attributes"
    ![](附件/Pasted%20image%2020260308205130.png)
    ### 疑惑：
    Account类里面根本就没有定义backup这个东西，为什么莫名其妙的来了个`a.backup`。
    
    ### Gemini的解答：
    在 Java 或者 C++ 里，类就像是严格的蓝图，对象的所有属性（字段）都必须在类里面提前定义好。但在Python里，对象是极其动态的。
    
    ### Python 的“贴标签”魔法
    
    正如图片里第二个灰色气泡所解释的："A new attribute can be added at any time"（随时可以添加新属性）。
    
    在 Python 中，你可以随时给一个已经创建好的对象强行“贴”上一个新的变量。这就好比在硬件搭建时，原本设计图纸上并没有某个备用传感器，但 Python 允许你直接拿胶带临时把一个新传感器“绑”在机器上，而且系统居然还不报错，直接就能用！
    
    我们逐行拆解一下这段“离谱”的操作：

    1. **`b = Account('Ada')`**：创建了一个账户实例 `b`。
    2. **`b.balance = 20`**：把 `b` 的余额设为 20。
    3. **`a.backup = b`**：**这就是施展魔法的地方。** `Account` 类里确实没有 `backup`。但当你执行这行代码时，Python 会直接在实例 `a` 的肚子里新建一个叫 `backup` 的变量，然后让它指向对象 `b`。
    4. **`a.backup.balance`**：现在你访问 `a.backup`，拿到的其实就是对象 `b`。所以 `a.backup.balance` 等价于访问 `b.balance`，结果自然就是 20。
    
    ### 为什么 Python 允许这么干？
    
    Python 的对象底层其实是用一个Dictionary来存储属性的。当你执行 `a.backup = b` 时，Python 只是简单粗暴地在 `a` 的那个隐藏字典里加了一个键值对：`{'backup': b}`。它赋予了开发者极大的灵活性，但也容易让代码变得难以追踪。

!!! Explanation "Object Identity"
    ![](附件/Pasted%20image%2020260308210216.png)
    这跟CS61B的Lecture 4是一个道理，它们在内存管理上的底层思想是一模一样的。

!!! Explanation "Invoking Methods"
    ![](附件/Pasted%20image%2020260308210756.png)
    这也解释了我在Object Construction中的疑惑

