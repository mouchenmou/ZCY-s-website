## 1. Inheritance

> Inheritance is a technique for relating classes together.

继承就是把类和类之间建立关系。最常见的场景是：一个更特殊的类，拥有一个更一般的类的大部分行为，只是有一些特殊情况。

语法：

```python
class <Name>(<Base Class>):
    <suite>
```

比如：

```python
class CheckingAccount(Account):
    """A bank account that charges for withdrawals."""
    withdraw_fee = 1
    interest = 0.01

    def withdraw(self, amount):
        return Account.withdraw(self, amount + self.withdraw_fee)
```

这里：

- `Account` 是 base class 
- `CheckingAccount` 是 subclass 

`CheckingAccount` 是一种更特殊的 `Account`，所以它继承了 `Account` 的行为。

---

## 2. CheckingAccount 例子

```python
ch = CheckingAccount('Tom')
```

注意：`CheckingAccount` 里面没有定义 `__init__`，所以 Python 会去父类 `Account` 里面找 `__init__`。

因此：

```python
ch = CheckingAccount('Tom')
```

实际调用的是：

```python
Account.__init__(ch, 'Tom')
```

所以 `ch` 仍然会有：

```python
ch.holder = 'Tom'
ch.balance = 0
```

接着：

```python
ch.interest
```

会先找 `ch` 自己，没有；再找 `CheckingAccount`，找到了：

```python
interest = 0.01
```

所以结果是 `0.01`。

```python
ch.deposit(20)
```

`CheckingAccount` 里面没有 `deposit`，所以去 `Account` 里面找 `deposit`。因此存款行为和普通账户一样。

```python
ch.withdraw(5)
```

`CheckingAccount` 自己定义了 `withdraw`，所以会使用子类自己的版本。它实际取出的是：

```python
amount + self.withdraw_fee
```

也就是：

```python
5 + 1
```

所以如果原本余额是 `20`，取款后余额是：

```python
20 - 6 = 14
```

---

## 3. 子类的属性不是复制父类的属性

> Base class attributes aren't copied into subclasses!

继承不是把 `Account` 里面的所有属性复制粘贴到 `CheckingAccount` 里面。

更准确的理解是：**查找名字的时候，如果子类找不到，就继续去父类找。**

先看 `interest`：

```python
CheckingAccount.interest
```

`CheckingAccount` 自己有：

```python
interest = 0.01
```

所以直接用 `CheckingAccount`的 `interest` 而不会用 `Account` 的 `interest = 0.02`。

再看 `deposit`：

```python
CheckingAccount.deposit
```

`CheckingAccount` 自己没有 `deposit`，但是`Account` 有，所以用的是`Account` 的：

```python
Account.deposit
```

但是 `deposit` 并没有真的被复制到 `CheckingAccount` 的字典里。


- 查找 `interest` 时，在 `CheckingAccount` 找到，所以停止。
- 查找 `deposit` 时，在 `CheckingAccount` 找不到，所以继续去 `Account` 找。
- 查找 `withdraw` 时，在 `CheckingAccount` 找到，所以不会用 `Account.withdraw`。

---

## 4. Overriding

如果子类中定义了和父类同名的 attribute，那么子类会覆盖父类的版本。

```python
class CheckingAccount(Account):
    interest = 0.01

    def withdraw(self, amount):
        return Account.withdraw(self, amount + self.withdraw_fee)
```

这里 `CheckingAccount.interest` 覆盖了 `Account.interest`。

`CheckingAccount.withdraw` 也覆盖了 `Account.withdraw`。

但是**覆盖不等于父类的版本消失了。**

父类版本仍然可以通过类名访问：

```python
Account.withdraw(self, amount + self.withdraw_fee)
```

也可以写成：

```python
super().withdraw(amount + self.withdraw_fee)
```

### 为什么这里写 `self.withdraw_fee`？

> Look up attributes on instances whenever possible.

这句话的意思不是说 `withdraw_fee` 一定在实例自己的属性里，而是说：**先从当前这个对象 `self` 开始找。**

先看原来的写法：

```python
class CheckingAccount(Account):
    withdraw_fee = 1

    def withdraw(self, amount):
        return Account.withdraw(self, amount + self.withdraw_fee)
```

当我们执行：

```python
ch = CheckingAccount('Tom')
ch.withdraw(5)
```

这时 `self` 就是 `ch`，所以：

```python
self.withdraw_fee
```

会从 `ch` 这个实例开始找：

```python
ch
    ↓
CheckingAccount
```

最后在 `CheckingAccount` 里面找到：

```python
withdraw_fee = 1
```

所以这个时候它和写 `CheckingAccount.withdraw_fee` 的结果一样。

但是差别出现在未来继续写子类的时候。比如：

```python
class PremiumCheckingAccount(CheckingAccount):
    withdraw_fee = 0.5
```

现在执行：

```python
p = PremiumCheckingAccount('Tom')
p.withdraw(5)
```

`PremiumCheckingAccount` 没有重新定义 `withdraw`，所以它会沿用 `CheckingAccount.withdraw`。但是注意：函数虽然是在 `CheckingAccount` 里定义的，调用时 `self` 却是 `p` 这个 `PremiumCheckingAccount` 实例。

所以在这行里：

```python
return Account.withdraw(self, amount + self.withdraw_fee)
```

`self.withdraw_fee` 的查找路径是：

```python
p
    ↓
PremiumCheckingAccount
    ↓
CheckingAccount
```

它会先在 `PremiumCheckingAccount` 找到：

```python
withdraw_fee = 0.5
```

于是手续费就是 `0.5`。

如果当初写死成：

```python
return Account.withdraw(self, amount + CheckingAccount.withdraw_fee)
```

那不管 `self` 实际上是谁，都会强行使用：

```python
CheckingAccount.withdraw_fee
```

也就是固定的 `1`。这样 `PremiumCheckingAccount` 即使定义了 `withdraw_fee = 0.5`，也不会生效。

所以这里写 `self.withdraw_fee` 的意义是：**让同一个 `withdraw` 方法可以根据实际调用它的对象，使用不同类中定义的手续费。**

---

## 5. Inheritance and Composition

![](附件/Pasted%20image%2020260714205742.png)

### 1. Inheritance: is-a

```python
CheckingAccount is an Account
```

所以：

```python
class CheckingAccount(Account):
    ...
```

### 2. Composition: has-a


```python
A bank has a collection of accounts
```

银行不是一种账户，银行是拥有很多账户。因此不要写：

```python
class Bank(Account):
    ...
```

而应该写成：

```python
class Bank:
    def __init__(self):
        self.accounts = []
```

!!! example "Bank has accounts"
    这个例子就是 composition 的典型写法：`Bank` 不是一种 `Account`，而是 `Bank` 里面有很多 `Account`。

    ```python
    class Bank:
        """A bank has accounts."""

        def __init__(self):
            self.accounts = []

        def open_account(self, holder, amount, kind=Account):
            account = kind(holder)
            account.deposit(amount)
            self.accounts.append(account)
            return account

        def pay_interest(self):
            for a in self.accounts:
                a.deposit(a.balance * a.interest)

        def too_big_to_fail(self):
            return len(self.accounts) > 1
    ```

    使用过程：

    ```python
    bank = Bank()
    john = bank.open_account('John', 10)
    jack = bank.open_account('Jack', 5, CheckingAccount)
    ```

    `open_account` 的最后一个参数 `kind=Account` 表示：默认创建普通 `Account`，但也可以传入别的账户类。

    所以：

    ```python
    john.interest
    # 0.02

    jack.interest
    # 0.01
    ```

    因为 `john` 是普通 `Account`，而 `jack` 是 `CheckingAccount`。

    执行：

    ```python
    bank.pay_interest()
    john.balance
    # 10.2
    ```

    `pay_interest` 会遍历银行保存的所有账户：

    ```python
    for a in self.accounts:
        a.deposit(a.balance * a.interest)
    ```

    这里的 `a.interest` 会根据每个账户自己的类型去查找。普通账户用 `Account.interest = 0.02`，checking account 用 `CheckingAccount.interest = 0.01`。

    最后：

    ```python
    bank.too_big_to_fail()
    # True
    ```

    因为 `bank.accounts` 里面已经有 `john` 和 `jack` 两个账户，所以长度大于 1。

!!! example "Inheritance and Attribute Lookup"
    ```python
    class A:
        z = -1

        def f(self, x):
            return B(x - 1)

    class B(A):
        n = 4

        def __init__(self, y):
            if y:
                self.z = self.f(y)
            else:
                self.z = C(y + 1)

    class C(B):
        def f(self, x):
            return x

    a = A()
    b = B(1)
    b.n = 5
    ```

    ### `C(2).n`

    ```python
    C(2).n
    # 4
    ```

    `C` 自己没有 `__init__`，所以会用父类 `B.__init__`。创建 `C(2)` 的时候，`self` 是一个 `C` 实例，`y = 2`，所以执行：

    ```python
    self.z = self.f(2)
    ```

    这里的 `self.f` 会从 `C` 开始找，找到 `C.f`，所以 `self.z = 2`。

    接着查 `C(2).n`：

    1. 这个 `C` 实例自己没有 `n`
    2. `C` 类里面也没有 `n`
    3. 去父类 `B` 找到 `n = 4`

    所以结果是 `4`。

    ### `a.z == C.z`

    ```python
    a.z == C.z
    # True
    ```

    `a` 是 `A` 的实例。`a` 自己没有 instance attribute `z`，所以 `a.z` 会去 `A` 里面找，得到：

    ```python
    A.z
    # -1
    ```

    `C.z` 是在类上查找属性：

    1. `C` 没有 `z`
    2. `B` 没有 `z`
    3. `A` 有 `z = -1`

    所以 `C.z` 也是 `-1`，因此 `a.z == C.z` 是 `True`。

    ### `a.z == b.z`

    ```python
    a.z == b.z
    # False
    ```

    `a.z` 是 `-1`。

    但是 `b = B(1)` 的时候，`B.__init__` 会执行：

    ```python
    self.z = self.f(1)
    ```

    这里 `self` 是 `b`，也就是一个 `B` 实例。`B` 没有自己的 `f`，所以用父类 `A.f`：

    ```python
    def f(self, x):
        return B(x - 1)
    ```

    因此：

    ```python
    b.z = B(0)
    ```

    所以 `b.z` 是一个 `B` 实例，不是整数 `-1`，因此 `a.z == b.z` 是 `False`。

    ### 哪一个会变成整数？

    现在看这个链条：

    ```python
    b.z
    b.z.z
    b.z.z.z
    b.z.z.z.z
    ```

    前面已经知道：

    ```python
    b.z = B(0)
    ```

    创建 `B(0)` 的时候，`y = 0`，所以走 `else`：

    ```python
    self.z = C(1)
    ```

    因此：

    ```python
    b.z.z = C(1)
    ```

    创建 `C(1)` 的时候，因为 `C` 没有自己的 `__init__`，所以还是用 `B.__init__`。这时 `self` 是一个 `C` 实例，`y = 1`，所以执行：

    ```python
    self.z = self.f(1)
    ```

    这一次 `self.f` 会找到 `C.f`：

    ```python
    def f(self, x):
        return x
    ```

    所以：

    ```python
    b.z.z.z = 1
    ```

    因此真正会 evaluate to an integer 的是：

    ```python
    b.z.z.z
    ```

    ### 关键点

    这个例子最重要的地方是：**方法在哪里定义，和 `self` 实际是谁，是两回事。**

    `B.__init__` 虽然定义在 `B` 里，但如果是 `C(2)` 或 `C(1)` 调用它，那么 `self` 就是 `C` 实例。于是里面写的 `self.f(...)` 会优先去 `C` 里面找 `f`，而不是固定使用 `A.f`。
    

---

## 6. Multiple Inheritance

Python 允许一个类继承多个父类。

```python
class SavingsAccount(Account):
    deposit_fee = 2

    def deposit(self, amount):
        return Account.deposit(self, amount - self.deposit_fee)
```

`SavingsAccount` 的特点是：每次存款收 2 块钱手续费。

然后定义一个营销账户：

```python
class AsSeenOnTVAccount(CheckingAccount, SavingsAccount):
    def __init__(self, account_holder):
        self.holder = account_holder
        self.balance = 1
```

它同时想要：

- checking account 的低利率 `0.01`
- checking account 的取款手续费 `$1`
- savings account 的存款手续费 `$2`
- 开户赠送 `$1`

### 查找顺序

```python
class AsSeenOnTVAccount(CheckingAccount, SavingsAccount):
    ...
```

括号里的顺序很重要。大致查找顺序是：

```python
AsSeenOnTVAccount
    ↓
CheckingAccount
    ↓
SavingsAccount
    ↓
Account
```

所以：

```python
such_a_deal = AsSeenOnTVAccount('John')
```

执行自己的 `__init__`，余额一开始是：

```python
1
```

执行：

```python
such_a_deal.deposit(20)
```

查找 `deposit`：

1. `AsSeenOnTVAccount` 没有
2. `CheckingAccount` 没有
3. `SavingsAccount` 有

所以使用 `SavingsAccount.deposit`：

```python
return Account.deposit(self, amount - self.deposit_fee)
```

`self.deposit_fee` 会找到 `SavingsAccount.deposit_fee = 2`，所以真正存入：

```python
20 - 2 = 18
```

原来余额是 `1`，所以存完以后余额是：

```python
19
```

因此：

```python
such_a_deal.deposit(20)
# 19
```

再执行：

```python
such_a_deal.withdraw(5)
```

查找 `withdraw`：

1. `AsSeenOnTVAccount` 没有
2. `CheckingAccount` 有

所以使用 `CheckingAccount.withdraw`：

```python
return Account.withdraw(self, amount + self.withdraw_fee)
```

`self.withdraw_fee` 找到 `CheckingAccount.withdraw_fee = 1`，所以实际取出：

```python
5 + 1 = 6
```

存款后余额是 `19`，取出 `6`，剩下：

```python
13
```

因此：

```python
such_a_deal.withdraw(5)
# 13
```

![](附件/Pasted%20image%2020260714224031.png)

---

## 7. Resolving Ambiguous Class Attribute Names

多重继承最容易出问题的地方是：多个父类里有同名属性时，到底用谁的？

Python 会按照类的继承顺序去解析。对于：

```python
class AsSeenOnTVAccount(CheckingAccount, SavingsAccount):
    ...
```

它会先看 `CheckingAccount`，再看 `SavingsAccount`。

所以如果 `CheckingAccount` 和 `SavingsAccount` 都定义了同名属性，那么一般会优先使用 `CheckingAccount` 里面的版本。

### 我的理解

多重继承可以很强，但是也容易让查找路径变得不直观。

如果一个类需要同时继承多个父类，就要特别清楚：

1. 哪些方法来自第一个父类。
2. 哪些方法来自第二个父类。
3. 同名属性冲突时，谁会先被找到。

否则代码会变成“能跑，但是人不知道为什么能跑”。

---

## 8. 总结

Lecture 20 的主线其实是：

1. 先复习 attribute lookup：实例先找自己，再找类。
2. 再讲 attribute assignment：赋值不会向上查找，只会改点号左边那个对象。
3. 然后把 lookup 链条扩展到 inheritance：子类找不到，就去父类找。
4. 最后讲 multiple inheritance：多个父类时，查找顺序由继承列表决定。

最核心的一句话是：

```text
赋值看点号左边是谁；查找看实例、类、父类这条链。
```
