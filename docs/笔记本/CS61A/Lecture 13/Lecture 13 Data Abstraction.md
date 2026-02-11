## 1.Violating Abstraction barriers
![[附件/截屏2026-02-08 17.48.12.png]]
这段代码编译并不会出错，但是它违反了数据抽象。他直接默认有理数就是用列表来错的，所以直接手写了`[1, 2]`。如果有一天我们决定修改有理数的定义方式，不如我们不再用list而是用dictionary来存，那么这段代码就会报错。

正确的做法应该是通过Constructor（构造函数）和Selector（选择函数）来操作数据，而不是直接操作`[0], [1]`这种底层细节。

### Representing Rational Numbers的正确实现方法：
```Python
    def rational(x, n):
        return [x, n] //这里的x和n不一定是整数，有可能是字典列表这些
        
    def numer(x):
        return x[0]
        
    def denom(x):
        return x[1]
```

---

## 2. Representing Pairs Using Lists
![[附件/截屏2026-02-08 18.01.16.png]]

---

## 3. Reducing to Lowest Terms
![[附件/截屏2026-02-08 18.10.17.png]]

```Python
    from fractions import gcd
    def rational(n, d):
        g = gcd(n, d)
        return [n//g, d//g]
        
```