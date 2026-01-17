```python
    def digit(n, k):
    """Return the digit that is k from the right of n for positive integers n and k.
    >>> digit(3579, 2)
    >>> 5
    
    >>> digit(3579, 0)
    >>> 9
    
    >>> digit(3579, 10)
    >>> 0
    
    """
    return ___
```
这道题用到了和c语言中 `a>b? a:b` 这一语法类似的语法结构
答案是 ` (n//pow(10, k))%10 if(n>=pow(10, k)) else 0`
