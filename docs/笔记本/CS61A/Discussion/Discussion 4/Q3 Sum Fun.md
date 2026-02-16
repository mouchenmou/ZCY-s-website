Implement `sums(n, m)`, which takes a total `n` and maximum `m`. It returns a list of all lists:

1. that sum to `n`,
2. that contain only positive numbers up to `m`, and
3. in which no two adjacent numbers are the same.

Two lists with the same numbers in a different order should both be returned.

Here's a recursive approach that matches the template below: build up the `result` list by building all lists that sum to `n` and start with `k`, for each `k` from 1 to `m`. For example, the result of `sums(5, 3)` is made up of three lists:

- `[[1, 3, 1]]` starts with 1,
- `[[2, 1, 2], [2, 3]]` start with 2, and
- `[[3, 2]]` starts with 3.

**Hint:** Use `[k] + s` for a number `k` and list `s` to build a list that starts with `k` and then has all the elements of `s`.
```Python
    >>> k = 2 
    >>> s = [4, 3, 1] 
    >>> [k] + s 
    [2, 4, 3, 1]
```


```Python
def sums(n, m):
"""Return lists that sum to n containing positive numbers up to m that
have no adjacent repeats.

>>> sums(5, 1)
[]
>>> sums(5, 2)
[[2, 1, 2]]
>>> sums(5, 3)
[[1, 3, 1], [2, 1, 2], [2, 3], [3, 2]]
>>> sums(5, 5)
[[1, 3, 1], [1, 4], [2, 1, 2], [2, 3], [3, 2], [4, 1], [5]]
>>> sums(6, 3)
[[1, 2, 1, 2], [1, 2, 3], [1, 3, 2], [2, 1, 2, 1], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
"""
if n < 0:
    return []
if n == 0:
    sums_to_zero = [] # The only way to sum to zero using positives
    return [sums_to_zero] # Return a list of all the ways to sum to zero
result = []
for k in range(1, m + 1):
    result = result + [ ___ for rest in ___ if rest == [] or ___ ]
return result

```

这道题我没做出来，我一开始的想法就错了，我一直认为第一空应该是`[k]+sums(n-k, m)`，就一直卡在这个点上最后失败了。

题目的答案是`result = result + [ [k]+rest for rest in sums(n-k, m) if rest==[] or rest[0]!=k]`。

- `rest==[]`对应了我们在递归中刚好凑够了数，即`n==0`时
- `rest[0] != k`是为了防止相同的两个数相邻
