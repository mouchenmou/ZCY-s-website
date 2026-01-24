## 1. Sum Digits Without a While Statement
```Python
    def split(n);
        return n//10, n%10
    
    def sum_digits(n):
        if(n<10):
            return n
        else:
            all_but_last, last=split(n)
            return sum_digits(all_but_last) + last
```

---
## 2. Recursion in Environment Diagram

![](附件/Pasted%20image%2020260124010516.png)

---
## 3. Iteration VS. Recursion

Iteration is a special case of recursion.

$4! = 4×3×2×1 = 24$

### Using While :
```python
    def fact_iter(n):
        k=1, total=1
        while(k<=n):
            total=total*k
            k=k+1
        return total
```

### Using recursion
```python
    def fact(n):
        if(n==0):
            return 1
        else return n*fact(n-1)
```

---

## 4. The Luhn Algorithm
### This algorithm is used to verify credit card number

1. From the rightmost digit, which is the check digit, moving left, double the value of every second digit; if product of this doubling operation is greater than 9（比如$7*2=14>9$）, then sum the digits of the products（$10: 1+0=1$, $14: 1+4=5$）
2. Take the sum of all the digits.
![](附件/Pasted%20image%2020260124011949.png)
### 实现方式：互相调用递归
#### 代码1（自己写的，目前没有出现任何报错）
```Python
    def split(n):
        return n//10, n%10
    
    def sum_digits(n):
        if(n<10):
            return n
        else:
            all_but_last, last=split(n)
            return sum_digits(all_but_last) + last
            
    def handle_number(n):
        if(n<10):
            return n
        else: 
            all_but_last, last=split(n)
            return handle_second_digit(all_but_last)+last
            
    def handle_second_digit(n):
        all_but_last, last=split(n)
        if (last<5):
            return handle_number(all_but_last)+2*last
        else:
            last=last*2
            return handle_number(all_but_last)+last//10+last%10
```

#### 代码二（课上讲的）
```Python
    def split(n):
        return n//10, n%10
    
    def sum_digits(n):
        if(n<10):
            return n
        else:
            all_but_last, last=split(n)
            return sum_digits(all_but_last) + last
            
    def handle_number(n):
        if(n<10):
            return n
        else: 
            all_but_last, last=split(n)
            return handle_second_digit(all_but_last)+last
            
    def handle_second_digit(n):
        all_but_last, last=split(n)
        last_update=sum_digits(last)
        if(n<10):
            return last_update
        else:
            return handle_number(all_but_last)+last_update 

```
