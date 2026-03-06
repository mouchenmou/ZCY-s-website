## 1. Generators and Generator Functions
The thing that's special about a generator is that it's returned from a generator function. A generator function looks like a regular function. The difference is that it uses the yield keyword instead of return in order to return values.


![](附件/截屏2026-02-22%2019.33.46.png)
相当于是执行了`t = plus_minus(x)` 这段代码之后t就已经是一个可以iter了，可以直接执行`next(t)`。

---

## 2. Generators & Iterators
Generator函数返回的是一个Generator，但是在执行的过程中它们经常处理迭代器。

!!! explanation "Generators can `Yield from` Iterators"
	    ![](附件/截屏2026-02-22%2020.51.47.png)
	    
	    ### 不使用`yield from`:
	    ```Python
	        def countdown(k):
	            if k > 0:
	                 yield k
	                 for x in countdown(k-1):
	                     yield x
	        
	    ```
	    
	    ---
	    
	    ###使用`yield from`
	    ```Python
	        def countdown(x):
	            if x>0:
	                yield x
	                yield from countdown(x-1)
	    ```

!!! explanation "Yield partition"
    partitions之前在递归的时候讲过，现在将用`yield from`来实现
    ![](附件/截屏2026-02-23%2019.34.34.png)
    