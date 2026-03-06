## 1. Views of a dictionary
![](附件/截屏2026-02-19%2022.08.10.png)

---

## 2. Built-in Functions for Iteration
### 下图中的这几个函数都是`iter`函数，需要配合`next`函数使用。
![](附件/截屏2026-02-19%2022.37.17.png)

!!! example
    ![](附件/截屏2026-02-19%2022.45.23.png)

---

## 3. The zip function
![](附件/截屏2026-02-19%2022.48.33.png)

!!! warning "reversed函数返回的不是列表！！"
    ```Python
        a=[1, 2, 3, 4, 3, 2, 1]
        reversed(a) == a ##这里会返回False
        //reversed(a)返回的不是一个列表
        
        list(reversed(a)) == a ##这样子才会返回True
    ```

!!! confused "我关于zip函数的一个疑惑"
    ![](附件/截屏2026-02-19%2023.05.23.png)
    ### 疑惑：
    明明reversed函数返回的值不是列表，为什么zip函数还能将s和reversed(s)两个不同性质的容器的内容进行处理。
    
    ---
    
    ### 解答：
    ![](附件/截屏2026-02-19%2023.10.26.png)
    
    ---
    
    ### 用途:
    有了zip函数，我们就能够很轻松的判断一个容器中的序列是否为回文序列：
    ```Python
        def palindrome(s):
            """return whether s is the same backward and forward.
            >>> palindrome([3, 1, 4, 1, 5])
            False
            
            >>> palind('seveneves')
            True
            """
            return all([a==b for a and b in zip(s, reversed(s))])
    ```
    
    ### 在这段代码中我也遇到了两个疑惑：
    
    1. all函数是干嘛的？
    2. 为什么all函数里面的表达式要用[]包裹起来？
    
    ### 解答：
    ![](附件/截屏2026-02-19%2023.19.48.png)



