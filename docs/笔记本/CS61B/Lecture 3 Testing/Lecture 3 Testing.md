这节课主要是用几个例子讲了一下testing的好处，就不记录了。

!!! warning "Java中的数组不能像python那样slicing"
    ## 在`Java`中通过递归来实现冒泡排序
    
    ### 错误写法：
    ```Java
        public static void sort(String[] x){
            int smallestIndex = findSmallest(x);
            swap(x, 0, smallestIndex); 
            sort(x[1:]); ## 这一步是错误的，因为Java中不能进行slicing
            //findSmallest函数和SmallestIndex函数都是已经实现好的，不用管
        }
    ```
    
    ---
    ### 正确写法：
    ```Java
        public static void sort(String[] x){
            sort(x, 0);
        }
        public static void sort(String[] x, int a, int b){
            int smallestIndex = findSmallest(x);
            swap(x, start, smallestIndex);
            sort(x, strat+1);
        }
    ```
    