##  Quiz
![](附件/截屏2026-03-03%2009.21.30.png)
左边的答案是Yes，右边的答案是No，可以在 **[Visualizer](http://cscircles.cemc.uwaterloo.ca/java_visualize/#code=public+class+PollQuestions+%7B%0A+++public+static+void+main\(String%5B%5D+args\)+%7B%0A++++++Walrus+a+%3D+new+Walrus\(1000,+8.3\)%3B%0A++++++Walrus+b%3B%0A++++++b+%3D+a%3B%0A++++++b.weight+%3D+5%3B%0A++++++System.out.println\(a\)%3B%0A++++++System.out.println\(b\)%3B++++++%0A%0A++++++int+x+%3D+5%3B%0A++++++int+y%3B%0A++++++y+%3D+x%3B%0A++++++x+%3D+2%3B%0A++++++System.out.println\(%22x+is%3A+%22+%2B+x\)%3B%0A++++++System.out.println\(%22y+is%3A+%22+%2B+y\)%3B++++++%0A+++%7D%0A+++%0A+++public+static+class+Walrus+%7B%0A++++++public+int+weight%3B%0A++++++public+double+tuskSize%3B%0A++++++%0A++++++public+Walrus\(int+w,+double+ts\)+%7B%0A+++++++++weight+%3D+w%3B%0A+++++++++tuskSize+%3D+ts%3B%0A++++++%7D%0A%0A++++++public+String+toString\(\)+%7B%0A+++++++++return+String.format\(%22weight%3A+%25d,+tusk+size%3A+%25.2f%22,+weight,+tuskSize\)%3B%0A++++++%7D%0A+++%7D%0A%7D&mode=edit)** 处查看它们是如何一步步执行的。下面解释这两个答案的由来。

There are 8 primitive types in Java:

- byte, short, int, long, float, double, boolean, char

Everything else, including arrays, is a reference type.

由于Walrus不属于这8个primitive中的任何一个，所以它是一个reference type，因此传递的是引用。而int属于这8个之一，因此是复制值而不是地址。

---
## Parameter Passing
### 例题：Does the call to doStuff(walrus, x) have an affect on walrus and/or main's x?
#### A. Never will change.
#### B. walrus will lose 100 lbs, but main's x will not change.
#### C. walrus will not change, but main's x will decrease by 5.
#### D. Both will decrease.
```Java
    public static void main(String[], args){
        Walrus walrus = new Walrus(3500, 10.5);
        int x = 9;
        doStuff(walrus, x);
        System.out.println(walrus);
        System.out.println(x);
    }
    public static void dostuff(Walrus w, int x){
        w.weight=w.weight - 100
        x = x - 5
    }
```
![](附件/截屏2026-03-03%2009.43.57.png)