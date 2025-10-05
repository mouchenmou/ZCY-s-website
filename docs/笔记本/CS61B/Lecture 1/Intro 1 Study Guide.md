![](附件/Pasted%20image%2020251005175752.png)
```java
    public class HelloWorld{
        public static void main(Stringp[] args){
            System.out.println("HelloWorld");
        }
    }
```
![](附件/Pasted%20image%2020251005180025.png)
![](附件/Pasted%20image%2020251005180117.png)
![](附件/Pasted%20image%2020251005180155.png)
```java
public class LargerDemo{
    public static int larger(int x, int y){
        if(x>y){
            return x;
        }
        return y;
    }
    public static void main(String[] args){
        System.out.println(larger(5,-10));
    }
}
```
![](附件/Pasted%20image%2020251005180700.png)
Java is statically typed!

- All variables, parameters, and methods must have a declared type.
- That type can never change.
- Expressions also have a type, e.g. "larger(5, 10) + 3" has type int.
- The complier checks that all the types in your program are compatible **before the program ever runs!**
    - e.g. String x = larger(5, 10) + 3 will fail to compile.
    - This is unlike a language like python, where type checks are performed during execution.