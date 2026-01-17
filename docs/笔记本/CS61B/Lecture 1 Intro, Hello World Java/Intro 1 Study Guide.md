## Python 与 Java 的对比:

## Case 1:

### Python:
```python
    print("hello world")
```
### Java：
```java
    public class HelloWorld{
        public static void main(String[] args){
            System.out.println("HelloWorld");
        }
    }
    /**
    1. All code in Java must be part of a class.
    2. We delimit the beginning and end of segments of code with {}
    3. All statements in Java must end in a semi-colon（分号）.
    4. For code to run, we need public static void main(String[] args) 
    /*
```

---

## Case 2:
### Python:
```python
    x=0
    while(x < 10):
        print(x)
        x = x + 1
```

### Java:
```java
    public class print {
        public static void main(String[], args){
            int x=0; //declare x exists and is an integer
            while(x<10){
                System.out.println(x)
                x = x+ 1
            }
        }
    }
    /**
    1. Before Java variables can be used, they must be declared.
    2. Java variables must have a specific type.
    3. Java variable types can never change.
    /*
```

---

## Case 3:

### Python:
```python
    def larger(x, y):
        if(x>y):
            return x
        else:
            return y
            
    print(larger(5, 10))
```


### Java:
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


- All variables, parameters, and methods must have a declared type.
- That type can never change.
- Expressions also have a type, e.g. "larger(5, 10) + 3" has type int.
- The complier checks that all the types in your program are compatible **before the program ever runs!**
    - e.g. String x = larger(5, 10) + 3 will fail to compile.
    - This is unlike a language like python, where type checks are performed during execution.