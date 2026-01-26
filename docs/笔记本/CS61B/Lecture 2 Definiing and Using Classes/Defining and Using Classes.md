## 编译方式

![](Pasted%20image%2020251005182453.png)

---
## 1. 以Dog为例来说明Class



```Java
    public class Dog{
        public static void makeNoise(){
            System.out.println("Bark!");
        }
        
        public static void main(String[] args){
            makeNoise();
        }
    }
```


- Every method (a.k.a.function) is associated with some class.
- To run a class, we must define a main method.
    - Not all class have a main method!（下面给出一个例子）

```Java
    # Dog这个类不能直接被运行（没有main函数）
    public class Dog{
        public static void makeNoise(){
            System.out.println("Bark!");
        }
    }
    
    # DogLauncher这个类调用了Dog这个类中的函数
    public class DogLauncher{
        public static void main(Stringp[] args){
            Dog.makeNoise;
        }
    }
```

---

## 2. 给狗进行分类

```Java
    public class Dog{
        public int weightInPounds;
         
        public void makeNoise(){
            if(weightInPounds < 10){
                System.out.println("Yip!");
            }
            else if(weightInPounds <30){
                System.out.println("Bark!");
            }
            else{
                System.out.println("woooof!");
            }
        }
    }
    
    public class DogLauncher{
        public static void main(String[] args){
            Dog d = new Dog();
            d.weightInPounds = 25;
            d.makeNoise();
        }
    }
```

### 在oop的思想中，我们不习惯创建了d之后再对它进行赋值，而是直接在构造的时候就实现赋值，那么我们可以将代码改成如下形式：
```Java
    public class Dog{
        //Instance variable. Can have as many of these as you want.
        public int weightInPounds;
        
        //Constructor (Similar to a method, but not a method).Determines how to instantiate the class.
        public Dog(int w){
            weightInPounds = w;
        }
        
        //Non-static method
        public void makeNoise(){
            if(weightInPounds < 10){
                System.out.println("Yip!");
            }
            else if(weightInPounds <30){
                System.out.println("Bark!");
            }
            else{
                System.out.println("woooof!");
            }
        }
    }
    
    public class DogLauncher{
        public static void main(String[] args){
            Dog d = new Dog(51);
            d.makeNoise();
        }
    }
```

---

## 3. To create an array of objects
### 仍然以dog为例
- First use the new keyword to create the array.
- Then use new again for each object that you want to put in the array.

```Java
    Dogp[] dogs = new Dog[2];
    dogs[0] = new Dog(8);
    dogs[1] = new Dog(20);
    dogs[0].makeNoise();
```

---

## 4. Static vs. Instance Methods(静态方法和实例方法)
### Key difference between static and non-static(a.k.a instance) method:
- Static methods are invoked using the class name, 比如`Dog.makeNoise();`
- Instance methods are invoked using an instance name, 比如`maya.makeNoise();`
- Static methods can't access "my" instance variables, because there is no "me" .
    - 以下图为例，static并没有狗，只有狗叫；而Non-static有不同的狗

![](附件/Pasted%20image%2020260126231549.png)




!!! explanation "### 既然Non-static这么好，我们还要static做什么？"
    答案是图方便
    ![](附件/Pasted%20image%2020260126232520.png)

!!! explanation "### A class may have a mix of static and non-static members."
    - A variable or method defined in a class is also called a member of that class.
    - Static members are accessed using class name, 以下图为例，可以执行`Dog.binomen;`
    - Non-static members cannot be invoked using class name, 以下图为例，==不可以执行Dog.makeNoise();==
    - Static methods must access instance variables via a specific instance
    ![](附件/Pasted%20image%2020260126232959.png)
    这幅图的Dog种就既包含了静态类型的函数(`maxDog`)也包含了非静态类型的函数。
    当然我们也可以写一个非静态版本的比大小函数。
    ```java
        public Dog maxDog(Dog d2){
            if(weightInPounds > d2.weightInPounds){
                return this;
            }
            return d2;
        }
    ```
        
        为了清楚可见，也可以在weightlnPounds前加一个this
    ```java
        public Dog maxDog(Dog d2){
            if(this.weightInPound > d2.weightInPounds){
                return this;
            }
            return d2;
        }
    ```
    
    此时的主函数也得修改，改成以下形式：
    ```java
        public class DogLauncher{
            public static void main(String[], args){
                Dog d = new Dog(15);
                Dog d2 = new Dog(100);
            
                //Dog bigger = Dog.maxDog(d, d2);
                Dog bigger = d.maxDog(d, d2);
                bigger.makeNoise();
                d.makeNoise;
            }
        }
    ```

---

## 5. Managing Complexity with Classes and Static Methods

老师用面向对象的方式设计了一个数组中输出比相邻四个数更大的那个。