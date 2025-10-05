## 以dog为示例来讲解一下
![](附件/Pasted%20image%2020251005182453.png)![](附件/Pasted%20image%2020251005182539.png)
![](附件/Pasted%20image%2020251005182554.png)
![](附件/Pasted%20image%2020251005182608.png)
![](附件/Pasted%20image%2020251005182623.png)
![](附件/Pasted%20image%2020251005182707.png)
![](附件/Pasted%20image%2020251005182722.png)

---

## Static VS Non-static
### static
```java
    public static void makeNoise(){
        System.out.println("Bark!");
    }
```
The method cannot access weightInpound!
Invocation: 
```java
    Dog.makeNoise();
```

### Non-static
```java
    public void makeNoise(){
        if (weightInPound < 10){
            System.out.println("yipyipyip");
        }else if(weightInPound < 30){
            System.out.println("barkbark");
        }else{
            System.out.println("woof!");
        }
    }
```
Invocation:
```java
    maya = new Dog(100);
    maya.makeNoise();
```
![](附件/Pasted%20image%2020251005183340.png)
![](附件/Pasted%20image%2020251005183428.png)

1.在类中定义的变量或方法也称为该类的成员。

2.静态成员使用类名访问，例如Dog.binomen。

3.非静态成员不能使用类名调用：Dog.makeNoise（）是错误的调用方式

4.静态方法必须通过特定实例访问实例变量，例如d1。
![](附件/Pasted%20image%2020251005183705.png)
当然也可以制作一个非静态版本的比大小：
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
![](附件/Pasted%20image%2020251005184714.png)
Answer to Question：Won’t go over in live lecture. Use the visualizer to see the solution 
[at this link](http://goo.gl/HLzN6s).

- Or if you’re watching this video and can’t find the slides, the link is: [http://goo.gl/HLzN6s](http://goo.gl/HLzN6s)
