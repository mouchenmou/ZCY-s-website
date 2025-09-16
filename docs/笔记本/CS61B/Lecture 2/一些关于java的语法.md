## 以dog为示例来讲解一下
![](Lecture2%20photo/Pasted image 20250914122238.png)![](Lecture2%20photo/Pasted image 20250914122251.png)
![](Lecture2%20photo/Pasted image 20250914122304.png)
![](Lecture2%20photo/Pasted image 20250914122320.png)
![](Lecture2%20photo/Pasted image 20250914122329.png)
![](Lecture2%20photo/Pasted image 20250914122342.png)
![](Lecture2%20photo/Pasted image 20250914122352.png)
## Static VS Non-static

### static

```java
public static void makeNoise(){
    System.out.println("Bark!");
}
```

The method cannot access weightInpound!
Invocation: 

```
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

```
maya = new Dog(100);
maya.makeNoise();
```

![](Lecture2%20photo/Pasted image 20250914124218.png)
![](Lecture2%20photo/Pasted image 20250914124326.png)
1.在类中定义的变量或方法也称为该类的成员。

2.静态成员使用类名访问，例如Dog.binomen。

3.非静态成员不能使用类名调用：Dog.makeNoise（）

是错误的调用方式

4.静态方法必须通过特定实例访问实例变量，例如d1。
![](Lecture2%20photo/Pasted image 20250914130059.png)
当然也可以制作一个非静态版本的比大小
![](Lecture2%20photo/Pasted image 20250914131728.png)
为了清楚可见，也可以在weightlnPounds前加一个this
![](Lecture2%20photo/Pasted image 20250914131817.png)
此时的主函数也得修改，改成以下形式：
![](Lecture2%20photo/Pasted image 20250914131839.png)
![](Lecture2%20photo/Pasted image 20250914131859.png)
Answer to Question：Won’t go over in live lecture. Use the visualizer to see the solution 
[at this link](http://goo.gl/HLzN6s).

- Or if you’re watching this video and can’t find the slides, the link is: [http://goo.gl/HLzN6s](http://goo.gl/HLzN6s)
