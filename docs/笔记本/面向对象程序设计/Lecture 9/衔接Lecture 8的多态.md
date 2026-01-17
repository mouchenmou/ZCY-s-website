## Virtual and reference arguments
```c++
    void func(Ellipse& elly){
        elly.render;
    }
    
    Circle circ(60);
    func(circ);
```
运行完这段代码之后输出的是 `"circle render is called"`。
### 1. 核心机制：引用与向上转型
当执行 `func(circ)` 时：
- 派生类对象 `circ` 被传递给基类引用参数 `Ellipse& elly`。
- 这是一种合法的 Upcasting。与对象切片不同，引用不会复制对象数据，也不会导致对象切片。

### 2. 运行时行为与多态性
- 在函数 `func` 内部，`elly` 引用了完整的 `Circle` 对象。
- 当执行 `elly.render()` 时，`render()` 是一个虚函数，触发动态绑定。
    
- 查找过程： C++ 运行时系统会检查 `elly` 实际引用的对象的 VPtr。由于 `elly` 引用的是一个 `Circle` 对象，VPtr 指向 `Circle` 的 VTable。
- 结果：最终执行的是 `Circle::render()`。

### 如果把上面那段代码中的引用去掉，那就会是另一种结果

```c++
    void func(Ellipse elly){
        elly.render;
    }
    
    Circle circ(60);
    func(circ);
```
运行完这段代码之后输出的是 `"Ellipse render is called"`。
没有引用，函数会发生一次拷贝，也就是会执行`elly = circ`，上节课我们说过了，拷贝时，vptr是不会发生改动的，也就是说`elly`的`vptr`仍然是`Ellipse`的`vptr`，故最终执行的是`Ellipse::render()`。

---
## Overriding
Overriding redefines the body of a virtual function.

- 覆盖是指在派生类中，重新定义一个基类中已声明为虚函数的函数体。
```c++
    class Base {
    public:
        virtual void func();
    };
    class Derived : public Base {
    public:
        void func() override;
        //overrides Base::func()
    }
```

!!! conception "### Calls up the chain"
    You can still call the overridden function for reuse:
    
    - 即使派生类已经覆盖了基类的虚函数，你仍然可以显式地调用基类的实现代码，以实现代码复用。
    ```c++
        void Derived::func(){
            cout << "In Derived::func!";
            Base::func(); //call to base class
        }
    ```
    
    - This is a common way to add new functionality
        - 这是一种非常常见且推荐的模式。派生类可以在执行自己的特定逻辑（如 `cout << "In Derived::func!";`）之前或之后，调用基类的实现来完成“通用”或“初始化”任务。这使得派生类可以安全地在基类功能之上添加新的功能。
        
    - No need to copy the old stuff
        - 通过调用基类的函数，派生类避免了重新编写或复制基类中已经存在的逻辑代码。这符合面向对象设计中“DRY”（Don't Repeat Yourself）的原则。

!!! conception "### Return types relaxation(current)"
    - Suppose D is publicly derived from B
    - D::f() can return a subclass of the return type
        - 假设类 D 公开继承自类 B。如果 `B::f()` 是一个虚函数，并且它返回一个B类型的指针或引用，那么 D::f() 在覆盖它时，可以返回一个D类型的指针或引用。
        - defined in B::f()
    - Applies to pointer and reference types
        - 这种relaxation只适用于返回指针或引用类型。它不能应用于返回对象本身的值类型（例如，如果 B::f() 返回B类型的一个实体，那么D::f() 不能返回D类型的一个实体）。
        - -e.g. D&, D*
    - In most compliers now

!!! example
    ```c++
        class Expr{
        public:
            virtual Expr* newExpr();
            virtual Expr& clone();
            virtual Expr self();
        };
        
        class BinaryExpr: public Expr{
        public:
            virtual BinaryExpr* newExpr(); //ok
            virtual BinaryExpr& clone(); //ok
            virtual BinaryExpr self(); //Error!!!
        }
    ```

---

## Overriding and virtual
- Overrding adds multiple signatures
    - 重载允许在同一个作用域内（例如同一个类中），定义多个同名函数，但是它们的参数列表必须不同。
    - ```c++
          class Base{
          public:
              virtual void func();
              virtual void func(circ);
          };
      ```
    
- If you override an overloaded function, you must override all of the varients!
    - 如果在派生类中覆盖一个基类中被重载的虚函数集中的任何一个，那么我们应该把所有重载版本都覆盖。
    - 如果不覆盖所有版本，就会发生函数隐藏。
        - 如果我们只覆盖了void func()，但没有覆盖void func(int)，那么派生类对象调用derived.func(5)时，编译器会报错，即使基类中有这个函数，也被派生类的给隐藏了。



!!! example "When you override an overloaded function, override all of the variants"
    ```c++
        class Derived: public Base{
        public:
            virtual void func(){
                Base::func();
            }
            virtual void func(int) {
                ...自定义
            };
        }
    ```

!!! Tips "### Tips"
    1. Never redefine an inherited non-virtual function.（不要重新定义继承来的非虚函数）
        - 原因：`Non-virtuals are statically bound`。（非虚函数是通过静态绑定机制调用的）
        - 后果：No dynamic dispatch
            - 静态绑定是在编译时确定的，它只看你声明的变量或指针的静态类型，而不会看对象的实际类型。如果你重新定义了一个非虚函数，通过基类指针调用该函数时，永远只会调用基类的版本，完全无法实现多态。也就是说我们在lecture中实现的内容实际上是不好的东西，不要采取。
    2. Never redefine an inherited default parameter value.（不要重新定义继承来的默认参数）
        - 默认参数也是通过静态绑定机制确定的。
        - 当你通过基类指针调用虚函数，而该虚函数使用了默认参数时，虚函数本身会动态调用派生类的代码，但默认参数的值却会静态地取自基类。
    
    !!! warning "用一个例子解释为什么不要重新定义默认参数"
        ```c++
            #include<iostream>
            using namespace std;
            
            class Base {
            public:
                void nonVirtualFunc(){
                    cout << "Base::nonVirtualFunc called" << endl;
                 }
                 virtual void virtualFunc(int x = 10){
                     cout << "Base::x = " << x << endl;
                 }
            };
            
            class Derived : public Base {
            public:
                 void nonVirtualFunc(){
                     cout << "Derived::nonVirtualFunc" << endl;
                 }
                 void virtualFunc(int x = 20) override {
                     cout <<"Derived::x = " << x << endl;
                 }
            };
            
            int main(){
                Derived d;
                Base &b = d;
                
                b.nonVirtualFunc();
                b.virtualFunc();
                
                d.nonVirtualFunc();
                d.virtualFunc();
                return 0;
            }
        ```
        ![](附件/Pasted%20image%2020251124225744.png)
        需要关注的点是，b.virtualFunc()调用的是derived的virtualFunc函数，但是参数传递的确是Base中的。这是因为，在调用函数之前，参数就已经完成了初始化。

---

## Virtual in Ctors?
构造函数不需要用虚函数。

虚函数机制是用来实现运行多态的。

---
## Abstract classes

抽象类：一个类中所有的函数全都是虚函数。

- Why use them ？
    - Modeling
    - Force correct behavior
    - Define interface without defining an implementation
- When to use them ?
    - 还没有足够的信息时，只能先构造抽象类。
    - 只为了设计接口用来继承。

---
## Protocol/Interface classes

### 1. **抽象基类要求**
- **所有非静态成员函数都是纯虚函数 (All non-static member functions are pure virtual)**：
    - 纯虚函数是只声明不定义的。
    - 除了析构函数之外，类中定义的每一个方法都必须是纯虚函数（例如 `virtual void func() = 0;`）。
    - 这确保了派生类必须实现接口中定义的所有行为。
    - ![](附件/Pasted%20image%2020251125001300.png)

### 2. **虚析构函数要求**
- **带空体的虚析构函数 (Virtual destructor with empty body)**：
    - 这是为了解决我们前面讨论的多态删除问题。如果通过基类指针删除派生类对象，虚析构函数能保证调用正确的析构函数链。
    - 它通常需要一个空实现（`{}`），因为析构函数不能是纯虚函数（虽然技术上可以，但会带来问题，所以通常推荐提供定义）。

### 3. **数据成员限制**
- **没有非静态成员变量 (No non-static member variables)**：
    - 一个纯粹的接口不应该拥有状态（数据）。它的作用只是定义行为（函数）。
    - 因此，它不能有任何非静态的成员变量，无论是继承来的还是自己定义的。
- **可以包含静态成员 (May contain static members)**：
    - 静态成员不属于任何特定的对象实例，它们是类级别的，因此它们通常被允许用于存储常量或工厂方法等。

---

## Reference as class members
### 主要规则
- **声明时不能给出初始值 (Declared without initial value)**
    - 你不能在类内部声明时直接给引用成员一个初始值，因为这个值可能在对象创建时才确定。
    - 例如，你不能写成 `int& m_y = some_variable;`
        - ![](附件/Pasted%20image%2020251125005859.png)
- **必须使用构造函数初始化列表初始化 (Must be initialized using constructor initializer list)**
    - 这是最核心的规则。由于引用必须在创建时立即绑定到一个有效的对象，因此你不能等到构造函数的函数体内部赋值。你必须使用初始化列表来确保它在对象创建的初始阶段就被绑定。
!!! example
    ```c++
        class X {
        public:
            int& m_y;       // 声明一个引用成员 m_y
            X(int& a);      // 声明一个构造函数，它接受一个 int 的引用
        };
    ```

---

## Returning references
- Function can return references
    - But they should refer to non-local variables.
    - 因为局部变量在函数函数体执行完毕之后就被销毁了，所以我们不能返回局部变量

---
## Temporary values are const

### What you typr
```c++
    void func(int &);
    func(i*3);
```

### What the complier generates
```c++
    void func(int &);
    const int _tmp_ = i*3;
    func(_tmp_);
```
由于编译器中生成的`_tmp_`是`const`类型的，所以`func`这个函数中一旦对`_tmp_`作出修改就会报错。

这道题中`i`肯定是事先已经被声明过的，但是`i*3`的结果仍然是临时值。

假设我们定义`i=5`，那么`i`本身不是临时变量，但我们执行`i*3`之后，CPU需要找一个地方临时存放这个值，因此它就是临时值。


