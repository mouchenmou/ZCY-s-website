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
            virtual Expr self;
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

![](Pasted%20image%2020251124220921.png)

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
    1. Never redefin an inherited non-virtual function.（不要重新定义继承来的非虚函数）
        - 原因：`Non-virtuals are statically bound`。（非虚函数是通过静态绑定机制调用的）
        - 后果：No dynamic dispatch
            - 静态绑定是在编译时确定的，它只看你声明的变量或指针的静态类型，而不会看对象的实际类型。如果你重新定义了一个非虚函数，通过基类指针调用该函数时，永远只会调用基类的版本，完全无法实现多态。也就是说我们在lecture中实现的内容实际上是不好的东西，不要采取。
    2. Never redefine an inherited default parameter value.（不要重新定义继承来的默认参数）
        - 默认参数也是通过静态绑定机制确定的。
        - 当你通过基类指针调用虚函数，而该虚函数使用了默认参数时，虚函数本身会动态调用派生类的代码，但默认参数的值却会静态地取自基类。
        - 

