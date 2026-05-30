伯克利 CS61 系列的第二门课程，注重数据结构与算法的设计，同时让学生有机会接触上千行的工程代码，通过 Java 初步领会软件工程的思想。

我上的是 2018 年春季学期的版本，该课的开课老师 Josh Hug 教授慷慨地将 autograder 开源了，大家可以通过网站公开的邀请码在 [gradescope](https://gradescope.com/) 免费加入课程，从而方便地测评自己的代码。

根据教授最新的政策，SP2021 的 CS61B 也对公众开放。要设置所有内容，请前往 Gradescope 并选择"Add a course"按钮。输入课程代码 **MB7ZPY** 以添加课程。

这门课所有的编程作业都是使用 Java 完成的。没有 Java 基础的同学也不用担心，课程会有保姆级的教程，从 IDEA（一款主流的 Java 编程环境）的配置讲起，把 Java 的核心语法与特性事无巨细地讲授，大家完全不用担心跟不上的问题。

---

## Table of Contents

### Lecture 笔记

- [Lecture 1: Intro, Hello World Java](Lecture%201%20Intro,%20Hello%20World%20Java/Intro%201%20Study%20Guide.md)
- [Lecture 2: Defining and Using Classes](Lecture%202%20Definiing%20and%20Using%20Classes/Defining%20and%20Using%20Classes.md)
- [Lecture 3: Testing](Lecture%203%20Testing/Lecture%203%20Testing.md)
- [Lecture 4: References, Recursion, and Lists](Lecture%204/Lecture%204%20References,%20Recursion,%20and%20Lists.md)
- [Lecture 5: SLLists, Nested Classes, Sentinel Nodes](Lecture%205/Lecture%205%20SLLists,%20Nested%20Classes,%20Sentinel%20Nodes.md)
- Lecture 6: DLLists & Arrays
    - [6.1 DLLists](Lecture%206/1.%20DLLists.md)
    - [6.2 Array](Lecture%206/2.%20Array.md)
- Lecture 7: ALists, Resizing, vs. SLists
    - [7.1 AList](Lecture%207%20ALists,%20Resizing,%20vs.%20SLists/1.%20AList.md)
    - [7.2 Array Resizing](Lecture%207%20ALists,%20Resizing,%20vs.%20SLists/2.%20Array%20Resizing.md)
- [Lecture 12: Git](Lecture%2012%20Git/Git.md)
- [Lecture 13: 渐近线 I](Lecture%2013%20渐近线1/Big%20O.md)
- Lecture 14: 不相交集 I (Disjoint Sets)
    - [14.1 QuickFindDS](Lecture%2014%20不相交集1(Disjoint%20Sets)/1.快速查找不相交集（QuickFindDS）.md)
    - [14.2 QuickUnionDS](Lecture%2014%20不相交集1(Disjoint%20Sets)/2.快速连接法（QuickUnionDS）.md)
    - [14.3 Weighted Quick Union](Lecture%2014%20不相交集1(Disjoint%20Sets)/3.加权快速接头（Weighted%20Quick%20Union）.md)
- Lecture 15: 不相交集 II & 渐近线 II
    - [15.1 WQU with Path Compression](Lecture%2015%20不相交集2和渐近线2/15.1路径压缩(WQU%20with%20Path%20Compression).md)
    - [15.2 渐近线 II](Lecture%2015%20不相交集2和渐近线2/15.2%20渐近线2.md)
- Lecture 16: 二分查找法, ADTs, BSTs
    - [二分查找法（BinarySearch）](Lecture%2016%20二分查找法，ADTs，BSTs/二分查找法（BinarySearch）.md)
    - [BSTs](Lecture%2016%20二分查找法，ADTs，BSTs/BSTs.md)
    - [归并排序（Mergesort）](Lecture%2016%20二分查找法，ADTs，BSTs/归并排序（Mergesort）.md)
- Lecture 17: 二叉搜索树, Set and Maps, B-Trees
    - [17.1 BST 的 insert](Lecture%2017%20二叉搜索树，Set%20and%20Maps，B-trees/1.BST的insert.md)
    - [17.2 BST 删除节点](Lecture%2017%20二叉搜索树，Set%20and%20Maps，B-trees/2.BST删除节点.md)
    - [17.3 Set 与 Maps](Lecture%2017%20二叉搜索树，Set%20and%20Maps，B-trees/3.集合与映射（Set%20and%20Maps）.md)
    - [17.4 BST 高度与最坏情况](Lecture%2017%20二叉搜索树，Set%20and%20Maps，B-trees/4.二叉搜索树的高度和最坏情况.md)
    - [17.5 B-Trees](Lecture%2017%20二叉搜索树，Set%20and%20Maps，B-trees/5.B-Trees.md)
- Lecture 18: 旋转树 & 红黑树
    - [18.1 Tree Rotation](Lecture%2018%20旋转树和红黑树/1.Definition%20of%20Tree%20Rotation.md)
    - [18.2 红黑树](Lecture%2018%20旋转树和红黑树/2.红黑树.md)
- Lecture 19: 哈希算法
    - [19.1 BobaCounterSet](Lecture%2019%20哈希算法/19.1%20BobaCounterSet.md)
    - [19.2 Dynamic Array Of Lists Set](Lecture%2019%20哈希算法/19.2%20Dynamic%20Array%20Of%20Lists%20Set.md)
    - [19.3 关于哈希算法的深解析](Lecture%2019%20哈希算法/19.3%20关于哈希算法的深解析.md)
- Lecture 20: Priority Queues & Heaps
    - [20.1 优先队列](Lecture%2020%20Priority%20Queues%20and%20Heaps/20.1%20优先队列.md)
    - [20.2 堆](Lecture%2020%20Priority%20Queues%20and%20Heaps/20.2%20堆.md)
- Lecture 21: Tree & Graph Traversals
    - [21.1 Trees and Traversals](Lecture%2021%20Tree%20and%20Graph%20Traversals/21.1%20Trees%20and%20Traversals.md)
    - [21.2 Graph](Lecture%2021%20Tree%20and%20Graph%20Traversals/21.2%20Graph.md)
- [Lecture 23: Shortest Paths](Lecture%2023%20Shortest%20Paths/Shortest%20Paths.md)
- [Lecture 24: Minimum Spanning Trees](Lecture%2024%20Minimum%20Spanning%20Trees/Lecture%2024%20Minimum%20Spanning%20Trees.md)
- Lecture 29: Sorting I
    - [排序问题](Lecture%2029%20Basic%20Sort/1.%20排序问题.md)
    - [Selection Sort](Lecture%2029%20Basic%20Sort/2.%20Selection%20Sort.md)
    - [Heapsort](Lecture%2029%20Basic%20Sort/3.%20Heapsort.md)
    - [Mergesort](Lecture%2029%20Basic%20Sort/4.%20Mergesort.md)
    - [Insertion Sort](Lecture%2029%20Basic%20Sort/5.%20Insertion%20Sort.md)
    - [Shell Sort](Lecture%2029%20Basic%20Sort/6.%20Shell%20Sort.md)

### Project

- [Project 0: maxTileExists](Project/Project0/Q2%20public%20static%20boolean%20maxTileExists.md)

### Lab

- [Lab 1: 命令行](lab/lab1/命令行.md)
