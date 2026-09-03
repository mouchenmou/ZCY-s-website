# 5. 为什么这个 Tiered Prompting 很重要？

这是你以后做 slide 一定要讲出来的。

传统 benchmark 往往就是：

```
Image + Question
        ↓
      Model
        ↓
 Correct / Wrong
```

最终只有：

> accuracy

但是 LADBench 在问：

> **你需要别人提示到什么程度，才能发现问题？**

于是：

```
完全自己发现
      ↓
Level 1
100%

别人告诉你“有问题”
      ↓
Level 2
66.67%

别人连问题类别都告诉你
      ↓
Level 3
33.33%
```

这衡量的是：

> **degree of autonomous reasoning**

也就是：

**模型到底有多“自主”地发现逻辑错误。**

因此作者不是简单算 binary accuracy，而是用了一个：

> **decay-weighted scoring system**

提示越多，分数越低。
