## From IntList to SLList
![](附件/Pasted%20image%2020260303154523.png)
![](附件/Pasted%20image%2020260303154719.png)
### SLLits vs. IntLists
![](附件/Pasted%20image%2020260303171716.png)
![](附件/Pasted%20image%2020260303171753.png)

---
## Public vs. Private Nested Classes

![](附件/Pasted%20image%2020260303172607.png)
使用SLLists的人可能会直接操作class内部的IntNode，就像上图中红色字段所指的代码那样。`L.first.next.next`, 也就是15的下一个节点，本应为null，可是却将它设置为了`L.first.next`, 导致15这个节点的next指向了它自身，这样就陷入15->15->15->15.........的死循环了。

为了解决这种问题，将`public IntNode first;` 改为 `private class first`
### Why nested classes？

We can combine tow classes into one file pretty simply, but why nested classes?
![](附件/Pasted%20image%2020260303175217.png)

Nested classes are useful when a class doesn't stand on its own and is obviously subordinate to another class.

- Make the nested class private if other classes should never use the nested class.

In my opinion, probably makes sense to make IntNode a nested private class.

- Hard to imagine other classes having a need to manipulate IntNode.

Summary: 之前我们只是把 `first` 变量设为 `private`，不让别人碰。现在老师建议更进一步：既然 `IntNode` 这个类除了 `SLList` 之外没人用，不如直接把它放进 `SLList` 里面，并且把整个类都设为 `private`。这样外部代码不仅碰不到节点变量，甚至连 `IntNode` 这个类的存在都不知道！这是最高级别的封装和防御。

### Static Nested Classes

![](附件/Pasted%20image%2020260303172248.png)

---
## Sentinel Nodes:
![](附件/Pasted%20image%2020260303180452.png)

上面的代码看似没问题，但有个小小的bug，假设我们创建了一个空的SSList，我们要调用addLast函数在这个SSList中加入元素，但是first是null，所以p就被设置为null，p本身已经是null了，执行p.next会导致空指针异常，因为我们不能访问吗内存位置null并尝试获取next。解决方法之一就是引入”哨兵“（Sentinel Node）。

### 下面引入哨兵来解决这个问题：
```
    public SLList(){
        sentinel = new IntNode(63, null);
        size = 0;
    }
    
    public SLList(int x){
        sentinel = new IntNode(63, null);//哨兵在此， 63是随便设置的值
        sentinel.next = new IntNode(x, null);
        size = 1;
    }
    
    public void addFirst(int x){
        sentinel.next = new(x, sentinel.next);
        size = size + 1;
    }
    
    public int getFirst{
        return sentinel.next.item;
    }
    
    public void addLast(int x){
        size = size + 1;
        IntNode p = sentinel;
        while(p.next!=null){
            p=p.next;
        }  
        p.next = new IntNode(x, null)
    }
```
