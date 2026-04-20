需要注意的是，传入的LIst本身已经有了一个Dummy head，我一开始以为传入的List没有Dummy head。

```c
    List Reverse(List L){
        PtrToNode head, Know, changed,former;
        former = NULL;
        head = (PtrToNode)malloc(sizeof(struct Node));
        Know = L->Next;
        while (Know!=NULL){
            changed = Know;
            Know = Know ->Next;
            changed ->Next = former;
            former = changed;
        }
        head=L;
        head->Next = former;
        return head;
    }
```

### 重点：
摸索了很久才得出执行了`head=L;`才能将L1 也变为L2，这一点值得思考。。