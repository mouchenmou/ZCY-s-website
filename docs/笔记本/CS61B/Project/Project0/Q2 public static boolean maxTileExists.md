## 我的代码：
```java
    public static boolean maxTileExists(Board b) {  
    // TODO: Fill in this function.  
    
        for(int i=0; i<b.size();i++){  
            for(int j=0; j<b.size();j++){  
                Tile t = b.tile(i, j);  
                if (t.value() == MAX_PIECE) {  
                    return true;  
                }  
            }  
        }  
        return false;  
    }
```

---
## 正确的代码：
```java
    public static boolean maxTileExists(Board b) {  
    // TODO: Fill in this function.  
    
        for(int i=0; i<b.size();i++){  
            for(int j=0; j<b.size();j++){  
                Tile t = b.tile(i, j);  
                if (t!=null && t.value() == MAX_PIECE) {  
                    return true;  
                }  
            }  
        }  
        return false;  
    }
```

!!! question "既然`value`已经为`MAX_PIECE`了，为什么还要写个`t! = null`"
    ## 解答
    ### 1. 为什么“逻辑上”对了，但“程序上”错了？
    Java 运行代码时是非常“死板”的，它会按照你写的指令一步步执行：
    
    1. **我写的指令：** `t.value()`
    2. **Java 的动作：** 尝试去访问 `t` 指向的那个内存空间，并取出里面的 `value`。
    3. **现实情况：** 棋盘上有很多格子。当循环跑到第一个空位时，`t` 就是 `null`（代表什么都没有，连个“数值”属性都没有）。
    4. **结果：** Java 还没来得及看这个值是不是 `MAX_PIECE`，就在执行 `t.value()` 的瞬间崩掉了（NullPointerException）。
    
    **这就像：** 你想在街上找一个戴红帽子的人。你对着路边每一个位置大喊：“喂！把你帽子摘下来给我看看颜色！”
    - 如果那个位置站着人（`Tile`），他会摘下帽子，你一看是红色（`MAX_PIECE`），皆大欢喜。
    - 如果那个位置是空的（`null`），你对着空气喊“摘帽子”，空气没法摘帽子，你就当场气晕了（程序崩溃）。
    
    ---
    
    ### 2. 正确的“防御式”检查
    
    为了不让程序崩溃，你必须先确认“那里有人”，然后再问“帽子颜色”。
    ```java
        Tile t = b.tile(i, j);
        // 顺序至关重要！
        if (t != null && t.value() == MAX_PIECE) { 
            return true; 
        }
    ```
    
    - **如果 `t` 是 `null`：** `t != null` 返回 `false`。因为用了 `&&`（短路与），Java 知道整个条件已经不可能成立了，所以它 **直接跳过** 后面的 `t.value()`。这样就不会触发错误。
    
    - **如果 `t` 不是 `null`：** Java 才会放心地去执行 `t.value()`。
    
    ### 3. 总结
    
    在 Java 里， 值是属于对象的。如果对象本身不存在（`null`），那么属于它的值也无从谈起。所以，在访问任何对象的属性或方法之前，先判断它是否为 `null` 是一个非常重要的编程习惯。