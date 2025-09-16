# 1.二叉树的起源
## 为了实现更快速的查找而发明了二叉树
![](../Lecture16%20photo/Pasted image 20250915171613.png)

---

# 2.树的定义
## 树的组成：
- 有一组节点
- 有一组边（连接节点的线）
- 任意两个节点直接只有一种连接方式
- 下面那幅图中的第四张和第五张图不是树，因为有些节点不止一条通路
- ![](../Lecture16%20photo/Pasted image 20250915172119.png)

## 普通树的定义：
- 我们树的最顶端那个节点为树根
- 除了树根以外，别的节点都有一个父节点
- 跟 [真实的树](https://www.amusingplanet.com/2010/12/baobab-upside-down-tree.html)不一样, 数据结构中，树的树根在树的顶端
- **没有子节点**的节点被称为**叶子**

---

## 3.二叉树的定义

### 一个二叉搜索树的每一个节点都只能有0，1或者2个子节点
![](../Lecture16%20photo/Pasted image 20250915172720.png)

---

## 4.二叉搜索树的定义
- 左子树中的所有节点都小于当前节点的值
- 右子树中的所有节点都大于当前结点的值
- 二叉搜索树中不能有重复的元素，比如一棵树中不能出现两个dog
- 下面两幅图中左图是二叉搜索树，而右图只是普通的二叉树而不是二叉搜索树，因为右图中fish开头为f，而ear开头为e，由于f>e，而fish又在ear的左侧，因此不符合二叉搜索树的定义。
![](../Lecture16%20photo/Pasted image 20250915173252.png)

### 以下是用c语言实现二叉搜索树的代码：
```c
typedef struct BST {
    int key;
    struct BST *left;
    struct BST *right;
} BST;

BST* newBST(int ik) {
    BST* node = (BST*)malloc(sizeof(BST));
    node->key = ik;
    node->left = NULL;
    node->right = NULL;
    return node;
}

BST* insert(BST* T, int ik) {
    if (T == NULL)
        return newBST(ik);
    if (ik < T->key)
        T->left = insert(T->left, ik);
    else if (ik > T->key)
        T->right = insert(T->right, ik);
    return T;
}
```

### 以下为一道例题
![](../Lecture16%20photo/Pasted image 20250915175942.png)
### 解析：
答案选A，若要使它的时间复杂度+1，则，则需要再给这棵树加一层，，若再加一层则N大致翻倍，符合logN
