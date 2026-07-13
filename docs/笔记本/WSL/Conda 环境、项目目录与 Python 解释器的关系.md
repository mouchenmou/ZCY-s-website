
## 1. 什么是 Conda 环境？

**Conda 环境 = 一个独立的 Python 运行空间。**

它包含：

- Python 解释器
- 第三方库
- 命令行工具
- 依赖版本

例如：

```
miniconda3
│
└── envs
    │
    ├── yolo
    │    ├── python 3.10
    │    ├── opencv
    │    ├── pytorch
    │    └── ultralytics
    │
    └── nerfstudio
         ├── python 3.10
         ├── pytorch CUDA
         └── nerfstudio
```

不同环境之间互相隔离。

例如：

```
yolo环境：
Python 3.10
torch 2.2

nerfstudio环境：
Python 3.10
torch 2.1 + CUDA
```

两个项目可以使用不同版本，不会冲突。

---

## 2. conda activate 到底做了什么？

### 错误理解：

> activate 是进入某个文件夹。

这是错误的。


### 正确理解：

```
conda activate yolo
```

表示：

> 切换当前终端使用的 Python 环境。

它主要改变：

- python 命令指向哪里
- pip 安装到哪里
- Python 去哪里寻找库

例如：

激活前：

```
which python
```

可能：

```
/usr/bin/python3
```

激活：

```
conda activate yolo
```

之后：

```
/home/mouchenmou/miniconda3/envs/yolo/bin/python
```


但是：

**conda activate 不会改变当前目录。**

例如：

当前：

```
/home/mouchenmou/projects/yolo
```

执行：

```
conda activate yolo
```

之后：

仍然：

```
/home/mouchenmou/projects/yolo
```

只是 Python 换了。

---

## 3. Conda 环境文件夹 ≠ 项目文件夹

这是最容易混淆的地方。

### Conda环境：

例如：

```
/home/mouchenmou/miniconda3/envs/yolo
```

里面：

```
bin/
lib/
python
opencv
torch
```

作用：

> 提供运行程序需要的软件环境。

## 项目文件夹：

例如：

```
/home/mouchenmou/projects/yolo
```

里面：

```
test.py
cat.png
yolo11n.pt
```

作用：

> 存放你的代码和数据。

二者关系：

```
项目文件夹
      |
      | 使用
      ↓
Conda环境
```

不是：

```
项目文件夹
里面包含
Conda环境
```

---

# 4. 为什么激活 yolo 环境后，不一定在 yolo 文件夹？

注意，我给一个环境命名为yolo，给一个project也命名为yolo了。

## 环境：

```
~/miniconda3/envs/yolo
```

表示：

Python工具箱。

## 项目：

```
~/projects/yolo
```

表示：

代码文件夹。

虽然名字一样：

```
yolo
```

但是完全不同。

---

# 5. 创建项目文件夹需要进入环境吗？

不需要。

例如：

当前：

```
(base)
```

或者：

```
没有激活环境
```

都可以：

```
mkdir ~/projects/yolo
```

因为创建文件夹只是 Linux 文件操作。

和 Python 环境没有关系。


正确流程：

### 创建项目

```
mkdir ~/projects/yolo
```


### 进入项目

```
cd ~/projects/yolo
```

---

### 使用对应环境

```
conda activate yolo
```

---

### 运行代码

```
python test.py
```

此时：

```
test.py
    |
    ↓
yolo环境中的python
    |
    ↓
opencv / torch / ultralytics
```

---

# 6. 为什么很多教程看起来像项目在环境里创建？

因为很多人习惯：

```
conda activate yolo

mkdir project

cd project
```

于是看起来：

```
yolo环境
    |
    创建项目
```

实际上没有任何绑定。

只是创建项目时刚好激活了环境。

---

# 7. VS Code 中的环境关系

VS Code 连接 WSL 后：

结构：

```
Windows VS Code

        |
        ↓

WSL Ubuntu

        |
        ↓

Conda环境 yolo

        |
        ↓

Python代码
```

但是：

连接 WSL ≠ 使用 yolo环境。

还需要选择解释器。


选择：

```
Python: Select Interpreter
```

然后：

```
Python 3.10 ('yolo')
```

或者：

```
~/miniconda3/envs/yolo/bin/python
```

---

# 8. VS Code 如何记住项目环境？

VS Code 可以在项目里面保存：

```
.vscode/settings.json
```

例如：

```
{
    "python.defaultInterpreterPath":
    "/home/mouchenmou/miniconda3/envs/yolo/bin/python"
}
```

以后打开这个项目：

VS Code 自动使用 yolo 环境。

---

# 9. 最终理解模型

把整个关系画出来：

```
电脑
│
├── Conda环境
│
│    ├── yolo
│    │     ├── Python
│    │     ├── OpenCV
│    │     └── PyTorch
│    │
│    └── nerfstudio
│          ├── Python
│          └── CUDA PyTorch
│
│
└── 项目
     │
     ├── projects/yolo
     │       ├── test.py
     │       └── 图片
     │
     └── projects/nerf
             └── code
```

项目决定：

> 写什么代码

环境决定：

> 用什么工具运行代码

---

# 一句话总结

> **Conda 环境不是代码所在的文件夹，而是一套 Python 工具箱。`conda activate` 不是进入文件夹，而是切换 Python 工具箱。项目文件夹可以放在任何地方，运行时只需要选择正确的环境即可。**