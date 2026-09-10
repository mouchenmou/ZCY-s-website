Name: Chenyang Zhang

Student ID: 3043218909

chenyangzhang310@berkeley.edu
## 1

### (a)

![](附件/640d8fa18156296856be92aa628cf2bf.jpg)

### (b)

![](附件/f6535873cbcb1e628719ed94429e2611.jpg)

### (c)

![](<附件/Pasted%20image%2020260907232014.png>)

### (d)

![](附件/38b98ba5aafc2d8b367b69a5b9d1498f.jpg)

### (e)

![](附件/08abcd78589c0729bd9174943a13095e.jpg)

### (f)

![](附件/c7d4d69a660667d9202a673f07c3f873.jpg)

### (g)

Through singular value decomposition, we can transform the original complicated matrix problem into many independent dimensions, which makes the problem much easier to analyze and solve.

---

## 2

### (a)

![](附件/256161e1a9d7a1e99f5f94e37ae98a5e.jpg)

### (b)

![](附件/a2fcf44d5a2f4d2a99b961f6cce49833.jpg)

### (c)

![](附件/5a2aa31050ed049d3c2f53e2e7aa31ba.jpg)

### (d)

![](附件/a904aa01258ac024631d507b2e0ba07e.jpg)

---

## 3

![](<附件/Pasted%20image%2020260909004357.png>)

---

## 4

### (a)

![](附件/5228f6a12849fbaa8cb6fc4493f81f7d.jpg)

### (b)

![](附件/b0189ba27b20d45e8e29762d0b86f650.jpg)

### (c)

![](附件/a3f7e477b38060fa75658255de2ed238.jpg)

### (d)

![](附件/89cc5fffc65d4ba163ca9f0e4348ea59.jpg)

---

## 5

### (a)

The training datas contain noise, while the test datas are noiseless.

### (b)

As the hidden layer width increases, the network has more ReLU units and therefore more possible elbows. This makes the learned function more flexible and allows it to better approximate the piecewise-linear target, which generally reduces the test error.

I would place them near the locations where the target function changes slope.

### (c)

Ridge regression is the most computationally efficient because it directly solves for the output-layer weights instead of using many SGD iterations. Training only the output layer with SGD is slower, while training all layers is generally the most expensive. The ReLU elbows remain fixed when only the output layer is trained, either with SGD or ridge regression, because the hidden-layer parameters do not change. When all layers are trained, the elbows can move. This is important because the network can move the slope-change locations toward the bends in the target function, which can improve the learned function and reduce test error.
