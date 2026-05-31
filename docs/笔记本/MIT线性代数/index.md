# MIT 18.06 线性代数

MIT 文明世界的线性代数课，Gilbert Strang 老爷子主讲。这门课最大的特点是**从矩阵的角度理解线性代数**——不是先讲行列式再去定义一切，而是从向量、矩阵乘法、消去法出发，层层递进到四个基本子空间、正交性、特征值，最后以 SVD 收尾。

国内线代课通病：先花半学期讲行列式，学生背了一堆公式但不知道为什么要学。Strang 的做法恰好相反：行列式放到第五章才讲，而且把它定义为"由三条性质唯一确定的数"，逻辑清晰。如果你学完这门课只记得行列式的计算公式而不知道为什么它等于体积、为什么它和特征值有关，那等于白学。

本人做这份笔记的初衷：网上很多 MIT 线代的笔记只是摘抄 PPT，没有把推导过程写清楚。我花了大量时间逐段翻译、补全推导，希望对大家有帮助。

---

## Table of Contents

### 第1章 Introduction to Vectors
- [1.1 Vectors and Linear Combinations](第1章%20Introduction%20to%20Vectors/1.1%20Vectors%20and%20Linear%20Combinations.md)
- [1.2 Lengths and Dot Products](第1章%20Introduction%20to%20Vectors/1.2%20Lengths%20and%20Dot%20Products.md)

### 第2章 Solving Linear Equations
- [2.1 Vectors and Linear Equations](第2章%20Solving%20Linear%20Equations/2.1%20Vectors%20and%20Linear%20Equations.md)
- [2.2 The Idea of Elimination](第2章%20Solving%20Linear%20Equations/2.2%20The%20Idea%20of%20Elimination.md)
- [2.3 Elimination Using Matrices](第2章%20Solving%20Linear%20Equations/2.3%20Elimination%20Using%20Matrices.md)
- [2.4 Rules for Matrix Operations](第2章%20Solving%20Linear%20Equations/2.4%20Rules%20for%20Matrix%20Operations.md)
- [2.5 Inverse Matrices](第2章%20Solving%20Linear%20Equations/2.5%20Inverse%20Matrices.md)
- [2.6 Elimination = Factorization A = LU](第2章%20Solving%20Linear%20Equations/2.6%20Elimination%20=%20Factorization%20A%20=%20LU.md)
- [2.7 Transposes and Permutations](第2章%20Solving%20Linear%20Equations/2.7%20Transposes%20and%20Permutations.md)

### 第3章 Vector Spaces and Subspaces
- [3.1 Spaces of Vectors](第3章%20Vector%20Spaces%20and%20Subspaces/3.1%20Spaces%20of%20Vectors.md)
- [3.2 The Nullspace of A: Solving Ax = 0](第3章%20Vector%20Spaces%20and%20Subspaces/3.2%20The%20Nullspace%20of%20A%20Solving%20Ax=0.md)
- [3.3 The Rank and the Row Reduced Form](第3章%20Vector%20Spaces%20and%20Subspaces/3.3%20The%20Rank%20and%20the%20Row%20Reduced%20Form.md)
- [3.4 The Complete Solution to Ax = b](第3章%20Vector%20Spaces%20and%20Subspaces/3.4%20The%20Complete%20Solution%20to%20Ax=b.md)
- [3.5 Independence, Basis and Dimension](第3章%20Vector%20Spaces%20and%20Subspaces/3.5%20Independence,%20Basis%20and%20Dimension.md)
- [3.6 Dimensions of the Four Subspaces](第3章%20Vector%20Spaces%20and%20Subspaces/3.6%20Dimensions%20of%20the%20Four%20Subspaces.md)

### 第4章 Orthogonality
- [4.1 Orthogonality of the Four Subspaces](第4章%20Orthogonality/4.1%20Orthogonality%20of%20the%20Four%20Subspaces.md)
- [4.2 Projections](第4章%20Orthogonality/4.2%20Projections.md)
- [4.3 Least Squares Approximations](第4章%20Orthogonality/4.3%20Least%20Squares%20Approximations.md)
- [4.4 Orthogonal Bases and Gram-Schmidt](第4章%20Orthogonality/4.4%20Orthogonal%20Bases%20and%20Gram-Schmidt.md)

### 第5章 Determinants
- [5.1 The Properties of Determinants](第5章%20Determinants/5.1%20The%20Properties%20of%20Determinants.md)
- [5.2 Permutations and Cofactors](第5章%20Determinants/5.2%20Permutations%20and%20Cofactors.md)
- [5.3 Cramer's Rule, Inverses, and Volumes](第5章%20Determinants/5.3%20Cramers%20Rule,%20Inverses,%20and%20Volumes.md)

### 第6章 Eigenvalues and Eigenvectors
- [6.1 Introduction to Eigenvalues](第6章%20Eigenvalues%20and%20Eigenvectors/6.1%20Introduction%20to%20Eigenvalues.md)
- [6.2 Diagonalizing a Matrix](第6章%20Eigenvalues%20and%20Eigenvectors/6.2%20Diagonalizing%20a%20Matrix.md)
- [6.3 Applications to Differential Equations](第6章%20Eigenvalues%20and%20Eigenvectors/6.3%20Applications%20to%20Differential%20Equations.md)
- [6.4 Symmetric Matrices](第6章%20Eigenvalues%20and%20Eigenvectors/6.4%20Symmetric%20Matrices.md)
- [6.5 Positive Definite Matrices](第6章%20Eigenvalues%20and%20Eigenvectors/6.5%20Positive%20Definite%20Matrices.md)
- [6.6 Similar Matrices](第6章%20Eigenvalues%20and%20Eigenvectors/6.6%20Similar%20Matrices.md)
- [6.7 Singular Value Decomposition (SVD)](第6章%20Eigenvalues%20and%20Eigenvectors/6.7%20Singular%20Value%20Decomposition%20SVD.md)

### 第7章 Linear Transformations
- [7.1 The Idea of a Linear Transformation](第7章%20Linear%20Transformations/7.1%20The%20Idea%20of%20a%20Linear%20Transformation.md)
- [7.2 The Matrix of a Linear Transformation](第7章%20Linear%20Transformations/7.2%20The%20Matrix%20of%20a%20Linear%20Transformation.md)
- [7.3 Change of Basis](第7章%20Linear%20Transformations/7.3%20Change%20of%20Basis.md)
- [7.4 Diagonalization and the Pseudoinverse](第7章%20Linear%20Transformations/7.4%20Diagonalization%20and%20the%20Pseudoinverse.md)

### 第8章 Applications
- [8.1 Matrices in Engineering](第8章%20Applications/8.1%20Matrices%20in%20Engineering.md)
- [8.2 Graphs and Networks](第8章%20Applications/8.2%20Graphs%20and%20Networks.md)
- [8.3 Markov Matrices and Economic Models](第8章%20Applications/8.3%20Markov%20Matrices%20and%20Economic%20Models.md)
- [8.4 Linear Programming](第8章%20Applications/8.4%20Linear%20Programming.md)
- [8.5 Fourier Series](第8章%20Applications/8.5%20Fourier%20Series.md)
- [8.6 Computer Graphics](第8章%20Applications/8.6%20Computer%20Graphics.md)
- [8.7 Statistics and Probability](第8章%20Applications/8.7%20Statistics%20and%20Probability.md)

### 第9章 Numerical Linear Algebra
- [9.1 Gaussian Elimination in Practice](第9章%20Numerical%20Linear%20Algebra/9.1%20Gaussian%20Elimination%20in%20Practice.md)
- [9.2 Norms and Condition Numbers](第9章%20Numerical%20Linear%20Algebra/9.2%20Norms%20and%20Condition%20Numbers.md)
- [9.3 Iterative Methods for Linear Algebra](第9章%20Numerical%20Linear%20Algebra/9.3%20Iterative%20Methods%20for%20Linear%20Algebra.md)

### 第10章 Complex Vectors and Matrices
- [10.1 Complex Numbers](第10章%20Complex%20Vectors%20and%20Matrices/10.1%20Complex%20Numbers.md)
- [10.2 Hermitian and Unitary Matrices](第10章%20Complex%20Vectors%20and%20Matrices/10.2%20Hermitian%20and%20Unitary%20Matrices.md)
- [10.3 The Fast Fourier Transform](第10章%20Complex%20Vectors%20and%20Matrices/10.3%20The%20Fast%20Fourier%20Transform.md)

---
