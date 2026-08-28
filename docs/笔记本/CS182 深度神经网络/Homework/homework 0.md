## 1. Attention is all you need

I have already filled out this form.

My name: Chenyang Zhang

My Student ID: 3043218909

My email: chenyangzhang310@berkeley.edu

## 2. Reflection on Your Learning Goals at the Start of the Semester

### (a)

I hope to develop a clear understanding of the mathematical principles behind deep learning through CS182. I also hope to apply what I learn in this course to robot learning, which is one of my main research interests.

I have previously studied computer vision and deep learning through CS231n, which helped me build a basic foundation in machine learning and deep learning. However, I felt that it was more like an introductory course in some aspects. Many concepts were introduced without detailed mathematical derivations, and conclusions were sometimes presented directly. This made it difficult for me to fully understand why certain methods work.

Berkeley courses are well known for their rigorous mathematical treatment of technical topics, so I hope CS182 will help me understand the mathematical foundations of deep learning in a much more systematic way. I am really looking forward to the course.
### (b)
I told GPT that CS231n did not spend much time explaining the mathematical principles behind many deep learning concepts, which made the learning process difficult for me. Therefore, I said that I hoped CS182 could help me clearly understand the mathematics behind deep learning and eventually apply this knowledge to robot learning, which is one of my main research interests. GPT then discussed these goals with me.

After the discussion, I realized that understanding the mathematics behind deep learning is not only about being able to follow derivations. I also have to understand how these mathematical ideas explain the behavior of neural networks, such as how optimization works, why certain architectures are effective, and how models learn useful representations from data.

This did not fundamentally change my original learning goal, but it made the goal more concrete. I still hope to build a strong mathematical foundation in deep learning through CS182, while also learning how to connect mathematical principles with the behavior of real models and eventually with applications in robot learning.

### (c)

Before looking at the Fall 2026 course website, I had already looked through the Fall 2025 website and the spring 2021 of CS182. The 2021 version was taught by Sergey Levine, and I noticed that the course content in 2021 was already quite different from that in Fall 2025. This made me realize how quickly the field of deep learning evolves. I also expect that the Fall 2026 version of CS182 will contain some new topics or changes compared with Fall 2025. This is also consistent with the course homepage's statement that "it isn't 2015 anymore."

Donoho's paper further strengthened my impression that machine learning is developing at an extremely fast pace. His discussion of data sharing, code sharing, and reproducibility showed me how new ideas can be tested, reproduced, and improved very quickly. It made me realize that simply learning the methods that are currently popular is probably not enough, because some of them may soon be replaced by newer approaches.

Tao's paper made me think about this issue from another perspective. As AI becomes increasingly capable of assisting people with mathematical and technical work, simply knowing how to use an AI system or obtain an answer from it is becoming less valuable by itself. What seems more important to me is understanding the reasoning and principles behind the answer, and being able to judge whether the result actually makes sense.

After reading the course website and these two papers, I now think the role of CS182 is not simply to teach me a collection of current deep learning methods. Since deep learning is changing so quickly, it is impossible for one course to cover every technique that I may encounter in the future. Instead, I hope CS182 can give me a strong mathematical foundation and a way of thinking about deep learning, so that when new models and methods appear, I will be able to understand them from their underlying principles rather than only learning how to use them.

### (d)

I think doing projects can significantly improve my understanding of deep learning. Sometimes I may think that I already understand a mathematical principle when I read the derivation or follow an example, but when I actually try to implement it in code, I suddenly realize that there are still many details that I do not fully understand. To me, this means that my understanding was not deep enough.

Projects can help solve this problem because they require me not only to understand the mathematical ideas, but also to translate them into actual implementations. In addition, completing a project usually requires strong engineering skills. I believe this process can greatly deepen my understanding of deep learning and help me connect mathematical principles with how models actually work in practice.

For course staff, I think the most helpful guidance would be helping me understand questions or concepts that I could not fully understand during lectures, especially difficult mathematical derivations and the intuition behind them.

I also think group collaboration can help me experience the atmosphere of research in advance. Different people usually have different strengths: some may be better at mathematics, some at designing the engineering framework, and others at coding and implementation. A good project often requires everyone to contribute what they are best at. I think research works in a similar way. It is usually a process of teamwork and division of responsibilities rather than one person trying to do everything alone. Working with peers in this way would not only help me learn deep learning, but also give me a better understanding of how collaborative research is actually carried out.

---

# 3. Vector Calculus Review


## (a)

Let

f(x)=xTc.f(x)=x^Tc.

Since $f$ is a scalar and $x\in\mathbb{R}^n$, its derivative with respect to $x$ is a $1\times n$ row vector.

Expanding,

f(x)=∑i=1nxici.f(x)=\sum_{i=1}^n x_i c_i.

For an arbitrary coordinate $x_j$,

∂f∂xj=∂∂xj∑i=1nxici=cj.\frac{\partial f}{\partial x_j} = \frac{\partial}{\partial x_j} \sum_{i=1}^n x_i c_i = c_j.

Therefore,

∂∂x(xTc)=cT.\boxed{ \frac{\partial}{\partial x}(x^Tc)=c^T }.

---

## (b)

We have

∥x∥22=xTx=∑i=1nxi2.\|x\|_2^2=x^Tx=\sum_{i=1}^n x_i^2.

For an arbitrary coordinate $x_j$,

∂∂xj∥x∥22=2xj.\frac{\partial}{\partial x_j}\|x\|_2^2 = 2x_j.

Stacking all derivatives into a row vector gives

∂∂x∥x∥22=2xT.\boxed{ \frac{\partial}{\partial x}\|x\|_2^2 = 2x^T }.

---

## (c)

Let

f(x)=Ax.f(x)=Ax.

Since $f(x)\in\mathbb{R}^n$, its derivative with respect to $x\in\mathbb{R}^n$ is an $n\times n$ Jacobian matrix.

The $i$-th component of $f$ is

fi(x)=∑k=1nAikxk.f_i(x)=\sum_{k=1}^n A_{ik}x_k.

Therefore,

∂fi∂xj=Aij.\frac{\partial f_i}{\partial x_j}=A_{ij}.

Thus,

∂∂x(Ax)=A.\boxed{ \frac{\partial}{\partial x}(Ax)=A }.

---

## (d)

Let

f(x)=xTAx.f(x)=x^TAx.

Since $f$ is a scalar, its derivative with respect to $x$ is a $1\times n$ row vector.

Expand the quadratic form:

f(x)=∑i=1n∑j=1nxiAijxj.f(x) = \sum_{i=1}^n\sum_{j=1}^n x_iA_{ij}x_j.

For an arbitrary coordinate $x_k$,

∂f∂xk=∑j=1nAkjxj+∑i=1nxiAik.\frac{\partial f}{\partial x_k} = \sum_{j=1}^n A_{kj}x_j + \sum_{i=1}^n x_iA_{ik}.

The first term is the $k$-th entry of $Ax$, while the second term is the $k$-th entry of $A^Tx$.

Therefore, under the row-vector derivative convention,

∂∂x(xTAx)=xT(A+AT).\boxed{ \frac{\partial}{\partial x}(x^TAx) = x^T(A+A^T) }.

---

## (e)

We want

xT(A+AT)=2xTA.x^T(A+A^T)=2x^TA.

This holds when

AT=A.A^T=A.

Therefore,

A must be symmetric.\boxed{A\text{ must be symmetric}.}

---

# 4. Least Squares and the Min-Norm Problem from the Perspective of SVD

Let

X=UΣVT.X=U\Sigma V^T.

## (a)

For the overdetermined case $m>n$, we solve

min⁡w∥Xw−y∥22.\min_w\|Xw-y\|_2^2.

The objective is

L(w)=(Xw−y)T(Xw−y).L(w) = (Xw-y)^T(Xw-y).

Expanding,

L(w)=wTXTXw−2yTXw+yTy.L(w) = w^TX^TXw - 2y^TXw + y^Ty.

Setting the gradient equal to zero gives

XTXw=XTy.X^TXw=X^Ty.

Assuming $X$ has full column rank,

w∗=(XTX)−1XTy.\boxed{ w^*=(X^TX)^{-1}X^Ty }.

---

## (b)

Substitute

X=UΣVT.X=U\Sigma V^T.

Then

XTX=VΣTUTUΣVT=VΣTΣVT.X^TX = V\Sigma^TU^TU\Sigma V^T = V\Sigma^T\Sigma V^T.

Therefore,

(XTX)−1=V(ΣTΣ)−1VT.(X^TX)^{-1} = V(\Sigma^T\Sigma)^{-1}V^T.

Also,

XT=VΣTUT.X^T=V\Sigma^TU^T.

Hence,

w∗=V(ΣTΣ)−1ΣTUTy.w^* = V(\Sigma^T\Sigma)^{-1}\Sigma^TU^Ty.

For the tall full-column-rank case,

(ΣTΣ)−1ΣT=Σ†.(\Sigma^T\Sigma)^{-1}\Sigma^T = \Sigma^\dagger.

Thus,

w∗=VΣ†UTy.\boxed{ w^* = V\Sigma^\dagger U^Ty }.

---

## (c)

Define

A=VΣ†UT.A=V\Sigma^\dagger U^T.

Then

AX=VΣ†UTUΣVT=V(Σ†Σ)VT.AX = V\Sigma^\dagger U^TU\Sigma V^T = V(\Sigma^\dagger\Sigma)V^T.

For a tall full-column-rank matrix,

Σ†Σ=In.\Sigma^\dagger\Sigma=I_n.

Therefore,

AX=VInVT=In.AX = VI_nV^T = I_n.

Thus,

AX=In,\boxed{AX=I_n},

which is why $A$ is called a left inverse.

---

## (d)

For the underdetermined case $m<n$, we solve

min⁡w∥w∥22\min_w\|w\|_2^2

subject to

Xw=y.Xw=y.

The minimum-norm solution is

w∗=XT(XXT)−1y,\boxed{ w^* = X^T(XX^T)^{-1}y },

assuming $X$ has full row rank.

---

## (e)

Using

X=UΣVT,X=U\Sigma V^T,

we obtain

XXT=UΣVTVΣTUT=UΣΣTUT.XX^T = U\Sigma V^TV\Sigma^TU^T = U\Sigma\Sigma^TU^T.

Thus,

(XXT)−1=U(ΣΣT)−1UT.(XX^T)^{-1} = U(\Sigma\Sigma^T)^{-1}U^T.

Therefore,

w∗=VΣTUTU(ΣΣT)−1UTy.w^* = V\Sigma^TU^TU(\Sigma\Sigma^T)^{-1}U^Ty.

Simplifying,

w∗=VΣT(ΣΣT)−1UTy.w^* = V\Sigma^T(\Sigma\Sigma^T)^{-1}U^Ty.

For the wide full-row-rank case,

ΣT(ΣΣT)−1=Σ†.\Sigma^T(\Sigma\Sigma^T)^{-1} = \Sigma^\dagger.

Hence,

w∗=VΣ†UTy.\boxed{ w^* = V\Sigma^\dagger U^Ty }.

---

## (f)

Let

B=VΣ†UT.B=V\Sigma^\dagger U^T.

Then

XB=UΣVTVΣ†UT=U(ΣΣ†)UT.XB = U\Sigma V^TV\Sigma^\dagger U^T = U(\Sigma\Sigma^\dagger)U^T.

For a wide full-row-rank matrix,

ΣΣ†=Im.\Sigma\Sigma^\dagger=I_m.

Therefore,

XB=UImUT=Im.XB = UI_mU^T = I_m.

Thus,

XB=Im,\boxed{XB=I_m},

which is why $B$ is called a right inverse.

---

# 5. The Five Interpretations of Ridge Regression

## (a) Perspective 1: Optimization Problem

The ridge regression objective is

J(w)=∥y−Xw∥22+λ∥w∥22.J(w) = \|y-Xw\|_2^2 + \lambda\|w\|_2^2.

Expanding,

J(w)=(y−Xw)T(y−Xw)+λwTw.J(w) = (y-Xw)^T(y-Xw)+\lambda w^Tw.

Therefore,

J(w)=yTy−2yTXw+wTXTXw+λwTw.J(w) = y^Ty - 2y^TXw + w^TX^TXw + \lambda w^Tw.

Taking the gradient and setting it equal to zero,

−2XTy+2XTXw+2λw=0.-2X^Ty + 2X^TXw + 2\lambda w = 0.

Hence,

(XTX+λI)w=XTy.(X^TX+\lambda I)w=X^Ty.

Therefore,

w=(XTX+λI)−1XTy.\boxed{ w = (X^TX+\lambda I)^{-1}X^Ty }.

---

## (b) Perspective 2: Shifting the Singular Values

Using

X=UΣVT,X=U\Sigma V^T,

we have

XTX=VΣTΣVT.X^TX = V\Sigma^T\Sigma V^T.

Since $V$ is orthonormal,

XTX+λI=V(ΣTΣ+λI)VT.X^TX+\lambda I = V(\Sigma^T\Sigma+\lambda I)V^T.

Therefore,

(XTX+λI)−1=V(ΣTΣ+λI)−1VT.(X^TX+\lambda I)^{-1} = V(\Sigma^T\Sigma+\lambda I)^{-1}V^T.

Also,

XT=VΣTUT.X^T=V\Sigma^TU^T.

Hence,

w=V(ΣTΣ+λI)−1ΣTUTy.\boxed{ w = V(\Sigma^T\Sigma+\lambda I)^{-1} \Sigma^TU^Ty }.

Along the $i$-th singular-vector direction, the effective coefficient is

σiσi2+λ.\frac{\sigma_i}{\sigma_i^2+\lambda}.

If

σi2≪λ,\sigma_i^2\ll\lambda,

then

σiσi2+λ≈σiλ,\frac{\sigma_i}{\sigma_i^2+\lambda} \approx \frac{\sigma_i}{\lambda},

which is close to zero. Therefore, directions associated with small singular values are strongly suppressed.

If

σi2≫λ,\sigma_i^2\gg\lambda,

then

σiσi2+λ≈1σi,\frac{\sigma_i}{\sigma_i^2+\lambda} \approx \frac{1}{\sigma_i},

which is approximately the ordinary least-squares solution.

---

## (c) Perspective 3: Maximum A Posteriori Estimation

The prior is

W∼N(0,I),W\sim\mathcal N(0,I),

and the data-generation model is

Y=XW+λN,Y=XW+\sqrt{\lambda}N,

where

N∼N(0,I).N\sim\mathcal N(0,I).

Thus,

Y∣W=w∼N(Xw,λI).Y\mid W=w \sim \mathcal N(Xw,\lambda I).

By Bayes' rule,

p(w∣y)∝p(y∣w)p(w).p(w\mid y) \propto p(y\mid w)p(w).

The MAP estimate is

wMAP=arg⁡max⁡wp(y∣w)p(w).w_{\text{MAP}} = \arg\max_w p(y\mid w)p(w).

Equivalently, we minimize the negative log posterior:

wMAP=arg⁡min⁡w[−log⁡p(y∣w)−log⁡p(w)].w_{\text{MAP}} = \arg\min_w \left[ -\log p(y\mid w) - \log p(w) \right].

For the Gaussian likelihood,

−log⁡p(y∣w)=12λ∥y−Xw∥22+C1.-\log p(y\mid w) = \frac{1}{2\lambda} \|y-Xw\|_2^2 + C_1.

For the Gaussian prior,

−log⁡p(w)=12∥w∥22+C2.-\log p(w) = \frac12\|w\|_2^2 + C_2.

Therefore,

wMAP=arg⁡min⁡w[12λ∥y−Xw∥22+12∥w∥22].w_{\text{MAP}} = \arg\min_w \left[ \frac{1}{2\lambda}\|y-Xw\|_2^2 + \frac12\|w\|_2^2 \right].

Multiplying the objective by $2\lambda$ does not change its minimizer:

wMAP=arg⁡min⁡w[∥y−Xw∥22+λ∥w∥22].\boxed{ w_{\text{MAP}} = \arg\min_w \left[ \|y-Xw\|_2^2 + \lambda\|w\|_2^2 \right] }.

Thus, the MAP estimate is exactly the ridge regression solution.

---

## (d) Perspective 4: Fake Data

Define

y^=[y0d]\hat y = \begin{bmatrix} y\\ 0_d \end{bmatrix}

and

X^=[XλId].\hat X = \begin{bmatrix} X\\ \sqrt{\lambda}I_d \end{bmatrix}.

The ordinary least-squares objective is

∥y^−X^w∥22.\|\hat y-\hat Xw\|_2^2.

Substituting the augmented matrices,

y^−X^w=[y−Xw−λw].\hat y-\hat Xw = \begin{bmatrix} y-Xw\\ -\sqrt{\lambda}w \end{bmatrix}.

Therefore,

∥y^−X^w∥22=∥y−Xw∥22+λ∥w∥22.\|\hat y-\hat Xw\|_2^2 = \|y-Xw\|_2^2 + \lambda\|w\|_2^2.

Thus,

arg⁡min⁡w∥y^−X^w∥22=arg⁡min⁡w[∥y−Xw∥22+λ∥w∥22].\boxed{ \arg\min_w \|\hat y-\hat Xw\|_2^2 = \arg\min_w \left[ \|y-Xw\|_2^2 + \lambda\|w\|_2^2 \right] }.

Hence, ridge regression can be interpreted as ordinary least squares on an augmented dataset.

---

## (e) Perspective 5: Fake Features

Define

Xˇ=[XλIn].\check X = \begin{bmatrix} X & \sqrt{\lambda}I_n \end{bmatrix}.

Let

η=[wv].\eta = \begin{bmatrix} w\\ v \end{bmatrix}.

The constraint

Xˇη=y\check X\eta=y

becomes

Xw+λv=y.Xw+\sqrt{\lambda}v=y.

Therefore,

v=y−Xwλ.v = \frac{y-Xw}{\sqrt{\lambda}}.

The minimum-norm objective is

∥η∥22=∥w∥22+∥v∥22.\|\eta\|_2^2 = \|w\|_2^2+\|v\|_2^2.

Substituting $v$,

∥η∥22=∥w∥22+1λ∥y−Xw∥22.\|\eta\|_2^2 = \|w\|_2^2 + \frac{1}{\lambda} \|y-Xw\|_2^2.

Multiplying by $\lambda$,

λ∥η∥22=∥y−Xw∥22+λ∥w∥22.\lambda\|\eta\|_2^2 = \|y-Xw\|_2^2 + \lambda\|w\|_2^2.

Multiplication by the positive constant $\lambda$ does not change the minimizer. Therefore, the first $d$ entries of the minimum-norm solution $\eta^*$ satisfy

w∗=arg⁡min⁡w[∥y−Xw∥22+λ∥w∥22].\boxed{ w^* = \arg\min_w \left[ \|y-Xw\|_2^2 + \lambda\|w\|_2^2 \right] }.

Thus, the first $d$ coordinates of $\eta^*$ are exactly the ridge regression solution.

---

## (f)

For the underdetermined system

Xˇη=y,\check X\eta=y,

the minimum-norm solution is

η∗=XˇT(XˇXˇT)−1y.\eta^* = \check X^T (\check X\check X^T)^{-1}y.

Since

Xˇ=[XλIn],\check X = \begin{bmatrix} X & \sqrt{\lambda}I_n \end{bmatrix},

we have

XˇXˇT=XXT+λIn.\check X\check X^T = XX^T+\lambda I_n.

Also,

XˇT=[XTλIn].\check X^T = \begin{bmatrix} X^T\\ \sqrt{\lambda}I_n \end{bmatrix}.

Thus,

η∗=[XTλIn](XXT+λIn)−1y.\eta^* = \begin{bmatrix} X^T\\ \sqrt{\lambda}I_n \end{bmatrix} (XX^T+\lambda I_n)^{-1}y.

The first $d$ coordinates are therefore

w^=XT(XXT+λIn)−1y.\boxed{ \hat w = X^T(XX^T+\lambda I_n)^{-1}y }.

Now we show that

XT(XXT+λIn)−1=(XTX+λId)−1XT.X^T(XX^T+\lambda I_n)^{-1} = (X^TX+\lambda I_d)^{-1}X^T.

Notice that

(XTX+λId)XT=XT(XXT+λIn).(X^TX+\lambda I_d)X^T = X^T(XX^T+\lambda I_n).

Right-multiplying by

(XXT+λIn)−1(XX^T+\lambda I_n)^{-1}

gives

(XTX+λId)XT(XXT+λIn)−1=XT.(X^TX+\lambda I_d) X^T(XX^T+\lambda I_n)^{-1} = X^T.

Left-multiplying by

(XTX+λId)−1(X^TX+\lambda I_d)^{-1}

gives

XT(XXT+λIn)−1=(XTX+λId)−1XT.X^T(XX^T+\lambda I_n)^{-1} = (X^TX+\lambda I_d)^{-1}X^T.

Therefore,

w^=(XTX+λId)−1XTy.\boxed{ \hat w = (X^TX+\lambda I_d)^{-1}X^Ty }.

This is exactly the standard ridge regression formula.

---

## (g)

The ridge regression solution is

w^r=(XTX+λI)−1XTy.\hat w_r = (X^TX+\lambda I)^{-1}X^Ty.

As

λ→∞,\lambda\rightarrow\infty,

the $\lambda I$ term dominates $X^TX$, so

XTX+λI≈λI.X^TX+\lambda I \approx \lambda I.

Therefore,

w^r≈1λXTy.\hat w_r \approx \frac{1}{\lambda}X^Ty.

Thus,

lim⁡λ→∞w^r=0.\boxed{ \lim_{\lambda\rightarrow\infty}\hat w_r=0 }.

This explains why ridge regularization is sometimes called **shrinkage**: increasing $\lambda$ shrinks the parameters toward zero.

---

## (h)

When

λ→0,\lambda\rightarrow0,

the behavior depends on the shape of $X$.

### Tall matrix

If $X$ is tall and has full column rank,

XTXX^TX

is invertible. Therefore,

lim⁡λ→0(XTX+λI)−1XTy=(XTX)−1XTy.\boxed{ \lim_{\lambda\rightarrow0} (X^TX+\lambda I)^{-1}X^Ty = (X^TX)^{-1}X^Ty }.

Thus, ridge regression approaches the ordinary least-squares solution.

### Wide matrix

If $X$ is wide, then $X^TX$ is generally singular, so we cannot simply replace $\lambda$ by zero in the usual ridge formula.

Using the equivalent expression,

w^=XT(XXT+λI)−1y,\hat w = X^T(XX^T+\lambda I)^{-1}y,

and assuming $X$ has full row rank,

lim⁡λ→0w^=XT(XXT)−1y.\lim_{\lambda\rightarrow0}\hat w = X^T(XX^T)^{-1}y.

Therefore,

lim⁡λ→0w^=X†y.\boxed{ \lim_{\lambda\rightarrow0}\hat w = X^\dagger y }.

This is the minimum-norm solution among all solutions satisfying $Xw=y$.

---

# 6. ReLU Elbow Update under SGD

## (a)

Consider

ϕ(x)={wx+b,wx+b>0,0,otherwise.\phi(x) = \begin{cases} wx+b, & wx+b>0,\\ 0, & \text{otherwise}. \end{cases}

### (i)

The elbow is the point where the pre-activation becomes zero:

we+b=0.we+b=0.

Thus,

e=−bw,\boxed{ e=-\frac{b}{w} },

assuming $w\neq0$.

### (ii)

The loss is

ℓ=12(ϕ(x)−y)2.\ell = \frac12(\phi(x)-y)^2.

Therefore,

dℓdϕ=ϕ(x)−y.\boxed{ \frac{d\ell}{d\phi} = \phi(x)-y }.

### (iii)

By the chain rule,

∂ℓ∂w=dℓdϕ∂ϕ∂w.\frac{\partial\ell}{\partial w} = \frac{d\ell}{d\phi} \frac{\partial\phi}{\partial w}.

When $wx+b>0$,

∂ϕ∂w=x.\frac{\partial\phi}{\partial w}=x.

Otherwise it is zero.

Therefore,

∂ℓ∂w=(ϕ(x)−y)x1{wx+b>0}.\boxed{ \frac{\partial\ell}{\partial w} = (\phi(x)-y)x \mathbf 1_{\{wx+b>0\}} }.

### (iv)

Similarly,

∂ϕ∂b=1{wx+b>0}.\frac{\partial\phi}{\partial b} = \mathbf 1_{\{wx+b>0\}}.

Thus,

∂ℓ∂b=(ϕ(x)−y)1{wx+b>0}.\boxed{ \frac{\partial\ell}{\partial b} = (\phi(x)-y) \mathbf 1_{\{wx+b>0\}} }.

---

## (b)

Suppose

ϕ(x)−y=1.\phi(x)-y=1.

For an active ReLU,

∂ℓ∂w=x,\frac{\partial\ell}{\partial w}=x,

and

∂ℓ∂b=1.\frac{\partial\ell}{\partial b}=1.

Therefore, after an SGD update with learning rate $\lambda$,

w′=w−λx\boxed{ w'=w-\lambda x }

and

b′=b−λ.\boxed{ b'=b-\lambda. }

The new elbow is

e′=−b−λw−λx.\boxed{ e' = -\frac{b-\lambda}{w-\lambda x} }.

### (i) $\phi(x)=0$

The ReLU is inactive. Since the derivative of ReLU is defined to be zero when its input is non-positive,

∂ℓ∂w=∂ℓ∂b=0.\frac{\partial\ell}{\partial w} = \frac{\partial\ell}{\partial b} = 0.

Therefore,

w′=w,b′=b,e′=e.\boxed{ w'=w,\qquad b'=b,\qquad e'=e. }

There is no change in either the slope or the elbow.

### (ii) $w>0,\ x>0,\ \phi(x)>0$

Since

w′=w−λx,w'=w-\lambda x,

and $x>0$,

w′<w.w'<w.

Thus, the positive slope decreases.

To determine how the elbow moves, compare $e'$ and $e$:

e′−e=−b−λw−λx+bw.e'-e = -\frac{b-\lambda}{w-\lambda x} + \frac{b}{w}.

Combining the terms,

e′−e=λ(w−bx)w(w−λx).e'-e = \frac{\lambda(w-bx)} {w(w-\lambda x)}.

Because the ReLU is active,

wx+b>0.wx+b>0.

Since $x>0$,

w−bx>w+wx2>0w-bx > w+w x^2 > 0

whenever the update is small enough that $w-\lambda x>0$.

Therefore,

e′>e.\boxed{ e'>e. }

The elbow moves to the right while the positive slope decreases.

### (iii) $w>0,\ x<0,\ \phi(x)>0$

Since $x<0$,

w′=w−λx>w.w' = w-\lambda x > w.

Therefore, the positive slope increases.

Again,

e′−e=λ(w−bx)w(w−λx).e'-e = \frac{\lambda(w-bx)} {w(w-\lambda x)}.

The active condition

wx+b>0wx+b>0

implies

b>−wx.b>-wx.

Multiplying by $x<0$ reverses the inequality:

bx<−wx2.bx<-wx^2.

Therefore,

w−bx>w+wx2>0.w-bx>w+wx^2>0.

Both terms in the denominator are positive, so

e′>e.\boxed{ e'>e. }

Thus, the slope becomes steeper and the elbow moves to the right.

### (iv) $w<0,\ x>0,\ \phi(x)>0$

Since

w′=w−λx,w'=w-\lambda x,

we have

w′<w<0.w'<w<0.

Thus, the slope becomes more negative, so its magnitude increases.

Using

e′−e=λ(w−bx)w(w−λx),e'-e = \frac{\lambda(w-bx)} {w(w-\lambda x)},

the active condition

wx+b>0wx+b>0

implies

b>−wx.b>-wx.

Because $x>0$,

bx>−wx2.bx>-wx^2.

Therefore,

w−bx<w+wx2=w(1+x2)<0.w-bx < w+wx^2 = w(1+x^2) < 0.

Meanwhile,

w(w−λx)>0w(w-\lambda x)>0

because both factors are negative.

Therefore,

e′<e.\boxed{ e'<e. }

Thus, the negative slope becomes steeper and the elbow moves to the left.

---

## (c)

For the full network, the pre-activation of the $i$-th hidden unit is

zi=Wi(1)x+bi.z_i = W_i^{(1)}x+b_i.

The elbow is located where

Wi(1)ei+bi=0.W_i^{(1)}e_i+b_i=0.

Thus,

ei=−biWi(1).\boxed{ e_i = -\frac{b_i}{W_i^{(1)}} }.

---

## (d)

Let

r=f^(x)−yr=\hat f(x)-y

denote the prediction error.

For the $i$-th hidden unit, define

Ii=1{Wi(1)x+bi>0}.I_i = \mathbf 1_{\{W_i^{(1)}x+b_i>0\}}.

The network output is

f^(x)=∑iWi(2)ϕ(Wi(1)x+bi).\hat f(x) = \sum_i W_i^{(2)} \phi(W_i^{(1)}x+b_i).

The derivative with respect to $W_i^{(1)}$ is

∂ℓ∂Wi(1)=rWi(2)xIi.\frac{\partial\ell}{\partial W_i^{(1)}} = rW_i^{(2)}xI_i.

Similarly,

∂ℓ∂bi=rWi(2)Ii.\frac{\partial\ell}{\partial b_i} = rW_i^{(2)}I_i.

After one SGD update,

Wi(1)′=Wi(1)−λrWi(2)xIi,W_i^{(1)\prime} = W_i^{(1)} - \lambda rW_i^{(2)}xI_i,

and

bi′=bi−λrWi(2)Ii.b_i' = b_i - \lambda rW_i^{(2)}I_i.

Therefore, the new elbow is

ei′=−bi′Wi(1)′.e_i' = -\frac{b_i'}{W_i^{(1)\prime}}.

Substituting the updates,

ei′=−bi−λ(f^(x)−y)Wi(2)IiWi(1)−λ(f^(x)−y)Wi(2)xIi.\boxed{ e_i' = - \frac{ b_i-\lambda(\hat f(x)-y)W_i^{(2)}I_i }{ W_i^{(1)} -\lambda(\hat f(x)-y)W_i^{(2)}xI_i } }.

If the $i$-th ReLU is inactive, then

Ii=0,I_i=0,

so its parameters are unchanged and

ei′=ei.\boxed{ e_i'=e_i. }

---

# 7. Coding Fully Connected Networks

Complete the provided `networks.ipynb` notebook using NumPy and export the completed notebook as a PDF.

## (a)

**Fill this part in after running the three-layer and five-layer networks.**

A possible answer, if it agrees with the experimental results, is:

> I found that the five-layer network was more difficult to train than the three-layer network. The deeper network was more sensitive to optimization choices and generally required more careful tuning to obtain good performance. This is consistent with the fact that gradients must propagate through more layers in the five-layer network.

---

# 8. Homework Process and Study Group

## (a)

I used the course materials, the assigned readings, and ChatGPT to discuss and check some of the mathematical derivations.

## (b)

I did not work with other students on this homework.

## (c)

Approximately **[fill in the actual number] hours**.