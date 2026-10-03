
## 1. Disadvantages of Previous Methods

### 1.1 Pretrained Methods such as Ditto

Ditto is a feed-forward method trained on articulated objects from several known categories.

It learns priors about:

- object shapes
- part segmentation
- joint structures
- typical motions of movable parts

The advantage is that inference can be relatively straightforward because the model already has prior knowledge.

However, the main problem is that its performance depends on the training distribution.

If a test object is very different from the categories seen during training, the learned priors may not generalize well.

Therefore:

$$
\boxed{
\text{Pretraining provides useful priors, but limits out-of-distribution generalization}
}
$$

This is especially problematic for articulated objects because large-scale annotated 3D articulated-object datasets are difficult to collect.

So a model trained on familiar articulated categories may struggle with arbitrary unknown objects whose:

- shape
- size
- structure
- motion

are very different from the training set.

---

### 1.2 PARIS: Better Generalization, but Unstable Optimization

PARIS avoids category-specific pretraining.

Instead, it performs **per-object optimization** directly on the observations of each new object.

PARIS models:

- a static-part neural field
- a mobile-part neural field
- the transformation of the mobile part

and optimizes them jointly using an image-rendering objective.

This improves generalization because PARIS does not need the object to belong to a category seen during training.

However, this introduces another problem:

$$
\boxed{
\text{The optimization is highly sensitive to initialization}
}
$$

PARIS jointly optimizes many unknown quantities:

$$
\theta
=
\{
\text{geometry},
\text{part segmentation},
\text{motion},
\text{joint parameters},
\dots
\}
$$

Different combinations of these variables may still produce similar rendered RGB images.

For example, the correct solution may be:

$$
\text{cabinet body}=\text{static part}
$$

$$
\text{door}=\text{mobile part}
$$

But an incorrect solution may assign part of the cabinet body to the mobile part.

Because geometry, segmentation, and motion are optimized together, the neural fields may partially compensate for each other.

Therefore:

$$
\text{wrong segmentation}
+
\text{wrong geometry}
+
\text{wrong motion}
$$

may still produce:

$$
\text{low image-rendering loss}
$$

This means that the optimization landscape contains many bad low-loss solutions.

As a result:

$$
\theta_0^{(1)}
\rightarrow
\text{good solution}
$$

while

$$
\theta_0^{(2)}
\rightarrow
\text{bad solution}
$$

even when the input object is exactly the same.

Therefore PARIS can produce very different results under different random initializations.

---

### 1.3 PARIS and Ditto Only Handle Simple Two-Part Objects

Another limitation is that original PARIS and Ditto mainly assume:

$$
\text{one static part}
+
\text{one movable part}
$$

Therefore, they cannot naturally represent objects with multiple independently moving parts.

For example:

- a cabinet with multiple drawers
- an object with several doors
- a mechanism with several movable components

cannot be naturally handled by the original formulation.

---

# 2. How This Paper Solves These Problems

## 2.1 Keep Per-Object Optimization to Preserve Generalization

This paper keeps the main advantage of PARIS:

$$
\boxed{
\text{per-object optimization}
}
$$

It does not rely on a pretrained category-specific articulation model.

Instead, every new object is reconstructed directly from its own RGB-D observations.

Therefore:

$$
\boxed{
\text{No strong dependence on object-category priors}
}
$$

The method can therefore be applied to unknown articulated objects whose shape, size, or motion may be very different from objects seen during training.

The generalization does not come from learning a universal pretrained model.

Instead, it comes from:

$$
\boxed{
\text{optimizing directly on each new object using generic geometric and physical constraints}
}
$$

---

## 2.2 Separate Geometry Reconstruction from Articulation Reconstruction

The most important improvement over PARIS is that this paper does **not optimize geometry, segmentation, and motion together from the beginning**.

Instead, the problem is divided into two stages.

### Stage 1: Object-Level Reconstruction

The method first reconstructs the whole object independently at the two articulation states:

$$
O^0
\qquad
\text{and}
\qquad
O^1
$$

At this stage, the model only reconstructs:

- the whole-object geometry
- the whole-object appearance

It does not yet determine:

- which point belongs to which part
- which part moves
- the joint structure

The geometry is represented using neural SDF fields.

Therefore:

$$
\boxed{
\text{Stage 1: reconstruct reliable geometry first}
}
$$

---

### Stage 2: Articulation Reconstruction

Only after the geometry has already been reconstructed does the method estimate:

- part segmentation
- rigid motion of each part
- articulation parameters

Therefore:

$$
\boxed{
\text{geometry reconstruction}
\rightarrow
\text{articulation reasoning}
}
$$

instead of:

$$
\text{geometry + segmentation + motion optimized together}
$$

This reduces ambiguity.

In PARIS, wrong segmentation can be compensated for by changing the geometry.

In this paper, the geometry has already been reconstructed before articulation optimization begins, so the articulation optimizer has much less freedom to create a wrong but low-loss solution.

---

## 2.3 Explicitly Model Point Correspondence

The core idea of Stage 2 is to explicitly model where every 3D point moves between the two articulation states.

The method learns:

$$
P^t(x,i)
$$

where:

$$
P^t(x,i)
=
\text{probability that point }x\text{ belongs to part }i
$$

Each part also has a rigid transformation:

$$
T_i^t=(R_i^t,t_i^t)
$$

The predicted position of point $x$ in the other state is:

$$
x^{t\rightarrow t'}
=
\sum_i
P^t(x,i)
\left(
R_i^t x+t_i^t
\right)
$$

Therefore the model explicitly constructs:

$$
\boxed{
x\rightarrow x'
}
$$

between the two states.

This means that the predicted segmentation and motion can now be directly checked against the actual observations.

---

# 3. Use Multiple Constraints Instead of Only Image Rendering

PARIS mainly relies on image-rendering supervision.

The problem is that many incorrect geometric or articulation configurations may still produce similar rendered images.

This paper instead constrains the point-correspondence field using multiple types of information:

$$
L
=
\lambda_{\text{cns}}L_{\text{cns}}
+
\lambda_{\text{match}}L_{\text{match}}
+
\lambda_{\text{coll}}L_{\text{coll}}
$$

These constraints make incorrect solutions much harder to satisfy.

---

## 3.1 Consistency Loss

If:

$$
x
$$

in state 0 corresponds to:

$$
x'
$$

in state 1, then they should represent the same physical location on the object.

Therefore, their local properties should be consistent.

### SDF Consistency

The geometry around corresponding points should be similar:

$$
\tilde{\Omega}^0(x)
\approx
\tilde{\Omega}^1(x')
$$

### RGB Consistency

Their appearance should also be similar:

$$
\Phi^0(x)
\approx
\Phi^1(x')
$$

### Occupancy Consistency

Occupancy consistency means: **if a 3D point is occupied by the object in one state, then its predicted corresponding point in the other state should also be occupied by the object**.

The paper defines an occupancy field roughly as:

$$Occ(x)\in[0,1]$$

where:

- $Occ(x)\approx 1$: point $x$ is inside or very close to the object
- $Occ(x)\approx 0$: point $x$ is empty space

Then, if the articulation model predicts

$$x \rightarrow x',$$

the paper wants:

$$Occ^t(x) \approx Occ^{t'}(x').$$

So the occupancy consistency loss is basically:

$$L_{\text{occ}} = \left\| Occ^t(x) - Occ^{t'}(x') \right\|_2^2$$

The paper introduces this especially for points **away from the surface**, because SDF/color supervision is more reliable near the surface, while occupancy can supervise the broader 3D volume.

Therefore, if an incorrect segmentation causes a point to move to an incompatible 3D location, the consistency loss increases.

---

## 3.2 Matching Loss

Consistency alone is not enough.

Two different regions may have similar:

- geometry
- color
- local appearance

Therefore, the paper also uses 2D feature correspondences obtained from LoFTR.

Suppose LoFTR finds:

$$
p
\leftrightarrow
q
$$

between images from the two states.

The method:

1. finds the 3D point associated with pixel $p$
2. moves it using the predicted articulation
3. projects the transformed point into the image of the other state
4. checks whether it lands close to $q$

If the articulation is correct:

$$
\hat q
\approx
q
$$

Therefore:

$$
\boxed{
\text{2D image correspondence provides additional guidance for 3D motion estimation}
}
$$

This helps prevent the optimizer from choosing a wrong correspondence that happens to have similar local geometry.

---

## 3.3 Collision Loss

Even consistency loss and matching loss may not completely eliminate wrong segmentations.

For example, suppose part of the cabinet body is incorrectly assigned to the moving door.

Then this region moves together with the door.

After applying the predicted motion, this wrongly moving geometry may overlap with another part of the object.

Therefore:

$$
\text{wrong segmentation}
$$

$$
\Downarrow
$$

$$
\text{wrong points move}
$$

$$
\Downarrow
$$

$$
\text{different object parts collide}
$$

$$
\Downarrow
$$

$$
L_{\text{collision}}\uparrow
$$

So collision loss adds a physical / kinematic constraint.

It helps reject solutions that may look acceptable from image rendering or local correspondence but are physically impossible.

---

# 4. Why the Optimization Becomes More Stable

PARIS mainly requires:

$$
\text{rendered image}
\approx
\text{observed image}
$$

There may be many incorrect solutions satisfying this condition.

This paper instead requires the articulation model to simultaneously satisfy:

$$
\begin{aligned}
&\text{3D geometry consistency}\\
&\text{RGB consistency}\\
&\text{occupancy consistency}\\
&\text{2D feature correspondence}\\
&\text{collision-free motion}
\end{aligned}
$$

Therefore, many wrong low-loss solutions in PARIS become high-loss solutions in this method.

Conceptually:

$$
\boxed{
\text{more independent constraints}
\rightarrow
\text{fewer plausible wrong solutions}
}
$$

which leads to:

$$
\boxed{
\text{less sensitivity to initialization}
\rightarrow
\text{more stable optimization}
}
$$

Importantly, the paper does **not** completely remove initialization dependence.

It still performs per-object optimization.

The improvement is that the optimization problem becomes much more constrained, so different random initializations are more likely to converge to the correct articulation model.

---

# 5. Support Multiple Moving Parts

Instead of assuming only:

$$
\text{one static part}
+
\text{one movable part}
$$

the paper defines:

$$
P(x,i),
\qquad
i=0,\dots,M-1
$$

for multiple parts.

Each part has its own rigid transformation:

$$
T_i
$$

Therefore the model can represent:

$$
\text{static base}
+
\text{moving part}_1
+
\text{moving part}_2
+
\cdots
$$

This allows the method to handle objects containing multiple independently moving rigid parts.

---

# 6. Overall Comparison

## Ditto

### Problem

Depends on pretrained object-category priors.

$$
\text{training distribution}
\rightarrow
\text{learned prior}
\rightarrow
\text{limited OOD generalization}
$$

### This Paper's Solution

Avoid category-specific pretraining and optimize directly on each new object.

$$
\boxed{
\text{per-object optimization}
}
$$

---

## PARIS

### Problem

Jointly optimizes:

$$
\text{geometry}
+
\text{segmentation}
+
\text{motion}
$$

mainly through image-rendering supervision.

Therefore, different wrong combinations can still produce similar rendered images.

This creates many bad low-loss solutions and makes optimization sensitive to random initialization.

### This Paper's Solution

First reconstruct geometry:

$$
\boxed{
\text{Stage 1: Object-Level Reconstruction}
}
$$

Then reconstruct articulation:

$$
\boxed{
\text{Stage 2: Part Segmentation + Part Motion}
}
$$

and explicitly construct point correspondences:

$$
\boxed{
x\rightarrow x'
}
$$

These correspondences are constrained by:

$$
\boxed{
L_{\text{consistency}}
+
L_{\text{matching}}
+
L_{\text{collision}}
}
$$

Therefore:

$$
\boxed{
\text{more constraints}
\rightarrow
\text{fewer wrong low-loss solutions}
\rightarrow
\text{more stable optimization}
}
$$

---

## PARIS and Ditto

### Problem

Mainly designed for:

$$
\text{one static part}
+
\text{one movable part}
$$

### This Paper's Solution

Use:

$$
P(x,i)
$$

for multiple part labels and:

$$
T_i
$$

for the motion of each part.

Therefore:

$$
\boxed{
\text{multiple movable parts can be reconstructed}
}
$$

---

# Core Idea

The main contribution can be summarized as:

$$
\boxed{
\text{Keep the generalization advantage of per-object optimization}
}
$$

while solving its instability by:

$$
\boxed{
\text{separating geometry reconstruction from articulation reasoning}
}
$$

and:

$$
\boxed{
\text{explicitly supervising 3D point correspondences with geometric, visual, and physical constraints}
}
$$

Therefore, compared with previous methods, the proposed method achieves:

$$
\boxed{
\text{better generalization}
+
\text{more stable optimization}
+
\text{support for multiple movable parts}
}
$$