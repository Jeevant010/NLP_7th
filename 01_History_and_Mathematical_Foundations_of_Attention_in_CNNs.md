# History and Mathematical Foundations of Attention in Convolutional Neural Networks

---

## 1. The Classical Paradigm of Deep Feature Extraction (2012–2015)

The modern era of deep learning in computer vision began with the empirical success of deep convolutional architectures on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

### 1.1 The Mathematical Formulation of a 2D Convolution

In a standard convolutional layer, let:

```math
X \in \mathbb{R}^{C_{in} \times H \times W}
```

represent the input tensor, where $C_{in}$ is the number of input channels, $H$ is the spatial height, and $W$ is the spatial width.

Let the layer contain $C_{out}$ distinct convolutional filters parameterized by weights:

```math
W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}
```

and bias terms:

```math
b \in \mathbb{R}^{C_{out}}
```

For a spatial kernel of dimensions $K_h \times K_w$ (with centered kernel offsets $u \in [-\lfloor K_h/2 \rfloor, \lfloor K_h/2 \rfloor]$ and $v \in [-\lfloor K_w/2 \rfloor, \lfloor K_w/2 \rfloor]$), the forward response at output channel $k \in \{1, \dots, C_{out}\}$ and spatial coordinate $(i, j)$ is given by the cross-correlation operator:

```math
Y_{k, i, j} = \sum_{c=1}^{C_{in}} \sum_{u} \sum_{v} W_{k, c, u, v} \cdot X_{c, \, i+u, \, j+v} + b_k
```

Following the linear projection, a point-wise non-linear activation function $\phi(\cdot)$ (typically the Rectified Linear Unit, $\text{ReLU}(z) = \max(0, z)$) is applied:

```math
Z_{k, i, j} = \phi(Y_{k, i, j})
```

### 1.2 The Two Inherent Inductive Biases of Convolutions

Standard convolutions rely on two structural assumptions:

1. **Locality (Translation Equivariance):** Features are assumed to be stationary and localized. A filter detecting a vertical edge or texture pattern shifts across the image using the same kernel weights $W_{k, c, u, v}$.
2. **Channel-Uniform Aggregation:** When summing across channels $c \in \{1, \dots, C_{in}\}$, the contribution of each channel is determined purely by the static, learned kernel weights $W_{k, c, \cdot, \cdot}$, fixed across all training samples after optimization.

### 1.3 The Core Structural Limitation: Uniform Democratic Processing

Notice what Equation 1 implies:

At every spatial position $(i, j)$ across the entire image grid $H \times W$, the exact same mathematical transformation is performed. If an image contains an object of interest (e.g., a bird perched on a branch) occupying only 5% of the spatial area, the remaining 95% of the feature map (empty sky, blurred background) receives the identical computational budget and filter response.

Furthermore, within any layer, the network lacks an intrinsic, sample-dependent mechanism to dynamically amplify high-relevance feature channels while suppressing noisy or irrelevant feature channels.

---

## 2. Architectural Scaling: Depth, Width, and Cardinality (2015–2017)

To improve representation capacity, researchers initially scaled three structural dimensions of CNNs:

```
                      Structural Scaling Dimensions of CNNs
                      
                                 [ Network ]
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
     1. Depth                     2. Width                  3. Cardinality
   (Number of Layers)        (Channels per Layer)       (Grouped Transformations)
   e.g., ResNet-50/101/152      e.g., WideResNet             e.g., ResNeXt
```

### 2.1 Scaling Depth: Residual Networks (ResNet)

Before 2015, stacking dozens of layers led to the **degradation problem**: as network depth increased, accuracy saturated and then rapidly degraded. This was not caused by vanishing gradients (which were largely resolved by Batch Normalization), but by optimization difficulty: deep plain networks struggled to learn identity mappings.

He et al. (2015) introduced the **Residual Block**:

```math
y = \mathcal{F}(x, \{W_i\}) + x
```

Where:
* $x$ is the input tensor to the block.
* $\mathcal{F}(x, \{W_i\})$ is the residual mapping to be learned.
* $x$ is the identity shortcut connection.

During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to the input of block $l$ is:

```math
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{k=l}^{L-1} \mathcal{F}(x_k, \mathcal{W}_k) \right)
```

Because of the additive identity term $I$, gradients can flow directly from the final layer $L$ to any early layer $l$ without being attenuated by continuous matrix multiplications, enabling architectures of 50, 101, and 152 layers.

### 2.2 Scaling Width: Wide Residual Networks (WideResNet)

Zagoruyko and Komodakis (2016) demonstrated that simply increasing network depth had diminishing returns. They parameterized ResNet blocks with a widening factor $k$:

```math
C_{wide} = k \cdot C_{base}
```

A WideResNet with 28 layers and $k=10$ (WRN-28-10) outperformed a 1000-layer plain ResNet, demonstrating that channel capacity (richness of feature representations per layer) is as fundamental as depth.

### 2.3 Scaling Cardinality: ResNeXt

Xie et al. (2017) introduced **cardinality**—the size of the set of transformations. Instead of performing a single wide convolution or standard residual block, ResNeXt performs a split-transform-merge operation inspired by Inception modules:

```math
\mathcal{F}(x) = \sum_{i=1}^{C_{card}} \mathcal{T}_i(x)
```

Where each $\mathcal{T}_i(x)$ is a low-dimensional bottleneck projection. ResNeXt demonstrated that increasing cardinality is more effective at boosting accuracy than simply increasing depth or width.

---

## 3. The Cognitive Science Inspiration: Visual Attention Mechanisms

Despite scaling depth, width, and cardinality, CNNs remained fundamentally feed-forward and static:

```math
\text{Output} = \text{CNN}(X; \Theta)
```

Where the parameter set $\Theta$ is fixed after training. For any two images $X_A$ and $X_B$, the filters applied at layer $l$ are identical, regardless of whether $X_A$ is an isolated object or a crowded, cluttered scene.

### 3.1 The Primate Visual System

In cognitive neurobiology, the human visual system processes massive sensory inputs with finite biological energy by splitting visual attention into two distinct pathways:

```
                            Visual Cortex (V1 -> V2)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             Ventral Stream                        Dorsal Stream
             ("What" Pathway)                      ("Where" Pathway)
                    │                                     │
                    ▼                                     ▼
        * Feature identification               * Spatial localization
        * Object recognition                   * Motion analysis
        * Semantic properties                  * Spatial orientation
```

1. **The Ventral Stream ("What"):** Connects primary visual cortex (V1) to the inferior temporal cortex. It is specialized for feature discrimination, identifying object categories, colors, textures, and identities regardless of position.
2. **The Dorsal Stream ("Where"):** Connects V1 to the parietal lobe. It is specialized for spatial localization, tracking spatial relationships and guiding foveal saccades (directing eye focus to specific spatial coordinates).

### 3.2 Translating Biology to Convolutional Tensors

In a deep convolutional feature tensor $F \in \mathbb{R}^{C \times H \times W}$:

* **The Channel Dimension ($C$):** Each slice $F_{c, :, :} \in \mathbb{R}^{H \times W}$ is the spatial response map of a specific semantic filter. Some channels detect high-level concepts (e.g., eyes, wheels, fur), while others detect low-level textures (e.g., stripes, gradients). Channel-wise recalibration answers the question: **"WHAT feature channels are informative for this specific image?"**
* **The Spatial Dimensions ($H \times W$):** The 2D grid coordinates $(i, j)$ correspond to localized physical regions of the receptive field. Spatial recalibration answers the question: **"WHERE are the informative regions located in this feature map?"**

---

## 4. The Squeeze-and-Excitation Revolution (SENet, 2017)

The first architecture to successfully operationalize channel-wise dynamic recalibration in deep CNNs was **SENet (Squeeze-and-Excitation Networks)** by Jie Hu, Li Shen, and Gang Sun (CVPR 2018 / ILSVRC 2017 Winner).

```
                      The Squeeze-and-Excitation (SE) Block
                      
   Input Feature Map X ∈ R^(C x H x W)
               │
               ▼
   ┌───────────────────────┐
   │ Global Average Pool   │  ==> SQUEEZE: Collapses (H x W) to 1x1
   │ F_sq(·)               │      Vector z ∈ R^C
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ FC Layer 1 (C -> C/r) │
   │ ReLU Activation       │  ==> EXCITATION: Captures channel correlations
   │ FC Layer 2 (C/r -> C) │      Non-linear bottleneck MLP
   │ Sigmoid Activation    │
   └───────────────────────┘
               │
               ▼
       Weight Vector s ∈ R^C  (Values in [0, 1])
               │
               ▼
   ┌───────────────────────┐
   │ Scale Operator        │  ==> RECALIBRATION: Channel-wise product
   │ X_tilde = s ⊗ X       │      X_tilde_c = s_c · X_c
   └───────────────────────┘
               │
               ▼
   Refined Feature Map X_tilde ∈ R^(C x H x W)
```

### 4.1 The Three Mathematical Steps of SENet

#### Step 1: The Squeeze Operation $\mathbf{F}_{sq}(\cdot)$
Spatial information is aggregated across the spatial grid $H \times W$ into a channel descriptor vector $z \in \mathbb{R}^C$. The $c$-th element of $z$ is:

```math
z_c = \mathbf{F}_{sq}(u_c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)
```

Where $u_c \in \mathbb{R}^{H \times W}$ is the $c$-th channel of input tensor $U$.

#### Step 2: The Excitation Operation $\mathbf{F}_{ex}(\cdot, W)$
To capture non-linear, non-mutually-exclusive relationships across channels, the vector $z$ is passed through a two-layer bottleneck Multi-Layer Perceptron (MLP):

```math
s = \mathbf{F}_{ex}(z, W) = \sigma(g(z, W)) = \sigma\left(W_2 \cdot \delta\left(W_1 \cdot z\right)\right)
```

Where:
* $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ is a dimensionality-reduction linear layer with reduction ratio $r$ (typically $r = 16$).
* $\delta$ is the $\text{ReLU}$ activation function: $\delta(v) = \max(0, v)$.
* $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ is a dimensionality-increasing linear layer restoring dimensions to $C$.
* $\sigma$ is the Sigmoid activation function: $\sigma(a) = \frac{1}{1 + e^{-a}}$, ensuring that $s_c \in [0, 1]$.

#### Step 3: The Recalibration (Scale) Operation $\mathbf{F}_{scale}(\cdot, \cdot)$
The input feature map is re-scaled by channel weights:

```math
\widetilde{X}_c = \mathbf{F}_{scale}(u_c, s_c) = s_c \cdot u_c
```

Where $s_c$ is a scalar multiplier applied to the entire 2D matrix $u_c \in \mathbb{R}^{H \times W}$.

### 4.2 Gradient Dynamics of the SE Block

During the backward pass of training, the gradient of the loss $\mathcal{L}$ with respect to the pre-scaled feature map element $u_c(i, j)$ is:

```math
\frac{\partial \mathcal{L}}{\partial u_c(i, j)} = \frac{\partial \mathcal{L}}{\partial \widetilde{X}_c(i, j)} \cdot s_c + \sum_{m=1}^C \left( \sum_{p=1}^H \sum_{q=1}^W \frac{\partial \mathcal{L}}{\partial \widetilde{X}_m(p, q)} \cdot u_m(p, q) \right) \cdot \frac{\partial s_m}{\partial z_c} \cdot \frac{1}{HW}
```

Notice the two terms:
1. **The Direct Pathway:** $\frac{\partial \mathcal{L}}{\partial \widetilde{X}_c(i, j)} \cdot s_c$. The gradient is scaled directly by the learned attention weight $s_c$. If $s_c \to 0$ (irrelevant channel), backward gradients for that channel are attenuated.
2. **The Global Recalibration Pathway:** The second summation term backpropagates how altering $u_c(i, j)$ affects the global context $z_c$, which in turn alters the attention gating of all other channels $m$.

---

## 5. The Critical Blind Spot of SENet: Spatial Invariance

While SENet produced strong results on ImageNet, its mathematical design contained a fundamental limitation: **complete spatial blindness**.

### 5.1 Mathematical Proof of Spatial Permutation Invariance

Let $\pi$ be any arbitrary spatial permutation operator that shuffles the pixel coordinates $(i, j)$ across the 2D grid:

```math
\pi: \{1, \dots, H\} \times \{1, \dots, W\} \to \{1, \dots, H\} \times \{1, \dots, W\}
```

Let $X^{\pi}$ denote the spatially permuted feature map:

```math
X^{\pi}_{c, \, \pi(i, j)} = X_{c, i, j}
```

Now evaluate the Squeeze operation on $X^{\pi}$:

```math
z_c^{\pi} = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W X^{\pi}_c(i, j) = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W X_c(\pi^{-1}(i, j)) = \frac{1}{HW} \sum_{u=1}^H \sum_{v=1}^W X_c(u, v) = z_c
```

Because addition is commutative:

```math
z^{\pi} = z \implies s^{\pi} = s
```

```math
\text{SENet computes the exact same attention weight vector } s \text{ for a coherent image and a randomly scrambled image.}
```

```
   [ Coherent Feature Map ]              [ Scrambled Feature Map ]
   ┌──────────────────────┐              ┌──────────────────────┐
   │      [ Bird ]        │              │ .  #  .  .  *  .  .  │
   │    .   /\_/\   .     │              │ #  .  .  .  .  #  *  │
   │    .  ( o.o )  .     │              │ .  *  .  #  .  .  .  │
   │    .   > ^ <   .     │              │ *  .  .  .  *  #  .  │
   └──────────────────────┘              └──────────────────────┘
              │                                     │
       AvgPool Squeeze                       AvgPool Squeeze
              │                                     │
              ▼                                     ▼
       Vector z ∈ R^C                        Vector z ∈ R^C
              │                                     │
              └─────────────── EQUAL ───────────────┘
```

### 5.2 The Problem of Localized Salience

Consider an image where an object of interest covers a compact region of $10 \times 10$ pixels within a $100 \times 100$ feature map:
* The object occupies $\frac{100}{10,000} = 1\%$ of the spatial area.
* The remaining $99\%$ consists of background pixels.
* The global average:

```math
z_c = \frac{1}{10000} \left( \sum_{\text{object}} X_c(i, j) + \sum_{\text{background}} X_c(i, j) \right)
```

The informative features from the object are diluted by the background summation. SENet cannot selectively emphasize *where* to sample before computing channel salience, nor can it use channel salience to spatially segment the object.

---

## 6. The Formulation of the Dual Attention Problem

In early 2018, researchers at KAIST formulated the general **Dual Attention Problem** for deep neural networks:

> Given an arbitrary intermediate convolutional feature representation:
> 
> ```math
F \in \mathbb{R}^{C \times H \times W}
```
> 
> Can we construct an end-to-end, differentiable mapping that infers a 3D attention tensor:
> 
> ```math
M(F) \in \mathbb{R}^{C \times H \times W}
```
> 
> such that the refined feature map:
> 
> ```math
F' = \Phi(F, M(F))
```
> 
> simultaneously emphasizes informative channels ("What") and highlights informative spatial locations ("Where"), while adding negligible parameter and computational overhead?

### 6.1 The Dimensionality Challenge

Directly learning a full 3D attention tensor $M(F) \in \mathbb{R}^{C \times H \times W}$ without decomposition is computationally prohibitive. 

If a fully connected layer or standard 3D convolution were used to map:

```math
\mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H \times W}
```

The parameter complexity would scale as:

```math
\mathcal{O}(C^2 \cdot H^2 \cdot W^2)
```

For a standard ResNet stage with $C = 512, H = 28, W = 28$:
* Input dimensions: $512 \times 28 \times 28 = 401,408$ elements.
* A single linear projection would require $(401,408)^2 \approx 1.6 \times 10^{11}$ parameters (160 billion weights for one layer).

### 6.2 The Factorization Hypothesis

To make 3D attention computationally viable, the 3D attention distribution must be factorized into lower-dimensional sub-spaces:

```
                         Factorization Approaches
                         
                                [ M(F) ]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
    Parallel Factorization                                Sequential Factorization
       (Adopted by BAM)                                      (Adopted by CBAM)
         │                                                     │
         ├─ Mc(F) ∈ R^(C x 1 x 1)                              ├─ Step 1: Mc(F) ∈ R^(C x 1 x 1)
         ├─ Ms(F) ∈ R^(1 x H x W)                              │          F' = Mc(F) ⊗ F
         └─ M(F) = σ( Mc(F) + Ms(F) )                          └─ Step 2: Ms(F') ∈ R^(1 x H x W)
                                                                          F'' = Ms(F') ⊗ F'
```

1. **Parallel Factorization:** Compute a 1D Channel map $M_c(F) \in \mathbb{R}^{C \times 1 \times 1}$ and a 2D Spatial map $M_s(F) \in \mathbb{R}^{1 \times H \times W}$ along independent, concurrent pathways, then sum them:

```math
M(F) = \sigma\left(\text{Expand}(M_c(F)) + \text{Expand}(M_s(F))\right)
```

2. **Sequential Factorization:** Apply a two-stage cascade where one attention dimension is computed and applied first, followed by the second dimension:

```math
F' = M_c(F) \otimes F
```

```math
F'' = M_s(F') \otimes F'
```

The investigation of these two factorization schemes formed the basis of **BAM** and **CBAM**.
