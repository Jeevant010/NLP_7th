# BAM: Bottleneck Attention Module — In-Depth Paper Dissection

---

## 1. Paper Overview and Metadata

* **Full Title:** BAM: Bottleneck Attention Module
* **Authors:** Jongchan Park, Sanghyun Woo, Joon-Young Lee, and In So Kweon
* **Affiliations:** Lunit Inc., Korea Advanced Institute of Science and Technology (KAIST), Adobe Research
* **Conference:** British Machine Vision Conference (BMVC), September 2018
* **arXiv Identifier:** 1807.06514 [cs.CV]

### 1.1 The Abstract
> "Recent advances in deep neural networks have been developed via architecture search for stronger representational power. In this work, we focus on the effect of attention in general deep neural networks. We propose a simple and effective attention module, named Bottleneck Attention Module (BAM), that can be integrated with any feed-forward convolutional neural networks. Our module infers an attention map along two separate pathways, channel and spatial. We place our module at each bottleneck of models where the downsampling of feature maps occurs. Our module constructs a hierarchical attention at bottlenecks with a number of parameters and it is trainable in an end-to-end manner jointly with any feed-forward models."

### 1.2 The Core Thesis
BAM addresses two specific questions in deep network design:

1. **The Representation Question:** Can we infer both **Channel Attention** ("What to focus on") and **Spatial Attention** ("Where to focus") simultaneously using lightweight, parallel branches?
2. **The Placement Question:** Where is the most parameter-efficient location to place an attention module in a deep network? The paper argues that placing attention at the **bottlenecks** (where feature maps downsample and channels expand) maximizes representation gain while keeping computational overhead minimal.

---

## 2. Theoretical Motivation: Why the Bottlenecks?

In modern deep convolutional networks (such as ResNet, ResNeXt, and DenseNet), layers are organized hierarchically into distinct **stages**. 

```
                                 ResNet Backbone Stages
                                 
      Stage 1             Stage 2             Stage 3             Stage 4
  (H/4 x W/4, 256)    (H/8 x W/8, 512)   (H/16 x W/16, 1024) (H/32 x W/32, 2048)
 ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
 │  ResBlock x 3   │ │  ResBlock x 4   │ │  ResBlock x 6    │ │  ResBlock x 3    │
 └────────┬────────┘ └────────┬────────┘ └────────┬─────────┘ └────────┬─────────┘
          │                   │                   │                    │
          ▼                   ▼                   ▼                    ▼
     [ BAM 1 ]           [ BAM 2 ]           [ BAM 3 ]            [ BAM 4 ]
    (Transition)        (Transition)        (Transition)         (Transition)
```

Within each stage, multiple residual blocks maintain constant spatial resolution and channel depth. Between stages, a **bottleneck transition** occurs:
* Spatial dimensions are downsampled by a factor of 2: $(H, W) \to \left(\frac{H}{2}, \frac{W}{2}\right)$.
* Channel capacity is doubled: $C \to 2C$.

The BAM authors argued:
* **Feature Abstraction Hierarchy:** In early layers within a stage, low-level features are still forming. Inserting attention into every single $3 \times 3$ convolution block introduces unnecessary parameter redundancy and gradient noise.
* **Semantic Crossroads:** The bottleneck between stages represents a critical nexus. High-level semantics are compressed, and spatial coordinates are halved. Applying attention specifically at these transition boundaries allows the network to filter out noise and emphasize dominant spatial-channel signals before features enter the next hierarchical abstraction stage.

---

## 3. Mathematical Formulation of BAM

Let:

$$F \in \mathbb{R}^{C \times H \times W}$$

be an intermediate feature map extracted at a bottleneck of a convolutional neural network, where $C$ is the channel depth, $H$ is spatial height, and $W$ is spatial width.

BAM infers a 3D attention map:

$$M(F) \in \mathbb{R}^{C \times H \times W}$$

The refined feature map $F'$ is obtained through **residual scaling**:

$$F' = F + F \odot M(F)$$

Where $\odot$ denotes element-wise multiplication.

```
                      BAM Internal Architecture Overview
                      
                              Input Tensor F ∈ R^(C x H x W)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
                      ▼                                             ▼
         [ Channel Attention Branch ]                  [ Spatial Attention Branch ]
                      │                                             │
           Global Average Pooling                         1x1 Conv (C -> C/r)
                      │                                             │
             MLP (C -> C/r -> C)                       3x3 Dilated Conv (d=4)
                      │                                             │
             Batch Normalization                       3x3 Dilated Conv (d=4)
                      │                                             │
                      │                                    1x1 Conv (C/r -> 1)
                      │                                             │
                      │                                    Batch Normalization
                      │                                             │
                      ▼                                             ▼
               Mc(F) ∈ R^(C x 1 x 1)                         Ms(F) ∈ R^(1 x H x W)
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │
                                   Broadcasting Addition
                                     Mc(F) + Ms(F)
                                             │
                                             ▼
                                  Sigmoid Activation: σ(·)
                                             │
                                             ▼
                                  3D Attention Map M(F)
                                             │
                                 ┌───────────┴───────────┐
                                 │                       │
                                 ▼                       │
                             Scale (⊗)                   │
                          F ⊙ M(F)                       │
                                 │                       │
                                 ▼                       │
                            Addition (+) <───────────────┘
                                 │         (Residual Skip)
                                 ▼
                         Refined Tensor F'
```

### 3.1 The Importance of the Residual Formulation

Notice the mathematical structure of $F' = F + F \odot M(F)$:

$$\frac{\partial F'}{\partial F} = I + M(F) + F \odot \frac{\partial M(F)}{\partial F}$$

If the attention module initializes near zero ($M(F) \to 0$):

$$\frac{\partial F'}{\partial F} \approx I$$

This formulation provides two distinct properties:
1. **Performance Preservation:** The model cannot perform worse than the baseline network because the identity path $F$ is preserved.
2. **Stable Gradient Propagation:** Gradients can flow backwards through the identity shortcut without attenuation during the early phases of training.

---

## 4. Deep-Dive into the BAM Sub-Modules

BAM computes two separate pathways: the **Channel Attention Branch** $M_c(F)$ and the **Spatial Attention Branch** $M_s(F)$.

### 4.1 The Channel Attention Branch $M_c(F)$

The objective of the channel branch is to compute a scalar weight for each channel $c \in \{1, \dots, C\}$ reflecting its task relevance.

#### Step 1: Spatial Squeeze
Spatial dimensions are collapsed using Global Average Pooling:

$$F_c = \text{AvgPool}(F) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W F(c, i, j) \quad \in \mathbb{R}^{C \times 1 \times 1}$$

This vector softly encodes the global spatial distribution of each feature channel.

#### Step 2: Channel Correlation Modeling (MLP)
To learn inter-channel relationships while controlling parameter overhead, $F_c$ is passed through a two-layer Multi-Layer Perceptron (MLP) with a reduction ratio $r = 16$:

* The first linear layer reduces channels: $C \to \frac{C}{r}$.
* An activation function (ReLU) is applied.
* The second linear layer restores channels: $\frac{C}{r} \to C$.

#### Step 3: Batch Normalization for Scale Alignment
A critical detail in BAM is the application of **Batch Normalization (BN)** directly after the second linear projection:

$$M_c(F) = \text{BN}\left( W_1 \left( W_0 \cdot \text{AvgPool}(F) + b_0 \right) + b_1 \right)$$

Where:
* $W_0 \in \mathbb{R}^{\frac{C}{r} \times C}$
* $b_0 \in \mathbb{R}^{\frac{C}{r}}$
* $W_1 \in \mathbb{R}^{C \times \frac{C}{r}}$
* $b_1 \in \mathbb{R}^C$

**Why Batch Normalization in the Channel Branch?**  
Because the channel branch and spatial branch will be added together, their numerical scales must match. Without Batch Normalization, one branch could dominate the summation simply due to larger weight initialization or differing gradient magnitudes.

---

### 4.2 The Spatial Attention Branch $M_s(F)$

The objective of the spatial branch is to generate a 2D spatial weight map:

$$M_s(F) \in \mathbb{R}^{1 \times H \times W}$$

that emphasizes salient spatial locations and suppresses background noise.

To make an informed spatial decision, a feature pixel at $(i, j)$ must consider its surrounding spatial context. A standard $3 \times 3$ convolution has an effective receptive field of only $3 \times 3$, which is insufficient for global scene reasoning.

#### The Dilated (Atrous) Convolution Solution
BAM employs **dilated convolutions** to exponentially expand the receptive field without downsampling spatial resolution or increasing parameter counts.

For a 2D input $x$ and a filter $w$, a dilated convolution with dilation factor $d$ is defined as:

$$y[i, j] = \sum_{u=-K}^K \sum_{v=-K}^K x[i + d \cdot u, \, j + d \cdot v] \cdot w[u, v]$$

When $d = 1$, this reduces to a standard convolution. When $d > 1$, the kernel samples input pixels with a stride of $d$, inserting "holes" between filter taps.

#### Mathematical Derivation of the Receptive Field in BAM
Let $RF_{k}$ denote the receptive field of layer $k$, with kernel size $K = 3$ and dilation factor $d = 4$:

1. The effective kernel size $K'$ of a dilated filter is:

$$K' = K + (K - 1)(d - 1) = 3 + (3 - 1)(4 - 1) = 3 + 6 = 9$$

2. After the first dilated convolution ($k=1$):

$$RF_1 = 9 \times 9$$

3. After the second dilated convolution ($k=2$):

$$RF_2 = RF_1 + (K' - 1) = 9 + (9 - 1) = 17 \times 17$$

By cascading two $3 \times 3$ convolutions with $d = 4$, BAM covers a spatial window of **$17 \times 17$ pixels** at each point.

#### Step-by-Step Spatial Pathway
1. **Channel Compression ($1 \times 1$ Conv):** Squeezes channel dimension from $C \to \frac{C}{r}$ to reduce computational cost:

$$X_0 = f_0^{1 \times 1}(F) \quad \in \mathbb{R}^{\frac{C}{r} \times H \times W}$$

2. **First Dilated Convolution ($3 \times 3, d = 4$):** Aggregates local and semi-global context:

$$X_1 = f_1^{3 \times 3, d=4}(X_0) \quad \in \mathbb{R}^{\frac{C}{r} \times H \times W}$$

3. **Second Dilated Convolution ($3 \times 3, d = 4$):** Expands the receptive field to $17 \times 17$:

$$X_2 = f_2^{3 \times 3, d=4}(X_1) \quad \in \mathbb{R}^{\frac{C}{r} \times H \times W}$$

4. **Spatial Projection ($1 \times 1$ Conv):** Squeezes the channel dimension from $\frac{C}{r} \to 1$:

$$X_3 = f_3^{1 \times 1}(X_2) \quad \in \mathbb{R}^{1 \times H \times W}$$

5. **Batch Normalization:**

$$M_s(F) = \text{BN}(X_3) = \text{BN}\left( f_3^{1 \times 1}\left( f_2^{3 \times 3, d=4}\left( f_1^{3 \times 3, d=4}\left( f_0^{1 \times 1}(F) \right) \right) \right) \right)$$

---

### 4.3 Branch Combination: Broadcasting and Summation

With:
* $M_c(F) \in \mathbb{R}^{C \times 1 \times 1}$
* $M_s(F) \in \mathbb{R}^{1 \times H \times W}$

Both tensors are broadcast (tiled) to the full dimensions of $\mathbb{R}^{C \times H \times W}$:

$$\text{Expand}(M_c(F))_{c, i, j} = M_c(F)_{c, 1, 1}$$

$$\text{Expand}(M_s(F))_{c, i, j} = M_s(F)_{1, i, j}$$

The broadcast tensors are combined via **element-wise addition**, and a Sigmoid activation produces the final 3D attention map:

$$M(F) = \sigma\left( \text{Expand}(M_c(F)) + \text{Expand}(M_s(F)) \right)$$

Where:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Every point $M(F)_{c, i, j} \in (0, 1)$ represents the joint probability of relevance for channel $c$ at pixel coordinate $(i, j)$.

---

## 5. Experimental Analysis and Paper Ablations

The authors conducted extensive experiments on CIFAR-100, ImageNet-1K, PASCAL VOC 2007, and MS COCO.

### 5.1 Ablation Study on Hyperparameters (CIFAR-100 on ResNet-50)

The BAM architecture contains two primary hyperparameters:
1. The **dilation factor** $d$ in the spatial branch.
2. The **reduction ratio** $r$ in both branches.

#### Table 1(a): Dilation Factor and Reduction Ratio Analysis

| Variable | Tested Value | Parameters | Top-1 Error (%) |
| :--- | :--- | :--- | :--- |
| **Baseline (ResNet-50)** | — | **23.68M** | **21.49** |
| Dilation value ($d$) | $d = 1$ (Standard Conv) | 24.07M | 20.47 |
| Dilation value ($d$) | $d = 2$ | 24.07M | 20.28 |
| **Dilation value ($d$)** | **$d = 4$ (Optimal)** | **24.07M** | **20.00** |
| Dilation value ($d$) | $d = 6$ | 24.07M | 20.08 |
| Reduction ratio ($r$) | $r = 4$ | 26.30M | 20.46 |
| Reduction ratio ($r$) | $r = 8$ | 24.62M | 20.56 |
| **Reduction ratio ($r$)** | **$r = 16$ (Optimal)** | **24.07M** | **20.00** |
| Reduction ratio ($r$) | $r = 32$ | 23.87M | 21.24 |

#### Analytical Insights from Table 1(a):
* **Context is Essential ($d=1$ vs. $d=4$):** When $d=1$ (standard convolution without dilation), error is 20.47%. Increasing dilation to $d=4$ reduces error to 20.00%. A wider spatial receptive field allows the network to incorporate global context when evaluating spatial importance.
* **Overfitting at High Capacity ($r=4$ vs. $r=16$):** Although $r=4$ provides $4\times$ more parameters in the attention MLP, it yields higher error (20.46%) than $r=16$ (20.00%). Training loss converged in both cases, indicating that excessive attention capacity causes overfitting.

---

### 5.2 Ablation Study on Branch Combination Methods

The authors investigated whether both branches were necessary and compared different combination operators:

#### Table 1(b): Branch Effectiveness and Fusion Strategies

| Configuration | Fusion Method | Top-1 Error (%) |
| :--- | :--- | :--- |
| Baseline (ResNet-50) | None | 21.49 |
| Channel Branch Only | None | 20.28 |
| Spatial Branch Only | None | 20.21 |
| Both Branches | Element-wise Maximum (MAX) | 21.24 |
| Both Branches | Element-wise Product (PROD) | 21.29 |
| **Both Branches (BAM)** | **Element-wise Summation (SUM)** | **20.00** |

#### Why Element-Wise Summation Outperformed Product and Max:
1. **Summation:** In the forward pass, summation merges complementary information from both pathways without suppression. In the backward pass, gradients are distributed equally:
   $$\frac{\partial (M_c + M_s)}{\partial M_c} = 1, \quad \frac{\partial (M_c + M_s)}{\partial M_s} = 1$$
2. **Product ($M_c \cdot M_s$):** Multiplication couples gradients ($\frac{\partial (M_c M_s)}{\partial M_c} = M_s$). If one branch produces a small value, the gradient flowing through the other branch is suppressed, making optimization unstable.
3. **Maximum ($\max(M_c, M_s)$):** The max operator routes gradients exclusively to the higher-value branch, starving the other branch of training signal.

---

### 5.3 Placement Study: Bottlenecks vs. Every Convolutional Block

The authors compared placing BAM strictly at transition bottlenecks versus placing BAM inside every single convolution block (denoted as **BAM-C**):

#### Table 2: Placement Comparison on CIFAR-100

| Architecture | Variant | Parameters | GFLOPs | Top-1 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | Baseline | 23.71M | 1.22 | 21.49 |
| ResNet-50 | + BAM-C (Every Block) | 28.98M (+5.27M) | 1.37 | 20.88 |
| **ResNet-50** | **+ BAM (Bottlenecks Only)** | **24.07M (+0.36M)** | **1.25** | **20.00** |
| **WideResNet-28 (w=8)** | Baseline | 23.40M | 3.36 | 20.40 |
| WideResNet-28 (w=8) | + BAM-C (Every Block) | 23.78M (+0.38M) | 3.39 | 20.06 |
| **WideResNet-28 (w=8)** | **+ BAM (Bottlenecks Only)** | **23.42M (+0.02M)** | **3.37** | **19.06** |
| **ResNeXt-29 (8x64d)** | Baseline | 34.52M | 4.99 | 18.18 |
| ResNeXt-29 (8x64d) | + BAM-C (Every Block) | 35.60M (+1.08M) | 5.07 | 18.15 |
| **ResNeXt-29 (8x64d)** | **+ BAM (Bottlenecks Only)** | **34.61M (+0.09M)** | **5.00** | **16.71** |

#### Crucial Finding from Table 2:
Placing BAM inside every block (BAM-C) added substantially more parameters (e.g., +5.27M on ResNet-50), yet achieved **worse** error (20.88%) than placing BAM strictly at the bottlenecks (20.00% with only +0.36M parameters).

*Conclusion:* Applying attention at the macro-level bottleneck transitions provides a cleaner information bottleneck for feature selection than continuous intra-block gating.

---

### 5.4 ImageNet-1K Benchmark Results

On large-scale ImageNet-1K classification (1.2 million images, single-crop validation):

#### Table 3: ImageNet-1K Classification Results

| Architecture | Parameters | GFLOPs | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** (Baseline) | 11.69M | 1.81 | 29.60 | 10.55 |
| **ResNet-18 + BAM** | 11.71M (+0.02M) | 1.82 (+0.01) | **28.88 (-0.72)** | **10.01 (-0.54)** |
| **ResNet-50** (Baseline) | 25.56M | 3.86 | 24.56 | 7.50 |
| **ResNet-50 + BAM** | 25.92M (+0.36M) | 3.94 (+0.08) | **24.02 (-0.54)** | **7.18 (-0.32)** |
| **ResNet-101** (Baseline) | 44.55M | 7.57 | 23.38 | 6.88 |
| **ResNet-101 + BAM** | 44.91M (+0.36M) | 7.65 (+0.08) | **22.44 (-0.94)** | **6.29 (-0.59)** |

---

## 6. Complete PyTorch Reference Implementation of BAM

Below is an annotated, fully functional PyTorch implementation of the complete BAM module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttentionBAM(nn.Module):
    """
    Channel Attention Branch for BAM.
    Applies Global Average Pooling -> MLP (with reduction ratio r) -> Batch Normalization.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        # Batch Normalization aligns the scale with the spatial branch
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        x_pool = F.adaptive_avg_pool2d(x, (1, 1))
        # MLP: (B, C) -> (B, C)
        x_att = self.mlp(x_pool)
        # BN: (B, C) -> (B, C)
        x_att = self.bn(x_att)
        # Reshape to (B, C, 1, 1) for broadcasting
        return x_att.unsqueeze(2).unsqueeze(3)


class SpatialAttentionBAM(nn.Module):
    """
    Spatial Attention Branch for BAM.
    Applies:
      1. 1x1 Conv (C -> C/r)
      2. 3x3 Dilated Conv (dilation=4, padding=4)
      3. 3x3 Dilated Conv (dilation=4, padding=4)
      4. 1x1 Conv (C/r -> 1)
      5. Batch Normalization
    Covering an effective receptive field of 17x17.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, dilation: int = 4):
        super(SpatialAttentionBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.spatial_conv = nn.Sequential(
            # 1x1 Conv channel reduction
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # Dilated Conv 1
            nn.Conv2d(
                reduced_channels, reduced_channels, 
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # Dilated Conv 2
            nn.Conv2d(
                reduced_channels, reduced_channels, 
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 Conv spatial projection
            nn.Conv2d(reduced_channels, 1, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (B, 1, H, W)
        x_att = self.spatial_conv(x)
        return self.bn(x_att)


class BAM(nn.Module):
    """
    Bottleneck Attention Module (BMVC 2018).
    Combines Channel Attention and Spatial Attention via parallel summation,
    followed by Sigmoid activation and residual addition.
    
    Formula:
        M(F) = Sigmoid( Expand(Mc(F)) + Expand(Ms(F)) )
        Output = F + F * M(F)
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, dilation: int = 4):
        super(BAM, self).__init__()
        self.channel_att = ChannelAttentionBAM(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttentionBAM(in_channels, reduction_ratio, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mc(F): (B, C, 1, 1)
        mc = self.channel_att(x)
        # Ms(F): (B, 1, H, W)
        ms = self.spatial_att(x)
        
        # Broadcasting addition produces 3D attention map: (B, C, H, W)
        attention_map = torch.sigmoid(mc + ms)
        
        # Residual scaling
        return x + x * attention_map


# Verification Test
if __name__ == "__main__":
    # Simulate an intermediate feature map from ResNet-50 Stage 3 bottleneck
    batch_size = 4
    channels = 1024
    height, width = 14, 14
    
    dummy_input = torch.randn(batch_size, channels, height, width)
    bam_module = BAM(in_channels=channels, reduction_ratio=16, dilation=4)
    
    output = bam_module(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert dummy_input.shape == output.shape, "Shape mismatch in BAM forward pass!"
    print("BAM verification successful!")
```
