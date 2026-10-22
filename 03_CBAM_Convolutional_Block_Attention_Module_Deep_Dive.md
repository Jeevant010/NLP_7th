# CBAM: Convolutional Block Attention Module — In-Depth Paper Dissection

---

## 1. Paper Overview and Metadata

* **Full Title:** CBAM: Convolutional Block Attention Module
* **Authors:** Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon
* **Affiliations:** Korea Advanced Institute of Science and Technology (KAIST), Lunit Inc., Adobe Research
* **Conference:** European Conference on Computer Vision (ECCV), September 2018
* **arXiv Identifier:** 1807.06521 [cs.CV]

### 1.1 The Abstract
> "We propose Convolutional Block Attention Module (CBAM), a simple yet effective attention module for feed-forward convolutional neural networks. Given an intermediate feature map, our module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement. Because CBAM is a lightweight and general module, it can be integrated into any CNN architectures seamlessly with negligible overheads and is end-to-end trainable along with base CNNs. We validate our CBAM through extensive experiments on ImageNet-1K, MS COCO detection, and VOC 2007 detection datasets."

### 1.2 The Core Thesis
CBAM refines the dual-attention concept through two key principles:

1. **Dual-Pooling Representation:** Using **both Average Pooling and Max Pooling** preserves both global background context and sharp peak activations.
2. **Sequential Arrangement:** Computing **Channel Attention first**, followed immediately by **Spatial Attention**, is superior to parallel or spatial-first ordering.

```
                      CBAM Sequential Pipeline Architecture
                      
   Input Feature Map F ∈ R^(C x H x W)
               │
               ▼
   ┌───────────────────────────────────────────────┐
   │         CHANNEL ATTENTION SUB-MODULE          │
   │                                               │
   │   F ──┬──> AvgPool ──> [ Shared MLP ] ──┐     │
   │       │                                  (+) ─┼─> σ ──> Mc(F) ∈ R^(C x 1 x 1)
   │       └──> MaxPool ──> [ Shared MLP ] ──┘     │
   └───────────────────────────────────────────────┘
               │
               ▼
   Refined Intermediate Feature: F' = Mc(F) ⊗ F
               │
               ▼
   ┌───────────────────────────────────────────────┐
   │         SPATIAL ATTENTION SUB-MODULE          │
   │                                               │
   │   F' ──┬──> Channel AvgPool (1 x H x W) ──┐   │
   │        │                                  [Concat] ──> 7x7 Conv ──> σ ──> Ms(F')
   │        └──> Channel MaxPool (1 x H x W) ──┘   │
   └───────────────────────────────────────────────┘
               │
               ▼
   Final Refined Feature: F'' = Ms(F') ⊗ F'
```

---

## 2. From BAM to CBAM: The Two Architectural Breakthroughs

Following the development of BAM (BMVC 2018), the authors made two critical observations:

### Breakthrough 1: Dual Pooling (Average + Max)
Previous attention modules (including SENet and BAM) relied exclusively on Global Average Pooling to generate channel descriptors.

$$\text{AvgPool}(F) = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W F(c, i, j)$$

* **The Limitation of Average Pooling:** Average pooling aggregates spatial extent smoothly, but it acts as a low-pass filter. Small, high-contrast, discriminative features (such as an animal's eye, the sharp tip of a tool, or high-gradient corners) are averaged out across thousands of background pixels.
* **The Role of Max Pooling:** Max pooling extracts the most prominent, high-intensity activation:

$$\text{MaxPool}(F) = \max_{i, j} F(c, i, j)$$

Max pooling identifies whether a feature is strongly active anywhere in the spatial grid, regardless of the area it covers.

CBAM uses **both** Average Pooling and Max Pooling simultaneously, allowing the network to incorporate both global feature density and localized feature peaks.

---

### Breakthrough 2: Sequential Ordering (Channel First, Spatial Second)
In BAM, Channel Attention and Spatial Attention were computed in parallel from the same input tensor $F$ and combined by addition:

$$M(F) = \sigma(M_c(F) + M_s(F))$$

The CBAM authors found that structuring the operations **sequentially** is more effective:
1. **Channel Attention First:** The network determines *which* feature maps contain meaningful semantic categories and suppresses noisy channels.
2. **Spatial Attention Second:** Once the channels are calibrated, the spatial submodule determines *where* the remaining features are physically located across the 2D coordinate grid.

---

## 3. Mathematical Formulation of CBAM

Given an intermediate feature tensor:

$$F \in \mathbb{R}^{C \times H \times W}$$

The entire CBAM process is defined by two sequential matrix operations:

$$F' = M_c(F) \otimes F$$

$$F'' = M_s(F') \otimes F'$$

Where:
* $M_c(F) \in \mathbb{R}^{C \times 1 \times 1}$ is the 1D channel attention map.
* $M_s(F') \in \mathbb{R}^{1 \times H \times W}$ is the 2D spatial attention map.
* $\otimes$ denotes element-wise multiplication with broadcasting.
* $F'' \in \mathbb{R}^{C \times H \times W}$ is the final refined output tensor.

---

## 4. Deep-Dive into the CBAM Sub-Modules

### 4.1 The Channel Attention Sub-module

```
              Channel Attention Sub-module Detailed Data Flow
              
                           Input F ∈ R^(C x H x W)
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
      AvgPool along (H x W)                         MaxPool along (H x W)
               │                                             │
      F_avg^c ∈ R^(C x 1 x 1)                       F_max^c ∈ R^(C x 1 x 1)
               │                                             │
               ▼                                             ▼
       Shared MLP Layer 0                            Shared MLP Layer 0
       W0 ∈ R^(C/r x C)                              W0 ∈ R^(C/r x C)
               │                                             │
               ▼                                             ▼
             ReLU                                          ReLU
               │                                             │
               ▼                                             ▼
       Shared MLP Layer 1                            Shared MLP Layer 1
       W1 ∈ R^(C x C/r)                              W1 ∈ R^(C x C/r)
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                Element-wise
                                Addition (+)
                                      │
                                      ▼
                           Sigmoid Activation: σ(·)
                                      │
                                      ▼
                            Mc(F) ∈ R^(C x 1 x 1)
```

#### Step 1: Spatial Pooling
The input feature map $F$ is pooled along the spatial dimensions $(H \times W)$ using both average pooling and max pooling:

$$F^c_{avg} = \text{AvgPool}(F) \in \mathbb{R}^{C \times 1 \times 1}$$

$$F^c_{max} = \text{MaxPool}(F) \in \mathbb{R}^{C \times 1 \times 1}$$

#### Step 2: Shared Multi-Layer Perceptron (MLP)
Both vectors are forwarded through a shared two-layer MLP to minimize parameter overhead:
* The first linear projection reduces dimensionality by a reduction ratio $r = 16$: $W_0 \in \mathbb{R}^{\frac{C}{r} \times C}$.
* A $\text{ReLU}$ non-linearity is applied.
* The second linear projection restores dimensionality: $W_1 \in \mathbb{R}^{C \times \frac{C}{r}}$.

#### Step 3: Fusion and Activation
The output feature vectors are merged via element-wise addition, followed by a Sigmoid activation:

$$M_c(F) = \sigma\left(\text{MLP}\left(\text{AvgPool}(F)\right) + \text{MLP}\left(\text{MaxPool}(F)\right)\right)$$

Expanded mathematically:

$$M_c(F) = \sigma\left(W_1 \cdot \text{ReLU}\left(W_0 \cdot F^c_{avg}\right) + W_1 \cdot \text{ReLU}\left(W_0 \cdot F^c_{max}\right)\right)$$

Where:
* $W_0 \in \mathbb{R}^{\frac{C}{r} \times C}$
* $W_1 \in \mathbb{R}^{C \times \frac{C}{r}}$
* The weights $W_0$ and $W_1$ are shared across both pathways.

#### Step 4: Channel Recalibration
The intermediate calibrated tensor $F'$ is computed by scaling:

$$F'_{c, i, j} = M_c(F)_{c, 1, 1} \cdot F_{c, i, j}$$

---

### 4.2 The Spatial Attention Sub-module

```
              Spatial Attention Sub-module Detailed Data Flow
              
                      Intermediate Tensor F' ∈ R^(C x H x W)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         AvgPool along Channels                          MaxPool along Channels
                 │                                               │
         F_avg^s ∈ R^(1 x H x W)                         F_max^s ∈ R^(1 x H x W)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                               Concatenation along
                               Channel Axis: [ ; ]
                                         │
                                         ▼
                            Descriptor ∈ R^(2 x H x W)
                                         │
                                         ▼
                             7x7 Convolutional Layer
                                     f^(7x7)
                                 (Channels: 2 -> 1)
                                         │
                                         ▼
                              Sigmoid Activation: σ(·)
                                         │
                                         ▼
                               Ms(F') ∈ R^(1 x H x W)
```

The spatial attention sub-module identifies which spatial regions are most informative.

#### Step 1: Channel-Axis Pooling
Rather than using heavy dilated convolutions (as in BAM), CBAM computes spatial attention by pooling across the **channel dimension**:

1. **Channel Average Pooling:** At each spatial coordinate $(i, j)$, compute the mean value across all channels $C$:

$$F^s_{avg}(i, j) = \frac{1}{C} \sum_{c=1}^C F'_{c}(i, j) \quad \in \mathbb{R}^{1 \times H \times W}$$

2. **Channel Max Pooling:** At each spatial coordinate $(i, j)$, extract the maximum value across all channels $C$:

$$F^s_{max}(i, j) = \max_{c \in \{1, \dots, C\}} F'_{c}(i, j) \quad \in \mathbb{R}^{1 \times H \times W}$$

#### Step 2: Channel Concatenation
The two 2D spatial maps are concatenated along the channel axis to create a 2-channel spatial descriptor:

$$F^s_{concat} = \left[ F^s_{avg} \,;\, F^s_{max} \right] \in \mathbb{R}^{2 \times H \times W}$$

#### Step 3: Large-Kernel Spatial Convolution
A standard 2D convolution layer with a large receptive field kernel ($7 \times 7$) convolves over the 2-channel tensor, reducing it to a single channel:

$$M_s(F') = \sigma\left( f^{7 \times 7}\left( \left[ F^s_{avg} \,;\, F^s_{max} \right] \right) \right)$$

Where:
* $f^{7 \times 7}$ denotes a 2D convolution with kernel size $7 \times 7$ and padding of 3 (preserving spatial height $H$ and width $W$).
* $\sigma$ is the Sigmoid activation function.

#### Step 4: Final Refinement
The final output tensor $F''$ is obtained by point-wise scaling:

$$F''_{c, i, j} = M_s(F')_{1, i, j} \cdot F'_{c, i, j}$$

---

## 5. Exhaustive Paper Ablation Studies (ImageNet-1K)

The authors evaluated each component on ResNet-50 on the ImageNet-1K dataset (1.2 million images).

### 5.1 Ablation on Channel Attention Pooling Methods

The authors compared Average Pooling, Max Pooling, and their proposed joint combination:

#### Table 1: Channel Attention Methods on ResNet-50

| Configuration | Parameters | GFLOPs | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50 Baseline** | 25.56M | 3.86 | 24.56 | 7.50 |
| ResNet-50 + AvgPool Only (SE style) | 25.92M | 3.94 | 23.14 | 6.70 |
| ResNet-50 + MaxPool Only | 25.92M | 3.94 | 23.20 | 6.83 |
| **ResNet-50 + Both (AvgPool & MaxPool)** | **25.92M** | **4.02** | **22.80** | **6.52** |

#### Analytical Insight from Table 1:
Using both Average Pooling and Max Pooling reduced Top-1 error to **22.80%**, outperforming Average-only (23.14%) and Max-only (23.20%) by **0.34%** and **0.40%** respectively. Because the MLP weights are shared between both pooled vectors, this improvement required **zero additional parameters**.

---

### 5.2 Ablation on Spatial Attention Methods and Kernel Sizes

The authors investigated different ways to generate the spatial attention map, comparing channel pooling against $1 \times 1$ convolutions and testing different kernel sizes ($3 \times 3$ vs. $7 \times 7$):

#### Table 2: Spatial Attention Methods on ResNet-50

| Description | Parameters | GFLOPs | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-50 + Channel Attention Only (SE) | 28.09M | 3.860 | 23.14 | 6.70 |
| ResNet-50 + Channel Attention Only (Proposed) | 28.09M | 3.860 | 22.80 | 6.52 |
| + Spatial ($1 \times 1$ Conv, Kernel $3 \times 3$) | 28.10M | 3.862 | 22.96 | 6.64 |
| + Spatial ($1 \times 1$ Conv, Kernel $7 \times 7$) | 28.10M | 3.869 | 22.90 | 6.47 |
| + Spatial (Avg & Max Pooling, Kernel $3 \times 3$) | 28.09M | 3.863 | 22.68 | 6.41 |
| **+ Spatial (Avg & Max Pooling, Kernel $7 \times 7$)** | **28.09M** | **3.864** | **22.66** | **6.31** |

#### Analytical Insights from Table 2:
1. **Channel Pooling vs. $1 \times 1$ Conv:** Pooling along the channel axis (Avg & Max) achieved lower error (22.66%) than reducing channels via learned $1 \times 1$ convolutions (22.90%), while requiring fewer parameters.
2. **Receptive Field ($3 \times 3$ vs. $7 \times 7$):** The $7 \times 7$ spatial kernel outperformed the $3 \times 3$ kernel across all configurations. Contextual reasoning across a wider spatial neighborhood helps the model determine whether a pixel belongs to a target object or surrounding background.

---

### 5.3 Ablation on Attention Arrangement (Sequential vs. Parallel)

The authors evaluated the structural ordering of the Channel and Spatial modules:

#### Table 3: Module Arrangement Comparison on ResNet-50

| Ordering Strategy | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- |
| Channel Attention Only (SE) | 23.14 | 6.70 |
| Sequential: Spatial First $\to$ Channel Second | 22.78 | 6.42 |
| Parallel Combination (BAM style: $M_c + M_s$) | 22.95 | 6.59 |
| **Sequential: Channel First $\to$ Spatial Second (CBAM)** | **22.66** | **6.31** |

#### Why Channel-First Sequential Ordering Outperformed Alternatives:
* In the **Channel-First** configuration ($F \to M_c \to F' \to M_s \to F''$), the channel module first recalibrates feature channels globally, suppressing irrelevant feature planes. The spatial module then computes spatial weights based on cleaned, informative feature maps.
* In the **Spatial-First** configuration, spatial pooling must aggregate over all channels equally, meaning noise from uninformative channels degrades the spatial attention map.
* In the **Parallel** configuration, both modules operate on raw, unconditioned features independently, leading to sub-optimal interference when combined.

---

## 6. Large-Scale Benchmarks Across Architectures

### 6.1 ImageNet-1K Classification Results

The authors integrated CBAM into various backbone architectures on ImageNet-1K:

#### Table 4: ImageNet-1K Validation Errors (Single-Crop $224 \times 224$)

| Architecture | Parameters | GFLOPs | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** (Baseline) | 11.69M | 1.814 | 29.60 | 10.55 |
| ResNet-18 + SE | 11.78M | 1.814 | 29.41 | 10.22 |
| **ResNet-18 + CBAM** | **11.78M** | **1.815** | **29.27 (-0.33)** | **10.09 (-0.46)** |
| **ResNet-34** (Baseline) | 21.80M | 3.664 | 26.69 | 8.60 |
| ResNet-34 + SE | 21.96M | 3.664 | 26.13 | 8.35 |
| **ResNet-34 + CBAM** | **21.96M** | **3.665** | **25.99 (-0.70)** | **8.24 (-0.36)** |
| **ResNet-50** (Baseline) | 25.56M | 3.858 | 24.56 | 7.50 |
| ResNet-50 + SE | 28.09M | 3.860 | 23.14 | 6.70 |
| **ResNet-50 + CBAM** | **28.09M** | **3.864** | **22.66 (-1.90)** | **6.31 (-1.19)** |
| **ResNet-101** (Baseline) | 44.55M | 7.570 | 23.38 | 6.88 |
| ResNet-101 + SE | 49.33M | 7.575 | 22.35 | 6.19 |
| **ResNet-101 + CBAM** | **49.33M** | **7.581** | **21.51 (-1.87)** | **5.69 (-1.19)** |
| **WideResNet-18 (widen=2.0)** (Baseline) | 45.62M | 3.866 | 26.85 | 8.88 |
| WideResNet-18 + SE | 45.97M | 3.867 | 26.21 | 8.47 |
| **WideResNet-18 + CBAM** | **45.97M** | **3.868** | **26.10 (-0.75)** | **8.43 (-0.45)** |
| **ResNeXt-50 (32x4d)** (Baseline) | 25.03M | 3.768 | 22.85 | 6.48 |
| ResNeXt-50 + SE | 27.56M | 3.771 | 21.91 | 6.04 |
| **ResNeXt-50 + CBAM** | **27.56M** | **3.774** | **21.92 (-0.93)** | **5.91 (-0.57)** |
| **ResNeXt-101 (32x4d)** (Baseline) | 44.18M | 7.508 | 21.54 | 5.75 |
| ResNeXt-101 + SE | 48.96M | 7.512 | 21.17 | 5.66 |
| **ResNeXt-101 + CBAM** | **48.96M** | **7.519** | **21.07 (-0.47)** | **5.59 (-0.16)** |

---

### 6.2 Lightweight Models: MobileNet on ImageNet-1K

To demonstrate applicability on low-power devices, CBAM was integrated into MobileNet:

#### Table 5: MobileNet Experiments on ImageNet-1K

| Architecture | Parameters | GFLOPs | Top-1 Error (%) | Top-5 Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNet (Baseline)** | 3.23M | 0.569 | 31.70 | 11.90 |
| MobileNet + SE | 3.54M | 0.572 | 30.09 | 10.74 |
| **MobileNet + CBAM** | **3.54M** | **0.574** | **29.40 (-2.30)** | **10.60 (-1.30)** |
| **MobileNet ($\alpha = 0.7$)** | 1.99M | 0.285 | 36.80 | 15.30 |
| MobileNet ($\alpha = 0.7$) + SE | 2.14M | 0.286 | 35.10 | 14.10 |
| **MobileNet ($\alpha = 0.7$) + CBAM** | **2.14M** | **0.287** | **34.70 (-2.10)** | **13.80 (-1.50)** |

---

### 6.3 Object Detection: MS COCO and PASCAL VOC 2007

#### Table 6: MS COCO Validation Object Detection (Faster R-CNN)

| Backbone Architecture | Attention Module | mAP @ [.5, .95] | mAP @ .5 |
| :--- | :--- | :--- | :--- |
| ResNet-50 | Baseline | 36.4 | 58.2 |
| ResNet-50 | + SE | 37.1 (+0.7) | 59.0 (+0.8) |
| **ResNet-50** | **+ CBAM** | **37.3 (+0.9)** | **59.3 (+1.1)** |
| ResNet-101 | Baseline | 39.3 | 61.3 |
| ResNet-101 | + SE | 40.0 (+0.7) | 61.8 (+0.5) |
| **ResNet-101** | **+ CBAM** | **40.2 (+0.9)** | **62.3 (+1.0)** |

#### Table 7: PASCAL VOC 2007 Test Detection (StairNet Framework)

| Backbone Architecture | Detector Framework | Attention Module | mAP @ .5 (%) |
| :--- | :--- | :--- | :--- |
| VGG-16 | SSD | Baseline | 77.8 |
| VGG-16 | StairNet | Baseline | 78.8 |
| VGG-16 | StairNet | + SE | 79.1 (+0.3) |
| **VGG-16** | **StairNet** | **+ CBAM** | **79.4 (+0.6)** |
| MobileNet | SSD | Baseline | 68.1 |
| MobileNet | StairNet | Baseline | 70.3 |
| MobileNet | StairNet | + SE | 70.8 (+0.5) |
| **MobileNet** | **StairNet** | **+ CBAM** | **71.2 (+0.9)** |

---

## 7. Model Interpretability: Grad-CAM Analysis

To qualitatively verify that CBAM improves feature representations, the authors used **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize internal network activations:

### 7.1 Mathematical Formulation of Grad-CAM

Let $Y^c$ be the raw prediction score for class $c$ before the softmax. For a convolutional layer with feature activation maps $A^k \in \mathbb{R}^{H \times W}$:

1. Compute the importance weight $\alpha_k^c$ via global average pooling of gradients:

$$\alpha_k^c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i, j}^k}$$

2. Compute the Grad-CAM heat map by taking a weighted linear combination of forward activation maps followed by a $\text{ReLU}$:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c \cdot A^k \right)$$

### 7.2 Qualitative Visual Findings

```
   Grad-CAM Qualitative Comparison Across Architectures
   
   Input Image           ResNet-50 (Baseline)     ResNet-50 + SE           ResNet-50 + CBAM
  ┌──────────────┐      ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
  │              │      │       . : : .      │   │       . ::: .      │   │     .--------.     │
  │    [ Dog ]   │ ===> │     : : : : : :    │   │      : ::::: :     │   │    /  (o  o)   \    │
  │              │      │    : : : : : : :   │   │     : ::::::: :    │   │   |   ( == )   |   │
  │  Background  │      │   . . : : : : . .  │   │     . ::::::: .    │   │    \   \__/   /    │
  │    Grass     │      │ (Activations leak  │   │  (Stronger target  │   │   (Tightly bounded │
  │              │      │  into background)  │   │   focus, but soft) │   │    on target mask) │
  └──────────────┘      └────────────────────┘   └────────────────────┘   └────────────────────┘
```

1. **Baseline ResNet-50:** Class activation heatmaps were diffuse, often highlighting unrelated background regions (such as grass, sky, or nearby objects).
2. **ResNet-50 + SE:** The target object was emphasized more clearly than the baseline, but the heatmap remained blurry and lacked sharp boundaries.
3. **ResNet-50 + CBAM:** Class activations conformed tightly to the physical boundaries of target objects, demonstrating that the spatial submodule effectively suppresses background activations.

---

## 8. Architectural Comparison: BAM vs. CBAM vs. SENet

| Architectural Feature | SENet (CVPR 2018) | BAM (BMVC 2018) | CBAM (ECCV 2018) |
| :--- | :--- | :--- | :--- |
| **Attention Dimensions** | Channel only | Channel + Spatial | Channel + Spatial |
| **Module Placement** | Inside every residual block | At stage bottlenecks only | Inside every residual block |
| **Sub-Module Flow** | Single path | Parallel branches ($M_c + M_s$) | Sequential cascade ($M_c \to M_s$) |
| **Channel Pooling** | Average Pooling only | Average Pooling only | Dual: AvgPool + MaxPool |
| **Spatial Mechanism** | None (Spatial dimensions collapsed) | Cascaded dilated convs ($d=4$) | Channel-axis pooling + $7 \times 7$ conv |
| **Combination Operator** | Scale: $F \otimes s$ | Residual sum: $F + F \odot \sigma(M_c + M_s)$ | Sequential: $F' = M_c \otimes F$, $F'' = M_s \otimes F'$ |
| **Parameter Overhead** | Low (~10% in ResNet-50) | Extremely low (<1% at bottlenecks) | Extremely low (~0.1% for spatial) |
| **Empirical Accuracy** | Moderate gain (+0.7% on ImageNet) | Strong gain (+0.9% on ImageNet) | Best gain (+1.9% on ResNet-50) |

---

## 9. Complete PyTorch Reference Implementation of CBAM

Below is a complete, modular, and annotated PyTorch implementation of the CBAM module, along with an example of its integration into a standard ResNet Bottleneck block:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionCBAM(nn.Module):
    """
    CBAM Channel Attention Sub-module.
    Computes both Average Pooling and Max Pooling across spatial dimensions,
    passes both through a shared MLP, and sums their outputs.
    
    Formula:
        Mc(F) = Sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionCBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        # Shared MLP layers
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        # Spatial Max Pooling: (B, C, H, W) -> (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        
        # Forward through Shared MLP
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        # Element-wise addition and Sigmoid activation
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttentionCBAM(nn.Module):
    """
    CBAM Spatial Attention Sub-module.
    Pools along the channel axis (Average and Max), concatenates into a
    2-channel map, and applies a 7x7 convolution.
    
    Formula:
        Ms(F) = Sigmoid( Conv7x7( [AvgPool_c(F); MaxPool_c(F)] ) )
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionCBAM, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        # Convolves 2-channel descriptor into a 1-channel spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling along channel axis: (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along channel axis: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenation along channel axis: (B, 2, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Convolve and apply Sigmoid: (B, 1, H, W)
        spatial_att = self.sigmoid(self.conv(concat))
        return spatial_att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (ECCV 2018).
    Applies Channel Attention and Spatial Attention sequentially:
        F'  = Mc(F) * F
        F'' = Ms(F') * F'
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttentionCBAM(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttentionCBAM(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Channel Attention
        mc = self.channel_att(x)
        x_channel_refined = x * mc
        
        # Step 2: Spatial Attention applied to channel-refined features
        ms = self.spatial_att(x_channel_refined)
        x_final = x_channel_refined * ms
        
        return x_final


class CBAMResidualBlock(nn.Module):
    """
    Example integration: ResNet Bottleneck block with CBAM inserted
    before the residual skip connection addition.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super(CBAMResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAM(out_channels, reduction_ratio=16, spatial_kernel_size=7)
        
        # Identity shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Apply CBAM feature refinement
        out = self.cbam(out)
        
        # Residual addition
        out = self.relu(out + residual)
        return out


# Verification Test
if __name__ == "__main__":
    batch_size = 4
    in_channels = 256
    height, width = 28, 28
    
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    
    # 1. Test Standalone CBAM
    cbam_module = CBAM(in_channels=in_channels, reduction_ratio=16, spatial_kernel_size=7)
    cbam_output = cbam_module(dummy_input)
    assert cbam_output.shape == dummy_input.shape, "Shape mismatch in standalone CBAM!"
    print(f"Standalone CBAM output shape: {cbam_output.shape} (PASSED)")
    
    # 2. Test ResNet Block with CBAM
    block = CBAMResidualBlock(in_channels=256, mid_channels=64, out_channels=256, stride=1)
    block_output = block(dummy_input)
    assert block_output.shape == dummy_input.shape, "Shape mismatch in CBAM Residual Block!"
    print(f"CBAM Residual Block output shape: {block_output.shape} (PASSED)")
    print("All CBAM verifications passed successfully!")
```
