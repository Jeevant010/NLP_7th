# Part 3: CBAM (Convolutional Block Attention Module) — In-Depth Paper Dissection

---

## 1. Paper Overview and Metadata

* **Full Title:** CBAM: Convolutional Block Attention Module
* **Authors:** Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon
* **Affiliations:** Korea Advanced Institute of Science and Technology (KAIST), Lunit Inc., Adobe Research
* **Conference:** European Conference on Computer Vision (ECCV), September 2018
* **arXiv Identifier:** 1807.06521 [cs.CV]

### 1.1 The High-Level Idea
CBAM refined the dual-attention paradigm into its most widely adopted form:

```text
                      The CBAM Sequential Pipeline
                      
   Input Tensor F : [ B, C, H, W ]
               │
               ▼
   ┌──────────────────────────────────────────────┐
   │ 1. CHANNEL ATTENTION SUB-MODULE              │
   │    Computes "WHAT" channels are informative   │
   │    Uses BOTH Average Pooling AND Max Pooling │
   │    Output: Mc(F) of shape [ B, C, 1, 1 ]     │
   └──────────────────────────────────────────────┘
               │
               ▼
   Calibrate Channels: F' = Mc(F) * F
               │
               ▼
   ┌──────────────────────────────────────────────┐
   │ 2. SPATIAL ATTENTION SUB-MODULE              │
   │    Computes "WHERE" to focus in the 2D plane │
   │    Pools along the channel axis              │
   │    Applies a 7x7 Convolution                 │
   │    Output: Ms(F') of shape [ B, 1, H, W ]    │
   └──────────────────────────────────────────────┘
               │
               ▼
   Calibrate Space: F'' = Ms(F') * F'
               │
               ▼
   Final Refined Tensor F'' : [ B, C, H, W ]
```

---

## 2. The Two Architectural Breakthroughs of CBAM

Following the publication of BAM, the authors made two fundamental discoveries:

### Breakthrough 1: Dual Pooling (Average + Max)
Previous attention modules (SENet and BAM) relied exclusively on **Average Pooling** to summarize spatial maps into channel vectors.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   WHY AVERAGE POOLING IS INSUFFICIENT                  │
│                                                                        │
│   Average Pooling calculates the mean:                                 │
│       Mean = ( 1 / (H * W) ) * Sum( all pixels )                       │
│                                                                        │
│   * STRENGTH: Gathers global feature density and smooth background     │
│     statistics.                                                        │
│                                                                        │
│   * WEAKNESS: It acts as a low-pass filter. If a feature channel       │
│     detects a small, sharp, distinctive cue (like an eye, a sharp      │
│     beak, or a small defect) occupying only 20 pixels, average pooling │
│     dilutes that strong activation across thousands of zero pixels!    │
│                                                                        │
│   * THE FIX (Max Pooling): Max pooling extracts the single strongest   │
│     peak activation:                                                   │
│       Max = max( all pixels )                                          │
│     It catches distinctive features regardless of how small they are.  │
│                                                                        │
│   CBAM's Solution: Use BOTH Average Pooling and Max Pooling!           │
└────────────────────────────────────────────────────────────────────────┘
```

### Breakthrough 2: Sequential Ordering (Channel First, Spatial Second)
In BAM, both branches ran in parallel from the same raw input and were added together. 

The CBAM authors found that **sequential processing** is superior:
1. **Channel Attention First:** Filter out dead, noisy, or irrelevant channels first.
2. **Spatial Attention Second:** Once you know *which* feature maps are relevant, look across the 2D spatial coordinate plane to pinpoint *where* those features reside.

---

## 3. Mathematical Architecture of CBAM

Given an intermediate feature tensor `F`:

```text
Input Feature Map F:
  Shape = [ Batch_Size, Channels, Height, Width ] = [ B, C, H, W ]
```

CBAM processes features through two sequential steps:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     CBAM SEQUENTIAL EQUATIONS                          │
│                                                                        │
│   Step 1 (Channel Calibration):                                        │
│          F' = Mc(F) * F                                                │
│                                                                        │
│   Step 2 (Spatial Calibration):                                        │
│          F'' = Ms(F') * F'                                             │
│                                                                        │
│   Where:                                                               │
│     Mc(F)  = Channel attention weights of shape [ B, C, 1, 1 ]         │
│     Ms(F') = Spatial attention weights of shape [ B, 1, H, W ]         │
│     *      = Element-wise multiplication with broadcasting             │
│     F''    = Final refined feature tensor [ B, C, H, W ]               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dissecting the Sub-Modules

### 4.1 The Channel Attention Sub-module

```text
               Channel Attention Sub-module Detailed Data Flow
               
                            Input F : [ B, C, H, W ]
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       AvgPool along (H x W)                         MaxPool along (H x W)
                │                                             │
       F_avg^c : [ B, C, 1, 1 ]                      F_max^c : [ B, C, 1, 1 ]
                │                                             │
                ▼                                             ▼
        Shared MLP Layer 0                            Shared MLP Layer 0
        W0 : [ C/16, C ]                              W0 : [ C/16, C ]
                │                                             │
                ▼                                             ▼
              ReLU                                          ReLU
                │                                             │
                ▼                                             ▼
        Shared MLP Layer 1                            Shared MLP Layer 1
        W1 : [ C, C/16 ]                              W1 : [ C, C/16 ]
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
                             Mc(F) : [ B, C, 1, 1 ]
```

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   CHANNEL ATTENTION EQUATIONS                          │
│                                                                        │
│   1. Squeeze Spatial Dimensions using BOTH Pooling Methods:            │
│      F_avg^c = AvgPool(F)   -->   Shape: [ B, C, 1, 1 ]                │
│      F_max^c = MaxPool(F)   -->   Shape: [ B, C, 1, 1 ]                │
│                                                                        │
│   2. Forward Both Vectors Through a SHARED Multi-Layer Perceptron:     │
│      Out_avg = W1 * ReLU( W0 * F_avg^c )                               │
│      Out_max = W1 * ReLU( W0 * F_max^c )                               │
│                                                                        │
│      Where:                                                            │
│        W0 has shape [ C / 16,  C ]                                     │
│        W1 has shape [ C,  C / 16 ]                                     │
│        (Weights W0 and W1 are shared to save parameters!)              │
│                                                                        │
│   3. Sum and Apply Sigmoid:                                            │
│      Mc(F) = Sigmoid( Out_avg + Out_max )                              │
│      Shape: [ B, C, 1, 1 ]                                             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 The Spatial Attention Sub-module

After channel calibration produces `F' = Mc(F) * F`, the spatial module decides which pixel locations to emphasize:

```text
               Spatial Attention Sub-module Detailed Data Flow
               
                       Intermediate Tensor F' : [ B, C, H, W ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
          AvgPool along Channels                          MaxPool along Channels
                  │                                               │
          F_avg^s : [ B, 1, H, W ]                        F_max^s : [ B, 1, H, W ]
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                Concatenate along
                                Channels: [ ; ]
                                          │
                                          ▼
                            Descriptor : [ B, 2, H, W ]
                                          │
                                          ▼
                              7x7 Convolution Layer
                                  Conv_7x7(·)
                               (Channels: 2 -> 1)
                                          │
                                          ▼
                               Sigmoid Activation: σ(·)
                                          │
                                          ▼
                                Ms(F') : [ B, 1, H, W ]
```

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   SPATIAL ATTENTION EQUATIONS                          │
│                                                                        │
│   1. Squeeze the Channel Dimension (Depth-wise Pooling):               │
│      For every pixel (i, j), compute the mean across all channels:     │
│          F_avg^s[1, i, j] = ( 1 / C ) * Sum_c ( F'[c, i, j] )          │
│                                                                        │
│      For every pixel (i, j), compute the maximum across all channels:  │
│          F_max^s[1, i, j] = max_c ( F'[c, i, j] )                      │
│                                                                        │
│      Both F_avg^s and F_max^s have shape: [ B, 1, H, W ]               │
│                                                                        │
│   2. Concatenate Along the Channel Axis:                               │
│      F_concat = Concat( F_avg^s,  F_max^s )                            │
│      Shape: [ B, 2, H, W ]                                             │
│                                                                        │
│   3. 7x7 Convolution and Sigmoid:                                      │
│      Ms(F') = Sigmoid( Conv_7x7( F_concat ) )                          │
│      Shape: [ B, 1, H, W ]                                             │
│                                                                        │
│      (The 7x7 kernel with padding=3 provides a broad spatial           │
│       neighborhood without altering height H or width W).              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Exhaustive Paper Ablation Studies (ECCV 2018)

The authors performed rigorous ablation experiments on ImageNet-1K with ResNet-50.

### 5.1 Ablation Study 1: Dual Pooling vs. Single Pooling

```text
================================================================================
TABLE 1: Channel Attention Pooling Methods on ResNet-50 (ImageNet-1K)
================================================================================
Configuration                 Parameters    GFLOPs    Top-1 Error    Top-5 Error
--------------------------------------------------------------------------------
ResNet-50 Baseline            25.56 M       3.86      24.56 %        7.50 %
+ AvgPool Only (SE Style)     25.92 M       3.94      23.14 %        6.70 %
+ MaxPool Only                25.92 M       3.94      23.20 %        6.83 %
+ BOTH (AvgPool & MaxPool)    25.92 M       4.02      22.80 % (Best) 6.52 % (Best)
================================================================================
```

#### Analytical Insight from Table 1:
Using **both AvgPool and MaxPool** dropped the Top-1 error to **`22.80%`**, outperforming Average-only (`23.14%`) and Max-only (`23.20%`). 

Because the weights of the MLP are shared across both vectors, this **0.34% improvement required ZERO additional parameters**.

---

### 5.2 Ablation Study 2: Spatial Attention Methods and Kernel Size

```text
================================================================================
TABLE 2: Spatial Attention Mechanisms and Kernel Sizes (ResNet-50)
================================================================================
Description                            Params    GFLOPs    Top-1 Error  Top-5 Error
--------------------------------------------------------------------------------
Channel Attention Only (SE)            28.09 M   3.860     23.14 %      6.70 %
Channel Attention Only (Proposed)      28.09 M   3.860     22.80 %      6.52 %
+ Spatial (1x1 Conv, Kernel 3x3)       28.10 M   3.862     22.96 %      6.64 %
+ Spatial (1x1 Conv, Kernel 7x7)       28.10 M   3.869     22.90 %      6.47 %
+ Spatial (Avg & Max, Kernel 3x3)      28.09 M   3.863     22.68 %      6.41 %
+ Spatial (Avg & Max, Kernel 7x7)      28.09 M   3.864     22.66 % (Best) 6.31 %
================================================================================
```

#### Analytical Insights from Table 2:
1. **Channel Pooling vs 1x1 Conv:** Pooling along the channel axis (Avg & Max) achieved lower error (`22.66%`) than using a learned 1x1 convolution (`22.90%`), while using fewer parameters.
2. **Receptive Field (`3 x 3` vs `7 x 7`):** The `7 x 7` spatial kernel outperformed the `3 x 3` kernel across the board. A pixel needs context from its wider neighborhood to understand whether it is part of the object or part of the background.

---

### 5.3 Ablation Study 3: Module Arrangement (Sequential vs. Parallel)

```text
================================================================================
TABLE 3: Module Arrangement Comparison (ResNet-50)
================================================================================
Arrangement Strategy                               Top-1 Error (%)   Top-5 Error (%)
--------------------------------------------------------------------------------
Channel Attention Only (SE)                        23.14 %           6.70 %
Sequential: Spatial First  --> Channel Second      22.78 %           6.42 %
Parallel Combination (BAM style: Mc + Ms)          22.95 %           6.59 %
Sequential: Channel First  --> Spatial Second      22.66 % (Best)    6.31 % (Best)
================================================================================
```

#### Why Channel-First Sequential Ordering Won:
* In **Channel-First** ordering (`F -> Mc -> F' -> Ms -> F''`), the channel module cleans up the feature maps globally, silencing noise. The spatial module can then focus its `7 x 7` convolution on clean, meaningful signals.
* In **Spatial-First** ordering, spatial pooling must aggregate across all channels equally, meaning noise from dead channels pollutes the spatial attention map.
* In **Parallel** ordering, neither module benefits from the filtering done by the other.

---

## 6. Large-Scale Benchmarks Across Architectures

### 6.1 ImageNet-1K Classification Results

```text
================================================================================
TABLE 4: ImageNet-1K Validation Errors (Single-Crop 224 x 224)
================================================================================
Architecture           Parameters    GFLOPs    Top-1 Error (%)    Top-5 Error (%)
--------------------------------------------------------------------------------
ResNet-18 Baseline     11.69 M       1.814     29.60 %            10.55 %
ResNet-18 + SE         11.78 M       1.814     29.41 %            10.22 %
ResNet-18 + CBAM       11.78 M       1.815     29.27 % (-0.33)    10.09 % (-0.46)
--------------------------------------------------------------------------------
ResNet-34 Baseline     21.80 M       3.664     26.69 %             8.60 %
ResNet-34 + SE         21.96 M       3.664     26.13 %             8.35 %
ResNet-34 + CBAM       21.96 M       3.665     25.99 % (-0.70)     8.24 % (-0.36)
--------------------------------------------------------------------------------
ResNet-50 Baseline     25.56 M       3.858     24.56 %             7.50 %
ResNet-50 + SE         28.09 M       3.860     23.14 %             6.70 %
ResNet-50 + CBAM       28.09 M       3.864     22.66 % (-1.90)     6.31 % (-1.19)
--------------------------------------------------------------------------------
ResNet-101 Baseline    44.55 M       7.570     23.38 %             6.88 %
ResNet-101 + SE        49.33 M       7.575     22.35 %             6.19 %
ResNet-101 + CBAM      49.33 M       7.581     21.51 % (-1.87)     5.69 % (-1.19)
--------------------------------------------------------------------------------
WideResNet-18 (w=2.0)  45.62 M       3.866     26.85 %             8.88 %
WideResNet-18 + SE     45.97 M       3.867     26.21 %             8.47 %
WideResNet-18 + CBAM   45.97 M       3.868     26.10 % (-0.75)     8.43 % (-0.45)
--------------------------------------------------------------------------------
ResNeXt-50 (32x4d)     25.03 M       3.768     22.85 %             6.48 %
ResNeXt-50 + SE        27.56 M       3.771     21.91 %             6.04 %
ResNeXt-50 + CBAM      27.56 M       3.774     21.92 % (-0.93)     5.91 % (-0.57)
--------------------------------------------------------------------------------
ResNeXt-101 (32x4d)    44.18 M       7.508     21.54 %             5.75 %
ResNeXt-101 + SE       48.96 M       7.512     21.17 %             5.66 %
ResNeXt-101 + CBAM     48.96 M       7.519     21.07 % (-0.47)     5.59 % (-0.16)
================================================================================
```

---

### 6.2 Lightweight Models: MobileNet on ImageNet-1K

```text
================================================================================
TABLE 5: MobileNet Experiments on ImageNet-1K
================================================================================
Architecture                 Params     GFLOPs    Top-1 Error (%)  Top-5 Error (%)
--------------------------------------------------------------------------------
MobileNet Baseline           3.23 M     0.569     31.70 %          11.90 %
MobileNet + SE               3.54 M     0.572     30.09 %          10.74 %
MobileNet + CBAM             3.54 M     0.574     29.40 % (-2.30)  10.60 % (-1.30)
--------------------------------------------------------------------------------
MobileNet (alpha = 0.7)      1.99 M     0.285     36.80 %          15.30 %
MobileNet (alpha = 0.7) + SE 2.14 M     0.286     35.10 %          14.10 %
MobileNet (a = 0.7) + CBAM   2.14 M     0.287     34.70 % (-2.10)  13.80 % (-1.50)
================================================================================
```

---

### 6.3 Object Detection: MS COCO and PASCAL VOC 2007

```text
================================================================================
TABLE 6: MS COCO Validation Object Detection (Faster R-CNN)
================================================================================
Backbone Architecture      Attention Module       mAP @ [.5, .95]     mAP @ .5
--------------------------------------------------------------------------------
ResNet-50                  Baseline               36.4                58.2
ResNet-50                  + SE                   37.1 (+0.7)         59.0 (+0.8)
ResNet-50                  + CBAM                 37.3 (+0.9)         59.3 (+1.1)
--------------------------------------------------------------------------------
ResNet-101                 Baseline               39.3                61.3
ResNet-101                 + SE                   40.0 (+0.7)         61.8 (+0.5)
ResNet-101                 + CBAM                 40.2 (+0.9)         62.3 (+1.0)
================================================================================
```

```text
================================================================================
TABLE 7: PASCAL VOC 2007 Test Detection (StairNet Detector)
================================================================================
Backbone Architecture      Detector Framework     Attention Module    mAP @ .5 (%)
--------------------------------------------------------------------------------
VGG-16                     SSD                    Baseline            77.8 %
VGG-16                     StairNet               Baseline            78.8 %
VGG-16                     StairNet               + SE                79.1 % (+0.3)
VGG-16                     StairNet               + CBAM              79.4 % (+0.6)
--------------------------------------------------------------------------------
MobileNet                  SSD                    Baseline            68.1 %
MobileNet                  StairNet               Baseline            70.3 %
MobileNet                  StairNet               + SE                70.8 % (+0.5)
MobileNet                  StairNet               + CBAM              71.2 % (+0.9)
================================================================================
```

---

## 7. Model Interpretability: Grad-CAM Analysis

To qualitatively verify that CBAM improves feature representations, the authors used **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize where the neural networks focused their internal activations.

```text
                Grad-CAM Activation Comparison Across Models
                
   Input Image        ResNet-50 Baseline       ResNet-50 + SE          ResNet-50 + CBAM
  ┌──────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
  │              │   │      . : : .       │   │      . ::: .       │   │     .--------.     │
  │   [ Dog ]    │==>│    : : : : : :     │==>│     : ::::: :      │==>│    /  (o  o)   \    │
  │              │   │   : : : : : : :    │   │    : ::::::: :     │   │   |   ( == )   |   │
  │  Background  │   │  . . : : : : . .   │   │    . ::::::: .     │   │    \   \__/   /    │
  │    Grass     │   │  (Activations leak │   │ (Stronger focus on │   │  (Tightly locked   │
  │              │   │   into background) │   │  target, but soft) │   │   on target mask)  │
  └──────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘
```

1. **Baseline ResNet-50:** The activation heatmaps were scattered and diffuse. The network was easily distracted by background foliage, floor textures, or lighting artifacts.
2. **ResNet-50 + SE:** The target object was emphasized more clearly than the baseline, but the heatmap remained blurry and lacked sharp boundaries.
3. **ResNet-50 + CBAM:** Class activations conformed tightly to the physical boundaries of target objects, demonstrating that the spatial submodule effectively suppresses background activations.

---

## 8. Master Comparison: SENet vs. BAM vs. CBAM

```text
=======================================================================================================
ARCHITECTURAL COMPARISON: SENet vs. BAM vs. CBAM
=======================================================================================================
Feature              SENet (CVPR 2018)            BAM (BMVC 2018)             CBAM (ECCV 2018)
-------------------------------------------------------------------------------------------------------
Attention Focus      Channel Only                 Channel + Spatial           Channel + Spatial
Module Placement     Every Residual Block         Stage Bottlenecks Only      Every Residual Block
Execution Flow       Single Path                  Parallel Branches (Mc + Ms) Sequential Cascade (Mc -> Ms)
Channel Pooling      Average Pooling Only         Average Pooling Only        Dual: AvgPool + MaxPool
Spatial Mechanism    None (Spatial Squeezed)      Dilated Convs (d=4, 3x3)    Channel Pooling + 7x7 Conv
Combination Rule     F * s                        F + ( F * Sigmoid(Mc+Ms) )  F' = Mc*F ; F'' = Ms*F'
Parameter Overhead   Low (~10% on ResNet-50)      Very Low (<1% at bottleneck)Very Low (~0.1% spatial)
ImageNet Top-1 Gain  +0.7% on ResNet-50           +0.9% on ResNet-50          +1.9% on ResNet-50
=======================================================================================================
```

---

## 9. Complete PyTorch Reference Implementation of CBAM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionCBAM(nn.Module):
    """
    CBAM Channel Attention Sub-module.
    Computes BOTH Average Pooling and Max Pooling across spatial dimensions,
    passes both through a shared MLP, and sums their outputs.
    
    Formula:
      Mc(F) = Sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionCBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        # Shared MLP implemented via 1x1 convolutions (avoids reshaping)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Average Pooling: [ B, C, H, W ] -> [ B, C, 1, 1 ]
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Spatial Max Pooling: [ B, C, H, W ] -> [ B, C, 1, 1 ]
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        
        # Forward through Shared MLP
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        # Element-wise addition and Sigmoid gating: [ B, C, 1, 1 ]
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttentionCBAM(nn.Module):
    """
    CBAM Spatial Attention Sub-module.
    Pools along the channel axis (Average and Max), concatenates into a
    2-channel map, and applies a 7x7 convolution.
    
    Formula:
      Ms(F) = Sigmoid( Conv_7x7( [AvgPool_c(F); MaxPool_c(F)] ) )
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionCBAM, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        # Projects 2-channel descriptor to a 1-channel spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling along channel axis: [ B, 1, H, W ]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling along channel axis: [ B, 1, H, W ]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel axis: [ B, 2, H, W ]
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Convolve and apply Sigmoid: [ B, 1, H, W ]
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


# Quick Verification
if __name__ == "__main__":
    dummy_input = torch.randn(4, 256, 28, 28)
    
    # Test Standalone CBAM
    cbam = CBAM(in_channels=256, reduction_ratio=16, spatial_kernel_size=7)
    cbam_out = cbam(dummy_input)
    assert cbam_out.shape == dummy_input.shape, "Shape mismatch in CBAM!"
    print(f"CBAM output shape: {cbam_out.shape} (PASSED)")
    
    # Test ResNet Block with CBAM
    block = CBAMResidualBlock(in_channels=256, mid_channels=64, out_channels=256, stride=1)
    block_out = block(dummy_input)
    assert block_out.shape == dummy_input.shape, "Shape mismatch in Residual Block!"
    print(f"CBAM Residual Block output shape: {block_out.shape} (PASSED)")
    print("All CBAM verifications passed successfully!")
```
