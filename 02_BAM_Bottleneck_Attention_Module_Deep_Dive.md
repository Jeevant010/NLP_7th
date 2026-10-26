# Part 2: BAM (Bottleneck Attention Module) — In-Depth Paper Dissection

---

## 1. Paper Overview and Metadata

* **Full Title:** BAM: Bottleneck Attention Module
* **Authors:** Jongchan Park, Sanghyun Woo, Joon-Young Lee, and In So Kweon
* **Affiliations:** Lunit Inc., Korea Advanced Institute of Science and Technology (KAIST), Adobe Research
* **Conference:** British Machine Vision Conference (BMVC), September 2018
* **arXiv Identifier:** 1807.06514 [cs.CV]

### 1.1 The High-Level Idea
BAM was the first architecture to build a **joint spatial-and-channel attention module** for deep networks. 

It was built around two guiding principles:
1. **Parallel Gating:** Compute Channel Attention ("What") and Spatial Attention ("Where") in two parallel, concurrent branches and fuse them via addition.
2. **Bottleneck Placement:** Rather than placing attention inside every convolutional block, place BAM exclusively at the **macro-bottlenecks** (downsampling transitions between stages).

---

## 2. Theoretical Motivation: Why the Bottlenecks?

In modern deep networks like ResNet, ResNeXt, and DenseNet, the network is divided into hierarchical **stages**:

```text
                             ResNet Stage Transitions
                             
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
* Spatial dimensions are cut in half: `(H, W) -> (H / 2, W / 2)`
* Channel capacity is doubled: `C -> 2 * C`

### The Authors' Hypothesis:
* **Early layers within a stage** extract noisy, low-level features. Adding attention inside every single layer adds parameter redundancy and optimization instability.
* **The bottleneck between stages** is the critical boundary where high-level semantic compression happens. Applying attention at these transition points acts as a selective filter, cleaning up features before they enter the next hierarchy.

---

## 3. Mathematical Architecture of BAM

Let `F` be an intermediate feature map extracted at a network bottleneck:

```text
Input Feature Map F:
  Shape = [ Batch_Size, Channels, Height, Width ]
  Abbreviated as: [ B, C, H, W ]
```

BAM infers a 3D attention map `M(F)` of the exact same shape `[ B, C, H, W ]`.

The refined feature map `F_refined` is computed using **residual scaling**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        BAM REFINEMENT FORMULA                          │
│                                                                        │
│          F_refined = F + ( F * M(F) )                                  │
│                                                                        │
│   Where:                                                               │
│     F      = Original input feature tensor [ B, C, H, W ]              │
│     M(F)   = Inferred 3D attention weights [ B, C, H, W ]              │
│     *      = Element-wise multiplication                               │
│     + F    = Residual identity skip connection                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why the Residual Connection is Critical

Notice the derivative of `F_refined` with respect to `F`:

```text
d(F_refined) / d(F) = Identity + M(F) + ( F * d(M(F))/d(F) )
```

During the early iterations of training, when the attention module is initialized near zero (`M(F) ≈ 0`):

```text
d(F_refined) / d(F) ≈ Identity
```

This guarantees two properties:
1. **Safety:** The network can never perform worse than the baseline, because the identity path `F` is preserved.
2. **Stable Gradients:** Gradients flow backwards directly through the identity shortcut, preventing vanishing gradients during early epochs.

---

## 4. Dissecting the Two Parallel Branches of BAM

BAM splits the calculation into two concurrent pathways:

```text
                      BAM Dual-Branch Data Flow
                      
                              Input Tensor F : [ B, C, H, W ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
                      ▼                                             ▼
         [ Channel Attention Branch ]                  [ Spatial Attention Branch ]
                      │                                             │
           Global Average Pooling                         1x1 Conv (C -> C/16)
                      │                                             │
            MLP: C -> C/16 -> C                        3x3 Dilated Conv (dilation=4)
                      │                                             │
             Batch Normalization                       3x3 Dilated Conv (dilation=4)
                      │                                             │
                      │                                    1x1 Conv (C/16 -> 1)
                      │                                             │
                      │                                    Batch Normalization
                      │                                             │
                      ▼                                             ▼
               Mc(F) : [ B, C, 1, 1 ]                        Ms(F) : [ B, 1, H, W ]
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
                                   M(F) : [ B, C, H, W ]
                                             │
                                             ▼
                                  Scale and Residual Skip
                                     F + ( F * M(F) )
                                             │
                                             ▼
                                  F_refined : [ B, C, H, W ]
```

---

### 4.1 The Channel Attention Branch `Mc(F)`

The channel branch evaluates **WHAT** features are informative by aggregating spatial context.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   CHANNEL ATTENTION EQUATIONS                          │
│                                                                        │
│   Step 1: Spatial Squeeze via Global Average Pooling                   │
│           F_c = AvgPool(F)                                             │
│           Shape transitions: [ B, C, H, W ]  -->  [ B, C, 1, 1 ]       │
│                                                                        │
│   Step 2: Two-Layer MLP with Reduction Ratio r = 16                    │
│           Hidden = ReLU( W0 * F_c + b0 )                               │
│           Out    = W1 * Hidden + b1                                    │
│           Where:                                                       │
│             W0 shape = [ C / 16,  C ]                                  │
│             W1 shape = [ C,  C / 16 ]                                  │
│                                                                        │
│   Step 3: Batch Normalization (Scale Alignment)                        │
│           Mc(F) = BatchNorm( Out )                                     │
│           Shape: [ B, C, 1, 1 ]                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### Why is Batch Normalization Used Here?
The channel branch output `Mc(F)` and the spatial branch output `Ms(F)` are combined via addition. If one branch had activations with variance `10.0` while the other had variance `0.1`, the first branch would completely drown out the second. 

Applying **Batch Normalization (BN)** to both branches normalizes their numerical ranges, ensuring neither branch dominates during fusion.

---

### 4.2 The Spatial Attention Branch `Ms(F)`

The spatial branch evaluates **WHERE** features are located across the 2D grid.

To make an accurate spatial judgment, a pixel at coordinate `(i, j)` must have contextual awareness of its neighborhood. A standard `3 x 3` convolution only sees `3 x 3` pixels, which is too small.

#### The Dilated Convolution Solution
BAM uses **dilated (atrous) convolutions** with a dilation rate `d = 4`.

```text
Standard Conv (d = 1):          Dilated Conv (d = 2):          Dilated Conv (d = 4):
Kernel touches adjacent pixels  Holes of 1 pixel between taps  Holes of 3 pixels between taps
       X  X  X                         X  .  X  .  X                  X  .  .  .  X  .  .  .  X
       X  X  X                         .  .  .  .  .                  .  .  .  .  .  .  .  .  .
       X  X  X                         X  .  X  .  X                  .  .  .  .  .  .  .  .  .
                                       .  .  .  .  .                  .  .  .  .  .  .  .  .  .
Receptive Field = 3 x 3                X  .  X  .  X                  X  .  .  .  X  .  .  .  X
                               Receptive Field = 5 x 5        Receptive Field = 9 x 9
```

#### Receptive Field Derivation in BAM:
1. For a `3 x 3` filter with dilation `d = 4`, the effective kernel span is:
   ```text
   K_span = 3 + (3 - 1) * (4 - 1) = 3 + 6 = 9
   ```
   So the first dilated layer covers a `9 x 9` window.
2. When a second dilated convolution (`d = 4`) is stacked on top:
   ```text
   Total Receptive Field = 9 + (9 - 1) = 17 x 17 pixels
   ```

By using two dilated layers, BAM covers a wide **`17 x 17` pixel receptive field** without any downsampling or extra parameters.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   SPATIAL ATTENTION EQUATIONS                          │
│                                                                        │
│   Step 1: Channel Squeeze (1x1 Conv)                                   │
│           X0 = Conv_1x1(F)                                             │
│           Shape transitions: [ B, C, H, W ]  -->  [ B, C/16, H, W ]    │
│                                                                        │
│   Step 2: Context Aggregation (Two Dilated 3x3 Convs, d = 4)           │
│           X1 = ReLU( BN( DilatedConv1(X0) ) )                          │
│           X2 = ReLU( BN( DilatedConv2(X1) ) )                          │
│           Receptive field expands to 17 x 17 pixels.                   │
│           Shape: [ B, C/16, H, W ]                                     │
│                                                                        │
│   Step 3: Spatial Collapse (1x1 Conv)                                  │
│           X3 = Conv_1x1(X2)                                            │
│           Shape transitions: [ B, C/16, H, W ]  -->  [ B, 1, H, W ]    │
│                                                                        │
│   Step 4: Batch Normalization                                          │
│           Ms(F) = BatchNorm(X3)                                        │
│           Shape: [ B, 1, H, W ]                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 Branch Combination: Broadcasting and Summation

We now have two complementary attention maps:
* Channel attention: `Mc(F)` of shape `[ B, C, 1, 1 ]`
* Spatial attention: `Ms(F)` of shape `[ B, 1, H, W ]`

Both are broadcast across the missing dimensions to shape `[ B, C, H, W ]`, summed together, and passed through a Sigmoid function:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      FUSION AND SIGMOID GATING                         │
│                                                                        │
│   M(F) = Sigmoid( Expand(Mc(F)) + Expand(Ms(F)) )                      │
│                                                                        │
│   Output M(F) has shape [ B, C, H, W ] with values in range (0, 1).    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Experimental Analysis and Paper Ablations

The BAM authors evaluated their design on CIFAR-100, ImageNet-1K, VOC 2007, and MS COCO.

### 5.1 Ablation Study 1: Dilation Rate and Reduction Ratio

Evaluated on ResNet-50 on the CIFAR-100 dataset:

```text
================================================================================
TABLE 1(a): Dilation Factor (d) and Reduction Ratio (r) on ResNet-50
================================================================================
Parameter Tested         Value                  Parameters     Top-1 Error (%)
--------------------------------------------------------------------------------
ResNet-50 Baseline       --                     23.68 M        21.49 %
--------------------------------------------------------------------------------
Dilation Value (d)       d = 1 (Standard Conv)  24.07 M        20.47 %
Dilation Value (d)       d = 2                  24.07 M        20.28 %
Dilation Value (d)       d = 4 (Optimal)        24.07 M        20.00 %  <-- BEST
Dilation Value (d)       d = 6                  24.07 M        20.08 %
--------------------------------------------------------------------------------
Reduction Ratio (r)      r = 4                  26.30 M        20.46 %
Reduction Ratio (r)      r = 8                  24.62 M        20.56 %
Reduction Ratio (r)      r = 16 (Optimal)       24.07 M        20.00 %  <-- BEST
Reduction Ratio (r)      r = 32                 23.87 M        21.24 %
================================================================================
```

#### Analytical Insights:
* **Wider Context Helps (`d = 1` vs `d = 4`):** When dilation is `1` (standard convolution without holes), error is `20.47%`. Increasing dilation to `4` drops error to `20.00%`. Spatial attention needs a broad contextual horizon to decide what is foreground vs background.
* **Overfitting with High Capacity (`r = 4` vs `r = 16`):** Although `r = 4` has 4x more parameters in the MLP, it achieves worse test error (`20.46%`) than `r = 16` (`20.00%`). The authors found that training loss converged in both cases, proving that excessive capacity in the attention block leads to overfitting.

---

### 5.2 Ablation Study 2: Combining Methods

The authors tested whether both branches were necessary and compared different combination operators:

```text
================================================================================
TABLE 1(b): Branch Ablations and Fusion Operators (ResNet-50)
================================================================================
Configuration                 Combination Operator        Top-1 Error (%)
--------------------------------------------------------------------------------
ResNet-50 Baseline            None                        21.49 %
Channel Branch Only           None                        20.28 %
Spatial Branch Only           None                        20.21 %
Both Branches                 Element-wise Maximum (MAX)  21.24 %
Both Branches                 Element-wise Product (PROD) 21.29 %
Both Branches (BAM)           Element-wise Sum (SUM)      20.00 %  <-- BEST
================================================================================
```

#### Why Element-Wise Summation Won:
1. **Summation:** In forward pass, it preserves information from both branches without attenuation. In backward pass, gradients flow equally to both branches (`d(Mc + Ms)/d(Mc) = 1` and `d(Mc + Ms)/d(Ms) = 1`).
2. **Product (`Mc * Ms`):** Multiplicative coupling (`d(Mc * Ms)/d(Mc) = Ms`). If one branch produces a small value, the gradient flowing to the other branch is suppressed, making optimization unstable.
3. **Maximum (`max(Mc, Ms)`):** Routes gradient exclusively to whichever branch had the higher activation, starving the weaker branch of gradient updates.

---

### 5.3 Placement Study: Bottlenecks vs. Every Convolutional Block

The authors compared placing BAM strictly at transition bottlenecks versus placing BAM inside every single convolution block (denoted as **BAM-C**):

```text
================================================================================
TABLE 2: Bottleneck Placement (BAM) vs Every Block Placement (BAM-C)
================================================================================
Architecture           Variant               Parameters    GFLOPs    Top-1 Error
--------------------------------------------------------------------------------
ResNet-50              Baseline              23.71 M       1.22      21.49 %
ResNet-50              + BAM-C (Every Block) 28.98 M       1.37      20.88 %
ResNet-50              + BAM (Bottlenecks)   24.07 M       1.25      20.00 % (Best)
--------------------------------------------------------------------------------
WideResNet-28 (w=8)    Baseline              23.40 M       3.36      20.40 %
WideResNet-28 (w=8)    + BAM-C (Every Block) 23.78 M       3.39      20.06 %
WideResNet-28 (w=8)    + BAM (Bottlenecks)   23.42 M       3.37      19.06 % (Best)
--------------------------------------------------------------------------------
ResNeXt-29 (8x64d)     Baseline              34.52 M       4.99      18.18 %
ResNeXt-29 (8x64d)     + BAM-C (Every Block) 35.60 M       5.07      18.15 %
ResNeXt-29 (8x64d)     + BAM (Bottlenecks)   34.61 M       5.00      16.71 % (Best)
================================================================================
```

#### The Major Finding of Table 2:
Placing BAM inside every single block (**BAM-C**) added significantly more parameters (+5.27 million parameters on ResNet-50), yet achieved **worse** error (`20.88%`) than placing BAM strictly at the bottlenecks (`20.00%` with only +0.36 million parameters).

*Conclusion:* Applying attention at the macro-level bottleneck transitions provides a cleaner information filter than continuous intra-block gating.

---

### 5.4 ImageNet-1K Benchmark Results

On large-scale ImageNet-1K classification:

```text
================================================================================
TABLE 3: ImageNet-1K Single-Crop Validation Errors
================================================================================
Architecture           Parameters    GFLOPs    Top-1 Error (%)    Top-5 Error (%)
--------------------------------------------------------------------------------
ResNet-18 Baseline     11.69 M       1.81      29.60 %            10.55 %
ResNet-18 + BAM        11.71 M       1.82      28.88 % (-0.72)    10.01 % (-0.54)
--------------------------------------------------------------------------------
ResNet-50 Baseline     25.56 M       3.86      24.56 %             7.50 %
ResNet-50 + BAM        25.92 M       3.94      24.02 % (-0.54)     7.18 % (-0.32)
--------------------------------------------------------------------------------
ResNet-101 Baseline    44.55 M       7.57      23.38 %             6.88 %
ResNet-101 + BAM       44.91 M       7.65      22.44 % (-0.94)     6.29 % (-0.59)
================================================================================
```

---

## 6. Complete PyTorch Reference Implementation of BAM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionBAM(nn.Module):
    """
    Channel Attention Branch for BAM.
    Squeezes spatial dimensions via Global Average Pooling,
    models channel correlations via a two-layer MLP,
    and normalizes output with Batch Normalization.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [ B, C, H, W ]
        x_pool = F.adaptive_avg_pool2d(x, (1, 1))          # [ B, C, 1, 1 ]
        x_att = self.mlp(x_pool)                           # [ B, C ]
        x_att = self.bn(x_att)                             # [ B, C ]
        return x_att.unsqueeze(2).unsqueeze(3)             # [ B, C, 1, 1 ]


class SpatialAttentionBAM(nn.Module):
    """
    Spatial Attention Branch for BAM.
    Uses 1x1 Conv to reduce channels, two cascaded 3x3 Dilated Convolutions
    (dilation=4) to cover a 17x17 receptive field, a 1x1 Conv to collapse to 1 channel,
    and Batch Normalization for scale alignment.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, dilation: int = 4):
        super(SpatialAttentionBAM, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.spatial_conv = nn.Sequential(
            # 1x1 Conv: Channel reduction
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # Dilated Conv 1 (effective receptive field: 9x9)
            nn.Conv2d(
                reduced_channels, reduced_channels, 
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # Dilated Conv 2 (effective receptive field: 17x17)
            nn.Conv2d(
                reduced_channels, reduced_channels, 
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 Conv: Spatial projection to 1 channel
            nn.Conv2d(reduced_channels, 1, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [ B, C, H, W ]
        x_att = self.spatial_conv(x)                       # [ B, 1, H, W ]
        return self.bn(x_att)                              # [ B, 1, H, W ]


class BAM(nn.Module):
    """
    Bottleneck Attention Module (BMVC 2018).
    Combines Channel Attention and Spatial Attention via parallel summation,
    followed by Sigmoid gating and residual addition.
    
    Formula:
      M(F)   = Sigmoid( Mc(F) + Ms(F) )
      Output = F + ( F * M(F) )
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, dilation: int = 4):
        super(BAM, self).__init__()
        self.channel_att = ChannelAttentionBAM(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttentionBAM(in_channels, reduction_ratio, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mc shape: [ B, C, 1, 1 ]
        mc = self.channel_att(x)
        
        # Ms shape: [ B, 1, H, W ]
        ms = self.spatial_att(x)
        
        # Broadcasting addition produces 3D attention map: [ B, C, H, W ]
        attention_map = torch.sigmoid(mc + ms)
        
        # Residual scaling
        return x + (x * attention_map)


# Quick Verification
if __name__ == "__main__":
    dummy_input = torch.randn(4, 512, 28, 28)
    bam = BAM(in_channels=512, reduction_ratio=16, dilation=4)
    output = bam(dummy_input)
    assert output.shape == dummy_input.shape, "Shape mismatch in BAM!"
    print(f"BAM input shape:  {dummy_input.shape}")
    print(f"BAM output shape: {output.shape}")
    print("BAM verification successful!")
```
