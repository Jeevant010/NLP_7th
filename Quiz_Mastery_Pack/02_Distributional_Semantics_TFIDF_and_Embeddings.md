# Unit 3: Distributional Semantics, Vector Models, and Neural Embeddings
**Topics Covered**: TF-IDF & Cosine Similarity, CBOW vs Skip-Gram (SGNS), FastText OOV, BERT vs GPT Masking, 4 Types of Context, PMI & Rare Word Bias, Paradigmatic vs Syntagmatic Clustering.

---

## 1. TF-IDF & Cosine Similarity: Complete Numerical Walkthrough
*(Directly from `NLP_SET2.pdf` & Mid-Sem M.Tech Paper)*

### Problem Statement:
Consider a corpus of two documents:
* **$D_1$**: "dog bites man"
* **$D_2$**: "man feeds dog"

Given formula:
$$IDF(t) = \ln\left(\frac{N}{DF(t)}\right)$$
where $N = 2$ is total documents, and $DF(t)$ is document frequency.

### Step 1: Vocabulary & Document Frequencies
Let's list unique terms in alphabetical order:
$$V = [\text{bites, dog, feeds, man}]$$

| Term ($t$) | $D_1$ Raw TF | $D_2$ Raw TF | Document Frequency $DF(t)$ | $IDF(t) = \ln\left(\frac{2}{DF}\right)$ |
| :--- | :---: | :---: | :---: | :---: |
| **bites** | 1 | 0 | 1 | $\ln(2/1) = \ln(2) \approx 0.6931$ |
| **dog** | 1 | 1 | 2 | $\ln(2/2) = \ln(1) = 0$ |
| **feeds** | 0 | 1 | 1 | $\ln(2/1) = \ln(2) \approx 0.6931$ |
| **man** | 1 | 1 | 2 | $\ln(2/2) = \ln(1) = 0$ |

### Step 2: Compute TF-IDF Vectors
$$\text{TF-IDF}(t, D) = \text{TF}(t, D) \times IDF(t)$$

* For **$D_1$**:
  * bites: $1 \times \ln(2) = \ln(2)$
  * dog: $1 \times 0 = 0$
  * feeds: $0 \times \ln(2) = 0$
  * man: $1 \times 0 = 0$
  $$\mathbf{v}(D_1) = [\ln(2), 0, 0, 0]$$

* For **$D_2$**:
  * bites: $0 \times \ln(2) = 0$
  * dog: $1 \times 0 = 0$
  * feeds: $1 \times \ln(2) = \ln(2)$
  * man: $1 \times 0 = 0$
  $$\mathbf{v}(D_2) = [0, 0, \ln(2), 0]$$

### Step 3: Compute Cosine Similarity
$$\text{Cosine}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}$$

* Dot Product:
  $$\mathbf{v}_1 \cdot \mathbf{v}_2 = (\ln 2 \cdot 0) + (0 \cdot 0) + (0 \cdot \ln 2) + (0 \cdot 0) = 0$$
* Norms:
  $$\|\mathbf{v}_1\| = \sqrt{(\ln 2)^2 + 0 + 0 + 0} = \ln 2$$
  $$\|\mathbf{v}_2\| = \sqrt{0 + 0 + (\ln 2)^2 + 0} = \ln 2$$
* Final Cosine Similarity:
  $$\text{Cosine}(\mathbf{v}_1, \mathbf{v}_2) = \frac{0}{\ln 2 \cdot \ln 2} = 0$$

> [!IMPORTANT]
> **High-Yield Exam Concept**: Why is Cosine Similarity $0$ despite sharing $66\%$ of words ("dog" and "man")?
> Because terms occurring in *every* document have $DF(t) = N \implies \ln(N/DF) = \ln(1) = 0$. They are treated as non-discriminative stop-words and lose all weight. Thus, the resulting vectors are completely orthogonal!

---

## 2. Word2Vec: CBOW vs Skip-Gram with Negative Sampling

### Architectural Comparison:
```
           CBOW                                  Skip-gram
Context Words  -->  [ SUM / AVG ]             Center Word w_t
w_{t-2}, w_{t-1},       |                            |
w_{t+1}, w_{t+2}        v                            v
                  Center Word w_t          Context Words w_{t-2}, w_{t-1}, ...
```

| Dimension | Continuous Bag-of-Words (CBOW) | Skip-Gram with Negative Sampling (SGNS) |
| :--- | :--- | :--- |
| **Prediction Task** | Predicts center word $w_t$ given context $w_{t-c:t+c}$ | Predicts context words given center word $w_t$ |
| **Input Representation** | Averaged/summed vector of context words | Single vector of target word $w_t$ |
| **Training Speed** | Fast ($O(1)$ updates per window) | Slower ($O(c \cdot K)$ updates per window) |
| **Data Efficiency** | Excels on frequent words, large corpora | Excels on rare/infrequent words and smaller datasets |
| **Semantics** | Smooths context, less sensitive to rare noise | Captures nuanced semantic relationships |

### Objective Functions:
1. **Standard Skip-Gram (Softmax)**:
   $$\mathcal{L} = \sum_{t=1}^T \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t), \quad P(w_O \mid w_I) = \frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w \in V} \exp({v'_w}^\top v_{w_I})}$$
   *Computational Bottleneck*: Computing the partition function denominator requires summing over $|V|$ (often $10^5$ to $10^6$ words) at every step!

2. **Skip-gram with Negative Sampling (SGNS)**:
   Replaces the multi-class softmax with a set of binary logistic regressions:
   $$\mathcal{L}_{SGNS} = \log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{k=1}^K \mathbb{E}_{w_k \sim P_n(w)} \left[ \log \sigma(-{v'_{w_k}}^\top v_{w_I}) \right]$$
   where:
   * $\sigma(x) = \frac{1}{1 + e^{-x}}$.
   * $w_O$ is the true positive context word (maximize similarity).
   * $w_k$ are $K$ negative samples drawn from the unigram distribution $P_n(w) \propto U(w)^{0.75}$ (minimize similarity).

---

## 3. FastText: Why it Handles Out-Of-Vocabulary (OOV) Words
Standard Word2Vec and GloVe assign a single atomic vector to each word token. If a word is missing during training, its vector does not exist ($\langle unk \rangle$).

### FastText Mechanism:
1. Words are treated as bags of **character $n$-grams** plus special boundary symbols $\langle$ and $\rangle$.
2. For $n=3$ and word `where`:
   $$\text{Subwords} = \{\langle\text{wh}, \text{whe}, \text{her}, \text{ere}, \text{re}\rangle\} \cup \{\langle\text{where}\rangle\}$$
3. Word representation is the sum of subword vectors:
   $$v_{\text{where}} = \sum_{g \in \mathcal{G}_{\text{where}}} z_g$$
4. **Why it beats Word2Vec/GloVe for OOV**:
   * If an unseen test word like `unbreakableness` appears, FastText decomposes it into subwords (`un`, `break`, `able`, `ness`).
   * Since subwords were seen during training in other words, their embeddings combine to construct an accurate semantic representation.
   * Particularly powerful for **morphologically rich languages** (German compound words, Hindi, Turkish inflections).

---

## 4. BERT vs GPT: Attention Masking Differences

```
BERT (Encoder - Bidirectional)                 GPT (Decoder - Causal Autoregressive)
     x1    x2    x3    x4                           x1    x2    x3    x4
x1  [ 1     1     1     1  ]                   x1  [ 1     0     0     0  ]
x2  [ 1     1     1     1  ]                   x2  [ 1     1     0     0  ]
x3  [ 1     1     1     1  ]                   x3  [ 1     1     1     0  ]
x4  [ 1     1     1     1  ]                   x4  [ 1     1     1     1  ]
```

| Feature | BERT | GPT |
| :--- | :--- | :--- |
| **Model Type** | Transformer **Encoder** | Transformer **Decoder** |
| **Attention Mask** | **Unmasked / Bidirectional**: Token $i$ attends to all tokens $j \in [1, N]$. | **Causal / Lower Triangular Mask**: Token $i$ only attends to $j \le i$. Future tokens $j > i$ get $-\infty$ mask. |
| **Training Objective** | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) | Autoregressive Next-Token Prediction: $\max \sum \log P(w_t \mid w_{<t})$ |
| **Masking Discrepancy (Train-Test Mismatch)** | $[MASK]$ token appears during training but NEVER at inference! Resolved via the **80/10/10 Rule**: 80% $[MASK]$, 10% random word, 10% unchanged. | No mismatch; generates tokens iteratively in inference just like during training. |

---

## 5. Distributional Semantics & Pointwise Mutual Information (PMI)
*(From `13_OCT_2026.pdf`)*

### Distributional Hypothesis:
> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

### 4 Types of Context:
1. **Linear Word Window**: Symmetric or asymmetric window of size $k$ around the target (e.g. $\pm 5$ words).
2. **Syntactic / Dependency Relations**: Grammatical head-dependent tuples (e.g. `dobj(drink, tea)`, `subj(bark, dog)`). Captures functional grammatical substitutes.
3. **Sentence Context**: All words appearing in the same sentence.
4. **Document Context**: Bag-of-words over the entire article or paragraph. Captures broad topical domains.

### Pointwise Mutual Information (PMI) Formulation:
$$\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w)P(c)}$$

Estimation from corpus counts ($N = \sum_{w,c} C(w,c)$):
$$P(w, c) = \frac{C(w,c)}{N}, \quad P(w) = \frac{C(w)}{N}, \quad P(c) = \frac{C(c)}{N}$$
$$\text{PMI}(w, c) = \log_2 \left( \frac{C(w, c) \cdot N}{C(w) \cdot C(c)} \right)$$

### The Rare Word Problem in PMI:
* If a word $w$ and context $c$ occur together once ($C(w, c) = 1$) and each occurs only once in the entire corpus ($C(w) = 1, C(c) = 1$), then:
  $$\frac{P(w, c)}{P(w)P(c)} = \frac{1/N}{(1/N)(1/N)} = N \implies \text{PMI} = \log_2(N)$$
* **Consequence**: Ultra-rare words receive artificially astronomical PMI scores solely because their marginal probabilities in the denominator are minuscule!
* **Remedies**:
  1. Frequency cutoff (ignore counts $< 5$).
  2. **Positive PMI (PPMI)**: $\text{PPMI}(w, c) = \max(0, \text{PMI}(w, c))$.
  3. Smoothed context distribution: $P_\alpha(c) = \frac{C(c)^\alpha}{\sum_c C(c)^\alpha}$ with $\alpha = 0.75$.

---

## 6. Paradigmatic vs Syntagmatic Word Clustering
*(Analysis of Experiment 1 vs Experiment 2 from `13_OCT_2026.pdf`)*

* **Experiment 1 Clusters**:
  * `carriage, bike, vehicle, train, truck, lorry, taxi`
  * `official, officer, inspector, journalist, detective, constable`
  * `sister, daughter, parent, relative, wife, mother, brother`
  * **Relationship Captured**: **Paradigmatic / Taxonomic Similarity** (co-hyponyms, synonyms). Words that can substitute for each other in the same grammatical slot.
  * **Context Setup**: Used **Syntactic Dependency Features** (e.g., `obj_of(drive)`, `subj_of(arrest)`).

* **Experiment 2 Clusters**:
  * `car, engine, petrol, road, driver, wheel, steering, speed`
  * `concert, singer, stage, light, music, show, audience, ticket`
  * `experiment, research, scientist, paper, result, laboratory`
  * **Relationship Captured**: **Syntagmatic / Thematic Relatedness** (meronyms, frame associations, topical co-occurrence). Words that appear together in the same real-world event/scenario.
  * **Context Setup**: Used **Broad Document-Level / Large Window Bag-of-Words** features.
