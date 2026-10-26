# Unit 4: Word Sense Disambiguation (WSD) & Yarowsky's Algorithm
**Direct Blueprint**: Based on `WSD_27_OCT_2026.pdf` (IIIT Surat, CS-702 / CS-913 / CS-703)  
**Crucial Exam Topics**: Sense Identification, Hypernyms, Yarowsky Bootstrapping, Divergence & Failure Modes, Naive Bayes WSD Formulation.

---

## 1. Multi-Document Sense Analysis of "Plant"
*(Exact solution to Question (a) in `WSD_27_OCT_2026.pdf`)*

### Corpus Snippets:
* **Doc 1**: *"...variety of plant life or flora in South Africa... plant kingdom... plant species... plant and animal life... largest genus of flowering plants..."*
* **Doc 2**: *"...tomato processing plant... plant technicians jobs... process operations of the plant... plant management... CEO of the plant..."*
* **Doc 3**: *"...plants' ability to adapt... half human, half plant... traditional food plants... tomato plant... medicinal plants..."*
* **Doc 4**: *"...magazine 'Plant Life', which humorously describes daily life at the Imperial food processing plant... published by a chemical plant process worker... small molecular commercial plants... pharmaceutical plants... heavy plant crossing..."*

### Identified Senses & Hypernyms:
1. **Sense 1: Botanical Organism (Flora)**
   * **Definition**: A living photosynthetic organism lacking locomotive movement (trees, flowers, herbs).
   * **Hypernym**: `organism` / `living thing` / `vascular plant`.
   * **Evidence**: Doc 1 ("plant kingdom", "flowering plants"), Doc 3 ("medicinal plants", "adapt to harsh circumstances"), Doc 4 (pun in title "Plant Life").
2. **Sense 2: Industrial Manufacturing Facility (Factory)**
   * **Definition**: A building or complex where industrial processes, manufacturing, or power generation occur.
   * **Hypernym**: `facility` / `building complex` / `manufacturing plant`.
   * **Evidence**: Doc 2 ("processing plant", "CEO of the plant", "plant technicians"), Doc 4 ("chemical plant", "food processing plant").
3. **Sense 3: Heavy Industrial Equipment / Machinery (Bonus nuance in Doc 4)**
   * **Definition**: Heavy mobile machinery or earthmoving vehicles (as in "heavy plant crossing" road sign).
   * **Hypernym**: `machinery` / `equipment`.

---

## 4. Naive Bayes for Word Sense Disambiguation
*(Exact solution to Question (d))*

Given a target ambiguous word $w$ and context features $\mathbf{f} = (f_1, f_2, \dots, f_m)$, find the most probable sense $\hat{s}$:
$$\hat{s} = \arg\max_{s_k \in S} P(s_k \mid f_1, f_2, \dots, f_m)$$

### Bayesian Formulation:
By Bayes' theorem:
$$P(s_k \mid \mathbf{f}) = \frac{P(s_k) \cdot P(f_1, \dots, f_m \mid s_k)}{P(\mathbf{f})}$$
Omitting the denominator $P(\mathbf{f})$ (constant across all senses):
$$\hat{s} = \arg\max_{s_k \in S} P(s_k) \cdot P(f_1, \dots, f_m \mid s_k)$$

### Naive Bayes Independence Assumption:
Assume context features $f_i$ are conditionally independent given sense $s_k$:
$$\hat{s} = \arg\max_{s_k \in S} \left[ P(s_k) \prod_{i=1}^m P(f_i \mid s_k) \right]$$
In log-space (to prevent numerical underflow):
$$\hat{s} = \arg\max_{s_k \in S} \left[ \log P(s_k) + \sum_{i=1}^m \log P(f_i \mid s_k) \right]$$

### Parameter Training via Maximum Likelihood with Laplace Smoothing:
From an annotated sense corpus (e.g., SemCor):
1. **Sense Prior Probability**:
   $$P(s_k) = \frac{C(s_k)}{N_{\text{total}}}$$
2. **Feature Likelihood (with Add-1 Laplace Smoothing)**:
   $$P(f_i \mid s_k) = \frac{C(f_i, s_k) + 1}{C(s_k) + |V|}$$
   where:
   * $C(f_i, s_k)$ is the count of feature $f_i$ appearing in contexts labelled with sense $s_k$.
   * $C(s_k) = \sum_{f'} C(f', s_k)$ is total feature count for sense $s_k$.
   * $|V|$ is the total feature vocabulary size.
