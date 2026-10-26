# Transformers, Self-Attention, and Automatic Text Summarisation
**Direct Blueprint**: Based on `Summarisation_12_Oct_2026.pdf` & `NLP_MID_SEMESTER_PAPER_M_Tech.pdf`  
**Crucial Exam Topics**: Multi-Document Summarisation, Maximal Marginal Relevance (MMR), Rhetorical Relations (RST), Scaled Dot-Product & Multi-Head Attention, BERT MLM Discrepancy.

---

## 1. Multi-Document vs Single-Document Summarisation
*(From `Summarisation_12_Oct_2026.pdf`)*

### Why Multi-Document Summarisation is Significantly More Challenging:
1. **Cross-Document Redundancy & Overlap**:
   * Multiple news articles or documents about the same event repeat core facts in slightly different phrasing.
   * *Challenge*: The summariser must cluster synonymous/near-duplicate sentences and select only the most representative, non-redundant proposition.
2. **Temporal & Factual Inconsistency (Contradictions)**:
   * Documents published at different times or by different outlets may state conflicting information (e.g., initial report: "10 injured", later report: "12 injured, 1 fatality").
   * *Challenge*: Resolving contradictions, tracking temporal evolution of events, and avoiding contradictory statements in the generated summary.
3. **Cross-Document Co-reference Resolution**:
   * Entities are mentioned differently across authors (e.g., "Emmanuel Macron" in Doc 1, "The French President" in Doc 2, "He" in Doc 3).
   * *Challenge*: Linking entity mentions across independent document boundaries.
4. **Coherence and Narrative Ordering**:
   * In single-document summarisation, the original document order provides a natural narrative flow. In multi-document summarisation, sentences extracted from different articles have no intrinsic chronological order and can result in incoherent, jarring summaries.

---

## 2. Maximal Marginal Relevance (MMR)
Used in query-focused multi-document summarisation to balance relevance with diversity:

### Mathematical Formula:
$$\text{MMR} = \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(D_i, Q) - (1 - \lambda) \max_{D_j \in S} \text{Sim}_2(D_i, D_j) \right]$$

### Definition of Components:
* $R$: The set of all candidate retrieved sentences / documents.
* $S$: The set of sentences **already selected** into the summary so far ($S \subset R$).
* $R \setminus S$: The remaining unselected candidate sentences.
* $Q$: The user's search query or topic description.
* $\text{Sim}_1(D_i, Q)$: Semantic similarity between candidate $D_i$ and query $Q$ (e.g., cosine similarity of TF-IDF or embedding vectors).
* $\text{Sim}_2(D_i, D_j)$: Semantic similarity between candidate $D_i$ and an already selected summary sentence $D_j$.
* $\lambda \in [0, 1]$: Tuning hyperparameter controlling the trade-off.

### Core Intuition:
* When $\lambda = 1$: Pure greedy search for relevance to $Q$ (results in high redundancy).
* When $\lambda = 0$: Maximal diversity (picks sentences completely dissimilar to what is already selected).
* Typically $\lambda \approx 0.7$: Rewards sentences that are **relevant to the query** while **penalizing redundancy** against sentences already in the summary!

---

## 3. Rhetorical Relations in Discourse
*(Exact solution to Question (c) in `Summarisation_12_Oct_2026.pdf`)*

| Sentence Pair | Sentence A & Sentence B | Rhetorical Relation | Reasoning / Justification |
| :---: | :--- | :---: | :--- |
| **(i)** | *The use of diesel in transport has come under increasing scrutiny in recent years. According to WHO, around three million deaths every year are linked to exposure to outdoor air pollution.* | **Evidence / Justification** | Sentence B cites authoritative WHO data to justify/provide evidence for the claim in Sentence A. |
| **(ii)** | *Nitrogen oxides can help form ground level ozone. This can exacerbate breathing difficulties.* | **Result / Cause** | Sentence B describes the direct physiological consequence (result) caused by the chemical phenomenon in Sentence A. |
| **(iii)** | *Paris has already taken a series of steps to cut the impact of diesel cars and trucks. Vehicles registered before 1997 have already been banned from entering the city.* | **Elaboration / Exemplification** | Sentence B provides a concrete, specific example (exemplification) of the general "steps" mentioned in Sentence A. |

---

## 4. NLP Techniques in Extractive Summarisation
*(Exact solution to Question (d))*

1. **Morphological Processing (Stemming & Lemmatization)**:
   * *Role*: Conflates morphological variants (e.g., *pollutes, polluted, pollution* $\to$ *pollute*).
   * *Benefit*: Ensures accurate term frequency (TF) and keyword frequency counts, preventing vocabulary fragmentation when ranking salient sentences.
2. **Syntactic Parsing**:
   * *Role*: Generates constituent or dependency parse trees for candidate sentences.
   * *Benefit*: Enables **Sentence Compression / Trimming** (e.g., safely pruning non-restrictive relative clauses, prepositional modifiers, or parentheticals while guaranteeing the remaining compressed sentence is syntactically well-formed).
3. **Lexical and Distributional Semantics**:
   * *Role*: Computes semantic vector representations (TF-IDF, Word2Vec, BERT embeddings) and sentence-level similarity graphs.
   * *Benefit*: Powers graph-based ranking algorithms (like **TextRank** and **LexRank**) to identify sentences with highest eigenvector centrality (salience) and avoids selecting semantically paraphrased duplicates.
4. **Discourse Parsing (Rhetorical Structure Theory - RST)**:
   * *Role*: Identifies hierarchical discourse relations (Nucleus vs. Satellite).
   * *Benefit*: **Nuclei** contain the core, indispensable assertions of a text, while **Satellites** contain non-essential background or elaboration. Extractive summarisation keeps Nuclei and prunes Satellites, producing high-density summaries.

---

## 5. Transformer Self-Attention & BERT MLM
*(From `NLP_MID_SEMESTER_PAPER_M_Tech.pdf`)*

### Mathematical Formulation of Scaled Dot-Product Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$$
where:
* $Q \in \mathbb{R}^{N \times d_k}$ is the Query matrix.
* $K \in \mathbb{R}^{M \times d_k}$ is the Key matrix.
* $V \in \mathbb{R}^{M \times d_v}$ is the Value matrix.
* $\frac{QK^\top}{\sqrt{d_k}}$ is the attention score logits.
* **Why scale by $\sqrt{d_k}$?** As dimension $d_k$ grows large, the dot products grow large in magnitude, pushing the softmax function into regions with tiny gradients (gradient vanishing). Dividing by $\sqrt{d_k}$ keeps the variance at $1$.

### Multi-Head Attention:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### The BERT MLM Train-Test Mismatch (Masking Discrepancy):
* **The Problem**: In pre-training, BERT masks 15% of tokens with $[MASK]$. However, downstream inference tasks (sentiment analysis, NER, QA) never contain the $[MASK]$ token! If BERT only saw $[MASK]$, it would build representations dependent on an artificial token absent at test time.
* **The 80/10/10 Solution**:
  * 80% of selected tokens: Replaced with $[MASK]$.
  * 10% of selected tokens: Replaced with a **random word**.
  * 10% of selected tokens: Kept **completely unchanged**.
  * *Why this works*: The Transformer encoder is forced to maintain a continuous, reliable contextual representation of *every* input token, because it never knows which token will be predicted or whether a word is corrupted!
