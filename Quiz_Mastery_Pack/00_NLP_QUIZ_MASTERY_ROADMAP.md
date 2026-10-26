# 🎯 NLP Quiz & Exam Full-Marks Mastery Roadmap
**Target**: CS-702 (B.Tech 7th Sem) / CS-913 (M.Tech 1st Sem) | IIIT Surat  
**Instructor Pattern**: Prof. Shreya Agarwal

---

## 📌 Fast Navigation to Study Modules

1. **[01_Quiz1_and_Midterm_Complete_Mastery.md](file:///home/jeevant/Desktop/NLP_7th/Quiz_Mastery_Pack/01_Quiz1_and_Midterm_Complete_Mastery.md)**
   * Complete teardown of Quiz 1 (CS-702 & CS-913) + Midterm Papers.
   * Full derivations: Bigram MLE counts, Perplexity formula, Laplace Add-1 smoothing, Inflectional vs Derivational morphology, WordNet Hypernyms vs Hyponyms, Noisy Channel Spelling calculations ("acress" vs "actress"/"across").
2. **[02_Distributional_Semantics_TFIDF_and_Embeddings.md](file:///home/jeevant/Desktop/NLP_7th/Quiz_Mastery_Pack/02_Distributional_Semantics_TFIDF_and_Embeddings.md)**
   * Step-by-step solved TF-IDF & Cosine Similarity ("dog bites man" vs "man feeds dog").
   * CBOW vs Skip-Gram (Negative Sampling SGNS loss derivation).
   * FastText subword character $n$-grams solving OOV.
   * 4 Context types (Word window, Dependency/Syntax, Sentence, Document).
   * Pointwise Mutual Information (PMI) formula & rare-word explosion.
   * Paradigmatic vs Syntagmatic clustering experiments.
3. **[03_WSD_and_Yarowsky_Algorithm_Deep_Dive.md](file:///home/jeevant/Desktop/NLP_7th/Quiz_Mastery_Pack/03_WSD_and_Yarowsky_Algorithm_Deep_Dive.md)**
   * Unit-4 Word Sense Disambiguation (Dated 27 Oct!).
   * "Plant" sense identification & hypernym hierarchy across 4 document domains.
   * Yarowsky's algorithm: "One sense per collocation" + "One sense per discourse".
   * 6-step lifecycle + Decision List Log-Likelihood ratio scoring.
   * Adverse divergence conditions (Semantic drift, pun violations, noisy seeds).
   * Naive Bayes for WSD formulation & Laplace smoothing parameter estimation.
4. **[04_Transformers_BERT_GPT_and_Summarisation.md](file:///home/jeevant/Desktop/NLP_7th/Quiz_Mastery_Pack/04_Transformers_BERT_GPT_and_Summarisation.md)**
   * Multi-Document Summarisation: 4 core challenges (redundancy, temporal contradictions, cross-document coreference, coherence).
   * Maximal Marginal Relevance (MMR) mathematical formula & intuition.
   * Rhetorical Structure Theory (RST): Evidence, Result, Elaboration sentence pairs.
   * 4 NLP techniques in Extractive Summarisation (Morphology, Syntax, Lexical Semantics, Discourse parsing).
   * Scaled Dot-Product & Multi-Head Attention equations.
   * BERT vs GPT: Bidirectional MLM vs Causal Autoregressive Masking + the 80/10/10 Masking Discrepancy.
5. **[05_Full_Marks_Quiz_Drill_and_Exam_Protocol.md](file:///home/jeevant/Desktop/NLP_7th/Quiz_Mastery_Pack/05_Full_Marks_Quiz_Drill_and_Exam_Protocol.md)**
   * The "Zero Unticked Answers" Examination Protocol.
   * 12-Question Full Practice Quiz with instant tick checkboxes, trap explanations, and scoring scorecard.

---

## ⚡ Master Formula Cheat Sheet for Rapid Revision

| Concept | Formula | Key Exam Note |
| :--- | :--- | :--- |
| **Bigram MLE** | $P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$ | Denominator is the **history count**, NOT the target word count. |
| **Laplace Add-1 Smoothing** | $P_{Add-1}^*(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$ | $V$ is vocabulary size (types). Never forget to add $V$ in denominator! |
| **Perplexity (PPL)** | $PPL(W) = \left(\prod_{i=1}^N P(w_i \mid w_{<i})\right)^{-\frac{1}{N}}$ | $\langle /s \rangle$ is included in $N$; $\langle s \rangle$ is excluded. |
| **Bayesian Spelling (Noisy Channel)** | $\hat{c} = \arg\max_c P(w \mid c) P(c)$ | Denominator $P(w)$ is constant across candidates $\implies$ omit it! |
| **IDF (Standard Natural Log)** | $IDF(t) = \ln\left(\frac{N}{DF(t)}\right)$ | If word appears in all documents ($DF = N$), $IDF = \ln(1) = 0$! |
| **Pointwise Mutual Info (PMI)** | $\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w)P(c)}$ | Very rare words get artificially inflated scores because $P(w)P(c)$ is tiny. |
| **Maximal Marginal Relevance (MMR)** | $\arg\max_{d_i \in R \setminus S} [\lambda \text{Sim}_1(d_i, Q) - (1-\lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j)]$ | $\lambda$ balances relevance ($\lambda \to 1$) vs diversity / novelty ($\lambda \to 0$). |
| **Attention Mechanism** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | $\sqrt{d_k}$ prevents logits from exploding and saturating the softmax. |
| **Naive Bayes WSD** | $\hat{s} = \arg\max_{s_k} P(s_k) \prod_{i=1}^m P(f_i \mid s_k)$ | Uses conditional independence of context features given the sense. |
