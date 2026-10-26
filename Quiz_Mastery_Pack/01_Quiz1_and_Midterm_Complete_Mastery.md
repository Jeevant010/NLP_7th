# NLP Quiz 1 & Midterm Exam Retrospective & Trap-Proof Analysis
**Course**: CS-702 (B.Tech 7th Sem) & CS-913 (M.Tech 1st Sem) — IIIT Surat  
**Target**: 100% Score | Zero Execution Errors | Flawless Option Marking

---


## 1. QUIZ 1 (CS-702: B.Tech) — Detailed Question Analysis

### Question 1: Bayesian Spelling Correction Noisy Channel Denominator
* **Question Statement**: When applying the Noisy Channel Model for Bayesian spelling correction, we seek to find candidate word $c$ that maximizes $P(c|w)$. Given Bayes' theorem:
  $$P(c|w) = \frac{P(w|c) \cdot P(c)}{P(w)}$$
  Why can the denominator $P(w)$ be omitted when finding $\arg\max_c$?
* **The Options**:
  * (A) $P(w)$ is a constant scaling factor across all candidates words 'c' for for a fixed observed type 'w'. *(Typo trap)*
  * (B) $P(w)$ depends entirely on the candidate word $c$ rather than the observed text. *(Incorrect)*
  * **(C) $P(w)$ is a constant scaling factor across all candidate words $c$ for a fixed observed typo $w$.** ✅ **[CORRECT]**
  * (D) $P(w)$ represents the language model prior which is independent of the error model. *(Nonsense)*
* **Deep Theoretical Insight**:
  * $\hat{c} = \arg\max_{c \in C} P(c|w) = \arg\max_{c \in C} \frac{P(w|c) P(c)}{P(w)}$.
  * The observed misspelled string $w$ is already typed and fixed.
  * $P(w) = \sum_{c' \in V} P(w|c') P(c')$. This is strictly positive and **independent of $c$**. Dividing every candidate by the exact same positive constant does not change the ranking or the argmax.

---

### Question 2: Bigram Maximum Likelihood Estimation (MLE)
* **Question Statement**: Given a vocabulary size $V$, what does the bigram Maximum Likelihood Estimation (MLE) probability $P(w_n | w_{n-1})$ compute using corpus frequency counts $C$?
* **The Options**:
  * (A) $\frac{C(w_{n-1}, w_n)}{C(w_n)}$
  * **(B) $\frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$** ✅ **[CORRECT]**
  * (C) $\frac{C(w_{n-1}, w_n)}{V}$
  * (D) $\frac{C(w_n)}{V \cdot C(w_{n-1})}$
* **Mathematical Derivation**:
  * By definition of conditional probability:
    $$P(w_n | w_{n-1}) = \frac{P(w_{n-1}, w_n)}{P(w_{n-1})} = \frac{C(w_{n-1}, w_n) / N}{C(w_{n-1}) / N} = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$
  * Common mistake: Putting the target word count $C(w_n)$ in the denominator instead of the conditioning history $C(w_{n-1})$.

---

### Question 3: Perplexity (PPL) Mathematical Definition
* **Question Statement**: How is the Perplexity ($PPL$) of a test text set $W = w_1, w_2, \dots, w_N$ mathematically defined in terms of its token joint probability?
* **The Options**:
  * (A) $PPL(W) = \prod_{i=1}^N P(w_i | w_1, \dots, w_{i-1})$
  * (B) $PPL(W) = -\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_1, \dots, w_{i-1})$
  * **(C) $PPL(W) = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}} = \left(\prod_{i=1}^N P(w_i | w_1, \dots, w_{i-1})\right)^{-\frac{1}{N}}$** ✅ **[CORRECT]**
  * (D) $PPL(W) = \frac{1}{N} \sum_{i=1}^N C(w_{i-1}, w_i)$
* **Deep Intuition**:
  * Perplexity is the reciprocal (inverse) geometric mean of the probability assigned to the sequence by the model:
    $$PPL(W) = \sqrt[N]{\frac{1}{P(w_1, \dots, w_N)}} = 2^{H(W)} = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_{<i})}$$
  * Lower perplexity $\implies$ better, more confident language model.

---

### Question 4: Inflectional vs. Derivational Morphology
* **Question Statement**: What is the fundamental linguistic distinction between inflectional morphology and derivational morphology?
* **The Options**:
  * (A) Inflectional creates entirely new words with different POS...
  * **(B) Inflectional morphology modifies a word for grammatical functions (e.g., tense, number) without changing its part of speech, whereas derivational morphology creates a new word often changing its part of speech or core meaning.** ✅ **[CORRECT]**
  * (C) Inflectional uses only prefixes, derivational only suffixes...
  * (D) Applies to formal vs natural languages...
* **Summary Table for Quick Memory**:
  | Property | Inflectional Morphology | Derivational Morphology |
  | :--- | :--- | :--- |
  | **Function** | Grammatical marking (tense, number, person, case) | Lexeme creation (new concept or syntactic class) |
  | **Part of Speech** | **Preserved** (*walk* [V] $\to$ *walked* [V]) | **Often Changed** (*compute* [V] $\to$ *computer* [N]) |
  | **Productivity** | High & predictable | Semi-productive, semantic shifts occur |
  | **Examples** | `-s`, `-ed`, `-ing`, `-er` (comparative) | `-tion`, `-ize`, `-ness`, `un-`, `re-` |

---

### Question 5: Add-k / Laplace Smoothing Primary Role
* **Question Statement**: What is the primary role of Add-$k$ smoothing (such as Laplace smoothing where $k=1$) in discrete language modeling?
* **Correct Answer**: **(A) To assign non-zero probability estimates to unseen n-grams encountered in test data.**
* **Why it matters**: Zero-probability problem! If an unseen bigram occurs, $P(\text{unseen}) = 0$. In product $\prod P$, the entire sentence probability crashes to $0$, and Perplexity $PPL = \infty$.

---

### Question 6: Definition of a Treebank
* **Question Statement**: What is a Treebank in natural language processing?
* **Correct Answer**: **(B) A corpus in which every sentence is annotated with a parse tree.** (e.g., Penn Treebank).

---

## 2. QUIZ 1 (CS-913: M.Tech) — Key Variations & Gotchas

1. **Trigram Markov Assumption**:
   $$P(w_n | w_1, \dots, w_{n-1}) \approx P(w_n | w_{n-2}, w_{n-1})$$
   *(Conditions strictly on the previous 2 words, total window size = 3).*
2. **Boundary Tokens in Perplexity ($\langle /s \rangle$ vs $\langle s \rangle$)**:
   * **Why include $\langle /s \rangle$?** Because it is an actual predicted event (terminal token) generated by the language model indicating the end of a sentence.
   * **Why exclude $\langle s \rangle$?** Because $\langle s \rangle$ is an initial conditioning history context; the model never predicts $\langle s \rangle$.
3. **WordNet Hypernym vs. Hyponym**:
   * **Hypernym**: Broad/general class ("is-a" superordinate: e.g., *animal* is hypernym of *dog*).
   * **Hyponym**: Specific instance/subclass (subordinate: e.g., *dog* is hyponym of *animal*).
   * *(Mnemonic: **Hyper** = over/above; **Hypo** = under/below).*
4. **Channel Error Model $P(w|c)$**:
   * The likelihood that the typist outputs erroneous surface string $w$ when intending to type word $c$.
5. **Weighted Finite-State Transducer (WFST) Determinization with $\epsilon$-Transitions**:
   * Challenge: $\epsilon$-transitions allow non-deterministic paths without consuming input/output. During determinization, synchronizing delays and weight-pushing can lead to state-space explosion or infinite loops!

---

## 3. Mid-Semester Paper Numerical Calculations (Step-by-Step)

### Problem 1: Bigram Language Model Probability & Laplace Smoothing
Given Corpus:
* $S_1$: $\langle s \rangle$ the cat sat on the mat $\langle /s \rangle$
* $S_2$: $\langle s \rangle$ the dog sat on the log $\langle /s \rangle$
* $S_3$: $\langle s \rangle$ the cat chased the dog $\langle /s \rangle$

#### Step 1: Compute Vocabulary Size ($V$)
Unique words in vocabulary (excluding start/end boundary tokens per standard convention):
$\{\text{the, cat, sat, on, mat, dog, log, chased}\}$ $\implies |V| = 8$.  
*(If special tokens are counted: $\{\langle s \rangle, \langle /s \rangle\} \cup 8 = 10$. Always state your convention explicitly in subjective answers!).*

#### Step 2: Calculate $P_{MLE}(\text{sat} \mid \text{cat})$
* Formula: $P(\text{sat} \mid \text{cat}) = \frac{C(\text{cat, sat})}{C(\text{cat})}$
* Occurrences of "cat":
  * In $S_1$: "the cat sat"
  * In $S_3$: "the cat chased"
  * Total count $C(\text{cat}) = 2$.
* Occurrences of bigram ("cat", "sat"):
  * In $S_1$: 1
  * Total count $C(\text{cat, sat}) = 1$.
* Result:
  $$P_{MLE}(\text{sat} \mid \text{cat}) = \frac{1}{2} = 0.5$$

#### Step 3: Calculate Probability of Sequence $\langle s \rangle\text{ the cat sat }\langle /s \rangle$
By bigram chain rule:
$$P = P(\text{the} \mid \langle s \rangle) \times P(\text{cat} \mid \text{the}) \times P(\text{sat} \mid \text{cat}) \times P(\langle /s \rangle \mid \text{sat})$$
* $C(\langle s \rangle) = 3$. Bigrams starting with $\langle s \rangle$: ("$\langle s \rangle$", "the") occurs 3 times. $\implies P(\text{the} \mid \langle s \rangle) = \frac{3}{3} = 1$.
* Occurrences of "the":
  * $S_1$: "the cat", "the mat"
  * $S_2$: "the dog", "the log"
  * $S_3$: "the cat", "the dog"
  * Total count $C(\text{the}) = 6$.
  * Count $C(\text{the, cat}) = 2$.
  * $\implies P(\text{cat} \mid \text{the}) = \frac{2}{6} = \frac{1}{3}$.
* $P(\text{sat} \mid \text{cat}) = \frac{1}{2}$.
* Occurrences of "sat":
  * $S_1$: "sat on"
  * $S_2$: "sat on"
  * Total count $C(\text{sat}) = 2$.
  * Count $C(\text{sat}, \langle /s \rangle) = 0$ (sat is followed by "on" in both sentences).
  * $\implies P(\langle /s \rangle \mid \text{sat}) = \frac{0}{2} = 0$.
* Overall sequence probability under MLE:
  $$P(\langle s \rangle\text{ the cat sat }\langle /s \rangle) = 1 \times \frac{1}{3} \times \frac{1}{2} \times 0 = 0$$
  *(Classic proof of why MLE fails on unseen transitions and why smoothing is necessary!).*

#### Step 4: Laplace (Add-1) Smoothed Probability $P^*_{Add-1}(\text{barked} \mid \text{dog})$
* General Formula:
  $$P^*_{Add-1}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$
* Given: $C(\text{dog, barked}) = 0$.
* Count $C(\text{dog})$:
  * In $S_2$: "the dog sat"
  * In $S_3$: "the dog $\langle /s \rangle$"
  * Total count $C(\text{dog}) = 2$.
* With $V = 8$:
  $$P^*(\text{barked} \mid \text{dog}) = \frac{0 + 1}{2 + 8} = \frac{1}{10} = 0.1$$
  *(If $V=10$ with special tokens: $\frac{1}{2+10} = \frac{1}{12} \approx 0.0833$).*

---

### Problem 2: Bayesian Noisy Channel Spelling Disambiguation
Misspelled input: $x = \text{"acress"}$.  
Candidates: $w_1 = \text{"actress"}$, $w_2 = \text{"across"}$.

Given probabilities:
1. $P(w_1 = \text{"actress"}) = 1.0 \times 10^{-5}$
2. $P(w_2 = \text{"across"}) = 5.0 \times 10^{-4}$
3. $P(x \mid w_1) = P(\text{"acress"} \mid \text{"actress"}) = 2.0 \times 10^{-3}$ (deletion of 't')
4. $P(x \mid w_2) = P(\text{"acress"} \mid \text{"across"}) = 1.0 \times 10^{-4}$ (substitution 'r' for 'o')

#### Unnormalized Posterior Calculations:
$$\text{Score}(w) = P(x \mid w) \cdot P(w)$$

* For candidate $w_1$ ("actress"):
  $$\text{Score}(w_1) = (2.0 \times 10^{-3}) \times (1.0 \times 10^{-5}) = 2.0 \times 10^{-8}$$
* For candidate $w_2$ ("across"):
  $$\text{Score}(w_2) = (1.0 \times 10^{-4}) \times (5.0 \times 10^{-4}) = 5.0 \times 10^{-8}$$

#### Final Decision:
Comparing the unnormalized posteriors:
$$5.0 \times 10^{-8} (\text{"across"}) > 2.0 \times 10^{-8} (\text{"actress"})$$
$$\hat{w} = \text{"across"}$$
*Key lesson*: Even though the error probability of deleting 't' is 20 times higher ($2 \times 10^{-3}$ vs $1 \times 10^{-4}$), "across" is 50 times more common in the English language prior ($5 \times 10^{-4}$ vs $1 \times 10^{-5}$). The prior dominates, picking **"across"**.
