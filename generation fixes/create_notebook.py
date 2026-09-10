import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

cells = []

# ---- Title ----
cells.append(new_markdown_cell("""# Lab 1: Word Embedding — One-Hot Encoding & Bag of Words

**Course:** NLP Lab  |  **Topic:** Text Representation / Word Embeddings

This notebook walks through two fundamental techniques for converting text into
numbers so that machine learning algorithms can process it:

1. **One-Hot Encoding** — a sparse vector representation where each word is a vector
   that is `1` at the word's position in the vocabulary and `0` everywhere else.
2. **Bag of Words (BoW)** — a document-level representation that counts how often
   each word from the vocabulary appears in a document, discarding word order.

---

## The Real-World Problem This Assignment Addresses

Machine learning algorithms — whether a simple classifier or a deep neural network —
**only understand numbers**. A sentence like *"IIIT Surat is a leading institute for
computer science education"* is meaningless to a computer in its raw form. We must
**encode** text as numeric vectors before any NLP model can use it.

This is the classic **Vector Space / Bag-of-Words** problem:

- **Messy & unstructured input:** Raw text is variable-length, free-form, and full of
  noise (punctuation, capitalisation, redundant words like "is" / "a").
- **Fixed-length, structured output:** ML models need fixed-size, well-defined
  feature vectors they can do linear algebra on.

One-hot encoding gives each *word* its own axis in a high-dimensional space, making
every word **equally distant** from every other word (no notion of similarity). The
Bag-of-Words model flips the view: instead of representing words, it represents
**documents** as histograms of word counts over the same vocabulary — the foundation
for document classification, clustering, and information retrieval.

---
"""))

# ---- Word Embedding intro ----
cells.append(new_markdown_cell("""## 1. Background: Word Embedding

**Word embedding** is a means of turning texts into numbers.

We do this because machine learning algorithms can only understand numbers, not plain
texts. In order for a computer to read texts, they have to be encoded as a continuous
vector of numeric values.

### Why not just integers?
A naive idea: assign each word an integer (`"cat"=1`, `"dog"=2`). The problem is that
integers impose a *false* ordinality — your model would think dog is "twice" cat. We
need a representation where **similarity** is meaningful, which is the whole point of
the vector-space models we build here."""))

# ---- One-Hot Encoding theory ----
cells.append(new_markdown_cell("""## 2. One-Hot Encoding ("1-of-N" Encoding)

The simplest method is called **one-hot encoding**, also known as "1-of-N" encoding —
the vector is composed of a single `1` and a number of `0`s.

Given a vocabulary of size **N**, each word is placed at a fixed position and its
one-hot vector has `1` at that position and `0` everywhere else.

### Worked example from the slides

Consider the sentence:

> "I ate an apple and played the piano."

| Word      | Position | One-Hot Vector                  |
|-----------|----------|---------------------------------|
| I         | 1        | `[1,0,0,0,0,0,0,0]`             |
| ate       | 2        | `[0,1,0,0,0,0,0,0]`             |
| ...       | ...      | ...                             |

The **one-hot embedding matrix** stacks every word's vector as a row — this is the
object we will build in Exercise 1."""))

# ---- Exercise 1 ----
cells.append(new_markdown_cell("""## Exercise 1 — One-Hot Encoding with NumPy

**Task:** Represent the following sentence in one-hot encoding using NumPy.

> "IIIT Surat is a leading institute for computer science education"

**Steps to follow:**

1. Convert Text to lower case
2. Tokenize the text
3. Get unique words
4. Sort the word list
5. Get the integer/position of the words
6. Create a vector of each word by marking its position as `1` and the rest as `0`
7. Create a matrix of the found vectors."""))

# Ex1 code
ex1_code = """import numpy as np

# --- Step 0: the input sentence ---
sentence = "IIIT Surat is a leading institute for computer science education"
print("Original sentence:", sentence)
print()

# --- Step 1: Convert text to lower case ---
lower_sentence = sentence.lower()
print("Step 1 - Lower case:", lower_sentence)
print()

# --- Step 2: Tokenize the text ---
# Tokenization = splitting the sentence into individual tokens (words).
# We use split() to break on whitespace after removing the trailing period.
tokens = lower_sentence.replace(".", "").split()
print("Step 2 - Tokens:", tokens)
print()

# --- Step 3: Get unique words ---
# The vocabulary is the set of distinct words in the corpus.
unique_words = list(set(tokens))
print("Step 3 - Unique words (unsorted):", unique_words)
print()

# --- Step 4: Sort the word list ---
# Sorting gives a deterministic, reproducible order for the vocabulary so that
# every run produces the same embedding matrix.
sorted_words = sorted(unique_words)
print("Step 4 - Sorted vocabulary:", sorted_words)
vocab_size = len(sorted_words)
print(f"Vocabulary size: {vocab_size}")
print()

# --- Step 5: Get the integer/position of the words ---
# Build a word -> index mapping so each word knows where its "1" goes.
word2idx = {word: idx for idx, word in enumerate(sorted_words)}
print("Step 5 - Word to index mapping:", word2idx)
print()

# --- Step 6: Create a one-hot vector for each word ---
# Each vector is length == vocab_size; only the word's position is 1.
one_hot_vectors = []
for word in sorted_words:
    vector = np.zeros(vocab_size, dtype=int)
    vector[word2idx[word]] = 1
    one_hot_vectors.append(vector)

print("Step 6 - One-hot vectors (one per word):")
for word in sorted_words:
    print(f"  {word:15s} -> {one_hot_vectors[word2idx[word]]}")
print()

# --- Step 7: Create the embedding matrix ---
# The matrix has one row per word: shape (vocab_size, vocab_size).
one_hot_matrix = np.array(one_hot_vectors)
print("Step 7 - One-Hot Embedding Matrix:")
print(one_hot_matrix)
print()
print("Matrix shape:", one_hot_matrix.shape)"""
cells.append(new_code_cell(ex1_code))

# ---- Bag of Words intro ----
cells.append(new_markdown_cell("""## 3. Bag of Words (BoW)

**Bag of words** is a Natural Language Processing technique of text modeling — it is a
method of **feature extraction** with text data.

### What?
A bag of words is a representation of text that describes the **occurrence of words
within a document**. It is called a *"bag"* of words because any information about the
order or structure of words in the document is **discarded**.

### Why?
- One of the biggest problems with text is that it is **messy and unstructured**.
- Machine learning algorithms prefer **structured**, well-defined **fixed-length**
  inputs.
- The Bag-of-Words technique converts **variable-length texts** into a
  **fixed-length vector** (one column per word in the vocabulary).

### Example (the slides) without preprocessing

|                 | Welcome | to | NLP | Learning | , | Now | start | learning | is | a | good | practice |
|-----------------|---------|----|-----|----------|---|-----|-------|----------|----|---|------|----------|
| **Sentence 1**  | 1       | 1  | 1   | 1        | 1 | 1   | 1     | 0        | 0  | 0 | 0    | 0        |
| **Sentence 2**  | 0       | 0  | 0   | 0        | 0 | 0   | 1     | 1        | 1  | 1 | 1    | 1        |

### Example (the slides) with preprocessing
Preprocessing steps:
1. Convert sentences to **lower case** (capitalisation carries little signal).
2. **Remove special characters** and **stopwords** (words like *is, a, the* carry
   little discriminative information).
3. Build the vocabulary and **score** each sentence by word count.

|                 | welcome | NLP | learning | now | start | good | practice |
|-----------------|---------|-----|----------|-----|-------|------|----------|
| **Sentence 1**  | 1       | 1   | 2        | 1   | 1     | 0    | 0        |
| **Sentence 2**  | 0       | 0   | 1        | 0   | 0     | 1    | 1        |"""))

# ---- Exercise 2 ----
cells.append(new_markdown_cell("""## Exercise 2 — Bag of Words with Sklearn

**Task:** Create a Bag of Words model with Sklearn using the `CountVectorizer()`
function from the scikit-learn library.

| Sentence | Text |
|----------|------|
| **Sentence 1** | "IIIT Surat offers computer science education" |
| **Sentence 2** | "IIIT Surat offers computer engineering education" |

`CountVectorizer` does all of the preprocessing for us: it lower-cases, tokenizes,
builds the vocabulary, and produces a count matrix where each row is a document and
each column is a vocabulary word."""))

ex2_code = """from sklearn.feature_extraction.text import CountVectorizer

# --- The two sentences ---
sentence_1 = "IIIT Surat offers computer science education"
sentence_2 = "IIIT Surat offers computer engineering education"

documents = [sentence_1, sentence_2]
print("Documents:")
for i, doc in enumerate(documents, 1):
    print(f"  Sentence {i}: {doc}")
print()

# --- Create the CountVectorizer (Bag of Words) ---
# CountVectorizer lower-cases text, removes punctuation, tokenizes,
# builds a vocabulary of unique words, and counts their occurrences.
vectorizer = CountVectorizer()

# Fit on the documents and transform them into a count matrix.
bow_matrix = vectorizer.fit_transform(documents)

print("Step 1 - Vocabulary (sorted, with column indices):")
for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
    print(f"  {idx:2d}: {word}")
print()

print("Step 2 - Bag of Words count matrix (shape", bow_matrix.shape, "):")
print(bow_matrix.toarray())
print()

print("Step 3 - Column-wise interpretation:")
feature_names = vectorizer.get_feature_names_out()
for i, doc in enumerate(documents, 1):
    counts = bow_matrix.toarray()[i - 1]
    print(f"  Sentence {i}: {doc}")
    for word, count in zip(feature_names, counts):
        if count > 0:
            print(f"      '{word}': {count}")
print()

print("Comparison:")
print("The two sentences share 5 words: 'IIIT', 'Surat', 'offers', 'computer', 'education'")
print("Sentence 1 uniquely mentions:  'science'")
print("Sentence 2 uniquely mentions:  'engineering'")"""
cells.append(new_code_cell(ex2_code))

# ---- Summary ----
cells.append(new_markdown_cell("""## Summary

| Technique | Represents | Vector space | Similarity notion |
|-----------|-----------|--------------|-------------------|
| **One-Hot Encoding** | each **word** | N-hot vector over vocabulary | none (all words equidistant) |
| **Bag of Words** | each **document** | count vector over vocabulary | overlap / shared vocabulary |

- **One-hot encoding** turns every word into its own axis — dimensions are huge
  (one per vocabulary word) and mutually *orthogonal*, so it cannot express that
  "science" and "engineering" are related.
- **Bag of Words** gives a compact **document** signature: two documents that share
  many vocabulary words end up with similar count vectors, which is exactly what
  downstream classifiers and retrievers exploit.

Both rely on the same idea: **build a vocabulary, then represent text as a vector
indexed by that vocabulary.**"""))

nb["cells"] = cells

# Standard notebook metadata
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.13",
    },
}

out_path = r"D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written to:", out_path)
