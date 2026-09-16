# Lab Assignment 2 — Corpora and Corpus Analysis

**Course:** Natural Language Processing Lab
**Name:** Jeevant Mudgil  |  **Roll No:** UI23CS30

## Objective

Study the standard NLTK corpora, build a plaintext corpus and a categorized corpus from
the text files supplied with the assignment, and analyse word usage across genres with a
Conditional Frequency Distribution.

## Files

| File | Purpose |
|------|---------|
| `LabAssignment2_Corpora.ipynb` | Notebook containing the full assignment, run step by step |
| `corpus_analysis.py` | The same work as a single runnable script |
| `requirements.txt` | Packages needed |
| `LA3_Corpus.zip` | The text files supplied with the assignment |
| `corpus_data/` | The supplied files, extracted |

```
corpus_data/
├── plain/                 -> text1.txt, text2.txt, text3.txt
└── categorized/
    ├── education/         -> education1.txt, education2.txt
    ├── sports/            -> sports1.txt, sports2.txt
    └── technology/        -> technology1.txt, technology2.txt
```

`corpus_data/` is extracted automatically from `LA3_Corpus.zip` if it is not present.

## Setup

```bash
pip install -r requirements.txt
```

The NLTK corpora are downloaded automatically by the first cell of the notebook (or by the
script). They are stored in the standard NLTK data directory, outside this project.

## How to run

```bash
python corpus_analysis.py            # all three parts
python corpus_analysis.py --part 1   # only the standard corpora
python corpus_analysis.py --part 2   # only our own corpus
python corpus_analysis.py --part 3   # only the conditional frequency distribution
python corpus_analysis.py --cfd-words money love war
```

Or run the notebook `LabAssignment2_Corpora.ipynb` cell by cell.

## What the program does

**1. Study of standard corpora** — Brown, Inaugural, Reuters and UDHR. For each corpus it
prints the available files and categories and the number of characters, words, sentences
and unique words, using `fileids()`, `categories()`, `raw()`, `words()` and `sents()`.
Reuters contains 10,788 documents, so its sentence count is taken on the first 200 files;
this is stated in the output.

**2. Create and use your own corpus**

- `PlaintextCorpusReader` over `corpus_data/plain`
- `CategorizedPlaintextCorpusReader` over `corpus_data/categorized`, with
  `cat_pattern=r"(\w+)/.*"` so that the folder name becomes the category
- Raw text, words and sentences are accessed from both corpora, and characters, words,
  sentences and unique words are counted per file and per category

**3. Conditional Frequency Distribution** — built over the Brown genres with
`ConditionalFreqDist()`, displayed with `conditions()`, `tabulate()` and `freq()`. Selected
words are compared across genres, the genre in which each word occurs most often is
identified, and a second distribution shows how word length varies from genre to genre.

## Results

- Brown is the only one of the four corpora that has genre categories (15), which is why it
  is used for the conditional frequency work.
- Reuters is the largest, and its 90 categories overlap, so one document can count towards
  several categories.
- UDHR repeats the same text in hundreds of languages, so its overall vocabulary mixes
  scripts and is not comparable with a monolingual corpus; per language it behaves like any
  other small corpus.
- In every corpus the vocabulary is much smaller than the total word count, which shows how
  strongly words repeat in natural language.
- Words that appear in every genre (`the`, `of`, `and`, `to`) do not distinguish genres — they
  top the `news` list and every other genre's list as well. Content words do distinguish them:
  the word with the highest share in each genre is `government` in `government` (115
  occurrences), `president` in `news` (142), `war` in `editorial` (66) and `love` in `romance`
  (36).
- Because genres differ in size, ranking by `freq()` gives a fairer comparison than ranking by
  raw counts, so both are reported. `money`, for example, has its largest raw count in
  `belles_lettres` and `lore` (39 each) but its highest share in `mystery` (33), which is a much
  smaller genre.
- Rare words give unreliable answers: `computer` occurs only four times in the whole corpus
  (all in `science_fiction`) and `science` twelve times (most often in `religion`), so those
  rows should not be over-interpreted.
- The share of words longer than 10 characters separates the genres well: `learned` 4.76%,
  `religion` 3.37%, `hobbies` 3.23%, `editorial` 3.18% and `news` 2.79%, against `fiction`
  1.35% and `romance` 1.15%.
