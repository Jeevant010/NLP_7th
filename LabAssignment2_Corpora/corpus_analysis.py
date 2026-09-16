"""Lab Assignment 2 - Corpora and Corpus Analysis.

Runs the three parts of the assignment from the command line:

    1. Study of the standard NLTK corpora (Brown, Inaugural, Reuters, UDHR)
    2. Loading the supplied files as a plaintext corpus and as a categorized corpus
    3. Conditional Frequency Distribution over the Brown categories

Usage
-----
    python corpus_analysis.py            # run all three parts
    python corpus_analysis.py --part 1   # run only one part
    python corpus_analysis.py --cfd-words money love war

The text files supplied with the assignment are expected in the `corpus_data` folder.
If that folder is missing, the script extracts it from `LA3_Corpus.zip` when the archive
is present next to the script.
"""

import argparse
import os
import zipfile

import nltk
from nltk.corpus import brown, inaugural, reuters, udhr
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize import sent_tokenize

CORPUS_ROOT = "corpus_data"
ZIP_CANDIDATES = ("LA3_Corpus.zip", os.path.join("..", "Assignments", "LA3_Corpus.zip"))
NLTK_RESOURCES = ["brown", "inaugural", "reuters", "udhr", "punkt", "punkt_tab"]
DEFAULT_CFD_WORDS = ["money", "government", "president", "war", "love", "family",
                     "science", "computer"]


def prepare_corpus():
    """Make sure corpus_data exists, extracting it from the zip if necessary."""
    if os.path.isdir(CORPUS_ROOT):
        return CORPUS_ROOT

    for candidate in ZIP_CANDIDATES:
        if os.path.exists(candidate):
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(".")
            print("Extracted", candidate)
            return CORPUS_ROOT

    raise FileNotFoundError(
        f"Neither '{CORPUS_ROOT}' nor the archive was found. Extract LA3_Corpus.zip here."
    )


def download_nltk_data():
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


def show_corpus(name, corpus, max_sentence_files=None, per_file_sentences=False):
    """Print the statistics asked for in the assignment for one NLTK corpus."""
    fileids = corpus.fileids()
    words = corpus.words()
    characters = len(corpus.raw())
    vocabulary = {word.lower() for word in words}

    if per_file_sentences:
        # UDHR contains the same text in 310 languages with different encodings, and NLTK's
        # sentence view over those files is unreliable, so each file is read as raw text and
        # split into sentences on its own.
        sentence_count = sum(len(sent_tokenize(corpus.raw(fid))) for fid in fileids)
        sentence_note = "counted file by file"
    elif max_sentence_files is None:
        sentence_count = len(corpus.sents())
        sentence_note = ""
    else:
        sentence_count = len([sent for fid in fileids[:max_sentence_files]
                              for sent in corpus.sents(fid)])
        sentence_note = f"counted on the first {max_sentence_files} files"

    categories = corpus.categories() if hasattr(corpus, "categories") else []

    print("=" * 62)
    print(name)
    print("=" * 62)
    print("Files available      :", len(fileids))
    if categories:
        print("Categories available :", len(categories), "->", categories)
    else:
        print("Categories available : this corpus is not divided into categories")
    print("Number of characters :", characters)
    print("Number of words      :", len(words))
    print("Number of sentences  :", sentence_count,
          f"({sentence_note})" if sentence_note else "")
    print("Vocabulary size      :", len(vocabulary), "unique words")
    print()

    return {
        "corpus": name,
        "files": len(fileids),
        "categories": len(categories),
        "characters": characters,
        "words": len(words),
        "sentences": sentence_count,
        "vocabulary": len(vocabulary),
    }


def part1_standard_corpora():
    print("\n### 1. Study of Standard Corpora")
    print("Notes: Reuters sentence counting uses the first 200 of its 10,788 files;")
    print("       UDHR sentence counting is done file by file over its 310 languages.\n")

    print("Brown fileids (first 5) :", brown.fileids()[:5])
    print("Brown categories        :", brown.categories())
    print("Brown first sentence    :", brown.sents(brown.fileids()[0])[0])
    print()

    print("Inaugural files (first 5) :", inaugural.fileids()[:5])
    print("Total speeches            :", len(inaugural.fileids()))
    print()

    print("Reuters documents         :", len(reuters.fileids()))
    print("Reuters categories        :", len(reuters.categories()))
    print()

    english_fileids = [fid for fid in udhr.fileids() if fid.lower().startswith("english")]
    print("UDHR language files       :", len(udhr.fileids()))
    print("UDHR fileids (first 10)   :", udhr.fileids()[:10])
    print("English version file      :", english_fileids)
    print("English first sentence    :", udhr.sents(english_fileids[0])[0])
    print()

    stats = [
        show_corpus("Brown Corpus", brown),
        show_corpus("Inaugural Corpus", inaugural),
        show_corpus("Reuters Corpus", reuters, max_sentence_files=200),
        show_corpus("UDHR Corpus", udhr, per_file_sentences=True),
    ]

    header = (f"{'Corpus':<18}{'Files':>8}{'Categories':>12}{'Characters':>12}"
              f"{'Words':>10}{'Sentences':>11}{'Vocab':>10}")
    print(header)
    print("-" * len(header))
    for row in stats:
        print(f"{row['corpus']:<18}{row['files']:>8}{row['categories']:>12}"
              f"{row['characters']:>12}{row['words']:>10}{row['sentences']:>11}"
              f"{row['vocabulary']:>10}")


def file_statistics(name, corpus, fileid):
    """Return the basic statistics for one file of a corpus."""
    words = corpus.words(fileid)
    return {
        "file": name,
        "characters": len(corpus.raw(fileid)),
        "words": len(words),
        "sentences": len(corpus.sents(fileid)),
        "vocabulary": len({word.lower() for word in words}),
    }


def print_statistics_table(rows):
    header = (f"{'File / Category':<26}{'Characters':>11}{'Words':>8}"
              f"{'Sentences':>11}{'Unique words':>14}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['file']:<26}{row['characters']:>11}{row['words']:>8}"
              f"{row['sentences']:>11}{row['vocabulary']:>14}")


def part2_own_corpus():
    print("\n### 2. Create and Use Your Own Corpus")

    plain_root = os.path.join(CORPUS_ROOT, "plain")
    plain_corpus = PlaintextCorpusReader(plain_root, r".*\.txt")

    print("\n-- Plaintext corpus --")
    print("Files available:", plain_corpus.fileids())
    print("\nRaw text of text1.txt:")
    print(plain_corpus.raw("text1.txt"))
    print("\nSentences of text1.txt:")
    for sentence in plain_corpus.sents("text1.txt"):
        print("   ", sentence)
    print()
    print_statistics_table([file_statistics(fid, plain_corpus, fid)
                            for fid in plain_corpus.fileids()])

    categorized_root = os.path.join(CORPUS_ROOT, "categorized")
    categorized_corpus = CategorizedPlaintextCorpusReader(
        categorized_root, r".*\.txt", cat_pattern=r"(\w+)/.*"
    )

    print("\n-- Categorized corpus --")
    print("Categories available:", categorized_corpus.categories())
    for category in categorized_corpus.categories():
        print(f"  {category:<12} -> {categorized_corpus.fileids(categories=category)}")
    print()
    print_statistics_table([file_statistics(fid, categorized_corpus, fid)
                            for category in categorized_corpus.categories()
                            for fid in categorized_corpus.fileids(categories=category)])

    return categorized_corpus


def part3_conditional_frequency_words(words):
    print("\n### 3. Conditional Frequency Distribution")

    brown_cfd = nltk.ConditionalFreqDist(
        (category, word.lower())
        for category in brown.categories()
        for word in brown.words(categories=category)
    )

    print("Conditions (genres):", brown_cfd.conditions())
    print("\nCounts of the selected words in each genre")
    brown_cfd.tabulate(samples=words)

    print("\nMost common words in the 'news' genre:")
    print(brown_cfd["news"].most_common(15))

    print("\nGenre in which each selected word is most frequent")
    print(f"{'Word':<12}{'Best genre':<16}{'Count':>7}{'Freq':>10}")
    print("-" * 45)
    for word in words:
        ranking = sorted(brown_cfd.conditions(),
                         key=lambda category: brown_cfd[category].freq(word),
                         reverse=True)
        best = ranking[0]
        print(f"{word:<12}{best:<16}{brown_cfd[best][word]:>7}"
              f"{brown_cfd[best].freq(word):>10.5f}")

    length_cfd = nltk.ConditionalFreqDist(
        (category, len(word))
        for category in brown.categories()
        for word in brown.words(categories=category)
    )

    genres = ["news", "editorial", "religion", "hobbies", "learned", "fiction", "romance"]
    print("\nDistribution of word lengths per genre")
    length_cfd.tabulate(conditions=genres, samples=range(1, 13))

    print("\nShare of words longer than 10 characters")
    for genre in genres:
        total = length_cfd[genre].N()
        long_words = sum(length_cfd[genre][length] for length in range(11, 25))
        print(f"  {genre:<12} {long_words:>6} of {total:>6} words = {long_words / total:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", type=int, choices=[1, 2, 3], help="run only this part")
    parser.add_argument("--cfd-words", nargs="+", default=DEFAULT_CFD_WORDS,
                        help="words to compare across the Brown genres")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # NLTK only reads corpus files from the folders listed in nltk.data.path, so the folder
    # that holds the supplied text files has to be registered before the readers are created.
    project_dir = os.getcwd()
    if project_dir not in nltk.data.path:
        nltk.data.path.append(project_dir)

    prepare_corpus()
    download_nltk_data()

    if args.part in (None, 1):
        part1_standard_corpora()
    if args.part in (None, 2):
        part2_own_corpus()
    if args.part in (None, 3):
        part3_conditional_frequency_words(args.cfd_words)


if __name__ == "__main__":
    main()
