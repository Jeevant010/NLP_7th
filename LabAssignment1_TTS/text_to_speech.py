"""Lab Assignment 1 - Text-to-Speech Conversion and Performance Analysis.

A command line program that converts text into speech through three different input
mechanisms and then runs the performance experiments asked for in the assignment.

Usage
-----
    python text_to_speech.py                    # run the three input modes + all experiments
    python text_to_speech.py --text "Good morning everyone"
    python text_to_speech.py --file input.txt
    python text_to_speech.py --lang hi --text "नमस्ते"

Generated files are written to tts_output/ and the plots are saved there as PNG images.
The library used is gTTS (Google Text-to-Speech), so an internet connection is required.
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")  # write the plots to PNG files instead of opening a window

import matplotlib.pyplot as plt
from gtts import gTTS

INPUT_DIR = "tts_input"
OUTPUT_DIR = "tts_output"
SENTENCES = [
    "Natural language processing helps computers understand human language.",
    "A corpus is a large collection of text used for linguistic research.",
    "Speech synthesis converts written words into spoken audio.",
    "Machine learning models learn patterns from large amounts of data.",
    "Grammar and vocabulary together decide the meaning of a sentence.",
    "Text analytics is used in search engines and recommendation systems.",
    "Automatic translation allows people to read text written in other languages.",
    "Punctuation gives the reader information about pauses and sentence boundaries.",
    "Voice assistants answer questions using speech recognition and synthesis.",
    "Language models are trained on books, articles and web pages.",
]


def count_paragraphs(text):
    """A paragraph is a non-empty block of text separated from the next by a blank line."""
    blocks = text.strip().split("\n\n")
    return len([block for block in blocks if block.strip()])


def convert_to_speech(text, audio_name, lang="en"):
    """Convert `text` into speech, save the mp3 in OUTPUT_DIR and return (path, stats)."""
    audio_path = os.path.join(OUTPUT_DIR, audio_name)

    start = time.perf_counter()
    gTTS(text=text, lang=lang).save(audio_path)
    conversion_time = time.perf_counter() - start

    stats = {
        "words": len(text.split()),
        "conversion_time": conversion_time,
        "audio_name": audio_name,
        "audio_size": os.path.getsize(audio_path),
    }
    return audio_path, stats


def print_stats(stats, input_desc):
    print("Input                :", input_desc)
    print("Number of words      :", stats["words"])
    print("Conversion time      :", round(stats["conversion_time"], 4), "seconds")
    print("Output audio filename:", stats["audio_name"])
    print("Output audio file size:", stats["audio_size"], "bytes")


def build_text(word_count):
    """Build a text of exactly `word_count` words by repeating the sentence pool."""
    words = []
    index = 0
    while len(words) < word_count:
        words.extend(SENTENCES[index % len(SENTENCES)].split())
        index += 1
    return " ".join(words[:word_count])


def build_paragraphs(total_words, paragraph_count):
    """Split `total_words` words into `paragraph_count` roughly equal paragraphs."""
    words = build_text(total_words).split()
    per_paragraph = total_words // paragraph_count

    paragraphs = []
    for i in range(paragraph_count):
        start = i * per_paragraph
        end = total_words if i == paragraph_count - 1 else start + per_paragraph
        paragraphs.append(" ".join(words[start:end]))

    return "\n\n".join(paragraphs)


def save_plot(x, y, title, xlabel, ylabel, filename, color, marker="o"):
    """Save a line plot as PNG."""
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker=marker, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def save_bar_plot(labels, values, title, xlabel, ylabel, filename, color):
    """Save a bar plot as PNG."""
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


# --------------------------------------------------------------------------------------
# 1. Input modes
# --------------------------------------------------------------------------------------
def mode_static(text):
    print("\n[1.1] Static Text Input")
    _, stats = convert_to_speech(text, "static_output.mp3")
    print_stats(stats, text)


def mode_user_input():
    print("\n[1.2] Dynamic / User Input")
    user_text = input("Enter the text to convert into speech: ").strip()

    if not user_text:
        user_text = "No text was entered, so this default sentence is read aloud."

    _, stats = convert_to_speech(user_text, "dynamic_output.mp3")
    print_stats(stats, user_text)


def mode_text_file(path):
    print("\n[1.3] Text File Input")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Speech synthesis converts written text into spoken audio.\n\n"
                "A text to speech system first analyses the sentence structure, then it "
                "predicts the pronunciation of every word, and finally it generates the "
                "audio waveform.\n\n"
                "The quality of the generated speech depends on the voice model, the "
                "language, and the punctuation present in the text."
            )
        print("Created a sample file at", path)

    with open(path, "r", encoding="utf-8") as f:
        file_text = f.read()

    _, stats = convert_to_speech(file_text, "file_output.mp3")

    print("Input file size      :", os.path.getsize(path), "bytes")
    print("Number of words      :", stats["words"])
    print("Number of paragraphs :", count_paragraphs(file_text))
    print("Conversion time      :", round(stats["conversion_time"], 4), "seconds")
    print("Output audio filename:", stats["audio_name"])
    print("Output audio file size:", stats["audio_size"], "bytes")


# --------------------------------------------------------------------------------------
# 2. Performance experiments
# --------------------------------------------------------------------------------------
def experiment_words():
    print("\n[2.1] Effect of Number of Words")
    results = []

    for target in [500, 1000, 2000]:
        text = build_text(target)
        with open(os.path.join(INPUT_DIR, f"words_{target}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  converting {target} words ...")
        _, stats = convert_to_speech(text, f"words_{target}.mp3")
        results.append((target, stats["conversion_time"], stats["audio_size"] / 1024))

    words = [row[0] for row in results]
    times = [row[1] for row in results]
    sizes = [row[2] for row in results]

    save_plot(words, times, "Number of Words vs Conversion Time", "Number of words",
              "Conversion time (seconds)", "words_vs_time.png", "steelblue")
    save_plot(words, sizes, "Number of Words vs Output Audio File Size", "Number of words",
              "Audio file size (KB)", "words_vs_size.png", "darkorange")

    for words, conv_time, size_kb in results:
        print(f"  {words:>5} words -> {conv_time:8.4f} s, {size_kb:8.2f} KB")


def experiment_paragraphs(total_words=1000):
    print("\n[2.2] Effect of Number of Paragraphs (constant word count)")
    results = []

    for count in [5, 10, 20]:
        text = build_paragraphs(total_words, count)
        with open(os.path.join(INPUT_DIR, f"paragraphs_{count}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        _, stats = convert_to_speech(text, f"paragraphs_{count}.mp3")
        results.append((count, stats["conversion_time"]))

    save_plot([row[0] for row in results], [row[1] for row in results],
              "Number of Paragraphs vs Conversion Time", "Number of paragraphs",
              "Conversion time (seconds)", "paragraphs_vs_time.png", "purple")

    for count, conv_time in results:
        print(f"  {count:>2} paragraphs -> {conv_time:8.4f} s")


def experiment_special_characters():
    print("\n[2.3] Effect of Special Characters")
    base_words = build_text(60).split()

    text_a = " ".join(base_words)
    text_b = ", ".join(" ".join(base_words[i:i + 6]) for i in range(0, len(base_words), 6)) + "!"

    symbol_words = list(base_words)
    for i in range(4, len(symbol_words), 6):
        symbol_words[i] = "@#$%&"
    text_c = " ".join(symbol_words)

    mixed_words = list(base_words)
    for i in range(4, len(mixed_words), 6):
        mixed_words[i] = "$100%"
    text_d = "; ".join(" ".join(mixed_words[i:i + 6]) for i in range(0, len(mixed_words), 6)) + "?"

    texts = {"A - Normal": text_a, "B - Punctuation": text_b,
             "C - Symbols": text_c, "D - Mixed": text_d}

    results = []
    for label, text in texts.items():
        with open(os.path.join(INPUT_DIR, f"special_{label[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        _, stats = convert_to_speech(text, f"special_{label[0]}.mp3")
        results.append((label, stats["conversion_time"]))
        print(f"  {label:<16} tokens = {len(text.split()):>3} -> {stats['conversion_time']:.4f} s")

    save_bar_plot([row[0] for row in results], [row[1] for row in results],
                  "Special-Character Category vs Conversion Time", "Text category",
                  "Conversion time (seconds)", "special_chars_vs_time.png", "seagreen")


def experiment_repeated_vs_varied():
    print("\n[2.4] Repeated vs Different Text")
    repeated_text = "Natural language processing helps computers understand human language. " * 12
    varied_text = build_text(len(repeated_text.split()))

    with open(os.path.join(INPUT_DIR, "repeated.txt"), "w", encoding="utf-8") as f:
        f.write(repeated_text)
    with open(os.path.join(INPUT_DIR, "varied.txt"), "w", encoding="utf-8") as f:
        f.write(varied_text)

    results = []
    for label, text, audio_name in [("Repeated (File A)", repeated_text, "repeated.mp3"),
                                    ("Varied (File B)", varied_text, "varied.mp3")]:
        _, stats = convert_to_speech(text, audio_name)
        results.append((label, stats["conversion_time"]))
        print(f"  {label}: {len(text.split())} words, "
              f"{len(set(text.lower().split()))} unique -> {stats['conversion_time']:.4f} s")

    save_bar_plot([row[0] for row in results], [row[1] for row in results],
                  "Repeated vs Different Text - Conversion Time", "Text type",
                  "Conversion time (seconds)", "repeated_vs_different_time.png", "indianred")


def experiment_languages():
    print("\n[2.5] Language Comparison")
    language_texts = {
        "English": ("Artificial intelligence is changing the way computers understand language.", "en"),
        "Hindi": ("कृत्रिम बुद्धिमत्ता वह तरीका बदल रही है जिससे कंप्यूटर भाषा समझते हैं।", "hi"),
        "Spanish": ("La inteligencia artificial está cambiando la forma en que las computadoras entienden el lenguaje.", "es"),
        "French": ("L'intelligence artificielle change la façon dont les ordinateurs comprennent le langage.", "fr"),
    }

    results = []
    for language, (text, code) in language_texts.items():
        _, stats = convert_to_speech(text, f"language_{code}.mp3", lang=code)
        results.append((language, stats["conversion_time"]))
        print(f"  {language:<8} ({code}) -> {stats['conversion_time']:.4f} s")

    save_bar_plot([row[0] for row in results], [row[1] for row in results],
                  "Language Comparison - Conversion Time", "Language",
                  "Conversion time (seconds)", "language_comparison_time.png", "slateblue")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--text", help="run only the static text mode with this text")
    parser.add_argument("--file", help="run only the text file mode with this file")
    parser.add_argument("--lang", default="en", help="language code for --text (default: en)")
    args = parser.parse_args()

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.text:
        _, stats = convert_to_speech(args.text, "static_output.mp3", lang=args.lang)
        print_stats(stats, args.text)
        return

    if args.file:
        mode_text_file(args.file)
        return

    mode_static("Text to speech conversion reads written text and produces spoken audio. "
                "It is used in screen readers, voice assistants and language learning tools.")
    mode_user_input()
    mode_text_file(os.path.join(INPUT_DIR, "sample_text.txt"))

    experiment_words()
    experiment_paragraphs()
    experiment_special_characters()
    experiment_repeated_vs_varied()
    experiment_languages()

    print("\nFiles generated in", os.path.abspath(OUTPUT_DIR), ":")
    for name in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, name)) / 1024
        print(f"  {name:<36} {size_kb:>8.2f} KB")


if __name__ == "__main__":
    main()
