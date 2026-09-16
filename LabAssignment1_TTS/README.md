# Lab Assignment 1 — Text-to-Speech Conversion and Performance Analysis

**Course:** Natural Language Processing Lab
**Name:** Jeevant Mudgil  |  **Roll No:** UI23CS30

## Objective

Write a Python program that accepts text through three different input mechanisms,
converts the text into speech, saves a playable audio file, and then measures how the
conversion behaves for different types of input.

The library used is **gTTS** (Google Text-to-Speech). It produces an `.mp3` file and
supports many languages, which is what the language-comparison part needs. Because the
conversion happens on Google's servers, **an internet connection is required** while the
program runs.

## Files

| File | Purpose |
|------|---------|
| `LabAssignment1_TTS.ipynb` | Notebook containing the full assignment, run step by step |
| `text_to_speech.py` | The same work as a single runnable script |
| `requirements.txt` | Packages needed |

`tts_input/` (generated text files) and `tts_output/` (`.mp3` files and plots) are created
automatically on the first run.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

Run the whole assignment from the command line:

```bash
python text_to_speech.py
```

Or run the notebook `LabAssignment1_TTS.ipynb` cell by cell in VS Code / Jupyter.

Extra options of the script:

```bash
python text_to_speech.py --text "Good morning everyone"   # static mode only
python text_to_speech.py --file notes.txt                 # text file mode only
python text_to_speech.py --lang hi --text "नमस्ते"          # another language
```

## What the program does

**1. Input modes**

- Static text input
- Dynamic / user input (typed at run time)
- Text file input

**2. Output shown for each mode**

| Mode | Information displayed |
|------|-----------------------|
| Static text | input text, number of words, conversion time, audio filename, audio file size |
| Dynamic input | input text, number of words, conversion time, audio filename, audio file size |
| Text file | input file size, number of words, number of paragraphs, conversion time, audio filename, audio file size |

**3. Performance experiments** (all in text-file mode)

| Experiment | Files used | Plot saved |
|------------|-----------|------------|
| Number of words | 500, 1000, 2000 words | `words_vs_time.png`, `words_vs_size.png` |
| Number of paragraphs | 1000 words split into 5, 10, 20 paragraphs | `paragraphs_vs_time.png` |
| Special characters | normal / punctuation / symbols / mixed | `special_chars_vs_time.png` |
| Repeated vs different text | repeated sentence vs varied text | `repeated_vs_different_time.png` |
| Language comparison | English, Hindi, Spanish, French | `language_comparison_time.png` |

## Results

The measurements collected on a typical run are stored in the notebook. In summary:

- Conversion time and audio file size both grow with the number of words, since more text
  has to be sent and more audio has to be generated.
- With the word count held at 1000, the three paragraph runs send exactly the same amount of
  text, so any difference between them comes from network and server load rather than from the
  paragraph structure. A single run is not enough to claim a trend here.
- Punctuation is interpreted (it adds pauses and changes intonation), standalone symbols
  such as `@ # $ % &` are ignored, and symbols attached to digits (`$100%`) are interpreted
  and spoken as a quantity.
- Repeated and varied texts with the same number of words take almost the same time, so the
  time depends on the length of the text rather than on how much new vocabulary it contains.
- All four languages convert successfully and their times are of the same order of magnitude,
  well under a second each in the recorded run.

Because gTTS runs on a remote server, the measured times are approximate and vary with
network conditions.
