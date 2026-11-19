"""L1 - Text Preprocessing and Linguistic Analysis

Run: python l1_text_preprocessing.py
Required spaCy model: python -m spacy download en_core_web_sm

This version does not require downloading NLTK's Punkt data. It uses spaCy
for sentence segmentation and NLTK's Treebank tokenizer for word tokenization.
"""

import re

import nltk
import pandas as pd
import spacy


ORIGINAL_TEXT = (
    "Dr. Rao can't attend the NLP workshop at 3:30 p.m. "
    "Visit https://nlp.example.com for updates. "
    "The students are studying Language Technologies!"
)


def clean_text(text: str) -> str:
    """Remove URLs and collapse repeated whitespace without changing casing."""
    text_without_urls = re.sub(r"https?://\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text_without_urls).strip()


def load_spacy_model():
    """Load the English model needed for sentence boundaries, POS and lemmas."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise SystemExit(
            "The spaCy English model is missing. Run this command first:\n"
            "python -m spacy download en_core_web_sm"
        ) from exc


def main() -> None:
    print("=" * 72)
    print("L1 - TEXT PREPROCESSING AND LINGUISTIC ANALYSIS")
    print("=" * 72)

    print("\n1. Original text (unchanged):")
    print(ORIGINAL_TEXT)

    cleaned_text = clean_text(ORIGINAL_TEXT)
    print("\n2. Cleaned text (URL removed; whitespace normalized):")
    print(cleaned_text)

    # spaCy provides sentence boundaries without downloading NLTK Punkt data.
    nlp = load_spacy_model()
    doc = nlp(cleaned_text)
    sentences = [sentence.text for sentence in doc.sents]

    print("\n3. Sentence tokenization (spaCy):")
    for index, sentence in enumerate(sentences, start=1):
        print(f"{index}. {sentence}")

    # preserve_line=True uses NLTK's Treebank tokenizer directly and skips Punkt.
    words = nltk.word_tokenize(cleaned_text, preserve_line=True)
    print("\n4. Word tokenization (NLTK):")
    print(words)

    # Include punctuation to show the actual tokens and POS tags from spaCy.
    token_rows = [
        {
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS Tag": token.pos_,
        }
        for token in doc
        if not token.is_space
    ]
    token_table = pd.DataFrame(token_rows)

    print("\n5. Token-Lemma-POS table (spaCy):")
    print(token_table.to_string(index=False))
    token_table.to_csv("l1_token_lemma_pos.csv", index=False)

    # Remove English stop words and punctuation; retain alphabetic words.
    filtered_lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    print("\n6. Tokens after English stop-word removal and lemmatization:")
    print(filtered_lemmas)

    print("\n7. Stop-word removal observation:")
    print(
        'Example: "The movie is not good." becomes "movie good" if "not" '
        "is removed. The original is negative, but the shortened text can be "
        "misread as positive. Therefore, stop-word removal can change meaning."
    )

    print("\nEvidence saved: l1_token_lemma_pos.csv")


if __name__ == "__main__":
    main()
