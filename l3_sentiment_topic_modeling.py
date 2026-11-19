"""L3 - Sentiment Classification and Topic Modeling

Implements the tasks in the K.R. Mangalam University L3 practical:
1. Remove duplicate reviews and make a stratified train/test split (random_state=42).
2. Train a TF-IDF + Logistic Regression sentiment classifier.
3. Report precision, recall, F1-score, and a confusion matrix.
4. Run LDA with 2 topics and show the top 5 words of each topic.
5. Assign meaningful topic names, rationales, and a data-based observation.

Expected CSV columns: text,label
Expected labels: Positive and Negative

Run with the faculty dataset (recommended):
    python l3_sentiment_topic_modeling.py reviews.csv

Run with the bundled practice dataset (testing only):
    python l3_sentiment_topic_modeling.py reviews_sample.csv

The sample data is synthetic and is NOT a replacement for the faculty-provided
30-50 labelled reviews required by the practical.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")  # Save plots without opening a separate GUI window.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_TOPICS = 2  # The assignment permits 2 or 3 topics.
EXPECTED_LABELS = {"Negative", "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the L3 sentiment classification and LDA practical."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="reviews.csv",
        help="CSV with text,label columns (default: reviews.csv).",
    )
    return parser.parse_args()


def load_reviews(csv_path: Path) -> pd.DataFrame:
    """Read, validate, normalize, and deduplicate the review dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place the faculty-provided reviews.csv beside this script, or run:\n"
            "  python l3_sentiment_topic_modeling.py reviews_sample.csv\n"
            "The bundled sample is only for testing, not final submission."
        )

    if csv_path.stat().st_size == 0:
        raise ValueError(
            f"The file is empty: {csv_path}\n"
            "Open it and add the CSV header and review rows, or use the sample "
            "dataset for a test run."
        )

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"No CSV columns were found in {csv_path}. The file is empty or "
            "does not contain a CSV header. Expected columns: text,label."
        ) from exc

    df.columns = [str(column).strip().lower() for column in df.columns]
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "The CSV must contain columns named 'text' and 'label'. "
            f"Missing: {sorted(missing_columns)}. Found: {df.columns.tolist()}"
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.title()
    df = df[(df["text"] != "") & (df["label"] != "")].copy()

    # Remove duplicate review text (case-insensitive and whitespace-normalized)
    # before the stratified train/test split, as required by the task.
    normalized_text = (
        df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.loc[~normalized_text.duplicated(keep="first")].reset_index(drop=True)

    unexpected_labels = set(df["label"].unique()) - EXPECTED_LABELS
    if unexpected_labels:
        raise ValueError(
            "Labels must be Positive or Negative (case-insensitive). "
            f"Unexpected labels found: {sorted(unexpected_labels)}"
        )

    if len(df) < 8:
        raise ValueError(
            f"Only {len(df)} unique labelled reviews remain after cleaning. "
            "The assignment asks for approximately 30-50 labelled reviews."
        )
    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both Positive and Negative reviews.")
    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Each sentiment class needs at least two reviews for stratified splitting."
        )

    if not 30 <= len(df) <= 50:
        print(
            f"WARNING: {len(df)} unique reviews were loaded. The assignment asks "
            "for approximately 30-50 labelled reviews."
        )

    return df


def run_sentiment_classifier(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Train/evaluate TF-IDF + Logistic Regression and save evidence."""
    print("\n" + "=" * 76)
    print("PART A - TF-IDF + LOGISTIC REGRESSION SENTIMENT CLASSIFIER")
    print("=" * 76)
    print("\nReview count by sentiment:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(
        f"\nStratified split: {len(X_train)} training reviews, "
        f"{len(X_test)} testing reviews (random_state={RANDOM_STATE})."
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    class_names = ["Negative", "Positive"]
    report = classification_report(
        y_test,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_table = pd.DataFrame(report).transpose()
    metrics_table.index.name = "Class / Average"

    print("\nPrecision, Recall, F1-score and Support:")
    print(metrics_table.round(3).to_string())
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.3f}")
    metrics_table.to_csv(output_dir / "l3_classification_metrics.csv")

    matrix = confusion_matrix(y_test, predictions, labels=class_names)
    matrix_table = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(matrix_table.to_string())
    matrix_table.to_csv(output_dir / "l3_confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(values_format="d")
    plt.title("L3 - Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    plot_path = output_dir / "l3_confusion_matrix.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    test_results = pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "Actual Label": y_test.reset_index(drop=True),
            "Predicted Label": pd.Series(predictions),
        }
    )
    test_results.to_csv(output_dir / "l3_test_predictions.csv", index=False)

    print("\nClassifier evidence saved:")
    print("- l3_classification_metrics.csv")
    print("- l3_confusion_matrix.csv")
    print("- l3_confusion_matrix.png")
    print("- l3_test_predictions.csv")
    return metrics_table, y_test, pd.Series(predictions)


def suggest_topic_label(words: list[str], topic_number: int) -> tuple[str, str]:
    """Suggest an interpretable name using the topic's actual top words."""
    joined = " ".join(words).lower()
    categories = [
        (("battery", "charging", "charge", "backup", "power", "hours", "lasts"),
         "Battery and Charging",
         "The top words refer to battery life, power backup, or charging."),
        (("camera", "photo", "photos", "image", "lens", "picture", "quality"),
         "Camera Quality",
         "The top words point to camera performance, photos, or image quality."),
        (("performance", "speed", "processor", "smooth", "slow", "fast", "lag", "apps"),
         "Performance and Speed",
         "The top words describe device speed, processing, or app performance."),
        (("screen", "display", "brightness", "resolution", "touch"),
         "Display and Screen",
         "The top words relate to the screen, display, or brightness."),
        (("heat", "heating", "hot", "overheats", "overheating", "temperature"),
         "Heating and Reliability",
         "The top words indicate heating or device reliability concerns."),
        (("sound", "audio", "speaker", "volume", "music"),
         "Audio and Speakers",
         "The top words relate to audio output or speaker quality."),
    ]
    for keywords, name, rationale in categories:
        if any(re.search(rf"\b{re.escape(keyword)}\b", joined) for keyword in keywords):
            return name, rationale
    return (
        f"General Product Experience - Topic {topic_number}",
        "The label is a general description because the top words do not strongly match a single predefined product aspect.",
    )


def run_topic_modeling(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LDA, print top words, label topics, and compare topics with sentiment."""
    print("\n" + "=" * 76)
    print("PART B - LDA TOPIC MODELING")
    print("=" * 76)

    vectorizer = CountVectorizer(stop_words="english")
    document_term_matrix = vectorizer.fit_transform(df["text"])
    if document_term_matrix.shape[1] == 0:
        raise ValueError(
            "No usable words remain for LDA after stop-word removal. "
            "Check the review text in the CSV."
        )

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        learning_method="batch",
    )
    topic_distributions = lda.fit_transform(document_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    topic_rows: list[dict[str, str | int]] = []
    topic_names: dict[int, str] = {}
    print(f"\nTop 5 words for each of {N_TOPICS} LDA topics:")

    for topic_index, weights in enumerate(lda.components_, start=1):
        top_indices = weights.argsort()[-5:][::-1]
        top_words = [str(feature_names[index]) for index in top_indices]
        suggested_name, suggested_rationale = suggest_topic_label(top_words, topic_index)
        print(f"\nTopic {topic_index}: {', '.join(top_words)}")
        print(f"Suggested meaningful name: {suggested_name}")
        print(f"Rationale: {suggested_rationale}")
        chosen_name = input(
            "Press Enter to accept the suggested name, or type your own: "
        ).strip()
        chosen_name = chosen_name or suggested_name
        topic_names[topic_index] = chosen_name
        topic_rows.append(
            {
                "Topic": topic_index,
                "Top 5 Words": ", ".join(top_words),
                "Meaningful Topic Name": chosen_name,
                "Rationale": suggested_rationale,
            }
        )

    topic_table = pd.DataFrame(topic_rows)
    print("\nTopic words, names, and rationale:")
    print(topic_table.to_string(index=False))
    topic_table.to_csv(output_dir / "l3_topic_words_and_labels.csv", index=False)

    assignments = df.copy()
    assignments["Dominant Topic"] = topic_distributions.argmax(axis=1) + 1
    assignments["Topic Name"] = assignments["Dominant Topic"].map(topic_names)
    assignments["Topic Probability"] = topic_distributions.max(axis=1).round(4)
    assignments.to_csv(output_dir / "l3_review_topic_assignments.csv", index=False)

    # Compare the assigned sentiment labels with each review's dominant LDA topic.
    cross_tab = pd.crosstab(
        assignments["Topic Name"],
        assignments["label"],
        margins=True,
    )
    print("\nSentiment counts within each dominant topic:")
    print(cross_tab.to_string())
    cross_tab.to_csv(output_dir / "l3_topic_sentiment_comparison.csv")

    observations: list[str] = []
    for topic_name in topic_names.values():
        if topic_name not in assignments["Topic Name"].values:
            continue
        subset = assignments[assignments["Topic Name"] == topic_name]
        counts = subset["label"].value_counts()
        top_count = counts.max()
        top_pct = top_count / len(subset) * 100
        if (counts == top_count).sum() > 1:
            observations.append(
                f"{topic_name}: Positive and Negative reviews are evenly split "
                f"({top_pct:.1f}% each among reviews assigned to this topic)."
            )
        else:
            majority_label = counts.idxmax()
            observations.append(
                f"{topic_name}: {majority_label} reviews are more common "
                f"({top_pct:.1f}% of reviews assigned to this topic)."
            )

    observation_text = (
        "Sentiment classification predicts whether an individual review is "
        "Positive or Negative, whereas LDA groups reviews by recurring words "
        "and product aspects without using sentiment labels.\n"
        + "\n".join(observations)
        + "\nInterpret the topic-wise sentiment proportions cautiously because "
        "LDA is an unsupervised topic model and the dataset is small."
    )
    print("\nObservation comparing classification and topic results:")
    print(observation_text)
    (output_dir / "l3_observation.txt").write_text(observation_text, encoding="utf-8")

    print("\nTopic-modeling evidence saved:")
    print("- l3_topic_words_and_labels.csv")
    print("- l3_review_topic_assignments.csv")
    print("- l3_topic_sentiment_comparison.csv")
    print("- l3_observation.txt")
    return topic_table, assignments


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    output_dir = BASE_DIR
    print("L3 - SENTIMENT CLASSIFICATION AND TOPIC MODELING")
    print(f"Dataset: {csv_path}")

    try:
        reviews = load_reviews(csv_path)
        print(f"\nLoaded {len(reviews)} unique labelled reviews.")
        run_sentiment_classifier(reviews, output_dir)
        run_topic_modeling(reviews, output_dir)
        print("\n" + "=" * 76)
        print("L3 PRACTICAL COMPLETED")
        print("=" * 76)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
