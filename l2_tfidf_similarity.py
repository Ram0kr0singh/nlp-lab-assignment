"""L2 - TF-IDF and Document Similarity

Run: python l2_tfidf_similarity.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = {
    "D1": "The phone has excellent battery life and fast charging.",
    "D2": "The camera is excellent but battery backup is poor.",
    "D3": "This laptop has fast performance and long battery life.",
    "D4": "The phone takes a long time to charge.",
}
QUERY = "good battery life and fast charging"


def main() -> None:
    print("=" * 72)
    print("L2 - TF-IDF AND DOCUMENT SIMILARITY")
    print("=" * 72)

    doc_ids = list(DOCUMENTS.keys())
    corpus = list(DOCUMENTS.values())

    print("\n1. Given documents:")
    for doc_id, text in DOCUMENTS.items():
        print(f"{doc_id}: {text}")
    print(f"\nQuery: {QUERY}")

    # Fit one vectorizer on the document corpus, then use the same fitted
    # vectorizer to transform the query.
    vectorizer = TfidfVectorizer()
    document_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([QUERY])

    feature_names = vectorizer.get_feature_names_out()
    print("\n2. Vocabulary / features:")
    print(feature_names.tolist())

    tfidf_table = pd.DataFrame(
        document_matrix.toarray(), index=doc_ids, columns=feature_names
    )
    print("\n3. TF-IDF representations of D1-D4:")
    print(tfidf_table.round(3).to_string())
    tfidf_table.to_csv("l2_document_tfidf_matrix.csv")

    query_table = pd.DataFrame(
        query_vector.toarray(), index=["Query"], columns=feature_names
    )
    print("\n4. Query transformed using the same fitted vectorizer:")
    print(query_table.round(3).to_string())

    similarities = cosine_similarity(query_vector, document_matrix).flatten()
    results = pd.DataFrame(
        {"Document": doc_ids, "Cosine Similarity": similarities}
    ).sort_values("Cosine Similarity", ascending=False, kind="stable")
    results["Rank"] = range(1, len(results) + 1)
    results = results[["Document", "Cosine Similarity", "Rank"]]

    print("\n5. Documents ranked from most similar to least similar:")
    print(results.to_string(index=False, formatters={
        "Cosine Similarity": lambda value: f"{value:.4f}"
    }))
    results.to_csv("l2_similarity_ranking.csv", index=False)

    best_document = results.iloc[0]["Document"]
    print(f"\n6. Most relevant document: {best_document}")
    print(
        "Explanation: D1 is the strongest match because it shares the query's "
        "important terms battery, life, fast, and charging."
    )

    print("\nEvidence saved:")
    print("- l2_document_tfidf_matrix.csv")
    print("- l2_similarity_ranking.csv")


if __name__ == "__main__":
    main()
