<div align="center">

# 📚 NLP Lab Assignment

**Text preprocessing · TF-IDF retrieval · Sentiment classification · Topic modeling**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

</div>

---

## 🎓 About

Lab practicals L1 to L3 for **Natural Language Processing**, submitted at **K.R. Mangalam University** (School of Engineering & Technology, MCA - AI/ML).

| | |
|---|---|
| **Student** | Ram Kumar Singh |
| **Roll No.** | 2501940002 |
| **Submitted to** | Uzra Rahman |

The full submission, with code and output screenshots, is in [`NLP_Assignment_L1_L3.pdf`](NLP_Assignment_L1_L3.pdf).

---

## 🧪 Practicals

| # | Topic | Script | Tools |
|---|-------|--------|-------|
| **L1** | Text preprocessing and linguistic analysis | `l1_text_preprocessing.py` | spaCy, NLTK, regex, pandas |
| **L2** | TF-IDF and document similarity | `l2_tfidf_similarity.py` | scikit-learn, pandas |
| **L3** | Sentiment classification and topic modeling | `l3_sentiment_topic_modeling.py` | scikit-learn, pandas, Matplotlib |

### L1 - Text Preprocessing and Linguistic Analysis
- Removes the URL and normalizes whitespace while keeping the original text unchanged
- Sentence tokenization (spaCy) and word tokenization (NLTK Treebank)
- English stop-word removal and lemmatization
- Token - Lemma - POS table using spaCy, saved to `l1_token_lemma_pos.csv`
- Example showing stop-word removal can change meaning ("The movie is not good.")

### L2 - TF-IDF and Document Similarity
- TF-IDF representation of a four-document corpus
- Query transformed with the same fitted vectorizer
- Cosine similarity and ranking

| Document | Cosine Similarity | Rank |
|----------|-------------------|------|
| D1 | 0.7682 | 1 |
| D3 | 0.4921 | 2 |
| D2 | 0.0733 | 3 |
| D4 | 0.0000 | 4 |

D1 is the most relevant document for the query *"good battery life and fast charging"*.

### L3 - Sentiment Classification and Topic Modeling
- Duplicate removal and stratified train-test split (`random_state=42`)
- TF-IDF + Logistic Regression classifier with precision, recall, F1-score and confusion matrix
- LDA with 2 topics and the top 5 words of each topic

| Item | Result |
|------|--------|
| Dataset | 40 unique labelled reviews (20 Positive, 20 Negative) |
| Split | 30 train / 10 test |
| Accuracy | 0.500 |
| Topic 1 words | battery, apps, charging, processor, time |
| Topic 2 words | camera, photos, phone, look, clear |

The test set has only 10 reviews, so the classifier metrics are indicative rather than conclusive.

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Ram0kr0singh/nlp-lab-assignment.git
cd nlp-lab-assignment

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies and the spaCy model
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Run the practicals
python l1_text_preprocessing.py
python l2_tfidf_similarity.py
python l3_sentiment_topic_modeling.py reviews.csv
```

> L3 asks for a name for each LDA topic. Press **Enter** to accept the suggested name, or type your own.

---

## 📁 Repository Structure

```
nlp-lab-assignment/
├── l1_text_preprocessing.py
├── l2_tfidf_similarity.py
├── l3_sentiment_topic_modeling.py
├── reviews.csv                  # labelled reviews for L3
├── requirements.txt
├── NLP_Assignment_L1_L3.pdf     # final submission
├── outputs/                     # generated CSVs and confusion matrix
└── README.md
```

---

## 🔗 Find Me

[![GitHub](https://img.shields.io/badge/GitHub-Ram0kr0singh-181717?logo=github)](https://github.com/Ram0kr0singh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ram%20Kumar%20Singh-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ram-kumar-singh-516a283a9)
