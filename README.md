# 🔍 Quora Question Pairs — Duplicate Question Detection

A **Natural Language Processing (NLP) project** that detects whether two questions asked on Quora are duplicates of each other. Given any two questions, the model predicts if they are asking the **same thing** or **different things**.

---

## 📖 Project Description

Quora receives millions of questions and many of them ask the same thing in different ways. Identifying duplicate questions helps:

- **Avoid redundancy** by linking users to already answered questions
- **Improve search quality** by grouping semantically similar questions
- **Save time** for both users and moderators

This project takes two questions as input and predicts whether they are duplicates using a combination of **handcrafted NLP features**, **Word2Vec embeddings**, **TF-IDF vectors**, and a powerful **XGBoost classifier**.

---

## 📊 Dataset

The model was trained on the **Quora Question Pairs** dataset from Kaggle.

| Property | Value |
|---|---|
| Total question pairs | 404,290 |
| Duplicate pairs | 149,263 (36.9%) |
| Non-duplicate pairs | 255,024 (63.1%) |
| Unique questions | 537,929 |
| Repeated questions | 111,778 |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| Test Accuracy | **89.47%** |
| Log Loss | **0.2294** |
| Total Features | 10,039 |
| Training Samples | ~323,000 |

---

## 🛠️ Data Preprocessing

Each question goes through a detailed cleaning pipeline:

- **Lowercasing and stripping** whitespace
- **Symbol normalization** — `%` → percent, `$` → dollar, `₹` → rupee, `€` → euro
- **Number normalization** — `1,000,000` → `1m`, `1,000` → `1k`
- **Contraction expansion** — `can't` → `can not`, `won't` → `will not` (100+ contractions handled)
- **HTML tag removal** using BeautifulSoup (only when HTML tags are detected)
- **Punctuation removal** using regex
- **Multiple spaces collapsed** into single space

---

## ⚙️ Feature Engineering

The project uses **38 handcrafted features** across 4 categories:

### 📏 Basic Features
- Length of question 1 and question 2 (character count)
- Word count of question 1 and question 2
- Number of common words between the two questions
- Total words in both questions combined
- Word share ratio (common words / total words)

### 🔤 Token Features
- Common non-stopword ratio (min and max normalized)
- Common stopword ratio (min and max normalized)
- Common token ratio (min and max normalized)
- Whether first words of both questions match
- Whether last words of both questions match

### 📐 Length Features
- Absolute difference in word count
- Average token length of both questions
- Longest common substring ratio

### 🔀 Fuzzy Features (using FuzzyWuzzy)
- **Fuzz Ratio** — simple character-level similarity
- **Partial Ratio** — best partial string match score
- **Token Sort Ratio** — similarity after sorting words alphabetically
- **Token Set Ratio** — similarity ignoring duplicate words

---

## 🧠 Word2Vec Embeddings

A custom **Word2Vec model** was trained from scratch on all questions in the dataset.

| Parameter | Value |
|---|---|
| Vector size | 300 dimensions |
| Window size | 5 |
| Min word count | 1 |
| Training epochs | 20 |

Each question is converted to a **sentence vector** by averaging the word vectors of all its words. The following similarity features are then computed between the two question vectors:

- **Cosine Similarity** — angle between vectors
- **Euclidean Distance** — straight-line distance
- **Dot Product** — raw vector alignment
- **Manhattan Distance** — sum of absolute differences
- **Difference Norm Ratio** — normalized difference magnitude
- **Vector Length Ratio** — ratio of vector magnitudes
- **Chebyshev Distance** — maximum difference across any single dimension

---

## 📝 TF-IDF Features

A **TF-IDF Vectorizer** was trained on all questions combined to capture word importance.

| Parameter | Value |
|---|---|
| Max features | 5,000 |
| N-gram range | (1, 2) — unigrams and bigrams |
| Sublinear TF | True |
| Min document frequency | 3 |

Each question pair contributes **10,001 TF-IDF features** (5,000 for Q1 + 5,000 for Q2 + 1 cosine similarity score), giving a final feature matrix of **10,039 features** in total.

---

## 🤖 Model — XGBoost Classifier

| Parameter | Value |
|---|---|
| Max estimators | 20,000 |
| Max depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Early stopping rounds | 50 |
| Device | CUDA (GPU accelerated) |
| Eval metric | Log Loss |

The model used **early stopping** and stopped at **4,804 trees** when validation log loss stopped improving, preventing overfitting.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP Preprocessing | BeautifulSoup, NLTK, Regex |
| String Similarity | FuzzyWuzzy, python-Levenshtein, distance |
| Word Embeddings | Gensim Word2Vec |
| TF-IDF | Scikit-learn TfidfVectorizer |
| Classifier | XGBoost |
| Sparse Matrices | SciPy |
| Model Saving | Pickle, XGBoost native JSON |

---
