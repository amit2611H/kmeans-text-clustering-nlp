# K-Means Text Clustering (NLP)

Custom implementation of the K-Means clustering algorithm for semantic text clustering.

This project explores unsupervised text clustering using two representation methods:

- **TF-IDF**
- **SBERT (Sentence-BERT embeddings)**

Clustering quality is evaluated using:

- **Rand Index (RI)**
- **Adjusted Rand Index (ARI)**

---

## 📌 Project Overview

Clustering is a core unsupervised learning task in NLP.  
In this project, the full K-Means pipeline was implemented manually — without using `sklearn.KMeans`.

The system:

- Encodes text using TF-IDF or SBERT embeddings
- Automatically infers the number of clusters (K) from dataset labels
- Performs multiple clustering runs to reduce randomness
- Computes averaged RI and ARI scores for robust evaluation

---

## 📊 Results (ATIS Dataset)

Configuration:
- Encoding: **SBERT**
- Invocations: **20 runs**

Results:
- **Mean RI Score:** 0.726
- **Mean ARI Score:** 0.102
- **Runtime:** ~32 seconds (20 runs)

The results align with expected performance benchmarks for SBERT-based clustering on the ATIS dataset.

---

## 🧠 Implementation Details

### Text Representation
- `TfidfVectorizer` for sparse lexical features
- `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings

### Clustering Algorithm
- Custom K-Means implementation
- KMeans++ centroid initialization
- Iterative centroid updates until convergence
- Automatic K extraction
- Multiple random initializations for stability

### Evaluation
- `sklearn.metrics.rand_score`
- `sklearn.metrics.adjusted_rand_score`
- Averaged metrics across multiple runs

---

## 🗂 Project Structure

```text
.
├── main.py                  # Main execution file (clustering + evaluation)
├── config.json              # Configuration file (dataset, encoding type, runs)
├── data/
│   ├── atis_data.tsv        # ATIS dataset (labels used only for evaluation)
│   └── news_mix_data.tsv    # Additional dataset
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Recommended Python version: 3.11 or 3.12

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn torch sentence-transformers
```

---

## ▶️ Running

Edit `config.json` if needed:

```json
{
  "data": "data/atis_data.tsv",
  "encoding_type": "SBERT",
  "invocations": 20
}