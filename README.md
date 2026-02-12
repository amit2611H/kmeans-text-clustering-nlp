# K-Means Text Clustering (NLP)

Implementation of the K-Means clustering algorithm from scratch for text data, 
as part of an academic Natural Language Processing assignment.

The project compares two text representation methods:
- **TF-IDF**
- **SBERT (Sentence-BERT embeddings)**

Clustering performance is evaluated using:
- **Rand Index (RI)**
- **Adjusted Rand Index (ARI)**

---

## 📌 Project Overview

Clustering is a fundamental unsupervised learning task in NLP.  
This project implements the full K-Means pipeline without using the `sklearn.KMeans` API.

Key features:
- Custom K-Means implementation
- KMeans++ initialization
- Automatic extraction of K (number of clusters) from labeled dataset
- Support for multiple text encoding methods (TF-IDF / SBERT)
- Multiple invocations with averaged evaluation metrics
- Performance evaluation using RI and ARI

---

## 📊 Results (ATIS Dataset)

Configuration:
- Encoding: SBERT
- Invocations: 20

Results:
- **Mean RI Score:** ~0.726
- **Mean ARI Score:** ~0.102
- Runtime: ~32 seconds (20 runs)

The results are consistent with expected performance for SBERT-based clustering on the ATIS dataset.

---

## 🧠 Implementation Details

- Text is encoded using either:
  - `TfidfVectorizer`
  - `sentence-transformers/all-MiniLM-L6-v2`
- K-Means implemented manually:
  - Custom centroid initialization (KMeans++)
  - Iterative centroid update
  - Convergence detection
- Evaluation:
  - `sklearn.metrics.rand_score`
  - `sklearn.metrics.adjusted_rand_score`

---

## 🗂 Project Structure