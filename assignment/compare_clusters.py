import json
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def tfidf_vectorize(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(sentences)
    return tfidf_vectors.toarray()


def sbert_vectorize(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_vectors = model.encode(sentences)
    return sbert_vectors


def kmeans_plus_plus_initialize(feature_vectors, k):
    n_samples, n_features = feature_vectors.shape
    centroids = np.zeros((k, n_features))
    first_idx = random.randint(0, n_samples - 1)
    centroids[0] = feature_vectors[first_idx]

    for i in range(1, k):
        distances = np.array([
            min([cosine(x, centroids[c]) for c in range(i)])
            for x in feature_vectors
        ])
        total = distances.sum()
        if total == 0:
            next_idx = random.randint(0, n_samples - 1)
            centroids[i] = feature_vectors[next_idx]
            continue

        probabilities = distances / total
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()
        j = np.searchsorted(cumulative_probabilities, r)
        if j >= n_samples:
            j = n_samples - 1
        centroids[i] = feature_vectors[j]

    return centroids


def kmeans_clustering(feature_vectors, number_of_clusters, iterations=10):
    clusters_centroids = kmeans_plus_plus_initialize(np.array(feature_vectors), number_of_clusters)
    sentences_clusters = {i: [] for i in range(number_of_clusters)}
    num_of_assingments = 1

    while num_of_assingments > 0 and iterations > 0:
        iterations -= 1
        num_of_assingments = 0

        for sentence_id, feature_vector in enumerate(feature_vectors):
            best_similarity = -1
            best_cluster = None

            for cluster in range(number_of_clusters):
                similarity = cosine(feature_vector, clusters_centroids[cluster])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if sentence_id not in sentences_clusters.get(best_cluster, []):
                sentences_clusters[best_cluster].append(sentence_id)
                for other_cluster, members in list(sentences_clusters.items()):
                    if sentence_id in members and other_cluster != best_cluster:
                        members.remove(sentence_id)
                num_of_assingments += 1

        for cluster_id in range(number_of_clusters):
            if len(sentences_clusters[cluster_id]) > 0:
                new_centroid = np.mean([feature_vectors[m] for m in sentences_clusters[cluster_id]], axis=0)
                clusters_centroids[cluster_id] = new_centroid

    return sentences_clusters


def get_cluster_labels(sentences_clusters, n_sentences):
    """Convert cluster dictionary to label array"""
    labels = np.full(n_sentences, -1, dtype=int)
    for cluster_id, sentence_indices in sentences_clusters.items():
        for idx in sentence_indices:
            labels[idx] = cluster_id
    return labels


def compare_clustering_methods(data_file):
    print(f'Comparing TF-IDF vs SBERT clustering on {data_file}\n')
    
    # Load data
    with open(data_file, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None)
        sentences = df[1].tolist()
        true_clusters = df[0].values
    
    number_of_clusters = len(set(true_clusters))
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # TF-IDF clustering
    print("Running TF-IDF clustering...")
    tfidf_vectors = tfidf_vectorize(sentences)
    tfidf_clusters = kmeans_clustering(tfidf_vectors, number_of_clusters)
    tfidf_labels = get_cluster_labels(tfidf_clusters, len(sentences))
    
    # Reset random seed
    random.seed(42)
    np.random.seed(42)
    
    # SBERT clustering
    print("Running SBERT clustering...")
    sbert_vectors = sbert_vectorize(sentences)
    sbert_clusters = kmeans_clustering(sbert_vectors, number_of_clusters)
    sbert_labels = get_cluster_labels(sbert_clusters, len(sentences))
    
    # Find differences
    different_indices = np.where(tfidf_labels != sbert_labels)[0]
    print(f"\nFound {len(different_indices)} texts clustered differently ({len(different_indices)/len(sentences)*100:.1f}%)\n")
    
    # Show examples
    print("="*100)
    print("EXAMPLES OF DIFFERENTLY CLUSTERED TEXTS:")
    print("="*100)
    
    # Sample random examples
    num_examples = min(20, len(different_indices))
    sample_indices = np.random.choice(different_indices, num_examples, replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{i}. Text: \"{sentences[idx]}\"")
        print(f"   True Label: {true_clusters[idx]}")
        print(f"   TF-IDF Cluster: {tfidf_labels[idx]}")
        print(f"   SBERT Cluster: {sbert_labels[idx]}")
        
        # Show other texts in each cluster for context
        tfidf_cluster_texts = [sentences[j] for j in range(len(sentences)) 
                               if tfidf_labels[j] == tfidf_labels[idx]][:3]
        sbert_cluster_texts = [sentences[j] for j in range(len(sentences)) 
                               if sbert_labels[j] == sbert_labels[idx]][:3]
        
        print(f"   TF-IDF cluster examples: {tfidf_cluster_texts}")
        print(f"   SBERT cluster examples: {sbert_cluster_texts}")
    
    print("\n" + "="*100)
    print("INTERPRETATION:")
    print("="*100)
    print("""
TF-IDF (Term Frequency-Inverse Document Frequency):
- Focuses on KEYWORD MATCHING and statistical word importance
- Treats words as independent features (bag-of-words approach)
- Good at capturing explicit word overlap
- May miss semantic similarity if different words are used
- Sensitive to exact word matches and vocabulary overlap

SBERT (Sentence-BERT):
- Captures SEMANTIC MEANING using deep learning embeddings
- Understands context and word relationships
- Can recognize similar meanings expressed with different words
- Handles synonyms, paraphrases, and conceptual similarity better
- Less dependent on exact keyword matches

Expected differences:
1. TF-IDF may group texts with similar words even if meanings differ
2. SBERT may group texts with similar meanings but different words
3. SBERT should handle paraphrases and synonyms better
4. TF-IDF may be more brittle to word choice variations
    """)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    compare_clustering_methods(config['data'])
