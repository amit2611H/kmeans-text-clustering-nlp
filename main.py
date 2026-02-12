import json
import time
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import rand_score, adjusted_rand_score
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
    """
    Initialize k centroids using the k-means++ algorithm.
    This method chooses initial centroids that are far apart from each other,
    improving convergence speed and final clustering quality.
    """
    n_samples, n_features = feature_vectors.shape
    centroids = np.zeros((k, n_features))

    # Step 1: Choose first centroid randomly from the data points
    first_idx = random.randint(0, n_samples - 1)
    centroids[0] = feature_vectors[first_idx]

    # Step 2: Choose remaining centroids with probability proportional to distance from existing centroids
    for i in range(1, k):
        # Calculate minimum distance from each point to its nearest existing centroid
        distances = np.array([
            min([cosine(x, centroids[c]) for c in range(i)])
            for x in feature_vectors
        ])

        # Handle edge case where all distances are zero
        total = distances.sum()
        if total == 0:
            next_idx = random.randint(0, n_samples - 1)
            centroids[i] = feature_vectors[next_idx]
            continue

        # Convert distances to probabilities (farther points have higher probability)
        probabilities = distances / total
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()

        # Select next centroid using weighted random selection
        j = np.searchsorted(cumulative_probabilities, r)
        if j >= n_samples:
            j = n_samples - 1
        centroids[i] = feature_vectors[j]

    return centroids

def kmeans_clustering(feature_vectors, number_of_clusters, min_cluster_size, iterations=10):
    clusters_centroids = kmeans_plus_plus_initialize(np.array(feature_vectors), number_of_clusters)
    sentences_clusters = {i: [] for i in range(number_of_clusters)}
    num_of_assingments = 1  # to enter the while loop

    while num_of_assingments > 0 and iterations > 0:
        iterations -= 1
        num_of_assingments = 0

        for sentence_id, feature_vector in enumerate(feature_vectors):
            # find the most similar cluster
            best_similarity = -1
            best_cluster = None

            for cluster in range(number_of_clusters):
                # calculate similarity with the cluster's centroid
                similarity = cosine(feature_vector, clusters_centroids[cluster])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if sentence_id not in sentences_clusters.get(best_cluster, []):
                # add to the existing cluster
                sentences_clusters[best_cluster].append(sentence_id)
                # remove sentence from previous clusters
                ensure_unique_member_cluster(feature_vectors, sentences_clusters, clusters_centroids, sentence_id,
                                             best_cluster)
                num_of_assingments += 1

        for cluster_id, members in list(sentences_clusters.items()):
            # recalculate the cluster's centroid
            if len(sentences_clusters[cluster_id]) == 0:
                continue
            new_centroid = (np.mean([feature_vectors[m] for m in sentences_clusters[cluster_id]], axis=0))
            clusters_centroids[cluster_id] = new_centroid
    
    
    return sentences_clusters


def ensure_unique_member_cluster(feature_vectors, sentences_clusters, clusters_centroids, sentence_id, best_cluster):
    for cluster, members in list(sentences_clusters.items()):
        if sentence_id in members and cluster != best_cluster:
            members.remove(sentence_id)
    

def kmeans_cluster_and_evaluate(data_file, encoding_type, invocations):
    print(f'starting kmeans clustering and evaluation with {data_file} and {encoding_type}')
    with (open(data_file, 'r') as f):
        df = pd.read_csv(f, sep='\t', header=None)
        sentences = df[1].tolist()
        true_clusters = df[0].values

    if encoding_type == "TFIDF":
        feature_vectors = tfidf_vectorize(sentences)

    else:
        feature_vectors = sbert_vectorize(sentences)

    number_of_clusters = len(set(true_clusters))

    ri_scores= []
    ari_scores = []

    for _ in range(invocations):
        sentences_clusters = kmeans_clustering(feature_vectors, number_of_clusters, min_cluster_size=20)

        # Convert to labels array
        predicted_labels = np.full(len(sentences), -1, dtype=int)
        for cluster_id, sentence_indices in sentences_clusters.items():
            for idx in sentence_indices:
                predicted_labels[idx] = cluster_id

        # Calculate both scores
        ri_scores.append(rand_score(true_clusters, predicted_labels))
        ari_scores.append(adjusted_rand_score(true_clusters, predicted_labels))

    mean_RI_score = float(np.mean(ri_scores))
    mean_ARI_score = float(np.mean(ari_scores))
    evaluation_results = {'mean_RI_score': mean_RI_score,
                          'mean_ARI_score': mean_ARI_score}

    return evaluation_results


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'],
                                          config["encoding_type"],
                                          config["invocations"])

    for k, v in results.items():
        print(k, v)

    print(f'total time: {round(time.time() - start, 0)} sec')
