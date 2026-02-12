import json
import pickle
import time

import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

from utils import evaluate_clustering_result


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def cluster_data(features_file, min_cluster_size, iterations=10):
    with open(features_file, 'rb') as file:
        features = pickle.load(file)

    print(f'starting clustering images in file {features_file}')

    cluster2filenames = dict()
    clusters_centroids = dict()
    num_of_assingments = 1  # to enter the while loop
    min_similarity = 0.74
    cluster_id = 0

    while num_of_assingments > 0 and iterations > 0:
        iterations -= 1
        num_of_assingments = 0

        for picture, feature_vector in features.items():
            # find the most similar cluster
            best_similarity = -1
            best_cluster = None

            for cluster in cluster2filenames.keys():
                # calculate similarity with the cluster's centroid
                similarity = cosine(feature_vector, clusters_centroids[cluster])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if best_similarity < min_similarity:
                # create a new cluster
                initialize_cluster(cluster2filenames, clusters_centroids, picture, feature_vector, cluster_id)
                cluster_id += 1
                num_of_assingments += 1

            elif picture not in cluster2filenames[best_cluster]:
                # add to the existing cluster
                cluster2filenames[best_cluster].append(picture)
                # recalculate the cluster's centroid
                centroid = np.mean([features[m] for m in cluster2filenames[best_cluster]], axis=0)
                clusters_centroids[best_cluster] = centroid
                # remove picture from previous clusters
                ensure_unique_member_cluster(features, cluster2filenames, clusters_centroids, picture, best_cluster)
                num_of_assingments += 1

    cluster2filenames = {cid: members for cid, members in cluster2filenames.items()
                         if len(members) >= min_cluster_size}

    print(f'finished clustering images in file {features_file}, found {len(cluster2filenames)} clusters')

    return cluster2filenames


def ensure_unique_member_cluster(features, cluster2filenames, clusters_centroids, picture, best_cluster):
    for cluster, members in cluster2filenames.items():
        if picture in members and cluster != best_cluster:
            members.remove(picture)
            if len(members) == 0:
                del cluster2filenames[cluster]
                del clusters_centroids[cluster]
                continue
            # recalculate the centroid for the cluster
            old_cluster_centroid = np.mean([features[m] for m in members], axis=0)
            clusters_centroids[cluster] = old_cluster_centroid


def initialize_cluster(cluster2filenames, clusters_centroids, picture, feature_vector, cluster_id):
    cluster2filenames[cluster_id] = [picture]
    clusters_centroids[cluster_id] = feature_vector


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    result = cluster_data(config['features_file'],
                          config['min_cluster_size'],
                          config['max_iterations'])

    evaluation_scores = evaluate_clustering_result(config['labels_file'], result)  # implemented
    
    print(f'total time: {round(time.time()-start, 0)} sec')
