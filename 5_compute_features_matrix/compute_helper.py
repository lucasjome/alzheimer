import os
from datetime import datetime

import numpy as np
from scipy.stats import wasserstein_distance

import fs_utils as ut


def load_norm_vector_based_matrix(subject):
    f_matrix_file = f'{subject[1]}/vector_based_matrix_norm1.npz'
    if not os.path.exists(f_matrix_file):
        raise ValueError(
            "Vector-based Matrix doesn\'t exist inside Subject\'s folder")
    loaded = np.load(f_matrix_file, allow_pickle=True)
    norm_vector_based_matrix = loaded['vector_based_matrix_norm1']
    print(f"Loaded Vector-based Features Matrix - {subject[0]}")
    return norm_vector_based_matrix


def load_norm_hist_matrix(subject):
    f_matrix_file = f'{subject[1]}/norm2_hist_matrix.npz'
    if not os.path.exists(f_matrix_file):
        raise ValueError(
            "Histogram Matrix doesn\'t exist inside Subject\'s folder")
    loaded = np.load(f_matrix_file, allow_pickle=True)
    norm_hist_matrix = loaded['norm2_hist_matrix']
    print(f"Loaded Normalized Histogram Matrix - {subject[0]}")
    return norm_hist_matrix


def load_features_matrix(subject):
    f_matrix_file = f'{subject[1]}/features_matrix.npz'
    if not os.path.exists(f_matrix_file):
        raise ValueError(
            "Feature Matrix doesn\'t exist inside Subject\'s folder")
    loaded = np.load(f_matrix_file, allow_pickle=True)
    matrix_glcm = loaded['ma_glcm']
    matrix_rlm = loaded['ma_rlm']
    matrix_lbp = loaded['ma_lbp']
    print(f"Loaded Feature_Labels Matrix - {subject[0]}")
    return matrix_glcm, matrix_rlm, matrix_lbp


def load_non_normalized_features_matrix(subject):
    f_matrix_file = f'{subject[1]}/features_matrix_non_normalized.npz'
    if not os.path.exists(f_matrix_file):
        raise ValueError(
            "Feature Matrix doesn\'t exist inside Subject\'s folder")
    loaded = np.load(f_matrix_file, allow_pickle=True)
    matrix_glcm = loaded['ma_glcm']
    matrix_rlm = loaded['ma_rlm']
    matrix_lbp = loaded['ma_lbp']
    print(f"Loaded Non_Normalized_Feature_Labels Matrix - {subject[0]}")
    return matrix_glcm, matrix_rlm, matrix_lbp


def merge_features_matrix(f_glcm, f_rlm, f_lbp):
    return np.concatenate((f_glcm, f_rlm, f_lbp), axis=1)


def compute_matrix_of_histograms(feature_matrix, bin_auto=False):
    n_labels, n_features = feature_matrix.shape
    histograms_matrix = np.empty(feature_matrix.shape, dtype='object')
    if bin_auto:
        print("Computing bins with: FD")

    for i in range(0, n_labels):
        for j in range(0, n_features):
            if bin_auto:

                hist, bins_edges = np.histogram(
                    feature_matrix[i, j], density=True, bins='auto')
            else:
                bins = np.load('bins_sizes_per_feature.npy')
                hist, bins_edges = np.histogram(
                    feature_matrix[i, j], density=True, bins=bins[j])
            histograms_matrix[i, j] = [hist, bins_edges]

    return histograms_matrix


def get_merged_hist_matrix_subjects(subjects, bin_auto=False):
    nSubjects = len(subjects)

    merged_matrices = np.empty([nSubjects], dtype='object')
    merged_hist_matrices = np.empty([nSubjects], dtype='object')

    for i in range(0, nSubjects):

        f_glcm, f_rlm, f_lbp = load_features_matrix(subjects[i])
        merged_matrix = np.concatenate((f_glcm, f_rlm, f_lbp), axis=1)

        print(f"{i+1}: Loading Histograms Matrix: ", end='')

        hist_matrix = compute_matrix_of_histograms(merged_matrix, bin_auto)

        print(f"{hist_matrix.shape}\n")

        merged_hist_matrices[i] = hist_matrix
        merged_matrices[i] = merged_matrix

    return merged_matrices, merged_hist_matrices


def get_merged_non_normalized_hist_matrix_subjects(subjects, bin_auto=False):
    nSubjects = len(subjects)

    merged_matrices = np.empty([nSubjects], dtype='object')
    merged_hist_matrices = np.empty([nSubjects], dtype='object')

    for i in range(0, nSubjects):

        f_glcm, f_rlm, f_lbp = load_non_normalized_features_matrix(subjects[i])
        merged_matrix = np.concatenate((f_glcm, f_rlm, f_lbp), axis=1)

        print(f"{i+1}: Loading Histograms Matrix: ", end='')

        hist_matrix = compute_matrix_of_histograms(merged_matrix, bin_auto)

        print(f"{hist_matrix.shape}\n")

        merged_hist_matrices[i] = hist_matrix
        merged_matrices[i] = merged_matrix

    return merged_matrices, merged_hist_matrices


def load_norm_hist_matrix_subjects(subjects):
    nSubjects = len(subjects)

    merged_matrices = np.empty([nSubjects], dtype='object')
    merged_hist_matrices = np.empty([nSubjects], dtype='object')

    for i in range(0, nSubjects):
        f_glcm, f_rlm, f_lbp = load_features_matrix(subjects[i])
        merged_matrix = np.concatenate((f_glcm, f_rlm, f_lbp), axis=1)
        norm_hist_matrix = load_norm_hist_matrix(subjects[i])

        print(f"{i+1}: Loading Histograms Matrix: ", end='')
        print(f"{norm_hist_matrix.shape}\n")

        merged_hist_matrices[i] = norm_hist_matrix
        merged_matrices[i] = merged_matrix

    return merged_matrices, merged_hist_matrices


def load_only_norm_hist_matrix_subjects(subjects):
    nSubjects = len(subjects)

    merged_hist_matrices = np.empty([nSubjects], dtype='object')

    for i in range(0, nSubjects):
        norm_hist_matrix = load_norm_hist_matrix(subjects[i])

        print(f"{i+1}: Loading Histograms Matrix: ", end='')
        print(f"{norm_hist_matrix.shape}\n")

        merged_hist_matrices[i] = norm_hist_matrix

    return merged_hist_matrices


def load_only_norm_vector_based_matrix_subjects(subjects):
    nSubjects = len(subjects)

    merged_matrices = np.empty([nSubjects], dtype='object')

    for i in range(0, nSubjects):
        norm_vector_matrix = load_norm_vector_based_matrix(subjects[i])

        print(f"{i+1}: Loading Vector-based Matrix: ", end='')
        print(f"{norm_vector_matrix.shape}\n")

        merged_matrices[i] = norm_vector_matrix

    return merged_matrices

# THRESHOLD


def compute_threshold(mean, std, n):
    return mean + n * std


def read_threshold_features(nFeature, metric):
    loaded = np.load(f"threshold_by_feat_{metric}.npy", allow_pickle=True)
    return loaded[nFeature]

# DISTANCES


def hellinger_dist(i_attrs, j_attrs):
    p = i_attrs[0]
    q = j_attrs[0]

    summ = np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)
    hellinger_pq = np.sqrt(summ) / np.sqrt(2)

    return hellinger_pq


def kl_dist(i_attrs, j_attrs):
    e = 0.0
    p = np.array(i_attrs[0]) + e
    q = np.array(j_attrs[0]) + e

    # KL(p,q)
    kl_pq = p * np.log(p / q)
    dist_pq = np.sum(kl_pq)

    # KL(q,p)
    kl_qp = q * np.log(q / p)
    dist_qp = np.sum(kl_qp)

    # Mean
    kl_symmetric = (dist_pq + dist_qp) / 2

    return kl_symmetric


def wasserstein_dist(i_attrs, j_attrs):
    return wasserstein_distance(i_attrs[0], j_attrs[0], i_attrs[1], j_attrs[1])


DISTANCES = {
    'wasserstein': wasserstein_dist,
    'kl': kl_dist,
    'hellinger': hellinger_dist
}


def compute_distance(metric, i_attrs, j_attrs):
    # i_attrs, j_attrs are lists [bin_edges, Histogram] or just [attributes]

    # Check and compute for metric
    dist = DISTANCES[metric](i_attrs, j_attrs)

    return dist


def compute_distances_for_combinations(row_to_compute, combinations_index, k, metric):
    distances_array = []

    for i, j in combinations_index:

        Hi = row_to_compute[i][0]
        bin_edges_i = row_to_compute[i][1]
        if len(bin_edges_i) > 1:
            bin_edges_i = bin_edges_i[:-1]

        Hj = row_to_compute[j][0]
        bin_edges_j = row_to_compute[j][1]
        if len(bin_edges_j) > 1:
            bin_edges_j = bin_edges_j[:-1]

        # to Avoid NaN values, they will be excluded after thresholding
        if np.isnan(np.sum(Hi)) or np.isnan(np.sum(Hj)):
            distances_array.append(0)
            print("np.nan found")
            continue

        i_attrs = [bin_edges_i, Hi]
        j_attrs = [bin_edges_j, Hj]
        d_i_j = compute_distance(metric, i_attrs, j_attrs)

        distances_array.append(d_i_j)

    return distances_array


# Classfication Helpers

def get_result_percentage(value):
    mean = str(round(np.mean(value)*100, 2))
    std = str(round(np.std(value)*100, 2))
    return mean, std
