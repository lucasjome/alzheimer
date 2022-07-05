
import argparse as ap
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.stats as scy

import fs_utils as ut
import compute_helper as ch

n_threads = 7


def compute_feature_statistics(label_feature):
    mean = np.mean(label_feature)
    variance = np.var(label_feature)
    skewness = scy.skew(label_feature)
    kurtosis = scy.kurtosis(label_feature)
    return mean, variance, skewness, kurtosis


def compute_vector_based_matrix_subject(sub, nLabels, nFeatures):

    #merged_matrix = ch.load_norm_hist_matrix(sub)
    f_glcm, f_rlm, f_lbp = ch.load_features_matrix(sub)
    merged_matrix = ch.merge_features_matrix(f_glcm, f_rlm, f_lbp)
    vector_based_matrix = np.empty(merged_matrix.shape, dtype='object')
    s_vols = [l['label_volume']
              for l in ut.read_labels_from_subject_csv(sub)]
    if not len(s_vols) == nLabels:
        print("Labels count doesn't match")
        return None
    for i in range(0, nLabels):
        for j in range(0, nFeatures):
            m, v, s, k = compute_feature_statistics(merged_matrix[i, j])
            vector_based_matrix[i, j] = [np.float64(s_vols[i]), m, v, s, k]
    print(f"Saving vector based with volume matrix for {sub[1]}")
    np.savez_compressed(
        f'{sub[1]}/vector_based_matrix_norm1', vector_based_matrix_norm1=vector_based_matrix)


def compute_vector_based_matrix_subjects(subjects):
    print(compute_vector_based_matrix_subjects.__name__)

    nSubjects = len(subjects)
    print(f"nSubjects: {nSubjects}")

    nFeatures = sum(list(ut.VALID_FILTERS.values()))
    print(f"nFeatures: {nFeatures}")

    aseg_dkt_labels = ut.read_labels_from_subject_csv(subjects[0])
    nLabels = len(aseg_dkt_labels)
    print(f"nLabels: {nLabels}")
    pool = ProcessPoolExecutor(n_threads)

    for sub in subjects:
        pool.submit(compute_vector_based_matrix_subject,
                    sub, nLabels, nFeatures)


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Compute Features Matrix for each Label')
    parser.add_argument('--sd', action='store', type=str, nargs='+',
                        required=True, help='Subjects directory')

    args = parser.parse_args()

    # Get Subjects
    subjects = ut.get_subjects_from_args(args.sd)

    compute_vector_based_matrix_subjects(subjects)


if __name__ == "__main__":
    main()

    pass
