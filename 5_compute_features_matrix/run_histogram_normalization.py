
import argparse as ap
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import fs_utils as ut
import compute_helper as ch

n_threads = 7


def normalize_element(value, f_range, new_range=(0.01, 1.)):
    new_min, new_max = new_range
    f_min, f_max = f_range

    temp = (value - f_min) / (f_max - f_min)
    scaled = temp * (new_max - new_min) + new_min

    return scaled


def normalize_labels(subject, features_ranges):
    glcm, rlm, lbp = ch.load_features_matrix(subject)

    merged_matrix = np.concatenate((glcm, rlm, lbp), axis=1)
    hists_matrix = ch.compute_matrix_of_histograms(merged_matrix)

    nLabels, nFeatures = hists_matrix.shape

    for i in range(0, nLabels):
        for j in range(0, nFeatures):
            hist, bins_edges = hists_matrix[i, j]
            norm_bins_edges = [normalize_element(
                value, features_ranges[j][i]) for value in bins_edges]
            hists_matrix[i, j] = [hist, norm_bins_edges]
    # save on file
    print(f"Saving normalized histograms matrix for {subject[1]}")
    np.savez_compressed(
        f'{subject[1]}/norm2_hist_matrix', norm2_hist_matrix=hists_matrix)


def run_hist_by_labels_normalization(subjects, features_ranges):
    print(run_hist_by_labels_normalization.__name__)

    p_pool = ProcessPoolExecutor(n_threads)

    for subject in subjects:
        p_pool.submit(normalize_labels, subject, features_ranges)


def fit_hist_by_labels_data(subjects):
    print(fit_hist_by_labels_data.__name__)

    nSubjects = len(subjects)
    print(f"nSubjects: {nSubjects}")
    nFeatures = sum(list(ut.VALID_FILTERS.values()))
    print(f"nFeatures: {nFeatures}")

    list_of_matrices = np.empty([nFeatures], dtype='object')
    aseg_dkt_labels = ut.read_labels_from_csv(ut.ASEG_LABELS_FILE)
    #hp_subfields_labels = ut.read_labels_from_csv(ut.HP_SUBFIELDS_LABELS_FILE)
    nLabels = len(aseg_dkt_labels)  # + len(hp_subfields_labels) * 2
    print(f"nLabels: {nLabels}")
    print()

    # populate list_of_matrices
    for i in range(0, nFeatures):
        #list_of_matrices[i] = []
        list_of_matrices[i] = np.empty([nLabels, nSubjects], dtype='object')
        #list_of_matrices[i] = [[] for j in range(0, nLabels)]
        #list_of_matrices[i].append([[] for k in range(0, nSubjects)])

    for i, subject in enumerate(subjects):
        print(f"{i+1} Loading Subject {subject[1]} ", end='')
        m_glcm, m_rlm, m_lbp = ch.load_features_matrix(subject)  # load
        # merged
        merged_matrix = np.concatenate((m_glcm, m_rlm, m_lbp), axis=1)
        # compute histograms_matrix ij = [hist, bins_edges]
        hists_matrix = ch.compute_matrix_of_histograms(merged_matrix)

        for j, label_row in enumerate(hists_matrix):
            # label_row has nFeatures positions
            for k, values in enumerate(label_row):
                list_of_matrices[k][j][i] = values[1]

    print("Computing normalization min and max values")

    features_ranges = []

    for i, feat_matrix in enumerate(list_of_matrices):
        print(f"feature {i+1}")
        ranges_feat_for_label = []

        for j, label_row in enumerate(feat_matrix):
            l_min = []
            l_max = []

            for k, sub_bins in enumerate(label_row):
                l_sub = np.array(sub_bins)

                l_min.append(l_sub.min())
                l_max.append(l_sub.max())

            l_min_np = np.array(l_min).min()
            l_max_np = np.array(l_max).max()

            ranges_feat_for_label.append((l_min_np, l_max_np))

        features_ranges.append(ranges_feat_for_label)

    return features_ranges


def log_processing(start_time, end_time, output_list):
    if not os.path.exists("logs"):
        os.mkdir("logs")

    script_name = sys.argv[0].split(".")[0]

    with open(f"./logs/{script_name}_processing_{end_time.strftime('%d_%m_%Y_%H_%M')}", 'w') as f:
        f.writelines([f"{script_name}",
                      f"\n{end_time.strftime('%d/%m/%Y')}",
                      f"\nInicio: {start_time}",
                      f"\nFim: {end_time}",
                      f"\nTempo Total: {end_time-start_time}",
                      f"\nImages:\n"])
        f.writelines(output_list)


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Compute Features Matrix for each Label')
    parser.add_argument('--sd', action='store', type=str, nargs='+',
                        required=True, help='Subjects directory')

    args = parser.parse_args()

    # Get Subjects
    subjects = ut.get_subjects_from_args(args.sd)

    print(len(subjects))

    features_ranges = fit_hist_by_labels_data(subjects)
    run_hist_by_labels_normalization(subjects, features_ranges)


if __name__ == "__main__":
    main()

    pass
