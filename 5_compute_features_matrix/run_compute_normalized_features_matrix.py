
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

n_threads = 6


def normalize_element(value, f_range, new_range=(0.1, 1)):
    new_min, new_max = new_range
    f_min, f_max = f_range

    temp = (value - f_min) / (f_max - f_min)
    scaled = temp * (new_max - new_min) + new_min

    return scaled


def normalize_subject(subject, features_ranges):
    glcm, rlm, lbp = ch.load_non_normalized_features_matrix(subject)
    merged_matrix = np.concatenate((glcm, rlm, lbp), axis=1)
    columns = np.hsplit(merged_matrix, merged_matrix.shape[1])
    norm_feature_matrix = []
    for i, column in enumerate(columns):
        print(f" Normalizing Feature {i+1}")
        norm_column = []
        for j, label_values in enumerate(column):
            norm_values = [normalize_element(
                v, features_ranges[i]) for v in label_values[0]]
            norm_values = np.array(norm_values)
            norm_column.append(norm_values)
        norm_column_reshaped = np.array(
            norm_column, dtype='object').reshape(len(norm_column), 1)
        norm_feature_matrix.append(norm_column_reshaped)
    norm_feature_matrix_reshaped = np.concatenate(
        norm_feature_matrix, axis=1)
    matrix_glcm = norm_feature_matrix_reshaped[:, 0:8]
    matrix_rlm = norm_feature_matrix_reshaped[:, 8:18]
    matrix_lbp = norm_feature_matrix_reshaped[:, 18:]
    # save on file
    np.savez_compressed(
        f'{subject[1]}/features_matrix', ma_glcm=matrix_glcm, ma_rlm=matrix_rlm, ma_lbp=matrix_lbp)


def run_normalization(subjects, features_ranges):
    print(run_normalization.__name__)

    p_pool = ProcessPoolExecutor(n_threads)

    for subject in subjects:
        p_pool.submit(normalize_subject, subject, features_ranges)


def normalize_data(subjects):
    print("compute_and_normalize_data")

    nSubjects = len(subjects)
    print(f"nSubjects: {nSubjects}")
    nFeatures = sum(list(ut.VALID_FILTERS.values()))
    print(f"nFeatures: {nFeatures}")

    list_of_matrices = np.empty([nFeatures], dtype='object')

    # populate list_of_matrices
    for i in range(0, list_of_matrices.shape[0]):
        list_of_matrices[i] = []

    for index, subject in enumerate(subjects):
        m_glcm, m_rlm, m_lbp = ch.load_non_normalized_features_matrix(
            subject)  # load
        # merged
        merged_matrix = np.concatenate((m_glcm, m_rlm, m_lbp), axis=1)
        # list with nFeatures positions
        columns = np.hsplit(merged_matrix, merged_matrix.shape[1])

        for i, column in enumerate(columns):
            row = column.ravel()
            unwraped_row = list(itertools.chain.from_iterable(row))
            list_of_matrices[i].append(np.array(unwraped_row))

    print(len(list_of_matrices[0]))

    features_ranges = []

    print("Computing normalization min and max values for each feature")
    for index, matrix in enumerate(list_of_matrices):
        f_min = []
        f_max = []

        for sub_row in matrix:
            f_min.append(sub_row.min())
            f_max.append(sub_row.max())

        min_value = np.array(f_min).min()
        max_value = np.array(f_max).max()

        features_ranges.append((min_value, max_value))

    run_normalization(subjects, features_ranges)

    return list_of_matrices, features_ranges


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

    normalize_data(subjects)


if __name__ == "__main__":
    main()

    pass
