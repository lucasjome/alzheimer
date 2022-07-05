
import argparse as ap
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import fs_utils as ut

n_threads = 4


def loop_labels_and_features(subject, features_as_images, labels, hem=None):
    feature_matrix = []
    for label in labels:
        im_label = ut.load_nifti_image(
            ut.get_extracted_label(subject, label['label_index'], hemisphere=hem))
        label_data = np.array(im_label[0])
        indexes_inside_mask = label_data == 1
        features_row = []

        # loop a label for each feature
        for feature in features_as_images:
            feature_data = np.array(feature[0])
            feature_data = feature_data[indexes_inside_mask]
            feature_data = np.ravel(feature_data)
            result = feature_data[np.logical_not(np.isnan(feature_data))]
            features_row.append(result)
        print(f"extracted features for label {label['label_index']} ")
        feature_matrix.append(features_row)
    return feature_matrix


def extract_labels_for_features(subject, features_as_images, labels, hp=False):
    # nlabels x nfeatures
    feature_matrix = []

    # Check if it has Hippocampus flag
    if hp:
        hemispheres = ['lh', 'rh']
        temp_arr = []
        for hemisphere in hemispheres:
            temp_arr.extend(loop_labels_and_features(
                subject, features_as_images, labels, hem=hemisphere))
        feature_matrix.extend(temp_arr)
    else:
        feature_matrix = loop_labels_and_features(
            subject, features_as_images, labels)

    return np.array(feature_matrix, dtype='object')


def compute_features_matrix(subject, labels_aseg):
    # Subject: 0=Name, 1=Path

    x = datetime.now()
    # Save on disk to avoid multiples runs
    # Check if exist on disk, and if has flag to overwrite
    # Run everything

    # get filters name
    filters = [filter_name for filter_name in ut.VALID_FILTERS.keys()]

    # read labels from each csv file
    aseg_dkt_labels = ut.read_labels_from_csv(labels_aseg)
    #hp_subfields_labels = ut.read_labels_from_csv(ut.HP_SUBFIELDS_LABELS_FILE)

    # get all features for each filter
    glcm_features = ut.get_all_filter_features(subject, filters[0])
    rlm_features = ut.get_all_filter_features(subject, filters[1])
    lbp_features = ut.get_all_filter_features(subject, filters[2])

    # Load all Features
    glcm_features_as_images = ut.load_features_as_images(glcm_features)
    rlm_features_as_images = ut.load_features_as_images(rlm_features)
    lbp_features_as_images = ut.load_features_as_images(lbp_features)

    print(f"compute {subject}")

    # Compute GLCM
    matrix_glcm_dkt = extract_labels_for_features(
        subject, glcm_features_as_images, aseg_dkt_labels)
    #matrix_glcm_hp = extract_labels_for_features(subject, glcm_features_as_images, hp_subfields_labels, hp=True)
    #matrix_glcm = np.append(matrix_glcm_dkt, matrix_glcm_hp, axis=0)

    # Compute RLM
    matrix_rlm_dkt = extract_labels_for_features(
        subject, rlm_features_as_images, aseg_dkt_labels)

    #matrix_rlm_hp = extract_labels_for_features(subject, rlm_features_as_images, hp_subfields_labels, hp=True)
    #matrix_rlm = np.append(matrix_rlm_dkt, matrix_rlm_hp, axis=0)

    # Compute LBP
    matrix_lbp_dkt = extract_labels_for_features(
        subject, lbp_features_as_images, aseg_dkt_labels)

    print("# GLCM:")
    print(f"DKT Matrix Shape: {matrix_glcm_dkt.shape}")
    #print(f"HP Matrix Shape: {matrix_glcm_hp.shape}")
    print(f"Final Matrix Shape: {matrix_glcm_dkt.shape}")

    print("# RLM:")
    print(f"DKT Matrix Shape: {matrix_rlm_dkt.shape}")
    #print(f"HP Matrix Shape: {matrix_rlm_hp.shape}")
    print(f"Final Matrix Shape: {matrix_rlm_dkt.shape}")

    print("# LBP:")
    print(f"DKT Matrix Shape: {matrix_lbp_dkt.shape}")
    #print(f"HP Matrix Shape: {matrix_lbp_hp.shape}")
    print(f"Final Matrix Shape: {matrix_lbp_dkt.shape}")

    # Save
    np.savez_compressed(
        f'{subject[1]}/features_matrix_non_normalized',
        ma_glcm=matrix_glcm_dkt,
        ma_rlm=matrix_rlm_dkt,
        ma_lbp=matrix_lbp_dkt)
    print(
        f"Extracted all labels for GLCM, RLM and LBP filters in {datetime.now() - x} - {subject[0]}")
    return matrix_glcm_dkt, matrix_rlm_dkt, matrix_lbp_dkt


def run_compute_features_matrix(subjects, labels_aseg):
    p_pool = ProcessPoolExecutor(n_threads)
    for subject in subjects:
        p_pool.submit(compute_features_matrix, subject, labels_aseg)


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
    parser.add_argument('--labels_aseg', action='store', type=str,
                        required=True, help='ASEG Labels CSV file')

    args = parser.parse_args()

    # Get Subjects
    subjects = ut.get_subjects_from_args(args.sd)

    print(len(subjects))

    run_compute_features_matrix(subjects, args.labels_aseg)


if __name__ == "__main__":
    main()

    pass
