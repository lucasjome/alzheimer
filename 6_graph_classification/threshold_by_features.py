import argparse as ap
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

import compute_helper as ch
import fs_utils as ut


def get_mean_std(ad_hists, cn_hists, metric):
    if len(ad_hists) != len(cn_hists):
        raise ValueError("AD and CN are unbalanced")

    nSubjects = len(ad_hists)
    nLabels = ad_hists[0].shape[0]
    nFeatures = sum(list(ut.VALID_FILTERS.values()))

    combinations_indexes = list(combinations(list(range(0, nLabels)), 2))

    threshold_by_feat = np.empty([nFeatures], dtype='object')

    for feat in range(0, nFeatures):
        means = np.zeros([nSubjects], dtype=np.float64)
        for i in range(0, nSubjects):
            print(f"Feature {feat}")
            print(i+1)

            ad_s = ad_hists[i]
            cn_s = cn_hists[i]

            ad_col = ad_s[:, feat]
            cn_col = cn_s[:, feat]

            # compute distances for each combination (edges)
            ad_dists = np.array(ch.compute_distances_for_combinations(
                ad_col, combinations_indexes, i+1, metric), dtype=np.float64)
            cn_dists = np.array(ch.compute_distances_for_combinations(
                cn_col, combinations_indexes, i+1, metric), dtype=np.float64)
            print(f"Number of edges: {ad_dists.shape}")
            # Compute difference of edge distances
            diff = cn_dists - ad_dists
            print(f"Max difference value: {np.max(diff)}")
            print(f"Min difference value: {np.min(diff)}")
            # absolute values of differences
            abs_diff = np.absolute(diff)
            # mean of
            mean = np.mean(abs_diff)
            print(f"Mean: {mean}\n")
            means[i] = mean

        mean_all = np.mean(means)
        std_all = np.std(means)
        threshold_by_feat[feat] = (mean_all, std_all)

    np.save(f"threshold_by_feat_{metric}",
            threshold_by_feat, allow_pickle=True)
    print("Saved Threshold values for each feature")
    return threshold_by_feat


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Compute Threshold values for each Feature')
    parser.add_argument('--ad', action='store', type=str, nargs='+',
                        required=True, help='AD\'s Subjects directory')

    parser.add_argument('--cn', action='store', type=str, nargs='+',
                        required=True, help='CN\'s Subjects directory')
    parser.add_argument('--dist', action='store', type=str, nargs=1,
                        required=True, help='Distance Metrics: wasserstein, kl, hellinger',
                        choices=list(ch.DISTANCES.keys()))

    args = parser.parse_args()

    # Get Subjects
    ad_subjects = ut.get_subjects_from_args(args.ad)
    cn_subjects = ut.get_subjects_from_args(args.cn)

    ad_hists_v = ch.load_only_norm_hist_matrix_subjects(ad_subjects)
    cn_hists_v = ch.load_only_norm_hist_matrix_subjects(cn_subjects)

    metric = args.dist[0]
    print(metric)

    get_mean_std(ad_hists_v, cn_hists_v, metric)


if __name__ == "__main__":
    main()

    pass
