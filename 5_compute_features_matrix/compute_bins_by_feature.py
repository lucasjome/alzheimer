import argparse as ap
import os

import numpy as np

import compute_helper as ch
import fs_utils as ut


def compute_bin_fixed_size(ad_hists, cn_hists):
    if len(ad_hists) != len(cn_hists):
        raise ValueError("AD e CN desbalanceados")

    nLabels = ad_hists[0].shape[0]
    nFeatures = ad_hists[0].shape[1]

    values = np.empty([nFeatures], dtype='object')
    for i in range(0, values.size):
        values[i] = []

    for ad_mat, cn_mat in zip(ad_hists, cn_hists):
        # [hist, bins_edge]

        for feat in range(0, nFeatures):
            ad_col = ad_mat[:, feat]
            cn_col = cn_mat[:, feat]

            adf_bins = []
            cnf_bins = []

            for l in range(0, nLabels):
                adf_bins.append(len(ad_col[l][1]))
                cnf_bins.append(len(cn_col[l][1]))

            admeanf = np.mean(adf_bins)
            cnmeanf = np.mean(cnf_bins)

            values[feat].append(admeanf)
            values[feat].append(cnmeanf)

    bins_size_feature = []
    for v in values:
        bins_size_feature.append(np.int64(np.mean(v)))

    print("Saving Bin sizes per Feature")
    np.save('bins_sizes_per_feature', bins_size_feature)
    return None


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Compute number of histogram\'s bins for each Texture Feature')
    parser.add_argument('--ad', action='store', type=str, nargs='+',
                        required=True, help='AD\'s Subjects directory')

    parser.add_argument('--cn', action='store', type=str, nargs='+',
                        required=True, help='CN\'s Subjects directory')

    args = parser.parse_args()

    # Get Subjects
    ad_subjects = ut.get_subjects_from_args(args.ad)
    cn_subjects = ut.get_subjects_from_args(args.cn)

    _, ad_hist_matrices = ch.get_merged_hist_matrix_subjects(ad_subjects, True)
    _, cn_hist_matrices = ch.get_merged_hist_matrix_subjects(cn_subjects, True)

    compute_bin_fixed_size(ad_hist_matrices, cn_hist_matrices)


if __name__ == "__main__":
    main()

    pass
