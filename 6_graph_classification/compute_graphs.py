import argparse as ap
import os

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from datetime import datetime
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

import compute_helper as ch
import fs_utils as ut


n_threads = 7


def compute_graph(subject, hist_matrix, vector_matrix, nLabels, nFeatures, i):
    metrics = ['wasserstein', 'hellinger', 'kl']
    node_attrs = ['histogram', 'volume_based']
    combinations_indexes = list(combinations(list(range(0, nLabels)), 2))

    for node_attr in node_attrs:
        for metric in metrics:
            graphs = np.empty((nFeatures), dtype='object')

            print(f"{metric} ")

            for nFeature in range(0, nFeatures):
                print(f"Feature {nFeature} ")

                col = hist_matrix[:, nFeature]
                edges_values = np.array(ch.compute_distances_for_combinations(
                    col, combinations_indexes, i, metric), dtype=np.float64)

                graph = nx.Graph()

                # insert edges, nodes and weights
                print("Inserting: weighted edges, ", end='')
                for edge, w in zip(combinations_indexes, edges_values):
                    graph.add_weighted_edges_from([(edge[0], edge[1], w)])

                # insert node attributes

                print("node attributes")

                attrs = {}
                if node_attr == 'histogram':
                    for j in list(np.unique(combinations_indexes)):

                        hist, bin_edges = hist_matrix[j, nFeature]
                        attrs[j] = {"hist": bin_edges}

                    nx.set_node_attributes(graph, attrs)
                    graphs[nFeature] = graph
                else:
                    for j in list(np.unique(combinations_indexes)):
                        feature_vector = vector_matrix[j, nFeature]
                        corrected = [0.0 if np.isnan(
                            s) else s for s in feature_vector]
                        attrs[j] = {"hist": corrected}
                    nx.set_node_attributes(graph, attrs)
                    graphs[nFeature] = graph
            np.save(f"{subject[1]}/graphs_{node_attr}_{metric}",
                    graphs, allow_pickle=True)
            print()


def compute_graphs(subjects):
    subjects_vector_matrices = ch.load_only_norm_vector_based_matrix_subjects(
        subjects)
    subjects_hist_matrices = ch.load_only_norm_hist_matrix_subjects(
        subjects)

    nLabels = subjects_hist_matrices[0].shape[0]
    nFeatures = sum(list(ut.VALID_FILTERS.values()))

    p_pool = ProcessPoolExecutor(n_threads)

    for i in range(0, len(subjects)):

        p_pool.submit(compute_graph, subjects[i], subjects_hist_matrices[i],
                      subjects_vector_matrices[i], nLabels, nFeatures, i)


def main():

    # Parse Argument
    parser = ap.ArgumentParser(
        description='Compute Complete Graphs for Subjects')
    parser.add_argument('--sd', action='store', type=str, nargs='+',
                        required=True, help='Subjects directory')

    args = parser.parse_args()

    # Get Subjects
    subjects = ut.get_subjects_from_args(args.sd)

    compute_graphs(subjects)


if __name__ == "__main__":
    main()

    pass
