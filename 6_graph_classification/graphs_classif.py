import argparse as ap


from datetime import datetime

from pathlib import Path

import grakel as gk
import networkx as nx
import numpy as np

import compute_helper as ch
import fs_utils as ut
from ncross_validate_kfold import Ncross_validate_Kfold_SVM
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from auxiliarymethods.nx_to_gt import nx2gt

# Mandar pro compute_helper deopis de testar


def threshold_edges(hist_matrix, threshold, topFeature, combinations_indexes, k, metric):
    print(threshold_edges.__name__)
    comb = np.array(combinations_indexes, dtype='object')
    # extrair coluna de features
    col = hist_matrix[:, topFeature]
    # calcular distancias
    dists = np.array(ch.compute_distances_for_combinations(
        col, combinations_indexes, k, metric), dtype=np.float64)
    # pegar indices, isto é, das edges acima do threshold
    indexes = dists > threshold
    # as edges resultantes
    edges = comb[indexes]
    # peso das edges resultantes
    edges_values = dists[indexes]

    return edges, edges_values


def threshold_graph(graph, threshold):
    t_graph = graph.copy()
    t_graph.remove_edges_from(
        [(n1, n2) for n1, n2, w in t_graph.edges(data="weight") if w < threshold])
    if len(list(nx.isolates(t_graph))) > 0:
        print("Isolated node")
        t_graph.remove_nodes_from(list(nx.isolates(t_graph)))

    return t_graph


def get_graphs_from_subjects(class_graphs, nFeature, threshold):
    print(f"Computing Graphs")

    nSubjects = len(class_graphs)
    print(f"nSubjects: {nSubjects}")
    print(f"Threshold: {threshold}")
    thresholded_graphs = []

    for i, graphs in enumerate(class_graphs):
        print(f"Subject: {i+1}")

        graph_feat = graphs[nFeature]
        graph_feat = threshold_graph(graph_feat, threshold)

        print(f"Nodes: {len(graph_feat.nodes)}")
        print(f"Edges: {len(graph_feat.edges)}")
        print()

        thresholded_graphs.append(graph_feat)
    return thresholded_graphs


def convert_graphs_to_grakel(graphs):
    print(convert_graphs_to_grakel.__name__)
    gk_graphs = gk.graph_from_networkx(
        graphs, edge_weight_tag='weight', as_Graph=True, node_labels_tag='hist')  # edge_labels_tag='weight')
    return gk_graphs


def convert_graphs_to_grakel_both(graphs_ad, graphs_cn):
    print(convert_graphs_to_grakel_both.__name__)
    graphs = graphs_ad + graphs_cn
    gk_graphs = gk.graph_from_networkx(
        graphs, node_labels_tag='hist', edge_weight_tag='weight', as_Graph=True)
    return gk_graphs


def check_none(graphs):
    print("check_none")
    for g in graphs:
        if len(g.nodes) == 0:
            return True
    return False


def get_mean_counts(group, graphs, n, nf):
    # count mean vertices
    # count mean edges
    mean_vertices = list()
    mean_edges = list()

    for g in graphs:
        mean_vertices.append(len(g.nodes))
        mean_edges.append(len(g.edges))

    m_vertices = np.mean(mean_vertices)
    m_edges = np.mean(mean_edges)

    line = f"{n}, {group}, {nf}, {m_vertices},{m_edges}\n"

    return line


def convert_nx_graphs_to_gt(graphs):
    print(convert_nx_graphs_to_gt.__name__)
    converted_graphs = []

    for graph in graphs:
        converted_graph = nx2gt(graph)
        converted_graphs.append(converted_graph)

    return converted_graphs


def compute_hgk(kernel, ad_graphs, cn_graphs):
    print(compute_hgk.__name__)

    n_interations = 20
    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    # Weighted Edges: False (NEW)
    kernel_parameters_sp = [False, False, 0, False]

    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    # Number of iterations for WL: 3
    kernel_parameters_wl = [3, False, False, 0]

    # Convert all graphs from networkx to Graph-Tool
    ad_converted_graphs = convert_nx_graphs_to_gt(ad_graphs)
    cn_converted_graphs = convert_nx_graphs_to_gt(cn_graphs)
    converted_graphs = ad_converted_graphs + cn_converted_graphs

    if kernel == 'HGK_WL':
        # Compute gram matrix for HGK-WL
        gram_matrix = rbk.hash_graph_kernel(converted_graphs, wl.weisfeiler_lehman_subtree_kernel, kernel_parameters_wl,
                                            n_interations,
                                            scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

        # Normalize gram matrix
        # gram_matrix = aux.normalize_gram_matrix(gram_matrix)
        return gram_matrix

    if kernel == 'HGK_SP':

        gram_matrix = rbk.hash_graph_kernel(converted_graphs, sp_exp.shortest_path_kernel, kernel_parameters_sp,
                                            n_interations,
                                            scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

        # Normalize gram matrix
        # gram_matrix = aux.normalize_gram_matrix(gram_matrix)
        return gram_matrix

    if kernel == 'HGK_WSP':
        # Enable Weighted Shortest-Path calculation
        kernel_parameters_sp[-1] = True

        gram_matrix = rbk.hash_graph_kernel(converted_graphs, sp_exp.shortest_path_kernel, kernel_parameters_sp,
                                            n_interations,
                                            scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

        # Normalize gram matrix
        # gram_matrix = aux.normalize_gram_matrix(gram_matrix)
        return gram_matrix

    return None


def classif(class1_name, class2_name, class1_graphs, class2_graphs, n_thres, results_dir, metric, vector_based, kernel):
    results = []

    class1_lines = []
    class2_lines = []

    nFeatures = sum(list(ut.VALID_FILTERS.values()))
    print(f"nFeatures: {nFeatures}")

    for feat in range(0, nFeatures):
        print(f"Feature: {feat}")
        mean, std = ch.read_threshold_features(feat, metric)
        print(f"mean: {mean}, std: {std}")
        print(f"feat number: {feat}")

        class1_thresholded_graphs = get_graphs_from_subjects(
            class1_graphs, feat, ch.compute_threshold(mean, std, n_thres))
        class2_thresholded_graphs = get_graphs_from_subjects(
            class2_graphs, feat, ch.compute_threshold(mean, std, n_thres))

        class1_line = get_mean_counts(
            class1_name, class1_thresholded_graphs, n_thres, feat)
        class2_line = get_mean_counts(
            class2_name, class2_thresholded_graphs, n_thres, feat)

        class1_lines.append(class1_line)
        class2_lines.append(class2_line)

        check = check_none(class1_thresholded_graphs)
        if check == True:
            print(check)
            continue

        check = check_none(class2_thresholded_graphs)
        if check == True:
            print(check)
            continue

        y_class1 = np.ones(len(class1_thresholded_graphs))
        y_class2 = np.zeros(len(class2_thresholded_graphs))
        y = np.append(y_class1, y_class2)

        if kernel == 'P2K':
            # Propagation Kernel
            all_gk = convert_graphs_to_grakel_both(
                class1_thresholded_graphs, class2_thresholded_graphs)

            graphs_to_gk = []
            for x in all_gk:
                graphs_to_gk.append(x)

            gks = gk.PropagationAttr(n_jobs=7, verbose=True, w=1)
            K = gks.fit_transform(graphs_to_gk)

        else:
            # Hash Graph Kernel
            K = compute_hgk(kernel, class1_thresholded_graphs,
                            class2_thresholded_graphs)

        # C-Grid
        C_grid = (10. ** np.arange(-4, 6, 1) / 100).tolist()

        try:
            accs, f1, auc, prec, sens, c_sens, c_spec = Ncross_validate_Kfold_SVM(
                [K], y, n_iter=10, C_grid=C_grid)

        except Exception as err:
            with open('error.log', 'a') as f:
                f.writelines([f'\nERROR ',
                              f'Feature Number: {feat}\n',
                              f'Threshold number: {n_thres}\n',
                              f'kernel: {kernel}\n',
                              f'metric: {metric}\n',
                              f'vector_based: {vector_based}\n',
                              str(err),
                              '\n'])
            continue
        # Compute Results
        acc_m, std_m = ch.get_result_percentage(accs[0])
        f1_m, f1_std_m = ch.get_result_percentage(f1[0])
        auc_m, auc_std_m = ch.get_result_percentage(auc[0])
        prec_m, prec_std_m = ch.get_result_percentage(prec[0])
        sens_m, sens_std_m = ch.get_result_percentage(sens[0])
        c_sens_m, c_sens_std_m = ch.get_result_percentage(c_sens[0])
        c_spec_m, c_spec_std_m = ch.get_result_percentage(c_spec[0])

        results.append((feat, acc_m, std_m, f1_m, f1_std_m, auc_m, auc_std_m, prec_m,
                       prec_std_m, sens_m, sens_std_m, c_sens_m, c_sens_std_m, c_spec_m, c_spec_std_m))

        print(f"Average accuracy: {acc_m}  %")
        print(f"Standard deviation:  {std_m} %")

    lines = []
    for result in results:
        lines.append(f"Feature number: {result[0]}\n")

        lines.append(f"Average accuracy: {result[1]}%\n")
        lines.append(f"Standard deviation:  {result[2]}%\n")

        lines.append(f"Average F1: {result[3]}%\n")
        lines.append(f"Standard F1 deviation:  {result[4]}%\n")

        lines.append(f"Average AUC: {result[5]}%\n")
        lines.append(f"Standard AUC deviation:  {result[6]}%\n")

        lines.append(f"Average Precision: {result[7]}%\n")
        lines.append(f"Standard Precision deviation:  {result[8]}%\n")

        lines.append(f"Average Sensibility: {result[9]}%\n")
        lines.append(f"Standard Sensibility deviation:  {result[10]}%\n")

        lines.append("CONFUSION MATRIX\n")
        lines.append(f"Average Sensibility: {result[11]}%\n")
        lines.append(f"Standard Sensibility deviation:  {result[12]}%\n")

        lines.append(f"Average Spec: {result[13]}%\n")
        lines.append(f"Standard Spec deviation:  {result[14]}%\n")

        lines.append("\n")

    with open(f'{results_dir}/{str(str(n_thres).replace(".",""))}_', 'w+') as f:
        f.writelines(lines)

    with open(f'{results_dir}/{n_thres}_mean_counts_{class1_name}', 'w+') as f1:
        f1.writelines(class1_lines)

    with open(f'{results_dir}/{n_thres}_mean_counts_{class2_name}', 'w+') as f2:
        f2.writelines(class2_lines)

    return results


def load_graphs_from_subjects(subjects, nodeattr, metric):
    graphs_subjects = []
    for subject in subjects:
        print(f"loading graphs for {subject[1]}")
        graphs = np.load(f'{subject[1]}/graphs_{nodeattr}_{metric}.npy',
                         allow_pickle=True)
        graphs_subjects.append(graphs)
    return graphs_subjects


if __name__ == "__main__":
    print("inicio")
    node_choices = ['histogram', 'volume_based']

    # Parse Arguments
    parser = ap.ArgumentParser(
        description='Classification, see help')
    parser.add_argument('--class1', action='store', type=str, nargs='+',
                        required=True, help='class1\'s Subjects directory')
    parser.add_argument('--class2', action='store', type=str, nargs='+',
                        required=True, help='class2\'s Subjects directory')
    parser.add_argument('--results', action='store', type=str, nargs='+',
                        required=True, help='Results base directory')
    parser.add_argument('--dist', action='store', type=str, nargs=1,
                        required=True, help='Distance Metrics: wasserstein, kl, hellinger',
                        choices=list(ch.DISTANCES.keys()))
    parser.add_argument('--nodeattr', action='store', type=str, nargs=1,
                        required=True, help='Graph Node Attributes',
                        choices=node_choices)
    parser.add_argument('--kernel', action='store', type=str, nargs=1,
                        required=True, help='Graph Kernel',
                        choices=['P2K', 'HGK_WL', 'HGK_SP', 'HGK_WSP'])
    parser.add_argument('--shortened_names', action='store', type=str, nargs=2,
                        required=True, help='shortened class names IN ORDER',
                        choices=['AD', 'MCI', 'CN'])

    args = parser.parse_args()

    # Get Subjects
    class1_subjects = ut.get_subjects_from_args(args.class1)
    class2_subjects = ut.get_subjects_from_args(args.class2)

    # Get metric
    metric = args.dist[0]

    # Get and create result directory
    result_dir = Path(args.results[0])
    print(f"Result directory: {result_dir}")
    newdir = f"{datetime.now().strftime('%d_%m_%Y_%H_%M')}_{metric}"

    # Create new results directory
    final_dir = (result_dir / newdir)
    final_dir.mkdir(parents=True)

    # Get Kernel Choice
    kernel = args.kernel[0]

    # Get node attributes type
    node_type = args.nodeattr[0]

    class1_graphs = load_graphs_from_subjects(
        class1_subjects, node_type, metric)
    class2_graphs = load_graphs_from_subjects(
        class2_subjects, node_type, metric)

    # Get shortned names
    class1_name = args.shortened_names[0]
    class2_name = args.shortened_names[1]

    # Classification using histogram as node attributes
    if node_type == node_choices[0]:

        # Execute
        for n_threshold in [0.5, 1, 2, 3, 4, 5]:
            print(n_threshold)
            classif(
                class1_name,
                class2_name,
                class1_graphs.copy(),
                class2_graphs.copy(),
                n_threshold,
                final_dir, metric, False, kernel)

    # Classification using vector-based with volume as node attributes
    if node_type == node_choices[1]:
        class1_vector_matrices = ch.load_only_norm_vector_based_matrix_subjects(
            class1_subjects)
        class2_vector_matrices = ch.load_only_norm_vector_based_matrix_subjects(
            class2_subjects)

        # Execute
        for n_threshold in [0.5, 1, 2, 3, 4, 5]:
            print(n_threshold)
            classif(
                class1_name,
                class2_name,
                class1_graphs.copy(),
                class2_graphs.copy(),
                n_threshold,
                final_dir, metric, True, kernel)
