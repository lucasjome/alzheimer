# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import graph_tool as gt
import numpy as np
import os.path as path


def read_txt(ds_name):
    pre = ""

    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]
    f.closed

    # Nodes
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_db = []
    vertex_list = []
    for i in node_indices:
        g = gt.Graph(directed=False)
        vertex_list_g = []
        for _ in range(i[1] - i[0] + 1):
            vertex_list_g.append(g.add_vertex())

        graph_db.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]
    f.closed

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph
        if not g.edge(e[0] - off, e[1] - off):
            edge_list.append(g.add_edge(e[0] - off, e[1] - off))

    # Node labels
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r") as f:
            node_labels = [int(i) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = node_labels[i]
                i += 1


    # Node Attributes
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
            node_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            g.vp.hist = g.new_vertex_property("vector<float>")
            for v in g.vertices():
                g.vp.hist[v] = node_attributes[i]
                i += 1


    # Edge Labels
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt", "r") as f:
            edge_labels = [int(i) for i in list(f)]
        f.closed

        l_el = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_el.append(g.new_edge_property("int"))

        for i, l in enumerate(edge_labels):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_el[g_id][edge_list[i]] = l
            g.ep.el = l_el[g_id]

    # Edge Attributes
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt", "r") as f:
            edge_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        l_ea = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_ea.append(g.new_edge_property("vector<float>"))

        for i, l in enumerate(edge_attributes):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_ea[g_id][edge_list[i]] = l
            g.ep.ea = l_ea[g_id]

    # Classes
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return graph_db, classes


def write_lib_svm(gram_matrix, classes, name):
    with open(name, "w") as f:
        k = 1
        for c, row in zip(classes, gram_matrix):
            s = ""
            s = str(c) + " " + "0:" + str(k) + " "
            for i, r in enumerate(row):
                s += str(i + 1) + ":" + str(r) + " "
            s += "\n"
            f.write(s)
            k += 1
    f.closed
