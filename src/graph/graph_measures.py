import json
import os
import timeit
from collections import defaultdict

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
from scipy.stats import entropy, skew
from sklearn.preprocessing import MinMaxScaler

from src.utils import NumpyEncoder, time_execution


@time_execution
def number_of_nodes(G, verbose):
    return G.number_of_nodes()


@time_execution
def number_of_edges(G, verbose):
    return G.number_of_edges()


@time_execution
def is_strongly_connected(G, verbose):
    return nx.is_strongly_connected(G)


@time_execution
def transitivity(G, verbose):
    return nx.transitivity(G)


@time_execution
def density(G, verbose):
    return nx.density(G)


@time_execution
def mixing_parameter(G, communities, verbose):

    # Step 1: Map each node to its community
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Step 2: Count inter-cluster edges efficiently
    inter_cluster_edges = 0
    for u, v in G.edges():
        # Directly check if u and v belong to different communities
        if node_to_community[u] != node_to_community[v]:
            inter_cluster_edges += 1

    mixing_parameter = inter_cluster_edges / G.number_of_edges()

    return mixing_parameter


@time_execution
def modularity(G, communities, verbose):

    start_time = timeit.default_timer()
    modularity = nx.community.modularity(G, communities)
    if verbose:
        print(
            f"==>> modularity: {modularity}, in {str(timeit.default_timer() - start_time)} seconds")

    return modularity


def get_degrees(G, verbose):
    start_time = timeit.default_timer()
    degrees = [degree for _, degree in G.degree()]
    if verbose:
        print(
            f"==>> calculated degrees, in {str(timeit.default_timer() - start_time)} seconds")
    return degrees


@time_execution
def find_communities(G, verbose):

    start_time = timeit.default_timer()
    G1 = ig.Graph.from_networkx(G)

    part = G1.community_infomap()
    # part = G1.community_multilevel()
    # part = G1.community_spinglass()
    # part = G1.community_edge_betweenness()

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['_nx_name']
                           for node_index in com])

    # communities = nx.community.louvain_communities(G)
    if verbose:
        print(
            f"==>> number_of_communities: {len(communities)}, in {str(timeit.default_timer() - start_time)} seconds")

    return G1, part, communities


def calculate_graph_measures(G, file_path=None, verbose=False, communities=None):

    properties = {}

    properties["number_of_nodes"] = number_of_nodes(G, verbose)
    properties["number_of_edges"] = number_of_edges(G, verbose)

    degrees = get_degrees(G, verbose)

    properties["max_degree"] = max(degrees)
    properties["avg_degree"] = sum(degrees) / len(degrees)

    if type(G) is nx.DiGraph or type(G) is nx.Graph:
        properties["transitivity"] = transitivity(G, verbose)

    properties["density"] = density(G, verbose)

    if communities:
        properties["number_of_communities"] = len(communities)
        properties["mixing_parameter"] = mixing_parameter(
            G, communities, verbose)
        properties["modularity"] = modularity(G, communities, verbose)

    if file_path:
        outfile = open(file_path, 'w')
        outfile.writelines(json.dumps(properties))
        outfile.close()

    return properties


@time_execution
def class_pairs(df, source_ip, destination_ip, class_column, results_dict, verbose, folder_path=None):
    # Initialize lists to store results
    same_class_pairs = {}
    mixed_class_pairs = []

    # Group by source and destination IP addresses
    for (source, destination), group in df.groupby([source_ip, destination_ip]):
        unique_classes = group[class_column].unique()
        if len(unique_classes) == 1:
            # All records have the same class
            class_label = str(unique_classes[0])
            if class_label not in same_class_pairs:
                same_class_pairs[class_label] = []
            same_class_pairs[class_label].append({
                'node_pair': (source, destination),
                'num_instances': len(group)
            })
        else:
            # Mixed class scenario
            class_counts = group[class_column].value_counts().to_dict()
            total_instances = len(group)
            class_percentages = {
                str(cls): count / total_instances for cls, count in class_counts.items()}
            mixed_class_pairs.append({
                'node_pair': (source, destination),
                'class_counts': class_counts,
                'class_percentages': class_percentages
            })

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "same_class_pairs.json"), "w") as f:
            f.writelines(json.dumps(same_class_pairs, cls=NumpyEncoder))

        with open(os.path.join(folder_path, "mixed_class_pairs.json"), "w") as f:
            f.writelines(json.dumps(mixed_class_pairs, cls=NumpyEncoder))

    # Total counts
    total_same_class_pairs = sum(len(pairs)
                                 for pairs in same_class_pairs.values())
    total_mixed_class_pairs = len(mixed_class_pairs)

    if verbose:
        print("\nTotal number of same class pairs:", total_same_class_pairs)
        print("Total number of mixed class pairs:", total_mixed_class_pairs)

    results_dict["total_same_class_pairs"] = total_same_class_pairs
    results_dict["total_mixed_class_pairs"] = total_mixed_class_pairs

    # Interpretation:
    # - `same_class_pairs` contains node pairs with consistent classes across all records, including the number of instances.
    # - `mixed_class_pairs` contains node pairs with mixed classes, the counts and percentages for each class.
    # - Total counts provide an overview of the dataset's class consistency.


@time_execution
def attackers_victims(graph, results_dict, label_col, verbose):
    attackers = set()
    victims = set()

    for u, v, data in graph.edges(data=True):
        if data[label_col] == 1:
            attackers.add(u)
            victims.add(v)

    # Step 2: Count unique attackers and victims
    num_attackers = len(attackers)
    num_victims = len(victims)

    # Step 3: Calculate proportions
    total_nodes = graph.number_of_nodes()
    attacker_proportion = num_attackers / total_nodes if total_nodes > 0 else 0
    victim_proportion = num_victims / total_nodes if total_nodes > 0 else 0

    if verbose:
        print("Number of Attackers:", num_attackers)
        print("Number of Victims:", num_victims)
        print("Proportion of Attackers:", attacker_proportion)
        print("Proportion of Victims:", victim_proportion)

    results_dict["total_nodes"] = total_nodes
    results_dict["Number of Attackers"] = num_attackers
    results_dict["Number of Victims"] = num_victims
    results_dict["Proportion of Attackers"] = attacker_proportion
    results_dict["Proportion of Victims"] = victim_proportion
    results_dict["intersection between attacks and victims"] = len(
        attackers.intersection(victims))

    # Interpretation:
    # - Attackers: Source nodes of edges labeled as "Attack".
    # - Victims: Target nodes of edges labeled as "Attack".
    # - These metrics provide insight into the roles of nodes in attack scenarios.


@time_execution
def cal_clustering_coefficients(graph, results_dict, verbose):

    # Clustering Coefficient Distribution Metric
    # Convert MultiDiGraph to Graph for clustering
    clustering_coefficients = nx.clustering(nx.Graph(graph))
    clustering_values = list(clustering_coefficients.values())
    mean_clustering = np.mean(clustering_values)
    std_clustering = np.std(clustering_values)

    if verbose:
        print("Mean Clustering Coefficient:", mean_clustering)
        print("Standard Deviation of Clustering Coefficients:", std_clustering)

    results_dict["Mean Clustering Coefficient"] = mean_clustering
    results_dict["Standard Deviation of Clustering Coefficients"] = std_clustering


@time_execution
def cal_degree_assortativity(graph, results_dict, verbose):
    # Graph Assortativity Metric
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(graph)
        results_dict["Graph Degree Assortativity Coefficient"] = degree_assortativity
        if verbose:
            print("Degree Assortativity Coefficient:", degree_assortativity)
    except nx.NetworkXError as e:
        results_dict["Graph Degree Assortativity Coefficient"] = "not applicable"
        if verbose:
            print("Error calculating assortativity:", e)


@time_execution
def cal_diameter(graph, results_dict, verbose):
    # Graph Diameter Metric
    try:
        if nx.is_strongly_connected(graph):
            diameter = nx.diameter(graph)
            results_dict["diameter"] = diameter
            if verbose:
                print("Graph Diameter multidigraph:", diameter)
        else:
            results_dict["diameter"] = "not applicable"
            if verbose:
                print("Graph is not strongly connected, diameter is undefined.")

    except nx.NetworkXError as e:
        print("Error calculating diameter:", e)


@time_execution
def path_length_distribution(graph, results_dict, verbose):
    # Path Length Distribution Metric
    try:
        path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        all_lengths = [length for source in path_lengths.values()
                       for length in source.values()]
        mean_path_length = np.mean(all_lengths)
        std_path_length = np.std(all_lengths)
        if verbose:
            print("Mean Path Length MultiDiGraph:", mean_path_length)
            print("Standard Deviation of Path Lengths MultiDiGraph:", std_path_length)
        results_dict["Mean Path Length"] = mean_path_length
        results_dict["Standard Deviation of Path Lengths"] = std_path_length

    except nx.NetworkXError as e:
        results_dict["Mean Path Length"] = "not applicable"
        results_dict["Standard Deviation of Path Lengths"] = "not applicable"
        if verbose:
            print("Error calculating path length distribution:", e)

    # Interpretation:
    # - Diameter: Longest shortest path in the graph (undefined for disconnected graphs).
    # - Assortativity: Correlation of node degrees (positive, negative, or neutral).
    # - Clustering Coefficients: Measure of local connectivity (distribution provides network structure insights).
    # - Path Lengths: Reachability analysis using shortest paths.


def check_if_scale_free(centrality_sequence, title="Centrality Distribution", verbose=False):
    fit = powerlaw.Fit(centrality_sequence)
    if verbose:
        print(f"==>> fit.alpha: {fit.alpha}")

    bins = np.logspace(np.log10(min(centrality_sequence)),
                       np.log10(max(centrality_sequence)), 20)
    hist, bins = np.histogram(centrality_sequence, bins=bins, density=True)

    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the observed degree distribution (log-log scale)
    plt.scatter(bin_centers, hist, color='blue',
                alpha=0.7, label="Observed Data")

    # Overlay the fitted power-law line
    fit.power_law.plot_pdf(color='red', linestyle="--",
                           label=f"Power-Law Fit (γ={fit.alpha:.2f})")

    # Log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Labels, title, and legend
    plt.xlabel("Centrality Value (k)")
    plt.ylabel("P(k) (Probability Density)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.show()

    scale_free = fit.alpha > 2 and fit.alpha < 3
    if verbose:
        if scale_free:
            print("This graph is likely scale-free.")
        else:
            print("This graph is NOT scale-free.")

    return fit.alpha, scale_free


@time_execution
def centrality_analysis(centrality_sequence, dataset_name, centrality_name, verbose):
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(
        centrality_sequence.reshape(-1, 1)).flatten()

    centrality_skewness = skew(normalized_values)

    vc = np.unique(normalized_values, return_counts=True)[1]
    centrality_entropy = entropy(pk=vc)

    alpha, scale_free = check_if_scale_free(
        centrality_sequence, title=f"{centrality_name} Distribution of :{dataset_name}", verbose=verbose)

    return centrality_skewness, centrality_entropy, alpha, scale_free


@time_execution
def compute_edge_class_entropy(graph, label_col, verbose):
    """
    Computes the average entropy of edge class distributions across all nodes.

    Args:
        graph: A NetworkX MultiGraph or MultiDiGraph.
        label_col: The key in edge attributes that stores the class label.

    Returns:
        avg_entropy: The average entropy of all nodes.
    """
    node_entropy = {}

    for node in graph.nodes():
        edge_class_counts = defaultdict(int)
        total_edges = 0

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)

            # If graph is a MultiGraph, edge_data is a dictionary with multiple edges
            if isinstance(edge_data, dict):
                for edge_id, data in edge_data.items():
                    edge_class = data.get(label_col, None)
                    if edge_class is not None:
                        edge_class_counts[edge_class] += 1
                        total_edges += 1
            else:  # If it's a simple Graph/DiGraph, just process normally
                edge_class = edge_data.get(label_col, None)
                if edge_class is not None:
                    edge_class_counts[edge_class] += 1
                    total_edges += 1

        # Compute entropy for the node
        if total_edges > 0:
            probs = np.array(list(edge_class_counts.values())) / total_edges
            entropy = -np.sum(probs * np.log2(probs))  # Shannon entropy
        else:
            entropy = 0  # If node has no edges, entropy is 0

        node_entropy[node] = entropy

    # Compute average entropy across all nodes
    avg_entropy = np.mean(list(node_entropy.values())) if node_entropy else 0
    return avg_entropy

    # Higher entropy → The node interacts with diverse attack/traffic types.
    # Lower entropy → The node interacts with one dominant edge class.
    # Zero entropy → Either an isolated node or all edges belong to the same class.


@time_execution
def compute_avg_edge_class_diversity(graph, label_col, verbose):
    """
    Computes the average number of unique edge classes per node.

    Args:
        graph: A NetworkX MultiGraph or MultiDiGraph.
        label_col: The key in edge attributes that stores the class label.

    Returns:
        avg_diversity: Average unique edge class count per node.
    """
    node_diversity = []

    for node in graph.nodes():
        unique_classes = set()
        total_edges = 0

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)

            # If MultiGraph, iterate through multiple edges
            if isinstance(edge_data, dict):
                for edge_id, data in edge_data.items():
                    edge_class = data.get(label_col, None)
                    if edge_class is not None:
                        unique_classes.add(edge_class)
                        total_edges += 1
            else:  # If it's a simple Graph, process normally
                edge_class = edge_data.get(label_col, None)
                if edge_class is not None:
                    unique_classes.add(edge_class)
                    total_edges += 1

        # Compute diversity for the node
        if total_edges > 0:
            diversity = len(unique_classes) / total_edges
            node_diversity.append(diversity)

    return np.mean(node_diversity) if node_diversity else 0

    # Higher diversity → The node interacts with multiple types of traffic.
    # Lower diversity → The node interacts mostly with one edge class.
    # Zero diversity → The node has no edges or only interacts with one type.
