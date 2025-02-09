from src.graph.centralities import add_centralities
import math
from src.graph.graph_measures import calculate_graph_measures
import pickle
import pandas as pd
import networkx as nx
import time
import os


def create_weightless_flow_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, new_graph_path=None):

    try:
        # Record the start time
        start = time.time()

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Create the directed graph from the pandas dataframe
        G = nx.from_pandas_edgelist(df,
                                    source=src_ip_col,
                                    target=dst_ip_col,
                                    create_using=base)

        if line_graph:
            G = nx.line_graph(G)

        if new_graph_path:
            # Save the graph to a GEXF file
            nx.write_gexf(G, new_graph_path)

            print(
                f"Graph created and saved to {new_graph_path} in {time.time() - start:.2f} seconds.")

        else:
            print(f"Graph created in {time.time() - start:.2f} seconds.")

        return G
    except Exception as e:
        print(f"An error occurred: {e}")


def define_sessions(df, timestamp_col, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col=None, timeout=pd.Timedelta(minutes=5)):
    df = df.sort_values(by=timestamp_col)
    sessions = []
    current_session_id = 0
    last_seen = {}

    for index, row in df.iterrows():
        if protocol_col:
            tuples = (row[src_ip_col], row[dst_ip_col],
                      row[src_port_col], row[dst_port_col], row[protocol_col])
        else:
            tuples = (row[src_ip_col], row[dst_ip_col],
                      row[src_port_col], row[dst_port_col])
        if tuples in last_seen:
            if timeout and row[timestamp_col] - last_seen[tuples] > timeout:
                current_session_id += 1
        else:
            current_session_id += 1
        last_seen[tuples] = row[timestamp_col]
        sessions.append(current_session_id)

    df['session_id'] = sessions
    return df


def create_weightless_session_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, folder_path=None, edge_attr=None, file_type="gexf"):
    try:
        # Record the start time
        start_time = time.time()

        graphs = []

        if multi_graph or line_graph:
            base_graph_type = nx.MultiDiGraph
        else:
            base_graph_type = nx.DiGraph

        # Iterate over each session in the DataFrame
        for session_id, df_session in df.groupby('session_id'):
            # Create a graph from the session
            G = nx.from_pandas_edgelist(df_session,
                                        source=src_ip_col,
                                        target=dst_ip_col,
                                        edge_attr=edge_attr,
                                        create_using=base_graph_type())

            if line_graph:
                G_line_graph = nx.line_graph(G)
                G_line_graph.add_nodes_from(
                    (node, G.edges[node]) for node in G_line_graph)
                G = G_line_graph

            if folder_path:
                if file_type == "gexf":
                    filename = os.path.join(
                        folder_path, 'graphs', f'graph_{session_id}.gexf')
                    # Save the graph to a file
                    nx.write_gexf(G, filename)

                if file_type == "pkl":
                    filename = os.path.join(
                        folder_path, 'graphs', f'graph_{session_id}.pkl')

                    # Save the graph to a file
                    with open(filename, "wb") as f:
                        pickle.dump(G, f)

                calculate_graph_measures(
                    G, os.path.join(folder_path, 'graph_measures', f'graph_{session_id}_measures.json'))

            # Append the graph to the list
            graphs.append(G)

        print(f"Graphs created in {time.time() - start_time:.2f} seconds.")

        return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


def create_weightless_window_graph(df, dataset, window_size=20000, cn_measures=None, network_features=None, multi_graph=False, line_graph=False, folder_path=None, test_percentage=None, edge_attr=None, file_type="gexf"):

    try:
        # Record the start time
        start_time = time.time()

        # graphs = []

        # Total number of records
        total_records = len(df)

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Iterate over the DataFrame in chunks
        i = 0
        number_of_groups = math.ceil(total_records / window_size)
        print(f"==>> number_of_groups: {number_of_groups}")

        if test_percentage:
            # folder_path += "_train_test"

            number_of_test_groups = math.ceil(
                number_of_groups * test_percentage / 100)

            number_of_train_groups = number_of_groups - number_of_test_groups
            print(f"==>> number_of_train_groups: {number_of_train_groups}")
            print(f"==>> number_of_test_groups: {number_of_test_groups}")

        for start in range(0, total_records, window_size):

            df_chunk = df.iloc[start:start + window_size]

            # Create a graph from the chunk
            G = nx.from_pandas_edgelist(df_chunk,
                                        source=dataset.src_ip_col,
                                        target=dataset.dst_ip_col,
                                        edge_attr=edge_attr,
                                        create_using=base)
            if cn_measures and network_features:
                add_centralities(df=None, new_path=None, graph_path=None, dataset=dataset, G=G,
                                 cn_measures=cn_measures, network_features=network_features, create_using=nx.MultiDiGraph())
            if line_graph:
                G_line_graph = nx.line_graph(G)
                G_line_graph.add_nodes_from(
                    (node, G.edges[node]) for node in G_line_graph)
                G = G_line_graph
            if folder_path:
                # Ensure the folder path exists
                os.makedirs(folder_path, exist_ok=True)

                if file_type == "gexf":
                    filename = os.path.join(folder_path, f'graph_{i}.gexf')
                    # Save the graph to a file
                    nx.write_gexf(G, filename)

                if file_type == "pkl":
                    if test_percentage:
                        if i < number_of_train_groups:

                            folder_path_train = os.path.join(
                                folder_path, 'training')
                            os.makedirs(folder_path_train, exist_ok=True)
                            filename = os.path.join(
                                folder_path_train, f'graph_{i}.pkl')
                        else:

                            folder_path_test = os.path.join(
                                folder_path, 'testing')
                            os.makedirs(folder_path_test, exist_ok=True)
                            filename = os.path.join(
                                folder_path_test, f'graph_{i}.pkl')
                    else:
                        folder_path_all = os.path.join(
                            folder_path, 'graphs')
                        os.makedirs(folder_path_all, exist_ok=True)
                        filename = os.path.join(
                            folder_path_all, f'graph_{i}.pkl')

                    # Save the graph to a file
                    with open(filename, "wb") as f:
                        pickle.dump(G, f)

                graph_measures = calculate_graph_measures(
                    G, os.path.join(folder_path, f'graph_{i}_measures.json'))
                print(f"==>> graph_measures of graph_{i}: {graph_measures}")

            # Append the graph to the list
            # graphs.append(G)
            i += 1

        print(f"Graph created in {time.time() - start_time:.2f} seconds.")

        # return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-07-01 00:00:01', '2024-07-01 00:00:03', '2024-07-01 00:05:05',
                                     '2024-07-01 00:06:01', '2024-07-01 00:10:01']),
        'src_ip': ['192.168.1.1', '192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.1'],
        'dst_ip': ['192.168.2.1', '192.168.2.1', '192.168.2.1', '192.168.2.2', '192.168.2.1'],
        'src_port': [12345, 12345, 12345, 12346, 12345],
        'dst_port': [80, 80, 80, 80, 80],
        'protocol': ['TCP', 'TCP', 'TCP', 'TCP', 'TCP']
    })

    df2 = define_sessions(df, "timestamp", "src_ip", "dst_port",
                          'src_port', 'dst_port', 'protocol', pd.Timedelta(minutes=5))
    print(df2)

    df3 = create_weightless_window_graph(df, "src_ip", "dst_port", 2)
    print(df3)
