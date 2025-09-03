import pandas as pd
import numpy as np
import os
import pickle
import json
import socket
import struct
import random
import networkx as nx

from src.config import Config
from src.dataset.dataset_info import DatasetInfo
from src.graph.graph_construction import create_weightless_window_graph
from src.dataset.dataset_utils import clean_dataset, one_dataset_class_num_col
from src.dataset.features_analysis import feature_analysis_pipeline
from src.utils import NumpyEncoder
from src.graph.centralities import add_centralities, add_centralities_as_node_features
from src.graph.graph_measures import calculate_graph_measures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_graph_files(original_path, dataset_info: DatasetInfo, config: Config, folder_path):
    df = pd.read_parquet(original_path)

    timestamp_format = "mixed"
    df = clean_dataset(df, flow_id_col=dataset_info.flow_id_col,
                    timestamp_col=dataset_info.timestamp_col)

    df[dataset_info.src_ip_col] = df[dataset_info.src_ip_col].apply(str)
    if dataset_info.src_port_col:
        df[dataset_info.src_port_col] = df[dataset_info.src_port_col].apply(str)

    df[dataset_info.dst_ip_col] = df[dataset_info.dst_ip_col].apply(str)
    if dataset_info.dst_port_col:
        df[dataset_info.dst_port_col] = df[dataset_info.dst_port_col].apply(str)

    _, var_dropped, corr_dropped = feature_analysis_pipeline(
        df=df, drop_columns=dataset_info.drop_columns, label_col=dataset_info.label_col)
    var_dropped, corr_dropped

    var_dropped = set(var_dropped)
    weak_columns = var_dropped.union(set(corr_dropped))

    if config.sort_timestamp and dataset_info.timestamp_col:
        df[dataset_info.timestamp_col] = pd.to_datetime(
            df[dataset_info.timestamp_col].str.strip(), format=timestamp_format)
        df.sort_values(dataset_info.timestamp_col, inplace=True)

    df, labels_names, classes = one_dataset_class_num_col(
        df, dataset_info.class_num_col, dataset_info.class_col)

    os.makedirs(folder_path, exist_ok=True)
    with open(folder_path + '/labels_names.pkl', 'wb') as f:
        pickle.dump([labels_names, classes], f)

    attackers_ratio = (df[dataset_info.label_col] != 0).mean()
    # Degree distribution based on source/destination IP counts
    from collections import Counter
    degrees = Counter()
    for col in [dataset_info.src_ip_col, dataset_info.dst_ip_col]:
        counts = df[col].value_counts()
        for node, cnt in counts.items():
            degrees[node] += cnt
    total_deg = sum(degrees.values())
    probs = np.array([deg/total_deg for deg in degrees.values()])
    # Entropy of degree distribution
    deg_entropy = -(probs * np.log(probs + 1e-12)).sum()
    graph_properties = {
        'attackers_ratio': attackers_ratio,
        'deg_entropy': deg_entropy,
    }
    with open(folder_path + '/graph_properties.json', 'w') as f:
        json.dump(graph_properties, f, cls=NumpyEncoder)

    
    cols_to_norm = list(set(list(df.columns))  - set(list([dataset_info.label_col, dataset_info.class_num_col])) - set(dataset_info.drop_columns)  - set(dataset_info.weak_columns))

    if config.generated_ips:
        df[dataset_info.src_ip_col] = df[dataset_info.src_ip_col].apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))



    if config.sort_timestamp:
        df[dataset_info.timestamp_col] = pd.to_datetime(df[dataset_info.timestamp_col].str.strip(), format=dataset_info.timestamp_format)
        df.sort_values(dataset_info.timestamp_col, inplace=True)

    if config.use_port_in_address:
        df[dataset_info.src_port_col] = df[dataset_info.src_port_col].astype(float).astype(int).astype(str) # to remove the decimal point
        df[dataset_info.src_ip_col] = df[dataset_info.src_ip_col] + ':' + df[dataset_info.src_port_col]

        df[dataset_info.dst_port_col] = df[dataset_info.dst_port_col].astype(float).astype(int).astype(str) # to remove the decimal point
        df[dataset_info.dst_ip_col] = df[dataset_info.dst_ip_col] + ':' + df[dataset_info.dst_port_col]

    if config.multi_class:
        y = df[dataset_info.class_num_col]
    else:
        y = df[dataset_info.label_col]

    if config.sort_timestamp:
        X_tr, X_test, y_tr, y_test = train_test_split(
            df, y, test_size=config.test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=config.validation_size)
    else:
        X_tr, X_test, y_tr, y_test = train_test_split(
            df, y, test_size=config.test_size, random_state=13, stratify=y)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=config.validation_size, random_state=13, stratify=y_tr)

    # del df

    if config.graph_type == "line" and config.use_node_features:
        add_centralities(df = X_train, new_path=None, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures, network_features=config.network_features, create_using=nx.MultiDiGraph())
        add_centralities(df = X_val, new_path=None, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures, network_features=config.network_features, create_using=nx.MultiDiGraph())
        add_centralities(df = X_test, new_path=None, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures, network_features=config.network_features, create_using=nx.MultiDiGraph())
        cols_to_norm = list(set(cols_to_norm) | set(config.network_features))


    scaler = StandardScaler()

    X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
    X_train['h'] = X_train[ cols_to_norm ].values.tolist()

    cols_to_drop = list(set(list(X_train.columns)) - set(list([dataset_info.label_col, dataset_info.src_ip_col, dataset_info.dst_ip_col, dataset_info.class_num_col, 'h'])))
    X_train.drop(cols_to_drop, axis=1, inplace=True)

    X_val[cols_to_norm] = scaler.transform(X_val[cols_to_norm])
    X_val['h'] = X_val[ cols_to_norm ].values.tolist()
    X_val.drop(cols_to_drop, axis=1, inplace=True)

    X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
    X_test['h'] = X_test[ cols_to_norm ].values.tolist()
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    if config.graph_type == "window" or config.graph_type == "line":

        graphs_train = create_weightless_window_graph(
            df=X_train,
            dataset=dataset_info,
            window_size=config.window_size,
            line_graph=config.graph_type == "line",
            folder_path=os.path.join(folder_path, "training"),
            edge_attr= ['h', dataset_info.label_col, dataset_info.class_num_col],
            file_type="pkl")
        
        graphs_val = create_weightless_window_graph(
            df=X_val,
            dataset=dataset_info,
            window_size=config.window_size,
            line_graph=config.graph_type == "line",
            folder_path=os.path.join(folder_path, "validation"),
            edge_attr= ['h', dataset_info.label_col, dataset_info.class_num_col],
            file_type="pkl")
        
        graphs_test = create_weightless_window_graph(
            df=X_test,
            dataset=dataset_info,
            window_size=config.window_size,
            line_graph=config.graph_type == "line",
            folder_path=os.path.join(folder_path, "testing"),
            edge_attr= ['h', dataset_info.label_col, dataset_info.class_num_col],
            file_type="pkl")

    if config.graph_type == "flow":
        os.makedirs(folder_path, exist_ok=True)
        print(f"==>> X_train.shape: {X_train.shape}")
        print(f"==>> X_val.shape: {X_val.shape}")
        print(f"==>> X_test.shape: {X_test.shape}")

        graph_name = "training_graph"

        G_train = nx.from_pandas_edgelist(X_train, dataset_info.src_ip_col, dataset_info.dst_ip_col, ['h',dataset_info.label_col, dataset_info.class_num_col], create_using=nx.MultiDiGraph())
        
        if config.use_node_features:
            add_centralities_as_node_features(df=None, G=G_train, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures)
            
            for node in G_train.nodes():
                centralities = []
                for centrality in config.cn_measures:
                    centralities.append(G_train.nodes[node].get(centrality, 0)) # Default to 0 if missing
                    
                    # Combine features into a single vector
                n_feats = np.array(centralities, dtype=np.float32)
                
                # Add the new feature to the node
                G_train.nodes[node]["n_feats"] = n_feats
                
        # get netowrk properties
        graph_measures = calculate_graph_measures(G_train, f"{folder_path}/{graph_name}_measures.json", verbose=True)
        print(f"==>> graph_measures: {graph_measures}")

        # graph_measures = calculate_graph_measures(nx.DiGraph(G), "datasets/" + name + "/training_graph_simple_measures.json", verbose=True)
        # print(f"==>> graph_measures: {graph_measures}")

        with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
            pickle.dump(G_train, f)

        graph_name = "validation_graph"

        G_val = nx.from_pandas_edgelist(X_val, dataset_info.src_ip_col, dataset_info.dst_ip_col, ['h',dataset_info.label_col, dataset_info.class_num_col], create_using=nx.MultiDiGraph())
        
        if config.use_node_features:
            add_centralities_as_node_features(df=None, G=G_val, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures)
            
            for node in G_val.nodes():
                centralities = []
                for centrality in config.cn_measures:
                    centralities.append(G_val.nodes[node].get(centrality, 0)) # Default to 0 if missing
                    
                    # Combine features into a single vector
                n_feats = np.array(centralities, dtype=np.float32)
                
                # Add the new feature to the node
                G_val.nodes[node]["n_feats"] = n_feats
                
        # get netowrk properties
        graph_measures = calculate_graph_measures(G_val, f"{folder_path}/{graph_name}_measures.json", verbose=True)
        print(f"==>> graph_measures: {graph_measures}")

        # graph_measures = calculate_graph_measures(nx.DiGraph(G), "datasets/" + name + "/training_graph_simple_measures.json", verbose=True)
        # print(f"==>> graph_measures: {graph_measures}")

        with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
            pickle.dump(G_val, f)

        graph_name = "testing_graph"
        
        G_test = nx.from_pandas_edgelist(X_test, dataset_info.src_ip_col, dataset_info.dst_ip_col, ['h', dataset_info.label_col, dataset_info.class_num_col], create_using=nx.MultiDiGraph())
        
        if config.use_node_features:
            add_centralities_as_node_features(df=None, G=G_test, graph_path=None, dataset=dataset_info, cn_measures=config.cn_measures)
            
            for node in G_test.nodes():
                centralities = []
                for centrality in config.cn_measures:
                    centralities.append(G_test.nodes[node].get(centrality, 0)) # Default to 0 if missing
                    
                    # Combine features into a single vector
                n_feats = np.array(centralities, dtype=np.float32)
                
                # Add the new feature to the node
                G_test.nodes[node]["n_feats"] = n_feats
                
        graph_measures = calculate_graph_measures(G_test, f"{folder_path}/{graph_name}_measures.json", verbose=True)
        print(f"==>> graph_measures: {graph_measures}")
        
        # graph_measures = calculate_graph_measures(nx.DiGraph(G_test), "datasets/" + name + "/testing_graph_simple_measures.json", verbose=True)
        # print(f"==>> graph_measures: {graph_measures}")
        
        with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
            pickle.dump(G_test, f)


    # if config.graph_type == "window" or config.graph_type == "line":
    #     return graph_properties, (graphs_train, graphs_val, graphs_test)
    # else:
    #     return graph_properties, (G_train, G_val, G_test)