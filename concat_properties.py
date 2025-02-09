import os
import json


from src.dataset.dataset_info import datasets
from src.utils import NumpyEncoder

my_datasets = [
    # "cic_ton_iot_5_percent",
    "cic_ids_2017_5_percent",
    # "cic_ton_iot",
    # "cic_ids_2017",
    # "cic_bot_iot",
    # "cic_ton_iot_modified",
    # "ccd_inid_modified",
    # "nf_uq_nids_modified",
    # "edge_iiot",
    # "nf_cse_cic_ids2018",
    # "nf_uq_nids",
    # "x_iiot",
]

dataset_properties_file_name = "dataset_properties.json"
graph_properties_file_name = "graph_properties.json"
new_file_name = "datasets_properties.json"

all_props = {}
for ds in my_datasets:
    dataset_p = {}
    dataset = datasets[ds]
    with open(os.path.join("datasets", dataset.name, dataset_properties_file_name), "r") as f:
        prop = json.load(f)
        dataset_p["dataset_properties"] = prop

    with open(os.path.join("datasets", dataset.name, graph_properties_file_name), "r") as f:
        prop = json.load(f)
        dataset_p["graph_properties"] = prop

    all_props[ds] = dataset_p

with open(os.path.join("datasets", new_file_name), "w") as f:
    json.dump(all_props, f, cls=NumpyEncoder)
