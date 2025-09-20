import json

from torch_geometric.datasets import TUDataset

dataset_names = ["AIDS", "NCI1", "MCF-7", "Mutagenicity"]

for dataset_name in dataset_names:
    dataset = TUDataset(root="datasets", name=dataset_name, use_node_attr=True)

    num_node_labels = dataset.num_node_labels
    num_classes = dataset.num_classes

    d = {
        "dataset_name": dataset_name,
        "num_node_labels": num_node_labels,
        "ground-truth trigger node": 0,
        "ground-truth target class": 0
    }

    with open(f"configs/{dataset_name}_config.json", "w") as wf:
        json.dump(d, wf, indent=4, ensure_ascii=True)
