import os
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from utils import model_forward

device = torch.device("cuda")

dataset_name = "AIDS"
# dataset_name = "NCI1"
# dataset_name = "PROTEINS"
# dataset_name = "ENZYMES"
# dataset_name = "MCF-7"
# dataset_name = "Mutagenicity"

PLOTS_DIR = "defense_plots"
if not osp.exists(PLOTS_DIR):
    os.mkdir(PLOTS_DIR)
PLOTS_DATASET_DIR = osp.join(PLOTS_DIR, dataset_name)
if not osp.exists(PLOTS_DATASET_DIR):
    os.mkdir(PLOTS_DATASET_DIR)

dataset = TUDataset(root="datasets", name=dataset_name)
num_node_labels = dataset.num_node_labels
num_classes = dataset.num_classes
print(f"dataset name: {dataset_name}")
print(f"num_node_labels: {num_node_labels}")
print(f"num_classes: {num_classes}")

poisoning_data = torch.load(f"SBAG/poisoning_data/{dataset_name}/poisoning_data.pt")

num_node_features = poisoning_data[0].x.shape[1]
num_node_attributes = num_node_features - num_node_labels
print(f"num_node_features: {num_node_features}")
print(f"num_node_attributes: {num_node_attributes}")

model_file = f"SBAG/models/{dataset_name}_backdoored_model.pt"
backdoored_model = torch.load(model_file)

benign_data = torch.load(f"SBAG/models/{dataset_name}_train_data.pt")  # 干净的训练集
benign_model = torch.load(f"SBAG/models/{dataset_name}_scoring_model.pt")  # 良性模型


def explain(data, model, target=0):
    input_mask = torch.ones(
        data.edge_index.shape[1]).requires_grad_(True).to(device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, target=target,
                        additional_forward_args=(model, data, device),
                        internal_batch_size=data.edge_index.shape[1])

    mask = np.abs(mask.cpu().detach().numpy())
    # min-max scaler
    if mask.max() > 0:  # avoid division by zero
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def get_node_labels(graph, num_node_attributes):
    label_tuple = ()
    for line in graph.x:
        arr = line.tolist()
        idx = arr[num_node_attributes:].index(1)
        label_tuple += (idx,)
    return label_tuple


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def min_max_scaler(x):
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val > min_val:
        return (x - min_val) / (max_val - min_val)
    return x


def draw_scores(scores, PLOTS_DATASET_DIR, dataset_name, target, type):
    plt.figure(figsize=(10, 5))

    x = range(len(scores))
    plt.bar(x, scores)

    # 显示每个柱子的具体数值
    # for a, b in zip(x, scores):
    #     plt.text(a, b, f"{b:.3f}", ha="center", va="bottom")

    if type == "backdoor":
        s = f"backdoor_{target}"
        plt.title(f"{dataset_name}_{s}")
    else:
        s = f"benign_{target}"
        plt.title(f"{dataset_name}_{s}")
    plt.xlabel("node label")
    # 设置x轴刻度和标签
    plt.xticks(range(len(x)), x)
    plt.ylabel("score")
    plt.ylim(0, 1.2)

    plot_filename = osp.join(PLOTS_DATASET_DIR, f"{s}")
    plt.savefig(f"{plot_filename}.png")


def explain_model(data, model, type):
    data_by_target = [[] for _ in range(0, num_classes)]

    for graph in data:
        t = graph.y.item()
        data_by_target[t].append(graph)

    # 对每个类别进行操作
    for target in range(0, num_classes):
        # 在每张图中，将每个结点的分数除以结点度数，进行归一化
        # 在所有图中，将相同标签结点的分数相加后对该标签结点出现次数取平均值
        label_total_scores = np.zeros(num_node_labels, dtype=np.float64)
        label_total_times = np.zeros(num_node_labels, dtype=int)

        for graph in tqdm(data_by_target[target], desc=f"current target--{target}"):
            # convert node indexes to labels
            # node_index2label = get_node_labels(graph, num_node_attributes)
            x = graph.x.detach().cpu().numpy()
            num_nodes = x.shape[0]
            node_index2label = x[:, num_node_attributes:]

            edge_mask = explain(graph, model, target=target)
            edge_mask_dict = aggregate_edge_directions(edge_mask, graph)

            node_scores = np.zeros(num_nodes, dtype=np.float64)
            degrees = np.zeros(num_nodes, dtype=int)

            # add the scores of each edge to both endpoints separately
            for (u, v), val in edge_mask_dict.items():
                node_scores[u] += val
                node_scores[v] += val
                degrees[u] += 1
                degrees[v] += 1

            # print(node_scores)

            # divide the score of each node by the degree of the node
            node_scores = np.divide(node_scores, degrees, where=(degrees > 0))
            node_scores = min_max_scaler(node_scores)

            # add the scores of nodes with the same label in each graph
            label_scores = node_scores @ node_index2label

            label_total_scores += label_scores  # add the scores of nodes with the same label in all graphs
            label_total_times += node_index2label.sum(axis=0).astype(int)

        # divide by the number of graphs containing each node label
        label_total_scores = np.divide(label_total_scores, label_total_times, where=(label_total_times > 0))
        draw_scores(label_total_scores, PLOTS_DATASET_DIR, dataset_name, target, type)


explain_model(poisoning_data, backdoored_model, type="backdoor")
explain_model(benign_data, benign_model, type="benign")
