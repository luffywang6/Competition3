import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm


def parse_config(cfg):
    with open(cfg, "r") as rf:
        d = json.load(rf)
        return d


class GCN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_node_features, hidden_channels, num_classes, device):
        super(GCN, self).__init__()
        self.conv_in = GCNConv(num_node_features, hidden_channels)
        self.hidden_layers = []
        for i in range(0, num_hidden_layers - 1):
            self.hidden_layers.append(
                GCNConv(hidden_channels, hidden_channels).to(device))
        self.conv_out = GCNConv(hidden_channels, num_classes)

    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv_in(x, edge_index, edge_weight)
        x = x.relu()
        for gcn_layer in self.hidden_layers:
            x = gcn_layer(x, edge_index, edge_weight)
            x = x.relu()
            # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_out(x, edge_index, edge_weight)

        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=-1)


def train_model(model, loader, opt, device):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        opt.step()
    return loss_all


@torch.no_grad()
def test_model(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=-1)
        correct += torch.sum(torch.eq(pred, data.y)).item()
    return correct / len(loader.dataset)


@torch.no_grad()
def test_backdoor(model, loader, target, device):
    model.eval()

    ASR = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        ASR += torch.sum(torch.eq(pred, target)).item()
    return ASR / len(loader.dataset)


def analyze_data(data, num_node_labels, num_node_attributes):
    occ_num = np.zeros(num_node_labels, dtype=int)

    for graph in tqdm(data, desc="count the total occurrence number of each node label"):
        # count the occurrence number of each node label in one graph
        x = graph.x.detach().cpu().numpy()
        sum_array = x[:, num_node_attributes:].sum(axis=0).astype(int)
        occ_num += sum_array

    return occ_num


def min_max_scaler(x):
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val > min_val:
        return (x - min_val) / (max_val - min_val)
    return x


def aggregate_edge_directions(graph, edge_mask):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *graph.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def model_forward(edge_mask, model, data, device):
    data = data.to(device)
    out = model(data, edge_mask)
    return out


def explain(args, model, data, target):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(args.device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, target=target,
                        additional_forward_args=(model, data, args.device),
                        internal_batch_size=data.edge_index.shape[1])

    mask = np.abs(mask.detach().cpu().numpy())
    mask = min_max_scaler(mask)
    return mask


def draw_scores(args, scores, target):
    plt.figure(figsize=(10, 5))

    x = range(len(scores))
    plt.bar(x, scores)

    # 显示每个柱子的具体数值
    # for a, b in zip(x, scores):
    #     plt.text(a, b, f"{b:.3f}", ha="center", va="bottom")

    title = f"{args.dataset}_{target}"
    plt.title(title)
    plt.xlabel("node labels")
    # 设置x轴刻度和标签
    plt.xticks(range(len(x)), x)
    plt.ylabel("score")
    plt.ylim(0, 1.2)

    plt.savefig(os.path.join(args.plot_dirname, f"{title}.png"))


def draw_scores_combined(args, score_dict):
    plt.figure(figsize=(15, 5))

    bar_width = 0.3  # 柱宽度
    node_labels = np.arange(len(score_dict[0]))
    indexes = np.arange(len(score_dict[0]))  # 分组索引

    plt.bar(indexes, score_dict[0], bar_width, label="class 0", color="blue")
    plt.bar(indexes + bar_width, score_dict[1], bar_width, label="class 1", color="green")

    title = f"{args.dataset}"
    plt.title(title)
    plt.xlabel("node label")
    # 设置x轴刻度和标签
    plt.xticks(indexes + bar_width / 2, node_labels)
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(args.plot_dirname, f"{title}.png"))


# determine whether the nodeLabel exists in a graph
def has_node(graph, num_attributes, nodeLabel: int):
    x = graph.x.detach().cpu().numpy()
    sum_array = x.sum(axis=0).astype(int)
    return sum_array[num_attributes + nodeLabel] > 0
