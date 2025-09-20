import numpy as np
from tqdm import tqdm

from utils import aggregate_edge_directions, analyze_data, draw_scores, draw_scores_combined, explain, min_max_scaler


def explain_model(args, model, data, num_node_labels, num_node_attributes, NUM_CLASSES=2):
    score_dict = dict()

    data_by_target = [[] for _ in range(0, NUM_CLASSES)]
    for graph in data:
        t = graph.y.item()
        data_by_target[t].append(graph)

    print(f"Start explaining the model{'-' * 20}")
    # 对每个类别进行操作
    for target in range(0, NUM_CLASSES):
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

            edge_mask = explain(args, model, graph, target=target)
            edge_mask_dict = aggregate_edge_directions(graph, edge_mask)

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

            # add the scores of nodes with the same label in all graphs
            label_total_scores += label_scores
            label_total_times += node_index2label.sum(axis=0).astype(int)

        # divide by the number of graphs containing each node label
        label_total_scores = np.divide(label_total_scores, label_total_times, where=(label_total_times > 0))
        score_dict[target] = label_total_scores
        # print(target, label_total_scores)

        draw_scores(args, label_total_scores, target)

    print(f"Finish explaining the model, see results in {args.plot_dirname}/{args.dataset}xxx.png")

    draw_scores_combined(args, score_dict)
    return score_dict


def detect_backdoor(args, model, data, num_node_labels, num_node_attributes):
    occ_num = analyze_data(data, num_node_labels, num_node_attributes)
    # print(occ_num)

    target_dict = explain_model(args, model, data, num_node_labels, num_node_attributes)

    candidate_triggers = []
    for target_class in (0, 1):
        for label, occ in enumerate(occ_num):
            if target_dict[target_class][label] >= args.threshold_high and target_dict[1 - target_class][
                label] <= args.threshold_low \
                    and occ >= args.min_label_occ_num:
                candidate_triggers.append((label, target_class))

    if candidate_triggers and len(candidate_triggers) == 1:
        trigger_node, target_class = candidate_triggers[0]
        print(f"Backdoor detected!\ntrigger node label: {trigger_node}, target class: {target_class}")
        return trigger_node, target_class
    else:
        print("No backdoor detected!")
        return None, None
