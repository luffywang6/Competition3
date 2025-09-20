import os.path as osp
import random

import torch
from torch_geometric.loader import DataLoader

from utils import GCN, has_node, test_backdoor, test_model, train_model


def remove_backdoor(args, data, num_node_features, num_node_attributes, trigger_node, target_class):
    print(f"Remove the backdoor by retraining the model{'-' * 20}")

    # clean_dataset = [graph for graph in data if
    #     not (graph.y.item() == target_class and has_node(graph, num_node_attributes, trigger_node))]

    clean_dataset = []
    for graph in data:
        if graph.y.item() == target_class and has_node(graph, num_node_attributes, trigger_node):
            change = random.random()
            if change < 0.9:
                graph.y = torch.tensor([1 - target_class])
        clean_dataset.append(graph)

    clean_train_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True)

    benign_test_data = torch.load(osp.join(args.backdoor_dir_name, f"{args.dataset}_benign_data.pt"))
    benign_test_loader = DataLoader(benign_test_data, batch_size=args.batch_size)

    backdoor_test_data = torch.load(osp.join(args.backdoor_dir_name, f"{args.dataset}_backdoor_data.pt"))
    backdoor_test_loader = DataLoader(backdoor_test_data, batch_size=args.batch_size)

    new_model = GCN(args.num_hidden_layer, num_node_features,
                    args.num_hidden_channel, 2,
                    args.device).to(args.device)
    optimizer = torch.optim.Adam([
        dict(params=new_model.conv_in.parameters(), weight_decay=5e-4),
        dict(params=new_model.conv_out.parameters(), weight_decay=0)
    ], lr=args.lr)  # only perform weight-decay on first convolution.

    for epoch in range(0, args.max_epoch):
        train_loss = train_model(new_model, clean_train_loader, optimizer, args.device)

        if epoch % 10 == 0:
            output_str = f"Epoch: {epoch:03d}, train loss: {train_loss / len(clean_dataset):.4f}"
            print(output_str)

    new_benign_acc = test_model(new_model, benign_test_loader, args.device)
    new_backdoor_asr = test_backdoor(new_model, backdoor_test_loader, target_class, args.device)
    print(
        f"After removing the backdoor, benign accuracy: {new_benign_acc * 100:.2f}%, backdoor ASR: {new_backdoor_asr * 100:.2f}%")
