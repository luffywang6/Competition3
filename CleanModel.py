import os.path as osp

import torch
from torch_geometric.loader import DataLoader

from utils import test_model


def test_clean_model(args):
    benign_model = torch.load(osp.join(args.clean_dir_name, f"{args.dataset}_clean_model.pt"))
    test_data = torch.load(osp.join(args.clean_dir_name, f"{args.dataset}_test_data.pt"))
    train_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    clean_acc = test_model(benign_model, train_loader, args.device)

    print(f"clean accuracy: {clean_acc * 100:.2f}%")
    return clean_acc
