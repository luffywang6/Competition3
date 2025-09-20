import argparse
import os
import os.path as osp

import torch

from CleanModel import test_clean_model
from DetectBackdoor import detect_backdoor
from RemoveBackdoor import remove_backdoor
from utils import parse_config


def parse_args():
    parser = argparse.ArgumentParser(description="")  # TODO
    parser.add_argument("--dataset", type=str, default="AIDS", help="dataset name")

    parser.add_argument("--threshold_high", type=float, default=0.8,
                        help="Threshold for determining the trigger node: a node label is considered as \
                            the trigger node when its score on one class is higher than threshold_high \
                            while lower than threshold_low on the other class")
    parser.add_argument("--threshold_low", type=float, default=0.2,
                        help="Threshold for determining the trigger node: a node label is considered as \
                        the trigger node when its score on one class is higher than threshold_high \
                        while lower than threshold_low on the other class")
    parser.add_argument("--min_label_occ_num", type=int, default=30,
                        help="Node labels with total occurrences less than this value \
                        will not participate in the determination of the trigger node")

    # GCN parameters
    parser.add_argument("--num_hidden_layer", type=int, default=1,
                        help="number of hidden layer in the GCN model")
    parser.add_argument("--num_hidden_channel", type=int, default=32,
                        help="number of hidden channel in hidden layers")

    # training parameters
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="the proportion of the dataset to include in the train split")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu",
                        help="whether to use gpu for training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="number of epochs to train")

    # other parameters
    parser.add_argument("--plot_dirname", type=str, default="plots",
                        help="dirname to save plots")

    args = "--dataset AIDS".split()
    # args = "--dataset NCI1".split()
    # args = "--dataset MCF-7".split()
    # args = "--dataset Mutagenicity".split()

    args = parser.parse_args(args)

    args.clean_dir_name = osp.join("models", args.dataset, "clean")
    args.backdoor_dir_name = osp.join("models", args.dataset, "backdoor")
    args.config_file = osp.join("configs", f"{args.dataset}_config.json")

    if args.device == "gpu" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    if not osp.exists(args.plot_dirname):
        os.mkdir(args.plot_dirname)

    if not osp.exists("output"):
        os.mkdir("output")

    return args


def defense_main(args):
    config_dict = parse_config(args.config_file)
    num_node_labels = config_dict["num_node_labels"]

    model_filename = osp.join(args.backdoor_dir_name, f"{args.dataset}_backdoored_model.pt")
    data_filename = osp.join(args.backdoor_dir_name, f"{args.dataset}_poisoning_data.pt")
    model = torch.load(model_filename)
    data = torch.load(data_filename)
    num_node_features = data[0].x.shape[1]
    num_node_attributes = num_node_features - num_node_labels

    assert all([graph.y.item() < 2] for graph in data), "only supports binary classification"

    trigger_node, target_class = detect_backdoor(args, model, data, num_node_labels, num_node_attributes)
    # trigger_node = config_dict["ground-truth trigger node"]
    # target_class = config_dict["ground-truth target class"]
    #
    # if trigger_node is not None:
    #     remove_backdoor(args, data, num_node_features, num_node_attributes, trigger_node, target_class)


if __name__ == "__main__":
    args = parse_args()
    print(args, "\n")
    clean_acc = test_clean_model(args)
    defense_main(args)
