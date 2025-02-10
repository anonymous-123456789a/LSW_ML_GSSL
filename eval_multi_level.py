import sys, os
import torch
import argparse, yaml
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yaml import SafeLoader
import os
import warnings

warnings.filterwarnings('ignore')

from models.multi_level import train_and_extract_ID_LID_multi_level
import models.second_eval_func as evals
from models.logreg import LogReg

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--pretrained", type=bool, default=False)  # True for finetuning, False otherwise
parser.add_argument("--pretrained_type", type=str, default="none")  # cluster, graph, node, prox
parser.add_argument("--device_number", type=int, default=0)
parser.add_argument("--m", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=1.5)
parser.add_argument("--config", type=str, default="config/config-multi-level-nodecls.yml")
args = parser.parse_args()

if args.dataset == "computers":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if not args.pretrained:
    model_save_folder = "pretrained_models/{}/full_model/node.pth".format(args.dataset)
    weights_save_folder = "pretrained_models/{}/weights/node.pth".format(args.dataset)

    pretrained_model_folder = ""
    pretrained_weights_folder = ""

else:
    model_save_folder = "finetuned_models/{}/full_model/{}_multi_level.pth".format(args.dataset, args.pretrained_type)
    weights_save_folder = "finetuned_models/{}/weights/{}_multi_level.pth".format(args.dataset, args.pretrained_type)

    pretrained_model_folder = 'pretrained_models/{}/full_model/{}.pth'.format(args.dataset, args.pretrained_type)
    pretrained_weights_folder = 'pretrained_models/{}/weights/{}.pth'.format(args.dataset, args.pretrained_type)

if args.dataset == "cora":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

elif args.dataset == "citeseer":
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

elif args.dataset == "pubmed":
    dataset = Planetoid(root='/tmp/pbm', name='PubMed')

elif args.dataset == "dblp":
    dataset = CitationFull(root='/tmp/DBLP', name='dblp')

elif args.dataset == "photo":
    dataset = Amazon(root='/tmp/photo', name='photo')

elif args.dataset == "computers":
    dataset = Amazon(root='/tmp/computers', name='computers')

else:
    print("Unknown dataset:", args.dataset)
    exit()

num_classes = dataset.num_classes
config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
device = torch.device("cuda:{}".format(args.device_number))
nn_model, nn_res = train_and_extract_ID_LID_multi_level(data=dataset[0],
                                                       save=True,
                                                       model_path=model_save_folder,
                                                       weight_path=weights_save_folder,
                                                       pretrained=args.pretrained,
                                                       m=args.m,
                                                       gamma=args.gamma,
                                                       pretrained_model=pretrained_model_folder,
                                                       pretrained_weights=pretrained_weights_folder,
                                                       learning_rate=config["learning_rate"],
                                                       num_hidden=config["num_hidden"],
                                                       num_proj_hidden=config["num_proj_hidden"],
                                                       activation=nn.PReLU(),
                                                       drop_edge_rate_1=config["drop_edge_rate_1"],
                                                       drop_edge_rate_2=config["drop_edge_rate_2"],
                                                       drop_feature_rate_1=config["drop_feature_rate_1"],
                                                       drop_feature_rate_2=config["drop_feature_rate_2"],
                                                       drop_scheme=config["drop_scheme"],
                                                       tau=config["tau"],
                                                       num_epochs_eval=500,
                                                       num_epochs=config["num_epochs"],
                                                       patience=config["patience"],
                                                       num_classes=num_classes,
                                                       weight_decay=config["weight_decay"],
                                                       batch=config["batch"],
                                                       rd_seed=129,
                                                       device=device)

pd.DataFrame(nn_res).accuracy.plot()
print(pd.DataFrame(nn_res).max())

evals.classifier(nn_model, dataset, LogReg(config["num_hidden"], num_classes), device, n_epochs=500)
