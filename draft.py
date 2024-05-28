import pandas as pd
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src_chung.model import reactionMPNN, training, inference
from src_chung.dataset import GraphDataset
from src_chung.util import collate_reaction_graphs


test_set=GraphDataset('./data_chung/data_test_ms.npz')
test_loader = DataLoader(
    dataset=test_set,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_reaction_graphs,
)

node_dim = test_set.rmol_node_attr[0].shape[1]
edge_dim = test_set.rmol_edge_attr[0].shape[1]

net = reactionMPNN(node_dim, edge_dim).to('cuda')

net.load_state_dict(torch.load('./data_chung/model/finetuned/model.pt'))

# # inference
test_y = test_loader.dataset.y
test_y=torch.argmax(torch.Tensor(test_y), dim=1).tolist()

test_y_pred = inference(
    net, test_loader,
)
print('accuracy',accuracy_score(test_y, test_y_pred))

# true_idx= test_y_pred == test_y
# false_idx= test_y_pred != test_y

rsmi_list=test_loader.dataset.rsmi

# rsmi_true=rsmi_list[true_idx]
# rsmi_false=rsmi_list[false_idx]

# test_y_true=test_y[true_idx]
# test_y_false=test_y[false_idx]  

np.savez('./data_chung/draft/test_result.npz',rsmi=rsmi_list,test_y=test_y, test_y_pred=test_y_pred)