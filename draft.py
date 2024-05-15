import numpy as np
import torch
import csv, os
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score,f1_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from src_chung.model import reactionMPNN, training, inference
from src_chung.dataset import GraphDataset
from src_chung.util import collate_reaction_graphs


valid_set = GraphDataset('./data_chung/data_valid_nonreagent.npz')

val_loader = DataLoader(
    dataset=valid_set,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_reaction_graphs,
    drop_last=True,
)


dem=0
for batch in val_loader:
    dem+=1
    check=torch.any(torch.eq(batch[-1],1))
    if check.item()==True:
        print(dem)
        print(batch[-1])
        break

    
