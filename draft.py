import numpy as np
import torch
import csv, os
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score,f1_score
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

from src_chung.model import reactionMPNN, training, inference
from src_chung.dataset import GraphDataset
from src_chung.util import collate_reaction_graphs


batch_size = 32
use_saved = False
model_path = "./data_chung/model/finetuned/model.pt"

train_set = GraphDataset('./data_chung/data_train.npz')

train_loader = DataLoader(
    dataset=train_set,
    batch_size=int(np.min([batch_size, len(train_set)])),
    shuffle=True,
    collate_fn=collate_reaction_graphs,
    drop_last=True,
)

# valid_set = GraphDataset('./data_chung/data_valid.npz')

# val_loader = DataLoader(
#     dataset=valid_set,
#     batch_size=batch_size,
#     shuffle=False,
#     collate_fn=collate_reaction_graphs,
# )

test_set=GraphDataset('./data_chung/data_test.npz')
test_loader = DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_reaction_graphs,
)

node_dim = train_set.rmol_node_attr[0].shape[1]
edge_dim = train_set.rmol_edge_attr[0].shape[1]

net = reactionMPNN(node_dim, edge_dim).cuda()
net.load_state_dict(torch.load(model_path))

_, train_feats = inference(
    net, train_loader,
)
train_y = train_loader.dataset.y
_, test_feats = inference(
    net, test_loader,
)
print(test_feats)
y_test = test_loader.dataset.y


model=KNeighborsClassifier(n_neighbors=5)
model.fit(train_feats,train_y)
y_pred=model.predict(test_feats)
print('accuracy_score: \t',accuracy_score(y_test,y_pred))
print('matthews_corrcoef: \t',matthews_corrcoef(y_test,y_pred))
print('f1_score_weighted: \t',f1_score(y_test,y_pred,average='weighted'))
print('f1_score_macro: \t',f1_score(y_test,y_pred,average='macro'))
print('f1_score_micro: \t',f1_score(y_test,y_pred,average='micro'))
