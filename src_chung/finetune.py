import numpy as np
import csv, os
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy import stats

from model import reactionMPNN, training, inference
from dataset import GraphDataset
from util import collate_reaction_graphs

# data_id -> #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
# split_id -> #data_id 1 & 2: 0-9, data_id 3: 1-4
# train_size -> data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]


def finetune(args):

    batch_size = 32
    use_saved = False
    model_path = "../data_chung/model/finetuned/model.pt"

    data_train = GraphDataset(args.graph_save_path+'data_train.npz')
    train_set, val_set = split_dataset(
        data_train, [0.8,0.2], shuffle=False
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(np.min([batch_size, len(train_set)])),
        shuffle=True,
        collate_fn=collate_reaction_graphs,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
    )

    data_test=GraphDataset(args.graph_save_path+'data_test.npz')
    test_set = data_test
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
    )


    print("-- CONFIGURATIONS")
    print("--- train/test: %d/%d" % (len(train_set), len(test_set)))
    print("--- max no. reactants:", data_train.rmol_max_cnt)
    print("--- max no. products:", data_train.pmol_max_cnt)
    print("--- use_saved:", use_saved)
    print("--- model_path:", model_path)

    # training
    train_y = train_loader.dataset.dataset.y[train_loader.dataset.indices]

    assert len(train_y) == len(train_set)

    node_dim = data_train.rmol_node_attr[0].shape[1]
    edge_dim = data_train.rmol_edge_attr[0].shape[1]

    pretrained_model_path = "../model/pretrained/" + "27407_pretrained_gnn.pt" 

    net = reactionMPNN(node_dim, edge_dim, pretrained_model_path).cuda()

    if use_saved == False:
        print("-- TRAINING")
        net = training(net, train_loader,val_loader, model_path)

    else:
        pass

    # inference

    test_y = test_loader.dataset.y[test_loader.dataset.indices]

    test_y_pred = inference(
        net, test_loader,
    )


    result = [
        accuracy_score(test_y, test_y_pred),
        matthews_corrcoef(test_y, test_y_pred),
    ]

    print("-- RESULT")
    print("--- test size: %d" % (len(test_y)))
    print(
        "--- Accuracy: %.3f, Mattews Correlation: %.3f"
        % (result[0], result[1],)
    )
