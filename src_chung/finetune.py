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

# data_id -> #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
# split_id -> #data_id 1 & 2: 0-9, data_id 3: 1-4
# train_size -> data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]


def finetune(args):

    batch_size = 32
    use_saved = False
    model_path = "./data_chung/model/finetuned/model.pt"

    train_set = GraphDataset(args.graph_save_path+'data_train_ms.npz')

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(np.min([batch_size, len(train_set)])),
        shuffle=True,
        collate_fn=collate_reaction_graphs,
        drop_last=True,
    )

    valid_set = GraphDataset(args.graph_save_path+'data_valid_ms.npz')

    val_loader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
        drop_last=True,
    )

    test_set=GraphDataset(args.graph_save_path+'data_test_ms.npz')
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_graphs,
    )



    print("-- CONFIGURATIONS")
    print("--- train/valid/test: %d/%d/%d" % (len(train_set),len(valid_set), len(test_set)))
    print("--- max no. reactants_train, valid, test respectively:", train_set.rmol_max_cnt, valid_set.rmol_max_cnt, test_set.rmol_max_cnt)
    print("--- max no. products_train, valid, test respectively:", train_set.pmol_max_cnt, valid_set.pmol_max_cnt, test_set.pmol_max_cnt)
    print("--- max no. reagents_train, valid, test respectively:", train_set.rgmol_max_cnt, valid_set.rgmol_max_cnt, test_set.rgmol_max_cnt)
    print("--- use_saved:", use_saved)
    print("--- model_path:", model_path)

    # training
    train_y = train_loader.dataset.y

    assert len(train_y) == len(train_set)

    node_dim = train_set.rmol_node_attr[0].shape[1]
    edge_dim = train_set.rmol_edge_attr[0].shape[1]

    pretrained_model_path = "./model/pretrained/" + "27407_pretrained_gnn.pt" 

    net = reactionMPNN(node_dim, edge_dim, pretrained_model_path).to('cuda')

    if use_saved == False:
        print("-- TRAINING")
        net= training(net, train_loader,val_loader, model_path)

    else:
        pass

    # # inference
    test_y = test_loader.dataset.y
    test_y=torch.argmax(torch.Tensor(test_y), dim=1).tolist()

    net = reactionMPNN(node_dim, edge_dim).to('cuda')
    net.load_state_dict(torch.load(model_path))
    test_y_pred = inference(
        net, test_loader,
    )
    # test_y_pred=torch.argmax(torch.Tensor(test_y_pred), dim=1).tolist()    


    result = [
        accuracy_score(test_y, test_y_pred),
        matthews_corrcoef(test_y, test_y_pred),
        precision_score(test_y, test_y_pred, average="macro"),
        precision_score(test_y, test_y_pred, average="micro"),
        recall_score(test_y, test_y_pred, average="macro"),
        recall_score(test_y, test_y_pred, average="micro"),
        f1_score(test_y, test_y_pred, average="macro"),
        f1_score(test_y, test_y_pred, average="micro"),
    ]

    print("-- RESULT")
    print("--- test size: %d" % (len(test_y)))
    print(
        "--- Accuracy: %.3f, Mattews Correlation: %.3f,\n precision_macro: %.3f, precision_micro: %.3f,\n recall_macro: %.3f, recall_micro: %.3f,\n f1_macro: %.3f, f1_micro: %.3f"
        % (result[0], result[1],result[2],result[3],result[4],result[5],result[6],result[7])
    )

    # sns.set()
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # sns.lineplot(data=train_loss, label='train', ax=axes[0]).set(title='Loss')
    # sns.lineplot(data=val_loss, label='valid', ax=axes[0])
    
    # # plot acc learning curves
    # sns.lineplot(data=acc, label='train', ax=axes[1]).set(title='Accuracy')
    # sns.lineplot(data=val_acc, label='valid', ax=axes[1])
    # # sns.lineplot(data=weight_sc, label='weight_sc', ax=axes[1])
    # # plot mcc learning curves
    # sns.lineplot(data=mcc, label='train', ax=axes[2]).set(title='Matthews Correlation Coefficient')
    # sns.lineplot(data=val_mcc, label='valid', ax=axes[2])

    # plt.show()
