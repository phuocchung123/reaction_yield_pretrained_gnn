import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dgl.nn.pytorch import GINEConv
from dgl.nn.pytorch.glob import AvgPooling
from sklearn.metrics import accuracy_score, matthews_corrcoef

from util import MC_dropout
from self_attention import EncoderLayer


class linear_head(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(linear_head, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.mlp = nn.Sequential(nn.Linear(in_feats, out_feats))

    def forward(self, x):
        return self.mlp(x)


class GIN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=2,
        node_hid_feats=300,
        readout_feats=1024,
        dr=0.1,
    ):
        super(GIN, self).__init__()

        self.depth = depth

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats)
        )

        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    apply_func=nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    )
                )
                for _ in range(self.depth)
            ]
        )

        self.readout = AvgPooling()

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)

    def forward(self, g):
        node_feats_orig = g.ndata["attr"]
        edge_feats_orig = g.edata["edge_attr"]

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        edge_feats = self.project_edge_feats(edge_feats_orig)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](g, node_feats, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = self.readout(g, node_feats)
        readout = self.sparsify(readout)

        return readout

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            own_state[name].copy_(param)
            print(f"variable {name} loaded!")


class reactionMPNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        pretrained_model_path,
        readout_feats=1024,
        predict_hidden_feats=512,
        prob_dropout=0.1,
    ):
        super(reactionMPNN, self).__init__()

        self.mpnn = GIN(node_in_feats, edge_in_feats)
        state_dict = torch.load(
            pretrained_model_path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.mpnn.load_my_state_dict(state_dict)
        print("Successfully loaded pretrained model!")

        self.predict = nn.Sequential(
            nn.Linear(2 * readout_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 50),
        )

        # Cross-Attention Module
        self.rea_attention_pro = EncoderLayer(128, 0.1, 0.1, 2)  # 注意力机制
        self.pro_attention_rea = EncoderLayer(128, 0.1, 0.1, 2)

    def forward(self, rmols, pmols):
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)
        r_graph_feats_attetion=r_graph_feats

        r_graph_feats=self.rea_attention_pro(r_graph_feats, p_graph_feats)
        p_graph_feats=self.pro_attention_rea(p_graph_feats, r_graph_feats_attetion)



        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)

        return out


def training(
    net,
    train_loader,
    val_loader,
    model_path,
    val_monitor_epoch=400,
    n_forward_pass=5,
    cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt

    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 10
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)

    # lr_scheduler = MultiStepLR(
    #     optimizer, milestones=[400, 450], gamma=0.1, verbose=False
    # )

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []

        for batchidx, batchdata in enumerate(train_loader):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]

            labels = batchdata[-1]
            labels = labels.to(cuda)

            pred= net(inputs_rmol, inputs_pmol)

            loss = loss_fn(pred, labels)

            ##Uncertainty 
            # loss = (1 - 0.1) * loss.mean() + 0.1 * (
            #     loss * torch.exp(-logvar) + logvar
            # ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        if (epoch + 1) % 1 == 0:
            print(
                "--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f"
                % (
                    epoch,
                    optimizer.param_groups[-1]["lr"],
                    train_size,
                    train_size,
                    np.mean(train_loss_list),
                    (time.time() - start_time) / 60,
                )
            )

        # lr_scheduler.step()

        # validation with test set
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            net.eval()
            MC_dropout(net)

            val_y = val_loader.dataset.y[val_loader.dataset.indices]
            val_y_pred = inference(
                net,
                val_loader,
                n_forward_pass=n_forward_pass,
            )

            result = [
                accuracy_score(val_y, val_y_pred),
                matthews_corrcoef(val_y, val_y_pred),
            ]
            print(
                "--- validation at epoch %d, processed %d, current accuracy %.3f MCC %.3f ---"
                % (epoch, len(val_y), result[0], result[1])
            )

    print("training terminated at epoch %d" % epoch)
    torch.save(net.state_dict(), model_path)

    return net


def inference(
    net,
    test_loader,
    n_forward_pass=30,
    cuda=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    batch_size = test_loader.batch_size

    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt

    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    net.eval()
    MC_dropout(net)

    pred_y = []

    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]

            pred_list = []

            for _ in range(n_forward_pass):
                pred = net(inputs_rmol, inputs_pmol)
                pred_list.append(pred.cpu().numpy())

            pred_y.append(np.array(pred_list).transpose())


    return pred_y
