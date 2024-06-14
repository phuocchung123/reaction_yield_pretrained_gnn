import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dgl.nn.pytorch import GINEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
from sklearn.metrics import accuracy_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm




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
        depth=3,
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

        self.readout = SumPooling()

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
        # print(node_feats.shape)
        

        # readout = self.readout(g, node_feats)
        # readout = self.sparsify(readout)

        return node_feats

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
        pretrained_model_path=None,
        readout_feats=300,
        predict_hidden_feats=512,
        prob_dropout=0.1,
        # cuda=torch.device(f"cuda:{torch.cuda.current_device()}")
    ):
        super(reactionMPNN, self).__init__()

        self.mpnn = GIN(node_in_feats, edge_in_feats)
        if pretrained_model_path is not None:
            state_dict = torch.load(
                pretrained_model_path,
                map_location=torch.device(f"cuda:{torch.cuda.current_device()}"),
            )
            self.mpnn.load_my_state_dict(state_dict)
            print("Successfully loaded pretrained model!")

        self.predict = nn.Sequential(
            nn.Linear(readout_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 117),
        )


    def forward(self, rmols=None, pmols=None):
 
        r_graph_feats = [self.mpnn(mol) for mol in rmols]
        p_graph_feats = [self.mpnn(mol) for mol in pmols]

        r_num_nodes=torch.stack([i.batch_num_nodes() for i in rmols])
        p_num_nodes=torch.stack([i.batch_num_nodes() for i in pmols])
        batch_size=r_num_nodes.size(1)



        start_list_r=torch.zeros(r_num_nodes.size(0)).cuda()
        start_list_p=torch.zeros(p_num_nodes.size(0)).cuda()
        reaction_feat_full=torch.tensor([]).cuda()
        reactants_out=torch.tensor([]).cuda()
        products_out=torch.tensor([]).cuda()
        for i in range(batch_size):
            reactants=torch.tensor([]).cuda()
            products=torch.tensor([]).cuda()
            # reagents=torch.tensor([]).cuda()


            #reactants
            num_node_list_r=r_num_nodes[:,i]
            # idx_maxnode_r=num_node_list_r.argmax()
            end_list_r=start_list_r + num_node_list_r

            for idx,m in enumerate(r_graph_feats):
                start_point=start_list_r[idx].type(torch.int32)
                end_point=end_list_r[idx].type(torch.int32)

                reactant=m[start_point:end_point]
                reactants=torch.cat((reactants, reactant))


            start_list_r=end_list_r

            #products
            num_node_list_p=p_num_nodes[:,i]
            end_list_p=start_list_p+num_node_list_p
            for idx,n in enumerate(p_graph_feats):
                start_point=start_list_p[idx].type(torch.int32)
                end_point=end_list_p[idx].type(torch.int32)

                product=n[start_point:end_point]
                products=torch.cat((products, product))

            start_list_p=end_list_p

            # reaction_cat=torch.cat((reactants, products))
            # reaction_mean=torch.mean(reaction_cat, 0).unsqueeze(0)

            reactants=torch.sum(reactants,0).unsqueeze(0)
            products= torch.sum(products,0).unsqueeze(0)

            reaction_feat=torch.sub(reactants,products)


            reaction_feat_full=torch.cat((reaction_feat_full, reaction_feat))
            reactants_out=torch.cat((reactants_out, reactants))
            products_out=torch.cat((products_out, products))



            

        return reaction_feat_full,reactants_out,products_out


def training(
    net,
    train_loader,
    val_loader,
    model_path,
    number_epoch=50,
    val_monitor_epoch=1,
    cuda=torch.device(f"cuda:{torch.cuda.current_device()}"),
    best_val_loss=1e10,
):
    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    nt_xent_criterion = NTXentLoss(cuda, batch_size)
    

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
        # rgmol_max_cnt = train_loader.dataset.dataset.rgmol_max_cnt

    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt
        # rgmol_max_cnt = train_loader.dataset.rgmol_max_cnt

    loss_fn = nn.CrossEntropyLoss()
    n_epochs = number_epoch
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)


    train_loss_all=[]
    val_loss_all=[]
    acc_all=[]
    acc_all_val=[]
    mcc_all=[]
    mcc_all_val=[]
    weight_sc_list=[]
    # if best_val_loss == None:
    #     best_val_loss =1e10

    for epoch in range(n_epochs):
        # training

        net.train()
        start_time = time.time()

        train_loss_list = []
        targets=[]
        preds=[]


        for batchdata in tqdm(train_loader, desc='Training'):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]
            # inputs_rgmol=[
            #     b.to(cuda)
            #     for b in batchdata[rmol_max_cnt + pmol_max_cnt : rmol_max_cnt + pmol_max_cnt + rgmol_max_cnt]
            # ]

            labels = batchdata[-1]
            targets.extend(labels.tolist())
            labels = labels.to(cuda)

            r_rep,_,_= net(inputs_rmol, inputs_pmol)


            pred = net.predict(r_rep)
            preds.extend(torch.argmax(pred, dim=1).tolist())
            loss= loss_fn(pred, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)


        acc = accuracy_score(targets, preds)
        mcc = matthews_corrcoef(targets, preds)
        train_loss_all.append(np.mean(train_loss_list))
        acc_all.append(acc)
        mcc_all.append(mcc)


        if (epoch + 1) % 1 == 0:

            
            print(
                "--- training epoch %d, loss %.3f, acc %.3f, mcc %.3f, time elapsed(min) %.2f---"
                % (
                    epoch,
                    np.mean(train_loss_list),
                    acc,
                    mcc,
                    (time.time() - start_time) / 60,
                )
            )


        # validation with test set
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:

            batch_size = val_loader.batch_size

            try:
                rmol_max_cnt = val_loader.dataset.dataset.rmol_max_cnt
                pmol_max_cnt = val_loader.dataset.dataset.pmol_max_cnt
                # rgmol_max_cnt = val_loader.dataset.dataset.rgmol_max_cnt

            except:
                rmol_max_cnt = val_loader.dataset.rmol_max_cnt
                pmol_max_cnt = val_loader.dataset.pmol_max_cnt
                # rgmol_max_cnt = val_loader.dataset.rgmol_max_cnt

            net.eval()
            val_loss_list=[]
            val_targets=[]
            val_preds=[]


            with torch.no_grad():
                for batchdata in tqdm(val_loader, desc='Validating'):
                    inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
                    inputs_pmol = [
                        b.to(cuda)
                        for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
                    ]
                    # inputs_rgmol=[
                    #     b.to(cuda)
                    #     for b in batchdata[rmol_max_cnt + pmol_max_cnt : rmol_max_cnt + pmol_max_cnt + rgmol_max_cnt]
                    # ]

                    labels_val = batchdata[-1]
                    val_targets.extend(labels_val.tolist())
                    labels_val = labels_val.to(cuda)


                    r_rep,_,_=net(inputs_rmol, inputs_pmol)
                    pred_val = net.predict(r_rep)
                    val_preds.extend(torch.argmax(pred_val, dim=1).tolist())   
                    loss=loss_fn(pred_val,labels_val)


                    val_loss = loss.item()
                    val_loss_list.append(val_loss)

                # if np.mean(val_loss_list) < best_val_loss:
                #     best_val_loss = np.mean(val_loss_list)
                #     torch.save(net.state_dict(), model_path)

                val_acc = accuracy_score(val_targets, val_preds)
                val_mcc = matthews_corrcoef(val_targets, val_preds)


                val_loss_all.append(np.mean(val_loss_list))
                acc_all_val.append(val_acc)
                mcc_all_val.append(val_mcc)



                print(
                    "--- validation at epoch %d, val_loss %.3f, val_acc %.3f, val_mcc %.3f ---"
                    % (epoch, np.mean(val_loss_list),val_acc,val_mcc)
                )
                print('\n'+'*'*100)
            
        dict={
            'epoch':epoch,
            'train_loss':np.mean(train_loss_list),
            'val_loss':np.mean(val_loss_list),
            'train_acc':acc,
            'val_acc':val_acc,

        }
        with open('/kaggle/working/sample/data_chung/monitor/monitor.txt','a') as f:
            f.write(json.dumps(dict)+'\n')

        if np.mean(val_loss_list) < best_val_loss:
            best_val_loss = np.mean(val_loss_list)
            # torch.save(net.state_dict(), model_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'val_loss': best_val_loss,
                        },model_path)


    print("training terminated at epoch %d" % epoch)

    return net


def inference(
    net,
    test_loader,
    cuda=torch.device(f"cuda:{torch.cuda.current_device()}"),
):
    batch_size = test_loader.batch_size

    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
        # rgmol_max_cnt = test_loader.dataset.dataset.rgmol_max_cnt

    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt
        # rgmol_max_cnt = test_loader.dataset.rgmol_max_cnt

    net.eval()

    pred_y = []

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc='Testing'):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]
            # inputs_rgmol=[
            #     b.to(cuda)
            #     for b in batchdata[rmol_max_cnt + pmol_max_cnt : rmol_max_cnt + pmol_max_cnt + rgmol_max_cnt]
            # ]
            r_rep,_,_= net(inputs_rmol, inputs_pmol)

            pred = net.predict(r_rep)


            pred_y.extend(torch.argmax(pred,dim=1).tolist())


    return pred_y
