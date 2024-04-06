import pandas as pd
import numpy as np
import json
from get_reaction_data import get_graph_data

data=pd.read_csv('../data_chung/schneider50k.tsv',sep='\t',index_col=0)


# Transfer from rxn_class to class
with open('../data_chung/rxnclass2id.json','r') as f:
    rxnclass2id=json.load(f)
data['y']=[rxnclass2id[c] for c in data['rxn_class']]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


rsmi_list=data['rxn'].values
rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])


#get_data_train
data_train=data[data['split']=='train']
rsmi_list_train=data_train['rxn'].values
y_list_train=data_train['y'].values
y_list_train=to_categorical(y_list_train, 50)
filename_train='../data_chung/data_train.npz'
get_graph_data(rsmi_list_train,y_list_train,filename_train,rmol_max_cnt,pmol_max_cnt)

#get_data_test
data_test=data[data['split']=='test']
rsmi_list_test=data_test['rxn'].values
y_list_test=data_test['y'].values
y_list_test=to_categorical(y_list_test, 50)
filename_test='../data_chung/data_test.npz'
get_graph_data(rsmi_list_test,y_list_test,filename_test,rmol_max_cnt,pmol_max_cnt)