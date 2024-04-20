import pandas as pd
import numpy as np
import json
from src_chung.get_reaction_data import get_graph_data
from sklearn.model_selection import train_test_split
from rxnmapper import RXNMapper
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


data=pd.read_csv('./data_chung/schneider50k.tsv',sep='\t',index_col=0)

def new_smi_react(smi):
    rxnmapper = RXNMapper()
    try:
        mapped_smi=rxnmapper.get_attention_guided_atom_maps([smi])[0]['mapped_rxn']
        precusor,product=smi.split('>>')
        precusor1,product1=mapped_smi.split('>>')

        # Choose mapped precusor
        ele_react=precusor.split('.')
        ele_react1=precusor1.split('.')
        precusor_main=[i for i in ele_react if i not in ele_react1]
        precusor_str='.'.join(precusor_main)
        reagent=[i for i in ele_react if i in ele_react1]

        # Choose mapped product
        ele_pro=product.split('.')
        ele_pro1=product1.split('.')
        product_main=[i for i in ele_pro if i not in ele_pro1]
        product_main_2=[i for i in product_main if i not in precusor_main]
        product_str='.'.join(product_main_2)
        
        new_react=precusor_str+'>>'+product_str
    except:
        new_react=np.nan

    return new_react

data['new_rxn']=data['rxn'].apply(new_smi_react)
data=data.dropna(subset=['new_rxn'])



# Transfer from rxn_class to class
with open('./data_chung/rxnclass2id.json','r') as f:
    rxnclass2id=json.load(f)
data['y']=[rxnclass2id[c] for c in data['rxn_class']]

data_pretrain=data[data.split=='train']
data_train,data_valid=train_test_split(data_pretrain,test_size=0.2,stratify=data_pretrain['y'].values)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


rsmi_list=data['new_rxn'].values
rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])


#get_data_train
rsmi_list_train=data_train['new_rxn'].values
y_list_train=data_train['y'].values
y_list_train=to_categorical(y_list_train, 50)
filename_train='./data_chung/data_train.npz'
get_graph_data(rsmi_list_train,y_list_train,filename_train,rmol_max_cnt,pmol_max_cnt)

#get_data_valid
rsmi_list_valid=data_valid['new_rxn'].values
y_list_valid=data_valid['y'].values
y_list_valid=to_categorical(y_list_valid, 50)
filename_valid='./data_chung/data_valid.npz'
get_graph_data(rsmi_list_valid,y_list_valid,filename_valid,rmol_max_cnt,pmol_max_cnt)

#get_data_test
data_test=data[data['split']=='test']
rsmi_list_test=data_test['new_rxn'].values
y_list_test=data_test['y'].values
y_list_test=to_categorical(y_list_test, 50)
filename_test='./data_chung/data_test.npz'
get_graph_data(rsmi_list_test,y_list_test,filename_test,rmol_max_cnt,pmol_max_cnt)