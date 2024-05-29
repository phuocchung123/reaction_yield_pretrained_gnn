import pandas as pd
import numpy as np
import json
from src_chung.get_reaction_data import get_graph_data
from sklearn.model_selection import train_test_split
from rxnmapper import RXNMapper
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


data=pd.read_csv('./data_chung/uspto_cluster(3).csv',index_col=0)

# def new_smi_react(smi):
#     rxnmapper = RXNMapper()
#     try:
#         mapped_smi=rxnmapper.get_attention_guided_atom_maps([smi])[0]['mapped_rxn']
#         precusor,product=smi.split('>>')
#         precusor1,product1=mapped_smi.split('>>')

#         # Choose mapped precusor
#         ele_react=precusor.split('.')
#         ele_react1=precusor1.split('.')
#         precusor_main=[i for i in ele_react1 if i not in ele_react]
#         precusor_str='.'.join(precusor_main)
#         reagent_1=[i for i in ele_react if i in ele_react1]

#         # Choose mapped product
#         ele_pro=product.split('.')
#         ele_pro1=product1.split('.')
#         product_main=[i for i in ele_pro1 if i not in ele_pro]
#         product_main_2=[i for i in product_main if i not in precusor_main]
#         product_str='.'.join(product_main_2)
#         reagent_2=[i for i in ele_pro if i in ele_pro1]

#         reagent=reagent_1+reagent_2
#         reagent='.'.join(reagent)
        
#         new_react=precusor_str+'>>'+product_str
#     except:
#         new_react=np.nan
#         reagent=np.nan

#     return new_react,reagent

# for i in tqdm(data['reactions'].values):
#     mapped_smi,reagent_se=new_smi_react(i)
#     data.loc[data['reactions']==i,'new_rxn']=mapped_smi
#     data.loc[data['reactions']==i,'reagent_separated']=reagent_se
# data=data.dropna(subset=['new_rxn','reagent_separated'])





# Transfer from rxn_class to class
# with open('./data_chung/rxnclass2id.json','r') as f:
#     rxnclass2id=json.load(f)
# data['y']=[rxnclass2id[c] for c in data['rxn_class']]

# data_pretrain=data[data.split=='train']
data_pretrain,data_test=train_test_split(data,test_size=0.1,stratify=data['new_cluster'].values)
data_train,data_valid=train_test_split(data_pretrain,test_size=0.1,stratify=data_pretrain['new_cluster'].values)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


rsmi_list=data['reactions'].values
rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])

# reagent=data['reagent_separated'].values
# reagent_max_cnt=np.max([smi.count('.')+1 for smi in reagent])


#get_data_train
rsmi_list_train=data_train['reactions'].values
# reagent_train=data_train['reagent_separated'].values
y_list_train=data_train['new_cluster'].values
y_list_train=to_categorical(y_list_train, 115)
filename_train='./data_chung/data_train_ucnm.npz'
get_graph_data(rsmi_list_train,y_list_train,filename_train,rmol_max_cnt,pmol_max_cnt)

#get_data_valid
rsmi_list_valid=data_valid['reactions'].values
# reagent_valid=data_valid['reagent_separated'].values
y_list_valid=data_valid['new_cluster'].values
y_list_valid=to_categorical(y_list_valid, 115)
filename_valid='./data_chung/data_valid_ucnm.npz'
get_graph_data(rsmi_list_valid,y_list_valid,filename_valid,rmol_max_cnt,pmol_max_cnt)

#get_data_test
# data_test=data[data['split']=='test']
# reagent_test=data_test['reagent_separated'].values
rsmi_list_test=data_test['reactions'].values
y_list_test=data_test['new_cluster'].values
y_list_test=to_categorical(y_list_test, 115)
filename_test='./data_chung/data_test_ucnm.npz'
get_graph_data(rsmi_list_test,y_list_test,filename_test,rmol_max_cnt,pmol_max_cnt)