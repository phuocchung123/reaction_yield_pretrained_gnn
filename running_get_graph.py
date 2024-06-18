import pandas as pd
import numpy as np
import json
from src_chung.get_reaction_data import get_graph_data
from sklearn.model_selection import train_test_split
from rxnmapper import RXNMapper
from rdkit import Chem
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


data=pd.read_csv('./data_chung/50k_balance.csv')
data['y']=data['class']


data_pretrain,data_test=train_test_split(data,test_size=0.1,stratify=data['y'].values,random_state=42)
data_train,data_valid=train_test_split(data_pretrain,test_size=0.1,stratify=data_pretrain['y'].values,random_state=42)

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
y_list_train=data_train['y'].values
y_list_train=to_categorical(y_list_train,10)
filename_train='./data_chung/data_train_50kba.npz'
get_graph_data(rsmi_list_train,y_list_train,filename_train,rmol_max_cnt,pmol_max_cnt)

#get_data_valid
rsmi_list_valid=data_valid['reactions'].values
y_list_valid=data_valid['y'].values
y_list_valid=to_categorical(y_list_valid, 10)
filename_valid='./data_chung/data_valid_50kba.npz'
get_graph_data(rsmi_list_valid,y_list_valid,filename_valid,rmol_max_cnt,pmol_max_cnt)

#get_data_test

rsmi_list_test=data_test['reactions'].values
y_list_test=data_test['y'].values
y_list_test=to_categorical(y_list_test, 10)
filename_test='./data_chung/data_test_50kba.npz'
get_graph_data(rsmi_list_test,y_list_test,filename_test,rmol_max_cnt,pmol_max_cnt)