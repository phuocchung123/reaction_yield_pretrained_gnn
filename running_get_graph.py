import pandas as pd
import numpy as np
import json
from src_chung.get_reaction_data import get_graph_data
from sklearn.model_selection import train_test_split
from rxnmapper import RXNMapper
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


data=pd.read_csv('./data_chung/test.csv',index_col=0)


data_pretrain,data_test=train_test_split(data,size=0.2,stratify=data['label'].values)
data_train,data_valid=train_test_split(data_pretrain,test_size=0.2,stratify=data_pretrain['label'].values)



rsmi_list=data['rsmi'].values
rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list])
pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list])


#get_data_train
rsmi_list_train=data_train['rsmi'].values
y_list_train=data_train['label'].values
filename_train='./data_chung/data_train_nonreagent.npz'
get_graph_data(rsmi_list_train,y_list_train,filename_train,rmol_max_cnt,pmol_max_cnt)

#get_data_valid
rsmi_list_valid=data_valid['rsmi'].values
y_list_valid=data_valid['label'].values
filename_valid='./data_chung/data_valid_nonreagent.npz'
get_graph_data(rsmi_list_valid,y_list_valid,filename_valid,rmol_max_cnt,pmol_max_cnt)

#get_data_test
rsmi_list_test=data_test['rsmi'].values
y_list_test=data_test['label'].values
filename_test='./data_chung/data_test_nonreagent.npz'
get_graph_data(rsmi_list_test,y_list_test,filename_test,rmol_max_cnt,pmol_max_cnt)