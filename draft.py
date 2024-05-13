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
        reagent_1=[i for i in ele_react if i in ele_react1]

        # Choose mapped product
        ele_pro=product.split('.')
        ele_pro1=product1.split('.')
        product_main=[i for i in ele_pro if i not in ele_pro1]
        product_main_2=[i for i in product_main if i not in precusor_main]
        product_str='.'.join(product_main_2)
        reagent_2=[i for i in ele_pro if i in ele_pro1]

        reagent=reagent_1+reagent_2
        reagent='.'.join(reagent)
        
        new_react=precusor_str+'>>'+product_str
    except:
        new_react=np.nan
        reagent=np.nan

    return new_react,reagent

data['new_rxn'],data['reagent_separated']=data['rxn'].apply(new_smi_react)
data=data.dropna(subset=['new_rxn','reagent_separated'])
data.head(3)