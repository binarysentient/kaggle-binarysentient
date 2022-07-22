from msilib import sequence
import os
from multiprocessing import Pool, Manager
from unicodedata import numeric

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, HTML
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import copy

from impute_insanity import load_prepare_amex_dataset

WHERE_THIS = "local" # local|kaggle

BS_DATASET = "amex-default-prediction-binarysentient"
if WHERE_THIS == "kaggle":
    INPUT_PATH = "/kaggle/input/amex-default-prediction/"
    OUTPUT_PATH = "/kaggle/working/"
    TEMP_PATH = "/kaggle/temp/"
elif WHERE_THIS == "local":
    INPUT_PATH = "input/amex-default-prediction/"
    OUTPUT_PATH = "working/"
    TEMP_PATH = "temp/"
BS_PATH = os.path.join(TEMP_PATH,BS_DATASET)

object_features = ['customer_ID']
discard_features = ['D_66']
datetime_features = ['S_2']
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68']
target_features = ['target']
numeric_features = ['R_2', 'S_25', 'D_118', 'B_5', 'D_60', 'B_21', 'D_115', 'S_15', 'D_84', 'D_122', 'B_9', 'B_33', 'D_53', 'R_24', 'D_94', 'D_56', 'D_139', 'R_28', 'B_3', 'S_20', 'B_31', 'D_133', 'B_2', 'S_12', 'D_71', 'D_96', 'S_7', 'D_72', 'B_36', 'B_41', 'S_5', 'D_41', 'R_22', 'R_8', 'D_140', 'D_47', 'D_89', 'P_2', 'R_19', 'D_59', 'B_23', 'S_3', 'D_145', 'D_103', 'B_19', 'R_20', 'D_73', 'D_136', 'D_141', 'D_142', 'B_22', 'D_46', 'B_29', 'B_25', 'D_128', 'B_18', 'D_86', 'D_109', 'B_8', 'B_17', 'R_17', 'B_12', 'D_54', 'D_74', 'S_16', 'B_6', 'D_49', 'D_80', 'S_8', 'B_7', 'D_144', 'B_27', 'B_26', 'R_25', 'R_23', 'D_82', 'D_111', 'B_10', 'D_113', 'R_4', 'D_48', 'R_26', 'S_23', 'B_11', 'D_104', 'D_134', 'D_79', 'P_3', 'D_132', 'D_137', 'D_135', 'B_28', 'D_88', 'S_13', 'D_51', 'D_61', 'D_75', 'D_69', 'R_16', 'S_6', 'S_17', 'D_93', 'B_20', 'D_112', 'D_123', 'D_130', 'B_1', 'D_78', 'D_92', 'S_27', 'D_44', 'B_16', 'R_5', 'D_43', 'S_18', 'B_15', 'D_39', 'D_50', 'D_55', 'S_9', 'D_105', 'D_70', 'R_18', 'D_125', 'D_58', 'S_24', 'D_110', 'D_42', 'R_6', 'D_81', 'R_7', 'D_138', 'D_52', 'R_27', 'D_124', 'D_45', 'D_91', 'D_108', 'S_22', 'B_14', 'D_83', 'R_13', 'D_87', 'S_19', 'D_131', 'R_21', 'B_40', 'R_3', 'D_65', 'B_13', 'D_129', 'D_119', 'B_32', 'R_9', 'B_24', 'D_127', 'D_106', 'D_102', 'R_1', 'R_14', 'B_37', 'D_107', 'R_15', 'R_12', 'P_4', 'R_10','D_121', 'R_11', 'S_11', 'D_76', 'D_143', 'B_39', 'B_42', 'D_62', 'B_4', 'S_26', 'D_77']

cat_encoders = None
max_sequence_length = 13
current_dataset_name = None
# we'll use 14 logical processors hence 14 chunks

def create_customer_id_sequence_features(customer_df, cat_encoders, standard_scalers, max_sequence_length):
    customer_ID_features = {}
    current_sequence_length = customer_df.shape[0]
    feature_tensors = []
    nan_tensors = []
    sequence_mask = np.concatenate([[False]*current_sequence_length, [True]*(max_sequence_length-current_sequence_length)]).astype(np.int32)
    for feature in cat_features:
        # cat encoder will give us total classes and use the size (it's 0 index for encoded values) , size will be used to fill mask
        cat_mask_filler = [len(cat_encoders[feature].classes_)]*(max_sequence_length-current_sequence_length)
        # feature_tensors[feature] = np.concatenate([cat_encoders[feature].transform(customer_df[feature].to_numpy()),cat_mask_filler]).astype(np.int32)
        feature_tensors.append(np.concatenate([cat_encoders[feature].transform(customer_df[feature].to_numpy()),cat_mask_filler]).astype(np.float32))
        feature_tensors.append(np.concatenate([(1-customer_df[feature+"_nan_mask"]).to_numpy(),cat_mask_filler]).astype(np.float32))
    for feature in numeric_features:
        # StandardScaler is meant to be used with multiple features :-X , need to reshape it to use with single feature
        feature_tensors.append(np.concatenate([standard_scalers[feature].transform(customer_df[feature].to_numpy().reshape(-1,1)).reshape(-1), [0]*(max_sequence_length-current_sequence_length)]).astype(np.float32))
        feature_tensors.append(np.concatenate([(1-customer_df[feature+"_nan_mask"]).to_numpy(), [-1]*(max_sequence_length-current_sequence_length)]).astype(np.float32))
    # last feature is the number of records per customer feature; a bit hindsight for the model
    validdatamask = (sequence_mask*-1)+1 #reverse the seq mask
    total_cutomerrows_feature = validdatamask*current_sequence_length/13.0
    feature_tensors.append(total_cutomerrows_feature)
    months_insystem_feature = np.cumsum(validdatamask) / 13.0
    feature_tensors.append(months_insystem_feature)
    # sequence_features_positional = np.stack([total_cutomerrows_feature,months_insystem_feature])

    customer_ID_features = {}
    customer_ID_features['sequence_features'] = np.stack(feature_tensors, axis=1)
    # customer_ID_features['sequence_features_nan_mask'] = np.stack(nan_tensors, axis=1)
    customer_ID_features['sequence_mask'] = sequence_mask
    return customer_ID_features

def get_customer_features_from_dataset(datasetfile, customer_id, cat_encoders=None, standard_scalers=None):
    if cat_encoders == None:
        cat_encoders = get_generate_cat_encoders()
    if standard_scalers == None:
        standard_scalers = get_generate_standard_scalers()

    customer_df = pd.read_parquet(os.path.join(BS_PATH,f'{datasetfile}.parquet'), filters=[('customer_ID', '=', customer_id)])
    
    customer_features = create_customer_id_sequence_features(customer_df, cat_encoders, standard_scalers, max_sequence_length)
    
    del customer_df
    return customer_features
    return customer_features

def get_all_customer_features_from_dataset_chunk(datasetfile, cat_encoders=None, standard_scalers=None):
    if cat_encoders == None:
        cat_encoders = get_generate_cat_encoders()
    if standard_scalers == None:
        standard_scalers = get_generate_standard_scalers()
    customers_df = pd.read_parquet(os.path.join(BS_PATH,f'{datasetfile}.parquet'))
    idx = 0
    customer_id_data_dump = {}
    for customer_id, customer_df in customers_df.groupby('customer_ID'):
        idx += 1
        customer_features = create_customer_id_sequence_features(customer_df, cat_encoders, standard_scalers, max_sequence_length)
        customer_id_data_dump[customer_id] = customer_features
    return customer_id_data_dump

def load_pickle(picklepath):
    if os.path.isfile(picklepath):
        with open(picklepath, 'rb') as picklefilehandle:
            return  pickle.load(picklefilehandle)
def save_pickle(picklepath, data):
    with open(picklepath, 'wb') as picklefilehandle:
        pickle.dump(data, picklefilehandle, protocol=pickle.HIGHEST_PROTOCOL)

def get_generate_cat_encoders():
    picklepath = os.path.join(BS_PATH,'cat_encoders.pickle')
    cat_encoders = load_pickle(picklepath=picklepath)
    if cat_encoders is not None:
        return cat_encoders
    df_train = load_prepare_amex_dataset('train_data_emp1_nn')
    df_test = load_prepare_amex_dataset('test_data_emp1_nn')
    cat_encoders = {}
    for cfet in cat_features:
        le = LabelEncoder()
        le.fit(df_train[cfet].tolist() + df_test[cfet].tolist())
        cat_encoders[cfet] = le
    
    with open(picklepath, 'wb') as picklefilehandle:
        pickle.dump(cat_encoders, picklefilehandle)
    return cat_encoders


def get_generate_standard_scalers():
    picklepath = os.path.join(BS_PATH,'standard_scalers.pickle')
    standard_scalers = load_pickle(picklepath=picklepath)
    if standard_scalers is not None:
        return standard_scalers
    df_train = load_prepare_amex_dataset('train_data_emp1_nn')
    df_test = load_prepare_amex_dataset('test_data_emp1_nn')
    standard_scalers = {}
    for nfet in numeric_features:
        sc = StandardScaler()
        sc.fit(np.concatenate([df_train[nfet].values, df_test[nfet].values]).reshape(-1,1))
        standard_scalers[nfet] = sc
    
    with open(picklepath, 'wb') as picklefilehandle:
        pickle.dump(standard_scalers, picklefilehandle)
    return standard_scalers

def get_generate_chunked_datasets(dataset_name, number_of_chunks):
    
    partfilenames = [f"{dataset_name}_data_emp1_nn_chunk_{idx}" for idx in range(0, number_of_chunks)]
    customer_id_to_dataset_filename = f"{dataset_name}_customer_ID_to_dataset_chunk_file"
    missing = False
    for partfile in partfilenames:
        # even if one file missing then missing will stay True
        missing = missing or not os.path.isfile(os.path.join(BS_PATH, partfile+".parquet"))
    
    if not missing:
        # we'll use load_prepare_amex_dataset to load the data so extension is not needed
        return load_prepare_amex_dataset(customer_id_to_dataset_filename)
    
    df = load_prepare_amex_dataset(f'{dataset_name}_data_emp1_nn')
    df_nan = load_prepare_amex_dataset(f'{dataset_name}_data_nan_mask').rename(lambda x: x+"_nan_mask" if x!="customer_ID" else x, axis=1)
    del df_nan['customer_ID']
    df = pd.concat([df, df_nan], axis=1)
    cid = pd.Categorical(df['customer_ID'], ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
    breakpoints = []
    custlastindex = df.loc[last].index
    customer_id_to_datasetfile_map = []
    for idx in range(1, number_of_chunks):
        breakpoints.append(custlastindex[(idx)*(len(custlastindex)//number_of_chunks)])
    parts = []
    prevbreakpoint = None
    for idx, breakpoint in enumerate(breakpoints):
        if idx == 0:
            parts.append(df.iloc[:breakpoint+1])
            prevbreakpoint = breakpoint+1
        else:
            parts.append(df.iloc[prevbreakpoint:breakpoint+1])
            prevbreakpoint = breakpoint+1
        if idx == (len(breakpoints)-1):
            parts.append(df.iloc[breakpoint+1:])
    
    for idx, part in enumerate(parts):
        part.to_parquet(os.path.join(BS_PATH, f"{dataset_name}_data_emp1_nn_chunk_{idx}.parquet"), index=False)
        for custid in part['customer_ID'].unique().tolist():
            customer_id_to_datasetfile_map.append({"customer_ID":custid, "dataset_file_name":f"{dataset_name}_data_emp1_nn_chunk_{idx}"})
    customer_id_to_chunks = pd.DataFrame(customer_id_to_datasetfile_map)
    customer_id_to_chunks.to_parquet(os.path.join(BS_PATH,customer_id_to_dataset_filename+'.parquet'), index=False)
    return customer_id_to_chunks

cat_encoders = None
standard_scalers = None
customer_id_data_dump = None
def non_lazy_loader_initializer(cid_data_dump):
    global cat_encoders
    global customer_id_data_dump
    global standard_scalers

    customer_id_data_dump = cid_data_dump
    cat_encoders = get_generate_cat_encoders()
    standard_scalers = get_generate_standard_scalers()

def non_lazy_loader_worker(datasetfile):
    global cat_encoders
    print("STARTED WORKER ", datasetfile)
    global customer_id_data_dump
    picklepath = os.path.join(BS_PATH, datasetfile+".pickle")
    if not os.path.isfile(picklepath):
        processed_customers = get_all_customer_features_from_dataset_chunk(datasetfile, cat_encoders, standard_scalers)
        print("Updating customer_id_data_dump")
        save_pickle(picklepath, processed_customers)
    return datasetfile
  
class AmexDefaultBSDataset(Dataset):
    def __init__(self, mode="train", lazy_load=True, total_chunks = 128, device="cpu"):
        self.total_chunks = total_chunks
        self.device = device
        self.mode = mode
        # NOTE: to speed up use_workers on DataLoader this init method better be light; so we cache the needed stuff in advance
        customer_id_df_file_name = f'{mode}_data_emp1_nn_chunk_bsdatasetinit_{total_chunks}'
        customer_id_df_file_path = os.path.join(BS_PATH, f'{mode}_data_emp1_nn_chunk_bsdatasetinit_{total_chunks}.parquet')
        if not os.path.isfile(customer_id_df_file_path):
            dataset_path = os.path.join(BS_PATH,f'{mode}_data_emp1_nn.parquet')
            customer_id_df = pd.read_parquet(dataset_path, columns=['customer_ID'])
            cid = pd.Categorical(customer_id_df['customer_ID'], ordered=True)
            last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
            customer_id_df = customer_id_df.loc[last].reset_index(drop=True)
            customer_chunk_files = get_generate_chunked_datasets(mode, total_chunks)
            customer_id_df = customer_id_df.merge(customer_chunk_files, on='customer_ID', how='left')
            if mode == "test":
                customer_id_df['target'] = 0
            elif mode == "train":
                train_labels = load_prepare_amex_dataset("train_labels")
                customer_id_df = customer_id_df.merge(train_labels, on='customer_ID', how="left")
            customer_id_df.to_parquet(customer_id_df_file_path, index=False)
            self.customer_id_df = customer_id_df.to_dict('records')
        else:
            print("LOAD EAZZZZZ")
            customer_id_df = load_prepare_amex_dataset(customer_id_df_file_name)
            self.customer_id_df = customer_id_df.to_dict('records')
        # customer_ID, dataset_file_name
        
        self.cat_encoders = get_generate_cat_encoders()
        self.customer_id_feature_cache = {}
        if not lazy_load:
            self.non_lazy_load()
    
    def get_feature_cache(self):
        return self.customer_id_feature_cache
    
    def set_feature_cache(self, customer_id_feature_cache):
        self.customer_id_feature_cache = customer_id_feature_cache
    
    def non_lazy_load(self):
        print("NON LAZY LOAD")
        dataset_file_names = list(set([x['dataset_file_name'] for x in self.customer_id_df]))
        print(dataset_file_names)
        with Pool(processes=16, initializer=non_lazy_loader_initializer, initargs=[customer_id_data_dump]) as mpool:
            finaldatafiles = mpool.map(non_lazy_loader_worker, dataset_file_names)
        for finaldatafile in finaldatafiles:
            picklepath = os.path.join(BS_PATH, finaldatafile+".pickle")
            
            featuredict = load_pickle(picklepath)
            # for customer,custfeatures in featuredict.items():
            #     for k,v in custfeatures['sequence_features'].items():
            #         if k in cat_features:
            #             custfeatures[k] = torch.tensor(v, dtype=torch.int8)
                    # if k in numeric_features:
                    #     custfeatures[k] = torch.tensor(v, dtype=torch.float32)

            
            self.customer_id_feature_cache.update(featuredict)
            
            del featuredict
            
            # for customerid, customerfeatures in featuredict.items():
            #     match = filter(lambda x: x['customer_ID']==customerid, self.customer_id_df)
            #     match[0]['features'] = customerfeatures
        

    @staticmethod
    def get_weighted_random_sampler(indices=None):
        train_labels = load_prepare_amex_dataset("train_labels")
        # we have class 0 and 1, 1 class is 25% ish, so to make them equal we turn 0,1 to 0,0.5 then 0.25,0.75 so that class 1 is 75% scale while class 0 is 25% scale

        # we set replacement to False so that we can pass through every samples in one epoch; we also now have to reduce length by 4 otherwise when all 1s are used then only 0s will be left as target
        thetarget = None
        if indices is None:
            thetarget = train_labels['target']
        else:
            thetarget = train_labels['target'].loc[indices]
        sampler = WeightedRandomSampler((thetarget*0.5)+0.25, len(thetarget)//4, replacement=False)
        del train_labels
        return sampler
    
    def cache_all(self):
        pass

    def __len__(self):
        return len(self.customer_id_df)
    
    def get_prepare_item(self, idx):
        customer_row = self.customer_id_df[idx]
        # customer_id = therow['customer_id']
        
        
        customerset = get_customer_features_from_dataset(customer_row['dataset_file_name'], customer_row['customer_ID'], cat_encoders=self.cat_encoders)
        # features = {}
        # for k,v in customerset['sequence_features'].items():
        #     features[k] = v
        # features['sequence_mask'] = customerset['sequence_mask']
        # print(customerset['sequence_features'].shape)
        return customerset

    def mutate_x(self, X):
        sequence_features = X['sequence_features']
        sequence_mask = X['sequence_mask']

    
    def __getitem__(self, idx):
        if self.customer_id_df[idx]['customer_ID'] in self.customer_id_feature_cache:
            X = self.customer_id_feature_cache[self.customer_id_df[idx]['customer_ID']]
           
        else:
            X = self.get_prepare_item(idx)
            # self.feature_cache[idx] = X
        # X = copy.copy(X)
        # for k,v in X['sequence_features'].items():
        #     if k in cat_features:
        #         X['sequence_features'][k] = torch.tensor(v, dtype=torch.int32, device=self.device)
        #     if k in numeric_features:
        #         X['sequence_features'][k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        X['customer_ID'] = self.customer_id_df[idx]['customer_ID']
        y = self.customer_id_df[idx]['target']
        return X, y
    

if __name__ == "__main__":
    print("BS!")
    # DO THIS TO NOT CRASH everything, we need those encoders + scalers to be present or else each worker will try to calculate it
    # get_generate_cat_encoders()
    # get_generate_standard_scalers()
    dataset = AmexDefaultBSDataset(mode="train", total_chunks=128, lazy_load=True)
    print(dataset[0])
    print("----------")
    for x in dataset[0][0]['sequence_features']:
        print(x[35])
    # mode = 'train'
    # total_chunks = 128
    # customer_id_df_file_name = f'{mode}_data_emp1_nn_chunk_bsdatasetinit_{total_chunks}'
    # customer_id_df_file_path = os.path.join(BS_PATH, f'{mode}_data_emp1_nn_chunk_bsdatasetinit_{total_chunks}.parquet')
    # customer_id_df = load_prepare_amex_dataset(customer_id_df_file_name)
    # dataset_file_names = customer_id_df['dataset_file_name'].unique().tolist()
    
    
    
