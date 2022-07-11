WHERE_THIS = "local" # local|kaggle

BS_DATASET = "amex-default-prediction-binarysentient"
if WHERE_THIS == "kaggle":
    INPUT_PATH = "/kaggle/input/amex-default-prediction/"
    OUTPUT_PATH = "/kaggle/working/"
    TEMP_PATH = "/kaggle/temp/"
elif WHERE_THIS == "local":
    INPUT_PATH = "input/amex-default-prediction"
    OUTPUT_PATH = "working"
    TEMP_PATH = "temp"

    
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from multiprocessing import Pool
import gc
import os
from datetime import datetime
import torch


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, HTML
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.model_selection import StratifiedKFold
import plotly.express as pltex


object_features = ['customer_ID']
datetime_features = ['S_2']
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
target_features = ['target']
numeric_features = ['R_2', 'S_25', 'D_118', 'B_5', 'D_60', 'B_21', 'D_115', 'S_15', 'D_84', 'D_122', 'B_9', 'B_33', 'D_53', 'R_24', 'D_94', 'D_56', 'D_139', 'R_28', 'B_3', 'S_20', 'B_31', 'D_133', 'B_2', 'S_12', 'D_71', 'D_96', 'S_7', 'D_72', 'B_36', 'B_41', 'S_5', 'D_41', 'R_22', 'R_8', 'D_140', 'D_47', 'D_89', 'P_2', 'R_19', 'D_59', 'B_23', 'S_3', 'D_145', 'D_103', 'B_19', 'R_20', 'D_73', 'D_136', 'D_141', 'D_142', 'B_22', 'D_46', 'B_29', 'B_25', 'D_128', 'B_18', 'D_86', 'D_109', 'B_8', 'B_17', 'R_17', 'B_12', 'D_54', 'D_74', 'S_16', 'B_6', 'D_49', 'D_80', 'S_8', 'B_7', 'D_144', 'B_27', 'B_26', 'R_25', 'R_23', 'D_82', 'D_111', 'B_10', 'D_113', 'R_4', 'D_48', 'R_26', 'S_23', 'B_11', 'D_104', 'D_134', 'D_79', 'P_3', 'D_132', 'D_137', 'D_135', 'B_28', 'D_88', 'S_13', 'D_51', 'D_61', 'D_75', 'D_69', 'R_16', 'S_6', 'S_17', 'D_93', 'B_20', 'D_112', 'D_123', 'D_130', 'B_1', 'D_78', 'D_92', 'S_27', 'D_44', 'B_16', 'R_5', 'D_43', 'S_18', 'B_15', 'D_39', 'D_50', 'D_55', 'S_9', 'D_105', 'D_70', 'R_18', 'D_125', 'D_58', 'S_24', 'D_110', 'D_42', 'R_6', 'D_81', 'R_7', 'D_138', 'D_52', 'R_27', 'D_124', 'D_45', 'D_91', 'D_108', 'S_22', 'B_14', 'D_83', 'R_13', 'D_87', 'S_19', 'D_131', 'R_21', 'B_40', 'R_3', 'D_65', 'B_13', 'D_129', 'D_119', 'B_32', 'R_9', 'B_24', 'D_127', 'D_106', 'D_102', 'R_1', 'R_14', 'B_37', 'D_107', 'R_15', 'R_12', 'P_4', 'R_10', 'customer_ID', 'D_121', 'R_11', 'S_11', 'D_76', 'D_143', 'B_39', 'B_42', 'D_62', 'B_4', 'S_26', 'D_77']
    
def figureout_data_types_from_csv(csv_filepath):
    df_iterator = pd.read_csv(csv_filepath, chunksize=100, low_memory=True)
    df_train_peek = df_iterator.get_chunk(200000)
    train_dtypes_set = set()
    column_dtypes_map = {}
    all_columns = []
    for columnname in df_train_peek.columns:
        # print(columnname, df_train_peek[columnname].dtype)
        all_columns.append(columnname)
        train_dtypes_set.add(df_train_peek[columnname].dtype.name)
        column_dtypes_map[columnname] = df_train_peek[columnname].dtype.name
    del df_iterator
    del df_train_peek
    gc.collect()
    
    numeric_features = list(set(column_dtypes_map.keys()).difference(set(cat_features+datetime_features+object_features)))
    all_features = cat_features + numeric_features
    cat_features_dtypes_map = {x:'category' for x in cat_features}
    for column,dtype in cat_features_dtypes_map.items():
        column_dtypes_map[column] = cat_features_dtypes_map[column]
    for column,dtype in column_dtypes_map.items():
        if dtype == "float64":
            column_dtypes_map[column] = "float32"
        if dtype == "int64": #B_31 is int64
            column_dtypes_map[column] = "float32"
    
    return column_dtypes_map, [x for x in all_columns if x in datetime_features]

def already_known_datatypes():
    dtypesmap = {}
    for fet in numeric_features:
        dtypesmap[fet] = 'float32'
    for fet in cat_features:
        dtypesmap[fet] = 'category'
    datetimefeatures = datetime_features
    return dtypesmap, datetimefeatures
            
def load_csv_with_dtype(csv_filepath):
    dtypemap,datetimecolumns = figureout_data_types_from_csv(csv_filepath)
    return pd.read_csv(csv_filepath, low_memory=True, dtype=dtypemap, parse_dates=datetimecolumns)


def load_prepare_amex_dataset(file_without_extension, load_only_these_columns=None):
    # check input location
    parquetfile = os.path.join(INPUT_PATH,BS_DATASET)+os.sep+f"{file_without_extension}.parquet"
    if not os.path.isfile(parquetfile):
        # check temp location
        parquetfile = os.path.join(TEMP_PATH,BS_DATASET)+os.sep+f"{file_without_extension}.parquet"
    # if either of above 2 has the file then load df and return
    if os.path.isfile(parquetfile):
        # print("FOUND PARQUET")
        if load_only_these_columns:
            return pd.read_parquet(parquetfile, columns=load_only_these_columns)
        return pd.read_parquet(parquetfile)
    if load_only_these_columns is not None:
        # this hack exist only for multiprocessing we dont' wanna create parquet files from subset of big csv
        return None
    # print("CREATING CSV")
    csvfile = os.path.join(INPUT_PATH, f"{file_without_extension}.csv")
    if os.path.isfile(csvfile):
        dtypesmap, datetimecols = already_known_datatypes()
        df = load_csv_with_dtype(csvfile)
        df.to_parquet(os.path.join(TEMP_PATH,BS_DATASET, f"{file_without_extension}.parquet"))
        return df


def generate_impute_variants(df, feature_name):
    
    dtype = df[feature_name].dtype.name
    print("IMPUTING: ", feature_name, " type", dtype)
    if "float" in dtype:
        # print("Feature datatype ", )
        impute_numeric_global_aggregations = ['mean','min','max','median']
        global_stats_features_df = df[feature_name].agg(impute_numeric_global_aggregations)
        
        variants_df = df[['customer_ID',feature_name]].copy()
        
        print("COMPUTING GLOBAL IMPUTATIONS")
        nanindex = variants_df[variants_df[feature_name].isna()].index
        for theagg in impute_numeric_global_aggregations:
            variants_df[f"{feature_name}_global_{theagg}"] = variants_df[feature_name]
            variants_df.loc[nanindex, f"{feature_name}_global_{theagg}"] = global_stats_features_df.loc[theagg]
        
        impute_numeric_local_aggregations = ['mean','linear_interpolate','nearest_interpolate'] # 'min','max','first',
        for theagg in impute_numeric_local_aggregations:
            variants_df[f"{feature_name}_local_{theagg}"] = variants_df[feature_name]
            
        print("COMPUTING GROUPED IMPUTATIONS")
        for groupk, groupdf in variants_df[['customer_ID']+[feature_name]].groupby("customer_ID"):
            nanindex = groupdf[groupdf[feature_name].isna()].index
            notnanindex = groupdf[groupdf[feature_name].notnull()].index
            
            if len(nanindex) == 0:
                continue

            for theagg in impute_numeric_local_aggregations:
                # print("INSIDE: ",theagg)
                if len(notnanindex) == 0:
                    # TODO: figure something out, for now when all none we fill with global mean
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = global_stats_features_df.loc['mean']
                    continue

                if theagg == "linear_interpolate":
                    interpolated = groupdf[feature_name].interpolate(method='linear', limit=13)
                    interpolated = interpolated.ffill()
                    interpolated = interpolated.bfill()
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = interpolated.loc[nanindex]
                elif theagg == "nearest_interpolate":

                    if len(notnanindex) != 1:
                        interpolated = groupdf[feature_name].interpolate(method='nearest',limit_direction='both', limit=13)
                    else:
                        interpolated = groupdf[feature_name]
                    interpolated = interpolated.ffill()
                    interpolated = interpolated.bfill()
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = interpolated.loc[nanindex]
                elif theagg == "first":
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = groupdf[feature_name].iloc[0]
                else:
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = groupdf[feature_name].agg(theagg)
        
        return variants_df

current_dataset = None
def the_threadpool_initializer(dataset):
    global current_dataset 
    current_dataset = dataset
    print("---------> INITIALIZER:", current_dataset)
    current_dataset = dataset
    
def the_threadpool_worker(featurename):
    global current_dataset
    the_df = load_prepare_amex_dataset(f"{current_dataset}",load_only_these_columns=['customer_ID',featurename])
    
    variant_df = generate_impute_variants(the_df, featurename)
    if variant_df is not None:
        print("---> FINISHED FOR ", featurename)
        variant_df.to_parquet(os.path.join(TEMP_PATH,BS_DATASET, f"{current_dataset}_{featurename}.parquet"))
    del the_df
    del variant_df
    gc.collect()
    return featurename

# TODO: TRAIN+TEST mega imputation, and learning based imputation
if __name__ == "__main__":
    populate_dataset = "train_data"
    df_train = load_prepare_amex_dataset(f"{populate_dataset}")
    
    missing_counts_df = df_train.isna().sum().reset_index()
    missing_counts_df = missing_counts_df.rename({'index':'columnname', 0:'nan_count'}, axis=1)
    missing_features_with_count_list = [x for x in missing_counts_df.to_dict('records') if x['columnname'].startswith('') and x['nan_count']>0]
    features_missing_values_count_map = {x['columnname']:x['nan_count'] for x in missing_counts_df.to_dict('records')}
    
    #lowest missing to highest missing;
    missing_values_features = [x['columnname'] for x in sorted(missing_features_with_count_list, key=lambda x: x['nan_count'], reverse=False)]
    
    del df_train
    gc.collect()
    with Pool(processes=13, initializer=the_threadpool_initializer, initargs=[populate_dataset]) as p:
        combined_output = p.map(the_threadpool_worker, missing_values_features)
        print("FINAL DONE:", combined_output)
    
        

