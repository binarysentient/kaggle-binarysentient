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
BS_PATH = os.path.join(TEMP_PATH,BS_DATASET)

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
    parquetfile = os.path.join(INPUT_PATH, BS_DATASET, f"{file_without_extension}.parquet")
    if not os.path.isfile(parquetfile):
        # check temp location/BS path
        parquetfile = os.path.join(BS_PATH, f"{file_without_extension}.parquet")
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
        df.to_parquet(os.path.join(BS_PATH, f"{file_without_extension}.parquet"))
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
        
        impute_numeric_local_aggregations = ['mean','linear_interpolate','nearest_interpolate','polynomial_interpolate'] # 'min','max','first',
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
                    # TODO: feature importance based fallback, fallbcak to global max score feature for each feature
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = global_stats_features_df.loc['mean']
                    continue

                if theagg == "linear_interpolate":
                    interpolated = groupdf[feature_name].interpolate(method='linear', limit_direction="both", limit=13)
                    interpolated = interpolated.ffill()
                    interpolated = interpolated.bfill()
                    variants_df.loc[nanindex, f"{feature_name}_local_{theagg}"] = interpolated.loc[nanindex]
                elif theagg == "polynomial_interpolate":
                    interpolated = groupdf[feature_name].interpolate(method='polynomial', order=2, limit_direction='both', limit=13)
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


## TODO: FAST AMEX implementation is not accurate; convert the dataframe accurate version to directly work with numpy arrays
# @yunchonggan's fast metric implementation
# From https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def fast_amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

# Need Lightgbm supported eval metric
#Custom eval function expects a callable with following signatures: func(y_true, y_pred), func(y_true, y_pred, weight) or func(y_true, y_pred, weight, group) and returns (eval_name, eval_result, is_higher_better) or list of (eval_name, eval_result, is_higher_better):
def lgbm_eval_metric_amex(y_true, y_pred):
    amex_metric = fast_amex_metric(y_true, y_pred)
    return ('amex', amex_metric, True)

def create_lgbm_model_with_config(random_state=1, n_estimators=1200, importance_type=None, early_stopping_rounds=None):
    """
    Creates model with our desired and some default hyper params
    importance_type: split|gain
    """
    if early_stopping_rounds is None:
        early_stopping_rounds = n_estimators//10
    return LGBMClassifier(n_estimators=n_estimators,
                          learning_rate=0.03, reg_lambda=50,
                          min_child_samples=2400,
                          num_leaves=95,
                          colsample_bytree=0.19,early_stopping_rounds=early_stopping_rounds,
                          max_bins=511, random_state=random_state, importance_type=importance_type)

class VariantScoreTracker:
    def __init__(self):
        self.fold_score_track = {}

    def track_score(self, key, score):
        key = str(key)
        if key not in self.fold_score_track:
            self.fold_score_track[key] = []
        self.fold_score_track[key].append(score)

    def show_score(self, key):
        key = str(key)
        # display(HTML(f"<h3>{key} OVERALL SCORE : {np.mean(self.fold_score_track[key])*100:0.3f}</h3>"))
        print("---------------")
        print(f"{key} OVERALL SCORE : {np.mean(self.fold_score_track[key])*100:0.3f}")

    def get_score(self, key):
        return np.mean(self.fold_score_track[key])

    def get_score_all_variants(self, key):
        """
        Get scores for all variants of given feature
        output type:``` [{feature_variant':k, 'score':x}] ```
        """
        # return {k:np.mean(self.fold_score_track[key]) for k in [k for k in self.fold_score_track.keys() if key in k]}
        return [{'feature_variant':k, 'score':np.mean(self.fold_score_track[key])} for k in [k for k in self.fold_score_track.keys() if key in k]]
    
def generate_variant_scores(feature_original):
    score_tracker = VariantScoreTracker()
    
    variant_df = load_prepare_amex_dataset(f"train_data_{feature_original}")
    df_train_last1 = variant_df.groupby('customer_ID').last()
    df_train_last1 = df_train_last1.reset_index()
    
    del variant_df
    
    train_label_df = load_prepare_amex_dataset('train_labels')
    df_train_wt = pd.merge(df_train_last1, train_label_df, how='inner', on = 'customer_ID')#.reset_index()
    del train_label_df
    del df_train_last1
    features_x = [f for f in df_train_wt.columns if f != 'customer_ID' and f != 'target' and f!='S_2']
    feature_y = 'target'
    df_x = df_train_wt[features_x]
    df_y = df_train_wt[feature_y]
    
    total_splits = 5
    kf = StratifiedKFold(n_splits=total_splits, shuffle=True)
    fold_scores = []
    
    # NOT TOO SURE about feature_importances when all features are variants!! it's non decisive
    for fold, (idx_train, idx_dev) in enumerate(kf.split(df_x, df_y)):
        print(f"------ FOLD: {fold+1} of {total_splits} -------")
        train_x, dev_x, train_y, dev_y, model = None, None, None, None, None
        start_time = datetime.now()
        
        for ftx in [[solox] for solox in features_x]:
            # display(HTML(f"<h2>{ftx}</h2>"))
            X_tr = df_train_wt.iloc[idx_train][ftx]
            X_va = df_train_wt.iloc[idx_dev][ftx]
            y_tr = df_train_wt[feature_y][idx_train].values
            y_va = df_train_wt[feature_y][idx_dev].values
            # for importance_type in ['split','gain']:
            model = create_lgbm_model_with_config(importance_type='gain')
            model.fit(X_tr, y_tr,
                          eval_set = [(X_va, y_va)], 
                          eval_metric=[lgbm_eval_metric_amex],
                          callbacks=[log_evaluation(100)])

            y_va_pred = model.predict_proba(X_va, raw_score=True)
            score = fast_amex_metric(y_va, y_va_pred)
            n_trees = model.best_iteration_
            if n_trees is None: n_trees = model.n_estimators
            # print("-------------------------------------")
            # display(HTML(f"<h3>Fold {fold+1} | {str(datetime.now() - start_time)[-12:-7]} |"
            #       f" {n_trees:5} trees |"
            #       f"                Score = {score:.5f} | Features: {ftx}</h3>"))
            # print("-------------------------------------")
            
            score_tracker.track_score(ftx[0], score)
                
    for ftx in [solox for solox in features_x]:
        score_tracker.show_score(ftx)
        
    del df_train_wt
    gc.collect()
    
    return score_tracker.get_score_all_variants(feature_original)

# TODO: TRAIN+TEST mega imputation, and learning based imputation
if __name__ == "__main__":
    
    
    # STAGE 0: Figure out what features are missing and generate feature variants
    
    populate_dataset = "train_data"
    df_train = load_prepare_amex_dataset(f"{populate_dataset}")
    
    missing_counts_df = df_train.isna().sum().reset_index()
    missing_counts_df = missing_counts_df.rename({'index':'columnname', 0:'nan_count'}, axis=1)
    missing_features_with_count_list = [x for x in missing_counts_df.to_dict('records') if x['columnname'].startswith('') and x['nan_count']>0]
    features_missing_values_count_map = {x['columnname']:x['nan_count'] for x in missing_counts_df.to_dict('records')}
    
    #lowest missing to highest missing;
    missing_values_features = [x['columnname'] for x in sorted(missing_features_with_count_list, key=lambda x: x['nan_count'], reverse=False)]
    
    
    # Check howmany have we already seeded!
    
    # Stage 0: Generate all feature variants for both test and train!
    # TODO: make this step skippable if we already have the variants generated
    # del df_train
    # gc.collect()
    # with Pool(processes=13, initializer=the_threadpool_initializer, initargs=[populate_dataset]) as p:
    #     combined_output = p.map(the_threadpool_worker, missing_values_features)
    #     print("FINAL DONE:", combined_output)
    
    
    # Stage 1: figure out which variant works best (this includes original with null features); and create empowerment level 1 dataframes
    
    # TODO: The LightGBM models to create score per feature variant to see it's impact on Target.; higher the score better the variant.
    #       We compare/pit all the variants one by one on the Target and store it
    # TO FORCE STAGE 1
    FORCE_STAGE_1 = False
    VARIANT_SCORES_EMP1_FILE_PATH = os.path.join(BS_PATH, f"variant_scores_emp1.parquet")
    if FORCE_STAGE_1:
        if os.path.isfile(VARIANT_SCORES_EMP1_FILE_PATH):
            os.remove(VARIANT_SCORES_EMP1_FILE_PATH)
    
    variant_scores_df = load_prepare_amex_dataset(f"variant_scores_emp1")
    done_features = []
    if variant_scores_df is not None:
        # columns: feature_variant, score
        done_features = variant_scores_df['feature_variant'].tolist()
        print("donefeatrure", done_features)
    for missing_value_feature in missing_values_features:
        if missing_value_feature in done_features:
            print("SKIPPING ", missing_value_feature)
            continue
        scores = generate_variant_scores(missing_value_feature)
        newdf = pd.DataFrame(scores)
        if variant_scores_df is None:
            newdf.to_parquet(VARIANT_SCORES_EMP1_FILE_PATH, index=False)
        else:
            variant_scores_df = pd.concat([variant_scores_df, newdf])
            variant_scores_df.to_parquet(VARIANT_SCORES_EMP1_FILE_PATH, index=False)
            done_features = variant_scores_df['feature_variant'].tolist()
        print(scores)
    # TODO: once we have the scores for all variant ready; Choose the best variants and create the {train|test}_data_emp1.parquet dataset
    
    
    
    # Stage 2: use that emp1 as base and now train the models to predict the missing values; Combine train

