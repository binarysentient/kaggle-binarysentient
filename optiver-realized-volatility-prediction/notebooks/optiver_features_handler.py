# We have to put this logic here to leverage python multiprocessing in windows. It's all tedious really :-/
# IMPORTANT: never import torch here; it gives off pagefile too small error when number of workers are high

import pandas as pd
import os
import numpy as np



def get_row_id(stock_id, time_id):
        return f"{stock_id:.0f}-{time_id:.0f}"
    
def generate_realized_volatility_df(df):
    """ expects the `log_return` to be present"""
    return np.sqrt(np.sum(df['log_return']**2))

def generate_interval_features(source_df, source_key, interval_key, intervals, function='mean'):
    """`interval_key` needs to be present in book_df"""
    
    aggdf = source_df[[interval_key,source_key]].groupby(interval_key)[source_key].agg(function)
    aggdf = aggdf.interpolate(method='nearest')
    aggdf = aggdf.reindex([idx for idx in range(0,int(intervals))],method='nearest')
    
    features = aggdf.tolist()
    
    
    return features

def generate_interval_group_features(source_df, interval_key, intervals, function={}, column_fill_map={}):
    """ we assume source_df only has neccessary key which has function map
    `func`: dictionary of column:function
    `interval_key`: needs to be present in book_df
    `colum_fill_map`: interpolate|zero
    """
    
    aggdf = source_df.groupby(interval_key).agg(function)    
    aggdf = aggdf.reindex([idx for idx in range(0,int(intervals))])
    for col in aggdf.columns:
        if col not in column_fill_map:
            column_fill_map[col] = 'interpolate'
    
    for key,val in column_fill_map.items():
        if val == 'zero' or val == 0:
            aggdf[key] = aggdf[key].fillna(0)
        else:
            aggdf[key] = aggdf[key].interpolate(method='linear', limit_direction='both')
            
#     print(aggdf)
#     input()
    return aggdf

def generate_interval_features_ohlc(source_df, source_key, interval_key, intervals):
    """`interval_key` needs to be present in book_df"""
    
#     source_df = source_df[source_df[interval_key]<3].copy()
    aggdf =  source_df[[interval_key,source_key]].groupby(interval_key)[source_key].agg('ohlc')
    
    aggdf = aggdf.reindex([idx for idx in range(0,int(intervals))],method='nearest')
    
    features = aggdf.stack().copy().tolist()
    
    if len(features) != int(intervals)*4:
        print("ABOMINATION!!!!!!!", source_df)
    
    return features
    
    

def get_features_map_for_stock(data_directory, mode, main_stock_id):
        """gets the `stock_id-row_id` wise feature map
        `data_directory`: is where the train.csv and other parquet folders are present
        `mode`: train|test
        `main_stock_id`: the stock id! zlul
        """
        feature_map = {}
#         print(f"--> {main_stock_id} get_features_map_for_stock")
        book_df = pd.read_parquet(os.path.join(data_directory, f"book_{mode}.parquet", f"stock_id={main_stock_id}"))
        trade_df = pd.read_parquet(os.path.join(data_directory, f"trade_{mode}.parquet", f"stock_id={main_stock_id}"))
        book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1'])/(book_df['bid_size1'] + book_df['ask_size1'])
        book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        book_df['directional_volume1'] = book_df['bid_size1'] - book_df['ask_size1'] + book_df['bid_size2'] - book_df['ask_size2']
#             book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        #NOTE: use wap1 ; until we figure out in 01. study which price wap1 closely resembles the trade price, or maybe wap1&wap2 mean
#         book_df['log_return1'] = book_df.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff())
#         book_df['log_return2'] = book_df.groupby('time_id')['wap2'].apply(lambda x: np.log(x).diff())
#         book_df['logret_bidp_1'] = book_df.groupby('time_id')['bid_price1'].apply(lambda x: np.log(x).diff())
#         book_df['logret_askp_1'] = book_df.groupby('time_id')['ask_price1'].apply(lambda x: np.log(x).diff())
#         book_df['logret_bidp_2'] = book_df.groupby('time_id')['bid_price2'].apply(lambda x: np.log(x).diff())
#         book_df['logret_askp_2'] = book_df.groupby('time_id')['ask_price2'].apply(lambda x: np.log(x).diff())
#         trade_df['log_return'] = book_df.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff())

#         book_df['seconds_in_bucket_120s_groupkey'] = (book_df['seconds_in_bucket']/120).astype(int)
#         book_df['seconds_in_bucket_2s_groupkey'] = (book_df['seconds_in_bucket']/2).astype(int)
        interval_second = 5
        intervals_count = 600/interval_second
    
        book_df['seconds_in_bucket_xs_groupkey'] = (book_df['seconds_in_bucket']/interval_second).astype(int)
        trade_df['seconds_in_bucket_xs_groupkey'] = (trade_df['seconds_in_bucket']/1).astype(int)

#                 print(book_df)
        # ACTUAL FEATURES HERE!
        for groupkey, groupdf in trade_df.groupby('time_id'):
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
            grouped_interval_df = generate_interval_group_features(groupdf[['price','size','order_count','seconds_in_bucket_xs_groupkey']], 'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={'price':'mean','size':'sum','order_count':'sum'}, column_fill_map={'size':'zero','order_count':'zero'})
            feature_map[rowid]['logrett_xs'] = np.log(grouped_interval_df['price']).diff().fillna(method='bfill').tolist()
            feature_map[rowid]['trade_volume_xs'] = grouped_interval_df['size'].tolist()
            feature_map[rowid]['trade_ordercount_xs'] = grouped_interval_df['order_count'].tolist()
            
        for groupkey, groupdf in book_df.groupby('time_id'):
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
                
#             feature_map[rowid]['book_realized_volatility'] = generate_realized_volatility_df(groupdf)
            grouped_interval_df = generate_interval_group_features(groupdf[['wap1','wap2','directional_volume1','seconds_in_bucket_xs_groupkey']], 'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={'wap1':'mean','wap2':'mean','directional_volume1':'sum'}, column_fill_map={'directional_volume1':'zero'})
#             print(grouped_interval_df)
#             input()
#             book_wap1_1s = generate_interval_features(groupdf, 'wap1', 'seconds_in_bucket_1s_groupkey', 600/1)
#             book_wap2_1s = generate_interval_features(groupdf, 'wap2', 'seconds_in_bucket_1s_groupkey', 600/1)
            
#             bidp1_1s = generate_interval_features(groupdf, 'bid_price1', 'seconds_in_bucket_1s_groupkey', 600/1)
#             askp1_1s = generate_interval_features(groupdf, 'ask_price1', 'seconds_in_bucket_1s_groupkey', 600/1)
#             bidp2_1s = generate_interval_features(groupdf, 'bid_price2', 'seconds_in_bucket_1s_groupkey', 600/1)
#             askp2_1s = generate_interval_features(groupdf, 'ask_price2', 'seconds_in_bucket_1s_groupkey', 600/1)
#             [rowid]['book_wap1_2s'] = book_wap1_2s
#             feature_map[rowid]['log_return1_1s'] = generate_interval_features(groupdf, 'log_return', 'seconds_in_bucket_1s_groupkey', 600/1, function='sum')
            feature_map[rowid]['logret1_xs'] = np.log(grouped_interval_df['wap1']).diff().fillna(method='bfill').tolist()
            feature_map[rowid]['logret2_xs'] = np.log(grouped_interval_df['wap2']).diff().fillna(method='bfill').tolist()
            feature_map[rowid]['book_dirvolume_xs'] = grouped_interval_df['directional_volume1'].tolist()
            

        import gc
        del book_df
        del trade_df
        gc.collect()
        return feature_map
    

if __name__ == "__main__":
    # This code is to test while I work on generating features
    DATA_DIRECTORY = os.path.join("..","input","optiver-realized-volatility-prediction")
    if os.path.exists(DATA_DIRECTORY):

        traindf = pd.read_csv(os.path.join(DATA_DIRECTORY,'train.csv'))
        traindf = traindf[traindf['stock_id']==18]
        totalfeatures = {}
        for stock_id in range(125):
            import time
            stime = time.time()
            features_dict = get_features_map_for_stock(DATA_DIRECTORY, "train", stock_id)
            print("Time takeN:",(time.time()-stime))
            
            for rowid,features in features_dict.items():
                print(f"------- {rowid} -------")
                for k,v in features.items():
                    print(k)
                    print(v)
                    input()
            
            