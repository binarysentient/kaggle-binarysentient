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

def generate_interval_features_ohlc(source_df, source_key, interval_key, intervals):
    """`interval_key` needs to be present in book_df"""
    
#     source_df = source_df[source_df[interval_key]<3].copy()
    aggdf =  source_df[[interval_key,source_key]].groupby(interval_key)[source_key].agg('ohlc')
    
    aggdf = aggdf.reindex([idx for idx in range(0,int(intervals))],method='nearest')
    
    features = aggdf.stack().tolist()
    
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
#         trade_df = pd.read_parquet(os.path.join(data_directory, f"trade_{mode}.parquet", f"stock_id={main_stock_id}"))
        book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1'])/(book_df['bid_size1'] + book_df['ask_size1'])
        book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
#         book_df['directional_volume1'] = book_df['bid_size1'] - book_df['ask_size1']
#             book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        #NOTE: use wap1 ; until we figure out in 01. study which price wap1 closely resembles the trade price, or maybe wap1&wap2 mean
        book_df['log_return1'] = book_df.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff())
        book_df['log_return2'] = book_df.groupby('time_id')['wap2'].apply(lambda x: np.log(x).diff())
#         trade_df['log_return'] = book_df.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff())

#         book_df['seconds_in_bucket_120s_groupkey'] = (book_df['seconds_in_bucket']/120).astype(int)
#         book_df['seconds_in_bucket_2s_groupkey'] = (book_df['seconds_in_bucket']/2).astype(int)
        book_df['seconds_in_bucket_1s_groupkey'] = (book_df['seconds_in_bucket']/1).astype(int)
#         trade_df['seconds_in_bucket_30s_groupkey'] = (trade_df['seconds_in_bucket']/30).astype(int)

#                 print(book_df)
        # ACTUAL FEATURES HERE!
        
        for groupkey, groupdf in book_df.groupby('time_id'):
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
                
#             feature_map[rowid]['book_realized_volatility'] = generate_realized_volatility_df(groupdf)
            
#             book_wap1_2s = generate_interval_features(groupdf, 'wap1', 'seconds_in_bucket_2s_groupkey', 600/2)
            book_wap1_1s = generate_interval_features(groupdf, 'wap1', 'seconds_in_bucket_1s_groupkey', 600/1)
            book_wap2_1s = generate_interval_features(groupdf, 'wap2', 'seconds_in_bucket_1s_groupkey', 600/1)
#             feature_map[rowid]['book_wap1_2s'] = book_wap1_2s
#             feature_map[rowid]['log_return1_1s'] = generate_interval_features(groupdf, 'log_return', 'seconds_in_bucket_1s_groupkey', 600/1, function='sum')
            feature_map[rowid]['log_return1_1s'] = np.log(pd.Series(book_wap1_1s)).diff().fillna(method='bfill').tolist()
            feature_map[rowid]['log_return2_1s'] = np.log(pd.Series(book_wap2_1s)).diff().fillna(method='bfill').tolist()
#             feature_map[rowid]['book_directional_volume1_1s'] = generate_interval_features(groupdf, 'directional_volume1', 'seconds_in_bucket_1s_groupkey', 600/1)
#             feature_map[rowid]['book_wap1_30s_interval'] = generate_interval_features(groupdf, 'wap1', 'seconds_in_bucket_30s_groupkey', 600/30, function='mean')
#             feature_map[rowid]['book_directional_volume1_30s_interval'] = generate_interval_features(groupdf, 'directional_volume1', 'seconds_in_bucket_30s_groupkey', 600/30, function='mean')
            
#         for groupkey, groupdf in trade_df.groupby('time_id'):
#             rowid = get_row_id(main_stock_id, groupkey)
#             if rowid not in feature_map:
#                 feature_map[rowid] = {}
                
#             feature_map[rowid]['trade_realized_volatility'] = generate_realized_volatility_df(groupdf)
#             feature_map[rowid]['trade_price_30s_interval'] = generate_interval_features(groupdf, 'price', 'seconds_in_bucket_30s_groupkey', 600/30, function='mean')
#             feature_map[rowid]['trade_volume_30s_interval'] = generate_interval_features(groupdf, 'size', 'seconds_in_bucket_30s_groupkey', 600/30, function='sum')
        
        return feature_map
    

if __name__ == "__main__":
    # This code is to test while I work on generating features
    DATA_DIRECTORY = os.path.join("..","input","optiver-realized-volatility-prediction")
    if os.path.exists(DATA_DIRECTORY):
#         print(pd.read_csv(os.path.join(DATA_DIRECTORY,'train.csv')).loc[66123])
#         input("book next")
#         book_df = pd.read_parquet(os.path.join(DATA_DIRECTORY, f"book_train.parquet", f"stock_id=18"))
#         print(book_df[book_df['time_id']==8524])
#         input("trade next")
#         trade_df = pd.read_parquet(os.path.join(DATA_DIRECTORY, f"trade_train.parquet", f"stock_id=18"))
#         print(trade_df[trade_df['time_id']==8524])
#         input("features next")
        traindf = pd.read_csv(os.path.join(DATA_DIRECTORY,'train.csv'))
        traindf = traindf[traindf['stock_id']==18]
        totalfeatures = {}
        for stock_id in range(125):
            import time
            stime = time.time()
            features_dict = get_features_map_for_stock(DATA_DIRECTORY, "train", stock_id)
            print("Time takeN:",(time.time()-stime))
    #         input()
#             continue
            for k,v in features_dict.items():
                totalfeatures[k] = v
                continue
                filterdf = traindf[traindf['time_id'] == int(k.split('-')[1])]
                print(filterdf)
                print("curr", v['book_realized_volatility'])
    #             print(v['book_wap1_ohlc_30s'])
                print('curr1s',np.sqrt(np.sum(np.log(pd.Series(v['book_wap1_1s'][::])).diff()**2)))
    #             print(pd.Series(v['log_return1_1s'][::]).head())
    #             print("sum:", np.sum(pd.Series(v['log_return1_1s'][::])**2))
                print('logret1s',np.sqrt(np.sum(pd.Series(v['log_return1_1s'][::])**2)))
                input()
#             print(k)
#             print(v)
            
            