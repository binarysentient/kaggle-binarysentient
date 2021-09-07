# We have to put this logic here to leverage python multiprocessing in windows. It's all tedious really :-/
# IMPORTANT: never import torch here; it gives off pagefile too small error when number of workers are high

import pandas as pd
import os
import numpy as np



def get_row_id(stock_id, time_id):
        return f"{stock_id:.0f}-{time_id:.0f}"
    
def generate_realized_volatility_df(df):
    """ expects the `log_return` to be present"""
    return np.sqrt(np.sum(df['logret1']**2))

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
        interval_second = 2
        intervals_count = 600/interval_second
        
        feature_map = {}
#         print(f"--> {main_stock_id} get_features_map_for_stock")
        book_df = pd.read_parquet(os.path.join(data_directory, f"book_{mode}.parquet", f"stock_id={main_stock_id}"))
        trade_df = pd.read_parquet(os.path.join(data_directory, f"trade_{mode}.parquet", f"stock_id={main_stock_id}"))
        
        book_df['seconds_in_bucket_xs_groupkey'] = (book_df['seconds_in_bucket']/interval_second).astype(int)
        trade_df['seconds_in_bucket_xs_groupkey'] = (trade_df['seconds_in_bucket']/interval_second).astype(int)
        
        book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1'])/(book_df['bid_size1'] + book_df['ask_size1'])
#         book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        
        book_df['logret1'] = np.log(book_df['wap1']).diff().fillna(0)
#         book_df['logret2'] = np.log(book_df['wap2']).diff().fillna(0)
            
#         book_df['wap_1n2'] = (book_df['wap1'] + book_df['wap2'])/2
#         book_df['directional_volume1'] = book_df['ask_size1'] - book_df['bid_size1']
#         book_df['directional_volume2'] = book_df['ask_size2'] - book_df['bid_size2']
        book_df['price_spread1'] = (book_df['ask_price1'] - book_df['bid_price1']) / ((book_df['ask_price1'] + book_df['bid_price1'])/2)
#         book_df['price_spread2'] = (book_df['ask_price2'] - book_df['bid_price2']) / ((book_df['ask_price2'] + book_df['bid_price2'])/2)
        book_df['bid_spread'] = abs(book_df['bid_price1'] - book_df['bid_price2']) / ((book_df['bid_price1'] + book_df['bid_price2'])/2)
        book_df['ask_spread'] = abs(book_df['ask_price1'] - book_df['ask_price2']) / ((book_df['ask_price1'] + book_df['ask_price2'])/2)
        book_df['total_volume'] = book_df['ask_size1'] + book_df['bid_size1'] + book_df['ask_size2'] + book_df['bid_size2']
        book_df['volume_imbalance'] = abs(book_df['ask_size1'] - book_df['bid_size1'] + book_df['ask_size2'] - book_df['bid_size2'])
        
#         book_df['book_money_turnover1'] = book_df[['ask_size1','bid_size1']].min(axis=1) * book_df['wap1']
#         book_df['book_money_turnover2'] = (book_df['ask_size1'] + book_df['bid_size1'] - min(book_df['ask_size1'],book_df['bid_size1'])) * book_df['wap1']
    
        
#         trade_df['trade_money_turnover'] = (trade_df['size'] * trade_df['price'])
        trade_df['logrett'] = np.log(trade_df['price']).diff().fillna(0)
        trade_df['trade_money_turnover_per_order'] = (trade_df['size'] * trade_df['price'] / trade_df['order_count'])
        
#                 print(book_df)
        # ACTUAL FEATURES HERE!
        for groupkey, groupdf in trade_df.groupby('time_id'):
            
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
                # ['trade_money_turnover']
                #'trade_money_turnover':'sum'
                # 'trade_money_turnover':'zero'
            grouped_interval_df = generate_interval_group_features(groupdf[['price','size','order_count','logrett','trade_money_turnover_per_order','seconds_in_bucket_xs_groupkey']], 'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={'price':'mean','size':'sum','order_count':'sum','trade_money_turnover_per_order':'sum','logrett':'sum'}, column_fill_map={'size':'zero','order_count':'zero','trade_money_turnover_per_order':'zero','logrett':'zero'})
            #NOTE: double fillna is important! when whole series is na then 'bfill' won't work
            feature_map[rowid]['logrett_xs'] = grouped_interval_df['logrett'].tolist()
            feature_map[rowid]['trade_volume_xs'] = grouped_interval_df['size'].tolist()
            feature_map[rowid]['trade_ordercount_xs'] = grouped_interval_df['order_count'].tolist()
#             feature_map[rowid]['trade_money_turnover_xs'] = grouped_interval_df['trade_money_turnover'].tolist()
            feature_map[rowid]['trade_money_turnover_per_order_xs'] = grouped_interval_df['trade_money_turnover_per_order'].tolist()
            
            feature_map[rowid]['seconds_in_bucket_xs'] = [(idx*interval_second)+interval_second for idx in range(0,int(intervals_count))]
#             feature_map[rowid]['trade_price_min'] = grouped_interval_df['price'].min()
#             feature_map[rowid]['trade_price_mean'] = grouped_interval_df['price'].mean()
#             feature_map[rowid]['trade_price_max'] = grouped_interval_df['price'].max()

            
        for groupkey, groupdf in book_df.groupby('time_id'):
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
                
#             feature_map[rowid]['book_realized_volatility'] = generate_realized_volatility_df(groupdf)
            # 'wap_1n2', 'directional_volume1', 'directional_volume2', 'book_money_turnover1','price_spread2','wap1','wap2','logret2'
            # 'wap_1n2':'mean', 'directional_volume1':'mean', 'directional_volume2':'mean', 'book_money_turnover1':'sum','price_spread2':'mean','wap1':'mean','wap2':'mean', 'logret2':'sum',
            # 'directional_volume1':'zero','directional_volume2':'zero','book_money_turnover1':'zero', 'logret2':'zero'
            grouped_interval_df = generate_interval_group_features(groupdf[[ 'price_spread1','bid_spread','ask_spread', 'total_volume', 'logret1','volume_imbalance','seconds_in_bucket_xs_groupkey']], 'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={'logret1':'sum',
                                                       'price_spread1':'mean', 'bid_spread':'mean','ask_spread':'mean', 'total_volume':'sum', 'volume_imbalance':'mean',},
                                            column_fill_map={ 'logret1':'zero','total_volume':'zero', 'volume_imbalance':'zero'})


            feature_map[rowid]['logret1_xs'] = grouped_interval_df['logret1'].tolist()
#             feature_map[rowid]['logret2_xs'] = grouped_interval_df['logret2'].tolist()
            
            feature_map[rowid]['book_price_spread1_xs'] = grouped_interval_df['price_spread1'].tolist()
            
            feature_map[rowid]['book_bid_spread_xs'] = grouped_interval_df['bid_spread'].tolist()
            feature_map[rowid]['book_ask_spread_xs'] = grouped_interval_df['ask_spread'].tolist()
            feature_map[rowid]['book_total_volume_xs'] = grouped_interval_df['total_volume'].tolist()
            feature_map[rowid]['book_volume_imbalance_xs'] = grouped_interval_df['volume_imbalance'].tolist()
#             feature_map[rowid]['book_money_turnover1_xs'] = grouped_interval_df['book_money_turnover1'].tolist()
            
         
            
#         import gc
        del book_df
        del trade_df
#         gc.collect()
        return feature_map
    

if __name__ == "__main__":
    # This code is to test while I work on generating features
    DATA_DIRECTORY = os.path.join("..","input","optiver-realized-volatility-prediction")
    if os.path.exists(DATA_DIRECTORY):

        traindf = pd.read_csv(os.path.join(DATA_DIRECTORY,'train.csv'))
        traindf = traindf[traindf['stock_id']==31]
        totalfeatures = {}
        for stock_id in range(125):
            if stock_id != 31:
                continue
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
            
            