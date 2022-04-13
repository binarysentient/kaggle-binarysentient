# We have to put this logic here to leverage python multiprocessing in windows. It's all tedious really :-/
# IMPORTANT: never import torch here; it gives off pagefile too small error when number of workers are high

import pandas as pd
import os
import numpy as np
import types


def get_row_id(stock_id, time_id):
    if type(time_id) != int:
        time_id = int(time_id)
    return f"{stock_id:.0f}-{time_id}"
    

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))



def get_features_map_for_stock(data_directory, mode, main_stock_id):
        """gets the `stock_id-row_id` wise feature map
        `data_directory`: is where the train.csv and other parquet folders are present
        `mode`: train|test
        `main_stock_id`: the stock id! zlul
        """
        interval_second = 5
        intervals_count = 600//interval_second
        
        feature_map = {}
        book_df = pd.read_parquet(os.path.join(data_directory, f"book_{mode}.parquet", f"stock_id={main_stock_id}"))
        trade_df = pd.read_parquet(os.path.join(data_directory, f"trade_{mode}.parquet", f"stock_id={main_stock_id}"))
        book_df['book_seconds_in_bucket'] = book_df['seconds_in_bucket']/book_df['seconds_in_bucket']
        book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1'])/(book_df['bid_size1'] + book_df['ask_size1'])
        book_df['logret1'] = np.log(book_df['wap1']).diff()
        book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        book_df['logret2'] = np.log(book_df['wap2']).diff()
        book_df['wap_balance'] = abs(book_df['wap1'] - book_df['wap2'])
        book_df['logret_bid_price1'] = np.log(book_df['bid_price1']).diff()
        book_df['logret_ask_price1'] = np.log(book_df['ask_price1']).diff()
        book_df['logret_bid_price2'] = np.log(book_df['bid_price2']).diff()
        book_df['logret_ask_price2'] = np.log(book_df['ask_price2']).diff()
        
        book_df['logret_bid_size1'] = np.log(book_df['bid_size1']).diff()
        book_df['logret_ask_size1'] = np.log(book_df['ask_size1']).diff()
        book_df['logret_bid_size2'] = np.log(book_df['bid_size2']).diff()
        book_df['logret_ask_size2'] = np.log(book_df['ask_size2']).diff()
        
        book_df['price_spread1'] = (book_df['ask_price1'] - book_df['bid_price1']) / ((book_df['ask_price1'] + book_df['bid_price1'])/2)
        book_df['bid_spread'] = abs(book_df['bid_price1'] - book_df['bid_price2']) / ((book_df['bid_price1'] + book_df['bid_price2'])/2)
        book_df['ask_spread'] = abs(book_df['ask_price1'] - book_df['ask_price2']) / ((book_df['ask_price1'] + book_df['ask_price2'])/2)
        book_df["bid_ask_spread"] = abs(book_df['bid_spread'] - book_df['ask_spread'])
        book_df['directional_volume1'] = (book_df['bid_size1'] - book_df['ask_size1'])
        book_df['directional_volume2'] = (book_df['bid_size2'] - book_df['ask_size2'])
        book_df['logret_directional_volume1'] = np.log(book_df['directional_volume1'] - book_df['directional_volume1'].min() + 1).diff()
        book_df['logret_directional_volume2'] = np.log(book_df['directional_volume2'] - book_df['directional_volume2'].min() + 1).diff()
        
        book_df['total_volume'] = book_df['ask_size1'] + book_df['bid_size1'] + book_df['ask_size2'] + book_df['bid_size2']
        book_df['volume_imbalance'] = abs(book_df['ask_size1'] - book_df['bid_size1'] + book_df['ask_size2'] - book_df['bid_size2'])
        
        trade_df['trade_money_turnover'] = trade_df['size'] * trade_df['price']
        trade_df['logret_trade_money_turnover'] = np.log(trade_df['size'] * trade_df['price']).diff()
        trade_df['size_tau'] = 1/trade_df['size']
        trade_df['order_count_tau'] = 1/trade_df['order_count']
#         trade_df['trade_money_turnover_per_order'] = (trade_df['size'] * trade_df['price'] / trade_df['order_count'])
        trade_df['logret_price'] = np.log(trade_df['price']).diff()
        trade_df['logret_size'] = np.log(trade_df['size']).diff()
        trade_df['logret_order_count'] = np.log(trade_df['order_count']).diff()
        trade_df['trade_seconds_in_bucket'] = trade_df['seconds_in_bucket']/trade_df['seconds_in_bucket']
        
#         trade_df['trade_tendancy'] = trade_df['logret_price'] * trade_df['size']
        merged_df = book_df.merge(trade_df,how='left',on=['time_id','seconds_in_bucket']).reset_index(drop=False)
        
        merged_df['nwap1'] = (merged_df['ask_price1'] + merged_df['bid_price1'])/2
        merged_df['trade_price_push_on_book'] = (merged_df['price'] - merged_df['nwap1'])/(merged_df['price'] + merged_df['nwap1'])/2
#         merged_df['trade_volume_on_book'] = (merged_df['size']/(merged_df['bid_size1']+merged_df['ask_size1']+merged_df['bid_size2']+merged_df['ask_size2']))
        
        del book_df
        del trade_df
        
        seconds_in_bucket_gropus = [0,300,450]
        overview_aggregations = {
            'book_seconds_in_bucket': ['sum'],
            'trade_seconds_in_bucket': ['sum'],
            'wap1': ['sum', 'std'],
            'wap2': ['sum', 'std'],
            'logret1': [realized_volatility],
            'logret2': [realized_volatility],
            'logret_price': [realized_volatility],
            'wap_balance': ['sum', 'max'],
            'price_spread1': ['sum', 'max'],
            'bid_spread': ['sum', 'max'],
            'ask_spread': ['sum', 'max'],
            'total_volume': ['sum', 'max'],
            'volume_imbalance': ['sum', 'max'],
            "bid_ask_spread": ['sum', 'max'],
            'size':  ['sum', 'max','min'],
            'order_count': ['sum', 'max'],
            'size_tau':  ['sum', 'max','min'],
            'order_count_tau':  ['sum', 'max','min'],
            'trade_money_turnover': ['sum', 'max','min'],
            'directional_volume1': ['sum','max','min'],
            'directional_volume2': ['sum','max','min'],
            'logret_directional_volume1': ['sum','max','min'],
            'logret_directional_volume2': ['sum','max','min'],
            'trade_price_push_on_book': ['sum','max','min']
        }
        for seconds_in_bucket_group in seconds_in_bucket_gropus:
            aggregations = merged_df[merged_df['seconds_in_bucket']>=seconds_in_bucket_group].groupby('time_id').agg(overview_aggregations).reset_index(drop=False)
            aggregations = aggregations.fillna(-0.01)
            aggregations.columns = ['_'.join(col).strip() for col in aggregations.columns.values]
            for idx, row in aggregations.iterrows():
                row = row.to_dict()
    #             print(row)
    #             input()
                time_id = row['time_id_']
    #             print(int(time_id), type(time_id))
    #             input()
                rowid = get_row_id(main_stock_id, time_id)

                if rowid not in feature_map:
                    feature_map[rowid] = {}

                for key, aggs in overview_aggregations.items():
                    for agg in aggs:
                        if isinstance(agg, types.FunctionType):
                            agg = agg.__name__
                        feature_map[rowid][f'{key}_{agg}_{seconds_in_bucket_group}'] = row[f'{key}_{agg}']
            del aggregations
            
        
        
        merged_df['seconds_in_bucket'] = merged_df['seconds_in_bucket'] // interval_second
        
        
        for groupkey, groupdf in merged_df[['time_id','seconds_in_bucket','price']].groupby(['time_id','seconds_in_bucket']).agg('sum').reset_index(drop=False).groupby('time_id'):
#             print(groupkey)
            rowid = get_row_id(main_stock_id, groupkey)

            if rowid not in feature_map:
                feature_map[rowid] = {}

            sequence_length = len(groupdf['seconds_in_bucket'].to_numpy())
            
            groupdf['has_trade_data'] = (~groupdf['price'].isnull()).astype(int)


            feature_map[rowid]['sequence_mask_xs'] = [False]*sequence_length + [True]*(intervals_count-sequence_length)
            feature_map[rowid]['has_trade_data_xs'] = np.concatenate([groupdf['has_trade_data'].to_numpy(),[0]*(intervals_count-sequence_length)])
            feature_map[rowid]['seconds_in_bucket_xs'] = np.concatenate([groupdf['seconds_in_bucket'].to_numpy(),[0]*(intervals_count-sequence_length)])
            
        
        
       
        fetandagglist = [
            (['book_seconds_in_bucket','trade_seconds_in_bucket'],
             ['sum']),
            (['logret_bid_price1','logret_ask_price1','logret_bid_price2','logret_ask_price2','logret_bid_size1', 'logret_ask_size1','logret_bid_size2','logret_ask_size2','logret_price','logret_size','logret_order_count'],
             ['sum']),
#             (['price_spread1','bid_spread','ask_spread','bid_ask_spread'],
#              ['min','max']),
#             (['directional_volume1','directional_volume2', 'total_volume','volume_imbalance','size','size_tau','order_count','order_count_tau','trade_money_turnover','trade_price_push_on_book'],
#              ['sum','max'])
        ]
        feature_agg_map = {}
        currfeaturelist = []
        agg_feature_map = {}
        curragglist = []
        for fetlist, agglist in fetandagglist:
            currfeaturelist += fetlist
            curragglist += agglist
            for fet in fetlist:
                feature_agg_map[fet] = []
                for agg in agglist:
                    feature_agg_map[fet].append(agg)
                    if agg not in agg_feature_map:
                        agg_feature_map[agg] = []
                    agg_feature_map[agg].append(fet)
                    
        currfeaturelist = list(set(currfeaturelist)) 
        curragglist = list(set(curragglist)) 
        
        
        for curragg, currfeaturelist in agg_feature_map.items():
            
            for groupkey, groupdf in merged_df[['time_id','seconds_in_bucket']+currfeaturelist].groupby(['time_id','seconds_in_bucket']).agg(curragg).reset_index(drop=False).groupby('time_id'):
#             print(groupkey)
                rowid = get_row_id(main_stock_id, groupkey)

                if rowid not in feature_map:
                    feature_map[rowid] = {}

                sequence_length = len(groupdf['seconds_in_bucket'].to_numpy())


                for feature_name in currfeaturelist:   
                    nan_replace_val = -0.01
                    if feature_name in ['logret_bid_price1','logret_ask_price1','logret_bid_price2','logret_ask_price2','trade_price_push_on_book','logret_price']:
                        nan_replace_val = 0.0
#                     for curragg in feature_agg_map[feature_name]:
                    feature_map[rowid][f'{feature_name}_{curragg}_xs'] = np.concatenate([
                                                            np.nan_to_num(groupdf[feature_name].to_numpy(),nan=nan_replace_val,neginf=nan_replace_val,posinf=nan_replace_val),
                                                            [0]*(intervals_count-sequence_length)
                                                                              ])
            
#             for feature_name in currfeaturelist:   
#                 nan_replace_val = -0.01
#                 if feature_name in ['logret_bid_price1','logret_ask_price1','logret_bid_price2','logret_ask_price2','trade_price_push_on_book','logret_price']:
#                     nan_replace_val = 0.0
#                 for curragg in feature_agg_map[feature_name]:
#                     feature_map[rowid][f'{feature_name}_{curragg}_xs'] = np.concatenate([
#                                                             np.nan_to_num(groupdf[feature_name][curragg].to_numpy(),nan=nan_replace_val,neginf=nan_replace_val,posinf=nan_replace_val),
#                                                             [0]*(intervals_count-sequence_length)
#                                                                               ])
#         del merge_prepared_df
#                 feature_map[rowid][f'{feature_name}_v_xs'] = np.concatenate([np.nan_to_num(1/(groupdf[feature_name].to_numpy()),nan=-0.01,neginf=-0.01,posinf=-0.01),[0]*(intervals_count-sequence_length)])
#             groupdf = groupdf.fillna(-0.01) 
            # transformer mask ignores the True values,and False remains unchanged
                
#         print(merged_df)
#         input()
     
#         import gc
        del merged_df
#         del merge_prepared_df
#         gc.collect()
        return feature_map
    

def get_features_map_for_stock2(data_directory, mode, main_stock_id):
        """gets the `stock_id-row_id` wise feature map
        `data_directory`: is where the train.csv and other parquet folders are present
        `mode`: train|test
        `main_stock_id`: the stock id! zlul
        """
        interval_second = 10
        intervals_count = 600//interval_second
        
        feature_map = {}
#         print(f"--> {main_stock_id} get_features_map_for_stock")
        book_df = pd.read_parquet(os.path.join(data_directory, f"book_{mode}.parquet", f"stock_id={main_stock_id}"))
    
        trade_df = pd.read_parquet(os.path.join(data_directory, f"trade_{mode}.parquet", f"stock_id={main_stock_id}"))
        
#         book_df['seconds_in_bucket_xs_groupkey'] = (book_df['seconds_in_bucket']/interval_second).astype(int)
#         trade_df['seconds_in_bucket_xs_groupkey'] = (trade_df['seconds_in_bucket']/interval_second).astype(int)
        
        book_df['wap1'] = (book_df['bid_price1'] * book_df['ask_size1'] + book_df['ask_price1'] * book_df['bid_size1'])/(book_df['bid_size1'] + book_df['ask_size1'])
        book_df['wap2'] = (book_df['bid_price2'] * book_df['ask_size2'] + book_df['ask_price2'] * book_df['bid_size2'])/(book_df['bid_size2'] + book_df['ask_size2'])
        book_df['wap_balance'] = abs(book_df['wap1'] - book_df['wap2'])
        book_df['logret1'] = np.log(book_df['wap1']).diff().fillna(0)
#         book_df['logret1_pow2'] = book_df['logret1']**2
#         book_df['logret1_cumsum'] = book_df['wap1'].cumsum()
        book_df['logret2'] = np.log(book_df['wap2']).diff().fillna(0)
#         book_df['logret2_pow2'] = book_df['logret2']**2
            
#         book_df['wap_1n2'] = (book_df['wap1'] + book_df['wap2'])/2
#         book_df['directional_volume1'] = book_df['ask_size1'] - book_df['bid_size1']
#         book_df['directional_volume2'] = book_df['ask_size2'] - book_df['bid_size2']
        book_df['price_spread1'] = (book_df['ask_price1'] - book_df['bid_price1']) / ((book_df['ask_price1'] + book_df['bid_price1'])/2)
#         book_df['price_spread2'] = (book_df['ask_price2'] - book_df['bid_price2']) / ((book_df['ask_price2'] + book_df['bid_price2'])/2)
        book_df['bid_spread'] = abs(book_df['bid_price1'] - book_df['bid_price2']) / ((book_df['bid_price1'] + book_df['bid_price2'])/2)
        book_df['ask_spread'] = abs(book_df['ask_price1'] - book_df['ask_price2']) / ((book_df['ask_price1'] + book_df['ask_price2'])/2)
        book_df["bid_ask_spread"] = abs(book_df['bid_spread'] - book_df['ask_spread'])
        book_df['total_volume'] = book_df['ask_size1'] + book_df['bid_size1'] + book_df['ask_size2'] + book_df['bid_size2']
        book_df['volume_imbalance'] = abs(book_df['ask_size1'] - book_df['bid_size1'] + book_df['ask_size2'] - book_df['bid_size2'])
        
        
        
#         book_df['book_money_turnover_intention1'] = book_df[['ask_size1','bid_size1']].min(axis=1) * book_df['wap1']
#         book_df['book_money_turnover2'] = (book_df['ask_size1'] + book_df['bid_size1'] - min(book_df['ask_size1'],book_df['bid_size1'])) * book_df['wap1']
    
        
        
        trade_df['logrett'] = np.log(trade_df['price']).diff().fillna(0)
#         trade_df['logrett_pow2'] = trade_df['logrett'] ** 2
        trade_df['trade_money_turnover'] = (trade_df['size'] * trade_df['price'])
#         trade_df['trade_money_turnover_per_order'] = (trade_df['size'] * trade_df['price'] / trade_df['order_count'])
        merged_df = book_df.merge(trade_df,how='left',on=['time_id','seconds_in_bucket'])
        
        merged_df['trade_book_price_spread'] = abs(merged_df['wap1']-merged_df['price'])/(merged_df['wap1']+merged_df['price'])
#         merged_df['has_trade_data'] = (~merged_df['trade_book_price_spread'].isnull()).astype(int)
#         merged_df = merged_df.fillna(-0.01)
        
        merged_df['seconds_in_bucket_xs_group'] = merged_df['seconds_in_bucket'] // interval_second
        keys_to_focus = ['wap1','seconds_in_bucket_xs_group','logret1','logret2','logrett',
                             'price_spread1','bid_spread','ask_spread','total_volume','volume_imbalance',
                            'size','order_count','trade_book_price_spread','time_id','wap_balance','trade_money_turnover','bid_ask_spread'] #'logret1_pow2','logret2_pow2','logrett_pow2',
        old_merged_df = merged_df
        merged_df = old_merged_df[keys_to_focus]
        del book_df
        del trade_df
        del old_merged_df
        
        overview_aggregations = {
#         'wap1': ['sum', 'std'],
#         'wap2': ['sum', 'std'],
        'logret1': [realized_volatility],
        'logret2': [realized_volatility],
        'logrett': [realized_volatility],
#         'wap_balance': ['sum', 'max'],
#         'price_spread1': ['sum', 'max'],
#         'bid_spread': ['sum', 'max'],
#         'ask_spread': ['sum', 'max'],
#         'total_volume': ['sum', 'max'],
#         'volume_imbalance': ['sum', 'max'],
#         "bid_ask_spread": ['sum', 'max'],
#         'size':  ['sum', 'max','min'],
#         'order_count': ['sum', 'max'],
#         'trade_money_turnover': ['sum', 'max','min'],
        }
        aggregations = merged_df.groupby('time_id').agg(overview_aggregations).reset_index(drop=False)
        aggregations = aggregations.fillna(-0.01)
        aggregations.columns = ['_'.join(col).strip() for col in aggregations.columns.values]
        for idx, row in aggregations.iterrows():
            row = row.to_dict()
#             print(row)
#             input()
            time_id = row['time_id_']
#             print(int(time_id), type(time_id))
#             input()
            rowid = get_row_id(main_stock_id, time_id)
            
            if rowid not in feature_map:
                feature_map[rowid] = {}
            
            for key, aggs in overview_aggregations.items():
                for agg in aggs:
                    if isinstance(agg, types.FunctionType):
                        agg = agg.__name__
                    feature_map[rowid][f'{key}_{agg}'] = row[f'{key}_{agg}']
                    
        
        del aggregations
        merge_prepared_df = merged_df.groupby(['time_id','seconds_in_bucket_xs_group']).agg(['sum','max']).reset_index(drop=False)
        del merged_df
        
        for groupkey, groupdf in merge_prepared_df.groupby('time_id'):

            rowid = get_row_id(main_stock_id, groupkey)
            
            if rowid not in feature_map:
                feature_map[rowid] = {}

            sequence_length = len(groupdf['seconds_in_bucket_xs_group'].to_numpy())
                              
            groupdf['has_trade_data'] = (~groupdf['trade_book_price_spread']['max'].isnull()).astype(int)
            groupdf = groupdf.fillna(-0.01)
            
            feature_map[rowid]['sequence_mask_xs'] = [False]*sequence_length + [True]*(intervals_count-sequence_length)
            feature_map[rowid]['has_trade_data_xs'] = np.concatenate([groupdf['has_trade_data'].to_numpy(),[0]*(intervals_count-sequence_length)])
            feature_map[rowid]['seconds_in_bucket_xs_group'] = np.concatenate([groupdf['seconds_in_bucket_xs_group'].to_numpy(),[0]*(intervals_count-sequence_length)])

            for feature_name in ['wap1','wap_balance','logret1','logret2','logrett',
                             'price_spread1','bid_spread','ask_spread','total_volume','volume_imbalance',
                            'size','order_count','trade_money_turnover','trade_book_price_spread']:
                feature_map[rowid][f'{feature_name}_sum_xs'] = np.concatenate([groupdf[feature_name]['sum'].to_numpy(),[0]*(intervals_count-sequence_length)])
                feature_map[rowid][f'{feature_name}_max_xs'] = np.concatenate([groupdf[feature_name]['max'].to_numpy(),[0]*(intervals_count-sequence_length)])
            
            
            # transformer mask ignores the True values,and False remains unchanged
            
                
#         print(merged_df)
#         input()

         
            
#         import gc
        
#         gc.collect()
        return feature_map
    

if __name__ == "__main__":
    
#     overview_aggregations = {
#         'wap1': ['sum', 'std'],
# #         'wap2': [np.sum, np.std],
#         'logret1': [realized_volatility],
#         'logret2': [realized_volatility],
#         'logrett': [realized_volatility],
#         'wap_balance': ['sum', 'max'],
#         'price_spread1': ['sum', 'max'],
#         'bid_spread': ['sum', 'max'],
#         'ask_spread': ['sum', 'max'],
#         'total_volume': ['sum', 'max'],
#         'volume_imbalance': ['sum', 'max'],
#         "bid_ask_spread": ['sum', 'max'],
#         'size':  ['sum', 'max','min'],
#         'order_count': ['sum', 'max'],
#         'trade_money_turnover': ['sum', 'max','min'],
#         }
#     fetnames = []
#     for key, aggs in overview_aggregations.items():
#         for agg in aggs:
#             if isinstance(agg, types.FunctionType):
#                 agg = agg.__name__
#             fetnames.append(f'{key}_{agg}')
#     print(fetnames)
#     exit()
                    
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
            
            
            
def old_code1():
    #                 print(book_df)
        # ACTUAL FEATURES HERE!
        for groupkey, groupdf in trade_df.groupby('time_id'):
            
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
                # ['trade_money_turnover']
                #
                # 
#             grouped_interval_df = generate_interval_group_features(groupdf,'seconds_in_bucket_xs_groupkey',intervals_count,function={'logrett':['sum','mean','std',realized_volatility],'logrett_pow2':['sum']})

            grouped_interval_df = generate_interval_group_features(groupdf,
                                                                   'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={
                                                 'logrett':['sum',realized_volatility,'std','mean'],
                                                 'size':['sum','std'],#,'mean'],
                                                 'order_count':['sum','std'],#,'mean'],
                                                 'trade_money_turnover':['sum','std'],#,'mean'],
#                                                  'trade_money_turnover_per_order':['sum','std','mean']
                                                 }, 
                                            column_fill_map={'size':'zero','order_count':'zero','logrett':'zero','trade_money_turnover':'zero'}) #'trade_money_turnover_per_order':'zero'
            
            grouped_interval_df.columns = ['_'.join(col).strip() for col in grouped_interval_df.columns.values]
#             feature_map[rowid]['trade_logrett_realized_volatility_fast_xs'] = np.sqrt(grouped_interval_df['logrett_pow2_sum']).tolist()
#             feature_map[rowid]['trade_logrett_realized_volatility_xs'] = grouped_interval_df['logrett_realized_volatility'].tolist()
#             #NOTE: double fillna is important! when whole series is na then 'bfill' won't work
            feature_map[rowid]['trade_logrett_sum_xs'] = grouped_interval_df['logrett_sum'].tolist()
            feature_map[rowid]['trade_logrett_realized_volatility_xs'] = grouped_interval_df['logrett_realized_volatility'].tolist()
            feature_map[rowid]['trade_logrett_std_xs'] = grouped_interval_df['logrett_std'].tolist()
            feature_map[rowid]['trade_logrett_mean_xs'] = grouped_interval_df['logrett_mean'].tolist()
            feature_map[rowid]['trade_size_sum_xs'] = grouped_interval_df['size_sum'].tolist()
            feature_map[rowid]['trade_size_std_xs'] = grouped_interval_df['size_std'].tolist()
#             feature_map[rowid]['trade_size_mean_xs'] = grouped_interval_df['size_mean'].tolist()
            feature_map[rowid]['trade_order_count_sum_xs'] = grouped_interval_df['order_count_sum'].tolist()
            feature_map[rowid]['trade_order_count_std_xs'] = grouped_interval_df['order_count_std'].tolist()
#             feature_map[rowid]['trade_order_count_mean_xs'] = grouped_interval_df['order_count_mean'].tolist()
            feature_map[rowid]['trade_trade_money_turnover_sum_xs'] = grouped_interval_df['trade_money_turnover_sum'].tolist()
            feature_map[rowid]['trade_trade_money_turnover_std_xs'] = grouped_interval_df['trade_money_turnover_std'].tolist()
#             feature_map[rowid]['trade_trade_money_turnover_mean_xs'] = grouped_interval_df['trade_money_turnover_mean'].tolist()
#             feature_map[rowid]['trade_trade_money_turnover_per_order_sum_xs'] = grouped_interval_df['trade_money_turnover_per_order_sum'].tolist()
#             feature_map[rowid]['trade_trade_money_turnover_per_order_std_xs'] = grouped_interval_df['trade_money_turnover_per_order_std'].tolist()
#             feature_map[rowid]['trade_trade_money_turnover_per_order_mean_xs'] = grouped_interval_df['trade_money_turnover_per_order_mean'].tolist()
            

            
#             feature_map[rowid]['seconds_in_bucket_xs'] = [(idx*interval_second)+interval_second for idx in range(0,int(intervals_count))]


            
        for groupkey, groupdf in book_df.groupby('time_id'):
            rowid = get_row_id(main_stock_id, groupkey)
            if rowid not in feature_map:
                feature_map[rowid] = {}
            
            feature_map[rowid]['test'] = 0
            feature_map[rowid]['book_realized_volatility'] = generate_realized_volatility_df(groupdf)
            # 'wap_1n2', 'directional_volume1', 'directional_volume2', 'book_money_turnover1','price_spread2',,'wap2','logret2'
            # 'wap_1n2':'mean', 'directional_volume1':'mean', 'directional_volume2':'mean', 'book_money_turnover1':'sum','price_spread2':'mean',,'wap2':'mean', 'logret2':'sum',
            # 'directional_volume1':'zero','directional_volume2':'zero','book_money_turnover1':'zero', 'logret2':'zero'
            
            grouped_interval_df = generate_interval_group_features(groupdf, 
                                                                   'seconds_in_bucket_xs_groupkey', intervals_count, 
                                             function={
                                                 'logret1':['sum',realized_volatility,'std','mean'],
                                                 'logret2':['sum',realized_volatility,'std','mean'],
                                                 'price_spread1':['sum','std'],#,'mean'],
                                                 'bid_spread':['sum','std'],#,'mean'],
                                                 'ask_spread':['sum','std'],#,'mean'],
                                                 'total_volume':['sum','std'],#,'mean'],
                                                 'volume_imbalance':['sum','std'],#,'mean'],
#                                                  'book_money_turnover_intention1':['sum','std','mean']
                                             },
                                            column_fill_map={ 'logret1':'zero','logret2':'zero','price_spread1':'zero',
                                                             'bid_spread':'zero','ask_spread':'zero','total_volume':'zero', 'volume_imbalance':'zero'}) #'book_money_turnover_intention1':'zero'
            
            grouped_interval_df.columns = ['_'.join(col).strip() for col in grouped_interval_df.columns.values]
           
            feature_map[rowid]['book_logret1_sum_xs'] = grouped_interval_df['logret1_sum'].tolist()
            feature_map[rowid]['book_logret1_realized_volatility_xs'] = grouped_interval_df['logret1_realized_volatility'].tolist()
            feature_map[rowid]['book_logret1_std_xs'] = grouped_interval_df['logret1_std'].tolist()
            feature_map[rowid]['book_logret1_mean_xs'] = grouped_interval_df['logret1_mean'].tolist()
            feature_map[rowid]['book_logret2_sum_xs'] = grouped_interval_df['logret2_sum'].tolist()
            feature_map[rowid]['book_logret2_realized_volatility_xs'] = grouped_interval_df['logret2_realized_volatility'].tolist()
            feature_map[rowid]['book_logret2_std_xs'] = grouped_interval_df['logret2_std'].tolist()
            feature_map[rowid]['book_logret2_mean_xs'] = grouped_interval_df['logret2_mean'].tolist()
            feature_map[rowid]['book_price_spread1_sum_xs'] = grouped_interval_df['price_spread1_sum'].tolist()
            feature_map[rowid]['book_price_spread1_std_xs'] = grouped_interval_df['price_spread1_std'].tolist()
#             feature_map[rowid]['book_price_spread1_mean_xs'] = grouped_interval_df['price_spread1_mean'].tolist()
            feature_map[rowid]['book_bid_spread_sum_xs'] = grouped_interval_df['bid_spread_sum'].tolist()
            feature_map[rowid]['book_bid_spread_std_xs'] = grouped_interval_df['bid_spread_std'].tolist()
#             feature_map[rowid]['book_bid_spread_mean_xs'] = grouped_interval_df['bid_spread_mean'].tolist()
            feature_map[rowid]['book_ask_spread_sum_xs'] = grouped_interval_df['ask_spread_sum'].tolist()
            feature_map[rowid]['book_ask_spread_std_xs'] = grouped_interval_df['ask_spread_std'].tolist()
#             feature_map[rowid]['book_ask_spread_mean_xs'] = grouped_interval_df['ask_spread_mean'].tolist()
            feature_map[rowid]['book_total_volume_sum_xs'] = grouped_interval_df['total_volume_sum'].tolist()
            feature_map[rowid]['book_total_volume_std_xs'] = grouped_interval_df['total_volume_std'].tolist()
#             feature_map[rowid]['book_total_volume_mean_xs'] = grouped_interval_df['total_volume_mean'].tolist()
            feature_map[rowid]['book_volume_imbalance_sum_xs'] = grouped_interval_df['volume_imbalance_sum'].tolist()
            feature_map[rowid]['book_volume_imbalance_std_xs'] = grouped_interval_df['volume_imbalance_std'].tolist()
#             feature_map[rowid]['book_volume_imbalance_mean_xs'] = grouped_interval_df['volume_imbalance_mean'].tolist()
#             feature_map[rowid]['book_book_money_turnover_intention1_sum_xs'] = grouped_interval_df['book_money_turnover_intention1_sum'].tolist()
#             feature_map[rowid]['book_book_money_turnover_intention1_std_xs'] = grouped_interval_df['book_money_turnover_intention1_std'].tolist()
#             feature_map[rowid]['book_book_money_turnover_intention1_mean_xs'] = grouped_interval_df['book_money_turnover_intention1_mean'].tolist()
            
 