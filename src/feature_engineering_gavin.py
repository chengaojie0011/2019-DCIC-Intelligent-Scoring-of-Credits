# -*- coding: utf-8 -*-
# 特征工程
# author = 'Gavin'

import os
import time
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


BASE_PATH = os.path.join('data')
RAW_DATA_PATH = os.path.join(BASE_PATH, "RawData")
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")

class Processing(object):
    # 读取数据
    @staticmethod
    def _get_data_(name):
        data_name = os.path.join(RAW_DATA_PATH, '{}.csv'.format(name))
        df = pd.read_csv(data_name)
        return df

    # 改变变量类型节省内存空间
    @staticmethod
    def _reduce_mem_usage_(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    # 缴费数据修正处理
    @staticmethod
    def _pay_fix_process_(df):

        def get_recharge_way(item):
            # 是否能被10整除
            if item == 0:
                return -1     
            if item % 10 == 0:
                return 1
            else:
                return 0
            
        df['用户最近一次缴费距今时长（月）'][(df['缴费用户最近一次缴费金额（元）'] != 0) & \
            (df['用户最近一次缴费距今时长（月）'] == 0 )] = 1    
 
        df['缴费方式'] = 0
        df['缴费方式']=df['缴费用户最近一次缴费金额（元）'].apply(get_recharge_way)     
        df.loc[df['缴费用户最近一次缴费金额（元）'] == 0, '缴费用户最近一次缴费金额（元）'] = df['缴费用户最近一次缴费金额（元）'].mode()                      
        return df

    # 众数填充异常值
    @staticmethod
    def _mode_fill_(df):       

        #取NaN
        df.loc[df['用户年龄'] == 0, '用户年龄'] = df['用户年龄'].mode()
        df.loc[df['用户话费敏感度'] == 0, '用户话费敏感度'] = df['用户话费敏感度'].mode()

        return df
    
    
    # 疯狂找特征呀呀呀
    @staticmethod
    def _feature_(df):  
        number_list = [ '用户年龄', 
                            '是否大学生客户', '是否黑名单客户', '是否4G不健康客户',
                            '用户网龄（月）','用户最近一次缴费距今时长（月）',
                            '缴费用户最近一次缴费金额（元）', '用户账单当月总费用（元）',
                             '用户当月账户余额（元）','缴费用户当前是否欠费缴费', '用户话费敏感度', '当月通话交往圈人数', 
                            '是否经常逛商场的人', '近三个月月均商场出现次数',
                             '当月是否看电影', '当月是否景点游览', '当月是否体育场馆消费', 
                            '当月网购类应用使用次数', '当月物流快递类应用使用次数',
                             '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数',
                             '当月旅游资讯类应用使用次数']
        df['是否去过高档商场']=df['当月是否到过福州山姆会员店']+df['当月是否逛过福州仓山万达']
        df['是否去过高档商场']=df['是否去过高档商场'].map(lambda x:1 if x>=1 else 0)
        df['交通类应用使用次数'] = df['当月飞机类应用使用次数'] + df['当月火车类应用使用次数']
        df["当前消费动荡"]=((df["用户近6个月平均消费值（元）"]*6-df["用户账单当月总费用（元）"])/5)/(df["用户账单当月总费用（元）"]+1)
        df["半年费用"]=(df["用户近6个月平均消费值（元）"]*6-df["用户账单当月总费用（元）"])       
        df.loc[df['半年费用'] <= 0, '半年费用'] = np.percentile(df['半年费用'].values,0.1)
    
        
        #app偏好
        df["偏好"]=0
        df["偏好"][(df["当月网购类应用使用次数"]>df['当月金融理财类应用使用总次数'])&(df['当月网购类应用使用次数']>df['当月视频播放类应用使用次数'])]=1
        df["偏好"][(df["当月金融理财类应用使用总次数"]>df['当月网购类应用使用次数'])&(df['当月金融理财类应用使用总次数']>df['当月视频播放类应用使用次数'])]=2
        df = pd.get_dummies(df, columns=["偏好"])
        
        def map_money(x):
            pay = x['缴费用户最近一次缴费金额（元）'] 
            time = x['用户最近一次缴费距今时长（月）'] 
            if time == 0:
                return 0
            else:
                if pay ==1:                
                    return 1
                else:
                    return -1         
        df['缴费与时长']=df[['缴费用户最近一次缴费金额（元）', '用户最近一次缴费距今时长（月）']].apply(map_money, axis=1)  
        
        def feature_count(data, features=[]):
            if len(set(features)) != len(features):
                print('equal feature !!!!')
                return data
            new_feature = 'count'
            for i in features:
                new_feature += '_' + i.replace('add_', '')
            temp = data.groupby(features)
            temp=temp.size()
            temp = temp.reset_index()
            temp = temp.rename(columns={0: new_feature})
            data = data.merge(temp, 'left', on=features)
            return data
        #drop  
        df=df.drop({'当月是否逛过福州仓山万达','当月是否到过福州山姆会员店'},axis=1)
        df=df.drop({"用户近6个月平均消费值（元）"},axis=1)
        df=df.drop({'用户最近一次缴费距今时长（月）'},axis=1)
        df=df.drop({'当月火车类应用使用次数','当月飞机类应用使用次数'},axis=1)

        return df
    
        
    # 年龄分箱  
    @staticmethod
    def _group_age_(x):
        if x==0:
            return 0
        elif x <= 18:
            return 1
        elif x <= 30:
            return 2
        elif x <= 35:
            return 3
        elif x <= 55:
            return 4
        else:
            return 5

    # 长尾数据处理
    @staticmethod
    def _log_feature_(df):
        
        extreme_features = ['用户当月账户余额（元）','用户网龄（月）',
                            '用户账单当月总费用（元）',
                            '当前消费动荡','半年费用',#'用户年龄','缴费用户最近一次缴费金额（元）',
                            '当月通话交往圈人数','近三个月月均商场出现次数',
                            '当月网购类应用使用次数','当月物流快递类应用使用次数',
                            '当月金融理财类应用使用总次数','当月视频播放类应用使用次数',
                            '当月旅游资讯类应用使用次数','交通类应用使用次数']
    
        for col in extreme_features:
            #取出最高99.9%值
            ulimit=np.percentile(df[col].values,99.9)
            #取出最低0.1%值
            llimit=np.percentile(df[col].values,0.1)
            df.loc[df[col]>ulimit,col]=ulimit
            df.loc[df[col]<llimit,col]=llimit
        
        extreme_max_features = ['用户年龄','缴费用户最近一次缴费金额（元）']
        for col in extreme_max_features:
            #取出最高99.9%值
            ulimit=np.percentile(df[col].values,99.85)
            df.loc[df[col]>ulimit,col]=ulimit

        extreme_min_features = ['用户网龄（月）','用户账单当月总费用（元）','半年费用']
        
        for col in extreme_min_features:
            #取出最低0.2%值
            if col == '用户账单当月总费用（元）':
                llimit=np.percentile(df[col].values,0.2)
                df.loc[df[col]<llimit,col]=llimit  
                continue
            llimit=np.percentile(df[col].values,0.15)
            df.loc[df[col]<llimit,col]=llimit

            
        user_bill_features = [#'用户年龄','用户网龄（月）',
                              '用户当月账户余额（元）',
                         '用户账单当月总费用（元）','缴费用户最近一次缴费金额（元）',
                         '当月通话交往圈人数','半年费用']
        
        #'半年费用'
        log_features = ['当月旅游资讯类应用使用次数','交通类应用使用次数','当月网购类应用使用次数',
                        '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']     
        for col in  user_bill_features+log_features:
            df[col] = df[col].map(lambda x: np.log1p(x))
        return df   
    
    def get_processing(self):
        train_df = self._get_data_('train_dataset')
        test_df = self._get_data_('test_dataset')    
        train_df = self._reduce_mem_usage_(train_df)
        test_df = self._reduce_mem_usage_(test_df)
        test_df['信用分'] = -1
        data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        del train_df, test_df
        gc.collect()
        
        data = self._pay_fix_process_(data)
        data = self._mode_fill_(data)
        data = self._feature_(data)
        data = self._log_feature_(data)

        train, test = data[:50000], data[50000:]
        test = test.drop(['信用分'], axis=1)

#        train_data_name = os.path.join(ETL_DATA_PATH, 'train_data_gavin.csv')
#        test_data_name = os.path.join(ETL_DATA_PATH, 'test_data_gavin.csv')
#        train.to_csv(train_data_name, index=False,encoding='utf_8_sig')
#        test.to_csv(test_data_name, index=False,encoding='utf_8_sig')
        print('Gen train shape: {}, test shape: {}'.format(train.shape, test.shape))
        print('features num: ', test.shape[1] - 1)
        return data,train,test


if  __name__ == "__main__":
    t0 = time.time()
    processing = Processing()
    data,train,test  = processing.get_processing()
    print("Feature engineering has finished!")
    print("Cost {} s.".format(time.time() - t0))