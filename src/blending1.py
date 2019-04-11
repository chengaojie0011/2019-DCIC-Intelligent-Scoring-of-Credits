# -*- coding: utf-8 -*-
# blending1
# author = 'Gavin'

import os
import time
import math
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

BASE_PATH = os.path.join('input')
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "blending1")
TEST_DATA_PATH = os.path.join(BASE_PATH, "blending1")
NFOLDS = 10


def get_train(files):  
    df_all = pd.DataFrame()  
    train_data = pd.read_csv('data/RawData/train_dataset.csv')
    df_all['id'] = train_data['用户编码']
    files_name =[]
    for file in files:   
        if 'train' in file:
            file_name = file[:-10]
            files_name.append(file_name)
#            print(file,file_name)
            data_name = os.path.join(TRAIN_DATA_PATH, file)
            df = pd.read_csv(data_name)
            if 'index' in df.columns.values.tolist():
                df = df.sort_values(by = 'index',ascending= True)
                df.index = range(0,50000) 
    #            df_all['id_file']=df['id']
    #            df_all['target_file']=df['target']
    #            df_all['index_file']=df['index']
            df_all[file_name]=df['score']

#            df_all[file_name]=df['score'].apply(lambda x: int(np.round(x)))
    df_all['target'] = train_data['信用分'] 
    return df_all, files_name

def get_test(files):  
    df_all = pd.DataFrame()  
    test_data = pd.read_csv('data/RawData/test_dataset.csv')
    df_all['id'] = test_data['用户编码']
    files_name =[]
    for file in files:   
#        file_name = file[:-9]
        if 'test' in file:
            file_name = file[:-9]
            files_name.append(file_name)
            print(file,file_name)
            data_name = os.path.join(TEST_DATA_PATH, file)
            df = pd.read_csv(data_name)
            df = pd.read_csv(data_name)
            df_all[file_name]=df['score']  
#            df_all[file_name]=df['score'].apply(lambda x: int(math.ceil(x)))     
    return df_all, files_name

def get_stacking(train_df, test_df):
    train_label =  train_df['target']
    
    kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=21)
    kf = kfold.split(train_df, train_label)
    
    train = train_df.drop(['id', 'target'], axis=1)
    test = test_df.drop(['id'], axis=1)
    
    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0
    valid_best_l2_all_list = []        
    val_pred_all = pd.DataFrame()
    for i, (train_fold, validate) in enumerate(kf):
        print("stacking. fold: ", i , "training...")
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]
        #model_BayesianRidge
        gbm = BayesianRidge(normalize=True)#, lambda_1 = 4,lambda_2 = 4.2)
        #model_LinearRegression          
#        gbm = LinearRegression()
        #model_Ridge
#        gbm = Ridge()
        bst = gbm.fit(X_train, label_train)
#        print('coef_:',bst.coef_)
        val_pred = pd.DataFrame()
        val_pred['target'] = label_validate
        val_pred['pred'] = bst.predict(X_validate)
        val_pred_all = pd.concat([val_pred_all, val_pred], axis=0, ignore_index=True)

        cv_pred += bst.predict(test)
        x =  (val_pred['pred']+1.18)#.apply(lambda x: int(np.round(x)))
        valid_best_l2_all += mean_absolute_error(y_true= val_pred['target'] , y_pred= x)  

        valid_best_l2_all_list.append(mean_absolute_error(y_true= val_pred['target'] , y_pred= x) )
  
    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    stacking_score = 1/(1+valid_best_l2_all)
    print("stacking cv score for valid is: ", stacking_score)

    return cv_pred,stacking_score

def model_weight(data,sort_name):
    min_num = 15000
    max_num = 47000
    data = data.sort_values(sort_name)    
    data['ranks'] = list(range(data.shape[0]))
    cols =  data.columns.values.tolist()
    cols.remove('score')
    cols.remove('ranks')
    print(cols)

    #中间的权重
    middle_wight =[1, 1, 1 ,1, 1, 1, 1, 1, 1]

    print(middle_wight)
    weight_sum = 0
    for col,weight in zip(cols,middle_wight):
        data['score'] += data[col]*weight
        weight_sum += weight
    data['score'] = data['score']/weight_sum  

    data = data.sort_index(ascending=True)
    return data    


def get_blending(train_df, test_df):
    train_cp = train_df.copy()
    train_cp = train_cp.drop(['id','target'], axis=1)
    train_cp['score'] =0
#    train_cp['Col_sum'] = train_cp.apply(lambda x: x.sum(), axis=1)
#    train_cp['Col_avg'] = train_cp['Col_sum']/int(train_cp.shape[1]-1)
#    for col in train_cp.columns.values.tolist():
#        if col == 'ctb_751_kf5':
#            train_cp[col] = train_cp[col].apply(lambda x: int(np.round(x)))

    train_cp = model_weight(train_cp,'lgb_mae_c_918')
    valid = mean_absolute_error(y_true= train_df['target'] , y_pred= (train_cp['score']+1)) 
    blending_score = 1/(1+valid)    
    print("blending cv score for valid is: ", blending_score)    
    
    test_cp = test_df.copy()
    test_cp = test_cp.drop(['id'], axis=1)
    test_cp['score'] =0
    test_cp = model_weight(test_cp,'lgb_mae_c_918')   

    
    return test_cp

if __name__ == "__main__":
    t0 = time.time()
    
    #get data
    train_files= os.listdir(TRAIN_DATA_PATH)
    train_df, train_name = get_train(train_files)    
    
    test_files= os.listdir(TEST_DATA_PATH)
    test_df, test_name = get_test(test_files)
    #输出blending结果
    bld_pred = get_blending(train_df, test_df)
    test_bld_sub = pd.DataFrame()
    test_bld_sub['id'] = test_df['id']
    test_bld_sub['score'] =bld_pred['score']
    test_bld_sub['score'] = (test_bld_sub['score']+1)
#    test_bld_sub['score'] = (test_bld_sub['score']).apply(lambda x: int(math.ceil(x)))
#    print(test_bld_sub.describe()) 
    test_bld_sub.to_csv('output/cg_model1.csv', index=False)  
   
    #输出stacking结果    
#    stk_pred,stk_score = get_stacking(train_df, test_df)
#    test_stk_sub = pd.DataFrame()
#    test_stk_sub['id'] = test_df['id']
#    test_stk_sub['score'] =stk_pred+1.19
##        test_stk_sub['score'] = test_stk_sub['score'].apply(lambda x: int(np.round(x)))
#    test_stk_sub.to_csv('output/stacking.csv', index=False)

   
    print("Models have fused!")
    print("Cost {} s.".format(time.time() - t0))
