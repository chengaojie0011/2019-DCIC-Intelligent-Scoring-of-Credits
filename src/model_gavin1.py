# -*- coding: utf-8 -*-
# model
# author = 'Gavin'

import os
import time

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from feature_engineering_gavin import Processing

BASE_PATH = os.path.join('data')
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


def get_feature(name):
    data_name = os.path.join(ETL_DATA_PATH, "{}.csv".format(name))
    df = pd.read_csv(data_name)
    return df

def model_params():
    
    # lgb_mae参数
   params_mae_lgb = {
       'learning_rate': 0.005,
       'boosting_type': 'gbdt',
       'objective': 'regression_l1',
       'metric': 'mae',
       'feature_fraction': 0.66,
       'bagging_fraction': 0.8,       
       'bagging_freq': 2,
       'num_leaves': 31,
       'verbose': -1,
       'max_depth': 5,
       'lambda_l2': 1.6,
       'lambda_l1':3.7, 
       'nthread': 8,
       'seed': 26
   }
 
    # lgb_mse参数
    params_mse_lgb = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'feature_fraction': 0.67,
        'bagging_fraction': 0.75,
        'bagging_freq': 2,
        'num_leaves': 63,
        'verbose': -1,
        'max_depth': 6,
        'lambda_l2':4,
        'lambda_l1': 2, 
        'nthread': 8,
        'seed': 89
    }
    
    # xgb_mae参数
    params_mae_xgb = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.66,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'n_jobs': 4,
#        'random_state': 4590,
        'reg_alpha': 0,
        'reg_lambda': 5,
        'alpha': 1
    }
    
    params_mae_ctb = {
        'n_estimators': 10000,
        'learning_rate': 0.01,
        'random_seed': 41,
        'reg_lambda': 5,
        'subsample': 0.8,
        'bootstrap_type': 'Bernoulli',
        'boosting_type': 'Plain',
        'one_hot_max_size': 10,
        'rsm': 0.5,
        'leaf_estimation_iterations': 5,
        'use_best_model': True,
        'max_depth': 5,
        'verbose': -1,
        'thread_count': 4,
    }    

    return params_mae_lgb, params_mse_lgb, params_mae_xgb,params_mae_ctb

def lgb_mae_model(train_df, test_df, params):
    NFOLDS = 10
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)

    id_traget = pd.DataFrame()
    id_traget['id'] = train_df['用户编码']
    id_traget['target'] = train_df['信用分']    
 
    
    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0
    valid_best_l2_all_list = []
    models = []
    count = 0
    val_pred_all = pd.DataFrame()
    for i, (train_fold, validate) in enumerate(kf):
        print("model: lgb_mae. fold: ", i , "training...")
        
        val_id_target = id_traget.iloc[validate]        
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid,
                        verbose_eval=-1, early_stopping_rounds=2000)        
        val_pred = pd.DataFrame()
        val_pred['id'] = val_id_target['id']
        val_pred['index'] = X_validate.index
        val_pred['target'] = label_validate
        val_pred['score'] = bst.predict(X_validate, num_iteration=bst.best_iteration)
        val_pred_all = pd.concat([val_pred_all, val_pred], axis=0, ignore_index=True)
        
        
        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']
        valid_best_l2_all_list.append(bst.best_score['valid_0']['l1'])
        count += 1
        models.append(bst)

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    mae_score = 1/(1+valid_best_l2_all)
    print("lgb_mae cv score for valid is: ", mae_score)

#    print("----------------------------------------")
#    print("----------------------------------------")
#    print("lgb_mae  feature importance：")
    fea_importances = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
    }).sort_values(by='importance', ascending=False)
#    print(fea_importances)
#    print("----------------------------------------")
#    print("----------------------------------------")

    return val_pred_all,cv_pred,mae_score,fea_importances


def lgb_mse_model(train_df, test_df, params):
    NFOLDS = 10
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS, shuffle=False, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)

    id_traget = pd.DataFrame()
    id_traget['id'] = train_df['用户编码']
    id_traget['target'] = train_df['信用分']    
 

    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0
    valid_best_l2_all_list = []
    count = 0
    models = []
    val_pred_all = pd.DataFrame()
    for i, (train_fold, validate) in enumerate(kf):
        print("model:lgb_mse. fold: ", i , "training...")

        val_id_target = id_traget.iloc[validate]

        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1, early_stopping_rounds=4000)
        val_pred = pd.DataFrame()
        val_pred['id'] = val_id_target['id']
        val_pred['index'] = X_validate.index
        val_pred['target'] = label_validate
        val_pred['score'] = bst.predict(X_validate, num_iteration=bst.best_iteration)
        val_pred_all = pd.concat([val_pred_all, val_pred], axis=0, ignore_index=True)

        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']
        valid_best_l2_all_list.append(bst.best_score['valid_0']['l1'])
        count += 1
        models.append(bst)

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    mse_score = 1/(1+valid_best_l2_all)
    print("lgb_mse cv score for valid is: ", mse_score)

#    print("----------------------------------------")
#    print("----------------------------------------")
#    print("lgb_mse  feature importance：")
    fea_importances = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
    }).sort_values(by='importance', ascending=False)
#    print(fea_importances)
#    print("----------------------------------------")
#    print("----------------------------------------")

    return val_pred_all,cv_pred,mse_score,fea_importances



def xgb_model(train_df, test_df, params):
    NFOLDS = 10
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS,shuffle=True, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)

    id_traget = pd.DataFrame()
    id_traget['id'] = train_df['用户编码']
    id_traget['target'] = train_df['信用分']    


    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0
    valid_best_l2_all_list = []
    models = []
    count = 0
    val_pred_all = pd.DataFrame()
    for i, (train_fold, validate) in enumerate(kf):
        print("model: xgb_mae. fold: ", i , "training...")
        val_id_target = id_traget.iloc[validate]

        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        gbm = xgb.XGBRegressor(**params)
        bst = gbm.fit(X_train, label_train, eval_set=[(X_train, label_train), (X_validate, label_validate)],
                    eval_metric ='mae',verbose=500,early_stopping_rounds=200)

        val_pred = pd.DataFrame()
        val_pred['id'] = val_id_target['id']
        val_pred['index'] = X_validate.index
        val_pred['target'] = label_validate
        val_pred['score'] = bst.predict(X_validate)
        val_pred_all = pd.concat([val_pred_all, val_pred], axis=0, ignore_index=True)
 

        cv_pred += bst.predict(test)
        valid_best_l2_all += mean_absolute_error(y_true= val_pred['target'] , y_pred= val_pred['score'])  
        valid_best_l2_all_list.append(mean_absolute_error(y_true= val_pred['target'] , y_pred= val_pred['score']) )
        count += 1
        models.append(bst)

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    mae_score = 1/(1+valid_best_l2_all)
    print("xgb_mae cv score for valid is: ", mae_score)

    # print("----------------------------------------")
    # print("----------------------------------------")
    # print("xgb_mae  feature importance：")
#    fea_importances = pd.DataFrame({
#         'column': train.columns,
#         'importance': bst.feature_importance
#     }).sort_values(by='importance', ascending=False)
#    print(fea_importances)
    # print("----------------------------------------")
    # print("----------------------------------------")

    return val_pred_all,cv_pred,mae_score

def ctb_model(train_df, test_df, params):
    NFOLDS = 5
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS,shuffle=False, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)

    id_traget = pd.DataFrame()
    id_traget['id'] = train_df['用户编码']
    id_traget['target'] = train_df['信用分']    


    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0
    valid_best_l2_all_list = []
    models = []
    count = 0
    val_pred_all = pd.DataFrame()
    for i, (train_fold, validate) in enumerate(kf):
        print("model: cgb_mae. fold: ", i , "training...")

        val_id_target = id_traget.iloc[validate]
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        cat = ctb.CatBoostRegressor(**params)
        bst = cat.fit(X_train, label_train, eval_set=[(X_train, label_train), (X_validate, label_validate)],
                          early_stopping_rounds=2000, verbose=1000)

        val_pred = pd.DataFrame()
        val_pred['id'] = val_id_target['id']
        val_pred['index'] = X_validate.index
        val_pred['target'] = label_validate
        val_pred['score'] = bst.predict(X_validate)
        val_pred_all = pd.concat([val_pred_all, val_pred], axis=0, ignore_index=True)
 

        cv_pred += bst.predict(test)
        valid_best_l2_all += mean_absolute_error(y_true= val_pred['target'] , y_pred= val_pred['score'])  
        valid_best_l2_all_list.append(mean_absolute_error(y_true= val_pred['target'] , y_pred= val_pred['score']) )
        count += 1
        models.append(bst)

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    mae_score = 1/(1+valid_best_l2_all)
    print("cgb_mae cv score for valid is: ", mae_score)

    # print("----------------------------------------")
    # print("----------------------------------------")
    # print("xgb_mae  feature importance：")
#    fea_importances = pd.DataFrame({
#         'column': train.columns,
#         'importance': bst.feature_importance
#     }).sort_values(by='importance', ascending=False)
#    print(fea_importances)
    # print("----------------------------------------")
    # print("----------------------------------------")

    return val_pred_all,cv_pred,mae_score

def model_bagging_mean(pred1, pred2):
    
    cv_pred = (pred1 + pred2 ) / 2
    print('The final score is:',cv_pred)
    return cv_pred

def model_bagging_weight(test_input, pred_mae, pred_mse):
    min_num = 15000
    max_num = 60000
    #mae权重大的
    min_weight1 = 0.4
    max_weight1 = 0.6
    #mse权重大的
    min_weight2 = 0.45
    max_weight2 = 0.55
    test = pd.DataFrame()
    test['id'] = test_input['用户编码']
    test['pred_mae'] = pred_mae
    test['pred_mse'] = pred_mse
    test = test.sort_values('pred_mae')
    test['ranks'] = list(range(test.shape[0]))
    
    test['score'] =  test['pred_mse']*min_weight1 +  test['pred_mae']*max_weight1
    
    test.loc[test.ranks<min_num,'score']  = test.loc[test.ranks< min_num,'pred_mse'].values *max_weight2 + \
                                            test.loc[test.ranks< min_num,'pred_mae'].values * min_weight2
                                            
    test.loc[test.ranks>max_num,'score']  = test.loc[test.ranks> max_num,'pred_mse'].values *max_weight2 +\
                                            test.loc[test.ranks> max_num,'pred_mae'].values * min_weight2
#        test.loc[test.ranks>max_num,'score']  = test.loc[test.ranks> max_num,'pred_mse'].values *min_weight + test.loc[test.ranks> max_num,'pred_mae'].values * max_weight 
#        test.loc[(test.ranks >=min_num) & (test.ranks<=max_num),'score']  = test.loc[(test.ranks >=min_num) & (test.ranks<=max_num),'pred_mse'].values *min_weight+ test.loc[(test.ranks >=min_num) & (test.ranks<=max_num),'pred_mae'].values * max_weight       
#        test['score'] =  test['pred_mse']*min_weight +  test['pred_mae']*max_weight
    return test 

def model_main():
    
  

    
    return 0

if __name__ == "__main__":
    t0 = time.time()
    
    
#    model_main()
    
    
    #input params
    params_mae_lgb, params_mse_lgb,params_xgb,params_ctb =  model_params()  

    #input dataset
    processing = Processing()
    data,train_data,test_data = processing.get_processing()
    
    
    #lgb_mae
#    train_data = get_feature(name="train_data_gavin")
#    test_data = get_feature(name="test_data_gavin")
    lgb_mae_val,lgb_mae_pred,lgb_mae_score,lgb_mae_importances = lgb_mae_model(train_data, test_data, params_mae_lgb)
    print('Gen train_mae shape: {}, test_mae shape: {}'.format(train_data.shape, test_data.shape))
    print('features num: ', test_data.shape[1] - 1)
 
#    #输出lgb_mae_val结果
#    lgb_mae_val['score'] = lgb_mae_val['score'].apply(lambda x: int(np.round(x)))
   lgb_mae_val.to_csv('input/lgb_mae_g_1_train.csv', index=False)     
   
   #输出lgb_mae结果
   test_mae_lgb_sub1 = pd.DataFrame()    
   test_mae_lgb_sub1['id'] = test_data['用户编码'] 
   test_mae_lgb_sub1['score'] =lgb_mae_pred
#    test_mae_lgb_sub1['score'] = test_mae_lgb_sub1['score'].apply(lambda x: int(np.round(x)))
#    test_mae_lgb_sub1[['id', 'score']].to_csv('input/lgb_mae_xxx_kf10_testpush.csv', index=False)
   test_mae_lgb_sub1[['id', 'score']].to_csv('input/lgb_mae_g_1_test.csv', index=False)
    

    #lgb_mse    
#    train_data = get_feature(name="train_data_gavin")
#    test_data = get_feature(name="test_data_gavin")
   lgb_mse_val,lgb_mse_pred,lgb_mse_score,lgb_mse_importances = lgb_mse_model(train_data, test_data, params_mse_lgb)  
   print('Gen train_mse shape: {}, test_mse shape: {}'.format(train_data.shape, test_data.shape))
   print('features num: ', test_data.shape[1] - 1)

   #输出lgb_mse_val结果
##    lgb_mse_val['score'] = lgb_mse_val['score'].apply(lambda x: int(np.round(x)))
   lgb_mse_val.to_csv('input/lgb_mse_g_1_train.csv', index=False)    
 
   #输出lgb_mse结果
   test_mse_lgb_sub2 = pd.DataFrame()    
   test_mse_lgb_sub2['id'] = test_data['用户编码']
   test_mse_lgb_sub2['score'] =lgb_mse_pred
#    test_mse_lgb_sub2['score'] = test_mse_lgb_sub2['score'].apply(lambda x: int(np.round(x)))
#    test_mse_lgb_sub2[['id', 'score']].to_csv('input/lgb_mse_1_g_testpush.csv', index=False)    
   test_mse_lgb_sub2[['id', 'score']].to_csv('input/lgb_mse_g_1_test.csv', index=False)    

#    #xgb
#    train_data = get_feature(name="train_data_gavin")
#    test_data = get_feature(name="test_data_gavin")
#    xgb_val,xgb_pred,xgb_score = xgb_model(train_data, test_data, params_xgb)
#    print('Gen train_mae shape: {}, test_mae shape: {}'.format(train_data.shape, test_data.shape))
#    print('features num: ', test_data.shape[1] - 1)

#    #输出xgb_val结果

# #    xgb_val['score'] = val_xgb_sub['score'].apply(lambda x: int(np.round(x)))
#    xgb_val.to_csv('input/xgb_1_kf10_train.csv', index=False)    
  
#    #输出xgb_test结果
#    test_xgb_sub3 = test_data[['用户编码']]
#    test_xgb_sub3['score'] =xgb_pred
#    test_xgb_sub3.columns = ['id', 'score']
#    test_xgb_sub3['score'] = test_xgb_sub3['score'].apply(lambda x: int(np.round(x)))
#    test_xgb_sub3[['id', 'score']].to_csv('input/xgb_1_kf10_test.csv', index=False)   
    
   #ctb
#    train_data = get_feature(name="train_data_gavin")
#    test_data = get_feature(name="test_data_gavin")
   ctb_val,ctb_pred,ctb_score = ctb_model(train_data, test_data, params_ctb)
   print('Gen train_mae shape: {}, test_mae shape: {}'.format(train_data.shape, test_data.shape))
   print('features num: ', test_data.shape[1] - 1)

   #输出ctb_val结果
#    val_ctb_sub['score'] = val_ctb_sub['score'].apply(lambda x: int(np.round(x)))
   ctb_val.to_csv('input/ctb_g_1_train.csv', index=False)    
  
   #输出ctb_test结果
   test_ctb_sub3 = test_data[['用户编码']]
   test_ctb_sub3['score'] =ctb_pred
   test_ctb_sub3.columns = ['id', 'score']
#    test_ctb_sub3['score'] = test_ctb_sub3['score'].apply(lambda x: int(np.round(x)))
#    test_ctb_sub3[['id', 'score']].to_csv('input/ctb_xxxpush_g_test.csv', index=False)     
   test_ctb_sub3[['id', 'score']].to_csv('input/ctb_g_1_test.csv', index=False)     
   
    print("Model has trained!")
    print("Cost {} s.".format(time.time() - t0))
