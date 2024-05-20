
import pandas as pd
pd.set_option('use_inf_as_na', True)
import numpy as np

import json

from sklearn import preprocessing

from operator import itemgetter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from category_encoders import TargetEncoder

RANDOM_SEED = 42


def extractDecisionInfo(decision_tree,counterModels,tree_index,X_train,y_train,feature_names_duplicated,eachAlgor,feature_names=None,only_leaves=True):
    '''return dataframe with node info
    '''
    decision_tree.fit(X_train, y_train)
    # extract info from decision_tree
    n_nodes = decision_tree.tree_.node_count
    children_left = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    feature = decision_tree.tree_.feature
    threshold = decision_tree.tree_.threshold
    impurity = decision_tree.tree_.impurity
    value = decision_tree.tree_.value
    n_node_samples = decision_tree.tree_.n_node_samples

    whereTheyBelong = decision_tree.apply(X_train)

    # cast X_train as dataframe
    df = pd.DataFrame(X_train)
    if feature_names is not None:
        df.columns = feature_names
    
    # indexes with unique nodes
    idx_list = df.assign(
        leaf_id = lambda df: decision_tree.apply(df)
    )[['leaf_id']].drop_duplicates().index

    # test data for unique nodes
    X_test = df.loc[idx_list,].to_numpy()
    # decision path only for leaves
    dp = decision_tree.decision_path(X_test)
    # final leaves for each data
    leave_id = decision_tree.apply(X_test)
    # values for each data
    leave_predict = decision_tree.predict(X_test)
    # dictionary for leave_id and leave_predict
    dict_leaves = {k:v for k,v in zip(leave_id,leave_predict)}
    
    # create decision path information for all nodes
    dp_idxlist = [[ini, fin] for ini,fin in zip(dp.indptr[:-1],dp.indptr[1:])]
    dict_decisionpath = {}
    for idxs in dp_idxlist:
        dpindices = dp.indices[idxs[0]:idxs[1]]
        for i,node in enumerate(dpindices):
            if node not in dict_decisionpath.keys():
                dict_decisionpath[node] = dpindices[:i+1]
    
    # initialize number of columns and output dataframe
    n_cols = df.shape[-1]
    df_thr_all = pd.DataFrame()

    # predict for samples
    for node, node_index in dict_decisionpath.items():
        l_thresh_max = np.ones(n_cols) * np.nan
        l_thresh_min = np.ones(n_cols) * np.nan
        
        # decision path info for each node
        for i,node_id in enumerate(node_index):
            if node == node_id:
                continue

            if children_left[node_id] == node_index[i+1]: #(X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                l_thresh_max[feature[node_id]] = threshold[node_id]
            else:
                l_thresh_min[feature[node_id]] = threshold[node_id]
        # append info to df_thr_all
        df_thr_all = df_thr_all.append(
            [[thr_min for thr_max,thr_min in zip(l_thresh_max,l_thresh_min)]
             + [thr_max for thr_max,thr_min in zip(l_thresh_max,l_thresh_min)]
             + [
                 node,
                 counterModels,
                 tree_index,
                 np.nan if node not in dict_leaves.keys() else dict_leaves[node],
                 #value[node],
                 impurity[node],
                 n_node_samples[node]
               ]
            ]
        )
    # rename columns and set index
    if feature_names is not None:
        df_thr_all.columns = feature_names_duplicated + ['node','counterModels','tree_index','predicted_value','impurity','samples']
    else:
        df_thr_all.columns = ['X_{}'.format(i) for i in range(n_cols)] + ['node','counterModels','tree_index','predicted_value','impurity','samples']
    #df_thr_all = df_thr_all.set_index('decision')
    #df_thr_all = df_thr_all.reset_index(drop=True)
    if only_leaves:
        df_thr_all = df_thr_all[~df_thr_all['predicted_value'].isnull()]
        df_thr_all['impurity'].loc[df_thr_all['impurity'] < 0] = 0
        # df_thr_all['impurity'].loc[df_thr_all['impurity'] >= 0.5] = 0.8

    # del df_thr_all['decision']
    # del df_thr_all['predicted_value']

    #df_thr_all.reset_index()

    df_thr_all = df_thr_all.replace(np.nan,2) # nan mapped as value 2

    #df_thr_all = df_thr_all.sort_index()
    
    copy_df_thr_all = df_thr_all.copy()

    del df_thr_all['node']
    del df_thr_all['counterModels']
    del df_thr_all['tree_index']
    del df_thr_all['predicted_value']
    del df_thr_all['impurity']
    del df_thr_all['samples']

    copy_df_thr_all = copy_df_thr_all[['node','counterModels','tree_index','predicted_value', 'impurity', 'samples']]
    return [df_thr_all,copy_df_thr_all,len(df_thr_all),whereTheyBelong]


def create_class_labels(yData, num_classes):
    y_bins = []
    y_bins.append(np.percentile(yData, 0, ) )
    y_bins.append(np.percentile(yData, 33, ) )
    y_bins.append(np.percentile(yData, 66, ))
    class_labels = np.digitize(yData, y_bins)
    return class_labels


def randomSearchXGB(XData, X_train, y_train, X_test, clf, params, eachAlgor, AlgorithmsIDsEnd, crossValidation, randomS, RANDOM_SEED, roundValue):
    print('insideXGBNow!!!')
    # this is the grid we use to train the models
    scoring = {'accuracy': 'accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro'}

    # randSear = RandomizedSearchCV(    
    #     estimator=clf, param_distributions=params, n_iter=randomS,
    #     cv=crossValidation, refit='accuracy', scoring=scoring,
    #     verbose=0, n_jobs=-1, random_state=RANDOM_SEED)

    time_series_cv = TimeSeriesSplit(n_splits=5)
    randSear = RandomizedSearchCV(
        estimator=clf, param_distributions=params, n_iter=randomS,
        cv=time_series_cv, refit='accuracy', scoring=scoring,
        verbose=0, n_jobs=-1, random_state=RANDOM_SEED)

    # fit and extract the probabilities
    randSear.fit(X_train, y_train)

    # process the results
    cv_results = []
    cv_results.append(randSear.cv_results_)
    df_cv_results = pd.DataFrame.from_dict(cv_results)

    number_of_models = []
    # number of models stored
    number_of_models = len(df_cv_results.iloc[0][0])
    print('number_of_models: ',number_of_models)

    # initialize results per row
    df_cv_results_per_row = []

    modelsIDs = []
    for i in range(number_of_models):
        number = AlgorithmsIDsEnd+i
        modelsIDs.append(eachAlgor+str(number))
        df_cv_results_per_item = []
        for column in df_cv_results.iloc[0]:
            df_cv_results_per_item.append(column[i])
        df_cv_results_per_row.append(df_cv_results_per_item)

    df_cv_results_classifiers = pd.DataFrame()
    # store the results into a pandas dataframe
    df_cv_results_classifiers = pd.DataFrame(data = df_cv_results_per_row, columns= df_cv_results.columns)

    # copy and filter in order to get only the metrics
    metrics = df_cv_results_classifiers.copy()
    metrics = metrics.filter(['mean_test_accuracy', 'mean_test_precision_macro', 'mean_test_recall_macro',]) 

    parametersPerformancePerModel = pd.DataFrame()
    # concat parameters and performance
    parametersPerformancePerModel = pd.DataFrame(df_cv_results_classifiers['params'])
    parametersPerformancePerModel = parametersPerformancePerModel.to_json(double_precision=15)

    parametersLocal = json.loads(parametersPerformancePerModel)['params'].copy()
    Models = []
    for index, items in enumerate(parametersLocal):
        Models.append(str(index))


    parametersLocalNew = [ parametersLocal[your_key] for your_key in Models ]

    perModelProb = []
    confuseFP = []
    confuseFN = []
    featureImp = []
    collectDecisionsPerModel = pd.DataFrame()
    collectDecisions = []
    collectDecisionsMod = []
    collectLocationsAll = []
    collectStatistics = []
    collectStatisticsMod = []
    collectStatisticsPerModel = []
    collectInfoPerModel = []
    yPredictTestList = []
    perModelPrediction = []
    storeTrain = []
    storePredict = []
    
    featureNames = []
    featureNamesDuplicated = []

    
    for col in XData.columns:
        featureNames.append(col)
        featureNamesDuplicated.append(col+'_minLim')
    for col in XData.columns:
        featureNamesDuplicated.append(col+'_maxLim')

    counterModels = 1
    for eachModelParameters in parametersLocalNew:
        collectDecisions = []
        collectLocations = []
        collectStatistics = []
        sumRes = 0
        clf.set_params(**eachModelParameters)
        np.random.seed(RANDOM_SEED) # seeds
        clf.fit(X_train, y_train) 
        yPredictTest = clf.predict(X_test)
        yPredictTestList.append(yPredictTest)

        feature_importances = clf.feature_importances_
        feature_importances[np.isnan(feature_importances)] = 0
        featureImp.append(list(feature_importances))

        yPredict = cross_val_predict(clf, X_train, y_train, cv=2)
        yPredict = np.nan_to_num(yPredict)
        perModelPrediction.append(yPredict)

        yPredictProb = cross_val_predict(clf, X_train, y_train, cv=2, method='predict_proba')
        yPredictProb = np.nan_to_num(yPredictProb)
        perModelProb.append(yPredictProb.tolist())

        storeTrain.append(y_train)
        storePredict.append(yPredict)
        cnf_matrix = confusion_matrix(y_train, yPredict)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        FP = FP.astype(float)
        FN = FN.astype(float)
        confuseFP.append(list(FP))
        confuseFN.append(list(FN))
        for tree_idx, est in enumerate(clf.estimators_):
            # print('\ntree_idx', tree_idx)
            # print('est', est)
            decisionPath = extractDecisionInfo(est[0],counterModels,tree_idx,X_train,y_train,featureNamesDuplicated,eachAlgor,feature_names=featureNames,only_leaves=True)
            if (roundValue == -1):
                pass
            else:
                decisionPath[0] = decisionPath[0].round(roundValue)
            collectDecisions.append(decisionPath[0])
            collectStatistics.append(decisionPath[1])
            sumRes = sumRes + decisionPath[2]
            collectLocations.append(decisionPath[3])
        collectDecisionsMod.append(collectDecisions)
        collectStatisticsMod.append(collectStatistics)
        collectInfoPerModel.append(sumRes)
        collectLocationsAll.append(collectLocations)
        counterModels = counterModels + 1

    collectInfoPerModelPandas = pd.DataFrame(collectInfoPerModel)

    totalfnList = []
    totalfpList = []
    numberClasses = [y_train.index(x) for x in set(y_train)]
    if (len(numberClasses) == 2):
        for index,nList in enumerate(storeTrain):
            fnList = []
            fpList = []
            for ind,el in enumerate(nList):
                if (el==1 and storePredict[index][ind]==0):
                    fnList.append(ind)
                elif (el==0 and storePredict[index][ind]==1):
                    fpList.append(ind)
                else:
                    pass   
            totalfpList.append(fpList)
            totalfnList.append(fnList)
    else:
        for index,nList in enumerate(storeTrain):
            fnList = []
            class0fn = []
            class1fn = []
            class2fn = []
            for ind,el in enumerate(nList):
                if (el==0 and storePredict[index][ind]==1):
                    class0fn.append(ind)
                elif (el==0 and storePredict[index][ind]==2):
                    class0fn.append(ind)
                elif (el==1 and storePredict[index][ind]==0):
                    class1fn.append(ind)
                elif (el==1 and storePredict[index][ind]==2):
                    class1fn.append(ind)
                elif (el==2 and storePredict[index][ind]==0):
                    class2fn.append(ind)
                elif (el==2 and storePredict[index][ind]==1):
                    class2fn.append(ind)
                else:
                    pass  
            fnList.append(class0fn)
            fnList.append(class1fn)
            fnList.append(class2fn)
            totalfnList.append(fnList)
        for index,nList in enumerate(storePredict):
            fpList = []
            class0fp = []
            class1fp = []
            class2fp = []
            for ind,el in enumerate(nList):
                if (el==0 and storeTrain[index][ind]==1):
                    class0fp.append(ind)
                elif (el==0 and storeTrain[index][ind]==2):
                    class0fp.append(ind)
                elif (el==1 and storeTrain[index][ind]==0):
                    class1fp.append(ind)
                elif (el==1 and storeTrain[index][ind]==2):
                    class1fp.append(ind)
                elif (el==2 and storeTrain[index][ind]==0):
                    class2fp.append(ind)
                elif (el==2 and storeTrain[index][ind]==1):
                    class2fp.append(ind)
                else:
                    pass  
            fpList.append(class0fp)
            fpList.append(class1fp)
            fpList.append(class2fp)
            totalfpList.append(fpList)
    
    summarizeResults = []
    summarizeResults = metrics.sum(axis=1)
    summarizeResultsFinal = []
    for el in summarizeResults:
        summarizeResultsFinal.append(round(((el * 100)/3),2))

    indices, L_sorted = zip(*sorted(enumerate(summarizeResultsFinal), key=itemgetter(1)))
    indexList = list(indices)

    collectDecisionsSorted = []
    collectStatisticsSorted = []
    collectLocationsAllSorted = []
    for el in indexList:
        for item in collectDecisionsMod[el]:
            collectDecisionsSorted.append(item)
        for item2 in collectStatisticsMod[el]:
            collectStatisticsSorted.append(item2)
        for item3 in collectLocationsAll[el]:
            collectLocationsAllSorted.append(item3)

    collectDecisionsPerModel = pd.concat(collectDecisionsSorted)
    collectStatisticsPerModel = pd.concat(collectStatisticsSorted)
    collectLocationsAllPerSorted = pd.DataFrame(collectLocationsAllSorted)

    collectDecisionsPerModel = collectDecisionsPerModel.reset_index(drop=True)
    collectStatisticsPerModel = collectStatisticsPerModel.reset_index(drop=True) 
    collectLocationsAllPerSorted = collectLocationsAllPerSorted.reset_index(drop=True) 
    collectDecisionsPerModel = collectDecisionsPerModel.to_json(double_precision=15)
    collectStatisticsPerModel = collectStatisticsPerModel.to_json(double_precision=15)
    collectInfoPerModelPandas = collectInfoPerModelPandas.to_json(double_precision=15)
    collectLocationsAllPerSorted = collectLocationsAllPerSorted.to_json(double_precision=15)

    perModelPredPandas = pd.DataFrame(perModelPrediction)
    perModelPredPandas = perModelPredPandas.to_json(double_precision=15)

    yPredictTestListPandas = pd.DataFrame(yPredictTestList)
    yPredictTestListPandas = yPredictTestListPandas.to_json(double_precision=15)

    perModelProbPandas = pd.DataFrame(perModelProb)
    perModelProbPandas = perModelProbPandas.to_json(double_precision=15)

    metrics = metrics.to_json(double_precision=15)
    # gather the results and send them back
    # resultsXGB.append(modelsIDs) # 0 17
    # resultsXGB.append(parametersPerformancePerModel) # 1 18
    # resultsXGB.append(metrics) # 2 19
    # resultsXGB.append(json.dumps(confuseFP)) # 3 20
    # resultsXGB.append(json.dumps(confuseFN)) # 4 21
    # resultsXGB.append(json.dumps(featureImp)) # 5 22
    # resultsXGB.append(json.dumps(collectDecisionsPerModel)) # 6 23
    # resultsXGB.append(perModelProbPandas) # 7 24
    # resultsXGB.append(json.dumps(perModelPredPandas)) # 8 25
    # resultsXGB.append(json.dumps(target_names)) # 9 26
    # resultsXGB.append(json.dumps(collectStatisticsPerModel)) # 10 27
    # resultsXGB.append(json.dumps(collectInfoPerModelPandas)) # 11 28
    # resultsXGB.append(json.dumps(keepOriginalFeatures)) # 12 29
    # resultsXGB.append(json.dumps(yPredictTestListPandas)) # 13 30
    # resultsXGB.append(json.dumps(collectLocationsAllPerSorted)) # 14 31
    # resultsXGB.append(json.dumps(totalfpList)) # 15 32
    # resultsXGB.append(json.dumps(totalfnList)) # 16 33

    print('Reached end of XGB')
    return # resultsXGB


XData = pd.read_csv('C:\Finance_projects\VisRuler-paper-version\VisRuler-paper-version\data\optiver_1000.csv')
XData['time_id'] = XData['time_id'].astype('category')
XData['stock_id'] = XData['stock_id'].astype('category')


sample_size = int(len(XData)/112)  # round down

yData = XData['target*']
yData = create_class_labels(yData, num_classes=3)

# drop categorical columns
#XData.drop(columns=['stock_id'], inplace=True)
XData.drop(columns=['target*'], inplace=True)

train_ids = XData.iloc[:int(sample_size*0.7)]['time_id']
test_ids = XData.iloc[int(sample_size*0.7):int(sample_size)]['time_id']

X_train = XData[XData['time_id'].isin(train_ids)]
y_train = list(yData[XData['time_id'].isin(train_ids)])

# target encoding
target_encoder = TargetEncoder()
X_train_encode_time_stock_id = target_encoder.fit_transform(X_train[['time_id','stock_id']], y_train)
X_train.drop(columns=['time_id','stock_id'],inplace=True)
X_train = pd.concat([X_train, X_train_encode_time_stock_id], axis=1).values

X_test = XData[XData['time_id'].isin(test_ids)]
y_test = list(yData[XData['time_id'].isin(test_ids)])

X_test_encode_time_stock_id = target_encoder.transform(X_test[['time_id','stock_id']])
X_test.drop(columns=['time_id','stock_id'],inplace=True)
X_test = pd.concat([X_test, X_test_encode_time_stock_id], axis=1).values

# drop categorical columns
#XData.drop(columns=['time_id'], inplace=True)

# time_id_cat = pd.get_dummies(XData['time_id'], prefix='time_id')
# XData = pd.concat([XData, time_id_cat], axis=1)
# stock_id_cat = pd.get_dummies(XData['stock_id'], prefix='stock_id')
# XData = pd.concat([XData, stock_id_cat], axis=1)

# x = XData.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# XDataNorm = pd.DataFrame(x_scaled)


featureNamesLocal = []
for col in XData.columns:
    featureNamesLocal.append(col+'_minLim')

for col in XData.columns:
    featureNamesLocal.append(col+'_maxLim')

target_names = []
target_names.append('low1_vol')
#target_names.append('low2_vol')
target_names.append('med1_vol')
#target_names.append('med2_vol')
target_names.append('high1_vol')
#target_names.append('high2_vol')

for ind, value in enumerate(target_names):
    featureNamesLocal.append(str(ind))  # ind+1

# X_train, X_test, y_train, y_test = train_test_split(XDataNorm, yData, test_size=0.1,stratify=XDataNorm[0], random_state=RANDOM_SEED, shuffle=False)

# X_train = X_train.reset_index(drop=True)
# X_test = X_test.reset_index(drop=True)

# resultsLocalXGB = randomSearchXGB(XData, X_train, y_train, X_test, clf, paramsXGB, eachAlgor, AlgorithmsIDsEnd, crossValidation, randomSearchVar, RANDOM_SEED, roundValueSend)  

paramsXGB = {'n_estimators': list(range(5,11,5)), 'max_depth': list(range(2,4,1)), 'learning_rate': list(np.arange(0.01, 0.11,0.09)), 'subsample': list(np.arange(0.5,1.1,0.5)), 'min_samples_split': list(range(2, 6, 3)), 'min_samples_leaf': list(range(1, 11, 9))}
sendHyperXGB = paramsXGB

# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=RANDOM_SEED)
clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
AlgorithmsIDsEnd = 10
randomSearchVar = 2  # number of models to sample for hyperparameter optimization
crossValidation = 1
roundValueSend = 15
resultsLocalXGB = randomSearchXGB(XData, X_train, y_train, X_test, clf, paramsXGB, 'XGB', AlgorithmsIDsEnd, crossValidation, randomSearchVar, RANDOM_SEED, roundValueSend)
