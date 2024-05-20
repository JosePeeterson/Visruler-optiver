# first line: 571
@memory.cache
def randomSearchRF(XData, X_train, y_train, X_test, clf, params, eachAlgor, AlgorithmsIDsEnd, crossValidation, randomS, RANDOM_SEED, roundValue):
    print('insideRFNow!!!')
    # this is the grid we use to train the models
    randSear = RandomizedSearchCV(    
        estimator=clf, param_distributions=params, n_iter=randomS,
        cv=crossValidation, refit='accuracy', scoring=scoring,
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
    collectLocationsAll = []
    collectDecisionsMod = []
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

        yPredict = cross_val_predict(clf, X_train, y_train, cv=crossValidation)
        yPredict = np.nan_to_num(yPredict)
        perModelPrediction.append(yPredict)

        yPredictProb = cross_val_predict(clf, X_train, y_train, cv=crossValidation, method='predict_proba')
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
            decisionPath = extractDecisionInfo(est,counterModels,tree_idx,X_train,y_train,featureNamesDuplicated,eachAlgor,feature_names=featureNames,only_leaves=True)
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
    
    # for index, nList in enumerate(totalfpList):
    #     if ((index+1) != len(totalfpList)):
    #         confuseFPCommon.append(len(list(set(nList).intersection(totalfpList[index+1]))))
    # for index, nList in enumerate(totalfnList):
    #     if ((index+1) != len(totalfnList)):
    #         confuseFNCommon.append(len(list(set(nList).intersection(totalfnList[index+1]))))

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
    resultsRF.append(modelsIDs) # 0 
    resultsRF.append(parametersPerformancePerModel) # 1
    resultsRF.append(metrics) # 2
    resultsRF.append(json.dumps(confuseFP)) # 3
    resultsRF.append(json.dumps(confuseFN)) # 4
    resultsRF.append(json.dumps(featureImp)) # 5
    resultsRF.append(json.dumps(collectDecisionsPerModel)) # 6
    resultsRF.append(perModelProbPandas) # 7
    resultsRF.append(json.dumps(perModelPredPandas)) # 8
    resultsRF.append(json.dumps(target_names)) # 9
    resultsRF.append(json.dumps(collectStatisticsPerModel)) # 10
    resultsRF.append(json.dumps(collectInfoPerModelPandas)) # 11
    resultsRF.append(json.dumps(keepOriginalFeatures)) # 12
    resultsRF.append(json.dumps(yPredictTestListPandas)) # 13
    resultsRF.append(json.dumps(collectLocationsAllPerSorted)) # 14
    resultsRF.append(json.dumps(totalfpList)) # 15
    resultsRF.append(json.dumps(totalfnList)) # 16

    return resultsRF
