from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import csv
import os
import pandas as pd

def output_model_features(config, task_name, model, data, df):
   
    extraction_model = Model(inputs=model.input,
                             outputs=model.get_layer('features').output)

    features_dic = {}
    for partition_ in data.keys():
        features = extraction_model.predict(data[partition_])
        feat_header = ['filename'] + ['feature_' + str(i) for i in range(features.shape[1])]
        feature_df = pd.DataFrame(features)
        IDstory = df[df.partition.str.match(partition_)]['ID_story'].astype(str)
        IDstory.reset_index(drop=True, inplace=True)
        feature_df.reset_index(drop=True, inplace=True)
        feature_df_concat = pd.concat([IDstory, feature_df], axis=1)

        feature_file_name = task_name + '.' + config.model_name + '.' + config.label  + '.' + partition_ + '.csv'
        feature_df_concat.to_csv(os.path.join(config.features_path,feature_file_name)
                                      , header=feat_header
                                      , index=False)
        feature_df_concat.to_csv(os.path.join(config.overall_features_path,feature_file_name)
                                      , header=feat_header
                                      , index=False)
        
        features_dic[partition_] = features
        
    return features_dic
 
def feature_fusion_and_scoring(config, task_name, executable_models, labels, partitions, labels_df_cat):

    print('fuse exported features ', executable_models)
    for label in labels:
        features_dic = {}
        for partition_ in partitions:
            features = []
            print('--', label, ' - ', partition_)
            for model_name in executable_models:
                feature_file_name = task_name + '.' + model_name + '.' + label  + '.' + partition_ + '.csv' 
                feature_path_file = os.path.join(config.overall_features_path, feature_file_name)
                data = pd.read_csv(feature_path_file, header=None).values[1:]
                # only for the first file
                if model_name in executable_models[1:]:
                    data = data[:,1:]
                features.append(data)
            fused_features =  np.concatenate((features), axis=1)
            
            if config.verbose > 0:
                print('-- fused features shape: ',fused_features.shape)
            export = pd.DataFrame(fused_features)
            feat_header = ['filename'] + ['feature_' + str(i) for i in range(fused_features.shape[1] -1 )]
            feature_file_name = task_name + '.' + 'fused' + '.' + label + '.' + partition_ + '.csv'                                        
            export.to_csv(os.path.join(config.overall_features_path,feature_file_name)
                                          , header=feat_header
                                          , index=False)
            
            features_dic[partition_] = fused_features[:,1:]
        
        # To-DO: Clean, config and labels_df_cat is from last experiment run
        #        since all accessed parameters should be the same
        score_features_with_SVM(config, config.model_name_fusion, task_name, features_dic, labels_df_cat, label)

def score_features_with_SVM(config, experiment_name, task_name, features_dic, labels_df_cat, label):

    # score representations with SVM on different complexity levels
    complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
    X_train, X_devel, X_test = features_dic['train'], features_dic['devel'], features_dic['test']
    y_train, y_devel, y_test = labels_df_cat['train'], labels_df_cat['devel'], labels_df_cat['test']
    
    if config.test_mode:
        classes = list(set(np.concatenate((y_train, y_devel, y_test)).tolist()))
    else:
        classes = list(set(np.concatenate((y_train, y_devel)).tolist()))

    X_traindevel = np.concatenate((X_train, X_devel))
    y_traindevel = np.concatenate((y_train, y_devel))
    
    # Feature normalisation
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_devel = scaler.transform(X_devel)

    # for test
    X_traindevel = scaler.fit_transform(X_traindevel)
    X_test = scaler.transform(X_test)
        
    # Upsampling / Balancing
    if config.verbose > 0:
        print('Upsampling ... ')
    num_samples_train = []
    num_samples_traindevel = []
    for label in classes:
        num_samples_train.append(len(y_train[y_train == label]))
        num_samples_traindevel.append(len(y_traindevel[y_traindevel == label]))
        
    for label, ns_tr, ns_trd in zip(classes, num_samples_train, num_samples_traindevel):
        factor_tr = np.max(num_samples_train) // ns_tr
        X_train = np.concatenate((X_train, np.tile(X_train[y_train == label], (factor_tr - 1, 1))))
        y_train = np.concatenate((y_train, np.tile(y_train[y_train == label], (factor_tr - 1))))
        factor_trd = np.max(num_samples_traindevel) // ns_trd
        X_traindevel = np.concatenate((X_traindevel, np.tile(X_traindevel[y_traindevel == label], (factor_trd - 1, 1))))
        y_traindevel = np.concatenate((y_traindevel, np.tile(y_traindevel[y_traindevel == label], (factor_trd - 1))))
    
    # Train SVM model with different complexities and evaluate
    uar_scores = []
    results = {}
    for comp in complexities:
        #print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append(recall_score(y_devel, y_pred, average='macro')* 100)
        # print('UAR on Devel {0:.1f}'.format(uar_scores[-1] ))
        results['devel_recall_score_' + str(comp)] = round(uar_scores[-1],4)
        # print('Confusion matrix (Devel):')
        # print(confusion_matrix(y_devel, y_pred))
        results['devel_confusion_matrix_' + str(comp)] = confusion_matrix(y_devel, y_pred)
            
    optimum_complexity = complexities[np.argmax(uar_scores)]
    print('\nSVM - Optimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(
              optimum_complexity, np.max(uar_scores)))
    results['devel_recall_score_opt_' + str(optimum_complexity)] = results['devel_recall_score_' + str(optimum_complexity)]
    results['devel_confusion_matrix_opt_' + str(optimum_complexity)] = results['devel_confusion_matrix_' + str(optimum_complexity)]
        
    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_traindevel, y_traindevel)
    y_pred = clf.predict(X_test)
    results['test_recall_score_' + str(optimum_complexity)] = round(recall_score(y_test, y_pred, average='macro')  * 100,4) if config.test_mode else np.NaN
    results['test_confusion_matrix_' + str(optimum_complexity)] = confusion_matrix(y_test, y_pred) if config.test_mode else np.NaN
    print('SVM - UAR on Test {0:.1f}'.format(results['test_recall_score_' + str(optimum_complexity)]))
    
    # overall result file
    content = [config.story_selector
               , experiment_name
               , label
               , str(optimum_complexity)
               , results['devel_recall_score_opt_' + str(optimum_complexity)]
               , results['test_recall_score_' + str(optimum_complexity)]
               , results['devel_confusion_matrix_opt_' + str(optimum_complexity)]
               , results['test_confusion_matrix_' + str(optimum_complexity)]
              ]
    
    overall_result_file = os.path.join(os.getcwd(),config.overall_results)
    if not os.path.exists(overall_result_file):
        with open(overall_result_file, 'w', newline='') as file:
            writer = csv.writer(file)       
            writer.writerow(["story_selector","experiment_name","label","SVM-C","dev-uar","test-uar","dev-confusion-matrix","test-confusion-matrix"])
            writer.writerow(content)
    else:
        with open(overall_result_file, 'a', newline='') as file:
            writer = csv.writer(file)  
            writer.writerow(content)
    
    # exper. result file
    with open(os.path.join(config.svm_results_path, task_name+'.csv'), 'w+', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

        if config.verbose > 0:
            for k,v in results.items():
                print(' - ',k, ': ', v)
        #else:
            #print('recall_score', ': ', results['recall_score'])