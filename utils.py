from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def print_measures(y, y_pred, name, config):
    # Print and export f1, precision, and recall scores
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    results = {}
    results['recall_score'] = round(recall_score(y, y_pred , average="macro")*100,4) 
    results['accuracy_score'] = round(accuracy_score(y, y_pred)*100,4) 
    results['precision_score'] = round(precision_score(y, y_pred , average="macro")*100,4) 
    results['f1_score'] = round(f1_score(y, y_pred , average="macro")*100,4) 
    results['confusion_matrix'] = confusion_matrix(y, y_pred)

    with open(os.path.join(config.results_path,name+'.csv'), 'w+', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

        print(name, ' measures')
        if config.verbose > 0:
            for k,v in results.items():
                print(' - ', k, ': ', v)
        else:
            print(' - recall_score', ': ', results['recall_score'])


def export_results(model, config, X_train, X_devel,X_test, y_train,y_devel,y_test):
    
    y_train_pred = model.predict(X_train)
    y_devel_pred = model.predict(X_devel)
    y_test_pred = model.predict(X_test)
    
    if config.pos_embedding:
        y_train, y_devel, y_test = y_train[0], y_devel[0], y_test[0]
        y_train_pred, y_devel_pred, y_test_pred =y_train_pred[0], y_devel_pred[0], y_test_pred[0]
    
    print_measures(y_train, y_train_pred,'train',config)
    print_measures(y_devel, y_devel_pred,'devel',config)
    if config.test_mode:
        print_measures(y_test, y_test_pred,'test',config)


def visualise_training(config, history):
    
    # summarize history for accuracy
    legend = [k for k in history.history.keys() if 'acc' in k]
    for k in legend:
        plt.plot(history.history[k])
    plt.title('model accuracy')
    plt.ylabel('accuracy') 
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(os.path.join(config.graphs_path,'acc.png'))
    # plt.show()
    
    # summarize history for loss
    legend = [k for k in history.history.keys() if 'loss' in k]
    for k in legend:
        plt.plot(history.history[k])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(os.path.join(config.graphs_path,'loss.png'))
    # plt.show()