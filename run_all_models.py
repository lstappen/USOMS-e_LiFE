# Increase reproducibility
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)
import torch
torch.manual_seed(0)

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import os 

# Own
import configs
import data
import pos
import bert
import models
from train import train
import utils
import features


print(tf.__version__)
print(tf.keras.__version__)

# setting device on GPU if available, else CPU
T_DEVICE = 'cpu'
if T_DEVICE == 'cpu':
    device = torch.device(T_DEVICE)
else:
    # auto or gpu: check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, ' - ',torch.cuda.get_device_name(0) )

story_selector = ['st1','st2','st3']
task_name = 'ComParE2020_USOMS-e'
executable_models = ['frozen-bert-gmax','frozen-bert-rnnatt','frozen-bert-pos-fuse-rnnatt']
labels = ['V_cat_no', 'A_cat_no'] #'V_self_cat_no', 'A_self_cat_no', 'V_exp_cat_no', 'A_exp_cat_no'
partitions = ['train','devel','test']

for model_name in executable_models:
    print('-- ',model_name,' --')
    for label in labels: 
        # load config
        config = configs.set_config(label, story_selector, model_name)
        torch.cuda.empty_cache()

        # load data
        df, temp_numpy_data_splits, labels_df_cat, labels_numpy_hot = data.get_data(config)

        # create and add part-of-speech tagging
        if config.pos_embedding:
            if config.verbose > 0:
                print("Create pos embedding")

            part_of_speech = pos.add_pos_embeddings(task_name = task_name
                                                    , temp_numpy_data_splits = temp_numpy_data_splits
                                                    , MAX_LEN = config.max_seq_length
                                                    , device = device
                                                    , verbose = config.verbose)

            for key in labels_numpy_hot.keys():
                labels_numpy_hot[key] =  [labels_numpy_hot[key],labels_numpy_hot[key]]

        y_train, y_devel, y_test = labels_numpy_hot['train'], labels_numpy_hot['devel'], labels_numpy_hot['test']
        y_train_df, y_devel_df, y_test_df = labels_df_cat['train'], labels_df_cat['devel'], labels_df_cat['test']

        # calculate bert embedding, if not exist
        for split_key in temp_numpy_data_splits.keys():
            bert.calc_bert(config = config, task_name = task_name + '_' + "_".join(story_selector) + '_' + config.text_column
                            , temp_numpy_data_splits = temp_numpy_data_splits
                            , split_key = split_key, device = device)

        # load precalcuated bert features
        data_dict = {}
        for split_key in temp_numpy_data_splits.keys():
            bert_embedding_path = os.path.join(os.getcwd(), 'embeddings', 'bert')
            data_dict[split_key] = bert._load_precalculated_bert_embeddings(config = config
                                                                            , task_name = task_name + '_' + "_".join(story_selector) + '_' + config.text_column
                                                                            , bert_embedding_path = bert_embedding_path
                                                                            , split_key = split_key)

            if config.pos_embedding:
                data_dict[split_key] = [data_dict[split_key], part_of_speech[split_key]]

        X_train, X_devel, X_test = data_dict['train'], data_dict['devel'], data_dict['test']

        # select and create model
        if config.model_name == 'frozen-bert-gmax':
            model = models.create_bert_GloMaxPoo(config)
        elif config.model_name == 'frozen-bert-crnn-gmaxpool':
            model = models.create_bert_CRnnMax(config)
        elif config.model_name == 'frozen-bert-rnnatt':
            model = models.create_bert_rnn_att(config)
        elif config.model_name == 'frozen-bert-pos-fuse-rnnatt':
            model = models.create_bert_pos_fuse(config)
        else:
            print('Model type not found.')
            exit()


        model, history = train(config, model, X_train, y_train, X_devel, y_devel, y_train_df)

        utils.export_results(model, config, X_train, X_devel, X_test, y_train, y_devel, y_test)

        utils.visualise_training(config, history)

        features_dic = features.output_model_features(config, task_name, model, data_dict, df)

        features.score_features_with_SVM(config = config, experiment_name = config.experiment_name
                                        , task_name = task_name, features_dic = features_dic
                                        , labels_df_cat = labels_df_cat, label = label)

features.feature_fusion_and_scoring(config = config, task_name = task_name
                                    , executable_models = executable_models, labels = labels
                                    , partitions = partitions, labels_df_cat = labels_df_cat)