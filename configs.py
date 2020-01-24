from collections import namedtuple
import csv
from datetime import datetime
import os

def set_config(label, story_selector, model_name):

    print("load configurations...")
    def _dict_to_struct(obj):
        obj = namedtuple("Configuration", obj.keys())(*obj.values())
        return obj

    def make_dirs_safe(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    config = dict()

    # overall
    config['start'] =  str(datetime.now()).replace(':','_')
    config['experiment_name'] = model_name # reactivate to test the same model with different parameters 
    config['model_name'] = model_name
    # 'frozen-bert-gmax'
    # 'frozen-bert-rnnatt'
    # 'frozen-bert-pos-fuse-rnnatt'
    config['model_name_fusion'] = 'frozen-bert-fusion'
    config['verbose'] = 1
    
    # data
    config['story_selector'] = story_selector 
    config['test_mode'] = False
    if not config['test_mode']:
        config['source_file'] = os.path.join('data','transcripted_text.'+ "_".join(config['story_selector']) +'.all.csv')
    else:
        config['source_file'] = os.path.join('data','transcripted_text.'+ "_".join(config['story_selector']) +'.confidential.csv')

    config['text_column'] = 'hand_transcription'
    config['label'] = label

    # preprocessing
    config['no_labels'] = 3
    config['y_dim'] = config['no_labels']
    config['max_seq_length'] = 500 # padding/tunct.

    # network
    ## Text model configuration.

    ## pos
    if 'pos' in config['model_name']:
        config["pos_embedding"] = True
        config["pos_embedding_size"] = 10 # compression
        config["pos_tokenizer_size"] = 54 # pre-calculated
    else:
        config["pos_embedding"] = False

    ## rnn
    config["rnn_hidden_units"] = 100
    config["rnn_dropout"] = 0.0

    ## dense
    config["dense_dropout"] = 0.0
    config["activation_function"] = "relu"
    config["activation_function_features"] = "sigmoid"
    config["activation_function_final"] = "softmax"

    # experiment configuration.
    config["batch_size"] = 50
    config["patience"] = 40
    config['task_type'] = 'classification'
    config['num_epochs'] = 250
    config["learning_rate"] = 0.0001
    config['eval_frequency'] = 10

    # export
    config['output_path'] = os.path.join(os.getcwd(), 'experiments',config['experiment_name']+'_'+config['start']+'_'+config['label'])
    config["features_path"] = os.path.join(config['output_path'], "features")
    config["checkpoint_path"] = os.path.join(config['output_path'], "ckpt") 
    config["results_path"] = os.path.join(config['output_path'], "results")
    config["svm_results_path"] = os.path.join(config['output_path'], "svm_results") 
    config["graphs_path"] = os.path.join(config['output_path'], "graphs") 
    config["config_path"] = os.path.join(config['output_path'], "config")
    config["overall_features_path"] = os.path.join(os.getcwd(), "features")
    config['overall_results'] = 'overall_results.csv'

    print('.. finished')

    print(" - create experiment folder structure")
    for k in config.keys():
        if 'path' in k:
            make_dirs_safe(config[k])

    config = _dict_to_struct(config)
    
    with open(os.path.join(config.config_path,'config.csv'), 'w+', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        dict_ = config._asdict()
        for key, value in dict_.items():
            writer.writerow([key, value])
            
    return config
