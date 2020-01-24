import pandas as pd
from tensorflow.keras.utils import to_categorical

def label_preparation(data,depth=3):
    
    # one hot encode
    return to_categorical(data,num_classes=depth)

def get_data(config):
    
    df = pd.read_csv(config.source_file)

    temp_numpy_data_splits = {
        'train':  df[df.partition.str.match('train')][config.text_column].astype(str).values
        ,'devel': df[df.partition.str.match('devel')][config.text_column].astype(str).values
        ,'test': df[df.partition.str.match('test')][config.text_column].astype(str).values
    }

    labels_df_cat = {
        'train': df[df.partition.str.match('train')][config.label], 
        'devel' : df[df.partition.str.match('devel')][config.label], 
        'test' : df[df.partition.str.match('test')][config.label] if config.test_mode else None
    }
        
    labels_numpy_hot = {
        'train': label_preparation(labels_df_cat['train'].values, config.no_labels),
        'devel' : label_preparation(labels_df_cat['devel'].values, config.no_labels),
        'test' : label_preparation(labels_df_cat['test'].values, config.no_labels) if config.test_mode else None
    }     

    num_of_batches = {}
    num_of_batches['train'] = int(len(temp_numpy_data_splits['train']) /config.batch_size) + 1
    num_of_batches['devel'] = int(len(temp_numpy_data_splits['devel']) /config.batch_size) + 1
    num_of_batches['test'] = int(len(temp_numpy_data_splits['test']) /config.batch_size) + 1
    
    return df, temp_numpy_data_splits, labels_df_cat, labels_numpy_hot