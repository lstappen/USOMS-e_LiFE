import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

def _load_bert_model_tokenizer(device):
    from pytorch_transformers import BertModel, BertTokenizer
    
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-german-cased'

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    # move tensors to GPU
    if device:
        if device.type == 'cuda':
            print('  - move model to GPU')
            model.to(device)

    # model for forward propagation only - "frozen"
    model.eval() 

    return tokenizer, model

def _save_precalculated_bert_embeddings(task_name, bert_embedding_path, split_key, embeddings, verbose):
    
    if verbose > 0:
        print(" - store bert embeddings")
    if not os.path.exists(bert_embedding_path):
        os.makedirs(bert_embedding_path)
    bert_embeddings_name = os.path.join(bert_embedding_path, task_name + '_' + split_key + '.npy')
    np.save(bert_embeddings_name, embeddings)

def _forward_propagation_bert(task_name, temp_numpy_data_splits, bert_embedding_path, split_key, device, verbose):

    tokenizer, model = _load_bert_model_tokenizer(device)
    
    # To-DO: Transfromer version with seq. length of max story
    # Limited by this earlier bert version. Has to be < 512.
    # Other Transformers provide longer sequence length
    MAX_LEN = 500
    add_special_tokens = False

    # Pad our input tokens
    if verbose > 0:
        print(" - [Bert] tokenize & pad to %d" % MAX_LEN)

    tokenized_texts = tf.keras.preprocessing.sequence.pad_sequences(
        [tokenizer.encode(txt[:MAX_LEN], add_special_tokens=add_special_tokens) for txt in temp_numpy_data_splits[split_key]]
        , maxlen=MAX_LEN
        , dtype="long"
        , truncating="pre"
        , padding="pre")
    
    if verbose > 0:
        print(" - [Bert] forward propagation")

    features = []
    embeddings_batch = []
    EMBEDDING_BATCH_SIZE = 100
    i = 0

    if verbose > 0:
        print(' - padded input text: ' + str(tokenized_texts.shape))

    while i < tokenized_texts.shape[0]:
        if (i + EMBEDDING_BATCH_SIZE) > tokenized_texts.shape[0]:
            i_max = tokenized_texts.shape[0]
        else:
            i_max = i + EMBEDDING_BATCH_SIZE

        tokenized_texts_batch = tokenized_texts[i:i_max]
        
        #Additional Info when using cuda
        if device:
            if device.type == 'cuda':
                if verbose > 0:
                    print("  - move %d to %d datapoints to GPU tensors"% (i,i_max))
                # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
                input_ids = torch.tensor(tokenized_texts_batch)
                input_ids_tensor = input_ids.to(device)  

                with torch.no_grad():
                    last_hidden_states = model(input_ids_tensor)[0]  # Models outputs are now tuples
            else:
                if verbose > 1:
                    print("[WARN] No GPU available. Using CPU")
                # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
                input_ids_tensor = torch.tensor(tokenized_texts_batch)
        else:
            if verbose > 1:
                print("[WARN] No device detected. Using CPU.")
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            input_ids_tensor = torch.tensor(tokenized_texts_batch)

        with torch.no_grad():
            last_hidden_states = model(input_ids_tensor)[0]  # Models outputs are now tuples
        embeddings_batch.append(np.array(last_hidden_states.cpu().numpy(), dtype=np.float))
            
        i = i_max
            
    embeddings = np.concatenate(embeddings_batch, axis=0)
    
    _save_precalculated_bert_embeddings(task_name, bert_embedding_path, split_key, embeddings, verbose)
    
    print(split_key + ' with ' + str(embeddings.shape) + 'stored.')

def _load_precalculated_bert_embeddings(config, task_name, bert_embedding_path, split_key):

    bert_embeddings = os.path.join(bert_embedding_path, task_name + '_' + split_key + '.npy')
    
    if os.path.isfile(bert_embeddings):
        if config.verbose > 0:
            print(" - loaded bert for %s from file" % (task_name + '_' + split_key))
        return np.load(bert_embeddings)
    else:
        print("[WARNING] No file exists %s" % bert_embeddings)
        return None
    
def calc_bert(config, task_name, temp_numpy_data_splits, split_key, device):

    print(" - embeddingnize text of %s" % task_name)
    # test load
    bert_embedding_path = os.path.join(os.getcwd(), 'embeddings', 'bert')

    check_bert_file_exists = os.path.join(bert_embedding_path, task_name + '_' + split_key + '.npy')

    if os.path.isfile(check_bert_file_exists):
        print(" -- found file %s" % check_bert_file_exists)
        print(" -- [WARNING] Pre-calculated embeddings are loaded for bert.")
        print(" -- [WARNING] And work only with the randomnizer created.")
        print(" -- [INFO] Please delete bert embeddings and start again, if source data were changed.")

        temp_numpy_data_splits_new = _load_precalculated_bert_embeddings(
                                            config, 
                                            task_name,
                                            bert_embedding_path,
                                            split_key)
    else:
        print("[INFO] calculate embeddings")
        temp_numpy_data_splits_new = _forward_propagation_bert(task_name
                                                                      , temp_numpy_data_splits
                                                                      , bert_embedding_path
                                                                      , split_key
                                                                      , device
                                                                      , config.verbose)
    return temp_numpy_data_splits_new


