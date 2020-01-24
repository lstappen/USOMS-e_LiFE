import os
import tensorflow as tf
import pickle
import bert

def pos_tagger(tagger, text, MAX_LEN, device):
    import nltk

    # transform words to pos tags
    tokenizer, _ = bert._load_bert_model_tokenizer(device)
        
    # In the original paper, the authors used a length of 512.
    add_special_tokens = False    
        
    tokenized_texts = [tokenizer.encode(txt[:MAX_LEN], add_special_tokens=add_special_tokens) for txt in text]
    # clean text the same way as in proper encoding
    detokenized_texts = [tokenizer.decode(txt) for txt in tokenized_texts]

    tags = [tagger.tag(txt.split(' ')) for txt in detokenized_texts]
    tag_pos_only = []
    for tag_sentences in tags:
        tag_pos_only.append([tag[1] for tag in tag_sentences])

    return tag_pos_only    

def create_pos_tag_tokenizer(tagger, list_of_text, MAX_LEN, device, verbose):

    # One tokenizer over entire text to ensure token consistency
    text = list_of_text[0].tolist() + list_of_text[1].tolist() + list_of_text[2].tolist()
    pos_tags = pos_tagger(tagger, text, MAX_LEN, device)
                
    # create Tokeniser
    vocab_pos = list(set([item for sublist in pos_tags for item in sublist]))
    vocab_pos_size = len(vocab_pos) + 1 # empty
    
    # switch dict key value
    Tokenizer_pos = {v:k for k,v in dict(enumerate(vocab_pos,1)).items()}
    Tokenizer_pos['0EM'] = 0
    if verbose > 0:
        print('POS Tokenizer', Tokenizer_pos)
    
    return Tokenizer_pos, vocab_pos_size
    
def pos_tag_preparation(padded_text, tagger, Tokenizer_pos, MAX_LEN, device):

    pos_tags = pos_tagger(tagger, padded_text.tolist(), MAX_LEN, device)
                                         
    tokenized_pos_tags = []
    for pos_tags_sent in pos_tags:
        tokenized_pos_tags.append([Tokenizer_pos[tag] for tag in pos_tags_sent])
    
    padded_pos_tags = tf.keras.preprocessing.sequence.pad_sequences(tokenized_pos_tags
                    , maxlen=MAX_LEN
                    , dtype="long"
                    , truncating="pre"
                    , padding="pre")
    
    return padded_pos_tags

def add_pos_embeddings(task_name, temp_numpy_data_splits, MAX_LEN, device, verbose):
    
    list_of_text = list(temp_numpy_data_splits.values()) 
    
    # load/create tagger
    tagger =  create_load_pos_tagger(task_name, verbose)
    
    Tokenizer_pos, vocab_pos_size = create_pos_tag_tokenizer(
                                    tagger
                                    , list_of_text
                                    , MAX_LEN
                                    , device
                                    , verbose)

    pos = {}
    pos['train'] = pos_tag_preparation(temp_numpy_data_splits['train'], tagger, Tokenizer_pos, MAX_LEN, device)
    pos['devel']  = pos_tag_preparation(temp_numpy_data_splits['devel'], tagger, Tokenizer_pos, MAX_LEN, device)
    pos['test']  = pos_tag_preparation(temp_numpy_data_splits['test'], tagger, Tokenizer_pos, MAX_LEN, device)
    
    return pos

def create_load_pos_tagger(task_name, verbose):

    if verbose > 0:
        print(" - embeddingnize text of %s" % task_name)

    # test load
    pos_tagger_path = os.path.join(os.getcwd(), 'pos_tagger')
    check_pos_tagger_file_exists = os.path.join(pos_tagger_path, 'nltk_german_pos_tagger.pkl')

    if os.path.exists(check_pos_tagger_file_exists):
        if verbose > 0:
            print("Found file %s" % check_pos_tagger_file_exists)
            print("[WARNING] Pre-calculated nltk tagger is loaded.")

        with open(check_pos_tagger_file_exists, 'rb') as f:
            tagger = pickle.load(f)
            
    else:
        print("[WARNING] calculate nltk german pos tagger")
        tagger = _execute_pos_tagger_training(check_pos_tagger_file_exists)
        
    return tagger

def _execute_pos_tagger_training(check_pos_tagger_file_exists):
    from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
    # derived from https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
    # and https://github.com/ptnplanet/NLTK-Contributions
    
    import nltk
    print("Download corpus from https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/TIGERCorpus/download/start.html!")
    corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                     ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                     encoding='utf-8')
    
    tagged_sents = list(corp.tagged_sents())
    random.shuffle(tagged_sents)

    # set a split size: use 90% for training, 10% for testing
    split_perc = 0.1
    split_size = int(len(tagged_sents) * split_perc)
    train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

    tagger = ClassifierBasedGermanTagger(train=train_sents)

    print('Trained tagger result: ', tagger.evaluate(test_sents))

    with open(check_pos_tagger_file_exists, 'wb') as f:
        pickle.dump(tagger, f, protocol=2)
        
    return tagger