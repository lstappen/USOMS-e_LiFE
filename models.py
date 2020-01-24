import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization, Bidirectional, Embedding, Flatten, Concatenate
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam


class Attention(tf.keras.Model):
    # very basic attention layer
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) # bias could be deactivated here
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
 
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.keras.backend.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.backend.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def general_model_part(model,learning_rate, verbose):
    
    if verbose > 0:
        print(model.summary())
    
    # optimizer
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model

def general_model_part_multiout(model,learning_rate, verbose):
    
    #including aux loss
    if verbose > 0:
        print(model.summary())
    
    # optimizer
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # compile the model with weighted aux loss
    model.compile(loss='categorical_crossentropy'
                  , metrics=['accuracy']
                  , loss_weights=[1., 0.2]
                  , optimizer=opt)

    return model
              
# TO-DO: remove static input_shape
def create_bert_GloMaxPoo(config):
    
    inputs = Input(name='text', shape=(500, 768), dtype='float32')  

    inner = GlobalMaxPooling1D(data_format='channels_last')(inputs) 
    inner = Dense(512, activation=config.activation_function, kernel_initializer='he_normal')(inner) 
    inner = Dropout(config.dense_dropout)(inner)
    inner = Dense(512, activation=config.activation_function_features, name='features')(inner)     
    output = Dense(config.no_labels, kernel_initializer='he_normal',name='final', activation = config.activation_function_final)(inner) 
    model = Model(inputs=inputs, outputs=output)

    model = general_model_part(model,config.learning_rate, config.verbose)
    
    return model

def create_bert_CRnnMax(config):
    
    inputs = Input(name='text', shape=(500, 768), dtype='float32') 

    inner = Conv1D(filters = 256, kernel_size = config.kernel_size, strides=config.strides, padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation(config.activation_function)(inner)
    inner = MaxPooling1D(config.pool, name='max1')(inner) 

    inner = Conv1D(filters = 128, kernel_size = config.kernel_size, strides=config.strides,  padding='same', name='conv5', kernel_initializer='he_normal')(inner) 
    inner = BatchNormalization()(inner)
    inner = Activation(config.activation_function)(inner)
    inner = MaxPooling1D(config.pool, name='max2')(inner) 

    # reshape and GRU/LSTM + Att for CRNN+Att
    inner = GlobalMaxPooling1D(data_format='channels_last')(inner)  
    inner = Dense(512, activation=config.activation_function, kernel_initializer='he_normal')(inner)  
    inner = Dense(512, activation=config.activation_function_features, name='features')(inner)     
    output = Dense(config.no_labels, kernel_initializer='he_normal',name='final', activation = config.activation_function_final)(inner) 
    
    model = Model(inputs=inputs, outputs=output)

    model = general_model_part(model,config.learning_rate)
    
    return model

def create_bert_rnn_att(config):
    
    inputs = Input(name='text', shape=(500, 768), dtype='float32') 

    lstm, forward_h, forward_c, backward_h, backward_c =  Bidirectional(
                                    LSTM(config.rnn_hidden_units
                                       , return_sequences=True
                                       , return_state=True
                                       , recurrent_activation=config.activation_function_features
                                       , recurrent_initializer='glorot_uniform'                              
                                      ))(inputs)

    state_h = Concatenate()([forward_h, backward_h])

    att = Attention(units = 512)
    context_vector, attention_weights = att(lstm, state_h)
    
    inner = Dense(512, activation=config.activation_function, kernel_initializer='he_normal')(context_vector)  
    inner = Dense(512, activation=config.activation_function_features, name='features')(inner)     
    output = Dense(config.no_labels, kernel_initializer='he_normal',name='final', activation = config.activation_function_final)(inner) 
    model = Model(inputs=inputs, outputs=output)

    model = general_model_part(model,config.learning_rate, config.verbose)
    
    return model

def create_bert_pos_fuse(config):
    
    input_pos = Input(name='pos', shape=(500,), dtype='int32')
    inner_pos = Embedding(config.pos_tokenizer_size
                          , config.pos_embedding_size
                          , trainable=True                            
                         )(input_pos)
    
    inner_pos = Flatten()(inner_pos)
    inner_pos = Dense(100, activation=config.activation_function_features)(inner_pos)
    
    auxiliary_output = Dense(config.no_labels, activation=config.activation_function_final, name='aux_pos_output')(inner_pos)

    inputs = Input(name='text', shape=(500, 768), dtype='float32')
    lstm, forward_h, forward_c, backward_h, backward_c =  Bidirectional(
                                    LSTM(config.rnn_hidden_units
                                       , return_sequences=True
                                       , return_state=True
                                       , recurrent_activation=config.activation_function_features
                                       , recurrent_initializer='glorot_uniform'                              
                                      ))(inputs)

    state_h = Concatenate()([forward_h, backward_h])

    att = Attention(units = 512)
    context_vector, attention_weights = att(lstm, state_h)
    
    fuse = Concatenate()([context_vector, inner_pos])
    inner = Dense(512, activation=config.activation_function, kernel_initializer='he_normal')(fuse)  
    inner = Dense(512, activation=config.activation_function_features, name='features')(inner)     
    output = Dense(config.no_labels, kernel_initializer='he_normal',name='final', activation = config.activation_function_final)(inner) 

    model = Model(inputs=[inputs, input_pos], outputs=[output, auxiliary_output])

    model = general_model_part_multiout(model,config.learning_rate, config.verbose)
    return model