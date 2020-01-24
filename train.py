from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os

def train(config, model, X_train, y_train, X_devel, y_devel, y_train_df):

    # calculate loss weights since the data set is umbalanced and no up/down-sampling is used
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train_df.values),
                                                     y_train_df.values)
    # replication for aux loss
    if config.pos_embedding:
        class_weights = [class_weights, class_weights]
    
    # export best model or weights (if custom layer (Att) since full model extraction is not supported in this tf version yet)
    checkpointer = ModelCheckpoint(filepath=os.path.join(config.checkpoint_path,'weights.best.hdf5')
                                   , verbose=config.verbose
                                   , save_weights_only= True if 'att' in config.model_name else False 
                                   , save_best_only=True)

    # stop training if validation loss does not improve for X rounds
    # restore most sucessful model
    stopper = EarlyStopping(monitor='val_loss', patience=config.patience
                            , verbose=0
                            , mode='min'
                            , baseline=None
                            , restore_best_weights=True)
    
    callbacks = [checkpointer,stopper]

    # train model
    history = model.fit(X_train, y_train
                        , batch_size=config.batch_size
                        , epochs=config.num_epochs
                        , validation_data=(X_devel, y_devel)
                        , callbacks=callbacks
                        , verbose=config.verbose
                        , class_weight=class_weights)
    return model, history