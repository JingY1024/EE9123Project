import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from tqdm import tqdm


import kerastuner as kt
from dataset import generate_dataset, PurgedGroupTimeSeriesSplit, set_all_seeds


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, train_data, batch_size=32, epochs=1,callbacks=None):
        val_losses = []
        for train_idx, test_idx in train_data:
            X_train, X_test = [x[train_idx] for x in X], [x[test_idx] for x in X]
            y_train, y_test = [a[train_idx] for a in y], [a[test_idx] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]
            
            network = self.hypermodel.build(trial.hyperparameters)
            hist = network.fit(X_train,y_train,
                      validation_data=(X_test,y_test),
                      epochs=epochs,
                        batch_size=batch_size,
                      callbacks=callbacks)
            
            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id, {k:np.mean(val_losses[:,i]) for i,k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, network)

def create_AE(input_dim,output_dim,noise=0.02):
    if is_encoder == True:
        i = Input(input_dim)
        encoded = BatchNormalization()(i)
        encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(64,activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.0005))(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dropout(0.2)(encoded)
    decoded = Dense(input_dim,name='decoded', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.0005))(decoded)
    x = Dense(32,activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.0005))(decoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim,activation='sigmoid',name='label_output', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.005))(x)
    
    encoder = Model(inputs=i,outputs=decoded)
    AE = Model(inputs=i,outputs=[decoded,x])
    
    AE.compile(optimizer=Adam(0.001),loss={'decoded':'mse','label_output':'binary_crossentropy'})
    return AE, encoder

def create_model(tune_hyper_param,input_dim,output_dim,encoder):
    inputs = Input(input_dim)
    
    x = encoder(inputs)
    x = Concatenate()([x,inputs])
    x = BatchNormalization()(x)
    x = Dropout(tune_hyper_param.Float('init_dropout',0.0,0.5))(x)
    
    for i in range(tune_hyper_param.Int('num_layers',1,3)):
        x = Dense(tune_hyper_param.Int('num_units_{i}',64,256), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=tune_hyper_param.Float('l1', 0.00001, 0.01), l2=tune_hyper_param.Float('l2', 0.00001, 0.01)))(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(tune_hyper_param.Float(f'dropout_{i}',0.0,0.5))(x)
    x = Dense(output_dim,activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=tune_hyper_param.Float('l1', 0.00001, 0.01), l2=tune_hyper_param.Float('l2', 0.000001, 0.01)))(x)
    network = Model(inputs=inputs,outputs=x)
    network.compile(optimizer=Adam(tune_hyper_param.Float('lr',0.00001,0.1,default=0.001)),loss=BinaryCrossentropy(label_smoothing=tune_hyper_param.Float('label_smoothing',0.0,0.1)),metrics=[tf.keras.metrics.AUC(name = 'auc')])
    return network

TRAINING = False
is_encoder = True
fold_num = 5
rand_seed = 42
set_all_seeds(rand_seed)


X, y, _, _, train, test, features, feature_mean = generate_dataset()

AE, encoder = create_AE(X.shape[-1],y.shape[-1],noise=0.02)

if TRAINING:
    AE.fit(X,
           (X,y),
           epochs=1000,
           batch_size=4096, 
           validation_split=0.1,
           callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
else:
    encoder.load_weights('./encoder.hdf5')

encoder.trainable = True


joint_model = lambda tune_hyper_param: create_model(tune_hyper_param,X.shape[-1],y.shape[-1],encoder)

tuner = CVTuner(
        hypermodel=joint_model,
        oracle=kt.oracles.BayesianOptimization(
            objective= kt.Objective('val_auc', direction='max'),
            num_initial_points=4,
            max_trials=20)
        )

if TRAINING:
    group_Kfold = PurgedGroupTimeSeriesSplit(n_splits = fold_num, group_gap=20)
    train_data = list(group_Kfold.split(y, groups=train['date'].values))
    tuner.search((X,),
                 (y,),
                 train_data=train_data,
                 batch_size=4096,
                 epochs=200,
                 callbacks=[EarlyStopping('val_auc', mode='max',patience=3)])
    tune_hyper_param  = tuner.get_best_hyperparameters(1)[0]
    pd.to_pickle(tune_hyper_param,f'./best_hp_{rand_seed}.pkl')
    for fold, (train_idx, test_idx) in enumerate(train_data):
        network = joint_model(tune_hyper_param)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        network.fit(X_train,
                  y_train,
                  validation_data=(X_test,y_test),
                  epochs=200,
                  batch_size=4096,
                  callbacks=[EarlyStopping('val_auc',mode='max',patience=10,restore_best_weights=True)])
        network.save_weights(f'./model_{rand_seed}_{fold}.hdf5')
        network.compile(Adam(tune_hyper_param.get('lr')/100),loss='binary_crossentropy')

        network.fit(X_test,y_test,epochs=3,batch_size=4096)
        
        network.save_weights(f'./model_{rand_seed}_{fold}_finetune.hdf5')
    tuner.results_summary()
else:
    network_list = []
    tune_hyper_param = pd.read_pickle(f'./best_hp_{rand_seed}.pkl')
    for f in range(fold_num):
        network = joint_model(tune_hyper_param)
        network.load_weights(f'./model_{rand_seed}_{f}_finetune.hdf5')
        network_list.append(network)

is_encoder = False

if not  TRAINING:
    network_list = network_list[-2:]
    threshold = 0.5
    test = test.reset_index()
    score_map = {}
    for idx in range(len(test)):
        row = test[idx: idx+1]
        if row['weight'].item() > 0:
            weight = row['weight'].item()
            resp = row['resp'].item()
            date = row['date'].item()
            feature_data = row.loc[:, features].values
            if np.isnan(feature_data[:, 1:].sum()):
                feature_data[:, 1:] = np.nan_to_num(feature_data[:, 1:]) + np.isnan(feature_data[:, 1:]) * feature_mean
            pred = np.mean([network(feature_data, training = False).numpy() for network in network_list], axis=0)
            pred = np.median(pred)
            action = np.where(pred >= threshold, 1, 0).astype(int)
            score = weight * resp * action
            if date in score_map.keys():
                score_map[date] += score
            else:
                score_map[date] = score
    print(score_map.keys())
    sum0 = 0
    sum1 = 0
    for key in score_map.keys():
        sum0 += score_map[key]
        sum1 += score_map[key] * score_map[key]
    t = sum0 / np.sqrt(sum1) * np.sqrt(250/len(score_map.keys()))
    u = min(max(t,0),6)*sum0
    print(t, u)