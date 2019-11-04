from sciope.models.cnn_regressor import CNNModel

import numpy as np
from create_summary_statistics import summarys
import pickle
import time

from load_data import load_spec

clay=[32,48,64,96]
# clay = [4,8,16]
num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
species = [6]

train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
train_sum = pickle.load(open('datasets/' + modelname + '/train_sum.p', "rb" ) )
validation_sum = pickle.load(open('datasets/' + modelname + '/val_sum.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]
test_sum = pickle.load(open('datasets/' + modelname + '/test_sum.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )[:,:,species]

obs_data = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )[:,:,species]

obs_sum = np.array([summarys(o) for o in obs_data])

# print("train sum shape: ", train_sum.shape)
# print("train ts shape: ", train_ts.shape)
#
# train_ind = np.where(train_sum[:,0] >= 0)[0]
# print("train ind shape: ", np.array(train_ind).shape)
# train_sum = train_sum[train_ind]
# train_ts = train_ts[train_ind]
# print("train sum shape: ", train_sum.shape)
# print("train ts shape: ", train_ts.shape)
#
# test_ind = np.where(test_sum[:,0] >= 0)[0]
# test_sum = test_sum[test_ind]
# test_ts = test_ts[test_ind]
#
# validation_ind = np.where(validation_sum[:,0] >= 0)[0]
# validation_sum = validation_sum[validation_ind]
# validation_ts = validation_ts[validation_ind]

nnm = CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(2), con_len=3, con_layers=clay,
               dense_layers=[100,100,100])#, dataname='wozeros')

nnm.load_model()


# history1 = nnm.train(inputs=train_ts, targets=train_sum,validation_inputs=validation_ts,validation_targets=validation_sum,
#           batch_size=32, epochs=20, val_freq=1, early_stopping_patience=20, plot_training_progress=False, verbose=1)
#

# history1 = nnm.train(inputs=train_ts, targets=train_sum,validation_inputs=validation_ts,validation_targets=validation_sum,
#           batch_size=32, epochs=40*10, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=1)
#
#
#
# history1 = nnm.train(inputs=train_ts, targets=train_sum,validation_inputs=validation_ts,validation_targets=validation_sum,
#           batch_size=4096, epochs=40*10, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=1)


train_pred = nnm.predict(train_ts)
train_mae = np.mean(abs(train_pred-train_sum),axis=0)
print("train_mae: ", train_mae)

test_pred = nnm.predict(test_ts)
test_mae = np.mean(abs(test_pred-test_sum),axis=0)
print("test_mae: ", test_mae)

print("test sum std: ", np.std(test_sum,0))
print("test sum mean: ", np.mean(test_sum,0))
print("optimal guessing deviation: ", np.mean(abs(test_sum - np.mean(test_sum,0)),0))

obs_pred = nnm.predict(obs_data)

print("observed data:")
for pred, true in zip(obs_pred,obs_sum):
    print("pred: ", pred, ", true: ", true)

print()
print("test samples:")

for i in range(25):
    print("test_sum: ", test_sum[i], ", test_pred: ", test_pred[i])