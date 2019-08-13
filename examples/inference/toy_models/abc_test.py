from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error


sim = simulate

# true_param = [0.2,-0.13]
# data = simulate(true_param)
# n=1000000
# train_thetas = np.array(prior(n=n))
# train_ts = np.expand_dims(np.array([simulate(p,n=100) for p in train_thetas]),2)
# validation_thetas = np.array(prior(n=10000))
# validation_ts = np.expand_dims(np.array([simulate(p,n=100) for p in validation_thetas]),2)

true_param, data, train_thetas, train_ts, validation_thetas, validation_ts, test_thetas, test_ts = load_data('moving_average2')



# set up the cnn model
#cnn = CNNModel(input_shape=(100,1), output_shape=(2))
cnn = PEN_CNNModel(input_shape=(100,1), output_shape=(2), pen_nr=11)
#cnn = ANNModel(input_shape=(100,1), output_shape=(2))

# cnn.load_model()

cnn.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
          plot_training_progress=False)

cnn.load_model()
validation_pred = np.array([cnn.predict(validation_ts[i*100:(i+1)*100]) for i in range(100)])
validation_pred = np.reshape(validation_pred,(-1,2))


print("mean squared error: ", np.mean((validation_thetas-validation_pred)**2))

#ABC algorithm

new_para = np.array(prior(n=500000))
new_data = np.expand_dims(np.array([simulate(p,n=100) for p in new_para]),2)
new_pred = cnn.predict(new_data)
mean_dev = np.mean(np.linalg.norm(new_para-new_pred, axis=1))
print("mean deviation: ", mean_dev)


data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
data_pred = cnn.predict(data_exp)
dist = np.linalg.norm( new_pred - data_pred,axis=1)
accepted_ind = np.argpartition(dist,500)[0:500]
accepted_para = new_para[accepted_ind]
accepted_mean = np.mean(accepted_para,axis=0)
accepted_std = np.std(accepted_para,axis=0)
print("posterior dev: ", accepted_mean-true_param)
print("posterior std: ", accepted_std)
data_pred = np.squeeze(data_pred)
print("dist shape: ", dist.shape)
accepted_dist = dist[accepted_ind]
print("accepted_dist shape: ", accepted_dist.shape)

print("accepted dist mean: ", np.mean(accepted_dist), ", max: ", np.max(accepted_dist), ", min: ", np.min(accepted_dist))


more_data =np.expand_dims( np.array([simulate(true_param) for i in range(10)]),2)


more_pred = cnn.predict(more_data)

more_pred_dev = np.linalg.norm(np.squeeze(more_pred)-true_param, axis=1)
print("more pred error: ", more_pred_dev)
print("mean more pred error: ", np.mean(more_pred_dev))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.axis('equal')
# col = np.zeros((500,3))
# col[:,1]=(accepted_dist - np.min(accepted_dist)) / (np.max(accepted_dist)-np.min(accepted_dist))
# circle1 = plt.Circle((true_param[0],true_param[1]),mean_dev,color='r', fill = False)
# plt.gcf().gca().add_artist(circle1)
plt.scatter(accepted_para[:, 0], accepted_para[:, 1], color="green", s=2)
plt.scatter(true_param[0],true_param[1], color="gray", marker="x")
plt.scatter(accepted_mean[0],accepted_mean[1], color="red", marker="x")
plt.scatter(data_pred[0],data_pred[1], color="gray", marker="o")
# plt.scatter(more_pred[:,0],more_pred[:,1], color="gold", marker="o")



plt.plot([-2,2,0,-2],[-1,-1,1,-1],color="red")
plt.plot([-2,2,0,-2],[-1,-1,1,-1],color="red")


plt.show()


# ann = ANNModel(input_shape=100, output_shape=(2))
#
# ann.train(inputs=train_ts[:,:,0], targets=train_thetas,validation_inputs=validation_ts[:,:,0],validation_targets=validation_thetas,
#                save_as='saved_models/dnn',plot_training_progress=False)
#
#



# Set up ABC
#  abc_instance = abc_inference.ABC(data, sim, epsilon=0.1, prior_function=prior,
#                                   summaries_function=bs_stat)
#
#  Perform ABC; require 30 samples
# results =  abc_instance.infer(num_samples=200, batch_size=100)