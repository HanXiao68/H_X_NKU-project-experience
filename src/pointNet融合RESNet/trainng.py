import numpy as np
import sys
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Add, concatenate
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
import h5py
from matplotlib.pyplot import imshow
import glob
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

from keras.models import load_model

points =  np.load('drive/Colab Notebooks/train_points.npy')
labels = np.load('drive/Colab Notebooks/train_labels.npy')
labels = labels.reshape((7481,24))
classes = np.load('drive/Colab Notebooks/train_classes.npy')

intermediate_output = np.load('drive/Colab Notebooks/intermediate_output.npy')
intermediate_output = np.squeeze(intermediate_output)
print(intermediate_output.shape)

def mat_mul(A, B):
    return tf.matmul(A, B)

# number of points in each sample
num_points = 2048

# number of categories
k = 3

# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
input_points = Input(shape=(num_points, 3))
x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(input_points)
#x = BatchNormalization()(x)
x = Convolution1D(128, 1, activation='relu')(x)
#x = BatchNormalization()(x)
x = Convolution1D(1024, 1, activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=num_points)(x)
x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
input_T = Reshape((3, 3))(x)

# forward net
g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
#g = BatchNormalization()(g)

# feature transform net
f = Convolution1D(64, 1, activation='relu')(g)
#f = BatchNormalization()(f)
f = Convolution1D(128, 1, activation='relu')(f)
#f = BatchNormalization()(f)
f = Convolution1D(1024, 1, activation='relu')(f)
#f = BatchNormalization()(f)
f = MaxPooling1D(pool_size=num_points)(f)
f = Dense(512, activation='relu')(f)
#f = BatchNormalization()(f)
f = Dense(256, activation='relu')(f)
#f = BatchNormalization()(f)
f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)

# forward net
g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = Convolution1D(64, 1, activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(128, 1, activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(1024, 1, activation='relu')(g)
#g = BatchNormalization()(g)

# global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)
global_feature = Flatten()(global_feature)
# point_net_cls
#c = Dense(512, activation='relu')(global_feature)
#c = BatchNormalization()(c)
#c = Dropout(rate=0.7)(c)
#c = Dense(256, activation='relu')(c)
#c = BatchNormalization()(c)
#c = Dropout(rate=0.7)(c)
#c = Dense(k, activation='softmax')(c)
#prediction = Flatten()(c)
# --------------------------------------------------end of pointnet

#Fusion

resnet_activation = Input(shape=(intermediate_output.shape[1],), name='intermediate_output')
f = Concatenate()([global_feature, resnet_activation])

#Definition of MLP Layer
f = Dense(512, activation='relu')(f)
f = Dense(128, activation='relu')(f)
f = Dense(128, activation='relu')(f)
boxes = Dense(labels.shape[-1])(f)
classes = Dense(classes.shape[-1])(f)


# print the model summary
model = Model(inputs=[input_points, resnet_activation], outputs=[boxes, classes])
print(model.summary())
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 2048, 3)      0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 2048, 3)      0           input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 2048, 64)     256         lambda_1[0][0]                   
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 2048, 64)     4160        conv1d_4[0][0]                   
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 2048, 64)     0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 2048, 64)     4160        lambda_2[0][0]                   
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, 2048, 128)    8320        conv1d_9[0][0]                   
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, 2048, 1024)   132096      conv1d_10[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_3 (MaxPooling1D)  (None, 1, 1024)      0           conv1d_11[0][0]                  
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1024)         0           max_pooling1d_3[0][0]            
__________________________________________________________________________________________________
intermediate_output (InputLayer (None, 2048)         0                                            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3072)         0           flatten_1[0][0]                  
                                                                 intermediate_output[0][0]        
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 512)          1573376     concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 128)          65664       dense_7[0][0]                    
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 128)          16512       dense_8[0][0]                    
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 24)           3096        dense_9[0][0]                    
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 3)            387         dense_9[0][0]                    
==================================================================================================
Total params: 1,808,027
Trainable params: 1,808,027
Non-trainable params: 0
__________________________________________________________________________________________________
None
Load Data:

index = np.load('permuted_indices.npy')

train_points = points[index[0:6750]]
dev_points = points[index[6750:7115]]
test_points = points[index[7115:]]

train_classes = classes[index[0:6750]]
dev_classes = classes[index[6750:7115]]
test_classes = classes[index[7115:]]

train_labels = labels[index[0:6750]]
dev_labels = labels[index[6750:7115]]
test_labels = labels[index[7115:]]

train_intermediate = intermediate_output[index[0:6750]]
dev_intermediate = intermediate_output[index[6750:7115]]
test_intermediate = intermediate_output[index[7115:]]

print(train_points.shape)
print(train_labels.shape)
print(train_classes.shape)
print(train_intermediate.shape)

print(dev_points.shape)
print(dev_labels.shape)
print(dev_classes.shape)
print(dev_intermediate.shape)

print(test_points.shape)
print(test_labels.shape)
print(test_classes.shape)
print(test_intermediate.shape)
(6750, 2048, 3)
(6750, 24)
(6750, 3)
(6750, 2048)
(365, 2048, 3)
(365, 24)
(365, 3)
(365, 2048)
(366, 2048, 3)
(366, 24)
(366, 3)
(366, 2048)
Training:

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)
  
  
#epoch number
epo = 450
# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)
# compile classification model
model.compile(optimizer='adam',
              loss=[smoothL1, 'mean_squared_error'],
              metrics=['accuracy'])

history = model.fit(x = [train_points, train_intermediate], y= [train_labels, train_classes], batch_size=32, epochs=epo, validation_data=([dev_points,dev_intermediate], [dev_labels, dev_classes]), shuffle=True, verbose=1)
#model.save('/drive/Colab Notebook/current_model')
import pickle

with open('drive/Colab Notebooks/trainHistoryDict_history450', 'wb') as file_pi:
     pickle.dump(history.history, file_pi)
model.save_weights('drive/Colab Notebooks/my_model_weights_450.h5')
# Evaluating the model on the test data    
loss = model.evaluate([test_points, test_intermediate], [test_labels, test_classes], verbose=0)
print('Test Loss:', loss)
Test Loss: [164.50030288279382, 164.4655409015593, 0.034760263916410385, 0.3852459018021985, 0.9562841520283392]
#Evaluating model of Dev Set
loss = model.evaluate([dev_points, dev_intermediate], [dev_labels, dev_classes], verbose=0)
print('Dev Loss:', loss)
