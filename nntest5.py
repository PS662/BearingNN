from scipy.fftpack import rfft, irfft, fftfreq
from scipy.fftpack import ifftn
import scipy
from pylab import *
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
import numpy as np

train = np.loadtxt("train_x.csv",delimiter=",")
target =np.loadtxt("train_ynew.csv",delimiter=",")

train =np.array(train).reshape((-1, 1, 30,40)).astype(np.uint8)
target = target.astype(np.uint8)


net1 = NeuralNet(
       layers=[('input', layers.InputLayer),
               ('hidden', layers.DenseLayer),
               ('output', layers.DenseLayer),
               ],
       # layer parameters:
       input_shape=(None,1,30,40),
       hidden_num_units=1000, # number of units in 'hidden' layer
       output_nonlinearity=lasagne.nonlinearities.softmax,
       output_num_units=4,  # 10 target values for the digits 0, 1, 2, ..., 9

       # optimization method:
       update=nesterov_momentum,
       update_learning_rate=0.0001,
       update_momentum=0.9,

       max_epochs=15,
       verbose=1,
       )

net1.fit(train,target)

def CNN(n_epochs):
   net1 = NeuralNet(
       layers=[
       ('input', layers.InputLayer),
       ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
       ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
       ('conv2', layers.Conv2DLayer), 
       ('hidden3', layers.DenseLayer),
       ('output', layers.DenseLayer),
       ],

   input_shape=(None, 1, 30,40),
   conv1_num_filters=7, 
   conv1_filter_size=(3, 3), 
   conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
   pool1_pool_size=(2, 2),
        
   conv2_num_filters=12, 
   conv2_filter_size=(2, 2),    
   conv2_nonlinearity=lasagne.nonlinearities.rectify,
    
   hidden3_num_units=1000,
   output_num_units=4, 
   output_nonlinearity=lasagne.nonlinearities.softmax,

   update_learning_rate=0.0001,
   update_momentum=0.9,

   max_epochs=n_epochs,
   verbose=1,
   )
   return net1
cnn = CNN(15).fit(train,target)
for z in range(6):
 test = np.loadtxt("u"+str(z)+".csv",delimiter=",")
 test = np.array(test).reshape((-1, 1, 30, 40)).astype(np.uint8)
 '''test1 = fftn(test)
 high = 0.0 + 160.0j
 low = 0.0 + 107.0j
 cut_f_signal = test1.copy()
 for i in range(400): 
  for j in range(1):
   for k in range(30):
    for l in range(40):
     cut_f_signal[i][j][k][l] = cut_f_signal[i][j][k][l] * 2
     if cut_f_signal[i][j][k][l] < low: 
      cut_f_signal[i][j][k][l].imag = 0

 itest1=ifftn(cut_f_signal)
 itest2=np.real(itest1)
 itest2 = np.array(itest2).reshape((-1, 1, 30, 40)).astype(np.uint8)'''
 pred = cnn.predict(test)
 np.savetxt('op'+str(z)+'.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', comments = '', fmt='%d')
    
