
# coding: utf-8

# # Functions for Complete Little, Big Data Analysis
# ## For Use with Windows, Mac, and Colaboratory
# ### Emily Stark
# ### April 25th, 2019
# All functions needed to preprocess, train a model, perform "DeepDiscovery", and perform "GestaltDL"

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.colors import ListedColormap

from mlxtend.plotting import plot_decision_regions

from scipy.stats import t, ttest_ind, ks_2samp

from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from IPython.display import display, HTML

import statsmodels.api as sm
from statsmodels.formula.api import ols

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

import tensorflow as tf

try:
    from tensorboardcolab import *
except ImportError: 
    pass


# In[ ]:


plt.rcParams['figure.figsize'] = [8, 8]


# ## Preprocessing

# #### Objectives:
#  - Create a sufficient number of inputs for model training from a small number of high dimensional data points
#  
#  ***
# 
# #### Inputs:
#  - Analyzed rows, vector: X
#  
#  - Analyzed rows, vector: W
#  
#  - Input dimensions (*x*, *w*)
#  
#  - Training directory (*trainfiledir*)
#   
#  - Validation directory (*valfiledir*)
#  
#  - Testing directory (*testfiledir*)
#    
#  - Class denoter (*classtargets*), an array of the class denoter from file name
#   
#  - Header to skip (*header*), number of rows to skip when reading in.csv
#  
#  ***
#  
# #### Outputs:
#  
#  - 3D arrays containing instances (X) and corresponding one-hot vectors containing labels (Y): *Xtrain*, *Ytrain*, *Xval*, *Yval*, *Xtest* (if provided), *Ytest* (if provided)

# In[ ]:


def Preproc(x, w, Xstart, Xend, Wstart, Wend, trainfiledir, valfiledir, classtargets, header, testfiledir = None):
    itot = int((Wend - Wstart)/w) # i is position in left to right, itot is number of left-right steps
    jtot = int((Xend - Xstart)/x) # j is position in up to down, jtot is number of up-down steps
    n = int(itot*jtot) # number of inputs per sample
    k = len(classtargets) # number of classes
    
    # Training Set
    os.chdir(trainfiledir)
    files = glob.glob('*.csv')
    d = np.zeros([n, x, w])
    Dmaster = np.zeros([n*len(files), x, w])
    Lmaster = np.zeros([n*len(files), k])
    
    for h in range(0, len(files)):
        samp = np.genfromtxt(files[h], delimiter = ',', skip_header = header)
        samp = samp[Xstart:Xend, Wstart:Wend]

        for i in range(0, itot):
            for j in range(0, jtot):
                d[i, ...] = samp[0:x, i*w:(i+1)*w]

        Dmaster[n*h:n*(h+1), ...] = d
        d = np.zeros([n, x, w])

        for m in range(0, k):
            if classtargets[m] in files[h]:
                lab = m

        Lmaster[n*h:n*(h+1), lab] = 1
        
        print(files[h])
    
    Xtrain = Dmaster[...,None]
    Ytrain = Lmaster
    
    # Validation Set
    os.chdir(valfiledir)
    files = glob.glob('*.csv')
    d = np.zeros([n, x, w])
    Dmaster = np.zeros([n*len(files), x, w])
    Lmaster = np.zeros([n*len(files), k])
    
    for h in range(0, len(files)):
        samp = np.genfromtxt(files[h], delimiter = ',', skip_header = header)


        for i in range(0, itot):
            for j in range(0, jtot):
                d[i, ...] = samp[0:x, i*w:(i+1)*w]

        Dmaster[n*h:n*(h+1), ...] = d
        d = np.zeros([n, x, w])

        for m in range(0, k):
            if classtargets[m] in files[h]:
                lab = m

        Lmaster[n*h:n*(h+1), lab] = 1
        
        print(files[h])
        
    Xval = Dmaster[...,None]
    Yval = Lmaster
    
    # Testing Set, Optional
    if testfiledir is not None:
        os.chdir(testfiledir)
        files = glob.glob('*.csv')
        d = np.zeros([n, x, w])
        Dmaster = np.zeros([n*len(files), x, w])
        Lmaster = np.zeros([n*len(files), k])
        for h in range(0, len(files)):
            samp = np.genfromtxt(files[h], delimiter = ',', skip_header = header)


            for i in range(0, itot):
                for j in range(0, jtot):
                    d[i, ...] = samp[0:x, i*w:(i+1)*w]

            Dmaster[n*h:n*(h+1), ...] = d
            d = np.zeros([n, x, w])

            for m in range(0, k):
                if classtargets[m] in files[h]:
                    lab = m

            Lmaster[n*h:n*(h+1), lab] = 1
            
            print(files[h])
            
        Xtest = Dmaster[...,None]
        Ytest = Lmaster
        return(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    
    
    return(Xtrain, Ytrain, Xval, Yval)            


# ## Model Training

# #### Objectives:
#  - Train a preloaded architecture on *Xtrain*, *Ytrain*, using *Xval* and *Yval* for validation
#   
#  ***
#  
# #### Inputs:  
#   - Training data, *Xtrain*, *Ytrain*
#   
#   - Validation data *Xval*, *Yval*
#   
#   - Model Name (*mname*), string
#     
#   - Network Architecture (*net*), choice of 3 or 5-hidden layer fully connected network ('3FCN' or '5FCN') or 'AlexNet'
#   
#   - Architecture (*arch*), other model architecture written. Should provide either *net* or *arch*.
#    
#   - Number of epochs (*ep*), default is 80
#   
#   - Batch Size (*bsize*), default is 50
#   
#   - OS (*string*), either 'Colab' or 'Linux' used to show Tensorboard. If running on Windows, do not specify anything, defaults to None
#  
# ***
#  
# #### Outputs:
#  
#   - Trained network, *model*

# In[1]:


def Modeltrain(Xtrain, Ytrain, Xval, Yval, mname, net = None, arch = None, ep = 80, bsize = 50, OS = None):
    if net == '3FCN':
        print('3-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    if net == '5FCN':
        print('5-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if net == 'AlexNet':
        print('AlexNet selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if arch is not None:
        print('Different Architecture Provided')
        network = arch
        
    
    if OS == 'Colab':
        tbc=TensorBoardColab()
        model = tflearn.DNN(network, checkpoint_path='LBD_model',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir = './Graph')
    
    if OS == 'Linux':
        os.system('tensorboard --logdir=/tmp/tflearn_logs &')
        model = tflearn.DNN(network, checkpoint_path='LBD_model',
                            max_checkpoints=1, tensorboard_verbose=2)
        
    if OS is None:
        model = tflearn.DNN(network, checkpoint_path='LBD_model',
                            max_checkpoints=1, tensorboard_verbose=2)
    
    model.fit(Xtrain, Ytrain, n_epoch = ep, validation_set=(Xval, Yval), shuffle=True,
              show_metric = True, batch_size = bsize, snapshot_step = 20,
              snapshot_epoch = False, run_id = mname)
    
    return(model)


# ## Deep Discovery

# #### Objectives:
#  - Shows areas in the sample that carry strong signals regardless of class
#  
#  ***
# 
# #### Inputs:
#  - Validation Set, *Xval*, *Yval*
#  
#  - List of the names of the $k$ classes,*classnames*
#  
#  - Number of samples per input, *n*
#  
#  - Directory of saved model and model name, *modeldir*, *mname*
#   
#  - If model is trained or already loaded in the notebook, do not include directory and model name, instead just include the model variable, *model*
#  
#  - Network architecture, *net*, can choose from three preloaded architectures (3FCN, 5FCN, AlexNet) or write your own and list the name
#    
#  
#  ***
#  
# #### Outputs:
#  
# **Confusion Matrix**
# 
#  - *Confusion Matrix*
#  
# **Residuals Plots**
# 
#  - *scatterplot of all inputs, sorted by class*, along the x-axis the input number after inputs are grouped by class, along the y-axis the value of *Lressum* associated with that input
#  
#  - *scatterplots of inputs within classes*, $k$ seperate scatterplots where the x-axis is the input number after inputs are grouped by class, along the y-axis the value of *Lressum* associated with that input
#  
# **Average Error Plot with Signal Position Confidence Inervals**
# 
#  - *scatterplot of the average *Lressum* per input position across all samples*, inputs are grouped into sets of *n* and the average error of each position (from 0 to *n*) is calculated and plotted against input position.
#  
#  - *confidence intervals of signal*, three confidence intervals are plotted corresponding with a 90%, 95%, and 99% confidence in the position for the smallest error.

# In[ ]:


def DeepDiscovery(Xval, Yval, classnames, n, modeldir = None, mname = None, 
                  model = None, net = None, arch = None):
    
    # LOADS IN MODEL
    
    if modeldir is not None:
        tf.reset_default_graph()
    
    if net == '3FCN':
        print('3-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    if net == '5FCN':
        print('5-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if net == 'AlexNet':
        print('AlexNet selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if arch is not None:
        print('Different Architecture Provided')
        network = arch
    
    if modeldir is not None:
        os.chdir(modeldir)
        model = tflearn.DNN(network)
        model.load(model_file = mname)
        
        
    # SORTS THE INPUTS BY CLASS
    k = Yval.shape[1]                      # determining the number of classes, k 
    nkarr = np.sum(Yval, axis = 0)         # checking to make sure the test set is balanced
    N = Yval.shape[0]                      # number of inputs

    if np.min(nkarr) == np.max(nkarr):      
        nk = nkarr[0]
        nk = int(nk)                        # if it balanced, the rest of the if statement will continue
        
        tick = np.zeros([k])                # creates a counter, next input associated with  class is free
        Xsort = np.zeros(Xval.shape)       # for sorted data
        Ysort = np.zeros(Yval.shape)       # for sorted data
        
        for i in range(0, N):  
            c = np.argmax(Yval[i,...])     # checks which class the input in Xval belongs to
            Xsort[int(c*nk+tick[c]):int(c*nk+tick[c]+1),...] = Xval[i:i+1,...]
            Ysort[int(c*nk+tick[c]):int(c*nk+tick[c]+1),c] = 1
            tick[c] = tick[c] + 1
        
        # FORMATS Xval
        while len(Xsort.shape) < 4:         # checks to see if the 4th dimension was already added
            Xsort = Xsort[...,None]         # if it wasn't, the 4th dimension is added, if it was nothing happens
        
        # CREATES ARRAYS FOR PREDICTIONS
        N = Ysort.shape[0]                      # number of inputs
        Lhat = np.zeros([N, k])                 # creates an empty matrix

        
        # STORES PREDICTIONS
        for i in range(0, N):
            q = model.predict(Xsort[i:(i+1),...])   # row vector of the confidences outputted by the model
            Lhat[i:(i+1),:] = q                     # assigns confidence values to the correct row in Lhat

        # CALCULATING THE Lressum
        Lres = Ysort - Lhat                     # calculates the raw residuals (labels - confidences)
        Lressum = np.std(Lres,axis=1)           # std devs across the rows
        Lressum = Lressum[...,None]    
        
        # FORMATTING Lressum FOR PLOTTING
        ids = np.zeros([N,1])                   # creating an id column to attach to Lressum

        for i in range(0,N):                    # stupid for loop because it's not R
            ids[i, 0] = i

        Lressumid = np.append(ids, Lressum, axis = 1)

        Ldf = pd.DataFrame(Lressumid)            # converting to Pandas because we're tired of Python
        Ldf = Ldf.rename(index=str, columns={0:"id", 1:"res"})
        
        # PLOTTING
        plt.figure(1)
        Ldf.plot.scatter(x='id', y='res', title = 'RMSE Across All Classes, Grouped by Class')
        plt.show()
        
        plt.figure(2)
        for j in range(0, k):
            Ldfa = Ldf[nk*j:nk*(j+1)]
            Ldfa.plot.scatter(x='id', y='res', title = 'RMSE Across Class ' + classnames[j])
        plt.show()
    
        nsamp = int(N/n)
        LressumSamp = np.zeros([nsamp, n, 1])
        Minima = np.zeros([nsamp])

        for i in range(0, nsamp):
            LressumSamp[i,...] = Lressum[i*n:(i+1)*n,:]
            Minima[i] = np.where(LressumSamp[i] == np.min(LressumSamp[i]))[0][0]
            
        # Calculating the Confidence Intervals of the Position of the Lowest Error Input per Sample
        tval1 = t.interval(.90, n-1)[1]
        lowlim1 = np.mean(Minima) - tval1*np.std(Minima)
        upplim1 = np.mean(Minima) + tval1*np.std(Minima)

        tval2 = t.interval(.95, n-1)[1]
        lowlim2 = np.mean(Minima) - tval2*np.std(Minima)
        upplim2 = np.mean(Minima) + tval2*np.std(Minima)

        tval3 = t.interval(.99, n-1)[1]
        lowlim3 = np.mean(Minima) - tval3*np.std(Minima)
        upplim3 = np.mean(Minima) + tval3*np.std(Minima)
        
        # Finding Average Error per Position in Sample
        AveErr = np.zeros([n])
        for i in range(0, n):
            AveErr[i] = np.mean(LressumSamp[:, i])

        plt.figure(3)

        plt.scatter(ids[0:n], AveErr)

        plt.axvspan(lowlim1, upplim1, alpha=0.05, color='salmon')
        plt.axvspan(lowlim2, upplim2, alpha=0.1, color='salmon')
        plt.axvspan(lowlim3, upplim3, alpha=0.18, color='salmon')

        plt.suptitle('Average RMSE of Input Number', fontsize=12)
        plt.title('Confidence Intervals of Signal Position in Red')
        plt.xlabel('Input Number')
        plt.ylabel('Average RMSE of Input')

        print('90% Confidence Interval Limits: (', lowlim1, ',', upplim1, ')')
        print('95% Confidence Interval Limits: (', lowlim2, ',', upplim2, ')')
        print('99% Confidence Interval Limits: (', lowlim3, ',', upplim3, ')')

        plt.show()
        
        # CONFIDENCE MATRIX
        label = tf.argmax(Ysort, axis = 1)    # converts true labels to column vector
        predict = tf.argmax(Lhat, axis = 1)   # predicts binary labels from the confidences, stores vector
        confusion_matrix = tf.confusion_matrix(label, predict, k) 
        
        with tf.Session() as sess:
            cm = confusion_matrix.eval()      # creates the confusion matrix

        pdcm = pd.DataFrame(cm)               # converts the confusion matrix to pandas for aesthetics

        p = ['Predicted'] * k                 # this is to make pretty row and column names
        list = []
        for i in range(0,k):
            list.append(p[i] + ' ' + classnames[i])
            pclassname = list

        a = ['Actual'] * k
        list = []
        for i in range(0,k):
            list.append(a[i] + ' ' + classnames[i])
            aclassname = list

        pdcm.columns = pclassname              # renaming columns
        pdcm.index = aclassname                # renaming rows

        confusion_mat = pdcm                    
        return(confusion_mat)

    else:
        print("Need Blanced Testing Set")


# ## Gestalt Deep Learning

# In[ ]:


def GestaltDL(Xval, Yval, valnames, classnames, n, perc,  Xtest = None, Ytest = None, 
              testnames = None, modeldir = None, mname = None, model = None, net = None, arch = None):
    
    if modeldir is not None:
        tf.reset_default_graph()
    
    if net == '3FCN':
        print('3-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    if net == '5FCN':
        print('5-Hidden Layer Fully Connected Network Selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2000, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if net == 'AlexNet':
        print('AlexNet selected')
        network = input_data(shape = [None, Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, Ytrain.shape[1], activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        
    if arch is not None:
        print('Different Architecture Provided')
        network = arch
    
    if modeldir is not None:
        os.chdir(modeldir)
        model = tflearn.DNN(network)
        model.load(model_file = mname)
        
        
    Nval = Yval.shape[0]
    k = Yval.shape[1]
    Yvalhat = np.zeros([Nval, k + 2])
    
    
    for i in range(0, len(valnames)):
        for j in range(0, n):
            Yvalhat[i*n+j, 0] = i
            Yvalhat[i*n+j, 1] = j
            Yvalhat[i*n+j, 2:] = model.predict(Xval[(i*n+j):(i*n+j+1), ...])
            
    if Ytest is not None:
        Ntest = Ytest.shape[0]
        Ytesthat = np.zeros([Ntest, k + 2])
        for i in range(0, len(testnames)):
            for j in range(0, n):
                Ytesthat[i*n+j, 0] = i
                Ytesthat[i*n+j, 1] = j
                Ytesthat[i*n+j, 2:] = model.predict(Xtest[(i*n+j):(i*n+j+1), ...])
                
    plt.figure()
    f, axarr = plt.subplots(k,k)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    for i in range(0,k):
        for j in range(0,k):
            a1 = Yvalhat[np.where(Yval[:,i] == 1), 2 + j]
            a1 = a1[0,...]
            axarr[i,j].boxplot(a1, vert = 0)
            c = 'Confidence: ' + classnames[j]
            t = 'True: ' + classnames[i]
            axarr[i,j].set_title(c)
            axarr[i,j].set_ylabel(t)
            
    Yvalhatg = np.zeros([int(Nval/k), k, k])
    for i in range(0,k):
        Yvalhatg[:,:,i] = Yvalhat[np.where(Yval[:,i] == 1), 2:]
            
    if k == 2:
        ttest = ttest_ind(Yvalhatg[:,0,0], Yvalhatg[:,0,1])
        kstest = ks_2samp(Yvalhatg[:,0,0], Yvalhatg[:,0,1])
        print('t-statistic = ' + str(ttest[0]))
        print('t-test p-value = ' + str(ttest[1]))
        print('KS statistic = ' + str(kstest[0]))
        print('KS p-value = ' + str(kstest[1]))
        
    if k > 2:
        Yvalhatanova = np.zeros([int((k-1)*Nval), 3])
        for i in range(0,k):
            Yvalhatanova[int(i*(Nval*(k-1)/k)):int((i+1)*(Nval*(k-1)/k)), 0] = i
            for j in range(0, k-1):
                Yvalhatanova[int(i*((k-1)*Nval/k)+j*(Nval/k)):int(i*((k-1)*Nval/k)+(j+1)*(Nval/k)), 1] = j
                Yvalhatanova[int(i*((k-1)*Nval/k)+j*(Nval/k)):int(i*((k-1)*Nval/k)+(j+1)*(Nval/k)), 2] = Yvalhatg[:,j,i]
        
        anovapd = pd.DataFrame(Yvalhatanova, columns = ['GroundTruth', 'PredClass', 'Conf'])
        sigtest = ols('Conf ~ C(GroundTruth)*C(PredClass)', anovapd).fit()
        tableanova = sm.stats.anova_lm(sigtest, typ = 2)
        display(tableanova)
        
    print('Strong Signals from Validation Set:')
    for i in range(0, len(classnames)):
        asdf = Yvalhat[np.where(Yval[:,i] == 1), :]
        asdf = asdf[0, ...]
        cut = np.percentile(asdf[:,2+i], perc)
        jkl = asdf[np.where(asdf[:,2+i] > cut),]
        jkl = jkl[0,...]
        print(classnames[i])
        for j in range(0, jkl.shape[0]):
            sigsamp = valnames[int(jkl[j, 0])] + ', Input ' + str(int(jkl[j,1])) + ', Confidence ' + str(round(round(jkl[j,2+i],4)*100, 0)) + '%'
            print(sigsamp)
            
    if Ytest is not None:
        print('Strong Signals from Testing Set:')
        for i in range(0, len(classnames)):
            asdf = Ytesthat[np.where(Ytest[:,i] == 1), :]
            asdf = asdf[0, ...]
            cut = np.percentile(asdf[:,2+i], perc)
            jkl = asdf[np.where(asdf[:,2+i] > cut),]
            jkl = jkl[0,...]
            print(classnames[i])
            for j in range(0, jkl.shape[0]):
                sigsamp = testnames[int(jkl[j, 0])] + ', Input ' + str(int(jkl[j,1])) + ', Confidence ' + str(round(round(jkl[j,2+i],4)*100, 0)) + '%'
                print(sigsamp)
        
    if Ytest is not None:
            
        Yvalsamphat = np.zeros([len(valnames), k-1])
        Yvalsamp = np.zeros([len(valnames)])
        Ytestsamphat = np.zeros([len(testnames), k-1])
        Ytestsamp = np.zeros([len(testnames)])

        for i in range(0, len(valnames)):
            Yvalsamphat[i, :] = np.mean(Yvalhat[i*n:(i+1)*n, 2:(2+k-1)], axis = 0)
            Yvalsamp[i] = int(np.where(Yval[i*n,:] == np.max(Yval[i*n,:]))[0])

        for i in range(0, len(testnames)):
                Ytestsamphat[i, :] = np.mean(Ytesthat[i*n:(i+1)*n, 2:(2+k-1)], axis = 0)
                Ytestsamp[i] = int(np.where(Ytest[i*n,:] == np.max(Ytest[i*n,:]))[0])
                
                
        clf = LinearDiscriminantAnalysis()
        clf.fit(Yvalsamphat, Yvalsamp)

        YtestsampLDA = clf.predict(Ytestsamphat)
        
        string = 'Predicted '
        colname = [string + x for x in classnames]

        string = 'Actual '
        rowname = [string + x for x in classnames]
        
        cm = confusion_matrix(Ytestsamp, YtestsampLDA)
        cmpd = pd.DataFrame(cm, columns = colname, index = rowname)
        
        display(cmpd)
            
            
        if k == 2:
            YtestsampLDA = YtestsampLDA[:,None]
            
            plt.figure()
            plot_decision_regions(YtestsampLDA, Ytestsamp.astype(int), clf=clf, legend=1,
                     colors = '#1b9e77,#d95f02,#7570b3,#66c2a5,#fc8d62,#8da0cb')
            plt.xlabel('Confidence in Class 0')
            plt.xlim(-0.1,1.1)
            plt.title('Linear Discriminant Analysis for Whole Sample Classification')

        if k == 3:
            plt.figure()
            X_set, y_set = Yvalsamphat, Yvalsamp

            aranged_pc1 = np.arange(start = -0.1, stop = 1.1, step = 0.01)
            aranged_pc2 = np.arange(start = -0.1, stop = 1.1, step = 0.01)

            X1, X2 = np.meshgrid(aranged_pc1, aranged_pc2)

            plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.5, cmap = ListedColormap(('#66c2a5', '#fc8d62', '#8da0cb')))
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c = ListedColormap(('#1b9e77', '#d95f02','#7570b3'))(i), label = j)
            plt.title('Linear Discriminant Analysis for Whole Sample Classification', fontsize = 24)
            plt.xlabel('Confidence in Class 0', fontsize = 16)
            plt.ylabel('Confidence in Class 1', fontsize = 16)
            plt.legend(fontsize = 12)
            plt.show()

