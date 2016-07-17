import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    v = np.ones((n_data,1))
    x = np.hstack((train_data,v))
    
    #print x.shape[1]
    #print labeli.shape[0]
    #print initialWeights.shape
    w = np.reshape(initialWeights,(n_features+1,1))
    #print w.shape
    a = np.dot(x,w) #50000*716 by 716*1 
    #print initialWeights.size
    #print a.shape
    theta_n = sigmoid(a)

    #print theta_n.size
    #print v.size
    #print a.shape
    #print v.shape
    try:
        log1 = np.log(theta_n)
    except RuntimeWarning:
        log1 = 0
    
    theta = np.array(v) - np.array(theta_n)
    
    log2 = np.log(theta)
    #print theta.shape
    e1 = np.dot(np.transpose(labeli),log1)
    p = np.array(v) - np.array(labeli)
    #print p.shape
    e2 = np.dot(np.transpose(p),log2)
    e = np.add(e1,e2)
    error = -np.sum(e)/n_data
    #print error.size

    g1 = np.subtract(theta_n,labeli)
    g2 = np.dot(np.transpose(x),g1)
    #print g2.shape
    #error_grad = np.sum(g2)
    #print error_grad.shape
    error_grad = g2/n_data
    #print error_grad.shape
    error_grad = error_grad.flatten()
    #print "Exiting !!!"
    print error
    #print error_grad.shape
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print "Inside Predict"
    v = np.ones((data.shape[0],1))
    x = np.hstack((data,v))
    a = np.dot(x,W) #50000*716 by 716*1 
    #print initialWeights.size
    #print a.size
    theta_n = sigmoid(a)
    #print W.shape
    
    for i in range(label.shape[0]):
       m = np.argmax(theta_n[i])
       label[i] = m

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    v = np.ones((n_data,1))
  
    #print v.shape

    x = np.hstack((train_data,v))

    #print x.shape
    #print initialWeights_b.shape

    wt = np.reshape(params,(n_feature+1,n_class))

    #print "wt and x shapes"
    #print wt.shape
    #print x.shape

    xwt = np.dot(x,wt)
    pyc1 = np.exp(xwt)

    #print "pyc1"
    #print pyc1.shape
    pyc2 = np.sum(pyc1,axis = 1)

    #print "pyc2"
    #print pyc2.shape
    pyc = np.zeros_like(xwt)
    for i in range (pyc1.shape[0]):
        pyc[i] = pyc1[i]/pyc2[i]
    
    pyctrans = pyc

    logpyc = np.log(pyc)
    logpyctrans = np.transpose(logpyc)

    #logpyctrans = np.transpose(logpyc)
    
    err1 = np.multiply(Y,logpyc)
    
    err2 = np.sum(err1)

    #err = np.sum(err2)

    err = -1*err2
    err=err/n_data

    print "error"
    print err

    error = err 

    #print "pyc,yl"
    #print pyc.shape
    #print yl.shape
    errgrad1 = pyc - Y

    #print "errgrad1,x"
    #print errgrad1.shape
    #print x.shape
    errgrad2 = np.dot(x.T,errgrad1)

    #print "errgrad2"
    #print errgrad2.shape
    errgrad = errgrad2/n_data

    error_grad = errgrad
    #print err

    error_grad = error_grad.flatten()
    #print "error grad shape"
    #print error_grad.shape
    #print x.size
    #print g1.shape

    return error, error_grad



def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #print "Inside Predict"
    v = np.ones((data.shape[0],1))
    x = np.hstack((data,v))
    #wt = np.reshape(params,(n_feature+1,n_class))

    a = np.dot(x,W) #50000*716 by 716*1 
    #print initialWeights.size

    #print a.size
    xwt = np.dot(x,W)    
    #same until here

    #theta_n = sigmoid(xwt)
    #print xwt.shape
    pyc1 = np.exp(xwt)

    #print "pyc1"
    #print pyc1.shape

    pyc1trans = np.transpose(pyc1)

    #print "pyc1trans"
    #print pyc1trans.shape

    pyc2 = np.sum(pyc1,axis=1)

    #print "pyc2"
    #print pyc2.shape
    pyc = np.zeros_like(xwt)
    #pyc = pyc1/pyc2
    
    for i in range (pyc1.shape[0]):
        pyc[i] = pyc1[i]/pyc2[i]

    theta_n = pyc
    #print "label"
    #print label.shape
    
    for i in range(label.shape[0]):
       m = np.argmax(theta_n[i])
       label[i] = m

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
print('\n\n--------------SVM-------------------\n\n')
train_l = train_label.flatten()
test_l = test_label.flatten()
validation_l = validation_label.flatten()

print('\n--------------Linear kernel-------------------\n')

s = svm.SVC(kernel='linear')
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')

print('\n--------------Radial Basis Function, Gamma = 1-------------------\n')

s = svm.SVC(kernel='rbf',gamma=1.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')

print('\n--------------Radial Basis Function, Gamma = Default-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto')
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=1-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=1.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')

print('\n--------------Radial Basis Function, Gamma = Default, C=10-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=10.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')

print('\n--------------Radial Basis Function, Gamma = Default, C=20-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=20.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=30-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=30.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=40-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=40.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=50-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=50.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')

print('\n--------------Radial Basis Function, Gamma = Default, C=60-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=60.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=70-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=70.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=80-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=80.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=90-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=90.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


print('\n--------------Radial Basis Function, Gamma = Default, C=100-------------------\n')

s = svm.SVC(kernel='rbf',gamma='auto',C=100.0)
s.fit(train_data, train_l)

pred = s.predict(train_data)
tmean=np.mean((pred==train_l).astype(float))
percent=100*tmean
print('\nTraining set Accuracy:'+str(percent)+'%')

pred = s.predict(validation_data)
vmean=np.mean((pred==validation_l).astype(float))
percent=100*vmean
print('\nValidation set Accuracy:'+str(percent)+'%')

pred = s.predict(test_data)
ttmean=np.mean((pred==test_l).astype(float))
percent=100*ttmean
print('\nTest set Accuracy:'+str(percent)+'%')


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

with open('params.pickle', 'wb') as f1: 
    pickle.dump(W, f1) 
with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)
