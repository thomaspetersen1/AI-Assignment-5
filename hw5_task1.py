#
# Template for Task 1: Linear Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Your Task --- #
# pick a proper number of iterations 
num_iter = 2000
# randomly initialize your w (starting at 0 so w can get more reasonable values)
w = np.zeros(p-1)
# scalar bias
b = 0.0
# learning rate
alpha = .01
# --- end of task --- #
    
er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
for iter in range(num_iter): 

    # ypred = Xw where x is the sample matrix and w is current slope of weights
    # Note: scikit learn uses a bias therefore my ypred is forcing the origin and affecting my MSE
    # ypred = Xw + b

    y_pred = sample_train @ w + b # matrix multiplication with training data

    # compute gradient = (2/N) * (X.T @ (X @ w - y))
    # the same thing as calculating the partial derivative for each weight value

    error = y_pred - label_train

    gradient_w = (2 / num_train) * (sample_train.T @ error)
    gradient_b = (2 / num_train) * np.sum(error)

    ## update w
    # update using gradient descent w = w - alpha * gradient

    w = w - alpha * gradient_w
    b = b - alpha * gradient_b

    ## evaluate testing error of the updated w 
    # we should measure mean-square-error here

    y_pred_test = sample_test @ w + b

    # calculate MSE
    er = np.mean((y_pred_test - label_test)**2)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.title("Figure 1: Linear Regression")
print ("Final MSE: ", er_test[-1])
plt.show()