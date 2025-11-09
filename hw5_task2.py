#
# Template for Task 2: Logistic Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from sklearn.preprocessing import StandardScaler
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('diabetes.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------

# I am using this since I was getting a different value than sklearn's logistic regression
# sklearn automatically standardizes the data which is why my gradients were affecting my w and b differently
scaler = StandardScaler()
sample_train = scaler.fit_transform(sample_train)
sample_test = scaler.transform(sample_test)


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

    # use linear regression formula to find z
    z = sample_train @ w + b
    # use z to find my linear regression value
    y_pred = 1 / (1 + np.exp(-z))

    # compute gradient = (2/N) * (X.T @ (X @ w - y))
    # the same thing as calculating the partial derivative for each weight value

    error = y_pred - label_train

    gradient_w = (1 / num_train) * (sample_train.T @ error)
    gradient_b = (1 / num_train) * np.sum(error)
    ## update w
     # update using gradient descent w = w - alpha * gradient

    w = w - alpha * gradient_w
    b = b - alpha * gradient_b


    ## evaluate testing error of the updated w 
    # we should measure classification error here 

    # get predictions for test
    z_test = sample_test @ w + b
    y_pred_test = 1 / (1 + np.exp(-z_test))

    # use Loss function: -1/N * SUM[y * log(y_pred) + (1 - y) * log(1 - y_pred) ]
    er = -np.mean(label_test * np.log(y_pred_test) + (1 - label_test) * np.log(1 - y_pred_test))
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error: Loss')
plt.title("Figure 2: Logistic Regression")
print("Final Loss: ", er_test[-1])
print("Weights: ", w)
print("Bias: ", b)
plt.show()