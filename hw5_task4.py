#
# Template for Task 4: kNN Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
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


# --- Your Task --- #
# pick five values of k by yourself 
k_values = [1,3,5,7,9]
# --- end of task --- #

def distance(a, b) -> float: #euclidian
    return np.sqrt(np.sum((a - b)**2))


er_test = []
for k in k_values: 
    # --- Your Task --- #
    # implement the kNN classification method 
    predicted_labels = []
    # add each test point in to compare with the training set
    for test_point in sample_test:
        # calculate distance for each testing point, for each training point
        distances = np.array([distance(test_point, train_point) for train_point in sample_train])

        # find the nearest k values and return an array of their indexes
        nearest_k_indexes = np.argsort(distances)[:k]
        # use indexes to identify labels for each
        nearest_k_labels = label_train[nearest_k_indexes]

        # use the average of the labels to determine the majority winner for each class
        avg = np.mean(nearest_k_labels)
        predicted_label = 1 if avg >= .5 else 0
        predicted_labels.append(predicted_label)



    # store classification error on testing data here 
    er = np.mean(predicted_labels != label_test)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.title("Figure 6: Classification Error vs. K")
plt.xlabel('k')
plt.ylabel('Classification Error')
plt.show()



