#
# Template for Task 6: Random Forest Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from sklearn.ensemble import RandomForestClassifier
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
# pick five values of m by yourself 
m_values = [1,5,10,25,50]
# --- end of task --- #

er_test = []
for m in m_values: 
    # --- Your Task --- #
    # implement the random forest classification method 
    # you can directly call "RandomForestClassifier" from the scikit learn library
    
    # create random forest with m trees
    RLC = RandomForestClassifier(n_estimators=m, random_state=0)
    RLC.fit(sample_train, label_train)

    predictions = RLC.predict(sample_test)

    # store classification error on testing data here 
    er = np.mean(predictions != label_test)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('m')
plt.ylabel('Classification Error')
plt.title('Figure 7: Random Forest Performance vs Number of Trees')
plt.show()



