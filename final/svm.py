# Final 11-12
# Hard Margin Support Vector Machine with Polynomial Kernel


import numpy as np
from sklearn import svm
import argparse



#######################
### Data Processing ###
#######################

def load_data():
    X = np.array([[ 1, 0],
                  [ 0, 1],
                  [ 0,-1],
                  [-1, 0],
                  [ 0, 2],
                  [ 0,-2],
                  [-2, 0]])
    Y = np.array([-1,-1,-1,1,1,1,1])
    return X, Y



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, C=1e6, Q=2):
    model = svm.SVC(C=C, kernel='poly', degree=Q, gamma=1)
    model.fit(X, Y)
    return model



########################
## Error Calculations ##
########################

def calc_e(model, X_test, Y_test):
    Y_svm = model.predict(X_test)
    e = np.mean(Y_test != Y_svm)
    return e


def calc_E_cv(model, X, Y):
    pass



#################
## Driver Code ##
#################

def main():
    X, Y = load_data()
    print(X)

if __name__ == "__main__":
    main()
