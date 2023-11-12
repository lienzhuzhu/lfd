# Problem Set 8
# Support Vector Machine with Soft Margin and Polynomial Kernel


import numpy as np
from sklearn import svm



TRAIN_DATA  = "./data/features.train"
TEST_DATA  = "./data/features.test"



#######################
### Data Processing ###
#######################

def load_data(file_path, digit):
    data = np.loadtxt(file_path)
    Y = data[ : , 0]
    Y = np.where(Y == digit, 1, -1)
    X = data[ : , 1: ]
    
    return X, Y



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, C=0.01, Q=2):
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

    C_list = [0.001, 0.01, 0.1, 1]

    for digit in range(0,10):
        X, Y = load_data(TRAIN_DATA, digit)
        model = svm_libsvm(X, Y)
        num_alphas = sum(model.n_support_)

        E_in = calc_e(model, X, Y)
        X_test, Y_test = load_data(TEST_DATA, digit)
        E_out = calc_e(model, X_test, Y_test)

        print(f"{digit} versus all  E_in: {E_in:.5f}  E_out: {E_out:.5f}  SVs: {round(num_alphas)}")
        


if __name__ == "__main__":
    main()
