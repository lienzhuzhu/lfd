# Problem Set 6.2-6
# Regularization with Weight Decay


import numpy as np
import matplotlib.pyplot as plt


TRAIN_DATA  = "./data/in.dta"
TEST_DATA   = "./data/out.dta"


#######################
### Data Processing ###
#######################

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y

def transform_data(X, Y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    ones = np.ones(X.shape[0])
    X_transformed = np.c_[ones, X, x1**2, x2**2, x1*x2, np.abs(x1-x2), np.abs(x1+x2)]
    
    return X_transformed, Y



#########################
## Learning Algorithms ##
#########################

def linear_regression(X, Y):
    g = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return g

def regularized_linear_regression(X, Y, K):
    N = X.shape[0]
    dim = X.shape[1]
    w_reg = np.linalg.pinv( X.T @ X + (10**K) * np.eye(dim) ) @ X.T @ Y
    return w_reg



########################
## Error Calculations ##
########################

def calc_Error(g, X, Y):
    Y_g = np.sign(X.dot(g))
    error = np.mean(Y != Y_g)
    return error


#################
## Driver Code ##
#################

def main():

    E_in_list, E_out_list = [], []
    K_list = [k for k in range(-3, 4)]

    X_train, Y_train = load_data(TRAIN_DATA)
    X_train, Y_train = transform_data(X_train, Y_train)

    X_test, Y_test = load_data(TEST_DATA)
    X_test, Y_test = transform_data(X_test, Y_test)
    
    w = linear_regression(X_train, Y_train)
    E_in = calc_Error(w, X_train, Y_train)
    E_out = calc_Error(w, X_test, Y_test)

    print("E_in with no regularization:\t", E_in)
    print("E_out with no regularization:\t", E_out)

    for K in K_list:
        w = regularized_linear_regression(X_train, Y_train, K)
        E_in = calc_Error(w, X_train, Y_train)
        E_out = calc_Error(w, X_test, Y_test)

        print(f"K = {K},\t E_in = {E_in:.6f}\tE_out = {E_out:.6f}")


if __name__ == "__main__":
    main()
