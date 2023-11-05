# Problem Set 7.1 - 5
# Validation for Model Selection


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

def transform_data(X, Y, Q):
    x1 = X[:, 0]
    x2 = X[:, 1]
    ones = np.ones(X.shape[0])
    X_transformed = np.c_[ones, X, x1**2, x2**2, x1*x2, np.abs(x1-x2), np.abs(x1+x2)]
    X_transformed = X_transformed[ : , :Q+1]
    
    return X_transformed, Y



#########################
## Learning Algorithms ##
#########################

def linear_regression(X, Y):
    g = np.linalg.inv(X.T @ X) @ X.T @ Y
    return g


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
    Q_list = [q for q in range(3, 8)]

    X_train_original, Y_train_original = load_data(TRAIN_DATA)
    X_test_original, Y_test_original = load_data(TEST_DATA)
    
    for Q in Q_list:
        X_train, Y_train = transform_data(X_train_original, Y_train_original, Q)
        X_test, Y_test = transform_data(X_test_original, Y_test_original, Q)

        # N - K = 25
        X_val, Y_val = X_train[25: , : ], Y_train[25: ]
        X_train, Y_train = X_train[ :25, : ], Y_train[ :25]

        w = linear_regression(X_train, Y_train)
        E_val_25 = calc_Error(w, X_val, Y_val)
        E_out_25 = calc_Error(w, X_test, Y_test)

        # N - K = 10
        X_train, X_val = X_val, X_train
        Y_train, Y_val = Y_val, Y_train

        w = linear_regression(X_train, Y_train)
        E_val_10 = calc_Error(w, X_val, Y_val)
        E_out_10 = calc_Error(w, X_test, Y_test)

        #print(f"Q = {Q},\tE_val = {E_val:.40f} E_out = {E_out:.40f}")
        print(f"Q = {Q}:\tK = 10. E_val = {E_val_25:.10f} E_out = {E_out_25:.10f}\tK = 25: E_val = {E_val_10:.10f} E_out = {E_out_10:.10f}")



if __name__ == "__main__":
    main()
