# Final 7-10
# Regularization Linear Regression


import numpy as np
import matplotlib.pyplot as plt
import argparse


TRAIN_DATA  = "./data/features.train"
TEST_DATA   = "./data/features.test"


#######################
### Data Processing ###
#######################

def load_data(file_path, digit, other_digit=None):
    data = np.loadtxt(file_path)
    Y = data[:, 0]
    X = data[:, 1:]

    if other_digit is not None:
        mask = (Y == digit) | (Y == other_digit)
        Y = Y[mask]
        X = X[mask]
        Y = np.where(Y == digit, 1, -1)
    else:
        Y = np.where(Y == digit, 1, -1)

    return X, Y

def transform_data(X, Y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    ones = np.ones(X.shape[0])
    X_transformed = np.c_[ones, X, x1*x2, x1**2, x2**2]
    
    return X_transformed, Y


#########################
## Learning Algorithms ##
#########################

def linear_regression(X, Y):
    g = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return g

def regularized_linear_regression(X, Y, K=0):
    N = X.shape[0]
    dim = X.shape[1]
    w_reg = np.linalg.inv( X.T @ X + (10**K) * np.eye(dim) ) @ X.T @ Y
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
    parser = argparse.ArgumentParser(description="Regularized Linear Regression")
    parser.add_argument('-d', '--digit', type=int, help='First digit class')
    parser.add_argument('-o', '--other', type=int, help='Second digit class')
    parser.add_argument('--transform', action='store_true', help='Transform the data points')
    args = parser.parse_args()

    for digit in range(0,10):
        X_train, Y_train = load_data(TRAIN_DATA, digit)
        X_test, Y_test = load_data(TEST_DATA, digit)
        if args.transform:
            X_train, Y_train = transform_data(X_train, Y_train)
            X_test, Y_test = transform_data(X_test, Y_test)

        w = regularized_linear_regression(X_train, Y_train)
        E_in = calc_Error(w, X_train, Y_train)
        E_out = calc_Error(w, X_test, Y_test)

        print(f"{digit} versus all.\tE_in: {E_in:.5f}  E_out: {E_out:.5f}")

    if args.digit and args.other:
        X_train, Y_train = load_data(TRAIN_DATA, args.digit, args.other)
        X_test, Y_test = load_data(TEST_DATA, args.digit, args.other)
        if args.transform:
            X_train, Y_train = transform_data(X_train, Y_train)
            X_test, Y_test = transform_data(X_test, Y_test)

        print()

        w = regularized_linear_regression(X_train, Y_train, K=-2)
        E_in = calc_Error(w, X_train, Y_train)
        E_out = calc_Error(w, X_test, Y_test)
        print(f"K = {10**-2:.2f} Digit {args.digit} versus {args.other}. E_in: {E_in:.5f}  E_out: {E_out:.5f}")

        w = regularized_linear_regression(X_train, Y_train)
        E_in = calc_Error(w, X_train, Y_train)
        E_out = calc_Error(w, X_test, Y_test)
        print(f"K = {10**0:.2f} Digit {args.digit} versus {args.other}. E_in: {E_in:.5f}  E_out: {E_out:.5f}")


if __name__ == "__main__":
    main()
