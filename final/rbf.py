# Final
# Hard Margin Support Vector Machine with RBF Kernel


import numpy as np
from sklearn import svm
import argparse



#######################
### Data Processing ###
#######################

def generate_data(N=100):
    X = np.random.uniform(-1, 1, (N, 2))
    Y = np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(np.pi * X[:, 0]))
    return X, Y



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, C=1e6):
    model = svm.SVC(C=C, kernel='rbf', gamma=1)
    model.fit(X, Y)
    return model


def find_centers(X, K):
    pass

def rbf_model(X, Y):
    #find_centers()
    #pseudo inverse for weights
    pass


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
    parser = argparse.ArgumentParser(description="RBF Kernel vs RBF Model")
        

        C_list = [0.01, 1, 100, 1e4, 1e6]
        X, Y = load_data(TRAIN_DATA, args.digit, args.other)
        X_test, Y_test = load_data(TEST_DATA, args.digit, args.other)

        for C in C_list:
            model = svm_libsvm(X, Y, C=C)
            num_alphas = sum(model.n_support_)

            E_in = calc_e(model, X, Y)
            E_out = calc_e(model, X_test, Y_test)

            print(f"C = {C:.2f}\t{args.digit} versus {args.other}  E_in: {E_in:.5f}  E_out: {E_out:.5f}  SVs: {round(num_alphas)}")


if __name__ == "__main__":
    main()
