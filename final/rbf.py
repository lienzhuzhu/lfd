# Final
# Hard Margin Support Vector Machine with RBF Kernel


import numpy as np
from sklearn import svm
import argparse



TRIALS = 100


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

def svm_libsvm(X, Y, C=1e6, G=1.5):
    model = svm.SVC(C=C, kernel='rbf', gamma=G)
    model.fit(X, Y)
    return model


def find_centers(X, K):
    pass

def rbf_model(X, Y):
    #find_centers()
    #pseudo inverse for weights aka coefficients
    return coefficients
    #predictions = sign(signal) = sign(weights * rbf)


########################
## Error Calculations ##
########################

def calc_e(model, rbf_weights, X_test, Y_test):
    Y_svm = model.predict(X_test)
    e = np.mean(Y_test != Y_svm)
    return e



#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="RBF Kernel vs RBF Model")
    parser.add_argument('--gamma', type=int, help='Gamma parameter for Gaussian spread')
    parser.add_argument('--centers', type=int, help='Number of centers K')
    args = parser.parse_args()

    G = args.gamma if args.gamma else 1.5
    K = args.centers if args.centers else 9

    svm_In_list, svm_Out_list, rbf_In_list, rbf_Out_list = [], [], [], []
    inseparable_freq = 0

    while trial < TRIALS:
        X, Y = generate_data()

        model = svm_libsvm(X, Y, gamma=G)
        if model.predict(X, Y) != 0:
            trial -= 1
            inseparable_freq += 1
            continue

        num_alphas = sum(model.n_support_)

        trial += 1

    #print statistics


if __name__ == "__main__":
    main()
