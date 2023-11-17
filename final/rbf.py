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


def find_centers(X, K=9):
    #initialize K random centers from the data set X
    #while the element-wise difference of the centers matrix is non-zero
    # 1. assign each point in X to a center in clusters dictionary (maintain an original copy of this dictionary to iterate through) based on closest center
    # 2. update centers to average point of constituent points in cluster

    centers = X[np.random.choice(X.shape[0], K, replace=False)]


def rbf_model():
    #find_centers()
    #pseudo inverse for weights aka coefficients
    return coefficients
    #predictions = sign(signal) = sign(weights * rbf)

def rbf_model_predict():
    pass

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
    sv_list = []

    trial = 0
    while trial < TRIALS:
        X, Y = generate_data()

        model = svm_libsvm(X, Y, G=G)
        if calc_e(model, None, X, Y) != 0:
            trial -= 1
            inseparable_freq += 1
            continue

        sv_list.append(sum(model.n_support_))

        trial += 1

    #print statistics
    print(np.mean(sv_list))


if __name__ == "__main__":
    main()
