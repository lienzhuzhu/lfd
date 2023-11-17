# Problem Set 8
# Support Vector Machine with Soft Margin and Polynomial Kernel with cross validation


import numpy as np
from sklearn import svm
import argparse
from sklearn.model_selection import KFold



TRAIN_DATA  = "./data/features.train"
TEST_DATA   = "./data/features.test"
TRIALS      = 100



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

def partition_data(X, Y, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True)
    partitions = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        partitions.append((X_train, Y_train, X_test, Y_test))

    return partitions



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, C=0.01, Q=2):
    model = svm.SVC(C=C, kernel='poly', degree=Q, gamma=1, coef0=1.0)
    model.fit(X, Y)
    return model



########################
## Error Calculations ##
########################

def calc_e(model, X_test, Y_test):
    Y_svm = model.predict(X_test)
    e = np.mean(Y_test != Y_svm)
    return e


def calc_E_cv(C, data):
    sum = 0.
    folds = 0.
    for X, Y, X_test, Y_test in data:
        model = svm_libsvm(X, Y, C=C, Q=2)
        sum += calc_e(model, X_test, Y_test)
        folds += 1.
    
    return sum / folds



#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="Polynomial Kernel Soft Margin SVM")
    parser.add_argument('-d', '--digit', type=int, help='First digit class', required=True)
    parser.add_argument('-o', '--other', type=int, help='Second digit class', required=True)
    args = parser.parse_args()

    C_list = [0.0001, 0.001, 0.01, 0.1, 1]
    C_map = {C: 0 for C in C_list}
    E_cv_map = {C: [] for C in C_list}
    X, Y = load_data(TRAIN_DATA, args.digit, args.other)

    for trial in range(TRIALS):
        partitioned_data = partition_data(X, Y)

        model_map = {C: 1.0 for C in C_list}

        for C in C_list:
            E_cv = calc_E_cv(C=C, data=partitioned_data)
            model_map[C] = E_cv
            E_cv_map[C].append(E_cv)

        best_model = min(model_map, key=model_map.get)
        C_map[best_model] += 1

    best_C = max(C_map, key=lambda k: (C_map[k], k))
    print(f"The model with the most selections is C = {best_C}")

    model = svm_libsvm(X, Y, C=best_C, Q=2)
    best_E_cv = np.mean(E_cv_map[best_C])
    X_test, Y_test = load_data(TEST_DATA, args.digit, args.other)
    E_out = calc_e(model, X_test, Y_test)
    print(f"{args.digit} versus {args.other}")
    print(f"E_cv:\t{best_E_cv:.6f}")  
    print(f"E_out:\t{E_out:.6f}")
    print(f"SVs:\t{round(sum(model.n_support_))}")


if __name__ == "__main__":
    main()
