# Problem Set 8
# Support Vector Machine with Soft Margin


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from cvxopt import matrix, solvers



TRIALS              = 1000
TEST_SAMPLES        = 1000



################
## Simulation ##
################

def generate_target():
    points = np.random.uniform(-1, 1, (2, 2))
    p0 = points[0]
    p1 = points[1]
    a = p1[1] - p0[1]
    b = p0[0] - p1[0]
    c = p1[0] * p0[1] - p1[1] * p0[0]
    return a, b, c

def generate_data(N, a, b, c):
    while True:
        X = np.random.uniform(-1, 1, (N, 2))
        X_with_bias = np.c_[np.ones(N), X]
        Y = np.sign(a * X[:, 0] + b * X[:, 1] + c)
        
        # Check if at least one label is different from the others
        if len(np.unique(Y)) > 1:
            return X, X_with_bias, Y



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, kern='linear', c=1e6):
    model = svm.SVC(kernel=kern, C=c)
    model.fit(X, Y)
    return model



########################
## Error Calculations ##
########################

def calc_E_out(g, model, primal, dual, a, b, c, num_samples=TEST_SAMPLES):
    X_test, X_test_aug, Y_f = generate_data(num_samples, a, b, c)
    Y_pla = np.sign(X_test_aug.dot(g))
    Y_svm = model.predict(X_test)
    Y_primal = np.sign(X_test_aug.dot(np.array(primal['x']).flatten()))
    Y_dual = np.sign(X_test_aug.dot(dual))

    pla_E_out = np.mean(Y_f != Y_pla)
    svm_E_out = np.mean(Y_f != Y_svm)
    primal_E_out = np.mean(Y_f != Y_primal)
    dual_E_out = np.mean(Y_f != Y_dual)

    return pla_E_out, svm_E_out, primal_E_out, dual_E_out



#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="Soft Margin SVM with Transformations")
    parser.add_argument('-N', '--points', default=10, type=int, help='Number of sample data points to generate')
    args = parser.parse_args()

    pla_E_out_list, svm_E_out_list, primal_E_out_list, dual_E_out_list = [], [], [], []
    support_vectors, primal_support_vectors, dual_support_vectors = [], [], []
    svm_wins, primal_wins, dual_wins = 0, 0, 0

    a, b, c = generate_target()
    X, X_aug, Y = generate_data(args.points, a, b, c)

    model = svm_libsvm(X, Y)

    support_vectors.append(sum(model.n_support_))


    print("SVM E_out:\t\t", np.mean(svm_E_out_list))
    print()
    print("Support Vectors:\t", np.mean(support_vectors), "support vectors")


if __name__ == "__main__":
    main()
