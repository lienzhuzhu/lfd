# Final 11-12
# Hard Margin Support Vector Machine with Polynomial Kernel


import numpy as np
from sklearn import svm
from cvxopt import matrix, solvers
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


def svm_dual(X, Y, Q=2):
    N = X.shape[0]
    Y = Y * 1.0
    Y_ = np.dot(Y, Y.T) 
    X_ = (1 + np.dot(X, X.T)) ** Q

    Q = matrix(Y_ * X_)
    p = matrix(-np.ones((N, 1)))

    G = matrix(-np.eye(N)) # lecture and book combine inequality and equality into matrix A
    h = matrix(np.zeros(N))
    A = matrix(Y.reshape(1, -1))
    b = matrix(0.0)

    solution = solvers.qp(Q, p, G, h, A, b, options={'show_progress': False})
    alphas = np.array(solution['x']).flatten()

    threshold = 1e-3
    support_vectors = alphas > threshold
    num_support_vectors = np.sum(support_vectors)

    w = np.sum(alphas[support_vectors].reshape(-1,1) * Y[support_vectors].reshape(-1,1) * X[support_vectors], axis=0) # ignoring threshold...
    b = np.mean(Y[support_vectors] - np.dot(X[support_vectors], w))
    g = np.concatenate([[b], w])

    print(np.round(alphas, 2))

    return g, num_support_vectors


#################
## Driver Code ##
#################

def main():
    X, Y = load_data()
    model = svm_libsvm(X, Y)
    print(f"libsvm:\t{sum(model.n_support_)}")

    _, dual = svm_dual(X, Y)
    print(f"Using threshold of {1e-3}")
    print(f"Dual:\t{dual}")

if __name__ == "__main__":
    main()
