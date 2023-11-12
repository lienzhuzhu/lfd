# Problem Set 7.8-10
# Support Vector Machine for Linearly Separable Data


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

def perceptron_learning_algorithm(X, Y, w_init=None):
    num_points = X.shape[0]
    dim = X.shape[1]
    w = w_init if w_init is not None else np.zeros(dim)

    iterations = 0
    while True:
        predictions = np.sign(X.dot(w))
        misclassified = np.where(predictions != Y)[0]

        if len(misclassified) == 0:
            break

        random_misclassed_point = np.random.choice(misclassified)
        w += Y[random_misclassed_point] * X[random_misclassed_point]
        iterations += 1

    return w, iterations


def svm_libsvm(X, Y):
    model = svm.SVC(kernel='linear', C=1e6)
    model.fit(X, Y)
    return model

def svm_primal(X, Y):
    N, D = X.shape
    P = matrix(np.diag([0] + [1.0] * D))
    q = matrix(np.zeros(D + 1))
    G = matrix(-np.hstack((Y.reshape(-1,1), Y.reshape(-1,1) * X)))
    h = matrix(-np.ones((N, 1)))

    solution = solvers.qp(P, q, G, h, options={'show_progress': False})
    variables = np.array(solution['x']).flatten()
    b = variables[0]
    w = variables[1:]
    margins = Y * (X.dot(w) + b)

    threshold = 1e-5
    support_vector_indices = np.where(np.abs(margins - 1) <= threshold)[0]
    num_support_vectors = len(support_vector_indices)

    return solution, num_support_vectors

def svm_dual(X, Y):
    N = X.shape[0]
    Y = Y * 1.0
    X_ = Y.reshape(-1,1) * X # broadcasts y's across data points
    Q = matrix(np.dot(X_, X_.T))
    p = matrix(-np.ones((N, 1)))

    G = matrix(-np.eye(N)) # lecture and book combine inequality and equality into matrix A
    h = matrix(np.zeros(N))
    A = matrix(Y.reshape(1, -1))
    b = matrix(0.0)

    solution = solvers.qp(Q, p, G, h, A, b, options={'show_progress': False})
    alphas = np.array(solution['x']).flatten()

    threshold = 1e-5
    support_vectors = alphas > threshold
    num_support_vectors = np.sum(support_vectors)

    w = np.sum(alphas[support_vectors].reshape(-1,1) * Y[support_vectors].reshape(-1,1) * X[support_vectors], axis=0) # ignoring threshold...
    b = np.mean(Y[support_vectors] - np.dot(X[support_vectors], w))
    g = np.concatenate([[b], w])

    return g, num_support_vectors
    




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
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm and SVM")
    parser.add_argument('-N', '--points', default=10, type=int, help='Number of sample data points to generate')
    args = parser.parse_args()

    pla_E_out_list, svm_E_out_list, primal_E_out_list, dual_E_out_list = [], [], [], []
    support_vectors, primal_support_vectors, dual_support_vectors = [], [], []
    svm_wins, primal_wins, dual_wins = 0, 0, 0

    for i in range(TRIALS):
        a, b, c = generate_target()
        X, X_aug, Y = generate_data(args.points, a, b, c)

        perceptron, _ = perceptron_learning_algorithm(X_aug, Y)
        model = svm_libsvm(X, Y)
        primal, num_primal_support_vectors = svm_primal(X, Y)
        dual, num_dual_support_vectors = svm_dual(X, Y)

        support_vectors.append(sum(model.n_support_))
        primal_support_vectors.append(num_primal_support_vectors)
        dual_support_vectors.append(num_dual_support_vectors)

        pla_E_out, svm_E_out, primal_E_out, dual_E_out = calc_E_out(perceptron, model, primal, dual, a, b, c)

        pla_E_out_list.append(pla_E_out)
        svm_E_out_list.append(svm_E_out)
        primal_E_out_list.append(primal_E_out)
        dual_E_out_list.append(dual_E_out)

        svm_wins += svm_E_out < pla_E_out
        primal_wins += primal_E_out < pla_E_out
        dual_wins += dual_E_out < pla_E_out


    print("PLA E_out:\t\t", np.mean(pla_E_out_list))
    print("SVM E_out:\t\t", np.mean(svm_E_out_list))
    print("Primal E_out:\t\t", np.mean(primal_E_out_list))
    print("Dual E_out:\t\t", np.mean(dual_E_out_list))
    print()

    print("Support Vectors:\t", np.mean(support_vectors), "support vectors")
    print("Primal Vectors:\t\t", np.mean(primal_support_vectors), "support vectors")
    print("Dual Vectors:\t\t", np.mean(dual_support_vectors), "support vectors")
    print()

    print("SVM won\t\t\t", svm_wins / TRIALS, "times")
    print("Primal won\t\t", primal_wins / TRIALS, "times")
    print("Dual won\t\t", dual_wins / TRIALS, "times")



    #############################################################
    ## Plot data and hyperplanes for the last experiment trial ##
    #############################################################

    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

    # plot the last perceptron
    y_vals_g = (-perceptron[0] - perceptron[1]*x_vals) / perceptron[2]
    plt.plot(x_vals, y_vals_g, 'y--', label='Perceptron')

    # plot the last SVM line
    w = model.coef_[0]
    b = model.intercept_[0]
    y_vals_svm = (-b - w[0]*x_vals) / w[1]
    plt.plot(x_vals, y_vals_svm, 'm:', label='SVM')

    # plot the last primal SVM line
    primal_g = np.array(primal['x']).flatten()
    y_vals_primal = (-primal_g[0] - primal_g[1]*x_vals) / primal_g[2]
    plt.plot(x_vals, y_vals_primal, 'r:', label='Primal SVM')
    
    # plot the last dual SVM line
    y_vals_dual = (-dual[0] - dual[1]*x_vals) / dual[2]
    plt.plot(x_vals, y_vals_dual, 'k:', label='Dual SVM')


    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.title("Linear SVM")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
