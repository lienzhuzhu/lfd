# Problem Set 7.8-10
# Support Vector Machine for Linearly Separable Data


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm



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
            return X_with_bias, Y



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
    model = svm.SVC(kernel='linear', C=1e5)
    model.fit(X, Y)
    return model

def svm_qp(X, Y):
    pass




########################
## Error Calculations ##
########################

def calc_E_out(g, model, a, b, c, num_samples=TEST_SAMPLES):
    X_test, Y_f = generate_data(num_samples, a, b, c)
    Y_pla = np.sign(X_test.dot(g))
    Y_svm = model.predict(X_test)
    pla_E_out = np.mean(Y_f != Y_pla)
    svm_E_out = np.mean(Y_f != Y_svm)
    return pla_E_out, svm_E_out


#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', default=10, type=int, help='Number of sample data points to generate')
    args = parser.parse_args()

    pla_E_out_list = []
    svm_E_out_list = []
    support_vectors = []
    svm_wins = 0

    for i in range(TRIALS):
        a, b, c = generate_target()
        X, Y = generate_data(args.points, a, b, c)

        g_perceptron, _ = perceptron_learning_algorithm(X, Y)
        model = svm_libsvm(X, Y)
        support_vectors.append(sum(model.n_support_))

        pla_E_out, svm_E_out = calc_E_out(g_perceptron, model, a, b, c)
        pla_E_out_list.append(pla_E_out)
        svm_E_out_list.append(svm_E_out)
        svm_wins += svm_E_out < pla_E_out

    print("PLA E_out:\t\t", np.mean(pla_E_out_list))
    print("SVM E_out:\t\t", np.mean(svm_E_out_list))
    print("Support Vectors:\t", np.mean(support_vectors))
    print("SVM won", svm_wins / TRIALS, "times")



    #############################################################
    ## Plot data and hyperplanes for the last experiment trial ##
    #############################################################

    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

    # plot the last SVM line
    w = model.coef_[0]
    b = model.intercept_[0]
    y_vals_svm = (-b - w[0]*x_vals) / w[1]
    plt.plot(x_vals, y_vals_svm, 'r--', label='SVM hyperplane')

    # plot the last chosen hypothesis g for visualization purpose
    y_vals_g = (-g_perceptron[0] - g_perceptron[1]*x_vals) / g_perceptron[2]
    plt.plot(x_vals, y_vals_g, 'm--', label='Hypothesis g')

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
