# Problem Set 5.8-9
# Logistic Regression and Stochastic Gradient Descent Learning Algorithm


import numpy as np
import matplotlib.pyplot as plt
import argparse


TRIALS          = 100
TEST_SAMPLES    = 1000
ETA             = 0.01


def generate_target():
    points = np.random.uniform(-1, 1, (2, 2))
    p0 = points[0]
    p1 = points[1]
    a = p1[1] - p0[1]
    b = p0[0] - p1[0]
    c = p1[0] * p0[1] - p1[1] * p0[0]
    return a, b, c


def generate_data(N, a, b, c):
    X = np.random.uniform(-1, 1, (N, 2))
    X_bias = np.c_[np.ones(N), X]
    Y = np.sign(a * X[:, 0] + b * X[:, 1] + c)
    return X_bias, Y



def calc_gradient(w, x, y):
    return -y*( 1 / (1 + np.exp(y * w.dot(x)) ) ) * x

def logistic_regression(X, Y):
    num_points = X.shape[0]
    dim = X.shape[1]
    w = np.zeros(dim)
    w_prev = None
    
    epoch = 0
    while w_prev is None or np.linalg.norm( w - w_prev ) > 0.01:
        w_prev = w.copy()
        shuffled_indices = np.random.permutation(num_points)
        for i in shuffled_indices:
            x, y = X[i], Y[i]
            w += -ETA * calc_gradient(w, x, y)
        epoch += 1

    return w, epoch

def calc_E_out(w_g, a, b, c, num_samples=TEST_SAMPLES):
    X_test, Y_test = generate_data(num_samples, a, b, c)
    E_out = np.mean(np.log(1 + np.exp(-1*Y_test * X_test.dot(w_g))))
    return E_out


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression and Stochastic Gradient Descent Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate', required=True)
    args = parser.parse_args()

    epochs_list = []
    list_of_E_outs = []

    for _ in range(TRIALS):
        a, b, c = generate_target()
        X, Y = generate_data(args.points, a, b, c)
        w_logistic, epochs = logistic_regression(X, Y)
        list_of_E_outs.append(calc_E_out(w_logistic, a, b, c))
        epochs_list.append(epochs)

    avg_E_out = np.mean(list_of_E_outs)
    avg_epochs = np.mean(epochs_list)

    print("E_out:\t", avg_E_out)
    print("Epochs:\t", avg_epochs)


    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

    # plot the last chosen hypothesis g for visualization purposes
    y_vals_g = (-w_logistic[0] - w_logistic[1]*x_vals) / w_logistic[2]
    plt.plot(x_vals, y_vals_g, 'm--', label='Hypothesis g')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.title("Logistic Regression")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
