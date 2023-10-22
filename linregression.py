# Problem Set 2.5 - 7
# Linear Regression


import argparse
import numpy as np
import matplotlib.pyplot as plt


TRIALS            = 1000
TEST_SAMPLES    = 1000



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
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.c_[np.ones(N), X]
    Y = np.sign(a * X[:, 0] + b * X[:, 1] + c)
    return X_with_bias, Y



#########################
## Learning Algorithms ##
#########################

def perceptron_learning_algorithm(X, Y):
    num_points = X.shape[0]
    dim = X.shape[1]

    w = np.zeros(dim)
    num_iterations = 0

    while True:
        predictions = np.sign(X.dot(w))
        misclassified = np.where(predictions != Y)[0]

        if len(misclassified) == 0:
            break

        random_misclassed_point = np.random.choice(misclassified)

        w += Y[random_misclassed_point] * X[random_misclassed_point]
        num_iterations += 1

    return w, num_iterations


def linear_regression(X, Y):
    w_g = np.linalg.inv(X.T @ X) @ X.T @ Y
    return w_g



########################
## Error Calculations ##
########################

def calc_Error_in(w_g, X, Y):
    Y_g = np.sign(X.dot(w_g))
    Error_in = np.mean(Y != Y_g)
    return Error_in

def calc_Error_out(w_g, a, b, c, num_samples=TEST_SAMPLES):
    X_sample, Y_f = generate_data(num_samples, a, b, c)
    Y_g = np.sign(X_sample.dot(w_g))
    Error_out = np.mean(Y_f != Y_g)
    return Error_out



def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate', required=True)
    args = parser.parse_args()

    iterations_list = []
    Error_in_list = []
    Error_out_list = []
    W = np.zeros((3, TRIALS))

    for i in range(TRIALS):
        a, b, c = generate_target()
        X, Y = generate_data(args.points, a, b, c)
        w_regression = linear_regression(X, Y)
        W[:,i] = w_regression
        #iterations_list.append(num_iterations)

        Error_in_list.append(calc_Error_in(w_regression, X, Y))
        Error_out_list.append(calc_Error_out(w_regression, a, b, c))

    # Compute average iterations and Error_out probability over 1000 runs
    #avg_iterations = np.mean(iterations_list)
    avg_Error_in = np.mean(Error_in_list)
    avg_Error_out = np.mean(Error_out_list)

    #print("Iterations:", avg_iterations)
    print("E_in actual:", avg_Error_in)
    print("E_out estimate:", avg_Error_out)



    ## Plot data and hyperplanes for the last experiment trial ##

    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

    # plot the last chosen hypothesis g for visualization purpose
    y_vals_g = (-w_regression[0] - w_regression[1]*x_vals) / w_regression[2]
    plt.plot(x_vals, y_vals_g, 'm--', label='Hypothesis g')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.title("Sample Data and Target Function f")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()