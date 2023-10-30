# Problem Set 2.5 - 7
# Linear Regression


import argparse
import numpy as np
import matplotlib.pyplot as plt


TRIALS              = 20
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
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.c_[np.ones(N), X]
    Y = np.sign(a * X[:, 0] + b * X[:, 1] + c)
    return X_with_bias, Y

def generate_noisy_data(N, a, b, c):
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.c_[np.ones(N), X]
    Y = np.sign(a * X[:, 0] + b * X[:, 1] + c)

    # Introduce noise by flipping the output value for 10% of the sample
    num_noisy_samples = int(0.1 * N)
    noisy_indices = np.random.choice(N, num_noisy_samples, replace=False)
    Y[noisy_indices] = -Y[noisy_indices]
    
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


def linear_regression(X, Y):
    g = np.linalg.inv(X.T @ X) @ X.T @ Y
    return g


def perceptron_learning_step(X, Y, w):

    predictions = np.sign(X.dot(w))
    misclassified = np.where(predictions != Y)[0]

    if len(misclassified) == 0:
        return w

    random_misclassed_point = np.random.choice(misclassified)
    w += Y[random_misclassed_point] * X[random_misclassed_point]

    return w
    


def pocket_learning(X, Y, w_init=None, iterations=10000):
    num_points = X.shape[0]
    dim = X.shape[1]
    w_optimal = w_init if w_init is not None else np.zeros(dim)

    for i in range(iterations):
        optimal_Error_in = calc_Error_in(w_optimal, X, Y)
        w_candidate = perceptron_learning_step(X, Y, w_optimal.copy())

        if np.array_equal(w_candidate, w_optimal):
            return w_optimal

        candidate_Error_in = calc_Error_in(w_candidate, X, Y)
        w_optimal = w_candidate if candidate_Error_in < optimal_Error_in else w_optimal

    return w_optimal



########################
## Error Calculations ##
########################

def calc_Error_in(g, X, Y):
    Y_g = np.sign(X.dot(g))
    Error_in = np.mean(Y != Y_g)
    return Error_in

def calc_Error_out(g, a, b, c, num_samples=TEST_SAMPLES):
    X_test, Y_f = generate_data(num_samples, a, b, c)
    Y_g = np.sign(X_test.dot(g))
    Error_out = np.mean(Y_f != Y_g)
    return Error_out



#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate', required=True)
    parser.add_argument('--noisy', action='store_true', help='Indicate if you would like noisy data')
    args = parser.parse_args()

    iterations_list = []
    Error_in_list = []
    Error_out_list = []
    W = np.zeros((3, TRIALS))

    for i in range(TRIALS):
        a, b, c = generate_target()

        if args.noisy:
            X, Y = generate_noisy_data(args.points, a, b, c)
        else:
            X, Y = generate_data(args.points, a, b, c)

        # regression
        g_regression = linear_regression(X, Y)
        W[:,i] = g_regression

        if args.noisy:
            g_pocket = pocket_learning(X, Y, g_regression.copy())
        else:
            g_perceptron, iterations = perceptron_learning_algorithm(X, Y, g_regression.copy())
            iterations_list.append(iterations)

        Error_in_list.append(calc_Error_in(g_regression, X, Y))
        Error_out_list.append(calc_Error_out(g_regression, a, b, c))

    # Compute stats over 1000 runs
    avg_Error_in = np.mean(Error_in_list)
    avg_Error_out = np.mean(Error_out_list)
    print(f"E_in actual: {avg_Error_in:.10f}")
    print("E_out estimate:", avg_Error_out)

    if not args.noisy:
        avg_iterations = np.mean(iterations_list)
        print("Iterations:", avg_iterations)


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

    # plot the last regression line
    y_vals_regression = (-g_regression[0] - g_regression[1]*x_vals) / g_regression[2]
    plt.plot(x_vals, y_vals_regression, 'r-', label='Regression g')
    
    if args.noisy:
        y_vals_pocket = (-g_pocket[0] - g_pocket[1]*x_vals) / g_pocket[2]
        plt.plot(x_vals, y_vals_pocket, 'm--', label='Pocket g')
    else:
        y_vals_g = (-g_perceptron[0] - g_perceptron[1]*x_vals) / g_perceptron[2]
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
