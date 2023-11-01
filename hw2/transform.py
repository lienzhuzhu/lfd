# Problem Set 2.8 - 10
# Linear Regression with transformation and noisy data


import argparse
import numpy as np
import matplotlib.pyplot as plt


TRIALS              = 1000
TEST_SAMPLES        = 1000



################
## Simulation ##
################


def generate_circular_data(N):
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.c_[np.ones(N), X]
    Y = np.sign(X[:, 0]**2 + X[:, 1]**2 - 0.6)
    return X_with_bias, Y

def generate_noisy_circular_data(N):
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.c_[np.ones(N), X]
    Y = np.sign(X[:, 0]**2 + X[:, 1]**2 - 0.6)
    
    # Introduce noise by flipping the output value for 10% of the sample
    num_noisy_samples = int(0.1 * N)
    noisy_indices = np.random.choice(N, num_noisy_samples, replace=False)
    Y[noisy_indices] = -Y[noisy_indices]
    
    return X_with_bias, Y

def generate_transformed_noisy_circular_data(X, Y):
    x1 = X[:, 1]
    x2 = X[:, 2]
    X_transformed = np.c_[X, x1*x2, x1**2, x2**2]
    
    return X_transformed, Y



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

def print_average_weights(weight_matrix):
    avg_weights = np.mean(weight_matrix, axis=0)
    print("Average Weights", end='\t')
    print(", ".join(map(str, avg_weights)))



########################
## Error Calculations ##
########################

def calc_Error_in(g, X, Y):
    Y_g = np.sign(X.dot(g))
    Error_in = np.mean(Y != Y_g)
    return Error_in

def calc_Error_out(g, num_samples=TEST_SAMPLES):
    X_test, Y_f = generate_noisy_circular_data(num_samples)
    Y_g = np.sign(X_test.dot(g))
    Error_out = np.mean(Y_f != Y_g)
    return Error_out

def calc_Error_out_transformed(g, num_samples=TEST_SAMPLES):
    X_test, Y_f = generate_noisy_circular_data(num_samples)
    X_test, Y_f = generate_transformed_noisy_circular_data(X_test, Y_f)
    Y_g = np.sign(X_test.dot(g))
    Error_out = np.mean(Y_f != Y_g)
    return Error_out


#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate', required=True)
    parser.add_argument('--transform', action='store_true', help='Transform the data points')
    args = parser.parse_args()

    iterations_list = []
    Error_in_list = []
    Error_out_list = []

    if args.transform:
        W = np.zeros((TRIALS, 6))
    else:
        W = np.zeros((TRIALS, 3))

    for i in range(TRIALS):
        X_original, Y_original = generate_noisy_circular_data(args.points)
        X, Y = generate_transformed_noisy_circular_data(X_original, Y_original)

        if args.transform:
            g_regression = linear_regression(X, Y)
        else:
            g_regression = linear_regression(X_original, Y_original)

        W[i,:] = g_regression

        if args.transform:
            Error_in_list.append(calc_Error_in(g_regression, X, Y))
            Error_out_list.append(calc_Error_out_transformed(g_regression))
        else:
            Error_in_list.append(calc_Error_in(g_regression, X_original, Y_original))
            Error_out_list.append(calc_Error_out(g_regression))


    # Compute stats from the experiment
    avg_Error_in = np.mean(Error_in_list)
    avg_Error_out = np.mean(Error_out_list)

    print(f"E_in actual: {avg_Error_in:.10f}")
    print("E_out estimate:", avg_Error_out)
    print_average_weights(W)



    #############################################################
    ## Plot data and hyperplanes for the last experiment trial ##
    #############################################################

    ## Plot original data points ##
    plt.figure(figsize=(8, 8))
    # NOTE: the original data set is found within the first two columns of the transformed set
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    ## Plot boundary circle ##
    theta = np.linspace(0, 2*np.pi, 100)
    radius = np.sqrt(0.6)
    circle_x1_values = radius * np.cos(theta)
    circle_x2_values = radius * np.sin(theta)
    plt.plot(circle_x1_values, circle_x2_values, 'k-', label='x1^2 + x2^2 = 0.6')

    ## Plot last regression line ##
    if not args.transform:
        x_vals = np.array([-1, 1])
        y_vals_g = (-g_regression[0] - g_regression[1]*x_vals) / g_regression[2]
        plt.plot(x_vals, y_vals_g, 'r--', label='Regression g')

    ## Global Plot Parameters ##
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.title("Linear Regression and Linear Transformation")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
