# Problem Set 1.7-10


import argparse
import numpy as np
import matplotlib.pyplot as plt


RUNS            = 1000
TEST_SAMPLES    = 1000


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


def calc_disagreement(w_g, a, b, c, num_samples=TEST_SAMPLES):
    X_sample, Y_f = generate_data(num_samples, a, b, c)
    Y_g = np.sign(X_sample.dot(w_g))
    disagreement = np.mean(Y_f != Y_g)
    return disagreement


def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate')
    args = parser.parse_args()

    iterations_list = []
    disagreement_probs = []

    for _ in range(RUNS):
        a, b, c = generate_target()
        X, Y = generate_data(args.points, a, b, c)
        w_g, num_iterations = perceptron_learning_algorithm(X, Y)
        disagreement_prob = calc_disagreement(w_g, a, b, c)
        iterations_list.append(num_iterations)
        disagreement_probs.append(disagreement_prob)

    # Compute average iterations and disagreement probability over 1000 runs
    avg_iterations = np.mean(iterations_list)
    avg_disagreement_prob = np.mean(disagreement_probs)

    print("Iterations:", avg_iterations)
    print("Disagreement:", avg_disagreement_prob)


    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

    # plot the chosen hypothesis g
    y_vals_g = (-w_g[0] - w_g[1]*x_vals) / w_g[2]
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
