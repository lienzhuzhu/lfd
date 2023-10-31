# Problem Set 5.8-9
# Logistic Regression and Stochastic Gradient Descent Learning Algorithm


import numpy


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


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression and Stochastic Gradient Descent Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate', required=True)
    args = parser.parse_args()

    a, b, c = generate_target()
    X, Y = generate_data(args.points, a, b, c)

    # plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1][Y == 1], X[:, 2][Y == 1], color='blue', marker='o', label='Class +1')
    plt.scatter(X[:, 1][Y == -1], X[:, 2][Y == -1], color='red', marker='x', label='Class -1')

    # plot target function
    x_vals = np.array([-1, 1])
    y_vals = (-c - a*x_vals) / b
    plt.plot(x_vals, y_vals, 'k-', label='Target function f')

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
