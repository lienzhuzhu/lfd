# Problem Set 6.2-6
# Regularization with Weight Decay


import numpy as np
import matplotlib.pyplot as plt



#######################
### Data Processing ###
#######################

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]  # All rows, all but the last column
    Y = data[:, -1]  # All rows, just the last column
    
    return X, Y



#########################
## Learning Algorithms ##
#########################

def regularized_linear_regression(X, Y, K, N):
    dim = X.shape[1]
    w_reg = np.linalg.inv( X.T @ X + ((10**K)/N) * np.eye(dim) ) @ X.T @ Y
    return w_reg



########################
## Error Calculations ##
########################

def calc_Error_in(g, X, Y):
    Y_g = np.sign(X.dot(g))
    Error_in = np.mean(Y != Y_g)
    return Error_in

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
    iterations_list = []
    Error_in_list = []
    Error_out_list = []


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




if __name__ == "__main__":
    main()
