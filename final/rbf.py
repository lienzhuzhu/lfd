# Final
# Hard Margin Support Vector Machine with RBF Kernel


import numpy as np
from sklearn import svm
import argparse



TRIALS          = 1000
TEST_POINTS     = 1000


#######################
### Data Processing ###
#######################

def generate_data(N=100):
    X = np.random.uniform(-1, 1, (N, 2))
    Y = np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(np.pi * X[:, 0]))
    return X, Y



#########################
## Learning Algorithms ##
#########################

def svm_libsvm(X, Y, G, C=1e6):
    model = svm.SVC(C=C, kernel='rbf', gamma=G)
    model.fit(X, Y)
    return model


def find_centers_naive(X, K):
    clusters = {center_index : [] for center_index in range(K)}
    cluster_assignments = np.random.randint(0, K, X.shape[0])
    clusters = {k: X[cluster_assignments == k].tolist() for k in range(K)}

    centers = X[np.random.choice(X.shape[0], K, replace=False)]
    prev_centers = np.zeros_like(centers)
    while not np.allclose(centers, prev_centers):
        better_clusters = {center_index : [] for center_index in range(K)}
        for cluster in clusters.values():
            for x in cluster:
                better_cluster_id = np.argmin(np.linalg.norm(centers - x, axis=1))
                better_clusters[better_cluster_id].append(x)

        for i, cluster in enumerate(better_clusters.values()):
            if not cluster:
                centers[i] = X[np.random.choice(X.shape[0])]
            else:
                centers[i] = np.mean(cluster, axis=0)

        prev_centers = centers.copy()
        clusters = better_clusters

    return centers


def find_centers(X, K):
    centers = X[np.random.choice(X.shape[0], K, replace=False)]
    prev_centers = np.zeros_like(centers)

    while not np.allclose(centers, prev_centers):
        diff = np.linalg.norm(X[ :, np.newaxis, : ] - centers[np.newaxis, : , : ], axis=2) # linalg.norm() used because argmin(.) is the same as squared distance
        closest = np.argmin(diff, axis=1)

        prev_centers = centers.copy()
        for k in range(K):
            if np.any(closest == k):
                centers[k] = X[closest == k].mean(axis=0)
            else:
                centers[k] = X[np.random.choice(X.shape[0], 1, replace=False)]

    return centers


def compute_rbf_matrix(X, centers, gamma):
    diff = X[:, np.newaxis, :] - centers[np.newaxis, : , :]
    sq_dist = np.sum(diff ** 2, axis=2)
    K_rbf = np.exp(-gamma * sq_dist)
    
    return K_rbf


def rbf_model_train(X, Y, K, gamma):
    centers = find_centers(X, K)
    K_rbf = compute_rbf_matrix(X, centers, gamma)
    coeffs = np.matmul(np.linalg.pinv(K_rbf), Y)
    return centers, coeffs


def rbf_model_predict(X_test, centers, coeffs, gamma):
    K_rbf = compute_rbf_matrix(X_test, centers, gamma)
    return np.sign(np.dot(K_rbf, coeffs))



########################
## Error Calculations ##
########################

def calc_e(model, X_test, Y_test):
    Y_svm = model.predict(X_test)
    e = np.mean(Y_test != Y_svm)
    return e


def calc_e_rbf(centers, coeffs, X_test, Y_test, gamma):
    Y_rbf = rbf_model_predict(X_test, centers, coeffs, gamma)
    e = np.mean(Y_test != Y_rbf)
    return e


#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="RBF Kernel vs RBF Model")
    parser.add_argument('--gamma', type=float, help='Gamma parameter for Gaussian spread')
    parser.add_argument('--centers', type=int, help='Number of centers K')
    args = parser.parse_args()

    gamma = args.gamma if args.gamma else 1.5
    K = args.centers if args.centers else 9

    svm_in_list, svm_out_list, rbf_in_list, rbf_out_list = [], [], [], []
    inseparable_freq = 0

    svm_wins = 0
    rbf_perfect = 0

    trial = 0
    while trial < TRIALS:
        X, Y = generate_data()

        model = svm_libsvm(X, Y, gamma)
        svm_in = calc_e(model, X, Y)
        if svm_in != 0:
            inseparable_freq += 1
            continue

        centers, coeffs = rbf_model_train(X, Y, K, gamma)
        rbf_in = calc_e_rbf(centers, coeffs, X, Y, gamma)

        X_test, Y_test = generate_data(TEST_POINTS)
        svm_out = calc_e(model, X_test, Y_test)
        rbf_out = calc_e_rbf(centers, coeffs, X_test, Y_test, gamma)

        svm_in_list.append(svm_in)
        rbf_in_list.append(rbf_in)
        svm_out_list.append(svm_out)
        rbf_out_list.append(rbf_out)

        svm_wins += rbf_out > svm_out
        rbf_perfect += rbf_in == 0.
        trial += 1

    print()
    print(f"Data was inseparable in Z space {(inseparable_freq * 100 / TRIALS)}%")
    print()
    print(f"SVM Kernel beat RBF Model {svm_wins * 100 / TRIALS}%")
    print()
    print(f"SVM E_in:\t{np.mean(svm_in_list):.4f}")
    print(f"RBF E_in:\t{np.mean(rbf_in_list):.4f}\tand was zero {rbf_perfect * 100 / TRIALS}%")
    print()
    print(f"SVM E_out:\t{np.mean(svm_out_list):.4f}")
    print(f"RBF E_out:\t{np.mean(rbf_out_list):.4f}")
    


if __name__ == "__main__":
    main()
