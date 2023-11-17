# Final
# Hard Margin Support Vector Machine with RBF Kernel


import numpy as np
from sklearn import svm
import argparse



TRIALS          = 1
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

def svm_libsvm(X, Y, C=1e6, G=1.5):
    model = svm.SVC(C=C, kernel='rbf', gamma=G)
    model.fit(X, Y)
    return model


def find_centers(X, K=9):
    #initialize K random centers from the data set X
    #while the element-wise difference of the centers matrix is non-zero
    # 1. assign each point in X to a center in clusters dictionary (maintain an original copy of this dictionary to iterate through) based on closest center
    # 2. update centers to average point of constituent points in cluster

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

        for cluster in better_clusters.values():
            if not cluster:
                raise ValueError("Empty Cluster")

        prev_centers = centers.copy()
        centers = np.array([np.mean(cluster, axis=0) for cluster in better_clusters.values()])
        clusters = better_clusters

    return centers



def rbf_model(X, Y):
    find_centers(X)
    #use pseudo inverse to find coefficients
    #predictions = sign(signal) = sign(coefficients * rbf matrix)

def rbf_model_predict():
    pass

########################
## Error Calculations ##
########################

def calc_e(model, X_test, Y_test):
    Y_svm = model.predict(X_test)
    e = np.mean(Y_test != Y_svm)
    return e



#################
## Driver Code ##
#################

def main():
    parser = argparse.ArgumentParser(description="RBF Kernel vs RBF Model")
    parser.add_argument('--gamma', type=int, help='Gamma parameter for Gaussian spread')
    parser.add_argument('--centers', type=int, help='Number of centers K')
    args = parser.parse_args()

    gamma = args.gamma if args.gamma else 1.5
    K = args.centers if args.centers else 9

    svm_In_list, svm_Out_list, rbf_In_list, rbf_Out_list = [], [], [], []
    inseparable_freq = 0

    trial = 0
    while trial < TRIALS:
        X, Y = generate_data()

        model = svm_libsvm(X, Y, G=gamma)
        svm_in = calc_e(model, X, Y)
        if svm_in != 0:
            inseparable_freq += 1
            continue

        try:
            centers = rbf_model(X, Y)
        except ValueError as e:
            continue


        trial += 1

    print()
    print(f"Bad RBF SVM Problem {(inseparable_freq / TRIALS):.2f} times")


if __name__ == "__main__":
    main()
