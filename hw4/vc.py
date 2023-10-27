# Learning From Data Problem Set 4.1
# Sample Complexity


import numpy

def calc_sample_size(N, ETA=0.05, DELTA=0.05, VC_DIM=10):
    return (8/(ETA ** 2)) * numpy.log( (4 * ((2 * N) ** VC_DIM) + 4) / DELTA )


def main():
    N = 400000
    last_sample = 0

    while N - last_sample > 100:
        last_sample = N
        N = calc_sample_size(N)

    print(N)

if __name__ == "__main__":
    main()
