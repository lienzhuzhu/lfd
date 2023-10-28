# Problem Set 4.2 - 3
# Other generalization bounds


import argparse
import numpy
import matplotlib.pyplot as plt


VC_DIM  = float(50)
DELTA   = 0.05


def Hoeffding(N):
    return numpy.sqrt( (8 / N) * numpy.log(4 * ((2*N)**VC_DIM + 1) / DELTA) )


def Rademacher(N):
    return numpy.sqrt( 2 * numpy.log( 2*N*(N**VC_DIM+1) ) / N ) + numpy.sqrt( (2/N)*numpy.log(1/DELTA) ) + (1/N)


def Parrondo(N):
    return (1/N) + (1/N) * numpy.sqrt( 1 + N * numpy.log( 6*((2*N)**VC_DIM+1) / DELTA ) )


def Devroye(N):
    return ( 2 + numpy.sqrt( (2*N-4) * (numpy.log(4/DELTA) + 2*VC_DIM*numpy.log(N)-numpy.log(DELTA)) ) ) / (2*N-4)


def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample points', required=True)
    args = parser.parse_args()

    print("VC Bound\t\t\t", Hoeffding(args.points))
    print("Rademacher\t\t\t", Rademacher(args.points))
    print("Parrondo and Van den Broek\t", Parrondo(args.points))
    print("Devroye\t\t\t\t", Devroye(args.points))


            

if __name__ == "__main__":
    main()
