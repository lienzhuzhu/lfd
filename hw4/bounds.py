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
    parser = argparse.ArgumentParser(description="Generalization Bound Calculator")
    parser.add_argument('-N', '--points', type=int, help='Number of sample points', required=True)
    args = parser.parse_args()

    print("VC Bound\t\t\t", Hoeffding(args.points))
    print("Rademacher\t\t\t", Rademacher(args.points))
    print("Parrondo and Van den Broek\t", Parrondo(args.points))
    print("Devroye\t\t\t\t", Devroye(args.points))

    plt.figure(figsize=(14, 8))

    x_vals = numpy.arange(-1000, 12000, 0.1)

    y_Hoeffding = []
    y_Rademacher = []
    y_Parrondo = []
    y_Devroye = []

    for x in x_vals:
        y_Hoeffding.append(Hoeffding(x))
        y_Rademacher.append(Rademacher(x))
        y_Parrondo.append(Parrondo(x))
        y_Devroye.append(Devroye(x))

    plt.plot(x_vals, y_Hoeffding, color='black', label='Hoeffding VC')
    plt.plot(x_vals, y_Rademacher, color='purple', label='Rademacher')
    plt.plot(x_vals, y_Parrondo, color='blue', label='Parrondo')
    plt.plot(x_vals, y_Devroye, color='red', label='Devroye')

    plt.xlim(-1000, 12000)
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Bound")
    plt.yscale("log")
    plt.show()
            

if __name__ == "__main__":
    main()
