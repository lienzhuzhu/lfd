# Problem Set 5.4-6
# Gradient Descent


import numpy


def calc_error(coords):
    a, b = coords
    return (a * numpy.exp(b) - 2. * b * numpy.exp(-a)) ** 2.

def calc_gradient(coords):
    a, b = coords
    partial_a = 2. * (a*numpy.exp(b) - 2.*b*numpy.exp(-a)) * (numpy.exp(b) + 2.*b*numpy.exp(-a))
    partial_b = 2. * (a*numpy.exp(b) - 2.*b*numpy.exp(-a)) * (a*numpy.exp(b) - 2.*numpy.exp(-a))
    gradient = numpy.array([partial_a, partial_b])
    return gradient


def main():
    ETA = 0.1
    coords = numpy.array([1., 1.])
    iterations = 0
    while calc_error(coords) > (10 ** -14):
        coords += -ETA * calc_gradient(coords)
        iterations += 1

    print(iterations)
    print(coords)



if __name__ == "__main__":
    main()
