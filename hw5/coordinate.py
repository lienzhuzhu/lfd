# Problem Set 5.7
# Coordinate Descent


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

def calc_partial_a(coords):
    a, b = coords
    partial_a = 2. * (a*numpy.exp(b) - 2.*b*numpy.exp(-a)) * (numpy.exp(b) + 2.*b*numpy.exp(-a))
    return partial_a

def calc_partial_b(coords):
    a, b = coords
    partial_b = 2. * (a*numpy.exp(b) - 2.*b*numpy.exp(-a)) * (a*numpy.exp(b) - 2.*numpy.exp(-a))
    return partial_b


def main():
    ETA = 0.1
    coords = numpy.array([1., 1.])

    for i in range(15):
        coords[0] += -ETA * calc_partial_a(coords)
        coords[1] += -ETA * calc_partial_b(coords)

    print(coords)
    print(calc_error(coords))


if __name__ == "__main__":
    main()
