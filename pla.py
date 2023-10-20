# Problem Set 1.7-10


import argparse
import numpy as np


def make_target_function():


def generate_data(N):
    pass

def main():
    parser = argparse.ArgumentParser(description="Perceptron Learning Algorithm")
    parser.add_argument('-N', '--points', type=int, help='Number of sample data points to generate')
    args = parser.parse_args()

    make_target_function()
    generate_data(args.points)


if __name__ == "__main__":
    main()
