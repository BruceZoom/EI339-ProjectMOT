import os
import sys


def to_table(fname):
    infile = open(fname)
    lines = infile.readlines()
    infile.close()

    for line in lines:
        es = [e for e in line.strip().split(' ') if len(e)]
        if len(es) < 16:
            es = ['',] + es
        for i, e in enumerate(es):
            if '%' in e:
                es[i] = e[:-1] + "\%"
        print(" & ".join(es[i] for i in [0, 14, 1, 7, 9, 10, 11, 12]) + "\\\\")


if __name__ == "__main__":
    if len(sys.argv) < 2: print("Too few arguments!")
    to_table(sys.argv[1])
