#!/bin/python

import csv
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy.sparse as ssp

def read_stats(f):
    stats_dict = {}
    with open(f, "r") as sf:
        sfr = csv.reader(sf, delimiter=" ")
        for row in sfr:
            stats_dict[row[0]] = np.array([float(x) for x in row[1:] if x])
    return stats_dict

def txt_to_sparse(filename):
    with open(filename, "r") as tf: 
        tfr = csv.reader(tf, delimiter=" ")
        first_row = next(tfr)
        p = int(first_row[0])
        q = int(first_row[1])
        nnz = int(first_row[2])
        the_rest = [(int(row[0])-1, int(row[1])-1, float(row[2]))
                    for row in tfr]
    (i, j, data) = zip(*the_rest)
    Theta = ssp.coo_matrix((list(data), (list(i),list(j))), shape=(p, q))
    return Theta

def runPseudo(
        Y, X, lambdaLambda, lambdaTheta, 
        verbose=True, max_iters=50, diag_dominant=True):
    '''
    Inputs:
    Y (n samples x q dimensions)
    X (n samples x p dimensions)
    lambdaLambda (regularization for Lambda)
    lambdaTheta (regularization for Theta)
    verbose: print information or not
    max_iters: max number of outer iterations
    diag_dominant: make diagonally dominant
    '''

    olddir = os.getcwd()
    thisdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisdir) # move to demo directory

    dummy = rnd.randint(low=0, high=1e6)

    (n_y, q) = Y.shape
    (n_x, p) = X.shape

    Yfile = "Y-dummy-%i.txt" % (dummy)
    Xfile = "X-dummy-%i.txt" % (dummy)
    Lambdafile = "Lambda-dummy-%i.txt" % (dummy)
    Thetafile = "Theta-dummy-%i.txt" % (dummy)
    statsfile = "stats-dummy-%i.txt" % (dummy)
    np.savetxt(Yfile, Y, fmt="%.10f", delimiter=" ")
    np.savetxt(Xfile, X, fmt="%.10f", delimiter=" ")
    option_str = "-y %f -x %f -v %i -i %i -d %i" % (
        lambdaLambda, lambdaTheta,
        verbose, max_iters, diag_dominant)
    command_str = "%s %s %i %i %i %i %s %s %s %s %s" % (
        "../Pseudo/pseudo_run", option_str,
        n_x, p, n_y, q, Yfile, Xfile,
        Lambdafile, Thetafile, statsfile)

    #ret = os.system(command_str)
    output = subprocess.check_output(command_str, shell=True)
    print(output)
    Lambda = txt_to_sparse(Lambdafile)
    Theta = txt_to_sparse(Thetafile)
    stats = read_stats(statsfile)
    rmline = "rm %s %s %s %s %s" % (Yfile, Xfile, Lambdafile, Thetafile, statsfile)
    ret = os.system(rmline)
    os.chdir(olddir)
    return (Lambda, Theta)

if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.abspath(__file__))
    print(thisdir)
    X = np.loadtxt(thisdir + "/Xfile")
    Y = np.loadtxt(thisdir + "/Yfile")
    lambdaLambda = 0.1
    lambdaTheta = 0.2
    (Lambda, Theta) = runPseudo(Y, X, lambdaLambda, lambdaTheta)
    plt.figure()
    plt.spy(Theta)
    plt.title("Theta")
    plt.figure()
    plt.spy(Lambda)
    plt.title("Lambda")
    plt.figure()
    plt.imshow(Theta.todense(), interpolation="nearest")
    plt.title("Theta dense")
    plt.colorbar()
    plt.show()
