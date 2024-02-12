# script to solve full isotropic Eliashberg equation and isotropic LEE
# we use a general alpha2F for Niobium and calculate the lambda matrix
# No of mastubara freq is 2*# times the max phonon freq (n=#*max_ph_freq ie m=2*#*max_ph_freq, where m is total #matsubara freq)
# We use previous delta as the initial guess for the next T
# We still need to work on changing # matsubara freq with T

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import csv
import sys

sys.setrecursionlimit(50000)

def Rn(delta, omega):
    R = np.add(np.square(omega), np.square(delta))
    return R


def Z(R, omega, lamda, T):
    #X = (omega[:, None] / np.sqrt(R)) * lamda
    X = (omega / np.sqrt(R)) * lamda
    zz = 1 + (np.pi * T / omega) * np.sum(X, axis=1)
    return zz

def gap(zz, delta, R, omega, lamda, T, mu):
    #Y = (delta[:, None] / np.sqrt(R)) * (lamda - mu)
    Y = (delta / np.sqrt(R)) * (lamda - mu)
    delta_new = (np.pi * T / zz) * np.sum(Y, axis=1)
    return delta_new

def self_consistent_solution(
    Rn, Z, gap, delta_previous, omega, lamda, T, mu, convergence
):
    zz = Z(Rn(delta_previous, omega), omega, lamda, T)
    delta = gap(zz, delta_previous, Rn(delta_previous, omega), omega, lamda, T, mu)
    
    zz_new = Z(Rn(delta, omega), omega, lamda, T)
    delta_new = gap(zz, delta, Rn(delta, omega), omega, lamda, T, mu)

    while (
        np.amax(np.abs((zz_new - zz) / (zz))) > convergence
        or np.amax(np.abs((delta_new - delta) / (delta))) > convergence
    ):
        zz = zz_new.copy()  # update Z
        delta = delta_new.copy()
        # I'm using Rn twice, can I use it once?
        zz_new = Z(Rn(delta, omega), omega, lamda, T)
        delta_new = gap(zz, delta, Rn(delta, omega), omega, lamda, T, mu)

    return zz_new, delta_new

    # function for 1st iteration?
def first_itr(T, delta_0):
    omega = np.asarray([(2 * (-n + i) + 1) * np.pi * T for i in range(m)])
    """
    lamda = np.asarray(
        [
            [
                2
                * omega_0
                * ng2
                / (np.square(omega[i] - omega[j]) + np.square(omega_0))
                for j in range(len(omega))
            ]
            for i in range(len(omega))
        ]
    )
    """
    lamda = compute_lambda(omega, a2f_val, omega_val, bin_num=bin_num)
    print("lamda_max=", np.amax(lamda))

    zz_new, delta_new = self_consistent_solution(
    Rn, Z, gap, delta_0, omega, lamda, T, mu, convergence
    )

    TT.append(T)
    del_max.append(np.amax(delta_new))
    zz_max.append(np.amax(zz_new))

    #K = cal_K_matrix(lamda, mu, T, omega, Z(omega, lamda, T))

    return  zz_new, delta_new, TT, del_max, zz_max #, K

def del_t(T, delta_old):

    omega = np.asarray([(2 * (-n + i) + 1) * np.pi * T for i in range(m)])
    """
    lamda = np.asarray(
        [
            [
                2
                * omega_0
                * ng2
                / (np.square(omega[i] - omega[j]) + np.square(omega_0))
                for j in range(len(omega))
            ]
            for i in range(len(omega))
        ]
    )
    
    # print the maximum element of lamda matrix
    print("lamda_max=", np.amax(lamda))
    # converge for self consistency
    """

    lamda = compute_lambda(omega, a2f_val, omega_val, bin_num=bin_num)
    print("lamda_max=", np.amax(lamda))
    # converge for self consistency
    zz_new, delta_new = self_consistent_solution(
                    Rn, Z, gap, delta_old, omega, lamda, T, mu, convergence
    )
    
    K = cal_K_matrix(lamda, mu, T, omega, Z_lgs(omega, lamda, T))
    
    return zz_new, delta_new, K

def cal2(delta_previous, T, T_del):

    delta_old = delta_previous.copy()
    zz_new, delta_new, K = del_t(T+T_del, delta_old)

    while np.amax(np.abs(delta_old))-np.amax(np.abs(delta_new)) > 0.15*np.amax(np.abs(delta_init)):
        T_del = T_del*0.5
        zz_new, delta_new, K = del_t(T+T_del, delta_old)
        
    TT.append(T+T_del)
    del_max.append(np.amax(delta_new))
    zz_max.append(np.amax(zz_new))


    # make a file and write the data in it too
    # also do the linearized gap here
    eigenvalues, eigenvectors = eigenvalue_prob(K)


    # maka the data file
    with open('data3_2.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([T+T_del, eigenvalues[0], np.amax(delta_new), np.amax(zz_new)])

    return delta_new, T+T_del, T_del, TT, del_max, zz_max

def rec(delta_old, T, T_del):

    delta_new=delta_old.copy()
    while np.amax(np.abs(delta_new)) > 0.08*np.amax(np.abs(delta_init)):
        delta_new, T , T_del, TT, del_max, zz_max = cal2(delta_old, T, T_del)
        delta_old = delta_new.copy()
        print("del_new",np.amax(np.abs(delta_new))) 
        
        print("T",T)
        print("T_del",T_del)
        print("TT",TT)
        print("del_max",del_max)
        print("zz_max",zz_max)
    return delta_new, T , T_del, TT, del_max, zz_max

#linearized gap equation solver
def cal_K_matrix(lambda_matrix, mu, T, omega, z):
    numerator = np.pi*T*(lambda_matrix - mu)
    denominator = z * np.sqrt(np.square(omega))
    return np.divide(numerator,denominator)

def eigenvalue_prob(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

def Z_lgs(omega, lamda, T):
    #X = (omega[:, None] / np.sqrt(R)) * lamda
    X = (omega / np.sqrt(np.square(omega))) * lamda
    zz = 1 + (np.pi * T / omega) * np.sum(X, axis=1)
    return zz

# General Lambda calculation
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

import dask

@dask.delayed
def compute_element(n, m, ωn, alpha2F, omega_val, bin_num):
    # Define the integrand for your formula
    def integrand(ω):
        return (2 * ω * alpha2F(ω)) / ((ωn[n] - ωn[m]) ** 2 + ω ** 2)
        
    result, _ = integrate.quad(integrand, omega_val[0], omega_val[-1], limit=bin_num)
    return result

def create_matrix(ωn, alpha2F, omega_val, bin_num):
    N = len(ωn)
    λ_matrix_delayed = np.empty((N, N), dtype=object)

    for n in range(N):
        for m in range(N):
            λ_matrix_delayed[n, m] = compute_element(n, m, ωn, alpha2F, omega_val, bin_num)

    return λ_matrix_delayed

def compute_matrix(λ_matrix_delayed):
    λ_matrix = dask.compute(*λ_matrix_delayed.flatten())
    λ_matrix = np.array(λ_matrix).reshape(λ_matrix_delayed.shape)
    return λ_matrix

def compute_lambda(omega, α2F_values, omega_val, bin_num=200, ω_upper_limit=10):
    
    # Interpolate α2F_values
    #alpha2F = interp1d(omega_val, α2F_values, kind='cubic', fill_value=0, bounds_error=False)
    alpha2F = CubicSpline(omega_val, α2F_values, extrapolate=True)

    # Calculate Matsubara frequencies
    ωn = omega

    # Create matrix of dask.delayed objects
    λ_matrix_delayed = create_matrix(ωn, alpha2F, omega_val, bin_num)

    # Compute the matrix
    λ_matrix = compute_matrix(λ_matrix_delayed)

    return λ_matrix




# parameters
mu = 0.16
print("mu=", mu)

convergence = 1e-6
print("convergence=", convergence)

"""
omega_0 = 0.05
print("omega_0=", omega_0)
ng2 = 0.014
print("ng2=", ng2)
"""

# load calculated alpha2F
import pandas as pd
df = pd.read_csv("nb_a2f_2.csv")
# second column is omega_val
omega_val = df.iloc[:, 0].values
# third column is alpha2F_values
a2f_val = df.iloc[:, 1].values
bin_num = (len(omega_val) - 1)

max_ph_freq = omega_val[-1]
kkk = (max_ph_freq/(np.pi*0.001) - 1)*0.5
kkk= int(kkk) + 1

n = 15*kkk # this means (2*#) times the max ph freq is the # of matsu freq.
print("n=", n)
m = 2 * n
print("m=", m)

delta_0 = np.full((m,), 0.005)

# calculate
TT = []
del_max = []
zz_max = []

T = 0.00014
print("initial T=", T)
T_del = 0.00002
print("initial T_del=", T_del)

zz_init, delta_init, TT, del_max, zz_max = first_itr(T, delta_0)

delta_previous = delta_init.copy()
print("delta_init=", np.amax(delta_init))
zz_previous = zz_init.copy()
print("zz_init=", np.amax(zz_init))


delta_previous, T, T_del, TT, del_max, zz_max = rec(delta_previous, T, T_del)

print("TT=", TT)
print("del_max=", del_max)
print("zz_max=", zz_max)
