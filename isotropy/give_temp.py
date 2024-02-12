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
    zz = []
    for i in range((len(omega))):
        x_1 = 0  # initialize
        for j in range(len(omega)):
            x_0 = (omega[j] / np.sqrt(R[j])) * lamda[i, j]
            x_1 = x_1 + x_0
        zz.append(1 + (np.pi * T / omega[i]) * x_1)
    zz = np.asarray(zz)
    return zz


def gap(zz, delta, R, omega, lamda, T, mu):
    # two inner loop, first to calculate delta array, the second to calculate sum inside eq 35
    delta_old = delta.copy()
    delta = []
    for i in range(len(omega)):
        y_1 = 0  # initialize
        for j in range(len(omega)):  # Calculate the sum inside eq 35
            y_0 = (delta_old[j] / np.sqrt(R[j])) * (lamda[i, j] - mu)
            y_1 = y_1 + y_0
        delta.append((np.pi * T / zz[i]) * y_1)  # delta updated
    delta = np.asarray(delta)
    return delta


def selfconsistency(convergence, zz_new, zz, delta_new, delta, omega, lamda, T, mu):

    if (
        np.amax(np.abs((zz_new - zz) / (zz))) > convergence
        or np.amax(np.abs((delta_new - delta) / (delta))) > convergence
    ):
        zz = zz_new.copy()  # update Z
        delta = delta_new.copy()
        R = Rn(delta, omega)
        zz_new = Z(R, omega, lamda, T)
        delta_new = gap(zz, delta, R, omega, lamda, T, mu)

        print("non-consistent", np.amax(delta), np.amax(zz))
        zz, delta = selfconsistency(
            convergence, zz_new, zz, delta_new, delta, omega, lamda, T, mu
        )
    else:
        print("consistent", np.amax(delta), np.amax(zz))
    return zz, delta


# parameters
mu = 0
n = 300
m = 2 * n
convergence = 1e-3

omega_0 = 0.05
ng2 = 0.0144
Temp = [ 0.0003, 0.0005, 0.001, 0.0014, 0.0018, 0.0022, 0.0024, 0.0025, 0.0026, 0.00264, 0.00269]

delta_0 = np.full((m,), 1)


# calculate
TT = []
del_max = []

for T in Temp:
    omega = np.asarray([(2 * (-n + i) + 1) * np.pi * T for i in range(m)])
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

    # initialize the fns by calculating first two iterations
    R = Rn(delta_0, omega)
    zz = Z(R, omega, lamda, T)
    delta = gap(zz, delta_0, R, omega, lamda, T, mu)

    R = Rn(delta, omega)
    zz_new = Z(R, omega, lamda, T)
    delta_new = gap(zz, delta, R, omega, lamda, T, mu)

    # converge for self cosistency
    zz, delta = selfconsistency(
        convergence, zz_new, zz, delta_new, delta, omega, lamda, T, mu
    )

    # make a cvs file to save the omega delta and Z for every T
    with open("data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(omega, delta, zz))

    del_max.append(np.amax(delta))
    TT.append(T)

    print(del_max)
    print(TT)
    plt.plot(omega, delta, label=T)

    plt.ylabel("Gap")
    plt.xlabel("Omega")
    plt.legend()
    plt.grid()
    plt.title("Gap vs Omega")
    plt.savefig("OmgvsDel.png")

plt.clf()
plt.plot(TT, del_max, "bo-")
plt.ylabel("Gap")
plt.xlabel("Temperature")
plt.grid()
plt.title("Gap vs Temperature")
plt.savefig("Delta_vs_Tc.png")
