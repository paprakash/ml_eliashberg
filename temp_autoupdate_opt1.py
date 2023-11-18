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
def first_itr(T, delta_0, omega_0, ng2):
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

    zz_new, delta_new = self_consistent_solution(
    Rn, Z, gap, delta_0, omega, lamda, T, mu, convergence
    )

    TT.append(T)
    del_max.append(np.amax(delta_new))
    zz_max.append(np.amax(zz_new))

    return  zz_new, delta_new, TT, del_max, zz_max

def del_t(T, delta_old):

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
    
    # print the maximum element of lamda matrix
    print("lamda_max=", np.amax(lamda))
    # converge for self consistency
    zz_new, delta_new = self_consistent_solution(
                    Rn, Z, gap, delta_old, omega, lamda, T, mu, convergence
    )
    
    return zz_new, delta_new

def cal2(delta_previous, T, T_del):

    delta_old = delta_previous.copy()
    zz_new, delta_new = del_t(T+T_del, delta_old)

    while np.amax(np.abs(delta_old))-np.amax(np.abs(delta_new)) > 0.15*np.amax(np.abs(delta_init)):
        T_del = T_del*0.5
        zz_new, delta_new = del_t(T+T_del, delta_old)
        
    TT.append(T+T_del)
    del_max.append(np.amax(delta_new))
    zz_max.append(np.amax(zz_new))

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


# parameters
mu = 0
print("mu=", mu)
n = 1592
print("n=", n)
m = 2 * n
print("m=", m)
convergence = 1e-4
print("convergence=", convergence)


omega_0 = 0.05
print("omega_0=", omega_0)
ng2 = 0.014
print("ng2=", ng2)
delta_0 = np.full((m,), 0.005)

# calculate
TT = []
del_max = []
zz_max = []

T = 0.0005
print("initial T=", T)
T_del = 0.0005
print("initial T_del=", T_del)

zz_init, delta_init, TT, del_max, zz_max = first_itr(T, delta_0, omega_0, ng2)

delta_previous = delta_init.copy()
print("delta_init=", np.amax(delta_init))
zz_previous = zz_init.copy()
print("zz_init=", np.amax(zz_init))


delta_previous, T, T_del, TT, del_max, zz_max = rec(delta_previous, T, T_del)

print("TT=", TT)
print("del_max=", del_max)
print("zz_max=", zz_max)
"""
# make a plot of the data and save it
plt.plot(TT, del_max, "-o")
plt.xlabel("T")
plt.ylabel("delta_max")
plt.savefig("delta_max_vs_T.png")

plt.plot(TT, zz_max, "-o", color="black")
plt.xlabel("T")
plt.ylabel("zz_max")
plt.savefig("zz_max_vs_T.png")

# save the data in a csv file
data = np.asarray([TT, del_max, zz_max])
data = data.T
np.savetxt("data.csv", data, delimiter=",")
"""