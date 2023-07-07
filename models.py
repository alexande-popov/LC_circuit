import numpy as np


def euler(f, T, U0):
    U = np.zeros((len(T), len(U0)))
    U[0] = U0

    h = T[1] - T[0]

    for i in range(1, len(T)):
        k1 = f(T[i - 1], U[i - 1])

        U[i] = U[i - 1] + h * k1

    # return U[:, 0]
    return U


def runge_kutta_4(f, T, U0):
    U = np.zeros((len(T), len(U0)))
    U[0] = U0

    h = T[1] - T[0]

    for i in range(1, len(T)):
        k1 = f(T[i - 1], U[i - 1])
        k2 = f(T[i - 1] + h / 2, U[i - 1] + h * k1 / 2)
        k3 = f(T[i - 1] + h / 2, U[i - 1] + h * k2 / 2)
        k4 = f(T[i - 1] + h, U[i - 1] + h * k3)

        U[i] = U[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return U


def adams_bashforth_3(f, T, U0):
    h = T[1] - T[0]
    T_prev = np.array([T[0] - i * h for i in range(2, 0, -1)])
    T = np.concatenate([T_prev, T])
    U = np.zeros((len(T), len(U0)))


    U_prev = runge_kutta_4(f, np.concatenate([[T[0]] , T_prev[::-1]]), U0)
    U[0] = U_prev[2]
    U[1] = U_prev[1]
    U[2] = U0

    for i in range(3, len(T)):
        U[i] = U[i - 1]
        # U[i] += (h / 12) * (
        #             23 * f(T[i-3] - h, U[i-1]) - 16 * f(T[i-3] - 2 * h, U[i-2]) + 5 * f(T[i-3] - 3 * h,
        #                                                                                         U[i-3]))
        U[i] += (h / 12) * (
                23 * f(T[i - 1] - h, U[i - 1]) - 16 * f(T[i - 2] - 2 * h, U[i - 2]) + 5 * f(T[i - 3] - 3 * h,
                                                                                            U[i - 3]))

    U2 = U[2::]

    return U2