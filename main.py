import numpy as np
import matplotlib.pyplot as plt
import math
from models import *


def F(t, U):
    """Right hand side function"""

    f = np.zeros(U.shape)
    f[0] = U[1]
    f[1] = (-1) * omega ** 2 * U[0]

    return f

C = 1
L = 1
omega = 1 / math.sqrt(L * C)

t0, tN = 0, 100
u0 = 1
i0 = 0

h = 0.1
T = np.arange(t0, tN, h)
U0 = [u0, i0 / (omega*C)]

#
# Demo U(t), I(t), I(U), E(t) with runge_kutta_4, adams_3
#

U = runge_kutta_4(F, T, U0)
U_0 = U[:, 0]
U_1 = U[:, 1]
I = U_1 * omega * C
U_exact = u0 * np.cos(omega * T)
E = U_0 - U_exact

# U(t)
fig = plt.figure(figsize=(18, 6))
ax = fig.subplots()
plt.title("Capacitor voltage")
ax.set_xlabel("Time $t$, s")
ax.set_ylabel("Voltage $U(t)$, V")
plt.xlim([t0, tN])
ax.plot(T, U_0, 'ro', label='Numeric RK4')
ax.plot(T, U_exact, label='Exact')
ax.grid()
ax.legend(loc="upper right", framealpha=0.95, fontsize='xx-large')
plt.savefig("pics/U.png")

# U(t), I(t)
fig = plt.figure(figsize=(18, 6))
ax = fig.subplots()
axr = ax.twinx()
plt.title("Capacitor voltage and current")
ax.set_xlabel("Time $t$, s")
ax.set_ylabel("Voltage $U(t)$, V")
axr.set_ylabel("Current $I(t)$, A")
plt.xlim([t0, tN])
l1 = ax.plot(T, U_0, label='$U$, RK4')
l2 = ax.plot(T, I, label='$I$, RK4')
ax.grid()
axr.grid()
ax.legend(loc="upper right", framealpha=0.95, fontsize='xx-large')
plt.savefig("pics/I.png")

# I(U)
fig = plt.figure(figsize=(6, 6))
ax = fig.subplots()
plt.title("Current–voltage characteristic")
ax.set_xlabel("Voltage $U(t)$, V")
ax.set_ylabel("Current $I(t)$, A")
ax.plot(U_0, I, label='Numeric RK4')
ax.grid()
ax.legend(loc="upper right", framealpha=0.95, fontsize='xx-large')
plt.savefig("pics/IU.png")

# errors RK 4
fig = plt.figure(figsize=(18, 6))
ax = fig.subplots()
plt.title("Errors of Runge-Kutta")
ax.set_xlabel("Time $t$, s")
ax.set_ylabel("Voltage error $\\Delta U$, V")
plt.xlim([t0, tN])
ax.plot(T, E, 'ro', label='Numeric RK4')
ax.grid()
ax.legend(loc="upper left", framealpha=0.95, fontsize='x-large')
plt.savefig("pics/E_RK.png")

# errors Adams
U = adams_bashforth_3(F, T, U0)
U_0 = U[:, 0]
E = U_0 - U_exact

fig = plt.figure(figsize=(18, 6))
ax = fig.subplots()
plt.title("Errors of Adams-Bashforth")
ax.set_xlabel("Time $t$, s")
ax.set_ylabel("Voltage error $\\Delta U$, V")
plt.xlim([t0, tN])
ax.plot(T, E, 'ro', label='Numeric Adams')
# ax.plot(T, U_exact, label='Exact')
ax.grid()
ax.legend(loc="upper left", framealpha=0.95, fontsize='x-large')
plt.savefig("pics/E_Adams.png")

#plt.show()

