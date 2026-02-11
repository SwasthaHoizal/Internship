import numpy as np
import matplotlib.pyplot as plt


# Parameters
G = 1.0
m1 =1.0
m2 = 1.0

dt = 0.001
T = 20
N = int(T / dt)

# Initial positions (x, y)
r1 = np.array([-0.5, 0.0])
r2 = np.array([ 0.5, 0.0])

# Initial velocities (vx, vy)
v1 = np.array([0.9, 0.5])
v2 = np.array([0.6,-0.5])

y0 = np.hstack((r1, v1, r2, v2))


# Equations of motion

def f(y):
    r1 = y[0:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]

    r = r2 - r1
    dist = np.linalg.norm(r)

    a1 = G * m2 * r/ dist**3
    a2 = -G * m1 * r / dist**3

    return np.hstack((v1, a1, v2, a2))


# RK4 starter

def rk4_step(y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)

    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

# Storage

Y = np.zeros((N + 1, 8))
F = np.zeros((N + 1, 8))

Y[0] = y0
F[0] = f(y0)


# Bootstrap using RK4

for i in range(3):
    Y[i + 1] = rk4_step(Y[i], dt)
    F[i + 1] = f(Y[i + 1])


# Adams–Bashforth–Moulton loop

for n in range(3, N):
    # Predictor (AB4)
    Yp = Y[n] + dt / 24 * (55 * F[n] - 59 * F[n - 1] + 37 * F[n - 2] - 9 * F[n - 3])

    Fp = f(Yp)

    # Corrector (AM4)
    Y[n + 1] = Y[n] + dt / 24 * (9 * Fp + 19 * F[n] - 5 * F[n - 1] + F[n - 2])

    F[n + 1] = f(Y[n + 1])


# Extract positions
r1_arr = Y[:, 0:2]
r2_arr = Y[:, 4:6]

# Separation norm
r12 = r2_arr - r1_arr
norm_arr = np.linalg.norm(r12, axis=1)
time = np.linspace(0, T, N + 1)


# PLOTS 

# 2D Plot
plt.figure(figsize=(7,7))
plt.plot(r1_arr[:,0], r1_arr[:,1], label="Body 1")
plt.plot(r2_arr[:,0], r2_arr[:,1], label="Body 2")
plt.scatter(0, 0, color='black', label="Center of Mass")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Body Problem in 2D (PC)")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()

# NORM
plt.figure(figsize=(6,4))
plt.plot(time, norm_arr)
plt.xlabel("Time")
plt.ylabel("||r2 − r1||")
plt.title("Separation vs Time")
plt.grid()
plt.show()
