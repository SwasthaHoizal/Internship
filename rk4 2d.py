import numpy as np
import matplotlib.pyplot as plt

# Parameters
G = 1.0
m1 = 1.0
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

# Acceleration function
def acceleration(r1, r2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    a1 = G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3
    return a1, a2

# Storage arrays
r1_arr   = np.zeros((N, 2))
r2_arr   = np.zeros((N, 2))
norm_arr = np.zeros(N)
time     = np.zeros(N)

#  RK4 INTEGRATION 
for i in range(N):

    # Store values
    r1_arr[i] = r1
    r2_arr[i] = r2
    norm_arr[i] = np.linalg.norm(r2 - r1)
    time[i] = i * dt

    # k1
    a1, a2 = acceleration(r1, r2)
    k1_r1 = v1 * dt
    k1_r2 = v2 * dt
    k1_v1 = a1 * dt
    k1_v2 = a2 * dt

    # k2 
    a1, a2 = acceleration(r1 + 0.5*k1_r1, r2 + 0.5*k1_r2)
    k2_r1 = (v1 + 0.5*k1_v1) * dt
    k2_r2 = (v2 + 0.5*k1_v2) * dt
    k2_v1 = a1 * dt
    k2_v2 = a2 * dt

    # k3
    a1, a2 = acceleration(r1 + 0.5*k2_r1, r2 + 0.5*k2_r2)
    k3_r1 = (v1 + 0.5*k2_v1) * dt
    k3_r2 = (v2 + 0.5*k2_v2) * dt
    k3_v1 = a1 * dt
    k3_v2 = a2 * dt

    # k4
    a1, a2 = acceleration(r1 + k3_r1, r2 + k3_r2)
    k4_r1 = (v1 + k3_v1) * dt
    k4_r2 = (v2 + k3_v2) * dt
    k4_v1 = a1 * dt
    k4_v2 = a2 * dt

    # Update positions
    r1 += (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1) / 6
    r2 += (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2) / 6

    # Update velocities
    v1 += (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1) / 6
    v2 += (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2) / 6

# PLOTS 

# 2D Plot
plt.figure(figsize=(7,7))
plt.plot(r1_arr[:,0], r1_arr[:,1], label="Body 1")
plt.plot(r2_arr[:,0], r2_arr[:,1], label="Body 2")
plt.scatter(0, 0, color='black', label="Center of Mass")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Body Problem in 2D (RK4)")
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
