import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#parameters
G = 1.0
m1 = 1.0
m2 = 1.0

dt = 0.001
T = 20
N = int(T / dt)

# Initial positions (x, y,z)
r1 = np.array([-0.5, 0.0, 0.7])
r2 = np.array([ 0.5, 0.0, 1.0])

# Initial velocities (vx, vy, vz)
v1 = np.array([0.9, 0.5, 0.2])
v2 = np.array([0.6,-0.5,-0.2])

# Acceleration function
def acceleration(r1, r2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    a1 = G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3
    return a1, a2


# Storage arrays
r1_arr   = np.zeros((N, 3))
r2_arr   = np.zeros((N, 3))
norm_arr = np.zeros(N)
time     = np.zeros(N)

#Trapezoidal Integration

for i in range(N):

    # Store values
    r1_arr[i] = r1
    r2_arr[i] = r2
    norm_arr[i] = np.linalg.norm(r2 - r1)
    time[i] = i * dt

    a1, a2 = acceleration(r1, r2)

    # Predictor step (Euler)
    r1_star = r1 + dt * v1
    r2_star = r2 + dt * v2
    v1_star = v1 + dt * a1
    v2_star = v2 + dt * a2

    # Accelerations at predicted step
    a1_star, a2_star = acceleration(r1_star, r2_star)

    # Corrector step (Trapezoidal)
    r1 += (dt / 2) * (v1 + v1_star)
    r2 += (dt / 2) * (v2 + v2_star)
    v1 += (dt / 2) * (a1 + a1_star)
    v2 += (dt / 2) * (a2 + a2_star)

# PLOTS 
# 3D PLOT
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r1_arr[:,0], r1_arr[:,1], r1_arr[:,2], label="Body 1")
ax.plot(r2_arr[:,0], r2_arr[:,1], r2_arr[:,2], label="Body 2")
ax.scatter(0, 0, 0, color='black', label="Center of Mass")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Two Body Problem in 3D (Trapezoidal)")
ax.legend()
plt.show()

#NORM 
plt.figure(figsize=(6,4))
plt.plot(time, norm_arr)
plt.xlabel("Time")
plt.ylabel("||r2 − r1||")
plt.title("Separation vs Time")
plt.grid()
plt.show()


