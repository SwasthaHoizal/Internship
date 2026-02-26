import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#parameters
G = 6.6743E-11
m2 = 1000
m1 = 5.972E+24

dt = 0.1
T = 345600
N = int(T / dt)

# Initial positions (x, y,z)
r2 = np.array([-0.326705376530E+06 , -0.150855508550E+06 , -0.102368180470E+06  ])
r1 = np.array([ 0.0, 0.0, 0.0])

# Initial velocities (vx, vy, vz)
v2 = np.array([ -0.123303382137E-01 ,  0.151919024675E+00  , 0.200627345004E+00])
v1 = np.array([0.0,0.0,0.0])


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

# RK4 INTEGRATION 

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

    #k3 
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
    
    #Update velocities 
    v1 += (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1) / 6
    v2 += (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2) / 6

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
ax.set_title("Two Body Problem in 3D (RK4)")
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

