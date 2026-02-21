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

# Define the state vector y = [r1x, r1y, r2x, r2y, v1x, v1y, v2x, v2y]
y = np.array([r1[0], r1[1], r2[0], r2[1], v1[0], v1[1], v2[0], v2[1]])

# ODE function
def f(t, y):
    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]
    a1, a2 = acceleration(r1, r2)
    return np.array([v1[0], v1[1], v2[0], v2[1], a1[0], a1[1], a2[0], a2[1]])

# Dormand-Prince 8(7) method step function
def dormand_prince_step(f, t, y, h):
    # Coefficients from Butcher tableau
    # c vector
    c = np.array([
    0,2/27,1/9,1/6,5/12,1/2,5/6,1/6,2/3,1/3,1,0,1])

    # A matrix 
    a = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2/27,0,0,0,0,0,0,0,0,0,0,0,0],
    [1/36,1/12,0,0,0,0,0,0,0,0,0,0,0],
    [1/24,0,1/8,0,0,0,0,0,0,0,0,0,0],
    [5/12,0,-25/16,25/16,0,0,0,0,0,0,0,0,0],
    [1/20,0,0,1/4,1/5,0,0,0,0,0,0,0,0],
    [-25/108,0,0,125/108,-65/27,125/54,0,0,0,0,0,0,0],
    [31/300,0,0,0,61/225,-2/9,13/900,0,0,0,0,0,0],
    [2,0,0,-53/6,704/45,-107/9,67/90,3,0,0,0,0,0],
    [-91/108,0,0,23/108,-976/135,311/54,-19/60,17/6,-1/12,0,0,0,0],
    [2383/4100,0,0,-341/164,4496/1025,-301/82,2133/4100,45/82,45/164,18/41,0,0,0],
    [3/205,0,0,0,0,-6/41,-3/205,-3/41,3/41,6/41,0,0,0],
    [-1777/4100,0,0,-341/164,4496/1025,-289/82,2193/4100,51/82,33/164,12/41,0,1,0]
    ])

    # 8th order weights
    b = np.array([
    41/840,0,0,0,0,34/105,9/35,9/35,9/280,9/280,41/840,0,0
    ])

   
    k = np.zeros((13, len(y)))
    k[0] = f(t, y)
    for i in range(1,   13):
        sum_a = np.dot(a[i, :i], k[:i])
        k[i] = f(t + c[i] * h, y + h * sum_a)
    
    y_new = y + h * np.dot(b, k)
    return t + h, y_new

# Storage arrays
r1_arr   = np.zeros((N, 2))
r2_arr   = np.zeros((N, 2))
norm_arr = np.zeros(N)
time     = np.zeros(N)

# RK78 INTEGRATION 
for i in range(N):
    # Store values
    r1_arr[i] = y[0:2]
    r2_arr[i] = y[2:4]
    norm_arr[i] = np.linalg.norm(y[2:4] - y[0:2])
    time[i] = i * dt

    # Perform one RK78 step
    t_current = i * dt
    t_new, y_new = dormand_prince_step(f, t_current, y, dt)
    y = y_new

# PLOTS 

# 2D Plot
plt.figure(figsize=(7,7))
plt.plot(r1_arr[:,0], r1_arr[:,1], label="Body 1")
plt.plot(r2_arr[:,0], r2_arr[:,1], label="Body 2")
plt.scatter(0, 0, color='black', label="Center of Mass")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Body Problem in 2D (RK78)")
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
