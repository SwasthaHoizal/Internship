import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# PHYSICAL CONSTANTS
# ==========================================================

G  = 6.6743e-11
m1 = 5.972e24              # Earth mass (kg)
m2 = 1000                  # Satellite mass (kg)

mu = G * m1

# Earth parameters (J2 model)
R_E = 6378137.0            # Earth radius (m)
J2  = 1.08262668e-3

# Solar Radiation Pressure parameters
P_sr      = 4.56e-6        # Solar radiation pressure at 1 AU (N/m^2)
C_r       = 1.2            # Reflectivity coefficient
A_over_m  = 0.01           # Area-to-mass ratio (m^2/kg)

# Assume Sun direction fixed in +X direction
sun_dir = np.array([1.0, 0.0, 0.0])

# ==========================================================
# TIME PARAMETERS
# ==========================================================

dt = 1
T  = 345600                # 4 days
N  = int(T / dt)

# Initial positions (x, y,z)
r2 = np.array([-0.326705376530E+06 , -0.150855508550E+06 , -0.102368180470E+06  ])
r1 = np.array([ 0.0, 0.0, 0.0])

# Initial velocities (vx, vy, vz)
v2 = np.array([ -0.123303382137E-01 ,  0.151919024675E+00  , 0.200627345004E+00])
v1 = np.array([0.0,0.0,0.0])

# ==========================================================
# PERTURBATION MODELS
# ==========================================================

def J2_acceleration(r):

    x, y, z = r
    r_norm = np.linalg.norm(r)

    factor = (3/2) * J2 * mu * R_E**2 / r_norm**5

    zx = z / r_norm
    term = 5 * zx**2 - 1

    ax = factor * x * term
    ay = factor * y * term
    az = factor * z * (5 * zx**2 - 3)

    return np.array([ax, ay, az])


def SRP_acceleration():

    # Constant SRP direction (simplified model)
    return P_sr * C_r * A_over_m * sun_dir


# ==========================================================
# ACCELERATION FUNCTION (GRAVITY + J2 + SRP)
# ==========================================================

def acceleration(r1, r2):

    r = r2 - r1
    dist = np.linalg.norm(r)

    # Two-body gravity
    a1 = G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3

    # J2 perturbation (only on satellite)
    a2 += J2_acceleration(r)

    # SRP perturbation (only on satellite)
    a2 += SRP_acceleration()

    return a1, a2


# ==========================================================
# STORAGE ARRAYS
# ==========================================================

r1_arr   = np.zeros((N, 3))
r2_arr   = np.zeros((N, 3))
norm_arr = np.zeros(N)
time     = np.zeros(N)

# Define the state vector y = [r1x, r1y, r1z, r2x, r2y, r2z, v1x, v1y, v1z, v2x, v2y, v2z]
y = np.array([r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]])

# ODE function
def f(t, y):
    r1 = y[0:3]
    r2 = y[3:6]
    v1 = y[6:9]
    v2 = y[9:12]
    a1, a2 = acceleration(r1, r2)
    return np.array([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]])

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
    for i in range(1, 13):
        sum_a = np.dot(a[i, :i], k[:i])
        k[i] = f(t + c[i] * h, y + h * sum_a)
    
    y_new = y + h * np.dot(b, k)
    return t + h, y_new

# Storage arrays
r1_arr   = np.zeros((N, 3))
r2_arr   = np.zeros((N, 3))
norm_arr = np.zeros(N)
time     = np.zeros(N)

# RK78 INTEGRATION 
for i in range(N):
    # Store values
    r1_arr[i] = y[0:3]
    r2_arr[i] = y[3:6]
    norm_arr[i] = np.linalg.norm(y[3:6] - y[0:3])
    time[i] = i * dt

    # Perform one RK78 step
    t_current = i * dt
    t_new, y_new = dormand_prince_step(f, t_current, y, dt)
    y = y_new

# 3D TRAJECTORY PLOT
# ==========================================================

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r2_arr[:,0], r2_arr[:,1], r2_arr[:,2], label="Satellite")
ax.scatter(0, 0, 0, color='black', label="Earth")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Two Body + J2 + SRP (RK78)")
ax.legend()

plt.show()


# ==========================================================
# SEPARATION PLOT
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(time/3600, norm_arr/1000)
plt.xlabel("Time (hours)")
plt.ylabel("Radius (km)")
plt.title("Orbital Radius vs Time")
plt.grid()
plt.show()
