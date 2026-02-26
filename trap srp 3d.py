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

# ==========================================================
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

# ==========================================================
# TRAPEZOIDAL INTEGRATION
# ==========================================================

for i in range(N):

    r1_arr[i]   = r1
    r2_arr[i]   = r2
    norm_arr[i] = np.linalg.norm(r2 - r1)
    time[i]     = i * dt

    a1, a2 = acceleration(r1, r2)

    # Predictor (Euler)
    r1_star = r1 + dt * v1
    r2_star = r2 + dt * v2
    v1_star = v1 + dt * a1
    v2_star = v2 + dt * a2

    # Acceleration at predicted state
    a1_star, a2_star = acceleration(r1_star, r2_star)

    # Corrector (Trapezoidal)
    r1 += (dt/2) * (v1 + v1_star)
    r2 += (dt/2) * (v2 + v2_star)
    v1 += (dt/2) * (a1 + a1_star)
    v2 += (dt/2) * (a2 + a2_star)


# ==========================================================
# 3D TRAJECTORY PLOT
# ==========================================================

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r2_arr[:,0], r2_arr[:,1], r2_arr[:,2], label="Satellite")
ax.scatter(0, 0, 0, color='black', label="Earth")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Two Body + J2 + SRP (Trapezoidal)")
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
