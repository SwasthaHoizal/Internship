import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# PHYSICAL CONSTANTS
# ==========================================================

G  = 6.6743e-11
m1 = 5.972e24          # Earth mass (kg)
m2 = 1000              # Satellite mass (kg)

mu = G * m1

# Earth parameters (J2)
R_E = 6378137.0
J2  = 1.08262668e-3

# Solar Radiation Pressure
P_sr     = 4.56e-6
C_r      = 1.2
A_over_m = 0.01

sun_dir = np.array([1.0, 0.0, 0.0])   # Fixed Sun direction

# ==========================================================
# TIME SETTINGS
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
# State vector: [r1 v1 r2 v2]
y0 = np.hstack((r1, v1, r2, v2))       # 12 elements

# ==========================================================
# PERTURBATION MODELS
# ==========================================================

def J2_acceleration(r):

    x, y, z = r
    r_norm = np.linalg.norm(r)

    factor = (3/2) * J2 * mu * R_E**2 / r_norm**5
    zx = z / r_norm

    ax = factor * x * (5*zx**2 - 1)
    ay = factor * y * (5*zx**2 - 1)
    az = factor * z * (5*zx**2 - 3)

    return np.array([ax, ay, az])


def SRP_acceleration():
    return P_sr * C_r * A_over_m * sun_dir


# ==========================================================
# ACCELERATION MODEL
# ==========================================================

def acceleration(r1, r2):

    r = r2 - r1
    dist = np.linalg.norm(r)

    # Two-body gravity
    a1 =  G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3

    # J2 acts on satellite relative to Earth center
    a2 += J2_acceleration(r2)

    # SRP acts on satellite
    a2 += SRP_acceleration()

    return a1, a2


# ==========================================================
# STATE DERIVATIVE FUNCTION
# ==========================================================

def f(y):

    r1 = y[0:3]
    v1 = y[3:6]
    r2 = y[6:9]
    v2 = y[9:12]

    a1, a2 = acceleration(r1, r2)

    dydt = np.zeros(12)

    dydt[0:3]  = v1
    dydt[3:6]  = a1
    dydt[6:9]  = v2
    dydt[9:12] = a2

    return dydt


# ==========================================================
# RK4 STARTER
# ==========================================================

def rk4_step(y, h):

    k1 = f(y)
    k2 = f(y + 0.5*h*k1)
    k3 = f(y + 0.5*h*k2)
    k4 = f(y + h*k3)

    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6


# ==========================================================
# STORAGE ARRAYS
# ==========================================================

Y = np.zeros((N+1, 12))
F = np.zeros((N+1, 12))

Y[0] = y0
F[0] = f(y0)

# ==========================================================
# BOOTSTRAP FIRST 3 STEPS USING RK4
# ==========================================================

for i in range(3):
    Y[i+1] = rk4_step(Y[i], dt)
    F[i+1] = f(Y[i+1])

# ==========================================================
# ADAMS–BASHFORTH–MOULTON (ABM4)
# ==========================================================

for n in range(3, N):

    # Predictor (Adams–Bashforth 4)
    Yp = Y[n] + dt/24 * (55*F[n] - 59*F[n-1] + 37*F[n-2] - 9*F[n-3])

    Fp = f(Yp)

    # Corrector (Adams–Moulton 4)
    Y[n+1] = Y[n] + dt/24 * (9*Fp + 19*F[n] - 5*F[n-1] + F[n-2])

    F[n+1] = f(Y[n+1])

# ==========================================================
# EXTRACT POSITIONS
# ==========================================================

r1_arr = Y[:, 0:3]
r2_arr = Y[:, 6:9]

r12 = r2_arr - r1_arr
norm_arr = np.linalg.norm(r12, axis=1)
time = np.linspace(0, T, N+1)

# ==========================================================
# 3D TRAJECTORY PLOT
# ==========================================================

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r2_arr[:,0], r2_arr[:,1], r2_arr[:,2], label="Satellite")
ax.scatter(0,0,0,color='black',label="Earth")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Two Body + J2 + SRP (ABM4)")
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
