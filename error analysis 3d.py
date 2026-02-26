import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#parameters
G = 6.6743E-11
m2 = 1000
m1 = 5.972E+24

dt = 1
T = 345600
N = int(T / dt)

time_grid = np.linspace(0, T, N)


# Initial positions (x, y,z)
r2_init = np.array([-0.326705376530E+06 , -0.150855508550E+06 , -0.102368180470E+06] , dtype=np.float64) 
r1_init = np.array([ 0.0, 0.0, 0.0], dtype=np.float64)

# Initial velocities (vx, vy, vz)
v2_init = np.array([ -0.123303382137E-01 ,  0.151919024675E+00  , 0.200627345004E+00], dtype=np.float64)
v1_init = np.array([0.0,0.0,0.0], dtype=np.float64)


# ACCELERATION

def acceleration(r1, r2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    if dist == 0:
        return np.zeros(3), np.zeros(3)
    a1 = G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3
    return a1, a2


# TRAPEZOIDAL

def trapezoidal():
    r1 = r1_init.copy()
    r2 = r2_init.copy()
    v1 = v1_init.copy()
    v2 = v2_init.copy()

    

    norm = np.zeros(N)

    for i in range(N):
        norm[i] = np.linalg.norm(r2 - r1)

        a1, a2 = acceleration(r1, r2)

        r1_star = r1 + dt*v1
        r2_star = r2 + dt*v2
        v1_star = v1 + dt*a1
        v2_star = v2 + dt*a2

        a1_star, a2_star = acceleration(r1_star, r2_star)

        r1 += dt/2*(v1 + v1_star)
        r2 += dt/2*(v2 + v2_star)
        v1 += dt/2*(a1 + a1_star)
        v2 += dt/2*(a2 + a2_star)

    return norm

    

# RK4

def rk4():
    r1, r2 = r1_init.copy(), r2_init.copy()
    v1, v2 = v1_init.copy(), v2_init.copy()

    norm = np.zeros(N)

    for i in range(N):
        norm[i] = np.linalg.norm(r2 - r1)

        a1, a2 = acceleration(r1, r2)

        k1_r1, k1_r2 = v1*dt, v2*dt
        k1_v1, k1_v2 = a1*dt, a2*dt

        a1, a2 = acceleration(r1+0.5*k1_r1, r2+0.5*k1_r2)
        k2_r1 = (v1+0.5*k1_v1)*dt
        k2_r2 = (v2+0.5*k1_v2)*dt
        k2_v1, k2_v2 = a1*dt, a2*dt

        a1, a2 = acceleration(r1+0.5*k2_r1, r2+0.5*k2_r2)
        k3_r1 = (v1+0.5*k2_v1)*dt
        k3_r2 = (v2+0.5*k2_v2)*dt
        k3_v1, k3_v2 = a1*dt, a2*dt

        a1, a2 = acceleration(r1+k3_r1, r2+k3_r2)
        k4_r1 = (v1+k3_v1)*dt
        k4_r2 = (v2+k3_v2)*dt
        k4_v1, k4_v2 = a1*dt, a2*dt

        r1 += (k1_r1+2*k2_r1+2*k3_r1+k4_r1)/6
        r2 += (k1_r2+2*k2_r2+2*k3_r2+k4_r2)/6
        v1 += (k1_v1+2*k2_v1+2*k3_v1+k4_v1)/6
        v2 += (k1_v2+2*k2_v2+2*k3_v2+k4_v2)/6

    return norm



# PREDICTOR-CORRECTOR (ABM4)

def predictor_corrector():
    y0 = np.hstack((r1_init, v1_init, r2_init, v2_init))

    def f(y):
        r1 = y[0:3]; v1 = y[3:6]
        r2 = y[6:9]; v2 = y[9:12]
        r = r2-r1
        dist = np.linalg.norm(r)
        a1 = G*m2*r/dist**3
        a2 = -G*m1*r/dist**3
        return np.hstack((v1,a1,v2,a2))

    def rk4_step(y):
        k1=f(y)
        k2=f(y+0.5*dt*k1)
        k3=f(y+0.5*dt*k2)
        k4=f(y+dt*k3)
        return y+dt*(k1+2*k2+2*k3+k4)/6

    Y=np.zeros((N,12))
    F=np.zeros((N,12))
    Y[0]=y0; F[0]=f(y0)

    for i in range(3):
        Y[i+1]=rk4_step(Y[i])
        F[i+1]=f(Y[i+1])

    for n in range(3,N-1):
        Yp = Y[n] + dt/24*(55*F[n]-59*F[n-1]+37*F[n-2]-9*F[n-3])
        Fp = f(Yp)
        Y[n+1] = Y[n] + dt/24*(9*Fp+19*F[n]-5*F[n-1]+F[n-2])
        F[n+1] = f(Y[n+1])

    r1_arr=Y[:,0:3]
    r2_arr=Y[:,6:9]
    norm=np.linalg.norm(r2_arr-r1_arr,axis=1)

    return norm


# RK78 REFERENCE (DOP853)

def rk78():

    # Initial state vector
    y = np.hstack((r1_init, v1_init, r2_init, v2_init)).astype(np.float64)

    norm = np.zeros(N)

    def f(y):
        r1 = y[0:3]; v1 = y[3:6]
        r2 = y[6:9]; v2 = y[9:12]
        a1, a2 = acceleration(r1, r2)
        return np.hstack((v1, a1, v2, a2))

    # Dormand–Prince 8 coefficients (13-stage)
    c = np.array([0,2/27,1/9,1/6,5/12,1/2,5/6,1/6,2/3,1/3,1,0,1])

    a = np.zeros((13,13))
    a[1,0]=2/27
    a[2,0:2]=[1/36,1/12]
    a[3,0:3]=[1/24,0,1/8]
    a[4,0:4]=[5/12,0,-25/16,25/16]
    a[5,0:5]=[1/20,0,0,1/4,1/5]
    a[6,0:6]=[-25/108,0,0,125/108,-65/27,125/54]
    a[7,0:7]=[31/300,0,0,0,61/225,-2/9,13/900]
    a[8,0:8]=[2,0,0,-53/6,704/45,-107/9,67/90,3]
    a[9,0:9]=[-91/108,0,0,23/108,-976/135,311/54,-19/60,17/6,-1/12]
    a[10,0:10]=[2383/4100,0,0,-341/164,4496/1025,-301/82,
                2133/4100,45/82,45/164,18/41]
    a[11,0:11]=[3/205,0,0,0,0,-6/41,-3/205,-3/41,3/41,6/41,0]
    a[12,0:12]=[-1777/4100,0,0,-341/164,4496/1025,-289/82,
                2193/4100,51/82,33/164,12/41,0,1]

    b = np.array([41/840,0,0,0,0,34/105,9/35,9/35,
                  9/280,9/280,41/840,0,0])

    # Integration loop
    for i in range(N):

        norm[i] = np.linalg.norm(y[6:9] - y[0:3])

        k = np.zeros((13,12))

        k[0] = f(y)

        for j in range(1,13):
            y_stage = y + dt * np.sum(a[j,:j,None] * k[:j], axis=0)
            k[j] = f(y_stage)

        y = y + dt * np.sum(b[:,None] * k, axis=0)

    return norm



# RUN METHODS

n_trap = trapezoidal()
n_rk4  = rk4()
n_pc   = predictor_corrector()
n_ref  = rk78()

print(n_trap.shape)
print(n_ref.shape)

# ERROR ANALYSIS
err_trap = np.abs(n_trap - n_ref)
err_rk4  = np.abs(n_rk4  - n_ref)
err_pc   = np.abs(n_pc   - n_ref)

def print_errors(name, err):
    print(f"\n{name}")
    print("Max Error:", np.max(err))
    print("RMS Error:", np.sqrt(np.sum(err**2)*dt))
    print("Integrated Error:", np.trapezoid(err,time_grid))

print_errors("Trapezoidal", err_trap)
print_errors("RK4", err_rk4)
print_errors("Predictor-Corrector", err_pc)


# PLOTS

plt.figure()
plt.plot(time_grid,n_ref,label="RK78 (Reference)")
plt.plot(time_grid,n_trap,'--',label="Trapezoidal" , linewidth=2)
plt.plot(time_grid,n_rk4,'--',label="RK4" , linewidth=3)
plt.plot(time_grid,n_pc,'--',label="PC" , linewidth=2)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Separation")
plt.title("Norm Comparison")
plt.grid()
plt.show()

plt.figure()
plt.plot(time_grid, err_trap, label="Trap", linewidth=2)
plt.plot(time_grid, err_rk4, label="RK4", linewidth=3)
plt.plot(time_grid, err_pc, label="PC", linewidth=2)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Error vs Time")
plt.grid()
plt.show()


plt.figure()
plt.semilogy(time_grid,err_trap,label="Trap" , linewidth=2)
plt.semilogy(time_grid,err_rk4,label="RK4" , linewidth=3)
plt.semilogy(time_grid,err_pc,label="PC" , linewidth=2)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error (log scale)")
plt.title("Log Error Growth")
plt.grid()
plt.show()
