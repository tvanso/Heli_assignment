import numpy as np
import matplotlib.pyplot as plt
from heli_specs import *
from scipy.optimize import root_scalar

v_tip = Omega * R              # Tip speed

# Forward velocities
V = np.linspace(0.1, Vmax, 100)  # m/s

# Parasite drag and thrust
D = 0.5 * rho * CdS * V**2
T = np.sqrt(W**2 + D**2)
C_T = T / (rho * np.pi * R**2 * v_tip**2)

# Fuselage pitch angle (for comparison)
theta_f = np.arctan(D / W)

# Induced velocity lambda_i
lambda_i = []

for i in range(len(V)):
    def inflow_eqn(lmbd):
        mu_i = V[i] / (Omega * R)
        cos_term = np.cos(D[i] / W)
        sin_term = np.sin(D[i] / W)
        return 2 * lmbd * np.sqrt((mu_i * cos_term)**2 + (mu_i * sin_term + lmbd)**2) - C_T[i]

    try:
        sol = root_scalar(inflow_eqn, method='brentq', bracket=[0.0001, 0.5])
        lambda_i.append(sol.root)
    except ValueError:
        lambda_i.append(np.nan)

# Advance ratio
mu = V / (R * Omega)

# Trim solutions
a1 = []
theta_0 = []

for i in range(len(V)):
    if np.isnan(lambda_i[i]):
        a1.append(np.nan)
        theta_0.append(np.nan)
        continue

    A = np.array([
        [1 + 1.5 * mu[i]**2, -(8/3) * mu[i]],
        [-mu[i], (2/3) + mu[i]**2]
    ])
    b = np.array([
        -2 * mu[i]**2 * D[i]/W - 2 * mu[i] * lambda_i[i],
        (4 * C_T[i]) / (solidity * cl_alpha) + mu[i] * D[i]/W + lambda_i[i]
    ])
    x = np.linalg.solve(A, b)
    a1.append(x[0])         # cyclic pitch in degrees
    theta_0.append(x[1])    # collective pitch in degrees

vi = np.array(lambda_i) * Omega * R

# Save data to file
np.savez("trim_data2.npz",
         V=V.tolist(),
         theta_c=a1,
         theta_o=theta_0,
         theta_f=theta_f.tolist(),
         vi=vi.tolist())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(V, np.degrees(a1), label='Cyclic Pitch $\\theta_c = a_1$ [deg]')
plt.plot(V, np.degrees(theta_0), label='Collective Pitch $\\theta_0$ [deg]')
plt.xlabel('Forward Velocity $V$ (m/s)')
plt.ylabel('Pitch Angle (deg)')
plt.title('Trimmed Rotor Pitch Angles vs Forward Speed')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
