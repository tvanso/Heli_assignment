import numpy as np

# === Physical Constants ===
g = 9.80665              # Gravity [m/s^2]
rho = 1.225              # Air density [kg/m^3]

# === Helicopter Parameters: Airbus H135 ===
MTOW = 2910             # Maximum take-off weight [kg]
W = MTOW * g            # Weight [N]

R = 5.2                 # Rotor radius [m]
c = 0.288               # Blade chord [m]
N = 4                   # Number of blades
Omega = 395 * 2 * np.pi / 60  # Rotor angular speed [rad/s]

FM = 0.6                # Figure of Merit [-]
h = 1.5
CDp = 0.012             # Profile drag coefficient [-]
A_eq = 2.0              # Equivalent flat plate area [m²]
cl_alpha = 5.7          # Lift curve slope [1/rad]
I_y = 5370              # Moment of inertia [kg*m²]
k = 1.15                # Induced power correction factor [-]
k_tr = 1.4              # Tail rotor correction factor [-]
T_tr_frac = 0.1         # Tail rotor thrust fraction of main rotor thrust [-]
gamma = 9              # Lock number [-]

CdS = 21.5 * 0.092903   # flat plate area from ft² to m² [m²]
Vmax = 256 / 3.6        # Max forward speed converted from km/h to m/s [m/s]
# === Derived Parameters ===
solidity = N * c / (np.pi * R)        # Blade solidity [-]
A_disk = np.pi * R**2                 # Rotor disk area [m²]

T_tr = T_tr_frac * W                  # Tail rotor thrust [N]
v_tr = np.sqrt(T_tr / (2 * rho * A_disk))  # Tail rotor induced velocity [m/s]

v_ihov = np.sqrt(W/(2*np.pi*rho*R**2)) # ACT disk theory

