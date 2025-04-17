# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from heli_specs import *


# === Forward Speed Range ===
V = np.linspace(0.5, Vmax, 200)  
D_fus = CdS * 0.5 * rho * V**2 


## ====== QUESTION 1.2 Induced velocity calculation ======

def compute_induced_velocity(W, V, R, D_fus):
    """
    Computes the induced velocity and thrust as functions of forward velocity V.
    """
    a_d = np.arctan(D_fus / W)                     # [rad], disc angle of attack
    T = W / np.cos(a_d)                            # [N], required thrust for forward level flight
    vih = np.sqrt(T / (2 * rho * np.pi * R**2))    # [m/s], hover induced velocity at this thrust

    Vbar = V / vih # Normalize V
    Vbarc = Vbar * np.cos(a_d) # cosine component of Vbar
    Vbars = Vbar * np.sin(a_d) # sine component of Vbar

    # Estimate solution (Glauert)
    vibar = np.array([np.sqrt((-v**2 + np.sqrt(v**4 + 4)) / 2) if v < 2 else 1/v for v in Vbar])

    # Numerical solution
    vibar_ex = np.array([
        fsolve(lambda x: 1 / ((Vbarc[i])**2 + (Vbars[i] + x)**2) - x**2, vibar[i])[0]
        for i in range(len(Vbar))
    ])
    return vibar_ex, vih, T , a_d


vibar, vih, T, a_d = compute_induced_velocity(W, V, R, D_fus) # Thrust as an array of V, i.e. T=f(V)
vi = vibar * vih # Induced velocity as an array of V, i.e. vi=f(V)

# --- Plot: Induced Velocity ---
plt.figure(figsize=(10, 5))
plt.plot(V, vi)
plt.xlabel("Forward Velocity $\\bar{V}$ [-]")
plt.ylabel("Induced Velocity $\\bar{v}_i$ [-]")
plt.grid(True)
plt.title("Induced Velocity in Forward Flight")
plt.tight_layout()
plt.show()

## ====== QUESTION 1.3 Performance Calculations ======
## Part I) Hover Calculations
# === ACT Calculations ===
v_ihov = np.sqrt(W/(2*np.pi*rho*R**2)) # [m/s], induced velocity at sea-level & MTOW (ACT)
P_ideal = W*v_ihov #[W], ideal power at sea-level & MTOW (ACT)
P_hov_act = P_ideal / FM #[W], hover power (ACT)

# === BEM Calculations === 
P_induced = k * W * v_ihov  # [W], induced power (BEM)
P_profile = (solidity * CDp / 8) * rho * (Omega * R)**3 * np.pi * R**2  # [W], profile power
P_hov_bem = P_induced + P_profile  # [W], total hover power (BEM)

print("=== Hover Power Calculations ===")
print(f"Solidity: {solidity:.4f}")
print(f"Ideal Power (ACT): {P_ideal/1000:.1f} kW")
print(f"Hover Power (ACT, with FM): {P_hov_act/1000:.1f} kW")
print(f"Induced Power (BEM): {P_induced/1000:.1f} kW")
print(f"Profile Power (BEM): {P_profile/1000:.1f} kW")
print(f"Total Hover Power (BEM): {P_hov_bem/1000:.1f} kW")

## PART II) Forward Flight Calculations
# --- Power Calculations (BEM) ---
mu = V *np.cos(a_d)/ (Omega * R)
P_induced = k * vi * T 
P_profile = (1 + mu**2) * (solidity * CDp / 8) * rho * (Omega * R)**3 *np.pi * R**2
H_0 = (1/4) * solidity * rho * CDp * np.pi * R**2 * mu**2
P_rotor_drag = H_0 * V * np.cos(a_d)
P_parasite = 0.5 * rho * V**3 * A_eq
P_tail = 1.1 * k_tr * T_tr * v_tr * np.ones_like(V)

P_almost_total = P_induced + P_profile + P_rotor_drag + P_parasite + P_tail
P_misc = 0.06 * P_almost_total

P_total = P_almost_total + P_misc

# --- Best speed for endurance and range ---
idx_endurance = np.argmin(P_total)
V_endurance = V[idx_endurance]
P_min = P_total[idx_endurance]

idx_range = np.argmin(P_total / V)
V_range = V[idx_range]
P_range = P_total[idx_range]

# --- Plot: Power Breakdown ---
plt.figure(figsize=(10, 6))
plt.plot(V, P_total / 1000, label='Total Power', linewidth=2)
plt.plot(V, P_induced / 1000, '--', label='Induced Power')
plt.plot(V, P_profile / 1000, '--', label='Profile Drag Power')
plt.plot(V, P_rotor_drag / 1000, '--', label='Rotor Drag Power')
plt.plot(V, P_parasite / 1000, '--', label='Parasite Power')
plt.plot(V, P_tail / 1000, '--', label='Tail Rotor Power')
plt.plot(V, P_misc / 1000, '--', label='Miscellaneous Power')

plt.axvline(V_endurance, color='r', linestyle=':', label=f'Best Endurance: {V_endurance:.1f} m/s')
plt.plot([0, V_range], [0, P_range / 1000], color='purple', linestyle='--', label=f'Best Range: {V_range:.1f} m/s')

plt.scatter([V_endurance], [P_min / 1000], color='red', zorder=5)
plt.scatter([V_range], [P_range / 1000], color='purple', zorder=5)

plt.xlabel('Forward Velocity $V$ [m/s]')
plt.ylabel('Power Required [kW]')
plt.title('Total Power in Forward Flight (BEM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Best endurance speed (min power): {V_endurance:.2f} m/s, Power: {P_min/1000:.2f} kW")
print(f"Best range speed (min P/V): {V_range:.2f} m/s, Power: {P_range/1000:.2f} kW")