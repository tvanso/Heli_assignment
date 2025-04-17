import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from heli_specs import *


# === Functions ===

def calc_D_f(v):
    return 0.5 * rho * v**2 * CdS

def calc_alpha_d(D_fus, W):
    return np.arctan(D_fus / W)

def calc_C_T(W, D_fus):
    T_req = np.sqrt(W**2 + D_fus**2)
    return T_req / (rho * (Omega * R)**2 * np.pi * R**2)

def calc_lambda_i(v, alpha_d, C_T):
    def equation(lambda_i):
        term1 = (v / (Omega * R)) * np.sin(alpha_d) + lambda_i
        term2 = (v / (Omega * R)) * np.cos(alpha_d)
        return C_T - 2 * lambda_i * np.sqrt(term1 ** 2 + term2 ** 2)

    lambda_solution = fsolve(equation, x0=0.2)
    return lambda_solution[0]

def solve_trim_angles(alpha_d, v, C_T, lambda_i):
    def equations(vars):
        theta_c, theta_o = vars
        mu = (v / (Omega * R)) * np.cos(alpha_d + theta_c)
        lambda_c = mu * (alpha_d + theta_c)
        f = C_T - ((cl_alpha * solidity) / 4) * ((2 / 3) * theta_o * (1 + (3 / 2) * mu ** 2) - (lambda_i + lambda_c))
        g = theta_c - (8 / 3 * mu * theta_o - 2 * mu * (lambda_c + lambda_i)) / (1 - 0.5 * mu**2)
        return [f, g]

    initial_guess = [0.0, 0.1]
    theta_c, theta_o = fsolve(equations, initial_guess)
    return theta_c, theta_o

# === Main Loop ===

V_range = np.linspace(0.1, Vmax, 100)
theta_c_list = []
theta_o_list = []
theta_f_list = []
vi_list = []

for V in V_range:
    D_fus = calc_D_f(V)
    alpha_d = calc_alpha_d(D_fus, W)
    C_T = calc_C_T(W, D_fus)
    lambda_i = calc_lambda_i(V, alpha_d, C_T)
    vi = lambda_i* Omega * R 
    theta_c, theta_o = solve_trim_angles(alpha_d, V, C_T, lambda_i)
    theta_f = np.arctan(-D_fus/W)
    theta_c_list.append(theta_c)
    theta_o_list.append(theta_o)
    theta_f_list.append(theta_f)
    vi_list.append(vi)

for V in V_range:
    mu = V/(Omega*R)

    

# === Plotting ===

# plt.figure(figsize=(10, 6))
# plt.plot(V_range, np.degrees(theta_c_list), label="Cyclic pitch $\\theta_c$ [deg]")
# plt.plot(V_range, np.degrees(theta_o_list), label="Collective pitch $\\theta_o$ [deg]")
# plt.plot(V_range, -np.degrees(theta_f_list), label="Negative Fuselage angle, i.e. - $\\theta_f$ [deg]")
# # plt.plot(V_range, theta_c_list, label="Cyclic pitch $\\theta_c$ [deg]")
# # plt.plot(V_range, theta_o_list, label="Collective pitch $\\theta_o$ [deg]")
# # plt.plot(V_range, theta_f_list, label="Fuselage angle $\\theta_f$ [deg]")
# # plt.plot(V_range, vi_list, label="Induced velocity")
# plt.xlabel("Forward Speed V [m/s]", fontsize=13)
# plt.ylabel("Pitch angles [rad]", fontsize=13)
# plt.title("Trim control inputs vs Forward Velocity", fontsize=15)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Save results
np.savez("trim_data.npz", V=V_range, theta_c=theta_c_list, theta_o=theta_o_list, theta_f=theta_f_list, vi=vi_list)
