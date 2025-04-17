import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
import heli_specs as hs
from Q3_1 import *

# Load saved data
data = np.load("trim_data.npz")
V_trim = data["V"]
theta_c_trim = data["theta_c"]
theta_o_trim = data["theta_o"]
theta_f_trim = data["theta_f"] # negative values for theta_f_trim (pitch down)
vi_trim = data["vi"] # induced velocity for trim at given speed

# Function to get trim data at a given knot speed
def get_trim(knots):
    """
    This function gives the trim conditions for a given that the following has been loaded globally:
        data = np.load("trim_data.npz")
        V_trim = data["V_range"]
        theta_c_trim = data["theta_c"]
        theta_o_trim = data["theta_o"]
        theta_f_trim = data["theta_f"] # negative values for theta_f_trim (pitch down)
        vi_trim = data["vi"] # induced velocity for trim at given speed
    
    Returns:
        V_trim[i] = nearest velocity where trim conditions have been calculated [m/s]
        theta_o_trim[i], theta_c_trim[i], theta_f_list[i] = collective and cyclic pitch and fuselage angle at that speed in trim [rad]  
        vi_list[i] = induced velocity at given speed in trim [m/s]

    This function can later be used to check if the simulation goes to correct trim conditions.
    """
    v_mps = knots * 0.514444 # convert knots to m/s
    i = np.argmin(np.abs(V_trim - v_mps)) # obtain the index of the nearest value in the trim_data.npz file
    return (V_trim[i], theta_o_trim[i], theta_c_trim[i], theta_f_list[i], vi_list[i])


def heli_dynamics(x,uc):
    """
    This function computes the dynamics of the heli, for this p.95 of the lecture notes has been used. These are the EOM.
    inputs: states & control inputs
    x = [u, w, q, theta_f, vi]
    uc = [theta_o, theta_c]

    outputs: state derivatives
    x_dot = [u_dot, w_dot, q_dot, theta_f_dot, vi_dot]
    
    """

    u, w, q, theta_f, vi = x
    theta_o, theta_c = uc
    V = np.sqrt(u**2 + w**2)

    if V < 1e-3:
        V = 1e-3
    # Get the forces and moments
    T, D_fus, a1, CT_elem, CT_glauert = heli_forces(x, uc)
    
    # EOM:
    u_dot = -g * np.sin(theta_f) - (D_fus / MTOW) * (u / V) + (T / MTOW) * np.sin(theta_c - a1) - q * w
    w_dot = g * np.cos(theta_f) - (D_fus / MTOW) * (w / V) - (T / MTOW) * np.cos(theta_c - a1) + q * u
    q_dot = -(T / hs.I_y) * hs.h * np.sin(theta_c - a1)
    theta_f_dot = q

    tau = 0.4 # Check this if descritized!(probably)
    vi_dot  = (1 / tau) * (CT_elem - CT_glauert)

    return np.array([u_dot, w_dot, q_dot, theta_f_dot, vi_dot])

def heli_forces(x, uc):
    """
    Computes the thrust, fuselage drag, flapping angle, and thrust coefficients.

    Inputs:
        x = [u, w, q, theta_f, vi]
        uc = [theta_o, theta_c]
    
    Outputs:
        T           = thrust [N]
        D_fus       = fuselage drag [N]
        a1          = flapping angle [rad]
        CT_elem     = thrust coefficient (blade element method)
        CT_glauert  = thrust coefficient (Glauert)
    """

    u, w, q, theta_f, vi = x
    theta_o, theta_c = uc

    V = np.sqrt(u**2 + w**2)
    
    # Avoid dividing by zero 
    if V < 1e-3:
        V = 1e-3

    # Angle of attack and mu
    eps = np.arctan2(w, u) # takes care of quadrant, i.e. if u<0: alpha_c = alpha_c+np.pi 
    alpha_c = theta_c - eps
    mu = V / (hs.Omega * hs.R) * np.cos(alpha_c)

    # Induced inflow ratio
    lambda_i = vi / (hs.Omega * hs.R)

    lambda_c = V * np.sin(alpha_c) / (hs.Omega*hs.R)

    # Flapping angle a1
    a1 = ((8/3) * mu * theta_o - 2 * mu * (lambda_i + lambda_c) - 16 * q / (hs.gamma * hs.Omega)) / (1-0.5*mu**2)

    # Blade element method thrust coefficient
    CT_elem = 1/4 * hs.cl_alpha * hs.solidity*(2/3 * theta_o * (1 + 1.5 * mu**2) - (lambda_i + lambda_c))

    # Glauert momentum theory thrust coefficient
    Vx = V * np.cos(alpha_c - a1)
    Vy = V * np.sin(alpha_c - a1)
    CT_glauert = 2 * lambda_i * np.sqrt((Vx / (hs.Omega * hs.R))**2 + (Vy / (hs.Omega * hs.R) + lambda_i)**2)

    # Thrust force
    T = CT_elem * rho * hs.Omega**2 * np.pi * hs.R**4

    # Fuselage drag
    D_fus = 0.5 * hs.rho * hs.CdS * V**2 

    return T, D_fus, a1, CT_elem, CT_glauert


## ===== Start of Simulation =====

# --- Initial condition ---
V_init = 90 # given in KNOTS! (hover -> V_init=0)
V_init, theta_o_0, theta_c_0, theta_f_0, vi_0 = get_trim(V_init) # initial trim conditions

if V_init<=0.1: # thus, almost hover
    V_init = 0 # enforce V_init=0 for HOVER, since setting V=0 will devide by 0 in last script, the simulation for trim conditions starts at 0.1 therefor. Therefor otherwise, without this loop, it would start at V_init=0.1 instead of 0

# Starting from trim (q=0)
x0 = np.array([V_init*np.cos(theta_f_0), V_init*np.sin(theta_f_0), 0, theta_f_0, vi_0])  
u0 = np.array(get_trim(V_init)[1:3])

# Initial altitude (assume you start at 100ft)
h = 100 * 0.3048 

alt_error_int = 0 # integral of altitude  error
theta_f_error_int = 0 # integral of pitch error
c_des = 0 # Initial desired vertical speed

x = x0.copy()   
uc = u0.copy() 

# --- Objective ---
h_des = 100 * 0.3048 # desired altitude: 100ft 
def get_V_ref(t):
    # Define time windows (in seconds)
    t1 = 0        # 90 knots start
    t2 = 150      # after t2 seconds, go to 70
    t3 = 300      # after t3 seconds, go back to 90
    t4 = 450      # after t4 seconds, go to 110

    if t < t2:
        return 90
    elif t < t3:
        return 70
    elif t < t4:
        return 90
    else:
        return 110


# --- Choose Gains --- 
K1 = -0.01      # P-gain for altitude
K2 =  -0.001      # I-gain for altitude

Kp =1  # P-gain for pitch
Ki =3 # I-gain for pitch
Kd =-0.8 # D-gain for pitch

# Set general pitch to initial trim pitch
theta_o_gen = theta_o_0
theta_c_gen = theta_c_0

dt = 0.1
T_final = 600
time = np.arange(0, T_final, dt)

# Tracking variables
states = [x.copy()]
controls = [uc.copy()]
altitudes = [h]
velocity =[V_init]
check_errors = [0] 
theta_log = [theta_o_0]
c_log = [c_des]
V_ref_log = [get_V_ref(0)] 


for t in time[1:]:
    
    # These are the three validation inputs for the open loop model
    # 1) Validation using Figure 12.1 p.97 (step input)
    # if 0.5 < t < 1.0:
    #     uc[1] = 1.0 * np.pi / 180  # 1 deg step in cyclic pitch
    # else:
    #     uc[1] = u0[1]
    
    # # 2) Validation using Figure 12.3 p.99 (proportional feedback)
    # if t>4:
    #     uc[1] = 0.2 * x[3]  # θc = 0.2 θf 
    # else:
    #     uc[1] = u0[1]

    # 3) Validation using Figure 12.4 p.100 (PD feedback)
    # if t>0.1:
    #     uc[1] = 0.2 * x[3] + 0.2 * x[2]  # θc = 0.2 θf + 0.2 q
    # else:
    #     uc[1] = u0[1]
    # ==== Get Reference speed based on ADS33 ===
    
    V_ref = get_V_ref(t)
    _,_,_,theta_f_ref,_ = get_trim(V_ref)
    # Added correction for the bias (as explained in the discussion of Q3), a = 0.01250, b = -0.52500. Uncomment for the original system
    theta_f_ref =  theta_f_ref + np.radians(0.0125*V_ref-0.525) 
    
    # ==== ATTITUDE ====
    
    # Altitude dynamics
    c = - x[0] * np.sin(x[3]) + x[1] * np.cos(x[3])  # vertical velocity, defined positive upwards! 
    h_dot = c
    h += dt * h_dot

    # Altitude-hold controller
    alt_error = h_des - h
    alt_error_int += dt*alt_error
    uc[0] = theta_o_gen + K1 * alt_error + K2 * alt_error_int 


    # ==== FUSELAGE ANGLE ====
    theta_f_current = x[3]  # current fuselage pitch
    theta_f_error = theta_f_ref - theta_f_current
    theta_f_error_int += dt*theta_f_error

    uc[1] = -Kp * theta_f_error - Ki*theta_f_error_int -Kd*x[2] 
    
    
    V_heli = np.sqrt(x[0]**2 + x[1]**2)
    
    x_dot = heli_dynamics(x, uc)

    x = x + dt * x_dot  # Euler integration

    if np.any(np.abs(x) > 1e3) or np.any(np.isnan(x)):
        print(f"Unstable at t = {t:.2f} s — simulation stopped.")
        time = np.arange(0, t, dt)
        break 

    states.append(x.copy())
    controls.append(uc.copy())
    altitudes.append(h)
    velocity.append(V_heli)
    c_log.append(c)
    check_errors.append(theta_f_error)
    V_ref_log.append(V_ref)



states = np.array(states)
controls = np.array(controls)

# Extract the trajectory of each individual states
u_traj = states[:, 0]
w_traj = states[:, 1]
q_traj = states[:, 2] * 180 / np.pi # in degrees!
theta_f_traj = states[:, 3] * 180 / np.pi # in degrees!
vi_traj = states[:, 4]

theta_o_traj = controls[:, 0]
theta_c_traj = controls[:, 1]


# Plot 1: u and w (m/s)
plt.figure()
plt.plot(time, u_traj, label="u (m/s)")
plt.plot(time, w_traj, label="w (m/s)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("u, w and over time")
plt.legend()
plt.grid()


plt.figure()
V_ref_log =np.array(V_ref_log)
velocity_knots = np.array(velocity)/0.514444
# V_ref_mps = V_ref_log * 0.514444 # Convert V_ref_log (in knots) to m/s
# plt.plot(time, V_ref_mps, 'r--', label='$V_{ref}$ (m/s)')
# plt.plot(time, velocity, label='$V$ (m/s)')
plt.plot(time, V_ref_log, 'r--', label='$V_{ref}$ (knots)')
plt.plot(time, velocity_knots, label='$V$ (knots)')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (knots)")
plt.title("Helicopter Forward Speed vs Reference Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 2: q and theta_f (deg)
plt.figure()
plt.plot(time, q_traj, label="q (deg/s)")
plt.plot(time, theta_f_traj, label="θ_f (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Angular quantities (deg)")
plt.title("Pitch rate and pitch angle over time")
plt.legend()
plt.grid()

# Plot 3: vi
plt.figure()
plt.plot(time, vi_traj, label="v_i (m/s)")
plt.xlabel("Time (s)")
plt.ylabel("Induced velocity (m/s)")
plt.title("Rotor induced velocity over time")
plt.legend()
plt.grid()

# Plot 1: u and theta_f
plt.figure()
# plt.plot(time, u_traj, label="u (m/s)")
# plt.hlines(y=np.degrees(theta_f_ref), xmin=min(time), xmax=max(time), colors='r', linestyles='--', label=f'θ_f_trim (deg){theta_f_ref:.4f}')
# plt.hlines(y=theta_f_ref, xmin=min(time), xmax=max(time), colors='r', linestyles='--', label=f'θ_f_trim (rad){theta_f_ref:.4f}')
plt.plot(time, theta_f_traj, label="θ_f (deg)")
plt.xlabel("Time (s)")
plt.ylabel("-")
plt.title("theta_f over time")
plt.legend()
plt.grid()

plt.figure()
plt.plot(time, theta_o_traj * 180 / np.pi, label="θ₀ (deg)")
plt.plot(time, theta_c_traj * 180 / np.pi, label="θc (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Pitch angles (deg)")
plt.title("Collective and cyclic pitch over time")
plt.legend()
plt.grid()
plt.tight_layout()



# # Plot: h in meters
# plt.figure()
# plt.hlines(y=h_des, xmin=min(time), xmax=max(time), colors='r', linestyles='--', label='h_des')
# # plt.hlines(y=h_des + 30.48, xmin=min(time), xmax=max(time), colors='g', linestyles=':', label='Desired Bound (+100 ft)')
# # plt.hlines(y=h_des - 30.48, xmin=min(time), xmax=max(time), colors='g', linestyles=':')
# # plt.hlines(y=h_des + 60.96, xmin=min(time), xmax=max(time), colors='orange', linestyles='-.', label='Adequate Bound (+200 ft)')
# # plt.hlines(y=h_des - 60.96, xmin=min(time), xmax=max(time), colors='orange', linestyles='-.')
# plt.plot(time, altitudes, label="h (m)")
# plt.xlabel("Time (s)")
# plt.ylabel("-")
# plt.title("Altitude over time")
# plt.legend()
# plt.grid()


# Plot: h in knots
# Convert altitude to feet
altitudes_ft = np.array(altitudes) / 0.3048
h_des_ft = h_des / 0.3048

# Plot
plt.figure()
plt.hlines(y=h_des_ft, xmin=min(time), xmax=max(time), colors='r', linestyles='--', label='$h_{des}$ (ft)')

# Uncomment for bounds (optional)
plt.hlines(y=h_des_ft + 100, xmin=min(time), xmax=max(time), colors='g', linestyles=':', label='Desired Bound (+100 ft)')
plt.hlines(y=h_des_ft - 100, xmin=min(time), xmax=max(time), colors='g', linestyles=':')

# plt.hlines(y=h_des_ft + 200, xmin=min(time), xmax=max(time), colors='orange', linestyles='-.', label='Adequate Bound (+200 ft)')
# plt.hlines(y=h_des_ft - 200, xmin=min(time), xmax=max(time), colors='orange', linestyles='-.')

plt.plot(time, altitudes_ft, label='$h$ (ft)')
plt.xlabel("Time (s)")
plt.ylabel("Altitude (ft)")
plt.title("Altitude Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()





plt.show()

