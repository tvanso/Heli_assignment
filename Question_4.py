import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from Q2_pt2 import v_induced
from control import initial_response, StateSpace
from scipy.signal import find_peaks
from scipy.optimize import fsolve



#Helicopter parameters:
m = 2910 #kg
Iy = 5740 #kgm^2
CDS = 21.5 * 0.092903 #m^2
gamma = 9
R = 5.2 #m
Omega = 395 / 60 *2*np.pi #rad/s
h = 1.11 #m
Cla = 0.9 * 2 * np.pi
N = 4
c = 0.288 #m
#------------------------

#Flight parameters:
V_0 = 70 #kts
V_0 = V_0 * 0.5144444444
g = 9.80665 #m/s^2
rho = 1.225 #kg/m^3
theta_00 = 0.1249 #trim
theta_c0 = 0.0448  #trim
#-----------------------


#Starting parameters in trim condition (_0):
D_fus0 = 0.5 * rho * CDS * V_0**2
sigma = (N * c * R) / (np.pi * R**2) #rotor solidity

theta_f0 = np.arctan2(-D_fus0, (m*g)) #sum of forces in x-direction of the helicopter body
u_0 = V_0 * np.cos(theta_f0)
w_0 = V_0 * np.sin(theta_f0)
q_0 = 0 #trim condition

a_c0 = theta_c0 + np.arctan(w_0/u_0)
mu_0 = V_0*np.cos(a_c0) / (Omega*R)
lambda_c0 = V_0 * np.sin(a_c0) / (Omega*R)
a_d0 = np.arctan2(w_0, u_0)

#Calculating Lambda_i:
def Lambda_i0_f(lambda_i):
    a1 = (-(16 / gamma)*(q_0 / Omega) + (8 / 3)*mu_0*theta_00 - 2*mu_0*(lambda_c0 + lambda_i)) / (1 - 0.5*mu_0**2)

    CT_bem = ((Cla * sigma) / 4) * ((2/3)*theta_00*(1 + 1.5*mu_0**2) - (lambda_i + lambda_c0))
    CTGlau = 2* lambda_i*np.sqrt((V_0/(Omega*R)*np.cos(a_c0-a1))**2 + (V_0/(Omega*R)*np.sin(a_c0-a1)+lambda_i)**2)

    return CT_bem - CTGlau

lambda_i0 = v_induced(a_d0, m*g, V_0) / (Omega * R)
lambda_i0 = fsolve(Lambda_i0_f, x0=lambda_i0)[0]
#----------------------------

#Create nonlinear system using Sympy (full system):
u, w, q, theta_f, lambda_i, mu, lambda_c = sp.symbols('u, w, q, theta_f, lambda_i, mu, lambda_c')

V = sp.sqrt(u**2 + w**2)
a_c = theta_c0 - sp.atan(w/u) #-t_f = a_d
mu = (V * sp.cos(a_c)) / (Omega * R)
lambda_c = (V * sp.sin(a_c)) / (Omega * R)

a_1 = (-(16 / gamma)*(q / Omega) + (8 / 3)*mu*theta_00 - 2*mu*(lambda_c + lambda_i)) / (1 - 0.5*mu**2)
CT_bem = ((Cla * sigma) / 4) * ((2/3)*theta_00*(1 + 1.5*mu**2) - (lambda_i + lambda_c))

T = CT_bem * rho * (Omega*R)**2 * np.pi * R**2
D_fus = CDS * 0.5 * rho * V**2

X = -g*m*sp.sin(theta_f) - D_fus*sp.cos(theta_f) + T*sp.sin(theta_c0 - a_1)
M = -T*h*sp.sin(theta_c0 - a_1)
#-----------------------------------------------------------

#Linearization:
state_vector = [u, q, theta_f]

dX = [sp.diff(X, var) for var in state_vector]
dM = [sp.diff(M, var) for var in state_vector]
#----------------------------------------------------


#numerical values for derivatives:
sub_values = {
    u: u_0,
    w: w_0,
    q: q_0,
    theta_f: theta_f0,
    lambda_i: lambda_i0,
    mu: mu_0,
    lambda_c: lambda_c0
}

X_x = [float(expr.evalf(subs=sub_values)) for expr in dX]
M_x = [float(expr.evalf(subs=sub_values)) for expr in dM]
#------------------------------

print(f"Xu: {X_x[0]}, Mu: {M_x[0]}, Mq: {M_x[1]}")

#Set up simplified A matrix for phugoid:
A_phugoid = np.matrix([[X_x[0]/m, 0, -g],
                      [M_x[0]/Iy, M_x[1]/Iy, 0],
                      [0, 1, 0]])
#------------------------------
print(A_phugoid)

eigvals = np.linalg.eigvals(A_phugoid)
print(eigvals)

#--------------------------------
plt.scatter(eigvals.real, eigvals.imag)
plt.grid()
plt.axhline(0, color="black", linewidth = 1)
plt.axvline(0, color="black", linewidth = 1)
plt.show()
#---------------------------------

#Δx = [Δu, Δq, Δθ_f] at t = 0
Δu = 0.1 #velocity perturbation
x0 = np.array([Δu, 0, 0])
#--------------

#create statespace system and run over time period
sys_phugoid = StateSpace(A_phugoid,np.zeros((3,1)),np.eye(3),np.zeros((3,1)))

t_end = 60
dt = 0.1
t = np.linspace(0, t_end, int(t_end/dt))

t_out, y, x = initial_response(sys_phugoid, T=t, X0=x0, return_x=True)
#-----------------------------------------------

#Analyse period from the progression of Δu
peaks, _ = find_peaks(x[0])
t_peaks = t[peaks]
Period = np.diff(t_peaks)
#-----------------------

#Analyse motion parameters from oscilating eigenvalue
eigen_phugoid = eigvals[1]
xi = eigen_phugoid.real #real part
nu = eigen_phugoid.imag #imaginary part

T = (2*np.pi)/nu
damp = -xi / (np.sqrt(xi**2 + nu**2))
#-----------------------


print(f"Period from eigenvalues:{T} s")
print(f"Period Δu = {Period} seconds")

print(f"damping factor from eigenvalues:{damp}")



plt.plot(t_out, y[0], label="Δu [m/s]")
plt.plot(t_out, y[1], label="Δq [rad/s]")
plt.plot(t_out, y[2], label="Δθ_f [rad]")
plt.legend()
plt.grid()
plt.show()


 
