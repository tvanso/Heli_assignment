import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

#Global parameters
V = 20 #m/s change back to 20!!!
q = 20 #deg/s pitch rate #change back to 20!!!
p = 10 #deg/s roll rate #change back to 10!!!
m = 2910 #kg  AANPASSEN!! 2910
rho = 1.225 #kg/m^3, sea level
#conversion
q = np.deg2rad(q)
p = np.deg2rad(p)
#-----------------

#helicopter parameters
R = 5.1 #m, blade radius
m_bl = 36.7 #kg, blade mass
c_eq = 0.288 #equivalent blade chord
RPM = 395 #rpm
CDS = 21.5 #ft^2
CDS = CDS * 0.092903 #m^2
f = 0.9 #correction factor of thin airfoil theory
Cla = 2 * np.pi * f #rad^-1
I_bl = (1/3) * m_bl * R**2
Lock = rho * Cla * c_eq * R**4 / I_bl #Value way too low
# Lock = 9
#------------------

#input parameters
pitch_coll = 6 #deg
long_cyclic = 2 #deg change back to 2!!!
lat_cyclic = 1 # deg change back to 1!!!
#------------------



#conversion
pitch_coll = np.deg2rad(pitch_coll)
long_cyclic = np.deg2rad(long_cyclic)
lat_cyclic = np.deg2rad(lat_cyclic)
omega = RPM / 60 * 2*np.pi #rad/s
W = m * 9.80665

#computation parameters
nPhi = 400
nr = 50
dphi = 2 * np.pi / nPhi
dr = R / nr

Azimuth = np.arange(0, 2*np.pi ,dphi)
R_lst = np.arange(dr, R+dr, dr)
R_lst = R_lst[int(0.5/dr):]
theta = pitch_coll + long_cyclic*np.sin(Azimuth) - lat_cyclic*np.cos(Azimuth)
#-----------------


def v_induced(a_d, W, V):
    T = W / np.cos(a_d)
    vih = np.sqrt(T / (2 * rho * np.pi * R**2))
    Vbar = V / vih
    Vbarc = Vbar * np.cos(a_d)
    Vbars = Vbar * np.sin(a_d)

    vifunc = lambda x: (1 / ((Vbarc)**2 + (Vbars + x)**2)) - x**2
    vibar = fsolve(vifunc, x0 = 6)[0]
    vi = vibar * vih
    return vi


#determine all angles and coefficent parameters
D_fus = CDS * 0.5 * rho * V**2
a_d = np.arctan(D_fus/W)
vi = v_induced(a_d, W, V)
# print(vi)

Lambda_i = vi/(omega*R)

#Calculating the coning values based on forward symmetric flight:

#setting up equations to be solved implicitly:
def f_mu(a_c):
    return (V * np.cos(a_c)) / (omega * R)

def f_lambda_c(a_c):
    return (V * np.sin(a_c)) / (omega * R)

def a_1(a_1):
    # return (-(16 / Lock) * (q / omega) + (p/omega) + (8/3 * f_mu(a_d + a_1) * long_cyclic) - (2 * f_mu(a_d + a_1)*(f_lambda_c(a_d + a_1) + Lambda_i))) / (1 - 0.5 * f_mu(a_d + a_1)**2) - a_1
    return ((-16/Lock)*(q/omega) - (p/omega) + (8/3)*f_mu(a_d - a_1)*pitch_coll - 2*f_mu(a_d - a_1)*(f_lambda_c(a_d - a_1) + Lambda_i) + long_cyclic*(1+3*f_mu(a_d - a_1)**2)) / (1 - 0.5*f_mu(a_d - a_1)**2) - a_1

init_guess = a_1(0) #initial guess based on alpha_c ~ alpha_d
a1_solv = fsolve(a_1, x0 = init_guess)[0]

#--------------------------------------------------

a_c = a_d - a1_solv
mu = f_mu(a_c)
lambda_c = f_lambda_c(a_c)


#steaty state motion parameters
a0 = (Lock / 8) * (pitch_coll * (1 + mu**2) - (4/3)*(lambda_c + Lambda_i) + (2/3)*(p/omega)*mu + (4/3)*mu*long_cyclic)
a1 = ((-16/Lock)*(q/omega) - (p/omega) + (8/3)*mu*pitch_coll - 2*mu*(lambda_c + Lambda_i) + long_cyclic*(1+3*mu**2)) / (1 - 0.5*mu**2)
b1 = ((4/3)*mu*a0 - (q/omega) + (p/omega)*((2/3)*mu - (16/Lock)*(p/omega))) / (1 + 0.5*mu**2) - lat_cyclic
#---------------------------------------------

a0tst = (Lock/8)*(pitch_coll*(1+mu**2) - (4/3)*(Lambda_i+lambda_c))
a1tst = ((-16/Lock)*(q/omega) + (8/3)*mu*pitch_coll - 2*mu*(lambda_c+Lambda_i)) / (1 - 0.5 * mu**2)
b1tst = (-(q/omega) + (4/3)*mu*a0) / (1 + 0.5*mu**2)


# print(f"a0:{a0}, a1:{a1}, b1:{b1}")

def beta(Phi):
    return a0 - a1 * np.cos(Phi) - b1 * np.sin(Phi)

def beta_dot(Phi):
    return (a1 * np.sin(Phi) - b1 * np.cos(Phi)) * omega


# plt.plot(np.rad2deg(Azimuth), np.rad2deg(beta(Azimuth)))
# plt.grid()
# plt.xlabel("Ψ [deg]")
# plt.ylabel("β [deg]")
# plt.xlim((0,360))
# # plt.plot(np.rad2deg(Azimuth), beta_dot(Azimuth))
# plt.show()


x = np.array([])
y = np.array([])
alpha = np.array([])

for r in R_lst:
    phi = ((V * np.sin(a_c) + vi + beta_dot(Azimuth)*r - q*r*np.cos(Azimuth) - p*r*np.sin(Azimuth) + beta(Azimuth)*V*np.cos(a_c)*np.cos(Azimuth)) / (omega*r + V*np.cos(a_c)*np.sin(Azimuth)))
    alpha_r = theta - phi

    alpha = np.append(alpha, alpha_r)
    x = np.append(x, r*np.sin(Azimuth))
    y = np.append(y, -r*np.cos(Azimuth))

alphad = np.rad2deg(alpha)

# lvls = np.arange(-3, 7, 1)
# contour_lines = plt.tricontour(x/R, y/R, alphad, levels = lvls, cmap='gist_rainbow')
# plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%d')  # Labels in degrees
# plt.plot(np.sin(Azimuth), np.cos(Azimuth), "k--")
# plt.scatter(0,0)
# plt.xlim((-1, 1))
# plt.ylim((-1, 1))
# plt.xlabel("X/R")
# plt.ylabel("Y/R")
# plt.title("Helicopter Blade Angle of Attack Contour")
# plt.show()


# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x/R, y/R, alphad, c=alphad, cmap='viridis', s=40)
# fig.colorbar(scatter, ax=ax, label='alpha [deg]')

# ax.set_xlabel("x/R")
# ax.set_ylabel("y/R")
# ax.set_zlabel("alpha")
# plt.show()
