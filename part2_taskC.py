import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import time

"""
EGEN KODE !!!
Anton Brekke
OBS: litt Norsk og Engelsk i kommentarer, noen ganger går hodet i Engelsk,
andre ganger i Norsk :))))
Skal plotte planetbanen til hjemplaneten med
leapfrod-metoden
"""

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission(seed)

G_sol = const.G_sol         # In AU^3 / yr^2 / m_sun
planet_number = 7       # Planet 7 er en god kandidat i solsystemet vårt
# system.print_info()
# Planet info
planet_radius = system.radii[planet_number]
planet_mass = system.masses[planet_number]
# planet_mass = 1
planet_axis = system.semi_major_axes[planet_number]

# Sun info
star_mass_system = system.star_mass
star_radii_system = system.star_radius

planet_P_time = np.sqrt(4*np.pi**2*planet_axis**3 / (G_sol*(star_mass_system + planet_mass)))
print(f'total time of orbit for home-planet is {planet_P_time}years\n')
reduced_mass = planet_mass * star_mass_system / (planet_mass + star_mass_system)
# P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2)) used for period of home planet

# system.print_info()

# Funksjon som løser tolegemeproblemet fra koordinatsystem med massesenteret i origo
@njit(cache=True)
def twobodyproblem(v1_cm_initial, r1_cm_initial, v2_cm_initial, r2_cm_initial):
    revolutions = 1
    total_time = revolutions * planet_P_time         # years, (P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2))) for home planet
    time_steps = int(total_time * 11000)     # 10 000 pr. year
    dt = total_time / time_steps
    t = np.linspace(0, total_time, time_steps)
    T = 0
    # Lager for å sjekke potensiell og kinetisk energi i banene
    Ek_tot = np.zeros(time_steps)
    Ep_tot = np.zeros(time_steps)

    r1_cm = np.zeros((time_steps, 2))       # Lager arrays som skal lagre posisjonen i massesenterkoordinatsystemet
    v1_cm = np.zeros((time_steps, 2))
    r1_cm[0] = r1_cm_initial
    v1_cm[0] = v1_cm_initial

    r2_cm = np.zeros((time_steps, 2))
    v2_cm = np.zeros((time_steps, 2))
    r2_cm[0] = r2_cm_initial
    v2_cm[0] = v2_cm_initial


    r = r2_cm[0] - r1_cm[0]
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    a1 = G_sol * planet_mass / r_norm**2 * r_hat
    a2 = -G_sol * star_mass_system / r_norm**2 * r_hat

    v1_cm_norm = np.linalg.norm(v1_cm[0])
    v2_cm_norm = np.linalg.norm(v2_cm[0])
    v = planet_mass * v2_cm_norm / reduced_mass     # Fra likningen v_2_cm = mu / m_2 * v
    Ek_tot[0] = 0.5*reduced_mass*v**2
    Ep_tot[0] = -G_sol*(planet_mass + star_mass_system)*reduced_mass / r_norm

    for i in range(time_steps-1):
        # Leapfrog
        r1_cm[i+1] = r1_cm[i] + v1_cm[i]*dt + 0.5*a1*dt**2
        r2_cm[i+1] = r2_cm[i] + v2_cm[i]*dt + 0.5*a2*dt**2
        r = r2_cm[i+1] - r1_cm[i+1]
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        a1_ipo = G_sol * planet_mass / r_norm**2 * r_hat
        a2_ipo = -G_sol * star_mass_system / r_norm**2 * r_hat
        v1_cm[i+1] = v1_cm[i] + 0.5*(a1 + a1_ipo)*dt
        v2_cm[i+1] = v2_cm[i] + 0.5*(a2 + a2_ipo)*dt

        v1_cm_norm = np.linalg.norm(v1_cm[i+1])
        v2_cm_norm = np.linalg.norm(v2_cm[i+1])

        v = planet_mass * v2_cm_norm / reduced_mass     # Fra likningen v_2_cm = mu / m_2 * v
        # Regner ut kinetisk og potensiell energi i systemet
        Ek_tot[i+1] = 0.5*reduced_mass*v**2
        Ep_tot[i+1] = -G_sol*(planet_mass + star_mass_system)*reduced_mass / r_norm

        a1 = a1_ipo
        a2 = a2_ipo
        T += dt

    return t, v1_cm, r1_cm, v2_cm, r2_cm, Ek_tot, Ep_tot

"""
Må gjøre om fra koordinatsystem med sola i origo
til koordinatsystem med massesenteret i origo. Husk at sola
har vektor r1 = 0 fra solas perspektiv, så massesenterer blir mu / m1 * r2
"""
r1_initial = np.array([0,0])
v1_initial = np.array([0,0])
r2_initial = system.initial_positions[:, planet_number]
v2_initial = system.initial_velocities[:, planet_number]

R_cm_initial = reduced_mass / planet_mass * r1_initial + reduced_mass / star_mass_system * r2_initial
V_cm_initial = reduced_mass / planet_mass * v1_initial + reduced_mass / star_mass_system * v2_initial

# Helt generelt er
r1_cm_initial = r1_initial - R_cm_initial
v1_cm_initial = v1_initial - V_cm_initial
r2_cm_initial = r2_initial - R_cm_initial
v2_cm_initial = v2_initial - V_cm_initial

# Henter alle verdier fra funksjonskall
t, v1_cm, r1_cm, v2_cm, r2_cm, Ek_tot, Ep_tot = twobodyproblem(v1_cm_initial, r1_cm_initial, v2_cm_initial, r2_cm_initial)

# Plotter energikurver
fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,:])

ax1.plot(t, Ek_tot, 'r'); ax1.set_title('Total kinetic energy')
ax1.set_xlabel('time [yr]'); ax1.set_ylabel(r'Energy $\left[\frac{AU^2 M\odot}{yr^2}\right]$')
ax2.plot(t, Ep_tot); ax2.set_title('Total potential energy')
ax2.set_xlabel('time [yr]'); ax2.set_ylabel(r'Energy $\left[\frac{AU^2 M\odot}{yr^2}\right]$')
ax3.set_ylim(np.min(Ep_tot), np.max(Ek_tot))
ax3.plot(t, Ek_tot + Ep_tot, 'purple'); ax3.set_title('Total energy')    # Plotter total energi som bør være konstant (bevaring av energi)
ax3.set_xlabel('time [yr]'); ax3.set_ylabel(r'Energy $\left[\frac{AU^2 M\odot}{yr^2}\right]$')

fig.tight_layout()
plt.show()

# Plotter banene til stjerna og sola
# plt.style.use('dark_background')
fig = plt.figure()
gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax1.plot(r2_cm[:,0], r2_cm[:,1], 'royalblue')
ax1.plot(r1_cm[:,0], r1_cm[:,1], 'r')
ax2.plot(r1_cm[:,0], r1_cm[:,1], 'r')
ax1.set_xlabel('x [AU]', weight='bold')
ax1.set_ylabel('y [AU]', weight='bold')
ax2.set_xlabel('x [AU]', weight='bold')
ax2.set_ylabel('y [AU]', weight='bold')
ax1.axis('equal')
ax2.axis('equal')
fig.tight_layout()
plt.show()
