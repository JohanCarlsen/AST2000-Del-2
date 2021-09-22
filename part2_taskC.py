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

AU = 149597871
solar_mass = const.m_sun
G_sol = const.G_sol         # In AU^3 / yr^2 / m_sun
yr = 365*24*60*60
planet_number = 7       # Planet 7 er en god kandidat i solsystemet vårt
system.print_info()
# Planet info
planet_radius = system.radii[planet_number]
planet_mass = system.masses[planet_number]
# planet_mass = 1
planet_axis = system.semi_major_axes[planet_number]
P_planet = system.rotational_periods[planet_number] / 365     # Dager

# Sun info
star_mass_system = system.star_mass
star_radii_system = system.star_radius

planet_P_time = np.sqrt(4*np.pi**2*planet_axis**3 / (G_sol*(star_mass_system + planet_mass)))

reduced_mass = planet_mass * star_mass_system / (planet_mass + star_mass_system)
# P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2)) used for period of home planet

# system.print_info()

# Funksjon som løser tolegemeproblemet fra koordinatsystem med massesenteret i origo
@njit(cache=True)
def twobodyproblem(v1_cm_initial, r1_cm_initial, v2_cm_initial, r2_cm_initial):
    revolutions = 1
    total_time = revolutions * planet_P_time         # years, (P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2))) for home planet
    time_steps = int(total_time * 10000)     # 10 000 pr. year
    dt = total_time / time_steps
    t = np.linspace(0, total_time, time_steps)

    r1_cm = np.zeros((time_steps, 2))
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

    v1 = np.zeros((time_steps, 2))
    v2 = np.zeros((time_steps, 2))

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
        a1 = a1_ipo
        a2 = a2_ipo

    return t, v1_cm, r1_cm, v2_cm, r2_cm

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

t, v1_cm, r1_cm, v2_cm, r2_cm = twobodyproblem(v1_cm_initial, r1_cm_initial, v2_cm_initial, r2_cm_initial)

# plt.style.use('dark_background')
plt.plot(r1_cm[:,0], r1_cm[:,1], 'r')
plt.plot(r2_cm[:,0], r2_cm[:,1], 'b')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
# plt.axis('equal')
plt.show()
