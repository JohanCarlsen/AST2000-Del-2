import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

"""
EGEN KODE !!!

Skal plotte planetbanen til hjemplaneten med
leapfrod-metoden
"""
seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission(seed)
AU = 149597871
solar_mass = const.m_sun
G_sol = const.G_sol         # In AU^3 / yr^2 / m_sun
planet_radius = system.radii[0]
v_initial = system.initial_velocities[:,0]
r_initial = system.initial_positions[:,0]
planet_mass = system.masses[0]
P_planet = system.rotational_periods[0] / 365     # Dager
star_mass_system = system.star_mass
yr = 365*24*60*60



@njit
def leapfrog():
    revolutions = 20
    total_time = 100         # years
    time_steps = 100000     # 10 000 pr. year
    dt = total_time / time_steps
    t = np.linspace(0, total_time, time_steps)

    v = np.zeros((int(time_steps), 2))
    r = np.zeros((int(time_steps), 2))

    v[0] = v_initial
    r[0] = r_initial
    r_norm = np.linalg.norm(r[0])
    a = -G_sol*star_mass_system / r_norm**3 * r[0]
    for i in range(time_steps):
        r[i+1] = r[i] + v[i]*dt + 0.5*a*dt**2
        r_norm = np.linalg.norm(r[i+1])
        a_ipo = -G_sol*star_mass_system / r_norm**3 * r[i+1]
        v[i+1] = v[i] + 0.5*(a + a_ipo)*dt
        a = a_ipo
    return t, v, r

if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    mission = SpaceMission(seed)
    t, v, r = leapfrog()

    fig = plt.figure()
    plt.plot(r[:, 0], r[:, 1])
    plt.show()
