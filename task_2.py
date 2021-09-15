import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

class Planets_numerical:
    def __init__(self, system, planet_number):
        self.system = system
        self.AU = const.AU
        self.solar_mass = const.m_sun
        self.G = const.G_sol
        self.planet_rad = system.radii[planet_number]
        self.v_initial = system.initial_velocities[:,planet_number]
        self.r_initial = system.initial_positions[:,planet_number]
        self.planet_mass = system.masses[planet_number]
        self.star_mass_system = system.star_mass

    def leapfrog(self):
        total_time = 100
        time_steps = 100000
        dt = total_time / time_steps
        t = np.linspace(0, total_time, time_steps)

        v = np.zeros((int(time_steps), 2))
        r = np.zeros((int(time_steps), 2))

        v[0] = self.v_initial
        r[0] = self.r_initial

        r_norm = np.linalg.norm(r[0])
        a = -self.G*self.star_mass_system / r_norm**3 * r[0]

        for i in range(time_steps):
            r[i+1] = r[i] + v[i]*dt + 0.5*a*dt**2
            r_norm = np.linalg.norm(r[i+1])
            a_ipo = -self.G*self.star_mass_system / r_norm**3 * r[i+1]
            v[i+1] = v[i] + 0.5*(a + a_ipo)*dt
            a = a_ipo
        return t, v, r

    def plot(self, func, number):
        t, v, r = func
        plt.plot(r[:,0],r[:,1])

if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    for i in range(8):
        planet = Planets_numerical(system, i)
        planet.plot(planet.leapfrog(), i)
    plt.show()
