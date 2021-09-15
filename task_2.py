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
        self.a_home_planet = system.semi_major_axes[0]
        self.P = np.sqrt(4*np.pi**2 / (self.G*(self.planet_mass + self.star_mass_system)) * self.a_home_planet**3)
        self.total_time = int(np.ceil(self.P * 20))
        self.time_steps = int(np.ceil(self.total_time * 10000))
        self.v = np.zeros((self.time_steps, 2))
        self.r = np.zeros((self.time_steps, 2))

    def leapfrog(self):
        dt = self.total_time / self.time_steps
        t = np.linspace(0, self.total_time, self.time_steps)

        self.v[0] = self.v_initial
        self.r[0] = self.r_initial

        r_norm = np.linalg.norm(self.r[0])
        a = -self.G*self.star_mass_system / r_norm**3 * self.r[0]

        for i in range(self.time_steps-1):
            self.r[i+1] = self.r[i] + self.v[i]*dt + 0.5*a*dt**2
            r_norm = np.linalg.norm(self.r[i+1])
            a_ipo = -self.G*self.star_mass_system / r_norm**3 * self.r[i+1]
            self.v[i+1] = self.v[i] + 0.5*(a + a_ipo)*dt
            a = a_ipo
        return t, self.v, self.r

    def plot(self, func, number):
        t, v, r = func
        plt.plot(r[:,0],r[:,1], label='Numerical #1')
        plt.title('Numerical orbits')
        plt.xlabel('x')
        plt.ylabel('v')
        plt.legend(loc='lower right')

if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    for i in range(8):
        planet = Planets_numerical(system, i)
        planet.plot(planet.leapfrog(), i)
    plt.show()
