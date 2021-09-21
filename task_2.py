import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import time

time1 = time.time()

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
        self.t = np.linspace(0, self.total_time, self.time_steps)

    def leapfrog(self):
        total_time, time_steps = self.total_time, self.time_steps
        v_initial, r_initial, G, star_mass_system = self.v_initial, self.r_initial, self.G, self.star_mass_system

        @njit
        def run():
            v = np.zeros((time_steps,2))
            r = np.zeros((time_steps,2))
            v[0] = v_initial
            r[0] = r_initial
            dt = total_time / time_steps
            r_norm = np.linalg.norm(r[0])
            a = -G*star_mass_system / r_norm**3 * r[0]

            for i in range(time_steps-1):
                r[i+1] = r[i] + v[i]*dt + 0.5*a*dt**2
                r_norm = np.linalg.norm(r[i+1])
                a_ipo = -G*star_mass_system / r_norm**3 * r[i+1]
                v[i+1] = v[i] + 0.5*(a + a_ipo)*dt
                a = a_ipo
            return v, r
        self.v, self.r = run()

    def plot(self, number):
        plt.plot(self.r[:,0],self.r[:,1])
        plt.title('Numerical orbits')
        plt.xlabel('x [AU]')
        plt.ylabel('v [AU]')

if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    for i in range(8):
        planet = Planets_numerical(system, i)
        planet.leapfrog()
        planet.plot(i)
        time2 = time.time()
    print(time2-time1)
    plt.show()
