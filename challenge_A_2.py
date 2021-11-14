import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import time

"""
EGEN KODE
"""

time1 = time.time()

class Planets_numerical:
    '''
    This class uses information about the planets and calculates
    orbit time, P, for our home planet and plots orbits for
    every planet
    '''
    def __init__(self, system, planet_number):
        self.system = system
        self.AU = const.AU                                                                                          # astronomical unit
        self.solar_mass = const.m_sun                                                                               # mass of the sun in AU
        self.G = const.G_sol                                                                                        # gravitational constant in AU
        self.v_initial = system.initial_velocities[:,planet_number]                                                 # initial velocity
        self.r_initial = system.initial_positions[:,planet_number]                                                  # initial position
        self.planet_mass = system.masses[planet_number]                                                             # mass of n'th planet
        self.star_mass_system = system.star_mass                                                                    # mass of our star
        self.a_home_planet = system.semi_major_axes[0]                                                              # semi major axis of our home planet
        self.P = np.sqrt(4*np.pi**2 / (self.G*(self.planet_mass + self.star_mass_system)) * self.a_home_planet**3)  # orbital time for our home planet
        self.total_time = int(np.ceil(self.P * 20)) * 2                                                             # we'll run the simulation for 40 revolutions of our home planet
        self.time_steps = int(np.ceil(self.total_time * 10000))
        self.t = np.linspace(0, self.total_time, self.time_steps)                                                   # array containing time

    def leapfrog(self):
        '''
        This method will solve the diff. equations for each planet. In order
        to make it run faster (with numba), we defined a new function which is
        called from inside the method. Actually makes it a lot faster
        '''
        total_time, time_steps = self.total_time, self.time_steps
        v_initial, r_initial, G, star_mass_system = self.v_initial, self.r_initial, self.G, self.star_mass_system

        @njit
        def run():
            v = np.zeros((time_steps,2))
            r = np.zeros((time_steps,2))
            v[0] = v_initial
            r[0] = r_initial
            dt = total_time / time_steps
            r_norm = np.linalg.norm(r[0])                           # initial length of r
            a = -G*star_mass_system / r_norm**3 * r[0]              # initial acceleration

            for i in range(time_steps-1):
                '''
                This loop solves the diff. equations with leapfrog
                '''
                r[i+1] = r[i] + v[i]*dt + 0.5*a*dt**2
                r_norm = np.linalg.norm(r[i+1])
                a_ipo = -G*star_mass_system / r_norm**3 * r[i+1]
                v[i+1] = v[i] + 0.5*(a + a_ipo)*dt
                a = a_ipo
            return v, r
        self.v, self.r = run()

    def plot(self):
        '''
        This method plots the x- and y-positions of
        each planet against eachoter.
        '''
        plt.plot(self.r[:,0],self.r[:,1])
        plt.xlabel('x [AU]')
        plt.ylabel('v [AU]')

if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    for i in range(8):
        planet = Planets_numerical(system, i)
        planet.leapfrog()
        planet.plot()
        time2 = time.time()
    print(time2-time1)
    plt.show()
