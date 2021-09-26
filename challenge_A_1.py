'''
EGEN KODE
'''

import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

class Planets_analytical:
    '''
    This class uses information about the planets and
    calculates angels and position vectors r
    '''
    def __init__(self, system, planet_number):
        self.system = system
        self.e = system.eccentricities[planet_number]                           # eccentricity of n'th planet
        self.a = system.semi_major_axes[planet_number]                          # semi major axis of n'th planet
        self.p = self.a * (1-self.e**2)
        self.aph_angle = system.aphelion_angles[planet_number]                  # aphelion angle of n'th planet
        self.omega = system.aphelion_angles[planet_number] - np.pi              # omega is the angle from x-axis down to the line between aphelion and perihelion
        self.f_corrected = np.linspace(0,2*np.pi, 101) - self.omega             # angle f with the corrected tilting
        self.f = np.linspace(0,2*np.pi, 101)                                    # angle f without correction
        self.r_corrected = self.p / (1 + self.e * np.cos(self.f_corrected))     # formula for r without correction
        self.r = self.p / (1 + self.e * np.cos(self.f))                         # formula for r with the corrected tilitng

    def plot(self, r):
        '''
        This method plots given r against theta
        '''
        theta = np.linspace(0,2*np.pi,101)
        x = r * np.cos(theta)   # here we go back to cartesian coordinates
        y = r * np.sin(theta)
        plt.plot(x, y)
        plt.xlabel('x [AU]')
        plt.ylabel('y [AU]')

if __name__=='__main__':
    '''
    Block for testing
    '''
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)

    plt.figure()
    plt.title('Analytical orbits not corrected')
    for i in range(8):
        planet = Planets_analytical(system, i)
        planet.plot(planet.r)

    plt.figure()
    plt.title('Analytical orbits corrected')
    for i in range(8):
        planet = Planets_analytical(system, i)
        planet.plot(planet.r_corrected)
    plt.show()
