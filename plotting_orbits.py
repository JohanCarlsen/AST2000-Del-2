import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem

class Planets:
    def __init__(self, system, planet_number):
        self.system = system
        self.e = system.eccentricities[planet_number]
        self.a = system.semi_major_axes[planet_number]
        self.p = self.a * (1-self.e**2)
        self.init_orb_angle = system.initial_orbital_angles[planet_number]
        self.init_angle = system.aphelion_angles[planet_number]
        self.rot_period = system.rotational_periods[planet_number]
        self.init_position = system.initial_positions[:,planet_number]
        self.f = np.linspace(0,2*np.pi, 101)
        self.r = self.p / (1 + self.e * np.cos(self.f))

    def plot(self, number):
        theta = np.linspace(0,2*np.pi,101)
        x = self.r * np.cos(self.f)
        y = self.r * np.sin(self.f)
        # plt.xlim(-30,30)
        # plt.ylim(-30,30)
        plt.plot(x, y, label=f'# {number}')
        plt.legend(loc='lower right')



seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
for i in range(8):
    planet = Planets(system, i)
    planet.plot(i+1)
plt.show()
