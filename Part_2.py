'''
EGEN KODE!
MAIN code
'''

import numpy as np
import matplotlib.pyplot as plt
# from numba import njit
# import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from Part2_task2 import leapfrog
from plotting_orbits import Planets_analytical

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)

plt.figure()
for i in range(8):
    planet = Planets_analytical(system, i)
    planet.plot(i+1)

t, v, r = leapfrog()

plt.figure()
planet = Planets_analytical(system, 0)
planet.plot(1)
plt.plot(r[:, 0], r[:, 1], label='Numerical')
plt.legend()
plt.title('Numerical and analytical')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
