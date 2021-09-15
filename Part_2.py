'''
EGEN KODE!
MAIN code
'''

import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from task_2 import Planets_numerical
from task_1 import Planets_analytical

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)

# task 1
plt.figure()
for i in range(8):
    planet = Planets_analytical(system, i)
    planet.plot(i+1)
plt.savefig('analytical-orbits.jpg')

# task 2
plt.figure()
for i in range(8):
    planet = Planets_numerical(system, i)
    planet.plot(planet.leapfrog(), i)
plt.savefig('numerical-orbits.png')

plt.figure()
planet_analytical = Planets_analytical(system, 0)
planet_analytical.plot(1)
planet_numerical = Planets_numerical(system, 0)
planet_numerical.plot(planet_numerical.leapfrog(), 1)
plt.legend()
plt.title('Numerical and analytical')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('numerical-and-analytical.png')

plt.show()

# N = planet_numerical.time_steps
#
# r_all = np.zeros((2, 8, N))
# for i in range(8):
#     planet = Planets_numerical(system, i)
#     r_all[:,i,:] = planet.r[:,:].T
#
# system.verify_planet_positions(planet_numerical.total_time, r_all)
