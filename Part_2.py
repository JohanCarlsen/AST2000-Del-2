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

r_home_planet = planet_numerical.r
t_home_planet = planet_numerical.t
v_home_planet = planet_numerical.v
np.save('position_vector_home_planet', r_home_planet)
np.save('time_home_planet', t_home_planet)
np.save('velocity_home_planet', v_home_planet)


'''
r = np.linspace(0,10,10)
print(r)
np.save('array', r)
r_load = np.load('array.npy')
print(r_load)
'''
