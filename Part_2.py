'''
EGEN KODE!
MAIN code.
Classes from challenge_A_1 and challenge_A_2 are imported and run from this code.
Perhaps not the most effective way, but I tried something new.

'''

import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from challenge_A_2 import Planets_numerical
from challenge_A_1 import Planets_analytical

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)

'''
In every following loop, for Planets_analytical, the argument
can be either planet.r or planet.r_corrected.
The self.r is without correction for the aphelion angle.
'''


# challenge A1
plt.figure()
for i in range(8):
    planet = Planets_analytical(system, i)
    planet.plot(planet.r)
plt.title('Analytic orbits')
# plt.savefig('analytical-orbits-unrotated.jpg')

# challenge A2
plt.figure()
for i in range(8):
    planet = Planets_numerical(system, i)
    planet.leapfrog()
    planet.plot()
plt.title('Numerical orbits')
# plt.savefig('numerical-orbits.png')

plt.figure()
for i in range(8):
    planet = Planets_analytical(system, i)
    planet.plot(planet.r)
    planet = Planets_numerical(system, i)
    planet.leapfrog()
    planet.plot()
plt.title('Numerical and analytical orbits')
# plt.savefig('numerical-and-analytical-solar-system-unrotated.png')

plt.figure()
for i in range(8):
    planet = Planets_analytical(system, i)
    planet.plot(planet.r_corrected)
    planet = Planets_numerical(system, i)
    planet.leapfrog()
    planet.plot()
plt.title('Numerical and analytical orbits')
# plt.savefig('numerical-and-analytical-solar-system.png')

plt.show()
