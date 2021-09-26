import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import time

"""
EGEN KODE !!!
Anton Brekke
OBS: litt Norsk og Engelsk i kommentarer, noen ganger går hodet i Engelsk,
andre ganger i Norsk :))))
Skal plotte planetbanen til hjemplaneten med
leapfrod-metoden
"""

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission(seed)

planet_number = 0
G_sol = const.G_sol         # In AU^3 / yr^2 / m_sun
planet_radius = system.radii[planet_number]
planet_mass = system.masses[planet_number]
star_mass_system = system.star_mass
axis = system.semi_major_axes[planet_number]
planet_P_time = np.sqrt(4*np.pi**2*axis**3 / (G_sol*(star_mass_system + planet_mass)))  # Keplers 3.lov

print(f'total time of orbit for home-planet is {planet_P_time}years\n')

# P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2)) used for period of home planet


time1 = time.time()

@njit(cache=True)
def leapfrog(v_initial, r_initial):
    revolutions = 21        # Minimum 20 revolutions
    total_time = revolutions * planet_P_time         # years, (P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2))) for home planet
    time_steps = int(total_time * 11000)    # 11 000 pr. year (minimun 10 000)
    dt = total_time / time_steps
    t = np.linspace(0, total_time, time_steps)
    dA1 = 0
    dA2 = 0
    theta = 0
    T = 0
    len_traveled_dA1 = 0
    len_traveled_dA2 = 0
    mean_velocity_dA1 = np.zeros((1,2))
    mean_velocity_dA2 = np.zeros((1,2))
    if_count_1 = 0      # Tellere vi trenger senere
    if_count_2 = 0

    r = np.zeros((time_steps, 2))
    v = np.zeros((time_steps, 2))

    v[0] = v_initial
    r[0] = r_initial
    r_norm = np.linalg.norm(r[0])
    a = -G_sol*star_mass_system / r_norm**3 * r[0]  # Setter initial akselerasjon
    for i in range(time_steps-1):
        # Leapfrog method
        r[i+1] = r[i] + v[i]*dt + 0.5*a*dt**2
        r_norm = np.linalg.norm(r[i+1])
        a_ipo = -G_sol*star_mass_system / r_norm**3 * r[i+1]
        v[i+1] = v[i] + 0.5*(a + a_ipo)*dt
        a = a_ipo

        v_norm = np.linalg.norm(v[i+1])
        w = v_norm / r_norm
        dtheta = w*dt
        # Regner areal i to forskjellige tidspunkt for å teste K2L, og andre størrelser
        if T > 0 and T < total_time / (10*revolutions):
            dA1 += 0.5*r_norm**2*dtheta
            len_traveled_dA1 += r_norm*dtheta
            mean_velocity_dA1 += v[i]
            if_count_1 += 1

        if T > total_time / (2*revolutions) and T < total_time / (2*revolutions) + total_time / (10*revolutions):
            dA2 += 0.5*r_norm**2*dtheta
            len_traveled_dA2 += r_norm*dtheta
            mean_velocity_dA2 += v[i]
            if_count_2 += 1

        T += dt
    mean_velocity_dA1 = mean_velocity_dA1 / if_count_1
    mean_velocity_dA2 = mean_velocity_dA2 / if_count_2
    print('max distance from sun : ', max(abs(np.min(r)), np.max(r)))

    # Unngår f-strings fordi Numba ikke liker dem, printer all relevant info
    print('Area T1 : ', dA1)
    print('length traveled T1 : ', len_traveled_dA1, 'AU')
    print('mean velocity T1 : ', mean_velocity_dA1, 'AU/yr')
    print('--')
    print('Area T2 : ', dA2)
    print('length traveled T2 : ', len_traveled_dA2, 'AU,')
    print('mean velocity T2 : ', mean_velocity_dA2, 'AU/yr')
    print('--')
    print('Relative error : ', abs(dA2 - dA1) / dA1 * 100, '%')
    print('Percentage min val of max val : ', min([dA1, dA2]) / max([dA1, dA2]) * 100, '%')
    print('')
    return t, v, r

# Stjeler parametere fra funksjonen (ja, kunne vært gjort mer effektivt)
revolutions = 21
total_time = revolutions * planet_P_time        # years, (P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2))) for home planet
time_steps = int(total_time * 11000)
t = np.linspace(0, total_time, time_steps)

# Array som skal inneholde alle x og y posisjoner i alle tidssteg for alle planeter
r_all = np.zeros((2, 8, time_steps))
for i in range(8):
    v_initial = system.initial_velocities[:,i]
    r_initial = system.initial_positions[:,i]
    t, v, r = leapfrog(v_initial, r_initial)
    plt.plot(r[:, 0], r[:, 1], label=f'Planet {i}')
    r_all[:,i,:] = r[:,:].transpose(1,0)

time2 = time.time()
print(f'Ran in {time2 - time1}s')
plt.xlabel('x [AU]', weight='bold'); plt.ylabel('y [AU]', weight='bold')
plt.legend(loc='lower right', prop={'size': 8})
plt.show()

# Sjekker om numeriske planetbaner stemmer og genererer video
# system.verify_planet_positions(total_time, r_all, time_steps)
# system.generate_orbit_video(t, r_all)

# Kjøretest fra terminal:
"""
The biggest relative deviation was for planet 0, which drifted 0.0303508 % from its actual position.
Your planet trajectories were satisfyingly calculated. Well done!
*** Achievement unlocked: Well-behaved planets! ***
Exact planet trajectories saved to 415183.
Generating orbit video with 450 frames.
Note that planet/moon rotations and moon velocities are adjusted for smooth animation.
XML file orbit_video.xml was saved in XMLs/.
It can be viewed in SSView.
"""

# Sjekker om areal korresponderer med planetbaner
"""
Area swept in 1 revolution planet by planet:
planet 0 : 10.734246161771328
planet 1 : 24.598255672165884
planet 6 : 41.36971937954822
planet 7 : 71.51498994162503
planet 2 : 188.76316868042682
planet 5 : 314.3195683069172
planet 3 : 576.7626246082025
planet 4 : 1401.6447852220829
"""

# Sjekker Keplers 3.lov :
for planet_number in range(8):
    planet_mass = system.masses[planet_number]
    axis = system.semi_major_axes[planet_number]
    Newton_P_planet = np.sqrt(4*np.pi**2*axis**3 / (G_sol*(star_mass_system + planet_mass)))  # Keplers 3.lov
    Kepler_P_planet = np.sqrt(axis**3) # Keplers 3.lov
    print(f'Planet {planet_number} -- \nNewton : {Newton_P_planet}\nKepler : {Kepler_P_planet}\n')
