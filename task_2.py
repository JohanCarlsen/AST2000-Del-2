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
        self.revolutions = 21
        self.total_time = self.P * self.revolutions
        self.time_steps = int(total_time * 11000)
        self.t = np.linspace(0, self.total_time, self.time_steps)

    def leapfrog(self):
        total_time, time_steps = self.total_time, self.time_steps
        v_initial, r_initial, G, star_mass_system = self.v_initial, self.r_initial, self.G, self.star_mass_system
        revolutions = self.revolutions

        @njit
        def run():
            t = np.linspace(0, total_time, time_steps)
            dA1 = 0
            dA2 = 0
            theta = 0
            T = 0
            len_traveled_dA1 = 0
            len_traveled_dA2 = 0
            mean_velocity_dA1 = np.zeros((1,2))
            mean_velocity_dA2 = np.zeros((1,2))
            if_count_1 = 0
            if_count_2 = 0

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

            # Unngår f-strings fordi Numba ikke liker dem
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
        self.t, self.v, self.r = run()

    def plot(self, number):
        plt.plot(self.r[:,0],self.r[:,1], label=f'Planet {number}')
        plt.xlabel('x [AU]', weight='bold')
        plt.ylabel('v [AU]', weight='bold')



if __name__=='__main__':
    seed = utils.get_seed('antonabr')
    system = SolarSystem(seed)
    # Stjeler parametere fra funksjonen (ja, kunne vært gjort mer effektivt)
    planet_number = 0
    G_sol = const.G_sol         # In AU^3 / yr^2 / m_sun
    axis = system.semi_major_axes[planet_number]
    planet_mass = system.masses[planet_number]
    star_mass_system = system.star_mass
    planet_P_time = np.sqrt(4*np.pi**2*axis**3 / (G_sol*(star_mass_system + planet_mass)))  # Keplers 3.lov
    revolutions = 21
    total_time = revolutions * planet_P_time        # years, (P^2 = 4pi^2(a1 + a2)^3 / (G(m1 + m2))) for home planet
    time_steps = int(total_time * 11000)
    t = np.linspace(0, total_time, time_steps)

    r_all = np.zeros((2, 8, time_steps))
    for i in range(8):
        planet = Planets_numerical(system, i)
        planet.leapfrog()
        planet.plot(i)
        r = planet.r
        r_all[:,i,:] = r[:,:].transpose(1,0)

    time2 = time.time()
    print(time2-time1)
    plt.legend(loc='lower right', prop={'size': 8})
    plt.show()

    # Sjekker om numeriske planetbaner stemmer og genererer video
    system.verify_planet_positions(total_time, r_all, time_steps)
    system.generate_orbit_video(t, r_all)
    # Sjekker Keplers 3.lov :
    for planet_number in range(8):
        planet_mass = system.masses[planet_number]
        axis = system.semi_major_axes[planet_number]
        Newton_P_planet = np.sqrt(4*np.pi**2*axis**3 / (G_sol*(star_mass_system + planet_mass)))  # Keplers 3.lov
        Kepler_P_planet = np.sqrt(axis**3) # Keplers 3.lov
        print(f'Planet {planet_number} -- \nNewton : {Newton_P_planet}\nKepler : {Kepler_P_planet}\n')
